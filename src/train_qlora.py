"""Part B2 - QLoRA instruction fine-tuning of the CPT checkpoint (Adapter B).

* Base weights: the Step-4 CPT checkpoint, loaded in 4-bit NF4 with double
  quantisation (bitsandbytes); compute in bf16 (A100/L4) or fp16 (T4).
* Adapter B (balanced): r=16, alpha=32, target q_proj + v_proj, dropout 0.05.
* SFTTrainer on prompt/completion *conversations*, rendered with the model's
  chat template; loss is computed on the assistant completion only.
"""
import json
import math
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src import config
from src.model_utils import native_bf16, save_json
from src.train_cpt import LossRecorderCallback


def load_chat_tokenizer(path=config.CPT_MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if not tokenizer.chat_template:
        tokenizer.chat_template = config.CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def to_conversation(row):
    return {
        "prompt": [{"role": "system", "content": config.SYSTEM_PROMPT},
                   {"role": "user", "content": row["instruction"]}],
        "completion": [{"role": "assistant", "content": row["response"]}],
    }


def load_split(path: Path):
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return Dataset.from_list([to_conversation(r) for r in rows])


def compute_dtype():
    return torch.bfloat16 if native_bf16() else torch.float16


def load_quantized(path=config.CPT_MODEL_DIR, quantize: bool = True):
    if quantize and torch.cuda.is_available():
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=compute_dtype())
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb,
                                                     device_map={"": 0}, dtype=compute_dtype())
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:  # CPU dry-run path: plain LoRA on full-precision weights
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    model.config.use_cache = False
    return model


def train_qlora(adapter_name: str = config.ADAPTER_NAME, base_path=config.CPT_MODEL_DIR,
                hparams: dict | None = None, quantize: bool = True):
    from src.instruction_quality import verify_published
    quality = verify_published()
    hp = {**config.SFT, **(hparams or {})}
    adapter_cfg = config.ADAPTERS[adapter_name]
    tokenizer = load_chat_tokenizer(base_path)
    train_ds = load_split(config.INSTRUCTION_DIR / "instruction_train.jsonl")
    eval_ds = load_split(config.INSTRUCTION_DIR / "instruction_eval.jsonl")
    model = load_quantized(base_path, quantize)

    lora = LoraConfig(r=adapter_cfg["r"], lora_alpha=adapter_cfg["lora_alpha"],
                      target_modules=adapter_cfg["target_modules"], lora_dropout=hp["lora_dropout"],
                      bias="none", task_type="CAUSAL_LM")
    steps_per_epoch = math.ceil(len(train_ds) / (hp["per_device_train_batch_size"]
                                                 * hp["gradient_accumulation_steps"]))
    total_steps = hp["max_steps"] if hp["max_steps"] > 0 else math.ceil(steps_per_epoch * hp["num_train_epochs"])
    on_gpu = torch.cuda.is_available()
    out_dir = config.ADAPTER_DIR / f"adapter_{adapter_name}"
    args = SFTConfig(
        output_dir=str(config.LOG_DIR / f"sft_adapter_{adapter_name}"),
        learning_rate=hp["learning_rate"],
        num_train_epochs=hp["num_train_epochs"],
        max_steps=hp["max_steps"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=hp["per_device_train_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        warmup_steps=max(1, round(hp["warmup_ratio"] * total_steps)),
        lr_scheduler_type="cosine",
        logging_steps=hp["logging_steps"],
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=max(1, total_steps // 5),
        save_strategy="no",
        max_length=hp["max_length"],
        completion_only_loss=True,
        gradient_checkpointing=on_gpu,
        optim="paged_adamw_8bit" if on_gpu and quantize else "adamw_torch",
        bf16=on_gpu and native_bf16(),
        fp16=on_gpu and not native_bf16(),
        report_to="none",
        seed=config.SEED,
    )
    recorder = LossRecorderCallback(config.EVAL_DIR / f"sft_adapter_{adapter_name}_loss_log.csv")
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tokenizer, peft_config=lora, callbacks=[recorder])
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"Adapter {adapter_name}: r={lora.r}, alpha={lora.lora_alpha}, targets={adapter_cfg['target_modules']}"
          f" -> {trainable:,} trainable / {total:,} params ({100 * trainable / total:.3f}%)")

    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    minutes = (time.time() - t0) / 60
    metrics = trainer.evaluate()

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    summary = {
        "instruction_dataset_fingerprint": quality["dataset_fingerprint"],
        "adapter": adapter_name, **adapter_cfg, "lora_dropout": hp["lora_dropout"],
        "quantization": "4-bit NF4 + double quant" if quantize and on_gpu else "none (CPU dry-run)",
        "train_pairs": len(train_ds), "eval_pairs": len(eval_ds),
        "trainable_params": trainable, "total_params": total,
        "trainable_percent": round(100 * trainable / total, 4),
        "optimizer_steps": result.global_step, "mean_train_loss": round(result.training_loss, 4),
        "final_eval_loss": round(metrics.get("eval_loss", float("nan")), 4),
        "train_minutes": round(minutes, 2),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if on_gpu else None,
        "hyperparameters": hp, "adapter_dir": str(out_dir),
    }
    save_json(summary, config.EVAL_DIR / f"sft_adapter_{adapter_name}_summary.json")
    print(json.dumps(summary, indent=2))
    return summary, recorder.rows


if __name__ == "__main__":
    train_qlora()
