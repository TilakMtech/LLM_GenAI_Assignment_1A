"""Step 4 - continual pre-training (CPT) on the packed HR-policy corpus.

Precision strategy
------------------
Step 3 loads the model in bfloat16 (as the assignment asks) for the audit and
baseline. For the optimiser, pure-bf16 weights are a trap: bf16 has ~3
significant digits, so an AdamW update of lr*~1 = 2e-5 on a weight of ~1e-2 is
below bf16 resolution and is silently rounded away. We therefore keep fp32
master weights and use mixed precision:
  * GPUs with native bf16 (A100 / L4 / L40S)  -> bf16 autocast
  * T4 (Turing, no native bf16)               -> fp16 autocast + loss scaling
plus gradient checkpointing and a paged 8-bit AdamW (bitsandbytes) so a
1.1B-parameter full fine-tune fits in 16 GB.
"""
import csv
import math
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments, default_data_collator

from src import config
from src.model_utils import load_model, native_bf16, save_json
from src.tokenize_pack import load_tokenizer


class PackedDataset(Dataset):
    """Wraps the packed Parquet file; labels = input_ids (causal LM shift is internal)."""

    def __init__(self, parquet_path: Path):
        self.blocks = pd.read_parquet(parquet_path)["input_ids"].tolist()

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        ids = torch.tensor(list(self.blocks[idx]), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


class LossRecorderCallback(TrainerCallback):
    """Captures training loss (and eval loss / LR / grad-norm) at every logging step."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.rows = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": state.global_step, "epoch": round(state.epoch or 0, 4),
               "loss": logs.get("loss"), "eval_loss": logs.get("eval_loss"),
               "learning_rate": logs.get("learning_rate"), "grad_norm": logs.get("grad_norm")}
        if row["loss"] is None and row["eval_loss"] is None:
            return
        self.rows.append(row)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.rows)


@torch.no_grad()
def mean_loss(model, dataset, max_blocks: int | None = None):
    """Token-weighted mean cross-entropy over (a subset of) a packed dataset."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in range(len(dataset) if max_blocks is None else min(max_blocks, len(dataset))):
        batch = {k: v.unsqueeze(0).to(model.device) for k, v in dataset[i].items()}
        n_predicted = batch["input_ids"].shape[1] - 1
        total_nll += model(**batch).loss.float().item() * n_predicted
        total_tokens += n_predicted
    return total_nll / total_tokens


def precision_flags():
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}, "fp32 (CPU)"
    if native_bf16():
        return {"bf16": True, "fp16": False}, "bf16 mixed precision"
    return {"bf16": False, "fp16": True}, "fp16 mixed precision (T4)"


def pick_optimizer():
    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401
            return "paged_adamw_8bit"
        except ImportError:
            pass
    return "adamw_torch"


CHECKPOINT_GB = 2.6  # bf16 TinyLlama-1.1B (~2.2 GB) + tokenizer + headroom


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def check_disk_space(output_dir: Path, need_gb: float = CHECKPOINT_GB):
    """Fail *before* training if the checkpoint will not fit (an old checkpoint in
    output_dir is counted as reclaimable because it is replaced)."""
    probe = output_dir if output_dir.exists() else output_dir.parent
    probe.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(probe).free + dir_size(output_dir)
    if free < need_gb * 2**30:
        raise RuntimeError(
            f"Only {free / 2**30:.1f} GiB free for {output_dir} but the CPT checkpoint needs "
            f"~{need_gb} GiB. Free space first (see `du -sh ~/.cache/* ~/datavol-1/* ~/.local`), "
            "e.g. `pip cache purge`, delete old checkpoints/venvs, or set MODELS_DIR to a bigger volume.")
    return free


def train_cpt(model_id: str = config.MODEL_ID, output_dir: Path = config.CPT_MODEL_DIR,
              hparams: dict | None = None):
    hp = {**config.CPT, **(hparams or {})}
    tokenizer = load_tokenizer(model_id)
    train_ds = PackedDataset(config.PROCESSED_DIR / "train_packed.parquet")
    eval_ds = PackedDataset(config.PROCESSED_DIR / "eval_packed.parquet")

    # fp32 master weights; autocast handles the low-precision compute.
    free = check_disk_space(output_dir)
    print(f"Disk space for checkpoint: {free / 2**30:.1f} GiB available (need ~{CHECKPOINT_GB} GiB)")
    model = load_model(model_id, dtype=torch.float32, gradient_checkpointing=True)
    start_loss = mean_loss(model, train_ds, max_blocks=4)
    start_eval_loss = mean_loss(model, eval_ds)
    print(f"Starting loss (pretrained, before CPT): train {start_loss:.3f} | held-out {start_eval_loss:.3f}")
    if start_loss > 8:
        raise RuntimeError("Starting loss looks like a randomly initialised model (~10.4 = ln 32000); "
                           "check MODEL_ID / loading.")

    updates_per_epoch = math.ceil(len(train_ds) / (hp["per_device_train_batch_size"]
                                                   * hp["gradient_accumulation_steps"]))
    total_updates = hp["max_steps"] if hp["max_steps"] > 0 else math.ceil(
        updates_per_epoch * hp["num_train_epochs"])
    flags, precision = precision_flags()
    args = TrainingArguments(
        output_dir=str(config.LOG_DIR / "cpt_trainer"),
        learning_rate=hp["learning_rate"],
        num_train_epochs=hp["num_train_epochs"],
        max_steps=hp["max_steps"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        warmup_steps=max(1, round(hp["warmup_ratio"] * total_updates)),
        weight_decay=hp["weight_decay"],
        lr_scheduler_type=hp["lr_scheduler_type"],
        max_grad_norm=hp["max_grad_norm"],
        logging_steps=hp["logging_steps"],
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=max(1, total_updates // 8),
        save_strategy="no",
        gradient_checkpointing=True,
        optim=pick_optimizer(),
        report_to="none",
        seed=config.SEED,
        dataloader_pin_memory=torch.cuda.is_available(),
        **flags,
    )
    recorder = LossRecorderCallback(config.EVAL_DIR / "cpt_loss_log.csv")
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=default_data_collator, callbacks=[recorder])

    print(f"CPT: {len(train_ds)} blocks x {config.BLOCK_SIZE} tokens, {total_updates} optimiser "
          f"updates, {precision}, optimiser {args.optim}")
    t0 = time.time()
    result = trainer.train()
    minutes = (time.time() - t0) / 60

    # Persist the CPT checkpoint (bf16 halves the size) + tokenizer for Step 5 / Part B.
    if output_dir.exists():
        shutil.rmtree(output_dir)  # drop the previous (or partially written) checkpoint first
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.config.use_cache = True
    # Small shards keep host-RAM use low while serialising (6 GiB pod limit).
    trainer.model.to(torch.bfloat16).save_pretrained(output_dir, safe_serialization=True,
                                                     max_shard_size="500MB")
    tokenizer.save_pretrained(output_dir)

    summary = {
        "model_id": model_id, "precision": precision, "optimizer": str(getattr(args.optim, "value", args.optim)),
        "train_blocks": len(train_ds), "block_size": config.BLOCK_SIZE,
        "tokens_per_update": config.BLOCK_SIZE * hp["per_device_train_batch_size"]
                             * hp["gradient_accumulation_steps"],
        "optimizer_updates": result.global_step, "hyperparameters": hp,
        "start_train_loss": round(start_loss, 4), "start_eval_loss": round(start_eval_loss, 4),
        "final_logged_train_loss": next((r["loss"] for r in reversed(recorder.rows) if r["loss"]), None),
        "mean_train_loss": round(result.training_loss, 4),
        "train_minutes": round(minutes, 2),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                              if torch.cuda.is_available() else None,
        "checkpoint_dir": str(output_dir),
    }
    save_json(summary, config.EVAL_DIR / "cpt_training_summary.json")
    print(f"Saved CPT model + tokenizer to {output_dir} ({minutes:.1f} min)")
    return summary, recorder.rows


if __name__ == "__main__":
    train_cpt()
