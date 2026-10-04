"""Step 3 - model loading, architecture audit, and text generation helpers."""
import copy
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from src import config


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def native_bf16() -> bool:
    """True only on GPUs with hardware bf16 (Ampere+: A100, L4, L40S). A T4 is Turing."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def load_model(model_id_or_path=config.MODEL_ID, dtype=torch.bfloat16,
               gradient_checkpointing: bool = False):
    """AutoModelForCausalLM.from_pretrained with the requested precision."""
    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, dtype=dtype)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # cache is incompatible with checkpointing
    return model.to(device())


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable,
            "trainable_percent": round(100 * trainable / total, 4)}


def architecture_audit(model, tokenizer):
    cfg = model.config
    hidden = cfg.hidden_size
    heads = cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", heads)
    lm_head = model.get_output_embeddings()
    lm_out = lm_head.weight.shape[0]
    audit = {
        "model_type": cfg.model_type,
        "architectures": cfg.architectures,
        **count_parameters(model),
        "num_decoder_layers": cfg.num_hidden_layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "hidden_size": hidden,
        "head_dim": getattr(cfg, "head_dim", None) or hidden // heads,
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "config_vocab_size": cfg.vocab_size,
        "tokenizer_vocab_size": len(tokenizer),
        "lm_head_shape": list(lm_head.weight.shape),
        "lm_head_out_equals_vocab": lm_out == cfg.vocab_size,
        "tied_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
        "dtype": str(next(model.parameters()).dtype),
    }
    assert audit["lm_head_out_equals_vocab"], "lm_head output dim must equal vocab size"
    return audit


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new_tokens=config.GEN_MAX_NEW_TOKENS,
             chat: bool = False, system_prompt: str | None = None, batch_size: int = 8):
    """Greedy, deterministic generation (repetition penalty keeps small models on track).

    chat=False -> raw completion of the prompt (base / CPT models).
    chat=True  -> wrap the prompt with the tokenizer's chat template (SFT model).
    Prompts are generated in left-padded batches for speed.
    """
    model.eval()
    if chat:
        texts = []
        for prompt in prompts:
            messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
            messages.append({"role": "user", "content": prompt})
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    else:
        texts = list(prompts)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # decoder-only models must be left-padded
    # All decoding settings go into one GenerationConfig. The length limit is
    # expressed only as max_length = prompt + max_new_tokens, so it never clashes
    # with the checkpoint's own max_length=2048 (avoids a warning per call).
    gen_config = copy.deepcopy(model.generation_config)
    gen_config.update(do_sample=False, repetition_penalty=1.15, pad_token_id=pad_id,
                      max_new_tokens=None, temperature=None, top_p=None, top_k=None)
    outputs = []
    try:
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True,
                              add_special_tokens=not chat).to(model.device)
            gen_config.max_length = batch["input_ids"].shape[1] + max_new_tokens
            generated = model.generate(**batch, generation_config=gen_config)
            new_tokens = generated[:, batch["input_ids"].shape[1]:]
            outputs.extend(t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = side
    return outputs


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
