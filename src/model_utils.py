"""Step 3 - model loading, architecture audit, and text generation helpers."""
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
             chat: bool = False, system_prompt: str | None = None):
    """Greedy, deterministic generation (repetition penalty keeps small models on track).

    chat=False -> raw completion of the prompt (base / CPT models).
    chat=True  -> wrap the prompt with the tokenizer's chat template (SFT model).
    """
    model.eval()
    outputs = []
    for prompt in prompts:
        if chat:
            messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
            messages.append({"role": "user", "content": prompt})
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=1.15, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
