"""Part B3 - evaluate the QLoRA adapter on the same 3 domain prompts (+ held-out pairs)."""
import json
import re

import pandas as pd
import torch
from peft import PeftModel

from src import config
from src.evaluate_cpt import free, keyword_verdict
from src.model_utils import generate, save_json
from src.train_qlora import load_chat_tokenizer, load_quantized


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 on lower-cased word tokens (LCS based, no extra dependency)."""
    p = re.findall(r"\w+", prediction.lower())
    r = re.findall(r"\w+", reference.lower())
    if not p or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(len(p)):
        for j in range(len(r)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if p[i] == r[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def load_adapter_model(adapter_name=config.ADAPTER_NAME, quantize=True):
    base = load_quantized(config.CPT_MODEL_DIR, quantize)
    base.config.use_cache = True
    model = PeftModel.from_pretrained(base, str(config.ADAPTER_DIR / f"adapter_{adapter_name}"))
    model.eval()
    return model


def evaluate_adapter(adapter_name=config.ADAPTER_NAME, n_eval_pairs: int = 30, quantize=True):
    tokenizer = load_chat_tokenizer(config.CPT_MODEL_DIR)
    tokenizer.padding_side = "left"
    with (config.INSTRUCTION_DIR / "instruction_eval.jsonl").open(encoding="utf-8") as handle:
        eval_rows = [json.loads(line) for line in handle][:n_eval_pairs]
    questions = [r["instruction"] for r in eval_rows]
    max_new = 120

    model = load_adapter_model(adapter_name, quantize)
    sft_domain = generate(model, tokenizer, config.DOMAIN_QUESTIONS, max_new, chat=True,
                          system_prompt=config.SYSTEM_PROMPT)
    sft_general = generate(model, tokenizer, [p for p, _ in config.GENERAL_PROMPTS], 40, chat=True,
                           system_prompt=config.SYSTEM_PROMPT)
    sft_eval = generate(model, tokenizer, questions, max_new, chat=True, system_prompt=config.SYSTEM_PROMPT)
    # Same weights with the adapter switched off = the CPT model answering through the chat template.
    with model.disable_adapter():
        cpt_domain = generate(model, tokenizer, config.DOMAIN_QUESTIONS, max_new, chat=True,
                              system_prompt=config.SYSTEM_PROMPT)
        cpt_eval = generate(model, tokenizer, questions, max_new, chat=True,
                            system_prompt=config.SYSTEM_PROMPT)
    free(model)

    # Raw-completion outputs of base / CPT models from Step 5 for a 3-way comparison.
    previous = config.EVAL_DIR / "domain_generation_base_vs_cpt.csv"
    prev = pd.read_csv(previous) if previous.exists() else None
    rows = []
    for i, question in enumerate(config.DOMAIN_QUESTIONS):
        refs = config.DOMAIN_REFERENCE[i]
        rows.append({
            "question": question,
            "reference_fact": " / ".join(refs[:2]),
            "base_completion (Step 3)": prev["base_output"][i] if prev is not None else "",
            "cpt_completion (Step 5)": prev["cpt_output"][i] if prev is not None else "",
            "cpt_chat_no_adapter": cpt_domain[i],
            f"sft_adapter_{adapter_name}": sft_domain[i],
            "cpt_has_fact": keyword_verdict(cpt_domain[i], refs),
            "sft_has_fact": keyword_verdict(sft_domain[i], refs),
        })
    domain = pd.DataFrame(rows)
    domain.to_csv(config.EVAL_DIR / f"sft_adapter_{adapter_name}_domain_prompts.csv", index=False)

    general = pd.DataFrame([{"prompt": p, "sft_output": o, "correct": keyword_verdict(o, k)}
                            for (p, k), o in zip(config.GENERAL_PROMPTS, sft_general)])
    general.to_csv(config.EVAL_DIR / f"sft_adapter_{adapter_name}_general_prompts.csv", index=False)

    scores = pd.DataFrame([{"instruction": r["instruction"], "reference": r["response"],
                            "cpt_no_adapter": c, "sft_adapter": s,
                            "rougeL_cpt": rouge_l(c, r["response"]),
                            "rougeL_sft": rouge_l(s, r["response"])}
                           for r, c, s in zip(eval_rows, cpt_eval, sft_eval)])
    scores.to_csv(config.EVAL_DIR / f"sft_adapter_{adapter_name}_heldout_scores.csv", index=False)
    summary = {
        "adapter": adapter_name, "heldout_pairs_scored": len(scores),
        "rougeL_cpt_no_adapter": round(scores["rougeL_cpt"].mean(), 4),
        "rougeL_sft_adapter": round(scores["rougeL_sft"].mean(), 4),
        "domain_prompts_with_reference_fact": {"cpt_no_adapter": int(domain["cpt_has_fact"].sum()),
                                               "sft_adapter": int(domain["sft_has_fact"].sum())},
        "general_prompts_correct_after_sft": int(general["correct"].sum()),
    }
    save_json(summary, config.EVAL_DIR / f"sft_adapter_{adapter_name}_eval_summary.json")
    print(json.dumps(summary, indent=2))
    return domain, general, scores, summary


if __name__ == "__main__":
    evaluate_adapter()
