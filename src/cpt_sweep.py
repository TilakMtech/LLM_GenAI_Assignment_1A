"""Step 4a - small learning-rate / update-count ablation for CPT.

Each candidate is trained in memory (no checkpoint is written - the lab disk has
room for only one), then scored on
  * held-out perplexity (last 10% of every document and the unseen documents), and
  * the three general-knowledge prompts of Step 5B (catastrophic-forgetting guard).
The best candidate = lowest held-out PPL among those that keep all three general
facts; it is then retrained and saved by train_cpt().
"""
import pandas as pd
import torch

from src import config
from src.evaluate_cpt import free, inference_dtype, keyword_verdict, perplexity
from src.model_utils import generate, load_model, save_json
from src.tokenize_pack import load_tokenizer
from src.train_cpt import PackedDataset, train_cpt

# Same token budget (2 epochs); they differ in step size and number of optimiser updates.
CANDIDATES = [
    {"name": "A: lr 5e-5, 16K tok/update", "learning_rate": 5e-5, "gradient_accumulation_steps": 8},
    {"name": "B: lr 1e-4, 8K tok/update", "learning_rate": 1e-4, "gradient_accumulation_steps": 4},
    {"name": "C: lr 2e-4, 8K tok/update", "learning_rate": 2e-4, "gradient_accumulation_steps": 4},
]


def _datasets():
    return {k: PackedDataset(config.PROCESSED_DIR / f"{k}_packed.parquet")
            for k in ("eval", "eval_unseen") if (config.PROCESSED_DIR / f"{k}_packed.parquet").exists()}


def run_sweep(candidates=CANDIDATES):
    tokenizer = load_tokenizer()
    data = _datasets()
    prompts = [p for p, _ in config.GENERAL_PROMPTS]

    base = load_model(config.MODEL_ID, dtype=inference_dtype())
    base_ppl = {k: perplexity(base, ds)[0] for k, ds in data.items()}
    free(base)
    print("Base PPL:", {k: round(v, 3) for k, v in base_ppl.items()})

    rows = []
    for i, cand in enumerate(candidates):
        hp = {k: v for k, v in cand.items() if k != "name"}
        print(f"\n=== {cand['name']} ===")
        summary, _, model = train_cpt(hparams=hp, save=False, tag=f"_sweep{i}", return_model=True)
        model.eval()
        model.config.use_cache = True
        with torch.autocast("cuda", dtype=inference_dtype(), enabled=torch.cuda.is_available()):
            ppl = {k: perplexity(model, ds)[0] for k, ds in data.items()}
            general = generate(model, tokenizer, prompts, 30)
        kept = sum(keyword_verdict(out, keys) for out, (_, keys) in zip(general, config.GENERAL_PROMPTS))
        row = {"candidate": cand["name"], **hp, "updates": summary["optimizer_updates"],
               "final_train_loss": summary["final_logged_train_loss"],
               "final_heldout_loss": summary["last_eval_loss"],
               "heldout_ppl": round(ppl["eval"], 3),
               "heldout_ppl_drop_%": round(100 * (1 - ppl["eval"] / base_ppl["eval"]), 2),
               "general_facts_kept": f"{kept}/3"}
        if "eval_unseen" in ppl:
            row["unseen_ppl_drop_%"] = round(100 * (1 - ppl["eval_unseen"] / base_ppl["eval_unseen"]), 2)
        rows.append(row)
        print({k: row[k] for k in ("heldout_ppl", "heldout_ppl_drop_%", "general_facts_kept")})
        del model
        free(None)

    table = pd.DataFrame(rows)
    # Most general facts kept first (forgetting guard), then lowest held-out PPL.
    table["_kept"] = table["general_facts_kept"].str[0].astype(int)
    pick = table.sort_values(["_kept", "heldout_ppl"], ascending=[False, True]).iloc[0]
    table = table.drop(columns="_kept")
    best = next(c for c in candidates if c["name"] == pick["candidate"])
    best_hp = {k: v for k, v in best.items() if k != "name"}
    table.to_csv(config.EVAL_DIR / "cpt_lr_sweep.csv", index=False)
    save_json({"selected": best["name"], "hyperparameters": best_hp,
               "rule": "most general facts kept (forgetting guard), then lowest held-out PPL"},
              config.EVAL_DIR / "cpt_lr_sweep_selection.json")
    print(f"\nSelected: {best['name']}")
    return table, best_hp
