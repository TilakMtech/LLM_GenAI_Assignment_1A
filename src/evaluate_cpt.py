"""Step 4 (loss curve) and Step 5 (perplexity + catastrophic forgetting)."""
import gc
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src import config  # noqa: E402
from src.model_utils import generate, load_model, native_bf16, save_json  # noqa: E402
from src.tokenize_pack import load_tokenizer  # noqa: E402
from src.train_cpt import PackedDataset  # noqa: E402


def inference_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if native_bf16() else torch.float16


def free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ Step 4 plot
def find_plateau(losses, window: int = 5, tolerance: float = 0.02):
    """First step after which the rolling-mean loss improves < tolerance (relative)
    over the next `window` logged steps. Returns None if it never flattens."""
    smooth = pd.Series(losses).rolling(window, min_periods=1).mean().tolist()
    for i in range(window, len(smooth) - window):
        if (smooth[i] - smooth[i + window]) / smooth[i] < tolerance:
            return i
    return None


def plot_loss_curve(log_csv=config.EVAL_DIR / "cpt_loss_log.csv",
                    out_png=config.FIGURE_DIR / "cpt_loss_curve.png"):
    log = pd.read_csv(log_csv)
    train = log.dropna(subset=["loss"])
    evals = log.dropna(subset=["eval_loss"])
    window = max(3, len(train) // 10)
    plateau_idx = find_plateau(train["loss"].tolist(), window=window)
    plateau_step = int(train["step"].iloc[plateau_idx]) if plateau_idx is not None else None

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(train["step"], train["loss"], color="#9db4d6", lw=1, label="train loss (every step)")
    ax.plot(train["step"], train["loss"].rolling(window, min_periods=1).mean(), color="#1f4e8c",
            lw=2, label=f"train loss (rolling mean, {window})")
    if len(evals):
        ax.plot(evals["step"], evals["eval_loss"], "o-", color="#c0504d", lw=1.5,
                label="held-out eval loss")
    if plateau_step is not None:
        ax.axvline(plateau_step, ls="--", color="grey")
        ax.annotate(f"plateau ≈ step {plateau_step}", (plateau_step, train["loss"].max()),
                    xytext=(5, -5), textcoords="offset points", color="grey")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Continual pre-training loss — TinyLlama-1.1B on HR policy corpus")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    stats = {
        "first_logged_loss": float(train["loss"].iloc[0]),
        "last_logged_loss": float(train["loss"].iloc[-1]),
        "min_loss": float(train["loss"].min()),
        "loss_drop_percent": float(round(100 * (1 - train["loss"].iloc[-window:].mean()
                                          / train["loss"].iloc[:window].mean()), 2)),
        "plateau_step": plateau_step,
        "first_eval_loss": float(evals["eval_loss"].iloc[0]) if len(evals) else None,
        "last_eval_loss": float(evals["eval_loss"].iloc[-1]) if len(evals) else None,
    }
    save_json(stats, config.EVAL_DIR / "cpt_loss_stats.json")
    return stats, out_png


# ------------------------------------------------------------------ Step 5A
@torch.no_grad()
def perplexity(model, dataset):
    """PPL = exp( -(1/N) * sum log P(t_i | t_<i) ) over every held-out token. No gradients."""
    model.eval()
    nll, n_tokens = 0.0, 0
    for i in range(len(dataset)):
        batch = {k: v.unsqueeze(0).to(model.device) for k, v in dataset[i].items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = batch["input_ids"][:, 1:]
        nll += torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
            reduction="sum").item()
        n_tokens += shift_labels.numel()
    return math.exp(nll / n_tokens), n_tokens


def domain_perplexity(base_id=config.MODEL_ID, cpt_dir=config.CPT_MODEL_DIR):
    eval_ds = PackedDataset(config.PROCESSED_DIR / "eval_packed.parquet")
    results = {}
    for name, path in [("base", base_id), ("cpt", str(cpt_dir))]:
        model = load_model(path, dtype=inference_dtype())
        ppl, n = perplexity(model, eval_ds)
        results[name] = ppl
        print(f"{name} model domain PPL: {ppl:.3f} over {n:,} held-out tokens")
        free(model)
    reduction = 100 * (results["base"] - results["cpt"]) / results["base"]
    out = {"eval_tokens": n, "base_ppl": round(results["base"], 4),
           "cpt_ppl": round(results["cpt"], 4), "ppl_reduction_percent": round(reduction, 2),
           "within_expected_10_40_percent": 10 <= reduction <= 40}
    save_json(out, config.EVAL_DIR / "perplexity.json")
    return out


# ------------------------------------------------------------------ Step 5B
def keyword_verdict(text: str, keywords) -> bool:
    text = text.lower().replace("°", " ")
    return any(k.lower() in text for k in keywords)


def forgetting_check(base_id=config.MODEL_ID, cpt_dir=config.CPT_MODEL_DIR):
    """Side-by-side generations on general prompts (and the 3 domain prompts)."""
    tokenizer = load_tokenizer(base_id)
    prompts = [p for p, _ in config.GENERAL_PROMPTS]
    generations = {}
    for name, path in [("base", base_id), ("cpt", str(cpt_dir))]:
        model = load_model(path, dtype=inference_dtype())
        generations[name] = {
            "general": generate(model, tokenizer, prompts),
            "domain": generate(model, tokenizer, config.DOMAIN_PROMPTS),
        }
        free(model)

    rows = []
    for i, (prompt, keywords) in enumerate(config.GENERAL_PROMPTS):
        base_out, cpt_out = generations["base"]["general"][i], generations["cpt"]["general"][i]
        base_ok, cpt_ok = keyword_verdict(base_out, keywords), keyword_verdict(cpt_out, keywords)
        verdict = ("Retained" if cpt_ok else "Degraded" if base_ok
                   else "Retained (base also wrong)")
        rows.append({"prompt": prompt, "base_output": base_out, "cpt_output": cpt_out,
                     "expected_fact": " / ".join(keywords[:2]),
                     "base_correct": base_ok, "cpt_correct": cpt_ok, "verdict": verdict})
    forgetting = pd.DataFrame(rows)
    forgetting.to_csv(config.EVAL_DIR / "forgetting_check.csv", index=False)

    domain_rows = []
    for i, prompt in enumerate(config.DOMAIN_PROMPTS):
        refs = config.DOMAIN_REFERENCE[i]
        b, c = generations["base"]["domain"][i], generations["cpt"]["domain"][i]
        domain_rows.append({"prompt": prompt, "base_output": b, "cpt_output": c,
                            "reference_fact": " / ".join(refs[:2]),
                            "base_mentions_fact": keyword_verdict(b, refs),
                            "cpt_mentions_fact": keyword_verdict(c, refs)})
    domain = pd.DataFrame(domain_rows)
    domain.to_csv(config.EVAL_DIR / "domain_generation_base_vs_cpt.csv", index=False)
    return forgetting, domain
