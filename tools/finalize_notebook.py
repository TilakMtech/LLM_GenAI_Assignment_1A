"""Finalise the executed submission notebook WITHOUT re-training anything.

1. Replaces the inference / observation markdown with text written for the final
   lab run (run 5, L40S, 2026-10-05), quoting its numbers and the reviewer's
   corrections (narrower claims, corrected verdicts).
2. Inserts three post-run cells and executes ONLY those cells in the
   `python_LLM` kernel (CPU + ~1 min of GPU inference, no training):
     * Step 4 - revised loss-curve reading from the saved log
     * B3a    - the adapter on the SAME three sentence-completion prompts as Steps 3/5
     * B3b    - manual review of the automatic keyword checks (full answers printed)
3. Replaces the stale-warning output of the Export cell and re-exports the HTML.

Usage (project root; notebook SAVED and its tab CLOSED in JupyterLab):
    python tools/finalize_notebook.py            # KERNEL=python_LLM by default
Running it twice is safe: previously inserted cells are replaced, not duplicated.
"""
import os
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "CorpPolicyLM_Assignment1A.ipynb"
KERNEL = os.environ.get("KERNEL", "python_LLM")
TAG = "finalize-inserted"

# ----------------------------------------------------------------------------- markdown
TEXT = {}

TEXT["**Inferences — Step 2**"] = r"""**Inferences — Step 2**
* **Tokens per word ≈ 1.62** (241,533 words → 391,053 tokens) — above the ~1.3 typical of general English. HR documents contain many strings the general-purpose LLaMA vocabulary splits into several pieces: upper-case headings (block 0 starts `▁Document ▁Type ▁Policy … Organ ization`), abbreviations (EL, CL, ICC, TA/DA), band codes (E1, J2, M3), clause numbers (4.1, a), ii.) and amounts (`Rs.10`, `3500/-`).
* **Held-out sets.** `eval` = the last 10 % of each of 25 documents = **9.3 % of words (41,584 tokens)**, cut at paragraph breaks; `eval_unseen` = **8 whole documents (6.5 % of words, 26,508 tokens)** never used for training. Taking the tail of *every* document keeps the main held-out set representative: documents span ~300 to ~58,000 words, so a few whole documents would be either tiny or dominated by one manual. Caveat: both sets were scored during the Step 4a sweep, so neither is an untouched final test set (see Step 5).
* **Packing.** Training: 349,469 tokens → **170 blocks × 2048**, zero padding; only 1,309 tail tokens (0.37 %) are dropped. Each document is wrapped `<s> … </s>`, so the 25 document boundaries inside the stream are explicit.
* **Average training document ≈ 14,000 tokens vs. a 2048-token context**, so long manuals span many consecutive blocks; packing (instead of one-document-per-sequence with truncation) is what lets CPT see the whole of each manual.
"""

TEXT["**Inferences — Step 3**"] = r"""**Inferences — Step 3**
* **Parameter count:** 1,100,048,384 (1.10 B), all trainable — CPT updates the full model, not adapters.
* **Architecture:** 22 decoder layers, hidden size 2048, 32 query heads × head dim 64 (2048 / 32), but only **4 key/value heads** — *grouped-query attention* (8 query heads share one KV head). That is why `k_proj`/`v_proj` are 2048→256 while `q_proj`/`o_proj` are 2048→2048; it shrinks the KV-cache 8×. SwiGLU MLP (5632), RMSNorm, RoPE up to 2048 positions.
* **lm_head check passes:** 2048 → 32,000 = tokenizer vocabulary = config `vocab_size`, so every logit maps to exactly one token id — required for correct cross-entropy/perplexity. Input embeddings and lm_head are *not* tied.
* **Baseline outputs are fluent but invented:** for probation leave the base model talks about pro-rata salary after 15 days of leave (MyGov: only casual/sick leave during probation); for own-vehicle travel it says “100 % of actual costs” (MyGov: Rs.10 per km); for the exit notice it produces generic contract boilerplate (MyGov: 60 days' written notice after 3 years' service, 90 days after 5 years). This is the reference point for Step 5 and Part B.
* **Precision:** the L40S (compute capability 8.9) has native bf16, so Step 4 trains with fp32 master weights + bf16 autocast; on a T4 (cc 7.5) the same code switches to fp16 autocast automatically.
"""

TEXT["**Inferences — Step 4**"] = r"""**Inferences — Step 4**
* **Ablation (4a) — larger, more frequent updates overfit and forget.** With the same 2-epoch token budget:

  | candidate | updates | final train loss | held-out loss | held-out PPL | general-fact probes kept |
  |---|---|---|---|---|---|
  | A: lr 5e-5, 16 K tok/update | 44 | 1.44 | **1.91** | **6.654 (−5.6 %)** | **3/3** |
  | B: lr 1e-4, 8 K tok/update | 86 | 1.18 | 2.04 | 7.588 (+7.6 %, worse) | 1/3 |
  | C: lr 2e-4, 8 K tok/update | 86 | 1.01 | 2.21 | 8.986 (+27.4 %, worse) | 0/3 |

  As the step size grows, the *training* loss falls (1.44 → 1.18 → 1.01) while the held-out loss *rises* (1.91 → 2.04 → 2.21) and fewer general-fact probes survive — overfitting of a 0.35 M-token corpus together with forgetting, the risk described in the assignment's tip. **A** is selected. **Selection caveat:** the held-out PPL and the three general-fact prompts were the selection criteria, so the Step 5 figures for these sets are *selection-set* results, not an independent test (A was also the pre-chosen default, which limits — but does not remove — this concern).
* **Correct starting point.** The pretrained loss on the training blocks is **1.9704** (held-out 1.9527) — at the low end of the expected 2–4 band, as expected for formulaic English prose; a randomly initialised LLaMA would start near ln 32,000 ≈ 10.4 (`train_cpt` aborts above 8).
* **How much the loss fell — two different numbers.** Endpoint change: **1.9704 → 1.4354 = −27.15 %**. The **32.55 %** printed by the loss-curve cell is a *different* quantity: the mean of the last few logged steps versus the mean of the first few (window = max(3, n/10) steps), which damps the per-step noise. The revised cell below prints both.
* **Where the curve plateaus — validation, not training.** The “train plateau ≈ step 8” annotation comes from an automatic rolling-window detector and is misleading: the training loss keeps falling sharply in the second epoch (epoch means 1.81 → 1.24). The meaningful turning point is the **held-out loss, which reaches its minimum 1.8897 at step 20 (end of epoch 1, 22 updates) and then rises to 1.9089 by the end**. The second epoch therefore reduced training loss but *worsened* validation loss relative to the first-epoch minimum — **overfitting**. We kept the final (end-of-epoch-2) checkpoint, which is permissible but ≈1 % worse in held-out loss than the best observed point; early stopping after epoch 1 would have been the better choice.
* **Resources:** 44 updates × 16,384 tokens ≈ 0.72 M tokens in 1.2 min on the L40S; peak GPU memory **14.73 GB** (fp32 weights + paged 8-bit AdamW + gradient checkpointing — tight for a 16 GB T4). The bf16 checkpoint (2.05 GiB) is saved to `models/cpt/`.
"""

TEXT["**Inferences — Step 5**"] = r"""**Inferences — Step 5**
* **5A — a positive but modest adaptation.** Held-out 10 %: PPL **7.051 → 6.655 (−5.62 %)**; unseen documents: **6.528 → 5.920 (−9.31 %)**. Neither set was used for training, so the drop is not memorisation of the evaluated text. It is below the 10–40 % the assignment describes as *typical*. Likely reasons, consistent with our results but not separately tested:
  1. **The base model already predicts this text well** (PPL ≈ 7): policy prose is formulaic English and TinyLlama's 3 T-token pre-training contains a lot of similar text.
  2. **The corpus is small** — 0.35 M training tokens, 44 updates.
  3. **More aggressive CPT did not help here** — in Step 4a, larger/more frequent updates made held-out PPL worse. Within the settings we tried, −5.6 % was the best result; a larger corpus is the obvious next step.
* **Caveat on independence.** Both PPL sets were reported during the Step 4a sweep and the held-out PPL drove the selection, so these are selection-set figures; a fresh, untouched test set would be needed for an unbiased estimate.
* **Unseen documents improved *more* (−9.31 %) than held-out tails (−5.62 %).** A plausible explanation (a hypothesis, not tested): the unseen set consists of short POSH / whistle-blower / code-of-conduct / human-rights policies whose wording follows statutory templates (POSH Act, SEBI vigil-mechanism rules) that recur across companies in the training data, while the held-out tails of long manuals end in annexures, pay tables and organisation-specific rules.
* **5B — the three general-fact probes were retained** (Paris; 100 °C; 300,000 km/s). These same three prompts were the forgetting guard in the Step 4a selection, so they are **selection probes, not independent evidence of broad knowledge retention**; a held-out set of general questions or a general-text perplexity would be needed for that claim. Only the continuation style changed (“10.2.3.4. The capital of India is New Delhi…”), a visible sign of domain adaptation. In the sweep, the same probes were lost at LR ≥ 1e-4 (1/3 and 0/3 kept).
* **Domain probes (raw completions).** The CPT completions sound like an HR manual but give other organisations' numbers (e.g. “30 days” notice instead of MyGov's 60/90 days, a percentage instead of Rs.10 per km). CPT teaches the *language* of the domain; recalling one organisation's specific rule needs instruction tuning or retrieval.
"""

TEXT["**Inferences — B1**"] = r"""**Inferences — B1**
* **Size and split:** **1,852** pairs from all 33 cleaned documents — **1,484 train / 368 eval (80.1 / 19.9 %)**, split by section group so no clause appears in both sets. The MyGov sections are kept in the training split so B3 tests recall of trained facts.
* **Grounding:** every response is copied from the source policy (no generated text), so responses carry the policy's wording; `source` and `section` give provenance for every pair. Grounded responses are not automatically *clean*, though — see limitations.
* **Mix:** clause-level pairs (769 cloze + 475 rule) dominate and teach precise single-rule answers; 381 section Q&A and 227 section summaries teach longer answers.
* **Quality controls** (added after inspecting earlier versions): tables of contents, annexures, forms and signature/approval blocks are skipped; dot-leader text, page numbers and colon-only intros are rejected (list items are merged into their intro); duplicate responses are removed; documents get curated titles from `data/sources.csv`.
* **Limitations:** quality was spot-checked on samples, not audited systematically. **PDF-extraction artefacts remain** — the conservative repair in Step 1 only joins a split word when the joined form occurs elsewhere in the corpus, so fragments such as “a re sent t o” and “per formance” are still present. Questions are templated (“What does the … require regarding …?”), so the model is less robust to free-form phrasing. The optional LLM method (prompt template above) would add natural paraphrases at the cost of a factuality check.
"""

TEXT["**Inferences — B2**"] = r"""**Inferences — B2**
* **Parameter efficiency:** Adapter B trains **2,252,800** parameters (22 layers × [16·(2048+2048) for `q_proj` + 16·(2048+256) for `v_proj`]) = **0.20 % of the 1.10 B model**. The printed 0.365 % is relative to 618 M because 4-bit weights are stored two per byte, so `numel()` of the quantised model counts roughly half the real parameters.
* **Memory and time:** peak GPU memory **2.74 GB** (vs 14.73 GB for full CPT) in 2.3 min — the 4-bit NF4 base needs ~0.7 GB and only the small LoRA matrices carry gradients and optimiser state. This is what makes QLoRA practical on a T4.
* **Learning curve:** 186 steps (2 epochs). Completion-only held-out loss improves **1.3862 → 1.3332** and mean token accuracy **0.686 → 0.695 (≈ 69.5 %)**, with most of the gain in the first epoch; train and held-out loss stay close (≈ 1.3–1.4), so there is no sign of over-fitting.
* **Why the loss stays ≈ 1.3:** responses are verbatim policy clauses, and an unseen clause cannot be predicted word-for-word — much of the remaining loss is content the model cannot know, not a failure to learn the format.
"""

TEXT["**Observations — B3**"] = r"""**Observations — B3** (read with the B3a and B3b cells above)
* **Answer format improved; lexical overlap rose.** On **30 held-out pairs — the first 30 rows of `instruction_eval.jsonl`** (section groups in seeded-random order, so a deterministic slice rather than a random sample of the 368; small n, no confidence interval) — ROUGE-L rises **0.063 → 0.382**. Without the adapter the CPT model continues a document (tables of contents, “1. Introduction 2. Definitions …”); with Adapter B it answers in one paragraph in the policies' register and stops with `</s>`.
* **ROUGE-L does not measure correctness.** One displayed answer scores **0.556** by copying the pay-scale prefix of the MSRLS clause but then *invents* “Insurance premium of Rs. 6000/- per child” where the reference says insurance cover (Mediclaim and Group Accident). High overlap with an invented fact is direct evidence that lexical overlap is not factual accuracy.
* **Manual review corrects the automatic checks (B3b).** The keyword checker only tests whether a keyword occurs and ignores contradictions:
  * MyGov probe questions: keyword check **1/3 → manual 0/3**. The leave answer describes extraordinary leave granted by management and does not state the probation entitlement (casual/sick leave only); the travel answer gives Rs. 1000/- per day (MyGov: Rs.10 per km); the notice answer gives 30 days (MyGov: 60 days after 3 years' service, 90 days after 5 years).
  * General questions through the chat template: keyword check **2/3 → manual 1/3**. Only Paris is correct; “100 °C at 100 mmHg” states the right number with a contradictory condition (at 100 mmHg water boils near 52 °C); the speed-of-light answer is wrong.
* **Same prompts as Steps 3/5 (B3a).** The assignment asks for the *same three prompts*; B3a runs the adapter on the identical sentence-completion text (raw, and as the user turn of the chat template) next to the base and CPT completions, with a strict check that the completion *begins* with the reference fact.
* **Why the facts are missed (hypotheses).** The MyGov policies are ~1.5 % of the corpus, many organisations state competing numbers for the same topics, and a rank-16 adapter on `q_proj`/`v_proj` trained for 2 epochs mainly learns the answer *format*. Questions that name the organisation helped in an earlier run but did not solve it.
* **General answers degraded with the adapter enabled** (1/3 under the HR system prompt and chat template). The experiment does not isolate the cause — the adapter, the system prompt, the chat format, or their interaction; the CPT model was only tested in raw-completion format, so the two results are not directly comparable.
* **What would help:** retrieval-augmented generation that puts the correct MyGov clause in the prompt (Assignment 2B), mixing general instruction data into SFT, a higher-capacity adapter (C), more epochs on the probe documents, LLM-paraphrased questions, and an evaluation based on manual or model-graded factual correctness rather than ROUGE.
"""

# ----------------------------------------------------------------------------- new cells
SETUP = r'''import os, sys, json, re
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import pandas as pd
pd.set_option("display.max_colwidth", None)
from src import config
EVAL = config.EVAL_DIR
'''

CELL_LOSS_MD = r"""#### Step 4 — revised loss-curve reading (added after the run; reads the saved log, no retraining)
The automatic “train plateau” annotation above is replaced by the held-out minimum, and the two loss-drop definitions are printed side by side."""

CELL_LOSS = SETUP + r'''import matplotlib.pyplot as plt
log = pd.read_csv(EVAL / "cpt_loss_log.csv")
tr, ev = log.dropna(subset=["loss"]), log.dropna(subset=["eval_loss"])
summary = json.load(open(EVAL / "cpt_training_summary.json"))
start, end = summary["start_train_loss"], float(tr["loss"].iloc[-1])
w = max(3, len(tr) // 10)
first_w, last_w = tr["loss"].iloc[:w].mean(), tr["loss"].iloc[-w:].mean()
best = ev.loc[ev["eval_loss"].idxmin()]
print(f"Endpoint change      : {start:.4f} -> {end:.4f} = -{100 * (start - end) / start:.2f}%")
print(f"Window means (w={w})   : {first_w:.4f} -> {last_w:.4f} = -{100 * (1 - last_w / first_w):.2f}%  (= 'loss_drop_percent' above)")
print(f"Held-out loss        : before {summary['start_eval_loss']:.4f} | minimum {best['eval_loss']:.4f} at step {int(best['step'])} "
      f"| final {ev['eval_loss'].iloc[-1]:.4f} (+{100 * (ev['eval_loss'].iloc[-1] / best['eval_loss'] - 1):.2f}% vs minimum)")
updates_per_epoch = summary["optimizer_updates"] / summary["hyperparameters"]["num_train_epochs"]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(tr["step"], tr["loss"], color="#9db4d6", lw=1, label="train loss (per step)")
ax.plot(ev["step"], ev["eval_loss"], "o-", color="#c0504d", label="held-out loss")
ax.axvline(updates_per_epoch, ls=":", color="grey"); ax.text(updates_per_epoch, ax.get_ylim()[1], " end of epoch 1", va="top", color="grey")
ax.annotate(f"held-out minimum {best['eval_loss']:.4f} (step {int(best['step'])})", (best["step"], best["eval_loss"]),
            xytext=(10, -30), textcoords="offset points", arrowprops=dict(arrowstyle="->", color="#c0504d"), color="#c0504d")
ax.set_xlabel("optimiser step"); ax.set_ylabel("cross-entropy loss"); ax.grid(alpha=.3); ax.legend()
ax.set_title("CPT loss: epoch 2 lowers training loss but raises held-out loss (overfitting)")
plt.tight_layout(); plt.savefig(config.FIGURE_DIR / "cpt_loss_curve_revised.png", dpi=150); plt.show()
'''

CELL_B3A_MD = r"""#### B3a — the adapter on the *same* three prompts as Steps 3 and 5 (added after the run; inference only)
The identical sentence-completion text is given to the adapter (i) raw, exactly as the base and CPT models received it, and (ii) as the user turn of the chat template. **Strict check:** the completion must *begin* with the reference fact (casual/sick leave; Rs.10 per km; 60 or 90 days), so a fact mentioned later alongside contradictions does not count."""

CELL_B3A = SETUP + r'''import torch
STRICT = [r"^\W*casual\s*(/|and|or|&)?\s*sick", r"^\W*rs\.?\s*10(?![\d,])", r"^\W*(60|90|sixty|ninety)\s*days"]
def strict(text, i):
    return bool(re.search(STRICT[i], str(text).strip().lower()))
prev = pd.read_csv(EVAL / "domain_generation_base_vs_cpt.csv")
if torch.cuda.is_available():
    from src.evaluate_sft import load_adapter_model
    from src.train_qlora import load_chat_tokenizer
    from src.model_utils import generate
    tok = load_chat_tokenizer(config.CPT_MODEL_DIR)
    model = load_adapter_model("B")
    raw = generate(model, tok, config.DOMAIN_PROMPTS, config.GEN_MAX_NEW_TOKENS)
    chat = generate(model, tok, config.DOMAIN_PROMPTS, config.GEN_MAX_NEW_TOKENS, chat=True,
                    system_prompt=config.SYSTEM_PROMPT)
    rows = []
    for i, prompt in enumerate(config.DOMAIN_PROMPTS):
        rows.append({"prompt (Steps 3/5, identical text)": prompt,
                     "base": prev["base_output"][i], "CPT": prev["cpt_output"][i],
                     "adapter (raw)": raw[i], "adapter (chat template)": chat[i],
                     "strict: base": strict(prev["base_output"][i], i), "strict: CPT": strict(prev["cpt_output"][i], i),
                     "strict: adapter raw": strict(raw[i], i), "strict: adapter chat": strict(chat[i], i)})
    same = pd.DataFrame(rows)
    same.to_csv(EVAL / "sft_adapter_B_same_prompts.csv", index=False)
    display(same)
    print("Strict-check totals (out of 3):", {c.replace("strict: ", ""): int(same[c].sum()) for c in same if c.startswith("strict")})
else:
    print("No GPU available - B3a needs the 4-bit adapter model; run this on the lab GPU.")
'''

CELL_B3B_MD = r"""#### B3b — manual review of the automatic keyword checks (added after the run)
The automatic check marks an answer correct if a keyword occurs anywhere, ignoring contradictions. Each answer was read in full; the manual verdict and its reason are listed below, and the aggregate counts are corrected."""

CELL_B3B = SETUP + r'''dom = pd.read_csv(EVAL / "sft_adapter_B_domain_prompts.csv")
gen = pd.read_csv(EVAL / "sft_adapter_B_general_prompts.csv")
# (manual verdict, reason, text that must appear in the reviewed answer)
MANUAL_DOMAIN = [
    ("Incorrect", "Describes extraordinary leave granted by management; does not state that a resource on probation may avail only casual/sick leave.", "xtraordinary"),
    ("Incorrect", "States Rs. 1000/- per day; MyGov reimburses use of an own vehicle @ Rs.10 per km.", "1000"),
    ("Incorrect", "States 30 days; MyGov requires 60 days' written notice after 3 years' service and 90 days after 5 years.", "30 days"),
]
MANUAL_GENERAL = [
    ("Correct", "Paris.", "Paris"),
    ("Incorrect", "100 °C is the boiling point at 1 atm (760 mmHg); at 100 mmHg water boils near 52 °C, so the stated condition contradicts the value.", "100 mmHg"),
    ("Incorrect", "The speed of light is about 299,792 km/s.", "km/s"),
]
answer_col = [c for c in dom.columns if c.startswith("sft_adapter")][0]
review = []
for i, (verdict, reason, anchor) in enumerate(MANUAL_DOMAIN):
    ans = str(dom[answer_col][i])
    review.append({"set": "MyGov probe question", "question": dom["question"][i], "full adapter answer": ans,
                   "keyword check": bool(dom["sft_has_fact"][i]), "manual verdict": verdict, "reason": reason,
                   "answer matches reviewed run": anchor in ans})
for i, (verdict, reason, anchor) in enumerate(MANUAL_GENERAL):
    ans = str(gen["sft_output"][i])
    review.append({"set": "general question", "question": gen["prompt"][i], "full adapter answer": ans,
                   "keyword check": bool(gen["correct"][i]), "manual verdict": verdict, "reason": reason,
                   "answer matches reviewed run": anchor in ans})
review = pd.DataFrame(review)
review.to_csv(EVAL / "sft_adapter_B_manual_review.csv", index=False)
display(review)
for name, part in review.groupby("set", sort=False):
    print(f"{name}: keyword check {int(part['keyword check'].sum())}/3 -> manual {int((part['manual verdict'] == 'Correct').sum())}/3")
if not review["answer matches reviewed run"].all():
    print("WARNING: some answers differ from the reviewed run - the manual verdicts must be re-checked.")
'''

INSERTIONS = [
    # (marker that identifies the cell to insert AFTER, [(type, source), ...])
    ("plot_loss_curve()", [("markdown", CELL_LOSS_MD), ("code", CELL_LOSS)]),
    ("evaluate_adapter(", [("markdown", CELL_B3A_MD), ("code", CELL_B3A),
                            ("markdown", CELL_B3B_MD), ("code", CELL_B3B)]),
]


def main():
    nb = nbformat.read(NB, as_version=4)
    # drop cells inserted by a previous run of this script
    nb.cells = [c for c in nb.cells if TAG not in c.get("metadata", {}).get("tags", [])]

    code = [c for c in nb.cells if c.cell_type == "code"]
    unexecuted = sum(c.get("execution_count") is None for c in code[:-1])
    if unexecuted:
        sys.exit(f"{unexecuted} code cells have no output in {NB} - save the notebook (Ctrl+S) first.")

    replaced = set()
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            first = cell.source.lstrip().split("\n", 1)[0].strip()
            key = next((k for k in TEXT if first.startswith(k)), None)
            if key:
                cell.source = TEXT[key].strip()
                replaced.add(key)
        elif cell.cell_type == "code" and "nbconvert" in cell.source and "stale" in cell.source:
            cell.outputs = [nbformat.v4.new_output(
                "stream", name="stdout",
                text="Exported: notebooks/CorpPolicyLM_Assignment1A.html (tools/finalize_notebook.py)\n")]
    missing = set(TEXT) - replaced
    if missing:
        sys.exit(f"Could not find these markdown cells: {sorted(missing)}")

    new_code_idx = []
    for marker, cells in INSERTIONS:
        pos = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and marker in c.source)
        for offset, (kind, source) in enumerate(cells, start=1):
            cell = (nbformat.v4.new_code_cell(source.strip()) if kind == "code"
                    else nbformat.v4.new_markdown_cell(source.strip()))
            cell.metadata["tags"] = [TAG]
            nb.cells.insert(pos + offset, cell)
    new_code_idx = [i for i, c in enumerate(nb.cells)
                    if c.cell_type == "code" and TAG in c.metadata.get("tags", [])]

    # Execute ONLY the inserted cells in the project kernel.
    from nbclient import NotebookClient
    os.environ["PYTHONNOUSERSITE"] = "1"
    client = NotebookClient(nb, kernel_name=KERNEL, timeout=900,
                            resources={"metadata": {"path": str(NB.parent)}})
    next_count = max((c.get("execution_count") or 0) for c in nb.cells if c.cell_type == "code") + 1
    with client.setup_kernel():
        for idx in new_code_idx:
            print(f"Executing inserted cell {idx} ...")
            client.execute_cell(nb.cells[idx], idx)
            nb.cells[idx].execution_count = next_count
            next_count += 1
            for out in nb.cells[idx].outputs:
                if out.get("output_type") == "stream":
                    print(out.get("text", "")[-1500:])

    nbformat.write(nb, NB)
    print(f"Updated {len(replaced)} inference cells and executed {len(new_code_idx)} new cells in {NB.name}")
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", str(NB),
                    "--output-dir", str(NB.parent)], check=True, env=env)
    print("Wrote", NB.with_suffix(".html"))
    print("\nPlease read the B3b table in the HTML once: confirm each full answer matches its manual verdict.")


if __name__ == "__main__":
    main()
