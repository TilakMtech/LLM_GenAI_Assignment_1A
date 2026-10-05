"""Finalise the executed submission notebook WITHOUT re-running it.

* Replaces the inference / observation markdown cells with text written for the
  final lab run (run 5: L40S, 2026-10-05), quoting its actual numbers.
* Clears the stale-warning output of the Export cell.
* Re-exports notebooks/CorpPolicyLM_Assignment1A.html with nbconvert.

Usage (from the project root, with the notebook tab CLOSED in JupyterLab so
autosave cannot overwrite the result):
    python tools/finalize_notebook.py
"""
import os
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "CorpPolicyLM_Assignment1A.ipynb"

TEXT = {}

TEXT["**Inferences — Step 2**"] = r"""**Inferences — Step 2**
* **Tokens per word ≈ 1.62** (241,533 words → 391,053 tokens) — above the ~1.3 typical of general English. HR documents are full of things the general-purpose LLaMA vocabulary splits into many pieces: upper-case headings (block 0 starts `▁Document ▁Type ▁Policy … Organ ization`), abbreviations (EL, CL, ICC, TA/DA), band codes (E1, J2, M3), clause numbers (4.1, a), ii.) and amounts (`Rs.10`, `3500/-`). These are exactly the patterns CPT should learn to predict cheaply.
* **Held-out sets.** `eval` = the last 10 % of each of 25 documents = **9.3 % of words (41,584 tokens)**, cut at paragraph breaks; `eval_unseen` = **8 whole documents (6.5 % of words, 26,508 tokens)** that CPT never sees. Taking the tail of *every* document keeps the main held-out set representative: documents span ~300 to ~58,000 words, so a few whole documents would be either tiny or dominated by one manual and would mostly measure a *different organisation's* style.
* **Packing.** Training: 349,469 tokens → **170 blocks × 2048**, zero padding; only 1,309 tail tokens (0.37 %) are dropped. Each document is wrapped `<s> … </s>`, so the 25 document boundaries inside the stream are explicit.
* **Average training document ≈ 14,000 tokens vs. a 2048-token context**, so long manuals span many consecutive blocks; packing (instead of one-document-per-sequence with truncation) is what lets CPT see the whole of each manual.
"""

TEXT["**Inferences — Step 3**"] = r"""**Inferences — Step 3**
* **Parameter count:** 1,100,048,384 (1.10 B), all trainable — CPT updates the full model, not adapters.
* **Architecture:** 22 decoder layers, hidden size 2048, 32 query heads × head dim 64 (2048 / 32), but only **4 key/value heads** — *grouped-query attention* (8 query heads share one KV head). That is why `k_proj`/`v_proj` are 2048→256 while `q_proj`/`o_proj` are 2048→2048; it shrinks the KV-cache 8×. SwiGLU MLP (5632), RMSNorm, RoPE up to 2048 positions.
* **lm_head check passes:** 2048 → 32,000 = tokenizer vocabulary = config `vocab_size`, so every logit maps to exactly one token id — required for correct cross-entropy/perplexity. Input embeddings and lm_head are *not* tied.
* **Baseline outputs are fluent but invented:** for probation leave the base model talks about pro-rata salary after 15 days of leave (MyGov: only casual/sick leave on probation); for own-vehicle travel it says “100 % of actual costs” (MyGov: Rs.10 per km); for the exit notice it produces generic contract boilerplate (MyGov: 60 days). This is the reference point for Step 5 and Part B.
* **Precision:** the L40S (compute capability 8.9) has native bf16, so Step 4 trains with fp32 master weights + bf16 autocast; on a T4 (cc 7.5) the same code switches to fp16 autocast automatically.
"""

TEXT["**Inferences — Step 4**"] = r"""**Inferences — Step 4**
* **Ablation (4a) — more aggressive updates overfit and forget.** With the same 2-epoch token budget:

  | candidate | updates | final train loss | held-out loss | held-out PPL | general facts |
  |---|---|---|---|---|---|
  | A: lr 5e-5, 16 K tok/update | 44 | 1.44 | **1.91** | **6.65 (−5.6 %)** | **3/3** |
  | B: lr 1e-4, 8 K tok/update | 86 | 1.18 | 2.04 | 7.59 (+7.6 %, worse) | 1/3 |
  | C: lr 2e-4, 8 K tok/update | 86 | 1.01 | 2.21 | 8.99 (+27 %, worse) | 0/3 |

  Larger and more frequent steps drive the *training* loss down (1.44 → 1.01) while the held-out loss goes *up* (1.91 → 2.21) and general facts are lost — textbook memorisation of a 0.35 M-token corpus plus catastrophic forgetting, exactly the risk the assignment's tip describes. The guard (keep 3/3 general facts) and the PPL both select **A**, so the final model uses lr 5e-5 with 16 K tokens per update. Because A was also the pre-chosen default, selecting on the held-out set introduces no optimistic bias here.
* **Correct starting point.** The pretrained loss on the training blocks is **1.97** (held-out 1.95) — at the low end of the expected 2–4 band, as expected for formulaic English prose; a randomly initialised LLaMA would start near ln 32,000 ≈ 10.4 (`train_cpt` aborts above 8).
* **Loss curve.** Training loss falls 1.97 → 1.44 (epoch-1 mean 1.81, epoch-2 mean 1.24). The held-out loss falls 1.95 → **1.889 at step 20 — the end of epoch 1 (22 updates) —** and then drifts back up slightly to 1.909 while the training loss keeps dropping (step 25: 1.11; final gap −0.66). The plateau is therefore at the end of the first epoch; the second epoch is mostly memorisation, and one epoch would capture all of the generalisable gain (early stopping).
* **Resources:** 2 epochs ≈ 0.7 M tokens in 1.2 min per run on the L40S; peak GPU memory 12 GB (fp32 weights + paged 8-bit AdamW + gradient checkpointing), which also fits a 16 GB T4. The checkpoint (bf16, 2.05 GiB) is saved to `models/cpt/`.
"""

TEXT["**Inferences — Step 5**"] = r"""**Inferences — Step 5**
* **5A — domain adaptation in the right direction, modest in size.** Held-out 10 %: PPL **7.05 → 6.66 (−5.6 %)**; unseen documents: **6.53 → 5.92 (−9.3 %)**. Both sets were never seen in CPT, so the gains are generalisation, not memorisation of the evaluated text. They are below the 10–40 % band the assignment cites as typical, for three reasons visible in our own results:
  1. **The base model is already good at this text** (PPL ≈ 7): policy prose is formulaic English, and TinyLlama's 3 T-token pre-training includes a lot of similar web text, so there is little left to gain.
  2. **The corpus is small** — 0.35 M training tokens, 44 updates.
  3. **More aggressive CPT does not help** — the Step 4a ablation shows that larger/more frequent updates make held-out PPL *worse* (+7.6 % and +27 %) and cause forgetting. With this data, −5.6 % is close to what CPT can achieve; a much larger corpus is the way to a bigger gain.
* **Unseen documents improve more than held-out tails.** The unseen set consists of short POSH / whistle-blower / code-of-conduct / human-rights policies whose wording follows shared statutory templates (POSH Act, SEBI vigil-mechanism rules) that recur across companies in the training data, so learning those templates transfers well. The held-out tails of long manuals end in annexures, pay tables and organisation-specific rules that are harder to predict.
* **5B — no catastrophic forgetting.** All three general facts are **Retained** (Paris; 100 °C; 300,000 km/s). Only the *continuation style* changed: after the fact the CPT model switches to numbered policy clauses (“10.2.3.4. The capital of India is New Delhi…”, “2.3.4. The following are the minimum qualifications…”) — a visible sign of domain adaptation, not knowledge loss. The ablation shows forgetting appears at LR ≥ 1e-4 (1/3 and 0/3 facts kept), which is why the conservative setting was selected.
* **Domain probes (raw completions).** The CPT completions now *sound* like an HR manual (numbered clauses, “resource”, “eligible travel amount”) but give other organisations' numbers (e.g. “30 days” notice instead of MyGov's 60, a percentage instead of Rs.10/km). CPT teaches the *language* of the domain; recalling one organisation's specific rule is the job of instruction tuning (Part B) — or retrieval.
"""

TEXT["**Inferences — B1**"] = r"""**Inferences — B1**
* **Size and split:** **1,852** grounded pairs from all 33 cleaned documents — **1,484 train / 368 eval (80.1 / 19.9 %)**, split by section group so no clause appears in both sets. Large enough for a 1.1 B model to learn the *format* “question → answer from the policy”, small enough for a ~2-minute QLoRA run.
* **Grounding:** every response is copied from the source policy (no generated text), so the dataset cannot contain hallucinated facts; `source` and `section` give provenance for every pair.
* **Mix:** clause-level pairs (769 cloze + 475 rule) dominate and teach precise single-rule answers; 381 section Q&A and 227 section summaries teach longer answers.
* **Quality controls** added after inspecting earlier versions: tables of contents, annexures, forms and signature/approval blocks are skipped; dot-leader text, page numbers and colon-only intros are rejected (list items are merged into their intro); PDF-split words are re-joined in Step 1; duplicate responses are removed; documents get curated titles from `data/sources.csv`. The MyGov sections are kept in the training split so B3 tests recall of trained facts.
* **Limitations:** questions are templated (“What does the … require regarding …?”), so the model is less robust to free-form phrasing; some responses still carry PDF artefacts (e.g. “o f”, “a re”). The optional LLM method (prompt template above) would add natural paraphrases at the cost of a factuality check.
"""

TEXT["**Inferences — B2**"] = r"""**Inferences — B2**
* **Parameter efficiency:** Adapter B trains **2,252,800** parameters (22 layers × [16·(2048+2048) for `q_proj` + 16·(2048+256) for `v_proj`]) = **0.20 % of the 1.10 B model**. The printed 0.365 % is relative to 618 M because 4-bit weights are stored two per byte, so `numel()` of the quantised model counts roughly half the real parameters.
* **Memory:** peak GPU memory **1.95 GB** versus 12 GB for full CPT — the 4-bit NF4 base needs ~0.7 GB and only the small LoRA matrices carry gradients and optimiser state. This is what makes QLoRA practical on a T4.
* **Learning curve:** 186 steps (2 epochs, 2.3 min). Completion-only held-out loss improves **1.386 → 1.333** and mean token accuracy **0.686 → 0.695**, with almost all of the gain in the first epoch; the training loss stays around 1.3–1.4, so there is no sign of over-fitting (train and held-out loss are close).
* **Why the loss stays ≈ 1.3:** responses are verbatim policy clauses, and an unseen clause cannot be predicted word-for-word — the remaining loss is mostly irreducible content, not a failure to learn the format.
"""

TEXT["**Observations — B3**"] = r"""**Observations — B3**
* **Instruction following — clear improvement.** On 30 held-out question–answer pairs, ROUGE-L against the reference answers rises **0.063 → 0.382 (≈ 6×)**. Without the adapter, the CPT model answers a chat question by continuing a document — it emits tables of contents and lists (“1. Introduction 2. Definitions & Interpretations 3. …”). With Adapter B it answers directly, in one paragraph, in the register of the policies, and stops with `</s>`.
* **Factual recall of the three MyGov probes — weak.** The keyword check credits 1/3, but on inspection none of the three answers states the MyGov rule: the leave answer describes extraordinary leave instead of “casual / sick leave only”, the travel answer gives “Rs. 1000/- per day” instead of Rs.10 per km, and the notice answer gives 30 days instead of 60. Each is a *real rule from another organisation's manual* in the corpus. Reasons: the MyGov policies are only ~1.5 % of the corpus, many organisations state competing numbers for the same topic, and a 1.1 B model with a rank-16 adapter on `q_proj`/`v_proj` learns the answer *format* far more readily than which organisation a number belongs to. Questions that name the organisation help (an earlier run without the name answered every probe from other manuals) but do not solve it.
* **General knowledge through the chat template: 2/3** (Paris ✓; “100 °C at 100 mmHg” — right value, wrong condition; speed of light ✗). The CPT model alone kept all three (5B), and LoRA does not modify the base weights, so this is *over-specialisation of the answer style* under the HR system prompt, not erased knowledge.
* **What would fix it:** retrieval-augmented generation that puts the correct MyGov clause in the prompt (Assignment 2B), mixing general instruction data into SFT, a higher-capacity adapter (C), more epochs on the probe documents, or LLM-paraphrased questions for more varied phrasing.
"""


def main():
    nb = nbformat.read(NB, as_version=4)
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
                text="Exported: CorpPolicyLM_Assignment1A.html (tools/finalize_notebook.py)\n")]
    missing = set(TEXT) - replaced
    if missing:
        sys.exit(f"Could not find these markdown cells: {sorted(missing)}")
    code = [c for c in nb.cells if c.cell_type == "code"]
    unexecuted = sum(c.get("execution_count") is None for c in code[:-1])
    if unexecuted:
        sys.exit(f"{unexecuted} code cells have no output in {NB} - save the notebook (Ctrl+S) first.")
    nbformat.write(nb, NB)
    print(f"Updated {len(replaced)} inference cells in {NB.name}")
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", str(NB),
                    "--output-dir", str(NB.parent)], check=True, env=env)
    print("Wrote", NB.with_suffix(".html"))


if __name__ == "__main__":
    main()
