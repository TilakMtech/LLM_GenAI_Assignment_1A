"""Phase 2 of the Part B revision (CPU only, < 1 min, no training).

Run AFTER tools/rerun_part_b.py. Adds the manual verdicts for the retrained
Adapter B (written after reading its full answers), the B2 inferences and the
B3 observations, makes the final summary self-contained, executes ONLY the B3b
and summary cells, and re-exports the HTML.

Usage (project root, notebook saved and its tab closed):
    KERNEL=python_llm python tools/complete_part_b.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

NB = ROOT / "notebooks" / "CorpPolicyLM_Assignment1A.ipynb"
KERNEL = os.environ.get("KERNEL", "python_llm")
FINGERPRINT = "90d6353c33461aac7fb9b2ce7a000b62ca0105bb0ef8a77eb136b7dbd882975e"


def one(nb, marker, kind):
    hits = [c for c in nb.cells if c.cell_type == kind and marker in c.source]
    if len(hits) != 1:
        sys.exit(f"Expected one {kind} cell containing {marker!r}; found {len(hits)}.")
    return hits[0]


B3B_MD = r"""#### B3b — manual review of the automatic keyword checks (retrained Adapter B)
The automatic check marks an answer correct if a keyword occurs anywhere, ignoring contradictions and wrong units. Each answer of the retrained adapter was read in full. The manual verdict and its reason are listed below, with corrected aggregate counts."""

B3B = r'''import os, sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import pandas as pd
from src import config
pd.set_option("display.max_colwidth", None)
EVAL = config.EVAL_DIR
assert json.loads((EVAL / "sft_adapter_B_summary.json").read_text())["instruction_dataset_fingerprint"] == \
    "''' + FINGERPRINT + r'''", "verdicts below belong to the adapter trained on the reviewed 66-pair dataset"
dom = pd.read_csv(EVAL / "sft_adapter_B_domain_prompts.csv")
gen = pd.read_csv(EVAL / "sft_adapter_B_general_prompts.csv")
# (manual verdict, reason, text that must appear in the reviewed answer)
MANUAL_DOMAIN = [
    ("Incorrect", "Says probation allows up to 15 days of unplanned leave of any kind, including maternity/paternity leave; "
                  "MyGov allows only casual and sick leave during probation.", "15 days of unplanned leave"),
    ("Incorrect", "Lists insurance and reasonableness conditions but gives no rate; MyGov reimburses use of an own vehicle "
                  "@ Rs.10 per km.", "adequate insurance cover"),
    ("Incorrect", "States 30 working days (15 for part-time) and then three months, which contradict each other; MyGov requires "
                  "60 days' written notice after 3 years' service and 90 days after 5 years.", "30 working days"),
]
MANUAL_GENERAL = [
    ("Correct", "Names Paris (the asked fact); the added population of 2.7 million is inaccurate (city about 2.1 million).", "Paris"),
    ("Incorrect", "Gives no boiling point (100 °C at sea-level pressure); only talks about temperature units.", "degrees Celsius"),
    ("Incorrect", "186,000 is the speed in miles per second; in km/s it is about 299,792.", "186,000 km/s"),
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

SUMMARY = r'''# Self-contained: reads the saved results of Part A and the revised Part B.
import json
import pandas as pd
from src import config
summary     = json.loads((config.REPORT_DIR / "corpus_summary.json").read_text())
tok_stats   = json.loads((config.REPORT_DIR / "tokenization_stats.json").read_text())
cpt_summary = json.loads((config.EVAL_DIR / "cpt_training_summary.json").read_text())
loss_stats  = json.loads((config.EVAL_DIR / "cpt_loss_stats.json").read_text())
ppl         = json.loads((config.EVAL_DIR / "perplexity.json").read_text())
forgetting  = pd.read_csv(config.EVAL_DIR / "forgetting_check.csv")
inst_stats  = json.loads((config.INSTRUCTION_DIR / "quality_manifest.json").read_text())
sft_summary = json.loads((config.EVAL_DIR / "sft_adapter_B_summary.json").read_text())
sft_eval    = json.loads((config.EVAL_DIR / "sft_adapter_B_eval_summary.json").read_text())
_c = (pd.read_csv(config.EVAL_DIR / "sft_adapter_B_manual_review.csv").groupby("set", sort=False)["manual verdict"]
      .apply(lambda s: f"{int((s == 'Correct').sum())}/{len(s)}"))
rows = {
  "documents (raw → clean)": f"{summary['documents_before']} → {summary['documents_after']}",
  "PDF pages (retained documents)": f"{inst_stats['pages_in_retained_corpus']}",
  "train tokens / blocks": f"{tok_stats['splits']['train']['total_tokens']:,} / {tok_stats['splits']['train']['packed_sequences']}",
  "CPT loss start → end": f"{cpt_summary['start_train_loss']:.3f} → {loss_stats['last_logged_loss']:.3f}",
  "domain PPL base → CPT (held-out 10%)": f"{ppl['base_ppl']:.2f} → {ppl['cpt_ppl']:.2f} (−{ppl['ppl_reduction_percent']}%)",
  "domain PPL base → CPT (unseen documents)": (f"{ppl['unseen_base_ppl']:.2f} → {ppl['unseen_cpt_ppl']:.2f} (−{ppl['unseen_ppl_reduction_percent']}%)"
                                             if "unseen_base_ppl" in ppl else "n/a"),
  "forgetting verdicts": ", ".join(forgetting.verdict),
  "instruction pairs train / eval": f"{inst_stats['train_pairs']} / {inst_stats['eval_pairs']}",
  "Adapter B held-out loss (final)": f"{sft_summary['final_eval_loss']:.4f}",
  "ROUGE-L no adapter → Adapter B": f"{sft_eval['rougeL_cpt_no_adapter']:.3f} → {sft_eval['rougeL_sft_adapter']:.3f} ({sft_eval['heldout_pairs_scored']} pairs)",
  "SFT manual correctness (B3b)": f"domain {_c['MyGov probe question']}; general {_c['general question']}",
}
display(pd.Series(rows).to_frame("result"))
'''

B2_INFER = r"""**Inferences — B2**
* **Parameter efficiency.** Adapter B trains **2,252,800** parameters, which is **0.20 % of the 1.10 B model**: 22 layers × [16·(2048+2048) for `q_proj` + 16·(2048+256) for `v_proj`]. The printed 0.365 % is relative to 618 M, because 4-bit weights are stored two per byte, so `numel()` of the quantised model counts roughly half of the real parameters.
* **Memory and time.** Peak GPU memory is **1.45 GB** (14.73 GB for full CPT), and 70 optimiser steps take 0.31 min. The 4-bit base needs about 0.7 GB, and only the small LoRA matrices carry gradients and optimiser state. This is what makes QLoRA practical on a T4.
* **Learning curve.** Completion-only training loss falls from 3.23 at the first step to about 2.0–2.2 in epochs 4–5 (mean 2.41). Held-out loss measured after each epoch:

  | after epoch | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|
  | held-out loss | 2.626 | 2.388 | 2.347 | 2.344 | 2.344 |

  Almost all of the gain comes in the first two epochs, and the curve is flat from epoch 3. Held-out loss never rises, so there is no sign of over-fitting, but epochs 4–5 add nothing; 3 epochs would have been enough.
* **Why the loss is higher than in the first version (2.34 vs 1.33).** The datasets differ, so the numbers are not directly comparable. The reviewed answers are paraphrases in another writer's words, which are harder to predict token by token than verbatim clauses the CPT model had already seen during continual pre-training. With only 54 examples, the adapter also sees far less of the answer style."""

B3_OBS = r"""**Observations — B3** (read with the B3a and B3b cells above)
* **Format learned.** Without the adapter, the CPT model answers every held-out chat question with a table of contents ("1. Introduction 2. Definitions & Interpretations …"). With Adapter B it answers in two or three fluent sentences in the register of the policies. ROUGE-L on all **12 held-out pairs** (the eval split; small n, no confidence interval) rises from **0.032 to 0.190**.
* **Facts not learned: lexical overlap is not correctness.** Many held-out answers state plausible policy details that are not in the source. For example:
  * Niramai's code is said to cover "consultants, contractors, … visitors"; the source covers payroll employees only.
  * A 30-day deadline for Bajaj Broking complaints is invented.
  * MyGov laptop users must "register with IT support"; this is invented.
  * Aurobindo's complaint is said to go to an internal committee, with an appeal; the source describes a 3-day preliminary enquiry.

  A few answers keep the gist (Astral's whistle-blower purpose, JioStar's coverage of retaliation). The ROUGE gain mostly reflects style and vocabulary.
* **Manual review (B3b).**
  * MyGov probe questions: keyword check 0/3, manual **0/3**. Probation leave: "15 days of unplanned leave … including maternity" (MyGov: casual and sick leave only). Own vehicle: insurance conditions but no rate (MyGov: Rs.10 per km). Notice: "30 working days … three months", which contradict each other (MyGov: 60 days after 3 years, 90 days after 5 years).
  * General questions through the chat template: keyword check 2/3, manual **1/3**. Paris is correct, though the added population is inaccurate. No boiling point is given. The speed of light is given as "186,000 km/s", which is the miles-per-second figure.
* **Training facts were not recalled.** The probation-leave rule and the Rs.10-per-km rate are both in the training pairs, phrased as comparative questions, yet neither was recalled for a differently worded question. Fifty-four pairs × 5 epochs with a rank-16 adapter on `q_proj`/`v_proj` teaches the answer *format*, not the facts. The 60/90-day notice is not in the SFT data at all.
* **Same prompts as Steps 3/5 (B3a).** None of the base, CPT and adapter (raw and chat) completions begins with the MyGov fact. The raw adapter continues like a manual, with other numbers (15 days after one year of service; 50 % of travel expenditure). Wrapped in the chat template, the adapter drifts off-topic.
* **Compared with the first dataset.** The first dataset had 1,852 verbatim pairs. The manual scores are the same (0/3 domain, 1/3 general), and the held-out ROUGE values cannot be compared because the eval sets differ. The reviewed dataset now meets the quality rules, but it gives the model about 28× fewer training examples.
* **What would help.**
  * Several hundred reviewed pairs, with the probe facts asked in several phrasings.
  * Retrieval-augmented generation that puts the correct MyGov clause in the prompt (Assignment 2B).
  * Mixing general instruction data into SFT to protect general answers.
  * A higher-capacity adapter (C).
  * Evaluation by manual or model-graded factual correctness rather than ROUGE."""


def main():
    nb = nbformat.read(NB, as_version=4)
    one(nb, "#### B3b", "markdown").source = B3B_MD
    b3b = one(nb, '"full adapter answer"', "code")
    b3b.source = B3B.strip()
    one(nb, "**Inferences — B2**", "markdown").source = B2_INFER
    one(nb, "**Observations — B3**", "markdown").source = B3_OBS
    summ = one(nb, '"ROUGE-L no adapter → Adapter B"', "code")
    summ.source = SUMMARY.strip()

    from nbclient import NotebookClient
    os.environ["PYTHONNOUSERSITE"] = "1"
    client = NotebookClient(nb, kernel_name=KERNEL, timeout=600,
                            resources={"metadata": {"path": str(NB.parent)}})
    count = max((c.get("execution_count") or 0) for c in nb.cells if c.cell_type == "code") + 1
    with client.setup_kernel():
        for cell in (b3b, summ):
            idx = nb.cells.index(cell)
            client.execute_cell(cell, idx)
            cell.execution_count = count
            count += 1
            for out in cell.outputs:
                if out.get("output_type") == "stream":
                    print(out.get("text", ""))
    nbformat.write(nb, NB)
    print("Updated B2/B3 text, manual verdicts and summary in", NB.name)
    shutil.copy2(config.INSTRUCTION_DIR / "instruction_dataset.jsonl", ROOT / "instruction_dataset.jsonl")
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    extra = [str(Path(p) / "share" / "jupyter") for p in {sys.base_prefix, "/opt/conda"}]
    env["JUPYTER_PATH"] = os.pathsep.join([env.get("JUPYTER_PATH", "")] + extra).strip(os.pathsep)
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", str(NB), "--output-dir", str(NB.parent)],
                   check=True, env=env)
    print("Wrote", NB.with_suffix(".html"))


if __name__ == "__main__":
    main()
