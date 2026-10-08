"""Phase 1 of the Part B revision: retrain Adapter B on the reviewed dataset and
re-run Part B INSIDE the submission notebook (Part A outputs are kept as they are).

What it does
  1. Applies tools/polish_notebook.py edits if they are not in the notebook yet.
  2. Backs up the notebook, the old Adapter B and its result files
     (outputs/backups/partB_before_revision/).
  3. Rewrites Part B:  B1 method + code (reviewed 66-pair dataset, page count,
     length / template / verbatim checks, independent 15-pair spot-check),
     B2 (5 epochs, grad-accum 1 for the small dataset), B3 (unchanged evaluation
     + same-prompt cell), and a B3b answer table WITHOUT verdicts.
     Inferences that depend on the new results are marked PENDING.
  4. Adds a cell that reloads the saved Part A results, then executes Part B and
     the final summary in the python_LLM kernel (GPU, ~5 min), and exports HTML.

Phase 2 (tools/complete_part_b.py, written after the new answers have been read)
adds the manual verdicts and the B2/B3 inferences.

Usage (project root, notebook saved and its tab closed, GPU server):
    KERNEL=python_llm python tools/rerun_part_b.py
"""
import csv
import json
import os
import re
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
TAG = "partb-rerun"
EVAL = config.EVAL_DIR
BACKUP = config.OUTPUTS_DIR / "backups" / "partB_before_revision"


# ----------------------------------------------------------------------------- helpers
def cells_with(nb, marker, kind=None):
    return [c for c in nb.cells if marker in c.source and (kind is None or c.cell_type == kind)]


def one(nb, marker, kind=None):
    hits = cells_with(nb, marker, kind)
    if len(hits) != 1:
        sys.exit(f"Expected one {kind or ''} cell containing {marker!r}; found {len(hits)}.")
    return hits[0]


def page_counts():
    rows = list(csv.DictReader((config.REPORT_DIR / "extraction_report.csv").open()))
    kept = {p.stem for p in config.DOMAIN_CORPUS_DIR.glob("*.txt")}
    total = sum(int(r["total_pages"]) for r in rows)
    retained = sum(int(r["total_pages"]) for r in rows if Path(r["file_name"]).stem in kept)
    return len(rows), total, retained, len(kept)


# ----------------------------------------------------------------------------- new cells
RELOAD = r'''# Part B was re-run after the instruction-dataset revision (Part A cells above were NOT re-executed).
# This cell reloads the saved Part A results that Part B and the final summary use.
import os, sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
from src import config
pd.set_option("display.max_colwidth", None)
summary     = json.loads((config.REPORT_DIR / "corpus_summary.json").read_text())
tok_stats   = json.loads((config.REPORT_DIR / "tokenization_stats.json").read_text())
cpt_summary = json.loads((config.EVAL_DIR / "cpt_training_summary.json").read_text())
loss_stats  = json.loads((config.EVAL_DIR / "cpt_loss_stats.json").read_text())
ppl         = json.loads((config.EVAL_DIR / "perplexity.json").read_text())
forgetting  = pd.read_csv(config.EVAL_DIR / "forgetting_check.csv")
print(f"Reloaded Part A results: {summary['documents_after']} documents, CPT loss "
      f"{cpt_summary['start_train_loss']:.4f} -> {loss_stats['last_logged_loss']:.4f}, "
      f"held-out PPL {ppl['base_ppl']:.3f} -> {ppl['cpt_ppl']:.3f}")
'''

B1_CODE = r'''import io, re, contextlib
from src.instruction_data import build_instruction_dataset
# validate -> check the seeded 15-pair audit -> check >= 300 pages -> grouped split -> publish
with contextlib.redirect_stdout(io.StringIO()):   # full report: data/instruction_revision/quality_report.json
    inst_stats = build_instruction_dataset("reviewed")
inst = pd.read_json(config.INSTRUCTION_DIR / "instruction_dataset.jsonl", lines=True)
corpus = {p.name: p.read_text(encoding="utf-8") for p in config.DOMAIN_CORPUS_DIR.glob("*.txt")}
words = inst.response.str.split().str.len()
sents = inst.response.apply(lambda t: len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", t.strip())))
fam = inst.template_family.value_counts()
opening = inst.instruction.str.lower().str.split().str[:2].str.join(" ").value_counts()

def toks(s): return re.findall(r"[a-z0-9]+", s.lower())
def longest_copy(resp, src):
    """Longest run of consecutive response words that also occurs in the source document."""
    doc, w, best = " " + " ".join(toks(src)) + " ", toks(resp), 0
    for i in range(len(w)):
        j = i + best + 1
        while j <= len(w) and f" {' '.join(w[i:j])} " in doc:
            best, j = j - i, j + 1
    return best
copied = pd.Series([longest_copy(r.response, corpus[r.source]) for r in inst.itertuples()])

print(f"PDF pages in the retained corpus: {inst_stats['pages_in_retained_corpus']} (minimum 300)")
print(f"Pairs: {inst_stats['total_pairs']} = train {inst_stats['train_pairs']} / eval {inst_stats['eval_pairs']}, "
      f"from {inst.source.nunique()} of {len(corpus)} documents")
print(f"Response length: {words.min()}-{words.max()} words (median {int(words.median())}); "
      f"responses < 30 words: {int((words < 30).sum())}; sentences per response: {sents.min()}-{sents.max()}")
print(f"Template families: {len(fam)}; largest '{fam.index[0]}' = {100 * fam.iloc[0] / len(inst):.1f}% | "
      f"two-word question openings: {len(opening)}; largest '{opening.index[0]}' = {100 * opening.iloc[0] / len(inst):.1f}% "
      f"(flag threshold 20%)")
print("Flagged (>20%): templates", inst_stats["templates_over_20_percent"] or "none",
      "| openings", inst_stats["question_openings_over_20_percent"] or "none")
print(f"Longest word sequence copied from the source document: median {int(copied.median())}, max {copied.max()} words")
display(inst.question_type.value_counts().to_frame("pairs"))
display(fam.to_frame("pairs").assign(share_percent=lambda d: (100 * d.pairs / len(inst)).round(1)))
spot = json.loads((config.DATA_DIR / "instruction_revision" / "independent_spotcheck_claude.json").read_text())
check = pd.DataFrame(spot["rows"]).merge(inst[["id", "instruction", "response"]], on="id")
print(f"Independent spot-check ({spot['reviewer']}): {len(check)} random pairs, "
      f"{check.verdict.str.startswith('pass').sum()} pass")
display(check[["n", "source", "question_type", "instruction", "response", "verdict", "note"]])
'''

B3B_PENDING_MD = r"""#### B3b — full answers for manual review
The automatic keyword check only tests whether a keyword occurs anywhere. Every answer below is read in full; the manual verdicts are added after reading them (PENDING)."""

B3B_PENDING = r'''import os, sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import pandas as pd
from src import config
dom = pd.read_csv(config.EVAL_DIR / "sft_adapter_B_domain_prompts.csv")
gen = pd.read_csv(config.EVAL_DIR / "sft_adapter_B_general_prompts.csv")
answer_col = [c for c in dom.columns if c.startswith("sft_adapter")][0]
review = pd.concat([
    pd.DataFrame({"set": "MyGov probe question", "question": dom["question"], "full adapter answer": dom[answer_col],
                  "keyword check": dom["sft_has_fact"]}),
    pd.DataFrame({"set": "general question", "question": gen["prompt"], "full adapter answer": gen["sft_output"],
                  "keyword check": gen["correct"]})], ignore_index=True)
review.to_csv(config.EVAL_DIR / "sft_adapter_B_answers_for_review.csv", index=False)
display(review)
print("Manual verdicts: PENDING")
'''

SUMMARY_MD = r"""## Summary
| Metric | Result |
|---|---|
| Corpus after cleaning / PDF pages | see Step 1 and B1 |
| Tokens / packed 2048-blocks | see Step 2 |
| CPT loss (start → end) | see Step 4 and the revised loss-curve reading |
| Domain PPL base → CPT | see Step 5A |
| Forgetting verdicts | see Step 5B |
| Instruction pairs train / eval | see B1 |
| Adapter B ROUGE-L (no adapter → adapter) | see B3 |
| SFT manual correctness | see B3b |"""

SUMMARY_CODE = r'''_mr_path = config.EVAL_DIR / "sft_adapter_B_manual_review.csv"
_c = (pd.read_csv(_mr_path).groupby("set", sort=False)["manual verdict"]
      .apply(lambda s: f"{int((s == 'Correct').sum())}/{len(s)}") if _mr_path.exists() else None)
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
  "ROUGE-L no adapter → Adapter B": f"{sft_eval['rougeL_cpt_no_adapter']:.3f} → {sft_eval['rougeL_sft_adapter']:.3f}",
  "SFT manual correctness (B3b)": (f"domain {_c['MyGov probe question']}; general {_c['general question']}"
                                   if _c is not None else "pending manual review"),
}
display(pd.Series(rows).to_frame("result"))
'''

PENDING = "*PENDING — to be written from the re-run results (phase 2).*"


def part_b_md(prompt, n_pdfs, pages, retained, n_docs):
    return rf"""# PART B — Instruction Fine-Tuning with QLoRA

## B1 — Instruction dataset creation  [2 marks]
**Revision.** The first version of this dataset (1,852 template-generated pairs whose responses were copied verbatim from the policies) did not meet the dataset rules: answers must be concise and in the writer's own words, instruction phrasings must not cycle through a few templates, and every response must have ≥ 30 words. It was replaced by the reviewed dataset below, and **Adapter B was retrained on it**. Part A (CPT) is unchanged; its outputs above come from the original run.

**Source:** {n_pdfs} public HR-policy PDFs = **{pages} pages** ({retained} pages in the {n_docs} documents kept after de-duplication), above the 300-page minimum.

**Method — evidence-grounded paraphrases written by an LLM assistant.** For every retained document, policy sections were given to an LLM assistant (Codex), which wrote the pairs following the prompt below. The prompt is saved in `data/instruction_revision/generation_prompt.txt`, with provenance in `generation_provenance.json`. It contains the required instruction *"Each response must be a concise 1–3 sentence answer in your own words. Do not copy text verbatim."* Each pair stores the `evidence_quote` it is based on, its `question_type` (factual / procedural / comparative) and its `template_family` (the reusable phrasing pattern).
```
{prompt.strip()}
```
**Automatic quality filters** (`src/instruction_quality.py`). A pair is rejected if:
* the response has fewer than 30 or more than 90 words, or is not 1–3 sentences;
* the instruction is not a question;
* the evidence quote is not found in the source text;
* the response occurs verbatim in the source;
* the question or answer duplicates another pair;
* it contains contents-page or form noise.

The report also flags any template family or two-word question opening used by more than 20 % of the pairs, and checks that every source document and all three question types are covered.

**Spot-checks** (neither is a human review):
1. The seeded 15-pair audit by the generating assistant (`data/instruction_revision/random_audit.json`).
2. An independent 15-pair random spot-check against the evidence (seed 2026, shown below).

**Split:** by source section group, so pairs written from the same evidence go to the same split. Sections of the probe documents (Leave, Travel, Exit) go to train, so B3 tests recall of trained facts."""


B1_INFER = r"""**Inferences — B1**
* **Requirements met.**
  * Pages: 902 PDF pages, against a 300-page minimum.
  * Size and coverage: 66 pairs (minimum 40–50) covering all 33 documents.
  * Question types: 30 factual, 19 procedural and 17 comparative.
  * Response length: every response is 34–44 words in two sentences (rules: at least 30 words, 1–3 sentences).
  * Variety: no template family exceeds 12.1 % and no two-word question opening exceeds 13.6 % (threshold 20 %).
* **Own words.** No response appears verbatim in its source. The longest word sequence any response shares with its source is short (median 4, max 10 words), against whole clauses copied in the first version.
* **Spot-check.** All 15 random pairs are grounded in their evidence and keep the numbers and conditions, for example the 3-day preliminary enquiry and a quorum of three that must include a woman member. Two weaknesses:
  * One answer (#13, Niramai) ends with a clause not stated in the source, padding to reach the 30-word minimum.
  * Two pairs from the same document can share one excerpt and overlap (#4/#6, #1/#5).
* **Trade-off.** The dataset is about 28× smaller than the first version: 66 vs 1,852 pairs, of which 54 are for training. It can teach answer style and selected facts, but not the whole corpus. The eval split has only 12 pairs from 6 documents, so held-out scores are noisy.
* **Limitations.**
  * The pairs were written by an LLM assistant and spot-checked by an LLM, not by a human expert.
  * The 30-word minimum sometimes forces padding.
  * The exit-notice period (60/90 days) is not in any training pair, so the B3 notice probe tests a fact the SFT data does not teach."""

B2_MD = r"""## B2 — QLoRA fine-tuning with **Adapter B (Balanced)**  [2 marks]
| Setting | Value |
|---|---|
| Base weights | CPT checkpoint from Step 4, **4-bit NF4**, double quantisation, compute dtype bf16 (L40S/A100) or fp16 (T4) |
| Adapter | **B — r = 16, α = 32, target `q_proj`, `v_proj`**, dropout 0.05 |
| Trainer | TRL `SFTTrainer`, prompt/completion conversations rendered with the chat template; loss on the assistant completion only |
| Optimiser / LR | paged 8-bit AdamW, 2e-4, cosine, 5 % warm-up, **5 epochs, batch 4 × grad-accum 1** (≈ 70 optimiser steps), max length 512 |

**Why these settings for 54 training pairs?** The original setting (2 epochs, batch 4 × grad-accum 4) would give only 7 optimiser steps on 54 pairs. Using grad-accum 1 and 5 epochs gives about 70 steps. Held-out loss is logged once per epoch so that over-fitting would be visible.

**Why Adapter B?**
* Adapter A (r = 8) risks under-fitting multi-clause answers.
* Adapter C (r = 32 + `o_proj`) roughly doubles the adapter parameters, and on a 54-pair dataset more capacity mostly raises the over-fitting risk.

B is the balance point between quality and cost. α/r = 2 keeps the effective LoRA scale the same as the common r = 8 / α = 16 setting.

**Chat template:** the TinyLlama *base* tokenizer has no chat template. The official TinyLlama-1.1B-Chat-v1.0 (Zephyr-style) template — `<|system|> … </s> <|user|> … </s> <|assistant|> …` — is therefore attached to the same tokenizer. This needs no vocabulary change: the markers are ordinary text tokens."""


# ----------------------------------------------------------------------------- main
def main():
    import torch
    if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU_TEST") != "1":
        sys.exit("No GPU visible: retraining Adapter B needs the 4-bit GPU path. Start a GPU server.")
    from src.instruction_quality import verify_published
    verify_published()
    if not (config.CPT_MODEL_DIR / "config.json").exists():
        sys.exit(f"CPT checkpoint missing in {config.CPT_MODEL_DIR}.")
    if not (config.DATA_DIR / "instruction_revision" / "independent_spotcheck_claude.json").exists():
        sys.exit("data/instruction_revision/independent_spotcheck_claude.json is missing - copy it to the lab.")

    nb = nbformat.read(NB, as_version=4)
    if not cells_with(nb, "training_summary = json.load", "code"):
        print("Applying tools/polish_notebook.py first ...")
        subprocess.run([sys.executable, str(ROOT / "tools" / "polish_notebook.py")], check=True)
        nb = nbformat.read(NB, as_version=4)

    # ---- backups (once)
    BACKUP.mkdir(parents=True, exist_ok=True)
    if not (BACKUP / NB.name).exists():
        shutil.copy2(NB, BACKUP / NB.name)
        shutil.copy2(NB.with_suffix(".html"), BACKUP / NB.with_suffix(".html").name)
        for f in EVAL.glob("sft_adapter_B_*"):
            shutil.copy2(f, BACKUP / f.name)
        adapter = config.ADAPTER_DIR / "adapter_B"
        if adapter.exists():
            shutil.copytree(adapter, BACKUP / "adapter_B", dirs_exist_ok=True)
        print("Backed up notebook, HTML, old Adapter B and its results to", BACKUP)
    stale = EVAL / "sft_adapter_B_manual_review.csv"
    if stale.exists():
        shutil.move(stale, BACKUP / stale.name)  # verdicts of the OLD adapter must not be reused

    # ---- Step 1 inference: page count
    n_pdfs, pages, retained, n_docs = page_counts()
    md = one(nb, "**Inferences — Step 1**", "markdown")
    if "300-page minimum" not in md.source:
        md.source = md.source.rstrip() + (f"\n* **PDF pages:** {n_pdfs} PDFs = **{pages} pages** ({retained} pages in the "
                                          f"{n_docs} documents kept after de-duplication), above the 300-page minimum.")

    # ---- Part B text and code
    prompt = (config.DATA_DIR / "instruction_revision" / "generation_prompt.txt").read_text(encoding="utf-8")
    one(nb, "# PART B", "markdown").source = part_b_md(prompt, n_pdfs, pages, retained, n_docs)
    one(nb, "build_instruction_dataset(", "code").source = B1_CODE.strip()
    one(nb, "**Inferences — B1**", "markdown").source = B1_INFER
    one(nb, "## B2 — QLoRA fine-tuning", "markdown").source = B2_MD
    b2 = one(nb, 'train_qlora("B"', "code")
    if "SFT_HP" not in b2.source:
        b2.source = b2.source.replace(
            'sft_summary, sft_rows = train_qlora("B")',
            '# 54 training pairs: grad-accum 1 and 5 epochs give ~70 optimiser steps (2 epochs x 4x4 would give 7)\n'
            'SFT_HP = {"num_train_epochs": 5, "gradient_accumulation_steps": 1, "logging_steps": 2}\n'
            'sft_summary, sft_rows = train_qlora("B", hparams=SFT_HP)')
    one(nb, "**Inferences — B2**", "markdown").source = "**Inferences — B2**\n" + PENDING
    one(nb, "**Observations — B3**", "markdown").source = "**Observations — B3**\n" + PENDING
    one(nb, "#### B3b", "markdown").source = B3B_PENDING_MD
    one(nb, '"full adapter answer"', "code").source = B3B_PENDING.strip()
    one(nb, "## Summary", "markdown").source = SUMMARY_MD
    kw = one(nb, "Automatic keyword counts", "code")
    kw.source = kw.source.replace("(B3b below)", "(B3b)")
    summary_cell = one(nb, '"ROUGE-L no adapter → Adapter B"', "code")
    summary_cell.source = SUMMARY_CODE.strip()

    # ---- reload cell right after the Part B heading
    nb.cells = [c for c in nb.cells if TAG not in c.metadata.get("tags", [])]
    head = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.startswith("# PART B"))
    reload_cell = nbformat.v4.new_code_cell(RELOAD.strip())
    reload_cell.metadata["tags"] = [TAG]
    nb.cells.insert(head + 1, reload_cell)

    # ---- execute Part B and the summary (not the export cell)
    start = head + 1
    stop = next(i for i, c in enumerate(nb.cells) if c is summary_cell)
    run_idx = [i for i in range(start, stop + 1) if nb.cells[i].cell_type == "code"]
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
    os.environ["PYTHONNOUSERSITE"] = "1"
    client = NotebookClient(nb, kernel_name=KERNEL, timeout=3600,
                            resources={"metadata": {"path": str(NB.parent)}})
    count = max((c.get("execution_count") or 0) for c in nb.cells if c.cell_type == "code") + 1
    try:
        with client.setup_kernel():
            for i in run_idx:
                print(f"Executing cell {i}: {nb.cells[i].source.splitlines()[0][:70]}")
                client.execute_cell(nb.cells[i], i)
                nb.cells[i].execution_count = count
                count += 1
                for out in nb.cells[i].outputs:
                    if out.get("output_type") == "stream":
                        print(out.get("text", "")[-1200:])
    except CellExecutionError as err:
        failed = NB.with_name("_partb_failed.ipynb")
        nbformat.write(nb, failed)
        sys.exit(f"A cell failed - the submission notebook was NOT changed. Partial run saved to {failed}.\n{err}")

    nbformat.write(nb, NB)
    shutil.copy2(config.INSTRUCTION_DIR / "instruction_dataset.jsonl", ROOT / "instruction_dataset.jsonl")
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    extra = [str(Path(p) / "share" / "jupyter") for p in {sys.base_prefix, "/opt/conda"}]
    env["JUPYTER_PATH"] = os.pathsep.join([env.get("JUPYTER_PATH", "")] + extra).strip(os.pathsep)
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", str(NB), "--output-dir", str(NB.parent)],
                   check=True, env=env)
    print("\nPart B re-run complete. Send the new HTML (or outputs/evaluations/sft_adapter_B_answers_for_review.csv"
          " + sft_adapter_B_same_prompts.csv + sft_adapter_B_summary.json) for the manual review and phase 2.")


if __name__ == "__main__":
    main()
