"""Team-review text fixes for the submitted Assignment 1A notebook (markdown cells only; no code
cell or output is changed, nothing is re-run). Re-exports the HTML from the saved notebook.

    python tools/review_fixes_1a.py

Fixes:
  1. Step 1a table: lists the 33 documents actually in the corpus; names the 12 organisations whose
     17 URLs failed to download.
  2. Step 1c: near-duplicate example no longer names a manual that is not in the corpus.
  3. Step 4: states clearly that the final, not the best, CPT checkpoint was kept.
  4. Step 5: note that the held-out PPL and general prompts were also used to choose the learning rate.
  5. Step 5 inferences: manual verdict table for 5B (Retained / Degraded only).
  6. B1: exact train/eval numbers and why exactly 80/20 is not possible; names the split files.
Second review round:
  7. B1 inferences: spot-check no longer claims all 15 pairs are grounded (one Niramai clause is unsupported).
  8. B1 split: the exit-notice probe tests a fact absent from the SFT data, not a trained fact.
Third review round (audit correction; dataset and adapter unchanged):
  9. data/instruction_revision/independent_spotcheck_claude.json: pair #13 (Niramai) is reclassified from
     "pass (minor ungrounded padding)" to "needs revision" -> 14 pass, 1 needs revision.
 10. B1: a new code cell after the original audit output shows the corrected table and counts
     (executed here, CPU only), and the B1 inference states the corrected result.
"""
import json
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "CorpPolicyLM_Assignment1A.ipynb"
DONE = "<!-- review-fixes-1a -->"
DONE2 = "<!-- review-fixes-1a-round2 -->"
DONE3 = "<!-- review-fixes-1a-round3 -->"
AUDIT = ROOT / "data" / "instruction_revision" / "independent_spotcheck_claude.json"
AUDIT_ID = "d28d954cc83933d3"   # pair #13, Niramai
CORRECTION = ("Audit correction after team review (8 Oct 2026): pair #13 (Niramai) was reclassified from "
              "'pass (minor ungrounded padding)' to 'needs revision', because its closing clause about outside "
              "stakeholders is not stated in the source. Corrected result: 14 pass, 1 needs revision. "
              "The pair is still in the dataset; the adapter was not retrained.")
OVERSTATE_OLD = ('The verdict column of the table above still marks it "pass", so the 15/15 count overstates the result.')
OVERSTATE_NEW = ('The audit was corrected after review: this pair is now marked "needs revision", giving '
                 '**14 pass, 1 needs revision** (corrected table in the cell after the original audit output). '
                 'The pair is still in the dataset and the adapter was not retrained.')
CORR_MD = ("#### B1 — audit correction (added after review; reads the corrected audit file, no retraining)\n"
           "The original output above counted pair #13 as a pass, because its verdict was \"pass (minor ungrounded "
           "padding)\". Its closing clause is not supported by the source, so it is now marked **needs revision**.\n" + DONE3)
CORR_CODE = """import json
from pathlib import Path
import pandas as pd
from IPython.display import display
pd.set_option("display.max_colwidth", None)
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
spot = json.loads((ROOT / "data/instruction_revision/independent_spotcheck_claude.json").read_text())
inst = pd.read_json(ROOT / "data/instruction/instruction_dataset.jsonl", lines=True)
check = pd.DataFrame(spot["rows"]).merge(inst[["id", "instruction", "response"]], on="id")
print(spot["correction"])
counts = check.verdict.value_counts()
print(f"Independent spot-check (corrected): {len(check)} random pairs - "
      + ", ".join(f"{v} {k}" for k, v in counts.items()))
display(check[["n", "source", "question_type", "instruction", "response", "verdict", "note"]])"""


def correct_audit():
    data = json.loads(AUDIT.read_text(encoding="utf-8"))
    row = next(r for r in data["rows"] if r["id"] == AUDIT_ID)
    if row["verdict"] != "needs revision":
        row["verdict"] = "needs revision"
        row["note"] += " Reclassified after team review: an unsupported clause means the pair needs revision."
        data["correction"] = CORRECTION
        AUDIT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("Audit file corrected")
    else:
        print("Audit file already corrected")

SPOT_OLD = ("* **Spot-check.** All 15 random pairs are grounded in their evidence and keep the numbers and conditions, "
            "for example the 3-day preliminary enquiry and a quorum of three that must include a woman member. Two weaknesses:\n"
            "  * One answer (#13, Niramai) ends with a clause not stated in the source, padding to reach the 30-word minimum.\n"
            "  * Two pairs from the same document can share one excerpt and overlap (#4/#6, #1/#5).\n")
SPOT_NEW = ("* **Spot-check.** The spot-check found that the sampled answers generally preserve the source facts and "
            "conditions, but identified an unsupported concluding clause in one Niramai answer. Examples of preserved "
            "facts are the 3-day preliminary enquiry and a quorum of three that must include a woman member. Weaknesses:\n"
            "  * The Niramai answer (#13) ends with a clause not stated in the source, padding to reach the 30-word minimum. "
            "The verdict column of the table above still marks it \"pass\", so the 15/15 count overstates the result.\n"
            "  * Two pairs from the same document can share one excerpt and overlap (#4/#6, #1/#5).\n")
PROBE_OLD = "Sections of the probe documents (Leave, Travel, Exit) go to train, so B3 tests recall of trained facts."
PROBE_NEW = ("Probe-document sections are assigned to training. The probation and travel probes test facts represented "
             "in SFT; the exit-notice probe tests a fact absent from the instruction dataset.")

TABLE_1A = """It was expanded with public HR / corporate-policy PDFs listed in `data/sources.csv`. The corpus actually used contains these **33 documents**:

| Category | Documents in the corpus | Count |
|---|---|---|
| MyGov HR policies (starting corpus) | Leave, Attendance & Work Protocols, Exit, Domestic Travel | 4 |
| MyGov policies (downloaded) | Laptop, Talent Development, Internship Programme | 3 |
| Government HR rules | CCS (Leave) Rules 1972, CCS (Conduct) Rules 1964 | 2 |
| Organisation HR manuals | MSRLS, Indus University, DAV University, IIM Raipur, IIHMR, All India Confederation of the Blind (AICB) | 6 |
| POSH policies | 3M India, Bajaj Broking, JioStar, BPTP, Mafatlal Industries | 5 |
| Code of conduct | Accelya, Niramai | 2 |
| Human rights | Home First Finance, EID Parry, Aurobindo Pharma, Oil India | 4 |
| Whistle-blower / vigil mechanism | AU Small Finance Bank, BEML, Astral, GHL, Khanna Paper, DreamFolks, GrowXCD | 7 |
| **Total** | | **33** |

`data/sources.csv` lists 47 URLs. 30 were downloaded. Together with the 4 starting MyGov PDFs, that gives 34 PDFs (902 pages), and de-duplication leaves 33 documents. **17 URLs failed** (HTTP 404/403 or TLS errors; see the log below), so **these 12 organisations are not in the corpus:**
* IIM Udaipur (3 manuals) and Bharat Forge (3 policies);
* IILM, INOXCVA (2 manuals) and IKS Health;
* NIT Tiruchirappalli, Shiv Nadar University, BASF, Sony Pictures Networks, Tata Elxsi, Sudarshan and Infosys.

"""

CKPT_OLD = ("We kept the final (end-of-epoch-2) checkpoint, which is permissible but ≈1 % worse in held-out loss "
            "than the best observed point; early stopping after epoch 1 would have been the better choice.")
CKPT_NEW = ("**Checkpoint used: the final (end-of-epoch-2, step 44) checkpoint, not the best one (step 20).** It was saved "
            "and used for Step 5 and Part B. Its held-out loss is 1.9089, against 1.8897 at step 20 (≈1 % worse). "
            "Saving the step-20 checkpoint (early stopping after epoch 1) would have been the better choice. The lab disk "
            "holds only one checkpoint, so the best one was not kept, and this was not re-run.")

STEP5_NOTE = ("\n\n> **Note on independence.** The held-out PPL set (5A) and the three general-knowledge prompts (5B) were "
              "**also used in Step 4a to choose the learning rate** (rule: lowest held-out PPL among the candidates that "
              "kept all three general facts). The Step 5 figures are therefore selection-set results, not an independent test.")

MANUAL_5B = """**5B — manual verdicts.** Each output was read in full. Only *Retained* (the CPT output still states the fact) and *Degraded* (the base model stated it and the CPT model lost it) are used. The automatic code also has a third label, *Retained (base also wrong)*, but it did not occur here.

| Prompt | Base output (fact) | CPT output (fact) | Automatic verdict | Manual verdict |
|---|---|---|---|---|
| The capital of France is | Paris | Paris (then lists other capitals in a numbered-clause style) | Retained | **Retained** |
| Water boils at | 212 degrees Fahrenheit | 100 degrees Celsius (then drifts into a qualifications table) | Retained | **Retained** |
| The speed of light is approximately | 300,000 km/s | 300,000 km/s (then repeats "1.1.1…") | Retained | **Retained** |

Manual result: **3 Retained, 0 Degraded**, the same as the automatic check. The facts are kept, but two CPT continuations degrade into manual-style numbering or tables after the fact.

"""

SPLIT_OLD = ("**Split:** by source section group, so pairs written from the same evidence go to the same split. Sections of "
             "the probe documents (Leave, Travel, Exit) go to train, so B3 tests recall of trained facts.")
SPLIT_NEW = ("**Split:** by source section group, so pairs written from the same evidence go to the same split. Sections of "
             "the probe documents (Leave, Travel, Exit) go to train, so B3 tests recall of trained facts.\n\n"
             "**Exact split: 54 train / 12 eval pairs = 81.8 % / 18.2 %** (27 / 6 section groups, from 27 / 6 documents). "
             "The split files are `data/instruction/instruction_train.jsonl` and `data/instruction/instruction_eval.jsonl`. "
             "Exactly 80/20 is not possible: 80 % of 66 pairs is 52.8, and each section group holds 2 pairs that are never "
             "split. The nearest options are 54/12 (81.8/18.2) and 52/14 (78.8/21.2). The split rule fills train until it "
             "reaches at least 80 %, which gives 54/12.")


def replace_once(src, old, new, where):
    if src.count(old) != 1:
        sys.exit(f"{where}: expected text found {src.count(old)} times")
    return src.replace(old, new)


def main():
    nb = nbformat.read(NB, as_version=4)
    md = [c for c in nb.cells if c.cell_type == "markdown"]
    if any(DONE in c.source for c in md):
        print("Round-1 fixes already applied.")
    else:
        def cell(marker):
            hits = [c for c in md if marker in c.source]
            if len(hits) != 1:
                sys.exit(f"Expected one markdown cell with '{marker}', found {len(hits)}")
            return hits[0]
        c = cell("### 1a. Data collection")
        a = c.source.index("It was expanded with public HR")
        b = c.source.index("One source (`mygov_travel_policy_dup.pdf`)")
        c.source = c.source[:a] + TABLE_1A + c.source[b:] + "\n" + DONE
        c = cell("### 1c. Cleaning pipeline")
        c.source = replace_once(c.source, "(e.g. two INOXCVA manuals)", "(e.g. two editions of the same manual)", "1c")
        c = cell("**Inferences — Step 4**")
        c.source = replace_once(c.source, CKPT_OLD, CKPT_NEW, "Step 4")
        c = cell("## Step 5 — Evaluation")
        c.source = replace_once(c.source, "for the base and the CPT model.", "for the base and the CPT model." + STEP5_NOTE, "Step 5")
        c = cell("**Inferences — Step 5**")
        c.source = replace_once(c.source, "**Inferences — Step 5**\n", "**Inferences — Step 5**\n\n" + MANUAL_5B, "5B")
        c = cell("## B1 — Instruction dataset creation")
        c.source = replace_once(c.source, SPLIT_OLD, SPLIT_NEW, "B1")
        print("Review fixes (round 1) applied")
    md = [c for c in nb.cells if c.cell_type == "markdown"]
    if any(DONE2 in c.source for c in md):
        print("Round-2 fixes already applied.")
    else:
        hits = [c for c in md if SPOT_OLD in c.source]
        if len(hits) != 1:
            sys.exit(f"B1 spot-check text found in {len(hits)} cells")
        hits[0].source = hits[0].source.replace(SPOT_OLD, SPOT_NEW) + "\n" + DONE2
        hits = [c for c in md if PROBE_OLD in c.source]
        if len(hits) != 1:
            sys.exit(f"B1 probe sentence found in {len(hits)} cells")
        hits[0].source = hits[0].source.replace(PROBE_OLD, PROBE_NEW)
        print("Review fixes (round 2) applied")
    correct_audit()
    md = [c for c in nb.cells if c.cell_type == "markdown"]
    if any(DONE3 in c.source for c in md):
        print("Round-3 fixes already applied.")
    else:
        idx = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "independent_spotcheck_claude.json" in c.source]
        if len(idx) != 1:
            sys.exit(f"Original audit cell found {len(idx)} times")
        code = nbformat.v4.new_code_cell(CORR_CODE)
        run = nbformat.v4.new_notebook(cells=[code])
        NotebookClient(run, kernel_name="python3", timeout=120,
                       resources={"metadata": {"path": str(NB.parent)}}).execute()
        code.execution_count = None                      # added after the original run
        nb.cells[idx[0] + 1:idx[0] + 1] = [nbformat.v4.new_markdown_cell(CORR_MD), code]
        hits = [c for c in nb.cells if c.cell_type == "markdown" and OVERSTATE_OLD in c.source]
        if len(hits) != 1:
            sys.exit(f"B1 overstatement sentence found in {len(hits)} cells")
        hits[0].source = hits[0].source.replace(OVERSTATE_OLD, OVERSTATE_NEW)
        if "14 pass" not in "".join(o.get("text", "") for o in code.outputs):
            sys.exit("Corrected audit cell did not report 14 pass")
        print("Review fixes (round 3) applied")
    nbformat.write(nb, NB)
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", str(NB), "--output-dir", str(NB.parent)], check=True)


if __name__ == "__main__":
    main()
