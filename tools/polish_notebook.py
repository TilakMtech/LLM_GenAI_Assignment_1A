"""Presentation clean-up of the finalised notebook - NO training, NO kernel.

Run AFTER tools/finalize_notebook.py. Edits code text and the matching saved
outputs together (values are recomputed from the saved result files, so the
edited outputs are exactly what the edited code prints), then re-exports HTML.

  1. Revised loss-curve cell: `summary` -> `training_summary`, so the corpus
     `summary` used by the final Summary cell is no longer overwritten.
  2. Original CPT loss statement: 27.15 % endpoint decrease; 32.55 % = window means.
  3. Original loss graph: labelled "Superseded by the revised loss analysis below".
  4. Epoch-statistics line: "held-out plateau ~ step 15" -> held-out minimum (step 20).
  5. Original B3 keyword counts: labelled as automatic keyword counts, superseded
     by the manual review (B3b).
  6. Final summary: adds "SFT manual correctness: domain 0/3; general 1/3".

Usage (project root; notebook saved and its tab closed):
    python tools/polish_notebook.py
Running it twice is safe: every edit checks whether it is already applied.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "CorpPolicyLM_Assignment1A.ipynb"
EVAL = ROOT / "outputs" / "evaluations"
DONE = []


def find_cell(nb, marker, kind="code"):
    hits = [c for c in nb.cells if c.cell_type == kind and marker in c.source]
    if len(hits) != 1:
        sys.exit(f"Expected exactly one {kind} cell containing {marker!r}, found {len(hits)}.")
    return hits[0]


def stream_outputs(cell):
    return [o for o in cell.outputs if o.get("output_type") == "stream"]


def main():
    nb = nbformat.read(NB, as_version=4)

    # ---------------------------------------------------------------- values
    train_sum = json.loads((EVAL / "cpt_training_summary.json").read_text())
    loss_stats = json.loads((EVAL / "cpt_loss_stats.json").read_text())
    log = pd.read_csv(EVAL / "cpt_loss_log.csv")
    ev = log.dropna(subset=["eval_loss"])
    start, end = train_sum["start_train_loss"], loss_stats["last_logged_loss"]
    endpoint = 100 * (1 - end / start)
    ev_min, ev_min_step = ev["eval_loss"].min(), int(ev.loc[ev["eval_loss"].idxmin(), "step"])
    review = pd.read_csv(EVAL / "sft_adapter_B_manual_review.csv")
    counts = review.groupby("set", sort=False)["manual verdict"].apply(
        lambda s: f"{int((s == 'Correct').sum())}/{len(s)}")
    manual = f"domain {counts['MyGov probe question']}; general {counts['general question']}"

    # ---------------------------------------------------------------- 1. variable overwrite
    cell = find_cell(nb, 'EVAL / "cpt_training_summary.json"')
    if "training_summary = json.load" not in cell.source:
        cell.source = re.sub(r"\bsummary\b(?=\s*=\s*json\.load|\[)", "training_summary", cell.source)
        DONE.append("1. revised loss-curve cell: summary -> training_summary")
    assert not re.search(r"(?<!training_)\bsummary\[", cell.source), "summary[...] left in revised cell"

    # ---------------------------------------------------------------- 2 + 3. original plot cell
    cell = find_cell(nb, "loss_stats, png = plot_loss_curve()")
    if "Superseded by the revised loss analysis below" not in cell.source:
        old_print = cell.source[cell.source.index('print(f"Starting loss (pretrained'):]
        new_print = (
            "endpoint = 100 * (1 - loss_stats['last_logged_loss'] / cpt_summary['start_train_loss'])\n"
            "print(f\"CPT loss {cpt_summary['start_train_loss']:.4f} → {loss_stats['last_logged_loss']:.4f}: \"\n"
            "      f\"{endpoint:.2f}% endpoint decrease. The {loss_stats['loss_drop_percent']}% \"\n"
            "      f\"('loss_drop_percent' above) refers to window means (first vs last logged steps).\")")
        cell.source = cell.source.replace(old_print, new_print)
        label = ("**Superseded by the revised loss analysis below.** The automatic “train plateau” "
                 "marker in this first plot is misleading; the revised loss-curve reading marks the "
                 "held-out minimum instead.")
        cell.source = cell.source.replace(
            "display(Image(str(png)))",
            "from IPython.display import Markdown\n"
            "display(Markdown(\"**Superseded by the revised loss analysis below.** The automatic “train plateau” \"\n"
            "                 \"marker in this first plot is misleading; the revised loss-curve reading marks the \"\n"
            "                 \"held-out minimum instead.\"))\n"
            "display(Image(str(png)))")
        img = next(i for i, o in enumerate(cell.outputs) if "image/png" in o.get("data", {}))
        cell.outputs.insert(img, nbformat.v4.new_output("display_data", data={
            "text/markdown": label, "text/plain": "<IPython.core.display.Markdown object>"}))
        new_text = (f"CPT loss {start:.4f} → {end:.4f}: {endpoint:.2f}% endpoint decrease. "
                    f"The {loss_stats['loss_drop_percent']}% ('loss_drop_percent' above) refers to window "
                    "means (first vs last logged steps).\n")
        out = next(o for o in stream_outputs(cell) if "Starting loss (pretrained" in o["text"])
        out["text"] = re.sub(r"Starting loss \(pretrained.*?\n", new_text, out["text"], flags=re.S)
        DONE.append("2. CPT loss statement: endpoint vs window-mean")
        DONE.append("3. original loss graph labelled 'Superseded'")

    # ---------------------------------------------------------------- 4. epoch-statistics cell
    cell = find_cell(nb, "held-out plateau ≈ step {loss_stats['eval_plateau_step']}", "code") \
        if any("held-out plateau ≈ step {loss_stats" in c.source for c in nb.cells if c.cell_type == "code") \
        else None
    if cell is not None:
        cell.source = cell.source.replace(
            "held-out plateau ≈ step {loss_stats['eval_plateau_step']}",
            "held-out minimum {ev.eval_loss.min():.4f} at step {int(ev.loc[ev.eval_loss.idxmin(), 'step'])}")
        for o in stream_outputs(cell):
            o["text"] = re.sub(r"held-out plateau ≈ step \S+",
                               f"held-out minimum {ev_min:.4f} at step {ev_min_step}", o["text"])
        DONE.append("4. epoch statistics: held-out plateau -> held-out minimum")

    # ---------------------------------------------------------------- 5. B3 keyword counts
    label = "Automatic keyword counts — not factual correctness; superseded by manual review (B3b below)."
    cell = find_cell(nb, "Held-out ROUGE-L ({sft_eval['heldout_pairs_scored']} pairs)")
    if label not in cell.source:
        cell.source = f'print("{label}")\n' + cell.source
        cell.source = (cell.source
                       .replace("* Domain probes stating the reference fact", "* Domain probes containing a reference keyword")
                       .replace("* General prompts still correct after SFT", "* General prompts containing an expected keyword after SFT"))
        out = stream_outputs(cell)[0]
        out["text"] = label + "\n" + (out["text"]
                                      .replace("* Domain probes stating the reference fact", "* Domain probes containing a reference keyword")
                                      .replace("* General prompts still correct after SFT", "* General prompts containing an expected keyword after SFT"))
        DONE.append("5. B3 keyword counts labelled")
    md = find_cell(nb, "## B3 — Evaluation", "markdown")
    note = ("\n\n**Note:** the `sft_has_fact` and `correct` columns and the counts printed below are "
            "*automatic keyword matches*, not factual correctness; they are superseded by the manual review in B3b.")
    if "automatic keyword matches" not in md.source:
        md.source += note

    # ---------------------------------------------------------------- 6. final summary
    cell = find_cell(nb, '"ROUGE-L no adapter → Adapter B"')
    key = "SFT manual correctness (B3b)"
    if key not in cell.source:
        cell.source = cell.source.replace(
            "rows = {",
            "_mr = pd.read_csv(config.EVAL_DIR / \"sft_adapter_B_manual_review.csv\")\n"
            "_c = _mr.groupby(\"set\", sort=False)[\"manual verdict\"].apply(lambda s: f\"{int((s == 'Correct').sum())}/{len(s)}\")\n"
            "rows = {", 1)
        new_row = ('  "SFT manual correctness (B3b)": '
                   'f"domain {_c[\'MyGov probe question\']}; general {_c[\'general question\']}",\n')
        cell.source = re.sub(r'(\n\s*"ROUGE-L no adapter → Adapter B":[^\n]*\n)',
                             lambda m: m.group(1) + new_row, cell.source)
        disp = next(o for o in cell.outputs if o.get("output_type") in ("display_data", "execute_result")
                    and "text/html" in o.get("data", {}))
        html = disp["data"]["text/html"]
        row = f"    <tr>\n      <th>{key}</th>\n      <td>{manual}</td>\n    </tr>\n  </tbody>"
        disp["data"]["text/html"] = html.replace("  </tbody>", row, 1) if "  </tbody>" in html \
            else html.replace("</tbody>", row, 1)
        if "text/plain" in disp["data"]:
            disp["data"]["text/plain"] = disp["data"]["text/plain"].rstrip("\n") + f"\n{key}  {manual}"
        DONE.append(f"6. final summary: {key} = {manual}")
    md = find_cell(nb, "| Adapter B ROUGE-L (no adapter → adapter) |", "markdown")
    if "SFT manual correctness" not in md.source:
        md.source = md.source.rstrip() + f"\n| SFT manual correctness | {manual} (B3b) |"

    nbformat.write(nb, NB)
    print("Edits applied:" if DONE else "Nothing to change - all edits already applied.")
    for d in DONE:
        print("  ", d)

    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    extra = [str(Path(p) / "share" / "jupyter") for p in {sys.base_prefix, "/opt/conda"}]
    env["JUPYTER_PATH"] = os.pathsep.join([env.get("JUPYTER_PATH", "")] + extra).strip(os.pathsep)
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", str(NB),
                    "--output-dir", str(NB.parent)], check=True, env=env)
    print("Wrote", NB.with_suffix(".html"))


if __name__ == "__main__":
    main()
