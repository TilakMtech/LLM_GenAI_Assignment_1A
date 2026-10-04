#!/usr/bin/env bash
# Run the whole Assignment 1A notebook headless, save every output into the
# .ipynb, then export the HTML deliverable. Run from the project root:
#     bash run_notebook.sh
# (Use this instead of "Run All" + manual save: nbconvert only sees the file on
#  disk, which during an interactive run holds just the last autosave.)
set -euo pipefail
cd "$(dirname "$0")"
NB=notebooks/CorpPolicyLM_Assignment1A.ipynb
LOG=outputs/logs/run_notebook_$(date +%Y%m%d_%H%M%S).log
mkdir -p outputs/logs

echo "Executing $NB (CPT + QLoRA; ~10-15 min on an L40S/A100) ... log: $LOG"
NOTEBOOK_HEADLESS=1 jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 "$NB" 2>&1 | tee "$LOG"

echo "Exporting HTML ..."
jupyter nbconvert --to html "$NB" --output-dir notebooks 2>&1 | tee -a "$LOG"
cp data/instruction/instruction_dataset.jsonl instruction_dataset.jsonl
echo "Done: notebooks/CorpPolicyLM_Assignment1A.ipynb (with outputs), notebooks/CorpPolicyLM_Assignment1A.html, instruction_dataset.jsonl"
