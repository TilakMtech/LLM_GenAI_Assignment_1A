#!/usr/bin/env bash
# Run the whole Assignment 1A notebook headless, save every output into the
# .ipynb, then export the HTML deliverable. From the project root:
#     bash run_notebook.sh
# (Use this instead of "Run All": nbconvert converts the .ipynb on disk, which
#  during an interactive run holds only the last autosave.)
set -euo pipefail
cd "$(dirname "$0")"
NB=notebooks/CorpPolicyLM_Assignment1A.ipynb
mkdir -p outputs/logs
LOG=outputs/logs/run_notebook_$(date +%Y%m%d_%H%M%S).log
PY=$(command -v python)

echo "Python: $PY" | tee "$LOG"
echo "1/4 Installing requirements (before any kernel starts) ..." | tee -a "$LOG"
"$PY" -m pip install -q -r requirements.txt 2>&1 | tee -a "$LOG"

echo "2/4 Checking the environment ..." | tee -a "$LOG"
"$PY" - <<'PYCHECK' 2>&1 | tee -a "$LOG"
import sys
try:
    import numpy, scipy, scipy.special, sklearn, pandas, pyarrow, torch, transformers, trl, peft
    from transformers import AutoModelForCausalLM  # pulls in generation -> sklearn/scipy
    pandas.DataFrame([{"a": 1, "b": "x"}])          # fails fast on numpy/pandas binary mismatch
except Exception as error:
    print(f"\nENVIRONMENT CHECK FAILED: {type(error).__name__}: {error}\n")
    print("numpy/scipy/scikit-learn/pandas were built against different numpy versions.")
    print("Repair them as one consistent set, then re-run this script:")
    print(f"    {sys.executable} -m pip install --upgrade --force-reinstall numpy scipy scikit-learn pandas")
    sys.exit(1)
print(f"numpy {numpy.__version__} | scipy {scipy.__version__} | sklearn {sklearn.__version__} | "
      f"pandas {pandas.__version__} | pyarrow {pyarrow.__version__} | torch {torch.__version__} "
      f"(CUDA {torch.cuda.is_available()}) | transformers {transformers.__version__} | "
      f"trl {trl.__version__} | peft {peft.__version__}")
PYCHECK

echo "3/4 Executing $NB (CPT + QLoRA; ~10-15 min on an L40S/A100) ..." | tee -a "$LOG"
NOTEBOOK_HEADLESS=1 jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$NB" 2>&1 | tee -a "$LOG"

echo "4/4 Exporting HTML ..." | tee -a "$LOG"
jupyter nbconvert --to html "$NB" --output-dir notebooks 2>&1 | tee -a "$LOG"
cp data/instruction/instruction_dataset.jsonl instruction_dataset.jsonl
echo "Done: $NB (with outputs), notebooks/CorpPolicyLM_Assignment1A.html, instruction_dataset.jsonl" | tee -a "$LOG"
