#!/usr/bin/env bash
# Run the whole Assignment 1A notebook headless, save every output into the
# .ipynb, then export the HTML deliverable. From the project root:
#     bash run_notebook.sh
# Progress is printed cell by cell; the full log is in outputs/logs/.
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
check_env() {
"$PY" - <<'PYCHECK' 2>&1 | tee -a "$LOG"
import sys
try:
    import numpy, scipy, scipy.special, sklearn, pandas, pyarrow, torch, transformers, trl, peft
    from transformers import AutoModelForCausalLM  # pulls in generation -> sklearn/scipy
    pandas.DataFrame([{"a": 1, "b": "x"}])          # fails fast on numpy/pandas binary mismatch
except Exception as error:
    print(f"ENVIRONMENT CHECK FAILED: {type(error).__name__}: {str(error)[:160]}")
    sys.exit(1)
print(f"numpy {numpy.__version__} | scipy {scipy.__version__} | sklearn {sklearn.__version__} | "
      f"pandas {pandas.__version__} | torch {torch.__version__} (CUDA {torch.cuda.is_available()}) | "
      f"transformers {transformers.__version__} | trl {trl.__version__} | peft {peft.__version__}")
import shutil
def gib(n): return f"{n / 2**30:.1f} GiB"
limit = None
for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
    try:
        raw = open(path).read().strip()
        limit = None if raw == "max" or int(raw) > 2**50 else int(raw)
        break
    except Exception:
        pass
avail = int([l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0]) * 1024
print(f"RAM available {gib(avail)} | container RAM limit {gib(limit) if limit else 'none'} | "
      f"free disk {gib(shutil.disk_usage('.').free)}")
PYCHECK
return "${PIPESTATUS[0]}"
}

if ! check_env; then
    # numpy, scipy, scikit-learn and pandas were built against different numpy
    # versions (typically after packages were upgraded inside a running kernel).
    # Reinstall them as one consistent set, keeping the installed numpy version.
    NUMPY_VER=$("$PY" -c "import numpy; print(numpy.__version__)")
    echo "Repairing numpy/scipy/scikit-learn/pandas (numpy $NUMPY_VER) - one-time, ~1-2 min ..." | tee -a "$LOG"
    "$PY" -m pip install -q --force-reinstall "numpy==$NUMPY_VER" scipy scikit-learn pandas 2>&1 | tee -a "$LOG"
    if ! check_env; then
        echo "Environment still broken after repair - paste the lines above to Claude." | tee -a "$LOG"
        exit 1
    fi
fi

echo "3/4 Executing $NB (CPT + QLoRA; ~10-15 min on an L40S/A100) ..." | tee -a "$LOG"
# papermill prints each cell's output live and saves the notebook after every
# cell, so progress is visible and a crash still leaves the finished cells on disk.
"$PY" -m pip install -q papermill 2>&1 | tee -a "$LOG"
TMP_NB=notebooks/.run_in_progress.ipynb
set +e
NOTEBOOK_HEADLESS=1 "$PY" -m papermill "$NB" "$TMP_NB" -k python3 --cwd notebooks \
    --log-output --progress-bar --request-save-on-cell-execute 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
if [ "$status" -ne 0 ] || [ ! -s "$TMP_NB" ]; then
    echo "Notebook run FAILED (exit $status). Finished cells are saved in $TMP_NB; see the log above." | tee -a "$LOG"
    exit 1
fi
mv "$TMP_NB" "$NB"

echo "4/4 Exporting HTML ..." | tee -a "$LOG"
jupyter nbconvert --to html "$NB" --output-dir notebooks 2>&1 | tee -a "$LOG"
cp data/instruction/instruction_dataset.jsonl instruction_dataset.jsonl
echo "Done: $NB (with outputs), notebooks/CorpPolicyLM_Assignment1A.html, instruction_dataset.jsonl" | tee -a "$LOG"
