#!/usr/bin/env bash
# Run the whole Assignment 1A notebook headless, save every output into the
# .ipynb, then export the HTML deliverable. From the project root:
#     bash run_notebook.sh
#
# Kernel: runs in the Jupyter kernel named by $KERNEL (default python_LLM, the
# lab's LLM environment). KERNEL=venv instead builds a project-local virtual
# environment (.venv-lab) that reuses conda's CUDA torch.
# ~/.local is ignored (PYTHONNOUSERSITE=1): on the BITS lab pod /opt/conda is
# read-only, so earlier `pip install`s landed in ~/.local and mixed numpy
# versions with conda's scipy ("All ufuncs must have type numpy.ufunc").
# Progress is printed cell by cell; the full log is in outputs/logs/.
set -euo pipefail
cd "$(dirname "$0")"
NB=notebooks/CorpPolicyLM_Assignment1A.ipynb
mkdir -p outputs/logs
LOG=outputs/logs/run_notebook_$(date +%Y%m%d_%H%M%S).log
BASE_PY=${BASE_PY:-$(command -v python3 || command -v python)}
KERNEL=${KERNEL:-python_LLM}
export PYTHONNOUSERSITE=1   # never import packages from ~/.local (kernel inherits this)

echo "1/4 Preparing kernel '$KERNEL' ..." | tee "$LOG"
if [ "$KERNEL" = "venv" ]; then
    VENV=.venv-lab
    PY=$VENV/bin/python
    [ -x "$PY" ] || "$BASE_PY" -m venv --system-site-packages "$VENV" 2>&1 | tee -a "$LOG"
    "$PY" -m pip install -q --upgrade pip 2>&1 | tee -a "$LOG"
    "$PY" -m pip install -q --ignore-installed numpy scipy scikit-learn pandas pyarrow matplotlib pillow \
        -r requirements.txt papermill ipykernel nbconvert 2>&1 | tee -a "$LOG"
    KERNEL=corppolicylm
    "$PY" -m ipykernel install --sys-prefix --name "$KERNEL" --display-name "CorpPolicyLM (.venv-lab)" \
        2>&1 | tee -a "$LOG"
else
    # Python interpreter behind the named Jupyter kernel.
    PY=$(jupyter kernelspec list --json 2>/dev/null | "$BASE_PY" -c "
import json, sys
specs = json.load(sys.stdin)['kernelspecs']
name = sys.argv[1]
print(specs[name]['spec']['argv'][0] if name in specs else '')" "$KERNEL")
    if [ -z "$PY" ]; then
        echo "Kernel '$KERNEL' not found. Available kernels:" | tee -a "$LOG"
        jupyter kernelspec list 2>&1 | tee -a "$LOG"
        echo "Re-run with one of them, e.g.  KERNEL=<name> bash run_notebook.sh  (or KERNEL=venv)" | tee -a "$LOG"
        exit 1
    fi
    case "$PY" in python|python3) PY=$(command -v "$PY");; esac
    echo "Kernel python: $PY" | tee -a "$LOG"
    # Install only what is missing, and only if that environment is writable
    # (never fall back to ~/.local).
    if ! "$PY" -c "import transformers, trl, peft, bitsandbytes, langdetect, pypdf, papermill" 2>/dev/null; then
        if "$PY" -c "import os, sysconfig, sys; sys.exit(0 if os.access(sysconfig.get_paths()['purelib'], os.W_OK) else 1)"; then
            "$PY" -m pip install -q -r requirements.txt papermill 2>&1 | tee -a "$LOG"
        else
            echo "Kernel '$KERNEL' is missing packages and its environment is read-only." | tee -a "$LOG"
            echo "Use the project venv instead:  KERNEL=venv bash run_notebook.sh" | tee -a "$LOG"
            exit 1
        fi
    fi
fi

echo "2/4 Checking the environment ..." | tee -a "$LOG"
"$PY" - <<'PYCHECK' 2>&1 | tee -a "$LOG"
import sys, shutil
try:
    import numpy, scipy, scipy.special, sklearn, pandas, pyarrow, torch, transformers, trl, peft
    from transformers import AutoModelForCausalLM  # pulls in generation -> sklearn/scipy
    pandas.DataFrame([{"a": 1, "b": "x"}])
except Exception as error:
    print(f"ENVIRONMENT CHECK FAILED: {type(error).__name__}: {str(error)[:200]}")
    print("Try the project venv instead:  KERNEL=venv bash run_notebook.sh")
    sys.exit(1)
print(f"python {sys.executable}")
print(f"numpy {numpy.__version__} ({numpy.__file__.split('/site-packages')[0]})")
print(f"scipy {scipy.__version__} | sklearn {sklearn.__version__} | pandas {pandas.__version__} | "
      f"torch {torch.__version__} (CUDA {torch.cuda.is_available()}) | transformers {transformers.__version__} | "
      f"trl {trl.__version__} | peft {peft.__version__}")
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
if not torch.cuda.is_available():
    print("WARNING: CUDA not available - training will run on CPU (very slow).")
PYCHECK
[ "${PIPESTATUS[0]}" -eq 0 ] || exit 1

echo "3/4 Executing $NB (CPT + QLoRA; ~10-15 min on an L40S/A100) ..." | tee -a "$LOG"
# papermill prints each cell's output live and saves the notebook after every
# cell, so progress is visible and a crash still leaves the finished cells on disk.
TMP_NB=notebooks/.run_in_progress.ipynb
set +e
NOTEBOOK_HEADLESS=1 "$PY" -m papermill "$NB" "$TMP_NB" -k "$KERNEL" --cwd notebooks \
    --log-output --progress-bar --request-save-on-cell-execute 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
if [ "$status" -ne 0 ] || [ ! -s "$TMP_NB" ]; then
    echo "Notebook run FAILED (exit $status). Finished cells are saved in $TMP_NB; see the log above." | tee -a "$LOG"
    exit 1
fi
mv "$TMP_NB" "$NB"

echo "4/4 Exporting HTML ..." | tee -a "$LOG"
{ "$PY" -m nbconvert --to html "$NB" --output-dir notebooks 2>&1 || jupyter nbconvert --to html "$NB" --output-dir notebooks 2>&1; } | tee -a "$LOG"
cp data/instruction/instruction_dataset.jsonl instruction_dataset.jsonl
echo "Done: $NB (with outputs), notebooks/CorpPolicyLM_Assignment1A.html, instruction_dataset.jsonl" | tee -a "$LOG"
