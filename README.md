# CorpPolicyLM — Assignment 1A (CPT + QLoRA SFT)

Domain LLM for **HR Policy & Corporate Documents** (Enterprise Variant 1 — Domain Q&A Assistant),
built on `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` and trained on a Colab **T4**.

## Pipeline → code

| Step | Module | Main outputs |
|---|---|---|
| 1a Collect PDFs | `src/download_corpus.py` (+ `data/sources.csv`) | `data/raw_pdfs/`, `outputs/reports/download_report.csv` |
| 1b Extract page-by-page | `src/extract.py` | `data/extracted_text/*.txt`, `extraction_report.csv` |
| 1c Clean (boilerplate → length → exact dedup → near dedup → English) | `src/clean.py` | `data/domain_corpus/*.txt`, `cleaning_report.csv`, `corpus_summary.json` |
| 2 Held-out split, BOS/EOS, packing, Parquet | `src/dataset.py`, `src/tokenize_pack.py` | `data/splits/`, `data/processed/*_packed.parquet`, `tokenization_stats.json` |
| 3 Load bf16 + audit + baseline | `src/model_utils.py` | `outputs/evaluations/architecture_audit.json`, `baseline_generations.json` |
| 4 CPT (custom loss callback) | `src/train_cpt.py` | `models/cpt/`, `cpt_loss_log.csv`, `figures/cpt_loss_curve.png` |
| 5 Perplexity + forgetting | `src/evaluate_cpt.py` | `perplexity.json`, `forgetting_check.csv` |
| B1 Instruction dataset (80/20) | `src/instruction_data.py` | `data/instruction/instruction_dataset.jsonl` (+ train/eval) |
| B2 QLoRA Adapter B (r16/α32, q,v) | `src/train_qlora.py` | `models/adapters/adapter_B/` |
| B3 Evaluation | `src/evaluate_sft.py` | `sft_adapter_B_*.csv/json` |

All settings live in `src/config.py` (override any of them with an environment variable of the same name,
e.g. `BLOCK_SIZE=1024`).

## How to run (Colab T4)

1. Copy the whole `CorpPolicyLM` folder to Google Drive as `MyDrive/CorpPolicyLM`.
2. Open `notebooks/CorpPolicyLM_Assignment1A.ipynb` in Colab → *Runtime → Change runtime type → T4 GPU*.
3. Run the install cell, restart the session once, then *Run all*. Approx. time on T4: CPT 10–25 min, QLoRA 5–10 min.
4. *File → Save*, then run the last cell to export the HTML.
5. Copy back to OneDrive: `notebooks/*.ipynb`, `notebooks/*.html`, `instruction_dataset.jsonl`,
   `data/domain_corpus/`, `data/instruction/`, `data/splits/`, `outputs/`.
   (`models/` stays on Drive — it is git-ignored.)

## Submission deliverables
* `notebooks/CorpPolicyLM_Assignment1A.ipynb` (with outputs) and `.html`
* `instruction_dataset.jsonl`
* `data/domain_corpus/*.txt`

## Design notes
* The training dtype differs from the load dtype on purpose: T4 has no native bf16, and pure-bf16 AdamW drops
  2e-5-sized updates, so CPT uses fp32 master weights + fp16 autocast + paged 8-bit AdamW (bf16 autocast on A100).
* `src/tokenize.py` was renamed to `src/tokenize_pack.py` so it never shadows Python's standard `tokenize` module.
* The corpus downloader is idempotent and never fails the run; failed URLs are listed in `download_report.csv`.
