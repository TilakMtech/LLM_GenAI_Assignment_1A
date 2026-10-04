"""Step 2a / 5A - deterministic document-level train / held-out split.

The split is made on whole documents *before* tokenisation so no sentence of
a held-out policy is ever seen during CPT (no leakage between the CPT
training stream and the perplexity evaluation set).
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from src import config


def split_corpus(input_dir: Path = config.DOMAIN_CORPUS_DIR,
                 output_dir: Path = config.SPLITS_DIR,
                 eval_fraction: float = config.EVAL_FRACTION, seed: int = config.SEED):
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between zero and one")
    documents, seen = [], set()
    for path in sorted(input_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Empty document: {path.name}")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            raise ValueError("Duplicate documents found; rerun cleaning first")
        seen.add(digest)
        documents.append({"document_id": path.stem, "source": path.name,
                          "sha256": digest, "words": len(text.split()), "text": text})
    if len(documents) < 2:
        raise ValueError("At least two documents are needed for a holdout")

    # Select held-out documents until they hold ~eval_fraction of all *words*
    # (documents range from 1-page policies to 200-page manuals, so a 10% share
    # of documents can be only ~3% of the text). Documents that the Step-3/5/B3
    # probe prompts ask about are pinned to train (their facts must be learnable).
    total_words = sum(r["words"] for r in documents)
    target = eval_fraction * total_words
    pinned = {p.lower() for p in config.PROBE_DOCUMENTS}
    candidates = [d for d in documents if d["source"].lower() not in pinned]
    random.Random(seed).shuffle(candidates)
    eval_docs, eval_words = [], 0
    for doc in candidates:
        if eval_words >= target:
            break
        if eval_words + doc["words"] > 1.5 * target:
            continue  # skip a document so large it would overshoot the target
        eval_docs.append(doc)
        eval_words += doc["words"]
    if not eval_docs:
        raise ValueError("Could not select a held-out set; add documents")
    eval_ids = {d["document_id"] for d in eval_docs}
    splits = {"train": [d for d in documents if d["document_id"] not in eval_ids], "eval": eval_docs}
    count = len(eval_docs)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    doc_fraction = count / len(documents)
    word_fraction = eval_words / total_words
    note = (f"Whole-document holdout = {word_fraction:.1%} of words (target {eval_fraction:.0%}); "
            f"probe documents pinned to train: {sorted(pinned)}")
    report = {
        "seed": seed, "split_unit": "whole_document, sized by word share",
        "requested_eval_fraction": eval_fraction,
        "documents_total": len(documents),
        "documents_train": len(splits["train"]), "documents_eval": count,
        "actual_eval_document_fraction": round(doc_fraction, 4),
        "actual_eval_word_fraction": round(eval_words / total_words, 4),
        "note": note,
        "documents": {name: [{k: v for k, v in r.items() if k != "text"} for r in rows]
                      for name, rows in splits.items()},
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Train: {len(splits['train'])} docs; held-out eval: {count} docs "
          f"({doc_fraction:.1%} of docs, {eval_words / total_words:.1%} of words)")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-fraction", type=float, default=config.EVAL_FRACTION)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    split_corpus(eval_fraction=args.eval_fraction, seed=args.seed)


if __name__ == "__main__":
    main()
