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

    random.Random(seed).shuffle(documents)
    count = min(len(documents) - 1, max(1, round(len(documents) * eval_fraction)))
    splits = {"train": documents[count:], "eval": documents[:count]}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_words = sum(r["words"] for r in documents)
    eval_words = sum(r["words"] for r in splits["eval"])
    doc_fraction = count / len(documents)
    note = ("Whole-document holdout close to the requested fraction."
            if abs(doc_fraction - eval_fraction) <= 0.05 else
            f"Only {len(documents)} documents: a whole-document holdout cannot hit "
            f"{eval_fraction:.0%} exactly; expand the corpus for a tighter split.")
    report = {
        "seed": seed, "split_unit": "whole_document",
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
