"""Create a deterministic document-level split before tokenization."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def split_corpus(input_dir, output_dir, eval_fraction=0.1, seed=42):
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between zero and one")
    documents = []
    seen = set()
    for path in sorted(input_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Empty document: {path.name}")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            raise ValueError("Duplicate documents found; rerun cleaning first")
        seen.add(digest)
        documents.append({"document_id": path.stem, "source": path.name,
                          "sha256": digest, "text": text})
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
    total_words = sum(len(row["text"].split()) for row in documents)
    report = {"seed": seed, "split_unit": "whole_document",
              "requested_eval_fraction": eval_fraction,
              "actual_eval_document_fraction": count / len(documents),
              "actual_eval_word_fraction": sum(len(r["text"].split()) for r in splits["eval"]) / total_words,
              "note": "Four documents cannot yield an exact 10% whole-document holdout. Expand the corpus or explicitly revise the split strategy.",
              "documents": {name: [{k: v for k, v in r.items() if k != "text"} for r in rows]
                            for name, rows in splits.items()}}
    (output_dir / "split_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Train: {len(splits['train'])}; evaluation: {count} documents")
    print(f"Actual evaluation fraction: {count / len(documents):.1%}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_corpus(ROOT / "data/domain_corpus", ROOT / "data/splits", args.eval_fraction, args.seed)


if __name__ == "__main__":
    main()
