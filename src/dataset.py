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


def split_tail(text: str, fraction: float):
    """Split text into (first 1-fraction, last fraction) by words, cutting at the
    paragraph (or, failing that, line) break closest to the target so no sentence
    is split across the train and held-out sets."""
    total = len(text.split())
    target = (1 - fraction) * total
    for separator in ("\n\n", "\n"):
        best, best_gap, position, words = None, None, 0, 0
        pieces = text.split(separator)
        for piece in pieces[:-1]:
            position += len(piece) + len(separator)
            words += len(piece.split())
            gap = abs(words - target)
            if 0 < words < total and (best_gap is None or gap < best_gap):
                best, best_gap = position, gap
        if best is not None and best_gap <= 0.05 * total:
            return text[:best].strip(), text[best:].strip()
    words = text.split()  # no break close enough: cut by words
    k = int(round(target))
    return " ".join(words[:k]), " ".join(words[k:])


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

    # Two held-out sets, both never seen during CPT:
    #  * eval        - the last EVAL_FRACTION (10%) of the words of every remaining
    #                  document, cut at a paragraph boundary: the assignment's
    #                  "10% of your domain corpus", same organisations/style as train.
    #  * eval_unseen - a few whole documents (~UNSEEN_FRACTION of words) that CPT
    #                  never sees at all: measures transfer to *other organisations'*
    #                  policies (a harder, secondary generalisation test).
    # Documents asked about by the probe prompts are never put in eval_unseen.
    total_words = sum(r["words"] for r in documents)
    pinned = {p.lower() for p in config.PROBE_DOCUMENTS}
    candidates = [d for d in documents if d["source"].lower() not in pinned]
    random.Random(seed).shuffle(candidates)
    unseen, unseen_words, target = [], 0, config.UNSEEN_FRACTION * total_words
    for doc in candidates:
        if unseen_words >= target:
            break
        if unseen_words + doc["words"] > 1.5 * target:
            continue
        unseen.append(doc)
        unseen_words += doc["words"]
    unseen_ids = {d["document_id"] for d in unseen}

    train, held = [], []
    for doc in documents:
        if doc["document_id"] in unseen_ids:
            continue
        head, tail = split_tail(doc["text"], eval_fraction)
        train.append({**doc, "text": head, "words": len(head.split()), "part": "first 90%"})
        held.append({**doc, "text": tail, "words": len(tail.split()), "part": "last 10%"})
    splits = {"train": train, "eval": held, "eval_unseen": unseen}
    eval_words = sum(d["words"] for d in held)
    count = len(held)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    doc_fraction = count / len(documents)
    word_fraction = eval_words / total_words
    note = (f"eval = last {eval_fraction:.0%} of each of {count} documents ({word_fraction:.1%} of words); "
            f"eval_unseen = {len(unseen)} whole documents never used in CPT "
            f"({unseen_words / total_words:.1%} of words); probe documents kept out of eval_unseen.")
    report = {
        "seed": seed, "split_unit": "eval: per-document tail (paragraph boundary); eval_unseen: whole documents",
        "requested_eval_fraction": eval_fraction,
        "documents_total": len(documents),
        "documents_train": len(splits["train"]), "documents_eval": count,
        "documents_eval_unseen": len(unseen),
        "actual_eval_unseen_word_fraction": round(unseen_words / total_words, 4),
        "actual_eval_document_fraction": round(doc_fraction, 4),
        "actual_eval_word_fraction": round(eval_words / total_words, 4),
        "note": note,
        "documents": {name: [{k: v for k, v in r.items() if k != "text"} for r in rows]
                      for name, rows in splits.items()},
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Train: first 90% of {len(train)} docs | eval (held-out last 10%): {eval_words / total_words:.1%} "
          f"of words | eval_unseen: {len(unseen)} whole docs ({unseen_words / total_words:.1%} of words)")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-fraction", type=float, default=config.EVAL_FRACTION)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    split_corpus(eval_fraction=args.eval_fraction, seed=args.seed)


if __name__ == "__main__":
    main()
