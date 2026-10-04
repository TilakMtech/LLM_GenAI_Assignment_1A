"""Step 2 - tokenisation with the model's own tokenizer and sequence packing.

* Every document is wrapped as  [BOS] + tokens + [EOS]  so document
  boundaries stay visible inside packed sequences.
* All wrapped documents are concatenated into one flat token stream and
  sliced into fixed BLOCK_SIZE chunks (the model's context window) - no padding.
* Train and held-out streams are packed separately and saved as Parquet.

(Named tokenize_pack.py, not tokenize.py, so it never shadows Python's
standard-library `tokenize` module, which torch/inspect import.)
"""
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from src import config


def load_tokenizer(model_id: str = config.MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define BOS and EOS tokens for packing.")
    if tokenizer.pad_token is None:  # only used for batched generation / SFT
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def read_split(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode_documents(documents, tokenizer):
    """Return one [BOS] ... [EOS] id list per document."""
    encoded = []
    for doc in documents:
        ids = tokenizer(doc["text"], add_special_tokens=False)["input_ids"]
        encoded.append([tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id])
    return encoded


def pack(sequences, block_size: int, keep_remainder: bool = False):
    """Concatenate into one stream and cut into full blocks.

    Training drops the short tail (every training block is exactly block_size);
    the held-out set keeps it so perplexity is computed on *every* eval token.
    """
    stream = [tok for seq in sequences for tok in seq]
    n_blocks = len(stream) // block_size
    blocks = [stream[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    remainder = len(stream) - n_blocks * block_size
    if keep_remainder and remainder > 1:
        blocks.append(stream[n_blocks * block_size:])
        remainder = 0
    return blocks, len(stream), remainder


def build_packed_dataset(model_id: str = config.MODEL_ID, block_size: int = config.BLOCK_SIZE,
                         splits_dir: Path = config.SPLITS_DIR,
                         output_dir: Path = config.PROCESSED_DIR):
    tokenizer = load_tokenizer(model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"model_id": model_id, "tokenizer_class": type(tokenizer).__name__,
             "vocab_size": len(tokenizer), "bos_token": tokenizer.bos_token,
             "eos_token": tokenizer.eos_token, "block_size": block_size, "splits": {}}

    for split in ("train", "eval", "eval_unseen"):
        path = splits_dir / f"{split}.jsonl"
        documents = read_split(path) if path.exists() else []
        if not documents and split == "eval_unseen":
            (output_dir / f"{split}_packed.parquet").unlink(missing_ok=True)
            continue  # optional secondary set (empty for very small corpora)
        encoded = encode_documents(documents, tokenizer)
        lengths = [len(seq) for seq in encoded]
        blocks, total, dropped = pack(encoded, block_size, keep_remainder=(split != "train"))
        if not blocks:
            raise ValueError(f"{split}: only {total} tokens - fewer than one {block_size}-token block. "
                             "Add documents or lower BLOCK_SIZE.")
        pd.DataFrame({"input_ids": blocks}).to_parquet(output_dir / f"{split}_packed.parquet",
                                                       index=False)
        pd.DataFrame({"document_id": [d["document_id"] for d in documents],
                      "tokens": lengths}).to_csv(output_dir / f"{split}_doc_tokens.csv", index=False)
        stats["splits"][split] = {
            "documents": len(documents),
            "total_tokens": total,
            "avg_tokens_per_document": round(total / len(documents), 1),
            "min_tokens_per_document": min(lengths),
            "max_tokens_per_document": max(lengths),
            "packed_sequences": len(blocks),
            "tokens_in_packed_sequences": sum(len(b) for b in blocks),
            "tokens_dropped_remainder": dropped,
            "bos_eos_pairs": len(documents),
        }
        print(f"{split}: {len(documents)} docs, {total:,} tokens "
              f"(avg {total / len(documents):,.0f}/doc) -> {len(blocks)} packed "
              f"sequences of {block_size}; {dropped} remainder tokens dropped")

    # Sanity check: every block is exactly block_size and contains BOS/EOS markers.
    train = pd.read_parquet(output_dir / "train_packed.parquet")
    assert train["input_ids"].map(len).eq(block_size).all()
    stats["bos_in_train_stream"] = int(sum(list(b).count(tokenizer.bos_token_id) for b in train["input_ids"]))
    stats["eos_in_train_stream"] = int(sum(list(b).count(tokenizer.eos_token_id) for b in train["input_ids"]))
    (config.REPORT_DIR / "tokenization_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


if __name__ == "__main__":
    build_packed_dataset()
