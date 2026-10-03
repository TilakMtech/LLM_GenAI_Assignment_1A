"""Step 1b - cleaning pipeline for the extracted policy text.

Stages (document counts are reported before/after each one):
  0. basic_cleaning   - drop page markers / "Page x of y", repeated page
                        headers & footers (document-control blocks), de-hyphenate
                        line breaks, strip control characters, collapse whitespace
  1. length_filter    - drop documents with fewer than MIN_WORDS words
  2. deduplication    - drop exact duplicates (SHA-256 of normalised text)
  3. near_dedup       - drop near duplicates (5-word shingle Jaccard >= threshold)
  4. english_filter   - keep only documents langdetect labels as English
"""
import csv
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

from langdetect import DetectorFactory, LangDetectException, detect

from src.config import (DOMAIN_CORPUS_DIR, EXTRACTED_TEXT_DIR, MIN_WORDS,
                        NEAR_DUP_THRESHOLD, REPORT_DIR)

DetectorFactory.seed = 0  # Make language detection repeatable.

PAGE_MARKER = re.compile(r"--- PAGE \d+ ---")
PAGE_NUMBER = re.compile(r"(Page\s*)?\d+\s*(of|/)\s*\d+|Page\s*\d+|-\s*\d{1,3}\s*-", re.I)


def repeated_lines(text: str, min_pages: int = 3) -> set:
    """Lines that repeat on many pages are running headers / footers."""
    pages = PAGE_MARKER.split(text)
    if len(pages) - 1 < min_pages:  # too few pages to call anything boilerplate
        return set()
    counts = Counter()
    for page in pages:
        counts.update({re.sub(r"\s+", " ", l).strip() for l in page.splitlines() if l.strip()})
    threshold = max(min_pages, int(0.5 * (len(pages) - 1)))
    return {line for line, n in counts.items() if n >= threshold and len(line) < 120}


def clean_text(text: str) -> str:
    """Remove page labels and boilerplate while preserving the line structure."""
    boilerplate = repeated_lines(text)
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks

    cleaned_lines = []
    for line in text.splitlines():
        line = re.sub(r"[^\S\r\n]+", " ", line).strip()
        if PAGE_MARKER.fullmatch(line) or PAGE_NUMBER.fullmatch(line):
            continue
        if line in boilerplate:
            continue
        if not line and (not cleaned_lines or cleaned_lines[-1] == ""):
            continue  # keep at most one blank line between paragraphs
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def shingles(text: str, n: int = 5) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def clean_corpus(input_dir: Path, output_dir: Path, report_path: Path,
                 min_words: int = MIN_WORDS, near_dup_threshold: float = NEAR_DUP_THRESHOLD):
    """Save documents passing every filter and return one report row per document."""
    if min_words < 1:
        raise ValueError("min_words must be at least 1.")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Extracted text folder not found: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Use a separate output folder to preserve extracted text.")
    text_files = sorted(p for p in input_dir.glob("*.txt") if p.is_file())
    if not text_files:
        raise FileNotFoundError(f"No text files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    seen_hashes = {}   # sha256 -> first file kept
    kept_shingles = {}  # file -> shingles, for near-duplicate search

    for text_path in text_files:
        original_text = text_path.read_text(encoding="utf-8")
        cleaned_text = clean_text(original_text)
        word_count = len(cleaned_text.split())
        digest = hashlib.sha256(" ".join(cleaned_text.split()).lower().encode()).hexdigest()
        duplicate_of, similarity, language = "", 0.0, "not_checked"

        if word_count < min_words:
            reason = "below_min_words"
        elif digest in seen_hashes:
            reason, duplicate_of, similarity = "exact_duplicate", seen_hashes[digest], 1.0
        else:
            seen_hashes[digest] = text_path.name
            current = shingles(cleaned_text)
            best = max(((jaccard(current, other), name) for name, other in kept_shingles.items()),
                       default=(0.0, ""))
            similarity = round(best[0], 3)
            if best[0] >= near_dup_threshold:
                reason, duplicate_of = "near_duplicate", best[1]
            else:
                try:
                    language = detect(cleaned_text)
                except LangDetectException:
                    language, reason = "unknown", "language_detection_failed"
                else:
                    reason = "passes_filters" if language == "en" else "non_english"
                if reason == "passes_filters":
                    kept_shingles[text_path.name] = current

        kept = reason == "passes_filters"
        output_path = output_dir / text_path.name
        if kept:
            output_path.write_text(cleaned_text + "\n", encoding="utf-8")
        elif output_path.exists():
            output_path.unlink()  # remove a stale copy from an earlier run

        before, after = len(original_text), len(cleaned_text)
        records.append({
            "file_name": text_path.name, "word_count": word_count, "min_words": min_words,
            "status": "kept" if kept else "rejected", "reason": reason,
            "duplicate_of": duplicate_of, "max_similarity": similarity, "language": language,
            "characters_before": before, "characters_after": after,
            "characters_removed": before - after,
            "reduction_percent": round(100 * (before - after) / before, 2) if before else 0.0,
        })
        print(f"{text_path.name}: {word_count} words - {reason} (language: {language})")

    # Remove outputs whose source text no longer exists.
    sources = {p.name for p in text_files}
    for stale in output_dir.glob("*.txt"):
        if stale.name not in sources:
            stale.unlink()

    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


STAGES = [
    ("length_filter", {"below_min_words"}),
    ("deduplication", {"exact_duplicate"}),
    ("near_dedup", {"near_duplicate"}),
    ("english_filter", {"non_english", "language_detection_failed"}),
]


def summarize_corpus(records):
    """Per-stage document/character counts, separating formatting from removal."""
    remaining = list(records)
    initial_chars = sum(r["characters_before"] for r in records)
    cleaned_chars = sum(r["characters_after"] for r in records)
    stages = [{
        "stage": "basic_cleaning", "documents_before": len(records),
        "documents_after": len(records), "documents_removed": 0,
        "characters_before": initial_chars, "characters_after": cleaned_chars,
        "characters_removed": initial_chars - cleaned_chars,
    }]
    for name, reasons in STAGES:
        before = remaining
        remaining = [r for r in before if r["reason"] not in reasons]
        chars_before = sum(r["characters_after"] for r in before)
        chars_after = sum(r["characters_after"] for r in remaining)
        stages.append({
            "stage": name, "documents_before": len(before), "documents_after": len(remaining),
            "documents_removed": len(before) - len(remaining),
            "characters_before": chars_before, "characters_after": chars_after,
            "characters_removed": chars_before - chars_after,
        })

    words = [r["word_count"] for r in remaining]
    max_docs = max(s["documents_removed"] for s in stages)
    max_chars = max(s["characters_removed"] for s in stages)
    final_chars = sum(r["characters_after"] for r in remaining)
    return {
        "documents_before": len(records),
        "documents_after": len(remaining),
        "stages": stages,
        "largest_document_removal_stages": [
            s["stage"] for s in stages if max_docs > 0 and s["documents_removed"] == max_docs],
        "largest_character_removal_stages": [
            s["stage"] for s in stages if max_chars > 0 and s["characters_removed"] == max_chars],
        "retained_document_words": {
            "total": sum(words),
            "minimum": min(words) if words else None,
            "maximum": max(words) if words else None,
            "mean": round(statistics.mean(words), 1) if words else None,
            "median": statistics.median(words) if words else None,
        },
        "characters_before": initial_chars,
        "characters_after": final_chars,
        "character_reduction_percent": (
            round(100 * (initial_chars - final_chars) / initial_chars, 2) if initial_chars else 0.0),
        "language_detection_failures": sum(r["reason"] == "language_detection_failed" for r in records),
    }


def main():
    report_path = REPORT_DIR / "cleaning_report.csv"
    records = clean_corpus(EXTRACTED_TEXT_DIR, DOMAIN_CORPUS_DIR, report_path)
    summary = summarize_corpus(records)
    summary_path = REPORT_DIR / "corpus_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nCorpus summary:")
    for stage in summary["stages"]:
        print(f"  {stage['stage']:<15} {stage['documents_before']:>3} -> "
              f"{stage['documents_after']:>3} documents; "
              f"{stage['characters_removed']:>8} characters removed")
    largest = summary["largest_document_removal_stages"]
    print(f"Largest document removal: {', '.join(largest) if largest else 'none'}")
    print(f"Retained document word counts: {summary['retained_document_words']}")
    return summary


if __name__ == "__main__":
    main()
