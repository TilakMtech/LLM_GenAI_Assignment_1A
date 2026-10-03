import csv
import json
import statistics
import re
from pathlib import Path

from langdetect import DetectorFactory, LangDetectException, detect


DetectorFactory.seed = 0  # Make language detection repeatable.


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_TEXT_DIR = PROJECT_ROOT / "data" / "extracted_text"
DOMAIN_CORPUS_DIR = PROJECT_ROOT / "data" / "domain_corpus"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
MIN_WORDS = 100  # Provisional document-level threshold; not model tokens.


def clean_text(text: str) -> str:
    """Remove page labels and excess whitespace while preserving text lines."""
    cleaned_lines = []

    for line in text.splitlines():
        # Normalize horizontal whitespace without joining separate lines.
        line = re.sub(r"[^\S\r\n]+", " ", line).strip()

        if re.fullmatch(r"--- PAGE \d+ ---", line):
            continue
        if re.fullmatch(r"Page \d+ of \d+", line):
            continue

        # Keep at most one blank line between nonempty lines.
        if not line and (not cleaned_lines or cleaned_lines[-1] == ""):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def clean_corpus(
    input_dir: Path, output_dir: Path, report_path: Path, min_words: int = MIN_WORDS
):
    """Save documents passing length, deduplication, and English filters."""
    if min_words < 1:
        raise ValueError("min_words must be at least 1.")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Extracted text folder not found: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Use a separate output folder to preserve extracted text.")

    text_files = sorted(input_dir.glob("*.txt"))
    text_files = [path for path in text_files if path.is_file()]
    if not text_files:
        raise FileNotFoundError(f"No text files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    seen_documents = {}  # Cleaned text -> first filename passing length filtering.

    for text_path in text_files:
        original_text = text_path.read_text(encoding="utf-8")
        cleaned_text = clean_text(original_text)
        word_count = len(cleaned_text.split())
        duplicate_of = ""
        language = "not_checked"
        if word_count < min_words:
            reason = "below_min_words"
        elif cleaned_text in seen_documents:
            reason = "exact_duplicate"
            duplicate_of = seen_documents[cleaned_text]
        else:
            seen_documents[cleaned_text] = text_path.name
            try:
                language = detect(cleaned_text)
            except LangDetectException:
                language = "unknown"
                reason = "language_detection_failed"
            else:
                reason = "passes_filters" if language == "en" else "non_english"

        kept = reason == "passes_filters"
        output_path = output_dir / text_path.name
        if kept:
            output_path.write_text(cleaned_text, encoding="utf-8")
        elif output_path.exists():
            # Remove this document's old generated copy if it now fails the filter.
            output_path.unlink()

        before = len(original_text)
        after = len(cleaned_text)
        records.append({
            "file_name": text_path.name,
            "word_count": word_count,
            "min_words": min_words,
            "status": "kept" if kept else "rejected",
            "reason": reason,
            "duplicate_of": duplicate_of,
            "language": language,
            "characters_before": before,
            "characters_after": after,
            "characters_removed": before - after,
            "reduction_percent": round(100 * (before - after) / before, 2) if before else 0.0,
        })
        print(f"{text_path.name}: {word_count} words — {reason} (language: {language})")

    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    return records


def summarize_corpus(records):
    """Summarize this run, separating formatting changes from document removal."""
    remaining = list(records)
    initial_chars = sum(row["characters_before"] for row in records)
    cleaned_chars = sum(row["characters_after"] for row in records)
    stages = [{
        "stage": "basic_cleaning",
        "documents_before": len(records),
        "documents_after": len(records),
        "documents_removed": 0,
        "characters_before": initial_chars,
        "characters_after": cleaned_chars,
        "characters_removed": initial_chars - cleaned_chars,
    }]
    for name, reasons in [
        ("length_filter", {"below_min_words"}),
        ("deduplication", {"exact_duplicate"}),
        ("english_filter", {"non_english", "language_detection_failed"}),
    ]:
        before = remaining
        remaining = [row for row in before if row["reason"] not in reasons]
        chars_before = sum(row["characters_after"] for row in before)
        chars_after = sum(row["characters_after"] for row in remaining)
        stages.append({
            "stage": name,
            "documents_before": len(before),
            "documents_after": len(remaining),
            "documents_removed": len(before) - len(remaining),
            "characters_before": chars_before,
            "characters_after": chars_after,
            "characters_removed": chars_before - chars_after,
        })

    words = [row["word_count"] for row in remaining]
    max_documents = max(stage["documents_removed"] for stage in stages)
    max_characters = max(stage["characters_removed"] for stage in stages)
    final_chars = sum(row["characters_after"] for row in remaining)
    return {
        "documents_before": len(records),
        "documents_after": len(remaining),
        "stages": stages,
        "largest_document_removal_stages": [
            stage["stage"] for stage in stages
            if max_documents > 0 and stage["documents_removed"] == max_documents
        ],
        "largest_character_removal_stages": [
            stage["stage"] for stage in stages
            if max_characters > 0 and stage["characters_removed"] == max_characters
        ],
        "retained_document_words": {
            "total": sum(words),
            "minimum": min(words) if words else None,
            "maximum": max(words) if words else None,
            "mean": statistics.mean(words) if words else None,
            "median": statistics.median(words) if words else None,
        },
        "characters_before": initial_chars,
        "characters_after": final_chars,
        "character_reduction_percent": (
            round(100 * (initial_chars - final_chars) / initial_chars, 2)
            if initial_chars else 0.0
        ),
        "language_detection_failures": sum(
            row["reason"] == "language_detection_failed" for row in records
        ),
    }


def main():
    report_path = REPORT_DIR / "cleaning_report.csv"
    records = clean_corpus(EXTRACTED_TEXT_DIR, DOMAIN_CORPUS_DIR, report_path)
    summary = summarize_corpus(records)
    summary_path = REPORT_DIR / "corpus_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nCorpus summary:")
    for stage in summary["stages"]:
        print(f"  {stage['stage']}: {stage['documents_before']} -> "
              f"{stage['documents_after']} documents; "
              f"{stage['characters_removed']} characters removed")
    largest = summary["largest_document_removal_stages"]
    print(f"Largest document removal: {', '.join(largest) if largest else 'none — no documents removed'}")
    print(f"Retained document word counts: {summary['retained_document_words']}")
    print(f"Cleaned text files: {DOMAIN_CORPUS_DIR}")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
