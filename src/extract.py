import csv
from pathlib import Path

from pypdf import PdfReader

from src.config import EXTRACTED_TEXT_DIR, RAW_PDF_DIR, REPORT_DIR


def extract_pdf_page_by_page(pdf_path: Path):
    """Return document text with page markers and extraction statistics."""
    reader = PdfReader(pdf_path)
    extracted_pages = []
    text_page_count = 0
    empty_page_count = 0
    characters_extracted = 0

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        characters_extracted += len(page_text)

        if page_text.strip():
            text_page_count += 1
        else:
            empty_page_count += 1

        extracted_pages.append(
            f"--- PAGE {page_number} ---\n{page_text.strip()}"
        )

    full_text = "\n\n".join(extracted_pages)
    stats = {
        "file_name": pdf_path.name,
        "total_pages": len(reader.pages),
        "text_pages": text_page_count,
        "empty_pages": empty_page_count,
        # Count raw extracted characters, excluding our added page markers.
        "characters_extracted": characters_extracted,
    }
    return full_text, stats


def extract_corpus(input_dir: Path, output_dir: Path, report_path: Path):
    """Save text for each PDF and return the rows written to the CSV report."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Raw PDF folder not found: {input_dir}")

    pdf_files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    for pdf_path in pdf_files:
        try:
            full_text, stats = extract_pdf_page_by_page(pdf_path)
        except Exception as error:  # corrupt / encrypted PDFs are reported, not fatal
            print(f"Skipped {pdf_path.name}: {type(error).__name__}: {error}")
            records.append({"file_name": pdf_path.name, "total_pages": 0, "text_pages": 0,
                            "empty_pages": 0, "characters_extracted": 0})
            continue
        output_path = output_dir / f"{pdf_path.stem}.txt"
        output_path.write_text(full_text, encoding="utf-8")
        records.append(stats)
        print(f"Saved {output_path.name}: {stats['total_pages']} pages")

    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    return records


def main():
    report_path = REPORT_DIR / "extraction_report.csv"
    records = extract_corpus(RAW_PDF_DIR, EXTRACTED_TEXT_DIR, report_path)

    print(f"\nExtracted {len(records)} documents.")
    print(f"Total pages: {sum(row['total_pages'] for row in records)}")
    print(f"Pages without text: {sum(row['empty_pages'] for row in records)}")
    print(f"Text files: {EXTRACTED_TEXT_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
