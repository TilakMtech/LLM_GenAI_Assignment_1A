"""Step 1a - download the public HR / corporate policy PDFs listed in data/sources.csv.

Uses only the standard library so it runs anywhere with internet access
(Colab, a laptop). Files that already exist are skipped, failures are logged
and never stop the run, and anything that is not a real PDF is discarded.

    python -m src.download_corpus
"""
import csv
import time
import urllib.request
from pathlib import Path

from src import config

HEADERS = {"User-Agent": "Mozilla/5.0 (CorpPolicyLM academic corpus builder)"}


def download(url: str, dest: Path, timeout: int = 60) -> str:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    if not payload.startswith(b"%PDF"):
        return "not_a_pdf"
    dest.write_bytes(payload)
    return "downloaded"


def download_corpus(sources_csv: Path = config.SOURCES_CSV,
                    output_dir: Path = config.RAW_PDF_DIR,
                    report_path: Path = config.REPORT_DIR / "download_report.csv"):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with sources_csv.open(encoding="utf-8") as handle:
        sources = list(csv.DictReader(handle))

    records = []
    for row in sources:
        dest = output_dir / row["file_name"]
        if dest.exists() and dest.stat().st_size > 0:
            status = "already_present"
        else:
            try:
                status = download(row["url"], dest)
            except Exception as error:  # network errors must not stop the run
                status = f"failed: {type(error).__name__}: {str(error)[:80]}"
            time.sleep(0.5)  # be polite to the hosts
        size = dest.stat().st_size if dest.exists() else 0
        records.append({**row, "status": status, "bytes": size})
        print(f"{row['file_name']:<45} {status}")

    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    ok = sum(r["status"] in {"downloaded", "already_present"} for r in records)
    print(f"\n{ok}/{len(records)} source PDFs available in {output_dir}")
    return records


if __name__ == "__main__":
    download_corpus()
