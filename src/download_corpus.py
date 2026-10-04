"""Step 1a - download the public HR / corporate policy PDFs listed in data/sources.csv.

Uses only the standard library so it runs anywhere with internet access
(Colab, a laptop). Files that already exist are skipped, failures are logged
and never stop the run, and anything that is not a real PDF is discarded.

    python -m src.download_corpus
"""
import csv
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from src import config

# Many institutional sites return 403/404 to non-browser user agents, so we
# send ordinary browser headers (plus a same-site Referer).
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _ssl_context():
    """Use certifi's CA bundle when available (fixes 'unable to get local issuer
    certificate' on some hosts). Verification is never disabled."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _fetch(url: str, timeout: int) -> bytes:
    parts = urllib.parse.urlsplit(url)
    safe_url = urllib.parse.urlunsplit(parts._replace(path=urllib.parse.quote(parts.path, safe="/%")))
    headers = {**HEADERS, "Referer": f"{parts.scheme}://{parts.netloc}/"}
    request = urllib.request.Request(safe_url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
        return response.read()


def download(url: str, dest: Path, timeout: int = 30, retries: int = 1) -> str:
    """Retry only transient network errors; HTTP 4xx and TLS failures are final."""
    for attempt in range(retries + 1):
        try:
            payload = _fetch(url, timeout)
            break
        except urllib.error.HTTPError:
            raise  # 403/404/410: the server answered - retrying will not help
        except urllib.error.URLError as error:
            if isinstance(error.reason, ssl.SSLError) or attempt == retries:
                raise
        except (TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        time.sleep(2)
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
