"""
Restore Justice Foundation website ingestion.

Scrapes two sections of restorejustice.org:
  /resources/             — reentry resource directory (org listings + descriptions)
  /our-work/reports-factsheets/  — policy reports and factsheets

Output schema (JSONL):
  id, source, section, page_title, url, text, scraped_at

Usage:
  python ingest/restorejustice_ingest.py --local-only
  python ingest/restorejustice_ingest.py   # full S3 pipeline
"""
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import boto3
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://www.restorejustice.org"
# Scrape these sub-trees of restorejustice.org.
# The /resources/ hub and /our-work/ tree are both crawled; PDFs found
# while crawling are downloaded and extracted separately.
SEED_URLS = [
    ("resources",        f"{BASE_URL}/resources/"),
    ("our-work",         f"{BASE_URL}/our-work/"),
]
OUTPUT_FILE = "data_files/corpus/restorejustice_corpus.jsonl"
DELAY       = 0.75
MAX_RETRIES = 3
MIN_TEXT    = 200

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; IL-LegalRAG/1.0; "
        "legal research/non-commercial; contact: your@email.com)"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def _get_s3_config() -> tuple[str, str]:
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        log.error("RAW_S3_BUCKET is not set."); sys.exit(1)
    prefix = os.environ.get("RESTOREJUSTICE_S3_PREFIX", "restorejustice/").rstrip("/") + "/"
    return bucket, prefix

def _upload_to_s3(local_path: Path, no_upload: bool = False) -> None:
    if no_upload:
        return
    bucket, prefix = _get_s3_config()
    s3_key = f"{prefix}{local_path.name}"
    boto3.client("s3").upload_file(str(local_path), bucket, s3_key)
    log.info(f"  Uploaded → s3://{bucket}/{s3_key}")

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update(HEADERS)

def _fetch(url: str) -> BeautifulSoup | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, timeout=30)
            resp.raise_for_status()
            time.sleep(DELAY)
            return BeautifulSoup(resp.text, "html.parser")
        except requests.HTTPError as e:
            log.warning(f"HTTP {e.response.status_code} (attempt {attempt}): {url}")
            if e.response.status_code in (429, 502, 503):
                time.sleep(15 * attempt)
            elif attempt == MAX_RETRIES:
                return None
        except requests.RequestException as e:
            log.warning(f"Error: {e} (attempt {attempt})")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(10)
    return None

def _fetch_bytes(url: str) -> bytes | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, timeout=60)
            resp.raise_for_status()
            time.sleep(DELAY)
            return resp.content
        except requests.HTTPError as e:
            log.warning(f"HTTP {e.response.status_code} (attempt {attempt}): {url}")
            if e.response.status_code in (429, 502, 503):
                time.sleep(15 * attempt)
            elif attempt == MAX_RETRIES:
                return None
        except requests.RequestException as e:
            log.warning(f"Error: {e} (attempt {attempt})")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(10)
    return None

def _pdf_to_text(content: bytes, label: str) -> str:
    try:
        import pypdf
    except ImportError:
        log.warning("pypdf not installed — skipping PDF extraction")
        return ""
    import io
    reader = pypdf.PdfReader(io.BytesIO(content))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_NOISE = ["nav", "header", "footer", "script", "style", "noscript", "aside"]

def _extract(soup: BeautifulSoup) -> str:
    for tag in soup.find_all(_NOISE):
        tag.decompose()
    main = (soup.find("main") or soup.find("article")
            or soup.find(class_=re.compile(r"content|entry|post", re.I))
            or soup.body)
    if not main:
        return ""
    for tag in main.find_all(["p", "li", "h1", "h2", "h3", "h4"]):
        tag.insert_before("\n")
    text = main.get_text()
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def _title(soup: BeautifulSoup, url: str) -> str:
    raw = soup.title.string if soup.title else url
    return re.sub(r"\s*[|\-–]\s*Restore Justice.*$", "", raw).strip()

def _make_id(url: str) -> str:
    path = urlparse(url).path.strip("/").replace("/", "-")
    return f"rj-{path[:80]}" if path else "rj-home"

def _is_internal(url: str) -> bool:
    parsed = urlparse(url)
    return not parsed.netloc or "restorejustice.org" in parsed.netloc


def _is_pdf(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")

# ---------------------------------------------------------------------------
# Scrape
# ---------------------------------------------------------------------------

def scrape() -> list[dict]:
    visited: set[str] = set()
    records: list[dict] = []

    for section, seed in SEED_URLS:
        log.info(f"\n── Section: {section}  ({seed})")
        queue = [seed]

        while queue:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            log.info(f"  {url}")

            if _is_pdf(url):
                # Download and extract PDF
                content = _fetch_bytes(url)
                if content is None:
                    log.warning(f"    PDF download failed: {url}")
                    continue
                text = _pdf_to_text(content, url)
                if len(text) >= MIN_TEXT:
                    filename = urlparse(url).path.split("/")[-1].replace("_", " ").replace("-", " ")
                    page_title = re.sub(r"\.pdf$", "", filename, flags=re.I).strip()
                    records.append({
                        "id":          _make_id(url),
                        "source":      "restore_justice",
                        "section":     section,
                        "page_title":  page_title,
                        "url":         url,
                        "text":        text,
                        "scraped_at":  datetime.now(timezone.utc).isoformat(),
                    })
                    log.info(f"    PDF: {len(text):,} chars")
                continue  # no links to follow from a PDF

            soup = _fetch(url)
            if soup is None:
                continue

            text = _extract(soup)
            if len(text) >= MIN_TEXT:
                records.append({
                    "id":          _make_id(url),
                    "source":      "restore_justice",
                    "section":     section,
                    "page_title":  _title(soup, url),
                    "url":         url,
                    "text":        text,
                    "scraped_at":  datetime.now(timezone.utc).isoformat(),
                })

            # Follow links within the seed subtree (HTML) or PDFs on same domain
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"]).split("#")[0].rstrip("/")
                if href in visited:
                    continue
                if not _is_internal(href):
                    continue
                if _is_pdf(href) and "restorejustice.org" in href:
                    queue.append(href)  # enqueue PDFs from this domain
                elif href.startswith(seed):
                    queue.append(href)  # enqueue HTML sub-pages within seed subtree

    log.info(f"\nTotal: {len(records)} pages from {len(visited)} visited")
    return records

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Ingest Restore Justice Foundation → JSONL")
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    if not args.local_only:
        _get_s3_config()

    records  = scrape()
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(f"Wrote {len(records)} records → {out_path}")
    _upload_to_s3(out_path, args.local_only)

if __name__ == "__main__":
    main()
