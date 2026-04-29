"""
Illinois Sentencing Policy Advisory Council (SPAC) publication ingestion.

Enumerates the ICJIA file server directory listing at:
  https://archive.icjia-api.cloud/files/spac/

Downloads each publication PDF, extracts text, and writes one JSONL record
per document. Meeting agendas and minutes are skipped by default (--include-meetings
to override). Duplicate files (timestamp-suffixed variants) are de-duplicated,
preferring the canonical filename over the timestamped copy.

Requires: pip install pypdf

Output schema (JSONL):
  id, source, agency, category, title, year, filename, url, text, scraped_at

Usage:
  python ingest/spac_ingest.py --local-only
  python ingest/spac_ingest.py --local-only --include-meetings
  python ingest/spac_ingest.py          # full S3 pipeline
"""
import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import posixpath

import boto3
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FILE_SERVER   = "https://archive.icjia-api.cloud/files/spac/"
OUTPUT_FILE   = "data_files/corpus/spac_corpus.jsonl"
DELAY         = 0.5
MAX_RETRIES   = 3

# Filenames matching these patterns are meeting logistics, not substantive research
_SKIP_PATTERNS = [
    re.compile(r"(?i)meeting.agenda"),
    re.compile(r"(?i)meeting.minutes"),
    re.compile(r"(?i)agenda.final"),
    re.compile(r"(?i)minutes.final"),
]

# Timestamp suffix added by the ICJIA archive system — marks duplicate copies
_TIMESTAMP_RE = re.compile(r"-\d{8}T\d+\.pdf$", re.IGNORECASE)

# Year extraction from filename
_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")

# Category inference from filename keywords
_CATEGORIES = [
    (re.compile(r"(?i)annual.report"),             "Annual Report"),
    (re.compile(r"(?i)baseline.+projection|prison.+projection|population.+projection"),
                                                   "Prison Population Projection"),
    (re.compile(r"(?i)impact.analysis"),           "Impact Analysis"),
    (re.compile(r"(?i)cost.benefit"),              "Cost-Benefit Analysis"),
    (re.compile(r"(?i)average.offender|average.joe"), "Offender Profile"),
    (re.compile(r"(?i)cannabis"),                  "Cannabis Policy"),
    (re.compile(r"(?i)retrospective"),             "Retrospective Analysis"),
    (re.compile(r"(?i)commission.recommendation"), "Commission Recommendation"),
    (re.compile(r"(?i)methodology"),               "Methodology"),
    (re.compile(r"(?i)projection.methodology"),    "Projection Methodology"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; IL-LegalRAG/1.0; "
        "legal research/non-commercial; contact: your@email.com)"
    ),
}

# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def _get_s3_config() -> tuple[str, str]:
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        log.error("RAW_S3_BUCKET is not set.")
        sys.exit(1)
    prefix = os.environ.get("SPAC_S3_PREFIX", "spac/").rstrip("/") + "/"
    return bucket, prefix


def _upload_to_s3(local_path: Path, no_upload: bool = False) -> None:
    if no_upload:
        return
    bucket, prefix = _get_s3_config()
    s3_key = f"{prefix}{local_path.name}"
    s3     = boto3.client("s3")
    log.info(f"Uploading {local_path.name} → s3://{bucket}/{s3_key}")
    s3.upload_file(str(local_path), bucket, s3_key)
    log.info("  Upload complete.")

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update(HEADERS)


def _get(url: str) -> requests.Response | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, timeout=60)
            resp.raise_for_status()
            time.sleep(DELAY)
            return resp
        except requests.HTTPError as e:
            log.warning(f"HTTP {e.response.status_code} on {url} (attempt {attempt})")
            if e.response.status_code in (429, 502, 503):
                time.sleep(15 * attempt)
            elif attempt == MAX_RETRIES:
                log.error(f"  Giving up: {url}")
                return None
        except requests.RequestException as e:
            log.warning(f"  Request error: {e} (attempt {attempt})")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(10)
    return None

# ---------------------------------------------------------------------------
# Directory enumeration
# ---------------------------------------------------------------------------

def list_pdfs(include_meetings: bool = False) -> list[dict]:
    """Fetch the file server directory listing and return de-duplicated PDF entries."""
    log.info(f"Fetching directory listing: {FILE_SERVER}")
    resp = _get(FILE_SERVER)
    if resp is None:
        log.error("Failed to fetch directory listing.")
        sys.exit(1)

    soup  = BeautifulSoup(resp.text, "html.parser")
    links = soup.find_all("a", href=re.compile(r"\.pdf$", re.I))

    # Build {canonical_name: url} — prefer non-timestamped filenames
    canonical: dict[str, str] = {}
    for a in links:
        href = a.get("href", "").strip()
        if not href or href == "..":
            continue
        # href may be an absolute path (/files/spac/name.pdf) or just a filename
        filename = posixpath.basename(href)
        if not filename:
            continue
        url = FILE_SERVER + filename

        if not include_meetings and any(p.search(filename) for p in _SKIP_PATTERNS):
            continue

        # Strip timestamp suffix to get canonical name for de-duplication
        canon = _TIMESTAMP_RE.sub(".pdf", filename)
        # Prefer the version without a timestamp (cleaner filename)
        if canon not in canonical or _TIMESTAMP_RE.search(canonical[canon]):
            canonical[canon] = url

    entries = []
    for canon_name, url in sorted(canonical.items()):
        entries.append({"filename": canon_name, "url": url})

    log.info(f"  {len(entries)} unique PDFs found (after de-duplication)")
    return entries

# ---------------------------------------------------------------------------
# Metadata inference
# ---------------------------------------------------------------------------

def _infer_title(filename: str) -> str:
    """Convert a filename into a human-readable title."""
    name = _TIMESTAMP_RE.sub("", filename)      # strip timestamp
    name = re.sub(r"\.pdf$", "", name, flags=re.I)
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _infer_year(filename: str) -> str:
    m = _YEAR_RE.search(filename)
    return m.group(1) if m else ""


def _infer_category(filename: str) -> str:
    for pattern, label in _CATEGORIES:
        if pattern.search(filename):
            return label
    return "Report"


def _make_id(filename: str) -> str:
    safe = re.sub(r"[^\w]", "_", _TIMESTAMP_RE.sub("", filename).replace(".pdf", "").lower())
    return f"spac-{safe[:80]}"

# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def _extract_pdf_text(content: bytes, filename: str) -> str:
    try:
        import pypdf
    except ImportError:
        log.error("pypdf is required. Install with: pip install pypdf")
        sys.exit(1)

    import io
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages  = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
    except Exception as e:
        log.warning(f"    PDF parse error for {filename}: {type(e).__name__}: {e}")
        return ""

    if not pages:
        log.warning(f"    No text extracted from {filename} — may be a scanned/encrypted PDF")

    return "\n\n".join(pages)

# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def run(
    output_file: str,
    no_upload: bool,
    include_meetings: bool,
    limit: int,
) -> None:
    if not no_upload:
        _get_s3_config()

    entries = list_pdfs(include_meetings)
    if limit:
        entries = entries[:limit]

    out_path  = Path(output_file)
    written   = 0
    skipped   = 0
    seen_ids: set[str] = set()

    with open(out_path, "w", encoding="utf-8") as out:
        for i, entry in enumerate(entries):
            filename = entry["filename"]
            url      = entry["url"]
            log.info(f"  [{i+1}/{len(entries)}] {filename}")

            resp = _get(url)
            if resp is None:
                log.warning(f"    Download failed — skipping")
                skipped += 1
                continue

            text = _extract_pdf_text(resp.content, filename)
            if not text.strip():
                skipped += 1
                continue

            rec_id = _make_id(filename)
            if rec_id in seen_ids:
                log.warning(f"    Duplicate ID {rec_id!r} (timestamp variant) — skipping")
                skipped += 1
                continue
            seen_ids.add(rec_id)

            record = {
                "id":          rec_id,
                "source":      "spac",
                "agency":      "Illinois Sentencing Policy Advisory Council",
                "category":    _infer_category(filename),
                "title":       _infer_title(filename),
                "year":        _infer_year(filename),
                "filename":    filename,
                "url":         url,
                "text":        text,
                "scraped_at":  datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            written += 1

    log.info(f"\nDone. {written} records written, {skipped} skipped → {out_path}")
    _upload_to_s3(out_path, no_upload)

    if not no_upload:
        bucket, prefix = _get_s3_config()
        log.info(f"S3: s3://{bucket}/{prefix}{out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest SPAC publications from ICJIA file server → JSONL"
    )
    parser.add_argument(
        "--output", default=OUTPUT_FILE,
        help=f"Output JSONL file (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--local-only", action="store_true",
        help="Write output locally without uploading to S3.",
    )
    parser.add_argument(
        "--include-meetings", action="store_true",
        help="Include meeting agendas and minutes (excluded by default).",
    )
    parser.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="Process only the first N PDFs (0 = all). For testing.",
    )
    args = parser.parse_args()

    log.info("SPAC Publication Ingestion")
    log.info(f"  Source     : {FILE_SERVER}")
    log.info(f"  Output     : {args.output}")
    log.info(f"  Meetings   : {'included' if args.include_meetings else 'excluded'}")
    log.info(f"  S3 upload  : {'disabled' if args.local_only else 'enabled'}")

    run(args.output, args.local_only, args.include_meetings, args.limit)


if __name__ == "__main__":
    main()
