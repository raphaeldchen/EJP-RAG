"""
Illinois Community College Board (ICCB) correctional education data ingestion.

Downloads ICCB Annual Student Enrollment and Completion reports, which include
data on credit programs in IDOC facilities (postsecondary education in prison).

Source: https://www.iccb.org/divisions/research-and-analytics/
Reports available FY2020–FY2025.

Requires: pip install pypdf

Output schema (JSONL):
  id, source, doc_type, fiscal_year, title, url, text, scraped_at

Usage:
  python ingest/iccb_ingest.py --local-only
  python ingest/iccb_ingest.py   # full S3 pipeline
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Report manifest
# ---------------------------------------------------------------------------

# Full annual reports (contain the IDOC facility data tables)
ANNUAL_REPORTS = [
    {"fy": "2025", "url": "https://www.iccb.org/wp-content/uploads/2025/12/Annual_Enroll_Comp_2025_Final.pdf"},
    {"fy": "2024", "url": "https://www.iccb.org/wp-content/uploads/2024/12/Annual_Enroll_Comp_2024_Final.pdf"},
    {"fy": "2023", "url": "https://www.iccb.org/wp-content/uploads/2023/12/Annual_Enroll_Comp_2023.pdf"},
    {"fy": "2022", "url": "http://www.iccb.org/wp-content/pdfs/data/Annual%20Enroll%20Comp%202022%20Final.pdf"},
    {"fy": "2021", "url": "http://www.iccb.org/wp-content/uploads/2022/03/Annual_Enroll_Comp_2021_Final.pdf"},
    {"fy": "2020", "url": "http://www.iccb.org/wp-content/uploads/2021/09/Annual_Enroll_Comp_2020_Final.pdf"},
]

# Executive summaries (shorter, higher-level overview)
EXEC_SUMMARIES = [
    {"fy": "2025", "url": "https://www.iccb.org/wp-content/uploads/2025/12/Annual_Enroll_Comp_2025_Executive_Summary.pdf"},
    {"fy": "2024", "url": "https://www.iccb.org/wp-content/uploads/2024/12/Annual_Enroll_Comp_2024_Executive_Summary.pdf"},
    {"fy": "2023", "url": "https://www.iccb.org/wp-content/uploads/2023/12/Annual_Enroll_Comp_2023_Executive_Summary.pdf"},
    {"fy": "2022", "url": "http://www.iccb.org/wp-content/pdfs/data/Annual%20Enroll%20Comp%202022%20Executive%20Summary%20Final.pdf"},
    {"fy": "2021", "url": "http://www.iccb.org/wp-content/uploads/2022/03/Annual%20Enroll%20Comp%20Executive%20Summary%202021%20Final.pdf"},
    {"fy": "2020", "url": "http://www.iccb.org/wp-content/uploads/2021/09/Annual_Enroll_Comp_Executive_Summary_2020_Final.pdf"},
]

OUTPUT_FILE = "data_files/corpus/iccb_corpus.jsonl"
DELAY       = 1.0
MAX_RETRIES = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; IL-LegalRAG/1.0; "
        "legal research/non-commercial; contact: your@email.com)"
    ),
    "Accept": "application/pdf,*/*",
}

# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def _get_s3_config() -> tuple[str, str]:
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        log.error("RAW_S3_BUCKET is not set."); sys.exit(1)
    prefix = os.environ.get("ICCB_S3_PREFIX", "iccb/").rstrip("/") + "/"
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

def _get(url: str) -> requests.Response | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, timeout=120)
            resp.raise_for_status()
            time.sleep(DELAY)
            return resp
        except requests.HTTPError as e:
            log.warning(f"HTTP {e.response.status_code} (attempt {attempt}): {url}")
            if e.response.status_code in (429, 502, 503):
                time.sleep(20 * attempt)
            elif attempt == MAX_RETRIES:
                log.error(f"  Giving up: {url}"); return None
        except requests.RequestException as e:
            log.warning(f"Error: {e} (attempt {attempt})")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(15)
    return None

# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _pdf_to_text(content: bytes, label: str) -> str:
    try:
        import pypdf
    except ImportError:
        log.error("pypdf required: pip install pypdf"); sys.exit(1)
    import io
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages  = [p.extract_text() or "" for p in reader.pages]
    except Exception as e:
        log.warning(f"  PDF parse error for {label}: {type(e).__name__}: {e}")
        return ""
    text   = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        log.warning(f"  No text extracted from {label} — may be scanned/image PDF")
    return text

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_reports(include_summaries: bool) -> list[dict]:
    reports = list(ANNUAL_REPORTS)
    if include_summaries:
        reports += EXEC_SUMMARIES

    records = []
    for entry in reports:
        fy    = entry["fy"]
        url   = entry["url"]
        is_summary = entry in EXEC_SUMMARIES
        label = f"FY{fy} {'Executive Summary' if is_summary else 'Annual Report'}"
        log.info(f"  {label}: {url}")

        resp = _get(url)
        if resp is None:
            log.warning(f"    Download failed — skipping {label}")
            continue

        text = _pdf_to_text(resp.content, label)
        if not text.strip():
            continue

        records.append({
            "id":          f"iccb-fy{fy}-{'summary' if is_summary else 'annual'}",
            "source":      "iccb",
            "doc_type":    "executive_summary" if is_summary else "annual_report",
            "fiscal_year": fy,
            "title":       label,
            "url":         url,
            "text":        text,
            "scraped_at":  datetime.now(timezone.utc).isoformat(),
        })
        log.info(f"    {len(text):,} chars extracted")

    return records

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ICCB Annual Enrollment & Completion reports → JSONL"
    )
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--local-only", action="store_true",
                        help="Write locally without S3 upload.")
    parser.add_argument("--include-summaries", action="store_true",
                        help="Also download executive summary PDFs (default: annual reports only).")
    args = parser.parse_args()

    if not args.local_only:
        _get_s3_config()

    log.info("ICCB Correctional Education Report Ingestion")
    log.info(f"  Reports    : FY2020–FY2025 annual reports")
    log.info(f"  Summaries  : {'included' if args.include_summaries else 'excluded'}")
    log.info(f"  Output     : {args.output}")

    records  = ingest_reports(args.include_summaries)
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(f"\nWrote {len(records)} records → {out_path}")
    _upload_to_s3(out_path, args.local_only)

if __name__ == "__main__":
    main()
