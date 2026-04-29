"""
Illinois Department of Corrections (IDOC) ingestion.

Two source types handled by this script:

1. IDOC Administrative Directives (policies.html)
   All 392 directives are available from a single AEM JSON endpoint:
     https://idoc.illinois.gov/content/soi/idoc/en/aboutus/policies/policies/
       jcr:content/responsivegrid/container/container_293684588/container/
       data_table_assets.datatableassets.json
   Each directive is a PDF. By default only categories relevant to criminal
   justice are downloaded (education, programs, medical, classification, etc.).
   Use --all-categories to download all 392.

2. IDOC Reentry / Community Resources page
   https://idoc.illinois.gov/communityresources.html
   Simple HTML page with reentry guides and resource links.

Requires: pip install pypdf

Output schema (JSONL):
  id, source, doc_type, category, sub_category, title, url, text, scraped_at

Usage:
  python ingest/idoc_ingest.py --local-only
  python ingest/idoc_ingest.py --all-categories --local-only
  python ingest/idoc_ingest.py --skip-directives --local-only   # reentry only
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
from urllib.parse import quote, urljoin

import boto3
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IDOC_BASE    = "https://idoc.illinois.gov"
DIRECTIVES_JSON_URL = (
    "https://idoc.illinois.gov/content/soi/idoc/en/aboutus/policies/policies"
    "/jcr:content/responsivegrid/container/container_293684588/container"
    "/data_table_assets.datatableassets.json"
)
REENTRY_URL  = "https://idoc.illinois.gov/communityresources.html"
OUTPUT_FILE  = "data_files/corpus/idoc_corpus.jsonl"
DELAY        = 0.75
MAX_RETRIES  = 3

# Categories most relevant to criminal justice research.
# The directive numbering scheme maps to categories:
#   1xxxxx = Administration, Organization & Management
#   2xxxxx = Fiscal & Business Management
#   3xxxxx = Personnel & Labor Relations
#   4xxxxx = Programs & Services  ← most relevant
#   5xxxxx = Operations           ← partially relevant
RELEVANT_SUBCATEGORIES = {
    "EDUCATION AND VOCATIONAL SERVICES",
    "PROGRAMS AND SERVICES",
    "MEDICAL AND HEALTH CARE",
    "MENTAL HEALTH",
    "SUBSTANCE ABUSE",
    "CLASSIFICATION",
    "RECEPTION AND ORIENTATION",
    "INDIVIDUAL IN CUSTODY RECORDS",
    "OFFENDER MANAGEMENT AND MOVEMENT",
    "ADULT TRANSITION CENTERS",
    "FIELD OPERATIONS",
    "LIBRARY SERVICES AND LEGAL MATERIAL",
    "RELIGIOUS ISSUES",
    "LEISURE TIME ACTIVITIES/RECREATION",
    "INTERNAL INVESTIGATIONS",
    "RESEARCH AND EVALUATION",
    "PLACEMENT RESOURCE UNIT",
    "STAFF DEVELOPMENT AND TRAINING",
}

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
    "Accept": "text/html,application/json,application/pdf",
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def _get_s3_config() -> tuple[str, str]:
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        log.error("RAW_S3_BUCKET is not set.")
        sys.exit(1)
    prefix = os.environ.get("IDOC_S3_PREFIX", "idoc/").rstrip("/") + "/"
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
            log.warning(f"  Error: {e} (attempt {attempt})")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(10)
    return None

# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _pdf_to_text(content: bytes, label: str) -> str:
    try:
        import pypdf
    except ImportError:
        log.error("pypdf required: pip install pypdf")
        sys.exit(1)
    import io
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages  = [page.extract_text() or "" for page in reader.pages]
    except Exception as e:
        log.warning(f"  PDF parse error for {label}: {type(e).__name__}: {e}")
        return ""
    text   = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        log.warning(f"  No text extracted from {label} — may be scanned image PDF")
    return text

# ---------------------------------------------------------------------------
# Source 1: IDOC Administrative Directives
# ---------------------------------------------------------------------------

def _fetch_directive_manifest() -> list[dict]:
    """Fetch the AEM JSON endpoint and return all directive entries."""
    log.info(f"Fetching directive manifest: {DIRECTIVES_JSON_URL}")
    resp = _get(DIRECTIVES_JSON_URL)
    if resp is None:
        log.error("Failed to fetch directive manifest.")
        return []

    payload = resp.json()
    rows    = payload.get("data", [])
    log.info(f"  {len(rows)} directives in manifest")

    entries = []
    for row in rows:
        # Row format: [[title, /path/to.pdf, "true"], "CATEGORY", "SUB-CATEGORY"]
        if not isinstance(row, list) or len(row) < 3:
            continue
        cell0 = row[0]
        if isinstance(cell0, list) and len(cell0) >= 2:
            title    = str(cell0[0]).strip()
            pdf_path = str(cell0[1]).strip()
        else:
            continue
        category     = str(row[1]).strip()
        sub_category = str(row[2]).strip()
        # Build full URL — path may contain unencoded spaces
        pdf_url = IDOC_BASE + pdf_path.replace(" ", "%20")
        entries.append({
            "title":        title,
            "pdf_url":      pdf_url,
            "category":     category,
            "sub_category": sub_category,
        })
    return entries


def ingest_directives(all_categories: bool) -> list[dict]:
    entries = _fetch_directive_manifest()
    if not all_categories:
        before = len(entries)
        entries = [e for e in entries if e["sub_category"] in RELEVANT_SUBCATEGORIES]
        log.info(f"  Filtered to {len(entries)}/{before} relevant directives")

    records = []
    for i, entry in enumerate(entries):
        log.info(f"  [{i+1}/{len(entries)}] {entry['title']}")
        resp = _get(entry["pdf_url"])
        if resp is None:
            log.warning("    Download failed — skipping")
            continue

        text = _pdf_to_text(resp.content, entry["title"])
        if not text.strip():
            continue

        # Directive number from title (e.g. "403103 Individual In Custody...")
        dir_num = re.match(r"^(\d+)", entry["title"])
        rec_id  = f"idoc-dir-{dir_num.group(1)}" if dir_num else (
            "idoc-dir-" + re.sub(r"[^\w]", "_", entry["title"])[:60].lower()
        )
        rec = {
            "id":           rec_id,
            "source":       "idoc_directive",
            "doc_type":     "administrative_directive",
            "category":     entry["category"],
            "sub_category": entry["sub_category"],
            "title":        entry["title"],
            "url":          entry["pdf_url"],
            "text":         text,
            "scraped_at":   datetime.now(timezone.utc).isoformat(),
        }
        # Keep longest-text copy when the same directive number appears twice
        existing = next((r for r in records if r["id"] == rec_id), None)
        if existing is None:
            records.append(rec)
        elif len(text) > len(existing["text"]):
            records[records.index(existing)] = rec
        else:
            log.warning(f"    Duplicate directive ID {rec_id!r} — keeping existing copy")

    log.info(f"  Directives ingested: {len(records)}")
    return records

# ---------------------------------------------------------------------------
# Source 2: IDOC Reentry / Community Resources
# ---------------------------------------------------------------------------

_NOISE_TAGS = ["nav", "header", "footer", "script", "style", "noscript"]


def ingest_reentry() -> list[dict]:
    log.info(f"Fetching IDOC reentry page: {REENTRY_URL}")
    resp = _get(REENTRY_URL)
    if resp is None:
        log.error("Failed to fetch reentry page.")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    main = soup.find("main") or soup.body
    if main is None:
        return []

    for tag in main.find_all(["p", "li", "h1", "h2", "h3", "h4"]):
        tag.insert_before("\n")

    text = re.sub(r"\n{3,}", "\n\n", main.get_text()).strip()

    # Also collect any linked sub-pages on the reentry section
    records = [{
        "id":           "idoc-reentry-hub",
        "source":       "idoc_reentry",
        "doc_type":     "reentry_resource",
        "category":     "REENTRY",
        "sub_category": "HUB",
        "title":        "IDOC Reentry Resources",
        "url":          REENTRY_URL,
        "text":         text,
        "scraped_at":   datetime.now(timezone.utc).isoformat(),
    }]

    # Follow any reentry sub-links within idoc.illinois.gov
    sub_links = [
        a["href"] for a in soup.find_all("a", href=True)
        if "idoc.illinois.gov" in a.get("href", "")
        and "/reentry" in a.get("href", "")
        and a["href"] != REENTRY_URL
    ]
    for url in list(dict.fromkeys(sub_links)):   # dedupe
        log.info(f"  Sub-page: {url}")
        sub_resp = _get(url)
        if sub_resp is None:
            continue
        sub_soup = BeautifulSoup(sub_resp.text, "html.parser")
        for tag in sub_soup.find_all(_NOISE_TAGS):
            tag.decompose()
        sub_main = sub_soup.find("main") or sub_soup.body
        if sub_main is None:
            continue
        for tag in sub_main.find_all(["p", "li", "h1", "h2", "h3"]):
            tag.insert_before("\n")
        sub_text = re.sub(r"\n{3,}", "\n\n", sub_main.get_text()).strip()
        title    = sub_soup.title.string.strip() if sub_soup.title else url
        title    = re.sub(r"\s*-\s*IDOC.*$", "", title).strip()
        if len(sub_text) > 200:
            slug = re.sub(r"[^\w]", "_", url.split("/")[-1] or "page")[:50]
            records.append({
                "id":           f"idoc-reentry-{slug}",
                "source":       "idoc_reentry",
                "doc_type":     "reentry_resource",
                "category":     "REENTRY",
                "sub_category": "SUB-PAGE",
                "title":        title,
                "url":          url,
                "text":         sub_text,
                "scraped_at":   datetime.now(timezone.utc).isoformat(),
            })

    log.info(f"  Reentry records: {len(records)}")
    return records

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    output_file: str,
    no_upload: bool,
    all_categories: bool,
    skip_directives: bool,
    skip_reentry: bool,
) -> None:
    if not no_upload:
        _get_s3_config()

    records: list[dict] = []

    if not skip_directives:
        records.extend(ingest_directives(all_categories))

    if not skip_reentry:
        records.extend(ingest_reentry())

    out_path = Path(output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(f"\nWrote {len(records)} records → {out_path}")
    _upload_to_s3(out_path, no_upload)

    if not no_upload:
        bucket, prefix = _get_s3_config()
        log.info(f"S3: s3://{bucket}/{prefix}{out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest IDOC Administrative Directives + Reentry Resources → JSONL"
    )
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help=f"Output file (default: {OUTPUT_FILE})")
    parser.add_argument("--local-only", action="store_true",
                        help="Write locally without S3 upload.")
    parser.add_argument("--all-categories", action="store_true",
                        help="Download all 392 directives (default: relevant categories only).")
    parser.add_argument("--skip-directives", action="store_true",
                        help="Skip administrative directives, ingest reentry only.")
    parser.add_argument("--skip-reentry", action="store_true",
                        help="Skip reentry page, ingest directives only.")
    args = parser.parse_args()

    log.info("IDOC Ingestion")
    log.info(f"  Directives  : {'all categories' if args.all_categories else 'relevant only'}")
    log.info(f"  Output      : {args.output}")
    log.info(f"  S3 upload   : {'disabled' if args.local_only else 'enabled'}")

    run(args.output, args.local_only, args.all_categories,
        args.skip_directives, args.skip_reentry)


if __name__ == "__main__":
    main()
