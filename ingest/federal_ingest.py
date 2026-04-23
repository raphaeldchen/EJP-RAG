"""
Federal document ingestion for Illinois legal RAG corpus.

Handles four source types:
  1. Federal Register — via the public JSON API (no key required)
  2. Congress.gov bill text — HTML scraping
  3. BOP Program Statements — PDF download + text extraction (requires: pypdf)
  4. ED Dear Colleague Letters — HTML scraping

Documents to ingest are declared in FEDERAL_SOURCES below. Each entry maps
to one output record (or multiple if a PDF spans many pages as separate chunks —
this script writes one record per whole document; chunking is downstream).

Output schema (JSONL):
  id, source, doc_type, title, citation, url, text, scraped_at

Usage:
  pip install pypdf          # required for BOP PDF parsing
  python ingest/federal_ingest.py --local-only
  python ingest/federal_ingest.py --ids second-chance-pell-rule bop-5300.21
  python ingest/federal_ingest.py   # all sources, full S3 pipeline
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
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Source manifest
# ---------------------------------------------------------------------------

# doc_type values: "regulation" | "statute" | "policy" | "guidance"

FEDERAL_SOURCES: list[dict] = [
    {
        "id":       "second-chance-pell-rule",
        "title":    "Second Chance Pell Grant Final Rule (34 CFR Part 668)",
        "citation": "34 CFR Part 668",
        "doc_type": "regulation",
        "fetch":    "federal_register",
        "doc_num":  "2023-20511",   # Federal Register document number
        "url":      (
            "https://www.federalregister.gov/documents/2023/09/28/"
            "2023-20511/prison-education-programs"
        ),
    },
    {
        "id":       "first-step-act",
        "title":    "First Step Act of 2018 (P.L. 115-391)",
        "citation": "P.L. 115-391",
        "doc_type": "statute",
        "fetch":    "congress_html",
        "url":      (
            "https://www.congress.gov/bill/115th-congress/senate-bill/756/text"
        ),
    },
    {
        "id":       "bop-5300.21",
        "title":    "BOP Program Statement 5300.21 — Education, Training and Leisure Time Programs",
        "citation": "BOP PS 5300.21",
        "doc_type": "policy",
        "fetch":    "pdf",
        "url":      "https://www.bop.gov/policy/progstat/5300_021.pdf",
    },
    {
        "id":       "ed-gen-22-15",
        "title":    "ED Dear Colleague Letter GEN-22-15 — Second Chance Pell Guidance",
        "citation": "GEN-22-15",
        "doc_type": "guidance",
        "fetch":    "html",
        "url":      (
            "https://fsapartners.ed.gov/knowledge-center/library/dear-colleague-letters"
            "/2022-09-28/second-chance-pell-experimental-sites-initiative-"
            "2022-23-award-year-guidance"
        ),
    },
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DELAY          = 1.0    # be polite
DELAY_ON_ERROR = 15.0
MAX_RETRIES    = 3
OUTPUT_FILE    = "data_files/corpus/federal_corpus.jsonl"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; IL-LegalRAG-Scraper/1.0; "
        "legal research/non-commercial; contact: your@email.com)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf",
    "Accept-Language": "en-US,en;q=0.9",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def _get_s3_config() -> tuple[str, str]:
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        log.error("RAW_S3_BUCKET is not set.")
        sys.exit(1)
    prefix = os.environ.get("FEDERAL_S3_PREFIX", "federal/").rstrip("/") + "/"
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
# HTTP helpers
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update(HEADERS)


def _get(url: str, **kwargs) -> requests.Response | None:
    """GET with retry and rate limiting. Returns None on terminal failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, timeout=60, **kwargs)
            resp.raise_for_status()
            time.sleep(DELAY)
            return resp
        except requests.HTTPError as e:
            code = e.response.status_code
            log.warning(f"HTTP {code} on {url} (attempt {attempt}/{MAX_RETRIES})")
            if code in (429, 502, 503):
                time.sleep(DELAY_ON_ERROR * attempt)
            elif attempt == MAX_RETRIES:
                log.error(f"  Giving up: {url}")
                return None
        except requests.RequestException as e:
            log.warning(f"Request error ({e.__class__.__name__}): {e} (attempt {attempt})")
            if attempt == MAX_RETRIES:
                log.error(f"  Giving up: {url}")
                return None
            time.sleep(DELAY_ON_ERROR)
    return None


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    for tag in soup.find_all(["p", "br", "div", "h1", "h2", "h3", "h4", "li"]):
        tag.insert_before("\n")
    text = soup.get_text()
    return re.sub(r"\n{3,}", "\n\n", text).strip()

# ---------------------------------------------------------------------------
# Fetchers (one per fetch strategy)
# ---------------------------------------------------------------------------

def _fetch_federal_register(source: dict) -> str:
    """Use the Federal Register JSON API to get full body text."""
    doc_num  = source["doc_num"]
    api_url  = f"https://www.federalregister.gov/api/v1/documents/{doc_num}.json"
    fields   = "body_html_url,full_text_xml_url,title,citation,document_number"
    api_url += f"?fields[]={fields.replace(',', '&fields[]=')}"

    log.info(f"  [Federal Register API] {api_url}")
    resp = _get(api_url, headers={"Accept": "application/json"})
    if resp is None:
        return ""

    data         = resp.json()
    body_html_url = data.get("body_html_url") or data.get("full_text_xml_url")

    if not body_html_url:
        log.warning("  No body_html_url in API response — falling back to source URL")
        resp2 = _get(source["url"])
        return _html_to_text(resp2.text) if resp2 else ""

    log.info(f"  Fetching body HTML: {body_html_url}")
    resp2 = _get(body_html_url)
    if resp2 is None:
        return ""

    return _html_to_text(resp2.text)


def _fetch_congress_html(source: dict) -> str:
    """Scrape bill text from congress.gov.

    The congress.gov bill-text page has the content in a <pre> or <div>
    element with id='bill-summary' or similar. We fetch the text tab directly.
    """
    url  = source["url"]
    # The /text endpoint may redirect to a specific version — accept that
    resp = _get(url, allow_redirects=True)
    if resp is None:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try the main bill text container (congress.gov wraps text in
    # <div class="generated-html-container"> or <pre class="bill-text">)
    for selector in [
        {"class": "generated-html-container"},
        {"class": "bill-text"},
        {"id":    "main"},
        {"class": "legisBody"},
    ]:
        container = soup.find(["div", "pre", "section"], selector)
        if container:
            text = container.get_text(separator="\n", strip=True)
            if len(text) > 500:
                return re.sub(r"\n{3,}", "\n\n", text).strip()

    # Fallback: full-page text with nav stripped
    return _html_to_text(resp.text)


def _fetch_pdf(source: dict) -> str:
    """Download a PDF and extract text with pypdf.

    Requires: pip install pypdf
    Falls back to an empty string with a clear warning if pypdf is not installed.
    """
    try:
        import pypdf
    except ImportError:
        log.error(
            "pypdf is required for PDF ingestion. "
            "Install it with: pip install pypdf"
        )
        return ""

    import io
    url  = source["url"]
    log.info(f"  Downloading PDF: {url}")
    resp = _get(url, headers={"Accept": "application/pdf"})
    if resp is None:
        return ""

    try:
        reader = pypdf.PdfReader(io.BytesIO(resp.content))
        pages  = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
            if (i + 1) % 20 == 0:
                log.info(f"    Extracted {i + 1}/{len(reader.pages)} pages...")
    except Exception as e:
        log.warning(f"  PDF parse error: {type(e).__name__}: {e}")
        return ""

    log.info(f"  Extracted {len(pages)} pages → {sum(len(p) for p in pages):,} chars")
    return "\n\n".join(pages)


def _fetch_html(source: dict) -> str:
    """Generic HTML scraper — strips boilerplate, returns clean text."""
    resp = _get(source["url"])
    if resp is None:
        return ""
    return _html_to_text(resp.text)


_FETCHERS = {
    "federal_register": _fetch_federal_register,
    "congress_html":    _fetch_congress_html,
    "pdf":              _fetch_pdf,
    "html":             _fetch_html,
}

# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest_source(source: dict) -> dict | None:
    """Ingest a single federal source and return a JSONL record, or None on failure."""
    log.info(f"\n── {source['id']}: {source['title']}")
    fetcher = _FETCHERS.get(source["fetch"])
    if fetcher is None:
        log.error(f"  Unknown fetch strategy: {source['fetch']!r}")
        return None

    text = fetcher(source)
    if not text:
        log.warning(f"  No text extracted for {source['id']}")
        return None

    log.info(f"  {len(text):,} chars extracted")
    return {
        "id":          source["id"],
        "source":      "federal",
        "doc_type":    source["doc_type"],
        "title":       source["title"],
        "citation":    source["citation"],
        "url":         source["url"],
        "text":        text,
        "scraped_at":  datetime.now(timezone.utc).isoformat(),
    }


def run(
    source_ids: list[str] | None,
    output_file: str,
    no_upload: bool,
) -> None:
    if not no_upload:
        _get_s3_config()

    sources = FEDERAL_SOURCES
    if source_ids:
        sources = [s for s in sources if s["id"] in source_ids]
        missing = set(source_ids) - {s["id"] for s in sources}
        if missing:
            log.error(f"Unknown source IDs: {missing}")
            log.error(f"Available IDs: {[s['id'] for s in FEDERAL_SOURCES]}")
            sys.exit(1)

    records = []
    for source in sources:
        record = ingest_source(source)
        if record:
            records.append(record)

    out_path = Path(output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(f"\nWrote {len(records)}/{len(sources)} records → {out_path}")
    _upload_to_s3(out_path, no_upload)

    if not no_upload:
        bucket, prefix = _get_s3_config()
        log.info(f"S3: s3://{bucket}/{prefix}{out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest federal documents (Federal Register, Congress, BOP PDFs, ED guidance) → JSONL"
    )
    parser.add_argument(
        "--ids", nargs="+", metavar="ID",
        help=(
            "Specific source IDs to ingest. "
            f"Available: {[s['id'] for s in FEDERAL_SOURCES]}"
        ),
    )
    parser.add_argument(
        "--output", default=OUTPUT_FILE, metavar="FILE",
        help=f"Output JSONL file (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--local-only", action="store_true",
        help="Write output locally without uploading to S3.",
    )
    args = parser.parse_args()

    log.info("Federal Document Ingestion")
    log.info(f"  Sources   : {args.ids or 'all'}")
    log.info(f"  Output    : {args.output}")
    log.info(f"  S3 upload : {'disabled' if args.local_only else 'enabled'}")

    run(args.ids, args.output, args.local_only)


if __name__ == "__main__":
    main()
