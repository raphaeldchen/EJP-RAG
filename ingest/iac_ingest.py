"""
Illinois Administrative Code (JCAR) scraper.

Scrapes Title 20 (Corrections, Criminal Justice, and Law Enforcement) from
ILGA's JCAR publication via the EntirePart endpoint. By default targets
15 parts relevant to IDOC operations, inmate rights, and reentry.

Additional parts can be added via --parts or DEFAULT_PARTS below.

Output schema (JSONL, one record per section):
  id, source, title_num, title_name, part_num, part_name,
  section_num, section_heading, section_citation, url, text, scraped_at

Usage:
  python ingest/iac_ingest.py --local-only
  python ingest/iac_ingest.py --parts 420 470 503 --local-only
  python ingest/iac_ingest.py          # full S3 pipeline
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
# Constants
# ---------------------------------------------------------------------------

JCAR_BASE       = "https://www.ilga.gov/agencies/JCAR"
ENTIREPART_URL  = f"{JCAR_BASE}/EntirePart?titlepart="
TITLE_NUM       = "20"
TITLE_NAME      = "Corrections, Criminal Justice, and Law Enforcement"
TITLE_PAD       = "020"   # 3-digit zero-padded title for PartID construction

# Default Title 20 parts relevant to criminal justice research.
# PartID = TITLE_PAD (3) + part_number.zfill(5) (5) = 8 chars.
DEFAULT_PARTS: dict[str, str] = {
    "107":  "Records of Offenders",
    "120":  "Rules of Conduct",
    "405":  "School District #428 (Correctional Education)",
    "410":  "Legal Programs for Committed Persons",
    "415":  "Health Care",
    "420":  "Assignment of Committed Persons",
    "425":  "Chaplaincy Services and Religious Practices",
    "430":  "Library Services and Legal Materials",
    "455":  "Work Release Programs",
    "470":  "Release of Committed Persons",
    "503":  "Classification and Transfers",
    "504":  "Discipline and Grievances",
    "525":  "Rights and Privileges",
    "1205": "Expungement Procedures",
    "1610": "Prisoner Review Board",
}

DELAY          = 0.75   # seconds between requests
DELAY_ON_ERROR = 15.0
MAX_RETRIES    = 3

OUTPUT_FILE = "data_files/corpus/iac_corpus.jsonl"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; IL-AdminCode-RAG-Scraper/1.0; "
        "legal research/non-commercial; contact: your@email.com)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# S3 helpers (mirrors ilga_ingest pattern)
# ---------------------------------------------------------------------------

def _get_s3_config() -> tuple[str, str]:
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        log.error("RAW_S3_BUCKET is not set.")
        sys.exit(1)
    prefix = os.environ.get("IAC_S3_PREFIX", "iac/").rstrip("/") + "/"
    return bucket, prefix


def _upload_to_s3(local_path: Path, no_upload: bool = False) -> None:
    if no_upload:
        return
    bucket, prefix = _get_s3_config()
    s3_key = f"{prefix}{local_path.name}"
    s3 = boto3.client("s3")
    size_mb = local_path.stat().st_size / 1_048_576
    log.info(f"Uploading {local_path.name} ({size_mb:.1f} MB) → s3://{bucket}/{s3_key}")
    s3.upload_file(str(local_path), bucket, s3_key)
    log.info(f"  Upload complete.")

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
            code = e.response.status_code
            log.warning(f"HTTP {code} on {url} (attempt {attempt}/{MAX_RETRIES})")
            if code in (429, 502, 503):
                time.sleep(DELAY_ON_ERROR * attempt)
            elif attempt == MAX_RETRIES:
                log.error(f"  Giving up: {url}")
                return None
        except requests.RequestException as e:
            log.warning(f"Request error: {e} (attempt {attempt}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES:
                log.error(f"  Giving up: {url}")
                return None
            time.sleep(DELAY_ON_ERROR)
    return None

# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

def _entirepart_url(part_num: str) -> str:
    """Return the JCAR EntirePart URL for a given part.

    PartID = title_pad (3 digits) + part_num.zfill(5).
    e.g. Part 420 → titlepart=02000420
    """
    part_id = TITLE_PAD + part_num.zfill(5)
    return f"{ENTIREPART_URL}{part_id}"

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# JCAR section headings follow:  "Section 401.10  Purpose"
# The part number prefix (401) and two-digit suffix (.10) may vary.
_SECTION_RE = re.compile(
    r"(?m)^[ \t]*Section\s+(\d{2,4}\.\d+(?:\.\d+)?)\s+(.*?)(?=^[ \t]*Section\s+\d|\Z)",
    re.DOTALL | re.IGNORECASE,
)

# JCAR footer markers to strip
_FOOTER_MARKERS = [
    "Illinois General Assembly",
    "Legislative Information System",
    "Contact ILGA",
    "ILGA.gov",
]


def _strip_footer(text: str) -> str:
    earliest = len(text)
    for marker in _FOOTER_MARKERS:
        idx = text.find(marker)
        if 0 < idx < earliest:
            earliest = idx
    return text[:earliest].strip()


def _parse_inline(text: str, part_num: str, source_url: str) -> list[dict]:
    """Extract sections from an inline text dump of a JCAR page."""
    text = _strip_footer(text)
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        log.warning(f"    No section headings matched in {source_url}")
        return []

    sections = []
    for m in matches:
        sec_num = m.group(1).strip()
        body    = m.group(2).strip()

        # First non-blank line is the section title
        lines   = [l for l in body.splitlines() if l.strip()]
        heading = lines[0].strip().rstrip(".") if lines else ""
        rest    = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        full_text = f"Section {sec_num}  {heading}\n{rest}".strip()

        sections.append({
            "section_num":     sec_num,
            "section_heading": heading,
            "text":            full_text,
            "url":             source_url,
        })

    return sections


def _make_id(part_num: str, section_num: str) -> str:
    safe = section_num.replace(".", "_")
    return f"iac-t{TITLE_NUM}-p{part_num}-s{safe}"


def _build_records(
    raw: list[dict], part_num: str, part_name: str
) -> list[dict]:
    records = []
    for sec in raw:
        if not sec.get("text", "").strip():
            continue
        records.append({
            "id":               _make_id(part_num, sec["section_num"]),
            "source":           "illinois_admin_code",
            "title_num":        TITLE_NUM,
            "title_name":       TITLE_NAME,
            "part_num":         part_num,
            "part_name":        part_name,
            "section_num":      sec["section_num"],
            "section_heading":  sec["section_heading"],
            "section_citation": f"{TITLE_NUM} Ill. Adm. Code {sec['section_num']}",
            "url":              sec["url"],
            "text":             sec["text"],
            "scraped_at":       datetime.now(timezone.utc).isoformat(),
        })
    return records

# ---------------------------------------------------------------------------
# Per-part scraping
# ---------------------------------------------------------------------------

def scrape_part(part_num: str, part_name: str, debug: bool = False) -> list[dict]:
    url = _entirepart_url(part_num)
    log.info(f"\n  ── Part {part_num}: {part_name}")
    log.info(f"     {url}")

    soup = _fetch(url)
    if soup is None:
        log.error(f"     Failed to fetch Part {part_num}")
        return []

    if debug:
        log.info(f"     [DEBUG] HTML (first 3000 chars):\n{soup.prettify()[:3000]}")

    page_text = soup.get_text(separator="\n", strip=True)
    log.info(f"     {len(page_text):,} chars")
    raw = _parse_inline(page_text, part_num, url)

    records = _build_records(raw, part_num, part_name)
    log.info(f"     {len(records)} sections extracted")
    return records

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def scrape(
    parts: dict[str, str],
    output_file: str,
    no_upload: bool,
    debug: bool,
) -> None:
    if not no_upload:
        _get_s3_config()   # fail fast if env vars missing

    all_records: list[dict] = []
    for part_num, part_name in parts.items():
        all_records.extend(scrape_part(part_num, part_name, debug))

    out_path = Path(output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(f"\nWrote {len(all_records)} sections → {out_path}")
    _upload_to_s3(out_path, no_upload)

    if not no_upload:
        bucket, prefix = _get_s3_config()
        log.info(f"S3: s3://{bucket}/{prefix}{out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Illinois Administrative Code (JCAR) → JSONL"
    )
    parser.add_argument(
        "--parts", nargs="+", metavar="NUM",
        help=(
            "Part numbers to scrape (Title 20). "
            f"Default: {list(DEFAULT_PARTS.keys())}"
        ),
    )
    parser.add_argument(
        "--part-names", nargs="+", metavar="NAME",
        help=(
            "Human-readable names for --parts (same order). "
            "Defaults to 'Part NNN' if omitted."
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
    parser.add_argument(
        "--debug", action="store_true",
        help="Print raw HTML to diagnose page structure issues.",
    )
    args = parser.parse_args()

    if args.parts:
        names = args.part_names or []
        parts = {
            p: (names[i] if i < len(names) else f"Part {p}")
            for i, p in enumerate(args.parts)
        }
    else:
        parts = DEFAULT_PARTS

    log.info("Illinois Administrative Code Scraper")
    log.info(f"  Title       : {TITLE_NUM} ({TITLE_NAME})")
    log.info(f"  Parts       : {list(parts.keys())}")
    log.info(f"  Output      : {args.output}")
    log.info(f"  S3 upload   : {'disabled' if args.local_only else 'enabled'}")

    scrape(parts, args.output, args.local_only, args.debug)


if __name__ == "__main__":
    main()
