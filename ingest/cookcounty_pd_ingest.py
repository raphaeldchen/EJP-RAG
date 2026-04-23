"""
Cook County Public Defender resources ingestion.

Crawls the Cook County Public Defender website, following resource sub-pages
from the resources hub. Targets pages most relevant to Illinois criminal law:
expungement/sealing, juvenile court, problem-solving courts, and the FAQ.

Skips external links, PDF downloads, and navigation-only pages.

Output schema (JSONL):
  id, source, page_title, category, url, text, scraped_at

Usage:
  python ingest/cookcounty_pd_ingest.py --local-only
  python ingest/cookcounty_pd_ingest.py   # full S3 pipeline
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
from urllib.parse import urljoin, urlparse

import boto3
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL      = "https://www.cookcountypublicdefender.org"
RESOURCES_URL = f"{BASE_URL}/resources"
OUTPUT_FILE   = "data_files/corpus/cookcounty_pd_corpus.jsonl"
DELAY         = 0.75
MAX_RETRIES   = 3

# Only follow links that stay on the same domain
ALLOWED_DOMAIN = "cookcountypublicdefender.org"

# Paths to skip — external tools, forms, and non-content pages
_SKIP_PATH_PATTERNS = [
    re.compile(r"/locations"),
    re.compile(r"/careers"),
    re.compile(r"/news"),          # press releases, not legal reference
    re.compile(r"/about"),
    re.compile(r"/contact"),
    re.compile(r"/search"),
    re.compile(r"^/$"),            # homepage
]

# Infer category from URL path or page title
_CATEGORIES = [
    (re.compile(r"(?i)expung|seal|record"),  "Expungement & Records Relief"),
    (re.compile(r"(?i)juvenile"),            "Juvenile Justice"),
    (re.compile(r"(?i)problem.solving|drug.court|mental.health.court"),
                                             "Problem-Solving Courts"),
    (re.compile(r"(?i)faq|frequently"),      "FAQ"),
    (re.compile(r"(?i)immigration"),         "Immigration"),
    (re.compile(r"(?i)jail|prison|inmate"),  "Detention & Incarceration"),
    (re.compile(r"(?i)appeal"),              "Appeals"),
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
    "Accept": "text/html,application/xhtml+xml",
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
    prefix = os.environ.get("COOKCOUNTY_PD_S3_PREFIX", "cookcounty-pd/").rstrip("/") + "/"
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


def _fetch(url: str) -> BeautifulSoup | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, timeout=30)
            resp.raise_for_status()
            time.sleep(DELAY)
            return BeautifulSoup(resp.text, "html.parser")
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
# Link filtering
# ---------------------------------------------------------------------------

def _is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if ALLOWED_DOMAIN not in parsed.netloc:
        return False
    if parsed.path.endswith(".pdf"):
        return False
    path = parsed.path.rstrip("/") or "/"
    if any(p.search(path) for p in _SKIP_PATH_PATTERNS):
        return False
    return True


def _discover_links(soup: BeautifulSoup, base: str) -> list[str]:
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        full = urljoin(base, href)
        # Normalise — drop fragment and trailing slash
        parsed = urlparse(full)
        clean  = parsed._replace(fragment="", query="").geturl().rstrip("/")
        if _is_allowed(clean):
            urls.append(clean)
    return list(dict.fromkeys(urls))   # preserve order, dedupe

# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

# Tags that contain nav/header/footer boilerplate
_NOISE_TAGS = ["nav", "header", "footer", "script", "style",
               "noscript", "aside", "form"]

# Minimum text length to bother writing a record
_MIN_TEXT_LEN = 200


def _extract_text(soup: BeautifulSoup) -> str:
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    # Try to isolate main content area
    main = (
        soup.find("main")
        or soup.find(id=re.compile(r"main|content|body", re.I))
        or soup.find(class_=re.compile(r"main|content|entry|article", re.I))
        or soup.body
    )
    if main is None:
        return ""

    for tag in main.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "dt", "dd"]):
        tag.insert_before("\n")

    text = main.get_text()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _infer_category(url: str, title: str) -> str:
    combined = url + " " + title
    for pattern, label in _CATEGORIES:
        if pattern.search(combined):
            return label
    return "General"


def _make_id(url: str) -> str:
    path = urlparse(url).path.strip("/").replace("/", "-")
    return f"ccpd-{path[:80]}" if path else "ccpd-home"

# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

def crawl(max_pages: int = 50) -> list[dict]:
    """BFS crawl from the resources hub. Returns list of JSONL records."""
    to_visit  = [RESOURCES_URL]
    visited   = set()
    records   = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        log.info(f"  [{len(visited)}/{max_pages}] {url}")
        soup = _fetch(url)
        if soup is None:
            continue

        title = soup.title.string.strip() if soup.title else url
        # Clean common suffixes from page titles
        title = re.sub(r"\s*[|\-–]\s*Law Office.*$", "", title).strip()

        text = _extract_text(soup)
        if len(text) >= _MIN_TEXT_LEN:
            records.append({
                "id":          _make_id(url),
                "source":      "cook_county_public_defender",
                "page_title":  title,
                "category":    _infer_category(url, title),
                "url":         url,
                "text":        text,
                "scraped_at":  datetime.now(timezone.utc).isoformat(),
            })
        else:
            log.info(f"    Skipping — too short ({len(text)} chars)")

        # Discover next pages from this page
        new_links = _discover_links(soup, url)
        for link in new_links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)

    log.info(f"  Crawl complete: {len(records)} pages with content from {len(visited)} visited")
    return records

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(output_file: str, no_upload: bool, max_pages: int) -> None:
    if not no_upload:
        _get_s3_config()

    records  = crawl(max_pages)
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
        description="Crawl Cook County Public Defender website → JSONL"
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
        "--max-pages", type=int, default=50, metavar="N",
        help="Maximum number of pages to crawl (default: 50).",
    )
    args = parser.parse_args()

    log.info("Cook County Public Defender Crawler")
    log.info(f"  Start URL  : {RESOURCES_URL}")
    log.info(f"  Max pages  : {args.max_pages}")
    log.info(f"  Output     : {args.output}")
    log.info(f"  S3 upload  : {'disabled' if args.local_only else 'enabled'}")

    run(args.output, args.local_only, args.max_pages)


if __name__ == "__main__":
    main()
