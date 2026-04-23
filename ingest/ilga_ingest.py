import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs
import boto3
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

BASE_URL     = "https://ilga.gov"
CHAPTERS_URL = f"{BASE_URL}/Legislation/ILCS/Chapters"

DELAY          = 0.75   # seconds between requests — be polite to ILGA servers
DELAY_ON_ERROR = 15.0
MAX_RETRIES    = 3

OUTPUT_FILE = "data_files/corpus/ilcs_corpus.jsonl"
LOG_FILE    = "ilcs_scraper.log"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ILCS-RAG-Scraper/1.0; "
        "legal research/non-commercial; contact: your@email.com)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

def get_s3_config() -> tuple[str, str]:
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        log.error("RAW_S3_BUCKET environment variable is not set.")
        sys.exit(1)
    prefix = os.environ.get("ILCS_S3_PREFIX", "ilcs/").rstrip("/") + "/"
    return bucket, prefix

def upload_to_s3(local_path: Path, no_upload: bool = False) -> bool:
    if no_upload:
        return True
    bucket, prefix = get_s3_config()
    s3_key = f"{prefix}{local_path.name}"
    try:
        s3 = boto3.client("s3")
        size_mb = local_path.stat().st_size / 1_048_576
        log.info(f"Uploading {local_path.name} ({size_mb:.1f} MB) → s3://{bucket}/{s3_key}")
        s3.upload_file(str(local_path), bucket, s3_key)
        log.info(f"  Upload complete: s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        log.error(f"  S3 upload failed: {e}")
        return False

session = requests.Session()
session.headers.update(HEADERS)

def fetch(url: str) -> BeautifulSoup | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            time.sleep(DELAY)
            return BeautifulSoup(resp.text, "html.parser")
        except requests.HTTPError as e:
            code = e.response.status_code
            log.warning(f"HTTP {code} on {url} (attempt {attempt}/{MAX_RETRIES})")
            if code in (429, 502, 503):
                wait = DELAY_ON_ERROR * attempt
                log.info(f"  Rate-limited — waiting {wait:.0f}s")
                time.sleep(wait)
            elif attempt == MAX_RETRIES:
                log.error(f"  Giving up: {url}")
                return None
        except requests.RequestException as e:
            log.warning(f"Request error on {url}: {e} (attempt {attempt}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES:
                log.error(f"  Giving up: {url}")
                return None
            time.sleep(DELAY_ON_ERROR)
    return None

CITATION_RE   = re.compile(r"\(\s*(\d+)\s+ILCS\s+([\w/.\-]+)\)", re.IGNORECASE)
SEC_MARKER_RE = re.compile(r"\bSec\.\s+[\w./\-]+\.", re.IGNORECASE)

# FIX 1: HEADING_RE — original pattern stopped at the first \n, so multi-line
# headings like "Short\ntitle." (produced when get_text splits across inline
# elements) were truncated to just the first word.
#
# The new pattern uses re.DOTALL so (.*?) can cross newlines, then stops at
# the first period followed by whitespace-or-end.  After extraction we collapse
# internal whitespace so "Short\ntitle" becomes "Short title".
HEADING_RE = re.compile(
    r"Sec\.\s+[\w./\-]+\.\s*(.*?)\.",
    re.IGNORECASE | re.DOTALL,
)

FOOTER_MARKERS = [
    "This site is maintained for the Illinois General Assembly",
    "Legislative Information System",
    "Contact ILGA Webmaster",
    "ILGA.gov | All Rights Reserved",
    "This site contains provisions of the Illinois Compiled Statutes",
]

def strip_footer(text: str) -> str:
    earliest = len(text)
    for marker in FOOTER_MARKERS:
        idx = text.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    result = text[:earliest].strip()
    # Sanity check: if stripping removed more than 20% of the text and the result
    # is suspiciously short, log a warning — the page structure may have changed.
    if earliest < len(text) * 0.8 and len(result) < 500:
        log.warning(
            "strip_footer removed >20%% of page text and result is very short "
            "(%d chars) — page structure may have changed", len(result)
        )
    return result

def _has_sec_marker_nearby(text: str, cite_end: int) -> bool:
    return bool(SEC_MARKER_RE.search(text[cite_end: cite_end + 300]))

def split_sections(page_text: str, source_url: str) -> list[dict]:
    all_matches    = list(CITATION_RE.finditer(page_text))
    section_starts = [m for m in all_matches if _has_sec_marker_nearby(page_text, m.end())]
    if not section_starts:
        log.debug(f"  No valid section openers found: {source_url}")
        return []
    sections = []
    for i, m in enumerate(section_starts):
        start = m.start()
        end   = section_starts[i + 1].start() if i + 1 < len(section_starts) else len(page_text)
        block = page_text[start:end].strip()
        chapter_num = m.group(1)
        section_num = m.group(2)
        citation    = f"{chapter_num} ILCS {section_num}"
        heading = ""
        hm = HEADING_RE.search(block)
        if hm:
            # Collapse newlines/extra whitespace — multi-word headings can span
            # multiple lines in the raw get_text() output (e.g. "Short\ntitle")
            heading = re.sub(r"\s+", " ", hm.group(1)).strip().rstrip(".")
        is_repealed = bool(re.search(r"\(Repealed\)", block, re.I))
        sections.append({
            "section_citation": citation,
            "section_num":      section_num,
            "section_heading":  heading,
            "text":             block,
            "is_repealed":      is_repealed,
        })
    return sections

def clean_act_name(name: str) -> str:
    name = name.replace("\xa0", " ")
    name = re.sub(r"^\d+\s+ILCS\s+[\w/]+\s*", "", name).strip()
    name = re.sub(r"\s+", " ", name).strip().rstrip(".")
    return name

def extract_act_name(soup: BeautifulSoup, fallback: str) -> str:
    h2 = soup.find("h2")
    if h2:
        txt = clean_act_name(h2.get_text(separator=" ", strip=True))
        if len(txt) > 5:
            return txt
    text = soup.get_text()
    m = re.search(r"may be cited as the\s+(.+?)(?:\.|Act\.|\n)", text, re.I)
    if m:
        name = clean_act_name(m.group(1))
        if name:
            return name
    return clean_act_name(fallback)

def make_id(chapter_num: str, act_id: str, section_citation: str) -> str:
    sec = re.sub(r"^\d+\s+ILCS\s+", "", section_citation)
    sec = sec.replace("/", "_S").replace("-", "_D").replace(".", "_P")
    return f"ch{chapter_num}-act{act_id}-sec{sec}"

def parse_chapters(soup: BeautifulSoup) -> list[dict]:
    results = []
    for a in soup.find_all("a", href=re.compile(r"ChapterID=\d+", re.I)):
        href   = a.get("href", "")
        url    = urljoin(BASE_URL, href)
        params = parse_qs(urlparse(url).query)
        results.append({
            "url":          url,
            "chapter_id":   params.get("ChapterID",    [""])[0],
            "chapter_num":  params.get("ChapterNumber", [""])[0],
            "chapter_name": params.get("Chapter",       [""])[0],
            "major_topic":  params.get("MajorTopic",    [""])[0],
        })
    return results

def parse_acts(soup: BeautifulSoup, chapter: dict, debug: bool) -> list[dict]:
    if debug:
        log.info(f"  [DEBUG] Acts page HTML (first 3000 chars):\n{soup.prettify()[:3000]}")
    results  = []
    seen_ids = set()
    for a in soup.find_all("a", href=re.compile(r"ActID=\d+", re.I)):
        href   = a.get("href", "")
        url    = urljoin(BASE_URL, href)
        params = parse_qs(urlparse(url).query)
        act_id = params.get("ActID", [""])[0]
        if not act_id or act_id in seen_ids:
            continue
        seen_ids.add(act_id)
        articles_url = (
            f"{BASE_URL}/Legislation/ILCS/Articles"
            f"?ActID={act_id}"
            f"&ChapterID={chapter['chapter_id']}"
            f"&Chapter={chapter['chapter_name']}"
            f"&MajorTopic={chapter['major_topic']}"
        )
        results.append({
            "act_id":       act_id,
            "act_name":     a.get_text(strip=True),
            "articles_url": articles_url,
        })
    return results

def is_type_b(soup: BeautifulSoup) -> bool:
    return bool(soup.find_all("a", class_="list-group-item",
                               href=re.compile(r"details", re.I)))

def parse_article_links(art_soup: BeautifulSoup) -> list[dict]:
    # Priority 1: look for a single full-text link
    fulltext_links = []
    for a in art_soup.find_all("a", class_="list-group-item",
                                href=re.compile(r"FullText|details", re.I)):
        href = a.get("href", "")
        label = a.get_text(strip=True)
        if "FullText" in href or "View Entire" in label:
            url = urljoin(BASE_URL, href)
            fulltext_links.append({"url": url, "article_name": label})

    if fulltext_links:
        # Use only the first full-text link — it covers the whole act
        log.debug(f"  Using full-text link: {fulltext_links[0]['url']}")
        return fulltext_links[:1]

    # Priority 2: fall back to per-article detail links (no FullText available)
    results   = []
    seen_urls = set()
    for a in art_soup.find_all("a", class_="list-group-item",
                                href=re.compile(r"details", re.I)):
        href = a.get("href", "")
        url = urljoin(BASE_URL, href)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        span  = a.find("span")
        raw   = span.get_text(separator=" ", strip=True) if span else a.get_text(strip=True)
        label = re.sub(r"\s+", " ", raw).strip()
        results.append({"url": url, "article_name": label})
    return results

def build_docs(sections: list[dict], act: dict, act_name: str,
               chapter: dict, source_url: str,
               article_name: str | None, skip_repealed: bool) -> list[dict]:
    docs = []
    for sec in sections:
        if skip_repealed and sec["is_repealed"]:
            continue
        doc = {
            "id":               make_id(chapter["chapter_num"], act["act_id"],
                                        sec["section_citation"]),
            "chapter_num":      chapter["chapter_num"],
            "chapter_name":     chapter["chapter_name"],
            "major_topic":      chapter["major_topic"],
            "act_name":         act_name,
            "act_id":           act["act_id"],
            "section_citation": sec["section_citation"],
            "section_num":      sec["section_num"],
            "section_heading":  sec["section_heading"],
            "url":              source_url,
            "text":             sec["text"],
            "scraped_at":       datetime.now(timezone.utc).isoformat(),
        }
        if article_name:
            doc["article_name"] = article_name
        docs.append(doc)
    return docs

def scrape_type_a(art_soup: BeautifulSoup, act: dict, chapter: dict,
                  skip_repealed: bool) -> list[dict]:
    act_name  = extract_act_name(art_soup, act["act_name"])
    page_text = strip_footer(art_soup.get_text(separator="\n", strip=True))
    sections  = split_sections(page_text, act["articles_url"])
    return build_docs(sections, act, act_name, chapter,
                      act["articles_url"], None, skip_repealed)

def scrape_type_b(art_soup: BeautifulSoup, act: dict, chapter: dict,
                  debug: bool, skip_repealed: bool,
                  limit: int | None, current_total: int) -> list[dict]:
    article_links = parse_article_links(art_soup)
    act_name      = extract_act_name(art_soup, act["act_name"])
    log.info(f"    [Type B] {len(article_links)} unique article pages to fetch")
    if debug:
        log.info(f"    [DEBUG] TOC HTML (first 3000 chars):\n{art_soup.prettify()[:3000]}")
    all_docs = []
    for article in article_links:
        if limit is not None and (current_total + len(all_docs)) >= limit:
            break
        log.info(f"    ── {article['article_name'][:70]}")
        detail_soup = fetch(article["url"])
        if not detail_soup:
            log.warning(f"      Fetch failed: {article['url']}")
            continue
        if debug:
            log.info(f"      [DEBUG] Detail HTML (first 3000 chars):\n{detail_soup.prettify()[:3000]}")
        page_text = strip_footer(detail_soup.get_text(separator="\n", strip=True))
        sections  = split_sections(page_text, article["url"])
        log.info(f"      {len(sections)} sections found")
        all_docs.extend(build_docs(sections, act, act_name, chapter,
                                   article["url"], article["article_name"],
                                   skip_repealed))
    return all_docs

def clean_docs(raw_docs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for doc in raw_docs:
        key = (doc["section_citation"], doc["act_id"])
        groups[key].append(doc)

    cleaned    = []
    dupe_count = 0
    for recs in groups.values():
        best = max(recs, key=lambda r: len(r.get("text", "")))
        cleaned.append(best)
        dupe_count += len(recs) - 1

    if dupe_count:
        log.info(f"  Cleanup: removed {dupe_count} duplicate sections")

    return cleaned

def load_done_acts(output_file: str) -> set[str]:
    sidecar = output_file + ".done_acts"
    done = set()
    if Path(sidecar).exists():
        with open(sidecar, encoding="utf-8") as f:
            for line in f:
                act_id = line.strip()
                if act_id:
                    done.add(act_id)
    return done

def mark_act_done(output_file: str, act_id: str):
    sidecar = output_file + ".done_acts"
    with open(sidecar, "a", encoding="utf-8") as f:
        f.write(act_id + "\n")

def scrape(
    filter_chapters: list[str] | None = None,
    debug: bool = False,
    limit: int | None = None,
    skip_repealed: bool = True,
    output_file: str = OUTPUT_FILE,
    no_upload: bool = False,
    upload_every: int | None = None,
):
    done_acts = load_done_acts(output_file)
    if done_acts:
        log.info(f"Resuming — {len(done_acts)} acts already completed")
    if not no_upload:
        get_s3_config()
    total_sections  = 0
    total_acts      = 0
    acts_since_last_upload = 0
    log.info(f"Fetching chapters: {CHAPTERS_URL}")
    soup = fetch(CHAPTERS_URL)
    if not soup:
        log.error("Failed to fetch chapters page. Check network connection.")
        sys.exit(1)
    chapters = parse_chapters(soup)
    log.info(f"Found {len(chapters)} chapters")
    if filter_chapters:
        chapters = [c for c in chapters if c["chapter_num"] in filter_chapters]
        log.info(f"Filtered to {len(chapters)} chapter(s): {[c['chapter_num'] for c in chapters]}")
    if not chapters:
        log.error("No chapters matched filter. Exiting.")
        sys.exit(1)
    with open(output_file, "a", encoding="utf-8") as out:
        for ch in chapters:
            log.info(f"\n══ Chapter {ch['chapter_num']}: {ch['chapter_name']}")
            acts_soup = fetch(ch["url"])
            if not acts_soup:
                log.warning(f"  Skipping chapter {ch['chapter_num']} — fetch failed")
                continue
            acts = parse_acts(acts_soup, ch, debug)
            log.info(f"  Found {len(acts)} acts")
            for act in acts:
                if act["act_id"] in done_acts:
                    log.debug(f"  Skipping completed act {act['act_id']}")
                    continue
                if limit is not None and total_sections >= limit:
                    log.info(f"Reached section limit of {limit}. Stopping.")
                    upload_to_s3(Path(output_file), no_upload)
                    return
                log.info(f"  ── Act {act['act_id']}: {act['act_name'][:70]}")
                art_soup = fetch(act["articles_url"])
                if not art_soup:
                    log.warning(f"    Fetch failed — skipping act {act['act_id']}")
                    continue
                if is_type_b(art_soup):
                    log.info(f"    [Type B] TOC page detected")
                    raw_docs = scrape_type_b(
                        art_soup, act, ch, debug, skip_repealed,
                        limit=limit, current_total=total_sections
                    )
                else:
                    log.info(f"    [Type A] Inline text page")
                    raw_docs = scrape_type_a(art_soup, act, ch, skip_repealed)
                docs = clean_docs(raw_docs)
                if not docs:
                    log.warning(
                        f"    No sections extracted from act {act['act_id']} "
                        f"({act['act_name'][:50]}). Try --debug to inspect HTML."
                    )
                for doc in docs:
                    out.write(json.dumps(doc) + "\n")
                out.flush()
                mark_act_done(output_file, act["act_id"])
                done_acts.add(act["act_id"])
                total_acts             += 1
                total_sections         += len(docs)
                acts_since_last_upload += 1
                log.info(f"    Wrote {len(docs)} sections  (running total: {total_sections})")
                if upload_every and acts_since_last_upload >= upload_every:
                    log.info(f"  [{acts_since_last_upload} acts since last upload] uploading snapshot...")
                    upload_to_s3(Path(output_file), no_upload)
                    acts_since_last_upload = 0
    log.info(f"\n{'='*60}")
    log.info(f"Scrape complete.")
    log.info(f"  Acts processed  : {total_acts}")
    log.info(f"  Sections written: {total_sections}")
    log.info(f"  Output          : {output_file}")
    upload_to_s3(Path(output_file), no_upload)
    if not no_upload:
        bucket, prefix = get_s3_config()
        log.info(f"  S3 location     : s3://{bucket}/{prefix}{Path(output_file).name}")

def main():
    parser = argparse.ArgumentParser(
        description="Scrape Illinois Compiled Statutes into a cleaned JSONL RAG corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chapters", nargs="+", metavar="NUM",
        help="Chapter numbers to scrape (e.g. --chapters 720 725 730). Default: all.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print raw HTML snippets at each level to diagnose issues.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Stop after N total sections (for testing).",
    )
    parser.add_argument(
        "--keep-repealed", action="store_true",
        help="Include sections marked (Repealed). Default: skip them.",
    )
    parser.add_argument(
        "--output", default=OUTPUT_FILE, metavar="FILE",
        help=f"Output JSONL file (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--local-only", action="store_true",
        help="Skip S3 upload and write locally only.",
    )
    parser.add_argument(
        "--upload-every", type=int, default=None, metavar="N",
        help="Upload an incremental snapshot to S3 every N acts (default: final upload only).",
    )
    global DELAY
    parser.add_argument(
        "--delay", type=float, default=DELAY, metavar="SECS",
        help=f"Seconds between HTTP requests (default: {DELAY})",
    )
    args = parser.parse_args()
    DELAY = args.delay
    log.info("ILCS Scraper")
    log.info(f"  Chapters     : {args.chapters or 'ALL'}")
    log.info(f"  Output       : {args.output}")
    log.info(f"  Request delay: {DELAY}s")
    log.info(f"  Skip repealed: {not args.keep_repealed}")
    log.info(f"  S3 upload    : {'disabled' if args.local_only else 'enabled (on completion)'}")
    if args.upload_every:
        log.info(f"  Upload every : {args.upload_every} acts")
    if args.limit:
        log.info(f"  Section limit: {args.limit}")
    scrape(
        filter_chapters=args.chapters,
        debug=args.debug,
        limit=args.limit,
        skip_repealed=not args.keep_repealed,
        output_file=args.output,
        no_upload=args.local_only,
        upload_every=args.upload_every,
    )

if __name__ == "__main__":
    main()