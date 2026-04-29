"""
Illinois CAP bulk case law download from static.case.law.

The CAP API was sunset on September 5, 2024. Full case data is available at
https://static.case.law/{reporter}/{vol}/cases/{file_name}.json

Each volume has a CasesMetadata.json index used for date pre-filtering,
then individual case JSON files are downloaded for qualifying cases.

Illinois reporters included by default:
  ill        Illinois Supreme Court, 1st series     ~334 vols
  ill-2d     Illinois Supreme Court, 2nd series     ~242 vols
  ill-app    Illinois Appellate Court, 1st series   ~310 vols
  ill-app-2d Illinois Appellate Court, 2nd series   ~133 vols
  ill-app-3d Illinois Appellate Court, 3rd series   ~297 vols

Checkpoint: completed case IDs are recorded in <output>.done so interrupted
runs resume without duplicating already-written records.

Output schema (JSONL) — same as cap_bulk_ingest.py:
  id, source, reporter, volume, case_id, case_name, case_name_abbr, court,
  court_id, jurisdiction, date_decided, citations, docket_number, url, text,
  scraped_at

Usage:
  python ingest/cap_static_download.py --local-only
  python ingest/cap_static_download.py --after 1980-01-01 --local-only
  python ingest/cap_static_download.py --reporters ill ill-2d --local-only
  python ingest/cap_static_download.py --limit 200 --local-only   # quick test
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

STATIC_ROOT       = "https://static.case.law"
OUTPUT_FILE       = "data_files/corpus/cap_bulk_corpus.jsonl"
CHECKPOINT_SUFFIX  = ".done"
PROGRESS_SUFFIX    = ".progress"
DELAY             = 0.15   # seconds between requests
DELAY_ERROR       = 15.0
MAX_RETRIES       = 3

# Reporter slug → human-readable description
REPORTER_COURT = {
    "ill":        "Illinois Supreme Court (1st series)",
    "ill-2d":     "Illinois Supreme Court (2nd series)",
    "ill-app":    "Illinois Appellate Court (1st series)",
    "ill-app-2d": "Illinois Appellate Court (2nd series)",
    "ill-app-3d": "Illinois Appellate Court (3rd series)",
}

DEFAULT_REPORTERS = list(REPORTER_COURT.keys())

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; IL-LegalRAG/1.0; "
        "legal research/non-commercial; contact: rdchen3@illinois.edu)"
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
    prefix = os.environ.get("CAP_S3_PREFIX", "cap/").rstrip("/") + "/"
    return bucket, prefix


def _upload_to_s3(local_path: Path, local_only: bool = False) -> None:
    if local_only:
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
            resp = _session.get(url, timeout=60)
            resp.raise_for_status()
            time.sleep(DELAY)
            return resp
        except requests.HTTPError as e:
            code = e.response.status_code
            if code == 404:
                return None
            log.warning(f"HTTP {code} on {url} (attempt {attempt})")
            if code in (429, 502, 503):
                time.sleep(DELAY_ERROR * attempt)
            elif attempt == MAX_RETRIES:
                log.error(f"  Giving up: {url}")
                return None
        except requests.RequestException as e:
            log.warning(f"  Request error: {e} (attempt {attempt})")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(DELAY_ERROR)
    return None

# ---------------------------------------------------------------------------
# Volume listing
# ---------------------------------------------------------------------------

def list_volumes(reporter: str) -> list[int]:
    """Parse the reporter directory page and return sorted list of volume numbers."""
    url  = f"{STATIC_ROOT}/{reporter}/"
    resp = _get(url)
    if resp is None:
        log.error(f"  Cannot fetch volume listing for {reporter}")
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    vols = []
    for a in soup.find_all("a", href=True):
        # hrefs are absolute: https://static.case.law/ill-2d/42/
        segment = a["href"].rstrip("/").rsplit("/", 1)[-1]
        if re.fullmatch(r"\d+", segment):
            vols.append(int(segment))
    vols = sorted(set(vols))
    log.info(f"  {reporter}: {len(vols)} volumes")
    return vols

# ---------------------------------------------------------------------------
# Case metadata + text extraction
# ---------------------------------------------------------------------------

def fetch_cases_metadata(reporter: str, vol: int) -> list[dict]:
    """Fetch CasesMetadata.json for a volume. Returns list of case metadata dicts."""
    url  = f"{STATIC_ROOT}/{reporter}/{vol}/CasesMetadata.json"
    resp = _get(url)
    if resp is None:
        return []
    try:
        data = resp.json()
        return data if isinstance(data, list) else []
    except (ValueError, KeyError):
        return []


def fetch_case(reporter: str, vol: int, file_name: str) -> dict | None:
    """Download a single case JSON file.

    file_name from CasesMetadata omits the .json extension (e.g. "0021-01").
    """
    if not file_name.endswith(".json"):
        file_name = file_name + ".json"
    url  = f"{STATIC_ROOT}/{reporter}/{vol}/cases/{file_name}"
    resp = _get(url)
    if resp is None:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def _extract_text(case: dict) -> str:
    """Extract opinion text from a CAP case record.

    Handles two formats:
      - Per-file format (static.case.law): casebody.opinions[]
      - Bulk zip format (cap_bulk_ingest):  casebody.data.opinions[]
    """
    try:
        casebody = case.get("casebody", {})

        # Bulk zip format has a nested .data dict
        data = casebody.get("data")
        if isinstance(data, str):
            return re.sub(r"<[^>]+>", " ", data).strip()
        elif isinstance(data, dict):
            opinions = data.get("opinions", [])
        else:
            # Per-file format: opinions at casebody level
            opinions = casebody.get("opinions", [])

        if not opinions:
            return ""

        ordered = sorted(
            opinions,
            key=lambda o: 0 if o.get("type", "") in ("majority", "unanimous") else 1,
        )
        parts = []
        for op in ordered:
            op_type = op.get("type", "").replace("_", " ").title()
            text    = op.get("text", "").strip()
            if text:
                parts.append(f"[{op_type}]\n{text}")
        return "\n\n".join(parts)

    except (KeyError, TypeError, AttributeError):
        return ""


def _extract_citations(case: dict) -> list[str]:
    return [c.get("cite", "") for c in case.get("citations", []) if c.get("cite")]


def _build_record(case: dict, reporter: str, volume: int) -> dict | None:
    text = _extract_text(case)
    if not text.strip():
        return None
    court      = case.get("court", {})
    # Per-file format uses numeric id; bulk format uses slug — store whichever is present
    court_id   = court.get("slug", "") or str(court.get("id", ""))
    court_name = court.get("name_abbreviation", "") or court.get("name", "")
    return {
        "id":             f"cap-{case['id']}",
        "source":         "cap_bulk",
        "reporter":       reporter,
        "volume":         volume,
        "case_id":        str(case["id"]),
        "case_name":      case.get("name", ""),
        "case_name_abbr": case.get("name_abbreviation", ""),
        "court":          court_name,
        "court_id":       court_id,
        "jurisdiction":   case.get("jurisdiction", {}).get("name", "Illinois"),
        "date_decided":   case.get("decision_date", ""),
        "citations":      _extract_citations(case),
        "docket_number":  case.get("docket_number", ""),
        "url":            case.get("frontend_url", "") or case.get("url", ""),
        "text":           text,
        "scraped_at":     datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(output_file: str) -> set[str]:
    cp = Path(output_file + CHECKPOINT_SUFFIX)
    if not cp.exists():
        return set()
    with open(cp, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _save_checkpoint(output_file: str, case_id: str) -> None:
    cp = Path(output_file + CHECKPOINT_SUFFIX)
    with open(cp, "a", encoding="utf-8") as f:
        f.write(case_id + "\n")


def _load_progress(output_file: str) -> dict:
    p = Path(output_file + PROGRESS_SUFFIX)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_progress(output_file: str, progress: dict) -> None:
    p = Path(output_file + PROGRESS_SUFFIX)
    tmp = p.with_suffix(".progress.tmp")
    tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    tmp.replace(p)  # atomic on POSIX


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def run(
    reporters: list[str],
    output_file: str,
    local_only: bool,
    after: str | None,
    before: str | None,
    limit: int,
) -> None:
    if not local_only:
        _get_s3_config()

    done_ids      = _load_checkpoint(output_file)
    progress      = _load_progress(output_file)
    total_written = 0
    total_skipped_date = total_skipped_text = total_skipped_done = total_failed = 0
    limit_hit     = False

    if done_ids:
        log.info(f"  Resuming — {len(done_ids):,} cases already written")

    with open(output_file, "a", encoding="utf-8") as out:
        for reporter in reporters:
            if limit_hit:
                break

            log.info(f"\n── Reporter: {reporter}  ({REPORTER_COURT[reporter]})")
            vols = list_volumes(reporter)

            rep_prog = {"total_volumes": len(vols), "done_volumes": 0, "done_cases": 0}
            progress[reporter] = rep_prog

            for vol in vols:
                if limit_hit:
                    break

                case_metas = fetch_cases_metadata(reporter, vol)
                if not case_metas:
                    rep_prog["done_volumes"] += 1
                    _save_progress(output_file, progress)
                    continue

                vol_written = 0
                for meta in case_metas:
                    if limit_hit:
                        break

                    case_id   = str(meta.get("id", ""))
                    file_name = meta.get("file_name", "")
                    if not case_id or not file_name:
                        continue

                    # Checkpoint skip
                    if case_id in done_ids:
                        total_skipped_done += 1
                        continue

                    # Date pre-filter using metadata (avoids downloading the full case)
                    date_decided = meta.get("decision_date", "")
                    if after and date_decided and date_decided < after:
                        total_skipped_date += 1
                        continue
                    if before and date_decided and date_decided > before:
                        total_skipped_date += 1
                        continue

                    # Download full case JSON
                    case = fetch_case(reporter, vol, file_name)
                    if case is None:
                        total_failed += 1
                        continue

                    rec = _build_record(case, reporter, vol)
                    if rec is None:
                        total_skipped_text += 1
                        continue

                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    _save_checkpoint(output_file, case_id)
                    done_ids.add(case_id)
                    total_written += 1
                    vol_written   += 1

                    if total_written % 500 == 0:
                        log.info(
                            f"  {total_written:,} written | "
                            f"{total_skipped_date:,} out-of-range | "
                            f"{total_skipped_text:,} no-text"
                        )

                    if limit and total_written >= limit:
                        log.info(f"  --limit {limit} reached, stopping.")
                        limit_hit = True

                rep_prog["done_volumes"] += 1
                rep_prog["done_cases"]   += vol_written
                _save_progress(output_file, progress)

                if vol_written:
                    out.flush()
                    log.info(f"  {reporter}/{vol}: +{vol_written} cases")

    log.info(
        f"\nDone. {total_written:,} records written → {output_file}\n"
        f"  Skipped: {total_skipped_date:,} out of date range | "
        f"{total_skipped_text:,} no opinion text | "
        f"{total_skipped_done:,} already done | "
        f"{total_failed:,} download failed"
    )
    _upload_to_s3(Path(output_file), local_only)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def status(output_file: str) -> None:
    """Print progress report from the local corpus and checkpoint files; no network access."""
    from collections import Counter, defaultdict

    corpus_path = Path(output_file)
    checkpoint_path = Path(output_file + CHECKPOINT_SUFFIX)
    progress = _load_progress(output_file)

    if not corpus_path.exists():
        print(f"Corpus file not found: {corpus_path}")
        return

    checkpoint_count = 0
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            checkpoint_count = sum(1 for l in f if l.strip())

    reporter_counts: Counter = Counter()
    reporter_dates: defaultdict = defaultdict(list)
    unresolved_count = 0  # legacy records whose reporter couldn't be inferred
    last_record: dict | None = None

    # Citation patterns ordered most-specific first to avoid false matches.
    # Each pattern anchors on the reporter abbreviation only — the more-specific
    # patterns (3d, 2d) must come before the base "Ill. App." and "Ill." patterns
    # so they short-circuit before the broader ones can match.
    # No trailing \b after periods — a period followed by whitespace has no
    # word boundary, so \bIll\.\b would never match "Ill. 456".
    _CITATION_REPORTER_PATTERNS = [
        (re.compile(r"\bIll\.\s+App\.\s+3d\s"),  "ill-app-3d"),
        (re.compile(r"\bIll\.\s+App\.\s+2d\s"),  "ill-app-2d"),
        (re.compile(r"\bIll\.\s+App\."),          "ill-app"),
        (re.compile(r"\bIll\.\s+2d\s"),           "ill-2d"),
        (re.compile(r"\bIll\."),                  "ill"),
    ]

    def _infer_reporter(citations: list) -> str | None:
        for cite in (citations or []):
            for pattern, slug in _CITATION_REPORTER_PATTERNS:
                if pattern.search(cite):
                    return slug
        return None

    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rep = d.get("reporter") or _infer_reporter(d.get("citations", []))
            if rep:
                reporter_counts[rep] += 1
                date = d.get("date_decided", "")
                if date:
                    reporter_dates[rep].append(date)
            else:
                unresolved_count += 1
            last_record = d

    total = sum(reporter_counts.values()) + unresolved_count
    print(f"\nCAP ingest progress — {corpus_path}")
    print(f"  Cases in corpus   : {total:,}")
    print(f"  Checkpoint entries: {checkpoint_count:,}")
    if unresolved_count:
        print(f"  Unresolved        : {unresolved_count:,}  (no reporter field and no parseable citation)")
    print()

    REPORTER_META = {
        "ill":        "IL Supreme Ct 1st series  (1819–1955)",
        "ill-2d":     "IL Supreme Ct 2nd series  (1955–2011)",
        "ill-app":    "IL Appellate 1st series   (historical–1979)",
        "ill-app-2d": "IL Appellate 2nd series   (~1968–2001)",
        "ill-app-3d": "IL Appellate 3rd series   (~1972–2011)",
    }
    print("  By reporter:")
    for rep in DEFAULT_REPORTERS:
        n = reporter_counts.get(rep, 0)
        dates = sorted(reporter_dates.get(rep, []))
        rep_prog = progress.get(rep)

        if rep_prog:
            done_v  = rep_prog.get("done_volumes", 0)
            total_v = rep_prog.get("total_volumes", 0)
            pct     = f"{100 * done_v / total_v:.0f}%" if total_v else "?%"
            vol_str = f"  [{done_v}/{total_v} vols  {pct}]"
        else:
            vol_str = ""

        if n == 0:
            case_str = "not started"
        else:
            case_str = f"{n:,} cases  {dates[0]} → {dates[-1]}" if dates else f"{n:,} cases"

        label = REPORTER_META.get(rep, rep)
        print(f"    {rep:<12}  {label:<42}  {case_str}{vol_str}")

    if last_record:
        print()
        print(f"  Last written: {last_record.get('id')}  |  {last_record.get('reporter', '?')} vol {last_record.get('volume', '?')}  |  {last_record.get('court')}  |  {last_record.get('date_decided')}")
        print(f"                {last_record.get('case_name_abbr', last_record.get('case_name', ''))[:80]}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download Illinois CAP case law from static.case.law → JSONL"
        )
    )
    parser.add_argument(
        "--reporters", nargs="+", default=DEFAULT_REPORTERS,
        metavar="SLUG",
        help=(
            "Reporter slugs to download. "
            f"Default: all IL Supreme + Appellate ({DEFAULT_REPORTERS})"
        ),
    )
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help=f"Output JSONL file (default: {OUTPUT_FILE})")
    parser.add_argument("--local-only", action="store_true",
                        help="Write locally without uploading to S3.")
    parser.add_argument("--after", metavar="YYYY-MM-DD",
                        help="Only include cases decided on or after this date.")
    parser.add_argument("--before", metavar="YYYY-MM-DD",
                        help="Only include cases decided on or before this date.")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="Stop after N records (0 = no limit). For testing.")
    parser.add_argument("--status", action="store_true",
                        help="Print progress report from local files (no network) and exit.")
    args = parser.parse_args()

    if args.status:
        status(args.output)
        return

    for slug in args.reporters:
        if slug not in REPORTER_COURT:
            log.error(f"Unknown reporter: {slug!r}. Available: {list(REPORTER_COURT)}")
            sys.exit(1)

    log.info("CAP Static Download")
    log.info(f"  Reporters   : {args.reporters}")
    log.info(f"  Output      : {args.output}")
    log.info(f"  Date range  : {args.after or 'beginning'} → {args.before or 'end'}")
    log.info(f"  Limit       : {args.limit or 'none'}")
    log.info(f"  S3 upload   : {'disabled' if args.local_only else 'enabled'}")

    run(
        reporters   = args.reporters,
        output_file = args.output,
        local_only  = args.local_only,
        after       = args.after,
        before      = args.before,
        limit       = args.limit,
    )


if __name__ == "__main__":
    main()
