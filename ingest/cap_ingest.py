"""
Harvard Caselaw Access Project (CAP) ingestion for Illinois opinions.

Fetches Illinois cases via the CAP REST API with pagination and checkpoint-based
resumption. Outputs one JSONL record per case, using a schema compatible with
the existing CourtListener opinion format so the same chunk pipeline can handle
both sources.

Requires:
  CASELAW_API_TOKEN=...  in .env  (register at https://case.law/user/register/)

Output schema (JSONL, one record per case):
  id, source, case_id, case_name, court, court_id, jurisdiction,
  date_decided, citations, docket_number, url, text, scraped_at

Usage:
  python ingest/cap_ingest.py --local-only --limit 500
  python ingest/cap_ingest.py --local-only                # all Illinois cases
  python ingest/cap_ingest.py                             # full S3 pipeline

Notes:
  - The CAP bulk download (https://case.law/bulk/download/) covers opinions
    through 2018. Cases after 2018 are not available via CAP — use CourtListener
    for more recent opinions.
  - Illinois jurisdiction slug in CAP is "ill". Appellate opinions are in the
    same jurisdiction; court_id distinguishes supreme vs appellate vs circuit.
  - The CAP API rate-limits unauthenticated requests heavily. An API token is
    effectively required for bulk access.
  - With a standard token, full_case=true is needed to get opinion text.
    Some case bodies may be restricted (pre-2018 copyrighted reporters) —
    those cases are skipped (no text returned).
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
# Constants
# ---------------------------------------------------------------------------

CAP_API_BASE  = "https://api.case.law/v1"
JURISDICTION  = "ill"           # CAP jurisdiction slug for Illinois
PAGE_SIZE     = 100             # max cases per API page
DELAY         = 0.5             # seconds between paginated requests
OUTPUT_FILE   = "data_files/corpus/cap_corpus.jsonl"
CHECKPOINT_SUFFIX = ".cap_done_pages"

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
    prefix = os.environ.get("CAP_S3_PREFIX", "cap/").rstrip("/") + "/"
    return bucket, prefix


def _upload_to_s3(local_path: Path, no_upload: bool = False) -> None:
    if no_upload:
        return
    bucket, prefix = _get_s3_config()
    s3_key = f"{prefix}{local_path.name}"
    s3     = boto3.client("s3")
    size_mb = local_path.stat().st_size / 1_048_576
    log.info(f"Uploading {local_path.name} ({size_mb:.1f} MB) → s3://{bucket}/{s3_key}")
    s3.upload_file(str(local_path), bucket, s3_key)
    log.info("  Upload complete.")

# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def _make_session(api_token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Token {api_token}",
        "Accept":        "application/json",
        "User-Agent":    (
            "Mozilla/5.0 (compatible; IL-LegalRAG/1.0; "
            "legal research/non-commercial; contact: your@email.com)"
        ),
    })
    return session


def _get_page(
    session: requests.Session, url: str, params: dict | None = None
) -> dict | None:
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=60)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 30))
                log.warning(f"  Rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(DELAY)
            return resp.json()
        except requests.HTTPError as e:
            log.warning(f"  HTTP {e.response.status_code} on {url} (attempt {attempt})")
            if attempt == max_retries:
                log.error(f"  Giving up: {url}")
                return None
            time.sleep(5 * attempt)
        except (requests.RequestException, ValueError) as e:
            log.warning(f"  Error ({e.__class__.__name__}): {e} (attempt {attempt})")
            if attempt == max_retries:
                log.error(f"  Giving up: {url}")
                return None
            time.sleep(5 * attempt)
    return None

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text(case: dict) -> str:
    """Pull plain text from a CAP case record.

    CAP returns opinions under case['casebody']['data']['opinions']
    Each opinion has 'type' (majority, concurrence, dissent, etc.) and 'text'.
    We concatenate all opinions in order; majority opinion first if present.
    """
    try:
        casebody = case.get("casebody", {})
        data     = casebody.get("data") or {}

        # CAP v2 API: data may be a string (XML) or dict with 'opinions'
        if isinstance(data, str):
            # Strip XML/HTML tags for plain text
            return re.sub(r"<[^>]+>", " ", data)

        opinions = data.get("opinions", [])
        if not opinions:
            return ""

        # Put majority/lead opinion first
        ordered = sorted(
            opinions,
            key=lambda o: (0 if o.get("type", "") in ("majority", "unanimous") else 1),
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


def _build_record(case: dict) -> dict | None:
    """Convert a CAP API case object to a JSONL record."""
    text = _extract_text(case)
    if not text.strip():
        return None   # no opinion text (restricted or stub)

    court     = case.get("court", {})
    court_id  = court.get("slug", "")
    court_name = court.get("name_abbreviation", "") or court.get("name", "")

    return {
        "id":            f"cap-{case['id']}",
        "source":        "cap",
        "case_id":       str(case["id"]),
        "case_name":     case.get("name", ""),
        "case_name_abbr": case.get("name_abbreviation", ""),
        "court":         court_name,
        "court_id":      court_id,
        "jurisdiction":  JURISDICTION,
        "date_decided":  case.get("decision_date", ""),
        "citations":     _extract_citations(case),
        "docket_number": case.get("docket_number", ""),
        "url":           case.get("frontend_url", ""),
        "text":          text,
        "scraped_at":    datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(output_file: str) -> set[str]:
    """Return the set of CAP case IDs already written."""
    cp = Path(output_file + CHECKPOINT_SUFFIX)
    if not cp.exists():
        return set()
    with open(cp, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _save_checkpoint(output_file: str, case_id: str) -> None:
    cp = Path(output_file + CHECKPOINT_SUFFIX)
    with open(cp, "a", encoding="utf-8") as f:
        f.write(case_id + "\n")

# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

def run(
    output_file: str,
    no_upload: bool,
    limit: int,
    after_date: str | None,
    before_date: str | None,
) -> None:
    api_token = os.environ.get("CASELAW_API_TOKEN", "").strip()
    if not api_token:
        log.error(
            "CASELAW_API_TOKEN is not set in .env. "
            "Register at https://case.law/user/register/ to get a free token."
        )
        sys.exit(1)

    if not no_upload:
        _get_s3_config()

    session   = _make_session(api_token)
    done_ids  = _load_checkpoint(output_file)
    if done_ids:
        log.info(f"Resuming — {len(done_ids):,} cases already written")

    # Build initial query params
    params: dict = {
        "jurisdiction": JURISDICTION,
        "full_case":    "true",
        "ordering":     "decision_date",    # stable pagination
        "page_size":    PAGE_SIZE,
    }
    if after_date:
        params["decision_date__gte"] = after_date
    if before_date:
        params["decision_date__lte"] = before_date

    out_path   = Path(output_file)
    next_url   = f"{CAP_API_BASE}/cases/"
    total      = 0
    page_num   = 0
    skipped    = 0

    log.info(f"\nFetching Illinois cases from CAP API...")
    log.info(f"  Jurisdiction : {JURISDICTION}")
    log.info(f"  Date range   : {after_date or 'beginning'} → {before_date or 'end'}")
    log.info(f"  Limit        : {limit or 'none'}")

    with open(out_path, "a", encoding="utf-8") as out:
        while next_url:
            page_num += 1
            log.info(f"  Page {page_num}  (written so far: {total:,})")

            data = _get_page(session, next_url, params if page_num == 1 else None)
            if data is None:
                log.error("  API request failed — stopping.")
                break

            # After the first page, CAP provides full next URLs in data['next']
            # so we clear params to avoid double-applying them
            params = {}

            results = data.get("results", [])
            if not results:
                log.info("  No results — done.")
                break

            for case in results:
                case_id = str(case.get("id", ""))

                if case_id in done_ids:
                    skipped += 1
                    continue

                record = _build_record(case)
                if record is None:
                    skipped += 1
                    continue

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                _save_checkpoint(output_file, case_id)
                done_ids.add(case_id)
                total += 1

                if limit and total >= limit:
                    log.info(f"  Reached limit of {limit} cases — stopping.")
                    next_url = None
                    break

            if next_url is None:
                break

            next_url = data.get("next")
            if next_url:
                time.sleep(DELAY)

    log.info(f"\nDone.")
    log.info(f"  Cases written : {total:,}")
    log.info(f"  Cases skipped : {skipped:,}  (no text or already done)")
    log.info(f"  Output        : {out_path}")
    _upload_to_s3(out_path, no_upload)

    if not no_upload:
        bucket, prefix = _get_s3_config()
        log.info(f"  S3            : s3://{bucket}/{prefix}{out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Harvard Caselaw Access Project (Illinois) → JSONL"
    )
    parser.add_argument(
        "--output", default=OUTPUT_FILE, metavar="FILE",
        help=f"Output JSONL file (default: {OUTPUT_FILE}). Appended to if it already exists.",
    )
    parser.add_argument(
        "--local-only", action="store_true",
        help="Write output locally without uploading to S3.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="Stop after N cases (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--after", metavar="YYYY-MM-DD",
        help="Only fetch cases decided on or after this date.",
    )
    parser.add_argument(
        "--before", metavar="YYYY-MM-DD",
        help="Only fetch cases decided on or before this date.",
    )
    args = parser.parse_args()

    log.info("Harvard Caselaw Access Project Ingestion")
    log.info(f"  Jurisdiction : Illinois (ill)")
    log.info(f"  Output       : {args.output}")
    log.info(f"  S3 upload    : {'disabled' if args.local_only else 'enabled'}")
    if args.limit:
        log.info(f"  Limit        : {args.limit} cases")

    run(
        output_file=args.output,
        no_upload=args.local_only,
        limit=args.limit,
        after_date=args.after,
        before_date=args.before,
    )


if __name__ == "__main__":
    main()
