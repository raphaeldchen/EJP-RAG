"""
Harvard Caselaw Access Project (CAP) bulk data ingestion.

The CAP API was sunset on September 5, 2024. Full case data remains available
for bulk download. This script processes the Illinois bulk zip (or a directory
of pre-extracted JSONL files) and outputs one record per case, matching the
schema used by cap_ingest.py so both feed the same downstream chunker.

How to get the Illinois bulk download:
  1. Go to bulk.case.law
  2. Find "Illinois" and download the with-text variant (~1-3 GB zip)
  3. Pass the zip path to --zip-path below

Output schema (JSONL):
  id, source, case_id, case_name, case_name_abbr, court, court_id,
  jurisdiction, date_decided, citations, docket_number, url, text, scraped_at

Usage:
  python ingest/cap_bulk_ingest.py --zip-path Illinois-20240101-text.zip --local-only
  python ingest/cap_bulk_ingest.py --zip-path Illinois-20240101-text.zip --after 1980-01-01 --local-only
  python ingest/cap_bulk_ingest.py --jsonl-dir ./cap_extracted/ --local-only
  python ingest/cap_bulk_ingest.py --zip-path Illinois.zip  # full S3 pipeline
"""
import argparse
import json
import logging
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE       = "data_files/corpus/cap_bulk_corpus.jsonl"
CHECKPOINT_SUFFIX = ".cap_bulk_done"

# CAP court slugs for Illinois courts we want by default.
# Use --all-courts to include circuit courts and specialty courts.
DEFAULT_COURT_SLUGS = {
    "ill",        # Illinois Supreme Court
    "illappct",   # Illinois Appellate Court
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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


def _upload_to_s3(local_path: Path, local_only: bool = False) -> None:
    if local_only:
        return
    bucket, prefix = _get_s3_config()
    s3_key = f"{prefix}{local_path.name}"
    boto3.client("s3").upload_file(str(local_path), bucket, s3_key)
    log.info(f"  Uploaded → s3://{bucket}/{s3_key}")

# ---------------------------------------------------------------------------
# Text extraction  (mirrors cap_ingest._extract_text / _extract_citations)
# ---------------------------------------------------------------------------

def _extract_text(case: dict) -> str:
    """Pull plain text from a CAP case record.

    Bulk format: case['casebody']['data']['opinions'] — list of opinion dicts,
    each with 'type' and 'text'.  'head_matter' is the syllabus/headnotes, not
    the opinion itself and is omitted.
    """
    try:
        casebody = case.get("casebody", {})
        data     = casebody.get("data") or {}

        if isinstance(data, str):
            # Older bulk format encodes casebody as raw XML/HTML string
            return re.sub(r"<[^>]+>", " ", data).strip()

        opinions = data.get("opinions", [])
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


def _build_record(case: dict) -> dict | None:
    text = _extract_text(case)
    if not text.strip():
        return None

    court      = case.get("court", {})
    court_id   = court.get("slug", "")
    court_name = court.get("name_abbreviation", "") or court.get("name", "")

    return {
        "id":             f"cap-{case['id']}",
        "source":         "cap_bulk",
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

# ---------------------------------------------------------------------------
# JSONL iteration
# ---------------------------------------------------------------------------

def _iter_jsonl_file(fileobj) -> "Iterator[dict]":
    """Yield parsed case dicts from a JSONL fileobj (text or binary)."""
    for raw_line in fileobj:
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8", errors="replace")
        line = raw_line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            log.warning(f"  JSON parse error: {e} — skipping line")


def _iter_zip(zip_path: Path) -> "Iterator[tuple[str, dict]]":
    """Yield (member_name, case_dict) from all JSONL members in a zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".jsonl", ".json"))
                   and not m.startswith("__MACOSX")]
        log.info(f"  {len(members)} JSONL file(s) in zip")
        for member in sorted(members):
            log.info(f"  Processing {member} ...")
            with zf.open(member) as f:
                for case in _iter_jsonl_file(f):
                    yield member, case


def _iter_jsonl_dir(jsonl_dir: Path) -> "Iterator[tuple[str, dict]]":
    """Yield (filename, case_dict) from all JSONL files in a directory."""
    files = sorted(jsonl_dir.glob("*.jsonl")) + sorted(jsonl_dir.glob("*.json"))
    log.info(f"  {len(files)} JSONL file(s) in directory")
    for path in files:
        log.info(f"  Processing {path.name} ...")
        with open(path, encoding="utf-8") as f:
            for case in _iter_jsonl_file(f):
                yield path.name, case

# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest(
    zip_path: Path | None,
    jsonl_dir: Path | None,
    output_file: str,
    local_only: bool,
    after: str | None,
    before: str | None,
    all_courts: bool,
    limit: int,
) -> None:
    if not local_only:
        _get_s3_config()

    done_ids  = _load_checkpoint(output_file)
    if done_ids:
        log.info(f"  Resuming — {len(done_ids):,} cases already written")

    out_path  = Path(output_file)
    written   = skipped_text = skipped_court = skipped_date = skipped_done = 0

    court_filter = None if all_courts else DEFAULT_COURT_SLUGS

    source = _iter_zip(zip_path) if zip_path else _iter_jsonl_dir(jsonl_dir)  # type: ignore[arg-type]

    with open(out_path, "a", encoding="utf-8") as out:
        for _member, case in source:
            case_id = str(case.get("id", ""))

            # Checkpoint skip
            if case_id in done_ids:
                skipped_done += 1
                continue

            # Court filter
            court_slug = case.get("court", {}).get("slug", "")
            if court_filter and court_slug not in court_filter:
                skipped_court += 1
                continue

            # Date filter
            date_decided = case.get("decision_date", "")
            if after and date_decided and date_decided < after:
                skipped_date += 1
                continue
            if before and date_decided and date_decided > before:
                skipped_date += 1
                continue

            rec = _build_record(case)
            if rec is None:
                skipped_text += 1
                continue

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            _save_checkpoint(output_file, case_id)
            written += 1

            if written % 5_000 == 0:
                log.info(
                    f"  {written:,} written | "
                    f"{skipped_court:,} wrong court | "
                    f"{skipped_date:,} out of range | "
                    f"{skipped_text:,} no text"
                )

            if limit and written >= limit:
                log.info(f"  --limit {limit} reached, stopping.")
                break

    log.info(
        f"\nDone. {written:,} records written → {out_path}\n"
        f"  Skipped: {skipped_court:,} wrong court | "
        f"{skipped_date:,} out of date range | "
        f"{skipped_text:,} no opinion text | "
        f"{skipped_done:,} already done"
    )
    _upload_to_s3(out_path, local_only)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest CAP Illinois bulk download → JSONL"
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--zip-path", type=Path, metavar="FILE",
        help="Path to the CAP Illinois bulk zip file (e.g. Illinois-20240101-text.zip).",
    )
    src.add_argument(
        "--jsonl-dir", type=Path, metavar="DIR",
        help="Directory containing pre-extracted JSONL files from the CAP bulk zip.",
    )

    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument(
        "--local-only", action="store_true",
        help="Write output locally without uploading to S3.",
    )
    parser.add_argument(
        "--after", metavar="YYYY-MM-DD",
        help="Only include cases decided on or after this date.",
    )
    parser.add_argument(
        "--before", metavar="YYYY-MM-DD",
        help="Only include cases decided on or before this date.",
    )
    parser.add_argument(
        "--all-courts", action="store_true",
        help=(
            f"Include all Illinois courts (default: only {sorted(DEFAULT_COURT_SLUGS)})."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="Stop after writing N records (0 = no limit). Useful for testing.",
    )
    args = parser.parse_args()

    if args.zip_path and not args.zip_path.exists():
        log.error(f"Zip file not found: {args.zip_path}")
        sys.exit(1)
    if args.jsonl_dir and not args.jsonl_dir.is_dir():
        log.error(f"Directory not found: {args.jsonl_dir}")
        sys.exit(1)

    log.info("CAP Bulk Ingestion")
    log.info(f"  Source      : {args.zip_path or args.jsonl_dir}")
    log.info(f"  Output      : {args.output}")
    log.info(f"  Courts      : {'all' if args.all_courts else sorted(DEFAULT_COURT_SLUGS)}")
    log.info(f"  Date range  : {args.after or 'beginning'} → {args.before or 'end'}")
    log.info(f"  Limit       : {args.limit or 'none'}")
    log.info(f"  S3 upload   : {'disabled' if args.local_only else 'enabled'}")

    ingest(
        zip_path   = args.zip_path,
        jsonl_dir  = args.jsonl_dir,
        output_file= args.output,
        local_only = args.local_only,
        after      = args.after,
        before     = args.before,
        all_courts = args.all_courts,
        limit      = args.limit,
    )


if __name__ == "__main__":
    main()
