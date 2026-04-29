"""
Corpus validator — runs after any ingest script to sanity-check the raw JSONL
on S3 before chunking begins.

Checks (universal):
  - Record count above expected minimum
  - No duplicate IDs
  - No empty / whitespace-only text fields
  - All required fields present on every record
  - Stub ratio: warns if >20% of records have text shorter than 100 chars

Usage:
    python3 ingest/validate_corpus.py --source iac
    python3 ingest/validate_corpus.py --source spac
    python3 ingest/validate_corpus.py --key some/custom/path.jsonl
    python3 ingest/validate_corpus.py --all

Exit codes: 0 = all checks passed (warnings OK), 1 = one or more failures.
"""

import argparse
import json
import os
import sys
from collections import Counter

import boto3
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

SOURCES: dict[str, dict] = {
    "iac": {
        "key":       "iac/iac_corpus.jsonl",
        "min_count": 200,
        "required":  {"id", "source", "text", "title_num", "part_num", "section_num", "section_citation"},
    },
    "ilcs": {
        "key":       "ilcs/ilcs_corpus.jsonl",
        "min_count": 500,
        "required":  {"id", "source", "text", "section_citation", "chapter_num", "act_id"},
    },
    "idoc": {
        "key":       "idoc/idoc_corpus.jsonl",
        "min_count": 50,
        "required":  {"id", "source", "text", "doc_type", "title"},
    },
    "spac": {
        "key":       "spac/spac_corpus.jsonl",
        "min_count": 50,
        "required":  {"id", "source", "text", "title", "agency"},
    },
    "iccb": {
        "key":       "iccb/iccb_corpus.jsonl",
        "min_count": 5,
        "required":  {"id", "source", "text", "doc_type", "title", "fiscal_year"},
    },
    "federal": {
        "key":       "federal/federal_corpus.jsonl",
        "min_count": 3,
        "required":  {"id", "source", "text", "doc_type", "title"},
    },
    "restorejustice": {
        "key":       "restorejustice/restorejustice_corpus.jsonl",
        "min_count": 10,
        "required":  {"id", "source", "text", "page_title"},
    },
    "cookcounty_pd": {
        "key":       "cookcounty-pd/cookcounty_pd_corpus.jsonl",
        "min_count": 5,
        "required":  {"id", "source", "text", "page_title"},
    },
}

STUB_THRESHOLD   = 100   # chars; records shorter than this are "stubs"
STUB_WARN_RATIO  = 0.20  # warn if >20% are stubs


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
RESET  = "\033[0m"

def _pass(msg: str) -> None:
    print(f"  {GREEN}PASS{RESET}  {msg}")

def _warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{RESET}  {msg}")

def _fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def load_records(bucket: str, key: str) -> list[dict]:
    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read().decode("utf-8")
    records = []
    for i, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"  {RED}FAIL{RESET}  Malformed JSON on line {i}: {exc}")
    return records


def validate(bucket: str, key: str, cfg: dict) -> bool:
    """Run all checks; return True if all pass (warnings don't fail)."""
    print(f"\n{'─' * 60}")
    print(f"  Source : {cfg.get('name', key)}")
    print(f"  S3 key : s3://{bucket}/{key}")
    print(f"{'─' * 60}")

    try:
        records = load_records(bucket, key)
    except Exception as exc:
        _fail(f"Could not load corpus: {exc}")
        return False

    failed = False

    # 1. Record count
    n = len(records)
    min_count = cfg.get("min_count", 1)
    if n == 0:
        _fail(f"No records found (expected ≥ {min_count})")
        return False
    elif n < min_count:
        _fail(f"Only {n} records (expected ≥ {min_count})")
        failed = True
    else:
        _pass(f"{n:,} records (min {min_count})")

    # 2. Duplicate IDs
    ids        = [r.get("id", "") for r in records]
    id_counts  = Counter(ids)
    duplicates = {k: v for k, v in id_counts.items() if v > 1}
    if duplicates:
        examples = list(duplicates.items())[:3]
        _fail(
            f"{len(duplicates)} duplicate IDs "
            f"(e.g. {', '.join(f'{k!r}×{v}' for k, v in examples)})"
        )
        failed = True
    else:
        _pass("No duplicate IDs")

    # 3. Empty text
    empty = [r.get("id", f"[line {i}]") for i, r in enumerate(records)
             if not r.get("text", "").strip()]
    if empty:
        _fail(f"{len(empty)} records with empty text (e.g. {empty[:3]})")
        failed = True
    else:
        _pass("No empty text fields")

    # 4. Required fields
    required = cfg.get("required", {"id", "source", "text"})
    missing_field_records = []
    for r in records:
        missing = required - r.keys()
        if missing:
            missing_field_records.append((r.get("id", "?"), missing))
    if missing_field_records:
        _fail(
            f"{len(missing_field_records)} records missing required fields "
            f"(e.g. {missing_field_records[0][0]!r}: {missing_field_records[0][1]})"
        )
        failed = True
    else:
        _pass(f"All required fields present ({', '.join(sorted(required))})")

    # 5. Stub ratio (warn only)
    stubs      = [r for r in records if len(r.get("text", "").strip()) < STUB_THRESHOLD]
    stub_ratio = len(stubs) / n
    if stub_ratio > STUB_WARN_RATIO:
        _warn(
            f"{len(stubs):,}/{n:,} records ({stub_ratio:.0%}) have text < {STUB_THRESHOLD} chars "
            f"— possible scraping stubs"
        )
    else:
        _pass(f"Stub ratio acceptable: {len(stubs)}/{n} records < {STUB_THRESHOLD} chars")

    # 6. Source field consistency
    source_values = set(r.get("source", "") for r in records)
    if len(source_values) > 3:
        _warn(f"Unexpected source field diversity: {source_values}")
    else:
        _pass(f"Source values consistent: {source_values}")

    print()
    return not failed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw corpus JSONL from S3")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--source",
        choices=list(SOURCES.keys()),
        help="Named source to validate",
    )
    group.add_argument(
        "--key",
        metavar="S3_KEY",
        help="Direct S3 key (relative to RAW_S3_BUCKET) for an unlisted corpus",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Validate all registered sources",
    )
    args = parser.parse_args()

    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        print("RAW_S3_BUCKET environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if args.all:
        targets = [(name, cfg) for name, cfg in SOURCES.items()]
    elif args.source:
        targets = [(args.source, SOURCES[args.source])]
    else:
        targets = [(args.key, {"key": args.key, "min_count": 1, "required": {"id", "source", "text"}})]

    all_passed = True
    for name, cfg in targets:
        cfg = dict(cfg, name=name)
        passed = validate(bucket, cfg["key"], cfg)
        if not passed:
            all_passed = False

    print("─" * 60)
    if all_passed:
        print(f"{GREEN}All checks passed.{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}One or more checks failed.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
