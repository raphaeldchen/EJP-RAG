#!/usr/bin/env python3
"""
Stream cap_opinion_chunks.jsonl from S3, filter to 1973+, validate, then overwrite.
Writes to a temp key first; only overwrites the canonical key after all checks pass.
"""

import json
import os
import sys
import tempfile
import boto3
from collections import Counter
from datetime import datetime

BUCKET = "illinois-legal-corpus-chunked"
SOURCE_KEY = "cap/cap_opinion_chunks.jsonl"
TEMP_KEY = "cap/cap_opinion_chunks_1973plus_TEMP.jsonl"
CUTOFF = "1973"

REQUIRED_FIELDS = {"chunk_id", "parent_id", "chunk_index", "chunk_total",
                   "source", "text", "enriched_text", "token_count", "metadata"}


def stream_filter(s3, tmp_path):
    """Stream S3 → filter ≥ 1973 → local temp file. Returns (kept, skipped, bad_date)."""
    kept = skipped = bad_date = 0
    obj = s3.get_object(Bucket=BUCKET, Key=SOURCE_KEY)
    body = obj["Body"]

    print(f"Streaming and filtering {SOURCE_KEY} …", flush=True)
    with open(tmp_path, "w") as out:
        for raw in body.iter_lines():
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_date += 1
                continue

            date = (rec.get("metadata") or {}).get("date_decided", "")
            if not date:
                bad_date += 1
                continue

            if date >= CUTOFF:
                out.write(line + "\n")
                kept += 1
            else:
                skipped += 1

            total = kept + skipped + bad_date
            if total % 100_000 == 0:
                print(f"  {total:,} processed — kept {kept:,}, skipped {skipped:,}", flush=True)

    return kept, skipped, bad_date


def validate(tmp_path):
    """Run quality checks on the filtered file. Returns (passed, issues list)."""
    issues = []
    counts = Counter()
    dates = []
    empty_text = 0
    missing_fields = 0
    source_counts = Counter()
    bad_json = 0

    print(f"\nValidating {tmp_path} …", flush=True)
    with open(tmp_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            counts["total"] += 1
            source_counts[rec.get("source", "MISSING")] += 1

            missing = REQUIRED_FIELDS - set(rec.keys())
            if missing:
                missing_fields += 1

            if not (rec.get("text") or "").strip():
                empty_text += 1

            date = (rec.get("metadata") or {}).get("date_decided", "")
            if date:
                dates.append(date)

    total = counts["total"]

    # Checks
    if bad_json:
        issues.append(f"BAD JSON lines: {bad_json}")
    if missing_fields:
        issues.append(f"Records missing required fields: {missing_fields}")
    if empty_text:
        issues.append(f"Records with empty text: {empty_text}")

    if dates:
        min_date = min(dates)
        max_date = max(dates)
        print(f"  Date range: {min_date} → {max_date}")
        if min_date < CUTOFF:
            issues.append(f"Pre-{CUTOFF} records leaked through — earliest: {min_date}")
    else:
        issues.append("No date_decided fields found at all")

    print(f"  Total records: {total:,}")
    print(f"  Source breakdown: {dict(source_counts)}")
    print(f"  Missing required fields: {missing_fields}")
    print(f"  Empty text records: {empty_text}")

    passed = len(issues) == 0
    return passed, issues, total, min(dates) if dates else None, max(dates) if dates else None


def upload(s3, tmp_path):
    print(f"\nUploading to s3://{BUCKET}/{SOURCE_KEY} …", flush=True)
    file_size = os.path.getsize(tmp_path)
    # Multipart upload for large files
    config = boto3.s3.transfer.TransferConfig(multipart_threshold=100 * 1024 * 1024)
    s3.upload_file(tmp_path, BUCKET, SOURCE_KEY,
                   ExtraArgs={"ContentType": "application/x-ndjson"},
                   Config=config)
    print(f"  Uploaded {file_size / 1e9:.2f} GB to s3://{BUCKET}/{SOURCE_KEY}")


def main():
    s3 = boto3.client("s3")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                     dir="/tmp", prefix="cap_chunks_1973_") as tmp:
        tmp_path = tmp.name

    print(f"Temp file: {tmp_path}\n")

    try:
        # Step 1: filter
        t0 = datetime.now()
        kept, skipped, bad_date = stream_filter(s3, tmp_path)
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"\nFilter complete in {elapsed:.0f}s")
        print(f"  Kept:    {kept:,}")
        print(f"  Skipped: {skipped:,}  (pre-{CUTOFF})")
        print(f"  Bad/missing date: {bad_date:,}")

        if kept == 0:
            print("ERROR: zero records kept — aborting", file=sys.stderr)
            sys.exit(1)

        # Step 2: validate
        passed, issues, total, min_date, max_date = validate(tmp_path)

        if not passed:
            print("\nVALIDATION FAILED:")
            for issue in issues:
                print(f"  ✗ {issue}")
            print("\nAborting — S3 not modified.")
            sys.exit(1)

        print(f"\nAll checks passed — {total:,} records, {min_date} → {max_date}")
        file_size = os.path.getsize(tmp_path)
        print(f"Filtered file size: {file_size / 1e9:.2f} GB")

        # Step 3: upload
        t1 = datetime.now()
        upload(s3, tmp_path)
        elapsed2 = (datetime.now() - t1).total_seconds()
        print(f"Upload complete in {elapsed2:.0f}s")
        print(f"\nDone. s3://{BUCKET}/{SOURCE_KEY} now contains {total:,} chunks ({min_date}–{max_date}).")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"Temp file cleaned up.")


if __name__ == "__main__":
    main()
