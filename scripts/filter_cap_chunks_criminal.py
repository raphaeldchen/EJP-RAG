#!/usr/bin/env python3
"""
Filter cap_opinion_chunks.jsonl to Illinois criminal-justice-relevant opinions,
using the same _is_cap_criminal logic as batch_embed.py.
Validates then overwrites the S3 key.
"""

import json
import os
import re
import sys
import tempfile
import boto3
from datetime import datetime

BUCKET = "illinois-legal-corpus-chunked"
SOURCE_KEY = "cap/cap_opinion_chunks.jsonl"

_CRIMINAL_CASE_RE = re.compile(r"\bPeople\b|^In re\b", re.IGNORECASE)
_CRIMINAL_STATUTE_RE = re.compile(r"\b(705|720|725|730)\s+ILCS\b")

REQUIRED_FIELDS = {"chunk_id", "parent_id", "chunk_index", "chunk_total",
                   "source", "text", "enriched_text", "token_count", "metadata"}


def is_criminal(record: dict) -> bool:
    case_name = record.get("metadata", {}).get("case_name", "")
    if _CRIMINAL_CASE_RE.search(case_name):
        return True
    text = record.get("text", "") + " " + record.get("enriched_text", "")
    return bool(_CRIMINAL_STATUTE_RE.search(text))


def stream_filter(s3, tmp_path):
    kept = skipped = bad = 0
    obj = s3.get_object(Bucket=BUCKET, Key=SOURCE_KEY)
    print(f"Streaming and filtering {SOURCE_KEY} …", flush=True)
    with open(tmp_path, "w") as out:
        for raw in obj["Body"].iter_lines():
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if is_criminal(rec):
                out.write(line + "\n")
                kept += 1
            else:
                skipped += 1
            total = kept + skipped + bad
            if total % 100_000 == 0:
                print(f"  {total:,} processed — kept {kept:,}, skipped {skipped:,}", flush=True)
    return kept, skipped, bad


def validate(tmp_path):
    issues = []
    total = empty_text = missing_fields = bad_json = 0
    dates = []

    print(f"\nValidating {tmp_path} …", flush=True)
    with open(tmp_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            total += 1
            if REQUIRED_FIELDS - set(rec.keys()):
                missing_fields += 1
            if not (rec.get("text") or "").strip():
                empty_text += 1
            date = (rec.get("metadata") or {}).get("date_decided", "")
            if date:
                dates.append(date)
            # Spot-check: every kept record must pass the filter
            if not is_criminal(rec):
                issues.append(f"Non-criminal record leaked: {rec.get('chunk_id')}")
                if len(issues) >= 3:
                    break

    if bad_json:
        issues.append(f"Bad JSON lines: {bad_json}")
    if missing_fields:
        issues.append(f"Records missing required fields: {missing_fields}")
    if empty_text:
        issues.append(f"Records with empty text: {empty_text}")
    if total == 0:
        issues.append("Zero records in output")

    min_date = min(dates) if dates else None
    max_date = max(dates) if dates else None
    print(f"  Total records: {total:,}")
    print(f"  Date range: {min_date} → {max_date}")
    print(f"  Missing required fields: {missing_fields}")
    print(f"  Empty text records: {empty_text}")
    print(f"  File size: {os.path.getsize(tmp_path)/1e9:.2f} GB")

    return len(issues) == 0, issues, total, min_date, max_date


def upload(s3, tmp_path):
    print(f"\nUploading to s3://{BUCKET}/{SOURCE_KEY} …", flush=True)
    config = boto3.s3.transfer.TransferConfig(multipart_threshold=100 * 1024 * 1024)
    s3.upload_file(tmp_path, BUCKET, SOURCE_KEY,
                   ExtraArgs={"ContentType": "application/x-ndjson"},
                   Config=config)
    print(f"  Uploaded {os.path.getsize(tmp_path)/1e9:.2f} GB")


def main():
    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False,
                                     dir="/tmp", prefix="cap_criminal_") as tmp:
        tmp_path = tmp.name
    print(f"Temp file: {tmp_path}\n")

    try:
        t0 = datetime.now()
        kept, skipped, bad = stream_filter(s3, tmp_path)
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"\nFilter complete in {elapsed:.0f}s — kept {kept:,}, skipped {skipped:,}, bad {bad}")

        if kept == 0:
            print("ERROR: zero records kept — aborting")
            sys.exit(1)

        passed, issues, total, min_date, max_date = validate(tmp_path)

        if not passed:
            print("\nVALIDATION FAILED:")
            for issue in issues:
                print(f"  ✗ {issue}")
            print("Aborting — S3 not modified.")
            sys.exit(1)

        print(f"\nAll checks passed — {total:,} records, {min_date}–{max_date}")

        t1 = datetime.now()
        upload(s3, tmp_path)
        print(f"Upload complete in {(datetime.now()-t1).total_seconds():.0f}s")
        print(f"\nDone. s3://{BUCKET}/{SOURCE_KEY} now contains {total:,} criminal chunks.")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print("Temp file cleaned up.")


if __name__ == "__main__":
    main()
