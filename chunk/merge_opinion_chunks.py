import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

LOCAL_DIR = Path("./data_files/chunked_output")

def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        log.error(f"Required environment variable {key!r} is not set.")
        sys.exit(1)
    return val

def get_config() -> dict:
    chunked_bucket = _require_env("CHUNKED_S3_BUCKET")
    cl_prefix = os.environ.get("RAW_COURTLISTENER_S3_PREFIX", "courtlistener").rstrip("/")
    prefix = f"{cl_prefix}/bulk"
    return {
        "chunked_bucket": chunked_bucket,
        "bulk_key":       f"{prefix}/opinion_chunks.jsonl",
        "api_key":        f"{prefix}/api_opinion_chunks.jsonl",
        "merged_key":     f"{prefix}/merged_opinion_chunks.jsonl",
    }

def read_jsonl_s3(bucket: str, key: str) -> list[dict]:
    log.info(f"  Reading s3://{bucket}/{key}")
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    lines = obj["Body"].read().decode("utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]

def write_jsonl_s3(records: list[dict], bucket: str, key: str):
    log.info(f"  Writing {len(records):,} records → s3://{bucket}/{key}")
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/x-ndjson",
    )
    log.info("  Upload complete.")

def read_jsonl_local(path: Path) -> list[dict]:
    log.info(f"  Reading {path}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl_local(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"  Saved {len(records):,} records → {path}")

def _opinion_key(chunk: dict) -> str:
    # bulk chunks (courtlistener_chunk.py) use parent_id; API chunks use opinion_id
    return chunk.get("parent_id") or chunk.get("opinion_id", "")


def merge(bulk: list[dict], api: list[dict]) -> list[dict]:
    bulk_by_opinion: dict[str, list[dict]] = defaultdict(list)
    api_by_opinion:  dict[str, list[dict]] = defaultdict(list)
    for chunk in bulk:
        bulk_by_opinion[_opinion_key(chunk)].append(chunk)
    for chunk in api:
        api_by_opinion[_opinion_key(chunk)].append(chunk)
    all_ids = set(bulk_by_opinion) | set(api_by_opinion)
    bulk_only = len(set(bulk_by_opinion) - set(api_by_opinion))
    api_only  = len(set(api_by_opinion)  - set(bulk_by_opinion))
    overlap   = len(set(bulk_by_opinion) & set(api_by_opinion))
    log.info(f"  Opinion overlap report:")
    log.info(f"    Bulk only : {bulk_only}")
    log.info(f"    API only  : {api_only}")
    log.info(f"    In both   : {overlap}")
    log.info(f"    Total unique opinions: {len(all_ids)}")
    merged: list[dict] = []
    kept_from_bulk = 0
    kept_from_api  = 0
    conflict_api_won = 0
    conflict_bulk_won = 0
    for oid in sorted(all_ids):
        b_chunks = bulk_by_opinion.get(oid, [])
        a_chunks = api_by_opinion.get(oid,  [])
        if b_chunks and not a_chunks:
            merged.extend(b_chunks)
            kept_from_bulk += 1
        elif a_chunks and not b_chunks:
            merged.extend(a_chunks)
            kept_from_api += 1
        else:
            b_tokens = sum(c.get("token_count", 0) for c in b_chunks)
            a_tokens = sum(c.get("token_count", 0) for c in a_chunks)
            if b_tokens > a_tokens:
                merged.extend(b_chunks)
                conflict_bulk_won += 1
            else:
                merged.extend(a_chunks)
                conflict_api_won += 1
    log.info(f"  Conflict resolution (opinion in both sources):")
    log.info(f"    API  won : {conflict_api_won}  (higher token count or tie)")
    log.info(f"    Bulk won : {conflict_bulk_won} (higher token count)")
    log.info(f"  Chunks kept from bulk: {kept_from_bulk + conflict_bulk_won} opinions")
    log.info(f"  Chunks kept from API : {kept_from_api  + conflict_api_won}  opinions")
    return merged

def run(local_only: bool = False):
    cfg  = get_config()
    bulk = read_jsonl_s3(cfg["chunked_bucket"], cfg["bulk_key"])
    api  = read_jsonl_s3(cfg["chunked_bucket"], cfg["api_key"])
    log.info(f"Loaded {len(bulk):,} bulk chunks, {len(api):,} API chunks.")
    merged = merge(bulk, api)
    log.info(f"Merged total: {len(merged):,} chunks")
    if merged:
        tokens = [c.get("token_count", 0) for c in merged]
        log.info(
            f"Token stats: avg {sum(tokens)/len(tokens):.0f} | "
            f"min {min(tokens)} | max {max(tokens)}"
        )
    if local_only:
        write_jsonl_local(merged, LOCAL_DIR / "merged_opinion_chunks.jsonl")
    else:
        write_jsonl_s3(merged, cfg["chunked_bucket"], cfg["merged_key"])

def main():
    parser = argparse.ArgumentParser(
        description="Merge bulk and API CourtListener chunks, deduplicating on opinion_id."
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Write output to ./chunked_output/ instead of uploading to S3. Input always reads from S3.",
    )
    args = parser.parse_args()
    run(local_only=args.local_only)

if __name__ == "__main__":
    main()