#!/usr/bin/env python3
"""
batch_chunk.py — run all source chunkers in parallel and push output to S3.

By default, overwrites any existing output on the chunked S3 bucket.
Pass --skip-existing to check S3 before running and skip already-chunked sources.

Usage:
    python3 batch_chunk.py
    python3 batch_chunk.py --skip-existing
    python3 batch_chunk.py --sources ilga iac spac
    python3 batch_chunk.py --merge-opinions
    python3 batch_chunk.py --skip-existing --merge-opinions
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

LOGS_DIR = Path("logs")

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

@dataclass
class Source:
    name: str
    module: str
    # For --skip-existing: check one specific S3 key
    s3_check_key: Optional[str] = None
    # For --skip-existing: list objects under this prefix (iscr writes one file per PDF)
    s3_check_prefix: Optional[str] = None


def _build_sources() -> list[Source]:
    cl_prefix   = os.environ.get("RAW_COURTLISTENER_S3_PREFIX", "courtlistener").rstrip("/")
    ilcs_prefix = os.environ.get(
        "ILCS_CHUNKED_S3_PREFIX",
        os.environ.get("ILCS_S3_PREFIX", "ilcs/"),
    ).rstrip("/")
    spac_prefix = os.environ.get("SPAC_S3_PREFIX", "spac/").rstrip("/")
    iscr_prefix = os.environ.get("SUPREME_COURT_RULES_S3_PREFIX", "illinois-supreme-court-rules/").rstrip("/")
    cap_prefix  = os.environ.get("CAP_S3_PREFIX", "cap").rstrip("/")

    return [
        Source(
            "ilga",
            "chunk.ilga_chunk",
            s3_check_key=f"{ilcs_prefix}/ilcs_chunks.jsonl",
        ),
        Source(
            "iscr",
            "chunk.iscr_chunk",
            # ISCR writes one file per PDF — check prefix for any existing output
            s3_check_prefix=f"{iscr_prefix}/",
        ),
        Source("iac",           "chunk.iac_chunk",            s3_check_key="iac/iac_chunks.jsonl"),
        Source("iccb",          "chunk.iccb_chunk",           s3_check_key="iccb/iccb_chunks.jsonl"),
        Source("idoc",          "chunk.idoc_chunk",           s3_check_key="idoc/idoc_chunks.jsonl"),
        Source(
            "spac",
            "chunk.spac_chunk",
            s3_check_key=f"{spac_prefix}/spac_chunks.jsonl",
        ),
        Source("federal",        "chunk.federal_chunk",        s3_check_key="federal/federal_chunks.jsonl"),
        Source("restorejustice", "chunk.restorejustice_chunk", s3_check_key="restorejustice/restorejustice_chunks.jsonl"),
        Source("cookcounty-pd",  "chunk.cookcounty_pd_chunk",  s3_check_key="cookcounty-pd/cookcounty_pd_chunks.jsonl"),
        Source(
            "courtlistener",
            "chunk.courtlistener_chunk",
            s3_check_key=f"{cl_prefix}/bulk/opinion_chunks.jsonl",
        ),
        Source(
            "cap",
            "chunk.cap_chunk",
            s3_check_key=f"{cap_prefix}/cap_opinion_chunks.jsonl",
        ),
    ]


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_key_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def _s3_prefix_has_objects(s3, bucket: str, prefix: str) -> bool:
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return resp.get("KeyCount", 0) > 0


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_one(
    source: Source,
    log_path: Path,
    skip_existing: bool,
    chunked_bucket: str,
) -> dict:
    result = {
        "name": source.name,
        "status": None,
        "elapsed": 0.0,
        "log": log_path,
        "returncode": None,
    }

    if skip_existing:
        s3 = boto3.client("s3")
        if source.s3_check_key:
            already_done = _s3_key_exists(s3, chunked_bucket, source.s3_check_key)
        elif source.s3_check_prefix:
            already_done = _s3_prefix_has_objects(s3, chunked_bucket, source.s3_check_prefix)
        else:
            already_done = False

        if already_done:
            result["status"] = "skipped"
            _log(f"  [−] {source.name:<20} skipped (already in S3)")
            return result

    cmd = [sys.executable, "-m", source.module]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    result["elapsed"] = time.monotonic() - t0
    result["returncode"] = proc.returncode
    result["status"] = "ok" if proc.returncode == 0 else "failed"

    elapsed_str = _fmt_elapsed(result["elapsed"])
    icon = "✓" if result["status"] == "ok" else "✗"
    _log(f"  [{icon}] {source.name:<20} {result['status']:<8} {elapsed_str}")
    return result


def _run_merge(log_path: Path) -> dict:
    result = {
        "name": "merge-opinions",
        "status": None,
        "elapsed": 0.0,
        "log": log_path,
        "returncode": None,
    }
    cmd = [sys.executable, "-m", "chunk.merge_opinion_chunks"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)

    result["elapsed"] = time.monotonic() - t0
    result["returncode"] = proc.returncode
    result["status"] = "ok" if proc.returncode == 0 else "failed"
    return result


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all source chunkers in parallel and push output to S3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sources: ilga, iscr, iac, iccb, idoc, spac, federal, "
            "restorejustice, cookcounty-pd, courtlistener, cap"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sources whose output already exists in the chunked S3 bucket.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        metavar="SOURCE",
        help="Run only these sources (default: all).",
    )
    parser.add_argument(
        "--merge-opinions",
        action="store_true",
        help=(
            "After the parallel batch, run merge_opinion_chunks.py to combine "
            "CourtListener bulk + API chunks into merged_opinion_chunks.jsonl. "
            "Requires courtlistener_api.py to have been run separately first."
        ),
    )
    args = parser.parse_args()

    chunked_bucket = os.environ.get("CHUNKED_S3_BUCKET")
    if not chunked_bucket:
        print("ERROR: CHUNKED_S3_BUCKET is not set", file=sys.stderr)
        sys.exit(1)

    all_sources = _build_sources()
    source_by_name = {s.name: s for s in all_sources}

    if args.sources:
        unknown = set(args.sources) - source_by_name.keys()
        if unknown:
            print(f"ERROR: unknown sources: {', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"  Valid: {', '.join(source_by_name)}", file=sys.stderr)
            sys.exit(1)
        selected = [source_by_name[n] for n in args.sources]
    else:
        selected = all_sources

    run_ts  = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_dir = LOGS_DIR / f"batch_chunk_{run_ts}"
    log_dir.mkdir(parents=True, exist_ok=True)

    mode = "skip-existing" if args.skip_existing else "overwrite"
    print(f"Batch chunking {len(selected)} source(s)  →  s3://{chunked_bucket}/")
    print(f"  Mode : {mode}")
    print(f"  Logs : {log_dir}/")
    print()

    batch_t0 = time.monotonic()
    results: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = {
            pool.submit(
                _run_one,
                src,
                log_dir / f"{src.name}.log",
                args.skip_existing,
                chunked_bucket,
            ): src
            for src in selected
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Optional sequential merge step
    if args.merge_opinions:
        print()
        print("Running merge_opinion_chunks (sequential — depends on both bulk + API chunks)...")
        r = _run_merge(log_dir / "merge-opinions.log")
        elapsed_str = _fmt_elapsed(r["elapsed"])
        icon = "✓" if r["status"] == "ok" else "✗"
        print(f"  [{icon}] merge-opinions       {r['status']:<8} {elapsed_str}")
        results.append(r)

    # Summary
    total_elapsed = _fmt_elapsed(time.monotonic() - batch_t0)
    ok      = sum(1 for r in results if r["status"] == "ok")
    failed  = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")

    print()
    print(f"Done in {total_elapsed}:  {ok} ok  {skipped} skipped  {failed} failed")

    if failed:
        print()
        print("Failed sources (see logs for details):")
        for r in results:
            if r["status"] == "failed":
                print(f"  {r['name']:<20} exit {r['returncode']}  →  {r['log']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
