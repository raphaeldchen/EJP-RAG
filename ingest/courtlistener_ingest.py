"""
CourtListener 7th Circuit supplement ingestion.

Downloads CourtListener public bulk CSVs for the 7th Circuit (ca7).
CAP bulk download covers Illinois state courts (ill, illappct) through
early 2024 — this script covers 7th Circuit federal opinions, which CAP
does not include.

Pipeline:
  1. List latest bulk files from CourtListener's public S3 bucket
  2. Download dockets, clusters, opinions (with resume support)
  3. Filter: dockets → ca7 only; clusters/opinions cascade from that
  4. Upload filtered CSVs to your S3 for courtlistener_api.py to consume

Usage:
  python ingest/courtlistener_ingest.py --local-only
  python ingest/courtlistener_ingest.py         # full S3 pipeline
"""
import argparse
import bz2
import csv
import logging
import os
import re
import sys
import time
from pathlib import Path

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COURTS = {"ca7"}   # 7th Circuit only — state courts covered by CAP bulk

CL_S3_BUCKET  = "com-courtlistener-storage"
CL_S3_PREFIX  = "bulk-data/"
CL_S3_REGION  = "us-west-2"

LOCAL_DOWNLOAD_DIR = Path("./data_files/cl_downloads")
LOCAL_OUTPUT_DIR   = Path("./data_files/cl_filtered")

# Only the three tables courtlistener_api.py actually uses
CL_FILE_MAP = {
    "dockets":          "dockets",
    "opinion-clusters": "clusters",
    "opinions":         "opinions",
}

DOCKETS_COLS = [
    "id", "court_id", "case_name", "case_name_short", "docket_number",
    "date_filed", "date_terminated", "nature_of_suit", "cause",
    "blocked", "date_modified",
]
CLUSTERS_COLS = [
    "id", "docket_id", "case_name", "case_name_short", "date_filed",
    "date_filed_is_approximate", "judges", "precedential_status",
    "citation_count", "blocked", "filepath_pdf_harvard", "date_modified",
]
OPINIONS_COLS = [
    "id", "cluster_id", "type", "author_str", "per_curiam",
    "html_with_citations", "plain_text", "download_url",
    "local_path", "date_modified",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_s3_config() -> tuple[str, str]:
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        log.error("RAW_S3_BUCKET is not set.")
        sys.exit(1)
    prefix = os.environ.get("COURTLISTENER_S3_PREFIX", "courtlistener/").rstrip("/") + "/"
    return bucket, prefix


def _safe_str(val) -> str:
    return "" if val is None else str(val).strip()


def _make_writer(fileobj, fieldnames: list[str]) -> csv.DictWriter:
    writer = csv.DictWriter(
        fileobj, fieldnames=fieldnames,
        extrasaction="ignore", lineterminator="\n", quoting=csv.QUOTE_ALL,
    )
    writer.writeheader()
    return writer


def _increase_csv_limit():
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int //= 2

_increase_csv_limit()

# ---------------------------------------------------------------------------
# S3 listing
# ---------------------------------------------------------------------------

def list_cl_bulk_files() -> dict[str, str]:
    """Return {canonical_name: url} for the latest version of each needed file."""
    log.info("Listing CourtListener S3 bucket for latest bulk files...")
    s3 = boto3.client(
        "s3", region_name=CL_S3_REGION,
        config=Config(signature_version=UNSIGNED),
    )
    paginator = s3.get_paginator("list_objects_v2")
    raw: dict[str, list[tuple[str, str]]] = {}
    for page in paginator.paginate(Bucket=CL_S3_BUCKET, Prefix=CL_S3_PREFIX):
        for obj in page.get("Contents", []):
            key      = obj["Key"]
            if obj["Size"] < 1_000:
                continue
            filename = key.split("/")[-1]
            if not filename.endswith(".csv.bz2"):
                continue
            m = re.match(r"^(.+)-(\d{4}-\d{2}-\d{2})\.csv\.bz2$", filename)
            if not m:
                continue
            file_type, date_str = m.group(1), m.group(2)
            canonical = CL_FILE_MAP.get(file_type)
            if not canonical:
                continue
            url = f"https://{CL_S3_BUCKET}.s3-{CL_S3_REGION}.amazonaws.com/{key}"
            raw.setdefault(canonical, []).append((date_str, url))

    resolved: dict[str, str] = {}
    for canonical, entries in raw.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        date, url = entries[0]
        resolved[canonical] = url
        log.info(f"  {canonical}: {date}  ({url.split('/')[-1]})")
    return resolved

# ---------------------------------------------------------------------------
# Download (with resume)
# ---------------------------------------------------------------------------

def download_cl_file(url: str, name: str) -> Path:
    """Download a CourtListener bulk file with resume support."""
    LOCAL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest     = LOCAL_DOWNLOAD_DIR / f"{name}.csv.bz2"
    dest_tmp = LOCAL_DOWNLOAD_DIR / f"{name}.csv.bz2.part"

    if dest.exists():
        log.info(f"  Already staged: {dest.name}")
        return dest

    MAX_RETRIES   = 10
    CHUNK_SIZE    = 2 * 1024 * 1024
    READ_TIMEOUT  = 120
    CONNECT_TIMEOUT = 30

    for attempt in range(1, MAX_RETRIES + 1):
        resume_pos = dest_tmp.stat().st_size if dest_tmp.exists() else 0
        headers    = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}

        log.info(
            f"  {'Resuming' if resume_pos else 'Downloading'} {name} "
            f"from {resume_pos // 1024 // 1024} MB (attempt {attempt}/{MAX_RETRIES})..."
            if resume_pos else
            f"  Downloading {name} (attempt {attempt}/{MAX_RETRIES})..."
        )

        try:
            with requests.get(
                url, stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                headers=headers,
            ) as r:
                if r.status_code == 416:
                    dest_tmp.rename(dest)
                    return dest
                r.raise_for_status()
                total      = int(r.headers.get("content-length", 0)) + resume_pos
                downloaded = resume_pos
                with open(dest_tmp, "ab") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(
                                f"\r    {downloaded / total * 100:.1f}%"
                                f"  ({downloaded // 1024 // 1024} MB / "
                                f"{total // 1024 // 1024} MB)",
                                end="", flush=True,
                            )
            print()
            dest_tmp.rename(dest)
            log.info(f"  Saved: {dest}")
            return dest
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            print()
            wait = 2 ** attempt
            if attempt < MAX_RETRIES:
                log.warning(f"  Interrupted ({e.__class__.__name__}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Download of {name} failed after {MAX_RETRIES} retries")

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_dockets(src: Path, dest: Path) -> set[str]:
    log.info(f"Filtering dockets → {TARGET_COURTS} only...")
    kept: set[str] = set()
    count = skipped = 0
    with bz2.open(src, "rt", encoding="utf-8") as inf, \
         open(dest, "w", newline="", encoding="utf-8") as outf:
        writer = _make_writer(outf, DOCKETS_COLS)
        for row in csv.DictReader(inf):
            if _safe_str(row.get("court_id")) not in TARGET_COURTS:
                skipped += 1
                continue
            writer.writerow({k: _safe_str(row.get(k)) for k in DOCKETS_COLS})
            kept.add(_safe_str(row.get("id")))
            count += 1
    log.info(f"  {count:,} kept, {skipped:,} skipped.")
    return kept


def filter_clusters(src: Path, dest: Path, docket_ids: set[str]) -> set[str]:
    log.info("Filtering clusters → ca7 dockets...")
    kept: set[str] = set()
    count = skipped = 0
    with bz2.open(src, "rt", encoding="utf-8") as inf, \
         open(dest, "w", newline="", encoding="utf-8") as outf:
        writer = _make_writer(outf, CLUSTERS_COLS)
        for row in csv.DictReader(inf):
            if _safe_str(row.get("docket_id")) not in docket_ids:
                skipped += 1
                continue
            writer.writerow({k: _safe_str(row.get(k)) for k in CLUSTERS_COLS})
            kept.add(_safe_str(row.get("id")))
            count += 1
    log.info(f"  {count:,} kept, {skipped:,} skipped.")
    return kept


def filter_opinions(src: Path, dest: Path, cluster_ids: set[str]) -> set[str]:
    """Filter opinions and resolve the best available text (plain_text > stripped HTML)."""
    log.info("Filtering opinions → ca7 clusters...")
    kept: set[str] = set()
    count = skipped = 0
    has_plain = has_html_only = has_neither = 0

    with bz2.open(src, "rt", encoding="utf-8") as inf, \
         open(dest, "w", newline="", encoding="utf-8") as outf:
        from bs4 import BeautifulSoup
        writer = _make_writer(outf, OPINIONS_COLS)
        for row in csv.DictReader(inf):
            if _safe_str(row.get("cluster_id")) not in cluster_ids:
                skipped += 1
                continue
            html      = row.get("html_with_citations", "") or ""
            src_plain = _safe_str(row.get("plain_text"))
            if src_plain:
                plain = src_plain
                has_plain += 1
            elif html.strip():
                soup  = BeautifulSoup(html, "html.parser")
                for tag in soup.find_all(["p", "br", "div", "h1", "h2", "h3", "h4", "h5"]):
                    tag.insert_before("\n")
                import re as _re
                plain = _re.sub(r"\n{3,}", "\n\n", soup.get_text()).strip()
                has_html_only += 1
            else:
                plain = ""
                has_neither += 1
            out_row = {k: _safe_str(row.get(k)) for k in OPINIONS_COLS}
            out_row["html_with_citations"] = html
            out_row["plain_text"]          = plain
            writer.writerow(out_row)
            kept.add(_safe_str(row.get("id")))
            count += 1
    log.info(
        f"  {count:,} kept, {skipped:,} skipped. "
        f"Text: {has_plain:,} plain | {has_html_only:,} html-only | {has_neither:,} neither."
    )
    return kept

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _upload_and_clean(path: Path, bucket: str, prefix: str, bz2_src: Path | None = None) -> None:
    """Upload a filtered CSV to S3, then delete it and its source .bz2."""
    s3_key = f"{prefix}bulk/{path.name}"
    log.info(f"  Uploading {path.name} → s3://{bucket}/{s3_key}")
    boto3.client("s3").upload_file(str(path), bucket, s3_key)
    path.unlink()
    if bz2_src and bz2_src.exists():
        bz2_src.unlink()
        log.info(f"  Deleted local {bz2_src.name}")


def run(local_only: bool = False) -> None:
    bucket = prefix = None
    if not local_only:
        bucket, prefix = _get_s3_config()

    cl_files = list_cl_bulk_files()
    missing  = [t for t in CL_FILE_MAP.values() if t not in cl_files]
    if missing:
        log.error(f"Missing bulk files: {missing}")
        sys.exit(1)

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("\n--- Dockets ---")
    dockets_bz2 = download_cl_file(cl_files["dockets"], "dockets")
    docket_ids  = filter_dockets(dockets_bz2, LOCAL_OUTPUT_DIR / "dockets.csv")
    if not local_only:
        _upload_and_clean(LOCAL_OUTPUT_DIR / "dockets.csv", bucket, prefix, bz2_src=dockets_bz2)

    log.info("\n--- Clusters ---")
    clusters_bz2 = download_cl_file(cl_files["clusters"], "clusters")
    cluster_ids  = filter_clusters(clusters_bz2, LOCAL_OUTPUT_DIR / "clusters.csv", docket_ids)
    if not local_only:
        _upload_and_clean(LOCAL_OUTPUT_DIR / "clusters.csv", bucket, prefix, bz2_src=clusters_bz2)

    log.info("\n--- Opinions ---")
    opinions_bz2 = download_cl_file(cl_files["opinions"], "opinions")
    filter_opinions(opinions_bz2, LOCAL_OUTPUT_DIR / "opinions.csv", cluster_ids)
    if not local_only:
        _upload_and_clean(LOCAL_OUTPUT_DIR / "opinions.csv", bucket, prefix, bz2_src=opinions_bz2)
        log.info(f"\nDone. s3://{bucket}/{prefix}bulk/")
    else:
        log.info(f"\nLocal-only mode. Filtered CSVs in: {LOCAL_OUTPUT_DIR.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CourtListener 7th Circuit bulk CSVs → S3"
    )
    parser.add_argument(
        "--local-only", action="store_true",
        help="Filter CSVs locally without uploading to S3.",
    )
    args = parser.parse_args()
    run(local_only=args.local_only)


if __name__ == "__main__":
    main()
