import argparse
import csv
import bz2
import logging
import os
import re
import sys
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

ILLINOIS_COURTS = {"ill", "illappct"}

CL_S3_BUCKET = "com-courtlistener-storage"
CL_S3_PREFIX = "bulk-data/"
CL_S3_REGION = "us-west-2"

LOCAL_DOWNLOAD_DIR = Path("./cl_downloads")  # staged .csv.bz2 files from CL
LOCAL_OUTPUT_DIR   = Path("./cl_filtered")   # filtered CSVs before upload


def increase_csv_field_limit():
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = max_int // 2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)
increase_csv_field_limit()

CL_FILE_MAP = {
    # Core case law
    "courts":               "courts",
    "courthouses":          "courthouses",
    "court-appeals-to":     "court_appeals_to",
    "dockets":              "dockets",
    "originating-court-information": "originating_court_info",
    "opinion-clusters":     "clusters",
    "opinions":             "opinions",
    "citation-map":         "citations",
    "parentheticals":       "parentheticals",

    # Opinion relationship tables
    "search_opinion_joined_by":                       "opinion_joined_by",
    "search_opinioncluster_panel":                    "cluster_panel",
    "search_opinioncluster_non_participating_judges": "cluster_non_participating",

    # Judge / people tables
    "people-db-people":                 "judges",
    "people-db-positions":              "judge_positions",
    "people-db-educations":             "judge_educations",
    "people-db-political-affiliations": "judge_political",
    "people-db-retention-events":       "judge_retention",
    "people-db-races":                  "judge_races",
    "people-db-schools":                "judge_schools",
}

COURTS_COLS = [
    "id", "full_name", "short_name", "jurisdiction",
    "position", "in_use", "date_modified",
]
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
CITATIONS_COLS = [
    "id", "citing_opinion_id", "cited_opinion_id", "depth",
]
PARENTHETICALS_COLS = [
    "id", "describing_opinion_id", "described_opinion_id",
    "text", "score", "date_modified",
]


def get_s3_config() -> tuple[str, str]:
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        log.error("RAW_S3_BUCKET environment variable is not set.")
        sys.exit(1)
    prefix = os.environ.get("COURTLISTENER_S3_PREFIX", "courtlistener/").rstrip("/") + "/"
    return bucket, prefix


def upload_to_s3(local_path: Path, s3_key: str):
    bucket, _ = get_s3_config()
    s3 = boto3.client("s3")
    log.info(f"  Uploading {local_path.name} → s3://{bucket}/{s3_key}")
    s3.upload_file(str(local_path), bucket, s3_key)


def strip_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["p", "br", "div", "h1", "h2", "h3", "h4", "h5"]):
        tag.insert_before("\n")
    text = soup.get_text()
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def safe_str(val) -> str:
    return "" if val is None else str(val).strip()


def make_writer(fileobj, fieldnames: list[str]) -> csv.DictWriter:
    writer = csv.DictWriter(
        fileobj,
        fieldnames=fieldnames,
        extrasaction="ignore",
        lineterminator="\n",
        quoting=csv.QUOTE_ALL,
    )
    writer.writeheader()
    return writer


def list_cl_bulk_files() -> dict[str, str]:
    log.info("Listing CourtListener S3 bucket for latest bulk files...")
    s3 = boto3.client(
        "s3",
        region_name=CL_S3_REGION,
        config=Config(signature_version=UNSIGNED),
    )
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(
        Bucket=CL_S3_BUCKET,
        Prefix=CL_S3_PREFIX,
    )
    raw: dict[str, list[tuple[str, str]]] = {}
    for page in pages:
        for obj in page.get("Contents", []):
            key  = obj["Key"]
            size = obj["Size"]
            if size < 1_000:
                continue
            filename = key.split("/")[-1]
            if not filename.endswith(".csv.bz2"):
                continue
            match = re.match(r"^(.+)-(\d{4}-\d{2}-\d{2})\.csv\.bz2$", filename)
            if not match:
                continue
            file_type, date_str = match.group(1), match.group(2)
            canonical = CL_FILE_MAP.get(file_type)
            if not canonical:
                continue
            url = f"https://{CL_S3_BUCKET}.s3-{CL_S3_REGION}.amazonaws.com/{key}"
            raw.setdefault(canonical, []).append((date_str, url))

    resolved: dict[str, str] = {}
    for canonical, entries in raw.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        best_date, best_url = entries[0]
        resolved[canonical] = best_url
        log.info(f"  {canonical}: {best_date}  ({best_url.split('/')[-1]})")

    return resolved


def download_cl_file(url: str, name: str) -> Path:
    """
    Download a CourtListener bulk file with resume support and automatic retries.

    Uses HTTP Range requests to resume interrupted downloads, so re-running
    after a timeout picks up where it left off rather than starting over.
    Retries up to MAX_RETRIES times with exponential backoff on each attempt.
    """
    import time

    LOCAL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest     = LOCAL_DOWNLOAD_DIR / f"{name}.csv.bz2"
    dest_tmp = LOCAL_DOWNLOAD_DIR / f"{name}.csv.bz2.part"

    if dest.exists():
        log.info(f"  Already staged: {dest.name} — skipping download.")
        return dest

    MAX_RETRIES = 10
    CHUNK_SIZE  = 2 * 1024 * 1024   # 2 MB
    READ_TIMEOUT = 120               # seconds per chunk read
    CONNECT_TIMEOUT = 30

    for attempt in range(1, MAX_RETRIES + 1):
        # Resume from wherever the .part file left off
        resume_pos = dest_tmp.stat().st_size if dest_tmp.exists() else 0
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}

        if resume_pos:
            log.info(f"  Resuming {name} from {resume_pos // 1024 // 1024} MB "
                     f"(attempt {attempt}/{MAX_RETRIES})...")
        else:
            log.info(f"  Downloading {name} (attempt {attempt}/{MAX_RETRIES})...")

        try:
            with requests.get(
                url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                headers=headers,
            ) as r:
                if r.status_code == 416:
                    # Server says range is out of bounds — file is complete
                    log.info(f"  Server returned 416 — treating as complete.")
                    dest_tmp.rename(dest)
                    return dest
                r.raise_for_status()

                total = int(r.headers.get("content-length", 0)) + resume_pos
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
            # Download complete — promote .part to final file
            dest_tmp.rename(dest)
            log.info(f"  Saved: {dest}")
            return dest

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            print()
            wait = 2 ** attempt
            if attempt < MAX_RETRIES:
                log.warning(f"  Download interrupted ({e.__class__.__name__}). "
                            f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                log.error(f"  Failed after {MAX_RETRIES} attempts.")
                raise

    raise RuntimeError(f"Download of {name} failed after {MAX_RETRIES} retries")


def filter_passthrough(src: Path, dest: Path, label: str):
    log.info(f"Writing {label} (all rows)...")
    count = 0
    with bz2.open(src, "rt", encoding="utf-8") as inf, \
         open(dest, "w", newline="", encoding="utf-8") as outf:
        reader = csv.DictReader(inf)
        if reader.fieldnames is None:
            log.warning(f"  {label}: empty or unreadable file, skipping.")
            return
        writer = csv.DictWriter(
            outf,
            fieldnames=reader.fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for row in reader:
            writer.writerow(row)
            count += 1
    log.info(f"  {count:,} rows written.")


def filter_dockets(src: Path, dest: Path) -> set[str]:
    log.info("Filtering dockets → Illinois courts only...")
    kept: set[str] = set()
    count = skipped = 0
    with bz2.open(src, "rt", encoding="utf-8") as inf, \
         open(dest, "w", newline="", encoding="utf-8") as outf:
        writer = make_writer(outf, DOCKETS_COLS)
        for row in csv.DictReader(inf):
            if safe_str(row.get("court_id")) not in ILLINOIS_COURTS:
                skipped += 1
                continue
            writer.writerow({k: safe_str(row.get(k)) for k in DOCKETS_COLS})
            kept.add(safe_str(row.get("id")))
            count += 1
            if count % 50_000 == 0:
                log.info(f"    {count:,} dockets kept...")
    log.info(f"  {count:,} kept, {skipped:,} skipped.")
    return kept


def filter_clusters(src: Path, dest: Path, docket_ids: set[str]) -> set[str]:
    log.info("Filtering clusters → Illinois dockets only...")
    kept: set[str] = set()
    count = skipped = 0
    with bz2.open(src, "rt", encoding="utf-8") as inf, \
         open(dest, "w", newline="", encoding="utf-8") as outf:
        writer = make_writer(outf, CLUSTERS_COLS)
        for row in csv.DictReader(inf):
            if safe_str(row.get("docket_id")) not in docket_ids:
                skipped += 1
                continue
            writer.writerow({k: safe_str(row.get(k)) for k in CLUSTERS_COLS})
            kept.add(safe_str(row.get("id")))
            count += 1
            if count % 10_000 == 0:
                log.info(f"    {count:,} clusters kept...")
    log.info(f"  {count:,} kept, {skipped:,} skipped.")
    return kept


def filter_opinions(src: Path, dest: Path, cluster_ids: set[str]) -> set[str]:
    """
    Filter opinions to Illinois clusters and resolve the best available text.

    Text priority (written to plain_text column):
      1. plain_text from source if non-empty  ← preferred, usually cleaner
      2. strip_html(html_with_citations)       ← fallback if plain_text absent
      3. empty string

    html_with_citations is always preserved as-is for downstream use.

    Keeping plain_text as the primary source matters because CourtListener's
    html_with_citations is citation-annotated XML that strips poorly for many
    older opinions, while plain_text is a pre-extracted clean version.
    """
    log.info("Filtering opinions → Illinois clusters only...")
    kept: set[str] = set()
    count = skipped = 0
    has_plain = has_html_only = has_neither = 0

    with bz2.open(src, "rt", encoding="utf-8") as inf, \
         open(dest, "w", newline="", encoding="utf-8") as outf:
        writer = make_writer(outf, OPINIONS_COLS)
        for row in csv.DictReader(inf):
            if safe_str(row.get("cluster_id")) not in cluster_ids:
                skipped += 1
                continue

            html      = row.get("html_with_citations", "") or ""
            src_plain = safe_str(row.get("plain_text"))

            # Prefer source plain_text; fall back to stripped HTML
            if src_plain:
                plain = src_plain
                has_plain += 1
            elif html.strip():
                plain = strip_html(html)
                has_html_only += 1
            else:
                plain = ""
                has_neither += 1

            out_row = {k: safe_str(row.get(k)) for k in OPINIONS_COLS}
            out_row["html_with_citations"] = html   # raw HTML preserved
            out_row["plain_text"] = plain            # best available plain text
            writer.writerow(out_row)
            kept.add(safe_str(row.get("id")))
            count += 1
            if count % 5_000 == 0:
                log.info(f"    {count:,} opinions kept...")

    log.info(
        f"  {count:,} kept, {skipped:,} skipped. "
        f"Text sources: {has_plain:,} plain_text, "
        f"{has_html_only:,} html-only, "
        f"{has_neither:,} neither."
    )
    return kept


def filter_citations(src: Path, dest: Path, opinion_ids: set[str]):
    log.info("Filtering citations → Illinois citing opinions only...")
    count = skipped = 0
    with bz2.open(src, "rt", encoding="utf-8") as inf, \
         open(dest, "w", newline="", encoding="utf-8") as outf:
        writer = make_writer(outf, CITATIONS_COLS)
        for row in csv.DictReader(inf):
            if safe_str(row.get("citing_opinion_id")) not in opinion_ids:
                skipped += 1
                continue
            writer.writerow({k: safe_str(row.get(k)) for k in CITATIONS_COLS})
            count += 1
            if count % 100_000 == 0:
                log.info(f"    {count:,} citation edges kept...")
    log.info(f"  {count:,} kept, {skipped:,} skipped.")


def filter_parentheticals(src: Path, dest: Path, opinion_ids: set[str]):
    log.info("Filtering parentheticals → Illinois describing opinions only...")
    count = skipped = 0
    with bz2.open(src, "rt", encoding="utf-8") as inf, \
         open(dest, "w", newline="", encoding="utf-8") as outf:
        writer = make_writer(outf, PARENTHETICALS_COLS)
        for row in csv.DictReader(inf):
            if safe_str(row.get("describing_opinion_id")) not in opinion_ids:
                skipped += 1
                continue
            writer.writerow({k: safe_str(row.get(k)) for k in PARENTHETICALS_COLS})
            count += 1
            if count % 10_000 == 0:
                log.info(f"    {count:,} parentheticals kept...")
    log.info(f"  {count:,} kept, {skipped:,} skipped.")


def filter_opinion_joined_by(src: Path, dest: Path, opinion_ids: set[str]):
    log.info("Filtering opinion_joined_by → Illinois opinions only...")
    count = skipped = 0
    with bz2.open(src, "rt", encoding="utf-8") as inf, \
         open(dest, "w", newline="", encoding="utf-8") as outf:
        reader = csv.DictReader(inf)
        writer = csv.DictWriter(
            outf, fieldnames=reader.fieldnames,
            extrasaction="ignore", lineterminator="\n", quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for row in reader:
            if safe_str(row.get("opinion_id")) not in opinion_ids:
                skipped += 1
                continue
            writer.writerow(row)
            count += 1
    log.info(f"  {count:,} kept, {skipped:,} skipped.")


def filter_cluster_panel(src: Path, dest: Path, cluster_ids: set[str]):
    log.info("Filtering cluster_panel → Illinois clusters only...")
    count = skipped = 0
    with bz2.open(src, "rt", encoding="utf-8") as inf, \
         open(dest, "w", newline="", encoding="utf-8") as outf:
        reader = csv.DictReader(inf)
        writer = csv.DictWriter(
            outf, fieldnames=reader.fieldnames,
            extrasaction="ignore", lineterminator="\n", quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for row in reader:
            if safe_str(row.get("opinioncluster_id")) not in cluster_ids:
                skipped += 1
                continue
            writer.writerow(row)
            count += 1
    log.info(f"  {count:,} kept, {skipped:,} skipped.")


def filter_cluster_non_participating(src: Path, dest: Path, cluster_ids: set[str]):
    log.info("Filtering cluster_non_participating → Illinois clusters only...")
    count = skipped = 0
    with bz2.open(src, "rt", encoding="utf-8") as inf, \
         open(dest, "w", newline="", encoding="utf-8") as outf:
        reader = csv.DictReader(inf)
        writer = csv.DictWriter(
            outf, fieldnames=reader.fieldnames,
            extrasaction="ignore", lineterminator="\n", quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for row in reader:
            if safe_str(row.get("opinioncluster_id")) not in cluster_ids:
                skipped += 1
                continue
            writer.writerow(row)
            count += 1
    log.info(f"  {count:,} kept, {skipped:,} skipped.")


ILLINOIS_FILTERED = [
    "dockets", "clusters", "opinions",
    "citations", "parentheticals",
    "opinion_joined_by", "cluster_panel", "cluster_non_participating",
]

PASSTHROUGH = [
    "courts", "courthouses", "court_appeals_to", "originating_court_info",
    "judges", "judge_positions", "judge_educations", "judge_political",
    "judge_retention", "judge_races", "judge_schools",
]

ALL_TABLES = PASSTHROUGH + ILLINOIS_FILTERED


def run(local_only: bool = False):
    cl_files = list_cl_bulk_files()
    if not cl_files:
        log.error("No bulk file URLs found in CourtListener S3 bucket.")
        sys.exit(1)

    missing = [t for t in ALL_TABLES if t not in cl_files]
    if missing:
        log.warning(f"No bulk file found for: {missing} — those tables will be skipped.")

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("\n--- Download ---")
    staged: dict[str, Path] = {}
    for name in ALL_TABLES:
        if name in cl_files:
            staged[name] = download_cl_file(cl_files[name], name)

    out = {name: LOCAL_OUTPUT_DIR / f"{name}.csv" for name in ALL_TABLES}

    log.info("\n--- Filter ---")
    for name in PASSTHROUGH:
        if name in staged:
            filter_passthrough(staged[name], out[name], label=name)

    docket_ids: set[str] = set()
    if "dockets" in staged:
        docket_ids = filter_dockets(staged["dockets"], out["dockets"])

    cluster_ids: set[str] = set()
    if "clusters" in staged and docket_ids:
        cluster_ids = filter_clusters(staged["clusters"], out["clusters"], docket_ids)

    opinion_ids: set[str] = set()
    if "opinions" in staged and cluster_ids:
        opinion_ids = filter_opinions(staged["opinions"], out["opinions"], cluster_ids)

    if "citations" in staged and opinion_ids:
        filter_citations(staged["citations"], out["citations"], opinion_ids)

    if "parentheticals" in staged and opinion_ids:
        filter_parentheticals(staged["parentheticals"], out["parentheticals"], opinion_ids)

    if "opinion_joined_by" in staged and opinion_ids:
        filter_opinion_joined_by(staged["opinion_joined_by"], out["opinion_joined_by"], opinion_ids)

    if "cluster_panel" in staged and cluster_ids:
        filter_cluster_panel(staged["cluster_panel"], out["cluster_panel"], cluster_ids)

    if "cluster_non_participating" in staged and cluster_ids:
        filter_cluster_non_participating(
            staged["cluster_non_participating"],
            out["cluster_non_participating"],
            cluster_ids,
        )

    if local_only:
        log.info(f"\nLocal-only mode. Filtered CSVs in: {LOCAL_OUTPUT_DIR.resolve()}")
        return

    log.info("\n--- Upload ---")
    _, prefix = get_s3_config()
    for name in ALL_TABLES:
        path = out[name]
        if path.exists():
            upload_to_s3(path, f"{prefix}bulk/{path.name}")

    bucket, prefix = get_s3_config()
    log.info(f"\nDone. Data available at: s3://{bucket}/{prefix}bulk/")


def main():
    parser = argparse.ArgumentParser(
        description="CourtListener Illinois bulk corpus ingestion → S3 CSV pipeline"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Filter CSVs locally without uploading to S3 (useful for testing).",
    )
    args = parser.parse_args()
    run(local_only=args.local_only)


if __name__ == "__main__":
    main()