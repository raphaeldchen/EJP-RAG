import argparse
import io
import json
import logging
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from core.models import Chunk
from chunk.opinion_utils import (
    TARGET_TOKENS,
    MAX_TOKENS,
    OVERLAP_TOKENS,
    MIN_CHUNK_TOKENS,
    count_tokens,
    token_split,
    strip_html,
    safe_str,
    safe_int,
    safe_float,
    is_noise_chunk,
    detect_sections,
    split_section,
    _opinion_enriched_text,
    _opinion_display_citation,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        log.error(f"Required environment variable {key!r} is not set.")
        sys.exit(1)
    return val

def get_config() -> dict:
    raw_bucket     = _require_env("RAW_S3_BUCKET")
    chunked_bucket = _require_env("CHUNKED_S3_BUCKET")
    cl_prefix      = os.environ.get("RAW_COURTLISTENER_S3_PREFIX", "courtlistener/").rstrip("/")
    return {
        "raw_bucket":     raw_bucket,
        "raw_prefix":     f"{cl_prefix}/bulk",
        "chunked_bucket": chunked_bucket,
        "chunked_prefix": f"{cl_prefix}/bulk",
    }

LOCAL_OUTPUT_DIR = Path("./data_files/chunked_output")

PRECEDENTIAL_WEIGHT = {
    "Published":   1.0,
    "Unpublished": 0.5,
    "Errata":      0.1,
    "Separate":    0.4,
    "In-chambers": 0.3,
    "Relating-to": 0.2,
    "Unknown":     0.3,
    "":            0.3,
}

OPINION_TYPE_LABELS = {
    "010combined":            "combined",
    "015unamimous":           "unanimous",
    "020lead":                "majority",
    "025plurality":           "plurality",
    "030concurrence":         "concurrence",
    "035concurrence-in-part": "concurrence_in_part",
    "040dissent":             "dissent",
    "050addendum":            "addendum",
    "060remittitur":          "remittitur",
    "070rehearing":           "rehearing",
    "080on-the-merits":       "on_the_merits",
    "090on-motion":           "on_motion",
}

MAJORITY_TYPES = {"combined", "unanimous", "majority", "plurality"}

COURT_LABELS = {
    "ill":      "Illinois Supreme Court",
    "illappct": "Illinois Appellate Court",
}

def _par_enriched_text(case_name: str, date_filed: str, court_label: str, text: str) -> str:
    header_parts = [x for x in [case_name, date_filed, court_label] if x]
    header = " | ".join(header_parts)
    prefix = f"{header} [parenthetical]" if header else "[parenthetical]"
    return f"{prefix}\n\n{text}"


def _par_display_citation(case_name: str, date_filed: str) -> str:
    year = date_filed[:4] if date_filed and len(date_filed) >= 4 else ""
    base = f"{case_name} ({year})" if case_name and year else case_name or ""
    return f"{base} — parenthetical" if base else "parenthetical"

def s3_client():
    return boto3.client("s3")

def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    log.info(f"  Reading s3://{bucket}/{key}")
    obj = s3_client().get_object(Bucket=bucket, Key=key)
    return pd.read_csv(
        io.BytesIO(obj["Body"].read()),
        low_memory=False,
        dtype=str,
        keep_default_na=False,
    )

def write_jsonl_to_s3(records: list[dict], bucket: str, key: str):
    log.info(f"  Writing {len(records):,} records → s3://{bucket}/{key}")
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/x-ndjson",
    )
    log.info("  Upload complete.")

def write_jsonl_local(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"  Saved locally: {path}  ({len(records):,} records)")

def row_get(row, key: str) -> str:
    try:
        return safe_str(row[key])
    except (KeyError, TypeError):
        return ""

def build_lookup_maps(
    clusters_df: pd.DataFrame,
    dockets_df: pd.DataFrame,
) -> tuple[dict, dict]:
    cluster_map = {safe_str(row["id"]): row for _, row in clusters_df.iterrows()}
    docket_map  = {safe_str(row["id"]): row for _, row in dockets_df.iterrows()}
    return cluster_map, docket_map

def chunk_opinion(
    opinion_row: pd.Series,
    cluster_map: dict,
    docket_map: dict,
) -> list[Chunk]:
    opinion_id   = safe_str(opinion_row.get("id"))
    cluster_id   = safe_str(opinion_row.get("cluster_id"))
    opinion_type = safe_str(opinion_row.get("type"))
    plain = safe_str(opinion_row.get("plain_text"))
    if not plain:
        html  = safe_str(opinion_row.get("html_with_citations"))
        plain = strip_html(html) if html else ""
    if not plain:
        return []
    cluster     = cluster_map.get(cluster_id, {})
    docket_id   = row_get(cluster, "docket_id")
    docket      = docket_map.get(docket_id, {})
    prec_status = row_get(cluster, "precedential_status")
    court_id    = row_get(docket, "court_id")
    op_label    = OPINION_TYPE_LABELS.get(opinion_type, opinion_type)
    case_name        = row_get(cluster, "case_name")
    case_name_short  = row_get(cluster, "case_name_short")
    date_filed       = row_get(cluster, "date_filed")
    court_label      = COURT_LABELS.get(court_id, court_id)
    display_citation = _opinion_display_citation(case_name_short, date_filed)
    author           = safe_str(opinion_row.get("author_str"))
    per_curiam       = safe_str(opinion_row.get("per_curiam")).lower() == "true"
    judges           = row_get(cluster, "judges")
    prec_weight      = PRECEDENTIAL_WEIGHT.get(prec_status, 0.3)
    citation_count   = safe_int(row_get(cluster, "citation_count"))
    docket_number    = row_get(docket, "docket_number")
    nature_of_suit   = row_get(docket, "nature_of_suit")
    cause            = row_get(docket, "cause")
    date_terminated  = row_get(docket, "date_terminated")

    sections = detect_sections(plain)
    flat: list[tuple[str, str, int]] = []
    for sec_idx, (heading, body) in enumerate(sections):
        for h, text in split_section(heading, body):
            flat.append((h, text, sec_idx))

    # Filter noise/short, carrying token_count to avoid recomputing
    prelim: list[tuple[str, str, int, int]] = []  # (heading, text, sec_idx, token_count)
    for heading, chunk_text, sec_idx in flat:
        token_count = count_tokens(chunk_text)
        if token_count < MIN_CHUNK_TOKENS:
            continue
        if is_noise_chunk(chunk_text, token_count):
            continue
        prelim.append((heading, chunk_text, sec_idx, token_count))

    chunk_total = len(prelim)
    result: list[Chunk] = []
    for chunk_index, (heading, chunk_text, sec_idx, token_count) in enumerate(prelim):
        enriched = _opinion_enriched_text(case_name_short, date_filed, court_label, heading, chunk_text)
        result.append(Chunk(
            chunk_id         = f"{opinion_id}_c{chunk_index}",
            parent_id        = opinion_id,
            chunk_index      = chunk_index,
            chunk_total      = chunk_total,
            text             = chunk_text,
            enriched_text    = enriched,
            source           = "courtlistener",
            token_count      = token_count,
            display_citation = display_citation,
            metadata={
                "chunk_type":          "opinion_section" if heading else "opinion_paragraph",
                "section_heading":     heading,
                "section_index":       sec_idx,
                "opinion_id":          opinion_id,
                "opinion_type":        op_label,
                "is_majority":         op_label in MAJORITY_TYPES,
                "author":              author,
                "per_curiam":          per_curiam,
                "cluster_id":          cluster_id,
                "case_name":           case_name,
                "case_name_short":     case_name_short,
                "date_filed":          date_filed,
                "judges":              judges,
                "precedential_status": prec_status,
                "precedential_weight": prec_weight,
                "citation_count":      citation_count,
                "docket_id":           docket_id,
                "court_id":            court_id,
                "court_label":         court_label,
                "docket_number":       docket_number,
                "nature_of_suit":      nature_of_suit,
                "cause":               cause,
                "date_terminated":     date_terminated,
            },
        ))
    return result

def chunk_parentheticals(
    parentheticals_df: pd.DataFrame,
    cluster_map: dict,
    docket_map: dict,
    opinion_to_cluster: dict[str, str],
) -> list[Chunk]:
    result: list[Chunk] = []
    for _, row in parentheticals_df.iterrows():
        text = safe_str(row.get("text"))
        if not text:
            continue
        describing_id = safe_str(row.get("describing_opinion_id"))
        cluster_id    = opinion_to_cluster.get(describing_id, "")
        cluster       = cluster_map.get(cluster_id, {})
        docket_id     = row_get(cluster, "docket_id")
        docket        = docket_map.get(docket_id, {})
        court_id      = row_get(docket, "court_id")
        prec_status   = row_get(cluster, "precedential_status")
        case_name     = row_get(cluster, "case_name")
        date_filed    = row_get(cluster, "date_filed")
        court_label   = COURT_LABELS.get(court_id, court_id)
        par_id        = safe_str(row.get("id"))
        result.append(Chunk(
            chunk_id         = f"par_{par_id}",
            parent_id        = describing_id,
            chunk_index      = 0,
            chunk_total      = 1,
            text             = text,
            enriched_text    = _par_enriched_text(case_name, date_filed, court_label, text),
            source           = "courtlistener",
            token_count      = count_tokens(text),
            display_citation = _par_display_citation(case_name, date_filed),
            metadata={
                "chunk_type":            "parenthetical",
                "describing_opinion_id": describing_id,
                "described_opinion_id":  safe_str(row.get("described_opinion_id")),
                "score":                 safe_float(row.get("score")),
                "case_name":             case_name,
                "date_filed":            date_filed,
                "court_id":              court_id,
                "court_label":           court_label,
                "precedential_status":   prec_status,
                "precedential_weight":   PRECEDENTIAL_WEIGHT.get(prec_status, 0.3),
            },
        ))
    return result

def run(local_only: bool = False, limit: int = 0):
    cfg            = get_config()
    raw_bucket     = cfg["raw_bucket"]
    raw_prefix     = cfg["raw_prefix"]
    chunked_bucket = cfg["chunked_bucket"]
    chunked_prefix = cfg["chunked_prefix"]
    log.info("Loading raw tables from S3...")
    clusters_df       = read_csv_from_s3(raw_bucket, f"{raw_prefix}/clusters.csv")
    dockets_df        = read_csv_from_s3(raw_bucket, f"{raw_prefix}/dockets.csv")
    opinions_df       = read_csv_from_s3(raw_bucket, f"{raw_prefix}/opinions.csv")
    par_key = f"{raw_prefix}/parentheticals.csv"
    try:
        parentheticals_df = read_csv_from_s3(raw_bucket, par_key)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            log.warning(f"  s3://{raw_bucket}/{par_key} not found — skipping parentheticals.")
            parentheticals_df = pd.DataFrame()
        else:
            raise
    log.info(
        f"  {len(opinions_df):,} opinions | {len(clusters_df):,} clusters | "
        f"{len(dockets_df):,} dockets | {len(parentheticals_df):,} parentheticals"
    )
    has_html  = (opinions_df["html_with_citations"].str.strip() != "").sum()
    has_plain = (opinions_df["plain_text"].str.strip() != "").sum() \
                if "plain_text" in opinions_df.columns else 0
    log.info(
        f"  Text coverage: {has_html:,} html_with_citations | "
        f"{has_plain:,} plain_text | "
        f"{len(opinions_df) - max(has_html, has_plain):,} have NO text"
    )
    log.info("Building metadata lookup maps...")
    cluster_map, docket_map = build_lookup_maps(clusters_df, dockets_df)
    opinion_to_cluster = {
        safe_str(row["id"]): safe_str(row["cluster_id"])
        for _, row in opinions_df.iterrows()
    }
    opinions_df = opinions_df.assign(
        _text_len=(
            opinions_df["html_with_citations"].str.len().fillna(0)
            + opinions_df["plain_text"].str.len().fillna(0)
        )
    ).sort_values("_text_len", ascending=False).drop(columns="_text_len")
    if limit:
        log.info(f"  Limiting to first {limit} opinions (by text length).")
        opinions_df = opinions_df.head(limit)
    log.info("Chunking opinions...")
    opinion_chunks: list[Chunk] = []
    opinions_with_chunks = 0
    skipped_no_text   = 0
    skipped_too_short = 0
    for i, (_, row) in enumerate(opinions_df.iterrows()):
        chunks = chunk_opinion(row, cluster_map, docket_map)
        if not chunks:
            skipped_no_text += 1
            continue
        # chunk_opinion already filters noise/short; all returned chunks are useful
        opinion_chunks.extend(chunks)
        opinions_with_chunks += 1
        if (i + 1) % 1000 == 0:
            log.info(f"  {i + 1:,} opinions processed → {len(opinion_chunks):,} chunks...")
    log.info(
        f"  Done. {len(opinion_chunks):,} chunks produced from {opinions_with_chunks:,} opinions. "
        f"Skipped: {skipped_no_text} (no text)."
    )
    log.info("Chunking parentheticals...")
    par_chunks = chunk_parentheticals(
        parentheticals_df, cluster_map, docket_map, opinion_to_cluster
    )
    log.info(f"  {len(par_chunks):,} parenthetical chunks produced.")
    opinion_records = [asdict(c) for c in opinion_chunks]
    par_records     = [asdict(c) for c in par_chunks]
    if opinion_records:
        tokens = [r["token_count"] for r in opinion_records]
        log.info(
            f"  Chunk token stats: avg {sum(tokens)/len(tokens):.0f} | "
            f"min {min(tokens)} | max {max(tokens)}"
        )
    if local_only:
        write_jsonl_local(opinion_records, LOCAL_OUTPUT_DIR / "opinion_chunks.jsonl")
        write_jsonl_local(par_records,     LOCAL_OUTPUT_DIR / "parenthetical_chunks.jsonl")
        log.info(f"  Output in: {LOCAL_OUTPUT_DIR.resolve()}")
    else:
        write_jsonl_to_s3(opinion_records, chunked_bucket, f"{chunked_prefix}/opinion_chunks.jsonl")
        write_jsonl_to_s3(par_records,     chunked_bucket, f"{chunked_prefix}/parenthetical_chunks.jsonl")
        log.info(f"  Output at s3://{chunked_bucket}/{chunked_prefix}/")

def main():
    parser = argparse.ArgumentParser(
        description="Chunk Illinois legal corpus CSV → JSONL (S3)"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Write output locally without uploading to S3.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N opinions by text length (0 = all). For testing.",
    )
    args = parser.parse_args()
    run(local_only=args.local_only, limit=args.limit)

if __name__ == "__main__":
    main()