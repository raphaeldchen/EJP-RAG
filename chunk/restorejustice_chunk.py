"""
Restore Justice IL website chunker.

Reads restorejustice/restorejustice_corpus.jsonl from the raw S3 bucket and
writes restorejustice/restorejustice_chunks.jsonl to the chunked S3 bucket.
Defaults to S3 I/O; use --local-only for development.

Corpus: 17 records — mix of HTML sub-pages and PDFs.  Most records are
already short (< 800 tokens) and become single chunks.  The handful of
longer records (advocacy trainings, self-advocacy notes) are split by
paragraph with token-aware accumulation.

Chunking strategy:
  1. Collapse excess whitespace (HTML extraction leaves blank-line clusters)
  2. If total tokens <= MAX: emit as single chunk
  3. Otherwise: paragraph-aware token accumulation
     (target 600 tokens, max 800, last-paragraph overlap)
  4. Merge micro-chunks (< MIN_CHUNK_TOKENS) into neighbors

Usage:
    python3 chunk/restorejustice_chunk.py            # S3 -> S3
    python3 chunk/restorejustice_chunk.py --local-only
"""

import argparse
import dataclasses
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Generator

from core.models import Chunk

import boto3
import tiktoken
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        log.error("Required environment variable %r is not set.", key)
        sys.exit(1)
    return val


def get_config() -> dict:
    return {
        "raw_bucket":     _require_env("RAW_S3_BUCKET"),
        "raw_key":        "restorejustice/restorejustice_corpus.jsonl",
        "chunked_bucket": _require_env("CHUNKED_S3_BUCKET"),
        "chunked_key":    "restorejustice/restorejustice_chunks.jsonl",
        "aws_region":     os.getenv("AWS_REGION"),
    }


LOCAL_OUTPUT_DIR = Path("./data_files/chunked_output")

TARGET_TOKENS    = int(os.getenv("TARGET_TOKENS",    "600"))
MAX_TOKENS       = int(os.getenv("MAX_TOKENS",       "800"))
OVERLAP_TOKENS   = 75
MIN_CHUNK_TOKENS = int(os.getenv("MIN_CHUNK_TOKENS", "40"))
ENCODING_NAME    = "cl100k_base"

_enc = tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def token_split(text: str) -> list[str]:
    tokens = _enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + MAX_TOKENS, len(tokens))
        chunks.append(_enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - OVERLAP_TOKENS
    return chunks


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Collapse runs of 3+ blank lines to a single blank line (HTML extraction noise)
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_NBSP_RE       = re.compile(r"\xa0")


def normalize_whitespace(text: str) -> str:
    text = _NBSP_RE.sub(" ", text)
    return _BLANK_RUN_RE.sub("\n\n", text).strip()


# ---------------------------------------------------------------------------
# Token-aware splitting
# ---------------------------------------------------------------------------

def _sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


def _accumulate(units: list[str]) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = count_tokens(unit)
        if unit_tokens > MAX_TOKENS:
            if current:
                result.append("\n\n".join(current))
                current, current_tokens = [], 0
            result.extend(token_split(unit))
            continue
        if current_tokens + unit_tokens > TARGET_TOKENS and current:
            result.append("\n\n".join(current))
            overlap        = current[-1]
            overlap_tokens = count_tokens(overlap)
            if overlap_tokens + unit_tokens <= MAX_TOKENS:
                current        = [overlap, unit]
                current_tokens = overlap_tokens + unit_tokens
            else:
                current        = [unit]
                current_tokens = unit_tokens
        else:
            current.append(unit)
            current_tokens += unit_tokens
    if current:
        result.append("\n\n".join(current))
    return result


def split_text(text: str) -> list[str]:
    """Split text into token-bounded chunks."""
    if count_tokens(text) <= MAX_TOKENS:
        return [text]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        sentences = _sentence_split(text)
        paragraphs = sentences if len(sentences) > 1 else [text]
    return _accumulate(paragraphs)


# ---------------------------------------------------------------------------
# Micro-chunk merge
# ---------------------------------------------------------------------------

def _merge_micro_chunks(chunks: list[str]) -> list[str]:
    if len(chunks) <= 1:
        return chunks
    result = list(chunks)
    i = 0
    while i < len(result):
        if count_tokens(result[i]) < MIN_CHUNK_TOKENS:
            if i + 1 < len(result):
                combined = result[i] + "\n\n" + result[i + 1]
                if count_tokens(combined) <= MAX_TOKENS:
                    result[i + 1] = combined.strip()
                    result.pop(i)
                    continue
            elif i > 0:
                combined = result[i - 1] + "\n\n" + result[i]
                if count_tokens(combined) <= MAX_TOKENS:
                    result[i - 1] = combined.strip()
                    result.pop(i)
                    continue
        i += 1
    return result


# ---------------------------------------------------------------------------
# Output schema  — uses shared Chunk dataclass from core.models
# ---------------------------------------------------------------------------

def _make_enriched(rec: dict, text: str) -> str:
    header = f"Restore Justice IL — {rec.get('page_title', '')}"
    section = rec.get("section", "")
    if section and section != rec.get("page_title", "").lower().replace(" ", "-"):
        header += f" ({section})"
    return f"{header}\n\n{text}"


# ---------------------------------------------------------------------------
# Per-record chunking
# ---------------------------------------------------------------------------

def chunk_record(rec: dict) -> list[Chunk]:
    raw_text = rec.get("text", "").strip()
    if not raw_text:
        return []

    rec_id     = rec.get("id", "")
    page_title = rec.get("page_title", "")
    section    = rec.get("section", "")
    url        = rec.get("url", "")

    text = normalize_whitespace(raw_text)

    if count_tokens(text) < MIN_CHUNK_TOKENS:
        return []

    raw_chunks = split_text(text)
    merged     = _merge_micro_chunks(raw_chunks)
    filtered   = [t for t in merged if count_tokens(t) >= MIN_CHUNK_TOKENS]
    if not filtered:
        return []

    total  = len(filtered)
    result = []
    for i, chunk_text in enumerate(filtered):
        result.append(Chunk(
            chunk_id        = f"{rec_id}_c{i}",
            parent_id       = rec_id,
            chunk_index     = i,
            chunk_total     = total,
            text            = chunk_text,
            enriched_text   = _make_enriched(rec, chunk_text),
            source          = "restorejustice",
            token_count     = count_tokens(chunk_text),
            display_citation= f"Restore Justice IL — {page_title}",
            metadata        = {
                "record_id":  rec_id,
                "page_title": page_title,
                "section":    section,
                "url":        url,
            },
        ))
    return result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_records(records: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for rec in records:
        rid = rec["id"]
        if rid not in seen or len(rec.get("text", "")) > len(seen[rid].get("text", "")):
            seen[rid] = rec
    return list(seen.values())


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def iter_records(raw: str) -> Generator[dict, None, None]:
    for i, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("Malformed JSON on line %d: %s", i, exc)


def read_s3(bucket: str, key: str, region: str | None) -> str:
    log.info("Reading s3://%s/%s", bucket, key)
    kwargs = {"region_name": region} if region else {}
    obj = boto3.client("s3", **kwargs).get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")


def write_local(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info("Saved locally: %s  (%d chunks)", path, len(chunks))


def write_s3(chunks: list[dict], bucket: str, key: str, region: str | None) -> None:
    log.info("Writing %d chunks → s3://%s/%s", len(chunks), bucket, key)
    payload = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n"
    kwargs = {"region_name": region} if region else {}
    boto3.client("s3", **kwargs).put_object(
        Bucket=bucket,
        Key=key,
        Body=payload.encode("utf-8"),
        ContentType="application/x-ndjson",
    )
    log.info("Upload complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(local_only: bool = False, limit: int = 0) -> None:
    log.info("=== Restore Justice chunking pipeline starting ===")
    cfg     = get_config()
    raw     = read_s3(cfg["raw_bucket"], cfg["raw_key"], cfg["aws_region"])
    records = list(iter_records(raw))
    log.info("Loaded %d raw records", len(records))

    deduped = deduplicate_records(records)
    if len(deduped) < len(records):
        log.info("De-duplicated %d → %d records", len(records), len(deduped))
    records = deduped

    if limit:
        log.info("Limiting to first %d records.", limit)
        records = records[:limit]

    all_chunks: list[Chunk] = []
    skipped = 0
    for rec in records:
        chunks = chunk_record(rec)
        if not chunks:
            skipped += 1
            continue
        all_chunks.extend(chunks)

    log.info(
        "Produced %d chunks from %d records (%d skipped — stubs/empty)",
        len(all_chunks), len(records), skipped,
    )
    single = sum(1 for c in all_chunks if c.chunk_total == 1)
    multi  = len({c.parent_id for c in all_chunks if c.chunk_total > 1})
    log.info("Single-chunk records: %d  |  Multi-chunk records: %d", single, multi)

    serialized = [dataclasses.asdict(c) for c in all_chunks]
    if local_only:
        write_local(serialized, LOCAL_OUTPUT_DIR / "restorejustice_chunks.jsonl")
    else:
        write_s3(serialized, cfg["chunked_bucket"], cfg["chunked_key"], cfg["aws_region"])

    log.info("=== Restore Justice chunking pipeline complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk Restore Justice JSONL → chunks JSONL")
    parser.add_argument("--local-only", action="store_true",
                        help="Write output locally instead of uploading to S3.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N records (0 = all). For testing.")
    args = parser.parse_args()
    run(local_only=args.local_only, limit=args.limit)


if __name__ == "__main__":
    main()
