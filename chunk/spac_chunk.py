"""
SPAC publication chunker.

Reads spac/spac_corpus.jsonl from the raw S3 bucket and writes
spac/spac_chunks.jsonl to the chunked S3 bucket. Defaults to S3 I/O;
use --local-only for development.

Chunking strategy:
  1. Strip page-repeat headers (Page N of M patterns)
  2. Strip ToC dot-leader lines
  3. Split at all-caps section headings (>= 3 words)
  4. Token-aware paragraph accumulation within sections
     (target 600 tokens, max 800, last-paragraph overlap)
  5. Merge micro-chunks (< MIN_CHUNK_TOKENS) into neighbors

Usage:
    python3 chunk/spac_chunk.py            # S3 -> S3
    python3 chunk/spac_chunk.py --local-only
    python3 chunk/spac_chunk.py --limit 20 --local-only
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

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
    prefix = os.environ.get("SPAC_S3_PREFIX", "spac/").rstrip("/") + "/"
    return {
        "raw_bucket":     _require_env("RAW_S3_BUCKET"),
        "raw_key":        f"{prefix}spac_corpus.jsonl",
        "chunked_bucket": _require_env("CHUNKED_S3_BUCKET"),
        "chunked_key":    f"{prefix}spac_chunks.jsonl",
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

# Matches lines containing "Page N of M" (PDF page-repeat headers)
_PAGE_HEADER_RE = re.compile(
    r"[^\n]*\bPage \d+ of \d+\b[^\n]*\n?",
    re.IGNORECASE,
)

# Matches dot-leader lines: "Introduction ............... 4"
# Strip if 4+ dot run AND (dot density > 20% OR line ends with a digit/page-number)
_DOT_RUN_RE = re.compile(r"\.{4,}")
_ENDS_WITH_DIGIT_RE = re.compile(r"\d\s*$")


def strip_page_headers(text: str) -> str:
    return _PAGE_HEADER_RE.sub("", text).strip()


_ENDS_WITH_DOTS_RE = re.compile(r"\.{4,}\s*$")


def strip_toc_lines(text: str) -> str:
    lines = text.split("\n")
    result = []
    for line in lines:
        if _DOT_RUN_RE.search(line):
            dot_count = line.count(".")
            line_len = len(line)
            pct = dot_count / line_len if line_len > 0 else 0
            # High dot density → clear ToC/dot-leader line
            if pct > 0.20:
                continue
            # Lower density but ends with a page number → ToC entry
            if _ENDS_WITH_DIGIT_RE.search(line):
                continue
            # Line ends with trailing dots (truncated ToC entry, page number on next line)
            if _ENDS_WITH_DOTS_RE.search(line):
                continue
        result.append(line)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

# All-caps line: starts with A-Z, remainder is A-Z / 0-9 / spaces / select punctuation
_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9\s\(\)\-/,]{3,59}$")


def is_section_heading(line: str) -> bool:
    stripped = line.strip()
    if not _HEADING_RE.match(stripped):
        return False
    words = stripped.split()
    return len(words) >= 3


def split_at_headings(text: str) -> list[tuple[str, str]]:
    """Split text at all-caps headings; return list of (heading, body) pairs."""
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in lines:
        if is_section_heading(line):
            body = "\n".join(current_lines).strip()
            if body or current_heading:
                sections.append((current_heading, body))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body or current_heading:
        sections.append((current_heading, body))

    return sections if sections else [("", text)]


# ---------------------------------------------------------------------------
# Token-aware splitting  (same pattern as idoc_chunk.py)
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


def split_body(body: str) -> list[str]:
    """Split a section body into token-bounded chunks."""
    if count_tokens(body) <= MAX_TOKENS:
        return [body]
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        sentences = _sentence_split(body)
        paragraphs = sentences if len(sentences) > 1 else [body]
    return _accumulate(paragraphs)


# ---------------------------------------------------------------------------
# Micro-chunk merge
# ---------------------------------------------------------------------------

def _merge_micro_chunks(
    sections: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Merge chunks below MIN_CHUNK_TOKENS into their nearest neighbor."""
    if len(sections) <= 1:
        return sections

    result: list[list] = [[h, b] for h, b in sections]
    i = 0
    while i < len(result):
        _, body = result[i]
        if count_tokens(body) < MIN_CHUNK_TOKENS:
            if i + 1 < len(result):
                combined = (body + "\n\n" + result[i + 1][1]).strip()
                if count_tokens(combined) <= MAX_TOKENS:
                    result[i + 1][1] = combined
                    result.pop(i)
                    continue
            elif i > 0:
                combined = (result[i - 1][1] + "\n\n" + body).strip()
                if count_tokens(combined) <= MAX_TOKENS:
                    result[i - 1][1] = combined
                    result.pop(i)
                    continue
        i += 1
    return [(h, b) for h, b in result]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class SpacChunk:
    chunk_id:        str
    chunk_index:     int
    chunk_total:     int
    source:          str
    text:            str
    enriched_text:   str
    token_count:     int
    section_heading: str
    record_id:       str
    title:           str
    category:        str
    year:            str
    url:             str
    chunked_at:      str


def _make_enriched(rec: dict, section_heading: str, text: str) -> str:
    header = f"SPAC {rec.get('category', '')} ({rec.get('year', '')}): {rec.get('title', '')}"
    if section_heading:
        header += f" — {section_heading}"
    return f"{header}\n\n{text}"


# ---------------------------------------------------------------------------
# Per-record chunking
# ---------------------------------------------------------------------------

def chunk_record(rec: dict) -> list[SpacChunk]:
    raw_text = rec.get("text", "").strip()
    if not raw_text:
        return []

    rec_id   = rec.get("id", "")
    title    = rec.get("title", "")
    category = rec.get("category", "")
    year     = rec.get("year", "")
    url      = rec.get("url", "")

    text = strip_page_headers(raw_text)
    text = strip_toc_lines(text)

    if count_tokens(text) < MIN_CHUNK_TOKENS:
        return []

    sections = split_at_headings(text)

    flat: list[tuple[str, str]] = []
    for heading, body in sections:
        # Skip heading-only sections (no body) — document artifacts, not substantive content
        if not body.strip():
            continue
        if count_tokens(body) <= MAX_TOKENS:
            flat.append((heading, body))
        else:
            for chunk_text in split_body(body):
                flat.append((heading, chunk_text))

    flat = _merge_micro_chunks(flat)
    filtered = [(h, t) for h, t in flat if count_tokens(t) >= MIN_CHUNK_TOKENS]
    if not filtered:
        return []

    total  = len(filtered)
    result = []
    for i, (heading, chunk_text) in enumerate(filtered):
        result.append(SpacChunk(
            chunk_id        = f"{rec_id}_c{i}",
            chunk_index     = i,
            chunk_total     = total,
            source          = "spac",
            text            = chunk_text,
            enriched_text   = _make_enriched(rec, heading, chunk_text),
            token_count     = count_tokens(chunk_text),
            section_heading = heading,
            record_id       = rec_id,
            title           = title,
            category        = category,
            year            = year,
            url             = url,
            chunked_at      = datetime.now(timezone.utc).isoformat(),
        ))
    return result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_records(records: list[dict]) -> list[dict]:
    """Keep the longest-text copy when the raw corpus has duplicate record IDs."""
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
    log.info("=== SPAC chunking pipeline starting ===")
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

    all_chunks: list[dict] = []
    skipped = 0
    for rec in records:
        chunks = chunk_record(rec)
        if not chunks:
            skipped += 1
            continue
        all_chunks.extend(asdict(c) for c in chunks)

    log.info(
        "Produced %d chunks from %d records (%d skipped — stubs/empty)",
        len(all_chunks), len(records), skipped,
    )
    single = sum(1 for c in all_chunks if c["chunk_total"] == 1)
    multi  = len({c["record_id"] for c in all_chunks if c["chunk_total"] > 1})
    log.info("Single-chunk records: %d  |  Multi-chunk records: %d", single, multi)

    if local_only:
        write_local(all_chunks, LOCAL_OUTPUT_DIR / "spac_chunks.jsonl")
    else:
        write_s3(all_chunks, cfg["chunked_bucket"], cfg["chunked_key"], cfg["aws_region"])

    log.info("=== SPAC chunking pipeline complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk SPAC publications JSONL → chunks JSONL")
    parser.add_argument("--local-only", action="store_true",
                        help="Write output locally instead of uploading to S3.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N records (0 = all). For testing.")
    args = parser.parse_args()
    run(local_only=args.local_only, limit=args.limit)


if __name__ == "__main__":
    main()
