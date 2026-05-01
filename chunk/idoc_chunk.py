"""
IDOC directives and reentry resources chunker.

Reads idoc/idoc_corpus.jsonl from the raw S3 bucket and writes
idoc/idoc_chunks.jsonl to the chunked S3 bucket.

Directive structure (PDF-extracted):
  - Initial header block  (number, title, effective date, authority, referenced forms)
  - I. POLICY             (Roman numeral sections)
  - II. PROCEDURE
      A. Purpose           (uppercase letter subsections)
      B. Applicability
      ...
  - Page repeat headers   (artifact of PDF→text: "Illinois Dept of Corrections /
                           Administrative Directive / Page N of M / Number: ...")
                           → stripped before any splitting

Chunking strategy:
  1. Strip page-repeat headers
  2. Emit initial header block as chunk 0 (contains authority citations + DOC form refs)
  3. Split remaining body at Roman numeral section boundaries
  4. Within large sections, split at uppercase letter subsection boundaries
  5. Token-aware packing: 600-token target, 800-token hard cap

Usage:
    python3 chunk/idoc_chunk.py            # S3 → S3
    python3 chunk/idoc_chunk.py --local-only
    python3 chunk/idoc_chunk.py --limit 10 --local-only
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
        "raw_key":        "idoc/idoc_corpus.jsonl",
        "chunked_bucket": _require_env("CHUNKED_S3_BUCKET"),
        "chunked_key":    "idoc/idoc_chunks.jsonl",
        "aws_region":     os.getenv("AWS_REGION"),
    }

LOCAL_OUTPUT_DIR = Path("./data_files/chunked_output")

TARGET_TOKENS    = 600
MAX_TOKENS       = 800
OVERLAP_TOKENS   = 75
MIN_CHUNK_TOKENS = 40
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
# Cleaning — strip PDF page-repeat headers
#
# Pattern that appears at every PDF page break:
#   Illinois Department of Corrections
#   Administrative Directive
#   Page N of M
#   Number:
#   04.XX.XXX
#   Title:
#   <title text>
#   Effective:
#   <date>
# ---------------------------------------------------------------------------

_PAGE_HEADER_RE = re.compile(
    # Leading \n+ means this never matches at position 0 (the genuine header block)
    r"\n+Illinois Department of Corrections[^\n]*\n"
    r"(?:[^\n]*\n){0,3}"               # 0–3 lines (blank line + "Administrative Directive")
    r"[^\n]*Administrative Directive[^\n]*\n"
    r"Page \d+ of \d+[^\n]*\n"        # "Page N of M" — key discriminator vs. initial header
    r"(?:[^\n]*\n){0,8}"               # Number / Title / Effective block (≤ 8 lines)
    r"Effective:[^\n]*\n"
    r"[^\n]*\n"                        # date line
    r"(?:\s*\n)*",                     # trailing blank lines
    re.IGNORECASE,
)

def strip_page_headers(text: str) -> str:
    return _PAGE_HEADER_RE.sub("\n", text).strip()


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

# Roman numeral sections: "I. POLICY", "II. PROCEDURE", "III. DEFINITIONS"
_ROMAN_RE = re.compile(
    r"^((?:X{0,3})(?:IX|IV|V?I{0,3}))\.\s+([A-Z][A-Z\s/\-]{1,60})\s*$",
    re.MULTILINE,
)

# Uppercase letter subsections: "A. Purpose", "B. Applicability", etc.
# Require title-case or all-caps title to avoid matching mid-sentence "A. Smith said..."
_LETTER_SUB_RE = re.compile(
    r"^([A-Z])\.\s+([A-Z][A-Za-z\s/\-]{2,60})\s*$",
    re.MULTILINE,
)

def _split_at_pattern(text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """Split text at regex boundaries; return list of (heading, body) pairs."""
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, m in enumerate(matches):
        heading = m.group(0).strip()
        start   = m.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body    = text[start:end].strip()
        if body:
            sections.append((heading, body))

    return sections if sections else [("", text)]


# ---------------------------------------------------------------------------
# Token-aware accumulation (same pattern as courtlistener_chunk.py)
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
# Output schema  — uses shared Chunk dataclass from core.models
# ---------------------------------------------------------------------------

def _make_enriched(rec: dict, section_heading: str, text: str) -> str:
    directive_num = rec.get("id", "").replace("idoc-dir-", "").replace("_", ".")
    return (
        f"IDOC Administrative Directive {directive_num}: {rec.get('title', '')}"
        + (f" — {section_heading}" if section_heading else "")
        + f"\n\n{text}"
    )


# ---------------------------------------------------------------------------
# Per-record chunking
# ---------------------------------------------------------------------------

def chunk_record(rec: dict) -> list[Chunk]:
    raw_text = rec.get("text", "").strip()
    if not raw_text:
        return []

    rec_id   = rec.get("id", "")
    title    = rec.get("title", "")
    source   = rec.get("source", "")
    doc_type = rec.get("doc_type", "")

    # Non-directive records (e.g. idoc_reentry): single chunk if long enough
    if source != "idoc_directive":
        tokens = count_tokens(raw_text)
        if tokens < MIN_CHUNK_TOKENS:
            return []
        chunks = token_split(raw_text) if tokens > MAX_TOKENS else [raw_text]
        result = []
        for i, text in enumerate(chunks):
            result.append(Chunk(
                chunk_id        = f"{rec_id}_c{i}",
                parent_id       = rec_id,
                chunk_index     = i,
                chunk_total     = len(chunks),
                text            = text,
                enriched_text   = f"{title}\n\n{text}" if title else text,
                source          = source,
                token_count     = count_tokens(text),
                display_citation= title or rec_id,
                metadata        = {
                    "record_id":       rec_id,
                    "doc_type":        doc_type,
                    "title":           title,
                    "section_heading": "",
                    "category":        rec.get("category", ""),
                    "sub_category":    rec.get("sub_category", ""),
                    "url":             rec.get("url", ""),
                },
            ))
        return result

    # Directives: strip page headers, then split by section structure
    text = strip_page_headers(raw_text)

    # Split at Roman numeral section boundaries
    roman_sections = _split_at_pattern(text, _ROMAN_RE)

    flat: list[tuple[str, str]] = []
    for heading, body in roman_sections:
        if count_tokens(body) <= MAX_TOKENS:
            flat.append((heading, body))
        else:
            # Try to split further at uppercase letter subsections
            letter_subs = _split_at_pattern(body, _LETTER_SUB_RE)
            if len(letter_subs) > 1:
                for sub_heading, sub_body in letter_subs:
                    full_heading = f"{heading} — {sub_heading}".strip(" —") if sub_heading else heading
                    for chunk_text in split_body(sub_body):
                        flat.append((full_heading, chunk_text))
            else:
                for chunk_text in split_body(body):
                    flat.append((heading, chunk_text))

    # Filter and build output
    filtered = [(h, t) for h, t in flat if count_tokens(t) >= MIN_CHUNK_TOKENS]
    if not filtered:
        return []

    total  = len(filtered)
    result = []
    for i, (heading, chunk_text) in enumerate(filtered):
        directive_num = rec_id.replace("idoc-dir-", "").replace("_", ".")
        display = f"IDOC Administrative Directive {directive_num}: {title}"
        result.append(Chunk(
            chunk_id        = f"{rec_id}_c{i}",
            parent_id       = rec_id,
            chunk_index     = i,
            chunk_total     = total,
            text            = chunk_text,
            enriched_text   = _make_enriched(rec, heading, chunk_text),
            source          = source,
            token_count     = count_tokens(chunk_text),
            display_citation= display,
            metadata        = {
                "record_id":       rec_id,
                "doc_type":        doc_type,
                "title":           title,
                "section_heading": heading,
                "category":        rec.get("category", ""),
                "sub_category":    rec.get("sub_category", ""),
                "url":             rec.get("url", ""),
            },
        ))
    return result


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
        Bucket=bucket, Key=key,
        Body=payload.encode("utf-8"),
        ContentType="application/x-ndjson",
    )
    log.info("Upload complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def deduplicate_records(records: list[dict]) -> list[dict]:
    """Keep the longest-text copy when the raw corpus has duplicate directive IDs."""
    seen: dict[str, dict] = {}
    for rec in records:
        rid = rec["id"]
        if rid not in seen or len(rec.get("text", "")) > len(seen[rid].get("text", "")):
            seen[rid] = rec
    return list(seen.values())


def run(local_only: bool = False, limit: int = 0) -> None:
    log.info("=== IDOC chunking pipeline starting ===")
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
        "Produced %d chunks from %d records (%d skipped)",
        len(all_chunks), len(records), skipped,
    )
    single = sum(1 for c in all_chunks if c.chunk_total == 1)
    multi  = len({c.parent_id for c in all_chunks if c.chunk_total > 1})
    log.info("Single-chunk records: %d  |  Multi-chunk records: %d", single, multi)

    serialized = [dataclasses.asdict(c) for c in all_chunks]
    if local_only:
        write_local(serialized, LOCAL_OUTPUT_DIR / "idoc_chunks.jsonl")
    else:
        write_s3(serialized, cfg["chunked_bucket"], cfg["chunked_key"], cfg["aws_region"])

    log.info("=== IDOC chunking pipeline complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk IDOC directives JSONL → chunks JSONL")
    parser.add_argument("--local-only", action="store_true",
                        help="Write output locally instead of uploading to S3.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N records (0 = all). For testing.")
    args = parser.parse_args()
    run(local_only=args.local_only, limit=args.limit)


if __name__ == "__main__":
    main()
