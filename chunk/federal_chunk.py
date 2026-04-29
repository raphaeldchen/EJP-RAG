"""
Federal documents chunker.

Reads federal/federal_corpus.jsonl from the raw S3 bucket and writes
federal/federal_chunks.jsonl to the chunked S3 bucket. Defaults to S3 I/O;
use --local-only for development.

Corpus contains 3 heterogeneous records:
  - second-chance-pell-rule: Federal Register regulation (HTML-extracted)
  - first-step-act:          Congress.gov statute summary page (short)
  - bop-5300.21:             BOP Program Statement policy (numbered sections)

Chunking strategy:
  1. Strip web-navigation boilerplate (Federal Register / Congress.gov artifacts)
  2. Strip BOP page-repeat headers ("PS XXXX.XX / DATE / Page N")
  3. Split at numbered section boundaries ("N.  HEADING") for policy/regulation docs
  4. Token-aware paragraph accumulation within sections
     (target 600 tokens, max 800, last-paragraph overlap)
  5. Merge micro-chunks (< MIN_CHUNK_TOKENS) into neighbors

Usage:
    python3 chunk/federal_chunk.py            # S3 -> S3
    python3 chunk/federal_chunk.py --local-only
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
    return {
        "raw_bucket":     _require_env("RAW_S3_BUCKET"),
        "raw_key":        "federal/federal_corpus.jsonl",
        "chunked_bucket": _require_env("CHUNKED_S3_BUCKET"),
        "chunked_key":    "federal/federal_chunks.jsonl",
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

# BOP Program Statement page-repeat header: "PS XXXX.XX\nDATE\nPage N"
_BOP_PAGE_HEADER_RE = re.compile(
    r"PS \d+\.\d+\s*\n"
    r"\d+/\d+/\d+\s*\n"
    r"Page \d+\s*\n",
)

# Federal Register / Congress.gov HTML navigation boilerplate that precedes
# actual regulatory text.  These blocks appear before the first real heading.
# Strategy: find the first substantive heading and discard everything before it.
_FR_BOILERPLATE_STOP_RE = re.compile(
    r"(?:^AGENCY:\s|^ACTION:\s|^SUMMARY:\s|^DATES:\s|^FOR FURTHER INFORMATION|"
    r"^SUPPLEMENTARY INFORMATION)",
    re.MULTILINE,
)

# Congress.gov summary pages have a brief "LawHide Overview" block of metadata.
_CONGRESS_OVERVIEW_RE = re.compile(
    r"LawHide Overview.*?(?=\n[A-Z]|\Z)",
    re.DOTALL,
)

# BOP-style numbered sections: "N.  HEADING TEXT"
# Require at least 2 trailing spaces after the period (BOP formatting convention).
_NUM_SECTION_RE = re.compile(
    r"^(\d+)\.\s{2,}([A-Z][A-Z0-9\s/\-\[\]§\.]{1,80}?)\s*$",
    re.MULTILINE,
)


def strip_bop_page_headers(text: str) -> str:
    return _BOP_PAGE_HEADER_RE.sub("", text).strip()


def strip_fr_boilerplate(text: str) -> str:
    """Trim Federal Register HTML navigation cruft that precedes AGENCY: block."""
    m = _FR_BOILERPLATE_STOP_RE.search(text)
    if m:
        return text[m.start():].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

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
# Token-aware accumulation
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

def _merge_micro_chunks(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
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
class FederalChunk:
    chunk_id:        str
    chunk_index:     int
    chunk_total:     int
    source:          str
    doc_type:        str
    text:            str
    enriched_text:   str
    token_count:     int
    section_heading: str
    record_id:       str
    title:           str
    citation:        str
    url:             str
    chunked_at:      str


def _make_enriched(rec: dict, section_heading: str, text: str) -> str:
    parts = [f"Federal: {rec.get('title', '')}"]
    if rec.get("citation"):
        parts[0] += f" ({rec['citation']})"
    if section_heading:
        parts.append(section_heading)
    return "\n\n".join(parts) + f"\n\n{text}"


# ---------------------------------------------------------------------------
# Per-record chunking
# ---------------------------------------------------------------------------

def _clean_text(rec: dict) -> str:
    """Apply source-appropriate cleaning to the raw text."""
    text = rec.get("text", "").strip()
    doc_type = rec.get("doc_type", "")
    rec_id   = rec.get("id", "")

    if doc_type == "policy" or "bop" in rec_id:
        text = strip_bop_page_headers(text)
    if doc_type == "regulation" or "federal" in rec_id.lower() or "pell" in rec_id.lower():
        text = strip_fr_boilerplate(text)
    return text


def chunk_record(rec: dict) -> list[FederalChunk]:
    raw_text = rec.get("text", "").strip()
    if not raw_text:
        return []

    rec_id   = rec.get("id", "")
    title    = rec.get("title", "")
    doc_type = rec.get("doc_type", "")
    citation = rec.get("citation", "")
    url      = rec.get("url", "")

    text = _clean_text(rec)

    if count_tokens(text) < MIN_CHUNK_TOKENS:
        return []

    # Split at numbered sections for policy docs; fall back to plain paragraphs.
    sections = _split_at_pattern(text, _NUM_SECTION_RE)
    if len(sections) == 1:
        # No numbered section boundaries — use paragraph-based splitting directly
        sections = [("", text)]

    flat: list[tuple[str, str]] = []
    for heading, body in sections:
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
        result.append(FederalChunk(
            chunk_id        = f"{rec_id}_c{i}",
            chunk_index     = i,
            chunk_total     = total,
            source          = "federal",
            doc_type        = doc_type,
            text            = chunk_text,
            enriched_text   = _make_enriched(rec, heading, chunk_text),
            token_count     = count_tokens(chunk_text),
            section_heading = heading,
            record_id       = rec_id,
            title           = title,
            citation        = citation,
            url             = url,
            chunked_at      = datetime.now(timezone.utc).isoformat(),
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
    log.info("=== Federal chunking pipeline starting ===")
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
        write_local(all_chunks, LOCAL_OUTPUT_DIR / "federal_chunks.jsonl")
    else:
        write_s3(all_chunks, cfg["chunked_bucket"], cfg["chunked_key"], cfg["aws_region"])

    log.info("=== Federal chunking pipeline complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk federal documents JSONL → chunks JSONL")
    parser.add_argument("--local-only", action="store_true",
                        help="Write output locally instead of uploading to S3.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N records (0 = all). For testing.")
    args = parser.parse_args()
    run(local_only=args.local_only, limit=args.limit)


if __name__ == "__main__":
    main()
