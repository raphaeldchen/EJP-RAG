"""
Illinois Administrative Code (IAC) chunker.

Reads iac/iac_corpus.jsonl from the raw S3 bucket and writes
iac/iac_chunks.jsonl to the chunked S3 bucket. Defaults to S3 I/O;
use --local-only for development.

Key differences from ilga_chunk.py:
  - IAC uses a) b) c) subsection notation, not (a) (b) (c)
  - JCAR scraper produces HTML line-wrap artifacts that must be collapsed
    before splitting (e.g. "Adjustment\\nCommittee Hearing Procedures")
  - Metadata carries IAC hierarchy: title → part → section

Usage:
    python3 chunk/iac_chunk.py                   # S3 → S3
    python3 chunk/iac_chunk.py --local-only       # S3 in, local out
    python3 chunk/iac_chunk.py --limit 50 --local-only
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import boto3
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
        "raw_bucket":    _require_env("RAW_S3_BUCKET"),
        "raw_key":       "iac/iac_corpus.jsonl",
        "chunked_bucket": _require_env("CHUNKED_S3_BUCKET"),
        "chunked_key":   "iac/iac_chunks.jsonl",
        "aws_region":    os.getenv("AWS_REGION"),
    }

LOCAL_OUTPUT_DIR = Path("./data_files/chunked_output")

CHUNK_SIZE:      int = int(os.getenv("CHUNK_SIZE",      "1500"))
CHUNK_OVERLAP:   int = int(os.getenv("CHUNK_OVERLAP",    "200"))
MIN_CHUNK_SIZE:  int = int(os.getenv("MIN_CHUNK_SIZE",   "100"))
MIN_MERGE_TOKENS: int = int(os.getenv("MIN_MERGE_TOKENS", "30"))


# ---------------------------------------------------------------------------
# Text normalisation
# IAC text from the JCAR scraper has HTML line-wrap artifacts:
#   "Section 504.80  Adjustment\nCommittee Hearing Procedures"
#   "a)         The\nFacility Publication Review Officer..."
# These must be collapsed to single lines before any splitting logic.
# ---------------------------------------------------------------------------

# Lines that start with these patterns begin a new structural unit.
_STRUCTURAL_START_RE = re.compile(
    r"^(?:"
    r"[a-z]\)"        # letter subsection: a) b) c)
    r"|[A-Z]\)"       # lettered sub-item: A) B) C)
    r"|\d+\)"         # numbered item:     1) 2) 3)
    r"|SUBPART\b"     # subpart header
    r"|Section\s+\d"  # section heading
    r")",
    re.IGNORECASE,
)

_CRLF_RE      = re.compile(r"\r\n?")
_EXCESS_WS_RE = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    """Collapse HTML line-wrap artifacts; preserve structural line breaks."""
    text = _CRLF_RE.sub("\n", text)
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
        elif _STRUCTURAL_START_RE.match(stripped):
            # Normalise excessive internal whitespace (e.g. "a)         The")
            out.append(_EXCESS_WS_RE.sub(" ", stripped))
        elif out and out[-1] != "":
            # Continuation wrap — join onto the previous line
            out[-1] = out[-1] + " " + stripped
        else:
            out.append(stripped)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_heading(heading: str) -> str:
    return re.sub(r"\s+", " ", heading).strip()


# ---------------------------------------------------------------------------
# Subsection splitting
# IAC uses a) b) c) (no opening paren), numbered sub-items 1) 2) 3),
# and lettered sub-sub-items A) B) C).
# Split only at letter-subsection boundaries; numeric/letter sub-items
# stay bound to their parent letter subsection.
# ---------------------------------------------------------------------------

_CANDIDATE_SUBSECTION_RE = re.compile(
    r"(?:^|\n)([a-z]\))",  # single lowercase letter + closing paren only
)

_ORPHAN_SUBSECTION_RE = re.compile(r"^\s*[a-z0-9]\)")

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _is_real_subsection_boundary(text: str, marker_start: int) -> bool:
    """Reject false positives like 'e.g.) ...' or mid-sentence 'a)'."""
    if marker_start == 0:
        return True
    preceding  = text[:marker_start]
    last_line  = preceding.rstrip().split("\n")[-1].strip()
    if not last_line:
        return True
    if last_line[-1] in ".!?:;":
        return True
    if last_line.lower().endswith((" or", " and", " or:", " and:")):
        return True
    return False


def split_on_subsections(text: str) -> list[str]:
    boundaries: list[int] = []
    for m in _CANDIDATE_SUBSECTION_RE.finditer(text):
        start = m.start()
        if start < len(text) and text[start] == "\n":
            start += 1
        if _is_real_subsection_boundary(text, start):
            boundaries.append(start)
    interior = [b for b in boundaries if b > 0]
    if not interior:
        return [text]
    segments: list[str] = []
    prev = 0
    for b in interior:
        seg = text[prev:b].strip()
        if seg:
            segments.append(seg)
        prev = b
    tail = text[prev:].strip()
    if tail:
        segments.append(tail)
    return segments or [text]


# ---------------------------------------------------------------------------
# Splitting helpers  (identical to ilga_chunk.py)
# ---------------------------------------------------------------------------

def _split_hard(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    parts: list[str] = []
    pos = 0
    while pos < len(text):
        end = pos + chunk_size
        if end >= len(text):
            parts.append(text[pos:].strip())
            break
        split_at = text.rfind(" ", pos, end)
        if split_at <= pos:
            split_at = end
        parts.append(text[pos:split_at].strip())
        next_pos = split_at - overlap if overlap else split_at
        pos = max(next_pos, pos + 1)
    return [p for p in parts if p.strip()]


def split_by_sentences(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = _SENTENCE_END_RE.split(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        s_len = len(sentence)
        if s_len > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            hard_parts = _split_hard(sentence, chunk_size, overlap)
            if hard_parts:
                chunks.extend(hard_parts[:-1])
                current     = [hard_parts[-1]]
                current_len = len(hard_parts[-1])
            continue
        if current_len + s_len > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_buf: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap and overlap_buf:
                    break
                overlap_buf.insert(0, s)
                overlap_len += len(s) + 1
            max_carry = max(0, chunk_size - s_len - 1)
            if overlap_len > max_carry:
                carried = " ".join(overlap_buf)
                overlap_buf = [carried[-max_carry:]] if max_carry else []
                overlap_len = max_carry
            current     = overlap_buf
            current_len = overlap_len
        current.append(sentence)
        current_len += s_len + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def merge_small_chunks(segments: list[str]) -> list[str]:
    def _one_pass(segs: list[str]) -> list[str]:
        if len(segs) <= 1:
            return segs
        merged = [segs[0]]
        for seg in segs[1:]:
            prev     = merged[-1]
            combined = prev + "\n" + seg
            if (
                (len(prev.split()) <= MIN_MERGE_TOKENS or len(seg.split()) <= MIN_MERGE_TOKENS)
                and len(combined) <= CHUNK_SIZE
            ):
                merged[-1] = combined
            else:
                merged.append(seg)
        return merged
    return _one_pass(_one_pass(segments))


def _pack_subsections(segments: list[str]) -> list[str]:
    """Greedy bin-pack subsection segments to minimise splits and orphaned starts."""
    if len(segments) <= 1:
        return segments
    packed: list[str] = []
    current_parts: list[str] = [segments[0]]
    current_len = len(segments[0])
    for seg in segments[1:]:
        seg_len = len(seg)
        if current_len + 1 + seg_len <= CHUNK_SIZE:
            current_parts.append(seg)
            current_len += 1 + seg_len
        else:
            packed.append("\n".join(current_parts))
            current_parts = [seg]
            current_len   = seg_len
    if current_parts:
        packed.append("\n".join(current_parts))
    return packed


# ---------------------------------------------------------------------------
# Chunk production
# ---------------------------------------------------------------------------

def chunk_section(record: dict) -> list[dict]:
    raw_text = record.get("text", "")
    text     = normalize_text(raw_text)
    if not text:
        log.warning("Empty text for %s — skipping", record.get("id"))
        return []

    section_citation = record.get("section_citation", "")
    section_heading  = clean_heading(record.get("section_heading", ""))

    metadata = {
        "section_id":       record["id"],
        "source":           "illinois_admin_code",
        "title_num":        record.get("title_num", ""),
        "title_name":       record.get("title_name", ""),
        "part_num":         record.get("part_num", ""),
        "part_name":        record.get("part_name", ""),
        "section_num":      record.get("section_num", ""),
        "section_heading":  section_heading,
        "section_citation": section_citation,
        "url":              record.get("url", ""),
        "scraped_at":       record.get("scraped_at", ""),
    }

    if len(text) <= CHUNK_SIZE:
        if len(text) < MIN_CHUNK_SIZE:
            return []
        return [_make_chunk(text, metadata, parent_id=record["id"], chunk_index=0, chunk_total=1)]

    subsections = _pack_subsections(split_on_subsections(text))
    segments: list[str] = []
    for sub in subsections:
        if len(sub) <= CHUNK_SIZE:
            segments.append(sub)
        else:
            segments.extend(split_by_sentences(sub, CHUNK_SIZE, CHUNK_OVERLAP))

    segments = [s for s in segments if len(s) >= MIN_CHUNK_SIZE]
    segments = merge_small_chunks(segments)

    if not segments:
        log.warning("No valid segments for %s — skipping", record.get("id"))
        return []

    # Prefix orphaned-start segments with citation context
    section_header = f"{section_citation} {section_heading}".strip()
    short_prefix   = f"[{section_citation}]\n" if section_citation else ""
    final_segs: list[str] = []
    for i, seg in enumerate(segments):
        if i > 0 and _ORPHAN_SUBSECTION_RE.match(seg):
            full = f"{section_header}\n{seg}" if section_header else seg
            if len(full) <= CHUNK_SIZE:
                seg = full
            elif short_prefix:
                available = CHUNK_SIZE - len(short_prefix)
                seg = short_prefix + seg[:available]
        final_segs.append(seg)

    total = len(final_segs)
    return [
        _make_chunk(seg, metadata, parent_id=record["id"], chunk_index=i, chunk_total=total)
        for i, seg in enumerate(final_segs)
    ]


def _make_chunk(
    text: str,
    metadata: dict,
    *,
    parent_id: str,
    chunk_index: int,
    chunk_total: int,
) -> dict:
    # Enriched text prepends IAC hierarchy for embedding quality
    enriched = (
        f"Illinois Administrative Code Title {metadata['title_num']} "
        f"({metadata['title_name']}), "
        f"Part {metadata['part_num']} ({metadata['part_name']}), "
        f"{metadata['section_citation']} {metadata['section_heading']}\n\n"
        f"{text}"
    )
    return {
        "chunk_id":       f"{parent_id}_c{chunk_index}",
        "parent_id":      parent_id,
        "chunk_index":    chunk_index,
        "chunk_total":    chunk_total,
        "text":           text,
        "enriched_text":  enriched,
        "token_estimate": len(text.split()),
        "chunked_at":     datetime.now(timezone.utc).isoformat(),
        "metadata":       metadata,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def iter_records(source: str) -> Generator[dict, None, None]:
    for line_no, line in enumerate(source.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("Skipping malformed line %d: %s", line_no, exc)


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

def deduplicate_records(records: list[dict]) -> list[dict]:
    """Keep the longest-text copy when the raw corpus has stub + full-content duplicates."""
    seen: dict[str, dict] = {}
    for rec in records:
        rid = rec["id"]
        if rid not in seen or len(rec.get("text", "")) > len(seen[rid].get("text", "")):
            seen[rid] = rec
    return list(seen.values())


def run(local_only: bool = False, limit: int = 0) -> None:
    log.info("=== IAC chunking pipeline starting ===")
    cfg = get_config()
    raw = read_s3(cfg["raw_bucket"], cfg["raw_key"], cfg["aws_region"])
    records = list(iter_records(raw))
    log.info("Loaded %d raw sections", len(records))

    deduped = deduplicate_records(records)
    if len(deduped) < len(records):
        log.info("De-duplicated %d → %d records", len(records), len(deduped))
    records = deduped

    if limit:
        log.info("Limiting to first %d sections.", limit)
        records = records[:limit]

    all_chunks: list[dict] = []
    skipped = 0
    for rec in records:
        chunks = chunk_section(rec)
        if not chunks:
            skipped += 1
            continue
        all_chunks.extend(chunks)

    log.info(
        "Produced %d chunks from %d sections (%d skipped — stubs/empty)",
        len(all_chunks), len(records), skipped,
    )
    single  = sum(1 for c in all_chunks if c["chunk_total"] == 1)
    multi   = len({c["parent_id"] for c in all_chunks if c["chunk_total"] > 1})
    log.info("Single-chunk sections: %d  |  Multi-chunk sections: %d", single, multi)

    if local_only:
        write_local(all_chunks, LOCAL_OUTPUT_DIR / "iac_chunks.jsonl")
    else:
        write_s3(all_chunks, cfg["chunked_bucket"], cfg["chunked_key"], cfg["aws_region"])

    log.info("=== IAC chunking pipeline complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk Illinois Administrative Code JSONL → chunks JSONL"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Write output locally instead of uploading to S3.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N sections (0 = all). For testing.",
    )
    args = parser.parse_args()
    run(local_only=args.local_only, limit=args.limit)


if __name__ == "__main__":
    main()
