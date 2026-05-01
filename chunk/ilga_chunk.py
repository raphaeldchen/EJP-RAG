import argparse
import dataclasses
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Generator

import boto3
import tiktoken
from core.models import Chunk
from dotenv import load_dotenv

load_dotenv()

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        log.error("Required environment variable %r is not set.", key)
        sys.exit(1)
    return val

def get_config() -> dict:
    ilcs_prefix         = os.getenv("ILCS_S3_PREFIX", "ilcs/").rstrip("/")
    ilcs_chunked_prefix = os.getenv("ILCS_CHUNKED_S3_PREFIX", "ilcs/").rstrip("/")
    return {
        "raw_bucket":      _require_env("RAW_S3_BUCKET"),
        "raw_key":         os.getenv("ILCS_RAW_OBJECT_KEY", f"{ilcs_prefix}/ilcs_corpus.jsonl"),
        "chunked_bucket":  _require_env("CHUNKED_S3_BUCKET"),
        "chunked_key":     os.getenv("ILCS_CHUNKED_OBJECT_KEY", f"{ilcs_chunked_prefix}/ilcs_chunks.jsonl"),
        "aws_region":      os.getenv("AWS_REGION"),
    }

LOCAL_OUTPUT_DIR = Path("./data_files/chunked_output")

CHUNK_SIZE: int     = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP: int  = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_CHUNK_SIZE: int = int(os.getenv("MIN_CHUNK_SIZE", "100"))
MIN_MERGE_TOKENS: int = int(os.getenv("MIN_MERGE_TOKENS", "30"))

_WHITESPACE_RE = re.compile(r"[ \t]+")
_CRLF_RE       = re.compile(r"\r\n?")

def clean_text(text: str) -> str:
    text = _CRLF_RE.sub("\n", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_heading(heading: str) -> str:
    return re.sub(r"\s+", " ", heading).strip()

_CANDIDATE_SUBSECTION_RE = re.compile(
    r"(?:^|\n)(\([a-z]\))",  # single lowercase letter only; numeric/compound sub-items stay with parent
)

_ORPHAN_SUBSECTION_RE = re.compile(r"^\s*\([a-z0-9]\)")

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

def _is_real_subsection_boundary(text: str, marker_start: int) -> bool:
    if marker_start == 0:
        return True
    preceding = text[:marker_start]
    last_line = preceding.rstrip().split("\n")[-1].strip()
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
    interior_boundaries = [b for b in boundaries if b > 0]
    if not interior_boundaries:
        return [text]
    segments: list[str] = []
    prev = 0
    for b in interior_boundaries:
        seg = text[prev:b].strip()
        if seg:
            segments.append(seg)
        prev = b
    tail = text[prev:].strip()
    if tail:
        segments.append(tail)
    return segments or [text]


def _split_hard(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split at whitespace when no sentence boundaries are available; carries overlap."""
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
        # Single sentence exceeds chunk_size: hard-split it inline with overlap
        if s_len > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            hard_parts = _split_hard(sentence, chunk_size, overlap)
            if hard_parts:
                chunks.extend(hard_parts[:-1])
                current = [hard_parts[-1]]
                current_len = len(hard_parts[-1])
            continue
        if current_len + s_len > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_buf: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap and overlap_buf:  # always carry at least one sentence
                    break
                overlap_buf.insert(0, s)
                overlap_len += len(s) + 1
            # Cap overlap so the next sentence always fits within chunk_size
            max_carry = max(0, chunk_size - s_len - 1)
            if overlap_len > max_carry:
                carried = " ".join(overlap_buf)
                overlap_buf = [carried[-max_carry:]] if max_carry else []
                overlap_len = max_carry
            current = overlap_buf
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
    """Greedy bin-pack subsection segments to minimize splits and orphaned starts."""
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
            current_len = seg_len
    if current_parts:
        packed.append("\n".join(current_parts))
    return packed


def chunk_section(record: dict) -> list[Chunk]:
    text = clean_text(record.get("text", ""))
    if not text:
        log.warning("Empty text for record %s — skipping", record.get("id"))
        return []
    raw_heading = record.get("section_heading", "")
    cleaned_heading = clean_heading(raw_heading)
    if raw_heading != cleaned_heading:
        log.debug(
            "Normalised heading for %s: %r → %r",
            record.get("id"), raw_heading, cleaned_heading,
        )
    metadata = {
        "section_id":       record["id"],
        "section_citation": record.get("section_citation", ""),
        "section_num":      record.get("section_num", ""),
        "section_heading":  cleaned_heading,
        "article_name":     record.get("article_name", ""),
        "act_name":         record.get("act_name", ""),
        "act_id":           record.get("act_id", ""),
        "chapter_num":      record.get("chapter_num", ""),
        "chapter_name":     record.get("chapter_name", ""),
        "major_topic":      record.get("major_topic", ""),
        "url":              record.get("url", ""),
        "scraped_at":       record.get("scraped_at", ""),
        "source":           "ilcs",
    }
    if len(text) <= CHUNK_SIZE:
        if len(text) < MIN_CHUNK_SIZE:
            return []
        return [_make_chunk(text, metadata,
                            parent_id=record["id"], chunk_index=0, chunk_total=1)]
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
    # Prefix orphaned-start segments with context so chunks at index > 0 don't start bare.
    # Tiered: full "citation heading\n" if it fits, else short "[citation]\n", else trim to fit.
    section_header = f"{metadata['section_citation']} {metadata['section_heading']}".strip()
    citation = metadata.get("section_citation", "")
    short_prefix = f"[{citation}]\n" if citation else ""
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
        _make_chunk(seg, metadata,
                    parent_id=record["id"], chunk_index=i, chunk_total=total)
        for i, seg in enumerate(final_segs)
    ]

def _make_chunk(
    text: str,
    metadata: dict,
    *,
    parent_id: str,
    chunk_index: int,
    chunk_total: int,
) -> Chunk:
    section_citation = metadata.get("section_citation", "")
    section_heading  = metadata.get("section_heading", "")
    act_name         = metadata.get("act_name", "")
    major_topic      = metadata.get("major_topic", "")

    display_citation = section_citation
    if section_heading:
        display_citation = f"{section_citation} — {section_heading}"

    enriched_parts = [p for p in [section_citation, section_heading, act_name, major_topic] if p]
    enriched = "\n".join(enriched_parts) + "\n\n" + text if enriched_parts else text

    return Chunk(
        chunk_id=f"{parent_id}_c{chunk_index}",
        parent_id=parent_id,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        text=text,
        enriched_text=enriched,
        source=metadata["source"],
        token_count=count_tokens(text),
        display_citation=display_citation,
        metadata=metadata,
    )

def iter_records(source: str) -> Generator[dict, None, None]:
    for line_no, line in enumerate(source.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("Skipping malformed line %d: %s", line_no, exc)

def read_raw_corpus_s3(bucket: str, key: str, region: str | None) -> str:
    log.info("Reading raw corpus from s3://%s/%s", bucket, key)
    kwargs = {"region_name": region} if region else {}
    obj = boto3.client("s3", **kwargs).get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")

def write_chunks_local(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info("Saved locally: %s  (%d chunks)", path, len(chunks))

def write_chunks_s3(chunks: list[dict], bucket: str, key: str, region: str | None) -> None:
    log.info("Writing %d chunks to s3://%s/%s", len(chunks), bucket, key)
    payload = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n"
    kwargs = {"region_name": region} if region else {}
    boto3.client("s3", **kwargs).put_object(
        Bucket=bucket,
        Key=key,
        Body=payload.encode("utf-8"),
        ContentType="application/x-ndjson",
    )
    log.info("Upload complete.")

def run(local_only: bool = False, local_input: Path | None = None, limit: int = 0) -> None:
    log.info("=== ILCS chunking pipeline starting ===")
    log.info(
        "Config: CHUNK_SIZE=%d  CHUNK_OVERLAP=%d  MIN_CHUNK_SIZE=%d  MIN_MERGE_TOKENS=%d",
        CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE, MIN_MERGE_TOKENS,
    )
    if local_input:
        log.info("Reading raw corpus from local file: %s", local_input)
        raw = local_input.read_text(encoding="utf-8")
    else:
        cfg = get_config()
        raw = read_raw_corpus_s3(cfg["raw_bucket"], cfg["raw_key"], cfg["aws_region"])
    records = list(iter_records(raw))
    log.info("Loaded %d raw sections", len(records))
    if limit:
        log.info("Limiting to first %d sections.", limit)
        records = records[:limit]
    all_chunks: list[Chunk] = []
    skipped = 0
    for rec in records:
        chunks = chunk_section(rec)
        if not chunks:
            skipped += 1
            continue
        all_chunks.extend(chunks)
    log.info(
        "Produced %d chunks from %d sections (%d skipped)",
        len(all_chunks), len(records), skipped,
    )
    single_chunk_sections = sum(1 for c in all_chunks if c.chunk_total == 1)
    multi_chunk_parents   = len({c.parent_id for c in all_chunks if c.chunk_total > 1})
    log.info(
        "Single-chunk sections: %d  |  Multi-chunk sections: %d",
        single_chunk_sections, multi_chunk_parents,
    )
    serialized = [dataclasses.asdict(c) for c in all_chunks]
    if local_only or local_input:
        write_chunks_local(serialized, LOCAL_OUTPUT_DIR / "ilcs_chunks.jsonl")
        log.info("Output in: %s", LOCAL_OUTPUT_DIR.resolve())
    else:
        cfg = get_config()
        write_chunks_s3(serialized, cfg["chunked_bucket"], cfg["chunked_key"], cfg["aws_region"])
        log.info("Output at s3://%s/%s", cfg["chunked_bucket"], cfg["chunked_key"])
    log.info("=== ILCS chunking pipeline complete ===")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk Illinois Compiled Statutes JSONL → chunks JSONL (S3 or local)"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Write output locally to ./chunked_output/ilcs_chunks.jsonl instead of S3.",
    )
    parser.add_argument(
        "--local-input",
        type=Path,
        default=None,
        metavar="FILE",
        help="Read raw corpus from a local JSONL file instead of S3. Implies --local-only for output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N sections (0 = all). For testing.",
    )
    args = parser.parse_args()
    run(local_only=args.local_only, local_input=args.local_input, limit=args.limit)

if __name__ == "__main__":
    main()