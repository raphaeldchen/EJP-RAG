import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

import boto3
from dotenv import load_dotenv

from chunk.opinion_utils import (
    MIN_CHUNK_TOKENS,
    count_tokens,
    detect_sections,
    is_noise_chunk,
    split_section,
    _opinion_enriched_text,
)
from core.models import Chunk

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

LOCAL_CORPUS_PATH = Path("data_files/corpus/cap_bulk_corpus.jsonl")
LOCAL_OUTPUT_DIR  = Path("data_files/chunked_output")

CAP_COURT_LABELS = {
    "Ill.":          "Illinois Supreme Court",
    "Ill. App. Ct.": "Illinois Appellate Court",
}

_OPINION_MARKER_RE = re.compile(
    r"^\[(Majority|Dissent|Concurrence(?:-In-Part)?|Concur|Rehearing|Per Curiam|"
    r"Opinion|Unanimous|Plurality|Combined|Addendum|Remittitur)\]",
    re.MULTILINE | re.IGNORECASE,
)

_OPINION_TYPE_NORM: dict[str, str] = {
    "majority":            "majority",
    "dissent":             "dissent",
    "concurrence":         "concurrence",
    "concur":              "concurrence",
    "concurrence-in-part": "concurrence",
    "rehearing":           "rehearing",
    "per curiam":          "majority",
    "opinion":             "majority",
    "unanimous":           "majority",
    "plurality":           "majority",
    "combined":            "majority",
    "addendum":            "addendum",
    "remittitur":          "remittitur",
}

_MAJORITY_TYPES = {"majority", "unanimous", "plurality", "combined"}


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        log.error(f"Required environment variable {key!r} is not set.")
        sys.exit(1)
    return val


def _get_config() -> dict:
    raw_bucket     = _require_env("RAW_S3_BUCKET")
    chunked_bucket = _require_env("CHUNKED_S3_BUCKET")
    cap_prefix     = os.environ.get("CAP_S3_PREFIX", "cap").rstrip("/")
    return {
        "raw_bucket":     raw_bucket,
        "raw_key":        f"{cap_prefix}/cap_bulk_corpus.jsonl",
        "chunked_bucket": chunked_bucket,
        "chunked_key":    f"{cap_prefix}/cap_opinion_chunks.jsonl",
    }


def _split_opinion_segments(text: str) -> list[tuple[str, str]]:
    """Split text at [Marker] boundaries into (opinion_type, segment_text) pairs."""
    matches = list(_OPINION_MARKER_RE.finditer(text))
    if not matches:
        return [("majority", text.strip())]
    segments = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            segments.append(("majority", preamble))
    for i, m in enumerate(matches):
        label     = m.group(1).lower()
        norm_type = _OPINION_TYPE_NORM.get(label, label)
        start     = m.end()
        end       = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body      = text[start:end].strip()
        if body:
            segments.append((norm_type, body))
    return segments or [("majority", text.strip())]


def _build_display_citation(entry: dict) -> str:
    citations = entry.get("citations") or []
    abbr      = entry.get("case_name_abbr") or entry.get("case_name") or ""
    date      = entry.get("date_decided") or ""
    year      = date[:4] if len(date) >= 4 else ""
    name_part = f"{abbr} ({year})" if abbr and year else abbr
    if citations:
        return f"{citations[0]} — {name_part}" if name_part else citations[0]
    return name_part


def chunk_entry(entry: dict) -> list[Chunk]:
    """Chunk a single CAP corpus entry into Chunk objects."""
    text = (entry.get("text") or "").strip()
    if not text:
        return []

    entry_id         = entry.get("id", "")
    case_id          = entry.get("case_id", "")
    case_name        = entry.get("case_name", "")
    case_name_abbr   = entry.get("case_name_abbr", "")
    date_decided     = entry.get("date_decided", "")
    court            = entry.get("court", "")
    court_label      = CAP_COURT_LABELS.get(court, court)
    citations        = entry.get("citations") or []
    display_citation = _build_display_citation(entry)

    segments = _split_opinion_segments(text)
    result: list[Chunk] = []

    for type_idx, (opinion_type, segment_text) in enumerate(segments):
        sections = detect_sections(segment_text)
        flat: list[tuple[str, str, int]] = []
        for sec_idx, (heading, body) in enumerate(sections):
            for h, t in split_section(heading, body):
                flat.append((h, t, sec_idx))

        prelim: list[tuple[str, str, int, int]] = []
        for heading, chunk_text, sec_idx in flat:
            token_count = count_tokens(chunk_text)
            if token_count < MIN_CHUNK_TOKENS:
                continue
            if is_noise_chunk(chunk_text, token_count):
                continue
            prelim.append((heading, chunk_text, sec_idx, token_count))

        chunk_total = len(prelim)
        for chunk_index, (heading, chunk_text, sec_idx, token_count) in enumerate(prelim):
            enriched = _opinion_enriched_text(
                case_name_abbr or case_name,
                date_decided,
                court_label,
                heading,
                chunk_text,
            )
            result.append(Chunk(
                chunk_id         = f"{entry_id}_{opinion_type}_c{chunk_index}",
                parent_id        = entry_id,
                chunk_index      = chunk_index,
                chunk_total      = chunk_total,
                text             = chunk_text,
                enriched_text    = enriched,
                source           = "cap_bulk",
                token_count      = token_count,
                display_citation = display_citation,
                metadata={
                    "chunk_type":      "opinion_section" if heading else "opinion_paragraph",
                    "section_heading": heading,
                    "section_index":   sec_idx,
                    "opinion_type":    opinion_type,
                    "is_majority":     opinion_type in _MAJORITY_TYPES,
                    "case_id":         case_id,
                    "case_name":       case_name,
                    "case_name_abbr":  case_name_abbr,
                    "date_decided":    date_decided,
                    "court":           court,
                    "court_label":     court_label,
                    "citations":       citations,
                },
            ))

    return result


def _read_jsonl_local(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_jsonl_s3(bucket: str, key: str) -> list[dict]:
    log.info(f"  Reading s3://{bucket}/{key}")
    obj   = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    lines = obj["Body"].read().decode("utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def _write_jsonl_local(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"  Saved locally: {path}  ({len(records):,} records)")


def _write_jsonl_s3(records: list[dict], bucket: str, key: str) -> None:
    log.info(f"  Writing {len(records):,} records → s3://{bucket}/{key}")
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/x-ndjson",
    )
    log.info("  Upload complete.")


def run(local_only: bool = False, limit: int = 0) -> None:
    if local_only:
        log.info(f"Reading corpus from {LOCAL_CORPUS_PATH} ...")
        entries = _read_jsonl_local(LOCAL_CORPUS_PATH)
    else:
        cfg     = _get_config()
        entries = _read_jsonl_s3(cfg["raw_bucket"], cfg["raw_key"])

    log.info(f"  {len(entries):,} entries loaded.")

    if limit:
        log.info(f"  Limiting to first {limit} entries by text length.")
        entries = sorted(entries, key=lambda e: len(e.get("text") or ""), reverse=True)[:limit]

    all_chunks: list[Chunk] = []
    skipped_no_text = 0

    for i, entry in enumerate(entries):
        chunks = chunk_entry(entry)
        if not chunks:
            skipped_no_text += 1
        else:
            all_chunks.extend(chunks)
        if (i + 1) % 5_000 == 0:
            log.info(f"  {i + 1:,} opinions processed → {len(all_chunks):,} chunks...")

    log.info(
        f"  Done. {len(all_chunks):,} chunks from "
        f"{len(entries) - skipped_no_text:,} opinions. "
        f"Skipped: {skipped_no_text} (no text)."
    )
    if all_chunks:
        tokens = [c.token_count for c in all_chunks]
        log.info(
            f"  Token stats: avg {sum(tokens) / len(tokens):.0f} | "
            f"min {min(tokens)} | max {max(tokens)}"
        )

    records = [asdict(c) for c in all_chunks]

    if local_only:
        _write_jsonl_local(records, LOCAL_OUTPUT_DIR / "cap_opinion_chunks.jsonl")
    else:
        cfg = _get_config()
        _write_jsonl_s3(records, cfg["chunked_bucket"], cfg["chunked_key"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk CAP bulk corpus JSONL → JSONL (local or S3)"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Read from data_files/corpus/ and write to data_files/chunked_output/ without S3.",
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
