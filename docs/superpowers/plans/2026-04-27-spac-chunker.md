# SPAC Chunker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `chunk/spac_chunk.py` and `tests/test_spac_chunk.py` to chunk SPAC publications into coherent, single-topic nodes suitable for vector+BM25 retrieval and future GraphRAG.

**Architecture:** Hard splits at all-caps section headings (≥ 3 words) create single-topic chunks; token-aware paragraph accumulation handles large sections; a merge guard collapses micro-chunks into neighbors. Follows the IDOC chunker pattern (token-based, dataclass output) rather than the ILCS/IAC character-based pattern.

**Tech Stack:** Python 3.12, `tiktoken` (cl100k_base), `boto3`, `dataclasses`, `pytest`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `chunk/spac_chunk.py` | Full chunking pipeline |
| Create | `tests/test_spac_chunk.py` | Unit + corpus + S3 tests |
| Modify | `tests/conftest.py` | Add `spac_records`, `spac_chunks`, `spac_chunks_s3` fixtures |

---

## Task 1: Add conftest fixtures for SPAC

**Files:**
- Modify: `tests/conftest.py` (append after the `iac_chunks_s3` fixture)

- [ ] **Step 1: Append the three SPAC fixtures to `tests/conftest.py`**

Add the following at the end of `tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def spac_records():
    """Download spac_corpus.jsonl from S3 and return parsed records."""
    bucket = os.environ.get("RAW_S3_BUCKET")
    if not bucket:
        pytest.skip("RAW_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="spac/spac_corpus.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download spac_corpus.jsonl from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


@pytest.fixture(scope="session")
def spac_chunks(spac_records):
    """Run chunk_record() over all SPAC records; return flat list of all chunks as dicts."""
    from chunk.spac_chunk import chunk_record
    from dataclasses import asdict
    chunks = []
    for rec in spac_records:
        chunks.extend(asdict(c) for c in chunk_record(rec))
    return chunks


@pytest.fixture(scope="session")
def spac_chunks_s3():
    """Load the actual chunked JSONL from the chunked S3 bucket."""
    bucket = os.environ.get("CHUNKED_S3_BUCKET")
    if not bucket:
        pytest.skip("CHUNKED_S3_BUCKET not set")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="spac/spac_chunks.jsonl")
        raw = obj["Body"].read().decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not download spac_chunks.jsonl from S3: {e}")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]
```

- [ ] **Step 2: Verify conftest still collects cleanly (chunk/spac_chunk.py doesn't exist yet, so spac_chunks fixture will fail on import — that's fine at collection time)**

```bash
python3 -m pytest tests/conftest.py --collect-only 2>&1 | tail -5
```

Expected: collection errors only for `spac_chunks` (import of `chunk.spac_chunk` fails). All other fixtures collect fine.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add spac_records, spac_chunks, spac_chunks_s3 fixtures to conftest"
```

---

## Task 2: Scaffold `chunk/spac_chunk.py`

**Files:**
- Create: `chunk/spac_chunk.py`

Create the full module in one shot. All functions are implemented (not stubs) — TDD here means writing tests that exercise the real implementation.

- [ ] **Step 1: Create `chunk/spac_chunk.py`**

```python
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
# Condition: sequence of 4+ dots where dots occupy >30% of the line
_DOT_RUN_RE = re.compile(r"\.{4,}")


def strip_page_headers(text: str) -> str:
    return _PAGE_HEADER_RE.sub("", text).strip()


def strip_toc_lines(text: str) -> str:
    lines = text.split("\n")
    result = []
    for line in lines:
        if _DOT_RUN_RE.search(line):
            dot_count = line.count(".")
            if len(line) > 0 and dot_count / len(line) > 0.30:
                continue
        result.append(line)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

# All-caps line: starts with A-Z, remainder is A-Z / spaces / select punctuation
_HEADING_RE = re.compile(r"^[A-Z][A-Z\s\(\)\-/,]{3,59}$")


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
                # Merge forward: body prepended to next section's body
                combined = (body + "\n\n" + result[i + 1][1]).strip()
                result[i + 1][1] = combined
                result.pop(i)
                continue
            elif i > 0:
                # Last chunk: merge backward
                combined = (result[i - 1][1] + "\n\n" + body).strip()
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
```

- [ ] **Step 2: Verify module imports cleanly**

```bash
python3 -c "from chunk.spac_chunk import chunk_record, SpacChunk, is_section_heading; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit scaffold**

```bash
git add chunk/spac_chunk.py
git commit -m "feat: scaffold chunk/spac_chunk.py — SPAC publication chunker"
```

---

## Task 3: Unit tests for text cleaning

**Files:**
- Create: `tests/test_spac_chunk.py`

- [ ] **Step 1: Create `tests/test_spac_chunk.py` with cleaning tests**

```python
"""
Test suite for chunk/spac_chunk.py.

Unit tests  — synthetic records; no S3 access
Corpus tests — full spac_corpus.jsonl from S3 via spac_chunks fixture
S3 output   — reads actual spac_chunks.jsonl from the chunked S3 bucket
"""

import re
from collections import defaultdict

import pytest

from chunk.spac_chunk import (
    MAX_TOKENS,
    MIN_CHUNK_TOKENS,
    SpacChunk,
    chunk_record,
    is_section_heading,
    split_at_headings,
    strip_page_headers,
    strip_toc_lines,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    text: str,
    rec_id: str = "spac-test-001",
    category: str = "Report",
    year: str = "2020",
    title: str = "Test Report",
) -> dict:
    return {
        "id":        rec_id,
        "source":    "spac",
        "agency":    "Illinois Sentencing Policy Advisory Council",
        "category":  category,
        "title":     title,
        "year":      year,
        "filename":  "test_2020.pdf",
        "url":       "https://example.com/test.pdf",
        "text":      text,
        "scraped_at": "2026-04-27T00:00:00+00:00",
    }

_PAGE_LEAK_RE = re.compile(r"\bPage \d+ of \d+\b", re.IGNORECASE)
_DOT_LEAK_RE  = re.compile(r"\.{4,}")


# ---------------------------------------------------------------------------
# Unit tests — strip_page_headers
# ---------------------------------------------------------------------------

def test_page_header_stripped():
    text = f"Content A.\nMay 2017 Sentencing Reform Page 1 of 78\nContent B."
    result = strip_page_headers(text)
    assert "Page 1 of 78" not in result
    assert "Content A." in result
    assert "Content B." in result


def test_multiple_page_headers_stripped():
    text = "Start.\nPage 3 of 20\nMiddle.\nPage 4 of 20\nEnd."
    result = strip_page_headers(text)
    assert "Page 3 of 20" not in result
    assert "Page 4 of 20" not in result
    assert "Start." in result
    assert "Middle." in result
    assert "End." in result


def test_non_page_content_preserved():
    text = "No page numbers here.\nJust regular content."
    assert strip_page_headers(text).replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# Unit tests — strip_toc_lines
# ---------------------------------------------------------------------------

def test_toc_dot_leader_stripped():
    text = "Introduction ............................................................... 4\nReal content here."
    result = strip_toc_lines(text)
    assert not _DOT_LEAK_RE.search(result), "Dot-leader line was not stripped"
    assert "Real content here." in result


def test_toc_multiple_dot_leader_lines_stripped():
    toc = (
        "Introduction ...................................... 4\n"
        "Background ........................................ 6\n"
        "Findings .......................................... 12\n"
    )
    text = toc + "Actual content here."
    result = strip_toc_lines(text)
    assert not _DOT_LEAK_RE.search(result)
    assert "Actual content here." in result


def test_non_toc_lines_with_few_dots_preserved():
    text = "See e.g. Smith v. Jones, 123 F.3d 456 (7th Cir. 2000).\nMore content."
    result = strip_toc_lines(text)
    assert "Smith v. Jones" in result


# ---------------------------------------------------------------------------
# Unit tests — is_section_heading
# ---------------------------------------------------------------------------

def test_three_word_all_caps_is_heading():
    assert is_section_heading("DRUG OFFENSE REFORM") is True


def test_four_word_heading():
    assert is_section_heading("LIMITATIONS AND ASSUMPTIONS SECTION") is True


def test_long_heading():
    assert is_section_heading("INSUFFICIENT DATA TO SUPPORT A FULL FISCAL IMPACT ANALYSIS") is True


def test_two_word_not_heading():
    assert is_section_heading("RETAIL THEFT") is False


def test_one_word_not_heading():
    assert is_section_heading("COMPONENT") is False


def test_mixed_case_not_heading():
    assert is_section_heading("Drug Offense Reform") is False


def test_lowercase_not_heading():
    assert is_section_heading("drug offense reform") is False


def test_heading_with_parenthetical():
    assert is_section_heading("CANNABIS POLICY REFORM (HB 1234)") is True


def test_empty_string_not_heading():
    assert is_section_heading("") is False


def test_blank_whitespace_not_heading():
    assert is_section_heading("   ") is False


# ---------------------------------------------------------------------------
# Unit tests — split_at_headings
# ---------------------------------------------------------------------------

def test_split_produces_preamble_then_sections():
    text = "Introduction text.\n\nDRUG OFFENSE REFORM\n\nBody of section."
    sections = split_at_headings(text)
    headings = [h for h, _ in sections]
    assert "" in headings, "Preamble section missing"
    assert "DRUG OFFENSE REFORM" in headings


def test_split_no_headings_returns_single_section():
    text = "Just a paragraph of text with no headings at all."
    sections = split_at_headings(text)
    assert len(sections) == 1
    assert sections[0][0] == ""
    assert "paragraph of text" in sections[0][1]


def test_split_multiple_headings():
    text = (
        "DRUG OFFENSE REFORM\n\nContent about drugs.\n\n"
        "MANDATORY SUPERVISED RELEASE\n\nContent about release.\n\n"
        "FISCAL IMPACT ANALYSIS\n\nContent about fiscal impact."
    )
    sections = split_at_headings(text)
    headings = [h for h, _ in sections if h]
    assert "DRUG OFFENSE REFORM" in headings
    assert "MANDATORY SUPERVISED RELEASE" in headings
    assert "FISCAL IMPACT ANALYSIS" in headings


# ---------------------------------------------------------------------------
# Unit tests — chunk_record
# ---------------------------------------------------------------------------

def test_empty_text_produces_no_chunks():
    assert chunk_record(_make_record("")) == []


def test_stub_below_min_tokens_skipped():
    assert chunk_record(_make_record("Short.")) == []


def test_short_record_produces_single_chunk():
    text = "This is a substantial paragraph with enough tokens. " * 15
    chunks = chunk_record(_make_record(text))
    assert len(chunks) == 1
    assert chunks[0].chunk_total == 1
    assert chunks[0].source == "spac"


def test_all_caps_headings_split_into_separate_chunks():
    body = "Some content about the policy. " * 30
    text = (
        f"DRUG OFFENSE REFORM\n\n{body}\n\n"
        f"MANDATORY SUPERVISED RELEASE\n\n{body}"
    )
    chunks = chunk_record(_make_record(text))
    headings = {c.section_heading for c in chunks}
    assert "DRUG OFFENSE REFORM" in headings
    assert "MANDATORY SUPERVISED RELEASE" in headings


def test_two_word_heading_does_not_split():
    body = "Content paragraph. " * 20
    text = f"RETAIL THEFT\n\n{body}\n\nMore content follows here."
    chunks = chunk_record(_make_record(text))
    headings = {c.section_heading for c in chunks}
    assert "RETAIL THEFT" not in headings, "Two-word heading incorrectly triggered a split"


def test_enriched_text_contains_category_year_title():
    text = "Some policy content. " * 20
    chunks = chunk_record(_make_record(text, category="Cannabis Policy", year="2019", title="Cannabis Reform"))
    assert chunks
    assert "Cannabis Policy" in chunks[0].enriched_text
    assert "2019" in chunks[0].enriched_text
    assert "Cannabis Reform" in chunks[0].enriched_text


def test_enriched_text_contains_section_heading():
    body = "Content. " * 30
    text = f"DRUG OFFENSE REFORM\n\n{body}"
    chunks = chunk_record(_make_record(text))
    drug_chunks = [c for c in chunks if c.section_heading == "DRUG OFFENSE REFORM"]
    assert drug_chunks
    assert "DRUG OFFENSE REFORM" in drug_chunks[0].enriched_text


def test_no_chunk_exceeds_max_tokens():
    big_para = "The sentencing commission shall review all applicable standards. " * 50
    text = "\n\n".join([big_para] * 5)
    chunks = chunk_record(_make_record(text))
    over = [(c.chunk_id, c.token_count) for c in chunks if c.token_count > MAX_TOKENS]
    assert not over, f"Chunks exceed MAX_TOKENS={MAX_TOKENS}: {over}"


def test_chunk_ids_contiguous():
    body = "Content paragraph. " * 20
    text = (
        f"SECTION ONE HEADING\n\n{body}\n\n"
        f"SECTION TWO ANALYSIS\n\n{body}\n\n"
        f"SECTION THREE REFORM\n\n{body}"
    )
    chunks = chunk_record(_make_record(text))
    indices = [c.chunk_index for c in chunks]
    assert sorted(indices) == list(range(len(indices))), f"Non-contiguous: {indices}"
    for c in chunks:
        assert c.chunk_total == len(chunks)


def test_toc_lines_not_in_chunk_text():
    toc = "Introduction ......................... 4\nBackground ............................ 6\n"
    body = "Real content paragraph. " * 20
    chunks = chunk_record(_make_record(toc + body))
    for c in chunks:
        assert not _DOT_LEAK_RE.search(c.text), (
            f"Dot-leader line leaked into {c.chunk_id}: {c.text[:100]}"
        )


def test_page_headers_not_in_chunk_text():
    text = "Content A. " * 20 + "\nPage 3 of 15\n" + "Content B. " * 20
    chunks = chunk_record(_make_record(text))
    for c in chunks:
        assert not _PAGE_LEAK_RE.search(c.text), (
            f"Page header leaked into {c.chunk_id}: {c.text[:100]}"
        )


def test_chunk_id_format():
    text = "Substantial content paragraph. " * 15
    chunks = chunk_record(_make_record(text, rec_id="spac-my-report-2020"))
    assert chunks[0].chunk_id == "spac-my-report-2020_c0"


def test_source_field_is_spac():
    text = "Substantial content paragraph. " * 15
    chunks = chunk_record(_make_record(text))
    assert all(c.source == "spac" for c in chunks)


# ---------------------------------------------------------------------------
# Corpus-level property tests (require S3 via spac_chunks fixture)
# ---------------------------------------------------------------------------

def test_no_toc_lines_in_corpus_chunks(spac_chunks):
    failures = [
        c["chunk_id"] for c in spac_chunks
        if _DOT_LEAK_RE.search(c.get("text", ""))
    ]
    assert not failures, (
        f"{len(failures)} corpus chunks contain dot-leader ToC lines: {failures[:5]}"
    )


def test_no_page_headers_in_corpus_chunks(spac_chunks):
    failures = [
        c["chunk_id"] for c in spac_chunks
        if _PAGE_LEAK_RE.search(c.get("text", ""))
    ]
    assert not failures, (
        f"{len(failures)} corpus chunks contain page headers: {failures[:5]}"
    )


def test_chunk_index_contiguous(spac_chunks):
    by_record = defaultdict(list)
    for c in spac_chunks:
        by_record[c["record_id"]].append(c["chunk_index"])
    failures = [
        f"{rid}: {sorted(idxs)}"
        for rid, idxs in by_record.items()
        if set(idxs) != set(range(len(idxs)))
    ]
    assert not failures, "Non-contiguous chunk indices:\n" + "\n".join(failures[:5])


def test_chunk_total_accurate(spac_chunks):
    by_record = defaultdict(list)
    for c in spac_chunks:
        by_record[c["record_id"]].append(c)
    failures = [
        f"{c['chunk_id']}: chunk_total={c['chunk_total']} actual={len(siblings)}"
        for siblings in by_record.values()
        for c in siblings
        if c["chunk_total"] != len(siblings)
    ]
    assert not failures, "chunk_total mismatches:\n" + "\n".join(failures[:5])


def test_no_empty_chunks(spac_chunks):
    failures = [
        c["chunk_id"] for c in spac_chunks
        if c.get("token_count", 0) < MIN_CHUNK_TOKENS
    ]
    assert not failures, (
        f"{len(failures)} chunks below MIN_CHUNK_TOKENS={MIN_CHUNK_TOKENS}: {failures[:5]}"
    )


def test_chunk_ids_unique(spac_chunks):
    ids = [c["chunk_id"] for c in spac_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in SPAC chunks"


def test_no_chunk_exceeds_max_tokens_corpus(spac_chunks):
    failures = [
        (c["chunk_id"], c["token_count"]) for c in spac_chunks
        if c.get("token_count", 0) > MAX_TOKENS
    ]
    assert not failures, (
        f"{len(failures)} chunks exceed MAX_TOKENS={MAX_TOKENS}: {failures[:5]}"
    )


def test_all_chunks_have_required_fields(spac_chunks):
    required = {
        "chunk_id", "chunk_index", "chunk_total", "source",
        "text", "enriched_text", "token_count", "record_id",
        "title", "category", "year", "url",
    }
    failures = [
        f"{c['chunk_id']}: missing {required - c.keys()}"
        for c in spac_chunks
        if not required.issubset(c.keys())
    ]
    assert not failures, "Chunks missing required fields:\n" + "\n".join(failures[:5])


def test_enriched_text_nonempty(spac_chunks):
    failures = [
        c["chunk_id"] for c in spac_chunks
        if not c.get("enriched_text", "").strip()
    ]
    assert not failures, f"{len(failures)} chunks have empty enriched_text: {failures[:5]}"


def test_source_field_is_spac_corpus(spac_chunks):
    failures = [c["chunk_id"] for c in spac_chunks if c.get("source") != "spac"]
    assert not failures, f"{len(failures)} chunks have wrong source: {failures[:5]}"


def test_large_report_splits_into_multiple_chunks(spac_chunks):
    """The largest SPAC record (hb3355, ~193k chars) must produce many chunks."""
    target = [c for c in spac_chunks if "hb3355" in c.get("record_id", "")]
    if not target:
        pytest.skip("hb3355 record not found in chunked output")
    assert len(target) > 10, f"Expected >10 chunks for hb3355, got {len(target)}"
    headings = {c["section_heading"] for c in target if c["section_heading"]}
    assert len(headings) > 3, "Large report produced too few distinct section headings"


# ---------------------------------------------------------------------------
# S3 output verification (require chunked S3 bucket)
# ---------------------------------------------------------------------------

def test_s3_output_record_count(spac_chunks, spac_chunks_s3):
    assert len(spac_chunks_s3) == len(spac_chunks), (
        f"S3 has {len(spac_chunks_s3)} chunks, in-memory produced {len(spac_chunks)}"
    )


def test_s3_output_no_corrupt_records(spac_chunks_s3):
    required = {"chunk_id", "chunk_index", "chunk_total", "source", "text", "enriched_text"}
    failures = [
        f"record {i}: missing {required - c.keys()}"
        for i, c in enumerate(spac_chunks_s3)
        if not required.issubset(c.keys())
    ]
    assert not failures, "Corrupt records in S3 output:\n" + "\n".join(failures[:5])


def test_s3_output_no_empty_text(spac_chunks_s3):
    failures = [c["chunk_id"] for c in spac_chunks_s3 if not c.get("text", "").strip()]
    assert not failures, f"{len(failures)} chunks with empty text in S3: {failures[:5]}"


def test_s3_output_source_fields(spac_chunks_s3):
    failures = [c["chunk_id"] for c in spac_chunks_s3 if c.get("source") != "spac"]
    assert not failures, f"{len(failures)} S3 chunks have wrong source: {failures[:5]}"
```

- [ ] **Step 2: Run unit tests (no S3 access required)**

```bash
python3 -m pytest tests/test_spac_chunk.py -v -k "not corpus and not s3"
```

Expected: all unit tests pass. If any fail, fix `chunk/spac_chunk.py` before continuing.

- [ ] **Step 3: Commit passing unit tests**

```bash
git add tests/test_spac_chunk.py
git commit -m "test: unit + corpus + S3 tests for spac_chunk.py"
```

---

## Task 4: Run corpus tests against live S3 data

**Files:**
- No changes — exercises existing code against the real corpus

- [ ] **Step 1: Run corpus-level tests (requires `RAW_S3_BUCKET` set)**

```bash
python3 -m pytest tests/test_spac_chunk.py -v -k "corpus" --tb=short 2>&1 | tail -30
```

Expected: all corpus tests pass. Common failures and fixes:

- `test_no_empty_chunks` fails → `MIN_CHUNK_TOKENS` filter not applied after merge guard; check `filtered` line in `chunk_record`
- `test_no_chunk_exceeds_max_tokens_corpus` fails → a paragraph without `\n\n` breaks isn't being sentence-split; check `split_body` falls back to `_sentence_split` correctly
- `test_large_report_splits_into_multiple_chunks` fails → all-caps heading regex not matching; add a `print` debug call to `is_section_heading` with a few lines from the large record to diagnose

- [ ] **Step 2: If any corpus test fails, fix and re-run**

```bash
python3 -m pytest tests/test_spac_chunk.py -v -k "corpus" --tb=long 2>&1 | tail -50
```

Fix the issue in `chunk/spac_chunk.py`, then re-run until green.

- [ ] **Step 3: Commit any fixes**

```bash
git add chunk/spac_chunk.py
git commit -m "fix: spac chunker corpus test fixes"
```

---

## Task 5: Run pipeline locally and verify S3 output tests

**Files:**
- No changes — run the pipeline end-to-end

- [ ] **Step 1: Run chunker locally (limit 20 records for a quick smoke test)**

```bash
python3 chunk/spac_chunk.py --limit 20 --local-only 2>&1 | tail -10
```

Expected output similar to:
```
2026-04-27T... [INFO] Loaded 20 raw records
2026-04-27T... [INFO] Produced NNN chunks from 20 records (M skipped — stubs/empty)
2026-04-27T... [INFO] Single-chunk records: X  |  Multi-chunk records: Y
2026-04-27T... [INFO] Saved locally: data_files/chunked_output/spac_chunks.jsonl  (NNN chunks)
```

- [ ] **Step 2: Spot-check the local output**

```bash
python3 -c "
import json
with open('data_files/chunked_output/spac_chunks.jsonl') as f:
    chunks = [json.loads(l) for l in f if l.strip()]
print(f'Total chunks: {len(chunks)}')
print(f'Sample chunk_id: {chunks[0][\"chunk_id\"]}')
print(f'Source: {chunks[0][\"source\"]}')
print(f'Section heading: {chunks[0][\"section_heading\"]!r}')
print(f'Token count: {chunks[0][\"token_count\"]}')
print(f'Enriched text (first 200): {chunks[0][\"enriched_text\"][:200]}')
"
```

Expected: `source` is `"spac"`, token counts are < 800, enriched text starts with `"SPAC "`.

- [ ] **Step 3: Run full pipeline and upload to S3**

```bash
python3 chunk/spac_chunk.py 2>&1 | tail -10
```

Expected: uploads `spac/spac_chunks.jsonl` to `CHUNKED_S3_BUCKET`.

- [ ] **Step 4: Run S3 output tests**

```bash
python3 -m pytest tests/test_spac_chunk.py -v -k "s3" --tb=short 2>&1 | tail -20
```

Expected: all S3 tests pass. `test_s3_output_record_count` verifies in-memory and S3 agree.

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/test_spac_chunk.py -v --tb=short 2>&1 | tail -30
```

Expected: all tests pass (S3 output tests will skip if `CHUNKED_S3_BUCKET` not set, which is fine).

- [ ] **Step 6: Commit final state**

```bash
git add chunk/spac_chunk.py tests/test_spac_chunk.py tests/conftest.py
git commit -m "feat: SPAC chunker — pipeline, unit, corpus, and S3 output tests passing"
```
