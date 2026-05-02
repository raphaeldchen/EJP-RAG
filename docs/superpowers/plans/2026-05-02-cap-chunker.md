# CAP Bulk Chunker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract shared opinion-chunking utilities from `courtlistener_chunk.py` into `chunk/opinion_utils.py`, then write `chunk/cap_chunk.py` to chunk the 172,760-opinion CAP bulk corpus into `Chunk` objects ready for embedding.

**Architecture:** A new `chunk/opinion_utils.py` module holds all pure chunking functions (token math, section detection, noise filtering, enriched-text assembly) shared by both CourtListener and CAP chunkers. `cap_chunk.py` reads `cap_bulk_corpus.jsonl` line-by-line, splits each entry's `text` field at `[Majority]`/`[Dissent]`/`[Concurrence]` markers, then runs the shared detect→split→filter pipeline on each opinion-type segment. `batch_chunk.py` gains a `cap` source entry.

**Tech Stack:** `tiktoken` (cl100k_base), `beautifulsoup4`, `boto3`, `python-dotenv`, `core.models.Chunk`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `chunk/opinion_utils.py` | **Create** | Shared constants + pure functions for opinion chunking |
| `chunk/courtlistener_chunk.py` | **Modify** | Remove extracted defs; import from `opinion_utils` |
| `chunk/cap_chunk.py` | **Create** | CAP-specific chunker: parse markers, chunk, emit Chunks |
| `batch_chunk.py` | **Modify** | Add `cap` source entry |

---

## Task 1: Create `chunk/opinion_utils.py`

**Files:**
- Create: `chunk/opinion_utils.py`

- [ ] **Step 1: Write `chunk/opinion_utils.py`**

```python
"""Shared utilities for opinion chunking (CAP and CourtListener)."""

import math
import re

import tiktoken
from bs4 import BeautifulSoup

ENCODING_NAME    = "cl100k_base"
TARGET_TOKENS    = 600
MAX_TOKENS       = 800
OVERLAP_TOKENS   = 75
MIN_CHUNK_TOKENS = 50

_enc = tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def token_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    tokens = _enc.encode(text)
    chunks = []
    start  = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(_enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap_tokens
    return chunks


def strip_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["p", "br", "div", "h1", "h2", "h3", "h4", "h5"]):
        tag.insert_before("\n")
    return re.sub(r"\n{3,}", "\n\n", soup.get_text()).strip()


def safe_str(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val).strip()


def safe_int(val, default: int = 0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


_NOISE_PATTERNS = [
    re.compile(r"^\s*-\s*\d+\s*-\s*$", re.MULTILINE),
    re.compile(r"\x0c"),
    re.compile(r"^\s*(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH)\s+DISTRICT\s*$",
               re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*(?:ILLINOIS\s+)?SUPREME\s+COURT\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"v\.\s{2,}[\)|\|]"),
    re.compile(r"\bNos?\.\s+\d+[-\w]+"),
    re.compile(r"^\s*NOTICE\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"Order filed|Decision filed", re.IGNORECASE),
    re.compile(r"This order was filed under|Rule 23", re.IGNORECASE),
    re.compile(r"^\s*_{5,}\s*$", re.MULTILINE),
    re.compile(r"^\s*Nos?\.\s+\d", re.MULTILINE),
]


def is_noise_chunk(text: str, token_count: int) -> bool:
    if token_count > 150:
        return False
    hits = sum(1 for p in _NOISE_PATTERNS if p.search(text))
    threshold = 1 if token_count < 60 else 2
    return hits >= threshold


_SECTION_PATTERNS = [
    re.compile(
        r"^\s*(X{0,3}(?:IX|IV|V?I{0,3}))\s*[.\-—]\s*(.{0,80})$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(BACKGROUND|FACTS|PROCEDURAL\s+HISTORY|PROCEDURAL\s+BACKGROUND|"
        r"ANALYSIS|DISCUSSION|HOLDING|DISPOSITION|CONCLUSION|JURISDICTION|"
        r"STANDARD\s+OF\s+REVIEW|APPLICABLE\s+LAW|RELEVANT\s+STATUTES?|"
        r"PRELIMINARY\s+MATTERS?|PRIOR\s+PROCEEDINGS?|OPINION)\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(r"^\s*([A-Z][A-Z\s]{4,60})\s*$", re.MULTILINE),
    re.compile(
        r"^\s*([A-Z]|\d+)\s*[.\-—]\s*([A-Z][^.\n]{5,60})\s*$",
        re.MULTILINE,
    ),
]


def detect_sections(text: str) -> list[tuple[str, str]]:
    hits: list[tuple[int, str]] = []
    for pattern in _SECTION_PATTERNS:
        for m in pattern.finditer(text):
            hits.append((m.start(), m.group(0).strip()))
    if not hits:
        return [("", text)]
    hits.sort(key=lambda x: x[0])
    deduped: list[tuple[int, str]] = []
    last_pos = -100
    for pos, heading in hits:
        if pos - last_pos > 20:
            deduped.append((pos, heading))
            last_pos = pos
    sections: list[tuple[str, str]] = []
    for i, (pos, heading) in enumerate(deduped):
        start = pos + len(heading)
        end   = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        body  = text[start:end].strip()
        if body:
            sections.append((heading, body))
    preamble = text[: deduped[0][0]].strip()
    if preamble:
        sections.insert(0, ("PREAMBLE", preamble))
    return sections if sections else [("", text)]


def _sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z¶])", text) if s.strip()]


def _accumulate(units: list[str], separator: str) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = count_tokens(unit)
        if unit_tokens > MAX_TOKENS:
            if current:
                result.append(separator.join(current))
                current, current_tokens = [], 0
            result.extend(token_split(unit, MAX_TOKENS, OVERLAP_TOKENS))
            continue
        if current_tokens + unit_tokens > TARGET_TOKENS and current:
            result.append(separator.join(current))
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
        result.append(separator.join(current))
    return result


def split_section(heading: str, body: str) -> list[tuple[str, str]]:
    if count_tokens(body) <= MAX_TOKENS:
        return [(heading, body)]
    normalised = re.sub(r"(?<=[.!?]) *\n(?=[A-Z¶])", "\n\n", body)
    paragraphs = [p.strip() for p in normalised.split("\n\n") if p.strip()]
    if len(paragraphs) <= 2:
        sentences = _sentence_split(body)
        if len(sentences) > 2:
            paragraphs = sentences
    chunks = _accumulate(paragraphs, "\n\n")
    return [(heading, chunk) for chunk in chunks]


def _opinion_enriched_text(
    case_name_short: str,
    date_filed: str,
    court_label: str,
    section_heading: str,
    chunk_text: str,
) -> str:
    header_parts = [x for x in [case_name_short, date_filed, court_label] if x]
    header = " | ".join(header_parts)
    if section_heading:
        return f"{header}\n{section_heading}\n\n{chunk_text}"
    return f"{header}\n\n{chunk_text}" if header else chunk_text


def _opinion_display_citation(case_name_short: str, date_filed: str) -> str:
    year = date_filed[:4] if date_filed and len(date_filed) >= 4 else ""
    if case_name_short and year:
        return f"{case_name_short} ({year})"
    return case_name_short or ""
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
python3 -c "
from chunk.opinion_utils import (
    count_tokens, token_split, detect_sections, split_section,
    is_noise_chunk, _opinion_enriched_text, _opinion_display_citation,
    TARGET_TOKENS, MAX_TOKENS, MIN_CHUNK_TOKENS,
)
# Quick smoke: token count of a known string
assert count_tokens('hello world') == 2, 'token count wrong'
# Section detection on a minimal text
secs = detect_sections('BACKGROUND\nSome text here.')
assert len(secs) >= 1
print('opinion_utils OK')
"
```

Expected output: `opinion_utils OK`

- [ ] **Step 3: Commit**

```bash
git add chunk/opinion_utils.py
git commit -m "feat: extract shared opinion chunking utilities to opinion_utils.py"
```

---

## Task 2: Refactor `chunk/courtlistener_chunk.py`

**Files:**
- Modify: `chunk/courtlistener_chunk.py`

This task removes the function bodies that moved to `opinion_utils.py` and replaces them with a single import block. The remaining logic (S3 I/O, cluster/docket maps, parenthetical handling, `run()`, `main()`) is untouched.

- [ ] **Step 1: Replace the extracted import lines and function bodies**

At the top of `courtlistener_chunk.py`, after the existing import block (after `from dotenv import load_dotenv`), add:

```python
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
```

Then delete the following blocks from `courtlistener_chunk.py` (they now live in `opinion_utils.py`):

- The five token constants (`TARGET_TOKENS`, `MAX_TOKENS`, `OVERLAP_TOKENS`, `MIN_CHUNK_TOKENS`, `ENCODING_NAME`) and `PRECEDENTIAL_WEIGHT` / `OPINION_TYPE_LABELS` / `MAJORITY_TYPES` / `COURT_LABELS` remain — keep them. Only remove the five token constants.
- Remove `_opinion_enriched_text()` definition
- Remove `_opinion_display_citation()` definition
- Remove `_enc = tiktoken.get_encoding(ENCODING_NAME)` and `count_tokens()`
- Remove `token_split()`
- Remove `strip_html()`
- Remove `safe_str()`, `safe_int()`, `safe_float()`
- Remove `_NOISE_PATTERNS` and `is_noise_chunk()`
- Remove `_SECTION_PATTERNS` and `detect_sections()`
- Remove `_sentence_split()`
- Remove `_accumulate()`
- Remove `split_section()`

Also remove these now-unused top-level imports from `courtlistener_chunk.py`:
- `import tiktoken`
- `from bs4 import BeautifulSoup`

Keep in `courtlistener_chunk.py` (do NOT remove):
- `_par_enriched_text()` and `_par_display_citation()` — parenthetical helpers, CL-specific
- `row_get()` — Pandas row accessor, CL-specific
- `PRECEDENTIAL_WEIGHT`, `OPINION_TYPE_LABELS`, `MAJORITY_TYPES`, `COURT_LABELS` — CL-specific constants
- `s3_client()`, `read_csv_from_s3()`, `write_jsonl_to_s3()`, `write_jsonl_local()` — I/O helpers
- `build_lookup_maps()`, `chunk_opinion()`, `chunk_parentheticals()`, `run()`, `main()` — core logic

- [ ] **Step 2: Verify the module still imports and the key function is reachable**

```bash
python3 -c "
from chunk.courtlistener_chunk import chunk_opinion, chunk_parentheticals
import pandas as pd
# Minimal smoke: chunk_opinion returns [] for an empty row
row = pd.Series({'id': '1', 'cluster_id': '2', 'type': '020lead',
                 'plain_text': '', 'html_with_citations': '',
                 'author_str': '', 'per_curiam': 'false'})
result = chunk_opinion(row, {}, {})
assert result == [], f'expected [] got {result}'
print('courtlistener_chunk refactor OK')
"
```

Expected output: `courtlistener_chunk refactor OK`

- [ ] **Step 3: Commit**

```bash
git add chunk/courtlistener_chunk.py
git commit -m "refactor: courtlistener_chunk imports shared utils from opinion_utils"
```

---

## Task 3: Write `chunk/cap_chunk.py`

**Files:**
- Create: `chunk/cap_chunk.py`

- [ ] **Step 1: Write `chunk/cap_chunk.py`**

```python
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
    r"^\[(Majority|Dissent|Concurrence|Concur|Rehearing|Per Curiam|Opinion)\]",
    re.MULTILINE | re.IGNORECASE,
)

_OPINION_TYPE_NORM: dict[str, str] = {
    "majority":   "majority",
    "dissent":    "dissent",
    "concurrence": "concurrence",
    "concur":     "concurrence",
    "rehearing":  "rehearing",
    "per curiam": "majority",
    "opinion":    "majority",
}

_MAJORITY_TYPES = {"majority"}


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
                chunk_id         = f"{entry_id}_t{type_idx}_c{chunk_index}",
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
```

- [ ] **Step 2: Smoke-test the chunk_entry function directly**

```bash
python3 -c "
from chunk.cap_chunk import chunk_entry, _split_opinion_segments

# Test: marker splitting
segs = _split_opinion_segments('[Majority]\nThis is the majority opinion.\n[Dissent]\nI dissent.')
assert len(segs) == 2, f'expected 2 segments, got {len(segs)}'
assert segs[0][0] == 'majority'
assert segs[1][0] == 'dissent'

# Test: single entry chunks correctly
entry = {
    'id': 'cap-999',
    'case_id': '999',
    'case_name': 'Test v. Case',
    'case_name_abbr': 'Test v. Case',
    'date_decided': '2000-01-01',
    'court': 'Ill.',
    'citations': ['1 Ill. 2d 100'],
    'text': '[Majority]\nJUSTICE SMITH delivered the opinion of the court:\n' + ('The defendant was charged. ' * 60),
}
chunks = chunk_entry(entry)
assert len(chunks) >= 1, f'expected at least 1 chunk, got {len(chunks)}'
c = chunks[0]
assert c.source == 'cap_bulk'
assert c.parent_id == 'cap-999'
assert c.metadata['opinion_type'] == 'majority'
assert c.metadata['is_majority'] is True
assert c.metadata['court_label'] == 'Illinois Supreme Court'
assert '1 Ill. 2d 100' in c.display_citation
assert c.token_count > 0
print(f'chunk_entry OK — {len(chunks)} chunk(s), first display_citation: {c.display_citation!r}')
"
```

Expected output: `chunk_entry OK — N chunk(s), first display_citation: '1 Ill. 2d 100 — Test v. Case (2000)'`

- [ ] **Step 3: Run the chunker on 20 opinions from the local corpus**

```bash
python3 -m chunk.cap_chunk --local-only --limit 20
```

Expected output (last few lines):
```
Done. N chunks from 20 opinions. Skipped: 0 (no text).
Token stats: avg NNN | min NN | max NNN
Saved locally: data_files/chunked_output/cap_opinion_chunks.jsonl  (N records)
```

- [ ] **Step 4: Verify the output structure**

```bash
python3 -c "
import json
from pathlib import Path

path = Path('data_files/chunked_output/cap_opinion_chunks.jsonl')
records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
assert records, 'no records written'

required = {'chunk_id', 'parent_id', 'chunk_index', 'chunk_total', 'text',
            'enriched_text', 'source', 'token_count', 'display_citation', 'metadata'}
for r in records:
    missing = required - r.keys()
    assert not missing, f'missing fields: {missing} in {r[\"chunk_id\"]}'
    assert r['source'] == 'cap_bulk'
    assert r['token_count'] > 0
    assert r['text'].strip()
    assert r['enriched_text'].strip()
    md = r['metadata']
    assert md['opinion_type'] in {'majority', 'dissent', 'concurrence', 'rehearing'}
    assert isinstance(md['is_majority'], bool)
    assert md['court_label'] in {'Illinois Supreme Court', 'Illinois Appellate Court'}

# IDs must be unique
ids = [r['chunk_id'] for r in records]
assert len(ids) == len(set(ids)), 'duplicate chunk_ids found'

print(f'Structure OK — {len(records)} records, all fields present, IDs unique')
"
```

Expected output: `Structure OK — N records, all fields present, IDs unique`

- [ ] **Step 5: Commit**

```bash
git add chunk/cap_chunk.py
git commit -m "feat: add CAP bulk opinion chunker"
```

---

## Task 4: Register `cap` in `batch_chunk.py`

**Files:**
- Modify: `batch_chunk.py`

- [ ] **Step 1: Add `cap_prefix` variable and `cap` Source to `_build_sources()`**

In `_build_sources()`, add `cap_prefix` near the top of the function alongside the other prefix variables:

```python
cap_prefix  = os.environ.get("CAP_S3_PREFIX", "cap").rstrip("/")
```

Then add the `cap` Source entry at the end of the returned list (after `courtlistener`):

```python
Source(
    "cap",
    "chunk.cap_chunk",
    s3_check_key=f"{cap_prefix}/cap_opinion_chunks.jsonl",
),
```

- [ ] **Step 2: Update the `--sources` help string**

Find the `epilog` string in `main()`:

```python
"Sources: ilga, iscr, iac, iccb, idoc, spac, federal, "
"restorejustice, cookcounty-pd, courtlistener"
```

Replace with:

```python
"Sources: ilga, iscr, iac, iccb, idoc, spac, federal, "
"restorejustice, cookcounty-pd, courtlistener, cap"
```

- [ ] **Step 3: Verify `cap` appears in help output**

```bash
python3 batch_chunk.py --help
```

Expected: the help text lists `cap` in the sources list, and the `--sources` argument accepts it without error.

```bash
python3 batch_chunk.py --sources cap --help 2>&1 | head -5
```

Expected: no `ERROR: unknown sources` message.

- [ ] **Step 4: Commit**

```bash
git add batch_chunk.py
git commit -m "feat: register cap source in batch_chunk.py"
```

---

## Self-Review Checklist

- [x] **`opinion_utils.py` created** with all extracted constants and functions — Task 1
- [x] **`courtlistener_chunk.py` refactored** — Task 2 removes inline defs, adds import block
- [x] **`cap_chunk.py` created** — Task 3 implements `_split_opinion_segments`, `chunk_entry`, `run`, `main`
- [x] **`batch_chunk.py` updated** — Task 4 adds `cap` Source and updates epilog
- [x] **Opinion-type splitting** (`[Majority]`/`[Dissent]`/`[Concurrence]`) — `_split_opinion_segments()` in Task 3
- [x] **Court label mapping** (`"Ill."` → `"Illinois Supreme Court"`) — `CAP_COURT_LABELS` in Task 3
- [x] **Display citation format** (`"309 Ill. App. 3d 542 — Weyland v. Manning (2000)"`) — `_build_display_citation()` in Task 3
- [x] **Chunk IDs** (`cap-N_t0_c0`) — `chunk_id` construction in `chunk_entry()` in Task 3
- [x] **`--local-only` reads from `data_files/corpus/`** — `run()` in Task 3
- [x] **`--limit N`** — `run()` in Task 3
- [x] **`CAP_S3_PREFIX` env var** — `_get_config()` in Task 3, `_build_sources()` in Task 4
- [x] **Metadata fields** all present — `chunk_entry()` metadata dict in Task 3
- [x] **Source field** is `"cap_bulk"` — matches the ingest `source` field value
- [x] **`_par_enriched_text` / `_par_display_citation` / `row_get`** not removed from `courtlistener_chunk.py` — Task 2 keep-list is explicit
