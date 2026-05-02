# CAP Bulk Chunker — Design Spec

**Date:** 2026-05-02
**Status:** Approved

## Overview

Write `chunk/cap_chunk.py` to chunk the CAP bulk corpus (`cap_bulk_corpus.jsonl`, 172,760 IL court opinions) into `Chunk` objects ready for embedding. Extract shared opinion-chunking utilities from `courtlistener_chunk.py` into a new `chunk/opinion_utils.py` module so both chunkers share a single implementation. Register `cap` in `batch_chunk.py`.

## Files

| File | Action |
|------|--------|
| `chunk/opinion_utils.py` | **New** — shared constants and pure functions extracted from `courtlistener_chunk.py` |
| `chunk/courtlistener_chunk.py` | **Refactor** — replace extracted inline defs with `from chunk.opinion_utils import ...` |
| `chunk/cap_chunk.py` | **New** — CAP-specific chunker |
| `batch_chunk.py` | **Update** — add `cap` source entry |

## `chunk/opinion_utils.py`

Contains everything that is purely computational with no I/O or source-specific logic. Both `courtlistener_chunk.py` and `cap_chunk.py` import from it.

**Extracted from `courtlistener_chunk.py`:**

- Constants: `TARGET_TOKENS = 600`, `MAX_TOKENS = 800`, `OVERLAP_TOKENS = 75`, `MIN_CHUNK_TOKENS = 50`, `ENCODING_NAME = "cl100k_base"`
- `_enc`, `count_tokens()`, `token_split()`
- `_NOISE_PATTERNS`, `_SECTION_PATTERNS`
- `is_noise_chunk()`, `detect_sections()`, `split_section()`, `_accumulate()`, `_sentence_split()`
- `strip_html()`, `safe_str()`, `safe_int()`, `safe_float()` — used by `courtlistener_chunk.py` only (Pandas rows can be NaN); `cap_chunk.py` does not import these
- `_opinion_enriched_text()`, `_opinion_display_citation()`

**`courtlistener_chunk.py` change:** Replace the extracted function bodies with a single import block. All remaining logic (CSV reading, S3 I/O, cluster/docket maps, parenthetical handling, `run()`, `main()`) stays in place.

## `chunk/cap_chunk.py`

### Input

- `--local-only`: reads `data_files/corpus/cap_bulk_corpus.jsonl`
- Default (S3): reads `s3://${RAW_S3_BUCKET}/${CAP_S3_PREFIX}/cap_bulk_corpus.jsonl`
  - Env var `CAP_S3_PREFIX` defaults to `"cap"`

### Opinion-type splitting

CAP bundles all opinion types (majority, dissent, concurrence) into a single `text` field separated by bracket markers. The chunker splits at these markers before section detection so each opinion type is processed independently.

```
_OPINION_MARKER_RE = re.compile(
    r'^\[(Majority|Dissent|Concurrence|Concur|Rehearing|Per Curiam|Opinion)\]',
    re.MULTILINE | re.IGNORECASE,
)
```

Labels map to normalized values: `"Majority"` → `"majority"`, `"Concurrence"/"Concur"` → `"concurrence"`, etc.

`is_majority` is True only for `"majority"`.

### Chunking pipeline (per opinion-type segment)

1. `detect_sections(segment_text)` — identify named sections (BACKGROUND, ANALYSIS, etc.)
2. `split_section(heading, body)` for each section — token-aware paragraph/sentence accumulation
3. Filter: drop chunks where `count_tokens(text) < MIN_CHUNK_TOKENS` or `is_noise_chunk(text, tokens)`
4. Emit one `Chunk` per surviving unit

### Court label mapping

```python
CAP_COURT_LABELS = {
    "Ill.":          "Illinois Supreme Court",
    "Ill. App. Ct.": "Illinois Appellate Court",
}
```

Unknown `court` values fall back to the raw string.

### Display citation

`citations[0]` (the official IL reporter citation, e.g. `"309 Ill. App. 3d 542"`) combined with `case_name_abbr + (year)`:

```
"309 Ill. App. 3d 542 — Weyland v. Manning (2000)"
```

Falls back to `case_name_abbr (year)` if `citations` is empty.

### Chunk IDs and indexing

- `parent_id`: the entry `id` field, e.g. `"cap-435690"`
- `chunk_id`: `f"{parent_id}_t{type_idx}_c{chunk_index}"`
  - `type_idx` disambiguates majority (0) from dissent (1), concurrence (2), etc. within the same case
  - `chunk_index` is local to each opinion-type segment; `chunk_total` reflects the count for that segment

### Metadata fields

| Field | Type | Description |
|-------|------|-------------|
| `chunk_type` | str | `"opinion_section"` if named heading, else `"opinion_paragraph"` |
| `section_heading` | str | Detected heading string, or `""` |
| `section_index` | int | Index of the parent section within the segment |
| `opinion_type` | str | `"majority"`, `"dissent"`, `"concurrence"`, `"rehearing"`, etc. |
| `is_majority` | bool | True only for `"majority"` type |
| `case_id` | str | Raw CAP `case_id` field |
| `case_name` | str | Full case name |
| `case_name_abbr` | str | Short case name (e.g. `"Weyland v. Manning"`) |
| `date_decided` | str | ISO date string, e.g. `"2000-01-07"` |
| `court` | str | Raw court field, e.g. `"Ill. App. Ct."` |
| `court_label` | str | Human-readable label, e.g. `"Illinois Appellate Court"` |
| `citations` | list[str] | All citation strings from the entry |

### Output

- `--local-only`: writes `data_files/chunked_output/cap_opinion_chunks.jsonl`
- Default (S3): writes `s3://${CHUNKED_S3_BUCKET}/${CAP_S3_PREFIX}/cap_opinion_chunks.jsonl`

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--local-only` | False | Read from local `data_files/`, write locally |
| `--limit N` | 0 (all) | Process only first N opinions by text length — for testing |

### Logging

Progress logged every 5,000 opinions. Final summary: chunk count, opinion count, skipped-no-text count, token stats (avg / min / max).

## `batch_chunk.py` update

Add to `_build_sources()`:

```python
Source(
    "cap",
    "chunk.cap_chunk",
    s3_check_key=f"{cap_prefix}/cap_opinion_chunks.jsonl",
),
```

Where `cap_prefix = os.environ.get("CAP_S3_PREFIX", "cap").rstrip("/")`.

Update the `--sources` help string to include `cap`.

## Out of scope

- `merge_opinion_chunks.py` — no change needed; CAP (IL state) and CourtListener (7th Circuit federal) are non-overlapping by design
- Embedding — `cap_opinion_chunks.jsonl` will be consumed by a future `embed/opinion_embed.py`; no embedding work here
- Test suite — a `tests/test_cap_chunk.py` is the natural follow-on but is not part of this spec
