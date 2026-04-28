# SPAC Chunker Design

**Date:** 2026-04-27
**Scope:** `chunk/spac_chunk.py` + `tests/test_spac_chunk.py`

---

## Context

The SPAC corpus (`spac/spac_corpus.jsonl`, 166 records) contains Illinois Sentencing Policy Advisory Council publications ingested from ICJIA's file server as PDFs. Records range from 173-char stub impact summaries to 193k-char comprehensive bill analyses. The corpus is narrative (no regulatory structure), with all-caps section headings as the primary structural signal.

The chunker must produce nodes suitable for both the existing vector+BM25 retrieval system and a future GraphRAG layer. The GraphRAG requirement drives the core design decision: **hard splits at section boundaries**, so each chunk represents one coherent topic and can be connected via typed edges.

---

## I/O and S3 Paths

| Direction | Path |
|-----------|------|
| Raw input | `s3://$RAW_S3_BUCKET/spac/spac_corpus.jsonl` |
| Chunked output | `s3://$CHUNKED_S3_BUCKET/spac/spac_chunks.jsonl` |
| Local output | `data_files/chunked_output/spac_chunks.jsonl` |

S3 prefix is controlled by `SPAC_S3_PREFIX` env var (default `"spac/"`).

CLI flags: `--local-only`, `--limit N` — identical to IAC/IDOC convention.

---

## Text Cleaning

Applied before any splitting:

1. **Page repeat headers** — strip via regex matching `Page \d+ of \d+` with surrounding context (header/footer lines that appear at every PDF page break, e.g. `"May 2017 Sentencing Reform Page 1 of 78\nHB3355 HA1"`).

2. **Table of contents dot-leader lines** — strip lines where `\.{4,}` occupies > 30% of the line's length (e.g. `"Introduction .............. 4"`). These are PDF navigation artifacts that degrade embedding quality.

No line-wrap collapsing needed — pypdf extracts SPAC text as full paragraphs (unlike the JCAR scraper that produced IAC's wrap artifacts).

---

## Section Detection

A line qualifies as an all-caps section heading if:
- Matches `^[A-Z][A-Z\s\(\)\-/,]{3,59}$` (all-caps, 4–60 chars)
- Contains ≥ 3 words (suppresses single-word/two-word table labels like `COMPONENT`, `RETAIL THEFT`)
- Is not purely numeric

---

## Splitting Pipeline

Per record:

1. Strip page headers and ToC dot-leader lines.
2. Split at qualifying all-caps headings → `(heading, body)` pairs. Text before the first heading becomes a preamble chunk with `section_heading = ""`.
3. Per section: if `token_count(body) ≤ MAX_TOKENS` (800), emit as-is. Otherwise split via token-aware paragraph accumulation:
   - Split body on `\n\n` → paragraphs
   - Accumulate greedily up to `TARGET_TOKENS` (600); flush when adding the next paragraph would exceed `TARGET_TOKENS`
   - Carry the last paragraph as overlap into the next chunk (capped at `MAX_TOKENS - next_paragraph_tokens`)
   - If a single paragraph exceeds `MAX_TOKENS`, fall back to sentence splitting, then hard token split
4. **Merge guard**: merge any chunk with `token_count < MIN_CHUNK_TOKENS` (40) forward into its next sibling within the same record. Applied as a single pass after all sections are split.
5. Skip records where the entire cleaned text is below `MIN_CHUNK_TOKENS`.

---

## Output Schema

Dataclass `SpacChunk` (serialized via `dataclasses.asdict`):

| Field | Type | Notes |
|-------|------|-------|
| `chunk_id` | str | `{record_id}_c{chunk_index}` |
| `chunk_index` | int | 0-based, contiguous |
| `chunk_total` | int | total chunks for this record |
| `source` | str | always `"spac"` |
| `text` | str | raw chunk text |
| `enriched_text` | str | see below |
| `token_count` | int | tiktoken cl100k_base |
| `section_heading` | str | all-caps heading or `""` for preamble |
| `record_id` | str | parent record `id` |
| `title` | str | from corpus record |
| `category` | str | inferred category from ingest |
| `year` | str | inferred year from filename |
| `url` | str | source PDF URL |
| `chunked_at` | str | ISO 8601 UTC |

**`enriched_text` format:**
```
SPAC {category} ({year}): {title}[ — {section_heading}]

{text}
```

---

## Token Budget

| Constant | Value | Env override |
|----------|-------|-------------|
| `TARGET_TOKENS` | 600 | `TARGET_TOKENS` |
| `MAX_TOKENS` | 800 | `MAX_TOKENS` |
| `OVERLAP_TOKENS` | 75 | — |
| `MIN_CHUNK_TOKENS` | 40 | `MIN_CHUNK_TOKENS` |
| Encoding | `cl100k_base` | — |

---

## Test Suite (`tests/test_spac_chunk.py`)

Two layers, following the IAC/IDOC convention:

### Unit tests (no S3, synthetic records)
- `test_empty_text_produces_no_chunks`
- `test_stub_record_below_min_tokens_skipped`
- `test_short_record_produces_single_chunk`
- `test_toc_lines_stripped` — dot-leader lines absent from all chunk text
- `test_page_headers_stripped` — `Page \d+ of \d+` absent from all chunk text
- `test_all_caps_heading_splits_sections` — two distinct headings → separate chunks
- `test_single_word_heading_not_split` — two-word heading does not trigger split
- `test_no_chunk_exceeds_max_tokens` — synthetic large record
- `test_enriched_text_contains_title_and_category`
- `test_chunk_ids_contiguous`

### Corpus-level property tests (via `spac_chunks` fixture, requires S3 raw)
- `test_no_toc_lines_in_corpus_chunks`
- `test_no_page_headers_in_corpus_chunks`
- `test_chunk_index_contiguous`
- `test_chunk_total_accurate`
- `test_no_empty_chunks`
- `test_chunk_ids_unique`
- `test_no_chunk_exceeds_max_tokens_corpus`
- `test_all_chunks_have_required_fields`
- `test_enriched_text_nonempty`
- `test_source_field_is_spac`

### S3 output verification (via `spac_chunks_s3` fixture)
- `test_s3_output_record_count`
- `test_s3_output_no_corrupt_records`
- `test_s3_output_no_empty_text`
- `test_s3_output_source_fields`

### Fixtures (added to `tests/conftest.py`)
- `spac_records` — downloads `spac/spac_corpus.jsonl` from `RAW_S3_BUCKET`
- `spac_chunks` — runs `chunk_record()` over `spac_records`
- `spac_chunks_s3` — downloads `spac/spac_chunks.jsonl` from `CHUNKED_S3_BUCKET`
