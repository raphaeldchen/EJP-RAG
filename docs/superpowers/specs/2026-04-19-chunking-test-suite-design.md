# Chunking Test Suite Design

**Date:** 2026-04-19
**Scope:** Test suite for `chunk/ilga_chunk.py` and `chunk/iscr_chunk.py`, plus targeted bug fixes identified during test design.

---

## Goals

1. Reproduce known chunking failures as regression tests so they cannot silently regress.
2. Add property-based coverage that sweeps the full corpus and catches classes of bugs (orphaned enumerations, oversized chunks, etc.).
3. Fix two confirmed bugs in `iscr_chunk.py` alongside the tests.
4. Diagnose the ILCS 705 ILCS 405/5-915 failure as chunking vs. retrieval — the test result determines whether a fix is needed.

---

## Directory Structure

```
tests/
  conftest.py                  # shared fixtures — S3 fetching, caching, corpus loading
  fixtures/                    # gitignored; populated on first run
    iscr/                      #   <pdf_stem>.txt per ISCR PDF (pdfplumber-extracted)
    ilcs/                      #   (reserved for future use)
  test_ilga_chunk.py           # ILCS chunker tests
  test_iscr_chunk.py           # ISCR chunker tests
```

`tests/fixtures/` is gitignored — cached text files are not committed.

---

## Fixture Strategy (`conftest.py`)

Both fixtures are `scope="session"` so S3 is hit at most once per test run.

**`iscr_texts`**
Lists ISCR PDFs in S3 under `SUPREME_COURT_RULES_S3_PREFIX`, downloads each, extracts text with `pdfplumber`, writes to `tests/fixtures/iscr/<stem>.txt`. On subsequent runs reads from the cache file and skips S3. Returns `dict[str, str]` mapping PDF stem → extracted text.

**`ilcs_records`**
Loads all records from the local `ilcs_corpus.jsonl` file. Returns `list[dict]`. No S3 dependency.

**`ilcs_chunks`**
Derived fixture: runs `chunk_section()` over all `ilcs_records` and returns the flat list of all output chunks. Computed once per session.

**`iscr_chunks`**
Derived fixture: runs `chunk_document()` over each text in `iscr_texts` and returns the combined flat list. Computed once per session.

---

## ILCS Tests (`test_ilga_chunk.py`)

### Regression tests

**`test_5_915_chunk_is_self_contained`**
Find the 705 ILCS 405/5-915 record in `ilcs_records`, chunk it, assert that at least one chunk contains the term `"expunge"` (or `"automatic"`) and that the chunk text ends with a sentence-terminating character (not mid-word). If this passes, the 5-915 issue is a retrieval/embedding problem, not chunking.

**`test_enumeration_not_severed`**
Construct a synthetic record with a section body of the form:
```
Opening clause that introduces the following:
(a) First item with enough text to pad it out.
(b) Second item with enough text to pad it out.
(c) Third item with enough text to pad it out.
```
long enough to trigger subsection splitting. Assert no output chunk at `chunk_index > 0` begins with a bare `\([a-z]\)` marker without the opening clause present in the same chunk.

### Property tests (sweep full corpus)

**`test_no_orphaned_subsection_starts`**
For every chunk where `chunk_index > 0`, assert its text does not match `^\s*\([a-z0-9]\)` at the start. A chunk beginning with a bare subsection marker signals the opening clause was severed.

**`test_chunk_index_contiguous`**
Group chunks by `parent_id`. For each group, assert the set of `chunk_index` values is exactly `{0, 1, ..., chunk_total - 1}`.

**`test_chunk_total_accurate`**
For every chunk, assert `chunk_total == len(siblings)` where siblings are all chunks sharing the same `parent_id`.

**`test_no_empty_chunks`**
Assert no chunk has `len(text) < MIN_CHUNK_SIZE`.

**`test_chunk_ids_unique`**
Assert `len({c["chunk_id"] for c in all_chunks}) == len(all_chunks)`.

**`test_no_chunk_exceeds_max_size`**
Assert `len(chunk["text"]) <= CHUNK_SIZE` for every chunk. Catches cases where `split_by_sentences` failed to further split an oversized subsection.

**`test_sentence_split_overlap`**
For any `parent_id` with `chunk_total > 1` where sentence splitting fired (detectable because the chunks lack subsection markers), assert that the last sentence of chunk N appears somewhere in the text of chunk N+1. Validates the `overlap_buf` logic in `split_by_sentences`.

---

## ISCR Tests (`test_iscr_chunk.py`)

### Regression tests

**`test_rule_401_enumeration_intact`**
Find all chunks where `rule_number == "401"`. Assert that the chunk containing `(1)`, `(2)`, `(3)` list items also contains the parent introductory clause (assert both `re.search(r"\(1\)", text)` and `re.search(r"shall|inform", text)` match the same chunk). This is the exact Rule 401(a) failure from CLAUDE.md.

**`test_rule_subsection_has_parent_context`**
For every chunk with `content_type == "rule_subsection"`, assert its text does not begin with `^\s*\(\d+\)` or `^\s*\([a-z]\)`. A subsection chunk starting with a bare enumeration marker has been orphaned from its parent clause.

### Property tests

**`test_rule_chunks_have_rule_number`**
Every chunk with `content_type` in `{"rule_text", "rule_subsection"}` has a non-empty `rule_number`.

**`test_hierarchical_path_consistent`**
For every chunk where `rule_number` is set, assert `hierarchical_path` contains `f"Rule {rule_number}"`. Catches cases where `DocumentHierarchy` state drifts out of sync with the path builder.

**`test_no_duplicate_chunk_ids`**
Across all chunks from all PDFs, assert no `chunk_id` repeats.

**`test_no_empty_chunks`**
No chunk has empty or whitespace-only `text`.

**`test_page_markers_stripped`**
No chunk text contains `[PAGE \d+]`. These markers are injected by `merge_pages_to_text` and should not bleed into chunk text.

**`test_no_chunk_exceeds_target_size`**
For every chunk with `content_type == "rule_subsection"`, assert `token_estimate` is below the split threshold (`len(text) > 1000` with `>= 3` subsections triggers splitting, so no subsection chunk should exceed that threshold). Catches cases where the split fired but produced oversized sub-chunks.

**`test_large_rule_produces_multiple_chunks`**
For any rule text where `should_split_rule(text)` returns `True`, assert the chunker produced more than one output chunk for that rule. Verifies the split actually fires end-to-end.

**`test_small_chunks_not_orphaned`**
Any chunk with `token_estimate < 10` must have `content_type` in `{"article_header", "part_header"}`. Rule text or subsection chunks this small are junk micro-chunks produced by misfiring splits.

---

## Bug Fixes

### Fix 1 — ISCR: Rule 401(a) enumeration severance

**Location:** `chunk/iscr_chunk.py`, `split_rule_into_subsections()`

**Problem:** When the loop hits the first subsection marker `(1)`, it saves the preceding lines as a separate `subsection_id="intro"` chunk. The intro (the parent clause that introduces the enumeration) is emitted standalone, leaving the `(1)`, `(2)`, `(3)` chunk without context.

**Fix:** Instead of emitting the intro as a standalone chunk, prepend its text to the first real subsection chunk. The intro and the first enumerated item become a single chunk.

### Fix 2 — ISCR: `[PAGE N]` markers bleeding into chunk text

**Location:** `chunk/iscr_chunk.py`, `chunk_document()` or `merge_pages_to_text()`

**Problem:** `merge_pages_to_text` injects `[PAGE N]\n` before each page's text. `chunk_document` never strips these markers, so they appear in output chunk text.

**Fix:** Strip lines matching `^\[PAGE \d+\]$` at the start of `chunk_document` before the line-by-line processing loop.

### Non-fix — ILCS 705 ILCS 405/5-915

The test `test_5_915_chunk_is_self_contained` will determine whether this is a chunking bug. If it passes, the failure is in retrieval/embedding — not the chunker — and no change to `ilga_chunk.py` is warranted. Fix deferred until test verdict.

---

## Out of Scope

- CAP bulk chunker (not yet written)
- CourtListener chunker (`courtlistener_chunk.py`) — no known failures; add tests after ILCS/ISCR are stable
- Embedding or retrieval fixes — separate concern
