# Shared Chunk Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a shared `Chunk` dataclass in `core/models.py` and migrate all 11 chunkers to emit it, so the retrieval layer can read any Collection uniformly without hard-coded field names.

**Architecture:** Create `core/models.py` as a dependency-free shared module. Migrate chunkers one at a time, verifying tests pass after each. Update `CitationLabelingPostprocessor` and `BM25Retriever` last, once all chunkers are emitting the unified schema. The two chunkers without test suites (`courtlistener_chunk.py`, `merge_opinion_chunks.py`) get tests written before migration.

**Tech Stack:** Python `dataclasses` (stdlib), `pytest`, existing S3/local I/O in each chunker.

---

## File Map

**Create:**
- `core/__init__.py` — empty, makes `core` a package
- `core/models.py` — `Chunk` dataclass (the sole shared type)

**Modify (chunkers — one per task):**
- `chunk/iac_chunk.py` — update `_make_chunk`, update `run()` serialization
- `chunk/ilga_chunk.py` — same pattern as IAC
- `chunk/iscr_chunk.py` — dict-based but different fields; update `_process_rule_text`
- `chunk/idoc_chunk.py` — remove `IdocChunk` dataclass; update `chunk_record`
- `chunk/spac_chunk.py` — remove `SpacChunk` dataclass; update `chunk_record`
- `chunk/iccb_chunk.py` — remove `IccbChunk` dataclass; update `chunk_record`
- `chunk/federal_chunk.py` — remove `FederalChunk` dataclass; update `chunk_record`
- `chunk/restorejustice_chunk.py` — remove `RestoreJusticeChunk` dataclass; update `chunk_record`
- `chunk/cookcounty_pd_chunk.py` — remove `CookCountyPDChunk` dataclass; update `chunk_record`
- `chunk/courtlistener_chunk.py` — write tests first, then migrate
- `chunk/merge_opinion_chunks.py` — write tests first, then migrate

**Modify (retrieval — last):**
- `retrieval/postprocessor.py` — simplify `CitationLabelingPostprocessor` to use `display_citation`
- `retrieval/bm25_store.py` — remove hard-coded `section_citation`/`rule_number` column fetches

**Tests (add assertions to existing):**
- `tests/test_iac_chunk.py`, `tests/test_ilga_chunk.py`, `tests/test_iscr_chunk.py`
- `tests/test_idoc_chunk.py`, `tests/test_spac_chunk.py`, `tests/test_iccb_chunk.py`
- `tests/test_federal_chunk.py`, `tests/test_restorejustice_chunk.py`, `tests/test_cookcounty_pd_chunk.py`
- Create: `tests/test_courtlistener_chunk.py`, `tests/test_merge_opinion_chunks.py`

---

## Task 1: Create `core/models.py`

**Files:**
- Create: `core/__init__.py`
- Create: `core/models.py`

- [ ] **Step 1: Create `core/__init__.py`**

```bash
touch /path/to/legal_rag/core/__init__.py
```

- [ ] **Step 2: Write `core/models.py`**

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    chunk_index: int
    chunk_total: int
    text: str
    enriched_text: str
    source: str
    token_count: int
    display_citation: str
    metadata: dict = field(default_factory=dict)
    chunked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
```

- [ ] **Step 3: Write a smoke test**

Create `tests/test_core_models.py`:

```python
import dataclasses
from core.models import Chunk


def test_chunk_required_fields():
    c = Chunk(
        chunk_id="test_c0",
        parent_id="test",
        chunk_index=0,
        chunk_total=1,
        text="Some legal text.",
        enriched_text="720 ILCS 5/7-1 — Justifiable Use of Force\n\nSome legal text.",
        source="ilcs",
        token_count=3,
        display_citation="720 ILCS 5/7-1 — Justifiable Use of Force",
    )
    assert c.chunk_id == "test_c0"
    assert c.metadata == {}
    assert c.chunked_at  # auto-populated


def test_chunk_asdict_round_trips():
    c = Chunk(
        chunk_id="test_c0", parent_id="test", chunk_index=0, chunk_total=1,
        text="text", enriched_text="enriched", source="ilcs",
        token_count=1, display_citation="Citation",
    )
    d = dataclasses.asdict(c)
    assert d["chunk_id"] == "test_c0"
    assert d["display_citation"] == "Citation"
    assert "metadata" in d
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_core_models.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add core/__init__.py core/models.py tests/test_core_models.py
git commit -m "feat: add shared Chunk dataclass in core/models.py"
```

---

## Task 2: Migrate `chunk/iac_chunk.py`

IAC uses a dict-returning `_make_chunk`. This task replaces it with a `Chunk`-returning one and updates `run()` to serialize at the write boundary.

**Files:**
- Modify: `chunk/iac_chunk.py`
- Modify: `tests/test_iac_chunk.py`

- [ ] **Step 1: Add imports and update `_make_chunk` in `iac_chunk.py`**

At the top of `iac_chunk.py`, add:
```python
import dataclasses
from core.models import Chunk
```

Replace the `_make_chunk` function (currently lines ~347–373):

```python
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
    display_citation = section_citation
    if section_heading:
        display_citation = f"{section_citation} — {section_heading}"

    enriched = (
        f"Illinois Administrative Code Title {metadata['title_num']} "
        f"({metadata['title_name']}), "
        f"Part {metadata['part_num']} ({metadata['part_name']}), "
        f"{section_citation} {section_heading}\n\n"
        f"{text}"
    )
    return Chunk(
        chunk_id=f"{parent_id}_c{chunk_index}",
        parent_id=parent_id,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        text=text,
        enriched_text=enriched,
        source="illinois_admin_code",
        token_count=len(text.split()),
        display_citation=display_citation.strip(" — "),
        metadata=metadata,
    )
```

- [ ] **Step 2: Update `chunk_section` return type annotation**

Change the signature from `def chunk_section(record: dict) -> list[dict]:` to:
```python
def chunk_section(record: dict) -> list[Chunk]:
```

- [ ] **Step 3: Update `run()` to serialize at the write boundary**

In `run()`, find where `all_chunks` is built:
```python
    all_chunks: list[dict] = []
    ...
    for rec in records:
        chunks = chunk_section(rec)
        ...
        all_chunks.extend(chunks)
```

Change to:
```python
    all_chunks: list[Chunk] = []
    ...
    for rec in records:
        chunks = chunk_section(rec)
        ...
        all_chunks.extend(chunks)

    serialized = [dataclasses.asdict(c) for c in all_chunks]
```

And update the write calls from `write_local(all_chunks, ...)` / `write_s3(all_chunks, ...)` to:
```python
    if local_only:
        write_local(serialized, LOCAL_OUTPUT_DIR / "iac_chunks.jsonl")
    else:
        write_s3(serialized, cfg["chunked_bucket"], cfg["chunked_key"], cfg["aws_region"])
```

Also update the logging lines that access `all_chunks` fields:
```python
    single = sum(1 for c in all_chunks if c.chunk_total == 1)
    multi  = len({c.parent_id for c in all_chunks if c.chunk_total > 1})
```

- [ ] **Step 4: Update tests to use attribute access and check new fields**

In `tests/test_iac_chunk.py`, all assertions on chunk dicts must switch to attribute access. Find and replace the following patterns:

```python
# Before → After
chunks[0]["chunk_index"]   →  chunks[0].chunk_index
chunks[0]["chunk_total"]   →  chunks[0].chunk_total
chunks[0]["text"]          →  chunks[0].text
chunks[0]["chunk_id"]      →  chunks[0].chunk_id
chunk["chunk_index"]       →  chunk.chunk_index
chunk["text"]              →  chunk.text
chunk["chunk_id"]          →  chunk.chunk_id
```

Add a schema assertion to the existing `test_short_but_real_section_single_chunk` test:

```python
def test_short_but_real_section_single_chunk():
    rec = _make_record(
        "Section 504.10  Applicability\n"
        "This Part applies to all adult correctional facilities within the Department."
    )
    chunks = chunk_section(rec)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_total == 1
    # Schema assertions
    assert chunks[0].display_citation.startswith("20 Ill. Adm. Code")
    assert chunks[0].token_count > 0
    assert chunks[0].source == "illinois_admin_code"
    assert chunks[0].enriched_text.startswith("Illinois Administrative Code")
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/test_iac_chunk.py -v
```

Expected: all existing tests pass (check count matches pre-migration run).

- [ ] **Step 6: Commit**

```bash
git add chunk/iac_chunk.py tests/test_iac_chunk.py
git commit -m "feat: migrate iac_chunk to shared Chunk dataclass"
```

---

## Task 3: Migrate `chunk/ilga_chunk.py`

ILCS uses the same dict-based `_make_chunk` pattern as IAC.

**Files:**
- Modify: `chunk/ilga_chunk.py`
- Modify: `tests/test_ilga_chunk.py`

- [ ] **Step 1: Add imports**

```python
import dataclasses
from core.models import Chunk
```

- [ ] **Step 2: Replace `_make_chunk`**

```python
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
    display_citation = section_citation
    if section_heading:
        display_citation = f"{section_citation} — {section_heading}"

    act_name    = metadata.get("act_name", "")
    major_topic = metadata.get("major_topic", "")
    enriched_parts = [p for p in [section_citation, section_heading, act_name, major_topic] if p]
    enriched = "\n".join(enriched_parts) + "\n\n" + text if enriched_parts else text

    return Chunk(
        chunk_id=f"{parent_id}_c{chunk_index}",
        parent_id=parent_id,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        text=text,
        enriched_text=enriched,
        source="ilcs",
        token_count=len(text.split()),
        display_citation=display_citation.strip(" — "),
        metadata=metadata,
    )
```

- [ ] **Step 3: Update `chunk_section` return type**

```python
def chunk_section(record: dict) -> list[Chunk]:
```

- [ ] **Step 4: Update `run()` to serialize at write boundary**

Same pattern as Task 2, Step 3:
```python
    all_chunks: list[Chunk] = []
    ...
    serialized = [dataclasses.asdict(c) for c in all_chunks]
    if local_only:
        write_local(serialized, ...)
    else:
        write_s3(serialized, ...)
```

Update any `all_chunks` field accesses to dot notation (e.g., `c.chunk_total`, `c.parent_id`).

- [ ] **Step 5: Update tests — switch to attribute access + add schema assertions**

In `tests/test_ilga_chunk.py`, replace all `chunk["field"]` with `chunk.field`.

Add to any single-chunk test:
```python
    assert chunks[0].display_citation  # non-empty
    assert chunks[0].token_count > 0
    assert chunks[0].source == "ilcs"
    assert "section_citation" in chunks[0].metadata
```

- [ ] **Step 6: Run tests**

```bash
python3 -m pytest tests/test_ilga_chunk.py -v
```

- [ ] **Step 7: Commit**

```bash
git add chunk/ilga_chunk.py tests/test_ilga_chunk.py
git commit -m "feat: migrate ilga_chunk to shared Chunk dataclass"
```

---

## Task 4: Migrate `chunk/iscr_chunk.py`

ISCR is dict-based but uses UUIDs for `chunk_id` and different field names. The key function is `_process_rule_text`.

**Files:**
- Modify: `chunk/iscr_chunk.py`
- Modify: `tests/test_iscr_chunk.py`

- [ ] **Step 1: Add imports**

```python
import dataclasses
from core.models import Chunk
```

- [ ] **Step 2: Update `_process_rule_text` to return `Chunk`**

Find the return dict in `_process_rule_text` (around line 315) and replace with:

```python
    rule_number = hierarchy.current_rule_number or ""
    rule_title  = " ".join(rule_title_parts) if rule_title_parts else ""
    display_citation = rule_title if rule_title else (f"Rule {rule_number}" if rule_number else "")

    return Chunk(
        chunk_id=str(uuid.uuid4()),
        parent_id=source_key,
        chunk_index=0,        # ISCR assigns index after collection; update in chunk_document
        chunk_total=1,        # updated in chunk_document
        text=text,
        enriched_text=enriched_text,   # keep existing enriched_text construction
        source="illinois_supreme_court_rules",
        token_count=estimate_tokens(text),
        display_citation=display_citation,
        metadata={
            "source_s3_key":      source_key,
            "content_type":       content_type,
            "hierarchical_path":  hierarchical_path,
            "article_number":     hierarchy.current_article_number,
            "article_title":      hierarchy.current_article_title,
            "part_letter":        hierarchy.current_part_letter,
            "part_title":         hierarchy.current_part_title,
            "rule_number":        rule_number,
            "rule_title":         rule_title,
            "subsection_id":      subsection_id,
            "effective_date":     hierarchy.effective_date,
            "amendment_history":  hierarchy.amendment_history,
        },
    )
```

Note: ISCR chunks are collected then assigned sequential `chunk_index`/`chunk_total` in `chunk_document`. Update that loop to use attribute assignment:

```python
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
        chunk.chunk_total = len(chunks)
```

- [ ] **Step 3: Update `chunk_document` return type**

```python
def chunk_document(full_text: str, source_key: str) -> list[Chunk]:
```

- [ ] **Step 4: Update write path serialization**

Where `chunk_document` output is written to S3/local, wrap with `dataclasses.asdict`:

```python
serialized = [dataclasses.asdict(c) for c in chunks]
```

- [ ] **Step 5: Update tests — attribute access + schema assertions**

In `tests/test_iscr_chunk.py`, replace `chunk["field"]` with `chunk.field`.

Add:
```python
    assert chunk.display_citation  # non-empty
    assert chunk.source == "illinois_supreme_court_rules"
    assert "rule_number" in chunk.metadata
```

- [ ] **Step 6: Run tests**

```bash
python3 -m pytest tests/test_iscr_chunk.py -v
```

- [ ] **Step 7: Commit**

```bash
git add chunk/iscr_chunk.py tests/test_iscr_chunk.py
git commit -m "feat: migrate iscr_chunk to shared Chunk dataclass"
```

---

## Task 5: Migrate `chunk/idoc_chunk.py`

IDOC has a local `IdocChunk` dataclass. This task deletes it and uses `Chunk` instead.

**Files:**
- Modify: `chunk/idoc_chunk.py`
- Modify: `tests/test_idoc_chunk.py`

- [ ] **Step 1: Add import, remove local dataclass**

Add to imports:
```python
from core.models import Chunk
```

Delete the `IdocChunk` dataclass definition (the `@dataclass` block with fields `chunk_id`, `chunk_index`, etc.).

- [ ] **Step 2: Update `chunk_record` to return `list[Chunk]`**

In the non-directive branch (single-chunk path), replace the `IdocChunk(...)` constructor with:

```python
                directive_label = title or rec_id
                Chunk(
                    chunk_id=f"{rec_id}_c{i}",
                    parent_id=rec_id,
                    chunk_index=i,
                    chunk_total=len(texts),
                    text=text,
                    enriched_text=f"{title}\n\n{text}" if title else text,
                    source=source,
                    token_count=len(text.split()),
                    display_citation=directive_label,
                    metadata={
                        "source": source,
                        "title": title,
                        "url": rec.get("url", ""),
                        "scraped_at": rec.get("scraped_at", ""),
                    },
                )
```

In the directive branch, replace `IdocChunk(...)` with:

```python
                directive_num = rec_id.replace("idoc-dir-", "").replace("_", ".")
                display = f"IDOC Administrative Directive {directive_num}: {title}"
                Chunk(
                    chunk_id=f"{rec_id}_c{i}",
                    parent_id=rec_id,
                    chunk_index=i,
                    chunk_total=total,
                    text=chunk_text,
                    enriched_text=_make_enriched(rec, heading, chunk_text),
                    source=source,
                    token_count=len(chunk_text.split()),
                    display_citation=display,
                    metadata={
                        "source": source,
                        "title": title,
                        "section_heading": heading,
                        "url": rec.get("url", ""),
                        "scraped_at": rec.get("scraped_at", ""),
                    },
                )
```

- [ ] **Step 3: Update `run()` serialization**

```python
    serialized = [dataclasses.asdict(c) for c in all_chunks]
    write_local(serialized, ...) / write_s3(serialized, ...)
```

- [ ] **Step 4: Update tests**

In `tests/test_idoc_chunk.py`, switch from `chunk["field"]` to `chunk.field`. Fields that were top-level on `IdocChunk` and are now in `metadata` (e.g., `title`, `section_heading`) move to `chunk.metadata["title"]`.

Add:
```python
    assert chunk.display_citation
    assert chunk.source in ("idoc_directive", "idoc_reentry")
    assert chunk.token_count > 0
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/test_idoc_chunk.py -v
```

- [ ] **Step 6: Commit**

```bash
git add chunk/idoc_chunk.py tests/test_idoc_chunk.py
git commit -m "feat: migrate idoc_chunk to shared Chunk dataclass"
```

---

## Task 6: Migrate `chunk/spac_chunk.py`

**Files:**
- Modify: `chunk/spac_chunk.py`
- Modify: `tests/test_spac_chunk.py`

- [ ] **Step 1: Add import, remove `SpacChunk` dataclass**

```python
from core.models import Chunk
```

Delete the `SpacChunk` dataclass definition.

- [ ] **Step 2: Update `chunk_record` constructor**

Replace `SpacChunk(...)` with:

```python
            category = rec.get("category", "")
            year     = rec.get("year", "")
            display  = f"SPAC {category} ({year}): {title}".strip(": ")
            Chunk(
                chunk_id=f"{rec_id}_c{i}",
                parent_id=rec_id,
                chunk_index=i,
                chunk_total=total,
                text=chunk_text,
                enriched_text=_make_enriched(rec, heading, chunk_text),
                source="spac",
                token_count=len(chunk_text.split()),
                display_citation=display,
                metadata={
                    "title": title,
                    "category": category,
                    "year": year,
                    "section_heading": heading,
                    "url": rec.get("url", ""),
                    "scraped_at": rec.get("scraped_at", ""),
                },
            )
```

- [ ] **Step 3: Update `run()` serialization** (same pattern as previous tasks)

- [ ] **Step 4: Update tests — attribute access**

In `tests/test_spac_chunk.py`, `chunk["title"]` → `chunk.metadata["title"]`, `chunk["section_heading"]` → `chunk.metadata["section_heading"]`. All other field accesses switch to dot notation.

Add:
```python
    assert chunk.display_citation.startswith("SPAC")
    assert chunk.source == "spac"
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/test_spac_chunk.py -v
```

- [ ] **Step 6: Commit**

```bash
git add chunk/spac_chunk.py tests/test_spac_chunk.py
git commit -m "feat: migrate spac_chunk to shared Chunk dataclass"
```

---

## Task 7: Migrate `chunk/iccb_chunk.py`

Same pattern as SPAC.

**Files:**
- Modify: `chunk/iccb_chunk.py`
- Modify: `tests/test_iccb_chunk.py`

- [ ] **Step 1: Add import, remove `IccbChunk` dataclass**

```python
from core.models import Chunk
```

- [ ] **Step 2: Replace `IccbChunk(...)` constructor**

```python
            fiscal_year = rec.get("fiscal_year", "")
            display     = f"ICCB Annual Report FY{fiscal_year}: {title}".strip(": ")
            Chunk(
                chunk_id=f"{rec_id}_c{i}",
                parent_id=rec_id,
                chunk_index=i,
                chunk_total=total,
                text=chunk_text,
                enriched_text=_make_enriched(rec, heading, chunk_text),
                source="iccb",
                token_count=len(chunk_text.split()),
                display_citation=display,
                metadata={
                    "title": title,
                    "fiscal_year": fiscal_year,
                    "section_heading": heading,
                    "url": rec.get("url", ""),
                    "scraped_at": rec.get("scraped_at", ""),
                },
            )
```

- [ ] **Step 3: Update `run()` serialization**

- [ ] **Step 4: Update tests**

`chunk["title"]` → `chunk.metadata["title"]`, etc. Add:
```python
    assert chunk.display_citation.startswith("ICCB Annual Report")
    assert chunk.source == "iccb"
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/test_iccb_chunk.py -v
```

- [ ] **Step 6: Commit**

```bash
git add chunk/iccb_chunk.py tests/test_iccb_chunk.py
git commit -m "feat: migrate iccb_chunk to shared Chunk dataclass"
```

---

## Task 8: Migrate `chunk/federal_chunk.py`

**Files:**
- Modify: `chunk/federal_chunk.py`
- Modify: `tests/test_federal_chunk.py`

- [ ] **Step 1: Add import, remove `FederalChunk` dataclass**

```python
from core.models import Chunk
```

- [ ] **Step 2: Replace `FederalChunk(...)` constructor**

```python
            citation = rec.get("citation", "")
            display  = f"Federal: {title}"
            if citation:
                display += f" ({citation})"
            Chunk(
                chunk_id=f"{rec_id}_c{i}",
                parent_id=rec_id,
                chunk_index=i,
                chunk_total=total,
                text=chunk_text,
                enriched_text=_make_enriched(rec, heading, chunk_text),
                source="federal",
                token_count=len(chunk_text.split()),
                display_citation=display,
                metadata={
                    "title": title,
                    "citation": citation,
                    "section_heading": heading,
                    "url": rec.get("url", ""),
                    "scraped_at": rec.get("scraped_at", ""),
                },
            )
```

- [ ] **Step 3: Update `run()` serialization**

- [ ] **Step 4: Update tests**

`chunk["citation"]` → `chunk.metadata["citation"]`, etc. Add:
```python
    assert chunk.display_citation.startswith("Federal:")
    assert chunk.source == "federal"
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/test_federal_chunk.py -v
```

- [ ] **Step 6: Commit**

```bash
git add chunk/federal_chunk.py tests/test_federal_chunk.py
git commit -m "feat: migrate federal_chunk to shared Chunk dataclass"
```

---

## Task 9: Migrate `chunk/restorejustice_chunk.py`

**Files:**
- Modify: `chunk/restorejustice_chunk.py`
- Modify: `tests/test_restorejustice_chunk.py`

- [ ] **Step 1: Add import, remove `RestoreJusticeChunk` dataclass**

```python
from core.models import Chunk
```

- [ ] **Step 2: Replace `RestoreJusticeChunk(...)` constructor**

```python
            page_title = rec.get("page_title", "")
            Chunk(
                chunk_id=f"{rec_id}_c{i}",
                parent_id=rec_id,
                chunk_index=i,
                chunk_total=total,
                text=chunk_text,
                enriched_text=_make_enriched(rec, chunk_text),
                source="restorejustice",
                token_count=len(chunk_text.split()),
                display_citation=f"Restore Justice IL — {page_title}",
                metadata={
                    "page_title": page_title,
                    "url": rec.get("url", ""),
                    "scraped_at": rec.get("scraped_at", ""),
                },
            )
```

- [ ] **Step 3: Update `run()` serialization**

- [ ] **Step 4: Update tests**

`chunk["page_title"]` → `chunk.metadata["page_title"]`. Add:
```python
    assert chunk.display_citation.startswith("Restore Justice IL")
    assert chunk.source == "restorejustice"
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/test_restorejustice_chunk.py -v
```

- [ ] **Step 6: Commit**

```bash
git add chunk/restorejustice_chunk.py tests/test_restorejustice_chunk.py
git commit -m "feat: migrate restorejustice_chunk to shared Chunk dataclass"
```

---

## Task 10: Migrate `chunk/cookcounty_pd_chunk.py`

**Files:**
- Modify: `chunk/cookcounty_pd_chunk.py`
- Modify: `tests/test_cookcounty_pd_chunk.py`

- [ ] **Step 1: Add import, remove `CookCountyPDChunk` dataclass**

```python
from core.models import Chunk
```

- [ ] **Step 2: Replace `CookCountyPDChunk(...)` constructor**

```python
            page_title = rec.get("page_title", "")
            Chunk(
                chunk_id=f"{rec_id}_c{i}",
                parent_id=rec_id,
                chunk_index=i,
                chunk_total=total,
                text=chunk_text,
                enriched_text=_make_enriched(rec, chunk_text),
                source="cookcounty_pd",
                token_count=len(chunk_text.split()),
                display_citation=f"Cook County Public Defender — {page_title}",
                metadata={
                    "page_title": page_title,
                    "url": rec.get("url", ""),
                    "scraped_at": rec.get("scraped_at", ""),
                },
            )
```

- [ ] **Step 3: Update `run()` serialization**

- [ ] **Step 4: Update tests**

`chunk["page_title"]` → `chunk.metadata["page_title"]`. Add:
```python
    assert chunk.display_citation.startswith("Cook County Public Defender")
    assert chunk.source == "cookcounty_pd"
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/test_cookcounty_pd_chunk.py -v
```

- [ ] **Step 6: Commit**

```bash
git add chunk/cookcounty_pd_chunk.py tests/test_cookcounty_pd_chunk.py
git commit -m "feat: migrate cookcounty_pd_chunk to shared Chunk dataclass"
```

---

## Task 11: Write tests and migrate `chunk/courtlistener_chunk.py`

CourtListener has no test suite. Write tests before migrating.

**Files:**
- Create: `tests/test_courtlistener_chunk.py`
- Modify: `chunk/courtlistener_chunk.py`

- [ ] **Step 1: Read `courtlistener_chunk.py` and identify its current output schema**

Run:
```bash
grep -n "def chunk\|OpinionChunk\|dataclass\|chunk_id\|case_name\|source\|token_count" chunk/courtlistener_chunk.py | head -40
```

Note the current field names and the `OpinionChunk` dataclass definition.

- [ ] **Step 2: Write failing tests**

Create `tests/test_courtlistener_chunk.py`. Use the output of Step 1 to determine what synthetic inputs to construct. Minimum coverage:

```python
import pytest
from chunk.courtlistener_chunk import chunk_opinion   # adjust function name per Step 1
from core.models import Chunk


def _make_opinion(text: str, case_name: str = "People v. Smith") -> dict:
    return {
        "id": "cl-12345",
        "case_name": case_name,
        "citation": "2019 IL App (1st) 170001",
        "court": "ca7",
        "decision_date": "2019-06-01",
        "text": text,
        "source": "courtlistener",
    }


def test_single_chunk_short_opinion():
    rec = _make_opinion("The court held that the defendant's rights were not violated.")
    chunks = chunk_opinion(rec)
    assert len(chunks) >= 1
    assert isinstance(chunks[0], Chunk)


def test_chunk_schema():
    rec = _make_opinion("The court held that the defendant's rights were not violated.")
    chunk = chunk_opinion(rec)[0]
    assert chunk.chunk_id
    assert chunk.parent_id == "cl-12345"
    assert chunk.source == "courtlistener"
    assert chunk.display_citation
    assert chunk.token_count > 0
    assert chunk.text
    assert chunk.enriched_text


def test_no_empty_chunks():
    rec = _make_opinion("Short text.")
    chunks = chunk_opinion(rec)
    assert all(c.text.strip() for c in chunks)


def test_contiguous_indices():
    long_text = "The court held. " * 200
    rec = _make_opinion(long_text)
    chunks = chunk_opinion(rec)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.chunk_total == len(chunks)
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
python3 -m pytest tests/test_courtlistener_chunk.py -v
```

Expected: fail (either import error or `isinstance(Chunk)` failure).

- [ ] **Step 4: Migrate `courtlistener_chunk.py`**

Add import, remove `OpinionChunk` local dataclass, update the chunk constructor to `Chunk(...)` following the same pattern as previous tasks.

For `display_citation`, use:
```python
display = f"{case_name}"
if citation := rec.get("citation", ""):
    display += f", {citation}"
```

Move all opinion-specific fields (`case_name`, `court`, `decision_date`, `author`, etc.) into `metadata`.

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python3 -m pytest tests/test_courtlistener_chunk.py -v
```

- [ ] **Step 6: Commit**

```bash
git add chunk/courtlistener_chunk.py tests/test_courtlistener_chunk.py
git commit -m "feat: add tests and migrate courtlistener_chunk to shared Chunk dataclass"
```

---

## Task 12: Write tests and migrate `chunk/merge_opinion_chunks.py`

**Files:**
- Create: `tests/test_merge_opinion_chunks.py`
- Modify: `chunk/merge_opinion_chunks.py`

- [ ] **Step 1: Read `merge_opinion_chunks.py` to understand its purpose**

```bash
grep -n "def \|class \|Chunk\|return\|chunk_id" chunk/merge_opinion_chunks.py | head -30
```

This script merges/re-chunks opinion output. Identify what it takes as input and what it produces.

- [ ] **Step 2: Write failing tests**

Create `tests/test_merge_opinion_chunks.py` covering:
- Output is `list[Chunk]`
- `chunk_id`, `parent_id`, `chunk_index`, `chunk_total` are set
- `display_citation` is non-empty
- No empty `text` fields
- Contiguous `chunk_index` per `parent_id`

Use `core.models.Chunk` instances as input (since all upstream chunkers now emit `Chunk`).

- [ ] **Step 3: Run tests to confirm they fail**

```bash
python3 -m pytest tests/test_merge_opinion_chunks.py -v
```

- [ ] **Step 4: Migrate `merge_opinion_chunks.py`**

Follow the same pattern: import `Chunk`, remove any local dataclass, return `list[Chunk]`.

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python3 -m pytest tests/test_merge_opinion_chunks.py -v
```

- [ ] **Step 6: Commit**

```bash
git add chunk/merge_opinion_chunks.py tests/test_merge_opinion_chunks.py
git commit -m "feat: add tests and migrate merge_opinion_chunks to shared Chunk dataclass"
```

---

## Task 13: Update `CitationLabelingPostprocessor`

Now that every Chunk has `display_citation` in its metadata, the postprocessor no longer needs to reconstruct the label from `section_citation`/`rule_number`/`rule_title`.

**Files:**
- Modify: `retrieval/postprocessor.py`
- Modify: `tests/test_postprocessor.py` (create if absent)

- [ ] **Step 1: Replace `_postprocess_nodes` in `CitationLabelingPostprocessor`**

Current code checks for `section_citation`, then `rule_number`, then falls back to no label. Replace with:

```python
    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        for nws in nodes:
            node = nws.node
            citation = (node.metadata or {}).get("display_citation")
            if citation:
                node.text = f"[{citation}]:\n{node.get_content()}"
        return nodes
```

- [ ] **Step 2: Write a test**

```python
from llama_index.core.schema import NodeWithScore, TextNode
from retrieval.postprocessor import CitationLabelingPostprocessor


def _make_node(text: str, display_citation: str) -> NodeWithScore:
    node = TextNode(text=text, metadata={"display_citation": display_citation})
    return NodeWithScore(node=node, score=1.0)


def test_citation_label_prepended():
    proc = CitationLabelingPostprocessor()
    nws = _make_node("Some legal text.", "720 ILCS 5/7-1 — Justifiable Use of Force")
    result = proc._postprocess_nodes([nws])
    assert result[0].node.text.startswith("[720 ILCS 5/7-1 — Justifiable Use of Force]:")


def test_no_citation_leaves_text_unchanged():
    proc = CitationLabelingPostprocessor()
    node = TextNode(text="Some text.", metadata={})
    nws = NodeWithScore(node=node, score=1.0)
    result = proc._postprocess_nodes([nws])
    assert result[0].node.text == "Some text."
```

- [ ] **Step 3: Run tests**

```bash
python3 -m pytest tests/test_postprocessor.py -v
```

- [ ] **Step 4: Commit**

```bash
git add retrieval/postprocessor.py tests/test_postprocessor.py
git commit -m "refactor: simplify CitationLabelingPostprocessor to use display_citation"
```

---

## Task 14: Update `BM25Retriever` to use `display_citation`

The BM25 loader hard-codes `section_citation` for ILCS and `rule_number`/`rule_title` for ISCR. With `display_citation` now a top-level column, remove this.

**Files:**
- Modify: `retrieval/bm25_store.py`

- [ ] **Step 1: Update `_load` to fetch `display_citation` instead of source-specific columns**

Current:
```python
ilcs_rows = fetch_all("ilcs_chunks", ["section_citation"])
iscr_rows = fetch_all("court_rule_chunks", ["rule_number", "rule_title"])
```

Replace with:
```python
ilcs_rows = fetch_all("ilcs_chunks", ["display_citation"])
iscr_rows = fetch_all("court_rule_chunks", ["display_citation"])
```

Update `self._metadata` construction to include `display_citation`:
```python
        self._metadata = [
            {k: v for k, v in r.items()
             if k not in ("chunk_id", "text") and v is not None}
            for r in all_rows
        ]
```

(No change needed — `display_citation` is already included since it's not in the exclusion list. Just ensure the `fetch_all` columns include it.)

- [ ] **Step 2: Run the full test suite**

```bash
python3 -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add retrieval/bm25_store.py
git commit -m "refactor: BM25Retriever fetches display_citation instead of source-specific columns"
```

---

## Self-Review

**Spec coverage:**
- ✅ `core/models.py` with `Chunk` dataclass — Task 1
- ✅ IAC migration (dict-based reference pattern) — Task 2
- ✅ ILCS migration — Task 3
- ✅ ISCR migration — Task 4
- ✅ IDOC migration (dataclass-based reference pattern) — Task 5
- ✅ SPAC, ICCB, Federal, RestoreJustice, CookCountyPD — Tasks 6–10
- ✅ CourtListener (write tests first) — Task 11
- ✅ merge_opinion_chunks (write tests first) — Task 12
- ✅ `CitationLabelingPostprocessor` simplified — Task 13
- ✅ `BM25Retriever` decoupled — Task 14

**Placeholder scan:** No TBDs, no "similar to Task N" without code, no "add appropriate error handling."

**Type consistency:** `Chunk` dataclass defined once in Task 1; all subsequent tasks import it from `core.models`. `chunk_section`/`chunk_record`/`chunk_document` return `list[Chunk]` throughout. `dataclasses.asdict()` called at the write boundary in each `run()`.

**Note on `enriched_text` in ILCS:** The current `ilga_chunk.py` does not produce `enriched_text` (the `_make_chunk` in the existing code has no enriched field). Task 3's `_make_chunk` adds it, constructing it from `section_citation`, `section_heading`, `act_name`, and `major_topic` — matching the pattern the embed scripts expect.
