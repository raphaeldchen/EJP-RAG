# Chunking Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pytest suite that catches chunking bugs in `ilga_chunk.py` and `iscr_chunk.py`, then fix two confirmed bugs in the ISCR chunker.

**Architecture:** Session-scoped pytest fixtures pull ISCR PDFs from S3 once and cache as `.txt` files; ILCS uses the local `ilcs_corpus.jsonl`. All chunking functions are pure text→list[dict], so every test runs without mocking. Bug fixes land in `chunk/iscr_chunk.py` alongside the tests that verify them.

**Tech Stack:** `pytest`, `boto3`, `pdfplumber`, `python-dotenv`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `pytest.ini` | Adds repo root to `pythonpath` so `chunk.*` imports resolve |
| Create | `tests/__init__.py` | Empty — marks tests as a package |
| Create | `tests/conftest.py` | Session-scoped fixtures: `iscr_texts`, `ilcs_records`, `ilcs_chunks`, `iscr_chunks` |
| Create | `tests/test_ilga_chunk.py` | 7 ILCS tests (2 regression, 5 property) |
| Create | `tests/test_iscr_chunk.py` | 10 ISCR tests (2 regression, 8 property) |
| Modify | `chunk/iscr_chunk.py` | Fix 1: letter-only split + intro carry-forward; Fix 2: strip `[PAGE N]` markers |
| Modify | `.gitignore` | Add `tests/fixtures/` |

---

## Task 1: Scaffold — pytest config, gitignore, directory structure

**Files:**
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create `pytest.ini`**

```ini
[pytest]
pythonpath = .
```

- [ ] **Step 2: Create `tests/__init__.py`**

Empty file. Run:
```bash
touch tests/__init__.py
```

- [ ] **Step 3: Add fixture cache to `.gitignore`**

Append to `.gitignore`:
```
# Test fixture cache (populated on first run)
tests/fixtures/
```

- [ ] **Step 4: Create fixture cache directories**

```bash
mkdir -p tests/fixtures/iscr
```

- [ ] **Step 5: Verify pytest resolves imports**

```bash
python3 -c "from chunk.ilga_chunk import chunk_section; print('ok')"
python3 -c "from chunk.iscr_chunk import chunk_document; print('ok')"
```

Expected: `ok` for both.

- [ ] **Step 6: Commit**

```bash
git add pytest.ini tests/__init__.py .gitignore
git commit -m "test: scaffold pytest config and test directory"
```

---

## Task 2: `conftest.py` — shared session fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import io
import json
from pathlib import Path

import boto3
import pdfplumber
import pytest
from dotenv import load_dotenv
import os

load_dotenv()

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ISCR_FIXTURES_DIR = FIXTURES_DIR / "iscr"


@pytest.fixture(scope="session")
def iscr_texts():
    """Download ISCR PDFs from S3 once, extract text with pdfplumber, cache as .txt files."""
    ISCR_FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    bucket = os.environ["RAW_S3_BUCKET"]
    prefix = os.environ["SUPREME_COURT_RULES_S3_PREFIX"].rstrip("/")
    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    pdf_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".pdf"):
                pdf_keys.append(obj["Key"])

    texts = {}
    for key in pdf_keys:
        stem = Path(key).stem
        cache_path = ISCR_FIXTURES_DIR / f"{stem}.txt"
        if cache_path.exists():
            texts[stem] = cache_path.read_text(encoding="utf-8")
        else:
            response = s3.get_object(Bucket=bucket, Key=key)
            pdf_bytes = response["Body"].read()
            page_texts = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if text:
                        page_texts.append(f"[PAGE {i}]\n{text.strip()}")
            full_text = "\n\n".join(page_texts)
            cache_path.write_text(full_text, encoding="utf-8")
            texts[stem] = full_text

    return texts


@pytest.fixture(scope="session")
def ilcs_records():
    """Load all records from the local ilcs_corpus.jsonl."""
    corpus_path = Path(__file__).parent.parent / "ilcs_corpus.jsonl"
    records = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="session")
def ilcs_chunks(ilcs_records):
    """Run chunk_section() over all ilcs_records; return flat list of all chunks."""
    from chunk.ilga_chunk import chunk_section
    chunks = []
    for rec in ilcs_records:
        chunks.extend(chunk_section(rec))
    return chunks


@pytest.fixture(scope="session")
def iscr_chunks(iscr_texts):
    """Run chunk_document() over all ISCR texts; return flat list of all chunks."""
    from chunk.iscr_chunk import chunk_document
    chunks = []
    for stem, text in iscr_texts.items():
        chunks.extend(chunk_document(text, source_key=f"{stem}.pdf"))
    return chunks
```

- [ ] **Step 2: Verify fixtures load without error**

```bash
pytest tests/ --collect-only 2>&1 | head -20
```

Expected: collection shows 0 items (no tests yet), no import errors.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add session fixtures for ILCS and ISCR corpora"
```

---

## Task 3: ILCS regression tests

**Files:**
- Create: `tests/test_ilga_chunk.py` (regression section only for now)

- [ ] **Step 1: Write the two regression tests**

```python
import re
import pytest
from chunk.ilga_chunk import chunk_section, CHUNK_SIZE, MIN_CHUNK_SIZE


def test_5_915_chunk_is_self_contained(ilcs_records):
    """705 ILCS 405/5-915 must produce at least one chunk with key expungement terms."""
    record = next(
        (r for r in ilcs_records if "405/5-915" in r.get("section_citation", "")),
        None,
    )
    assert record is not None, "705 ILCS 405/5-915 record not found in corpus"
    chunks = chunk_section(record)
    assert chunks, "No chunks produced for 5-915"
    texts = [c["text"] for c in chunks]
    has_term = any(re.search(r"expunge|automatic", t, re.IGNORECASE) for t in texts)
    assert has_term, "No chunk contains 'expunge' or 'automatic' — section may be split incorrectly"
    for chunk in chunks:
        last_char = chunk["text"].rstrip()[-1]
        assert last_char in ".!?;:\"')", (
            f"Chunk {chunk['chunk_id']} ends mid-word (last char: {last_char!r})"
        )


def test_enumeration_not_severed():
    """A section long enough to trigger splitting must not orphan subsection markers."""
    filler = "Word " * 80  # ~400 chars per subsection
    text = (
        "The court shall consider all of the following factors:\n"
        f"(a) {filler}\n"
        f"(b) {filler}\n"
        f"(c) {filler}\n"
    )
    record = {
        "id": "test_enum",
        "text": text,
        "section_citation": "TEST/1-1",
        "section_heading": "Test Section",
        "section_num": "1-1",
        "article_name": "",
        "act_name": "",
        "act_id": "",
        "chapter_num": "",
        "chapter_name": "",
        "major_topic": "",
        "url": "",
        "scraped_at": "",
    }
    chunks = chunk_section(record)
    orphan_re = re.compile(r"^\s*\([a-z0-9]\)")
    for chunk in chunks:
        if chunk["chunk_index"] > 0:
            assert not orphan_re.match(chunk["text"]), (
                f"Chunk {chunk['chunk_id']} (index {chunk['chunk_index']}) "
                f"starts with orphaned subsection marker:\n{chunk['text'][:100]}"
            )
```

- [ ] **Step 2: Run to check current state**

```bash
pytest tests/test_ilga_chunk.py -v
```

Expected: `test_5_915_chunk_is_self_contained` should PASS (tells us 5-915 is a retrieval issue not chunking). `test_enumeration_not_severed` should PASS (ILCS chunker handles this correctly). If either FAILs, note which one and investigate before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ilga_chunk.py
git commit -m "test: ILCS regression tests for 5-915 and enumeration severance"
```

---

## Task 4: ILCS property tests

**Files:**
- Modify: `tests/test_ilga_chunk.py` (append property tests)

- [ ] **Step 1: Append property tests to `tests/test_ilga_chunk.py`**

```python
from collections import defaultdict


def test_no_orphaned_subsection_starts(ilcs_chunks):
    orphan_re = re.compile(r"^\s*\([a-z0-9]\)")
    failures = []
    for chunk in ilcs_chunks:
        if chunk["chunk_index"] > 0 and orphan_re.match(chunk["text"]):
            failures.append(chunk["chunk_id"])
    assert not failures, (
        f"{len(failures)} chunks start with orphaned subsection markers: {failures[:5]}"
    )


def test_chunk_index_contiguous(ilcs_chunks):
    by_parent = defaultdict(list)
    for chunk in ilcs_chunks:
        by_parent[chunk["parent_id"]].append(chunk["chunk_index"])
    failures = []
    for parent_id, indices in by_parent.items():
        expected = set(range(len(indices)))
        if set(indices) != expected:
            failures.append(f"{parent_id}: {sorted(indices)}")
    assert not failures, f"Non-contiguous chunk indices:\n" + "\n".join(failures[:5])


def test_chunk_total_accurate(ilcs_chunks):
    by_parent = defaultdict(list)
    for chunk in ilcs_chunks:
        by_parent[chunk["parent_id"]].append(chunk)
    failures = []
    for parent_id, siblings in by_parent.items():
        actual = len(siblings)
        for chunk in siblings:
            if chunk["chunk_total"] != actual:
                failures.append(
                    f"{chunk['chunk_id']}: chunk_total={chunk['chunk_total']} actual={actual}"
                )
    assert not failures, f"chunk_total mismatches:\n" + "\n".join(failures[:5])


def test_no_empty_chunks(ilcs_chunks):
    failures = [c["chunk_id"] for c in ilcs_chunks if len(c["text"]) < MIN_CHUNK_SIZE]
    assert not failures, f"{len(failures)} chunks below MIN_CHUNK_SIZE: {failures[:5]}"


def test_chunk_ids_unique(ilcs_chunks):
    ids = [c["chunk_id"] for c in ilcs_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids found in ILCS chunks"


def test_no_chunk_exceeds_max_size(ilcs_chunks):
    failures = [
        (c["chunk_id"], len(c["text"]))
        for c in ilcs_chunks
        if len(c["text"]) > CHUNK_SIZE
    ]
    assert not failures, (
        f"{len(failures)} chunks exceed CHUNK_SIZE={CHUNK_SIZE}: {failures[:5]}"
    )


def test_sentence_split_overlap(ilcs_chunks):
    """For sentence-split parents, chunk N+1 should contain the last sentence of chunk N."""
    by_parent = defaultdict(list)
    for chunk in ilcs_chunks:
        by_parent[chunk["parent_id"]].append(chunk)

    sentence_end_re = re.compile(r"(?<=[.!?])\s+")
    subsection_re = re.compile(r"^\([a-z0-9]\)", re.MULTILINE)
    failures = []

    for parent_id, siblings in by_parent.items():
        if len(siblings) <= 1:
            continue
        siblings_sorted = sorted(siblings, key=lambda c: c["chunk_index"])
        # Only check sentence-split parents (no subsection markers in any chunk)
        if any(subsection_re.search(c["text"]) for c in siblings_sorted):
            continue
        for i in range(len(siblings_sorted) - 1):
            chunk_a = siblings_sorted[i]
            chunk_b = siblings_sorted[i + 1]
            sentences = sentence_end_re.split(chunk_a["text"])
            last_sentence = sentences[-1].strip() if sentences else ""
            if len(last_sentence) < 15:
                continue
            if last_sentence not in chunk_b["text"]:
                failures.append(
                    f"No overlap: {chunk_a['chunk_id']} → {chunk_b['chunk_id']}"
                )

    assert not failures, (
        f"{len(failures)} consecutive chunk pairs have no overlap:\n"
        + "\n".join(failures[:5])
    )
```

- [ ] **Step 2: Run all ILCS tests**

```bash
pytest tests/test_ilga_chunk.py -v
```

Expected: all 7 tests PASS. If any property test fails, inspect the failing chunk IDs — they indicate real bugs. Fix them before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ilga_chunk.py
git commit -m "test: ILCS property tests — size, uniqueness, overlap, contiguity"
```

---

## Task 5: ISCR Fix 1 — write failing regression tests, then fix enumeration severance

**Files:**
- Create: `tests/test_iscr_chunk.py` (regression tests only for now)
- Modify: `chunk/iscr_chunk.py`

- [ ] **Step 1: Write the two ISCR regression tests**

Create `tests/test_iscr_chunk.py`:

```python
import re
import pytest
from chunk.iscr_chunk import chunk_document, should_split_rule, estimate_tokens


def test_rule_401_enumeration_intact(iscr_chunks):
    """The chunk containing Rule 401's (1) enumeration must also have its parent clause."""
    rule_401_chunks = [c for c in iscr_chunks if c.get("rule_number") == "401"]
    assert rule_401_chunks, "No chunks found for Rule 401 — check PDF extraction"

    enum_chunks = [c for c in rule_401_chunks if re.search(r"\(1\)", c["text"])]
    assert enum_chunks, (
        "No Rule 401 chunk contains '(1)' — enumeration may be split or missing"
    )
    for chunk in enum_chunks:
        assert re.search(r"shall|inform|advise|address|charged|offense", chunk["text"], re.IGNORECASE), (
            f"Rule 401 chunk with (1) lacks parent introductory clause:\n{chunk['text'][:400]}"
        )


def test_rule_subsection_has_parent_context(iscr_chunks):
    """No rule_subsection chunk should begin with a bare numeric enumeration marker."""
    orphan_re = re.compile(r"^\s*\(\d+\)")
    failures = []
    for chunk in iscr_chunks:
        if chunk.get("content_type") == "rule_subsection":
            if orphan_re.match(chunk.get("text", "")):
                failures.append(
                    f"{chunk['chunk_id']} (rule {chunk.get('rule_number')}):\n"
                    f"{chunk['text'][:150]}"
                )
    assert not failures, (
        f"{len(failures)} rule_subsection chunks start with orphaned numeric marker:\n"
        + "\n\n".join(failures[:3])
    )
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/test_iscr_chunk.py::test_rule_401_enumeration_intact tests/test_iscr_chunk.py::test_rule_subsection_has_parent_context -v
```

Expected: both FAIL (confirming the bug is present before the fix).

- [ ] **Step 3: Add `LETTER_SUBSECTION_RE` to `chunk/iscr_chunk.py`**

In `chunk/iscr_chunk.py`, after the existing `SUBSECTION_RE` definition (line 145), add:

```python
# Split rules only at letter-level subsections — numeric items (1)(2)(3) stay
# with their parent letter subsection to preserve introductory context.
LETTER_SUBSECTION_RE = re.compile(
    r"^\(([a-z])\)\s+",
    re.MULTILINE
)
```

- [ ] **Step 4: Rewrite `split_rule_into_subsections` in `chunk/iscr_chunk.py`**

Replace the entire `split_rule_into_subsections` function (lines 231–265) with:

```python
def split_rule_into_subsections(rule_text: str, rule_number: str) -> List[Tuple[str, str, str]]:
    """Split a rule at letter-subsection boundaries only.

    Numeric items like (1)(2)(3) stay with their parent letter subsection.
    Intro text (before the first letter subsection) is prepended to the first
    letter subsection so it is never emitted as a standalone orphan chunk.
    """
    sections = []
    lines = rule_text.split('\n')
    current_section: List[str] = []
    current_subsection: Optional[str] = None
    current_title: Optional[str] = None
    carry_forward: List[str] = []  # intro lines to prepend to the first subsection

    for line in lines:
        subsection_match = LETTER_SUBSECTION_RE.match(line.strip())
        if subsection_match and current_section:
            saved_text = '\n'.join(current_section).strip()
            if current_subsection is None:
                # This is intro text — carry forward rather than emit standalone
                carry_forward = current_section[:]
            else:
                prefix = '\n'.join(carry_forward).strip()
                full_text = (prefix + '\n' + saved_text).strip() if prefix else saved_text
                sections.append((current_subsection, current_title or f"Rule {rule_number}", full_text))
                carry_forward = []
            current_section = []
        if subsection_match:
            current_subsection = subsection_match.group(1)
            title_part = line.strip()[len(subsection_match.group(0)):].strip()
            current_title = f"Rule {rule_number}({current_subsection})"
            if title_part and len(title_part) < 100:
                current_title += f" — {title_part}"
            current_section.append(line)
        else:
            current_section.append(line)

    # Emit the final section
    if current_section:
        saved_text = '\n'.join(current_section).strip()
        prefix = '\n'.join(carry_forward).strip()
        full_text = (prefix + '\n' + saved_text).strip() if prefix else saved_text
        sections.append((
            current_subsection or "full",
            f"Rule {rule_number}" + (f"({current_subsection})" if current_subsection else ""),
            full_text,
        ))

    return sections
```

- [ ] **Step 5: Run regression tests to confirm they now pass**

```bash
pytest tests/test_iscr_chunk.py::test_rule_401_enumeration_intact tests/test_iscr_chunk.py::test_rule_subsection_has_parent_context -v
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add chunk/iscr_chunk.py tests/test_iscr_chunk.py
git commit -m "fix: ISCR chunker splits only on letter subsections, preserving numeric enumeration context"
```

---

## Task 6: ISCR Fix 2 — write failing page-marker test, then fix

**Files:**
- Modify: `tests/test_iscr_chunk.py` (append test)
- Modify: `chunk/iscr_chunk.py` (strip markers in `chunk_document`)

- [ ] **Step 1: Append `test_page_markers_stripped` to `tests/test_iscr_chunk.py`**

```python
def test_page_markers_stripped(iscr_chunks):
    """[PAGE N] markers injected by merge_pages_to_text must not bleed into chunk text."""
    page_marker_re = re.compile(r"\[PAGE \d+\]")
    failures = [
        chunk["chunk_id"]
        for chunk in iscr_chunks
        if page_marker_re.search(chunk.get("text", ""))
    ]
    assert not failures, (
        f"{len(failures)} chunks contain [PAGE N] markers: {failures[:5]}"
    )
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pytest tests/test_iscr_chunk.py::test_page_markers_stripped -v
```

Expected: FAIL — `[PAGE N]` markers are currently bleeding into chunk text.

If it unexpectedly PASSes, the PDF text for this corpus happens not to have page boundaries mid-rule. Still keep the test as a guard and continue.

- [ ] **Step 3: Strip `[PAGE N]` markers in `chunk_document`**

In `chunk/iscr_chunk.py`, at the very start of `chunk_document` (after the `chunks = []` and `hierarchy = DocumentHierarchy()` lines, before `lines = full_text.split('\n')`), add:

```python
    # Strip [PAGE N] markers injected by merge_pages_to_text before line processing.
    full_text = re.sub(r"^\[PAGE \d+\]\n?", "", full_text, flags=re.MULTILINE)
```

- [ ] **Step 4: Run to confirm it passes**

```bash
pytest tests/test_iscr_chunk.py::test_page_markers_stripped -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add chunk/iscr_chunk.py tests/test_iscr_chunk.py
git commit -m "fix: strip [PAGE N] markers from ISCR chunk text before processing"
```

---

## Task 7: ISCR property tests

**Files:**
- Modify: `tests/test_iscr_chunk.py` (append 8 property tests)

- [ ] **Step 1: Append property tests to `tests/test_iscr_chunk.py`**

```python
from collections import defaultdict


def test_rule_chunks_have_rule_number(iscr_chunks):
    failures = [
        chunk["chunk_id"]
        for chunk in iscr_chunks
        if chunk.get("content_type") in {"rule_text", "rule_subsection"}
        and not chunk.get("rule_number")
    ]
    assert not failures, (
        f"{len(failures)} rule_text/rule_subsection chunks missing rule_number: {failures[:5]}"
    )


def test_hierarchical_path_consistent(iscr_chunks):
    failures = []
    for chunk in iscr_chunks:
        rule_number = chunk.get("rule_number")
        path = chunk.get("hierarchical_path") or ""
        if rule_number and f"Rule {rule_number}" not in path:
            failures.append(
                f"chunk_id={chunk['chunk_id']} rule_number={rule_number!r} path={path!r}"
            )
    assert not failures, (
        f"{len(failures)} chunks have inconsistent hierarchical_path:\n"
        + "\n".join(failures[:5])
    )


def test_no_duplicate_chunk_ids(iscr_chunks):
    ids = [c["chunk_id"] for c in iscr_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids found in ISCR chunks"


def test_no_empty_chunks(iscr_chunks):
    failures = [
        chunk["chunk_id"]
        for chunk in iscr_chunks
        if not chunk.get("text", "").strip()
    ]
    assert not failures, f"{len(failures)} ISCR chunks have empty text: {failures[:5]}"


def test_no_chunk_exceeds_target_size(iscr_chunks):
    """rule_subsection chunks must not exceed the 1000-char split threshold."""
    failures = [
        (chunk["chunk_id"], len(chunk["text"]))
        for chunk in iscr_chunks
        if chunk.get("content_type") == "rule_subsection" and len(chunk["text"]) > 1000
    ]
    assert not failures, (
        f"{len(failures)} rule_subsection chunks exceed 1000 chars: {failures[:5]}"
    )


def test_large_rule_produces_multiple_chunks(iscr_chunks):
    """Any rule that produces subsection chunks must have produced more than one."""
    by_rule: dict[str, list] = defaultdict(list)
    for chunk in iscr_chunks:
        rule_number = chunk.get("rule_number")
        if rule_number:
            by_rule[rule_number].append(chunk)

    failures = []
    for rule_number, rule_chunks in by_rule.items():
        has_subsections = any(
            c.get("content_type") == "rule_subsection" for c in rule_chunks
        )
        if has_subsections and len(rule_chunks) < 2:
            failures.append(f"Rule {rule_number}: only {len(rule_chunks)} chunk despite subsection split")

    assert not failures, "\n".join(failures)


def test_small_chunks_not_orphaned(iscr_chunks):
    """Chunks with < 10 tokens must be header chunks, not rule content."""
    failures = []
    for chunk in iscr_chunks:
        if estimate_tokens(chunk.get("text", "")) < 10:
            ct = chunk.get("content_type")
            if ct not in {"article_header", "part_header"}:
                failures.append(
                    f"chunk_id={chunk['chunk_id']} content_type={ct!r} "
                    f"text={chunk['text'][:80]!r}"
                )
    assert not failures, (
        f"{len(failures)} junk micro-chunks found:\n" + "\n".join(failures[:5])
    )
```

- [ ] **Step 2: Run the full ISCR test suite**

```bash
pytest tests/test_iscr_chunk.py -v
```

Expected: all 10 tests PASS. If any property test fails, inspect the failing chunk IDs and investigate whether they indicate a latent bug or a test that needs refinement. Fix any real bugs before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_iscr_chunk.py
git commit -m "test: ISCR property tests — size, uniqueness, hierarchy, page markers, micro-chunks"
```

---

## Task 8: Full suite run and final commit

- [ ] **Step 1: Run the complete test suite**

```bash
pytest tests/ -v --tb=short 2>&1 | tee test_results.txt
```

Expected: all 17 tests PASS (7 ILCS + 10 ISCR). Review `test_results.txt` for any unexpected failures.

- [ ] **Step 2: Clean up and final commit**

```bash
rm -f test_results.txt
git add -u
git commit -m "test: complete chunking test suite — 17 tests, 2 ISCR bugs fixed"
```
