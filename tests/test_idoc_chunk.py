"""
Test suite for chunk/idoc_chunk.py.

Unit tests  — synthetic records; no S3 access; test IDOC-specific logic
Corpus tests — full idoc_corpus.jsonl from S3 via iac_records-style fixture
S3 output   — reads actual idoc_chunks.jsonl from the chunked S3 bucket
"""

import re
from collections import defaultdict

import pytest

from chunk.idoc_chunk import (
    MAX_TOKENS,
    MIN_CHUNK_TOKENS,
    chunk_record,
    strip_page_headers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_directive(text: str, rec_id: str = "idoc-dir-403103") -> dict:
    return {
        "id":           rec_id,
        "source":       "idoc_directive",
        "doc_type":     "administrative_directive",
        "category":     "Health Care",
        "sub_category": "Medical",
        "title":        "Test Directive",
        "url":          "https://example.com/test.pdf",
        "text":         text,
        "scraped_at":   "2026-04-20T00:00:00+00:00",
    }

_PAGE_LEAK_RE = re.compile(r"Page \d+ of \d+")

# A realistic page repeat header as it appears between PDF pages
_SAMPLE_PAGE_HEADER = (
    "\n\nIllinois Department of Corrections \n"
    "Administrative Directive \n"
    "Page 2 of 5 \n"
    "Number:  \n"
    "04.03.103 \n"
    "Title: \n"
    "Offender Health Care Services \n"
    "Effective: \n"
    "1/1/2020 \n\n"
)

# Filler text large enough to push over MAX_TOKENS
_FILLER = ("The Department shall ensure compliance with all applicable standards. " * 30)


# ---------------------------------------------------------------------------
# Unit tests — strip_page_headers
# ---------------------------------------------------------------------------

def test_strip_removes_mid_document_page_header():
    text = f"Section A content.\n{_SAMPLE_PAGE_HEADER}Section B content."
    result = strip_page_headers(text)
    assert "Page 2 of 5" not in result
    assert "Section A content." in result
    assert "Section B content." in result


def test_strip_preserves_initial_header_block():
    """The genuine header at position 0 must NOT be stripped."""
    initial = (
        "Illinois Department of Corrections \n\n"
        "Administrative Directive \n"
        "Number:  \n04.03.103 \n"
        "Title: \nOffender Health Care Services \n"
        "Effective: \n1/1/2020 \n\n"
    )
    text = initial + "I. POLICY\n\nThe policy text."
    result = strip_page_headers(text)
    assert "Illinois Department of Corrections" in result
    assert "Administrative Directive" in result
    assert "Effective:" in result


def test_strip_removes_multiple_page_headers():
    header2 = _SAMPLE_PAGE_HEADER
    header3 = _SAMPLE_PAGE_HEADER.replace("Page 2 of 5", "Page 3 of 5")
    text = f"Content A.{header2}Content B.{header3}Content C."
    result = strip_page_headers(text)
    assert "Page 2 of 5" not in result
    assert "Page 3 of 5" not in result
    assert "Content A." in result
    assert "Content B." in result
    assert "Content C." in result


# ---------------------------------------------------------------------------
# Unit tests — chunk_record
# ---------------------------------------------------------------------------

def test_empty_text_produces_no_chunks():
    assert chunk_record(_make_directive("")) == []


def test_short_directive_produces_chunks():
    text = (
        "Illinois Department of Corrections \n\n"
        "Administrative Directive \nNumber: \n04.03.103 \n"
        "Title: \nTest \nEffective: \n1/1/2026 \n\n"
        "I. POLICY\n\n" + ("The Department shall maintain compliance. " * 10) + "\n\n"
        "II. PROCEDURE\n\nA. Purpose\n\n" + ("The purpose is to establish guidelines. " * 10) + "\n"
    )
    chunks = chunk_record(_make_directive(text))
    assert chunks, "No chunks produced"


def test_roman_sections_become_separate_chunks():
    """A directive with multiple Roman numeral sections should split at those boundaries."""
    text = (
        "Illinois Department of Corrections \n\nAdministrative Directive \n"
        "Number: \n04.03.103 \nTitle: \nTest \nEffective: \n1/1/2026 \n\n"
        f"I. POLICY\n\n{_FILLER}\n\n"
        f"II. PROCEDURE\n\nA. Purpose\n\n{_FILLER}\n"
    )
    chunks = chunk_record(_make_directive(text))
    headings = [c.section_heading for c in chunks]
    assert any("I. POLICY" in h for h in headings), "I. POLICY section not found in any chunk"
    assert any("II. PROCEDURE" in h for h in headings or "A. Purpose" in h for h in headings), \
        "II. PROCEDURE / A. Purpose not found in any chunk"


def test_no_page_header_in_chunk_text():
    """Page repeat headers must never appear in any chunk's text."""
    text = (
        "Illinois Department of Corrections \n\nAdministrative Directive \n"
        "Number: \n04.03.103 \nTitle: \nTest \nEffective: \n1/1/2026 \n\n"
        f"I. POLICY\n\n{_FILLER}"
        f"{_SAMPLE_PAGE_HEADER}"
        f"II. PROCEDURE\n\n{_FILLER}\n"
    )
    chunks = chunk_record(_make_directive(text))
    for chunk in chunks:
        assert not _PAGE_LEAK_RE.search(chunk.text), (
            f"Page header leaked into chunk {chunk.chunk_id}:\n{chunk.text[:200]}"
        )


def test_no_chunk_exceeds_max_tokens():
    text = (
        "Illinois Department of Corrections \n\nAdministrative Directive \n"
        "Number: \n04.03.103 \nTitle: \nTest \nEffective: \n1/1/2026 \n\n"
        f"II. PROCEDURE\n\nA. Requirements\n\n{_FILLER * 3}\n"
    )
    chunks = chunk_record(_make_directive(text))
    over = [(c.chunk_id, c.token_count) for c in chunks if c.token_count > MAX_TOKENS]
    assert not over, f"Chunks exceed MAX_TOKENS={MAX_TOKENS}: {over}"


def test_enriched_text_contains_directive_number_and_title():
    text = (
        "Illinois Department of Corrections \n\nAdministrative Directive \n"
        "Number: \n04.03.103 \nTitle: \nOffender Health Care \nEffective: \n1/1/2026 \n\n"
        "I. POLICY\n\n" + ("The policy text ensures compliance with applicable standards. " * 10) + "\n"
    )
    chunks = chunk_record(_make_directive(text, rec_id="idoc-dir-403103"))
    assert chunks
    has_ref = any("403103" in c.enriched_text for c in chunks)
    assert has_ref, "No chunk enriched_text contains the directive number"


def test_chunk_ids_contiguous():
    text = (
        "Illinois Department of Corrections \n\nAdministrative Directive \n"
        "Number: \n04.50.151 \nTitle: \nTest \nEffective: \n1/1/2026 \n\n"
        f"I. POLICY\n\n{_FILLER}\n\n"
        f"II. PROCEDURE\n\nA. Purpose\n\n{_FILLER}\n\nB. Definitions\n\n{_FILLER}\n"
    )
    chunks = chunk_record(_make_directive(text, rec_id="idoc-dir-450151"))
    indices = [c.chunk_index for c in chunks]
    assert sorted(indices) == list(range(len(indices))), f"Non-contiguous indices: {indices}"
    for c in chunks:
        assert c.chunk_total == len(chunks)


def test_non_directive_record_chunked_as_single():
    rec = {
        "id":         "idoc-reentry-hub",
        "source":     "idoc_reentry",
        "doc_type":   "",
        "category":   "",
        "sub_category": "",
        "title":      "IDOC Reentry Resources",
        "url":        "https://example.com",
        "text":       "Reentry resources text. " * 20,
        "scraped_at": "2026-04-20T00:00:00+00:00",
    }
    chunks = chunk_record(rec)
    assert chunks, "Reentry record produced no chunks"
    assert all(c.source == "idoc_reentry" for c in chunks)


# ---------------------------------------------------------------------------
# Corpus-level property tests (require S3 via idoc_chunks fixture)
# ---------------------------------------------------------------------------

def test_no_page_headers_in_corpus_chunks(idoc_chunks):
    failures = [c["chunk_id"] for c in idoc_chunks if _PAGE_LEAK_RE.search(c.get("text", ""))]
    assert not failures, (
        f"{len(failures)} chunks contain page headers: {failures[:5]}"
    )


def test_chunk_index_contiguous(idoc_chunks):
    by_record = defaultdict(list)
    for c in idoc_chunks:
        by_record[c["record_id"]].append(c["chunk_index"])
    failures = [
        f"{rid}: {sorted(idxs)}"
        for rid, idxs in by_record.items()
        if set(idxs) != set(range(len(idxs)))
    ]
    assert not failures, "Non-contiguous chunk indices:\n" + "\n".join(failures[:5])


def test_chunk_total_accurate(idoc_chunks):
    by_record = defaultdict(list)
    for c in idoc_chunks:
        by_record[c["record_id"]].append(c)
    failures = [
        f"{c['chunk_id']}: chunk_total={c['chunk_total']} actual={len(siblings)}"
        for siblings in by_record.values()
        for c in siblings
        if c["chunk_total"] != len(siblings)
    ]
    assert not failures, "chunk_total mismatches:\n" + "\n".join(failures[:5])


def test_no_empty_chunks(idoc_chunks):
    failures = [c["chunk_id"] for c in idoc_chunks if c["token_count"] < MIN_CHUNK_TOKENS]
    assert not failures, f"{len(failures)} chunks below MIN_CHUNK_TOKENS: {failures[:5]}"


def test_chunk_ids_unique(idoc_chunks):
    ids = [c["chunk_id"] for c in idoc_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in IDOC chunks"


def test_no_chunk_exceeds_max_tokens_corpus(idoc_chunks):
    failures = [(c["chunk_id"], c["token_count"]) for c in idoc_chunks if c["token_count"] > MAX_TOKENS]
    assert not failures, (
        f"{len(failures)} chunks exceed MAX_TOKENS={MAX_TOKENS}: {failures[:5]}"
    )


def test_all_chunks_have_required_fields(idoc_chunks):
    required = {"chunk_id", "chunk_index", "chunk_total", "source", "text",
                "enriched_text", "token_count", "record_id", "title"}
    failures = [
        f"{c['chunk_id']}: missing {required - c.keys()}"
        for c in idoc_chunks
        if not required.issubset(c.keys())
    ]
    assert not failures, "Chunks missing required fields:\n" + "\n".join(failures[:5])


def test_directive_chunks_have_correct_source(idoc_chunks):
    directive_chunks = [c for c in idoc_chunks if c.get("source") == "idoc_directive"]
    assert directive_chunks, "No idoc_directive chunks found"
    failures = [
        c["chunk_id"] for c in directive_chunks
        if not c.get("enriched_text", "").strip()
    ]
    assert not failures, f"{len(failures)} directive chunks have empty enriched_text"


def test_long_directive_splits_into_sections(idoc_chunks):
    """The longest directive (04.50.151, ~68k chars) must produce many chunks
    with distinct section headings."""
    target = [c for c in idoc_chunks if "450151" in c.get("record_id", "")
              or "505105" in c.get("record_id", "")]
    if not target:
        pytest.skip("Long directive not found in chunked output")
    assert len(target) > 5, f"Expected >5 chunks for a long directive, got {len(target)}"
    headings = {c["section_heading"] for c in target if c["section_heading"]}
    assert len(headings) > 1, "Long directive produced only one distinct section heading"


# ---------------------------------------------------------------------------
# S3 output verification
# ---------------------------------------------------------------------------

def test_s3_output_record_count(idoc_chunks, idoc_chunks_s3):
    assert len(idoc_chunks_s3) == len(idoc_chunks), (
        f"S3 has {len(idoc_chunks_s3)} chunks, in-memory produced {len(idoc_chunks)}"
    )


def test_s3_output_no_corrupt_records(idoc_chunks_s3):
    required = {"chunk_id", "chunk_index", "chunk_total", "source", "text", "enriched_text"}
    failures = [
        f"record {i}: missing {required - c.keys()}"
        for i, c in enumerate(idoc_chunks_s3)
        if not required.issubset(c.keys())
    ]
    assert not failures, "Corrupt records in S3 output:\n" + "\n".join(failures[:5])


def test_s3_output_no_empty_text(idoc_chunks_s3):
    failures = [c["chunk_id"] for c in idoc_chunks_s3 if not c.get("text", "").strip()]
    assert not failures, f"{len(failures)} chunks with empty text in S3: {failures[:5]}"


def test_s3_output_no_page_headers(idoc_chunks_s3):
    failures = [c["chunk_id"] for c in idoc_chunks_s3 if _PAGE_LEAK_RE.search(c.get("text", ""))]
    assert not failures, f"{len(failures)} S3 chunks contain page headers: {failures[:5]}"


def test_s3_output_source_fields(idoc_chunks_s3):
    valid_sources = {"idoc_directive", "idoc_reentry"}
    failures = [
        c["chunk_id"] for c in idoc_chunks_s3
        if c.get("source") not in valid_sources
    ]
    assert not failures, f"{len(failures)} S3 chunks have unexpected source: {failures[:5]}"
