"""
Test suite for chunk/federal_chunk.py.

Unit tests   — synthetic records; no S3 access; test federal-specific logic
Corpus tests — full federal_corpus.jsonl from S3 via federal_chunks fixture
S3 output    — reads actual federal_chunks.jsonl from the chunked S3 bucket
"""

import re
from collections import defaultdict

import pytest

from chunk.federal_chunk import (
    MAX_TOKENS,
    MIN_CHUNK_TOKENS,
    chunk_record,
    strip_bop_page_headers,
    strip_fr_boilerplate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    text: str,
    rec_id: str = "bop-5300.21",
    doc_type: str = "policy",
    title: str = "BOP Program Statement 5300.21",
    citation: str = "",
) -> dict:
    return {
        "id":         rec_id,
        "source":     "federal",
        "doc_type":   doc_type,
        "title":      title,
        "citation":   citation,
        "url":        "https://example.com/bop.pdf",
        "text":       text,
        "scraped_at": "2026-04-20T00:00:00+00:00",
    }


_BOP_PAGE_HEADER = "PS 5300.21\n2/18/2002\nPage 3\n"

_FILLER = (
    "The Bureau of Prisons shall maintain standards for inmate education programs. "
    "All institutions must operate a full range of activities as required by statute. "
) * 25


# ---------------------------------------------------------------------------
# Unit tests — strip_bop_page_headers
# ---------------------------------------------------------------------------

def test_bop_page_header_stripped():
    text = f"Content A.\n{_BOP_PAGE_HEADER}Content B."
    result = strip_bop_page_headers(text)
    assert "Content A." in result
    assert "Content B." in result
    assert "Page 3" not in result


def test_bop_multiple_page_headers_stripped():
    h1 = "PS 5300.21\n2/18/2002\nPage 2\n"
    h2 = "PS 5300.21\n2/18/2002\nPage 4\n"
    text = f"Intro.\n{h1}Middle.\n{h2}End."
    result = strip_bop_page_headers(text)
    assert "Page 2" not in result
    assert "Page 4" not in result
    assert "Intro." in result
    assert "Middle." in result
    assert "End." in result


def test_non_bop_content_preserved():
    text = "The regulation amends 34 CFR Part 668.\nMore content follows."
    assert "34 CFR Part 668" in strip_bop_page_headers(text)


# ---------------------------------------------------------------------------
# Unit tests — strip_fr_boilerplate
# ---------------------------------------------------------------------------

def test_fr_boilerplate_trim_at_agency():
    text = (
        "Document Headings\n\n"
        "This document may contain the following:\n\n"
        "AGENCY: Department of Education.\n\n"
        "ACTION: Final rule.\n\n"
        "SUMMARY: This rule establishes prison education programs."
    )
    result = strip_fr_boilerplate(text)
    assert "AGENCY:" in result
    assert "Document Headings" not in result


def test_fr_boilerplate_trim_at_summary():
    text = (
        "skip to main content\nAlert: JavaScript required\n\n"
        "SUMMARY: This rule concerns prison education programs.\n\n"
        "The rule text follows here."
    )
    result = strip_fr_boilerplate(text)
    assert "SUMMARY:" in result
    assert "skip to main content" not in result


def test_fr_boilerplate_no_marker_returns_full_text():
    """If no AGENCY/SUMMARY/etc. marker, return the full text unchanged."""
    text = "Some regulatory text without standard markers.\nMore text."
    result = strip_fr_boilerplate(text)
    assert "Some regulatory text" in result


# ---------------------------------------------------------------------------
# Unit tests — chunk_record
# ---------------------------------------------------------------------------

def test_empty_text_no_chunks():
    assert chunk_record(_make_record("")) == []


def test_stub_below_min_tokens_skipped():
    assert chunk_record(_make_record("Short.")) == []


def test_short_statute_single_chunk():
    text = "The First Step Act of 2018 reduces mandatory minimum sentences. " * 15
    chunks = chunk_record(_make_record(text, rec_id="first-step-act", doc_type="statute"))
    assert chunks
    assert chunks[0].source == "federal"


def test_bop_numbered_sections_split():
    """BOP-style numbered sections (N.  HEADING) should be used as split points."""
    text = (
        f"1.  PURPOSE AND SCOPE\n\n{_FILLER}\n\n"
        f"2.  SUMMARY OF CHANGES\n\n{_FILLER}\n\n"
        f"3.  PROGRAM OBJECTIVES\n\n{_FILLER}"
    )
    chunks = chunk_record(_make_record(text))
    headings = {c.section_heading for c in chunks}
    # At least one numbered section boundary should be reflected in chunk headings
    assert any(re.match(r"^\d+\.", h) for h in headings if h), (
        f"No numbered-section headings found. Headings: {headings}"
    )


def test_page_header_not_in_chunks():
    text = _FILLER + "\n" + _BOP_PAGE_HEADER + _FILLER
    chunks = chunk_record(_make_record(text))
    for c in chunks:
        assert "Page 3" not in c.text, (
            f"BOP page header leaked into {c.chunk_id}"
        )


def test_no_chunk_exceeds_max_tokens():
    text = "\n\n".join([_FILLER] * 4)
    chunks = chunk_record(_make_record(text))
    over = [(c.chunk_id, c.token_count) for c in chunks if c.token_count > MAX_TOKENS]
    assert not over, f"Chunks exceed MAX_TOKENS={MAX_TOKENS}: {over}"


def test_chunk_ids_contiguous():
    text = "\n\n".join([_FILLER] * 3)
    chunks = chunk_record(_make_record(text))
    indices = [c.chunk_index for c in chunks]
    assert sorted(indices) == list(range(len(indices)))
    for c in chunks:
        assert c.chunk_total == len(chunks)


def test_enriched_text_contains_title_and_citation():
    text = "The Second Chance Pell rule amends 34 CFR Part 668. " * 20
    chunks = chunk_record(_make_record(
        text,
        rec_id="second-chance-pell-rule",
        doc_type="regulation",
        title="Second Chance Pell Grant Final Rule",
        citation="34 CFR Part 668",
    ))
    assert chunks
    assert "Second Chance Pell Grant Final Rule" in chunks[0].enriched_text
    assert "34 CFR Part 668" in chunks[0].enriched_text


def test_source_field_is_federal():
    text = "Federal education program requirements. " * 20
    chunks = chunk_record(_make_record(text))
    assert all(c.source == "federal" for c in chunks)


def test_chunk_id_format():
    text = "Prison education program requirements. " * 20
    chunks = chunk_record(_make_record(text, rec_id="bop-5300.21"))
    assert chunks[0].chunk_id == "bop-5300.21_c0"


def test_doc_type_preserved():
    text = "Statutory text here. " * 20
    chunks = chunk_record(_make_record(text, doc_type="statute"))
    assert all(c.doc_type == "statute" for c in chunks)


# ---------------------------------------------------------------------------
# Corpus-level property tests (require S3 via federal_chunks fixture)
# ---------------------------------------------------------------------------

def test_all_three_records_produce_chunks(federal_chunks):
    record_ids = {c["record_id"] for c in federal_chunks}
    assert "second-chance-pell-rule" in record_ids, "Missing second-chance-pell-rule chunks"
    assert "first-step-act" in record_ids, "Missing first-step-act chunks"
    assert "bop-5300.21" in record_ids, "Missing bop-5300.21 chunks"


def test_bop_doc_splits_into_multiple_chunks(federal_chunks):
    bop = [c for c in federal_chunks if c["record_id"] == "bop-5300.21"]
    assert len(bop) > 5, f"Expected >5 chunks for BOP 5300.21, got {len(bop)}"


def test_chunk_index_contiguous(federal_chunks):
    by_record = defaultdict(list)
    for c in federal_chunks:
        by_record[c["record_id"]].append(c["chunk_index"])
    failures = [
        f"{rid}: {sorted(idxs)}"
        for rid, idxs in by_record.items()
        if set(idxs) != set(range(len(idxs)))
    ]
    assert not failures, "Non-contiguous chunk indices:\n" + "\n".join(failures[:5])


def test_chunk_total_accurate(federal_chunks):
    by_record = defaultdict(list)
    for c in federal_chunks:
        by_record[c["record_id"]].append(c)
    failures = [
        f"{c['chunk_id']}: chunk_total={c['chunk_total']} actual={len(siblings)}"
        for siblings in by_record.values()
        for c in siblings
        if c["chunk_total"] != len(siblings)
    ]
    assert not failures, "chunk_total mismatches:\n" + "\n".join(failures[:5])


def test_no_empty_chunks(federal_chunks):
    failures = [c["chunk_id"] for c in federal_chunks if c.get("token_count", 0) < MIN_CHUNK_TOKENS]
    assert not failures, (
        f"{len(failures)} chunks below MIN_CHUNK_TOKENS={MIN_CHUNK_TOKENS}: {failures[:5]}"
    )


def test_chunk_ids_unique(federal_chunks):
    ids = [c["chunk_id"] for c in federal_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in federal chunks"


def test_no_chunk_exceeds_max_tokens_corpus(federal_chunks):
    failures = [
        (c["chunk_id"], c["token_count"])
        for c in federal_chunks
        if c.get("token_count", 0) > MAX_TOKENS
    ]
    assert not failures, (
        f"{len(failures)} chunks exceed MAX_TOKENS={MAX_TOKENS}: {failures[:5]}"
    )


def test_all_chunks_have_required_fields(federal_chunks):
    required = {
        "chunk_id", "chunk_index", "chunk_total", "source",
        "text", "enriched_text", "token_count", "record_id",
        "title", "doc_type", "url",
    }
    failures = [
        f"{c['chunk_id']}: missing {required - c.keys()}"
        for c in federal_chunks
        if not required.issubset(c.keys())
    ]
    assert not failures, "Chunks missing required fields:\n" + "\n".join(failures[:5])


def test_enriched_text_nonempty(federal_chunks):
    failures = [c["chunk_id"] for c in federal_chunks if not c.get("enriched_text", "").strip()]
    assert not failures, f"{len(failures)} chunks have empty enriched_text: {failures[:5]}"


def test_source_field_is_federal_corpus(federal_chunks):
    failures = [c["chunk_id"] for c in federal_chunks if c.get("source") != "federal"]
    assert not failures, f"{len(failures)} chunks have wrong source: {failures[:5]}"


def test_no_bop_page_headers_in_corpus(federal_chunks):
    """BOP page repeat headers (PS XXXX.XX / date / Page N) must not appear in any chunk."""
    bop_header_re = re.compile(r"PS \d+\.\d+\s*\n\d+/\d+/\d+\s*\nPage \d+")
    failures = [
        c["chunk_id"] for c in federal_chunks
        if bop_header_re.search(c.get("text", ""))
    ]
    assert not failures, (
        f"{len(failures)} chunks contain BOP page headers: {failures[:5]}"
    )


# ---------------------------------------------------------------------------
# S3 output verification
# ---------------------------------------------------------------------------

def test_s3_output_record_count(federal_chunks, federal_chunks_s3):
    assert len(federal_chunks_s3) == len(federal_chunks), (
        f"S3 has {len(federal_chunks_s3)} chunks, in-memory produced {len(federal_chunks)}"
    )


def test_s3_output_no_corrupt_records(federal_chunks_s3):
    required = {"chunk_id", "chunk_index", "chunk_total", "source", "text", "enriched_text"}
    failures = [
        f"record {i}: missing {required - c.keys()}"
        for i, c in enumerate(federal_chunks_s3)
        if not required.issubset(c.keys())
    ]
    assert not failures, "Corrupt records in S3 output:\n" + "\n".join(failures[:5])


def test_s3_output_no_empty_text(federal_chunks_s3):
    failures = [c["chunk_id"] for c in federal_chunks_s3 if not c.get("text", "").strip()]
    assert not failures, f"{len(failures)} chunks with empty text in S3: {failures[:5]}"


def test_s3_output_source_field(federal_chunks_s3):
    failures = [c["chunk_id"] for c in federal_chunks_s3 if c.get("source") != "federal"]
    assert not failures, f"{len(failures)} S3 chunks have wrong source: {failures[:5]}"
