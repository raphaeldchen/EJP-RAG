"""
Test suite for chunk/restorejustice_chunk.py.

Unit tests   — synthetic records; no S3 access
Corpus tests — full restorejustice_corpus.jsonl from S3 via restorejustice_chunks fixture
S3 output    — reads actual restorejustice_chunks.jsonl from the chunked S3 bucket
"""

import re
from collections import defaultdict

import pytest

from chunk.restorejustice_chunk import (
    MAX_TOKENS,
    MIN_CHUNK_TOKENS,
    chunk_record,
    normalize_whitespace,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    text: str,
    rec_id: str = "rj-our-work-advocacy",
    section: str = "our-work",
    page_title: str = "Advocacy",
) -> dict:
    return {
        "id":          rec_id,
        "source":      "restore_justice",
        "section":     section,
        "page_title":  page_title,
        "url":         "https://www.restorejustice.org/our-work/advocacy/",
        "text":        text,
        "scraped_at":  "2026-04-20T00:00:00+00:00",
    }


_FILLER = (
    "Restore Justice advocates for people who received extreme sentences as children. "
    "The organization works with families to reform Illinois sentencing laws. "
) * 60


# ---------------------------------------------------------------------------
# Unit tests — normalize_whitespace
# ---------------------------------------------------------------------------

def test_triple_blank_lines_collapsed():
    text = "Paragraph one.\n\n\n\n\nParagraph two."
    result = normalize_whitespace(text)
    assert "\n\n\n" not in result
    assert "Paragraph one." in result
    assert "Paragraph two." in result


def test_double_blank_lines_preserved():
    text = "Para one.\n\nPara two."
    result = normalize_whitespace(text)
    assert "Para one." in result
    assert "Para two." in result


def test_nbsp_replaced_with_space():
    text = "Text with\xa0non-breaking\xa0spaces."
    result = normalize_whitespace(text)
    assert "\xa0" not in result
    assert "non-breaking" in result


def test_leading_trailing_stripped():
    text = "\n\n  Content here.  \n\n"
    assert normalize_whitespace(text) == "Content here."


# ---------------------------------------------------------------------------
# Unit tests — chunk_record
# ---------------------------------------------------------------------------

def test_empty_text_no_chunks():
    assert chunk_record(_make_record("")) == []


def test_stub_below_min_tokens_skipped():
    assert chunk_record(_make_record("Short.")) == []


def test_short_page_single_chunk():
    text = "Restore Justice works to reform Illinois sentencing laws. " * 15
    chunks = chunk_record(_make_record(text))
    assert len(chunks) == 1
    assert chunks[0].chunk_total == 1
    assert chunks[0].source == "restore_justice"


def test_long_text_produces_multiple_chunks():
    chunks = chunk_record(_make_record(_FILLER))
    assert len(chunks) > 1, "Long text should produce multiple chunks"


def test_no_chunk_exceeds_max_tokens():
    text = _FILLER * 3
    chunks = chunk_record(_make_record(text))
    over = [(c.chunk_id, c.token_count) for c in chunks if c.token_count > MAX_TOKENS]
    assert not over, f"Chunks exceed MAX_TOKENS={MAX_TOKENS}: {over}"


def test_chunk_ids_contiguous():
    chunks = chunk_record(_make_record(_FILLER))
    indices = [c.chunk_index for c in chunks]
    assert sorted(indices) == list(range(len(indices)))
    for c in chunks:
        assert c.chunk_total == len(chunks)


def test_enriched_text_contains_page_title():
    text = "Content about criminal justice reform. " * 20
    chunks = chunk_record(_make_record(text, page_title="Advocacy", section="our-work"))
    assert chunks
    assert "Advocacy" in chunks[0].enriched_text


def test_enriched_text_contains_restore_justice():
    text = "Advocacy content. " * 20
    chunks = chunk_record(_make_record(text))
    assert "Restore Justice" in chunks[0].enriched_text


def test_chunk_id_format():
    text = "Advocacy paragraph content. " * 20
    chunks = chunk_record(_make_record(text, rec_id="rj-our-work-advocacy"))
    assert chunks[0].chunk_id == "rj-our-work-advocacy_c0"


def test_source_field_is_restore_justice():
    text = "Content paragraph. " * 20
    chunks = chunk_record(_make_record(text))
    assert all(c.source == "restore_justice" for c in chunks)


def test_page_title_preserved_in_all_chunks():
    chunks = chunk_record(_make_record(_FILLER, page_title="Future Leaders Program"))
    assert all(c.page_title == "Future Leaders Program" for c in chunks)


def test_section_preserved_in_all_chunks():
    chunks = chunk_record(_make_record(_FILLER, section="our-work"))
    assert all(c.section == "our-work" for c in chunks)


def test_no_chunk_below_min_tokens():
    text = "Restore Justice works to reduce extreme sentences. " * 40
    chunks = chunk_record(_make_record(text))
    under = [c for c in chunks if c.token_count < MIN_CHUNK_TOKENS]
    assert not under, f"Chunks below MIN_CHUNK_TOKENS: {[(c.chunk_id, c.token_count) for c in under]}"


def test_blank_line_clusters_not_in_text():
    """Triple+ blank lines from HTML extraction should not appear in chunk text."""
    text = "Paragraph one.\n\n\n\n\nParagraph two.\n\n\n\n\nParagraph three."
    chunks = chunk_record(_make_record(text))
    for c in chunks:
        assert "\n\n\n" not in c.text, f"Triple blank line in {c.chunk_id}"


# ---------------------------------------------------------------------------
# Corpus-level property tests (require S3 via restorejustice_chunks fixture)
# ---------------------------------------------------------------------------

def test_all_17_records_produce_chunks(restorejustice_chunks):
    record_ids = {c["record_id"] for c in restorejustice_chunks}
    assert len(record_ids) == 17, (
        f"Expected 17 distinct record IDs, got {len(record_ids)}"
    )


def test_resources_record_produces_chunks(restorejustice_chunks):
    resources = [c for c in restorejustice_chunks if c["record_id"] == "rj-resources"]
    assert resources, "rj-resources record produced no chunks"


def test_long_record_splits(restorejustice_chunks):
    """The IDOC transfers self-advocacy notes (~10k chars) should produce multiple chunks."""
    long_rec = [
        c for c in restorejustice_chunks
        if "loved-ones-self-advocacy" in c.get("record_id", "")
           or "idoc-transfers" in c.get("record_id", "")
    ]
    if not long_rec:
        pytest.skip("Long advocacy record not found in chunks")
    assert len(long_rec) > 1, f"Long record should produce multiple chunks, got {len(long_rec)}"


def test_chunk_index_contiguous(restorejustice_chunks):
    by_record = defaultdict(list)
    for c in restorejustice_chunks:
        by_record[c["record_id"]].append(c["chunk_index"])
    failures = [
        f"{rid}: {sorted(idxs)}"
        for rid, idxs in by_record.items()
        if set(idxs) != set(range(len(idxs)))
    ]
    assert not failures, "Non-contiguous chunk indices:\n" + "\n".join(failures[:5])


def test_chunk_total_accurate(restorejustice_chunks):
    by_record = defaultdict(list)
    for c in restorejustice_chunks:
        by_record[c["record_id"]].append(c)
    failures = [
        f"{c['chunk_id']}: chunk_total={c['chunk_total']} actual={len(siblings)}"
        for siblings in by_record.values()
        for c in siblings
        if c["chunk_total"] != len(siblings)
    ]
    assert not failures, "chunk_total mismatches:\n" + "\n".join(failures[:5])


def test_no_empty_chunks(restorejustice_chunks):
    failures = [
        c["chunk_id"] for c in restorejustice_chunks
        if c.get("token_count", 0) < MIN_CHUNK_TOKENS
    ]
    assert not failures, (
        f"{len(failures)} chunks below MIN_CHUNK_TOKENS={MIN_CHUNK_TOKENS}: {failures[:5]}"
    )


def test_chunk_ids_unique(restorejustice_chunks):
    ids = [c["chunk_id"] for c in restorejustice_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in Restore Justice chunks"


def test_no_chunk_exceeds_max_tokens_corpus(restorejustice_chunks):
    failures = [
        (c["chunk_id"], c["token_count"])
        for c in restorejustice_chunks
        if c.get("token_count", 0) > MAX_TOKENS
    ]
    assert not failures, (
        f"{len(failures)} chunks exceed MAX_TOKENS={MAX_TOKENS}: {failures[:5]}"
    )


def test_all_chunks_have_required_fields(restorejustice_chunks):
    required = {
        "chunk_id", "chunk_index", "chunk_total", "source",
        "text", "enriched_text", "token_count", "record_id",
        "page_title", "section", "url",
    }
    failures = [
        f"{c['chunk_id']}: missing {required - c.keys()}"
        for c in restorejustice_chunks
        if not required.issubset(c.keys())
    ]
    assert not failures, "Chunks missing required fields:\n" + "\n".join(failures[:5])


def test_enriched_text_nonempty(restorejustice_chunks):
    failures = [
        c["chunk_id"] for c in restorejustice_chunks
        if not c.get("enriched_text", "").strip()
    ]
    assert not failures, f"{len(failures)} chunks have empty enriched_text: {failures[:5]}"


def test_source_field_is_restore_justice_corpus(restorejustice_chunks):
    failures = [
        c["chunk_id"] for c in restorejustice_chunks
        if c.get("source") != "restore_justice"
    ]
    assert not failures, f"{len(failures)} chunks have wrong source: {failures[:5]}"


def test_most_short_records_are_single_chunks(restorejustice_chunks):
    """Most RJ records are short HTML pages; the majority should be single chunks."""
    by_record = defaultdict(list)
    for c in restorejustice_chunks:
        by_record[c["record_id"]].append(c)
    single_chunk_records = sum(1 for chunks in by_record.values() if len(chunks) == 1)
    total_records = len(by_record)
    assert single_chunk_records / total_records >= 0.4, (
        f"Expected ≥40% of records to be single chunks, got {single_chunk_records}/{total_records}"
    )


# ---------------------------------------------------------------------------
# S3 output verification
# ---------------------------------------------------------------------------

def test_s3_output_record_count(restorejustice_chunks, restorejustice_chunks_s3):
    assert len(restorejustice_chunks_s3) == len(restorejustice_chunks), (
        f"S3 has {len(restorejustice_chunks_s3)} chunks, in-memory produced {len(restorejustice_chunks)}"
    )


def test_s3_output_no_corrupt_records(restorejustice_chunks_s3):
    required = {"chunk_id", "chunk_index", "chunk_total", "source", "text", "enriched_text"}
    failures = [
        f"record {i}: missing {required - c.keys()}"
        for i, c in enumerate(restorejustice_chunks_s3)
        if not required.issubset(c.keys())
    ]
    assert not failures, "Corrupt records in S3 output:\n" + "\n".join(failures[:5])


def test_s3_output_no_empty_text(restorejustice_chunks_s3):
    failures = [c["chunk_id"] for c in restorejustice_chunks_s3 if not c.get("text", "").strip()]
    assert not failures, f"{len(failures)} chunks with empty text in S3: {failures[:5]}"


def test_s3_output_source_field(restorejustice_chunks_s3):
    failures = [
        c["chunk_id"] for c in restorejustice_chunks_s3
        if c.get("source") != "restore_justice"
    ]
    assert not failures, f"{len(failures)} S3 chunks have wrong source: {failures[:5]}"
