"""
Test suite for chunk/cookcounty_pd_chunk.py.

Unit tests   — synthetic records; no S3 access
Corpus tests — full cookcounty_pd_corpus.jsonl from S3 via cookcounty_pd_chunks fixture
S3 output    — reads actual cookcounty_pd_chunks.jsonl from the chunked S3 bucket
"""

import re
from collections import defaultdict

import pytest

from chunk.cookcounty_pd_chunk import (
    MAX_TOKENS,
    MIN_CHUNK_TOKENS,
    _LEGAL_AID_DIRECTORY_ID,
    chunk_record,
    normalize_whitespace,
    split_directory_into_entries,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    text: str,
    rec_id: str = "ccpd-Resources-public-defender-faq",
    page_title: str = "Public Defender FAQ",
    category: str = "FAQ",
) -> dict:
    return {
        "id":          rec_id,
        "source":      "cook_county_public_defender",
        "page_title":  page_title,
        "category":    category,
        "url":         "https://www.cookcountypublicdefender.org/Resources/public-defender-faq",
        "text":        text,
        "scraped_at":  "2026-04-20T00:00:00+00:00",
    }


def _make_directory_record(text: str) -> dict:
    return _make_record(
        text,
        rec_id=_LEGAL_AID_DIRECTORY_ID,
        page_title="Free And Low-Cost Legal Assistance Resources",
        category="General",
    )


_FILLER = (
    "The Cook County Public Defender provides free legal representation to indigent defendants. "
    "Public Defenders are licensed attorneys who specialize in criminal defense. "
) * 50

_ORG_ENTRY = (
    "Type of Legal Assistance\n"
    "Criminal defense and expungement services for low-income residents.\n"
    "To Request Legal Assistance\n"
    "Call: (312) 555-1234\n"
    "Eligibility Requirements\n"
    "Household income at or below 200% of the Federal Poverty Guidelines.\n"
)


# ---------------------------------------------------------------------------
# Unit tests — normalize_whitespace
# ---------------------------------------------------------------------------

def test_triple_blank_lines_collapsed():
    text = "Paragraph one.\n\n\n\n\nParagraph two."
    result = normalize_whitespace(text)
    assert "\n\n\n" not in result
    assert "Paragraph one." in result
    assert "Paragraph two." in result


def test_nbsp_replaced():
    text = "Free\xa0Legal\xa0Aid services available."
    result = normalize_whitespace(text)
    assert "\xa0" not in result
    assert "Free Legal Aid" in result


def test_leading_trailing_stripped():
    text = "   Legal content.   "
    assert normalize_whitespace(text) == "Legal content."


# ---------------------------------------------------------------------------
# Unit tests — split_directory_into_entries
# ---------------------------------------------------------------------------

def test_directory_splits_at_type_of_legal_assistance():
    text = (
        "Preamble text about the directory.\n\n"
        + _ORG_ENTRY
        + "\nAnother Org Name\n"
        + _ORG_ENTRY
    )
    entries = split_directory_into_entries(text)
    # Should produce the preamble + at least 2 org entries
    assert len(entries) >= 2


def test_directory_no_marker_returns_single():
    text = "General information without any org entries."
    entries = split_directory_into_entries(text)
    assert len(entries) == 1
    assert entries[0] == text


def test_directory_each_entry_contains_type_label():
    text = _ORG_ENTRY * 3
    entries = split_directory_into_entries(text)
    for entry in entries:
        if entry.strip():
            assert "Type of Legal Assistance" in entry, (
                f"Entry missing 'Type of Legal Assistance': {entry[:80]}"
            )


# ---------------------------------------------------------------------------
# Unit tests — chunk_record
# ---------------------------------------------------------------------------

def test_empty_text_no_chunks():
    assert chunk_record(_make_record("")) == []


def test_stub_below_min_tokens_skipped():
    assert chunk_record(_make_record("Short.")) == []


def test_short_page_single_chunk():
    text = "The Public Defender provides free legal representation. " * 15
    chunks = chunk_record(_make_record(text))
    assert len(chunks) == 1
    assert chunks[0].chunk_total == 1
    assert chunks[0].source == "cookcounty_pd"


def test_long_faq_produces_multiple_chunks():
    chunks = chunk_record(_make_record(_FILLER))
    assert len(chunks) > 1, "Long FAQ text should produce multiple chunks"


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


def test_directory_record_splits_at_org_boundaries():
    """The legal-aid directory record should split at 'Type of Legal Assistance' boundaries."""
    text = (_ORG_ENTRY * 10) + _FILLER
    chunks = chunk_record(_make_directory_record(text))
    # With 10 org entries there should be more than 1 chunk
    assert len(chunks) > 1, f"Directory record should produce multiple chunks, got {len(chunks)}"


def test_enriched_text_contains_page_title():
    text = "The Public Defender handles criminal, juvenile, and post-conviction cases. " * 15
    chunks = chunk_record(_make_record(text, page_title="About Our Divisions"))
    assert chunks
    assert "About Our Divisions" in chunks[0].enriched_text


def test_enriched_text_contains_cook_county():
    text = "FAQ content about the Public Defender office. " * 15
    chunks = chunk_record(_make_record(text))
    assert "Cook County Public Defender" in chunks[0].enriched_text


def test_category_in_enriched_text_when_not_general():
    text = "Juvenile court handles cases involving minors. " * 20
    chunks = chunk_record(_make_record(text, category="Juvenile Justice"))
    assert chunks
    assert "Juvenile Justice" in chunks[0].enriched_text


def test_general_category_not_duplicated_in_enriched_text():
    """Category='General' should not be appended as a bracketed label."""
    text = "General information about the Public Defender. " * 15
    chunks = chunk_record(_make_record(text, category="General"))
    assert "[General]" not in chunks[0].enriched_text


def test_source_field_correct():
    chunks = chunk_record(_make_record("Content paragraph. " * 20))
    assert all(c.source == "cookcounty_pd" for c in chunks)


def test_chunk_id_format():
    chunks = chunk_record(_make_record("Content. " * 20, rec_id="ccpd-Resources-public-defender-faq"))
    assert chunks[0].chunk_id == "ccpd-Resources-public-defender-faq_c0"


def test_category_preserved_in_all_chunks():
    chunks = chunk_record(_make_record(_FILLER, category="Expungement & Records Relief"))
    assert all(c.metadata["category"] == "Expungement & Records Relief" for c in chunks)


def test_no_chunk_below_min_tokens():
    chunks = chunk_record(_make_record(_FILLER))
    under = [c for c in chunks if c.token_count < MIN_CHUNK_TOKENS]
    assert not under, f"Chunks below MIN_CHUNK_TOKENS: {[(c.chunk_id, c.token_count) for c in under]}"


def test_chunk_schema_fields():
    """Shared Chunk schema: display_citation, token_count, source, record_id in metadata."""
    text = "The Public Defender provides free legal representation to indigent defendants. " * 15
    chunks = chunk_record(_make_record(text, rec_id="ccpd-Resources-public-defender-faq", page_title="Public Defender FAQ"))
    assert chunks
    c = chunks[0]
    assert c.display_citation.startswith("Cook County Public Defender"), (
        f"display_citation should start with 'Cook County Public Defender', got: {c.display_citation!r}"
    )
    assert c.token_count > 0
    assert c.source == "cookcounty_pd"
    assert c.metadata.get("record_id") == "ccpd-Resources-public-defender-faq"
    assert c.parent_id == "ccpd-Resources-public-defender-faq"


# ---------------------------------------------------------------------------
# Corpus-level property tests (require S3 via cookcounty_pd_chunks fixture)
# ---------------------------------------------------------------------------

def test_all_12_records_produce_chunks(cookcounty_pd_chunks):
    record_ids = {c["metadata"]["record_id"] for c in cookcounty_pd_chunks}
    assert len(record_ids) == 12, (
        f"Expected 12 distinct record IDs, got {len(record_ids)}: {record_ids}"
    )


def test_legal_aid_directory_splits_into_many_chunks(cookcounty_pd_chunks):
    """The large legal-aid directory (~28k chars) should produce many chunks."""
    dir_chunks = [c for c in cookcounty_pd_chunks if c["metadata"]["record_id"] == _LEGAL_AID_DIRECTORY_ID]
    assert dir_chunks, f"No chunks found for {_LEGAL_AID_DIRECTORY_ID}"
    assert len(dir_chunks) > 10, (
        f"Expected >10 chunks for legal-aid directory, got {len(dir_chunks)}"
    )


def test_faq_record_produces_chunks(cookcounty_pd_chunks):
    faq = [c for c in cookcounty_pd_chunks if "public-defender-faq" in c["metadata"]["record_id"]]
    assert faq, "FAQ record produced no chunks"


def test_chunk_index_contiguous(cookcounty_pd_chunks):
    by_record = defaultdict(list)
    for c in cookcounty_pd_chunks:
        by_record[c["metadata"]["record_id"]].append(c["chunk_index"])
    failures = [
        f"{rid}: {sorted(idxs)}"
        for rid, idxs in by_record.items()
        if set(idxs) != set(range(len(idxs)))
    ]
    assert not failures, "Non-contiguous chunk indices:\n" + "\n".join(failures[:5])


def test_chunk_total_accurate(cookcounty_pd_chunks):
    by_record = defaultdict(list)
    for c in cookcounty_pd_chunks:
        by_record[c["metadata"]["record_id"]].append(c)
    failures = [
        f"{c['chunk_id']}: chunk_total={c['chunk_total']} actual={len(siblings)}"
        for siblings in by_record.values()
        for c in siblings
        if c["chunk_total"] != len(siblings)
    ]
    assert not failures, "chunk_total mismatches:\n" + "\n".join(failures[:5])


def test_no_empty_chunks(cookcounty_pd_chunks):
    failures = [
        c["chunk_id"] for c in cookcounty_pd_chunks
        if c.get("token_count", 0) < MIN_CHUNK_TOKENS
    ]
    assert not failures, (
        f"{len(failures)} chunks below MIN_CHUNK_TOKENS={MIN_CHUNK_TOKENS}: {failures[:5]}"
    )


def test_chunk_ids_unique(cookcounty_pd_chunks):
    ids = [c["chunk_id"] for c in cookcounty_pd_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in Cook County PD chunks"


def test_no_chunk_exceeds_max_tokens_corpus(cookcounty_pd_chunks):
    failures = [
        (c["chunk_id"], c["token_count"])
        for c in cookcounty_pd_chunks
        if c.get("token_count", 0) > MAX_TOKENS
    ]
    assert not failures, (
        f"{len(failures)} chunks exceed MAX_TOKENS={MAX_TOKENS}: {failures[:5]}"
    )


def test_all_chunks_have_required_fields(cookcounty_pd_chunks):
    required = {
        "chunk_id", "parent_id", "chunk_index", "chunk_total", "source",
        "text", "enriched_text", "token_count", "display_citation", "metadata",
    }
    failures = [
        f"{c['chunk_id']}: missing {required - c.keys()}"
        for c in cookcounty_pd_chunks
        if not required.issubset(c.keys())
    ]
    assert not failures, "Chunks missing required fields:\n" + "\n".join(failures[:5])


def test_metadata_has_required_fields(cookcounty_pd_chunks):
    meta_required = {"record_id", "page_title", "category", "url"}
    failures = [
        f"{c['chunk_id']}: metadata missing {meta_required - c['metadata'].keys()}"
        for c in cookcounty_pd_chunks
        if not meta_required.issubset(c.get("metadata", {}).keys())
    ]
    assert not failures, "Chunks metadata missing fields:\n" + "\n".join(failures[:5])


def test_enriched_text_nonempty(cookcounty_pd_chunks):
    failures = [
        c["chunk_id"] for c in cookcounty_pd_chunks
        if not c.get("enriched_text", "").strip()
    ]
    assert not failures, f"{len(failures)} chunks have empty enriched_text: {failures[:5]}"


def test_source_field_correct_corpus(cookcounty_pd_chunks):
    failures = [
        c["chunk_id"] for c in cookcounty_pd_chunks
        if c.get("source") != "cookcounty_pd"
    ]
    assert not failures, f"{len(failures)} chunks have wrong source: {failures[:5]}"


def test_short_records_are_single_chunks(cookcounty_pd_chunks):
    """Short stub pages (< 400 chars) should produce a single chunk each."""
    by_record = defaultdict(list)
    for c in cookcounty_pd_chunks:
        by_record[c["metadata"]["record_id"]].append(c)
    # "ccpd-resources" (289 chars) and "ccpd-Resources-someones-need-services" (317 chars)
    stub_ids = ["ccpd-resources", "ccpd-Resources-someones-need-services"]
    for rid in stub_ids:
        if rid in by_record:
            assert len(by_record[rid]) == 1, (
                f"Stub record {rid} should be a single chunk, got {len(by_record[rid])}"
            )


# ---------------------------------------------------------------------------
# S3 output verification
# ---------------------------------------------------------------------------

def test_s3_output_record_count(cookcounty_pd_chunks, cookcounty_pd_chunks_s3):
    assert len(cookcounty_pd_chunks_s3) == len(cookcounty_pd_chunks), (
        f"S3 has {len(cookcounty_pd_chunks_s3)} chunks, in-memory produced {len(cookcounty_pd_chunks)}"
    )


def test_s3_output_no_corrupt_records(cookcounty_pd_chunks_s3):
    required = {"chunk_id", "chunk_index", "chunk_total", "source", "text", "enriched_text"}
    failures = [
        f"record {i}: missing {required - c.keys()}"
        for i, c in enumerate(cookcounty_pd_chunks_s3)
        if not required.issubset(c.keys())
    ]
    assert not failures, "Corrupt records in S3 output:\n" + "\n".join(failures[:5])


def test_s3_output_no_empty_text(cookcounty_pd_chunks_s3):
    failures = [c["chunk_id"] for c in cookcounty_pd_chunks_s3 if not c.get("text", "").strip()]
    assert not failures, f"{len(failures)} chunks with empty text in S3: {failures[:5]}"


def test_s3_output_source_field(cookcounty_pd_chunks_s3):
    # Accept both the legacy value ("cook_county_public_defender") and the migrated value
    # ("cookcounty_pd") until the batch rechunk pass regenerates the S3 output.
    valid_sources = {"cookcounty_pd", "cook_county_public_defender"}
    failures = [
        c["chunk_id"] for c in cookcounty_pd_chunks_s3
        if c.get("source") not in valid_sources
    ]
    assert not failures, f"{len(failures)} S3 chunks have unexpected source: {failures[:5]}"
