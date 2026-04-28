"""
Test suite for chunk/spac_chunk.py.

Unit tests  — synthetic records; no S3 access
Corpus tests — full spac_corpus.jsonl from S3 via spac_chunks fixture
S3 output   — reads actual spac_chunks.jsonl from the chunked S3 bucket
"""

import re
from collections import defaultdict

import pytest

from chunk.spac_chunk import (
    MAX_TOKENS,
    MIN_CHUNK_TOKENS,
    chunk_record,
    is_section_heading,
    split_at_headings,
    strip_page_headers,
    strip_toc_lines,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    text: str,
    rec_id: str = "spac-test-001",
    category: str = "Report",
    year: str = "2020",
    title: str = "Test Report",
) -> dict:
    return {
        "id":        rec_id,
        "source":    "spac",
        "agency":    "Illinois Sentencing Policy Advisory Council",
        "category":  category,
        "title":     title,
        "year":      year,
        "filename":  "test_2020.pdf",
        "url":       "https://example.com/test.pdf",
        "text":      text,
        "scraped_at": "2026-04-27T00:00:00+00:00",
    }


_PAGE_LEAK_RE = re.compile(r"\bPage \d+ of \d+\b", re.IGNORECASE)
_DOT_LEAK_RE  = re.compile(r"\.{4,}")


# ---------------------------------------------------------------------------
# Unit tests — strip_page_headers
# ---------------------------------------------------------------------------

def test_page_header_stripped():
    text = "Content A.\nMay 2017 Sentencing Reform Page 1 of 78\nContent B."
    result = strip_page_headers(text)
    assert "Page 1 of 78" not in result
    assert "Content A." in result
    assert "Content B." in result


def test_multiple_page_headers_stripped():
    text = "Start.\nPage 3 of 20\nMiddle.\nPage 4 of 20\nEnd."
    result = strip_page_headers(text)
    assert "Page 3 of 20" not in result
    assert "Page 4 of 20" not in result
    assert "Start." in result
    assert "Middle." in result
    assert "End." in result


def test_non_page_content_preserved():
    text = "No page numbers here.\nJust regular content."
    assert "regular content" in strip_page_headers(text)


# ---------------------------------------------------------------------------
# Unit tests — strip_toc_lines
# ---------------------------------------------------------------------------

def test_toc_dot_leader_stripped():
    text = "Introduction ............................................................... 4\nReal content here."
    result = strip_toc_lines(text)
    assert not _DOT_LEAK_RE.search(result), "Dot-leader line was not stripped"
    assert "Real content here." in result


def test_toc_multiple_dot_leader_lines_stripped():
    toc = (
        "Introduction ...................................... 4\n"
        "Background ........................................ 6\n"
        "Findings .......................................... 12\n"
    )
    text = toc + "Actual content here."
    result = strip_toc_lines(text)
    assert not _DOT_LEAK_RE.search(result)
    assert "Actual content here." in result


def test_non_toc_lines_with_few_dots_preserved():
    text = "See e.g. Smith v. Jones, 123 F.3d 456 (7th Cir. 2000).\nMore content."
    result = strip_toc_lines(text)
    assert "Smith v. Jones" in result


# ---------------------------------------------------------------------------
# Unit tests — is_section_heading
# ---------------------------------------------------------------------------

def test_three_word_all_caps_is_heading():
    assert is_section_heading("DRUG OFFENSE REFORM") is True


def test_four_word_heading():
    assert is_section_heading("LIMITATIONS AND ASSUMPTIONS SECTION") is True


def test_long_heading():
    assert is_section_heading("INSUFFICIENT DATA TO SUPPORT A FULL FISCAL IMPACT ANALYSIS") is True


def test_heading_with_year():
    assert is_section_heading("FINDINGS AND RECOMMENDATIONS FOR 2025") is True


def test_two_word_not_heading():
    assert is_section_heading("RETAIL THEFT") is False


def test_one_word_not_heading():
    assert is_section_heading("COMPONENT") is False


def test_mixed_case_not_heading():
    assert is_section_heading("Drug Offense Reform") is False


def test_lowercase_not_heading():
    assert is_section_heading("drug offense reform") is False


def test_heading_with_parenthetical():
    assert is_section_heading("CANNABIS POLICY REFORM (HB 1234)") is True


def test_empty_string_not_heading():
    assert is_section_heading("") is False


def test_blank_whitespace_not_heading():
    assert is_section_heading("   ") is False


# ---------------------------------------------------------------------------
# Unit tests — split_at_headings
# ---------------------------------------------------------------------------

def test_split_produces_preamble_then_sections():
    text = "Introduction text.\n\nDRUG OFFENSE REFORM\n\nBody of section."
    sections = split_at_headings(text)
    headings = [h for h, _ in sections]
    assert "" in headings, "Preamble section missing"
    assert "DRUG OFFENSE REFORM" in headings


def test_split_no_headings_returns_single_section():
    text = "Just a paragraph of text with no headings at all."
    sections = split_at_headings(text)
    assert len(sections) == 1
    assert sections[0][0] == ""
    assert "paragraph of text" in sections[0][1]


def test_split_multiple_headings():
    text = (
        "DRUG OFFENSE REFORM\n\nContent about drugs.\n\n"
        "MANDATORY SUPERVISED RELEASE\n\nContent about release.\n\n"
        "FISCAL IMPACT ANALYSIS\n\nContent about fiscal impact."
    )
    sections = split_at_headings(text)
    headings = [h for h, _ in sections if h]
    assert "DRUG OFFENSE REFORM" in headings
    assert "MANDATORY SUPERVISED RELEASE" in headings
    assert "FISCAL IMPACT ANALYSIS" in headings


# ---------------------------------------------------------------------------
# Unit tests — chunk_record
# ---------------------------------------------------------------------------

def test_empty_text_produces_no_chunks():
    assert chunk_record(_make_record("")) == []


def test_stub_below_min_tokens_skipped():
    assert chunk_record(_make_record("Short.")) == []


def test_short_record_produces_single_chunk():
    text = "This is a substantial paragraph with enough tokens. " * 15
    chunks = chunk_record(_make_record(text))
    assert len(chunks) == 1
    assert chunks[0].chunk_total == 1
    assert chunks[0].source == "spac"


def test_all_caps_headings_split_into_separate_chunks():
    body = "Some content about the policy. " * 30
    text = (
        f"DRUG OFFENSE REFORM\n\n{body}\n\n"
        f"MANDATORY SUPERVISED RELEASE\n\n{body}"
    )
    chunks = chunk_record(_make_record(text))
    headings = {c.section_heading for c in chunks}
    assert "DRUG OFFENSE REFORM" in headings
    assert "MANDATORY SUPERVISED RELEASE" in headings


def test_two_word_heading_does_not_split():
    body = "Content paragraph. " * 20
    text = f"RETAIL THEFT\n\n{body}\n\nMore content follows here."
    chunks = chunk_record(_make_record(text))
    headings = {c.section_heading for c in chunks}
    assert "RETAIL THEFT" not in headings, "Two-word heading incorrectly triggered a split"


def test_enriched_text_contains_category_year_title():
    text = "Some policy content. " * 20
    chunks = chunk_record(_make_record(text, category="Cannabis Policy", year="2019", title="Cannabis Reform"))
    assert chunks
    assert "Cannabis Policy" in chunks[0].enriched_text
    assert "2019" in chunks[0].enriched_text
    assert "Cannabis Reform" in chunks[0].enriched_text


def test_enriched_text_contains_section_heading():
    body = "Content. " * 30
    text = f"DRUG OFFENSE REFORM\n\n{body}"
    chunks = chunk_record(_make_record(text))
    drug_chunks = [c for c in chunks if c.section_heading == "DRUG OFFENSE REFORM"]
    assert drug_chunks
    assert "DRUG OFFENSE REFORM" in drug_chunks[0].enriched_text


def test_no_chunk_exceeds_max_tokens():
    big_para = "The sentencing commission shall review all applicable standards. " * 50
    text = "\n\n".join([big_para] * 5)
    chunks = chunk_record(_make_record(text))
    over = [(c.chunk_id, c.token_count) for c in chunks if c.token_count > MAX_TOKENS]
    assert not over, f"Chunks exceed MAX_TOKENS={MAX_TOKENS}: {over}"


def test_chunk_ids_contiguous():
    body = "Content paragraph. " * 20
    text = (
        f"SECTION ONE HEADING\n\n{body}\n\n"
        f"SECTION TWO ANALYSIS\n\n{body}\n\n"
        f"SECTION THREE REFORM\n\n{body}"
    )
    chunks = chunk_record(_make_record(text))
    indices = [c.chunk_index for c in chunks]
    assert sorted(indices) == list(range(len(indices))), f"Non-contiguous: {indices}"
    for c in chunks:
        assert c.chunk_total == len(chunks)


def test_toc_lines_not_in_chunk_text():
    toc = "Introduction ......................... 4\nBackground ............................ 6\n"
    body = "Real content paragraph. " * 20
    chunks = chunk_record(_make_record(toc + body))
    for c in chunks:
        assert not _DOT_LEAK_RE.search(c.text), (
            f"Dot-leader line leaked into {c.chunk_id}: {c.text[:100]}"
        )


def test_page_headers_not_in_chunk_text():
    text = "Content A. " * 20 + "\nPage 3 of 15\n" + "Content B. " * 20
    chunks = chunk_record(_make_record(text))
    for c in chunks:
        assert not _PAGE_LEAK_RE.search(c.text), (
            f"Page header leaked into {c.chunk_id}: {c.text[:100]}"
        )


def test_chunk_id_format():
    text = "Substantial content paragraph. " * 15
    chunks = chunk_record(_make_record(text, rec_id="spac-my-report-2020"))
    assert chunks[0].chunk_id == "spac-my-report-2020_c0"


def test_source_field_is_spac():
    text = "Substantial content paragraph. " * 15
    chunks = chunk_record(_make_record(text))
    assert all(c.source == "spac" for c in chunks)


# ---------------------------------------------------------------------------
# Corpus-level property tests (require S3 via spac_chunks fixture)
# ---------------------------------------------------------------------------

def test_no_toc_lines_in_corpus_chunks(spac_chunks):
    failures = [
        c["chunk_id"] for c in spac_chunks
        if _DOT_LEAK_RE.search(c.get("text", ""))
    ]
    assert not failures, (
        f"{len(failures)} corpus chunks contain dot-leader ToC lines: {failures[:5]}"
    )


def test_no_page_headers_in_corpus_chunks(spac_chunks):
    failures = [
        c["chunk_id"] for c in spac_chunks
        if _PAGE_LEAK_RE.search(c.get("text", ""))
    ]
    assert not failures, (
        f"{len(failures)} corpus chunks contain page headers: {failures[:5]}"
    )


def test_chunk_index_contiguous(spac_chunks):
    by_record = defaultdict(list)
    for c in spac_chunks:
        by_record[c["record_id"]].append(c["chunk_index"])
    failures = [
        f"{rid}: {sorted(idxs)}"
        for rid, idxs in by_record.items()
        if set(idxs) != set(range(len(idxs)))
    ]
    assert not failures, "Non-contiguous chunk indices:\n" + "\n".join(failures[:5])


def test_chunk_total_accurate(spac_chunks):
    by_record = defaultdict(list)
    for c in spac_chunks:
        by_record[c["record_id"]].append(c)
    failures = [
        f"{c['chunk_id']}: chunk_total={c['chunk_total']} actual={len(siblings)}"
        for siblings in by_record.values()
        for c in siblings
        if c["chunk_total"] != len(siblings)
    ]
    assert not failures, "chunk_total mismatches:\n" + "\n".join(failures[:5])


def test_no_empty_chunks(spac_chunks):
    failures = [
        c["chunk_id"] for c in spac_chunks
        if c.get("token_count", 0) < MIN_CHUNK_TOKENS
    ]
    assert not failures, (
        f"{len(failures)} chunks below MIN_CHUNK_TOKENS={MIN_CHUNK_TOKENS}: {failures[:5]}"
    )


def test_chunk_ids_unique(spac_chunks):
    ids = [c["chunk_id"] for c in spac_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in SPAC chunks"


def test_no_chunk_exceeds_max_tokens_corpus(spac_chunks):
    failures = [
        (c["chunk_id"], c["token_count"]) for c in spac_chunks
        if c.get("token_count", 0) > MAX_TOKENS
    ]
    assert not failures, (
        f"{len(failures)} chunks exceed MAX_TOKENS={MAX_TOKENS}: {failures[:5]}"
    )


def test_all_chunks_have_required_fields(spac_chunks):
    required = {
        "chunk_id", "chunk_index", "chunk_total", "source",
        "text", "enriched_text", "token_count", "record_id",
        "title", "category", "year", "url",
    }
    failures = [
        f"{c['chunk_id']}: missing {required - c.keys()}"
        for c in spac_chunks
        if not required.issubset(c.keys())
    ]
    assert not failures, "Chunks missing required fields:\n" + "\n".join(failures[:5])


def test_enriched_text_nonempty(spac_chunks):
    failures = [
        c["chunk_id"] for c in spac_chunks
        if not c.get("enriched_text", "").strip()
    ]
    assert not failures, f"{len(failures)} chunks have empty enriched_text: {failures[:5]}"


def test_source_field_is_spac_corpus(spac_chunks):
    failures = [c["chunk_id"] for c in spac_chunks if c.get("source") != "spac"]
    assert not failures, f"{len(failures)} chunks have wrong source: {failures[:5]}"


def test_large_report_splits_into_multiple_chunks(spac_chunks):
    """The largest SPAC record (hb3355, ~193k chars) must produce many chunks."""
    target = [c for c in spac_chunks if "hb3355" in c.get("record_id", "")]
    if not target:
        pytest.skip("hb3355 record not found in chunked output")
    assert len(target) > 10, f"Expected >10 chunks for hb3355, got {len(target)}"
    headings = {c["section_heading"] for c in target if c["section_heading"]}
    assert len(headings) > 3, "Large report produced too few distinct section headings"


# ---------------------------------------------------------------------------
# S3 output verification (require chunked S3 bucket)
# ---------------------------------------------------------------------------

def test_s3_output_record_count(spac_chunks, spac_chunks_s3):
    assert len(spac_chunks_s3) == len(spac_chunks), (
        f"S3 has {len(spac_chunks_s3)} chunks, in-memory produced {len(spac_chunks)}"
    )


def test_s3_output_no_corrupt_records(spac_chunks_s3):
    required = {"chunk_id", "chunk_index", "chunk_total", "source", "text", "enriched_text"}
    failures = [
        f"record {i}: missing {required - c.keys()}"
        for i, c in enumerate(spac_chunks_s3)
        if not required.issubset(c.keys())
    ]
    assert not failures, "Corrupt records in S3 output:\n" + "\n".join(failures[:5])


def test_s3_output_no_empty_text(spac_chunks_s3):
    failures = [c["chunk_id"] for c in spac_chunks_s3 if not c.get("text", "").strip()]
    assert not failures, f"{len(failures)} chunks with empty text in S3: {failures[:5]}"


def test_s3_output_source_fields(spac_chunks_s3):
    failures = [c["chunk_id"] for c in spac_chunks_s3 if c.get("source") != "spac"]
    assert not failures, f"{len(failures)} S3 chunks have wrong source: {failures[:5]}"
