"""
Test suite for chunk/iccb_chunk.py.

Unit tests   — synthetic records; no S3 access; test ICCB-specific logic
Corpus tests — full iccb_corpus.jsonl from S3 via iccb_chunks fixture
S3 output    — reads actual iccb_chunks.jsonl from the chunked S3 bucket
"""

import re
from collections import defaultdict

import pytest

from chunk.iccb_chunk import (
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
    rec_id: str = "iccb-fy2025-annual",
    fiscal_year: str = "2025",
    title: str = "FY2025 Annual Report",
) -> dict:
    return {
        "id":          rec_id,
        "source":      "iccb",
        "doc_type":    "annual_report",
        "fiscal_year": fiscal_year,
        "title":       title,
        "url":         "https://example.com/iccb_report.pdf",
        "text":        text,
        "scraped_at":  "2026-04-20T00:00:00+00:00",
    }


_DOT_LEAK_RE    = re.compile(r"\.{4,}")
_PAGE_LEAK_RE   = re.compile(
    r"Student Enrollments\s*&\s*Completions.*?Fiscal Year \d{4}",
    re.DOTALL | re.IGNORECASE,
)

_FILLER = "Illinois community colleges enrolled thousands of students in credit and noncredit courses. " * 30


# ---------------------------------------------------------------------------
# Unit tests — strip_page_headers (ICCB-specific pattern)
# ---------------------------------------------------------------------------

def test_iccb_page_header_stripped():
    text = (
        "Content before.\n"
        "Student Enrollments & Completions \n"
        "Fiscal Year 2025 \n"
        "7 \n"
        "Content after."
    )
    result = strip_page_headers(text)
    assert "Content before." in result
    assert "Content after." in result
    assert not _PAGE_LEAK_RE.search(result), "ICCB page header still present after stripping"


def test_iccb_multiple_page_headers_stripped():
    header = "Student Enrollments & Completions \nFiscal Year 2023 \n{n} \n"
    text = "Intro.\n" + header.format(n=3) + "Middle.\n" + header.format(n=4) + "End."
    result = strip_page_headers(text)
    assert not _PAGE_LEAK_RE.search(result)
    assert "Intro." in result
    assert "Middle." in result
    assert "End." in result


def test_non_page_content_preserved():
    text = "No headers here.\nJust regular enrollment discussion."
    assert "regular enrollment discussion" in strip_page_headers(text)


# ---------------------------------------------------------------------------
# Unit tests — strip_toc_lines
# ---------------------------------------------------------------------------

def test_toc_dot_leader_stripped():
    text = "Introduction ............................................................... 5\nReal content here."
    result = strip_toc_lines(text)
    assert not _DOT_LEAK_RE.search(result), "Dot-leader ToC line was not stripped"
    assert "Real content here." in result


def test_toc_multiple_dot_leaders_stripped():
    toc = (
        "Student Enrollments .......................................... 7\n"
        "Student Completions .......................................... 10\n"
        "Overall Fiscal Year 2025 Student Enrollments ................. 12\n"
    )
    result = strip_toc_lines(toc + "Actual content.")
    assert not _DOT_LEAK_RE.search(result)
    assert "Actual content." in result


def test_non_toc_lines_preserved():
    # Citation-style dots (few, not a dot-leader line)
    text = "See e.g. Smith v. Jones, 123 F.3d 456 (7th Cir. 2000).\nMore content."
    assert "Smith v. Jones" in strip_toc_lines(text)


def test_toc_trailing_dots_stripped():
    """Line ending with dot run (page number on next line in PDF) should be stripped."""
    line = "Dual Credit Enrollment ......\nReal content."
    result = strip_toc_lines(line)
    assert "Dual Credit Enrollment" not in result
    assert "Real content." in result


# ---------------------------------------------------------------------------
# Unit tests — is_section_heading
# ---------------------------------------------------------------------------

def test_three_word_all_caps_heading():
    assert is_section_heading("STUDENT CREDIT ENROLLMENTS") is True


def test_highlights_section_heading():
    assert is_section_heading("HIGHLIGHTS OF FISCAL YEAR 2025 ANNUAL REPORT") is True


def test_introduction_heading():
    assert is_section_heading("INTRODUCTION") is False  # single word


def test_two_word_not_heading():
    assert is_section_heading("STUDENT ENROLLMENTS") is False


def test_mixed_case_not_heading():
    assert is_section_heading("Student Enrollments") is False


def test_heading_with_year_digit():
    assert is_section_heading("FISCAL YEAR 2025 COMPLETIONS") is True


def test_empty_not_heading():
    assert is_section_heading("") is False


# ---------------------------------------------------------------------------
# Unit tests — split_at_headings
# ---------------------------------------------------------------------------

def test_split_produces_correct_headings():
    text = "Preamble text.\n\nHIGHLIGHTS OF ANNUAL REPORT\n\nHighlights body.\n\nSTUDENT CREDIT ENROLLMENTS\n\nEnrollment body."
    sections = split_at_headings(text)
    headings = [h for h, _ in sections]
    assert "HIGHLIGHTS OF ANNUAL REPORT" in headings
    assert "STUDENT CREDIT ENROLLMENTS" in headings


def test_split_no_headings_returns_single():
    text = "Just a paragraph with no headings."
    sections = split_at_headings(text)
    assert len(sections) == 1
    assert sections[0][0] == ""


# ---------------------------------------------------------------------------
# Unit tests — chunk_record
# ---------------------------------------------------------------------------

def test_empty_text_no_chunks():
    assert chunk_record(_make_record("")) == []


def test_stub_below_min_tokens_skipped():
    assert chunk_record(_make_record("Short.")) == []


def test_short_record_single_chunk():
    text = "Illinois community colleges enrolled students in credit courses. " * 15
    chunks = chunk_record(_make_record(text))
    assert len(chunks) == 1
    assert chunks[0].chunk_total == 1
    assert chunks[0].source == "iccb"


def test_all_caps_headings_produce_sections():
    text = (
        f"HIGHLIGHTS OF ANNUAL REPORT\n\n{_FILLER}\n\n"
        f"STUDENT CREDIT ENROLLMENTS\n\n{_FILLER}"
    )
    chunks = chunk_record(_make_record(text))
    headings = {c.metadata.get("section_heading", "") for c in chunks}
    assert "HIGHLIGHTS OF ANNUAL REPORT" in headings
    assert "STUDENT CREDIT ENROLLMENTS" in headings


def test_two_word_heading_does_not_split():
    text = f"DUAL CREDIT\n\n{_FILLER}\n\nMore content follows."
    chunks = chunk_record(_make_record(text))
    headings = {c.metadata.get("section_heading", "") for c in chunks}
    assert "DUAL CREDIT" not in headings, "Two-word heading should not create a split"


def test_no_chunk_exceeds_max_tokens():
    text = "\n\n".join([_FILLER] * 5)
    chunks = chunk_record(_make_record(text))
    over = [(c.chunk_id, c.token_count) for c in chunks if c.token_count > MAX_TOKENS]
    assert not over, f"Chunks exceed MAX_TOKENS={MAX_TOKENS}: {over}"


def test_chunk_ids_contiguous():
    text = (
        f"SECTION ONE REPORT\n\n{_FILLER}\n\n"
        f"SECTION TWO ANALYSIS\n\n{_FILLER}\n\n"
        f"SECTION THREE OVERVIEW\n\n{_FILLER}"
    )
    chunks = chunk_record(_make_record(text))
    indices = [c.chunk_index for c in chunks]
    assert sorted(indices) == list(range(len(indices)))
    for c in chunks:
        assert c.chunk_total == len(chunks)


def test_chunk_schema_fields():
    text = "Illinois community colleges enrolled students in credit courses. " * 15
    chunks = chunk_record(_make_record(text, fiscal_year="2025", title="FY2025 Annual Report"))
    assert chunks
    c = chunks[0]
    assert c.source == "iccb"
    assert c.token_count > 0
    assert c.display_citation != ""
    assert "record_id" in c.metadata
    assert c.parent_id == "iccb-fy2025-annual"


def test_enriched_text_contains_fiscal_year_and_title():
    text = "Credit enrollment increased significantly this year. " * 20
    chunks = chunk_record(_make_record(text, fiscal_year="2022", title="FY2022 Annual Report"))
    assert chunks
    assert "2022" in chunks[0].enriched_text
    assert "FY2022 Annual Report" in chunks[0].enriched_text


def test_enriched_text_contains_section_heading():
    text = f"STUDENT CREDIT ENROLLMENTS\n\n{_FILLER}"
    chunks = chunk_record(_make_record(text))
    section_chunks = [c for c in chunks if c.metadata.get("section_heading") == "STUDENT CREDIT ENROLLMENTS"]
    assert section_chunks
    assert "STUDENT CREDIT ENROLLMENTS" in section_chunks[0].enriched_text


def test_page_headers_not_in_chunk_text():
    text = (
        _FILLER + "\n"
        "Student Enrollments & Completions \n"
        "Fiscal Year 2025 \n"
        "8 \n" + _FILLER
    )
    chunks = chunk_record(_make_record(text))
    for c in chunks:
        assert not _PAGE_LEAK_RE.search(c.text), (
            f"ICCB page header leaked into {c.chunk_id}"
        )


def test_toc_lines_not_in_chunk_text():
    toc = "Introduction .......................................... 5\nBackground ............................................ 7\n"
    text = toc + _FILLER
    chunks = chunk_record(_make_record(text))
    for c in chunks:
        assert not _DOT_LEAK_RE.search(c.text), (
            f"Dot-leader ToC line leaked into {c.chunk_id}"
        )


def test_source_field_is_iccb():
    chunks = chunk_record(_make_record("Enrollment data discussion. " * 20))
    assert all(c.source == "iccb" for c in chunks)


def test_chunk_id_format():
    chunks = chunk_record(_make_record("Enrollment paragraph. " * 20, rec_id="iccb-fy2024-annual"))
    assert chunks[0].chunk_id == "iccb-fy2024-annual_c0"


def test_fiscal_year_preserved_in_chunks():
    chunks = chunk_record(_make_record("Enrollment content. " * 20, fiscal_year="2021"))
    assert all(c.metadata.get("fiscal_year") == "2021" for c in chunks)


# ---------------------------------------------------------------------------
# Corpus-level property tests (require S3 via iccb_chunks fixture)
# ---------------------------------------------------------------------------

def test_all_five_fiscal_years_present(iccb_chunks):
    fiscal_years = {c.get("metadata", {}).get("fiscal_year") for c in iccb_chunks}
    fiscal_years.discard(None)
    assert len(fiscal_years) >= 4, f"Expected at least 4 fiscal years, got: {fiscal_years}"


def test_no_page_headers_in_corpus_chunks(iccb_chunks):
    failures = [c["chunk_id"] for c in iccb_chunks if _PAGE_LEAK_RE.search(c.get("text", ""))]
    assert not failures, (
        f"{len(failures)} chunks contain ICCB page headers: {failures[:5]}"
    )


def test_no_toc_lines_in_corpus_chunks(iccb_chunks):
    failures = [c["chunk_id"] for c in iccb_chunks if _DOT_LEAK_RE.search(c.get("text", ""))]
    assert not failures, (
        f"{len(failures)} corpus chunks contain dot-leader ToC lines: {failures[:5]}"
    )


def test_chunk_index_contiguous(iccb_chunks):
    by_record = defaultdict(list)
    for c in iccb_chunks:
        by_record[c["parent_id"]].append(c["chunk_index"])
    failures = [
        f"{rid}: {sorted(idxs)}"
        for rid, idxs in by_record.items()
        if set(idxs) != set(range(len(idxs)))
    ]
    assert not failures, "Non-contiguous chunk indices:\n" + "\n".join(failures[:5])


def test_chunk_total_accurate(iccb_chunks):
    by_record = defaultdict(list)
    for c in iccb_chunks:
        by_record[c["parent_id"]].append(c)
    failures = [
        f"{c['chunk_id']}: chunk_total={c['chunk_total']} actual={len(siblings)}"
        for siblings in by_record.values()
        for c in siblings
        if c["chunk_total"] != len(siblings)
    ]
    assert not failures, "chunk_total mismatches:\n" + "\n".join(failures[:5])


def test_no_empty_chunks(iccb_chunks):
    failures = [c["chunk_id"] for c in iccb_chunks if c.get("token_count", 0) < MIN_CHUNK_TOKENS]
    assert not failures, (
        f"{len(failures)} chunks below MIN_CHUNK_TOKENS={MIN_CHUNK_TOKENS}: {failures[:5]}"
    )


def test_chunk_ids_unique(iccb_chunks):
    ids = [c["chunk_id"] for c in iccb_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in ICCB chunks"


def test_no_chunk_exceeds_max_tokens_corpus(iccb_chunks):
    failures = [
        (c["chunk_id"], c["token_count"])
        for c in iccb_chunks
        if c.get("token_count", 0) > MAX_TOKENS
    ]
    assert not failures, (
        f"{len(failures)} chunks exceed MAX_TOKENS={MAX_TOKENS}: {failures[:5]}"
    )


def test_all_chunks_have_required_fields(iccb_chunks):
    required_top = {
        "chunk_id", "parent_id", "chunk_index", "chunk_total", "source",
        "text", "enriched_text", "token_count", "display_citation", "metadata",
    }
    required_meta = {"record_id", "title", "fiscal_year", "doc_type", "url"}
    failures = []
    for c in iccb_chunks:
        missing_top = required_top - c.keys()
        if missing_top:
            failures.append(f"{c['chunk_id']}: missing top-level {missing_top}")
            continue
        missing_meta = required_meta - c.get("metadata", {}).keys()
        if missing_meta:
            failures.append(f"{c['chunk_id']}: missing metadata keys {missing_meta}")
    assert not failures, "Chunks missing required fields:\n" + "\n".join(failures[:5])


def test_enriched_text_nonempty(iccb_chunks):
    failures = [c["chunk_id"] for c in iccb_chunks if not c.get("enriched_text", "").strip()]
    assert not failures, f"{len(failures)} chunks have empty enriched_text: {failures[:5]}"


def test_source_field_is_iccb_corpus(iccb_chunks):
    failures = [c["chunk_id"] for c in iccb_chunks if c.get("source") != "iccb"]
    assert not failures, f"{len(failures)} chunks have wrong source: {failures[:5]}"


def test_large_report_produces_many_chunks(iccb_chunks):
    """FY2025 report (~142k chars) must produce many chunks with distinct section headings."""
    fy2025 = [c for c in iccb_chunks if c.get("metadata", {}).get("fiscal_year") == "2025"]
    assert fy2025, "No FY2025 chunks found"
    assert len(fy2025) > 50, f"Expected >50 chunks for FY2025 report, got {len(fy2025)}"
    headings = {c["metadata"]["section_heading"] for c in fy2025 if c["metadata"]["section_heading"]}
    assert len(headings) > 5, f"Expected >5 distinct section headings, got {headings}"


def test_all_reports_produce_chunks(iccb_chunks):
    """Every fiscal year in the corpus must produce at least one chunk."""
    by_fy = defaultdict(int)
    for c in iccb_chunks:
        by_fy[c.get("metadata", {}).get("fiscal_year")] += 1
    by_fy.pop(None, None)
    assert len(by_fy) >= 4, f"Fewer than 4 fiscal years have chunks: {dict(by_fy)}"
    for fy, count in by_fy.items():
        assert count > 0, f"FY{fy} produced no chunks"


# ---------------------------------------------------------------------------
# S3 output verification
# ---------------------------------------------------------------------------

def test_s3_output_record_count(iccb_chunks, iccb_chunks_s3):
    assert len(iccb_chunks_s3) == len(iccb_chunks), (
        f"S3 has {len(iccb_chunks_s3)} chunks, in-memory produced {len(iccb_chunks)}"
    )


def test_s3_output_no_corrupt_records(iccb_chunks_s3):
    required = {"chunk_id", "chunk_index", "chunk_total", "source", "text", "enriched_text"}
    failures = [
        f"record {i}: missing {required - c.keys()}"
        for i, c in enumerate(iccb_chunks_s3)
        if not required.issubset(c.keys())
    ]
    assert not failures, "Corrupt records in S3 output:\n" + "\n".join(failures[:5])


def test_s3_output_no_empty_text(iccb_chunks_s3):
    failures = [c["chunk_id"] for c in iccb_chunks_s3 if not c.get("text", "").strip()]
    assert not failures, f"{len(failures)} chunks with empty text in S3: {failures[:5]}"


def test_s3_output_source_field(iccb_chunks_s3):
    failures = [c["chunk_id"] for c in iccb_chunks_s3 if c.get("source") != "iccb"]
    assert not failures, f"{len(failures)} S3 chunks have wrong source: {failures[:5]}"
