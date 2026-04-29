"""
Test suite for chunk/iac_chunk.py.

Two layers:
  Unit tests   — synthetic records; no S3 access; fast; catch IAC-specific logic
  Corpus tests — run against the full iac_corpus.jsonl pulled from S3 via fixtures
"""

import re
from collections import defaultdict

import pytest

from chunk.iac_chunk import (
    CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    chunk_section,
    normalize_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(text: str, section_num: str = "504.80") -> dict:
    return {
        "id":               f"iac-t20-p504-s{section_num.replace('.', '_')}",
        "source":           "illinois_admin_code",
        "title_num":        "20",
        "title_name":       "Corrections, Criminal Justice, and Law Enforcement",
        "part_num":         "504",
        "part_name":        "Inmate Discipline",
        "section_num":      section_num,
        "section_heading":  "Test Section",
        "section_citation": f"20 Ill. Adm. Code {section_num}",
        "url":              "https://example.com",
        "scraped_at":       "2026-04-20T00:00:00+00:00",
        "text":             text,
    }


_ORPHAN_RE = re.compile(r"^\s*[a-z]\)")


# ---------------------------------------------------------------------------
# Unit tests — normalize_text
# ---------------------------------------------------------------------------

def test_normalize_collapses_mid_heading_wrap():
    """HTML line-wrap inside a section heading should be joined to one line."""
    raw = "Section 504.80  Adjustment\nCommittee Hearing Procedures"
    result = normalize_text(raw)
    assert "\n" not in result
    assert "Adjustment Committee Hearing Procedures" in result


def test_normalize_collapses_mid_sentence_wrap():
    """HTML line-wrap mid-sentence should be joined; not treated as a paragraph break."""
    raw = "a) The Facility Publication Review Officer, upon receiving\na publication for review from mailroom staff, shall notify the individual."
    result = normalize_text(raw)
    lines = [l for l in result.splitlines() if l.strip()]
    # Should be exactly one line: the a) line with the continuation joined on
    assert len(lines) == 1
    assert "mailroom staff" in lines[0]


def test_normalize_preserves_letter_subsection_breaks():
    """a) b) c) boundaries must remain on their own lines after normalisation."""
    raw = "a) First subsection text.\nb) Second subsection text.\nc) Third subsection text."
    result = normalize_text(raw)
    lines = [l for l in result.splitlines() if l.strip()]
    assert lines[0].startswith("a)")
    assert lines[1].startswith("b)")
    assert lines[2].startswith("c)")


def test_normalize_preserves_numbered_item_breaks():
    """1) 2) 3) sub-items must land on their own lines."""
    raw = "a) Parent text.\n1) First item.\n2) Second item."
    result = normalize_text(raw)
    lines = [l for l in result.splitlines() if l.strip()]
    assert any(l.startswith("1)") for l in lines)
    assert any(l.startswith("2)") for l in lines)


def test_normalize_collapses_excessive_internal_whitespace():
    """'a)         The' should become 'a) The' (single space after marker)."""
    raw = "a)         The offender shall receive written notice."
    result = normalize_text(raw)
    assert not re.search(r"a\)\s{2,}", result), "Excessive whitespace after a) not collapsed"


def test_normalize_preserves_subpart_header():
    """SUBPART headers should be kept as distinct lines."""
    raw = "Some text.\nSUBPART B:  DIMINUTION OF SENTENCE\nMore text."
    result = normalize_text(raw)
    assert any("SUBPART B" in l for l in result.splitlines())


# ---------------------------------------------------------------------------
# Unit tests — chunk_section
# ---------------------------------------------------------------------------

def test_stub_section_skipped():
    """A record whose text is only the section heading produces no chunks."""
    rec = _make_record("Section 107.10  Applicability")
    assert chunk_section(rec) == []


def test_short_but_real_section_single_chunk():
    """A section with real content under CHUNK_SIZE produces exactly one chunk."""
    rec = _make_record(
        "Section 504.10  Applicability\n"
        "This Part applies to all adult correctional facilities within the Department."
    )
    chunks = chunk_section(rec)
    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["chunk_total"] == 1


def test_enumeration_not_severed():
    """
    A section with multiple a) b) c) subsections that together exceed CHUNK_SIZE
    must not produce any chunk at index > 0 that starts with a bare letter marker.
    """
    filler = "Word " * 80  # ~400 chars per subsection
    text = (
        "Section 504.80  Hearing Procedures\n"
        f"a) {filler}\n"
        f"b) {filler}\n"
        f"c) {filler}\n"
    )
    chunks = chunk_section(_make_record(text))
    assert chunks, "No chunks produced"
    for chunk in chunks:
        if chunk["chunk_index"] > 0:
            assert not _ORPHAN_RE.match(chunk["text"]), (
                f"Chunk {chunk['chunk_id']} (index {chunk['chunk_index']}) starts with "
                f"orphaned subsection marker:\n{chunk['text'][:120]}"
            )


def test_numeric_subitems_stay_with_parent():
    """
    Numbered sub-items (1) 2) 3)) must not trigger a split — they belong
    to their parent letter subsection.
    """
    filler = "Word " * 20
    text = (
        "Section 504.80  Hearing Procedures\n"
        f"a) {filler}\n"
        f"1) {filler}\n"
        f"2) {filler}\n"
        f"b) {filler}\n"
    )
    chunks = chunk_section(_make_record(text))
    # Combined text should fit in one chunk; the 1) 2) items must not start a chunk
    assert len(chunks) == 1, (
        f"Expected 1 chunk (sub-items should stay with parent), got {len(chunks)}"
    )


def test_enriched_text_contains_hierarchy():
    """enriched_text must include the IAC title/part/section context for embedding."""
    rec = _make_record(
        "Section 504.80  Adjustment Committee Hearing Procedures\n"
        "a) The hearing shall be convened within 14 days after the commission of the offense."
    )
    chunks = chunk_section(rec)
    assert chunks
    enriched = chunks[0]["enriched_text"]
    assert "Title 20" in enriched
    assert "20 Ill. Adm. Code 504.80" in enriched
    assert "Part 504" in enriched


def test_chunk_ids_have_correct_format():
    """chunk_id must be '{parent_id}_c{index}'."""
    rec = _make_record(
        "Section 504.80  Applicability\n"
        "a) This Part applies to all adult correctional facilities."
    )
    chunks = chunk_section(rec)
    for chunk in chunks:
        expected_prefix = f"{rec['id']}_c{chunk['chunk_index']}"
        assert chunk["chunk_id"] == expected_prefix, (
            f"Unexpected chunk_id: {chunk['chunk_id']!r}"
        )


def test_wrap_artifact_does_not_produce_stub_chunk():
    """
    A section where a line-wrap artifact would otherwise look like a paragraph
    break must not produce undersized chunks after normalisation.
    """
    # Simulate "Adjustment\nCommittee" as seen in real IAC data
    text = (
        "Section 525.233  Procedures for\n"
        "Review of Publications\n"
        "a) The Facility Publication Review Officer, upon receiving a publication for\n"
        "an individual in custody to review from mailroom staff, shall notify\n"
        "the individual in custody."
    )
    chunks = chunk_section(_make_record(text, section_num="525.233"))
    assert chunks, "No chunks produced"
    for chunk in chunks:
        assert len(chunk["text"]) >= MIN_CHUNK_SIZE, (
            f"Chunk {chunk['chunk_id']} is undersized ({len(chunk['text'])} chars): "
            f"{chunk['text']!r}"
        )


# ---------------------------------------------------------------------------
# Corpus-level property tests (require S3 access via iac_chunks fixture)
# ---------------------------------------------------------------------------

def test_no_orphaned_subsection_starts(iac_chunks):
    """No chunk at index > 0 should start with a bare IAC letter subsection marker."""
    failures = [
        c["chunk_id"]
        for c in iac_chunks
        if c["chunk_index"] > 0 and _ORPHAN_RE.match(c["text"])
    ]
    assert not failures, (
        f"{len(failures)} chunks start with orphaned subsection markers: {failures[:5]}"
    )


def test_chunk_index_contiguous(iac_chunks):
    by_parent = defaultdict(list)
    for chunk in iac_chunks:
        by_parent[chunk["parent_id"]].append(chunk["chunk_index"])
    failures = [
        f"{pid}: {sorted(idxs)}"
        for pid, idxs in by_parent.items()
        if set(idxs) != set(range(len(idxs)))
    ]
    assert not failures, "Non-contiguous chunk indices:\n" + "\n".join(failures[:5])


def test_chunk_total_accurate(iac_chunks):
    by_parent = defaultdict(list)
    for chunk in iac_chunks:
        by_parent[chunk["parent_id"]].append(chunk)
    failures = [
        f"{c['chunk_id']}: chunk_total={c['chunk_total']} actual={len(siblings)}"
        for siblings in by_parent.values()
        for c in siblings
        if c["chunk_total"] != len(siblings)
    ]
    assert not failures, "chunk_total mismatches:\n" + "\n".join(failures[:5])


def test_no_empty_chunks(iac_chunks):
    failures = [c["chunk_id"] for c in iac_chunks if len(c["text"]) < MIN_CHUNK_SIZE]
    assert not failures, f"{len(failures)} chunks below MIN_CHUNK_SIZE: {failures[:5]}"


def test_chunk_ids_unique(iac_chunks):
    ids = [c["chunk_id"] for c in iac_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in IAC chunks"


def test_no_chunk_exceeds_max_size(iac_chunks):
    failures = [
        (c["chunk_id"], len(c["text"]))
        for c in iac_chunks
        if len(c["text"]) > CHUNK_SIZE
    ]
    assert not failures, (
        f"{len(failures)} chunks exceed CHUNK_SIZE={CHUNK_SIZE}: {failures[:5]}"
    )


def test_all_chunks_have_required_metadata(iac_chunks):
    required = {"section_id", "source", "title_num", "part_num", "section_citation"}
    failures = [
        f"{c['chunk_id']}: missing {required - c['metadata'].keys()}"
        for c in iac_chunks
        if not required.issubset(c["metadata"].keys())
    ]
    assert not failures, "Chunks missing required metadata fields:\n" + "\n".join(failures[:5])


def test_all_chunks_enriched_text_contains_citation(iac_chunks):
    """Every enriched_text should contain the section citation for retrieval context."""
    failures = [
        c["chunk_id"]
        for c in iac_chunks
        if c["metadata"]["section_citation"] not in c.get("enriched_text", "")
    ]
    assert not failures, (
        f"{len(failures)} chunks missing section citation in enriched_text: {failures[:5]}"
    )


def test_adjustment_committee_section_splits_correctly(iac_chunks):  # noqa: F811
    """
    20 Ill. Adm. Code 504.80 (Adjustment Committee Hearing Procedures) is the
    longest section in the corpus (~14k chars). It must produce multiple chunks
    and none at index > 0 should start with a bare letter-subsection marker.
    """
    target_chunks = [
        c for c in iac_chunks
        if "504.80" in c["metadata"].get("section_citation", "")
    ]
    if not target_chunks:
        pytest.skip("20 Ill. Adm. Code 504.80 not found in chunked output")
    assert len(target_chunks) > 1, "504.80 should produce multiple chunks given its length"
    for chunk in target_chunks:
        if chunk["chunk_index"] > 0:
            assert not _ORPHAN_RE.match(chunk["text"]), (
                f"504.80 chunk {chunk['chunk_id']} starts with orphaned marker:\n"
                f"{chunk['text'][:120]}"
            )


# ---------------------------------------------------------------------------
# S3 output verification — reads the actual chunked file from the chunked bucket
# ---------------------------------------------------------------------------

def test_s3_output_record_count(iac_chunks, iac_chunks_s3):
    """Chunk count in S3 must match what the chunker produces in memory."""
    assert len(iac_chunks_s3) == len(iac_chunks), (
        f"S3 has {len(iac_chunks_s3)} chunks, in-memory produced {len(iac_chunks)}"
    )


def test_s3_output_no_corrupt_records(iac_chunks_s3):
    """Every record in S3 must deserialise cleanly and have the required top-level keys."""
    required = {"chunk_id", "chunk_index", "chunk_total", "text", "enriched_text", "metadata"}
    failures = [
        f"record {i}: missing {required - c.keys()}"
        for i, c in enumerate(iac_chunks_s3)
        if not required.issubset(c.keys())
    ]
    assert not failures, "Corrupt records in S3 output:\n" + "\n".join(failures[:5])


def test_s3_output_no_empty_text(iac_chunks_s3):
    """No chunk in S3 should have empty text — would indicate a write truncation."""
    failures = [c["chunk_id"] for c in iac_chunks_s3 if not c.get("text", "").strip()]
    assert not failures, f"{len(failures)} chunks with empty text in S3: {failures[:5]}"


def test_s3_output_source_is_iac(iac_chunks_s3):
    """All S3 chunks must have source=illinois_admin_code in metadata."""
    failures = [
        c["chunk_id"] for c in iac_chunks_s3
        if c.get("metadata", {}).get("source") != "illinois_admin_code"
    ]
    assert not failures, f"{len(failures)} chunks with wrong source in S3: {failures[:5]}"
