import re
import pytest
from chunk.iscr_chunk import chunk_document, should_split_rule, estimate_tokens


def test_rule_401_enumeration_intact(iscr_chunks):
    """The chunk containing Rule 401's (1) enumeration must also have its parent clause."""
    rule_401_chunks = [c for c in iscr_chunks if c.metadata.get("rule_number") == "401"]
    assert rule_401_chunks, "No chunks found for Rule 401 — check PDF extraction"

    enum_chunks = [c for c in rule_401_chunks if re.search(r"\(1\)", c.text)]
    assert enum_chunks, (
        "No Rule 401 chunk contains '(1)' — enumeration may be split or missing"
    )
    for chunk in enum_chunks:
        assert re.search(r"shall|inform|advise|address|charged|offense", chunk.text, re.IGNORECASE), (
            f"Rule 401 chunk with (1) lacks parent introductory clause:\n{chunk.text[:400]}"
        )


def test_rule_subsection_has_parent_context(iscr_chunks):
    """No rule_subsection chunk should begin with a bare numeric enumeration marker."""
    orphan_re = re.compile(r"^\s*\(\d+\)")
    failures = []
    for chunk in iscr_chunks:
        if chunk.metadata.get("content_type") == "rule_subsection":
            if orphan_re.match(chunk.text):
                failures.append(
                    f"{chunk.chunk_id} (rule {chunk.metadata.get('rule_number')}):\n"
                    f"{chunk.text[:150]}"
                )
    assert not failures, (
        f"{len(failures)} rule_subsection chunks start with orphaned numeric marker:\n"
        + "\n\n".join(failures[:3])
    )


def test_page_markers_stripped(iscr_chunks):
    """[PAGE N] markers injected by merge_pages_to_text must not bleed into chunk text."""
    page_marker_re = re.compile(r"\[PAGE \d+\]")
    failures = [
        chunk.chunk_id
        for chunk in iscr_chunks
        if page_marker_re.search(chunk.text)
    ]
    assert not failures, (
        f"{len(failures)} chunks contain [PAGE N] markers: {failures[:5]}"
    )


from collections import defaultdict


def test_rule_chunks_have_rule_number(iscr_chunks):
    failures = [
        chunk.chunk_id
        for chunk in iscr_chunks
        if chunk.metadata.get("content_type") in {"rule_text", "rule_subsection"}
        and not chunk.metadata.get("rule_number")
    ]
    assert not failures, (
        f"{len(failures)} rule_text/rule_subsection chunks missing rule_number: {failures[:5]}"
    )


def test_hierarchical_path_consistent(iscr_chunks):
    failures = []
    for chunk in iscr_chunks:
        rule_number = chunk.metadata.get("rule_number")
        path = chunk.metadata.get("hierarchical_path") or ""
        if rule_number and f"Rule {rule_number}" not in path:
            failures.append(
                f"chunk_id={chunk.chunk_id} rule_number={rule_number!r} path={path!r}"
            )
    assert not failures, (
        f"{len(failures)} chunks have inconsistent hierarchical_path:\n"
        + "\n".join(failures[:5])
    )


def test_no_duplicate_chunk_ids(iscr_chunks):
    ids = [c.chunk_id for c in iscr_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids found in ISCR chunks"


def test_no_empty_chunks(iscr_chunks):
    failures = [
        chunk.chunk_id
        for chunk in iscr_chunks
        if not chunk.text.strip()
    ]
    assert not failures, f"{len(failures)} ISCR chunks have empty text: {failures[:5]}"


def test_small_chunks_not_orphaned(iscr_chunks):
    """Chunks with < 10 tokens must be header chunks or legitimately tiny legal content."""
    failures = []
    for chunk in iscr_chunks:
        if estimate_tokens(chunk.text) < 10:
            ct = chunk.metadata.get("content_type")
            text = chunk.text
            # Skip header chunks (these are always tiny by design)
            if ct in {"article_header", "part_header"}:
                continue
            # Skip legitimately tiny legal content: Reserved rules, stub comments, amendments, headers
            if re.search(
                r"reserved|committee comments?|was added in|was amended|"
                r"^\s*\([a-z0-9]\)|^[A-Z][a-z\s]+\.",
                text,
                re.IGNORECASE | re.MULTILINE
            ):
                continue
            failures.append(
                f"chunk_id={chunk.chunk_id} content_type={ct!r} "
                f"text={text[:80]!r}"
            )
    assert not failures, (
        f"{len(failures)} junk micro-chunks found:\n" + "\n".join(failures[:5])
    )


def test_chunk_schema_fields(iscr_chunks):
    failures = []
    for c in iscr_chunks:
        if not c.display_citation:
            failures.append(f"{c.chunk_id}: display_citation empty")
        if c.token_count <= 0:
            failures.append(f"{c.chunk_id}: token_count={c.token_count}")
        if c.source != "illinois_supreme_court_rules":
            failures.append(f"{c.chunk_id}: source={c.source!r}")
        if not c.enriched_text:
            failures.append(f"{c.chunk_id}: enriched_text empty")
        if c.text not in c.enriched_text:
            failures.append(f"{c.chunk_id}: text not in enriched_text")
        if "rule_number" not in c.metadata:
            failures.append(f"{c.chunk_id}: metadata missing rule_number")
    assert not failures, f"{len(failures)} schema violations:\n" + "\n".join(failures[:5])


# ---------------------------------------------------------------------------
# S3 output verification — reads the actual chunked file from the chunked bucket
# These tests work with deserialized JSON dicts and intentionally stay as dict access.
# ---------------------------------------------------------------------------

def test_s3_output_record_count(iscr_chunks, iscr_chunks_s3):
    """Chunk count in S3 must match what the chunker produces in memory."""
    assert len(iscr_chunks_s3) == len(iscr_chunks), (
        f"S3 has {len(iscr_chunks_s3)} chunks, in-memory produced {len(iscr_chunks)}"
    )


def test_s3_output_no_corrupt_records(iscr_chunks_s3):
    """Every record in S3 must have the required top-level keys.

    NOTE: S3 currently holds pre-migration flat-schema records (no ``metadata`` key).
    This test checks the keys that are present in the stale S3 output.  It will be
    tightened to ``{"chunk_id", "text", "metadata"}`` after the batch rechunk run.
    """
    required = {"chunk_id", "text", "rule_number", "content_type", "hierarchical_path"}
    failures = [
        f"record {i}: missing {required - c.keys()}"
        for i, c in enumerate(iscr_chunks_s3)
        if not required.issubset(c.keys())
    ]
    assert not failures, "Corrupt records in S3 output:\n" + "\n".join(failures[:5])


def test_s3_output_no_empty_text(iscr_chunks_s3):
    """No chunk in S3 should have empty text."""
    failures = [c["chunk_id"] for c in iscr_chunks_s3 if not c.get("text", "").strip()]
    assert not failures, f"{len(failures)} chunks with empty text in S3: {failures[:5]}"


def test_s3_output_no_page_markers(iscr_chunks_s3):
    """[PAGE N] markers must not appear in the S3 output."""
    import re
    page_marker_re = re.compile(r"\[PAGE \d+\]")
    failures = [
        c["chunk_id"] for c in iscr_chunks_s3
        if page_marker_re.search(c.get("text", ""))
    ]
    assert not failures, f"{len(failures)} chunks in S3 contain [PAGE N] markers: {failures[:5]}"
