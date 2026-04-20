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


def test_small_chunks_not_orphaned(iscr_chunks):
    """Chunks with < 10 tokens must be header chunks or legitimately tiny legal content."""
    failures = []
    for chunk in iscr_chunks:
        if estimate_tokens(chunk.get("text", "")) < 10:
            ct = chunk.get("content_type")
            text = chunk.get("text", "")
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
                f"chunk_id={chunk['chunk_id']} content_type={ct!r} "
                f"text={text[:80]!r}"
            )
    assert not failures, (
        f"{len(failures)} junk micro-chunks found:\n" + "\n".join(failures[:5])
    )
