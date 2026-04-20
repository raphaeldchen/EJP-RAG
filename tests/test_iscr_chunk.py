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
