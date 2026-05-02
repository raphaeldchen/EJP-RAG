"""
Tests for chunk/merge_opinion_chunks.py.

Pure unit tests — operates on synthetic dicts; no S3 access.
Tests cover: bulk-only, API-only, conflict resolution, and the
backward-compatibility shim that handles both parent_id (new) and
opinion_id (legacy courtlistener_api.py output) field names.
"""

import pytest

from chunk.merge_opinion_chunks import merge


def _bulk(opinion_id: str, n_chunks: int = 1, tokens_each: int = 100) -> list[dict]:
    return [
        {
            "parent_id":   opinion_id,
            "chunk_id":    f"{opinion_id}_c{i}",
            "chunk_index": i,
            "chunk_total": n_chunks,
            "token_count": tokens_each,
            "source":      "courtlistener",
            "text":        f"bulk chunk {i}",
        }
        for i in range(n_chunks)
    ]


def _api(opinion_id: str, n_chunks: int = 1, tokens_each: int = 100) -> list[dict]:
    """Legacy format from courtlistener_api.py — uses opinion_id instead of parent_id."""
    return [
        {
            "opinion_id":  opinion_id,
            "chunk_id":    f"api_{opinion_id}_{i}",
            "chunk_index": i,
            "chunk_total": n_chunks,
            "token_count": tokens_each,
            "source":      "courtlistener",
            "text":        f"api chunk {i}",
        }
        for i in range(n_chunks)
    ]


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------


def test_both_empty():
    assert merge([], []) == []


def test_bulk_only():
    chunks = _bulk("op-1", n_chunks=3)
    result = merge(bulk=chunks, api=[])
    assert len(result) == 3
    assert all("bulk chunk" in c["text"] for c in result)


def test_api_only():
    chunks = _api("op-2", n_chunks=2)
    result = merge(bulk=[], api=chunks)
    assert len(result) == 2
    assert all("api chunk" in c["text"] for c in result)


# ---------------------------------------------------------------------------
# Conflict resolution (same opinion in both)
# ---------------------------------------------------------------------------


def test_bulk_wins_when_more_tokens():
    bulk = _bulk("op-3", n_chunks=2, tokens_each=300)   # total 600
    api  = _api("op-3",  n_chunks=1, tokens_each=100)   # total 100
    result = merge(bulk=bulk, api=api)
    assert len(result) == 2
    assert all("bulk chunk" in c["text"] for c in result)


def test_api_wins_when_more_tokens():
    bulk = _bulk("op-4", n_chunks=1, tokens_each=50)    # total 50
    api  = _api("op-4",  n_chunks=3, tokens_each=200)   # total 600
    result = merge(bulk=bulk, api=api)
    assert len(result) == 3
    assert all("api chunk" in c["text"] for c in result)


def test_api_wins_on_tie():
    bulk = _bulk("op-5", n_chunks=2, tokens_each=100)   # total 200
    api  = _api("op-5",  n_chunks=2, tokens_each=100)   # total 200 (equal)
    result = merge(bulk=bulk, api=api)
    assert len(result) == 2
    assert all("api chunk" in c["text"] for c in result)


# ---------------------------------------------------------------------------
# Multiple opinions, some overlapping
# ---------------------------------------------------------------------------


def test_multiple_opinions_no_overlap():
    bulk = _bulk("op-A") + _bulk("op-B")
    api  = _api("op-C")  + _api("op-D")
    result = merge(bulk=bulk, api=api)
    assert len(result) == 4


def test_multiple_opinions_partial_overlap():
    bulk = _bulk("op-A", n_chunks=2) + _bulk("op-B", n_chunks=1)
    api  = _api("op-A",  n_chunks=1) + _api("op-C",  n_chunks=3)
    # op-A: bulk=2×100=200, api=1×100=100 → bulk wins
    # op-B: bulk only
    # op-C: api only
    result = merge(bulk=bulk, api=api)
    assert len(result) == 2 + 1 + 3  # op-A(bulk) + op-B + op-C


# ---------------------------------------------------------------------------
# Backward-compatibility: new parent_id (bulk) vs old opinion_id (api)
# ---------------------------------------------------------------------------


def test_new_bulk_parent_id_field():
    """Bulk chunks now use parent_id; merge must still group them correctly."""
    bulk = _bulk("op-NEW", n_chunks=2)
    assert "parent_id" in bulk[0]
    assert "opinion_id" not in bulk[0]
    result = merge(bulk=bulk, api=[])
    assert len(result) == 2


def test_old_api_opinion_id_field():
    """API chunks still use opinion_id; merge must still group them correctly."""
    api = _api("op-OLD", n_chunks=2)
    assert "opinion_id" in api[0]
    assert "parent_id" not in api[0]
    result = merge(bulk=[], api=api)
    assert len(result) == 2


def test_conflict_resolution_across_field_name_formats():
    """Bulk (parent_id) vs API (opinion_id) for same opinion → correct winner."""
    bulk = _bulk("op-X", n_chunks=1, tokens_each=50)    # total 50
    api  = _api("op-X",  n_chunks=2, tokens_each=100)   # total 200 → api wins
    result = merge(bulk=bulk, api=api)
    assert len(result) == 2
    assert all("api chunk" in c["text"] for c in result)


def test_output_preserves_all_chunk_fields():
    """merge() must not strip any fields from the winning chunks."""
    bulk = _bulk("op-Z")
    result = merge(bulk=bulk, api=[])
    assert result[0]["chunk_id"] == "op-Z_c0"
    assert result[0]["chunk_total"] == 1
    assert result[0]["source"] == "courtlistener"
