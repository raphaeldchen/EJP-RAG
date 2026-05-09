"""
Tests for BulkLoader.bulk_upsert SQL generation and value formatting.

These tests mock the psycopg2 connection so no real database is needed.
"""
import json
from unittest.mock import MagicMock, call, patch

import pytest

from embed.supabase_bulk_loader import BulkLoader, _format_value


# ---------------------------------------------------------------------------
# _format_value
# ---------------------------------------------------------------------------

def test_format_value_serializes_dict_to_json_string():
    result = _format_value("metadata", {"key": "val"})
    assert isinstance(result, str)
    assert json.loads(result) == {"key": "val"}


def test_format_value_formats_embedding_list_as_vector_string():
    result = _format_value("embedding", [0.1, 0.2, 0.3])
    assert result == "[0.1,0.2,0.3]"


def test_format_value_passes_strings_through():
    assert _format_value("chunk_id", "abc-123") == "abc-123"


def test_format_value_passes_none_through():
    assert _format_value("parent_id", None) is None


def test_format_value_passes_int_through():
    assert _format_value("chunk_index", 5) == 5


# ---------------------------------------------------------------------------
# BulkLoader.bulk_upsert — SQL construction and value formatting
# ---------------------------------------------------------------------------

def _make_payloads(n: int) -> list[dict]:
    return [
        {
            "chunk_id": f"c{i}",
            "source": "cap_bulk",
            "text": f"text {i}",
            "enriched_text": f"[header]\n\ntext {i}",
            "metadata": {"case_name": f"People v. Case {i}"},
            "embedding": [0.1] * 768,
        }
        for i in range(n)
    ]


def _loader_with_mock_conn():
    loader = BulkLoader.__new__(BulkLoader)
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    loader.conn = mock_conn
    loader.db_url = "postgresql://fake"
    loader.statement_timeout_ms = 600_000
    return loader, mock_conn, mock_cursor


def test_bulk_upsert_calls_execute_values():
    loader, _, _ = _loader_with_mock_conn()
    payloads = _make_payloads(3)

    with patch("embed.supabase_bulk_loader.execute_values") as mock_ev:
        count = loader.bulk_upsert(payloads, "opinion_chunks")

    assert count == 3
    mock_ev.assert_called_once()
    assert mock_ev.call_args.args[0] is not None  # cursor was passed


def test_bulk_upsert_template_casts_embedding_column():
    """The SQL template for the embedding column must contain ::vector."""
    loader, _, _ = _loader_with_mock_conn()
    payloads = _make_payloads(2)

    with patch("embed.supabase_bulk_loader.execute_values") as mock_ev:
        loader.bulk_upsert(payloads, "opinion_chunks")

    template = mock_ev.call_args.kwargs.get("template") or mock_ev.call_args.args[3]
    assert "::vector" in template


def test_bulk_upsert_serializes_metadata_dict():
    """metadata dicts must be JSON strings in the values tuples passed to execute_values."""
    loader, _, _ = _loader_with_mock_conn()
    payloads = _make_payloads(1)

    with patch("embed.supabase_bulk_loader.execute_values") as mock_ev:
        loader.bulk_upsert(payloads, "opinion_chunks")

    rows = mock_ev.call_args.args[2]
    assert len(rows) == 1
    row = rows[0]
    cols = list(payloads[0].keys())
    meta_idx = cols.index("metadata")
    assert isinstance(row[meta_idx], str)
    assert json.loads(row[meta_idx]) == {"case_name": "People v. Case 0"}


def test_bulk_upsert_formats_embedding_as_vector_string():
    """Embedding lists must be formatted as '[0.1,0.2,...]' in the values tuples."""
    loader, _, _ = _loader_with_mock_conn()
    payloads = _make_payloads(1)

    with patch("embed.supabase_bulk_loader.execute_values") as mock_ev:
        loader.bulk_upsert(payloads, "opinion_chunks")

    rows = mock_ev.call_args.args[2]
    cols = list(payloads[0].keys())
    embed_idx = cols.index("embedding")
    emb_val = rows[0][embed_idx]
    assert isinstance(emb_val, str)
    assert emb_val.startswith("[")
    assert emb_val.endswith("]")


def test_bulk_upsert_returns_zero_for_empty_input():
    loader, _, _ = _loader_with_mock_conn()
    assert loader.bulk_upsert([], "opinion_chunks") == 0


# ---------------------------------------------------------------------------
# BulkLoader.bulk_upsert — retry on OperationalError
# ---------------------------------------------------------------------------

def test_bulk_upsert_retries_on_operational_error():
    """OperationalError (connection drop) must trigger rollback + reconnect + retry."""
    import psycopg2

    loader, mock_conn, mock_cursor = _loader_with_mock_conn()

    call_count = 0

    def execute_values_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise psycopg2.OperationalError("connection reset by peer")
        # second call succeeds (do nothing)

    with patch("embed.supabase_bulk_loader.execute_values", side_effect=execute_values_side_effect), \
         patch.object(loader, "connect") as mock_reconnect, \
         patch("time.sleep"):
        count = loader.bulk_upsert(
            _make_payloads(5), "opinion_chunks",
            _max_retries=3, _base_delay=0.01,
        )

    assert count == 5
    assert call_count == 2  # first failed, second succeeded
    mock_conn.rollback.assert_called_once()
    mock_reconnect.assert_called_once()


def test_bulk_upsert_deduplicates_by_chunk_id():
    """Duplicate chunk_ids in the same batch must be collapsed to the last occurrence."""
    loader, _, _ = _loader_with_mock_conn()

    payloads = _make_payloads(2)
    # Add a duplicate of chunk c0 with different text (last wins)
    duplicate = dict(payloads[0])
    duplicate["text"] = "updated text"
    payloads.append(duplicate)

    with patch("embed.supabase_bulk_loader.execute_values") as mock_ev:
        count = loader.bulk_upsert(payloads, "opinion_chunks")

    rows = mock_ev.call_args.args[2]
    assert len(rows) == 2  # 3 payloads collapsed to 2 unique chunk_ids
    assert count == 2


def test_bulk_upsert_raises_after_max_retries():
    """After exhausting retries, the final OperationalError is re-raised."""
    import psycopg2

    loader, mock_conn, _ = _loader_with_mock_conn()

    with patch("embed.supabase_bulk_loader.execute_values",
               side_effect=psycopg2.OperationalError("persistent failure")), \
         patch.object(loader, "connect"), \
         patch("time.sleep"):
        with pytest.raises(psycopg2.OperationalError, match="persistent failure"):
            loader.bulk_upsert(_make_payloads(3), "opinion_chunks", _max_retries=2, _base_delay=0.01)

    assert mock_conn.rollback.call_count == 2  # rolled back on each failed attempt
