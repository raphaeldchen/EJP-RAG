# Robust Bulk Embedding for Supabase Pro Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the critical `57014` statement-timeout misclassification in `flush_batch`, add `--batch-size` and `--max-chunks` flags, and add a `--direct-db` flag that routes bulk loads through a psycopg2 connection with statement-timeout override and optional vector-index drop/recreate — eliminating the class of server-timeout failures entirely for large loads.

**Architecture:** Three-layer fix. (1) Surgical bug fix: add `57014`/`statement timeout` to the availability-error path in `flush_batch` so timeouts get wait-and-retry instead of binary-split. Binary splitting is correctly **preserved** for non-availability errors (encoding errors, malformed data) — those still benefit from halving the batch. (2) CLI tuning: `--batch-size` and `--max-chunks` flags. (3) Direct PostgreSQL path: new `BulkLoader` class (psycopg2) that bypasses PostgREST, overrides `statement_timeout` per-session, retries on connection drops, and optionally drops+rebuilds vector indexes around the bulk load.

**Tech Stack:** psycopg2-binary, existing supabase Python client (unchanged for checkpoint queries and REST fallback), pytest + unittest.mock for tests

---

## Files

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `embed/batch_embed.py` | Fix `_is_availability_error`, add `--batch-size`/`--max-chunks`, add jitter, wire `BulkLoader` |
| Create | `embed/supabase_bulk_loader.py` | `BulkLoader`: connect, checkpoint, `bulk_upsert` (with retry), index management |
| Modify | `tests/test_batch_embed.py` | Tests for `flush_batch` error-classification + jitter |
| Create | `tests/test_supabase_bulk_loader.py` | Tests for `BulkLoader.bulk_upsert` payload formatting and retry logic |

---

## Task 1: Fix 57014 error classification and add jitter

**Files:**
- Modify: `embed/batch_embed.py:287-293` (`_SCHEMA_CACHE_ERRORS`, `_is_availability_error`)
- Modify: `embed/batch_embed.py:315-345` (`flush_batch` backoff delay)
- Test: `tests/test_batch_embed.py`

### Background

`_is_availability_error()` currently watches for `"PGRST002"`, `"schema cache"`, `"upstream connect error"`, `"503"`. PostgreSQL's statement-timeout exception message is `"canceling statement due to statement timeout"` and carries error code `57014`. Neither string appears in `_SCHEMA_CACHE_ERRORS`, so 57014 errors fall through to the binary-split branch — making timeout pressure *worse* by adding more round-trips. The fix is to add both the code and the message fragment to the same tuple.

Jitter is added to the exponential backoff to prevent multiple parallel processes from retrying in lockstep.

- [ ] **Step 1: Write failing tests for 57014 handling**

Add to `tests/test_batch_embed.py`:

```python
import time
from unittest.mock import MagicMock, patch, call

from embed.batch_embed import _is_availability_error, flush_batch


# ---------------------------------------------------------------------------
# _is_availability_error
# ---------------------------------------------------------------------------

def test_is_availability_error_catches_57014_code():
    assert _is_availability_error(Exception("canceling statement due to statement timeout (57014)"))


def test_is_availability_error_catches_statement_timeout_text():
    assert _is_availability_error(Exception("statement timeout exceeded"))


def test_is_availability_error_still_catches_pgrst002():
    assert _is_availability_error(Exception("PGRST002: schema reload pending"))


def test_is_availability_error_still_catches_503():
    assert _is_availability_error(Exception("upstream returned 503"))


def test_is_availability_error_false_for_encoding_error():
    assert not _is_availability_error(Exception("invalid byte sequence for encoding UTF8"))


# ---------------------------------------------------------------------------
# flush_batch — 57014 must NOT trigger binary split
# ---------------------------------------------------------------------------

def _make_batch(n: int) -> list[dict]:
    return [{"chunk_id": f"c{i}", "embedding": [0.1] * 768} for i in range(n)]


def test_flush_batch_retries_full_batch_on_57014():
    """A 57014 error must retry the same full batch, not split it."""
    supabase = MagicMock()
    supabase.table.return_value.upsert.return_value.execute.side_effect = [
        Exception("canceling statement due to statement timeout (57014)"),
        MagicMock(data=[]),  # success on retry
    ]
    batch = _make_batch(10)

    with patch("time.sleep"):
        result = flush_batch(
            supabase, batch, "opinion_chunks",
            _max_avail_retries=2, _base_avail_delay=0.01,
        )

    assert len(result) == 10
    # Only two calls — the initial attempt and one retry (full batch both times)
    assert supabase.table.return_value.upsert.call_count == 2
    # Both calls received the full 10-row batch, not halves
    for c in supabase.table.return_value.upsert.call_args_list:
        assert len(c.args[0]) == 10


def test_flush_batch_splits_on_non_timeout_error():
    """Non-timeout errors (e.g. encoding) should still trigger binary split."""
    supabase = MagicMock()
    supabase.table.return_value.upsert.return_value.execute.side_effect = [
        Exception("invalid byte sequence"),  # triggers split
        MagicMock(data=[]),                  # first half succeeds
        MagicMock(data=[]),                  # second half succeeds
    ]
    batch = _make_batch(4)

    with patch("time.sleep"):
        result = flush_batch(supabase, batch, "opinion_chunks")

    assert len(result) == 4
    # Three calls: 1 failed (4-row), then 2 halves (2-row each)
    assert supabase.table.return_value.upsert.call_count == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /Users/raphaelchen/Desktop/legal_rag
python3 -m pytest tests/test_batch_embed.py::test_is_availability_error_catches_57014_code \
                  tests/test_batch_embed.py::test_is_availability_error_catches_statement_timeout_text \
                  tests/test_batch_embed.py::test_flush_batch_retries_full_batch_on_57014 \
                  -v
```

Expected: `FAILED` — `assert _is_availability_error(Exception("...57014..."))` returns `False` with current code.

- [ ] **Step 3: Fix `_is_availability_error` in `batch_embed.py`**

Replace lines 287–292 in `embed/batch_embed.py`:

```python
_SCHEMA_CACHE_ERRORS = (
    "PGRST002", "schema cache", "upstream connect error", "503",
    "57014", "statement timeout",
)
```

- [ ] **Step 4: Add jitter to the availability-error backoff in `flush_batch`**

In `flush_batch`, replace the existing `delay = ...` / `time.sleep(delay)` lines (currently around line 327–329) with:

```python
import random  # add at top of file if not already present

delay = min(_base_avail_delay * (2 ** attempt), 300.0)
jitter = random.uniform(0, delay * 0.2)
time.sleep(delay + jitter)
```

- [ ] **Step 5: Run all new tests plus the full test_batch_embed suite**

```bash
python3 -m pytest tests/test_batch_embed.py -v
```

Expected: all tests pass (no regressions).

- [ ] **Step 6: Commit**

```bash
git add embed/batch_embed.py tests/test_batch_embed.py
git commit -m "fix: treat 57014 statement-timeout as availability error; add backoff jitter"
```

---

## Task 2: Add `--batch-size` and `--max-chunks` CLI flags

**Files:**
- Modify: `embed/batch_embed.py` — `embed_source` signature, `main`

The current hard-coded `BATCH_SIZE = 200` is conservative (free-tier heritage). Pro tier benefits from batches of 500 that amortize HTTP round-trip cost and Supabase transaction overhead. `--max-chunks` limits how many records are processed, enabling test runs against real data without running the full corpus.

- [ ] **Step 1: Add `batch_size` and `max_chunks` parameters to `embed_source`**

In `embed_source`, change the signature and the `if len(batch) >= BATCH_SIZE:` guard:

```python
def embed_source(
    source_id: str,
    supabase: "Client",
    embed_model,
    chunked_bucket: str,
    aws_region: str | None,
    local_input: Path | None = None,
    batch_delay: float = 1.0,
    batch_size: int = BATCH_SIZE,
    max_chunks: int | None = None,
    bulk_loader: "BulkLoader | None" = None,
    manage_indexes: bool = False,
) -> None:
```

Add the early-exit guard at the top of the `for record in iter_records(lines):` loop body (before the `chunk_id in processed` check):

```python
        if max_chunks is not None and embedded >= max_chunks:
            log.info("[%s] --max-chunks=%d reached, stopping early.", source_id, max_chunks)
            break
```

Replace the hard-coded reference to `BATCH_SIZE` inside the loop:

```python
        if len(batch) >= batch_size:
```

- [ ] **Step 2: Wire `--batch-size` and `--max-chunks` into the CLI argument parser in `main`**

Add after the existing `--batch-delay` argument:

```python
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        metavar="N",
        help=(
            "Number of records per Supabase upsert batch (default: %(default)s). "
            "Pro tier handles 500–1000 well; free tier should stay at 50–200."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N chunks per source (for test runs). Omit to process all.",
    )
```

And pass both through in the `embed_source(...)` call:

```python
        embed_source(
            source_id, supabase, embed_model,
            chunked_bucket, aws_region,
            local_input=args.local_input,
            batch_delay=args.batch_delay,
            batch_size=args.batch_size,
            max_chunks=args.max_chunks,
        )
```

- [ ] **Step 3: Smoke-test the flags parse without error**

```bash
python3 -m embed.batch_embed --help
```

Expected: `--batch-size N` and `--max-chunks N` appear in the output.

- [ ] **Step 4: Commit**

```bash
git add embed/batch_embed.py
git commit -m "feat: add --batch-size and --max-chunks CLI flags to embed/batch_embed.py"
```

---

## Task 3: Create `BulkLoader` (psycopg2 direct-DB path)

**Files:**
- Create: `embed/supabase_bulk_loader.py`
- Create: `tests/test_supabase_bulk_loader.py`

### Background

PostgREST (Supabase REST API) has an edge-layer HTTP timeout (~30s) that cannot be raised from the Python client side. A direct psycopg2 connection to the Postgres instance allows:
- `SET statement_timeout = '10min'` per-session before bulk operations
- `execute_values` for bulk inserts — one round-trip for N rows instead of N/batch_size round-trips
- DDL (`DROP INDEX` / `CREATE INDEX`) for the optional index-management path

The `SUPABASE_DB_URL` is available in the Supabase Pro dashboard under **Settings → Database → Connection string → URI** (use the direct connection, not the pooler).

**Dependency:** `psycopg2-binary` must be installed. If not present, `--direct-db` will fail with `ModuleNotFoundError`.

```bash
pip install psycopg2-binary
```

### `execute_values` format for pgvector

psycopg2 does not know the `vector` type natively. Pass embeddings as `"[0.1,0.2,...]"` strings and cast in the SQL template: `%s::vector`.

### Metadata JSONB

PostgreSQL requires JSONB values to arrive as JSON strings when passed via `%s`. Dicts must be serialized with `json.dumps()` before being placed in the values tuple.

### Retry logic in `bulk_upsert`

If the direct DB connection drops mid-upload, `execute_values` raises `psycopg2.OperationalError`. A dropped connection leaves the session in an aborted-transaction state — rollback is required before reconnecting. `psycopg2.ProgrammingError` (bad schema, type mismatch) and `psycopg2.IntegrityError` are not retried — these are bugs.

### Index definitions backup

Before dropping indexes, `drop_embedding_indexes` writes the index definitions to `.embed_index_backup.json`. If the script crashes after dropping but before recreating, the operator can run `--recreate-indexes` (Task 4) to rebuild without manually copying from logs.

- [ ] **Step 1: Write tests for `BulkLoader.bulk_upsert` payload formatting and retry logic**

Create `tests/test_supabase_bulk_loader.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail (module does not exist yet)**

```bash
python3 -m pytest tests/test_supabase_bulk_loader.py -v
```

Expected: `ModuleNotFoundError: No module named 'embed.supabase_bulk_loader'`

- [ ] **Step 3: Create `embed/supabase_bulk_loader.py`**

```python
"""
BulkLoader — direct psycopg2 path for bulk embedding into Supabase Pro.

Bypasses PostgREST to:
  - Override statement_timeout per-session (default: 10 minutes)
  - Use execute_values for one-round-trip bulk upserts
  - Retry on connection drops (OperationalError) with exponential backoff + reconnect
  - Optionally drop + rebuild vector indexes around large loads

Usage:
  loader = BulkLoader(os.environ["SUPABASE_DB_URL"])
  loader.connect()
  try:
      checkpoint = loader.load_checkpoint("opinion_chunks")
      index_defs = loader.drop_embedding_indexes("opinion_chunks")
      loader.bulk_upsert(payloads, "opinion_chunks")
      loader.recreate_embedding_indexes(index_defs)
  finally:
      loader.close()

Requires: psycopg2-binary  (pip install psycopg2-binary)

SUPABASE_DB_URL: find under Supabase dashboard → Settings → Database →
  Connection string → URI  (use the Direct connection, not the pooler)
"""
import json
import logging
import pathlib
import random
import time

import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values

log = logging.getLogger(__name__)

_VECTOR_INDEX_QUERY = """
    SELECT indexname, indexdef
    FROM pg_indexes
    WHERE tablename = %s
      AND (   indexdef ILIKE '%%vector_cosine_ops%%'
           OR indexdef ILIKE '%%vector_l2_ops%%'
           OR indexdef ILIKE '%%vector_ip_ops%%')
"""

_INDEX_BACKUP_PATH = pathlib.Path(".embed_index_backup.json")


def _format_value(col: str, val):
    """Prepare a single column value for psycopg2 execute_values.

    PostgreSQL JSONB columns require a JSON string, not a Python dict.
    pgvector columns require a '[x,y,z]' string, not a Python list.
    """
    if col == "embedding" and isinstance(val, list):
        return "[" + ",".join(str(x) for x in val) + "]"
    if col == "metadata" and isinstance(val, dict):
        return json.dumps(val)
    return val


class BulkLoader:
    def __init__(self, db_url: str, statement_timeout_ms: int = 600_000):
        self.db_url = db_url
        self.statement_timeout_ms = statement_timeout_ms
        self.conn = None

    def connect(self):
        if self.conn and not self.conn.closed:
            self.conn.close()
        self.conn = psycopg2.connect(self.db_url)
        self.conn.autocommit = False
        with self.conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {self.statement_timeout_ms}")
        self.conn.commit()
        log.info("BulkLoader connected (statement_timeout=%dms)", self.statement_timeout_ms)

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()

    def load_checkpoint(self, table: str) -> set[str]:
        """Return chunk_ids already in the table — single query, no pagination.

        For tables with >500k rows, this loads all IDs into memory (~20-40MB).
        Acceptable for current corpus sizes; revisit if tables exceed 1M rows.
        """
        with self.conn.cursor() as cur:
            cur.execute(f'SELECT chunk_id FROM "{table}"')
            result = {row[0] for row in cur.fetchall()}
        log.info("Checkpoint: %d chunks already in %s.", len(result), table)
        return result

    def bulk_upsert(
        self,
        payloads: list[dict],
        table: str,
        _max_retries: int = 5,
        _base_delay: float = 2.0,
    ) -> int:
        """Upsert payloads in a single execute_values call. Returns count upserted.

        Retries on psycopg2.OperationalError (connection drop) with exponential
        backoff + reconnect. Does NOT retry ProgrammingError or IntegrityError
        (those are bugs, not transient failures).
        """
        if not payloads:
            return 0

        cols = list(payloads[0].keys())
        col_str = ", ".join(f'"{c}"' for c in cols)
        updates = ", ".join(
            f'"{c}" = EXCLUDED."{c}"' for c in cols if c != "chunk_id"
        )
        sql = (
            f'INSERT INTO "{table}" ({col_str}) VALUES %s '
            f"ON CONFLICT (chunk_id) DO UPDATE SET {updates}"
        )
        placeholders = ["%s::vector" if c == "embedding" else "%s" for c in cols]
        template = "(" + ", ".join(placeholders) + ")"
        rows = [tuple(_format_value(c, p[c]) for c in cols) for p in payloads]

        for attempt in range(_max_retries):
            try:
                with self.conn.cursor() as cur:
                    execute_values(cur, sql, rows, template=template, page_size=len(rows))
                self.conn.commit()
                log.info("Upserted %d rows into %s.", len(payloads), table)
                return len(payloads)
            except psycopg2.OperationalError as exc:
                self.conn.rollback()
                if attempt == _max_retries - 1:
                    raise
                delay = min(_base_delay * (2 ** attempt), 60.0)
                jitter = random.uniform(0, delay * 0.2)
                log.warning(
                    "Connection error on attempt %d/%d: %s — reconnecting in %.1fs",
                    attempt + 1, _max_retries, exc, delay + jitter,
                )
                time.sleep(delay + jitter)
                self.connect()

        return 0  # unreachable; satisfies type checker

    def drop_embedding_indexes(self, table: str) -> list[tuple[str, str]]:
        """Drop vector indexes on table; return (name, CREATE statement) pairs for recreation.

        Writes index definitions to .embed_index_backup.json before dropping.
        If the process crashes before recreate_embedding_indexes runs, use:
            python3 -m embed.batch_embed --recreate-indexes <table>
        to rebuild from the backup file without re-running embedding.
        """
        with self.conn.cursor() as cur:
            cur.execute(_VECTOR_INDEX_QUERY, (table,))
            indexes = cur.fetchall()

        if not indexes:
            log.info("No vector indexes found on %s — nothing to drop.", table)
            return []

        _INDEX_BACKUP_PATH.write_text(json.dumps(indexes))
        log.warning(
            "ADVISORY: About to drop %d vector index(es) on %s. "
            "Definitions saved to %s. "
            "If this script crashes before completion, run: "
            "python3 -m embed.batch_embed --recreate-indexes %s",
            len(indexes), table, _INDEX_BACKUP_PATH, table,
        )

        dropped = []
        for name, indexdef in indexes:
            log.info("Dropping index %s: %s", name, indexdef)
            with self.conn.cursor() as cur:
                cur.execute(f'DROP INDEX IF EXISTS "{name}"')
            self.conn.commit()
            dropped.append((name, indexdef))

        return dropped

    def recreate_embedding_indexes(self, index_defs: list[tuple[str, str]]):
        """Recreate indexes that were dropped before bulk load.

        Uses CREATE INDEX CONCURRENTLY so the table stays readable during the build.
        CONCURRENTLY requires autocommit mode — psycopg2 autocommit is toggled here.
        """
        if not index_defs:
            return

        old_autocommit = self.conn.autocommit
        self.conn.autocommit = True
        try:
            for name, indexdef in index_defs:
                create_sql = indexdef.replace("CREATE INDEX", "CREATE INDEX CONCURRENTLY IF NOT EXISTS", 1)
                log.info("Recreating index %s (CONCURRENTLY)…", name)
                with self.conn.cursor() as cur:
                    cur.execute(create_sql)
                log.info("Index %s rebuilt.", name)
        finally:
            self.conn.autocommit = old_autocommit
        _INDEX_BACKUP_PATH.unlink(missing_ok=True)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python3 -m pytest tests/test_supabase_bulk_loader.py -v
```

Expected: all 13 tests pass.

- [ ] **Step 5: Commit**

```bash
git add embed/supabase_bulk_loader.py tests/test_supabase_bulk_loader.py
git commit -m "feat: add BulkLoader psycopg2 path with retry logic and index management"
```

---

## Task 4: Integrate `BulkLoader` into `embed_source` and `main`

**Files:**
- Modify: `embed/batch_embed.py` — `embed_source`, `main`

Wire the new `BulkLoader` as an opt-in path activated by `--direct-db`. When active:
- `load_checkpoint` uses `BulkLoader.load_checkpoint` (single SQL query, not paginated REST)
- `flush_batch` is replaced by `BulkLoader.bulk_upsert` for each batch
- If `--manage-indexes` is also set, vector indexes are dropped before the first batch and recreated after the final one
- `--recreate-indexes TABLE` is a standalone recovery mode that reads `.embed_index_backup.json` and rebuilds without embedding

The existing REST API path (`flush_batch`) is preserved unchanged as the default.

Progress is logged every 10 batches so large runs (337k chunks = ~337 batches at batch-size=1000) stay visible.

- [ ] **Step 1: Update `embed_source` signature and body**

The signature was already updated in Task 2. The `bulk_loader` and `manage_indexes` parameters are added here. Full updated body of `embed_source` after the file-reading block:

```python
    if bulk_loader is not None:
        processed = bulk_loader.load_checkpoint(table)
    else:
        processed = load_checkpoint(supabase, table)

    batch: list[dict] = []
    embedded = skipped = filtered = failed = 0
    dropped_indexes: list[tuple[str, str]] = []
    batch_count = 0

    for record in iter_records(lines):
        if max_chunks is not None and embedded >= max_chunks:
            log.info("[%s] --max-chunks=%d reached, stopping early.", source_id, max_chunks)
            break

        chunk_id = record["chunk_id"]
        if chunk_id in processed:
            skipped += 1
            continue
        if entry.filter_fn and not entry.filter_fn(record):
            filtered += 1
            continue

        if source_id == "courtlistener":
            record = _normalize_cl_chunk(record)

        record["text"] = record["text"].replace("\x00", "")
        enriched = record["enriched_text"].replace("\x00", "")
        if len(enriched) > MAX_EMBED_CHARS:
            log.debug("Chunk %s truncated from %d to %d chars.", chunk_id, len(enriched), MAX_EMBED_CHARS)
            enriched = enriched[:MAX_EMBED_CHARS]
        record["enriched_text"] = enriched

        embedding = embed_model.get_text_embedding(enriched)
        batch.append(build_payload(record, embedding, table))
        embedded += 1

        if len(batch) >= batch_size:
            batch_count += 1
            if batch_count == 1 or batch_count % 10 == 0:
                log.info("[%s] Progress: %d rows embedded (batch %d)…", source_id, embedded, batch_count)

            if bulk_loader is not None:
                if manage_indexes and not dropped_indexes:
                    dropped_indexes = bulk_loader.drop_embedding_indexes(table)
                count = bulk_loader.bulk_upsert(batch, table)
                failed += len(batch) - count
                processed.update(p["chunk_id"] for p in batch)
            else:
                flushed = flush_batch(supabase, batch, table)
                failed += len(batch) - len(flushed)
                processed.update(flushed)
            batch = []
            if batch_delay > 0:
                time.sleep(batch_delay)

    # Flush remaining records
    if batch:
        if bulk_loader is not None:
            if manage_indexes and not dropped_indexes:
                dropped_indexes = bulk_loader.drop_embedding_indexes(table)
            count = bulk_loader.bulk_upsert(batch, table)
            failed += len(batch) - count
        else:
            flushed = flush_batch(supabase, batch, table)
            failed += len(batch) - len(flushed)

    if dropped_indexes and bulk_loader is not None:
        bulk_loader.recreate_embedding_indexes(dropped_indexes)

    log.info(
        "[%s] Done. embedded=%d  skipped=%d  filtered=%d  failed=%d  → %s",
        source_id, embedded, skipped, filtered, failed, table,
    )
```

- [ ] **Step 2: Add `--direct-db`, `--manage-indexes`, and `--recreate-indexes` flags to the CLI parser**

Add these arguments after `--max-chunks` in `main()`:

```python
    parser.add_argument(
        "--direct-db",
        action="store_true",
        default=False,
        help=(
            "Use a direct psycopg2 connection instead of PostgREST. "
            "Requires SUPABASE_DB_URL in .env. "
            "Sets statement_timeout=10min per-session, eliminating 57014 errors. "
            "Recommended for large sources (cap_bulk, courtlistener)."
        ),
    )
    parser.add_argument(
        "--manage-indexes",
        action="store_true",
        default=False,
        help=(
            "Drop vector indexes before bulk load and rebuild after. "
            "Only valid with --direct-db. "
            "Use for first-time loads of large sources (>50k rows). "
            "Skips per-insert HNSW maintenance overhead; rebuilds once at the end. "
            "If the script crashes, run --recreate-indexes <table> to rebuild."
        ),
    )
    parser.add_argument(
        "--recreate-indexes",
        metavar="TABLE",
        default=None,
        help=(
            "Standalone recovery mode: read .embed_index_backup.json and rebuild "
            "the vector indexes on TABLE. Use if the script crashed after --manage-indexes "
            "dropped indexes but before recreating them. Requires --direct-db."
        ),
    )
```

- [ ] **Step 3: Instantiate `BulkLoader` in `main` and pass to `embed_source`**

Add the `TYPE_CHECKING` import guard near the top of the file:

```python
if TYPE_CHECKING:
    from embed.supabase_bulk_loader import BulkLoader
```

In `main()`, after `embed_model = get_embedding_model()`:

```python
    if args.manage_indexes and not args.direct_db:
        parser.error("--manage-indexes requires --direct-db")
    if args.recreate_indexes and not args.direct_db:
        parser.error("--recreate-indexes requires --direct-db")

    bulk_loader = None
    if args.direct_db:
        from embed.supabase_bulk_loader import BulkLoader, _INDEX_BACKUP_PATH, _VECTOR_INDEX_QUERY
        db_url = _require_env("SUPABASE_DB_URL")
        bulk_loader = BulkLoader(db_url)
        bulk_loader.connect()

    # Standalone index recovery mode — skip embedding
    if args.recreate_indexes:
        import json as _json

        if _INDEX_BACKUP_PATH.exists():
            index_defs = _json.loads(_INDEX_BACKUP_PATH.read_text())
            log.info("Using index definitions from %s (%d indexes).", _INDEX_BACKUP_PATH, len(index_defs))
        else:
            # Backup missing — check whether indexes are already present in pg_indexes.
            # If they exist, nothing to do. If absent, we cannot reconstruct the exact
            # definition; print a template the operator can paste into the Supabase SQL editor.
            log.warning("%s not found — querying pg_indexes for existing vector indexes on %s.",
                        _INDEX_BACKUP_PATH, args.recreate_indexes)
            with bulk_loader.conn.cursor() as cur:
                cur.execute(_VECTOR_INDEX_QUERY, (args.recreate_indexes,))
                existing = cur.fetchall()
            if existing:
                log.info("Vector indexes already present on %s — no action needed.", args.recreate_indexes)
                bulk_loader.close()
                return
            log.error(
                "Vector indexes are missing on %s and no backup file exists. "
                "Re-create them manually via the Supabase SQL editor:\n"
                "  CREATE INDEX CONCURRENTLY IF NOT EXISTS %s_embedding_idx\n"
                "    ON %s USING hnsw (embedding vector_cosine_ops)\n"
                "    WITH (m = 16, ef_construction = 64);",
                args.recreate_indexes, args.recreate_indexes, args.recreate_indexes,
            )
            bulk_loader.close()
            raise SystemExit(1)

        log.info("Recreating %d index(es) on %s…", len(index_defs), args.recreate_indexes)
        try:
            bulk_loader.recreate_embedding_indexes(index_defs)
        finally:
            bulk_loader.close()
        return

    # Auto-recover missing indexes from a previous crashed run before embedding begins.
    # If the backup file exists and any of its indexes are absent from pg_indexes,
    # rebuild them now rather than silently embedding into an unindexed table.
    if bulk_loader is not None and _INDEX_BACKUP_PATH.exists():
        import json as _json
        backed_up = _json.loads(_INDEX_BACKUP_PATH.read_text())
        if backed_up:
            with bulk_loader.conn.cursor() as cur:
                cur.execute(
                    "SELECT indexname FROM pg_indexes WHERE indexname = ANY(%s)",
                    ([name for name, _ in backed_up],),
                )
                existing_names = {row[0] for row in cur.fetchall()}
            missing = [(name, defn) for name, defn in backed_up if name not in existing_names]
            if missing:
                log.warning(
                    "Auto-recovery: %d vector index(es) missing from a previous crashed run — "
                    "rebuilding before embedding begins.",
                    len(missing),
                )
                bulk_loader.recreate_embedding_indexes(missing)
```

Wrap the `for source_id in args.source:` loop in a try/finally:

```python
    try:
        log.info("Embedding %d source(s): %s", len(args.source), ", ".join(args.source))
        for source_id in args.source:
            embed_source(
                source_id, supabase, embed_model,
                chunked_bucket, aws_region,
                local_input=args.local_input,
                batch_delay=args.batch_delay,
                batch_size=args.batch_size,
                max_chunks=args.max_chunks,
                bulk_loader=bulk_loader,
                manage_indexes=args.manage_indexes,
            )
    finally:
        if bulk_loader is not None:
            bulk_loader.close()
```

- [ ] **Step 4: Add `SUPABASE_DB_URL` to the env-var documentation comment in `batch_embed.py`**

Add near the `SUPABASE_URL`/`SUPABASE_SERVICE_KEY` references at the top of the file:

```python
# Optional — only needed for --direct-db:
#   SUPABASE_DB_URL: postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres
#   (Supabase dashboard → Settings → Database → Connection string → URI, Direct connection)
#   Note: this contains the raw DB password — keep it out of version control (.gitignore .env).
```

- [ ] **Step 5: Smoke-test the CLI flag wiring**

```bash
python3 -m embed.batch_embed --help
```

Expected: `--direct-db`, `--manage-indexes`, `--recreate-indexes TABLE`, and `--max-chunks N` all appear in the output.

- [ ] **Step 6: Run full test suite**

```bash
python3 -m pytest tests/test_batch_embed.py tests/test_supabase_bulk_loader.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add embed/batch_embed.py
git commit -m "feat: integrate BulkLoader direct-db path; add --direct-db, --manage-indexes, --recreate-indexes flags"
```

---

## Self-Review

**Spec coverage check:**

| Problem from diagnosis | Task that addresses it |
|------------------------|----------------------|
| 57014 misclassified → binary split | Task 1: `_SCHEMA_CACHE_ERRORS` + `_is_availability_error` |
| No jitter → thundering herd on retry | Task 1: `random.uniform(0, delay * 0.2)` |
| Small batch sizes hurt Pro tier | Task 2: `--batch-size` flag, default stays at BATCH_SIZE |
| HTTP round-trip overhead × N/batch | Task 3: `execute_values` one-call path |
| statement_timeout capped at ~30s | Task 3: `SET statement_timeout = 600000` per-session |
| Index maintenance per-insert | Task 3 + 4: `drop_embedding_indexes` / `recreate_embedding_indexes` via `--manage-indexes` |
| No distinction timeout vs real failure | Task 1: separate `_is_availability_error` branch now catches 57014 |
| Connection drop mid-upload: no retry | Task 3: `bulk_upsert` retries on `OperationalError` with rollback + reconnect |
| Crash after drop, before recreate (backup exists) | Task 3: `.embed_index_backup.json` written before drop; Task 4: auto-recovery on next `--direct-db` run detects missing indexes via `pg_indexes` diff and rebuilds automatically |
| Crash after drop, backup file lost | Task 4: `--recreate-indexes` checks `pg_indexes`; if indexes already present does nothing; if absent, logs actionable `CREATE INDEX CONCURRENTLY IF NOT EXISTS` template for manual Supabase SQL editor paste |
| Partial drop (crash mid-drop, some indexes still exist) | Task 3: `CREATE INDEX CONCURRENTLY IF NOT EXISTS` in `recreate_embedding_indexes` — idempotent; already-present indexes are skipped without error |
| No way to test without full corpus run | Task 2: `--max-chunks N` stops after N chunks |
| psycopg2-binary not in deps | Task 3: explicit install step; documented in module docstring |
| Binary split removal over-promised | Clarified in Architecture: binary split is preserved for non-availability errors |

**Placeholder scan:** None found — all steps include full code blocks.

**Type consistency check:**
- `BulkLoader` typed as `"BulkLoader | None"` in `embed_source` signature (string forward-reference, consistent with existing `"Client"` pattern)
- `bulk_loader.bulk_upsert(batch, table)` returns `int` (count) — matches `count = ...` usage in Task 4
- `bulk_loader.drop_embedding_indexes(table)` returns `list[tuple[str, str]]` — matches `dropped_indexes: list[tuple[str, str]]` in Task 4
- `_format_value(col, val)` used in `BulkLoader.bulk_upsert` and tested directly in Task 3 — consistent
- `_INDEX_BACKUP_PATH` is a module-level `pathlib.Path` imported in both `BulkLoader.drop_embedding_indexes` and `main()` — consistent

---

## Usage after implementation

```bash
# REST API path (unchanged default — still works)
python3 -m embed.batch_embed --source cap_bulk --batch-size 500

# Test run: first 100 chunks only
python3 -m embed.batch_embed --source cap_bulk --direct-db --max-chunks 100 --batch-size 50

# Direct DB path (eliminates 57014 errors)
python3 -m embed.batch_embed --source cap_bulk --direct-db --batch-size 1000

# Direct DB path + drop indexes (first-time load of 337k CAP chunks)
python3 -m embed.batch_embed --source cap_bulk --direct-db --manage-indexes --batch-size 1000

# Recovery: if script crashed after --manage-indexes dropped indexes
python3 -m embed.batch_embed --direct-db --recreate-indexes opinion_chunks

# .env addition required for --direct-db:
# SUPABASE_DB_URL=postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres
```
