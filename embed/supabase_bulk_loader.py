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
