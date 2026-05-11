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

# Like _VECTOR_INDEX_QUERY but filters out INVALID indexes left by interrupted
# CONCURRENTLY builds — used to decide whether a rebuild is actually needed.
_VALID_VECTOR_INDEX_QUERY = """
    SELECT pi.indexname, pi.indexdef
    FROM pg_indexes pi
    JOIN pg_class c  ON c.relname = pi.indexname AND c.relkind = 'i'
    JOIN pg_index ix ON ix.indexrelid = c.oid
    WHERE pi.tablename = %s
      AND ix.indisvalid
      AND (   pi.indexdef ILIKE '%%vector_cosine_ops%%'
           OR pi.indexdef ILIKE '%%vector_l2_ops%%'
           OR pi.indexdef ILIKE '%%vector_ip_ops%%')
"""

# Fallback definition used when the backup file is empty and no index exists.
# Parameters match the codebase default (m=16, ef_construction=64).
_STANDARD_HNSW_INDEXDEF = (
    "CREATE INDEX {name} ON public.{table} "
    "USING hnsw (embedding vector_cosine_ops) "
    "WITH (m = 16, ef_construction = 64)"
)

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
        self.conn = psycopg2.connect(
            self.db_url,
            keepalives=1,
            keepalives_idle=60,      # seconds of inactivity before first probe
            keepalives_interval=10,  # seconds between probes
            keepalives_count=6,      # failed probes before declaring connection dead
        )
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

        # Deduplicate by chunk_id — ON CONFLICT DO UPDATE cannot affect the
        # same row twice within a single statement. Keep last occurrence so
        # that re-runs with updated embeddings naturally win.
        seen: dict[str, dict] = {}
        for p in payloads:
            seen[p["chunk_id"]] = p
        payloads = list(seen.values())

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
            except psycopg2.DatabaseError as exc:
                if isinstance(exc, (psycopg2.ProgrammingError, psycopg2.IntegrityError,
                                    psycopg2.DataError, psycopg2.NotSupportedError)):
                    raise
                try:
                    self.conn.rollback()
                except (psycopg2.OperationalError, psycopg2.InterfaceError):
                    pass
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

    def recreate_embedding_indexes(
        self,
        index_defs: list[tuple[str, str]],
        maintenance_work_mem: str = "",
        max_parallel_workers: int = 4,
    ):
        """Recreate indexes that were dropped before bulk load.

        Uses CREATE INDEX IF NOT EXISTS (non-concurrent) which works through
        the Supabase session pooler. Locks the table for writes during the build,
        but that is acceptable for bulk loads where no concurrent writers exist.

        Checkpointing: the backup file is updated after each successful rebuild
        (completed index removed from the list). On timeout/crash, rerun with
        --recreate-indexes — only the remaining indexes will be built.

        maintenance_work_mem: passed directly to SET — larger values reduce
            disk-spill passes during HNSW graph construction.
        max_parallel_workers: parallel background workers for the index build;
            pgvector 0.6+ supports this for HNSW (default: 4).
        """
        if not index_defs:
            return

        remaining = list(index_defs)

        for name, indexdef in index_defs:
            # Skip indexes already present — handles resume after a partial run.
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_indexes WHERE indexname = %s", (name,))
                if cur.fetchone():
                    log.info("Index %s already exists — skipping.", name)
                    remaining = [(n, d) for n, d in remaining if n != name]
                    if remaining:
                        _INDEX_BACKUP_PATH.write_text(json.dumps(remaining))
                    else:
                        _INDEX_BACKUP_PATH.unlink(missing_ok=True)
                    continue

            create_sql = indexdef.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS", 1)
            log.info(
                "Recreating index %s "
                "(maintenance_work_mem=%s, max_parallel_maintenance_workers=%d, no timeout)…",
                name, maintenance_work_mem or "default", max_parallel_workers,
            )
            with self.conn.cursor() as cur:
                cur.execute("SET statement_timeout = 0")
                if maintenance_work_mem:
                    cur.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}'")
                cur.execute(f"SET max_parallel_maintenance_workers = {max_parallel_workers}")
                cur.execute(create_sql)
            self.conn.commit()
            # Restore session settings for subsequent statements.
            with self.conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {self.statement_timeout_ms}")
                if maintenance_work_mem:
                    cur.execute("RESET maintenance_work_mem")
                cur.execute("RESET max_parallel_maintenance_workers")
            self.conn.commit()
            log.info("Index %s rebuilt.", name)

            # Checkpoint: remove this index from the backup file so a retry
            # skips it. Non-concurrent CREATE INDEX is transactional — a
            # connection drop rolls back the build, so any entry still in the
            # file means the index needs to be (re)built.
            remaining = [(n, d) for n, d in remaining if n != name]
            if remaining:
                _INDEX_BACKUP_PATH.write_text(json.dumps(remaining))
            else:
                _INDEX_BACKUP_PATH.unlink(missing_ok=True)

    def get_or_generate_index_defs(
        self, table: str, backup_path: pathlib.Path
    ) -> list[tuple[str, str]]:
        """Return index definitions to rebuild.

        Priority order:
        1. Backup file (if non-empty) — preserves original index parameters.
        2. Valid indexes already on the table — returns [] (nothing to do).
        3. Generated standard HNSW definition — used when backup is gone and
           no valid index exists (e.g. after a crash with an empty backup file).
           Writes the generated definition to backup_path for crash recovery.
        """
        if backup_path.exists():
            defs = json.loads(backup_path.read_text())
            if defs:
                log.info("Loaded %d index definition(s) from %s.", len(defs), backup_path)
                return defs
            log.info("Backup file is empty — checking table for valid indexes.")
        else:
            log.info("No backup file — checking table for valid indexes.")

        with self.conn.cursor() as cur:
            cur.execute(_VALID_VECTOR_INDEX_QUERY, (table,))
            valid = cur.fetchall()
        if valid:
            log.info("Valid vector index(es) already exist on %s — nothing to do.", table)
            return []

        index_name = f"{table}_embedding_idx"
        indexdef = _STANDARD_HNSW_INDEXDEF.format(name=index_name, table=table)
        log.warning(
            "No backup and no valid index on %s. "
            "Generating standard HNSW definition (m=16, ef_construction=64). "
            "If the original index used different parameters, create it manually instead.",
            table,
        )
        defs = [(index_name, indexdef)]
        backup_path.write_text(json.dumps(defs))
        log.info("Wrote generated definition to %s for crash recovery.", backup_path)
        return defs

    def recreate_embedding_indexes_concurrently(
        self,
        index_defs: list[tuple[str, str]],
        maintenance_work_mem: str = "",
        max_parallel_workers: int = 4,
    ):
        """Rebuild indexes using CREATE INDEX CONCURRENTLY.

        Runs outside a transaction so Supabase's long-connection timeout cannot
        roll back the build. If a previous CONCURRENTLY build was interrupted,
        it leaves an INVALID index behind; this method detects and drops it
        before rebuilding.

        Checkpointing: backup file updated after each successful build so
        --recreate-indexes resumes from where it left off on retry.
        """
        if not index_defs:
            return

        remaining = list(index_defs)

        # Switch to autocommit — required for CONCURRENTLY statements.
        # Roll back any pending implicit transaction first (e.g. from a prior
        # read-only SELECT in get_or_generate_index_defs) — psycopg2 raises
        # ProgrammingError if autocommit is changed inside an open transaction.
        try:
            self.conn.rollback()
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            pass
        # Session-level SET persists across autocommit transactions.
        self.conn.autocommit = True
        try:
            with self.conn.cursor() as cur:
                cur.execute("SET statement_timeout = 0")
                if maintenance_work_mem:
                    cur.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}'")
                cur.execute(f"SET max_parallel_maintenance_workers = {max_parallel_workers}")

            for name, indexdef in index_defs:
                # Check for existing valid index — skip if already done.
                with self.conn.cursor() as cur:
                    cur.execute(
                        "SELECT ix.indisvalid FROM pg_class c "
                        "JOIN pg_index ix ON ix.indexrelid = c.oid "
                        "WHERE c.relname = %s AND c.relkind = 'i'",
                        (name,),
                    )
                    row = cur.fetchone()

                if row is not None:
                    if row[0]:  # indisvalid = True
                        log.info("Index %s already exists and is valid — skipping.", name)
                        remaining = [(n, d) for n, d in remaining if n != name]
                        if remaining:
                            _INDEX_BACKUP_PATH.write_text(json.dumps(remaining))
                        else:
                            _INDEX_BACKUP_PATH.unlink(missing_ok=True)
                        continue
                    else:  # indisvalid = False — interrupted CONCURRENTLY build
                        log.warning(
                            "Dropping INVALID index %s left by a previous interrupted build…", name
                        )
                        with self.conn.cursor() as cur:
                            cur.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{name}"')
                        log.info("Dropped invalid index %s.", name)

                create_sql = indexdef.replace(
                    "CREATE INDEX", "CREATE INDEX CONCURRENTLY IF NOT EXISTS", 1
                )
                log.info(
                    "Building index %s CONCURRENTLY "
                    "(maintenance_work_mem=%s, max_parallel_maintenance_workers=%d)…",
                    name, maintenance_work_mem or "default", max_parallel_workers,
                )
                with self.conn.cursor() as cur:
                    cur.execute(create_sql)
                log.info("Index %s built successfully.", name)

                remaining = [(n, d) for n, d in remaining if n != name]
                if remaining:
                    _INDEX_BACKUP_PATH.write_text(json.dumps(remaining))
                else:
                    _INDEX_BACKUP_PATH.unlink(missing_ok=True)

        finally:
            # Restore session settings, then return to non-autocommit mode.
            # Guard against a dead connection (e.g. Supabase restart during the
            # long CONCURRENTLY build) — a connection drop here is non-fatal
            # because the INVALID index will be detected and dropped on the next run.
            try:
                with self.conn.cursor() as cur:
                    cur.execute(f"SET statement_timeout = {self.statement_timeout_ms}")
                    if maintenance_work_mem:
                        cur.execute("RESET maintenance_work_mem")
                    cur.execute("RESET max_parallel_maintenance_workers")
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                log.warning(
                    "Connection lost during index build — session settings not restored. "
                    "If an INVALID index was left behind, rerun --recreate-indexes to clean up."
                )
            try:
                self.conn.autocommit = False
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                pass
