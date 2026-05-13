# Audit Feedback Deduplication

**Date:** 2026-05-11
**Status:** Approved

## Problem

`submit_feedback` calls `.insert()` unconditionally. If a lawyer clicks a label button twice, or changes their label (e.g. RELEVANT → BINDING), a new row is appended each time. The `audit_feedback` table accumulates multiple rows per `(query_id, chunk_id, expert_id)` triple with no way to know which is canonical.

## Goal

Enforce at most one feedback row per `(query, candidate, expert)` triple. The latest feedback overwrites any prior row for that triple.

## Unique Key

`(query_id, chunk_id, expert_id)`

- `query_id` — SHA-256 prefix of query text, already computed in `submit_feedback`
- `chunk_id` — identifies the candidate chunk
- `expert_id` — identifies the lawyer; normalized to `"anonymous"` when blank so the unique constraint behaves correctly for NULL-equivalent inputs (PostgreSQL NULLs are never equal in unique constraints)

Not included: `retrieval_mode` — a lawyer may switch between hybrid/vector/BM25 for the same query and their latest label on a given chunk should still win regardless of mode.

## Changes

### 1. Supabase (one-time migration)

Backfill any existing NULL expert_id values, then add the unique constraint:

```sql
UPDATE audit_feedback SET expert_id = 'anonymous' WHERE expert_id IS NULL;
ALTER TABLE audit_feedback
  ADD CONSTRAINT audit_feedback_unique UNIQUE (query_id, chunk_id, expert_id);
```

### 2. `mcp_server/server.py` — `submit_feedback`

Two changes:
- Normalize: `expert_id = expert_id or "anonymous"` (before building the record dict)
- Replace `.insert({...}).execute()` with `.upsert({...}, on_conflict="query_id,chunk_id,expert_id").execute()`

No changes to `audit_app.py`.

## Future compatibility

When the login system is added, `expert_id` will be set from the authenticated user's ID rather than a free-text field. The unique constraint and upsert logic remain identical — only the source of `expert_id` changes.
