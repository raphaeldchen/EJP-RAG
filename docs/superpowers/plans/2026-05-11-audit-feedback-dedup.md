# Audit Feedback Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce at most one `audit_feedback` row per `(query_id, chunk_id, expert_id)` triple so that re-clicking a label button or changing a label overwrites rather than appends.

**Architecture:** Add a `UNIQUE(query_id, chunk_id, expert_id)` constraint to Supabase, then change `submit_feedback` from `.insert()` to `.upsert(on_conflict=...)`. Normalize blank `expert_id` to `"anonymous"` so the constraint behaves correctly for unauthenticated users (PostgreSQL treats NULLs as never-equal in unique constraints).

**Tech Stack:** Supabase (PostgreSQL + PostgREST), supabase-py v2, pytest

---

## Files

- Modify: `mcp_server/server.py` — `submit_feedback` function (~line 322)
- Modify: `tests/mcp_server/test_audit_tool.py` — extend `test_submit_feedback_writes_to_supabase`

---

### Task 1: Write the failing dedup test

**Files:**
- Modify: `tests/mcp_server/test_audit_tool.py`

- [ ] **Step 1: Replace the existing smoke test with a stricter dedup test**

Open `tests/mcp_server/test_audit_tool.py` and replace `test_submit_feedback_writes_to_supabase` (lines 40–67) with:

```python
def test_submit_feedback_deduplicates_by_query_chunk_expert():
    """Second call for same (query, chunk, expert) must update, not insert."""
    from mcp_server.server import submit_feedback, _get_state

    shared = dict(
        query="good time credit",
        chunk_id="test-chunk-dedup",
        citation="730 ILCS 5/3-6-3",
        source="ilcs",
        retrieval_mode="hybrid",
        persona="researcher",
        pre_rerank_rank=1,
        post_rerank_rank=1,
        rrf_score=0.05,
        ce_score=2.1,
        expert_id="test-expert",
    )

    # First submission
    submit_feedback(**shared, label="RELEVANT", comment="first")
    # Second submission — different label, same triple
    submit_feedback(**shared, label="BINDING", comment="changed my mind")

    state = _get_state()
    rows = (
        state.client.table("audit_feedback")
        .select("label, comment")
        .eq("chunk_id", "test-chunk-dedup")
        .eq("expert_id", "test-expert")
        .execute()
        .data
    )
    try:
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)} — dedup not working"
        assert rows[0]["label"] == "BINDING", "Latest label should have won"
        assert rows[0]["comment"] == "changed my mind"
    finally:
        state.client.table("audit_feedback").delete().eq("chunk_id", "test-chunk-dedup").execute()


def test_submit_feedback_anonymous_normalizes_expert_id():
    """Blank expert_id should be stored as 'anonymous', enabling dedup for unauthenticated users."""
    from mcp_server.server import submit_feedback, _get_state

    submit_feedback(
        query="good time credit anon",
        chunk_id="test-chunk-anon",
        citation="730 ILCS 5/3-6-3",
        source="ilcs",
        retrieval_mode="hybrid",
        persona="",
        pre_rerank_rank=1,
        post_rerank_rank=None,
        rrf_score=0.05,
        ce_score=None,
        label="RELEVANT",
        comment="",
        expert_id="",   # blank — should normalize to "anonymous"
    )

    state = _get_state()
    rows = (
        state.client.table("audit_feedback")
        .select("expert_id")
        .eq("chunk_id", "test-chunk-anon")
        .execute()
        .data
    )
    try:
        assert len(rows) == 1
        assert rows[0]["expert_id"] == "anonymous"
    finally:
        state.client.table("audit_feedback").delete().eq("chunk_id", "test-chunk-anon").execute()
```

- [ ] **Step 2: Run the new tests to confirm they fail**

```bash
cd /Users/raphaelchen/Desktop/legal_rag
source venv/bin/activate
pytest tests/mcp_server/test_audit_tool.py::test_submit_feedback_deduplicates_by_query_chunk_expert \
       tests/mcp_server/test_audit_tool.py::test_submit_feedback_anonymous_normalizes_expert_id \
       -v
```

Expected: both FAIL. `test_submit_feedback_deduplicates_by_query_chunk_expert` should fail with `AssertionError: Expected 1 row, got 2`. `test_submit_feedback_anonymous_normalizes_expert_id` should fail because `expert_id` is stored as `NULL` not `"anonymous"`.

---

### Task 2: Run the Supabase migration

> Run this SQL in the Supabase SQL Editor (dashboard → SQL Editor → New query).

- [ ] **Step 1: Backfill NULL expert_id values**

```sql
UPDATE audit_feedback
SET expert_id = 'anonymous'
WHERE expert_id IS NULL;
```

Verify: `SELECT COUNT(*) FROM audit_feedback WHERE expert_id IS NULL;` should return 0.

- [ ] **Step 2: Add the unique constraint**

```sql
ALTER TABLE audit_feedback
  ADD CONSTRAINT audit_feedback_unique UNIQUE (query_id, chunk_id, expert_id);
```

Verify: `SELECT conname FROM pg_constraint WHERE conname = 'audit_feedback_unique';` should return 1 row.

- [ ] **Step 3: Confirm the constraint exists**

```sql
SELECT conname, contype, pg_get_constraintdef(oid)
FROM pg_constraint
WHERE conrelid = 'audit_feedback'::regclass
  AND conname = 'audit_feedback_unique';
```

Expected output:
```
conname                    | contype | pg_get_constraintdef
audit_feedback_unique      | u       | UNIQUE (query_id, chunk_id, expert_id)
```

---

### Task 3: Update `submit_feedback` to upsert

**Files:**
- Modify: `mcp_server/server.py:322-357`

- [ ] **Step 1: Apply the two changes to `submit_feedback`**

Open `mcp_server/server.py`. In `submit_feedback`, make these two edits:

**Before** (lines ~338–357):
```python
    state = _get_state()
    query_id = _hashlib.sha256(query.encode()).hexdigest()[:16]
    collection_id = _SOURCE_TO_COLLECTION.get(source, "unknown")
    state.client.table("audit_feedback").insert({
        "query_text": query,
        "query_id": query_id,
        "chunk_id": chunk_id,
        "citation": citation,
        "source": source,
        "collection_id": collection_id,
        "retrieval_mode": retrieval_mode,
        "persona": persona,
        "pre_rerank_rank": pre_rerank_rank,
        "post_rerank_rank": post_rerank_rank,
        "rrf_score": rrf_score,
        "ce_score": ce_score,
        "label": label,
        "comment": comment or None,
        "expert_id": expert_id or None,
    }).execute()
```

**After:**
```python
    state = _get_state()
    query_id = _hashlib.sha256(query.encode()).hexdigest()[:16]
    collection_id = _SOURCE_TO_COLLECTION.get(source, "unknown")
    expert_id = expert_id or "anonymous"
    state.client.table("audit_feedback").upsert({
        "query_text": query,
        "query_id": query_id,
        "chunk_id": chunk_id,
        "citation": citation,
        "source": source,
        "collection_id": collection_id,
        "retrieval_mode": retrieval_mode,
        "persona": persona,
        "pre_rerank_rank": pre_rerank_rank,
        "post_rerank_rank": post_rerank_rank,
        "rrf_score": rrf_score,
        "ce_score": ce_score,
        "label": label,
        "comment": comment or None,
        "expert_id": expert_id,
    }, on_conflict="query_id,chunk_id,expert_id").execute()
```

- [ ] **Step 2: Run the tests — both should now pass**

```bash
pytest tests/mcp_server/test_audit_tool.py::test_submit_feedback_deduplicates_by_query_chunk_expert \
       tests/mcp_server/test_audit_tool.py::test_submit_feedback_anonymous_normalizes_expert_id \
       -v
```

Expected: both PASS.

- [ ] **Step 3: Run the full mcp_server test suite to check for regressions**

```bash
pytest tests/mcp_server/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add mcp_server/server.py tests/mcp_server/test_audit_tool.py
git commit -m "feat: deduplicate audit_feedback by (query_id, chunk_id, expert_id) via upsert"
```
