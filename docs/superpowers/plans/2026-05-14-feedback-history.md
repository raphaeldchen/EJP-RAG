# Feedback History Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a right-side history panel to the audit app so lawyers can review all their past feedback organized by query, with a drill-down detail view showing chunk-level labels, ranks, and scores.

**Architecture:** Add `get_feedback_history(expert_id)` to `mcp_server/server.py` (reads `audit_feedback` from Supabase). In `audit_app.py`, add a toggle button in the header, conditional `st.columns([3, 1])` layout, and a `_render_history_panel()` function that renders a list view or drill-down detail view controlled by session state.

**Tech Stack:** Streamlit 1.55, Supabase Python client, Python 3.11

---

## File Map

| File | Change |
|---|---|
| `mcp_server/server.py` | Add `get_feedback_history(expert_id)` (read function alongside `submit_feedback`) |
| `audit_app.py` | Add import, session state keys, history toggle button, conditional column layout, `_group_history()` helper, `_render_history_panel()`, `_render_history_list()`, `_render_history_detail()` |
| `tests/mcp_server/test_audit_tool.py` | Add `test_get_feedback_history_returns_expert_rows` |

---

## Task 1: Add `get_feedback_history` to `server.py` (with test)

**Files:**
- Modify: `mcp_server/server.py` (after `submit_feedback`, ~line 400)
- Modify: `tests/mcp_server/test_audit_tool.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/mcp_server/test_audit_tool.py`:

```python
def test_get_feedback_history_returns_expert_rows():
    """get_feedback_history returns only rows for the given expert_id."""
    from mcp_server.server import submit_feedback, get_feedback_history, _get_state

    state = _get_state()
    try:
        submit_feedback(
            query="test history query",
            chunk_id="test-chunk-history",
            citation="730 ILCS 5/3-6-3",
            source="ilcs",
            retrieval_mode="hybrid",
            persona="",
            pre_rerank_rank=1,
            post_rerank_rank=1,
            rrf_score=0.05,
            ce_score=2.1,
            label="BINDING",
            comment="test note",
            expert_id="test-history-expert",
        )

        rows = get_feedback_history("test-history-expert")
        our_row = next((r for r in rows if r["chunk_id"] == "test-chunk-history"), None)
        assert our_row is not None, "Row not returned for correct expert_id"
        assert our_row["label"] == "BINDING"
        assert our_row["query_text"] == "test history query"
        assert our_row["citation"] == "730 ILCS 5/3-6-3"
        assert our_row["comment"] == "test note"
        assert our_row["post_rerank_rank"] == 1
        assert "query_id" in our_row
        assert "created_at" in our_row

        other_rows = get_feedback_history("different-expert")
        assert not any(r["chunk_id"] == "test-chunk-history" for r in other_rows), \
            "Row leaked to a different expert_id"
    finally:
        state.client.table("audit_feedback").delete() \
            .eq("chunk_id", "test-chunk-history") \
            .eq("expert_id", "test-history-expert") \
            .execute()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/raphaelchen/Desktop/legal_rag && source venv/bin/activate
python3 -m pytest tests/mcp_server/test_audit_tool.py::test_get_feedback_history_returns_expert_rows -v
```

Expected: `FAILED` with `ImportError: cannot import name 'get_feedback_history'`

- [ ] **Step 3: Implement `get_feedback_history` in `server.py`**

Add the following function immediately after the closing brace of `submit_feedback` (around line 399, before `_eager_init`):

```python
def get_feedback_history(expert_id: str) -> list[dict]:
    """Return all audit_feedback rows for expert_id, ordered by created_at desc."""
    state = _get_state()
    expert_id = (expert_id or "").strip() or "anonymous"
    result = (
        state.client.table("audit_feedback")
        .select(
            "query_id,query_text,chunk_id,citation,label,comment,"
            "pre_rerank_rank,post_rerank_rank,rrf_score,ce_score,"
            "retrieval_mode,created_at"
        )
        .eq("expert_id", expert_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/mcp_server/test_audit_tool.py::test_get_feedback_history_returns_expert_rows -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add mcp_server/server.py tests/mcp_server/test_audit_tool.py
git commit -m "feat: add get_feedback_history to server.py"
```

---

## Task 2: Session state, button, and layout restructuring in `audit_app.py`

**Files:**
- Modify: `audit_app.py`

This task wires up the toggle button and conditional column layout. `_render_history_panel` is a stub (just shows "History panel" text) until Task 3.

- [ ] **Step 1: Update the import line**

In `audit_app.py`, replace line 3:

```python
from mcp_server.server import _audit_retrieval, submit_feedback, is_bm25_ready
```

with:

```python
from mcp_server.server import _audit_retrieval, submit_feedback, is_bm25_ready, get_feedback_history
```

- [ ] **Step 2: Add the stub `_render_history_panel` function**

Add this function before `_show_login()` (after the CSS block, around line 23):

```python
def _render_history_panel(expert_id: str) -> None:
    st.markdown("**📋 Feedback History**  (coming soon)")
```

- [ ] **Step 3: Replace the header block with the new three-button version**

The current header (lines 91–101) is:

```python
_title_col, _logout_col = st.columns([5, 1])
with _title_col:
    st.title("Retrieval Audit")
    st.caption("Illinois Legal RAG — Expert Labeling")
with _logout_col:
    st.write("")
    if st.button("Logout", use_container_width=True):
        for key in ["authenticated", "user_email", "audit_result", "audit_query",
                    "audit_mode", "audit_expert", "audit_top_k", "saved_labels"]:
            st.session_state.pop(key, None)
        st.rerun()
```

Replace it with:

```python
_title_col, _logout_col = st.columns([5, 1])
with _title_col:
    st.title("Retrieval Audit")
    st.caption("Illinois Legal RAG — Expert Labeling")
with _logout_col:
    st.write("")
    _hist_open = st.session_state.get("history_open", False)
    if st.button("📋 History" if not _hist_open else "✕ History", use_container_width=True):
        _now_open = not _hist_open
        st.session_state["history_open"] = _now_open
        if _now_open:
            st.session_state["history_data"] = get_feedback_history(expert_id)
            st.session_state["history_view"] = "list"
            st.session_state["history_selected_qid"] = None
        else:
            for k in ["history_data", "history_view", "history_selected_qid"]:
                st.session_state.pop(k, None)
        st.rerun()
    if st.button("Logout", use_container_width=True):
        for key in ["authenticated", "user_email", "audit_result", "audit_query",
                    "audit_mode", "audit_expert", "audit_top_k", "saved_labels",
                    "history_open", "history_view", "history_selected_qid", "history_data"]:
            st.session_state.pop(key, None)
        st.rerun()
```

- [ ] **Step 4: Wrap all remaining content in a conditional column layout**

After the header block and `expert_id = st.session_state["user_email"]` line, insert:

```python
# -- Layout: conditional right-panel split -----------------------------------

if st.session_state.get("history_open"):
    _main_col, _hist_col = st.columns([3, 1])
else:
    _main_col = st.container()
```

Then wrap everything from `# -- Query input` (the `query_input = st.text_area(...)` line) to the end of the file inside `with _main_col:` (indent all those lines by 4 spaces). The header block (`_title_col`/`_logout_col`) stays outside `with _main_col:` so it always renders at full width. After the `with _main_col:` block, add:

```python
if st.session_state.get("history_open"):
    with _hist_col:
        _render_history_panel(expert_id)
```

- [ ] **Step 5: Manually verify the toggle works**

```bash
streamlit run audit_app.py
```

1. Log in. The header should show "📋 History" and "Logout" stacked in the top-right.
2. Click "📋 History" — the button label becomes "✕ History", the page splits into a wide left column and a narrow right column showing "History panel (coming soon)".
3. Click "✕ History" — panel closes, layout returns to full width.
4. Click "Logout" — returns to login screen.

- [ ] **Step 6: Commit**

```bash
git add audit_app.py
git commit -m "feat: add history panel toggle and column layout to audit app"
```

---

## Task 3: Implement `_render_history_panel` — list view

**Files:**
- Modify: `audit_app.py`

Implement `_group_history()` and `_render_history_list()`, then wire `_render_history_panel` to use them.

- [ ] **Step 1: Add `_group_history` helper**

Add the following function immediately before the stub `_render_history_panel`:

```python
def _group_history(rows: list[dict]) -> list[dict]:
    """Group audit_feedback rows by query_id, sorted by most recent created_at."""
    groups: dict[str, dict] = {}
    for row in rows:
        qid = row["query_id"]
        if qid not in groups:
            groups[qid] = {
                "query_id": qid,
                "query_text": row["query_text"],
                "labels": [],
                "latest": row["created_at"],
                "retrieval_mode": row["retrieval_mode"],
            }
        groups[qid]["labels"].append(row["label"])
        if row["created_at"] > groups[qid]["latest"]:
            groups[qid]["latest"] = row["created_at"]
    return sorted(groups.values(), key=lambda g: g["latest"], reverse=True)
```

- [ ] **Step 2: Add `_render_history_list` function**

Add immediately after `_group_history`:

```python
def _render_history_list(rows: list[dict], expert_id: str) -> None:
    groups = _group_history(rows)

    if not groups:
        st.caption("No feedback submitted yet.")
        return

    n_labels = len(rows)
    n_queries = len(groups)
    st.caption(f"{n_labels} label{'s' if n_labels != 1 else ''} across "
               f"{n_queries} quer{'ies' if n_queries != 1 else 'y'} — all time")

    for g in groups:
        binding = g["labels"].count("BINDING")
        relevant = g["labels"].count("RELEVANT")
        irrelevant = g["labels"].count("IRRELEVANT")
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(g["latest"].replace("Z", "+00:00"))
            date_str = dt.strftime("%b %-d")
        except Exception:
            date_str = g["latest"][:10]

        truncated = g["query_text"][:58] + ("…" if len(g["query_text"]) > 58 else "")
        btn_label = f"{truncated}  \n🟢 {binding}  🟡 {relevant}  🔴 {irrelevant}  ·  {date_str}"

        if st.button(btn_label, key=f"hist_q_{g['query_id']}", use_container_width=True):
            st.session_state["history_view"] = "detail"
            st.session_state["history_selected_qid"] = g["query_id"]
            st.rerun()

    st.divider()
    st.caption("Labels as of when the panel opened.")
    if st.button("↻ Refresh", key="hist_refresh", use_container_width=True):
        st.session_state["history_data"] = get_feedback_history(expert_id)
        st.rerun()
```

- [ ] **Step 3: Replace the `_render_history_panel` stub**

Replace the stub:

```python
def _render_history_panel(expert_id: str) -> None:
    st.markdown("**📋 Feedback History**  (coming soon)")
```

with:

```python
def _render_history_panel(expert_id: str) -> None:
    col_title, col_close = st.columns([4, 1])
    with col_title:
        st.markdown("**📋 Feedback History**")
    with col_close:
        if st.button("✕", key="hist_close", use_container_width=True):
            for k in ["history_open", "history_view", "history_selected_qid", "history_data"]:
                st.session_state.pop(k, None)
            st.rerun()

    rows = st.session_state.get("history_data") or []
    view = st.session_state.get("history_view", "list")

    if view == "list":
        _render_history_list(rows, expert_id)
    else:
        qid = st.session_state.get("history_selected_qid")
        detail_rows = [r for r in rows if r["query_id"] == qid]
        groups = _group_history(rows)
        _render_history_detail(detail_rows, len(groups), expert_id)
```

Note: `_render_history_detail` is a stub added in Step 4 below — add it now so the panel doesn't crash when the detail view is triggered.

- [ ] **Step 4: Add a stub `_render_history_detail`**

Add immediately after `_render_history_list`:

```python
def _render_history_detail(rows: list[dict], total_queries: int, expert_id: str) -> None:
    if st.button(f"← All queries ({total_queries})", key="hist_back"):
        st.session_state["history_view"] = "list"
        st.session_state["history_selected_qid"] = None
        st.rerun()
    st.caption("(detail view coming in next task)")
```

- [ ] **Step 5: Manually verify the list view**

```bash
streamlit run audit_app.py
```

1. Open the history panel. The panel header shows "📋 Feedback History" and a "✕" close button.
2. If the logged-in expert has past labels: rows appear showing query text, colored dot counts, and date.
3. If no past labels: "No feedback submitted yet." caption appears.
4. The "↻ Refresh" button at the bottom re-fetches from Supabase and reruns.
5. The "✕" button in the panel header closes the panel (same as the header button).

- [ ] **Step 6: Commit**

```bash
git add audit_app.py
git commit -m "feat: implement history panel list view"
```

---

## Task 4: Implement `_render_history_detail` — drill-down view

**Files:**
- Modify: `audit_app.py`

Replace the stub `_render_history_detail` with the full implementation.

- [ ] **Step 1: Replace `_render_history_detail` stub with the full implementation**

Replace:

```python
def _render_history_detail(rows: list[dict], total_queries: int, expert_id: str) -> None:
    if st.button(f"← All queries ({total_queries})", key="hist_back"):
        st.session_state["history_view"] = "list"
        st.session_state["history_selected_qid"] = None
        st.rerun()
    st.caption("(detail view coming in next task)")
```

with:

```python
def _render_history_detail(rows: list[dict], total_queries: int, expert_id: str) -> None:
    if st.button(f"← All queries ({total_queries})", key="hist_back"):
        st.session_state["history_view"] = "list"
        st.session_state["history_selected_qid"] = None
        st.rerun()

    if not rows:
        st.caption("No data for this query.")
        return

    query_text = rows[0]["query_text"]
    retrieval_mode = rows[0]["retrieval_mode"] or "hybrid"
    n_labels = len(rows)
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(rows[0]["created_at"].replace("Z", "+00:00"))
        date_str = dt.strftime("%b %-d")
    except Exception:
        date_str = rows[0]["created_at"][:10]

    st.markdown(f"**{query_text}**")
    st.caption(f"{n_labels} labels · {date_str} · {retrieval_mode}")

    if st.button("↻ Refresh", key="hist_detail_refresh", use_container_width=True):
        st.session_state["history_data"] = get_feedback_history(expert_id)
        st.rerun()

    post_rows = sorted(
        [r for r in rows if r.get("post_rerank_rank") is not None],
        key=lambda r: r["post_rerank_rank"],
    )
    dropped_rows = sorted(
        [r for r in rows if r.get("post_rerank_rank") is None],
        key=lambda r: r.get("pre_rerank_rank") or 0,
    )

    _LABEL_ICON = {"BINDING": "🟢", "RELEVANT": "🟡", "IRRELEVANT": "🔴"}

    def _chunk_card(row: dict, stage: str) -> None:
        rank = row["post_rerank_rank"] if stage == "post" else row["pre_rerank_rank"]
        rank_str = f"post #{rank}" if stage == "post" else f"pre #{rank}"
        rrf_str = f"RRF {row['rrf_score']:.4f}" if row.get("rrf_score") is not None else ""
        ce_str = f"CE {row['ce_score']:.2f}" if row.get("ce_score") is not None else ""
        scores = "  ·  ".join(s for s in [rank_str, rrf_str, ce_str] if s)
        icon = _LABEL_ICON.get(row["label"], "⚪")
        with st.container(border=True):
            st.markdown(f"**{row['citation']}**  {icon} {row['label']}")
            st.caption(scores)
            if row.get("comment"):
                st.caption(f"_{row['comment']}_")

    tab_post, tab_dropped = st.tabs([
        f"Post-rerank ({len(post_rows)})",
        f"Dropped ({len(dropped_rows)})",
    ])

    with tab_post:
        if post_rows:
            for row in post_rows:
                _chunk_card(row, "post")
        else:
            st.caption("No post-rerank labels for this query.")

    with tab_dropped:
        if dropped_rows:
            for row in dropped_rows:
                _chunk_card(row, "pre")
        else:
            st.caption("No dropped-candidate labels for this query.")
```

- [ ] **Step 2: Manually verify the detail view end-to-end**

```bash
streamlit run audit_app.py
```

1. Open the history panel. Click a query row. The panel switches to the detail view.
2. Verify the query text appears at the top, followed by label count, date, and retrieval mode.
3. Post-rerank tab shows chunks with `post_rerank_rank` set, sorted by rank ascending. Each card shows citation, label icon + text, rank/RRF/CE scores, and (if present) italicized notes.
4. Dropped tab shows chunks with `post_rerank_rank` null, sorted by `pre_rerank_rank` ascending.
5. "← All queries (N)" button returns to the list view.
6. "↻ Refresh" re-fetches from Supabase and stays in detail view.

- [ ] **Step 3: Test with a real labeling round-trip**

In the main search area:
1. Run a query (e.g. "good time credit Illinois").
2. Label two candidates — one BINDING with a note, one IRRELEVANT.
3. Open the history panel. The query appears in the list with 🟢 1 🔴 1.
4. Click the query. Both labeled chunks appear in the appropriate tab with the correct label and note.

- [ ] **Step 4: Commit**

```bash
git add audit_app.py
git commit -m "feat: implement history panel drill-down detail view"
```

---

## Task 5: Deploy to server and smoke-test

**Files:** none (deployment only)

- [ ] **Step 1: Push to remote**

```bash
git push
```

- [ ] **Step 2: Deploy to server**

```bash
ssh ubuntu@163.192.97.229
cd legal_rag && git pull
sudo systemctl restart audit-app
```

- [ ] **Step 3: Smoke-test on production**

Open `https://ejp-rag-audit.com`, log in, and verify:

1. Header shows "📋 History" and "Logout" stacked in top-right.
2. History panel opens and closes without page errors.
3. Past labels load correctly from Supabase (if any exist for the account).
4. Drill-down view shows correct chunks, labels, and scores.
5. BM25 banner still appears and WebSocket stays alive (panel should not interfere with the `@st.fragment` keepalive).
6. Logout clears the history panel cleanly.

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add audit_app.py
git commit -m "fix: <describe any prod-only fix>"
git push
ssh ubuntu@163.192.97.229 "cd legal_rag && git pull && sudo systemctl restart audit-app"
```
