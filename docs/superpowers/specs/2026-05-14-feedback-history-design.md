# Feedback History Panel — Design Spec

**Date:** 2026-05-14
**Status:** Approved

## Summary

Add a "📋 Feedback History" button to the audit app that opens an inline right-side panel. The panel shows the logged-in lawyer's complete labeling history (all-time, from Supabase), organized by query. Clicking a query drills into a detail view of every chunk they labeled for that query, with ranks, scores, and notes.

## Motivation

Lawyers currently have no way to review or cross-reference their past feedback. As labeling sessions accumulate across days, being able to revisit earlier decisions — and see what was labeled BINDING vs. IRRELEVANT for a given query — is essential for calibration and for building trust in the ground-truth dataset.

## Scope

- Read-only. No editing of past labels from the history panel.
- Shows only labeled chunks (what is in `audit_feedback`) — not every candidate from a retrieval run, since unlabeled candidates are not stored.
- History is per-expert (filtered by `expert_id = user_email`).
- Data is all-time (not scoped to the current session).

## Data Source

`audit_feedback` table (Supabase). Relevant columns per row:

| Column | Used for |
|---|---|
| `query_id` | grouping rows into queries |
| `query_text` | display in list + detail header |
| `chunk_id` | uniqueness |
| `citation` | chunk identifier in detail view |
| `label` | BINDING / RELEVANT / IRRELEVANT |
| `comment` | notes in detail view |
| `pre_rerank_rank` | rank in detail card |
| `post_rerank_rank` | null = dropped; non-null = post-rerank |
| `rrf_score` | score in detail card |
| `ce_score` | score in detail card |
| `retrieval_mode` | shown in detail header |
| `created_at` | sorting queries by recency, date display |

## Backend — `mcp_server/server.py`

Add one new public function alongside `submit_feedback`:

```python
def get_feedback_history(expert_id: str) -> list[dict]:
```

- Queries `audit_feedback` where `expert_id = expert_id`, ordered by `created_at desc`.
- Selects: `query_id, query_text, chunk_id, citation, label, comment, pre_rerank_rank, post_rerank_rank, rrf_score, ce_score, retrieval_mode, created_at`.
- Returns the raw list of row dicts. Grouping and sorting by query is done in the caller.
- No pagination for now — lawyers will have tens to low-hundreds of labels, not thousands.

## Frontend — `audit_app.py`

### New imports

```python
from mcp_server.server import _audit_retrieval, submit_feedback, is_bm25_ready, get_feedback_history
```

### New session state keys

| Key | Type | Default | Purpose |
|---|---|---|---|
| `history_open` | bool | False | Whether the panel is visible |
| `history_view` | str | "list" | `"list"` or `"detail"` |
| `history_selected_qid` | str\|None | None | Which query is in detail view |
| `history_data` | list\|None | None | Cached rows from Supabase; fetched once on open |

### Button placement

The "📋 Feedback History" button goes in the existing `_logout_col` column, stacked above Logout:

```python
_title_col, _logout_col = st.columns([5, 1])
with _logout_col:
    st.write("")
    if st.button("📋 History", use_container_width=True):
        st.session_state["history_open"] = not st.session_state.get("history_open", False)
        if st.session_state["history_open"]:
            # fetch on open; reset to list view
            st.session_state["history_data"] = get_feedback_history(expert_id)
            st.session_state["history_view"] = "list"
            st.session_state["history_selected_qid"] = None
        st.rerun()
    if st.button("Logout", use_container_width=True):
        ...  # existing logic
```

Toggling the button again closes the panel (sets `history_open = False`).

### Layout restructuring

When the panel is open, split into two columns and render all existing content inside the left one:

```python
if st.session_state.get("history_open"):
    _main_col, _hist_col = st.columns([3, 1])
    with _hist_col:
        _render_history_panel(expert_id)
else:
    _main_col = st.container()  # full-width passthrough

with _main_col:
    # all existing content: BM25 banner, query input, settings row, search results
```

`st.container()` returns a `DeltaGenerator` that supports `with`, so `with _main_col:` works identically whether the panel is open or closed. No content is duplicated.

### History panel — `_render_history_panel(expert_id)`

A new top-level function, called inside `_hist_col` when open.

**Panel header:**
- Title: "📋 Feedback History"
- Close button (✕): sets `history_open = False`, clears `history_data`, reruns

**Data preparation (done once, cached in `history_data`):**
- Group rows by `query_id`
- Per group: extract `query_text`, count labels by type, find most recent `created_at`
- Sort groups by most recent `created_at` descending

**List view** (`history_view == "list"`):

Summary line: `"{total_labels} labels across {n_queries} queries — all time"`

Per query row (clickable `st.button` styled as a card via markdown/container):
- Query text truncated to ~60 chars
- Dot counts: 🟢 N  🟡 N  🔴 N  (BINDING / RELEVANT / IRRELEVANT)
- Date: `created_at.strftime("%b %-d")`
- On click: set `history_view = "detail"`, `history_selected_qid = query_id`, rerun

**Detail view** (`history_view == "detail"`):

- Back link: `← All queries ({n_queries})` — sets `history_view = "list"`, reruns
- Query text (full, wrapped)
- Meta line: `"{n_labels} labels · {date} · {retrieval_mode}"`
- Refresh button: re-fetches `get_feedback_history(expert_id)`, reruns

Tab strip (`st.tabs(["Post-rerank (N)", "Dropped (N)"])`):
- Post-rerank tab: rows where `post_rerank_rank IS NOT NULL`, sorted by `post_rerank_rank`
- Dropped tab: rows where `post_rerank_rank IS NULL`, sorted by `pre_rerank_rank`

Per chunk card (inside each tab):
- Citation + label badge (color-coded: green=BINDING, amber=RELEVANT, red=IRRELEVANT)
- Stage tag + rank: e.g. `[post #1]` or `[pre #4]`
- Scores: `RRF {rrf_score:.4f} · CE {ce_score:.2f}` (CE omitted if None)
- Notes (if `comment` is non-empty): italic, indented

**Stale data note:** A small caption at the bottom of the list view:
`"Showing labels as of when the panel was opened. Click ↻ Refresh to update."`

## Logout state cleanup

Add `history_open`, `history_view`, `history_selected_qid`, `history_data` to the logout key-clear list in the existing logout handler.

## Non-goals

- No editing of labels from the history panel (read-only).
- No export / CSV download.
- No filtering or searching within the history panel.
- No admin view of other lawyers' history (each lawyer sees only their own).
