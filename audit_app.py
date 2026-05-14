import json
from datetime import datetime
import streamlit as st
from mcp_server.server import _audit_retrieval, submit_feedback, is_bm25_ready, get_feedback_history
from auth.accounts import signup, login

st.set_page_config(page_title="Retrieval Audit", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .stApp { background: #ffffff; }
    .score-high  { background: #d1fae5; border-radius: 6px; padding: 2px 8px; font-size: 0.8rem; }
    .score-mid   { background: #fef3c7; border-radius: 6px; padding: 2px 8px; font-size: 0.8rem; }
    .score-low   { background: #fee2e2; border-radius: 6px; padding: 2px 8px; font-size: 0.8rem; }
    .citation-badge {
        display: inline-block; background: #f0f0f0; color: #374151;
        font-size: 0.75rem; padding: 2px 10px; border-radius: 12px;
        font-family: monospace; margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


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


def _show_login():
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)
    if submitted:
        if not email:
            st.error("Email is required.")
        else:
            success, msg = login(email, password)
            if success:
                st.session_state["authenticated"] = True
                st.session_state["user_email"] = email.lower().strip()
                st.rerun()
            else:
                st.error(msg)


def _show_signup():
    with st.form("signup_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create Account", use_container_width=True)
    if submitted:
        if not email:
            st.error("Email is required.")
        elif len(password) < 8:
            st.error("Password must be at least 8 characters.")
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            success, msg = signup(email, password)
            if success:
                st.success("Account created — you'll receive an email when it's approved.")
            else:
                st.error(msg)


# -- Auth gate -----------------------------------------------------------------

if not st.session_state.get("authenticated"):
    st.title("Retrieval Audit")
    st.caption("Illinois Legal RAG — Expert Labeling")
    tab_login, tab_signup, tab_admin = st.tabs(["Login", "Sign Up", "Admin"])
    with tab_login:
        _show_login()
    with tab_signup:
        _show_signup()
    with tab_admin:
        st.write("Admin access only.")
        if st.button("Go to Admin Panel", type="primary"):
            st.switch_page("pages/admin.py")
    st.stop()

# -- Sidebar (authenticated) ---------------------------------------------------

with st.sidebar:
    st.write(f"Logged in as **{st.session_state['user_email']}**")
    if st.button("⚙️ Admin Panel"):
        st.switch_page("pages/admin.py")

expert_id = st.session_state["user_email"]


# -- Header --------------------------------------------------------------------

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
            for k in ["history_open", "history_data", "history_view", "history_selected_qid"]:
                st.session_state.pop(k, None)
        st.rerun()
    if st.button("Logout", use_container_width=True):
        for key in ["authenticated", "user_email", "audit_result", "audit_query",
                    "audit_mode", "audit_expert", "audit_top_k", "saved_labels",
                    "history_open", "history_view", "history_selected_qid", "history_data"]:
            st.session_state.pop(key, None)
        st.rerun()

# -- Layout: conditional right-panel split -----------------------------------

if st.session_state.get("history_open"):
    _main_col, _hist_col = st.columns([3, 1])
else:
    _main_col = st.container()

with _main_col:
    # -- Query input ---------------------------------------------------------------

    query_input = st.text_area(
        "Legal Query",
        height=80,
        placeholder="e.g. What is good-time credit in Illinois?",
        label_visibility="collapsed",
    )

    # -- Settings row under search bar ---------------------------------------------

    s1, s2, s3 = st.columns([2, 2, 1])

    with s1:
        mode = st.selectbox(
            "Retrieval Mode",
            ["Hybrid (production)", "Vector only", "BM25 only"],
            help="Hybrid = production. Vector/BM25 = diagnostic modes.",
        )
    with s2:
        top_k = st.slider("Candidates to show", min_value=5, max_value=60, value=20, step=5)
    with s3:
        st.write("")
        search_btn = st.button("Search", type="primary", use_container_width=True)


# -- Helpers (module-level, not inside any column context) --------------------

def _mode_key(mode_label: str) -> str:
    return {"Hybrid (production)": "hybrid", "Vector only": "vector", "BM25 only": "bm25"}.get(mode_label, "hybrid")


def _score_class(ce_score):
    if ce_score is None:
        return ""
    if ce_score >= 2.0:
        return "score-high"
    if ce_score >= 0.0:
        return "score-mid"
    return "score-low"


def _render_card(chunk, position, stage, query, mode_key, expert_id, post_rerank_position=None):
    ce = chunk.get("ce_score")
    css_class = _score_class(ce)
    ce_label = f"CE: {ce:.2f}" if ce is not None else "no CE score"
    rrf_label = f"RRF: {chunk['rrf_score']:.4f}"
    key = f"{chunk['chunk_id']}_{stage}_{position}"

    if "saved_labels" not in st.session_state:
        st.session_state["saved_labels"] = {}

    with st.expander(
        f"#{position}  {chunk['citation']}  ·  {chunk['source']}  ·  {rrf_label}  ·  {ce_label}",
        expanded=False,
    ):
        st.markdown(
            f'<span class="citation-badge">{chunk["citation"]}</span>  '
            f'<span class="citation-badge">{chunk["source"]}</span>  '
            f'<span class="{css_class}">{ce_label}</span>',
            unsafe_allow_html=True,
        )
        st.text(chunk["text"])

        col1, col2, col3, col_note, col_save = st.columns([1, 1, 1, 3, 1])
        with col1:
            binding = st.button("BINDING", key=f"b_{key}", type="primary")
        with col2:
            relevant = st.button("RELEVANT", key=f"r_{key}")
        with col3:
            irrelevant = st.button("IRREL.", key=f"i_{key}")
        with col_note:
            comment = st.text_input("Notes", key=f"c_{key}",
                                    label_visibility="collapsed", placeholder="Optional notes...")
        with col_save:
            save_note = st.button("Save Note", key=f"n_{key}", use_container_width=True)

        def _do_submit(lbl, cmt):
            submit_feedback(
                query=query,
                chunk_id=chunk["chunk_id"],
                citation=chunk["citation"],
                source=chunk["source"],
                retrieval_mode=mode_key,
                pre_rerank_rank=position if stage == "pre_rerank" else 0,
                post_rerank_rank=post_rerank_position,
                rrf_score=chunk["rrf_score"],
                ce_score=chunk.get("ce_score"),
                label=lbl,
                comment=cmt,
                expert_id=expert_id,
            )

        label = "BINDING" if binding else ("RELEVANT" if relevant else ("IRRELEVANT" if irrelevant else None))
        if label:
            try:
                _do_submit(label, comment)
                st.session_state["saved_labels"][key] = label
                st.success(f"Saved: {label}")
            except Exception as e:
                st.error(f"Save failed: {e}")
        elif save_note:
            prior_label = st.session_state["saved_labels"].get(key)
            if not prior_label:
                st.warning("Assign a label (BINDING / RELEVANT / IRREL.) before saving a note.")
            else:
                try:
                    _do_submit(prior_label, comment)
                    st.success("Note saved.")
                except Exception as e:
                    st.error(f"Save failed: {e}")


# -- Search and display --------------------------------------------------------

@st.fragment(run_every=5)
def _bm25_status_banner():
    if not is_bm25_ready():
        st.info("BM25 index is loading in the background — searches work now (vector-only) and will automatically include BM25 once ready.")
    else:
        # Always render something once BM25 is ready.  The fragment firing every 5 s
        # keeps the WebSocket alive through Cloudflare's idle-timeout window; an empty
        # render may not generate a delta large enough to reset that timer.
        st.session_state.setdefault("_bm25_banner_dismissed", True)
        st.caption("● Hybrid retrieval active")


with _main_col:
    _bm25_status_banner()

    if search_btn and query_input:
        mk = _mode_key(mode)
        with st.spinner("Searching..."):
            raw = _audit_retrieval(query_input, mode=mk, top_k=top_k)
        st.session_state["audit_result"] = json.loads(raw)
        st.session_state["audit_query"] = query_input
        st.session_state["audit_mode"] = mk
        st.session_state["audit_expert"] = expert_id
        st.session_state["audit_top_k"] = top_k

    elif search_btn and not query_input:
        st.warning("Enter a query first.")

    if "audit_result" in st.session_state:
        result = st.session_state["audit_result"]
        q = st.session_state["audit_query"]
        mk = st.session_state["audit_mode"]
        eid = st.session_state["user_email"]
        saved_top_k = st.session_state.get("audit_top_k", top_k)

        st.divider()
        st.subheader(f"Results for: {q}")
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        col_meta1.metric("Candidates (pre-rerank)", len(result["candidates"]))
        col_meta2.metric("Survived reranking", len(result["reranked"]))
        col_meta3.metric("Dropped", len(result["dropped"]))

        post_rerank_map = {c["chunk_id"]: i + 1 for i, c in enumerate(result["reranked"])}

        # Dropped = candidates that did not survive reranking, with their original pre-rerank rank preserved.
        dropped = [(i + 1, chunk) for i, chunk in enumerate(result["candidates"])
                   if chunk["chunk_id"] not in post_rerank_map]

        tab_post, tab_pre = st.tabs([
            f"Post-Rerank ({len(result['reranked'])} survived)",
            f"Pre-Rerank ({len(dropped)} dropped)",
        ])

        with tab_post:
            st.caption("Chunks after CrossEncoder reranking — exactly what the LLM sees in production.")
            for i, chunk in enumerate(result["reranked"]):
                _render_card(chunk, i + 1, "post_rerank", q, mk, eid,
                             post_rerank_position=i + 1)

        with tab_pre:
            st.caption(
                f"Candidates dropped by the reranker. "
                f"The {len(result['reranked'])} post-rerank survivors are excluded — label them in the Post-Rerank tab."
            )
            num_dropped = len(dropped)
            show_all = st.toggle(
                f"Show all {num_dropped} dropped candidates",
                value=False,
                key="show_all_candidates",
            )
            visible = dropped if show_all else dropped[:saved_top_k]
            if not show_all and num_dropped > saved_top_k:
                st.caption(f"Showing top {saved_top_k} of {num_dropped}. Toggle above to show all.")
            for orig_rank, chunk in visible:
                _render_card(chunk, orig_rank, "pre_rerank", q, mk, eid,
                             post_rerank_position=None)

    else:
        st.info("Enter a query above and click Search to begin.")

if st.session_state.get("history_open"):
    with _hist_col:
        _render_history_panel(expert_id)
