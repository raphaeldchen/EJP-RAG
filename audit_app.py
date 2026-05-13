import json
import streamlit as st
from mcp_server.server import _audit_retrieval, submit_feedback
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
    if st.button("Logout"):
        for key in ["authenticated", "user_email", "audit_result", "audit_query",
                    "audit_mode", "audit_expert", "audit_top_k", "saved_labels"]:
            st.session_state.pop(key, None)
        st.rerun()

expert_id = st.session_state["user_email"]


# -- Header --------------------------------------------------------------------

st.title("Retrieval Audit")
st.caption("Illinois Legal RAG — Expert Labeling")

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


# -- Helpers -------------------------------------------------------------------

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
        show_all = st.toggle(
            f"Show all {len(dropped)} dropped candidates",
            value=False,
            key="show_all_candidates",
        )
        visible = dropped if show_all else dropped[:saved_top_k]
        if not show_all and len(dropped) > saved_top_k:
            st.caption(f"Showing top {saved_top_k} of {len(dropped)}. Toggle above to show all.")
        for orig_rank, chunk in visible:
            _render_card(chunk, orig_rank, "pre_rerank", q, mk, eid,
                         post_rerank_position=None)

else:
    st.info("Enter a query above and click Search to begin.")
