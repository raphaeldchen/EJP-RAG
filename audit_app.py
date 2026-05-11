import json
import streamlit as st
from mcp_server.server import _audit_retrieval, submit_feedback

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

s1, s2, s3, s4, s5 = st.columns([2, 2, 2, 2, 1])

with s1:
    mode = st.selectbox(
        "Retrieval Mode",
        ["Hybrid (production)", "Vector only", "BM25 only"],
        help="Hybrid = production. Vector/BM25 = diagnostic modes.",
    )
with s2:
    persona = st.selectbox("Persona", ["Researcher", "Practitioner", "Incarcerated Person"])
with s3:
    top_k = st.slider("Candidates to show", min_value=5, max_value=60, value=20, step=5)
with s4:
    expert_id = st.text_input("Your name / email", placeholder="Optional — for attribution")
with s5:
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


def _render_card(chunk, position, stage, query, mode_key, persona, expert_id, post_rerank_position=None):
    ce = chunk.get("ce_score")
    css_class = _score_class(ce)
    ce_label = f"CE: {ce:.2f}" if ce is not None else "no CE score"
    rrf_label = f"RRF: {chunk['rrf_score']:.4f}"
    key = f"{chunk['chunk_id']}_{stage}_{position}"

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

        col1, col2, col3, col_note = st.columns([1, 1, 1, 3])
        with col1:
            binding = st.button("BINDING", key=f"b_{key}", type="primary")
        with col2:
            relevant = st.button("RELEVANT", key=f"r_{key}")
        with col3:
            irrelevant = st.button("IRREL.", key=f"i_{key}")
        with col_note:
            comment = st.text_input("Notes", key=f"c_{key}",
                                    label_visibility="collapsed", placeholder="Optional notes...")

        label = "BINDING" if binding else ("RELEVANT" if relevant else ("IRRELEVANT" if irrelevant else None))
        if label:
            try:
                submit_feedback(
                    query=query,
                    chunk_id=chunk["chunk_id"],
                    citation=chunk["citation"],
                    source=chunk["source"],
                    retrieval_mode=mode_key,
                    persona=persona.lower().replace(" ", "_"),
                    pre_rerank_rank=position if stage == "pre_rerank" else 0,
                    post_rerank_rank=post_rerank_position,
                    rrf_score=chunk["rrf_score"],
                    ce_score=chunk.get("ce_score"),
                    label=label,
                    comment=comment,
                    expert_id=expert_id,
                )
                st.success(f"Saved: {label}")
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
    st.session_state["audit_persona"] = persona
    st.session_state["audit_expert"] = expert_id
    st.session_state["audit_top_k"] = top_k

elif search_btn and not query_input:
    st.warning("Enter a query first.")

if "audit_result" in st.session_state:
    result = st.session_state["audit_result"]
    q = st.session_state["audit_query"]
    mk = st.session_state["audit_mode"]
    p = st.session_state["audit_persona"]
    eid = st.session_state["audit_expert"]
    saved_top_k = st.session_state.get("audit_top_k", top_k)

    st.divider()
    st.subheader(f"Results for: {q}")
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    col_meta1.metric("Candidates (pre-rerank)", len(result["candidates"]))
    col_meta2.metric("Survived reranking", len(result["reranked"]))
    col_meta3.metric("Dropped", len(result["dropped"]))

    if result.get("rewritten_query"):
        st.caption(f"Query rewritten to: {result['rewritten_query']}")

    post_rerank_map = {c["chunk_id"]: i + 1 for i, c in enumerate(result["reranked"])}

    tab_post, tab_pre = st.tabs([
        f"Post-Rerank ({len(result['reranked'])} survived)",
        f"Pre-Rerank ({len(result['candidates'])} candidates)",
    ])

    with tab_post:
        st.caption("Chunks after CrossEncoder reranking — exactly what the LLM sees in production.")
        for i, chunk in enumerate(result["reranked"]):
            _render_card(chunk, i + 1, "post_rerank", q, mk, p, eid,
                         post_rerank_position=i + 1)

    with tab_pre:
        st.caption("All chunks before reranking. Label here to flag retrieval failures.")
        show_all = st.toggle(
            f"Show all {len(result['candidates'])} candidates",
            value=False,
            key="show_all_candidates",
        )
        visible = result["candidates"] if show_all else result["candidates"][:saved_top_k]
        if not show_all and len(result["candidates"]) > saved_top_k:
            st.caption(f"Showing top {saved_top_k} of {len(result['candidates'])}. Toggle above to show all.")
        for i, chunk in enumerate(visible):
            _render_card(chunk, i + 1, "pre_rerank", q, mk, p, eid,
                         post_rerank_position=post_rerank_map.get(chunk["chunk_id"]))

else:
    st.info("Enter a query above and click Search to begin.")
