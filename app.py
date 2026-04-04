import streamlit as st
from retrieval.main import build_rag, query

st.set_page_config(
    page_title="Illinois Legal Research",
    page_icon="⚖️",
    layout="centered",
)

st.markdown("""
<style>
    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Pure white background everywhere */
    .stApp, .stAppViewContainer, section.main { background-color: #ffffff; }

    /* No background on message bubbles — just clean white like Ollama */
    .stChatMessage {
        background-color: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.25rem 0;
    }

    /* Chat input — rounded pill, light gray, no border, like Ollama */
    .stChatInputContainer {
        background-color: #ffffff !important;
        border: none !important;
        padding: 0.5rem 0 1.5rem 0;
    }
    div[data-testid="stChatInput"] {
        background-color: #f0f0f0 !important;
        border: none !important;
        border-radius: 24px !important;
        padding: 0.25rem 1rem !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        box-shadow: none !important;
        border: none !important;
        outline: none !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: #f0f0f0 !important;
        color: #111111 !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        min-height: 24px !important;
        max-height: 120px !important;
        padding: 0.35rem 0 !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        overflow-y: auto !important;
        resize: none !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #9ca3af !important;
    }
    /* Kill the orange focus ring at every level */
    div[data-testid="stChatInput"] *:focus {
        outline: none !important;
        box-shadow: none !important;
        border-color: transparent !important;
    }

    /* Source citation pills — subtle gray */
    .source-pill {
        display: inline-block;
        background-color: #f0f0f0;
        color: #374151;
        font-size: 0.72rem;
        padding: 2px 10px;
        border-radius: 12px;
        margin: 3px 3px;
        font-family: monospace;
    }

    /* Center the empty-state title like Ollama */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 55vh;
        color: #111111;
    }
    .empty-state h1 {
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0;
    }
    .empty-state p {
        color: #9ca3af;
        font-size: 0.9rem;
        margin-top: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load RAG engine (cached so it only initialises once) ─────────────────────
@st.cache_resource(show_spinner="Loading legal database…")
def load_engine():
    return build_rag(use_local=True)

engine, retrievers = load_engine()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Empty state — centred title (hidden once chat starts) ─────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <h1>Illinois Legal Research</h1>
        <p>Illinois Compiled Statutes · Supreme Court Rules · Case Law</p>
    </div>
    """, unsafe_allow_html=True)

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚖️"):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg.get("sources"):
            st.markdown(
                " ".join(f'<span class="source-pill">{s}</span>' for s in msg["sources"]),
                unsafe_allow_html=True,
            )

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Send a message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="⚖️"):
        try:
            with st.spinner("Researching…"):
                raw = query(engine, retrievers, prompt)

            sources = []
            if "\n\nSources:\n" in raw:
                answer, sources_block = raw.split("\n\nSources:\n", 1)
                sources = [
                    line.strip().lstrip("•").strip()
                    for line in sources_block.splitlines()
                    if line.strip()
                ]
            else:
                answer = raw

            st.markdown(answer)
            if sources:
                st.markdown(
                    " ".join(f'<span class="source-pill">{s}</span>' for s in sources),
                    unsafe_allow_html=True,
                )

        except Exception as exc:
            error_type = type(exc).__name__
            if "Connect" in error_type or "connect" in str(exc).lower():
                answer = "_Connection to the database failed. Please try again in a moment._"
            else:
                answer = f"_An error occurred: {error_type}. Please try again._"
            sources = []
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
