"""Shared CSS for the EJP-RAG Streamlit apps.

Inject once per page, after st.set_page_config and before any rendering:

    from ui.styles import SHARED_CSS
    st.markdown(SHARED_CSS, unsafe_allow_html=True)

Color tokens, type stack, pill/badge classes, and Streamlit chrome overrides
all live here so the audit, chat, and admin apps stay visually aligned.
"""

SHARED_CSS = """
<style>
    /* ---------- Design tokens ---------- */
    :root {
        --bg:            #fafaf8;   /* warm off-white parchment */
        --surface:       #ffffff;
        --surface-alt:   #f4f2ee;
        --border:        #e2e0db;
        --border-strong: #c9c6bf;
        --text-primary:  #1a1a18;
        --text-muted:    #6b6860;
        --accent:        #1e3a5f;   /* deep navy */
        --accent-hover:  #15293f;
        --accent-light:  #eef2f7;
        --binding:       #166534;   /* ILCS deep green */
        --binding-bg:    #dcfce7;
        --relevant:      #92400e;   /* amber-brown */
        --relevant-bg:   #fef3c7;
        --irrelevant:    #991b1b;   /* dark red */
        --irrelevant-bg: #fee2e2;
        --score-high:    #dcfce7;
        --score-mid:     #fef9c3;
        --score-low:     #fee2e2;

        --font-serif:  "Georgia", "Times New Roman", serif;
        --font-sans:   -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        --font-mono:   "Courier New", "Consolas", "Menlo", monospace;

        --radius-sm: 4px;
        --radius-md: 6px;
        --radius-lg: 8px;
        --shadow-card: 0 2px 8px rgba(0, 0, 0, 0.06);
    }

    /* ---------- Streamlit chrome ---------- */
    #MainMenu, footer, header { visibility: hidden; }
    .stApp,
    .stAppViewContainer,
    section.main { background-color: var(--bg) !important; }

    /* Body type baseline */
    html, body, .stApp, [class*="st-"] {
        color: var(--text-primary);
        font-family: var(--font-sans);
    }

    /* Headings — serif for legal authority */
    h1, h2, h3, h4 {
        font-family: var(--font-serif) !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.01em;
    }

    /* ---------- Brand header (injected) ---------- */
    .ejp-brand {
        text-align: center;
        font-family: var(--font-serif);
        margin: 0 0 1.5rem 0;
    }
    .ejp-brand .ejp-mark {
        font-size: 2.25rem;
        color: var(--accent);
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .ejp-brand .ejp-title {
        font-size: 2rem;
        font-weight: 400;
        color: var(--accent);
        margin: 0;
        letter-spacing: -0.015em;
    }
    .ejp-brand .ejp-rule {
        width: 56px;
        height: 1px;
        background: var(--border-strong);
        margin: 0.85rem auto;
    }
    .ejp-brand .ejp-subtitle {
        font-family: var(--font-sans);
        font-size: 0.85rem;
        color: var(--text-muted);
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin: 0;
    }

    /* Page-section serif heading used in admin + history */
    .ejp-section-title {
        font-family: var(--font-serif);
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 0.75rem 0;
        letter-spacing: -0.005em;
    }

    /* ---------- Login card wrapper ---------- */
    .ejp-card {
        max-width: 480px;
        margin: 0 auto;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 2rem;
        box-shadow: var(--shadow-card);
    }

    /* ---------- Pills (replaces .citation-badge / .source-pill) ---------- */
    .pill {
        display: inline-block;
        font-family: var(--font-mono);
        font-size: 0.7rem;
        color: var(--accent);
        background: var(--surface);
        border: 1px solid var(--accent);
        border-radius: var(--radius-sm);
        padding: 1px 8px;
        margin: 2px 3px 2px 0;
        line-height: 1.5;
        letter-spacing: 0.02em;
        white-space: nowrap;
    }
    .pill.pill-muted {
        color: var(--text-muted);
        border-color: var(--border-strong);
    }

    /* ---------- Label badges (post-save) ---------- */
    .badge-binding,
    .badge-relevant,
    .badge-irrelevant {
        display: inline-block;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: var(--radius-sm);
        letter-spacing: 0.04em;
    }
    .badge-binding    { background: var(--binding-bg);    color: var(--binding); }
    .badge-relevant   { background: var(--relevant-bg);   color: var(--relevant); }
    .badge-irrelevant { background: var(--irrelevant-bg); color: var(--irrelevant); }

    /* ---------- Score badges ---------- */
    .score-badge {
        display: inline-block;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        padding: 1px 8px;
        border-radius: var(--radius-sm);
        margin: 2px 3px 2px 0;
        color: var(--text-primary);
        background: var(--surface-alt);
        border: 1px solid var(--border);
    }
    .score-badge.score-high { background: var(--score-high); border-color: #b6e6c8; }
    .score-badge.score-mid  { background: var(--score-mid);  border-color: #efe49a; }
    .score-badge.score-low  { background: var(--score-low);  border-color: #f3c3c3; }

    /* ---------- Query echo block ---------- */
    .query-echo {
        border-left: 3px solid var(--accent);
        padding: 0.6rem 0.9rem;
        margin: 1.25rem 0 1rem 0;
        background: var(--surface);
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    }
    .query-echo .query-label {
        display: block;
        font-family: var(--font-mono);
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--accent);
        margin-bottom: 0.25rem;
    }
    .query-echo .query-text {
        font-family: var(--font-serif);
        font-size: 1.05rem;
        font-style: italic;
        color: var(--text-primary);
        line-height: 1.45;
    }

    /* ---------- Chunk-card internals ---------- */
    .chunk-meta-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 4px 6px;
        margin-bottom: 0.5rem;
    }
    pre.chunk-text {
        font-family: var(--font-mono);
        font-size: 0.82rem;
        line-height: 1.5;
        overflow-x: auto;
        border-left: 3px solid var(--border-strong);
        padding: 0.65rem 0.9rem;
        background: var(--surface-alt);
        color: var(--text-primary);
        white-space: pre-wrap;
        word-break: break-word;
        margin: 0.4rem 0 0.9rem 0;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    }

    /* Streamlit native widgets — quiet them down to match palette */
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted) !important;
        font-family: var(--font-sans) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: var(--font-serif) !important;
        color: var(--accent) !important;
    }

    /* Active tab underline */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--border);
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-sans);
        color: var(--text-muted);
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    /* Expanders */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        font-family: var(--font-mono);
        font-size: 0.85rem;
        color: var(--text-primary);
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        background: var(--surface);
        margin-bottom: 0.5rem;
    }

    /* Primary buttons — navy */
    .stButton button[kind="primary"],
    button[kind="primary"] {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #ffffff !important;
        font-family: var(--font-sans) !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }
    .stButton button[kind="primary"]:hover {
        background-color: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
    }
    .stButton button[kind="secondary"],
    .stButton button {
        font-family: var(--font-sans);
        border-radius: var(--radius-sm) !important;
    }

    /* Form inputs — replace orange focus with navy */
    input:focus,
    textarea:focus,
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    [data-baseweb="input"]:focus-within,
    [data-baseweb="textarea"]:focus-within,
    [data-baseweb="select"]:focus-within {
        outline: 2px solid var(--accent) !important;
        outline-offset: -1px !important;
        border-color: var(--accent) !important;
        box-shadow: none !important;
    }

    /* History panel column — bordered left rail with navy cap */
    div[data-testid="stColumn"]:has(#hist-panel-root) {
        background: var(--surface);
        border-left: 1px solid var(--border);
        border-top: 3px solid var(--accent);
        padding: 0.85rem 1rem 0.5rem 1rem;
        border-radius: 0;
    }
    /* Dark close button in history panel */
    div:has(#hist-close-btn) button {
        background: var(--text-primary);
        color: #ffffff;
        border-color: var(--text-primary);
        font-family: var(--font-sans);
    }
    div:has(#hist-close-btn) button:hover {
        background: #000;
        border-color: #000;
    }

    /* History row cards */
    .hist-row {
        border-top: 1px solid var(--border);
        padding-top: 0.5rem;
        margin-top: 0.5rem;
    }

    /* Chat (app.py) empty-state */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 55vh;
        color: var(--text-primary);
        text-align: center;
    }
    .empty-state .es-mark {
        font-size: 2.5rem;
        color: var(--accent);
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .empty-state h1 {
        font-family: var(--font-serif) !important;
        font-size: 1.9rem;
        font-weight: 400;
        color: var(--accent) !important;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .empty-state .es-rule {
        width: 48px;
        height: 1px;
        background: var(--border-strong);
        margin: 0.8rem auto;
    }
    .empty-state p {
        font-family: var(--font-sans);
        color: var(--text-muted);
        font-size: 0.85rem;
        letter-spacing: 0.03em;
        margin: 0;
    }

    /* Chat input pill — keep Ollama-like minimal but navy focus */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.25rem 0;
    }
    .stChatInputContainer {
        background-color: var(--bg) !important;
        border: none !important;
        padding: 0.5rem 0 1.5rem 0;
    }
    div[data-testid="stChatInput"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 24px !important;
        padding: 0.25rem 1rem !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        outline: 2px solid var(--accent) !important;
        outline-offset: -1px !important;
        border-color: var(--accent) !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: var(--surface) !important;
        color: var(--text-primary) !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        min-height: 24px !important;
        max-height: 120px !important;
        padding: 0.35rem 0 !important;
        font-family: var(--font-sans) !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        overflow-y: auto !important;
        resize: none !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-muted) !important;
    }

    /* Assistant avatar badge — wrap ⚖️ in a navy circle */
    [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="chatAvatarIcon-assistant"] {
        background: var(--accent) !important;
        color: #ffffff !important;
        border-radius: 50% !important;
    }

    /* Dividers */
    hr, [data-testid="stDivider"] { border-color: var(--border) !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--surface-alt) !important;
        border-right: 1px solid var(--border);
    }
</style>
"""
