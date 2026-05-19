import os
import streamlit as st
from dotenv import load_dotenv
from auth.accounts import list_accounts, approve_account
from ui.styles import SHARED_CSS

load_dotenv()

st.set_page_config(page_title="Admin — Retrieval Audit", page_icon="⚖️", layout="wide")
st.markdown(SHARED_CSS, unsafe_allow_html=True)

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

if not st.session_state.get("admin_authenticated"):
    st.markdown(
        '<div class="ejp-brand">'
        '<div class="ejp-mark">⚖️</div>'
        '<h1 class="ejp-title">Illinois Legal Research</h1>'
        '<div class="ejp-rule"></div>'
        '<p class="ejp-subtitle">Admin Console</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    if ADMIN_PASSWORD in ("", "CHANGE_ME_BEFORE_DEPLOY"):
        st.error("ADMIN_PASSWORD is not configured. Replace the placeholder in your .env file.")
        st.stop()
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        with st.form("admin_login"):
            pwd = st.text_input("Admin password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
        if submitted:
            if pwd and pwd == ADMIN_PASSWORD:
                st.session_state["admin_authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        if st.button("← Retrieval Audit", use_container_width=True):
            st.switch_page("audit_app.py")
    st.stop()

# Admin authenticated

_title_col, _back_col, _logout_col = st.columns([4, 1, 1])
with _title_col:
    st.markdown(
        '<h1 style="font-family:Georgia,serif;color:#1e3a5f;'
        'font-size:1.8rem;margin:0 0 0.15rem 0;font-weight:400;">Account Management</h1>'
        '<p style="font-family:-apple-system,Segoe UI,sans-serif;'
        'font-size:0.78rem;color:#6b6860;letter-spacing:0.05em;'
        'text-transform:uppercase;margin:0 0 1rem 0;">Illinois Legal Research · Admin Console</p>',
        unsafe_allow_html=True,
    )
with _back_col:
    if st.button("← Audit", use_container_width=True):
        st.switch_page("audit_app.py")
with _logout_col:
    if st.button("Logout", use_container_width=True):
        st.session_state.pop("admin_authenticated", None)
        st.rerun()

accounts = list_accounts()

if not accounts:
    st.info("No accounts registered yet.")
    st.stop()

pending = [a for a in accounts if not a["approved"]]
approved = [a for a in accounts if a["approved"]]

if pending:
    st.markdown(
        f'<h3 class="ejp-section-title">Pending Approval '
        f'<span class="ejp-section-count">({len(pending)})</span></h3>',
        unsafe_allow_html=True,
    )
    for acct in pending:
        with st.container(border=True):
            col_email, col_btn = st.columns([5, 1])
            with col_email:
                registered = acct["created_at"][:10] if acct["created_at"] else "unknown"
                st.markdown(
                    f'<div class="ejp-account-email">{acct["email"]}</div>'
                    f'<div class="ejp-account-date">Registered {registered}</div>',
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("Approve", key=f"approve_{acct['id']}",
                             type="primary", use_container_width=True):
                    approve_account(acct["id"])
                    st.success(f"Approved {acct['email']}")
                    st.rerun()
else:
    st.info("No pending accounts.")

if approved:
    st.markdown(
        f'<h3 class="ejp-section-title" style="margin-top:1.5rem;">Approved Accounts '
        f'<span class="ejp-section-count">({len(approved)})</span></h3>',
        unsafe_allow_html=True,
    )
    # Table header
    h_email, h_date, h_ok = st.columns([5, 1, 1])
    h_email.markdown('<div class="ejp-table-header">Email</div>', unsafe_allow_html=True)
    h_date.markdown('<div class="ejp-table-header">Approved</div>', unsafe_allow_html=True)
    h_ok.markdown('<div class="ejp-table-header" style="text-align:center;">Status</div>', unsafe_allow_html=True)
    st.markdown('<div style="border-bottom:1px solid var(--border);margin:4px 0 6px 0;"></div>', unsafe_allow_html=True)

    for acct in approved:
        approved_date = acct["approved_at"][:10] if acct["approved_at"] else "—"
        c_email, c_date, c_ok = st.columns([5, 1, 1])
        c_email.markdown(
            f'<div class="ejp-account-email ejp-table-cell">{acct["email"]}</div>',
            unsafe_allow_html=True,
        )
        c_date.markdown(
            f'<div class="ejp-account-date ejp-table-cell">{approved_date}</div>',
            unsafe_allow_html=True,
        )
        c_ok.markdown('<div class="ejp-check">✓</div>', unsafe_allow_html=True)
