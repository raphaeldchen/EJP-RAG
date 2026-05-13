import os
import streamlit as st
from dotenv import load_dotenv
from auth.accounts import list_accounts, approve_account

load_dotenv()

st.set_page_config(page_title="Admin — Retrieval Audit", page_icon="⚖️", layout="wide")

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

if not st.session_state.get("admin_authenticated"):
    st.title("Admin Login")
    if not ADMIN_PASSWORD:
        st.error("ADMIN_PASSWORD is not configured. Add it to your .env file.")
        st.stop()
    with st.form("admin_login"):
        pwd = st.text_input("Admin password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)
    if submitted:
        if pwd and pwd == ADMIN_PASSWORD:
            st.session_state["admin_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# Admin authenticated

with st.sidebar:
    st.write("Admin panel")
    if st.button("Logout"):
        st.session_state.pop("admin_authenticated", None)
        st.rerun()

st.title("Account Management")

accounts = list_accounts()

if not accounts:
    st.info("No accounts registered yet.")
    st.stop()

pending = [a for a in accounts if not a["approved"]]
approved = [a for a in accounts if a["approved"]]

if pending:
    st.subheader(f"Pending Approval ({len(pending)})")
    for acct in pending:
        col_email, col_btn = st.columns([5, 1])
        with col_email:
            registered = acct["created_at"][:10] if acct["created_at"] else "unknown"
            st.write(f"**{acct['email']}** — registered {registered}")
        with col_btn:
            if st.button("Approve", key=f"approve_{acct['id']}"):
                approve_account(acct["id"])
                st.success(f"Approved {acct['email']}")
                st.rerun()
else:
    st.info("No pending accounts.")

if approved:
    st.subheader(f"Approved Accounts ({len(approved)})")
    for acct in approved:
        approved_date = acct["approved_at"][:10] if acct["approved_at"] else "—"
        st.write(f"✓ **{acct['email']}** — approved {approved_date}")
