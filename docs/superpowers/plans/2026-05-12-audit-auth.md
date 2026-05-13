# Audit App Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add email/password login and signup to `audit_app.py`, with an admin-approval gate and an in-app admin page for approving pending accounts.

**Architecture:** A `lawyer_accounts` Supabase table stores bcrypt-hashed credentials and an `approved` boolean. `auth/accounts.py` contains all auth business logic; `audit_app.py` wraps existing content with a session-state gate; `pages/admin.py` is a separate Streamlit page protected by an `ADMIN_PASSWORD` env var.

**Tech Stack:** Python, Streamlit, Supabase (existing), bcrypt

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `migrations/002_create_lawyer_accounts.sql` | Create | Schema for `lawyer_accounts` table |
| `auth/__init__.py` | Create | Package marker (empty) |
| `auth/accounts.py` | Create | `signup`, `login`, `list_accounts`, `approve_account` |
| `tests/test_auth_accounts.py` | Create | Unit tests for all auth functions |
| `audit_app.py` | Modify | Auth gate, login/signup UI, remove expert_id input |
| `pages/admin.py` | Create | Admin password gate + account approval UI |
| `.env` | Modify | Add `ADMIN_PASSWORD` |

---

## Task 1: Install bcrypt and create migration

**Files:**
- Create: `migrations/002_create_lawyer_accounts.sql`

- [ ] **Step 1: Install bcrypt into the venv**

```bash
source venv/bin/activate
venv/bin/pip install bcrypt
```

Expected: `Successfully installed bcrypt-X.X.X`

- [ ] **Step 2: Create the migration file**

Create `migrations/002_create_lawyer_accounts.sql`:

```sql
CREATE TABLE IF NOT EXISTS lawyer_accounts (
    id            uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    email         text UNIQUE NOT NULL,
    password_hash text NOT NULL,
    approved      boolean DEFAULT false,
    created_at    timestamptz DEFAULT now(),
    approved_at   timestamptz
);

CREATE INDEX IF NOT EXISTS lawyer_accounts_email_idx ON lawyer_accounts (email);
CREATE INDEX IF NOT EXISTS lawyer_accounts_approved_idx ON lawyer_accounts (approved);
```

- [ ] **Step 3: Run the migration in Supabase**

Open the Supabase dashboard → SQL editor → paste the contents of `migrations/002_create_lawyer_accounts.sql` → Run. Verify the `lawyer_accounts` table appears in the Table Editor.

- [ ] **Step 4: Commit**

```bash
git add migrations/002_create_lawyer_accounts.sql
git commit -m "feat: add lawyer_accounts migration for audit app auth"
```

---

## Task 2: Write failing tests for auth/accounts.py

**Files:**
- Create: `auth/__init__.py`
- Create: `tests/test_auth_accounts.py`

- [ ] **Step 1: Create the auth package**

Create `auth/__init__.py` as an empty file.

- [ ] **Step 2: Write the test file**

Create `tests/test_auth_accounts.py`:

```python
import bcrypt
import pytest
from unittest.mock import patch, MagicMock
from auth.accounts import signup, login, list_accounts, approve_account


def _hashed(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


class TestSignup:
    def test_creates_account_with_normalized_email(self):
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []

        with patch("auth.accounts._client", return_value=mock_client):
            success, msg = signup("Lawyer@Example.com", "password123")

        assert success is True
        inserted = mock_client.table.return_value.insert.call_args[0][0]
        assert inserted["email"] == "lawyer@example.com"
        assert inserted["approved"] is False
        assert bcrypt.checkpw(b"password123", inserted["password_hash"].encode())

    def test_rejects_duplicate_email(self):
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [{"id": "x"}]

        with patch("auth.accounts._client", return_value=mock_client):
            success, msg = signup("existing@example.com", "password123")

        assert success is False
        assert "already exists" in msg


class TestLogin:
    def test_success(self):
        hashed = _hashed("correct")
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"email": "a@b.com", "password_hash": hashed, "approved": True}
        ]

        with patch("auth.accounts._client", return_value=mock_client):
            success, _ = login("a@b.com", "correct")

        assert success is True

    def test_wrong_password(self):
        hashed = _hashed("correct")
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"email": "a@b.com", "password_hash": hashed, "approved": True}
        ]

        with patch("auth.accounts._client", return_value=mock_client):
            success, msg = login("a@b.com", "wrong")

        assert success is False
        assert "Incorrect password" in msg

    def test_pending_approval(self):
        hashed = _hashed("pw")
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"email": "a@b.com", "password_hash": hashed, "approved": False}
        ]

        with patch("auth.accounts._client", return_value=mock_client):
            success, msg = login("a@b.com", "pw")

        assert success is False
        assert "pending" in msg.lower()

    def test_no_account(self):
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []

        with patch("auth.accounts._client", return_value=mock_client):
            success, msg = login("nobody@example.com", "pw")

        assert success is False
        assert "No account" in msg

    def test_normalizes_email_before_lookup(self):
        hashed = _hashed("pw")
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"email": "a@b.com", "password_hash": hashed, "approved": True}
        ]

        with patch("auth.accounts._client", return_value=mock_client):
            login("A@B.COM", "pw")

        mock_client.table.return_value.select.return_value.eq.assert_called_with(
            "email", "a@b.com"
        )


class TestListAccounts:
    def test_returns_all_accounts(self):
        rows = [
            {"id": "1", "email": "a@b.com", "approved": True,
             "created_at": "2026-05-12T00:00:00Z", "approved_at": None}
        ]
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.order.return_value.execute.return_value.data = rows

        with patch("auth.accounts._client", return_value=mock_client):
            result = list_accounts()

        assert result == rows


class TestApproveAccount:
    def test_sets_approved_and_timestamp(self):
        mock_client = MagicMock()

        with patch("auth.accounts._client", return_value=mock_client):
            approve_account("some-uuid")

        updated = mock_client.table.return_value.update.call_args[0][0]
        assert updated["approved"] is True
        assert "approved_at" in updated
        mock_client.table.return_value.update.return_value.eq.assert_called_with(
            "id", "some-uuid"
        )
```

- [ ] **Step 3: Run tests — verify they all fail with ImportError**

```bash
python3 -m pytest tests/test_auth_accounts.py -v
```

Expected: `ModuleNotFoundError: No module named 'auth.accounts'`

- [ ] **Step 4: Commit**

```bash
git add auth/__init__.py tests/test_auth_accounts.py
git commit -m "test: add failing tests for auth.accounts (signup, login, list, approve)"
```

---

## Task 3: Implement auth/accounts.py

**Files:**
- Create: `auth/accounts.py`

- [ ] **Step 1: Create the implementation**

Create `auth/accounts.py`:

```python
import bcrypt
from datetime import datetime, timezone
from supabase import create_client, Client
from retrieval.config import SUPABASE_URL, SUPABASE_SERVICE_KEY


def _client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def signup(email: str, password: str) -> tuple[bool, str]:
    email = email.lower().strip()
    client = _client()
    existing = client.table("lawyer_accounts").select("id").eq("email", email).execute()
    if existing.data:
        return False, "An account with that email already exists."
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    client.table("lawyer_accounts").insert({
        "email": email,
        "password_hash": password_hash,
        "approved": False,
    }).execute()
    return True, "Account created."


def login(email: str, password: str) -> tuple[bool, str]:
    email = email.lower().strip()
    rows = _client().table("lawyer_accounts").select("*").eq("email", email).execute()
    if not rows.data:
        return False, "No account found with that email."
    row = rows.data[0]
    if not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return False, "Incorrect password."
    if not row["approved"]:
        return False, "Your account is pending approval. You'll receive an email when it's ready."
    return True, "ok"


def list_accounts() -> list[dict]:
    return (
        _client()
        .table("lawyer_accounts")
        .select("id,email,approved,created_at,approved_at")
        .order("created_at")
        .execute()
        .data
    )


def approve_account(account_id: str) -> None:
    _client().table("lawyer_accounts").update({
        "approved": True,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", account_id).execute()
```

- [ ] **Step 2: Run tests — verify they all pass**

```bash
python3 -m pytest tests/test_auth_accounts.py -v
```

Expected: 9 tests, all `PASSED`.

- [ ] **Step 3: Commit**

```bash
git add auth/accounts.py
git commit -m "feat: implement auth.accounts — signup, login, list, approve"
```

---

## Task 4: Add auth gate and sidebar to audit_app.py

**Files:**
- Modify: `audit_app.py`

- [ ] **Step 1: Add the import and helper functions**

At the top of `audit_app.py`, after the existing imports, add:

```python
from auth.accounts import signup, login
```

After the `st.markdown(...)` CSS block (around line 20), add the two auth helper functions:

```python
def _show_login():
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)
    if submitted:
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
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            success, msg = signup(email, password)
            if success:
                st.success("Account created — you'll receive an email when it's approved.")
            else:
                st.error(msg)
```

- [ ] **Step 2: Add the auth gate after the CSS block**

After the two helper functions and before the `# -- Header ---` comment, add:

```python
# -- Auth gate -----------------------------------------------------------------

if not st.session_state.get("authenticated"):
    st.title("Retrieval Audit")
    st.caption("Illinois Legal RAG — Expert Labeling")
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
    with tab_login:
        _show_login()
    with tab_signup:
        _show_signup()
    st.stop()

# -- Sidebar (authenticated) ---------------------------------------------------

with st.sidebar:
    st.write(f"Logged in as **{st.session_state['user_email']}**")
    if st.button("Logout"):
        st.session_state.pop("authenticated", None)
        st.session_state.pop("user_email", None)
        st.rerun()

expert_id = st.session_state["user_email"]
```

- [ ] **Step 3: Remove the expert_id text input and collapse the settings row to 3 columns**

Find the settings row (currently lines ~39–53). Replace it with:

```python
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
```

(The `expert_id` variable was set above from session state, so no change needed elsewhere in the file.)

- [ ] **Step 4: Smoke-test in the browser**

```bash
streamlit run audit_app.py
```

Verify:
- Unauthenticated: login/signup tabs appear, no audit content visible
- Sign up with a new email → "Account created — you'll receive an email when it's approved."
- Try to log in with that account → "Your account is pending approval."
- Manually set `approved = true` in the Supabase dashboard for that account
- Log in → audit content visible, sidebar shows email, Logout works

- [ ] **Step 5: Commit**

```bash
git add audit_app.py
git commit -m "feat: add auth gate to audit_app — login/signup, session, logout"
```

---

## Task 5: Create admin page

**Files:**
- Create: `pages/admin.py`

- [ ] **Step 1: Create the pages directory and admin page**

```bash
mkdir -p pages
```

Create `pages/admin.py`:

```python
import os
import streamlit as st
from auth.accounts import list_accounts, approve_account

st.set_page_config(page_title="Admin — Retrieval Audit", page_icon="⚖️", layout="wide")

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

if not st.session_state.get("admin_authenticated"):
    st.title("Admin Login")
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
            st.write(f"**{acct['email']}** — registered {acct['created_at'][:10]}")
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
```

- [ ] **Step 2: Add ADMIN_PASSWORD to .env**

Open `.env` and add:

```
ADMIN_PASSWORD=<choose-a-strong-password>
```

Replace `<choose-a-strong-password>` with an actual password. This is not committed to git.

- [ ] **Step 3: Smoke-test the admin page**

```bash
streamlit run audit_app.py
```

Navigate to the Admin page via the sidebar nav. Verify:
- Without the password: locked out with error
- With the correct `ADMIN_PASSWORD`: pending accounts appear with Approve buttons
- Clicking Approve moves the account to the Approved section and the lawyer can now log in
- Admin logout clears the admin session

- [ ] **Step 4: Commit**

```bash
git add pages/admin.py
git commit -m "feat: add admin page for approving lawyer accounts"
```

---

## Self-Review

**Spec coverage:**
- ✓ Signup form (email + password + confirm) → Task 4
- ✓ Account stored pending approval → Task 3 (`approved=False`)
- ✓ Admin manually approves → Task 5 (admin page Approve button)
- ✓ Only approved accounts can log in → Task 3 (`login()` checks `approved`)
- ✓ Admin page protected by `ADMIN_PASSWORD` env var → Task 5
- ✓ `expert_id` auto-populated from session → Task 4 Step 2/3
- ✓ Logout → Task 4 Step 2 (sidebar logout)
- ✓ bcrypt password hashing → Task 3
- ✓ Migration SQL → Task 1
- ✓ Server deployment (session state, no cookies) → architecture matches

**Placeholder scan:** No TBDs or vague steps. All code blocks are complete.

**Type consistency:**
- `signup(email, password) -> tuple[bool, str]` — used consistently in tests and audit_app.py
- `login(email, password) -> tuple[bool, str]` — same
- `list_accounts() -> list[dict]` — admin page iterates over `acct["id"]`, `acct["email"]`, `acct["approved"]`, `acct["created_at"]`, `acct["approved_at"]` — all returned by the SELECT in `accounts.py`
- `approve_account(account_id: str) -> None` — called with `acct["id"]` in admin page ✓
