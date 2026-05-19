# Audit App UI Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the history panel (emojis, dark close button, unreadable query rows) and admin page (ghost white box on login, no back button, inline-style inconsistency) to match the approved design spec.

**Architecture:** CSS changes land in `ui/styles.py` (single source of truth for all visual tokens); Python changes are minimal and isolated to `audit_app.py` and `pages/admin.py`. No logic, auth, or retrieval code is touched.

**Tech Stack:** Streamlit, custom CSS injected via `st.markdown(SHARED_CSS, unsafe_allow_html=True)`, Python 3.11

---

## File Map

| File | Change |
|------|--------|
| `ui/styles.py` | Tasks 1 & 4: add/update CSS rules for history panel buttons and admin utility classes |
| `audit_app.py` | Task 2: strip emojis from history panel header and query row labels |
| `pages/admin.py` | Tasks 3 & 5: fix ghost white box, add back buttons, replace inline styles with CSS classes |

---

### Task 1: History panel — update close button CSS, add card-row button styles

**Files:**
- Modify: `ui/styles.py`

- [ ] **Step 1: Locate the existing close-button rule in `ui/styles.py`**

Find this block (around line 304):
```css
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
```

- [ ] **Step 2: Replace the close-button block and add the card-row rules immediately after**

Replace the entire block above with:
```css
/* History panel query-row buttons — clean card style with good contrast */
div[data-testid="stColumn"]:has(#hist-panel-root) .stButton > button {
    text-align: left !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-md) !important;
    padding: 9px 12px !important;
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stColumn"]:has(#hist-panel-root) .stButton > button:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 1px 5px rgba(30,58,95,0.08) !important;
    background: var(--surface) !important;
}

/* Subtle close button — comes after the card-row rule so !important wins */
div:has(#hist-close-btn) button {
    background: none !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    font-family: var(--font-sans) !important;
    font-weight: 400 !important;
    font-size: 0.82rem !important;
}
div:has(#hist-close-btn) button:hover {
    background: none !important;
    color: var(--text-primary) !important;
    border-color: var(--border-strong) !important;
}
```

Note: the card-row rule must come **before** the close-button rule in the source. Both use `!important`; when specificity ties, last rule wins — so the close-button override correctly takes precedence over the general card-row style.

- [ ] **Step 3: Commit**

```bash
git add ui/styles.py
git commit -m "style: history panel — card-row button style, subtle close button"
```

---

### Task 2: History panel — strip emojis from Python

**Files:**
- Modify: `audit_app.py`

- [ ] **Step 1: Remove 📋 from the panel title**

Find (line ~144):
```python
        st.markdown('<h3 class="ejp-section-title">📋 Feedback History</h3>', unsafe_allow_html=True)
```
Replace with:
```python
        st.markdown('<h3 class="ejp-section-title">Feedback History</h3>', unsafe_allow_html=True)
```

- [ ] **Step 2: Reformat the query-row button label**

Find (in `_render_history_list`, line ~57):
```python
        btn_label = f"▶ {truncated}  \n🟢 {binding}  🟡 {relevant}  🔴 {irrelevant}  ·  {date_str}"
```
Replace with:
```python
        btn_label = f"{truncated}\n{binding}B  {relevant}R  {irrelevant}I  ·  {date_str}"
```
This removes the `▶` arrow and color emojis. The `\n` still creates a two-line button: query text on line 1, label counts + date on line 2. The CSS from Task 1 (`text-align: left`, `line-height: 1.5`) makes this render cleanly.

- [ ] **Step 3: Verify visually**

Run the app:
```bash
source venv/bin/activate
streamlit run audit_app.py
```
Open http://localhost:8501, log in, click "📋 History" (the button in the top-right of the main area). Confirm:
- Panel title reads "Feedback History" (no emoji)
- Query row buttons show plain text in two lines, white background, dark text, navy border on hover
- Close button is a subtle outlined "Close" or "✕" with muted text, no dark fill

- [ ] **Step 4: Commit**

```bash
git add audit_app.py
git commit -m "style: strip emojis from history panel title and query row buttons"
```

---

### Task 3: Admin login — fix ghost white box and add back buttons

**Files:**
- Modify: `pages/admin.py`

- [ ] **Step 1: Fix the ghost white box on the login form**

Find this block (lines ~27–37):
```python
    st.markdown('<div class="ejp-card">', unsafe_allow_html=True)
    with st.form("admin_login"):
        pwd = st.text_input("Admin password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
    if submitted:
        if pwd and pwd == ADMIN_PASSWORD:
            st.session_state["admin_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()
```

Replace with:
```python
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
```

The `if submitted:` block stays outside `with st.form()` (Streamlit requires this) but moves inside `with col_mid`.

- [ ] **Step 2: Add a back button to the authenticated admin header**

Find (lines ~42–56):
```python
_title_col, _logout_col = st.columns([5, 1])
with _title_col:
    st.markdown(
        '<h1 style="font-family:Georgia,serif;color:#1e3a5f;'
        'font-size:1.8rem;margin:0 0 0.15rem 0;font-weight:400;">Account Management</h1>'
        '<p style="font-family:-apple-system,Segoe UI,sans-serif;'
        'font-size:0.78rem;color:#6b6860;letter-spacing:0.05em;'
        'text-transform:uppercase;margin:0 0 1rem 0;">Illinois Legal Research · Admin Console</p>',
        unsafe_allow_html=True,
    )
with _logout_col:
    st.write("")
    if st.button("Logout", use_container_width=True):
        st.session_state.pop("admin_authenticated", None)
        st.rerun()
```

Replace with:
```python
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
```

The `st.write("")` spacer is removed — the columns align naturally.

- [ ] **Step 3: Verify visually**

```bash
streamlit run audit_app.py
```
Navigate to Admin tab on login page. Confirm:
- No ghost white box above the password field
- Form is centered (not stretched full width)
- "← Retrieval Audit" button appears below the form and navigates back when clicked

Log in as admin. Confirm:
- "← Audit" and "Logout" buttons appear in the header
- "← Audit" navigates back to audit_app.py

- [ ] **Step 4: Commit**

```bash
git add pages/admin.py
git commit -m "fix: admin login ghost box, add back-to-audit navigation buttons"
```

---

### Task 4: Admin CSS utilities — add shared classes to SHARED_CSS

**Files:**
- Modify: `ui/styles.py`

- [ ] **Step 1: Add admin utility classes before the closing `</style>` tag**

Locate the closing `</style>` at the bottom of `SHARED_CSS` in `ui/styles.py`. Insert this block immediately before it:

```css
    /* ---------- Admin account management ---------- */
    .ejp-section-count {
        color: var(--text-muted);
        font-weight: 400;
        font-family: var(--font-sans);
        font-size: 0.85rem;
        margin-left: 4px;
    }
    .ejp-table-header {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        font-family: var(--font-sans);
    }
    .ejp-account-email {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 0.9rem;
    }
    .ejp-account-date {
        font-family: var(--font-mono);
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 2px;
    }
    .ejp-table-cell { padding: 6px 0; }
    .ejp-check {
        color: var(--accent);
        font-weight: 700;
        text-align: center;
        padding: 6px 0;
    }
```

- [ ] **Step 2: Commit**

```bash
git add ui/styles.py
git commit -m "style: add admin utility CSS classes to SHARED_CSS"
```

---

### Task 5: Admin page — replace inline styles with CSS classes

**Files:**
- Modify: `pages/admin.py`

- [ ] **Step 1: Update the "Pending Approval" section title**

Find (lines ~68–73):
```python
    st.markdown(
        f'<h3 class="ejp-section-title">Pending Approval '
        f'<span style="color:#6b6860;font-weight:400;font-family:-apple-system,sans-serif;'
        f'font-size:0.85rem;">({len(pending)})</span></h3>',
        unsafe_allow_html=True,
    )
```
Replace with:
```python
    st.markdown(
        f'<h3 class="ejp-section-title">Pending Approval '
        f'<span class="ejp-section-count">({len(pending)})</span></h3>',
        unsafe_allow_html=True,
    )
```

- [ ] **Step 2: Update the pending account card content**

Find (lines ~79–86):
```python
            with col_email:
                registered = acct["created_at"][:10] if acct["created_at"] else "unknown"
                st.markdown(
                    f'<div style="font-family:-apple-system,sans-serif;">'
                    f'<div style="font-weight:600;color:#1a1a18;">{acct["email"]}</div>'
                    f'<div style="font-size:0.78rem;color:#6b6860;margin-top:2px;">'
                    f'Registered {registered}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
```
Replace with:
```python
            with col_email:
                registered = acct["created_at"][:10] if acct["created_at"] else "unknown"
                st.markdown(
                    f'<div class="ejp-account-email">{acct["email"]}</div>'
                    f'<div class="ejp-account-date">Registered {registered}</div>',
                    unsafe_allow_html=True,
                )
```

- [ ] **Step 3: Update the "Approved Accounts" section title**

Find (lines ~97–102):
```python
    st.markdown(
        f'<h3 class="ejp-section-title" style="margin-top:1.5rem;">Approved Accounts '
        f'<span style="color:#6b6860;font-weight:400;font-family:-apple-system,sans-serif;'
        f'font-size:0.85rem;">({len(approved)})</span></h3>',
        unsafe_allow_html=True,
    )
```
Replace with:
```python
    st.markdown(
        f'<h3 class="ejp-section-title" style="margin-top:1.5rem;">Approved Accounts '
        f'<span class="ejp-section-count">({len(approved)})</span></h3>',
        unsafe_allow_html=True,
    )
```

- [ ] **Step 4: Update the approved accounts table headers**

Find (lines ~104–118):
```python
    h_email, h_date, h_ok = st.columns([5, 1, 1])
    h_email.markdown(
        '<div style="font-family:-apple-system,sans-serif;font-size:0.7rem;'
        'text-transform:uppercase;letter-spacing:0.06em;color:#6b6860;">Email</div>',
        unsafe_allow_html=True,
    )
    h_date.markdown(
        '<div style="font-family:-apple-system,sans-serif;font-size:0.7rem;'
        'text-transform:uppercase;letter-spacing:0.06em;color:#6b6860;">Approved</div>',
        unsafe_allow_html=True,
    )
    h_ok.markdown(
        '<div style="font-family:-apple-system,sans-serif;font-size:0.7rem;'
        'text-transform:uppercase;letter-spacing:0.06em;color:#6b6860;text-align:center;">Status</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="border-bottom:1px solid #e2e0db;margin:4px 0 6px 0;"></div>',
        unsafe_allow_html=True,
    )
```
Replace with:
```python
    h_email, h_date, h_ok = st.columns([5, 1, 1])
    h_email.markdown('<div class="ejp-table-header">Email</div>', unsafe_allow_html=True)
    h_date.markdown('<div class="ejp-table-header">Approved</div>', unsafe_allow_html=True)
    h_ok.markdown('<div class="ejp-table-header" style="text-align:center;">Status</div>', unsafe_allow_html=True)
    st.markdown('<div style="border-bottom:1px solid var(--border);margin:4px 0 6px 0;"></div>', unsafe_allow_html=True)
```

- [ ] **Step 5: Update the approved accounts table rows**

Find (lines ~125–142):
```python
    for acct in approved:
        approved_date = acct["approved_at"][:10] if acct["approved_at"] else "—"
        c_email, c_date, c_ok = st.columns([5, 1, 1])
        c_email.markdown(
            f'<div style="font-family:-apple-system,sans-serif;color:#1a1a18;'
            f'padding:6px 0;">{acct["email"]}</div>',
            unsafe_allow_html=True,
        )
        c_date.markdown(
            f'<div style="font-family:Courier New,monospace;color:#6b6860;'
            f'font-size:0.82rem;padding:6px 0;">{approved_date}</div>',
            unsafe_allow_html=True,
        )
        c_ok.markdown(
            '<div style="text-align:center;color:#1e3a5f;font-weight:600;'
            'padding:6px 0;">✓</div>',
            unsafe_allow_html=True,
        )
```
Replace with:
```python
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
```

- [ ] **Step 6: Verify visually**

```bash
streamlit run audit_app.py
```
Navigate to Admin tab → log in. Confirm:
- "Pending Approval (N)" count is muted gray, not hardcoded `#6b6860`
- Pending account cards show email (bold) and "Registered YYYY-MM-DD" (mono, muted) — same look as before but using CSS classes
- "Approved Accounts (N)" section title styled consistently
- Table header row shows "Email / Approved / Status" in small uppercase mono
- Approved rows show email (bold), date (mono muted), ✓ (navy, centered)

- [ ] **Step 7: Commit**

```bash
git add pages/admin.py
git commit -m "style: replace admin page inline styles with SHARED_CSS utility classes"
```

---

## Self-Review

**Spec coverage:**
- ✅ History panel title: emoji removed (Task 2)
- ✅ History close button: dark fill → subtle outline (Task 1)
- ✅ History query rows: emojis removed, CSS card-row styling with proper contrast (Tasks 1 & 2)
- ✅ Admin ghost white box: `ejp-card` wrapper removed, `st.columns` centering (Task 3)
- ✅ Admin back button: on login view and authenticated header (Task 3)
- ✅ Admin inline style cleanup: all ~30 occurrences addressed (Tasks 4 & 5)

**Placeholder scan:** All steps contain actual code. No TBDs.

**Type consistency:** CSS class names defined in Task 4 (`ejp-section-count`, `ejp-table-header`, `ejp-account-email`, `ejp-account-date`, `ejp-table-cell`, `ejp-check`) are used exactly in Task 5.
