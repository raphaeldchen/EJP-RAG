# Audit App Authentication — Design Spec

**Date:** 2026-05-12
**Status:** Approved

## Overview

Add login/signup authentication to `audit_app.py` before deploying it to lawyers. Lawyers self-register with email + password; accounts start in a pending state until the admin manually approves them via an in-app admin page. Only approved accounts can log in.

## Constraints

- Deployment target: server (e.g. Fly.io, Railway, EC2)
- No email notifications — admin emails lawyers manually after approving
- Admin protected by a single `ADMIN_PASSWORD` env var (only one admin: the project owner)
- No "remember me" / persistent cookies — session lives for the duration of the browser tab
- No external auth service — credentials stored in Supabase alongside existing tables

## Database Schema

New table `lawyer_accounts` in Supabase:

```sql
CREATE TABLE lawyer_accounts (
    id            uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    email         text UNIQUE NOT NULL,
    password_hash text NOT NULL,   -- bcrypt
    approved      boolean DEFAULT false,
    created_at    timestamptz DEFAULT now(),
    approved_at   timestamptz      -- set when admin clicks Approve
);
```

- Email normalized to lowercase on write
- Password hashed with bcrypt (default work factor 12)
- `approved = false` by default; admin flips to `true` via the admin page

Migration lives at `migrations/002_create_lawyer_accounts.sql`.

## Auth Flow

### Signup
1. Lawyer fills in email + password + confirm password
2. Validation: passwords match, email not already registered
3. Row inserted with `approved = false`
4. UI shows: "Account created — you'll receive an email when it's approved."

### Login
1. Lawyer enters email + password
2. Row looked up by email; bcrypt hash verified
3. If hash matches but `approved = false` → "Your account is pending approval."
4. If hash matches and `approved = true` → session granted:
   - `st.session_state["authenticated"] = True`
   - `st.session_state["user_email"] = email`

### Session
- State held in `st.session_state` for the browser tab lifetime
- Logout button in sidebar clears both session keys, returns to login screen
- No cookies; no cross-session persistence

## File Structure

### New files

**`migrations/002_create_lawyer_accounts.sql`**
SQL migration for the `lawyer_accounts` table.

**`auth/accounts.py`**
Auth business logic, isolated from UI:
- `signup(email, password) -> dict` — validates uniqueness, bcrypt-hashes password, inserts row
- `login(email, password) -> tuple[bool, str]` — verifies hash, checks approval, returns `(success, message)`
- `list_accounts() -> list[dict]` — returns all rows ordered by `created_at` (for admin page)
- `approve_account(account_id: str)` — sets `approved = true`, `approved_at = now()`

**`pages/admin.py`**
Streamlit admin page (auto-discovered by Streamlit's multi-page system via the `pages/` directory). The page link appears in the sidebar nav for all logged-in lawyers — this is fine, since the page itself requires the admin password before showing anything sensitive.
- Renders a password form on first load
- On correct `ADMIN_PASSWORD` entry: sets `st.session_state["admin_authenticated"] = True`
- When admin-authenticated: shows all `lawyer_accounts` rows (email, created_at, approved status) with an Approve button per pending account
- Logout button clears `admin_authenticated`

### Modified files

**`audit_app.py`**
- Auth gate added at the top (after `st.set_page_config`): if `st.session_state["authenticated"]` is not set, render login/signup tabs and call `st.stop()`
- Login tab: email + password form, calls `auth.accounts.login()`
- Signup tab: email + password + confirm password form, calls `auth.accounts.signup()`
- Existing audit content is otherwise untouched
- `expert_id` text input removed; `expert_id` auto-populated from `st.session_state["user_email"]`
- Logout button added to sidebar (visible only when authenticated)

**`.env`**
Add: `ADMIN_PASSWORD=<your-chosen-password>`

## Dependencies

- `bcrypt` — add to venv (`venv/bin/pip install bcrypt`)
- No other new packages

## Security Notes

- bcrypt with default cost factor (12 rounds) — appropriate for a low-traffic internal tool
- `SUPABASE_SERVICE_KEY` (already in env) used for all `lawyer_accounts` operations — service key bypasses row-level security, which is acceptable here since all DB access goes through the app
- No rate limiting on login attempts — acceptable given the small, known user base; can add later if needed
- Admin password is compared via `== ` string equality (constant-time comparison not required for a single-admin env var gate)

## Out of Scope

- Email confirmation on signup
- Password reset flow
- "Remember me" / persistent sessions
- Per-lawyer persona selection (deferred to post-lawyer-collaboration per CLAUDE.md)
- Rate limiting or account lockout
