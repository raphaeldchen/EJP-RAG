# Audit App UI Fixes — Design Spec
**Date:** 2026-05-18

## Problem

Two areas of the audit app are visually unpolished and need fixing:

1. **History side panel** — emoji-heavy query rows, a jarring dark close button, and button text that doesn't contrast well with button backgrounds
2. **Admin login page** — a ghost white box above the password field (caused by an uncloseable `ejp-card` div wrapper), and no way to navigate back to the main audit app

Separately, a sweep for inline-style inconsistencies in `admin.py` is needed to bring it in line with `SHARED_CSS`.

---

## Scope

Three files change:
- `ui/styles.py` — add CSS rules for history panel buttons and close button
- `audit_app.py` — strip emojis from history panel header and query row labels
- `pages/admin.py` — fix ghost white box on login, add back button, replace inline styles with CSS classes

No changes to retrieval logic, data model, or other pages.

---

## Design Decisions

### 1. History Panel — header and close button

**Current:** `📋 Feedback History` as the title (emoji), close button styled as a filled dark block (`background: #1f2937; color: #fff`).

**Fix:**
- Title becomes `Feedback History` — plain serif text, no emoji, using the existing `ejp-section-title` CSS class
- Close button becomes a subtle outlined text button: `border: 1px solid var(--border)`, `background: none`, `color: var(--text-muted)` — matching the visual weight of the caption text around it
- The existing `#hist-close-btn` marker div approach is retained; the CSS rule that targets it in `SHARED_CSS` is updated

### 2. History Panel — query row buttons

**Current:** `st.button()` labels use `▶ {truncated}\n🟢 {binding}  🟡 {relevant}  🔴 {irrelevant}  · {date_str}`. The `\n` doesn't render as a proper line break in all Streamlit versions. Emojis are ugly. Button background and text contrast is unreliable due to Streamlit's default secondary button styles not being explicitly overridden.

**Fix (two parts):**

*Python (`audit_app.py`):*
- Remove `▶` and `🟢 🟡 🔴` from the label
- Format: `f"{truncated}\n{binding}B  {relevant}R  {irrelevant}I  · {date_str}"`
- The `\n` still acts as a soft separator between query text and label counts — Streamlit does render it as a line break inside the button element

*CSS (`ui/styles.py`):*
- Add a rule scoped to the history panel column (already targeted by `div[data-testid="stColumn"]:has(#hist-panel-root)`) that overrides secondary buttons to look like clean card rows:
  - `background: var(--surface)`, `border: 1px solid var(--border)`, `color: var(--text-primary)`
  - `text-align: left`, `padding: 9px 12px`, `border-radius: var(--radius-md)`
  - Hover: `border-color: var(--accent)`, subtle box-shadow
- This scoping ensures these overrides don't affect buttons anywhere else in the app

### 3. Admin login — ghost white box

**Cause:** `st.markdown('<div class="ejp-card">')` injects an open HTML tag, but Streamlit renders each `st.markdown()` call in isolation — the div doesn't wrap the subsequent `st.form()`. The result is an empty white box rendered above the form.

**Fix:** Remove the `st.markdown('<div class="ejp-card">')` and `st.markdown('</div>')` calls. Center the form using `st.columns([1, 2, 1])` and render the form inside the middle column. The `.ejp-card` CSS class is preserved for potential future use but not applied here.

### 4. Admin login — back button

**Current:** No navigation back to the main audit app from the admin login page.

**Fix:** Add `st.button("← Retrieval Audit")` below the login form that calls `st.switch_page("audit_app.py")`. This appears on both the unauthenticated login view and the authenticated admin view (in the sidebar or header area).

### 5. Admin page — inline style cleanup

**Current:** `pages/admin.py` has ~30 occurrences of hardcoded inline `style="font-family:...; color:#..."` attributes that bypass the design token system.

**Fix:** Extract recurring patterns to named CSS classes in `SHARED_CSS`:
- `.ejp-table-header` — uppercase mono label for table column headers
- `.ejp-account-email` — account email display (semibold, primary text color)
- `.ejp-account-date` — date display (mono, muted color)
- `.ejp-check` — green checkmark cell (centered, accent color)

Replace all inline style attributes in `admin.py` with these class references.

---

## What Does Not Change

- The `ejp-brand` login header (⚖️ mark, serif title, rule, subtitle) — this already looks good
- The main audit app header, query input, chunk cards, metrics, tabs — no changes
- The `ejp-card` CSS class definition — kept for potential future use
- Retrieval logic, data model, auth logic

---

## Files Changed

| File | Change |
|------|--------|
| `ui/styles.py` | Add history panel button styles, update `#hist-close-btn` CSS, add admin table CSS classes |
| `audit_app.py` | Strip emojis from history panel title and query row `btn_label` |
| `pages/admin.py` | Fix ghost box (remove card div wrappers), add back button, replace inline styles with CSS classes |
