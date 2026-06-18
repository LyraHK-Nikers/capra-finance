"""Email/password authentication for CAPRA Finance, backed by Supabase (PostgREST),
with admin approval and a super-admin role.

Design notes
------------
- Activates ONLY when Supabase secrets are present. With no secrets the app runs
  open (so nothing breaks before you configure it).
- Passwords are hashed with PBKDF2-SHA256 (Python stdlib — no extra dependency).
- The Streamlit server is a trusted backend, so it talks to Supabase with the
  service_role key (kept in st.secrets, never sent to the browser).
- New sign-ups land as role='user', status='pending'. The email in SUPERADMIN_EMAIL
  is auto-promoted to role='superadmin', status='approved' on registration so the
  owner can bootstrap and then approve everyone else.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets as _secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import streamlit as st

ROLE_RANK = {"user": 0, "admin": 1, "superadmin": 2}
COOKIE_NAME = "capra_auth"
COOKIE_DAYS = 30


# --------------------------------------------------------------------------
# Config (from st.secrets) — all access guarded so a missing file never crashes.
# --------------------------------------------------------------------------
def _secret(name: str) -> str:
    try:
        return (st.secrets.get(name) or "").strip()
    except Exception:
        return ""


def _cfg() -> tuple[str, str]:
    return _secret("SUPABASE_URL").rstrip("/"), _secret("SUPABASE_KEY")


def is_configured() -> bool:
    url, key = _cfg()
    return bool(url and key)


def _superadmin_email() -> str:
    return _secret("SUPERADMIN_EMAIL").lower()


def _cookie_secret() -> str:
    """Secret used to sign the remember-me cookie.

    Prefer a dedicated COOKIE_SECRET so cookie signing is decoupled from the
    database key (rotating the DB key then won't silently invalidate sessions,
    and a leak of one secret doesn't compromise the other). Falls back to
    SUPABASE_KEY when COOKIE_SECRET isn't set, so existing setups keep working.
    """
    return _secret("COOKIE_SECRET") or _cfg()[1]


# --------------------------------------------------------------------------
# Password hashing (PBKDF2-SHA256, stdlib)
# --------------------------------------------------------------------------
def hash_password(pw: str) -> str:
    salt = _secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000)
    return f"pbkdf2_sha256$200000${salt}${dk.hex()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _algo, iters, salt, h = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), int(iters))
        return hmac.compare_digest(dk.hex(), h)
    except Exception:
        return False


# --------------------------------------------------------------------------
# Supabase PostgREST helpers (urllib — no extra dependency)
# --------------------------------------------------------------------------
def _req(method: str, path: str, body=None, params=None):
    url, key = _cfg()
    endpoint = f"{url}/rest/v1/{path}"
    if params:
        endpoint += "?" + urllib.parse.urlencode(params)
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(endpoint, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=12) as r:
        txt = r.read().decode()
        return json.loads(txt) if txt else []


def get_user(email: str):
    try:
        rows = _req("GET", "users", params={"email": f"eq.{email.lower()}", "select": "*"})
        return rows[0] if rows else None
    except Exception:
        return None


def _explain_db_error(exc) -> str:
    """Map a request failure to a short cause the UI can turn into advice."""
    if isinstance(exc, urllib.error.HTTPError):
        code = getattr(exc, "code", 0)
        if code in (401, 403):
            return "auth"        # bad/missing key, or RLS blocking
        if code == 404:
            return "table"       # users table missing / wrong URL path
        if code == 409:
            return "duplicate"   # unique email constraint
        return "http"
    if isinstance(exc, urllib.error.URLError):
        return "unreachable"     # DNS/connection — usually a wrong SUPABASE_URL
    return "unknown"


def _error_detail(exc) -> str:
    """Best-effort human detail (status + PostgREST body) to debug setup errors."""
    try:
        if isinstance(exc, urllib.error.HTTPError):
            try:
                body = exc.read().decode(errors="replace")
            except Exception:
                body = ""
            return f"HTTP {exc.code} — {body[:400]}".strip()
        return f"{type(exc).__name__}: {exc}"[:400]
    except Exception:
        return type(exc).__name__


def create_user(email: str, pw: str, full_name: str):
    email = email.lower()
    is_super = bool(email) and email == _superadmin_email()
    row = {
        "email": email,
        "password_hash": hash_password(pw),
        "full_name": full_name or email.split("@")[0],
        "role": "superadmin" if is_super else "user",
        "status": "approved" if is_super else "pending",
    }
    try:
        res = _req("POST", "users", body=row)
        if res:
            return res[0]
        st.session_state["_auth_db_error"] = "empty"
        st.session_state["_auth_db_detail"] = (
            "Supabase accepted the request but returned no row — the insert may have "
            "failed silently, or row representation is turned off."
        )
        return None
    except Exception as exc:
        st.session_state["_auth_db_error"] = _explain_db_error(exc)
        st.session_state["_auth_db_detail"] = _error_detail(exc)
        return None


def list_users():
    try:
        return _req("GET", "users", params={"select": "*", "order": "created_at.desc"})
    except Exception:
        return []


def update_user(uid, fields: dict):
    try:
        return _req("PATCH", "users", body=fields, params={"id": f"eq.{uid}"})
    except Exception:
        return None


def delete_user(uid):
    try:
        return _req("DELETE", "users", params={"id": f"eq.{uid}"})
    except Exception:
        return None


# --------------------------------------------------------------------------
# "Remember me" — signed token in a first-party cookie (survives refresh).
# Fail-safe: if the cookie component isn't available, auth falls back to
# session-only (you'd re-login on refresh) — it never breaks login.
# --------------------------------------------------------------------------
_CM = None  # CookieManager for the current run (re-created each run)


def _init_cookies() -> None:
    """Instantiate ONE CookieManager per script run (components must be fresh)."""
    global _CM
    try:
        import extra_streamlit_components as stx
        _CM = stx.CookieManager(key="capra_cookies")
    except Exception:
        _CM = None


def _sign_token(email: str) -> str:
    key = _cookie_secret()
    exp = int(time.time()) + COOKIE_DAYS * 86400
    msg = f"{email}|{exp}"
    sig = hmac.new(key.encode(), msg.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{msg}|{sig}".encode()).decode()


def _verify_token(token: str) -> str | None:
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        email, exp, sig = raw.split("|")
        if int(exp) < time.time():
            return None
        key = _cookie_secret()
        good = hmac.new(key.encode(), f"{email}|{exp}".encode(), hashlib.sha256).hexdigest()
        return email if hmac.compare_digest(good, sig) else None
    except Exception:
        return None


def _set_cookie(email: str) -> None:
    if not _CM:
        return
    try:
        _CM.set(COOKIE_NAME, _sign_token(email),
                expires_at=datetime.now(timezone.utc) + timedelta(days=COOKIE_DAYS),
                key="capra_set_cookie")
    except Exception:
        pass


def _read_cookie_email() -> str | None:
    if not _CM:
        return None
    try:
        tok = _CM.get(COOKIE_NAME)
        return _verify_token(tok) if tok else None
    except Exception:
        return None


def _clear_cookie() -> None:
    if not _CM:
        return
    try:
        _CM.delete(COOKIE_NAME, key="capra_del_cookie")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Session helpers
# --------------------------------------------------------------------------
def current_user() -> dict | None:
    return st.session_state.get("_auth_user")


def logout() -> None:
    _clear_cookie()
    st.session_state.pop("_auth_user", None)


def is_admin(user: dict | None) -> bool:
    return ROLE_RANK.get((user or {}).get("role", "user"), 0) >= 1


# --------------------------------------------------------------------------
# Screens
# --------------------------------------------------------------------------
def _auth_screen() -> None:
    st.markdown("<h1 style='text-align:center;margin-top:8vh;'>CAPRA Finance</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#9ca3af;font-family:\"JetBrains Mono\",monospace;"
        "letter-spacing:.1em;text-transform:uppercase;font-size:.8rem;'>"
        "Sign in or create an account to continue</p>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        tab_login, tab_signup = st.tabs(["🔐 Log in", "✍️ Sign up"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email").strip().lower()
                pw = st.text_input("Password", type="password")
                if st.form_submit_button("Log in", use_container_width=True):
                    u = get_user(email)
                    if not u or not verify_password(pw, u.get("password_hash", "")):
                        st.error("Wrong email or password.")
                    else:
                        st.session_state["_auth_user"] = u
                        if u.get("status") == "approved":
                            _set_cookie(u["email"])  # remember me
                        st.rerun()

        with tab_signup:
            with st.form("signup_form", clear_on_submit=False):
                name = st.text_input("Full name")
                email2 = st.text_input("Email", key="su_email").strip().lower()
                pw1 = st.text_input("Password (min 8 chars)", type="password", key="su_pw1")
                pw2 = st.text_input("Confirm password", type="password", key="su_pw2")
                if st.form_submit_button("Create account", use_container_width=True):
                    if not email2 or "@" not in email2 or "." not in email2:
                        st.error("Enter a valid email address.")
                    elif len(pw1) < 8:
                        st.error("Password must be at least 8 characters.")
                    elif pw1 != pw2:
                        st.error("Passwords don't match.")
                    elif get_user(email2):
                        st.error("An account with that email already exists.")
                    else:
                        st.session_state.pop("_auth_db_error", None)
                        st.session_state.pop("_auth_db_detail", None)
                        u = create_user(email2, pw1, name)
                        if not u:
                            kind = st.session_state.get("_auth_db_error")
                            detail = st.session_state.get("_auth_db_detail", "")
                            if kind == "unreachable":
                                st.error("Can't reach Supabase. Double-check the **SUPABASE_URL** secret — "
                                         "it should look like `https://xxxx.supabase.co` (your real project ref, "
                                         "not the placeholder), with no trailing path.")
                            elif kind == "auth":
                                st.error("Supabase rejected the request. Check **SUPABASE_KEY** — it must be the "
                                         "**service_role** secret key (not the `anon` public key).")
                            elif kind == "table":
                                st.error("The `users` table wasn't found. Run the setup SQL in the Supabase "
                                         "SQL editor, then try again.")
                            elif kind == "duplicate":
                                st.error("An account with that email already exists.")
                            else:
                                st.error("Couldn't create the account (database error). Try again shortly.")
                            if detail:
                                st.caption(f"🔎 Technical detail (for setup): {detail}")
                        elif u.get("status") == "approved":
                            st.session_state["_auth_user"] = u
                            _set_cookie(u["email"])  # remember me (super-admin auto-approved)
                            st.success("Welcome — you're in!")
                            st.rerun()
                        else:
                            st.success("✅ Account created! An administrator needs to approve you "
                                       "before you can log in. You'll get access once they do.")
        st.caption("Educational tool · not investment advice")
    st.stop()


def _pending_screen(u: dict) -> None:
    st.markdown("<h2 style='text-align:center;margin-top:12vh;'>⏳ Awaiting approval</h2>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.info(f"Hi **{u.get('full_name') or u.get('email')}** — your account is **pending admin approval**. "
                "You'll get full access the moment an administrator approves it.")
        if st.button("Log out", use_container_width=True):
            logout()
            st.rerun()
    st.stop()


def require_auth() -> dict:
    """Gate the app. Returns the user dict (or a guest superadmin if auth is OFF).

    Halts the script (st.stop) to show the login or pending screen when needed.
    """
    if not is_configured():
        # Auth not set up yet → run open. Treated as superadmin locally so you can
        # still preview the (empty) admin panel and see setup instructions there.
        return {"role": "superadmin", "email": "local", "full_name": "Guest", "_auth_disabled": True}

    _init_cookies()  # one CookieManager per run

    u = current_user()
    if not u:
        # "Remember me": restore the session from a valid signed cookie.
        cookie_email = _read_cookie_email()
        if cookie_email:
            fresh = get_user(cookie_email)
            if fresh and fresh.get("status") == "approved":
                st.session_state["_auth_user"] = fresh
                u = fresh

    if not u:
        _auth_screen()  # halts

    if u.get("status") != "approved":
        fresh = get_user(u.get("email", ""))          # maybe approved since they logged in
        if fresh and fresh.get("status") == "approved":
            st.session_state["_auth_user"] = fresh
            return fresh
        _pending_screen(u)  # halts
    return u


# --------------------------------------------------------------------------
# Admin panel (super-admin: full control; admin: approve/reject)
# --------------------------------------------------------------------------
def render_admin_panel() -> None:
    me = current_user() or {"role": "superadmin", "_auth_disabled": True}
    my_role = me.get("role", "user")
    if not is_admin(me):
        st.error("Admins only.")
        return

    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
        '<h2 style="margin:0;">🛡️ Admin · User Management</h2>'
        f'<span style="color:#9ca3af;font-size:0.8rem;">You are <b>{my_role}</b></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not is_configured():
        st.warning(
            "Authentication isn't configured yet, so there are no accounts to manage. "
            "Add **SUPABASE_URL**, **SUPABASE_KEY** (service_role), and **SUPERADMIN_EMAIL** "
            "to your Streamlit secrets, create the `users` table, then register with your "
            "super-admin email — it's auto-approved and promoted."
        )
        return

    import pandas as pd
    users = list_users()
    if not users:
        st.info("No users yet. Share the app — new sign-ups will appear here for approval.")
        return

    approved = [u for u in users if u.get("status") == "approved"]
    pending = [u for u in users if u.get("status") == "pending"]
    k = st.columns(3)
    k[0].metric("Total users", len(users))
    k[1].metric("Approved", len(approved))
    k[2].metric("Pending", len(pending))

    # ---- Pending approvals ----
    if pending:
        st.markdown("##### ⏳ Pending approvals")
        for u in pending:
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.markdown(f"**{u.get('full_name') or '—'}** · `{u.get('email')}`")
            if c2.button("✅ Approve", key=f"appr_{u['id']}", use_container_width=True):
                update_user(u["id"], {"status": "approved"})
                st.rerun()
            if c3.button("❌ Reject", key=f"rej_{u['id']}", use_container_width=True):
                update_user(u["id"], {"status": "rejected"})
                st.rerun()
        st.divider()

    # ---- All users table ----
    st.markdown("##### All users")
    df = pd.DataFrame([{
        "Email": u.get("email"), "Name": u.get("full_name"),
        "Role": u.get("role"), "Status": u.get("status"),
        "Joined": (u.get("created_at") or "")[:10],
    } for u in users])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- Manage a single user (role/delete = super-admin only) ----
    st.markdown("##### Manage a user")
    by_email = {u["email"]: u for u in users}
    pick = st.selectbox("Select a user", [""] + list(by_email.keys()),
                        format_func=lambda e: e or "Pick a user…")
    if not pick:
        return
    target = by_email[pick]
    is_self = target.get("email") == me.get("email")
    is_super_admin = my_role == "superadmin"

    cols = st.columns(4)
    # Approve / reject (any admin)
    if target.get("status") != "approved":
        if cols[0].button("✅ Approve", key="mng_appr", use_container_width=True):
            update_user(target["id"], {"status": "approved"}); st.rerun()
    else:
        if cols[0].button("⏸ Set pending", key="mng_pend", use_container_width=True):
            update_user(target["id"], {"status": "pending"}); st.rerun()

    # Promote / demote (super-admin only, not on a fellow super-admin or self)
    if is_super_admin and target.get("role") != "superadmin":
        if target.get("role") == "admin":
            if cols[1].button("⬇ Make user", key="mng_demote", use_container_width=True):
                update_user(target["id"], {"role": "user"}); st.rerun()
        else:
            if cols[1].button("⬆ Make admin", key="mng_promote", use_container_width=True):
                update_user(target["id"], {"role": "admin"}); st.rerun()

    # Delete (super-admin only, never self or another super-admin)
    if is_super_admin and not is_self and target.get("role") != "superadmin":
        if cols[2].button("🗑 Delete", key="mng_delete", use_container_width=True):
            delete_user(target["id"]); st.rerun()

    if not is_super_admin:
        st.caption("Role changes and deletion are restricted to the super-admin.")
