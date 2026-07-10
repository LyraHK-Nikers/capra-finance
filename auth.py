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
- Optional email confirmation: when RESEND_API_KEY + APP_URL are set, sign-up emails
  a signed confirmation link (via Resend); users must click it (email_verified=true)
  AND be approved before they can log in. The super-admin is always exempt. With those
  secrets absent, the email step is skipped entirely and nothing changes.
  Requires a `email_verified boolean not null default false` column on `users`.
- Forgot password: when email is configured, the login screen offers a reset flow that
  emails a single-use signed link (?reset=...) to set a new password. The link is bound
  to the current password hash, so it stops working once used or once the password
  changes. Admins can also issue a temporary password from the admin panel (works even
  without email configured), so account recovery is always possible.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets as _secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import streamlit as st

ROLE_RANK = {"user": 0, "admin": 1, "superadmin": 2}
COOKIE_NAME = "capra_auth"
COOKIE_DAYS = 30
VERIFY_DAYS = 3      # how long an email-confirmation link stays valid
RESET_MINUTES = 60  # how long a password-reset link stays valid


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
# Email config (Resend) — verification emails are sent ONLY when configured.
# --------------------------------------------------------------------------
def _app_url() -> str:
    """Public base URL of the app, used to build the confirmation link."""
    return _secret("APP_URL").rstrip("/")


def _mail_from() -> str:
    """The From address. Defaults to Resend's test sender until a domain is verified."""
    return _secret("MAIL_FROM") or "CAPRA Finance <onboarding@resend.dev>"


def email_configured() -> bool:
    """True only when we can actually send a working confirmation link.

    Requires a Resend API key AND a public APP_URL (the link target). When this is
    False the app behaves exactly as before — no email step — so partial setup
    never locks anyone out.
    """
    return bool(_secret("RESEND_API_KEY") and _app_url())


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


def _insert_user(row: dict):
    """POST one user row; return the created row, or None (recording error detail)."""
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


def create_user(email: str, pw: str, full_name: str):
    email = email.lower()
    is_super = bool(email) and email == _superadmin_email()
    row = {
        "email": email,
        "password_hash": hash_password(pw),
        "full_name": full_name or email.split("@")[0],
        "role": "superadmin" if is_super else "user",
        "status": "approved" if is_super else "pending",
        # Super-admin and (until email is configured) everyone are auto-verified.
        "email_verified": bool(is_super or not email_configured()),
    }
    u = _insert_user(row)
    # Graceful path if the table hasn't had the email_verified column added yet:
    # retry the insert without it so sign-up never breaks pre-migration.
    if u is None and "email_verified" in (st.session_state.get("_auth_db_detail") or ""):
        row.pop("email_verified", None)
        u = _insert_user(row)
    return u


def mark_email_verified(email: str) -> bool:
    """Flip email_verified=true for an address (used when a confirmation link is clicked).

    Returns True only if a row was actually updated (return=representation is set, so an
    empty result means no matching/affected row, e.g. unknown email or missing column)."""
    try:
        res = _req("PATCH", "users", body={"email_verified": True},
                   params={"email": f"eq.{email.lower()}"})
        return bool(res)
    except Exception:
        return False


def reset_password(email: str, new_pw: str) -> bool:
    """Set a new password hash for an address. Returns True if a row was updated."""
    try:
        res = _req("PATCH", "users", body={"password_hash": hash_password(new_pw)},
                   params={"email": f"eq.{email.lower()}"})
        return bool(res)
    except Exception:
        return False


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
    # Bind the cookie to a fingerprint of the current password hash so that a
    # password reset (self-service OR admin) transitively revokes every remember-me
    # session on all devices — a stolen cookie stops working the moment the password changes.
    key = _cookie_secret()
    exp = int(time.time()) + COOKIE_DAYS * 86400
    fp = _reset_fingerprint(email) or ""
    sig = hmac.new(key.encode(), f"{email}|{exp}|{fp}".encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{email}|{exp}|{fp}|{sig}".encode()).decode()


def _verify_token(token: str) -> str | None:
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        email, exp, fp, sig = raw.split("|")
        if int(exp) < time.time():
            return None
        key = _cookie_secret()
        good = hmac.new(key.encode(), f"{email}|{exp}|{fp}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(good, sig):
            return None
        cur = _reset_fingerprint(email)   # revoke if the password changed since issue
        if cur is None or fp != cur:
            return None
        return email
    except Exception:
        return None


def _sign_verify_token(email: str) -> str:
    """A short-lived, namespaced HMAC token for the email-confirmation link.

    Signed over a 'verify|...' prefix so it can never be replayed as a login cookie
    (and vice-versa), using the same secret as the cookie.
    """
    key = _cookie_secret()
    exp = int(time.time()) + VERIFY_DAYS * 86400
    sig = hmac.new(key.encode(), f"verify|{email}|{exp}".encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{email}|{exp}|{sig}".encode()).decode()


def _verify_verify_token(token: str) -> str | None:
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        email, exp, sig = raw.split("|")
        if int(exp) < time.time():
            return None
        key = _cookie_secret()
        good = hmac.new(key.encode(), f"verify|{email}|{exp}".encode(), hashlib.sha256).hexdigest()
        return email if hmac.compare_digest(good, sig) else None
    except Exception:
        return None


def _reset_fingerprint(email: str) -> str | None:
    """Short hash of the user's CURRENT password hash.

    Baked into reset tokens so a link becomes single-use: once the password changes
    (or was changed by any other means), the fingerprint no longer matches and old
    reset links stop working. Returns None if the account doesn't exist.
    """
    u = get_user(email)
    if not u:
        return None
    return hashlib.sha256((u.get("password_hash") or "").encode()).hexdigest()[:16]


def _sign_reset_token(email: str) -> str | None:
    fp = _reset_fingerprint(email)
    if fp is None:
        return None
    key = _cookie_secret()
    exp = int(time.time()) + RESET_MINUTES * 60
    sig = hmac.new(key.encode(), f"reset|{email}|{exp}|{fp}".encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{email}|{exp}|{sig}".encode()).decode()


def _verify_reset_token(token: str) -> str | None:
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        email, exp, sig = raw.split("|")
        if int(exp) < time.time():
            return None
        fp = _reset_fingerprint(email)        # current state → single-use enforcement
        if fp is None:
            return None
        key = _cookie_secret()
        good = hmac.new(key.encode(), f"reset|{email}|{exp}|{fp}".encode(), hashlib.sha256).hexdigest()
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
# Email verification (Resend HTTP API — no extra dependency)
# --------------------------------------------------------------------------
def _verification_email_html(link: str, full_name: str) -> str:
    name = full_name or "there"
    return (
        '<div style="font-family:Arial,Helvetica,sans-serif;background:#000;padding:32px;color:#f3f4f6;">'
        '<div style="max-width:480px;margin:0 auto;background:#0c0c0c;border:1px solid #222;border-radius:6px;padding:32px;">'
        '<h1 style="color:#ff6b35;margin:0 0 6px;font-size:24px;">CAPRA Finance</h1>'
        '<p style="color:#9ca3af;margin:0 0 24px;font-size:12px;letter-spacing:.1em;text-transform:uppercase;">Confirm your email</p>'
        f'<p>Hi {name},</p>'
        '<p>Thanks for signing up. Please confirm your email address to continue — '
        'after that an administrator will approve your account.</p>'
        f'<p style="text-align:center;margin:28px 0;"><a href="{link}" '
        'style="background:#ff6b35;color:#000;text-decoration:none;font-weight:bold;'
        'padding:12px 28px;border-radius:4px;display:inline-block;">Confirm my email</a></p>'
        '<p style="color:#9ca3af;font-size:12px;">Or paste this link into your browser:</p>'
        f'<p style="color:#9ca3af;font-size:12px;word-break:break-all;">{link}</p>'
        '<p style="color:#6b7280;font-size:11px;margin-top:24px;">This link expires in 3 days. '
        "If you didn't create this account, you can safely ignore this email.</p>"
        '</div></div>'
    )


def send_verification_email(email: str, full_name: str = "") -> bool:
    """Email a signed confirmation link via Resend. Returns True on success."""
    if not email_configured():
        return False
    token = _sign_verify_token(email.lower())
    link = f"{_app_url()}/?verify={urllib.parse.quote(token, safe='')}"
    payload = {
        "from": _mail_from(),
        "to": [email],
        "subject": "Confirm your email · CAPRA Finance",
        "html": _verification_email_html(link, full_name),
    }
    try:
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {_secret('RESEND_API_KEY')}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=12) as r:
            r.read()
        st.session_state.pop("_mail_error", None)
        return True
    except Exception as exc:
        st.session_state["_mail_error"] = _error_detail(exc)
        return False


def handle_verification_link() -> None:
    """If the URL carries ?verify=<token>, confirm that email then strip the param."""
    try:
        token = st.query_params.get("verify")
    except Exception:
        token = None
    if not token:
        return
    email = _verify_verify_token(token)
    if email:
        st.session_state["_verify_flash"] = ("ok" if mark_email_verified(email) else "dberr", email)
    else:
        st.session_state["_verify_flash"] = ("bad", None)
    try:
        del st.query_params["verify"]
    except Exception:
        try:
            st.query_params.clear()
        except Exception:
            pass


def _render_verify_flash() -> None:
    """Show the result of clicking a confirmation/reset link (once)."""
    flash = st.session_state.pop("_verify_flash", None)
    if not flash:
        return
    kind, em = flash
    if kind == "ok":
        st.success(f"✅ Email confirmed for **{em}**! You can log in once an administrator approves your account.")
    elif kind == "pwreset":
        st.success(f"✅ Password updated for **{em}**. Log in with your new password.")
    elif kind == "dberr":
        st.warning("We couldn't save your confirmation just now. Please click the link again, or contact the admin.")
    else:
        st.error("This confirmation link is invalid or has expired. Log in and request a fresh one.")


# --------------------------------------------------------------------------
# Password reset (signed, single-use link via Resend)
# --------------------------------------------------------------------------
def _reset_email_html(link: str) -> str:
    return (
        '<div style="font-family:Arial,Helvetica,sans-serif;background:#000;padding:32px;color:#f3f4f6;">'
        '<div style="max-width:480px;margin:0 auto;background:#0c0c0c;border:1px solid #222;border-radius:6px;padding:32px;">'
        '<h1 style="color:#ff6b35;margin:0 0 6px;font-size:24px;">CAPRA Finance</h1>'
        '<p style="color:#9ca3af;margin:0 0 24px;font-size:12px;letter-spacing:.1em;text-transform:uppercase;">Reset your password</p>'
        '<p>We received a request to reset your password. Click below to choose a new one:</p>'
        f'<p style="text-align:center;margin:28px 0;"><a href="{link}" '
        'style="background:#ff6b35;color:#000;text-decoration:none;font-weight:bold;'
        'padding:12px 28px;border-radius:4px;display:inline-block;">Reset my password</a></p>'
        '<p style="color:#9ca3af;font-size:12px;">Or paste this link into your browser:</p>'
        f'<p style="color:#9ca3af;font-size:12px;word-break:break-all;">{link}</p>'
        '<p style="color:#6b7280;font-size:11px;margin-top:24px;">This link expires in 1 hour and can be used once. '
        "If you didn't request this, you can safely ignore this email — your password won't change.</p>"
        '</div></div>'
    )


def _post_email_async(payload: dict) -> None:
    """Fire a Resend send on a daemon thread so the request latency doesn't depend
    on whether an account exists (defeats timing-based email enumeration)."""
    key = _secret("RESEND_API_KEY")

    def _send():
        try:
            req = urllib.request.Request(
                "https://api.resend.com/emails",
                data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=12) as r:
                r.read()
        except Exception:
            pass  # nothing to surface — the UI shows the same generic message regardless

    threading.Thread(target=_send, daemon=True).start()


def send_password_reset(email: str) -> bool:
    """Email a single-use reset link IF the account exists. Stays silent about whether
    an account exists — same message and (roughly) same latency either way, since the
    actual send happens off the request path."""
    email = (email or "").lower()
    if not email_configured():
        return False
    token = _sign_reset_token(email)   # None when the account doesn't exist
    if not token:
        return False
    link = f"{_app_url()}/?reset={urllib.parse.quote(token, safe='')}"
    _post_email_async({
        "from": _mail_from(),
        "to": [email],
        "subject": "Reset your password · CAPRA Finance",
        "html": _reset_email_html(link),
    })
    return True


def handle_reset_link() -> None:
    """If the URL carries ?reset=<token>, stash the verified email so the
    set-new-password screen can render, then strip the param."""
    try:
        token = st.query_params.get("reset")
    except Exception:
        token = None
    if not token:
        return
    st.session_state["_reset_email"] = _verify_reset_token(token)  # email, or None if invalid
    try:
        del st.query_params["reset"]
    except Exception:
        try:
            st.query_params.clear()
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
    st.session_state.pop("_verify_flash", None)  # don't show a stale confirmation banner later


def is_admin(user: dict | None) -> bool:
    return ROLE_RANK.get((user or {}).get("role", "user"), 0) >= 1


def _is_super(user: dict | None) -> bool:
    return (user or {}).get("role") == "superadmin"


def _email_verified_ok(user: dict | None) -> bool:
    """Email requirement satisfied?

    True if verification is off, the user is the super-admin (always exempt), or the
    user is verified. Also True when the row has NO email_verified key at all — that
    only happens before the column migration has run, and we fail OPEN there so a
    half-finished setup can never lock out existing approved users.
    """
    if (not email_configured()) or _is_super(user):
        return True
    u = user or {}
    if "email_verified" not in u:   # column not added yet (pre-migration) → don't block
        return True
    return bool(u.get("email_verified"))


def _fully_cleared(user: dict | None) -> bool:
    """A user who may enter the app: approved AND past the email gate."""
    return (user or {}).get("status") == "approved" and _email_verified_ok(user)


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
        _render_verify_flash()
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
                        if _fully_cleared(u):
                            _set_cookie(u["email"])  # remember me
                        st.rerun()

            # Forgot password — always visible; self-service when email is configured,
            # otherwise it points the user to the administrator.
            with st.expander("🔑 Forgot your password?"):
                if email_configured():
                    with st.form("forgot_form", clear_on_submit=True):
                        fp_email = st.text_input("Your account email", key="fp_email").strip().lower()
                        if st.form_submit_button("Send reset link", use_container_width=True):
                            send_password_reset(fp_email)
                            st.success("If an account exists for that email, we've sent a reset link. "
                                       "Check your inbox (and spam).")
                else:
                    st.info("Password reset by email isn't switched on yet. Please contact the "
                            "administrator at **contact@caprahk.com** — they can reset it for you "
                            "from the admin panel in seconds.")

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
                        elif email_configured():
                            sent = send_verification_email(u["email"], u.get("full_name", ""))
                            if sent:
                                st.success(f"✅ Account created! We've emailed a confirmation link to **{email2}**. "
                                           "Click it to verify your address — then an administrator will approve you. "
                                           "(Check your spam folder if it's not in your inbox.)")
                            else:
                                st.warning("✅ Account created, but we couldn't send the confirmation email right now. "
                                           "Try logging in to resend it, or contact the administrator.")
                                if st.session_state.get("_mail_error"):
                                    st.caption(f"🔎 Mail detail (for setup): {st.session_state['_mail_error']}")
                        else:
                            st.success("✅ Account created! An administrator needs to approve you "
                                       "before you can log in. You'll get access once they do.")
        st.caption("Educational tool · not investment advice")
    st.stop()


def _pending_screen(u: dict) -> None:
    st.markdown("<h2 style='text-align:center;margin-top:12vh;'>⏳ Awaiting approval</h2>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        _render_verify_flash()
        st.info(f"Hi **{u.get('full_name') or u.get('email')}** — your account is **pending admin approval**. "
                "You'll get full access the moment an administrator approves it.")
        if st.button("Log out", use_container_width=True):
            logout()
            st.rerun()
    st.stop()


def _unverified_screen(u: dict) -> None:
    st.markdown("<h2 style='text-align:center;margin-top:12vh;'>✉️ Confirm your email</h2>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        _render_verify_flash()
        st.info(f"We sent a confirmation link to **{u.get('email')}**. Open it to verify your address — "
                "then an administrator will approve you. Check your spam folder if it hasn't arrived.")
        c1, c2 = st.columns(2)
        if c1.button("📧 Resend email", use_container_width=True):
            if send_verification_email(u.get("email", ""), u.get("full_name", "")):
                st.success("Sent! Check your inbox (and spam).")
            else:
                st.error("Couldn't send right now. " + (st.session_state.get("_mail_error") or "Try again shortly."))
        if c2.button("Log out", use_container_width=True):
            logout()
            st.rerun()
        st.caption("Already clicked the link? Just refresh this page.")
    st.stop()


def _reset_password_screen() -> None:
    """Shown when the URL carried a ?reset token. Lets the user set a new password."""
    email = st.session_state.get("_reset_email")
    st.markdown("<h2 style='text-align:center;margin-top:12vh;'>🔑 Set a new password</h2>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if not email:
            st.error("This reset link is invalid or has expired (reset links work once and last 1 hour). "
                     "Request a fresh one from the login screen.")
            if st.button("← Back to login", use_container_width=True):
                st.session_state.pop("_reset_email", None)
                st.rerun()
            st.stop()

        st.info(f"Choose a new password for **{email}**.")
        with st.form("reset_form", clear_on_submit=False):
            p1 = st.text_input("New password (min 8 chars)", type="password", key="rs_pw1")
            p2 = st.text_input("Confirm new password", type="password", key="rs_pw2")
            if st.form_submit_button("Update password", use_container_width=True):
                if len(p1) < 8:
                    st.error("Password must be at least 8 characters.")
                elif p1 != p2:
                    st.error("Passwords don't match.")
                elif reset_password(email, p1):
                    st.session_state.pop("_reset_email", None)
                    logout()  # force a fresh login with the new password (also kills old sessions)
                    st.session_state["_verify_flash"] = ("pwreset", email)
                    st.rerun()
                else:
                    st.error("Couldn't update the password just now. Please request a new link and try again.")
        if st.button("Cancel", use_container_width=True):
            st.session_state.pop("_reset_email", None)
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

    _init_cookies()              # one CookieManager per run
    handle_verification_link()   # process ?verify=<token>
    handle_reset_link()          # process ?reset=<token>

    # A valid (or invalid) reset link takes priority — show the set-password screen.
    if "_reset_email" in st.session_state:
        _reset_password_screen()  # halts

    u = current_user()
    if not u:
        # "Remember me": restore the session from a valid signed cookie.
        cookie_email = _read_cookie_email()
        if cookie_email:
            fresh = get_user(cookie_email)
            if fresh and _fully_cleared(fresh):
                st.session_state["_auth_user"] = fresh
                u = fresh

    if not u:
        _auth_screen()  # halts

    # Email-confirmation gate (only when email sending is configured).
    if not _email_verified_ok(u):
        fresh = get_user(u.get("email", ""))          # maybe confirmed in another tab
        if fresh and _email_verified_ok(fresh):
            st.session_state["_auth_user"] = fresh
            u = fresh
        else:
            _unverified_screen(u)  # halts

    # Admin-approval gate.
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
            if email_configured():
                ver = "✅ email confirmed" if u.get("email_verified") else "✉️ not confirmed yet"
                c1.markdown(f"**{u.get('full_name') or '—'}** · `{u.get('email')}` — {ver}")
            else:
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
    _emailing = email_configured()
    df = pd.DataFrame([{
        "Email": u.get("email"), "Name": u.get("full_name"),
        "Role": u.get("role"), "Status": u.get("status"),
        **({"Confirmed": "✅" if u.get("email_verified") else "—"} if _emailing else {}),
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

    # Manually confirm email (any admin) — rescue path if mail delivery is broken.
    if email_configured() and not target.get("email_verified"):
        if cols[3].button("✉️ Mark confirmed", key="mng_verify", use_container_width=True):
            update_user(target["id"], {"email_verified": True}); st.rerun()

    # Reset password (any admin) — issues a temporary password to hand over.
    # Works even when email isn't configured, so password recovery is always possible.
    if st.button("🔑 Reset password (issue a temporary one)", key="mng_pwreset", use_container_width=True):
        temp = _secrets.token_urlsafe(9)
        if reset_password(target["email"], temp):
            st.session_state["_temp_pw_for"] = (target["email"], temp)
        else:
            st.session_state.pop("_temp_pw_for", None)
            st.error("Couldn't reset that password just now. Try again shortly.")
        st.rerun()
    tp = st.session_state.get("_temp_pw_for")
    if tp and tp[0] == target.get("email"):
        st.warning(f"Temporary password for **{tp[0]}**: `{tp[1]}`\n\n"
                   "Share it securely. They can log in with it, then use **Forgot your password?** "
                   "to set their own (once email is enabled). This is shown once.")
        st.session_state.pop("_temp_pw_for", None)

    if not is_super_admin:
        st.caption("Role changes and deletion are restricted to the super-admin.")
