"""Simple shared-password gate for CAPRA Finance.

No database, no email, no accounts — just one password to enter the app.

- Set APP_PASSWORD in Streamlit secrets to require it. With no APP_PASSWORD the
  app runs open (nothing breaks before you configure it).
- Optional COOKIE_SECRET signs a 30-day "remember me" cookie so a refresh doesn't
  re-prompt. If it's not set, the password itself is used as the signing key. If the
  cookie component isn't available it falls back to session-only (fail-safe — the
  gate still works, you'd just re-enter the password after a full refresh).
- The cookie is bound to a fingerprint of the current password, so changing
  APP_PASSWORD instantly logs everyone out.

Public API kept stable for app.py: require_auth(), is_configured(), is_admin(),
logout(), current_user(), render_admin_panel().
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import time
from datetime import datetime, timedelta, timezone

import streamlit as st

COOKIE_NAME = "capra_gate"
COOKIE_DAYS = 30
_CM = None  # CookieManager for the current run (re-created each run)

_MEMBER = {"role": "member", "email": "", "full_name": "Member", "_auth_disabled": False}


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def _secret(name: str) -> str:
    try:
        return (st.secrets.get(name) or "").strip()
    except Exception:
        return ""


def _password() -> str:
    return _secret("APP_PASSWORD")


def is_configured() -> bool:
    return bool(_password())


def _cookie_secret() -> str:
    return _secret("COOKIE_SECRET") or _password()


def _pw_fingerprint() -> str:
    return hashlib.sha256(_password().encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Remember-me cookie (signed; bound to the current password)
# --------------------------------------------------------------------------
def _init_cookies() -> None:
    global _CM
    try:
        import extra_streamlit_components as stx
        _CM = stx.CookieManager(key="capra_cookies")
    except Exception:
        _CM = None


def _sign() -> str:
    key = _cookie_secret()
    exp = int(time.time()) + COOKIE_DAYS * 86400
    fp = _pw_fingerprint()
    sig = hmac.new(key.encode(), f"gate|{exp}|{fp}".encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{exp}|{fp}|{sig}".encode()).decode()


def _valid(token: str) -> bool:
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        exp, fp, sig = raw.split("|")
        if int(exp) < time.time():
            return False
        if fp != _pw_fingerprint():           # password changed -> cookie dead
            return False
        key = _cookie_secret()
        good = hmac.new(key.encode(), f"gate|{exp}|{fp}".encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(good, sig)
    except Exception:
        return False


def _set_cookie() -> None:
    if not _CM:
        return
    try:
        _CM.set(COOKIE_NAME, _sign(),
                expires_at=datetime.now(timezone.utc) + timedelta(days=COOKIE_DAYS),
                key="capra_set_cookie")
    except Exception:
        pass


def _cookie_ok() -> bool:
    if not _CM:
        return False
    try:
        tok = _CM.get(COOKIE_NAME)
        return _valid(tok) if tok else False
    except Exception:
        return False


def _clear_cookie() -> None:
    if not _CM:
        return
    try:
        _CM.delete(COOKIE_NAME, key="capra_del_cookie")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def current_user() -> dict | None:
    return st.session_state.get("_auth_user")


def logout() -> None:
    _clear_cookie()
    st.session_state.pop("_auth_user", None)


def is_admin(user: dict | None) -> bool:
    # Shared-password mode has no per-user roles / admin panel.
    return False


def require_auth() -> dict:
    """Gate the app behind the shared password. Halts (st.stop) to show the
    password screen when needed. Returns a lightweight user dict."""
    if not is_configured():
        # No password set → run open (also lets you preview locally).
        return {"role": "superadmin", "email": "local", "full_name": "Guest", "_auth_disabled": True}

    _init_cookies()

    u = current_user()
    if u:
        return u
    if _cookie_ok():
        st.session_state["_auth_user"] = dict(_MEMBER)
        return st.session_state["_auth_user"]

    _password_screen()  # halts
    return dict(_MEMBER)  # unreachable (st.stop above)


def _password_screen() -> None:
    st.markdown("<h1 style='text-align:center;margin-top:10vh;'>CAPRA Finance</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#9ca3af;font-family:\"JetBrains Mono\",monospace;"
        "letter-spacing:.1em;text-transform:uppercase;font-size:.8rem;'>"
        "Enter the access password to continue</p>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        with st.form("gate_form", clear_on_submit=False):
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Enter", use_container_width=True):
                if pw and hmac.compare_digest(pw, _password()):
                    st.session_state["_auth_user"] = dict(_MEMBER)
                    _set_cookie()  # remember me
                    st.rerun()
                else:
                    st.error("Wrong password.")
        st.caption("Educational tool · not investment advice")
    st.stop()


def render_admin_panel() -> None:
    # Kept for API compatibility; not reachable while is_admin() is False.
    st.info("This app uses a single shared access password — there are no individual "
            "accounts to manage.")
