"""BRAIN authentication, .env loading, session cache and account profile helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timezone
from functools import cache
from os import environ
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import cache_store
from check_evidence import normalize_checks, platform_check_names
import machine_lib as _ml
import research_paths
from .errors import BrainError


def _load_dotenv(path=None):
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Runs once per process. Concurrent callers wait for the complete file to
    load. Real environment variables always win over .env, so CI/shell
    configuration is never silently overridden.
    """
    pass
    if _ml._DOTENV_LOADED:
        return
    with _ml._DOTENV_LOCK:
        if _ml._DOTENV_LOADED:
            return
        env_path = Path(_ml.__file__).resolve().parent / ".env" if path is None else Path(path)
        if env_path.is_file():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                if key and key not in environ:
                    environ[key] = value
        _ml._DOTENV_LOADED = True


def configured_username():
    """Return the configured login name without requiring a password or session."""
    _ml._load_dotenv()
    return environ.get("BRAIN_USERNAME", "").strip()


def get_credentials():
    """Return (username, password) for the WorldQuant BRAIN API.

    Reads BRAIN_USERNAME / BRAIN_PASSWORD from the environment, after
    loading a .env file next to this module when present. Never hardcode
    credentials in the source tree.
    """
    username = _ml.configured_username()
    password = environ.get("BRAIN_PASSWORD", "").strip()
    if not username or not password:
        raise _ml.MissingCredentials(
            "Missing credentials: set BRAIN_USERNAME and BRAIN_PASSWORD in the "
            "environment, or copy .env.example to .env next to machine_lib.py "
            "and fill it in."
        )
    return username, password


def login():
    """Authenticate and return a ``requests.Session`` for the BRAIN API.

    Raises ``MissingCredentials`` when no username/password is configured and
    ``AuthError`` when BRAIN rejects them. Prints nothing: use
    ``authenticate()`` if you want to know which account you are using.
    """
    session, _ = _ml.authenticate()
    return session


def authenticate(*, deadline=None):
    """Return ``(session, info)`` for the configured BRAIN credentials.

    ``info`` is ``{"user_id", "permissions", "token_expiry"}`` -- deliberately
    not the ``/users/self`` payload, which carries account PII (email, phone).
    """
    username, password = _ml.get_credentials()
    from requests import Session
    from requests.exceptions import RequestException

    if deadline is not None and _ml.time.monotonic() >= deadline:
        raise _ml.AuthenticationDeadline("authentication wait budget expired before login")
    session = Session()
    session.auth = (username, password)
    try:
        response = session.post(_ml.BRAIN_API_URL + "/authentication", timeout=_ml._request_timeout(deadline))
    except RequestException as exc:
        message = "authentication request failed (%s)" % type(exc).__name__
        if _ml._transient_read_error(exc):
            if deadline is not None and _ml.time.monotonic() >= deadline:
                raise _ml.AuthenticationDeadline("authentication wait budget expired during login") from exc
            raise _ml.AuthenticationUnavailable(message) from exc
        raise _ml.ServiceError(message) from exc
    if response.status_code != 201:
        status = response.status_code
        if "captcha" in response.text[:300].lower():
            raise _ml.CaptchaRequired(
                "BRAIN is asking for a captcha before issuing a session (HTTP %s)." % status
            )
        if status in (400, 401):
            raise _ml.AuthError("authentication failed (HTTP %s)" % status)
        if status == 403:
            raise _ml.PermissionDenied("BRAIN denied access to authentication (HTTP 403)")
        if status == 429:
            raise _ml.RateLimited("BRAIN authentication is rate limited")
        if status == 408 or status >= 500:
            raise _ml.AuthenticationUnavailable("authentication request failed (HTTP %s)" % status)
        raise _ml.ServiceError("authentication request failed (HTTP %s)" % status)
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    token = payload.get("token")
    info = {
        "user_id": (payload.get("user") or {}).get("id"),
        "permissions": payload.get("permissions") or [],
        "token_expiry": (token or {}).get("expiry") if isinstance(token, dict)
        else payload.get("tokenExpiry"),
    }
    session._brain_account_id = info["user_id"]
    session._brain_permissions = info["permissions"]
    return session, info


def is_python_language(language):
    """Recognize the Python variant in platform settings."""
    return isinstance(language, str) and language.upper() == "PYTHON"


def normalize_python_source(source):
    """Normalize line endings while preserving Python indentation and whitespace."""
    return source.replace("\r\n", "\n").replace("\r", "\n")


def alpha_language(alpha):
    """Read the language marker from a platform alpha or ledger record."""
    for source in (alpha.get("settings"), alpha.get("regular"), alpha):
        language = source.get("language") if isinstance(source, dict) else None
        if isinstance(language, str) and language:
            return language.upper()
    return None


def regular_expression(alpha):
    """Return ``(code, description, operator_count)`` from an alpha payload.

    BRAIN returns ``regular`` as a plain string for some alphas and as
    ``{"code", "description", "operatorCount"}`` for others; callers should not
    have to branch on that.
    """
    regular = alpha.get("regular")
    if isinstance(regular, dict):
        code, description, count = (regular.get("code"), regular.get("description"),
                                    regular.get("operatorCount"))
    else:
        code, description, count = regular, None, None
    if isinstance(code, str):
        code = _ml.normalize_python_source(code) if _ml.is_python_language(_ml.alpha_language(alpha)) else code.strip()
    return code, description, count


def session_cache_path():
    """Where a reused BRAIN session cookie is kept.

    Lives under the cache root (outside the git working tree, on the persistent
    volume) rather than next to this module, and is written mode ``0600``: the
    file holds live session cookies, which are credentials.
    """
    return Path(_ml.research_paths.session_path())


def _session_payload(session, info):
    from requests.utils import dict_from_cookiejar

    auth = getattr(session, "auth", None)
    username = auth[0] if isinstance(auth, (tuple, list)) and auth else None
    return {
        "cookies": dict_from_cookiejar(session.cookies),
        "user_id": (info or {}).get("user_id"),
        "username": username.strip().casefold() if isinstance(username, str) else None,
        "permissions": (info or {}).get("permissions"),
        "token_expiry": (info or {}).get("token_expiry"),
    }


def save_session(session, info=None, path=None, *, meta=None):
    """Atomically store the authenticated user's cookies in a private envelope."""
    target = Path(path or _ml.session_cache_path())
    ttl = None
    expiry = (info or {}).get("token_expiry")
    if isinstance(expiry, (int, float)) and expiry > 0:
        ttl = float(expiry)
    entry = _ml.cache_store.build_entry(
        "brain-session", _ml._session_payload(session, info), ttl=ttl, source="login")
    _ml.cache_store.write_json_atomic(target, entry, mode=0o600)
    if meta is not None:
        meta["generation"] = entry["_cache"]["created_at"]
    return target


def _session_from_entry(payload, credentials):
    from requests import Session

    session = Session()
    session.auth = credentials
    session.cookies.update(payload["cookies"])
    session._brain_account_id = payload.get("user_id")
    session._brain_permissions = payload.get("permissions")
    return session


def session_status(cache_path=None):
    """Report cached session metadata and local expiry for auth and cache diagnostics."""
    path = Path(cache_path or _ml.session_cache_path())
    entry = _ml.cache_store.read_entry(path)
    payload = entry["payload"]
    if entry["kind"] != "brain-session" or not isinstance(payload, dict):
        payload = {}
    expires_in = entry.get("expires_in")
    status = {
        "path": str(path),
        "present": entry["present"],
        "fresh": entry["fresh"],
        "created_at": entry.get("created_at"),
        "expires_at": entry.get("expires_at"),
        "age_seconds": None if entry.get("age_seconds") is None
        else round(entry["age_seconds"], 1),
        "expires_in_seconds": None if expires_in is None else round(expires_in, 1),
        "user_id": payload.get("user_id"),
        "permissions": payload.get("permissions"),
        "has_cookies": bool(payload.get("cookies")),
        "source": entry.get("source"),
        "error": entry.get("error"),
    }
    return status


def invalidate_session(cache_path=None):
    """Drop the cached session (e.g. after BRAIN rejected it). Returns True if removed."""
    return _ml.cache_store.remove(Path(cache_path or _ml.session_cache_path()))


def logout_session(s=None, *, cache_path=None, remote=False):
    """Forget the local session cache and optionally revoke it via ``DELETE /authentication``."""
    from requests.exceptions import RequestException

    target = Path(cache_path or _ml.session_cache_path())
    remote_session = s
    if remote and remote_session is None:
        entry = _ml.cache_store.read_entry(target)
        payload = entry.get("payload")
        if (entry.get("present") and entry.get("kind") == "brain-session"
                and isinstance(payload, dict) and isinstance(payload.get("cookies"), dict)
                and payload["cookies"]):
            remote_session = _ml._session_from_entry(payload, None)
    remote_attempted = False
    remote_revoked = False
    status_code = None
    remote_error = None
    if remote and remote_session is not None:
        remote_attempted = True
        try:
            response = remote_session.delete(_ml.BRAIN_API_URL + "/authentication", timeout=10.0)
            status_code = response.status_code
            remote_revoked = status_code in (200, 202, 204, 401)
        except RequestException as exc:
            remote_error = type(exc).__name__
    removed = _ml.invalidate_session(target)
    result = {
        "logged_out": removed,
        "remote_attempted": remote_attempted,
        "remote_revoked": remote_revoked,
        "http_status": status_code,
        "session_path": str(target),
    }
    if remote_error:
        result["remote_error"] = remote_error
    return result


def fetch_user_profile(s, *, deadline=None):
    """Fetch the authenticated user's BRAIN profile (``GET /users/self``)."""
    limits = {} if deadline is None else {"deadline": deadline}
    payload = _ml.request_json(s, "GET", "/users/self", **limits)
    if not isinstance(payload, dict):
        raise _ml.ServiceError("user profile returned a non-object payload")
    return payload


def fetch_competitions(s, *, mine=True, competition_id=None, deadline=None):
    """Fetch enrolled competitions, all platform challenges, or one competition's details."""
    limits = {} if deadline is None else {"deadline": deadline}
    if competition_id is not None:
        if not isinstance(competition_id, str) or not competition_id.strip():
            raise ValueError("competition_id must be a nonempty string")
        payload = _ml.request_json(
            s, "GET", "/competitions/" + quote(competition_id.strip(), safe=""), **limits
        )
        if not isinstance(payload, dict):
            raise _ml.ServiceError("competition detail returned a non-object payload")
        return payload
    path = "/users/self/competitions" if mine else "/competitions"
    payload = _ml.request_json(s, "GET", path, **limits)
    return _ml._as_results(payload, "competitions")


def fetch_user_surface(s, path, *, deadline=None):
    """Read an account surface while distinguishing ``available``, ``level_gated`` and ``absent``."""
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("path must start with '/'")
    limits = {} if deadline is None else {"deadline": deadline}
    try:
        payload = _ml.request_json(s, "GET", path, **limits)
    except _ml.PermissionDenied:
        return {"path": path, "state": "level_gated", "http_status": 403, "data": None}
    except _ml.NotFound:
        return {"path": path, "state": "absent", "http_status": 404, "data": None}
    return {"path": path, "state": "available", "http_status": 200, "data": payload}


def fetch_consultant_summary(s, *, deadline=None):
    """Inspect level-gated consultant, pyramid and activity surfaces without failing on HTTP 403/404."""
    return {
        "consultant": _ml.fetch_user_surface(s, "/users/self/consultant", deadline=deadline),
        "pyramids": _ml.fetch_user_surface(s, "/users/self/pyramids", deadline=deadline),
        "activities": _ml.fetch_user_surface(s, "/users/self/activities", deadline=deadline),
    }


def login_lock_timeout():
    """Seconds a caller waits for another process to finish authentication."""
    try:
        timeout = float(environ.get("BRAIN_LOGIN_LOCK_TIMEOUT", "5") or 5)
    except ValueError as exc:
        raise _ml.ServiceError("BRAIN_LOGIN_LOCK_TIMEOUT must be finite and non-negative") from exc
    if not math.isfinite(timeout) or timeout < 0:
        raise _ml.ServiceError("BRAIN_LOGIN_LOCK_TIMEOUT must be finite and non-negative")
    return timeout


def isolated_session(source):
    """Copy authenticated state into a session with independent cookies and adapters."""
    from copy import deepcopy
    import requests

    session = requests.Session()
    for name in ("auth", "headers", "cookies", "params", "proxies", "hooks", "verify", "cert",
                 "trust_env", "max_redirects", "stream"):
        setattr(session, name, deepcopy(getattr(source, name)))
    return session


def load_session(cache_path=None, use_cache=True, refresh=False, meta=None, *,
                 deadline=None, rejected_generation=None):
    """Return an authenticated session, reusing a cached cookie when still valid.

    BRAIN issues sessions that last hours but challenges repeated logins with a
    captcha, so re-authenticating on every CLI call is both slow and fragile.
    A saved cookie is reusable only for the configured account, until it
    expires. ``refresh`` forces a fresh login.

    The login itself is single-flight: concurrent callers take a file lock, and
    the ones that lose the race re-check the cache and reuse the winner's
    session instead of adding a second login. A forced refresh may share a
    session written by another caller after the refresh started. If the caller
    knows its rejected cookie's generation, a refresh may reuse a session
    published since that cookie was read. A lock timeout with no reusable
    session raises ``SessionBusy``. Pass ``meta`` (a dict) to receive ``cached``,
    ``reason`` and the session's ``generation`` when cache-backed.
    """
    meta = meta if meta is not None else {}
    meta.pop("generation", None)
    if deadline is not None and _ml.time.monotonic() >= deadline:
        raise _ml.AuthenticationDeadline("authentication wait budget expired before login")
    path = Path(cache_path or _ml.session_cache_path())
    reuse_cache = use_cache and not refresh
    credentials = _ml.get_credentials() if reuse_cache else None
    previous_creation = rejected_generation
    if not reuse_cache and previous_creation is None:
        previous_creation = _ml.cache_store.read_entry(path).get("created_at")

    def _reuse(*, locked=False):
        entry = _ml.cache_store.read_entry(path)
        if not reuse_cache and entry.get("created_at") == previous_creation:
            return None
        if entry["present"] and entry["fresh"]:
            payload = entry["payload"]
            if (entry["kind"] != "brain-session" or not isinstance(payload, dict)
                    or not isinstance(payload.get("cookies"), dict) or not payload["cookies"]):
                meta["cache_state"] = "invalid"
                return None
            account_credentials = credentials if credentials is not None else _ml.get_credentials()
            if payload.get("username") != account_credentials[0].casefold():
                meta["cache_state"] = "account-mismatch"
                return None
            meta.update({"cached": True, "reason": "reused-cache",
                         "cache_state": "fresh",
                         "generation": entry["created_at"],
                         "expires_in": entry.get("expires_in")})
            return _ml._session_from_entry(payload, account_credentials)
        if entry["present"]:
            meta.setdefault("cache_state", "expired")
        elif entry.get("error") not in (None, "missing"):
            # Keep corrupt samples under the login lock for later diagnosis.
            meta["cache_state"] = "corrupt"
            if locked and reuse_cache:
                quarantined = _ml.cache_store.quarantine(path)
                if quarantined:
                    meta["quarantined_to"] = quarantined
                    _ml.note("session cache was unusable (%s); moved to %s"
                         % (entry["error"], quarantined))
        else:
            meta.setdefault("cache_state", "empty")
        return None

    def _login_and_save(reason):
        if deadline is None:
            session, info = _ml.authenticate()
        else:
            session, info = _ml.authenticate(deadline=deadline)
        try:
            _ml.save_session(session, info, path, meta=meta)
        except OSError:
            pass  # a read-only volume must not break a working session
        meta.update({"cached": False, "reason": reason, "user_id": (info or {}).get("user_id")})
        return session

    if reuse_cache or rejected_generation is not None:
        reused = _reuse()
        if reused is not None:
            return reused

    timeout = _ml.login_lock_timeout()
    if deadline is not None:
        remaining = deadline - _ml.time.monotonic()
        if remaining <= 0:
            raise _ml.AuthenticationDeadline("authentication wait budget expired before login")
        timeout = min(timeout, remaining)
    try:
        with _ml.cache_store.file_lock(_ml.research_paths.lock_path("brain-session"),
                                   timeout=timeout):
            reused = _reuse(locked=True)  # another caller may have refreshed the cache
            if reused is not None:
                meta["reason"] = "waited-for-other-login"
                return reused
            return _login_and_save("logged-in")
    except _ml.cache_store.LockTimeout as exc:
        # The holder may have published a reusable cookie just before timeout.
        reused = _reuse()
        if reused is not None:
            meta["reason"] = "cache-after-lock-timeout"
            return reused
        if deadline is not None and _ml.time.monotonic() >= deadline:
            raise _ml.AuthenticationDeadline("authentication wait budget expired waiting for login") from exc
        meta.update(cached=False, reason="session-busy")
        raise _ml.SessionBusy("another process is signing in to BRAIN") from exc
