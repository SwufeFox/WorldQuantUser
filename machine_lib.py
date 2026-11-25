"""BRAIN account sessions, catalog metadata, simulations and Alpha results.

Use load_session to reuse a private, account-bound session. Catalog datasets,
fields and categories are cached per account and query scope.

request_json handles ordinary API requests. Simulation progress and Alpha
recordsets have endpoint-specific readiness protocols. simulate_alphas uses
durable reservations to continue accepted work after interruption.

Connection setup imports the HTTP client on demand. Command help and offline
previews can run without it. DataFrame helpers import optional pandas.
"""

import hashlib
import json
import math
import os
import random
import re
import threading
import time
from functools import cache
from os import environ
from pathlib import Path
from time import sleep
from urllib.parse import quote, urljoin, urlsplit

import sys
from datetime import datetime, timezone

import cache_store
from check_evidence import normalize_checks, platform_check_names
import research_paths

# ---------------------------------------------------------------------------
# Diagnostics
#
# stdout is reserved for the single JSON document the CLI prints, so anything
# narrative goes to stderr. The CLI installs its own handler to emit structured
# events; a plain library user gets readable lines.
# ---------------------------------------------------------------------------
_diagnostic_handler = None


def set_diagnostic_handler(handler):
    """Install a diagnostic handler and return the previous one for restoration."""
    global _diagnostic_handler
    previous = _diagnostic_handler
    _diagnostic_handler = handler
    return previous


def note(message):
    """Emit one diagnostic line. Never writes to stdout."""
    handler = _diagnostic_handler
    if handler is not None:
        handler(message)
    else:
        print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Errors
#
# Everything this module raises on purpose derives from BrainError and carries
# a stable ``code``. Callers map ``code`` -> process exit code, so failures stay
# machine-readable instead of leaking a traceback shape.
# ---------------------------------------------------------------------------
class BrainError(RuntimeError):
    code = "BRAIN_ERROR"
    #: Optional actionable next step, surfaced verbatim by the CLI.
    hint = None


class MissingCredentials(BrainError):
    code = "AUTH_MISSING"


class AuthError(BrainError):
    code = "AUTH_FAILED"


class CaptchaRequired(AuthError):
    """BRAIN wants a human captcha before it will hand out a session."""

    code = "AUTH_CAPTCHA"
    hint = ("too many logins in a short window; wait a few minutes, then retry. "
            "Reusing the cached session avoids this.")


class NotFound(BrainError):
    code = "NOT_FOUND"


class RateLimited(BrainError):
    code = "RATE_LIMITED"

    def __init__(self, *args, retry_after=None):
        super().__init__(*args)
        self.retry_after = retry_after


class SimulationError(BrainError):
    code = "SIMULATION_FAILED"


SUBMIT_UNLOCK_ENV = "ALLOW_AGENT_SUBMIT"


class SubmissionDisabled(BrainError):
    """Raised when a real submission is attempted while the lock is closed."""

    code = "SUBMIT_DISABLED"
    hint = ("this is a policy lock, not a failure: set ALLOW_AGENT_SUBMIT=true in the "
            "environment or .env to allow real submissions on this account")


def agent_submit_allowed():
    """Whether an agent may take the irreversible step of submitting an alpha.

    Only an explicit ``ALLOW_AGENT_SUBMIT=true`` opens the lock. Surrounding whitespace is
    forgiven - a stray space from a copy-paste says nothing about intent - but every other
    spelling (``1``, ``yes``, ``TRUE``, ``on``) leaves it closed: there the intent is
    ambiguous, and ambiguity resolves to "no".
    """
    _load_dotenv()
    return (environ.get(SUBMIT_UNLOCK_ENV, "") or "").strip() == "true"


class ServiceError(BrainError):
    """A transport or unexpected HTTP failure while talking to BRAIN."""

    code = "SERVICE_ERROR"


class ReadInterrupted(ServiceError):
    """A read could not finish after transient transport or gateway errors."""


class AuthenticationUnavailable(ServiceError):
    """A transient transport or server failure interrupted authentication."""


class AuthenticationDeadline(ServiceError):
    """The caller's budget for starting or waiting for authentication expired."""


class SessionBusy(ServiceError):
    """Another process is acquiring the shared account session."""

    code = "SESSION_BUSY"
    hint = "retry shortly to reuse the session created by the active login"


class PermissionDenied(ServiceError):
    """BRAIN denied access to an endpoint with HTTP 403."""

    code = "ACCESS_DENIED"


# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------
#: Base payload for ``POST /simulations``. Callers override individual keys;
#: see ``resolve_settings`` for merge + validation rules.
DEFAULT_SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 6,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
    "pasteurization": "ON",
    "testPeriod": "P0Y",
    "unitHandling": "VERIFY",
    "nanHandling": "ON",
    "language": "FASTEXPR",
    "visualization": False,
}

#: Regions the API accepts. Which ones actually return data depends on the
#: account's permissions (a TUTORIAL account only sees USA).
REGIONS = [
    "USA", "GLB", "EUR", "ASI", "CHN", "HKG", "JPN", "KOR", "TWN", "IND",
    "AMR", "EMEA", "APAC",
]

#: Known neutralization values accepted by the API. **A hint list, not a local gate.**
#: The platform owns this enum; well-formed uppercase tokens outside this list are still
#: sent so new platform options work without client code changes, while malformed tokens
#: fail locally as input typos.
NEUTRALIZATIONS = ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY",
                   "STATISTICAL", "FAST"]

#: Shape of a settings enum token: upper case, digits, underscores.
_SETTING_TOKEN = re.compile(r"^[A-Z][A-Z0-9_]*$")

#: Non-exhaustive list of common universes, used for input validation hints.
UNIVERSES = [
    "TOP3000", "TOP1000", "TOP600", "TOP500", "TOP200", "TOP100",
    "TOPSP500", "TOPSP1000", "MINVOL1M", "ILLIQUID_MINVOL1M",
]


_DOTENV_LOADED = False
_DOTENV_LOCK = threading.Lock()


def _load_dotenv(path=None):
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Runs once per process. Concurrent callers wait for the complete file to
    load. Real environment variables always win over .env, so CI/shell
    configuration is never silently overridden.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    with _DOTENV_LOCK:
        if _DOTENV_LOADED:
            return
        env_path = Path(__file__).resolve().parent / ".env" if path is None else Path(path)
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
        _DOTENV_LOADED = True


def configured_username():
    """Return the configured login name without requiring a password or session."""
    _load_dotenv()
    return environ.get("BRAIN_USERNAME", "").strip()


def get_credentials():
    """Return (username, password) for the WorldQuant BRAIN API.

    Reads BRAIN_USERNAME / BRAIN_PASSWORD from the environment, after
    loading a .env file next to this module when present. Never hardcode
    credentials in the source tree.
    """
    username = configured_username()
    password = environ.get("BRAIN_PASSWORD", "").strip()
    if not username or not password:
        raise MissingCredentials(
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
    session, _ = authenticate()
    return session


def authenticate(*, deadline=None):
    """Return ``(session, info)`` for the configured BRAIN credentials.

    ``info`` is ``{"user_id", "permissions", "token_expiry"}`` -- deliberately
    not the ``/users/self`` payload, which carries account PII (email, phone).
    """
    username, password = get_credentials()
    from requests import Session
    from requests.exceptions import RequestException

    if deadline is not None and time.monotonic() >= deadline:
        raise AuthenticationDeadline("authentication wait budget expired before login")
    session = Session()
    session.auth = (username, password)
    try:
        response = session.post(BRAIN_API_URL + "/authentication", timeout=_request_timeout(deadline))
    except RequestException as exc:
        message = "authentication request failed (%s)" % type(exc).__name__
        if _transient_read_error(exc):
            if deadline is not None and time.monotonic() >= deadline:
                raise AuthenticationDeadline("authentication wait budget expired during login") from exc
            raise AuthenticationUnavailable(message) from exc
        raise ServiceError(message) from exc
    if response.status_code != 201:
        status = response.status_code
        if "captcha" in response.text[:300].lower():
            raise CaptchaRequired(
                "BRAIN is asking for a captcha before issuing a session (HTTP %s)." % status
            )
        if status in (400, 401):
            raise AuthError("authentication failed (HTTP %s)" % status)
        if status == 403:
            raise PermissionDenied("BRAIN denied access to authentication (HTTP 403)")
        if status == 429:
            raise RateLimited("BRAIN authentication is rate limited")
        if status == 408 or status >= 500:
            raise AuthenticationUnavailable("authentication request failed (HTTP %s)" % status)
        raise ServiceError("authentication request failed (HTTP %s)" % status)
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
        code = normalize_python_source(code) if is_python_language(alpha_language(alpha)) else code.strip()
    return code, description, count


def session_cache_path():
    """Where a reused BRAIN session cookie is kept.

    Lives under the cache root (outside the git working tree, on the persistent
    volume) rather than next to this module, and is written mode ``0600``: the
    file holds live session cookies, which are credentials.
    """
    return Path(research_paths.session_path())


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
    target = Path(path or session_cache_path())
    ttl = None
    expiry = (info or {}).get("token_expiry")
    if isinstance(expiry, (int, float)) and expiry > 0:
        ttl = float(expiry)
    entry = cache_store.build_entry(
        "brain-session", _session_payload(session, info), ttl=ttl, source="login")
    cache_store.write_json_atomic(target, entry, mode=0o600)
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
    path = Path(cache_path or session_cache_path())
    entry = cache_store.read_entry(path)
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
    return cache_store.remove(Path(cache_path or session_cache_path()))


def login_lock_timeout():
    """Seconds a caller waits for another process to finish authentication."""
    try:
        timeout = float(environ.get("BRAIN_LOGIN_LOCK_TIMEOUT", "5") or 5)
    except ValueError as exc:
        raise ServiceError("BRAIN_LOGIN_LOCK_TIMEOUT must be finite and non-negative") from exc
    if not math.isfinite(timeout) or timeout < 0:
        raise ServiceError("BRAIN_LOGIN_LOCK_TIMEOUT must be finite and non-negative")
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
    if deadline is not None and time.monotonic() >= deadline:
        raise AuthenticationDeadline("authentication wait budget expired before login")
    path = Path(cache_path or session_cache_path())
    reuse_cache = use_cache and not refresh
    credentials = get_credentials() if reuse_cache else None
    previous_creation = rejected_generation
    if not reuse_cache and previous_creation is None:
        previous_creation = cache_store.read_entry(path).get("created_at")

    def _reuse(*, locked=False):
        entry = cache_store.read_entry(path)
        if not reuse_cache and entry.get("created_at") == previous_creation:
            return None
        if entry["present"] and entry["fresh"]:
            payload = entry["payload"]
            if (entry["kind"] != "brain-session" or not isinstance(payload, dict)
                    or not isinstance(payload.get("cookies"), dict) or not payload["cookies"]):
                meta["cache_state"] = "invalid"
                return None
            account_credentials = credentials if credentials is not None else get_credentials()
            if payload.get("username") != account_credentials[0].casefold():
                meta["cache_state"] = "account-mismatch"
                return None
            meta.update({"cached": True, "reason": "reused-cache",
                         "cache_state": "fresh",
                         "generation": entry["created_at"],
                         "expires_in": entry.get("expires_in")})
            return _session_from_entry(payload, account_credentials)
        if entry["present"]:
            meta.setdefault("cache_state", "expired")
        elif entry.get("error") not in (None, "missing"):
            # Keep corrupt samples under the login lock for later diagnosis.
            meta["cache_state"] = "corrupt"
            if locked and reuse_cache:
                quarantined = cache_store.quarantine(path)
                if quarantined:
                    meta["quarantined_to"] = quarantined
                    note("session cache was unusable (%s); moved to %s"
                         % (entry["error"], quarantined))
        else:
            meta.setdefault("cache_state", "empty")
        return None

    def _login_and_save(reason):
        if deadline is None:
            session, info = authenticate()
        else:
            session, info = authenticate(deadline=deadline)
        try:
            save_session(session, info, path, meta=meta)
        except OSError:
            pass  # a read-only volume must not break a working session
        meta.update({"cached": False, "reason": reason, "user_id": (info or {}).get("user_id")})
        return session

    if reuse_cache or rejected_generation is not None:
        reused = _reuse()
        if reused is not None:
            return reused

    timeout = login_lock_timeout()
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AuthenticationDeadline("authentication wait budget expired before login")
        timeout = min(timeout, remaining)
    try:
        with cache_store.file_lock(research_paths.lock_path("brain-session"),
                                   timeout=timeout):
            reused = _reuse(locked=True)  # another caller may have refreshed the cache
            if reused is not None:
                meta["reason"] = "waited-for-other-login"
                return reused
            return _login_and_save("logged-in")
    except cache_store.LockTimeout as exc:
        # The holder may have published a reusable cookie just before timeout.
        reused = _reuse()
        if reused is not None:
            meta["reason"] = "cache-after-lock-timeout"
            return reused
        if deadline is not None and time.monotonic() >= deadline:
            raise AuthenticationDeadline("authentication wait budget expired waiting for login") from exc
        meta.update(cached=False, reason="session-busy")
        raise SessionBusy("another process is signing in to BRAIN") from exc



# ---------------------------------------------------------------------------
# Metadata cache
#
# Catalog responses are quota-free, paginated across many requests, and change on
# the order of days rather than seconds. Caching them with a TTL avoids redundant
# pagination across preflight and catalog queries, while serving a clearly marked
# stale snapshot if a transient network error interrupts a refresh.
# ---------------------------------------------------------------------------
DEFAULT_METADATA_TTL = float(environ.get("BRAIN_METADATA_TTL", 24 * 3600) or 0)

_metadata_policy = {"enabled": True, "max_age": DEFAULT_METADATA_TTL, "refresh": False}


def configure_metadata_cache(max_age=None, refresh=None, enabled=None):
    """Set the process-wide catalog cache policy (the CLI maps its flags here)."""
    if max_age is not None:
        _metadata_policy["max_age"] = float(max_age)
    if refresh is not None:
        _metadata_policy["refresh"] = bool(refresh)
    if enabled is not None:
        _metadata_policy["enabled"] = bool(enabled)
    return dict(_metadata_policy)


def metadata_cache_policy():
    """Current catalog cache policy, for `cache info` and `schema`."""
    return dict(_metadata_policy)


def metadata_cache_path(kind, params):
    """Deterministic path for one cached catalog response.

    The key is a hash of the query parameters, so different regions/universes/
    datasets never collide, while a repeat of the same query lands on the same
    file and hits.
    """
    blob = json.dumps({"kind": kind, "params": params}, sort_keys=True, default=str)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", kind)[:40] or "meta"
    return Path(research_paths.metadata_root()) / label / (digest + ".json")


def cached_metadata(kind, params, fetch, max_age=None, refresh=None, report=None, *,
                    use_cache=True, account=None, derive_from=None):
    """Read catalog data and return it with cache evidence.

    A hit must satisfy both the stored expiry and the requested maximum age.
    Read the source on a miss, a forced refresh, or when caching is disabled.
    On BrainError, reuse an available snapshot from the same cache key and mark
    it stale and degraded. A derived query may use a separate, fresh full
    snapshot before attempting the network. Pass a report dict for cache evidence.
    """
    policy = _metadata_policy
    ttl = policy["max_age"] if max_age is None else float(max_age)
    do_refresh = policy["refresh"] if refresh is None else bool(refresh)

    def cache_path(query):
        path = metadata_cache_path(kind, query)
        return path.with_name(f"{account}-{path.name}") if account is not None else path

    path = cache_path(params) if use_cache else None
    status = {
        "kind": kind,
        "path": None if path is None else str(path),
        "hit": False,
        "stale": False,
        "degraded": False,
        "disabled": False,
        "age_seconds": None,
        "expires_in_seconds": None,
        "refreshed_at": None,
        "error": None,
        "complete": not bool(params.get("limit")),
        "account_bound": account is not None,
    }

    def done(data):
        if isinstance(report, dict):
            report.clear()
            report.update(status)
        return {"data": data, "cache": status}

    if path is None or not policy["enabled"] or ttl <= 0:
        status["disabled"] = True
        data = fetch()
        status["refreshed_at"] = datetime.now(timezone.utc).isoformat()
        return done(data)

    entry = cache_store.read_entry(path)
    usable = entry["present"] and entry["kind"] == kind

    def cached_result(source, data, error=None):
        remaining = source["expires_in"]
        status.update(hit=True, stale=error is not None, degraded=error is not None,
                      age_seconds=round(source["age_seconds"], 1), refreshed_at=source["created_at"],
                      expires_in_seconds=None if remaining is None else round(remaining, 1),
                      error=error)
        return done(data)

    if usable and not do_refresh and entry["fresh"] and entry["age_seconds"] <= ttl:
        return cached_result(entry, entry["payload"])
    if not do_refresh and derive_from is not None:
        source_params, derive = derive_from
        source_path = cache_path(source_params)
        source = cache_store.read_entry(source_path)
        if (source["present"] and source["kind"] == kind and source["fresh"]
                and source["age_seconds"] <= ttl):
            derived = derive(source["payload"])
            if derived is not None:
                status.update(path=str(source_path), derived_from_complete=True)
                return cached_result(source, derived)
    if entry.get("error") not in (None, "missing"):
        cache_store.quarantine(path)

    try:
        data = fetch()
    except BrainError as exc:
        if usable:
            note("metadata cache: serving stale %s snapshot (%s)" % (kind, exc.code))
            return cached_result(entry, entry["payload"], exc.code)
        raise
    entry = cache_store.build_entry(kind, data, ttl=ttl, source="brain-api")
    try:
        cache_store.write_json_atomic(path, entry)
    except OSError as exc:
        note("metadata cache: could not write %s (%s)" % (path, exc))
    status.update(refreshed_at=entry["_cache"]["created_at"], age_seconds=0.0,
                  expires_in_seconds=ttl)
    return done(data)


class CatalogAccess:
    """Pair the catalog account identity with a runner that opens the session on demand."""

    def __init__(self, username, run):
        self.username = username
        self.run = run


def _catalog_account(username):
    name = username.strip().casefold() if isinstance(username, str) else ""
    return hashlib.sha256(name.encode()).hexdigest() if name else None


def catalog_field_cache_paths():
    """Yield (path, account_bound) for current-account and legacy field snapshots."""
    account = _catalog_account(configured_username())
    directory = Path(research_paths.metadata_root()) / "datafields"
    for path in sorted(directory.glob("*.json")):
        namespace, separator, _ = path.stem.partition("-")
        if not separator or namespace == account:
            yield path, bool(separator)


def _cached_account_catalog(session, kind, params, fetch, refresh, report, *, derive_from=None):
    if isinstance(session, CatalogAccess):
        username, run = session.username, session.run
    else:
        auth = getattr(session, "auth", None)
        username = auth[0] if isinstance(auth, (tuple, list)) and auth else None

        def run(work):
            return work(session)

    account = _catalog_account(username)
    return cached_metadata(kind, params, lambda: run(fetch), refresh=refresh, report=report,
                           use_cache=account is not None, account=account,
                           derive_from=derive_from)["data"]


def get_datasets(s, instrument_type="EQUITY", region="USA", delay=1,
                 universe="TOP3000", limit=None, refresh=None, report=None, theme=None, *, deadline=None):
    """List datasets available for a region/universe/delay as ``list[dict]``.

    Quota-free and cached (see :func:`cached_metadata`). An empty result concerns
    this exact combination; it does not establish access for other universes or
    delays, or distinguish an unsupported combination from an empty catalog.
    """
    if theme is not None and not isinstance(theme, bool):
        raise ValueError("theme must be true, false or unspecified")
    params = {"instrumentType": instrument_type, "region": region,
              "delay": delay, "universe": universe, "limit": limit}
    if theme is not None:
        params["theme"] = theme
    request_limits = {} if deadline is None else {"deadline": deadline}
    return _cached_account_catalog(
        s, "datasets", params,
        lambda session: _get_datasets(session, instrument_type, region, delay, universe, limit, theme,
                                      **request_limits),
        refresh, report,
    )


def _get_datasets(s, instrument_type="EQUITY", region="USA", delay=1,
                  universe="TOP3000", limit=None, theme=None, *, deadline=None):
    """List datasets available for a region/universe/delay as ``list[dict]``.

    Quota-free. An empty result concerns only this exact combination.
    """
    results = []
    params = {"instrumentType": instrument_type, "region": region,
              "delay": delay, "universe": universe}
    if theme is not None:
        params["theme"] = "true" if theme else "false"
    request_limits = {} if deadline is None else {"deadline": deadline}
    for batch in _catalog_pages(s, "/data-sets", params, keys=("data-sets",), **request_limits):
        if any(not isinstance(row.get("id"), str) or not row["id"] for row in batch):
            raise ServiceError("catalog returned an invalid dataset ID")
        results.extend(batch)
        if limit and len(results) >= limit:
            return results[:limit]
    return results


def _as_results(payload, *keys):
    """Normalize a BRAIN collection response into a list.

    Some endpoints wrap rows in ``{"results": [...]}`` and some return a bare
    JSON array, so callers should not have to care which.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        for key in ("results", "operators", "categories", "data-sets", "datafields"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _catalog_pages(session, path, params=None, keys=(), page_size=None, *, deadline=None):
    """Yield complete collection pages; incomplete or repeated pages are errors."""
    offset = 0
    seen = set()
    total = None
    request_limits = {} if deadline is None else {"deadline": deadline, "max_retries": 1}
    while True:
        query = dict(params or {})
        if page_size is not None or offset:
            query.update(limit=page_size or 50, offset=offset)
        payload = request_json(session, "GET", path, params=query or None, **request_limits)
        collections = (*keys, "results", "operators", "categories", "data-sets", "datafields")
        if not isinstance(payload, list) and not (
            isinstance(payload, dict) and any(isinstance(payload.get(key), list) for key in collections)
        ):
            raise ServiceError("catalog %s returned no collection" % path)
        batch = _as_results(payload, *keys)
        if any(not isinstance(row, dict) for row in batch):
            raise ServiceError("catalog %s returned an invalid row" % path)
        reported = payload.get("count") if isinstance(payload, dict) else None
        if reported is not None:
            if isinstance(reported, bool) or not isinstance(reported, int) or reported < 0:
                raise ServiceError("catalog %s returned an invalid count" % path)
            if total is not None and total != reported:
                raise ServiceError("catalog %s changed during pagination" % path)
            total = reported
        more = bool(payload.get("next")) if isinstance(payload, dict) else False
        if not batch:
            if more or (total is not None and offset < total):
                raise ServiceError("catalog %s ended before all rows were received" % path)
            return
        signature = tuple(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in batch)
        if signature in seen:
            raise ServiceError("catalog %s repeated a page" % path)
        seen.add(signature)
        offset += len(batch)
        if total is not None and offset > total:
            raise ServiceError("catalog %s returned more rows than its count" % path)
        yield batch
        if total is not None and offset == total:
            return
        if total is None and not more and (page_size is None or len(batch) < page_size):
            return
        if page_size is None:
            page_size = len(batch)


def fetch_operators(s, search="", group="", refresh=None, report=None, *, deadline=None):
    """List FASTEXPR operators as ``list[dict]`` (``name``, ``category``, ...).

    Quota-free and cached. ``search`` matches name or description, ``group``
    selects an exact category such as ``Time Series``.
    """
    operators = cached_metadata(
        "operators", {}, lambda: _fetch_operators(s, deadline=deadline), refresh=refresh, report=report)["data"]
    if group:
        operators = [o for o in operators if (o.get("category") or "") == group]
    if search:
        needle = search.lower()
        operators = [
            o for o in operators
            if needle in ((o.get("name") or "") + " " + (o.get("description") or "")).lower()
        ]
    return operators


def _fetch_operators(s, search="", group="", *, deadline=None):
    """List FASTEXPR operators as ``list[dict]`` (``name``, ``category``, ...).

    Quota-free. ``search`` matches the name or description, ``group`` an exact
    category such as ``Time Series``.
    """
    operators = [row for batch in _catalog_pages(s, "/operators", deadline=deadline) for row in batch]
    if group:
        operators = [o for o in operators if (o.get("category") or "") == group]
    if search:
        needle = search.lower()
        operators = [
            o for o in operators
            if needle in ((o.get("name") or "") + " " + (o.get("description") or "")).lower()
        ]
    return operators


def fetch_operator_doc(s, name, refresh=None, report=None):
    """Return one operator's documentation page (markdown body + metadata).

    Cached: the wording changes rarely and this is the most repeated lookup in
    expression work.
    """
    return cached_metadata(
        "operator-doc", {"name": name},
        lambda: _fetch_operator_doc(s, name), refresh=refresh, report=report)["data"]


def _fetch_operator_doc(s, name):
    """Return one operator's documentation page (markdown body + metadata)."""
    return request_json(s, "GET", "/operators/" + name)


def fetch_data_categories(s, region="USA", universe="TOP3000", delay=1,
                          refresh=None, report=None):
    """Return data categories with their field/alpha counts and value score.

    ``alphaCount`` is a rough crowding signal: a high count means many people
    have already mined that category. Cached.
    """
    params = {"region": region, "universe": universe, "delay": delay}
    return _cached_account_catalog(
        s, "data-categories", params,
        lambda session: _fetch_data_categories(session, region, universe, delay),
        refresh, report,
    )


def _fetch_data_categories(s, region="USA", universe="TOP3000", delay=1):
    """Return the data categories with their field/alpha counts and value score.

    ``alphaCount`` is a rough crowding signal: high counts mean "many people
    already mined this".
    """
    payload = request_json(
        s, "GET", "/data-categories",
        params={"region": region, "universe": universe, "delay": delay},
    )
    return _as_results(payload, "categories")


def get_datafields(s, instrument_type="EQUITY", region="USA", delay=1,
                   universe="TOP3000", dataset_id="", search="", field_type=None,
                   limit=None, refresh=None, report=None):
    """List datafields as ``list[dict]`` (``id``, ``description``, ``type``, ...).

    Paginates over ``GET /data-fields`` and stops once ``limit`` rows are
    collected. ``field_type`` filters on the BRAIN type (MATRIX, VECTOR, GROUP).
    Quota-free and cached per query, so re-auditing 4367 fields costs nothing.
    """
    params = {"instrumentType": instrument_type, "region": region, "delay": delay,
              "universe": universe, "dataset": dataset_id, "search": search,
              "type": field_type, "limit": limit}
    derive_from = None
    if search or field_type or limit:
        def filter_complete(rows):
            if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
                try:
                    return _filter_datafields(rows, field_type, search, limit)
                except ServiceError:
                    return None

        derive_from = ({**params, "search": "", "type": None, "limit": None}, filter_complete)
    return _cached_account_catalog(
        s, "datafields", params,
        lambda session: _get_datafields(session, instrument_type, region, delay, universe,
                                        dataset_id, search, field_type, limit),
        refresh, report, derive_from=derive_from,
    )


def _get_datafields(s, instrument_type="EQUITY", region="USA", delay=1,
                   universe="TOP3000", dataset_id="", search="", field_type=None,
                   limit=None):
    """List datafields as ``list[dict]`` (``id``, ``description``, ``type``, ...).

    Paginates over ``GET /datafields`` and stops early once ``limit`` rows are
    collected. ``field_type`` filters on the BRAIN type (MATRIX, VECTOR, GROUP).
    Quota-free.
    """
    params = {"instrumentType": instrument_type, "region": region, "delay": delay,
              "universe": universe, "dataset.id": dataset_id}
    # BRAIN's MATRIX filter has been verified against complete scoped catalogs.
    if field_type == "MATRIX":
        params["type"] = field_type
    pages = _catalog_pages(s, "/data-fields", params, keys=("datafields",), page_size=50)
    return _filter_datafields((item for page in pages for item in page), field_type, search, limit)


def _filter_datafields(rows, field_type, search, limit):
    fields = []
    search = search.lower() if search else ""
    for item in rows:
        field_id = item.get("id")
        if not isinstance(field_id, str) or not field_id:
            raise ServiceError("catalog returned an invalid field ID")
        if field_type and item.get("type") != field_type:
            continue
        if search and search not in (
            field_id + " " + (item.get("description") or "")
        ).lower():
            continue
        fields.append(item)
        if limit and len(fields) >= limit:
            break
    return fields


def search_datafield(s, field, scope, *, deadline):
    """Read one scoped catalog search page for an exact field ID, without caching."""
    params = {key: scope[key] for key in ("instrumentType", "region", "delay", "universe")}
    params.update(search=field, limit=50, offset=0)
    payload = request_json(s, "GET", "/data-fields", params=params, max_retries=1, deadline=deadline)
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ServiceError("field search returned no valid results page")
    count = payload.get("count")
    if type(count) is not int or count < len(rows):
        count = None
    return {"matches": [row for row in rows if row.get("id") == field],
            "returned_count": len(rows), "reported_count": count,
            "complete": count == len(rows) and not bool(payload.get("next"))}


def field_ids(fields, field_type=None, search=""):
    """Return the ``id`` of each field, optionally filtered by type/substring."""
    return [
        f["id"]
        for f in fields
        if f.get("id")
        and (not field_type or f.get("type") == field_type)
        and (
            not search
            or search.lower() in ((f.get("id") or "") + " " + (f.get("description") or "")).lower()
        )
    ]


def fields_to_df(fields):
    """Wrap ``list[dict]`` metadata in a DataFrame when pandas is installed."""
    try:
        import pandas
    except ImportError as exc:
        raise RuntimeError("pandas is required for fields_to_df(); pip install pandas") from exc
    return pandas.DataFrame(fields)


def process_datafields(fields):
    """Turn field metadata into directly usable FASTEXPR snippets.

    Accepts the ``list[dict]`` from :func:`get_datafields` (or a DataFrame, for
    older notebooks). MATRIX fields are winsorized + backfilled, VECTOR fields
    are converted to a scalar via ``vec_avg``.
    """
    if hasattr(fields, "to_dict"):
        fields = fields.to_dict("records")
    matrix = [
        'winsorize(ts_backfill(%s, 120), std=4)' % f["id"]
        for f in fields if f.get("type") == "MATRIX" and f.get("id")
    ]
    vector = [
        'vec_avg(%s)' % f["id"]
        for f in fields if f.get("type") == "VECTOR" and f.get("id")
    ]
    return matrix + vector


BRAIN_API_URL = "https://api.worldquantbrain.com"
_SIMULATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def valid_simulation_id(value):
    """Check the shape of a bare id before using it as a path segment."""
    return isinstance(value, str) and bool(_SIMULATION_ID.fullmatch(value))


def simulation_reference_id(reference):
    """Extract a bare id or the id from one exact BRAIN API simulation URL."""
    if valid_simulation_id(reference):
        return reference
    message = "expected a simulation id or https://api.worldquantbrain.com/simulations/ID"
    if not isinstance(reference, str) or any(
            character.isspace() or ord(character) < 32 or character in "\\?#"
            for character in reference):
        raise ValueError(message)
    try:
        url = urlsplit(reference)
    except ValueError as exc:
        raise ValueError(message) from exc
    base, separator, simulation_id = url.path.removesuffix("/").rpartition("/")
    if (url.scheme != "https" or url.netloc.lower() not in {
            "api.worldquantbrain.com", "api.worldquantbrain.com:443"}
            or not separator or base != "/simulations" or not valid_simulation_id(simulation_id)):
        raise ValueError(message)
    return simulation_id


def _validated_progress_url(location):
    """Resolve a BRAIN Location header and reject unsafe resume targets.

    Progress URLs are persisted in local checkpoints. Treat those files as
    untrusted input: a forged URL must not make the authenticated session send
    cookies to an arbitrary host when ``resume`` polls it.
    """
    if not isinstance(location, str) or not location or any(
        ord(character) < 32 or character == "\\" for character in location
    ):
        raise SimulationError("invalid simulation progress URL")
    resolved = urljoin(BRAIN_API_URL + "/", location)
    try:
        parsed = urlsplit(resolved)
        port = parsed.port
    except ValueError as exc:
        raise SimulationError("invalid simulation progress URL") from exc
    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() != "api.worldquantbrain.com"
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or not parsed.path.startswith("/simulations/")
        or parsed.fragment
    ):
        raise SimulationError(
            "simulation progress URL must target the BRAIN HTTPS simulations endpoint"
        )
    return resolved


def _quick_progress_target(location):
    """Return the original request URL and exact simulation ID for a Quick receipt."""
    url = _validated_progress_url(location)
    try:
        simulation_id = simulation_reference_id(urlsplit(url)._replace(query="").geturl())
    except ValueError as exc:
        raise SimulationError("Quick Location must identify one BRAIN simulation") from exc
    return url, simulation_id


def _request_timeout(deadline):
    """Cap a Requests connect/read timeout at the remaining phase budget."""
    if deadline is None:
        return 60
    return max(0.001, min(60, deadline - time.monotonic()))


def _retry_after_seconds(response, cap_error_wait=True):
    """Seconds BRAIN wants us to wait, or 0 when the request is ready.

    ``Retry-After`` drives both the progress protocol and submission rate
    limiting. It is parsed leniently; callers may cap non-2xx backoff for
    ordinary requests, while bounded simulation polling honors the full delay
    and checkpoints instead of sleeping past its deadline.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return 0.0
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not response.ok and cap_error_wait:
        return min(seconds, 120.0)
    return max(seconds, 0.0)


_TRANSIENT_READ_STATUSES = {408, 502, 503, 504}


def _transient_read_error(exc):
    from requests.exceptions import ChunkedEncodingError, ConnectionError, ContentDecodingError, SSLError, Timeout

    return (isinstance(exc, (ChunkedEncodingError, ConnectionError, ContentDecodingError, Timeout))
            and not isinstance(exc, SSLError))


def _wait_for_read_retry(attempt, attempts, deadline, retry_after=0):
    wait = retry_after if math.isfinite(retry_after) and retry_after > 0 else min(2.0, 0.5 * attempt)
    if attempt >= attempts or (deadline is not None and wait >= deadline - time.monotonic()):
        return False
    sleep(wait)
    return True


def _get_poll_response(s, url, deadline):
    """Retry interrupted reads while leaving the endpoint's readiness protocol intact."""
    from requests.exceptions import RequestException

    attempts = 3
    last = None
    for attempt in range(1, attempts + 1):
        if last is not None and deadline is not None and time.monotonic() >= deadline:
            raise last
        try:
            response = s.get(url, timeout=_request_timeout(deadline))
        except RequestException as exc:
            last = ReadInterrupted(f"BRAIN read interrupted ({type(exc).__name__})")
            if _transient_read_error(exc) and _wait_for_read_retry(attempt, attempts, deadline):
                continue
            raise last from exc
        status = response.status_code
        if status != 429 and status not in _TRANSIENT_READ_STATUSES:
            return response
        delay = _retry_after_seconds(response, cap_error_wait=False)
        if status in (429, 503) and math.isfinite(delay) and delay > 0:
            return response
        last = ReadInterrupted(f"BRAIN read interrupted (HTTP {status})")
        if not _wait_for_read_retry(attempt, attempts, deadline, delay):
            raise last


def _poll_json_when_ready(s, url, on_event, deadline, *, endpoint, status_errors):
    """Return the JSON payload or the delay for the caller to checkpoint."""
    while deadline is None or time.monotonic() < deadline:
        response = _get_poll_response(s, url, deadline)
        error = status_errors.get(response.status_code)
        if error is not None:
            raise error

        delay = _retry_after_seconds(response, cap_error_wait=False)
        if not math.isfinite(delay):
            raise ReadInterrupted("%s returned an invalid Retry-After" % endpoint)
        if response.status_code not in (429, 503) or not delay:
            response.raise_for_status()
        if not delay:
            try:
                return response.json(), None
            except ValueError as exc:
                raise SimulationError("%s endpoint returned non-JSON: %s"
                                      % (endpoint, response.text[:300])) from exc
        if deadline is not None and delay >= deadline - time.monotonic():
            return None, delay
        if on_event:
            on_event("polling", {"retry_after": delay})
        sleep(delay)
    return None, 0


def request_json(s, method, path, params=None, json_body=None, max_retries=4, *, deadline=None):
    """Decode a BRAIN API response or raise a typed HTTP failure.

    Responses with status 429 use backoff within the attempt and time budgets.
    Safe reads also retry interrupted transfers and transient gateway responses.
    Other failures raise AuthError, PermissionDenied, NotFound or ServiceError.
    Successful responses without decodable JSON yield an empty dict.
    """
    from requests.exceptions import RequestException

    url = path if path.startswith("http") else BRAIN_API_URL + path
    read_only = method.upper() in {"GET", "HEAD"}
    if deadline is None:
        deadline = time.monotonic() + _request_timeout(None)
    last = None
    for attempt in range(1, max_retries + 1):
        if time.monotonic() >= deadline:
            raise last or ServiceError("request wait budget expired for %s %s" % (method, url))
        try:
            response = s.request(method, url, params=params, json=json_body,
                                 timeout=_request_timeout(deadline))
        except RequestException as exc:
            last = ServiceError("BRAIN request failed (%s) for %s %s"
                                % (type(exc).__name__, method, url))
            if (read_only and _transient_read_error(exc)
                    and _wait_for_read_retry(attempt, max_retries, deadline)):
                continue
            raise last from exc
        if response.status_code == 401:
            raise AuthError("BRAIN rejected the session (HTTP 401) for %s %s" % (method, url))
        if response.status_code == 403:
            raise PermissionDenied("BRAIN denied access (HTTP 403) for %s %s" % (method, url))
        if response.status_code == 404:
            raise NotFound("%s %s -> 404 (unknown id or endpoint)" % (method, url))
        if response.status_code == 429:
            wait = _retry_after_seconds(response)
            if not math.isfinite(wait) or wait <= 0:
                wait = min(60.0, 5.0 * attempt)
            retry_after = _retry_after_seconds(response, cap_error_wait=False)
            last = RateLimited("%s %s is rate limited" % (method, url),
                               retry_after=retry_after if math.isfinite(retry_after) and retry_after > 0 else wait)
            if attempt < max_retries and wait < deadline - time.monotonic():
                sleep(wait)
                continue
            raise last
        if not response.ok:
            last = ServiceError(
                "HTTP %s from %s %s: %s"
                % (response.status_code, method, url, response.text[:300])
            )
            if (read_only and response.status_code in _TRANSIENT_READ_STATUSES
                    and _wait_for_read_retry(attempt, max_retries, deadline,
                                             _retry_after_seconds(response, cap_error_wait=False))):
                continue
            raise last
        try:
            return response.json()
        except ValueError:
            return {}
    raise last or RateLimited("%s %s" % (method, url))


def read_simulation_snapshot(s, simulation_id, *, deadline):
    """Read a snapshot, preserving terminal failure evidence in HTTP 400 replies."""
    if not valid_simulation_id(simulation_id):
        raise ValueError("invalid simulation id")
    url = f"{BRAIN_API_URL}/simulations/{simulation_id}"
    return _read_simulation_snapshot(s, url, simulation_id, deadline=deadline)


def _read_simulation_snapshot(s, url, simulation_id, *, deadline):
    from requests.exceptions import RequestException

    if deadline is not None and time.monotonic() >= deadline:
        raise ServiceError("simulation inspection wait budget expired")
    try:
        response = s.get(url, timeout=_request_timeout(deadline), allow_redirects=False)
    except RequestException as exc:
        raise ServiceError("simulation inspection request failed") from exc
    status = response.status_code
    if status in (401, 403):
        raise AuthError("BRAIN denied access to this simulation")
    if status == 404:
        raise NotFound("simulation not found")
    if status not in (200, 400, 429):
        raise ServiceError(f"simulation inspection returned HTTP {status}")
    seconds = _retry_after_seconds(response, cap_error_wait=False)
    if not math.isfinite(seconds):
        raise ServiceError("simulation inspection returned an invalid Retry-After")
    retry_after = seconds if seconds > 0 else None
    if status == 429:
        wait = f", retry after {retry_after:g} seconds" if retry_after is not None else ""
        raise RateLimited(f"simulation inspection rate limited{wait}", retry_after=retry_after)
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if status == 400:
        from simulation_slots import TERMINAL_FAILURE_STATUSES

        task = payload if isinstance(payload, dict) else {}
        echoed_id, children = task.get("id"), task.get("children")
        failed = str(task.get("status")).upper() in TERMINAL_FAILURE_STATUSES
        identified = echoed_id == simulation_id or (
            echoed_id is None and isinstance(children, list) and bool(children))
        if not (failed and identified):
            raise ServiceError("simulation inspection returned HTTP 400")
    if not isinstance(payload, dict):
        if retry_after is None:
            raise ServiceError("simulation inspection returned an invalid response")
        payload = {}
    return payload, retry_after


def simulation_progress_receipt(response, body, simulation_id):
    """Return a terminal snapshot that belongs to the requested simulation."""
    import simulation_slots

    if not isinstance(body, dict) or response.status_code not in (200, 400):
        return None
    echoed_id = body.get("id")
    status = str(body.get("status", "")).upper()
    failed = status in simulation_slots.TERMINAL_FAILURE_STATUSES
    if echoed_id not in (None, simulation_id) or status not in simulation_slots.TERMINAL_STATUSES:
        return None
    if response.status_code == 400 and (echoed_id != simulation_id or not failed):
        return None
    alpha_id = body.get("alpha") or body.get("alphaId")
    if (alpha_id is not None and not valid_simulation_id(alpha_id)) or (not failed and not alpha_id):
        return None
    return {"status": status, "alpha_id": alpha_id}


QUICK_ORPHAN_SWEEP_SECONDS = 5


def recover_accepted_slots(s, account, *, deadline=None):
    """Read ordinary/native acceptances before declaring the shared slots full.

    The sweep has one short budget, never posts, and never turns age or a
    missing response into proof of completion. Native reservations point to
    their parent, so child metrics cannot release or multiply parent slots.
    """
    import simulation_capabilities
    import simulation_slots

    fingerprint = account.get("fingerprint")
    if not fingerprint or simulation_capabilities.account_provenance(s).get("fingerprint") != fingerprint:
        return {"state": "account_unverified", "released": [], "kept": [], "checked": 0}
    end = time.monotonic() + QUICK_ORPHAN_SWEEP_SECONDS
    if deadline is not None:
        end = min(end, deadline)
    remaining = max(1, simulation_slots.confirmed_accepted_count(account) or 0)
    observed = {}

    def poll(url):
        nonlocal remaining
        try:
            simulation_id = simulation_reference_id(url)
        except ValueError:
            return None
        if simulation_id in observed:
            return observed[simulation_id]
        now = time.monotonic()
        request_deadline = now + max(0, end - now) / remaining
        remaining = max(1, remaining - 1)
        try:
            payload, delay = read_simulation_snapshot(s, simulation_id, deadline=request_deadline)
        except AuthError:
            raise
        except NotFound:
            answer = "NOT_FOUND"
        except Exception:
            answer = None
        else:
            status = str(payload.get("status") or "").upper()
            alpha = payload.get("alpha") or payload.get("alphaId")
            if (payload.get("id") not in (None, simulation_id)
                    or (alpha is not None and not valid_simulation_id(alpha))):
                answer = None
            elif delay:
                answer = "PENDING"
            else:
                answer = {"status": status, "alpha": alpha}
        observed[simulation_id] = answer
        return answer

    try:
        report = simulation_slots.reconcile_orphans(account, poll, deadline=end, ordinary_only=True)
    except cache_store.LockTimeout:
        report = {"state": "journal_busy", "released": [], "kept": []}
    report.setdefault("state", "observed")
    report["checked"] = len(observed)
    return report


def recover_quick_orphans(s, account, *, deadline=None, exclude_job=None, limit=None, offset=0):
    """Reclaim orphaned Quick slots within a short, shared recovery budget."""
    import simulation_slots
    from requests.exceptions import RequestException

    if not account.get("fingerprint"):
        return 0
    short_deadline = time.monotonic() + QUICK_ORPHAN_SWEEP_SECONDS
    deadline = min(deadline, short_deadline) if deadline is not None else short_deadline
    if time.monotonic() >= deadline:
        return 0
    jobs = simulation_slots.quick_orphans(account, exclude_job=exclude_job)
    if not jobs:
        return 0
    start = offset % len(jobs)
    jobs = (jobs[start:] + jobs[:start])[:limit]
    released = 0
    for position, job in enumerate(jobs):
        now = time.monotonic()
        if now >= deadline:
            break
        try:
            url, simulation_id = _quick_progress_target(job["progress_url"])
            # Leave time to inspect later reservations if this endpoint times out.
            request_deadline = now + (deadline - now) / (len(jobs) - position)
            response = s.get(url, timeout=_request_timeout(request_deadline), allow_redirects=False)
        except (SimulationError, RequestException):
            continue
        if response.status_code in (401, 403):
            raise AuthError("session rejected while recovering a Quick simulation")
        delay = _retry_after_seconds(response, cap_error_wait=False)
        if math.isfinite(delay) and delay > 0:
            try:
                simulation_slots.quick_defer(account, job["id"], time.time() + delay, deadline=deadline)
            except cache_store.LockTimeout:
                break
            continue
        if response.status_code not in (200, 400):
            continue
        try:
            payload = response.json()
        except ValueError:
            continue
        receipt = simulation_progress_receipt(response, payload, simulation_id)
        if not receipt:
            continue
        try:
            released += simulation_slots.quick_update(account, job["id"], "terminal", retain=True,
                                                      terminal_result=receipt, deadline=deadline)
        except cache_store.LockTimeout:
            break
    return released


def attach_quick_location(s, account, job_key, location, *, deadline=None, confirm_failure=False):
    """Attach the original, independently recovered Location of an ambiguous Quick POST."""
    import simulation_capabilities
    import simulation_slots

    fingerprint = account.get("fingerprint")
    if not fingerprint or simulation_capabilities.account_provenance(s).get("fingerprint") != fingerprint:
        raise ValueError("Quick Location recovery requires its authenticated account")
    if not isinstance(job_key, str) or not job_key:
        raise ValueError("Quick Location recovery requires a job key")
    url, simulation_id = _quick_progress_target(location)
    payload, delay = _read_simulation_snapshot(s, url, simulation_id, deadline=deadline)
    status = str(payload.get("status") or "").upper()
    progress = payload.get("progress")
    if (payload.get("id") not in (None, simulation_id)
            or not (payload.get("id") == simulation_id or status
                    or type(progress) in (int, float) and 0 <= progress <= 1)):
        raise ServiceError("BRAIN did not confirm the recovered Quick simulation identity")
    if confirm_failure:
        if (delay or payload.get("id") != simulation_id
                or status not in simulation_slots.TERMINAL_FAILURE_STATUSES
                or payload.get("alpha") or payload.get("alphaId")):
            raise ServiceError("BRAIN did not confirm a terminal failure without an Alpha for this Location")
        return simulation_slots.quick_attach_location(account, job_key, url, terminal_failure=status)
    next_poll_at = time.time() + delay if delay else None
    return simulation_slots.quick_attach_location(account, job_key, url, next_poll_at=next_poll_at)


def poll_simulation(s, progress_url, on_event=None, deadline=None):
    """Poll a simulation until completion or an optional monotonic deadline.

    If BRAIN's ``Retry-After`` would run past ``deadline``, return a resumable
    ``PENDING`` payload instead of sleeping for hours. The progress URL is kept
    so callers can checkpoint it and continue in a later invocation.
    """
    progress_url = _validated_progress_url(progress_url)
    payload, delay = _poll_json_when_ready(
        s, progress_url, on_event, deadline, endpoint="progress",
        status_errors={status: AuthError("session rejected while polling (HTTP %s)" % status)
                       for status in (401, 403)},
    )
    if delay is None:
        return payload
    return {"status": "PENDING", "progress_url": progress_url, "next_poll_after_seconds": delay}


def submit_single_simulation(
    s, simulation_data, max_retries=5, on_event=None, deadline=None
):
    """POST one single-alpha simulation and return its progress URL.

    Daily-quota headers are retained separately from transient HTTP throttling.
    Confirmed exhaustion defers new work; accepted jobs remain recoverable.
    """
    return _submit_simulation_request(s, simulation_data, max_retries, on_event, deadline)


def submit_multi_simulation(s, simulations, max_retries=5, on_event=None, deadline=None):
    """POST an array of REGULAR simulation requests and return the parent URL."""
    if not isinstance(simulations, list) or not 2 <= len(simulations) <= 10:
        raise ValueError("native multi-simulation needs 2–10 requests")
    if any(not isinstance(request, dict) or request.get("type") != "REGULAR"
           or not isinstance(request.get("settings"), dict) for request in simulations):
        raise ValueError("native multi-simulation accepts REGULAR request objects")
    first = simulations[0]["settings"]
    common = ("instrumentType", "region", "delay", "language")
    if any(any(request["settings"].get(key) != first.get(key) for key in common)
           for request in simulations):
        raise ValueError("native multi-simulation needs common instrumentType, region, delay and language")
    if _has_multi_simulation_permission(s) is False:
        raise PermissionDenied("native multi-simulation requires MULTI_SIMULATION permission")
    parent_url = _submit_simulation_request(s, simulations, max_retries, on_event, deadline)
    try:
        simulation_reference_id(parent_url)
    except ValueError as exc:
        raise SimulationError("native multi-simulation returned an invalid parent Location") from exc
    return parent_url


def _has_multi_simulation_permission(session):
    permissions = getattr(session, "_brain_permissions", None)
    if isinstance(permissions, (list, tuple, set, frozenset)):
        return "MULTI_SIMULATION" in permissions
    return None


def _submit_simulation_request(s, simulation_data, max_retries, on_event, deadline):
    import simulation_capabilities
    import simulation_slots

    native = isinstance(simulation_data, list)
    body = ([format_simulation_payload(item) for item in simulation_data]
            if native else format_simulation_payload(simulation_data))
    redirect_limits = {"allow_redirects": False} if native else {}
    account = simulation_capabilities.account_provenance(s)
    for attempt in range(1, max_retries + 1):
        if deadline is not None and deadline - time.monotonic() <= 0:
            raise RateLimited("submission deferred by the run time limit")
        quota = simulation_slots.quota_snapshot(account)
        if quota["blocked"]:
            if on_event:
                on_event("quota", quota)
            raise RateLimited("daily simulation quota exhausted; resume after the reported reset")
        if simulation_slots.submission_retry_at(account) is not None:
            raise RateLimited("simulation submission deferred by shared Retry-After cooldown")
        response = s.post(
            BRAIN_API_URL + "/simulations", json=body,
            timeout=_request_timeout(deadline), **redirect_limits,
        )
        observation = (simulation_slots.observe_quota(account, response.headers)
                       if response.status_code not in (401, 403) else None)
        if observation is not None and on_event:
            on_event("quota", observation)
        if response.status_code == 429 and observation is not None and observation["remaining"] == 0:
            raise RateLimited("daily simulation quota exhausted; resume after the reported reset")
        if response.status_code == 429:
            wait = _retry_after_seconds(response, cap_error_wait=False)
            if not math.isfinite(wait) or wait <= 0:
                wait = min(60.0, 5.0 * attempt)
            retry_at = simulation_slots.defer_submission(account, wait, deadline=deadline)
            if retry_at is not None:
                wait = max(wait, retry_at - time.time())
            if on_event:
                request = ({"child_count": len(simulation_data)} if isinstance(simulation_data, list)
                           else {"expression": simulation_data.get("regular")})
                on_event("rate_limited", {"attempt": attempt, "wait_seconds": wait, **request})
            if attempt < max_retries:
                if deadline is not None:
                    remaining = max(0.0, deadline - time.monotonic())
                    if wait >= remaining:
                        raise RateLimited(
                            "submission deferred until after the run time limit"
                        )
                sleep(wait)
                continue
            raise RateLimited(
                "submission still rate limited after %d attempts" % max_retries
            )
        if response.status_code in (401, 403):
            raise AuthError(
                "submission rejected (HTTP %s)" % response.status_code
            )
        if native and 300 <= response.status_code < 400:
            raise SimulationError("native multi-simulation returned an unexpected redirect")
        response.raise_for_status()
        location = response.headers.get("Location")
        if not location:
            raise SimulationError(
                "submit response has no Location header (HTTP %s): %s"
                % (response.status_code, response.text[:500])
            )
        return _validated_progress_url(location)


def fetch_alpha_metrics(s, alpha_id, on_event=None, deadline=None):
    """Read one alpha's settings, in-sample metrics and platform checks.

    Returns the raw ``GET /alphas/{id}`` payload (``regular``, ``settings``,
    ``is``, ``checks``, ...). Quota-free: it never consumes a simulation slot.
    Raises ``NotFound`` for an unknown id. Returns a private ``_deferred``
    sentinel when a Retry-After would exceed ``deadline``.
    """
    payload, delay = _poll_json_when_ready(
        s, BRAIN_API_URL + "/alphas/" + alpha_id, on_event, deadline, endpoint="alpha",
        status_errors={404: NotFound("alpha %s not found" % alpha_id)},
    )
    if delay is None:
        return payload
    return {"_deferred": True, "next_poll_after_seconds": delay}


def is_quick_simulation(alpha):
    """Identify a Quick simulation from its recorded settings."""
    settings = alpha.get("settings")
    mode = settings.get("simulationMode") if isinstance(settings, dict) else None
    return isinstance(mode, str) and mode.strip().upper() == "QUICK"


def alpha_classification_ids(alpha):
    """Return native classification IDs, preserving missing versus empty evidence."""
    classifications = alpha.get("classifications")
    if not isinstance(classifications, list):
        return None
    identifiers = [item.get("id") if isinstance(item, dict) else item for item in classifications]
    if not all(isinstance(identifier, str) and identifier for identifier in identifiers):
        return None
    return list(dict.fromkeys(identifiers))


def defer_alpha_metrics(record, payload):
    """Keep a simulation resumable when the metrics response asks us to wait."""
    if not payload.get("_deferred"):
        return False
    wait = max(0.0, float(payload.get("next_poll_after_seconds") or 0))
    record.update(status="PENDING", metrics_pending=True, poll_after_seconds=wait)
    if wait:
        record["next_poll_at"] = time.time() + wait
    else:
        record.pop("next_poll_at", None)
    return True


def normalize_setting_keys(overrides=None):
    """Canonicalize known key spellings within one input layer, without defaults.

    This must run before merging files, arms or CLI flags: spelling must not
    decide which layer wins. Unknown platform extensions remain untouched.
    """
    canonical_keys = {key.replace("_", "").casefold(): key for key in (*DEFAULT_SETTINGS, "simulationMode")}
    settings, origins = {}, {}
    for key, value in (overrides or {}).items():
        canonical = canonical_keys.get(str(key).replace("_", "").casefold(), key)
        if canonical == "simulationMode" and isinstance(value, str):
            mode = value.strip().upper()
            if mode in {"QUICK", "FULL"}:
                value = mode
        if canonical in settings:
            previous = settings[canonical]
            if value is None and previous is not None:
                continue
            if previous is not None and value is not None and previous != value:
                raise ValueError("conflicting settings %r and %r for %r" %
                                 (origins[canonical], key, canonical))
        settings[canonical], origins[canonical] = value, key
    return settings


def resolve_settings(overrides=None):
    """Merge defaults and validate structure, leaving platform support to evidence.

    Observed enum lists are hints, not permission gates. Explicit null extension
    values are retained; null known settings retain their historical default.
    """
    settings = dict(DEFAULT_SETTINGS)
    for key, value in normalize_setting_keys(overrides).items():
        if value is not None or key not in DEFAULT_SETTINGS:
            settings[key] = value
    if is_python_language(settings.get("language")):
        settings["language"] = "PYTHON"
    for key in ("region", "instrumentType", "universe", "neutralization",
                "pasteurization", "nanHandling"):
        value = settings.get(key)
        if not isinstance(value, str) or not _SETTING_TOKEN.fullmatch(value):
            raise ValueError("%s must be a well-formed settings token, got %r" % (key, value))
    delay = settings["delay"]
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
        raise ValueError("delay must be a non-negative integer, got %r" % (delay,))
    return settings


#: Settings the OPTIONS schema advertises as required but the POST contract
#: rejects for ``type=REGULAR`` ("Selection handling ... is not allowed for
#: simulation type REGULAR"). They belong to a selection simulation type this
#: REGULAR path never sends, even if legacy specs include them. Preflight and
#: dispatch check the actual REGULAR settings against the account schema.
_REGULAR_UNSENT_SETTINGS = ("selectionHandling", "selectionLimit")
#: The native schema advertises these FASTEXPR controls, but the Python
#: REGULAR endpoint rejects them. Keep the original settings in the spec and
#: checkpoint; use the wire subset for dispatch and Python identity.
_PYTHON_UNSENT_SETTINGS = ("unitHandling", "nanHandling")


def _regular_unsent_settings(settings):
    if is_python_language(settings.get("language")):
        return _REGULAR_UNSENT_SETTINGS + _PYTHON_UNSENT_SETTINGS
    return _REGULAR_UNSENT_SETTINGS


def _omit_regular_settings(mapping, excluded):
    spellings = {name.casefold() for name in excluded}
    return {key: value for key, value in mapping.items()
            if str(key).replace("_", "").casefold() not in spellings}


def regular_wire_settings(settings):
    """Return the writable settings present in a REGULAR simulation request."""
    return _omit_regular_settings(settings, _regular_unsent_settings(settings))


def format_simulation_payload(payload):
    """Apply safe default FASTEXPR formatting to a native REGULAR request."""
    if (not isinstance(payload, dict) or payload.get("type") != "REGULAR"
            or not isinstance(payload.get("regular"), str)
            or (alpha_language(payload) or "FASTEXPR") != "FASTEXPR"):
        return payload
    from expression_lint import format_fastexpr

    source, _reason = format_fastexpr(payload["regular"])
    return {**payload, "regular": source} if source != payload["regular"] else payload


def _single_simulation_data(expression, decay, settings, *, format_source=True):
    """Build one ``POST /simulations`` payload from a resolved settings dict."""
    body = regular_wire_settings(settings)
    body["decay"] = decay
    payload = {"type": "REGULAR", "settings": body, "regular": expression}
    return format_simulation_payload(payload) if format_source else payload


def regular_settings_evidence(settings, snapshot):
    """Check the settings that the REGULAR request will actually send."""
    from simulation_capabilities import settings_evidence

    wire = regular_wire_settings(settings)
    schema = snapshot.get("settings_schema")
    if isinstance(schema, dict):
        snapshot = {**snapshot, "settings_schema":
                    _omit_regular_settings(schema, _regular_unsent_settings(settings))}
    report = settings_evidence(wire, snapshot)
    report["omitted_from_payload"] = [key for key in settings if key not in wire]
    return report


#: Keys a per-alpha dict in a research file may carry. Anything else is an
#: error rather than a silent drop: a config that says
#: ``{"expression": ..., "neutralization": "SUBINDUSTRY"}`` looks like it
#: overrides the setting, but neutralisation is a settings-level field, so the
#: value would vanish and the round would quietly test nothing. Raising on the
#: unknown key turns a wasted experiment into a message.
_ALPHA_ENTRY_KEYS = frozenset({
    "expression", "decay", "mechanism", "label", "name",
    "family", "candidate_id", "idea", "hypothesis", "note", "role",
})


def normalize_alphas(alphas, settings):
    """Normalize a research-file ``alphas`` entry list into ``(expr, decay)`` pairs.

    Accepts plain strings, ``{"expression": ..., "decay": ...}`` dicts, or
    ``(expression, decay)`` tuples. The default decay comes from ``settings``.
    Unknown keys in a dict entry raise, because silently ignoring them lets a
    config appear to control something it does not.
    """
    default_decay = settings.get("decay", DEFAULT_SETTINGS["decay"])
    normalized = []
    for index, item in enumerate(alphas):
        if isinstance(item, str):
            expression, decay = item, default_decay
        elif isinstance(item, dict):
            unknown = set(item) - _ALPHA_ENTRY_KEYS
            if unknown:
                raise ValueError(
                    "alphas[%d] has unsupported key(s) %s; only %s are read"
                    % (index, ", ".join(sorted(unknown)),
                       ", ".join(sorted(_ALPHA_ENTRY_KEYS))))
            expression = item.get("expression")
            decay = item.get("decay", default_decay)
            if "role" in item and (not isinstance(item["role"], str)
                                   or item["role"] not in {"candidate", "diagnostic"}):
                raise ValueError("alphas[%d] role must be candidate or diagnostic" % index)
        elif isinstance(item, (list, tuple)) and item:
            expression, decay = item[0], (item[1] if len(item) > 1 else default_decay)
        else:
            raise ValueError("alphas[%d] is not a usable expression entry" % index)
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("alphas[%d] has an empty or non-string expression" % index)
        if not isinstance(decay, int) or isinstance(decay, bool) or decay < 0:
            raise ValueError("alphas[%d] has an invalid decay %r" % (index, decay))
        if is_python_language(settings.get("language")):
            expression = normalize_python_source(expression)
        else:
            expression = expression.strip()
        normalized.append((expression, decay))
    return normalized


def alpha_candidate_ids(alphas, count):
    """Per-expression ``candidate_id`` declarations, aligned with pairs.

    Same contract as :func:`alpha_families`: when the list does not line up with
    the normalized pairs it is dropped wholesale. A misaligned label is worse
    than a missing one, because it silently attaches an ablation variant's
    numbers to the wrong leg.
    """
    if not alphas or not isinstance(alphas[0], dict):
        return [None] * count
    ids = [item.get("candidate_id") for item in alphas if isinstance(item, dict)]
    if len(ids) != count:
        return [None] * count
    return ids


def alpha_families(alphas, count):
    """Per-expression ``family`` declarations from a config, aligned with pairs.

    Same contract as :func:`alpha_labels`: a config entry may carry ``family``,
    and the list is dropped wholesale when it does not line up with the
    normalized pairs. A wrong family silently corrupts the diversity count that
    ``alpha book`` reports, so dropping is safer than guessing.
    """
    if not alphas or not isinstance(alphas[0], dict):
        return [None] * count
    families = [item.get("family") for item in alphas if isinstance(item, dict)]
    if len(families) != count:
        return [None] * count
    return families


def alpha_labels(alphas, count):
    """Per-expression labels from a config, aligned with the normalized pairs.

    A config entry may carry ``mechanism`` / ``label`` / ``name``. When the
    shapes do not line up with the normalized pairs (an entry expanded into
    several expressions, plain strings, tuples) the labels are dropped rather
    than mis-assigned: a wrong label in the ledger is worse than no label.
    """
    if not alphas or not isinstance(alphas[0], dict):
        return [None] * count
    labels = [
        item.get("mechanism") or item.get("label") or item.get("name")
        for item in alphas
        if isinstance(item, dict)
    ]
    if len(labels) != count:
        return [None] * count
    return labels


# Local fallback policy only; the platform's unpublished capacity stays unknown.
DEFAULT_LOCAL_CONCURRENCY = 3


def concurrency_plan(session, requested=None, capability_snapshot=None):
    """Resolve an invocation budget from supplied/cached account evidence.

    Never discovers capabilities online. Three is a conservative local default,
    not a platform allowance; any explicit positive integer budget is valid.
    Only a verified capacity bound to the current account can clamp the budget.
    """
    import simulation_capabilities

    if requested is not None and (
        isinstance(requested, bool) or not isinstance(requested, int) or requested < 1
    ):
        raise ValueError("concurrency must be a positive integer")
    evidence = capability_snapshot if capability_snapshot is not None else simulation_capabilities.cached()
    account = simulation_capabilities.account_provenance(session) if session is not None else evidence.get("account", {})
    capacity = evidence.get("concurrency") or {}
    evidence_account = evidence.get("account") or {}
    matched = bool(account.get("fingerprint")) and account.get("fingerprint") == evidence_account.get("fingerprint")
    limit = capacity.get("limit")
    verified = (evidence.get("state") == "verified" and capacity.get("state") == "verified"
                and matched and isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0)
    budget = requested if requested is not None else (limit if verified else DEFAULT_LOCAL_CONCURRENCY)
    return {
        "requested": requested,
        "effective": min(budget, limit) if verified else budget,
        "platform_limit": limit if verified else None,
        "source": ("explicit_local_budget" if requested is not None else
                   "verified_platform_limit" if verified else "conservative_local_default"),
        "platform_source": capacity.get("source") if verified else None,
        "account": account,
        "capability_id": evidence.get("capability_id"),
    }


def simulate_alphas(
    s,
    alphas,
    settings=None,
    *,
    dry_run=False,
    on_event=None,
    max_wait_seconds=None,
    deadline=None,
    initial_records=None,
    labels=None,
    families=None,
    candidates=None,
    concurrency=None,
    capability_snapshot=None,
    scheduling_report=None,
    fetch_metrics=True,
    multi_size=None,
    recover_only=False,
    native_poll_once=False,
):
    """Simulate a list of alphas and return one record per expression.

    Each record is ``{"expression", "decay", "alpha_id", "status", "metrics",
    "checks", "error"}``. ``metrics`` is the in-sample block from
    ``GET /alphas/{id}`` for completed runs; ``checks`` maps platform check
    names to ``{"result", "limit", "value"}``. A failed expression is recorded
    with a populated ``error`` instead of aborting the run. When
    ``max_wait_seconds`` or an absolute monotonic ``deadline`` is set, unfinished work is returned as ``PENDING`` (or
    ``QUEUED`` if it has not yet been submitted) instead of keeping the caller
    blocked. Pass those records back as ``initial_records`` to resume without
    resubmitting accepted jobs.

    ``dry_run`` validates and builds the payloads but never touches the network
    (and requires no credentials). ``on_event(kind, payload)`` receives
    ``submitted`` / ``polling`` / ``quota`` / ``rate_limited`` / ``completed`` events.
    Feeding it the result of ``resolve_settings`` is the caller's job.
    ``concurrency`` is a positive local budget, clamped only by verified account
    capacity. A terminal progress response releases its slot independently
    of the Alpha metrics read. With ``fetch_metrics=False``, keep Alpha IDs in
    the checkpoint for a later metrics-only read of those records.
    ``multi_size`` sends native batches of compatible arms with a maximum size
    between two and ten. ``recover_only`` reads existing work without creating it.
    ``native_poll_once`` checks each due native parent once and saves its next poll time.
    Reservations survive interruptions and are
    reconciled by ``simulation_slots.reconcile_record``; other workspaces and
    unidentified accounts cannot share these reservations.
    """
    if native_poll_once and dry_run:
        raise ValueError("native_poll_once requires live simulation or recovery")
    settings = resolve_settings(settings)
    pairs = normalize_alphas(alphas, settings)
    roles = [item.get("role") if isinstance(item, dict) else None for item in alphas]
    if labels is None:
        labels = alpha_labels(alphas, len(pairs))
    elif len(labels) != len(pairs):
        labels = [None] * len(pairs)
    if families is None:
        families = alpha_families(alphas, len(pairs))
    elif len(families) != len(pairs):
        families = [None] * len(pairs)
    if candidates is None:
        candidates = alpha_candidate_ids(alphas, len(pairs))
    elif len(candidates) != len(pairs):
        candidates = [None] * len(pairs)

    import simulation_capabilities
    import simulation_slots

    native_enabled = multi_size is not None
    if native_enabled or (initial_records and any(row.get("multi_request_id") for row in initial_records)):
        import simulation_multi
    if native_enabled and (type(multi_size) is not int or not 2 <= multi_size <= simulation_multi.MAX_CHILDREN):
        raise ValueError("native multi-simulation size must be an integer between 2 and 10")

    def original_payload_for(index):
        expression, decay = pairs[index]
        return _single_simulation_data(expression, decay, settings, format_source=False)

    @cache
    def payload_for(index):
        return format_simulation_payload(original_payload_for(index))

    if capability_snapshot is None and not dry_run:
        capability_snapshot = simulation_capabilities.cached()
    plan = concurrency_plan(None if dry_run else s, concurrency,
                            (capability_snapshot or {}) if dry_run else capability_snapshot)
    if native_enabled:
        plan["multi_parent_limit"] = simulation_multi.CONSULTANT_PARENT_LIMIT
        plan["effective"] = min(plan["effective"], plan["multi_parent_limit"])
    batch_size = max(1, plan["effective"])
    if scheduling_report is not None:
        scheduling_report.update(plan)
        scheduling_report["shared_slots"] = {
            "state": "dry_run", "scope": "not_reserved", "cross_workspace_coordinated": False,
        }
        scheduling_report["quota"] = simulation_slots.quota_snapshot({} if dry_run else plan["account"])
    if max_wait_seconds is not None and (
        not math.isfinite(max_wait_seconds) or max_wait_seconds < 0
    ):
        raise ValueError("max_wait_seconds must be a finite, non-negative number")

    if dry_run:
        if native_enabled and scheduling_report is not None:
            requests = simulation_multi.request_groups(range(len(pairs)), payload_for, multi_size)
            scheduling_report["multi_requests"] = [
                {"indices": list(group), "payload": [payload_for(index) for index in group]
                 if len(group) > 1 else payload_for(group[0])}
                for group in requests
            ]
        return [
            {
                "expression": expression,
                "decay": decay,
                "candidate_id": candidates[index],
                "mechanism": labels[index],
                "family": families[index],
                **({"role": roles[index]} if roles[index] is not None else {}),
                "alpha_id": None,
                "status": "DRY_RUN",
                "metrics": None,
                "checks": None,
                "error": None,
                "payload": payload_for(index),
            }
            for index, (expression, decay) in enumerate(pairs)
        ]

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from requests.exceptions import HTTPError, RequestException

    if initial_records is None:
        records = [
            {
                "expression": expression,
                "decay": decay,
                "candidate_id": candidates[index],
                "mechanism": labels[index],
                "family": families[index],
                **({"role": roles[index]} if roles[index] is not None else {}),
                "alpha_id": None,
                "status": "QUEUED",
                "submission_attempted": False,
                "metrics": None,
                "checks": None,
                "error": None,
            }
            for index, (expression, decay) in enumerate(pairs)
        ]
    else:
        records = [dict(record) for record in initial_records]
        if len(records) != len(pairs):
            raise ValueError("initial_records must match the number of alphas")
        for index, record in enumerate(records):
            if "role" not in record and roles[index] is not None:
                record["role"] = roles[index]


    def shared_state():
        state = simulation_slots.snapshot(plan["account"], plan["effective"])
        if scheduling_report is not None:
            scheduling_report["shared_slots"] = state
            scheduling_report["submission_retry_at"] = simulation_slots.submission_retry_at(plan["account"])
        return state

    if max_wait_seconds is not None:
        relative_deadline = time.monotonic() + max_wait_seconds
        deadline = relative_deadline if deadline is None else min(deadline, relative_deadline)

    def emit(kind, index, payload=None):
        if on_event:
            data = dict(payload or {})
            data["index"] = index
            on_event(kind, data)

    def checkpoint_submission(index, record, identity):
        if identity is not None:
            record.update(slot_reservation=identity, slot_reservation_state="submitting")
        record.update(submission_attempted=True, status="UNKNOWN_SUBMISSION")
        records[index] = record
        emit("submitting", index, record)

    quota_exhausted = False

    def submission_event(kind, index, payload):
        nonlocal quota_exhausted
        if kind == "quota":
            payload = simulation_slots.quota_snapshot(plan["account"], observation=payload)
            quota_exhausted = quota_exhausted or payload["remaining"] == 0
            if scheduling_report is not None:
                scheduling_report["quota"] = payload
        emit(kind, index, payload)

    has_saved_native = any(record.get("multi_request_id") for record in records)
    native = (simulation_multi.MultiRuns(s, records, payload_for, plan["account"],
              plan["effective"], deadline if deadline is not None else float("inf"), emit,
              submission_event, original_payload_for=original_payload_for, poll_once=native_poll_once)
              if native_enabled or has_saved_native else None)
    saved_groups = list(native.restore()) if has_saved_native else []
    for record in records:
        if not record.get("multi_request_id"):
            simulation_slots.reconcile_record(record, plan["account"])
    shared_state()

    def fetch_metrics_record(index):
        record = records[index]
        next_poll_at = record.pop("next_poll_at", None)
        if next_poll_at:
            wait = max(0.0, float(next_poll_at) - time.time())
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and wait >= remaining:
                record["status"] = "PENDING"
                record["metrics_pending"] = True
                record["poll_after_seconds"] = wait
                record["next_poll_at"] = next_poll_at
                emit("pending", index, record)
                return
            if wait:
                sleep(wait)

        def metrics_event(kind, payload):
            emit(kind, index, payload)

        try:
            payload = fetch_alpha_metrics(
                s,
                record["alpha_id"],
                on_event=metrics_event,
                deadline=deadline,
            )
            if record.get("multi_request_id"):
                if not isinstance(payload, dict) or payload.get("id") not in (None, record["alpha_id"]):
                    raise ServiceError("multi-simulation Alpha metrics identity mismatch")
                metrics = payload.get("is")
                if not payload.get("_deferred") and (not isinstance(metrics, dict) or not metrics):
                    raise ReadInterrupted("multi-simulation Alpha metrics are not available yet")
        except (AuthError, ReadInterrupted) as exc:
            record["status"] = "PENDING"
            record["metrics_pending"] = True
            record["error"] = str(exc)
            emit("pending", index, record)
            return
        except Exception as exc:
            record["error"] = "metrics fetch failed: %s" % exc
            if record.get("multi_request_id"):
                record.update(status="PENDING", metrics_pending=True)
                emit("pending", index, record)
                return
        else:
            if defer_alpha_metrics(record, payload):
                emit("pending", index, record)
                return
            record["error"] = record.pop("simulation_error", None)
            record["metrics"] = payload.get("is")
            record["checks"] = _extract_checks(payload)
            classifications = alpha_classification_ids(payload)
            if classifications is not None:
                record["classification_ids"] = classifications
            split = extract_split(payload)
            if split:
                record["out_of_sample"] = split

        record["status"] = record.pop("simulation_status", record.get("status"))
        record.pop("metrics_pending", None)
        record.pop("next_poll_at", None)
        record.pop("poll_after_seconds", None)
        emit("completed", index, record)

    def poll_record(index):
        record = records[index]
        progress_url = record.get("progress_url")
        if not progress_url:
            return

        next_poll_at = record.pop("next_poll_at", None)
        if next_poll_at:
            wait = max(0.0, float(next_poll_at) - time.time())
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and wait >= remaining:
                record["status"] = "PENDING"
                record["poll_after_seconds"] = wait
                record["next_poll_at"] = next_poll_at
                emit("pending", index, record)
                return
            if wait:
                sleep(wait)

        def poll_event(kind, payload):
            emit(kind, index, payload)

        try:
            progress = poll_simulation(
                s, progress_url, on_event=poll_event, deadline=deadline
            )
            if progress.get("status") == "PENDING":
                wait = max(
                    0.0, float(progress.get("next_poll_after_seconds") or 0)
                )
                record["status"] = "PENDING"
                record["progress_url"] = progress_url
                record["poll_after_seconds"] = wait
                if wait:
                    record["next_poll_at"] = time.time() + wait
                emit("pending", index, record)
                return

            record["status"] = progress.get("status") or "UNKNOWN"
            record["alpha_id"] = progress.get("alpha")
            if not record["alpha_id"] and record["status"] not in simulation_slots.TERMINAL_STATUSES:
                record["status"] = "PENDING"
                record["progress_url"] = progress_url
                emit("pending", index, record)
                return
            simulation_slots.reconcile_record(record, plan["account"], outcome="terminal")
            record["error"] = None
            if progress.get("status") in simulation_slots.TERMINAL_FAILURE_STATUSES or progress.get("message"):
                record["error"] = progress.get("message") or "simulation ended with %s" % record["status"]
                if not is_python_language(settings.get("language")):
                    note_operator_rejection(record.get("expression"), record["error"])
            if record["alpha_id"]:
                record["simulation_status"] = record["status"]
                if record["error"]:
                    record["simulation_error"] = record["error"]
                record["status"] = "PENDING"
                record["metrics_pending"] = True
            record.pop("progress_url", None)
            if record.get("metrics_pending"):
                emit("pending", index, record)
            else:
                record.pop("next_poll_at", None)
                record.pop("poll_after_seconds", None)
                emit("completed", index, record)
        except (AuthError, ReadInterrupted) as exc:
            record["status"] = "PENDING"
            record["error"] = str(exc)
            record["progress_url"] = progress_url
            emit("pending", index, record)
        except Exception as exc:  # preserve the accepted job for a later resume
            record["status"] = "FAILED"
            record["error"] = "poll failed: %s" % exc
            record["progress_url"] = progress_url
            emit("failed", index, record)

    snapshot_account = (capability_snapshot or {}).get("account") or {}
    settings_snapshot = capability_snapshot if (
        plan["account"].get("fingerprint")
        and plan["account"]["fingerprint"] == snapshot_account.get("fingerprint")
    ) else {}
    settings_checks = {}

    def settings_allowed(index):
        decay = pairs[index][1]
        if decay not in settings_checks:
            settings_checks[decay] = regular_settings_evidence(
                {**settings, "decay": decay}, settings_snapshot)
        capability_check = settings_checks[decay]
        if not capability_check["unsupported"]:
            return True
        record = records[index]
        record.update(status="FAILED", submission_attempted=False, submission_refused=True,
                      error="settings are unsupported by current account metadata",
                      settings_capabilities=capability_check)
        emit("failed", index, record)
        return False

    def submit_record(index):
        record = dict(records[index])
        payload = payload_for(index)
        identity, slots = simulation_slots.prepare_submission(
            plan["account"], plan["effective"],
            lambda reservation: checkpoint_submission(index, record, reservation),
        )
        if scheduling_report is not None:
            scheduling_report["shared_slots"] = slots
        if identity is None and slots["available"] == 0:
            record["error"] = "shared simulation slots are occupied; reconcile or resume existing attempts"
            records[index] = record
            emit("deferred", index, record)
            return "contended"
        try:
            location = submit_single_simulation(
                s, payload,
                on_event=lambda kind, payload, i=index: submission_event(kind, i, payload),
                deadline=deadline,
            )
        except RateLimited as exc:
            simulation_slots.reconcile_record(record, plan["account"], outcome="refused")
            record["status"] = "QUEUED"
            record["submission_attempted"] = False
            record["error"] = "%s: %s" % (exc.code, exc)
            records[index] = record
            emit("deferred", index, record)
            return "blocked"
        except AuthError as exc:
            simulation_slots.reconcile_record(record, plan["account"], outcome="refused")
            shared_state()
            record.update(status="QUEUED", submission_attempted=False, error=str(exc))
            records[index] = record
            emit("deferred", index, record)
            raise
        except SimulationError as exc:
            record.update(status="UNKNOWN_SUBMISSION", error=str(exc))
            records[index] = record
            emit("pending", index, record)
            return "blocked"
        except BrainError as exc:
            record["status"] = "UNKNOWN_SUBMISSION"
            record["error"] = "%s: %s" % (exc.code, exc)
            records[index] = record
            emit("pending", index, record)
            return "blocked"
        except HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            refused = status_code is not None and 400 <= status_code < 500 and status_code != 408
            record["status"] = "FAILED" if refused else "UNKNOWN_SUBMISSION"
            record["error"] = "submit failed: %s" % exc
            if refused:
                detail = getattr(exc.response, "text", "")
                if is_python_language(settings.get("language")) and isinstance(detail, str) and detail.strip():
                    record["error"] += " (" + detail.strip()[:500] + ")"
                simulation_slots.reconcile_record(record, plan["account"], outcome="refused")
                record.update(status="FAILED", submission_refused=True)
                if not is_python_language(settings.get("language")):
                    note_operator_rejection(record.get("expression"), record["error"])
            records[index] = record
            emit("failed" if refused else "pending", index, record)
            return "skipped" if refused else "blocked"
        except RequestException as exc:
            record.update(status="UNKNOWN_SUBMISSION", error="submit transport failed: %s" % type(exc).__name__)
            records[index] = record
            emit("pending", index, record)
            return "blocked"
        record["status"] = "PENDING"
        record["progress_url"] = location
        record["error"] = None
        simulation_slots.reconcile_record(record, plan["account"], outcome="accepted")
        records[index] = record
        emit("submitted", index, record)
        return "accepted"

    def read_result(future):
        try:
            future.result()
        except Exception as exc:  # One read failure should not discard the other jobs.
            note("simulation read worker raised %s: %s" % (type(exc).__name__, exc))

    def dispatch(group):
        for index in group:
            record = records[index]
            if record.get("submission_attempted"):
                record["status"] = "UNKNOWN_SUBMISSION"
                record["error"] = "submission outcome needs reconciliation before another POST"
                emit("pending", index, record)
                return "blocked", ()
        if quota_exhausted:
            for index in group:
                record = records[index]
                record["error"] = "daily simulation quota exhausted; resume after the reported reset"
                emit("deferred", index, record)
            return "blocked", ()
        if simulation_slots.submission_retry_at(plan["account"]) is not None:
            for index in group:
                record = records[index]
                record["error"] = "simulation submission deferred by shared Retry-After cooldown"
                emit("deferred", index, record)
            return "blocked", ()
        allowed = tuple(index for index in group if records[index].get("status") != "FAILED"
                        and settings_allowed(index))
        if not allowed:
            return "skipped", ()
        if len(allowed) == 1:
            return submit_record(allowed[0]), allowed
        return native.submit(allowed), allowed

    queued = [index for index, record in enumerate(records)
              if record.get("status") in {"QUEUED", None}
              and not (record.get("progress_url") or record.get("metrics_pending") or record.get("multi_request_id"))]
    groups = (simulation_multi.request_groups(queued, payload_for, multi_size) if native_enabled
              else ((index,) for index in queued))
    next_group = next(groups, None)
    stop_submitting = recover_only or any(record.get("status") == "UNKNOWN_SUBMISSION" for record in records)
    unresolved_polls = 0
    read_workers = max(2, batch_size)
    attempted_orphan_recovery = launched_here = False

    with (ThreadPoolExecutor(max_workers=read_workers) as progress_pool,
          ThreadPoolExecutor(max_workers=read_workers) as metrics_pool):
        progress_reads = {}
        metrics_reads = []

        def read_progress(group):
            future = (progress_pool.submit(poll_record, group[0]) if len(group) == 1
                      else progress_pool.submit(native.poll_parent, group))
            progress_reads[future] = group

        def read_metrics(index):
            if fetch_metrics and records[index].get("metrics_pending"):
                metrics_reads.append(metrics_pool.submit(fetch_metrics_record, index))

        def read_children_and_metrics(group):
            native.collect_children(group)
            for index in group:
                if fetch_metrics and records[index].get("metrics_pending"):
                    fetch_metrics_record(index)

        def read_native_results(group):
            if native.needs_children(group):
                metrics_reads.append(metrics_pool.submit(read_children_and_metrics, group))
            else:
                for index in group:
                    read_metrics(index)

        for group in saved_groups:
            if native.needs_parent_read(group):
                read_progress(group)
            else:
                read_native_results(group)
        for index, record in enumerate(records):
            if record.get("multi_request_id"):
                continue
            if record.get("progress_url"):
                read_progress((index,))
            else:
                read_metrics(index)

        # Authenticated recovery restores shared reservations before dispatch. With
        # unidentified legacy jobs, wait for the original reads before using slots.
        untracked_reads = set() if plan["account"].get("fingerprint") else set(progress_reads)
        while True:
            while (next_group is not None and not stop_submitting and not untracked_reads
                   and len(progress_reads) + unresolved_polls < plan["effective"]
                   and (deadline is None or time.monotonic() < deadline)):
                outcome, accepted = dispatch(next_group)
                if outcome == "accepted":
                    launched_here = True
                    read_progress(accepted)
                elif outcome in {"contended", "blocked"}:
                    if outcome == "contended" and not attempted_orphan_recovery:
                        attempted_orphan_recovery = True
                        try:
                            recovery_deadline = time.monotonic() + QUICK_ORPHAN_SWEEP_SECONDS
                            if deadline is not None:
                                recovery_deadline = min(recovery_deadline, deadline)
                            recovered = recover_accepted_slots(s, plan["account"], deadline=recovery_deadline)
                            if scheduling_report is not None:
                                scheduling_report["accepted_slot_recovery"] = recovered
                            if recovered["released"]:
                                continue
                            if recover_quick_orphans(s, plan["account"], deadline=recovery_deadline):
                                continue
                        except AuthError as exc:
                            if not launched_here:
                                raise
                            # Keep this invocation's accepted jobs available to its caller.
                            note(str(exc))
                    stop_submitting = outcome == "blocked"
                    break
                next_group = next(groups, None)

            if not progress_reads:
                break
            ready, _ = wait(progress_reads, return_when=FIRST_COMPLETED)
            for future in ready:
                group = progress_reads.pop(future)
                read_result(future)
                pending = (records[group[0]].get("progress_url") if len(group) == 1
                           else native.pending_on_platform(group))
                if future in untracked_reads:
                    untracked_reads.remove(future)
                    stop_submitting = stop_submitting or bool(pending)
                if pending:
                    # The attempt still occupies a slot even though this read has yielded.
                    unresolved_polls += 1
                elif len(group) > 1:
                    read_native_results(group)
                else:
                    read_metrics(group[0])

        for future in metrics_reads:
            read_result(future)

    shared_state()
    return records


_OPERATOR_MENTION = re.compile(r"[Oo]perator\s+[`'\"]?([A-Za-z_][A-Za-z0-9_]*)")
_OPERATOR_FATAL = re.compile(
    r"unknown operator|not a valid operator|operator .{0,40}(not found|does not "
    r"exist|is not supported|not available|not permitted|is not allowed)",
    re.IGNORECASE,
)


def note_operator_rejection(expression, message, context=None):
    """Learn an operator-level rejection so the linter flags it next time.

    BRAIN answers some rejections with ``Operator <name> ...``. Two flavours
    matter and they are not equally severe:

    * the operator is unknown or unavailable in this config -- ``block``;
    * the operator exists but does not accept this kind of input, e.g.
      ``Operator subtract does not support event inputs`` -- ``warn``.

    Blanket-denying the second kind would disable perfectly good operators, so
    the two are stored with different severities.
    """
    if not message:
        return None
    match = _OPERATOR_MENTION.search(message)
    if not match:
        return None
    operator = match.group(1)
    severity = "block" if _OPERATOR_FATAL.search(message) else "warn"
    try:
        import expression_lint
        return expression_lint.record_rejection(
            operator, message, severity=severity,
            context=context or {"expression": expression},
        )
    except Exception:
        # Learning from a rejection must never break a simulation run.
        return None


# Checks whose limit is a ceiling: PASS when value <= limit. Every other check
# in the catalog uses a floor (value >= limit). The displayed value and formal
# verdict can differ, including when platform exceptions apply. Numeric
# comparison is diagnostic evidence and never replaces the formal result.
_CEILING_CHECKS = frozenset({
    "SELF_CORRELATION",
    "PROD_CORRELATION",
    "CONCENTRATED_WEIGHT",
    "HIGH_TURNOVER",
})


def reconcile_check(name, check):
    """Compare a check's value and limit without changing its formal verdict.

    Returns ``{"expected", "consistent"}``, or None when the check carries no
    value/limit pair to compare (CONCENTRATED_WEIGHT often has neither) or is
    still pending.
    """
    check = check or {}
    value = check.get("value")
    limit = check.get("limit")
    result = check.get("result")
    if value is None or limit is None or result in (None, "PENDING"):
        return None
    lower_is_better = (name or "").upper() in _CEILING_CHECKS
    passed = value <= limit if lower_is_better else value >= limit
    expected = "PASS" if passed else "FAIL"
    return {"expected": expected, "consistent": expected == result}


def _extract_checks(alpha_payload):
    """Normalize platform checks, preserving ambiguity and numeric diagnostics."""
    checks = normalize_checks((alpha_payload.get("is") or {}).get("checks"))
    for name, info in checks.items():
        info.pop("expected_result", None)
        info.pop("value_consistent", None)
        for key in ("result", "limit", "value"):
            info.setdefault(key, None)
        verdict = reconcile_check(name, info)
        if verdict is not None:
            info["expected_result"] = verdict["expected"]
            info["value_consistent"] = verdict["consistent"]
    return checks or None


def fetch_alpha_checks(s, alpha_id, max_wait_seconds=180.0, on_event=None):
    """Return live platform checks, including computed SELF_CORRELATION.

    ``GET /alphas/{id}`` embeds a *cached* check snapshot, which reports
    SELF_CORRELATION as PENDING or omits it entirely. The dedicated
    ``GET /alphas/{id}/check`` endpoint triggers the computation and answers
    with ``Retry-After`` until the job finishes, so it is the only way to read
    the real correlation value. This polls it within a bounded budget.

    ``complete`` indicates a nonempty check payload was received, not that every
    check resolved. ``payload_received`` also records successful empty payloads.
    The CLI reports check resolution separately from transport completion.
    A zero wait budget allows one request with the default transport timeout.
    """
    if max_wait_seconds is not None and (not math.isfinite(max_wait_seconds) or max_wait_seconds < 0):
        raise ValueError("max_wait_seconds must be finite and non-negative")
    deadline = time.monotonic() + max_wait_seconds if max_wait_seconds else None
    url = BRAIN_API_URL + "/alphas/" + alpha_id + "/check"
    payload = None
    checks = None
    while True:
        response = s.get(url, timeout=_request_timeout(deadline))
        if response.status_code in (401, 403):
            raise AuthError("BRAIN rejected the session (HTTP %s) for alpha %s checks"
                            % (response.status_code, alpha_id))
        if response.status_code == 404:
            raise NotFound("alpha %s not found" % alpha_id)
        if response.status_code == 429:
            raise RateLimited("checks for alpha %s are rate limited" % alpha_id)
        if not 200 <= response.status_code < 300:
            raise ServiceError("HTTP %s fetching checks for alpha %s"
                               % (response.status_code, alpha_id))
        retry_after = _retry_after_seconds(response, cap_error_wait=False)
        if retry_after > 0:
            wait = retry_after
            if on_event and deadline is not None and time.monotonic() < deadline:
                on_event({"event": "check_pending", "alpha_id": alpha_id,
                          "retry_after_seconds": wait})
        else:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if payload is not None:
                try:
                    checks = _extract_checks(payload)
                except (AttributeError, TypeError) as exc:
                    raise ServiceError("invalid check payload for alpha %s" % alpha_id) from exc
                if checks:
                    if not platform_check_names(checks):
                        raise ServiceError("invalid check payload for alpha %s" % alpha_id)
                    if any(check["result"] is not None and not isinstance(check["result"], str)
                           for check in checks.values()):
                        raise ServiceError("invalid check result for alpha %s" % alpha_id)
                    break
            wait = 5.0
        if deadline is None or wait >= deadline - time.monotonic():
            break
        time.sleep(wait)
        if time.monotonic() >= deadline:
            break
    return {
        "alpha_id": alpha_id,
        "checks": checks,
        "self_correlation": (checks or {}).get("SELF_CORRELATION"),
        "complete": bool(checks),
        "payload_received": payload is not None,
    }


def _performance_request(alpha_id, team_id=None, competition_id=None):
    if team_id is not None and competition_id is not None:
        raise ValueError("choose either a team or a competition, not both")
    for name, value in (("alpha_id", alpha_id), ("team_id", team_id), ("competition_id", competition_id)):
        if (name == "alpha_id" or value is not None) and (not isinstance(value, str) or not value.strip()):
            raise ValueError("%s must be a nonempty identifier" % name)
    kind, identifier = ("competition", competition_id) if competition_id is not None else (
        ("team", team_id) if team_id is not None else ("user", "self"))
    collection = {"competition": "competitions", "team": "teams", "user": "users"}[kind]
    path = "/%s/%s/alphas/%s/before-and-after-performance" % (
        collection, quote(identifier, safe=""), quote(alpha_id, safe=""))
    return {"scope": {"kind": kind, "id": identifier}, "source": BRAIN_API_URL + path}


def fetch_merged_performance(s, alpha_id, team_id=None, competition_id=None,
                             max_wait_seconds=180.0, on_event=None):
    """Read the platform's merged before/after comparison without submitting an alpha.

    The provider owns the partition and metric period. A ready response is not
    proof of train-only evidence or of a causal marginal contribution.
    """
    request = _performance_request(alpha_id, team_id, competition_id)
    if (isinstance(max_wait_seconds, bool) or not isinstance(max_wait_seconds, (int, float))
            or not math.isfinite(max_wait_seconds) or max_wait_seconds < 0):
        raise ValueError("max_wait_seconds must be a finite, non-negative number")
    deadline = time.monotonic() + max_wait_seconds
    while True:
        response = s.get(request["source"], timeout=_request_timeout(deadline))
        status = response.status_code
        if status in (401, 403):
            raise AuthError("merged performance access rejected (HTTP %s)" % status)
        if status == 404:
            raise NotFound("merged performance is unavailable for the requested alpha and scope")
        retry_after = _retry_after_seconds(response, cap_error_wait=False)
        if not math.isfinite(retry_after):
            raise ServiceError("merged performance returned an invalid Retry-After")
        observed = {**request, "retrieved_at": datetime.now(timezone.utc).isoformat(), "http_status": status}
        if retry_after > 0 and (200 <= status < 300 or status in (429, 503)):
            if retry_after >= deadline - time.monotonic():
                return {**observed, "complete": False, "payload": None,
                        "next_poll_after_seconds": retry_after}
            if on_event:
                on_event("performance_pending", {"alpha_id": alpha_id, "retry_after_seconds": retry_after})
            sleep(retry_after)
            if time.monotonic() >= deadline:
                return {**observed, "complete": False, "payload": None,
                        "next_poll_after_seconds": 0.0}
            continue
        if status == 429:
            raise RateLimited("merged performance read was rate limited")
        if not 200 <= status < 300:
            raise ServiceError("HTTP %s fetching merged performance" % status)
        try:
            payload = response.json()
            json.dumps(payload, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ServiceError("merged performance response was not finite JSON") from exc
        stats = payload.get("stats") if isinstance(payload, dict) else None
        if not isinstance(stats, dict) or any(
            side not in stats or (stats[side] is not None and not isinstance(stats[side], dict))
            for side in ("before", "after")
        ):
            raise ServiceError("merged performance response has no usable before/after statistics object")
        return {**observed, "complete": True, "payload": payload, "next_poll_after_seconds": None}


def _recordset_paths(alpha_id, names):
    if not isinstance(alpha_id, str) or not re.fullmatch(r"[A-Za-z0-9]+", alpha_id):
        raise ValueError("recordset reads require a BRAIN alpha ID")
    if any(not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) for name in names):
        raise ValueError("invalid recordset name; use e.g. yearly-stats")
    base = f"{BRAIN_API_URL}/alphas/{alpha_id}/recordsets"
    return base, {name: f"{base}/{name}" for name in names}


def _read_recordset(s, url, deadline, *, one_shot=False):
    """Read a native response within the shared wait budget or in one attempt."""
    from requests.exceptions import RequestException

    result = {"source": url, "state": "unavailable", "payload": None}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 and not one_shot:
            return {**result, "reason": "wait_budget_exhausted"}
        timeout = 12.0 if one_shot else min(12.0, remaining)
        try:
            response = s.get(url, timeout=timeout, allow_redirects=False)
        except RequestException:
            return {**result, "reason": "transport_error"}
        status = response.status_code
        result.update(http_status=status, retrieved_at=cache_store.utc_now().isoformat())
        if status == 401:
            raise AuthError("BRAIN rejected the session while reading alpha recordsets")
        if status in (403, 404):
            return {**result, "reason": "access_denied" if status == 403 else "not_found"}
        delay = _retry_after_seconds(response, cap_error_wait=False)
        if not math.isfinite(delay):
            return {**result, "reason": "invalid_retry_after"}
        if delay > 0 and (200 <= status < 300 or status in (429, 503)):
            if delay >= deadline - time.monotonic():
                return {**result, "state": "pending", "reason": "retry_after",
                        "next_poll_after_seconds": delay}
            sleep(delay)
            if time.monotonic() >= deadline:
                return {**result, "state": "pending", "reason": "retry_after",
                        "next_poll_after_seconds": 0.0}
            continue
        if not 200 <= status < 300:
            return {**result, "reason": "rate_limited" if status == 429 else "http_error"}
        if not response.content.strip():
            return {**result, "reason": "empty_response"}
        try:
            payload = response.json()
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError):
            return {**result, "reason": "invalid_json"}
        return {**result, "state": "available", "payload": payload}


def fetch_alpha_recordsets(s, alpha_id, names=(), max_wait_seconds=180.0):
    """Discover native series and read explicitly named recordsets.

    The discovery response and each named read have separate availability
    evidence. Schemas and rows retain platform dates and units. Completeness
    stays unknown when the response has no pagination evidence.
    """
    base, paths = _recordset_paths(alpha_id, names)
    if (type(max_wait_seconds) not in (int, float) or not math.isfinite(max_wait_seconds)
            or max_wait_seconds < 0):
        raise ValueError("max_wait_seconds must be a finite, non-negative number")
    deadline = time.monotonic() + max_wait_seconds
    one_shot = max_wait_seconds == 0
    discovery = _read_recordset(s, base, deadline, one_shot=one_shot)
    payload = discovery.pop("payload")
    rows = payload if isinstance(payload, list) else (payload.get("results") if isinstance(payload, dict) else None)
    discovery["complete"] = None
    if isinstance(rows, list) and all(isinstance(row, dict) and isinstance(row.get("name"), str)
                                     and row["name"] for row in rows):
        discovery["available"] = rows
        if isinstance(payload, list):
            discovery["complete"] = True
        elif payload.get("next"):
            discovery["complete"] = False
        elif type(payload.get("count")) is int:
            discovery["complete"] = payload["count"] == len(rows)
    else:
        if discovery["state"] == "available":
            discovery.update(state="unavailable", reason="missing_catalog" if not payload else "invalid_catalog")
        discovery["available"] = None
    advertised = discovery["available"]
    advertised_names = {row["name"] for row in advertised or []}
    datasets = {}
    for name, url in paths.items():
        # A catalog can omit an available series, including yearly statistics.
        entry = _read_recordset(s, url, deadline, one_shot=one_shot)
        payload = entry.pop("payload")
        entry.update(advertised=True if name in advertised_names else False if discovery["complete"] else None,
                     schema=None, records=None, records_returned=None, reported_count=None, complete=None)
        if entry["state"] == "available":
            if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
                entry.update(state="unavailable", reason="missing_records" if not payload else "invalid_records")
            else:
                records = payload["records"]
                count = payload.get("count")
                count = count if type(count) is int and count >= 0 else None
                complete = False if payload.get("next") else (
                    count == len(records) if count is not None else None)
                entry.update(schema=payload.get("schema"), records=records, records_returned=len(records),
                             reported_count=count, complete=complete)
        datasets[name] = entry
    return {"alpha_id": alpha_id, "train_only_verified": False,
            "discovery": discovery, "series": datasets}


_PNL_EVIDENCE_FORMAT = "worldquantuser.pnl-evidence"


def _pnl_digest(series):
    return hashlib.sha256(json.dumps(series, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _matching_pnl_evidence(metadata, alpha_id, digest):
    for candidate in (metadata, metadata.get("previous")):
        if (isinstance(candidate, dict) and candidate.get("format") == _PNL_EVIDENCE_FORMAT
                and candidate.get("alpha_id") == alpha_id and candidate.get("data_sha256") == digest):
            return candidate
    return None


def pnl_cache_status(cache_dir, alpha_id, series=None):
    """Describe cache evidence without assuming a legacy file was fully fetched."""
    path = Path(cache_dir) / (alpha_id + ".json")
    metadata_path = Path(cache_dir) / ".meta" / (alpha_id + ".json")
    evidence = {"source": "cache", "path": str(path), "metadata_path": str(metadata_path),
                "complete": None, "as_of": None, "error": None, "points": 0}
    if series is None:
        series, data_error = cache_store.load_json(path)
        if data_error:
            evidence["error"] = data_error
            return evidence
    if not isinstance(series, dict):
        evidence.update(complete=False, error="invalid_pnl_cache")
        return evidence
    evidence["points"] = len(series)
    metadata, evidence["error"] = cache_store.load_json(metadata_path)
    if not isinstance(metadata, dict) or metadata.get("format") != _PNL_EVIDENCE_FORMAT:
        return evidence
    try:
        digest = _pnl_digest(series)
    except (TypeError, ValueError):
        evidence.update(complete=False, error="invalid_pnl_values")
        return evidence
    metadata = _matching_pnl_evidence(metadata, alpha_id, digest)
    if metadata is None:
        evidence.update(complete=False, error="pnl_metadata_mismatch")
        return evidence
    evidence.update({key: metadata.get(key) for key in
                     ("complete", "as_of", "pages", "start", "end", "missing_values")})
    evidence["error"] = None
    return evidence


def _write_pnl_cache(path, metadata_path, series, metadata, deadline):
    """Stage the next digest while retaining proof for the still-visible data file."""
    lock_timeout = max(0, min(1, deadline - time.monotonic()))
    with cache_store.file_lock(metadata_path.with_suffix(".lock"), timeout=lock_timeout):
        prior, _ = cache_store.load_json(metadata_path)
        if isinstance(prior, dict) and prior.get("format") == _PNL_EVIDENCE_FORMAT:
            current, _ = cache_store.load_json(path)
            if isinstance(current, dict):
                try:
                    match = _matching_pnl_evidence(prior, metadata["alpha_id"], _pnl_digest(current))
                except (TypeError, ValueError):
                    match = None
                if match is not None and match["data_sha256"] != metadata["data_sha256"]:
                    metadata = {**metadata, "previous": {key: value for key, value in match.items()
                                                         if key != "previous"}}
        cache_store.write_json_atomic(metadata_path, metadata)
        cache_store.write_json_atomic(path, series)


def fetch_daily_pnl(s, alpha_id, cache_dir=None, max_age_hours=24.0,
                    page_limit=10000, attempts=3, report=None, max_wait_seconds=180.0, *, deadline=None):
    """Read daily PnL; never return or cache a silently truncated recordset.

    Raw cache files remain date/value maps. A hash-bound metadata sidecar records
    successful pagination; older cache files have unknown completeness until refreshed.
    Retry-After takes precedence over a nonempty body. The total wait is bounded;
    pending pages raise RateLimited without replacing complete cached evidence.
    """
    if not isinstance(page_limit, int) or isinstance(page_limit, bool) or page_limit < 1:
        raise ValueError("page_limit must be a positive integer")
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    if (isinstance(max_wait_seconds, bool) or not isinstance(max_wait_seconds, (int, float))
            or not math.isfinite(max_wait_seconds) or max_wait_seconds < 0):
        raise ValueError("max_wait_seconds must be a finite, non-negative number")
    evidence = {"source": "platform", "complete": False, "pages": 0,
                "points": 0, "as_of": None, "error": None, "missing_values": 0}

    def publish():
        if isinstance(report, dict):
            report.clear()
            report.update(evidence)

    if cache_dir and max_age_hours > 0:
        path = Path(cache_dir) / (alpha_id + ".json")
        try:
            with path.open(encoding="utf-8") as handle:
                age = time.time() - os.fstat(handle.fileno()).st_mtime
                cached = json.load(handle) if age < max_age_hours * 3600 else None
            if isinstance(cached, dict) and cached and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) for value in cached.values()
            ):
                evidence = pnl_cache_status(cache_dir, alpha_id, cached)
                if evidence["complete"] is True:
                    publish()
                    return {key: float(value) for key, value in cached.items()}
        except (OSError, ValueError, TypeError):
            pass

    evidence = {"source": "platform", "complete": False, "pages": 0,
                "points": 0, "as_of": None, "error": None, "missing_values": 0}
    from requests.exceptions import RequestException

    relative_deadline = time.monotonic() + max_wait_seconds
    deadline = relative_deadline if deadline is None else min(deadline, relative_deadline)
    out = {}
    offset = 0
    total = None
    seen_pages = set()
    try:
        while True:
            url = BRAIN_API_URL + "/alphas/" + alpha_id + "/recordsets/daily-pnl"
            for attempt in range(attempts):
                if time.monotonic() >= deadline:
                    raise RateLimited("daily PnL read exceeded its wait budget")
                try:
                    response = s.get(url, params={"limit": page_limit, "offset": offset},
                                     timeout=_request_timeout(deadline))
                except RequestException as exc:
                    if (_transient_read_error(exc)
                            and _wait_for_read_retry(attempt + 1, attempts, deadline)):
                        continue
                    raise
                if response.status_code in (401, 403):
                    raise AuthError("session rejected while reading daily PnL")
                if response.status_code == 404:
                    raise NotFound("daily PnL is unavailable for %s" % alpha_id)
                retry_after = _retry_after_seconds(response, cap_error_wait=False)
                if not math.isfinite(retry_after):
                    raise ServiceError("daily PnL returned an invalid Retry-After")
                if retry_after > 0:
                    evidence["retry_after_seconds"] = retry_after
                    if attempt + 1 == attempts or retry_after >= deadline - time.monotonic():
                        raise RateLimited("daily PnL is not ready; retry after the reported delay")
                    time.sleep(retry_after)
                    continue
                if response.status_code == 200 and response.text.strip():
                    evidence.pop("retry_after_seconds", None)
                    break
                if attempt + 1 < attempts:
                    if 3 >= deadline - time.monotonic():
                        raise RateLimited("daily PnL read exceeded its wait budget")
                    time.sleep(3)
            if response.status_code == 429:
                raise RateLimited("daily PnL read was rate limited")
            if response.status_code != 200 or not response.text.strip():
                raise ServiceError("daily PnL page unavailable (HTTP %s)" % response.status_code)
            try:
                payload = response.json()
            except ValueError as exc:
                raise ServiceError("daily PnL page was not JSON") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
                raise ServiceError("daily PnL page has no record array")
            records = payload["records"]
            evidence["pages"] += 1
            count = payload.get("count")
            if count is not None:
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ServiceError("daily PnL page has an invalid count")
                if total is not None and total != count:
                    raise ServiceError("daily PnL changed during pagination")
                total = count
            more = bool(payload.get("next"))
            encoded_page = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
            signature = hashlib.sha256(encoded_page).digest()
            if records and signature in seen_pages:
                raise ServiceError("daily PnL repeated a page")
            seen_pages.add(signature)
            for row in records:
                if not isinstance(row, (list, tuple)) or len(row) < 2 or not isinstance(row[0], str):
                    raise ServiceError("daily PnL contains a malformed record")
                value = row[1]
                if value is None:
                    evidence["missing_values"] += 1
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ServiceError("daily PnL contains a non-finite or nonnumeric value")
                if row[0] in out and out[row[0]] != float(value):
                    raise ServiceError("daily PnL changed within an overlapping page")
                out[row[0]] = float(value)
            offset += len(records)
            evidence["points"] = len(out)
            if total is not None:
                if offset > total:
                    raise ServiceError("daily PnL exceeded its declared count")
                if offset == total:
                    break
                if not records:
                    raise ServiceError("daily PnL ended before its declared count")
            elif not more and len(records) < page_limit:
                break
            elif not records:
                raise ServiceError("daily PnL pagination did not advance")
    except RequestException as exc:
        evidence["error"] = "SERVICE_ERROR"
        publish()
        raise ServiceError("daily PnL transport failed: %s" % type(exc).__name__) from exc
    except BrainError as exc:
        evidence["error"] = exc.code
        publish()
        raise

    evidence.update(complete=True, as_of=datetime.now(timezone.utc).isoformat(),
                    start=min(out) if out else None, end=max(out) if out else None)
    if cache_dir:
        path = Path(cache_dir) / (alpha_id + ".json")
        metadata_path = Path(cache_dir) / ".meta" / (alpha_id + ".json")
        metadata = {**evidence, "format": _PNL_EVIDENCE_FORMAT, "version": 1,
                    "alpha_id": alpha_id, "data_sha256": _pnl_digest(out)}
        try:
            _write_pnl_cache(path, metadata_path, out, metadata, deadline)
            evidence.update(path=str(path), metadata_path=str(metadata_path), cached=True)
        except (OSError, cache_store.CacheError) as exc:
            evidence.update(cached=False, cache_error=type(exc).__name__)
    publish()
    return out


def read_alpha_submission_status(s, alpha_id, *, max_wait_seconds=30.0):
    """Read an earlier submission's outcome without issuing another POST.

    An UNSUBMITTED alpha snapshot is not a rejection receipt: evaluation may
    still be running. Only a ready submission endpoint with resolved failing
    checks proves that this evaluation ended in rejection.
    """
    from check_evidence import summarize_checks
    from requests.exceptions import RequestException

    if not valid_simulation_id(alpha_id):
        raise ValueError("invalid Alpha ID")
    report = {"state": "unknown", "source": "submission_status", "alpha_id": alpha_id}
    if max_wait_seconds is not None and max_wait_seconds <= 0:
        return {**report, "reason": "read_budget_exhausted"}
    deadline = time.monotonic() + min(30, max_wait_seconds or 30)
    try:
        response = s.get(BRAIN_API_URL + "/alphas/" + alpha_id + "/submit",
                         timeout=_request_timeout(deadline), allow_redirects=False)
    except RequestException as exc:
        raise ReadInterrupted("submission status read failed (%s)" % type(exc).__name__) from exc
    status = response.status_code
    report["status_code"] = status
    if status in (401, 403):
        raise AuthError("BRAIN denied the submission status read")
    delay = _retry_after_seconds(response, cap_error_wait=False)
    if not math.isfinite(delay):
        return {**report, "reason": "invalid_retry_after"}
    if delay > 0:
        return {**report, "state": "pending", "reason": "platform_retry_after",
                "retry_at": time.time() + delay}
    if status not in (200, 400):
        return {**report, "reason": "submission_status_unavailable"}
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not isinstance(payload, dict) or payload.get("id") not in (None, alpha_id):
        return {**report, "reason": "submission_response_unverified"}
    platform_status = str(payload.get("status") or "").upper()
    if platform_status in {"PENDING", "RUNNING", "SIMULATING", "CHECKING", "ACCEPTED"}:
        return {**report, "state": "pending", "reason": "platform_evaluating"}
    if payload.get("is") is not None and not isinstance(payload["is"], dict):
        return {**report, "reason": "submission_response_unverified"}
    checks = _extract_checks(payload)
    summary = summarize_checks(checks)
    report["checks"] = checks
    if summary["resolved"] and summary["failed"]:
        return {**report, "state": "rejected", "failed_checks": summary["failed"],
                "reason": "completed_submission_checks_failed"}
    return {**report, "reason": "no_terminal_submission_evidence"}


def submit_alpha(s, alpha_id, confirm=False, max_wait_seconds=180.0, on_event=None):
    """Submit one alpha to its competition, or report what would happen.

    BRAIN exposes no separate "test submission": the pre-submission test *is*
    ``GET /alphas/{id}/check`` (see :func:`fetch_alpha_checks`), and the only
    write action is ``POST /alphas/{id}/submit``. Submission is irreversible
    from a user's point of view, so ``confirm`` must be True to send it;
    otherwise a dry-run document describing the current checks is returned.
    """
    if confirm and not agent_submit_allowed():
        raise SubmissionDisabled(
            "submission is disabled: %s is not 'true' in the environment or .env, so an "
            "agent may not submit alphas on this account. Backtesting, checks and "
            "self-correlation analysis remain available; set %s=true to allow real "
            "submissions." % (SUBMIT_UNLOCK_ENV, SUBMIT_UNLOCK_ENV)
        )
    current = fetch_alpha_metrics(s, alpha_id)
    status = current.get("status")
    document = {
        "alpha_id": alpha_id,
        "status": status,
        "submitted": False,
        "post_attempted": False,
    }
    if status in ("ACTIVE", "SUBMITTED"):
        document.update({
            "checks": _extract_checks(current) or {},
            "submission": "CONFIRMED",
            "submitted": True,
            "date_submitted": current.get("dateSubmitted"),
            "reason": "already %s" % status,
        })
        return document
    if is_quick_simulation(current):
        document.update(
            submission="NOT_SENT",
            post_attempted=False,
            reason="Quick simulation results are for screening; rerun with simulationMode=FULL",
        )
        return document
    if not confirm:
        # the platform has no separate test-submission call: the test *is* the
        # check flow, so run it live instead of echoing the cached snapshot.
        live = fetch_alpha_checks(s, alpha_id,
                                  max_wait_seconds=max_wait_seconds,
                                  on_event=on_event)
        document["checks"] = live.get("checks")
        document["self_correlation"] = live.get("self_correlation")
        document["checks_complete"] = live.get("complete")
        document["submission"] = "NOT_SENT"
        document["reason"] = "confirmation required: rerun with --yes"
        return document
    deadline = time.monotonic() + (max_wait_seconds or 0)
    url = BRAIN_API_URL + "/alphas/" + alpha_id + "/submit"
    if on_event:
        # The caller checkpoints this boundary before transport can accept the
        # irreversible request. A failed callback prevents the POST entirely.
        on_event({"event": "submit_posting", "alpha_id": alpha_id})
    document["post_attempted"] = True
    response = s.post(url, timeout=_request_timeout(deadline))
    post_status = response.status_code
    retry_after = _retry_after_seconds(response, cap_error_wait=False)
    outcome_error = None
    if on_event:
        on_event({"event": "submit_sent", "alpha_id": alpha_id,
                  "status_code": post_status,
                  "retry_after_seconds": retry_after if math.isfinite(retry_after) else None})
    while 200 <= post_status < 300 and 0 < retry_after < deadline - time.monotonic():
        time.sleep(retry_after)
        if time.monotonic() >= deadline:
            break
        try:
            response = _get_poll_response(s, url, deadline)
        except ReadInterrupted as exc:
            outcome_error = str(exc)
            break
        if not 200 <= response.status_code < 300:
            outcome_error = "submission status read returned HTTP %s" % response.status_code
            break
        retry_after = _retry_after_seconds(response, cap_error_wait=False)
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    if post_status >= 500 or post_status == 408 or 300 <= post_status < 400:
        outcome_error = "submission request returned HTTP %s" % post_status
    if outcome_error:
        document.update(submission="UNCERTAIN", status_code=response.status_code,
                        reason=outcome_error + "; verify platform status before another attempt",
                        response=payload)
        return document

    # Four outcomes, not two. A 2xx from this endpoint means "accepted for
    # evaluation", not "submitted": the platform answers 201 with an empty body while
    # the checks are still being computed, and an alpha that then fails them stays
    # UNSUBMITTED with no error anywhere. Treating the status code as success is the
    # mistake this function used to make, and it is the same mistake that produced a
    # false "twenty submitted" report during the campaign.
    checks = (payload.get("is") or {}).get("checks") or []
    failed = [check for check in checks
              if check.get("result") not in (None, "PASS", "PENDING")]

    if post_status >= 400:
        document.update({
            "submission": "REJECTED",
            "submitted": False,
            "status_code": response.status_code,
            "checks": _extract_checks({"is": {"checks": checks}}) or {},
            "failed_checks": [check.get("name") for check in failed],
            "reason": ("refused: " + ", ".join(
                "%s=%s" % (check.get("name"), check.get("value")) for check in failed)
                if failed else "refused with HTTP %s" % response.status_code),
            "response": payload,
        })
        return document

    # A 2xx only buys an evaluation. The read-back decides.
    try:
        refreshed = fetch_alpha_metrics(s, alpha_id)
    except BrainError as exc:
        document.update({
            "submission": "UNCERTAIN",
            "submitted": False,
            "status_code": response.status_code,
            "reason": "accepted but the read-back failed (%s); re-check dateSubmitted "
                      "before treating this as submitted" % type(exc).__name__,
            "response": payload,
        })
        return document

    date_submitted = refreshed.get("dateSubmitted")
    document.update({
        "submitted": bool(date_submitted),
        "submission": "CONFIRMED" if date_submitted else "ACCEPTED",
        "status": refreshed.get("status"),
        "date_submitted": date_submitted,
        "status_code": response.status_code,
        "checks": _extract_checks(refreshed) or {},
        "response": payload,
    })
    if not date_submitted:
        document["reason"] = (
            "accepted for asynchronous evaluation but dateSubmitted is still empty; "
            "reconcile this attempt with read-only status checks before another submission")
    return document


def list_alphas(s, limit=100, offset=0, order="-is.sharpe", status="UNSUBMITTED",
                region=None, universe=None, min_sharpe=None, min_fitness=None,
                alpha_type_not="SUPER", hidden=False, search=None, *, deadline=None):
    """List the account's own alphas (``GET /users/self/alphas``) as ``list[dict]``.

    Metadata only: this never re-simulates anything and consumes no quota.
    Filters are pushed server-side, so paging a large account stays cheap.
    ``order`` uses the API syntax, e.g. ``-is.sharpe`` or ``is.fitness``.

    Careful: ``status`` defaults to ``"UNSUBMITTED"``, which **hides the
    submitted book**. Paging the default filter and then reporting "every alpha
    is unsubmitted" is a wrong conclusion that has already cost one analysis
    pass. Pass ``status="ACTIVE"`` to read what is actually submitted, or
    ``status=None`` for no filter at all.
    """
    params = {
        "limit": min(int(limit), 100),
        "offset": int(offset),
        "order": order,
        "hidden": "true" if hidden else "false",
    }
    if alpha_type_not:
        params["type!="] = alpha_type_not
    if status:
        params["status"] = status
    if region:
        params["settings.region"] = region
    if universe:
        params["settings.universe"] = universe
    if min_sharpe is not None:
        params["is.sharpe>"] = min_sharpe
    if min_fitness is not None:
        params["is.fitness>"] = min_fitness
    if search:
        params["search"] = search
    limits = {} if deadline is None else {"deadline": deadline}
    payload = request_json(s, "GET", "/users/self/alphas", params=params, **limits)
    return _as_results(payload, "alphas")


# ---------------------------------------------------------------------------
# Out-of-sample split
# ---------------------------------------------------------------------------
SPLIT_METRIC_KEYS = (
    "sharpe", "fitness", "turnover", "returns", "drawdown", "margin",
    "longCount", "shortCount", "pnl",
)


def extract_split(payload):
    """Pull BRAIN's own train/test split out of an alpha payload.

    A regular simulation always runs the platform's fixed window (this account
    reports 2019-01-01..2023-12-31; ``startDate``/``endDate`` in the settings
    are assigned by BRAIN and are silently ignored when a caller sends their
    own). The only out-of-sample evidence the API exposes is ``testPeriod``:
    the tail of that window becomes ``test`` and everything before it ``train``.

    Returns ``{"test_period", "train", "test"}`` with ``None`` blocks when the
    alpha was simulated with ``testPeriod: P0Y`` (no split exists).
    """
    if not isinstance(payload, dict):
        return None
    settings = payload.get("settings") or {}
    train = payload.get("train")
    test = payload.get("test")
    if not isinstance(train, dict) and not isinstance(test, dict):
        return None

    def block(source):
        if not isinstance(source, dict):
            return None
        picked = {k: source.get(k) for k in SPLIT_METRIC_KEYS if k in source}
        for key in ("startDate", "endDate"):
            value = source.get(key)
            if value is not None:
                picked[key] = value
        return picked or None

    return {
        "test_period": settings.get("testPeriod"),
        "train": block(train),
        "test": block(test),
    }

#: Values the platform accepts for an alpha's category. Discovered by probing the
#: API: an invalid value returns 400 without listing the choices, so the set is
#: assembled from what is accepted.
ALPHA_CATEGORIES = ("ANALYST", "FUNDAMENTAL", "PRICE_VOLUME", "SENTIMENT", "VOLUME")


def set_alpha_metadata(s, alpha_id, name=None, category=None, tags=None,
                       description=None):
    """Set an alpha's writable metadata and return the updated payload.

    ``name``, ``category`` and ``tags`` are top-level properties; ``tags`` must be a
    list, not a string.

    ``description`` is writable too, but **nested under ``regular``** - the payload is
    ``{"regular": {"description": ...}}``. This was missed for a long time on the
    strength of eighteen *top-level* spellings all answering "Unexpected property", and
    the wrong conclusion ("description does not exist") was written down here. Reading
    another project's client showed the nested shape; a PATCH with it returns 200 and the
    text reads back. The lesson is about the shape of a payload, not about the field.

    The update is verified by re-reading rather than trusting the PATCH status, for
    the same reason a submission is verified by ``dateSubmitted``: this API answers
    with success codes for requests it has only accepted.
    """
    payload = {}
    if name is not None:
        payload["name"] = name
    if category is not None:
        if category not in ALPHA_CATEGORIES:
            raise ValueError("category must be one of %s, got %r"
                             % (", ".join(ALPHA_CATEGORIES), category))
        payload["category"] = category
    if tags is not None:
        payload["tags"] = list(tags)
    if description is not None:
        payload["regular"] = {"description": str(description)}
    if not payload:
        raise ValueError("nothing to update")

    response = s.patch(BRAIN_API_URL + "/alphas/" + alpha_id, json=payload,
                       timeout=_request_timeout(None))
    if response.status_code != 200:
        raise ServiceError("metadata update failed: HTTP %d %s"
                           % (response.status_code, (response.text or "")[:200]))

    current = fetch_alpha_metrics(s, alpha_id)
    mismatched = []
    for key, value in payload.items():
        if key == "regular":
            # `regular` is a whole object on the way back: code, language, limits and the
            # description among them. Comparing objects would always fail; the only part
            # this function writes is the description.
            if (current.get("regular") or {}).get("description") != value.get("description"):
                mismatched.append("regular.description")
        elif current.get(key) != value:
            mismatched.append(key)
    if mismatched:
        raise ServiceError("metadata did not stick: %s" % ", ".join(mismatched))
    return current

def fetch_self_correlation(s, alpha_id, tries=6, max_wait=5.0):
    """Return the platform's own self-correlation view for one alpha.

    The counterpart's Sharpe determines whether the substitute route clears its
    required margin; a bare maximum correlation cannot answer that question.

    The endpoint frequently answers HTTP 200 with an *empty body* plus a Retry-After
    header. That means "not now", not "no data": reading it as absence silently
    reports an alpha as uncorrelated. The response is retried while the body is blank.
    """
    deadline = time.monotonic() + max_wait * tries
    for _ in range(tries):
        response = s.get(BRAIN_API_URL + "/alphas/" + alpha_id + "/correlations/self",
                         timeout=_request_timeout(None))
        if response.text.strip():
            payload = response.json()
            return {
                "alpha_id": alpha_id,
                "max": payload.get("max"),
                "min": payload.get("min"),
                "records": payload.get("records") or [],
            }
        if time.monotonic() > deadline:
            break
        try:
            wait = float(response.headers.get("Retry-After", 2) or 2)
        except (TypeError, ValueError):
            wait = 2.0
        time.sleep(min(wait, max_wait))
    raise ServiceError("self-correlation stayed empty after %d attempts" % tries)


def self_correlation_verdict(payload, limit=0.7, sharpe_margin=1.10):
    """Decide whether a submission is blocked, and whether the substitute route holds.

    The rule allows a correlated alpha through when its Sharpe exceeds, by the margin,
    the Sharpe of *every* alpha it correlates with above the limit - not merely the
    worst one. Checking only the maximum correlation's counterpart is the tempting
    shortcut and it is wrong in the direction that matters: a second, higher-Sharpe
    counterpart raises the bar.
    """
    records = payload.get("records") or []
    correlated = [row for row in records
                  if len(row) > 5 and isinstance(row[5], (int, float))
                  and abs(row[5]) > limit]
    if not correlated:
        # `known` separates two things that look identical in `blocked`: a real
        # measurement where everything sits under the limit, and a payload with no
        # records at all because the platform has not scored this alpha yet. Reading
        # the second as "clear" is the same mistake as reading an empty body as
        # zero correlation, one layer up -- and it is how an unscored alpha gets
        # reported as submittable.
        return {"blocked": False, "known": bool(records),
                "reason": "nothing above the correlation limit" if records
                          else "no correlation records: not scored yet",
                "counterparts": [], "sharpe_needed": None, "records": len(records)}

    sharpes = [row[6] for row in correlated if len(row) > 6 and isinstance(row[6], (int, float))]
    needed = max(sharpes) * sharpe_margin if sharpes else None
    return {
        "blocked": True,
        "known": True,
        "records": len(records),
        "reason": "self-correlation above the limit",
        "counterparts": [
            {"alpha_id": row[0], "name": row[1], "correlation": row[5],
             "sharpe": row[6] if len(row) > 6 else None,
             "fitness": row[9] if len(row) > 9 else None}
            for row in correlated
        ],
        "sharpe_needed": round(needed, 4) if needed else None,
    }
