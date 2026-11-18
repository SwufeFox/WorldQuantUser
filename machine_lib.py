"""The BRAIN transport: sessions, the metadata cache, and the simulation loop.

This module is the only place that talks HTTP to BRAIN. Everything above it --
`research_cli`, `guarded_sim`, `tools/*` -- is policy, and the split is deliberate:
a rule about *what* to simulate belongs where it can be read in ten lines, not
buried in a 2,000-line client.

What lives here, in the order a request flows through it:

* **errors** -- one ``BrainError`` family, so a caller can catch the class instead
  of matching strings. ``CaptchaRequired`` is separate because it is the one
  failure a retry makes worse.
* **credentials and the session cache** -- ``login`` / ``authenticate`` read the
  configured account and cache the session so repeated invocations do not re-login;
  BRAIN answers rapid logins with a captcha, so a cache miss is cheaper than a
  second POST. Nothing here prompts: an interactive prompt is a hang for an
  automated caller, so every failure raises with what the caller would need.
* **the metadata cache** -- datasets, operators and datafields, with a TTL, because
  the catalogue changes slowly and a field list is not worth a request per run.
  ``cached_metadata`` handles freshness and marks stale fallback explicitly.
* **JSON transport** -- ``request_json`` applies the retry policy, honours
  ``Retry-After``, and raises ``RateLimited`` only when the budget is spent. Every
  endpoint wrapper below goes through it.
* **the simulation loop** -- ``simulate_alphas`` submits in batches, polls, and
  checkpoints, so an interrupted batch resumes instead of re-spending quota. This is
  the expensive call; ``guarded_sim`` is what decides whether it may run.

What is deliberately *not* here: no linting (that is ``expression_lint`` via
``guarded_sim``), no record keeping (that is ``alpha_registry``), and no opinion
about which alphas are worth running. A transport that also judges is a transport
nobody can test.

The HTTP client is imported where it is needed. Local CLI introspection and
previews can run without it. ``pandas`` is optional for dataframe helpers.
"""

import hashlib
import json
import math
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from os import environ
from pathlib import Path
from time import sleep
from urllib.parse import quote, urljoin, urlsplit

import sys
from datetime import datetime, timezone

import cache_store
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
    """BRAIN answered with a non-2xx status we could not classify further."""

    code = "SERVICE_ERROR"


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

#: Neutralizations seen accepted by the API. **A record, not a gate.** The platform
#: owns this enum and answers for it; until 2026-11-04 a value missing here was
#: refused before a request was ever sent, which turned "we never tried STATISTICAL"
#: into "the tool cannot say the word". Well-formed values that are not on this list
#: are now *sent*, and the platform's 400 is the authority. Malformed values still
#: fail locally, because that is a typo and not a discovery.
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


def authenticate():
    """Return ``(session, info)`` for the configured BRAIN credentials.

    ``info`` is ``{"user_id", "permissions", "token_expiry"}`` -- deliberately
    not the ``/users/self`` payload, which carries account PII (email, phone).
    """
    username, password = get_credentials()
    from requests import Session

    session = Session()
    session.auth = (username, password)
    response = session.post(BRAIN_API_URL + "/authentication", timeout=_request_timeout(None))
    if response.status_code != 201:
        body = response.text[:300]
        if "captcha" in body.lower():
            raise CaptchaRequired(
                "BRAIN is asking for a captcha before issuing a session (HTTP 400)."
            )
        raise AuthError(
            "authentication failed (HTTP %s): %s" % (response.status_code, body)
        )
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


def save_session(session, info=None, path=None):
    """Persist an authenticated session's cookies so later calls can reuse it.

    Atomic and mode ``0600``. Research runs do get killed mid-write, and a
    half-written cache used to look like "no session", which in turn triggered a
    fresh login -- and repeated logins are what BRAIN answers with a captcha.
    """
    target = Path(path or session_cache_path())
    ttl = None
    expiry = (info or {}).get("token_expiry")
    if isinstance(expiry, (int, float)) and expiry > 0:
        ttl = float(expiry)
    entry = cache_store.build_entry(
        "brain-session", _session_payload(session, info), ttl=ttl, source="login")
    cache_store.write_json_atomic(target, entry, mode=0o600)
    return target


def _session_from_entry(payload, credentials):
    from requests import Session

    session = Session()
    session.auth = credentials
    session.cookies.update(payload["cookies"])
    session._brain_account_id = payload.get("user_id")
    return session


def session_status(cache_path=None):
    """Describe the cached session without touching the network.

    Used by `auth check` and `cache info` so a caller can see whether a session
    is cached, how long it has left, and why it would be re-created -- instead
    of inferring all that from a file mtime.
    """
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


def load_session(cache_path=None, use_cache=True, refresh=False, meta=None):
    """Return an authenticated session, reusing a cached cookie when still valid.

    BRAIN issues sessions that last hours but challenges repeated logins with a
    captcha, so re-authenticating on every CLI call is both slow and fragile.
    A saved cookie is reusable only for the configured account, until it
    expires. ``refresh`` forces a fresh login.

    The login itself is single-flight: concurrent callers take a file lock, and
    the ones that lose the race re-check the cache and reuse the winner's
    session instead of adding a second login. Pass ``meta`` (a dict) to receive
    ``cached`` and ``reason`` for diagnostics.
    """
    meta = meta if meta is not None else {}
    path = Path(cache_path or session_cache_path())
    reuse_cache = use_cache and not refresh
    credentials = get_credentials() if reuse_cache else None

    def _reuse(*, locked=False):
        entry = cache_store.read_entry(path)
        if entry["present"] and entry["fresh"]:
            payload = entry["payload"]
            if (entry["kind"] != "brain-session" or not isinstance(payload, dict)
                    or not isinstance(payload.get("cookies"), dict) or not payload["cookies"]):
                meta["cache_state"] = "invalid"
                return None
            if payload.get("username") != credentials[0].casefold():
                meta["cache_state"] = "account-mismatch"
                return None
            meta.update({"cached": True, "reason": "reused-cache",
                         "cache_state": "fresh",
                         "expires_in": entry.get("expires_in")})
            return _session_from_entry(payload, credentials)
        if entry["present"]:
            meta.setdefault("cache_state", "expired")
        elif entry.get("error") not in (None, "missing"):
            # Keep a sample instead of quietly dropping it: a truncated file is
            # a bug worth seeing, not a fact to hide behind a fresh login.
            meta["cache_state"] = "corrupt"
            if locked:
                quarantined = cache_store.quarantine(path)
                if quarantined:
                    meta["quarantined_to"] = quarantined
                    note("session cache was unusable (%s); moved to %s"
                         % (entry["error"], quarantined))
        else:
            meta.setdefault("cache_state", "empty")
        return None

    def _login_and_save(reason):
        session, info = authenticate()
        try:
            save_session(session, info, path)
        except OSError:
            pass  # a read-only volume must not break a working session
        meta.update({"cached": False, "reason": reason, "user_id": (info or {}).get("user_id")})
        return session

    if reuse_cache:
        reused = _reuse()
        if reused is not None:
            return reused

    timeout = float(environ.get("BRAIN_LOGIN_LOCK_TIMEOUT", "60") or 60)
    try:
        with cache_store.file_lock(research_paths.lock_path("brain-session"),
                                   timeout=timeout):
            if reuse_cache:
                reused = _reuse(locked=True)  # another caller may have refreshed the cache
                if reused is not None:
                    meta["reason"] = "waited-for-other-login"
                    return reused
            return _login_and_save("logged-in")
    except cache_store.LockTimeout as exc:
        note("session cache: %s; logging in anyway" % exc)
        return _login_and_save("lock-timeout-login")



# ---------------------------------------------------------------------------
# Metadata cache
#
# Catalog responses are quota-free but slow: the USA field catalog is 4367 rows
# over fourteen paginated calls, and one field audit plus a preflight lint reads
# most of it. Those answers also change on the order of days, not seconds. So
# they are cached with a TTL, and a stale snapshot is served when the network
# fails -- a flaky link should degrade into "older catalog", not into "the audit
# could not run at all".
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
                    use_cache=True, account=None):
    """Return ``{"data": ..., "cache": {...}}`` for one catalog query.

    ``fetch`` runs only on a miss, on an expired entry, or when ``refresh`` is
    set. If it raises :class:`BrainError` while a stale snapshot exists, the
    stale snapshot wins and the report says ``stale``/``degraded``: the catalog
    is descriptive, so yesterday's copy is worth more than no answer.

    Pass ``report`` (a dict) to also receive the cache report, so a CLI command
    can tell its caller whether the answer was cached and how old it is.
    """
    policy = _metadata_policy
    ttl = policy["max_age"] if max_age is None else float(max_age)
    do_refresh = policy["refresh"] if refresh is None else bool(refresh)
    path = metadata_cache_path(kind, params) if use_cache else None
    if path is not None and account is not None:
        path = path.with_name(f"{account}-{path.name}")
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
    age = entry["age_seconds"]

    def cached_result(error=None):
        remaining = entry["expires_in"]
        status.update(hit=True, stale=error is not None, degraded=error is not None,
                      age_seconds=round(age, 1), refreshed_at=entry["created_at"],
                      expires_in_seconds=None if remaining is None else round(remaining, 1),
                      error=error)
        return done(entry["payload"])

    if usable and not do_refresh and entry["fresh"] and age <= ttl:
        return cached_result()
    if entry.get("error") not in (None, "missing"):
        cache_store.quarantine(path)

    try:
        data = fetch()
    except BrainError as exc:
        if usable:
            note("metadata cache: serving stale %s snapshot (%s)" % (kind, exc.code))
            return cached_result(exc.code)
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
    """Read account metadata through a loader invoked on a cache miss."""

    def __init__(self, username, run):
        self.username = username
        self.run = run


def _catalog_account(username):
    name = username.strip().casefold() if isinstance(username, str) else ""
    return hashlib.sha256(name.encode()).hexdigest() if name else None


def catalog_field_cache_paths():
    """Yield current-account snapshots and legacy files of unknown provenance."""
    account = _catalog_account(configured_username())
    directory = Path(research_paths.metadata_root()) / "datafields"
    for path in sorted(directory.glob("*.json")):
        namespace, separator, _ = path.stem.partition("-")
        if not separator or namespace == account:
            yield path, bool(separator)


def _cached_account_catalog(session, kind, params, fetch, refresh, report):
    if isinstance(session, CatalogAccess):
        username, run = session.username, session.run
    else:
        auth = getattr(session, "auth", None)
        username = auth[0] if isinstance(auth, (tuple, list)) and auth else None

        def run(work):
            return work(session)

    account = _catalog_account(username)
    return cached_metadata(kind, params, lambda: run(fetch), refresh=refresh, report=report,
                           use_cache=account is not None, account=account)["data"]


def get_datasets(s, instrument_type="EQUITY", region="USA", delay=1,
                 universe="TOP3000", limit=None, refresh=None, report=None, theme=None):
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
    return _cached_account_catalog(
        s, "datasets", params,
        lambda session: _get_datasets(session, instrument_type, region, delay, universe, limit, theme),
        refresh, report,
    )


def _get_datasets(s, instrument_type="EQUITY", region="USA", delay=1,
                  universe="TOP3000", limit=None, theme=None):
    """List datasets available for a region/universe/delay as ``list[dict]``.

    Quota-free. An empty result concerns only this exact combination.
    """
    results = []
    params = {"instrumentType": instrument_type, "region": region,
              "delay": delay, "universe": universe}
    if theme is not None:
        params["theme"] = "true" if theme else "false"
    for batch in _catalog_pages(s, "/data-sets", params, keys=("data-sets",)):
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


def _catalog_pages(session, path, params=None, keys=(), page_size=None):
    """Yield complete collection pages; incomplete or repeated pages are errors."""
    offset = 0
    seen = set()
    total = None
    while True:
        query = dict(params or {})
        if page_size is not None or offset:
            query.update(limit=page_size or 50, offset=offset)
        payload = request_json(session, "GET", path, params=query or None)
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


def fetch_operators(s, search="", group="", refresh=None, report=None):
    """List FASTEXPR operators as ``list[dict]`` (``name``, ``category``, ...).

    Quota-free and cached. ``search`` matches name or description, ``group``
    selects an exact category such as ``Time Series``.
    """
    operators = cached_metadata(
        "operators", {}, lambda: _fetch_operators(s), refresh=refresh, report=report)["data"]
    if group:
        operators = [o for o in operators if (o.get("category") or "") == group]
    if search:
        needle = search.lower()
        operators = [
            o for o in operators
            if needle in ((o.get("name") or "") + " " + (o.get("description") or "")).lower()
        ]
    return operators


def _fetch_operators(s, search="", group=""):
    """List FASTEXPR operators as ``list[dict]`` (``name``, ``category``, ...).

    Quota-free. ``search`` matches the name or description, ``group`` an exact
    category such as ``Time Series``.
    """
    operators = [row for batch in _catalog_pages(s, "/operators") for row in batch]
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
    return _cached_account_catalog(
        s, "datafields", params,
        lambda session: _get_datafields(session, instrument_type, region, delay, universe,
                                        dataset_id, search, field_type, limit),
        refresh, report,
    )


def _get_datafields(s, instrument_type="EQUITY", region="USA", delay=1,
                   universe="TOP3000", dataset_id="", search="", field_type=None,
                   limit=None):
    """List datafields as ``list[dict]`` (``id``, ``description``, ``type``, ...).

    Paginates over ``GET /datafields`` and stops early once ``limit`` rows are
    collected. ``field_type`` filters on the BRAIN type (MATRIX, VECTOR, GROUP).
    Quota-free.
    """
    fields = []
    params = {"instrumentType": instrument_type, "region": region, "delay": delay,
              "universe": universe, "dataset.id": dataset_id}
    for batch in _catalog_pages(s, "/data-fields", params, keys=("datafields",), page_size=50):
        for item in batch:
            if field_type and item.get("type") != field_type:
                continue
            if search and search.lower() not in (
                (item.get("id") or "") + " " + (item.get("description") or "")
            ).lower():
                continue
            fields.append(item)
            if limit and len(fields) >= limit:
                return fields
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


def _request_timeout(deadline):
    """Keep a network request within the caller's remaining wait budget."""
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


def request_json(s, method, path, params=None, json_body=None, max_retries=4, *, deadline=None):
    """Call BRAIN and return decoded JSON, translating failures into typed errors.

    The single HTTP choke point, so every caller sees the same behaviour:
    401/403 -> AuthError, 404 -> NotFound, repeated 429 -> RateLimited (with
    backoff), other errors -> requests.HTTPError.
    """
    url = path if path.startswith("http") else BRAIN_API_URL + path
    last = None
    for attempt in range(1, max_retries + 1):
        if deadline is not None and time.monotonic() >= deadline:
            raise last or ServiceError("request wait budget expired for %s %s" % (method, url))
        response = s.request(method, url, params=params, json=json_body,
                             timeout=_request_timeout(deadline))
        if response.status_code in (401, 403):
            raise AuthError(
                "BRAIN rejected the session (HTTP %s) for %s %s"
                % (response.status_code, method, url)
            )
        if response.status_code == 404:
            raise NotFound("%s %s -> 404 (unknown id or endpoint)" % (method, url))
        if response.status_code == 429:
            wait = _retry_after_seconds(response) or min(60.0, 5.0 * attempt)
            last = RateLimited("%s %s is rate limited" % (method, url))
            if attempt < max_retries and (deadline is None or wait < deadline - time.monotonic()):
                sleep(wait)
                continue
            raise last
        if not response.ok:
            raise ServiceError(
                "HTTP %s from %s %s: %s"
                % (response.status_code, method, url, response.text[:300])
            )
        try:
            return response.json()
        except ValueError:
            return {}
    raise last or RateLimited("%s %s" % (method, url))


def read_simulation_snapshot(s, simulation_id, *, deadline):
    """GET one simulation response without polling or following redirects."""
    from requests.exceptions import RequestException

    if not valid_simulation_id(simulation_id):
        raise ValueError("invalid simulation id")
    if time.monotonic() >= deadline:
        raise ServiceError("simulation inspection wait budget expired")
    try:
        response = s.get(f"{BRAIN_API_URL}/simulations/{simulation_id}",
                         timeout=_request_timeout(deadline), allow_redirects=False)
    except RequestException as exc:
        raise ServiceError("simulation inspection request failed") from exc
    status = response.status_code
    if status in (401, 403):
        raise AuthError("BRAIN denied access to this simulation")
    if status == 404:
        raise NotFound("simulation not found")
    if status == 429:
        raise RateLimited("simulation inspection rate limited")
    if status != 200:
        raise ServiceError(f"simulation inspection returned HTTP {status}")
    seconds = _retry_after_seconds(response, cap_error_wait=False)
    retry_after = seconds if math.isfinite(seconds) and seconds > 0 else None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        if retry_after is None:
            raise ServiceError("simulation inspection returned an invalid response")
        payload = {}
    return payload, retry_after


def poll_simulation(s, progress_url, on_event=None, deadline=None):
    """Poll a simulation until completion or an optional monotonic deadline.

    If BRAIN's ``Retry-After`` would run past ``deadline``, return a resumable
    ``PENDING`` payload instead of sleeping for hours. The progress URL is kept
    so callers can checkpoint it and continue in a later invocation.
    """
    progress_url = _validated_progress_url(progress_url)
    while True:
        if deadline is not None and deadline - time.monotonic() <= 0:
            return {
                "status": "PENDING",
                "progress_url": progress_url,
                "next_poll_after_seconds": 0,
            }
        progress = s.get(progress_url, timeout=_request_timeout(deadline))
        if progress.status_code in (401, 403):
            raise AuthError(
                "session rejected while polling (HTTP %s)" % progress.status_code
            )
        retry_after = _retry_after_seconds(progress, cap_error_wait=False)
        if progress.status_code in (429, 503) and retry_after:
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
                if retry_after >= remaining:
                    return {
                        "status": "PENDING",
                        "progress_url": progress_url,
                        "next_poll_after_seconds": retry_after,
                    }
            if on_event:
                on_event("polling", {"retry_after": retry_after})
            sleep(retry_after)
            continue
        progress.raise_for_status()
        if not retry_after:
            break
        if deadline is not None:
            remaining = max(0.0, deadline - time.monotonic())
            if retry_after >= remaining:
                return {
                    "status": "PENDING",
                    "progress_url": progress_url,
                    "next_poll_after_seconds": retry_after,
                }
        if on_event:
            on_event("polling", {"retry_after": retry_after})
        sleep(retry_after)
    try:
        return progress.json()
    except ValueError:
        raise SimulationError(
            "progress endpoint returned non-JSON: %s" % progress.text[:300]
        )


def submit_single_simulation(
    s, simulation_data, max_retries=5, on_event=None, deadline=None
):
    """POST one single-alpha simulation and return its progress URL.

    Daily-quota headers are retained separately from transient HTTP throttling.
    Confirmed exhaustion defers new work; accepted jobs remain recoverable.
    """
    import simulation_capabilities
    import simulation_slots

    account = simulation_capabilities.account_provenance(s)
    for attempt in range(1, max_retries + 1):
        if deadline is not None and deadline - time.monotonic() <= 0:
            raise RateLimited("submission deferred by the run time limit")
        quota = simulation_slots.quota_snapshot(account)
        if quota["blocked"]:
            if on_event:
                on_event("quota", quota)
            raise RateLimited("daily simulation quota exhausted; resume after the reported reset")
        response = s.post(
            BRAIN_API_URL + "/simulations", json=simulation_data,
            timeout=_request_timeout(deadline),
        )
        observation = (simulation_slots.observe_quota(account, response.headers)
                       if response.status_code not in (401, 403) else None)
        if observation is not None and on_event:
            on_event("quota", observation)
        if response.status_code == 429 and observation is not None and observation["remaining"] == 0:
            raise RateLimited("daily simulation quota exhausted; resume after the reported reset")
        if response.status_code == 429:
            wait = _retry_after_seconds(response, cap_error_wait=False) or min(
                60.0, 5.0 * attempt
            )
            if on_event:
                on_event(
                    "rate_limited",
                    {"attempt": attempt, "wait_seconds": wait, "expression":
                    simulation_data.get("regular")},
                )
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
    while True:
        if deadline is not None and deadline - time.monotonic() <= 0:
            return {"_deferred": True, "next_poll_after_seconds": 0}
        response = s.get(
            BRAIN_API_URL + "/alphas/" + alpha_id,
            timeout=_request_timeout(deadline),
        )
        if response.status_code == 404:
            raise NotFound("alpha %s not found" % alpha_id)
        retry_after = _retry_after_seconds(response, cap_error_wait=False)
        if response.status_code in (429, 503) and retry_after:
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
                if retry_after >= remaining:
                    return {"_deferred": True, "next_poll_after_seconds": retry_after}
            if on_event:
                on_event("polling", {"retry_after": retry_after})
            sleep(retry_after)
            continue
        response.raise_for_status()
        if not retry_after:
            break
        if deadline is not None:
            remaining = max(0.0, deadline - time.monotonic())
            if retry_after >= remaining:
                return {"_deferred": True, "next_poll_after_seconds": retry_after}
        if on_event:
            on_event("polling", {"retry_after": retry_after})
        sleep(retry_after)
    try:
        return response.json()
    except ValueError:
        raise SimulationError(
            "alpha endpoint returned non-JSON: %s" % response.text[:300]
        )


def normalize_setting_keys(overrides=None):
    """Canonicalize known key spellings within one input layer, without defaults.

    This must run before merging files, arms or CLI flags: spelling must not
    decide which layer wins. Unknown platform extensions remain untouched.
    """
    canonical_keys = {key.replace("_", "").casefold(): key for key in DEFAULT_SETTINGS}
    settings, origins = {}, {}
    for key, value in (overrides or {}).items():
        canonical = canonical_keys.get(str(key).replace("_", "").casefold(), key)
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


def _single_simulation_data(expression, decay, settings):
    """Build one ``POST /simulations`` payload from a resolved settings dict."""
    body = regular_wire_settings(settings)
    body["decay"] = decay
    return {"type": "REGULAR", "settings": body, "regular": expression}


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
    initial_records=None,
    labels=None,
    families=None,
    candidates=None,
    concurrency=None,
    capability_snapshot=None,
    scheduling_report=None,
):
    """Simulate a list of alphas and return one record per expression.

    Each record is ``{"expression", "decay", "alpha_id", "status", "metrics",
    "checks", "error"}``. ``metrics`` is the in-sample block from
    ``GET /alphas/{id}`` for completed runs; ``checks`` maps platform check
    names to ``{"result", "limit", "value"}``. A failed expression is recorded
    with a populated ``error`` instead of aborting the run. When
    ``max_wait_seconds`` is set, unfinished work is returned as ``PENDING`` (or
    ``QUEUED`` if it has not yet been submitted) instead of keeping the caller
    blocked. Pass those records back as ``initial_records`` to resume without
    resubmitting accepted jobs.

    ``dry_run`` validates and builds the payloads but never touches the network
    (and requires no credentials). ``on_event(kind, payload)`` receives
    ``submitted`` / ``polling`` / ``quota`` / ``rate_limited`` / ``completed`` events.
    Feeding it the result of ``resolve_settings`` is the caller's job.
    ``concurrency`` is a positive local budget, clamped only by verified account
    capacity. A full shared workspace journal leaves work queued until the
    current wave can release a slot.
    Reservations survive interruptions and are
    reconciled by ``simulation_slots.reconcile_record``; other workspaces and
    unidentified accounts cannot share these reservations.
    """
    settings = resolve_settings(settings)
    pairs = normalize_alphas(alphas, settings)
    roles = [item.get("role") if isinstance(item, dict) else None for item in alphas]
    if labels is None:
        labels = alpha_labels(alphas, len(pairs))
    elif len(labels) != len(pairs):
        labels = [None] * len(pairs)
    if families is None:
        families = alpha_families(alphas, len(pairs))
    if candidates is None:
        candidates = alpha_candidate_ids(alphas, len(pairs))
    elif len(families) != len(pairs):
        families = [None] * len(pairs)

    import simulation_capabilities
    import simulation_slots

    if capability_snapshot is None and not dry_run:
        capability_snapshot = simulation_capabilities.cached()
    plan = concurrency_plan(None if dry_run else s, concurrency,
                            (capability_snapshot or {}) if dry_run else capability_snapshot)
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
                "payload": _single_simulation_data(expression, decay, settings),
            }
            for index, (expression, decay) in enumerate(pairs)
        ]

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


    for record in records:
        simulation_slots.reconcile_record(record, plan["account"])

    def shared_state():
        state = simulation_slots.snapshot(plan["account"], plan["effective"])
        if scheduling_report is not None:
            scheduling_report["shared_slots"] = state
        return state

    shared_state()

    deadline = (
        time.monotonic() + max_wait_seconds
        if max_wait_seconds is not None
        else None
    )

    def emit(kind, index, payload=None):
        if on_event:
            data = dict(payload or {})
            data["index"] = index
            on_event(kind, data)

    quota_exhausted = False

    def submission_event(kind, index, payload):
        nonlocal quota_exhausted
        if kind == "quota":
            payload = simulation_slots.quota_snapshot(plan["account"], observation=payload)
            quota_exhausted = quota_exhausted or payload["remaining"] == 0
            if scheduling_report is not None:
                scheduling_report["quota"] = payload
        emit(kind, index, payload)

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
        except AuthError as exc:
            record["status"] = "PENDING"
            record["metrics_pending"] = True
            record["error"] = str(exc)
            emit("pending", index, record)
            return
        except Exception as exc:
            record["error"] = "metrics fetch failed: %s" % exc
        else:
            if payload.get("_deferred"):
                wait = max(0.0, float(payload.get("next_poll_after_seconds") or 0))
                record["status"] = "PENDING"
                record["metrics_pending"] = True
                record["poll_after_seconds"] = wait
                if wait:
                    record["next_poll_at"] = time.time() + wait
                emit("pending", index, record)
                return
            record["metrics"] = payload.get("is")
            record["checks"] = _extract_checks(payload)
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
            shared_state()
            if progress.get("status") in simulation_slots.TERMINAL_FAILURE_STATUSES or progress.get("message"):
                record["error"] = progress.get("message") or "simulation ended with %s" % record["status"]
                if not is_python_language(settings.get("language")):
                    note_operator_rejection(record.get("expression"), record["error"])
            if record["alpha_id"]:
                record["simulation_status"] = record["status"]
                record["metrics_pending"] = True
            record.pop("progress_url", None)
            if record.get("metrics_pending"):
                fetch_metrics_record(index)
            else:
                record.pop("next_poll_at", None)
                record.pop("poll_after_seconds", None)
                emit("completed", index, record)
        except AuthError as exc:
            record["status"] = "PENDING"
            record["error"] = str(exc)
            record["progress_url"] = progress_url
            emit("pending", index, record)
        except Exception as exc:  # preserve the accepted job for a later resume
            record["status"] = "FAILED"
            record["error"] = "poll failed: %s" % exc
            record["progress_url"] = progress_url
            emit("failed", index, record)

    def poll_many(indices):
        """Poll several simulations at once.

        Serial polling makes the slowest arm set the pace for every arm behind it, and an
        arm whose progress endpoint keeps answering with an empty body blocks the batch
        outright - which is exactly what happened to a batch of three whose other two had
        already finished on the platform.

        Each poll writes only its own record, so the workers do not share state; the event
        sink writes whole lines, which is what it is for.
        """
        indices = [index for index in indices if records[index].get("progress_url")
                   or records[index].get("metrics_pending")]
        if not indices:
            return
        if len(indices) == 1:
            index = indices[0]
            if records[index].get("progress_url"):
                poll_record(index)
            elif records[index].get("metrics_pending"):
                fetch_metrics_record(index)
            return
        workers = min(len(indices), max(2, batch_size))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for index in indices:
                if records[index].get("progress_url"):
                    futures.append(pool.submit(poll_record, index))
                elif records[index].get("metrics_pending"):
                    futures.append(pool.submit(fetch_metrics_record, index))
            for future in futures:
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - one arm failing is not the batch
                    note("poll worker raised %s: %s" % (type(exc).__name__, exc))

    # Resume accepted jobs first. An unresolved job still occupies one of the
    # platform's limited active slots, so do not submit the next batch yet.
    poll_many(range(len(records)))
    shared_state()
    if any(
        record.get("progress_url") or record.get("metrics_pending")
        or record.get("status") == "UNKNOWN_SUBMISSION"
        for record in records
    ):
        return records

    if not plan["effective"]:
        return records

    snapshot_account = (capability_snapshot or {}).get("account") or {}
    settings_snapshot = capability_snapshot if (
        plan["account"].get("fingerprint")
        and plan["account"]["fingerprint"] == snapshot_account.get("fingerprint")
    ) else {}
    settings_checks = {}

    start = 0
    while start < len(pairs):
        end = min(len(pairs), start + batch_size)
        submitted = []
        submission_blocked = False
        contended_index = None
        for index in range(start, end):
            expression, decay = pairs[index]
            record = {
                **records[index],
            }
            if record.get("status") not in {"QUEUED", None}:
                continue
            if record.get("submission_attempted"):
                record["status"] = "UNKNOWN_SUBMISSION"
                record["error"] = "submission outcome needs reconciliation before another POST"
                records[index] = record
                emit("pending", index, record)
                return records
            if deadline is not None and deadline - time.monotonic() <= 0:
                break
            if quota_exhausted:
                submission_blocked = True
                record["error"] = "daily simulation quota exhausted; resume after the reported reset"
                records[index] = record
                emit("deferred", index, record)
                break
            if decay not in settings_checks:
                settings_checks[decay] = regular_settings_evidence(
                    {**settings, "decay": decay}, settings_snapshot)
            capability_check = settings_checks[decay]
            if capability_check["unsupported"]:
                record.update(status="FAILED", submission_attempted=False,
                              error="settings are unsupported by current account metadata",
                              settings_capabilities=capability_check)
                records[index] = record
                emit("failed", index, record)
                continue
            identity, slots = simulation_slots.reserve(plan["account"], plan["effective"])
            if scheduling_report is not None:
                scheduling_report["shared_slots"] = slots
            if identity is None and slots["available"] == 0:
                contended_index = index
                record["error"] = "shared simulation slots are occupied; reconcile or resume existing attempts"
                records[index] = record
                emit("deferred", index, record)
                break
            if identity is not None:
                record["slot_reservation"] = identity
                record["slot_reservation_state"] = "submitting"
            record["submission_attempted"] = True
            record["status"] = "UNKNOWN_SUBMISSION"
            records[index] = record
            emit("submitting", index, record)
            try:
                location = submit_single_simulation(
                    s, _single_simulation_data(expression, decay, settings),
                    on_event=lambda kind, payload, i=index: submission_event(kind, i, payload),
                    deadline=deadline,
                )
            except RateLimited as exc:
                submission_blocked = True
                simulation_slots.reconcile_record(record, plan["account"], outcome="refused")
                record["status"] = "QUEUED"
                record["submission_attempted"] = False
                record["error"] = "%s: %s" % (exc.code, exc)
                records[index] = record
                emit("deferred", index, record)
                break
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
                break
            except BrainError as exc:
                record["status"] = "UNKNOWN_SUBMISSION"
                record["error"] = "%s: %s" % (exc.code, exc)
                records[index] = record
                emit("pending", index, record)
                break
            except HTTPError as exc:
                status_code = getattr(exc.response, "status_code", None)
                refused = status_code is not None and 400 <= status_code < 500
                record["status"] = "FAILED" if refused else "UNKNOWN_SUBMISSION"
                record["error"] = "submit failed: %s" % exc
                if refused:
                    detail = getattr(exc.response, "text", "")
                    if is_python_language(settings.get("language")) and isinstance(detail, str) and detail.strip():
                        record["error"] += " (" + detail.strip()[:500] + ")"
                    simulation_slots.reconcile_record(record, plan["account"], outcome="refused")
                    record["status"] = "FAILED"
                    if not is_python_language(settings.get("language")):
                        note_operator_rejection(record.get("expression"), record["error"])
                records[index] = record
                emit("failed" if refused else "pending", index, record)
                if not refused:
                    break
                continue
            except RequestException as exc:
                record.update(status="UNKNOWN_SUBMISSION", error="submit transport failed: %s" % type(exc).__name__)
                records[index] = record
                emit("pending", index, record)
                break
            record["status"] = "PENDING"
            record["progress_url"] = location
            record["error"] = None
            simulation_slots.reconcile_record(record, plan["account"], outcome="accepted")
            records[index] = record
            emit("submitted", index, record)
            submitted.append(index)

        poll_many(submitted)
        slots = shared_state()
        if submission_blocked:
            return records
        if any(record.get("status") == "UNKNOWN_SUBMISSION" for record in records):
            break
        if any(
            records[index].get("progress_url")
            or records[index].get("metrics_pending")
            for index in submitted
        ):
            # Avoid exceeding the platform's concurrent simulation limit.
            break
        if contended_index is not None:
            # A full journal can include this wave's own reservations. After
            # polling these to completion, retry only the unsent part of the wave.
            if not submitted or slots["available"] == 0:
                break
            start = contended_index
        else:
            start = end

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
# in the catalog uses a floor (value >= limit). BRAIN sometimes serves a cached
# ``result`` beside a refreshed ``value``, so a cached PASS can contradict a value
# above the ceiling.
# ``reconcile_check`` recomputes the verdict from the number so a decision is
# never made on a stale label.
_CEILING_CHECKS = frozenset({
    "SELF_CORRELATION",
    "CONCENTRATED_WEIGHT",
    "HIGH_TURNOVER",
})


def reconcile_check(name, check):
    """Recompute a check's verdict from its value and limit.

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
    """Flatten the platform checks of an alpha payload into a name -> info map."""
    checks = {}
    for check in (alpha_payload.get("is") or {}).get("checks") or []:
        name = check.get("name")
        if not name:
            continue
        info = {key: value for key, value in check.items()
                if key not in {"name", "expected_result", "value_consistent"}}
        for key in ("result", "limit", "value"):
            info.setdefault(key, None)
        verdict = reconcile_check(name, info)
        if verdict is not None:
            info["expected_result"] = verdict["expected"]
            info["value_consistent"] = verdict["consistent"]
        checks[name] = info
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
    """
    if max_wait_seconds is not None and (not math.isfinite(max_wait_seconds) or max_wait_seconds < 0):
        raise ValueError("max_wait_seconds must be finite and non-negative")
    deadline = time.monotonic() + (max_wait_seconds or 0)
    url = BRAIN_API_URL + "/alphas/" + alpha_id + "/check"
    payload = None
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
        if retry_after:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait = retry_after if retry_after else 5.0
            if on_event:
                on_event({"event": "check_pending", "alpha_id": alpha_id,
                          "retry_after_seconds": wait})
            time.sleep(min(wait, remaining))
            continue
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if (payload or {}).get("is", {}).get("checks"):
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(5.0, remaining))
    checks = _extract_checks(payload or {})
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
    """Discover native series and read explicitly requested recordsets.

    Each result describes one native response. Completeness is unknown without
    pagination evidence, and the server's dates and units are left untouched.
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


def pnl_cache_status(cache_dir, alpha_id, series=None):
    """Describe cache evidence without assuming a legacy file was fully fetched."""
    path = Path(cache_dir) / (alpha_id + ".json")
    metadata_path = Path(cache_dir) / ".meta" / (alpha_id + ".json")
    metadata, error = cache_store.load_json(metadata_path)
    evidence = {"source": "cache", "path": str(path), "metadata_path": str(metadata_path),
                "complete": None, "as_of": None, "error": error, "points": 0}
    if series is None:
        series, data_error = cache_store.load_json(path)
        if data_error:
            evidence["error"] = data_error
            return evidence
    if not isinstance(series, dict):
        evidence.update(complete=False, error="invalid_pnl_cache")
        return evidence
    evidence["points"] = len(series)
    if not isinstance(metadata, dict) or metadata.get("format") != "worldquantuser.pnl-evidence":
        return evidence
    try:
        digest = hashlib.sha256(json.dumps(series, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode("utf-8")).hexdigest()
    except (TypeError, ValueError):
        evidence.update(complete=False, error="invalid_pnl_values")
        return evidence
    if metadata.get("alpha_id") != alpha_id or metadata.get("data_sha256") != digest:
        evidence.update(complete=False, error="pnl_metadata_mismatch")
        return evidence
    evidence.update({key: metadata.get(key) for key in
                     ("complete", "as_of", "pages", "start", "end", "missing_values")})
    evidence["error"] = None
    return evidence


def fetch_daily_pnl(s, alpha_id, cache_dir=None, max_age_hours=24.0,
                    page_limit=10000, attempts=3, report=None, max_wait_seconds=180.0):
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

    if cache_dir:
        path = Path(cache_dir) / (alpha_id + ".json")
        if path.is_file() and time.time() - path.stat().st_mtime < max_age_hours * 3600:
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
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

    deadline = time.monotonic() + max_wait_seconds
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
                response = s.get(url, params={"limit": page_limit, "offset": offset},
                                 timeout=_request_timeout(deadline))
                if response.status_code in (401, 403):
                    raise AuthError("session rejected while reading daily PnL")
                if response.status_code == 404:
                    raise NotFound("daily PnL is unavailable for %s" % alpha_id)
                retry_after = _retry_after_seconds(response, cap_error_wait=False)
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
            signature = json.dumps(records, sort_keys=True, separators=(",", ":"))
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
        metadata = {**evidence, "format": "worldquantuser.pnl-evidence", "version": 1,
                    "alpha_id": alpha_id,
                    "data_sha256": hashlib.sha256(json.dumps(out, sort_keys=True, separators=(",", ":"),
                                                              allow_nan=False).encode("utf-8")).hexdigest()}
        try:
            cache_store.write_json_atomic(path, out)
            cache_store.write_json_atomic(metadata_path, metadata)
            evidence.update(path=str(path), metadata_path=str(metadata_path), cached=True)
        except OSError as exc:
            evidence.update(cached=False, cache_error=type(exc).__name__)
    publish()
    return out


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
    response = s.post(url, timeout=_request_timeout(deadline))
    retry_after = _retry_after_seconds(response, cap_error_wait=False)
    if on_event:
        on_event({"event": "submit_sent", "alpha_id": alpha_id,
                  "status_code": response.status_code,
                  "retry_after_seconds": retry_after})
    while retry_after and time.monotonic() < deadline:
        time.sleep(min(retry_after, max(0.0, deadline - time.monotonic())))
        response = s.get(url, timeout=_request_timeout(deadline))
        retry_after = _retry_after_seconds(response, cap_error_wait=False)
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    # Four outcomes, not two. A 2xx from this endpoint means "accepted for
    # evaluation", not "submitted": the platform answers 201 with an empty body while
    # the checks are still being computed, and an alpha that then fails them stays
    # UNSUBMITTED with no error anywhere. Treating the status code as success is the
    # mistake this function used to make, and it is the same mistake that produced a
    # false "twenty submitted" report during the campaign.
    checks = (payload.get("is") or {}).get("checks") or []
    failed = [check for check in checks
              if check.get("result") not in (None, "PASS", "PENDING")]

    if response.status_code >= 400:
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
                alpha_type_not="SUPER", hidden=False, search=None):
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
    payload = request_json(s, "GET", "/users/self/alphas", params=params)
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
