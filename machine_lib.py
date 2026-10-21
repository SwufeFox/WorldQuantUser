import hashlib
import json
import math
import os
import pickle
import random
import re
import time
from collections import defaultdict
from itertools import combinations, product
from os import environ
from pathlib import Path
from time import sleep
from urllib.parse import urljoin, urlsplit

import sys
from datetime import datetime, timezone

import requests

import cache_store
import research_paths

try:  # pandas is optional: only the dataframe helpers and the notebook need it
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only when pandas is absent
    pd = None


# ---------------------------------------------------------------------------
# Diagnostics
#
# stdout is reserved for the single JSON document the CLI prints, so anything
# narrative goes to stderr. The CLI installs its own handler to emit structured
# events; a plain library user gets readable lines.
# ---------------------------------------------------------------------------
_diagnostic_handler = None


def set_diagnostic_handler(handler):
    """Route this module's diagnostics through ``handler`` (``None`` resets)."""
    global _diagnostic_handler
    _diagnostic_handler = handler


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

NEUTRALIZATIONS = ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]

#: Non-exhaustive list of common universes, used for input validation hints.
UNIVERSES = [
    "TOP3000", "TOP1000", "TOP600", "TOP500", "TOP200", "TOP100",
    "TOPSP500", "TOPSP1000", "MINVOL1M", "ILLIQUID_MINVOL1M",
]


_DOTENV_LOADED = False


def _load_dotenv(path=None):
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Runs once per process. Real environment variables always win over .env,
    so CI/shell configuration is never silently overridden.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    env_path = Path(__file__).resolve().parent / ".env" if path is None else Path(path)
    if not env_path.is_file():
        return

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


def get_credentials(prefix="BRAIN"):
    """Return (username, password) for the WorldQuant BRAIN API.

    Reads <prefix>_USERNAME / <prefix>_PASSWORD from the environment, after
    loading a .env file next to this module when present. Never hardcode
    credentials in the source tree.
    """
    _load_dotenv()
    username = environ.get(prefix + "_USERNAME", "").strip()
    password = environ.get(prefix + "_PASSWORD", "").strip()
    if not username or not password:
        raise MissingCredentials(
            "Missing credentials: set {}_USERNAME and {}_PASSWORD in the "
            "environment, or copy .env.example to .env next to machine_lib.py "
            "and fill it in.".format(prefix, prefix)
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
    session = requests.Session()
    session.auth = (username, password)
    response = session.post(BRAIN_API_URL + "/authentication")
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
    return session, info


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
        code = code.strip()
    return code, description, count


def session_cache_path():
    """Where a reused BRAIN session cookie is kept.

    Lives under the cache root (outside the git working tree, on the persistent
    volume) rather than next to this module, and is written mode ``0600``: the
    file holds live session cookies, which are credentials.
    """
    return Path(research_paths.session_path())


def legacy_session_cache_path():
    """Pre-2.0 location, kept only so an existing session can be migrated."""
    return Path(__file__).resolve().parent / ".brain_session.json"


def _session_payload(session, info):
    return {
        "cookies": requests.utils.dict_from_cookiejar(session.cookies),
        "user_id": (info or {}).get("user_id"),
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


def _migrate_legacy_session(target):
    """Move a pre-2.0 session file into the new location without re-logging in.

    The old file was a flat ``{cookies, saved_at, expiry_seconds, ...}`` record.
    Converting it is worth the few lines: dropping it would force one extra
    login, and that is exactly the request that gets captcha-challenged.
    """
    legacy = Path(legacy_session_cache_path())
    if target.exists() or not legacy.is_file():
        return None
    data, error = cache_store.load_json(legacy)
    if error or not isinstance(data, dict) or not data.get("cookies"):
        cache_store.quarantine(legacy)
        return None
    saved_at = data.get("saved_at")
    expiry = data.get("expiry_seconds")
    try:
        created = (datetime.fromtimestamp(float(saved_at), timezone.utc)
                   if saved_at else cache_store.utc_now())
    except (TypeError, ValueError, OSError):
        created = cache_store.utc_now()
    ttl = None
    if isinstance(expiry, (int, float)) and expiry > 0:
        ttl = float(expiry)
    entry = cache_store.build_entry(
        "brain-session",
        {
            "cookies": data.get("cookies") or {},
            "user_id": data.get("user_id"),
            "permissions": data.get("permissions"),
            "token_expiry": expiry,
        },
        ttl=ttl,
        source="legacy-migration",
        now=created,
    )
    try:
        cache_store.write_json_atomic(target, entry, mode=0o600)
    except OSError:
        return None
    try:
        legacy.unlink()
    except OSError:
        pass
    return str(legacy)


def _session_from_entry(payload):
    session = requests.Session()
    session.auth = get_credentials()
    session.cookies.update(payload.get("cookies") or {})
    return session


def session_status(cache_path=None):
    """Describe the cached session without touching the network.

    Used by `auth check` and `cache info` so a caller can see whether a session
    is cached, how long it has left, and why it would be re-created -- instead
    of inferring all that from a file mtime.
    """
    path = Path(cache_path or session_cache_path())
    entry = cache_store.read_entry(path)
    payload = entry.get("payload") or {}
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
        "legacy_present": Path(legacy_session_cache_path()).is_file(),
    }
    return status


def invalidate_session(cache_path=None):
    """Drop the cached session (e.g. after BRAIN rejected it). Returns True if removed."""
    return cache_store.remove(Path(cache_path or session_cache_path()))


def load_session(cache_path=None, use_cache=True, refresh=False, meta=None):
    """Return an authenticated session, reusing a cached cookie when still valid.

    BRAIN issues sessions that last hours but challenges repeated logins with a
    captcha, so re-authenticating on every CLI call is both slow and fragile.
    With ``use_cache`` a saved cookie is reused until just before it expires and
    ``refresh`` forces a fresh login.

    The login itself is single-flight: concurrent callers take a file lock, and
    the ones that lose the race re-check the cache and reuse the winner's
    session instead of adding a second login. Pass ``meta`` (a dict) to receive
    ``cached``/``migrated_from``/``reason`` for diagnostics.
    """
    meta = meta if meta is not None else {}
    path = Path(cache_path or session_cache_path())
    if cache_path is None:
        migrated = _migrate_legacy_session(path)
        if migrated:
            meta["migrated_from"] = migrated
            note("session cache: migrated %s -> %s" % (migrated, path))

    def _reuse():
        entry = cache_store.read_entry(path)
        if entry["present"] and entry["fresh"]:
            meta.update({"cached": True, "reason": "reused-cache",
                         "cache_state": "fresh",
                         "expires_in": entry.get("expires_in")})
            return _session_from_entry(entry["payload"])
        if entry["present"]:
            meta.setdefault("cache_state", "expired")
        elif entry.get("error") not in (None, "missing"):
            # Keep a sample instead of quietly dropping it: a truncated file is
            # a bug worth seeing, not a fact to hide behind a fresh login.
            quarantined = cache_store.quarantine(path)
            # setdefault: the retry inside the lock sees the file gone and would
            # otherwise relabel a corrupt cache as an empty one.
            meta.setdefault("cache_state", "corrupt")
            if quarantined:
                meta["quarantined_to"] = quarantined
                note("session cache was unusable (%s); moved to %s"
                     % (entry.get("error"), quarantined))
        else:
            meta.setdefault("cache_state", "empty")
        return None

    if use_cache and not refresh:
        reused = _reuse()
        if reused is not None:
            return reused

    timeout = float(environ.get("BRAIN_LOGIN_LOCK_TIMEOUT", "60") or 60)
    try:
        with cache_store.file_lock(research_paths.lock_path("brain-session"),
                                   timeout=timeout):
            if use_cache and not refresh:
                reused = _reuse()  # someone else may have logged in while we waited
                if reused is not None:
                    meta["reason"] = "waited-for-other-login"
                    return reused
            session, info = authenticate()
            try:
                save_session(session, info, path)
            except OSError:
                pass  # a read-only volume must not break a working session
            meta.update({"cached": False, "reason": "logged-in",
                         "user_id": (info or {}).get("user_id")})
            return session
    except cache_store.LockTimeout as exc:
        note("session cache: %s; logging in anyway" % exc)
        session, info = authenticate()
        try:
            save_session(session, info, path)
        except OSError:
            pass
        meta.update({"cached": False, "reason": "lock-timeout-login"})
        return session



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


def cached_metadata(kind, params, fetch, max_age=None, refresh=None, report=None):
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
    path = metadata_cache_path(kind, params)
    status = {
        "kind": kind,
        "path": str(path),
        "hit": False,
        "stale": False,
        "degraded": False,
        "disabled": False,
        "age_seconds": None,
        "expires_in_seconds": None,
        "refreshed_at": None,
        "error": None,
    }

    def done(data):
        if isinstance(report, dict):
            report.clear()
            report.update(status)
        return {"data": data, "cache": status}

    if not policy["enabled"] or ttl <= 0:
        status["disabled"] = True
        return done(fetch())

    entry = cache_store.read_entry(path)
    if entry["present"] and not do_refresh:
        age = entry.get("age_seconds")
        status["age_seconds"] = None if age is None else round(age, 1)
        remaining = entry.get("expires_in")
        status["expires_in_seconds"] = None if remaining is None else round(remaining, 1)
        if entry["fresh"]:
            status["hit"] = True
            return done(entry["payload"])
        status["stale"] = True
    elif entry.get("error") not in (None, "missing"):
        cache_store.quarantine(path)

    try:
        data = fetch()
    except BrainError as exc:
        if entry["present"]:
            status.update({"hit": True, "stale": True, "degraded": True,
                           "error": exc.code})
            note("metadata cache: serving stale %s snapshot (%s)" % (kind, exc.code))
            return done(entry["payload"])
        raise
    try:
        cache_store.write_json_atomic(
            path, cache_store.build_entry(kind, data, ttl=ttl, source="brain-api"))
    except OSError as exc:
        note("metadata cache: could not write %s (%s)" % (path, exc))
    status["refreshed_at"] = cache_store.read_entry(path).get("created_at")
    status["age_seconds"] = 0.0
    status["expires_in_seconds"] = ttl
    return done(data)


def get_datasets(s, instrument_type="EQUITY", region="USA", delay=1,
                 universe="TOP3000", limit=None, refresh=None, report=None):
    """List datasets available for a region/universe/delay as ``list[dict]``.

    Quota-free and cached (see :func:`cached_metadata`). Returns ``[]`` when the
    account has no data for that combination, which is what a TUTORIAL account
    sees outside USA.
    """
    params = {"instrumentType": instrument_type, "region": region,
              "delay": delay, "universe": universe, "limit": limit}
    return cached_metadata(
        "datasets", params,
        lambda: _get_datasets(s, instrument_type, region, delay, universe, limit),
        refresh=refresh, report=report,
    )["data"]


def _get_datasets(s, instrument_type="EQUITY", region="USA", delay=1,
                 universe="TOP3000", limit=None):
    """List datasets available for a region/universe/delay as ``list[dict]``.

    Quota-free. Returns ``[]`` when the account has no data for that
    combination (which is what a TUTORIAL account sees outside USA).
    """
    payload = request_json(
        s, "GET", "/data-sets",
        params={
            "instrumentType": instrument_type,
            "region": region,
            "delay": delay,
            "universe": universe,
        },
    )
    results = _as_results(payload, "data-sets")
    return results[:limit] if limit else results


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
    operators = _as_results(request_json(s, "GET", "/operators"))
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
    return cached_metadata(
        "data-categories", params,
        lambda: _fetch_data_categories(s, region, universe, delay),
        refresh=refresh, report=report,
    )["data"]


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
    return cached_metadata(
        "datafields", params,
        lambda: _get_datafields(s, instrument_type, region, delay, universe,
                                dataset_id, search, field_type, limit),
        refresh=refresh, report=report,
    )["data"]


def _get_datafields(s, instrument_type="EQUITY", region="USA", delay=1,
                   universe="TOP3000", dataset_id="", search="", field_type=None,
                   limit=None):
    """List datafields as ``list[dict]`` (``id``, ``description``, ``type``, ...).

    Paginates over ``GET /datafields`` and stops early once ``limit`` rows are
    collected. ``field_type`` filters on the BRAIN type (MATRIX, VECTOR, GROUP).
    Quota-free.
    """
    fields = []
    offset = 0
    page = 50  # BRAIN rejects larger pages with HTTP 400
    while True:
        payload = request_json(
            s, "GET", "/data-fields",
            params={
                "instrumentType": instrument_type,
                "region": region,
                "delay": delay,
                "universe": universe,
                "dataset.id": dataset_id,
                "limit": page,
                "offset": offset,
            },
        )
        results = _as_results(payload, "datafields")
        if not results:
            break
        for item in results:
            if field_type and item.get("type") != field_type:
                continue
            if search and search.lower() not in (
                (item.get("id") or "") + " " + (item.get("description") or "")
            ).lower():
                continue
            fields.append(item)
            if limit and len(fields) >= limit:
                return fields
        offset += page
        if offset >= (payload.get("count") or 0):
            break
    return fields


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
    if pd is None:
        raise RuntimeError("pandas is required for fields_to_df(); pip install pandas")
    return pd.DataFrame(fields)


def get_vec_fields(fields):

    # 请在此处添加获得权限的Vector操作符
    vec_ops = ["vec_avg", "vec_sum"]
    vec_fields = []
 
    for field in fields:
        for vec_op in vec_ops:
            if vec_op == "vec_choose":
                vec_fields.append("%s(%s, nth=-1)"%(vec_op, field))
                vec_fields.append("%s(%s, nth=0)"%(vec_op, field))
            else:
                vec_fields.append("%s(%s)"%(vec_op, field))
 
    return(vec_fields)


def process_datafields(fields):
    """Turn field metadata into directly usable FASTEXPR snippets.

    Accepts the ``list[dict]`` from :func:`get_datafields` (or a DataFrame, for
    older notebooks). MATRIX fields are winsorized + backfilled, VECTOR fields
    are converted to a scalar via ``vec_avg``.
    """
    if pd is not None and hasattr(fields, "to_dict"):
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


def ts_factory(op, field):
    output = []
    #days = [3, 5, 10, 20, 60, 120, 240]
    days = [5, 22, 66, 120, 240]
    
    for day in days:
    
        alpha = "%s(%s, %d)"%(op, field, day)
        output.append(alpha)
    
    return output

def first_order_factory(fields, ops_set):
    alpha_set = []
    #for field in fields:
    for field in fields:
        #reverse op does the work
        alpha_set.append(field)
        #alpha_set.append("-%s"%field)
        for op in ops_set:
 
            if op == "ts_percentage":
 
                alpha_set += ts_comp_factory(op, field, "percentage", [0.5])
 
 
            elif op == "ts_decay_exp_window":
 
                alpha_set += ts_comp_factory(op, field, "factor", [0.5])
 
 
            elif op == "ts_moment":
 
                alpha_set += ts_comp_factory(op, field, "k", [2, 3, 4])
 
            elif op == "ts_entropy":
 
                alpha_set += ts_comp_factory(op, field, "buckets", [10])
 
            elif op.startswith("ts_") or op == "inst_tvr":
 
                alpha_set += ts_factory(op, field)
 
            elif op.startswith("vector"):
 
                alpha_set += vector_factory(op, field)
 
            elif op == "signed_power":
 
                alpha = "%s(%s, 2)"%(op, field)
                alpha_set.append(alpha)
 
            else:
                alpha = "%s(%s)"%(op, field)
                alpha_set.append(alpha)
 
    return alpha_set

def load_task_pool_single(alpha_list, limit_of_single_simulations):

    '''
    Input:
        alpha_list : list of (alpha, decay) tuples
        limit_of_single_simulations : number of concurrent single simulations
    Output:
        task : [3 * (alpha, decay)] for 3 single simulations
        pool : [ alpha_num/3 * [3 * (alpha, decay)] ] 
    '''

    pool = [alpha_list[i:i + limit_of_single_simulations] for i in range(0, len(alpha_list), limit_of_single_simulations)]
    return pool


BRAIN_API_URL = "https://api.worldquantbrain.com"


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


def request_json(s, method, path, params=None, json_body=None, max_retries=4):
    """Call BRAIN and return decoded JSON, translating failures into typed errors.

    The single HTTP choke point, so every caller sees the same behaviour:
    401/403 -> AuthError, 404 -> NotFound, repeated 429 -> RateLimited (with
    backoff), other errors -> requests.HTTPError.
    """
    url = path if path.startswith("http") else BRAIN_API_URL + path
    last = None
    for attempt in range(1, max_retries + 1):
        response = s.request(method, url, params=params, json=json_body)
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
            if attempt < max_retries:
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

    BRAIN applies a per-account submission rate limit (HTTP 429) and exposes no
    queue to inspect, so 429 is absorbed here with the advertised backoff
    instead of reaching the caller.
    """
    for attempt in range(1, max_retries + 1):
        if deadline is not None and deadline - time.monotonic() <= 0:
            raise RateLimited("submission deferred by the run time limit")
        response = s.post(
            BRAIN_API_URL + "/simulations", json=simulation_data,
            timeout=_request_timeout(deadline),
        )
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


def resolve_settings(overrides=None):
    """Merge ``overrides`` onto :data:`DEFAULT_SETTINGS` and validate the result.

    Returns a fresh dict. Raises ``ValueError`` on an unknown region,
    neutralization or delay so callers fail before spending quota.
    """
    settings = dict(DEFAULT_SETTINGS)
    for key, value in (overrides or {}).items():
        if value is not None:
            settings[key] = value
    if settings["region"] not in REGIONS:
        raise ValueError(
            "unknown region %r (known: %s)" % (settings["region"], ", ".join(REGIONS))
        )
    if settings["neutralization"] not in NEUTRALIZATIONS:
        raise ValueError(
            "unknown neutralization %r (known: %s)"
            % (settings["neutralization"], ", ".join(NEUTRALIZATIONS))
        )
    if settings["delay"] not in (0, 1):
        raise ValueError("delay must be 0 or 1, got %r" % (settings["delay"],))
    return settings


def _single_simulation_data(expression, decay, settings):
    """Build one ``POST /simulations`` payload from a resolved settings dict."""
    body = dict(settings)
    body["decay"] = decay
    return {"type": "REGULAR", "settings": body, "regular": expression}


#: Keys a per-alpha dict in a research file may carry. Anything else is an
#: error rather than a silent drop: a config that says
#: ``{"expression": ..., "neutralization": "SUBINDUSTRY"}`` looks like it
#: overrides the setting, but neutralisation is a settings-level field, so the
#: value would vanish and the round would quietly test nothing. Raising on the
#: unknown key turns a wasted experiment into a message.
_ALPHA_ENTRY_KEYS = frozenset({
    "expression", "decay", "mechanism", "label", "name",
    "family", "candidate_id", "idea", "hypothesis", "note",
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
        elif isinstance(item, (list, tuple)) and item:
            expression, decay = item[0], (item[1] if len(item) > 1 else default_decay)
        else:
            raise ValueError("alphas[%d] is not a usable expression entry" % index)
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("alphas[%d] has an empty or non-string expression" % index)
        if not isinstance(decay, int) or isinstance(decay, bool) or decay < 0:
            raise ValueError("alphas[%d] has an invalid decay %r" % (index, decay))
        normalized.append((expression.strip(), decay))
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


def simulate_alphas(
    s,
    alphas,
    settings=None,
    batch_size=3,
    dry_run=False,
    on_event=None,
    max_wait_seconds=None,
    initial_records=None,
    labels=None,
    families=None,
    candidates=None,
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
    ``submitted`` / ``polling`` / ``rate_limited`` / ``completed`` events.
    Feeding it the result of ``resolve_settings`` is the caller's job.
    """
    settings = resolve_settings(settings)
    pairs = alphas if alphas and isinstance(alphas[0], tuple) else normalize_alphas(
        alphas, settings
    )
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

    if batch_size < 1 or batch_size > 3:
        raise ValueError("batch_size must be between 1 and 3")
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
                "alpha_id": None,
                "status": "DRY_RUN",
                "metrics": None,
                "checks": None,
                "error": None,
                "payload": _single_simulation_data(expression, decay, settings),
            }
            for index, (expression, decay) in enumerate(pairs)
        ]

    if initial_records is None:
        records = [
            {
                "expression": expression,
                "decay": decay,
                "candidate_id": candidates[index],
                "mechanism": labels[index],
                "family": families[index],
                "alpha_id": None,
                "status": "QUEUED",
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
            if progress.get("status") == "ERROR" or progress.get("message"):
                record["error"] = progress.get("message") or "simulation error"
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

    # Resume accepted jobs first. An unresolved job still occupies one of the
    # platform's limited active slots, so do not submit the next batch yet.
    for index, record in enumerate(records):
        if record.get("progress_url"):
            poll_record(index)
        elif record.get("metrics_pending"):
            fetch_metrics_record(index)
    if any(
        record.get("progress_url") or record.get("metrics_pending")
        for record in records
    ):
        return records

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        submitted = []
        for offset, (expression, decay) in enumerate(batch):
            index = start + offset
            record = {
                **records[index],
            }
            if record.get("status") not in {"QUEUED", None}:
                continue
            if deadline is not None and deadline - time.monotonic() <= 0:
                break
            try:
                location = submit_single_simulation(
                    s, _single_simulation_data(expression, decay, settings),
                    on_event=lambda kind, payload, i=index: emit(kind, i, payload),
                    deadline=deadline,
                )
            except RateLimited as exc:
                record["status"] = "QUEUED"
                record["error"] = "%s: %s" % (exc.code, exc)
                records[index] = record
                emit("deferred", index, record)
                break
            except AuthError:
                raise
            except BrainError as exc:
                record["status"] = "FAILED"
                record["error"] = "%s: %s" % (exc.code, exc)
                note_operator_rejection(record.get("expression"), record["error"])
                records[index] = record
                emit("failed", index, record)
                continue
            except requests.HTTPError as exc:
                record["status"] = "FAILED"
                record["error"] = "submit failed: %s" % exc
                note_operator_rejection(record.get("expression"), record["error"])
                records[index] = record
                emit("failed", index, record)
                continue
            record["status"] = "PENDING"
            record["progress_url"] = location
            record["error"] = None
            records[index] = record
            emit("submitted", index, record)
            submitted.append(index)

        for index in submitted:
            poll_record(index)
        if any(
            records[index].get("progress_url")
            or records[index].get("metrics_pending")
            for index in submitted
        ):
            # Avoid exceeding the platform's concurrent simulation limit.
            break

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
# ``result`` beside a refreshed ``value``, so the two can contradict each other
# -- observed values 0.7526 and 0.7911 both labelled PASS against limit 0.7.
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
        info = {
            "result": check.get("result"),
            "limit": check.get("limit"),
            "value": check.get("value"),
        }
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

    Returns ``{"alpha_id", "checks", "self_correlation", "complete"}``.
    ``complete`` is False when the budget expired before a finished payload.
    """
    if max_wait_seconds is not None and max_wait_seconds < 0:
        raise ValueError("max_wait_seconds must be non-negative")
    deadline = time.monotonic() + (max_wait_seconds or 0)
    url = BRAIN_API_URL + "/alphas/" + alpha_id + "/check"
    payload = None
    while True:
        response = s.get(url, timeout=_request_timeout(deadline))
        if response.status_code == 404:
            raise NotFound("alpha %s not found" % alpha_id)
        retry_after = _retry_after_seconds(response, cap_error_wait=False)
        if response.status_code in (429, 503) or retry_after:
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
    }


def fetch_daily_pnl(s, alpha_id, cache_dir=None, max_age_hours=24.0,
                    page_limit=10000, attempts=3):
    """Return ``{date: pnl}`` for one alpha from the daily-pnl recordset.

    This is the only view of an alpha's returns the platform exposes, and it is
    what makes an independence claim testable instead of a guess. The
    ``SELF_CORRELATION`` check reports one number against an opaque comparison
    set and never names the counterpart, so reasoning about a pool needs the
    series itself.

    The endpoint answers 200 with an empty body when called too quickly in
    succession. An empty reply is retried rather than returned as "this alpha has
    no PnL", because treating throttling as absence silently produces correlation
    tables that are missing exactly the rows that were fetched fastest.

    ``cache_dir`` keeps a successful fetch for ``max_age_hours``: the series is
    immutable history, and re-pulling it for every comparison wastes requests.
    """
    if cache_dir:
        path = Path(cache_dir) / (alpha_id + ".json")
        if path.is_file():
            age = time.time() - path.stat().st_mtime
            if age < max_age_hours * 3600:
                try:
                    cached = json.loads(path.read_text(encoding="utf-8"))
                    if cached:
                        return {k: float(v) for k, v in cached.items()}
                except (OSError, ValueError, TypeError):
                    pass

    out = {}
    offset = 0
    remaining = attempts
    while True:
        url = BRAIN_API_URL + "/alphas/" + alpha_id + "/recordsets/daily-pnl"
        params = {"limit": page_limit, "offset": offset} if offset else {}
        response = s.get(url, params=params, timeout=_request_timeout(None))
        if response.status_code != 200 or not response.text.strip():
            if not out and remaining > 1:
                remaining -= 1
                time.sleep(3)
                continue
            break
        try:
            payload = response.json()
        except ValueError:
            break
        records = payload.get("records") or []
        for row in records:
            if not row:
                continue
            value = row[1] if len(row) > 1 else None
            if isinstance(value, (int, float)):
                out[row[0]] = float(value)
        if len(records) < page_limit:
            break
        offset += len(records)

    if cache_dir and out:
        try:
            path = Path(cache_dir) / (alpha_id + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(out, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except OSError:
            pass
    return out


def submit_alpha(s, alpha_id, confirm=False, max_wait_seconds=180.0, on_event=None):
    """Submit one alpha to its competition, or report what would happen.

    BRAIN exposes no separate "test submission": the pre-submission test *is*
    ``GET /alphas/{id}/check`` (see :func:`fetch_alpha_checks`), and the only
    write action is ``POST /alphas/{id}/submit``. Submission is irreversible
    from a user's point of view, so ``confirm`` must be True to send it;
    otherwise a dry-run document describing the current checks is returned.
    """
    current = fetch_alpha_metrics(s, alpha_id)
    status = current.get("status")
    document = {
        "alpha_id": alpha_id,
        "status": status,
        "submitted": False,
    }
    if status in ("ACTIVE", "SUBMITTED"):
        document["checks"] = _extract_checks(current) or {}
        document["reason"] = "already %s; nothing to do" % status
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
    if response.status_code >= 400:
        raise ServiceError(
            "submission rejected with HTTP %s: %s"
            % (response.status_code, json.dumps(payload)[:300])
        )
    refreshed = fetch_alpha_metrics(s, alpha_id)
    document.update({
        "submitted": True,
        "status": refreshed.get("status"),
        "response": payload,
    })
    return document


def single_simulate(alpha_pool, neut="SUBINDUSTRY", region="USA", universe="TOP3000",
                    start=0, settings=None, dry_run=False, on_event=None):
    """Back-compat wrapper: simulate nested pools of ``(expression, decay)``.

    ``alpha_pool`` is a list of batches (as produced by ``load_task_pool_single``).
    Prefer :func:`simulate_alphas` for new code. ``start`` is accepted and
    ignored for call compatibility.
    """
    merged = dict(settings or {})
    merged.setdefault("neutralization", neut)
    merged.setdefault("region", region)
    merged.setdefault("universe", universe)
    merged = resolve_settings(merged)

    pairs = []
    for batch in alpha_pool:
        pairs.extend(normalize_alphas(batch, merged))

    if dry_run:
        session = None
    else:
        session = login()
    return simulate_alphas(
        session, pairs, settings=merged, dry_run=dry_run, on_event=on_event
    )


def set_alpha_properties(
    s,
    alpha_id,
    name: str = None,
    color: str = None,
    selection_desc: str = "None",
    combo_desc: str = "None",
    tags: str = ["ace_tag"],
):
    """
    Function changes alpha's description parameters
    """
 
    params = {
        "color": color,
        "name": name,
        "tags": tags,
        "category": None,
        "regular": {"description": None},
        "combo": {"description": combo_desc},
        "selection": {"description": selection_desc},
    }
    response = s.patch(
        "https://api.worldquantbrain.com/alphas/" + alpha_id, json=params
    )


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
        "type!=": alpha_type_not,
    }
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


def alpha_record_fields(record):
    """Return ``(expression, sharpe, decay)`` from a :func:`list_alphas` record.

    Also accepts the legacy tuples the old ``get_alphas`` returned, so old
    notebooks keep working.
    """
    if isinstance(record, dict):
        expression, _, _ = regular_expression(record)
        metrics = record.get("is") or {}
        settings = record.get("settings") or {}
        return expression, metrics.get("sharpe"), settings.get("decay")
    if isinstance(record, (list, tuple)) and len(record) >= 3:
        return record[1], record[2], record[-1]
    return None, None, None


def prune(alpha_records, prefix, keep_num):
    """Keep the top ``keep_num`` alphas per datafield, ordered by sharpe.

    ``alpha_records`` is what :func:`list_alphas` returns. Fields are read out
    of the expression after ``prefix`` (e.g. ``fnd6``, ``anl4``); a negative
    sharpe goes into a separate ``-field`` bucket so one bad field cannot crowd
    out a good one. Returns ``[[expression, decay], ...]``.
    """
    output = []
    counts = defaultdict(int)
    for record in alpha_records:
        expression, sharpe, decay = alpha_record_fields(record)
        if not expression:
            continue
        field = expression.split(prefix)[-1].split(",")[0]
        if sharpe is not None and sharpe < 0:
            field = "-%s" % field
        if counts[field] < keep_num:
            counts[field] += 1
            output.append([expression, decay])
    return output


def get_group_second_order_factory(first_order, group_ops, region):
    second_order = []
    for fo in first_order:
        for group_op in group_ops:
            second_order += group_factory(group_op, fo, region)
    return second_order


def group_factory(op, field, region):
    output = []
    vectors = ["cap"] 
    
    usa_group_13 = ['pv13_h_min2_3000_sector','pv13_r2_min20_3000_sector','pv13_r2_min2_3000_sector',
                    'pv13_r2_min2_3000_sector', 'pv13_h_min2_focused_pureplay_3000_sector']
    
    cap_group = "bucket(rank(cap), range='0.1, 1, 0.1')"
    asset_group = "bucket(rank(assets),range='0.1, 1, 0.1')"
    sector_cap_group = "bucket(group_rank(cap, sector),range='0.1, 1, 0.1')"
    sector_asset_group = "bucket(group_rank(assets, sector),range='0.1, 1, 0.1')"

    vol_group = "bucket(rank(ts_std_dev(returns,20)),range = '0.1, 1, 0.1')"

    liquidity_group = "bucket(rank(close*volume),range = '0.1, 1, 0.1')"

    groups = ["market","sector", "industry", "subindustry",
              cap_group, asset_group, sector_cap_group, sector_asset_group, vol_group, liquidity_group]
    
    groups += usa_group_13
        
    for group in groups:
        if op.startswith("group_vector"):
            for vector in vectors:
                alpha = "%s(%s,%s,densify(%s))"%(op, field, vector, group)
                output.append(alpha)
        elif op.startswith("group_percentage"):
            alpha = "%s(%s,densify(%s),percentage=0.5)"%(op, field, group)
            output.append(alpha)
        else:
            alpha = "%s(%s,densify(%s))"%(op, field, group)
            output.append(alpha)
        
    return output

def trade_when_factory(op,field,region):
    output = []
    open_events = ["ts_arg_max(volume, 5) == 0", "ts_corr(close, volume, 20) < 0",
                   "ts_corr(close, volume, 5) < 0", "ts_mean(volume,10)>ts_mean(volume,60)",
                   "group_rank(ts_std_dev(returns,60), sector) > 0.7", "ts_zscore(returns,60) > 2",
                   "ts_arg_min(volume, 5) > 3",
                   "ts_std_dev(returns, 5) > ts_std_dev(returns, 20)",
                   "ts_arg_max(close, 5) == 0", "ts_arg_max(close, 20) == 0",
                   "ts_corr(close, volume, 5) > 0", "ts_corr(close, volume, 5) > 0.3", "ts_corr(close, volume, 5) > 0.5",
                   "ts_corr(close, volume, 20) > 0", "ts_corr(close, volume, 20) > 0.3", "ts_corr(close, volume, 20) > 0.5",
                   "ts_regression(returns, %s, 5, lag = 0, rettype = 2) > 0"%field,
                   "ts_regression(returns, %s, 20, lag = 0, rettype = 2) > 0"%field,
                   "ts_regression(returns, ts_step(20), 20, lag = 0, rettype = 2) > 0",
                   "ts_regression(returns, ts_step(5), 5, lag = 0, rettype = 2) > 0"]

    exit_events = ["abs(returns) > 0.1", "-1"]

    usa_events = ["rank(rp_css_business) > 0.8", "ts_rank(rp_css_business, 22) > 0.8", "rank(vec_avg(mws82_sentiment)) > 0.8",
                  "ts_rank(vec_avg(mws82_sentiment),22) > 0.8", "rank(vec_avg(nws48_ssc)) > 0.8",
                  "ts_rank(vec_avg(nws48_ssc),22) > 0.8", "rank(vec_avg(mws50_ssc)) > 0.8", "ts_rank(vec_avg(mws50_ssc),22) > 0.8",
                  "ts_rank(vec_sum(scl12_alltype_buzzvec),22) > 0.9", "pcr_oi_270 < 1", "pcr_oi_270 > 1",]

    asi_events = ["rank(vec_avg(mws38_score)) > 0.8", "ts_rank(vec_avg(mws38_score),22) > 0.8"]

    eur_events = ["rank(rp_css_business) > 0.8", "ts_rank(rp_css_business, 22) > 0.8",
                  "rank(vec_avg(oth429_research_reports_fundamental_keywords_4_method_2_pos)) > 0.8",
                  "ts_rank(vec_avg(oth429_research_reports_fundamental_keywords_4_method_2_pos),22) > 0.8",
                  "rank(vec_avg(mws84_sentiment)) > 0.8", "ts_rank(vec_avg(mws84_sentiment),22) > 0.8",
                  "rank(vec_avg(mws85_sentiment)) > 0.8", "ts_rank(vec_avg(mws85_sentiment),22) > 0.8",
                  "rank(mdl110_analyst_sentiment) > 0.8", "ts_rank(mdl110_analyst_sentiment, 22) > 0.8",
                  "rank(vec_avg(nws3_scores_posnormscr)) > 0.8",
                  "ts_rank(vec_avg(nws3_scores_posnormscr),22) > 0.8",
                  "rank(vec_avg(mws36_sentiment_words_positive)) > 0.8",
                  "ts_rank(vec_avg(mws36_sentiment_words_positive),22) > 0.8"]

    glb_events = ["rank(vec_avg(mdl109_news_sent_1m)) > 0.8",
                  "ts_rank(vec_avg(mdl109_news_sent_1m),22) > 0.8",
                  "rank(vec_avg(nws20_ssc)) > 0.8",
                  "ts_rank(vec_avg(nws20_ssc),22) > 0.8",
                  "vec_avg(nws20_ssc) > 0",
                  "rank(vec_avg(nws20_bee)) > 0.8",
                  "ts_rank(vec_avg(nws20_bee),22) > 0.8",
                  "rank(vec_avg(nws20_qmb)) > 0.8",
                  "ts_rank(vec_avg(nws20_qmb),22) > 0.8"]

    chn_events = ["rank(vec_avg(oth111_xueqiunaturaldaybasicdivisionstat_senti_conform)) > 0.8",
                  "ts_rank(vec_avg(oth111_xueqiunaturaldaybasicdivisionstat_senti_conform),22) > 0.8",
                  "rank(vec_avg(oth111_gubanaturaldaydevicedivisionstat_senti_conform)) > 0.8",
                  "ts_rank(vec_avg(oth111_gubanaturaldaydevicedivisionstat_senti_conform),22) > 0.8",
                  "rank(vec_avg(oth111_baragedivisionstat_regi_senti_conform)) > 0.8",
                  "ts_rank(vec_avg(oth111_baragedivisionstat_regi_senti_conform),22) > 0.8"]

    kor_events = ["rank(vec_avg(mdl110_analyst_sentiment)) > 0.8",
                  "ts_rank(vec_avg(mdl110_analyst_sentiment),22) > 0.8",
                  "rank(vec_avg(mws38_score)) > 0.8",
                  "ts_rank(vec_avg(mws38_score),22) > 0.8"]

    twn_events = ["rank(vec_avg(mdl109_news_sent_1m)) > 0.8",
                  "ts_rank(vec_avg(mdl109_news_sent_1m),22) > 0.8",
                  "rank(rp_ess_business) > 0.8",
                  "ts_rank(rp_ess_business,22) > 0.8"]

    for oe in open_events:
        for ee in exit_events:
            alpha = "%s(%s, %s, %s)"%(op, oe, field, ee)
            output.append(alpha)
    return output


def check_submission(alpha_bag, gold_bag, start):
    depot = []
    s = login()
    for idx, g in enumerate(alpha_bag):
        if idx < start:
            continue
        if idx % 5 == 0:
            print(idx)
        if idx % 200 == 0:
            s = login()
        #print(idx)
        pc = get_check_submission(s, g)
        if pc == "sleep":
            sleep(100)
            s = login()
            alpha_bag.append(g)
        elif pc != pc:
            # pc is nan
            print("check self-corrlation error")
            sleep(100)
            alpha_bag.append(g)
        elif pc == "fail":
            continue
        elif pc == "error":
            depot.append(g)
        else:
            print(g)
            gold_bag.append((g, pc))
    print(depot)
    return gold_bag


def get_check_submission(s, alpha_id):
    while True:
        result = s.get("https://api.worldquantbrain.com/alphas/" + alpha_id + "/check")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    try:
        if result.json().get("is", 0) == 0:
            print("logged out")
            return "sleep"
        checks_df = pd.DataFrame(
                result.json()["is"]["checks"]
        )
        pc = checks_df[checks_df.name == "SELF_CORRELATION"]["value"].values[0]
        if not any(checks_df["result"] == "FAIL"):
            return pc
        else:
            return "fail"
    except:
        print("catch: %s"%(alpha_id))
        return "error"
    
def view_alphas(gold_bag):
    s = login()
    sharp_list = []
    for gold, pc in gold_bag:

        triple = locate_alpha(s, gold)
        info = [triple[0], triple[2], triple[3], triple[4], triple[5], triple[6], triple[1]]
        info.append(pc)
        sharp_list.append(info)

    sharp_list.sort(reverse=True, key = lambda x : x[1])
    for i in sharp_list:
        print(i)
 
def locate_alpha(s, alpha_id):
    while True:
        alpha = s.get("https://api.worldquantbrain.com/alphas/" + alpha_id)
        if "retry-after" in alpha.headers:
            time.sleep(float(alpha.headers["Retry-After"]))
        else:
            break
    string = alpha.content.decode('utf-8')
    metrics = json.loads(string)
    #print(metrics["regular"]["code"])
    
    dateCreated = metrics["dateCreated"]
    sharpe = metrics["is"]["sharpe"]
    fitness = metrics["is"]["fitness"]
    turnover = metrics["is"]["turnover"]
    margin = metrics["is"]["margin"]
    decay = metrics["settings"]["decay"]
    exp = metrics['regular']['code']
    
    triple = [alpha_id, exp, sharpe, turnover, fitness, margin, dateCreated, decay]
    return triple
            

# Consultant methods
def load_task_pool(alpha_list, limit_of_children_simulations, limit_of_multi_simulations):
    '''
    Input:
        alpha_list : list of (alpha, decay) tuples
        limit_of_multi_simulations : number of children simulation in a multi-simulation
        limit_of_multi_simulations : number of simultaneous multi-simulations
    Output:
        task : [10 * (alpha, decay)] for a multi-simulation
        pool : [10 * [10 * (alpha, decay)]] for simultaneous multi-simulations
        pools : [[10 * [10 * (alpha, decay)]]]

    '''
    tasks = [alpha_list[i:i + limit_of_children_simulations] for i in range(0, len(alpha_list), limit_of_children_simulations)]
    pools = [tasks[i:i + limit_of_multi_simulations] for i in range(0, len(tasks), limit_of_multi_simulations)]
    return pools


# some other factory for other operators
def vector_factory(op, field):
    output = []
    vectors = ["cap"]
    
    for vector in vectors:
    
        alpha = "%s(%s, %s)"%(op, field, vector)
        output.append(alpha)
    
    return output
 
def ts_comp_factory(op, field, factor, paras):
    output = []
    #l1, l2 = [3, 5, 10, 20, 60, 120, 240], paras
    l1, l2 = [5, 22, 66, 240], paras
    comb = list(product(l1, l2))
    
    for day,para in comb:
        
        if type(para) == float:
            alpha = "%s(%s, %d, %s=%.1f)"%(op, field, day, factor, para)
        elif type(para) == int:
            alpha = "%s(%s, %d, %s=%d)"%(op, field, day, factor, para)
            
        output.append(alpha)
    
    return output
 
def twin_field_factory(op, field, fields):
    
    output = []
    #days = [3, 5, 10, 20, 60, 120, 240]
    days = [5, 22, 66, 240]
    outset = list(set(fields) - set([field]))
    
    for day in days:
        for counterpart in outset:
            alpha = "%s(%s, %s, %d)"%(op, field, counterpart, day)
            output.append(alpha)
    
    return output
 
def login_hk():
    """Authenticate against the Hong Kong endpoint.

    BRAIN answers the first POST with a biometrics URL instead of a session.
    That step needs a human in a browser, so this raises with the URL rather
    than calling ``input()`` -- an interactive prompt is a hang for any
    automated caller.
    """
    session = requests.Session()
    response = session.post("https://api.worldquantbrain.com/authentication")
    location = response.headers.get("Location")
    if not location:
        return session
    raise AuthError(
        "Hong Kong accounts need browser authorization first: open %s, then "
        "retry with a session created from those cookies." % location
    )


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
