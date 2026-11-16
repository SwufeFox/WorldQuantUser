"""Account-bound simulation capability evidence, with no application dependencies.

Only native OPTIONS settings metadata is interpreted. BRAIN does not publish a
concurrency contract here; missing capacity remains unknown, never a tier guess.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os

import cache_store


_SCOPE = {"endpoint": "/simulations", "method": "POST"}
_SOURCE = "authenticated OPTIONS /simulations"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def account_provenance(session=None):
    """Hash a confirmed principal, excluding permissions and credential material.

    Authentication restores ``_brain_account_id`` on sessions. Offline cache
    identity is usable only if an explicit username still matches that cache.
    """
    import machine_lib as ml

    auth = getattr(session, "auth", None)
    username = (auth[0] if isinstance(auth, (tuple, list)) and auth else
                os.environ.get("BRAIN_USERNAME") if session is None else None)
    username = username.strip().casefold() if isinstance(username, str) else None
    user_id = getattr(session, "_brain_account_id", None)
    sources = []
    if type(user_id) in (str, int) and str(user_id).strip():
        sources = ["authenticated_user_id"]
    else:
        user_id = None
        entry = cache_store.read_entry(ml.session_cache_path())
        payload = entry.get("payload")
        if entry.get("fresh") and isinstance(payload, dict) and type(payload.get("user_id")) in (str, int):
            cookies = getattr(session, "cookies", None)
            stored = payload.get("cookies")
            cached_username = payload.get("username")
            cached_username = cached_username.strip().casefold() if isinstance(cached_username, str) else None
            matches = ((not username or username == cached_username) if session is None else
                       isinstance(stored, dict) and bool(stored) and hasattr(cookies, "get_dict")
                       and cookies.get_dict() == stored)
            if matches and str(payload["user_id"]).strip():
                user_id = payload["user_id"]
                sources = ["cached_user_id"]
    identity = {"user_id": str(user_id)} if user_id is not None else {"username": username} if username else {}
    return {"state": "known" if identity else "unknown",
            "fingerprint": _digest(identity) if identity else None,
            "sources": sources or (["username"] if username else [])}


def choice_values(spec, context):
    """Resolve native dependent choices; None is unknown, [] is advertised empty."""
    choices = spec.get("choices") if isinstance(spec, dict) else None
    while isinstance(choices, dict):
        keys = [key for key in context if key in choices]
        if len(keys) != 1 or not isinstance(choices[keys[0]], dict):
            return None
        key = keys[0]
        branches = choices[key]
        value = context[key]
        choices = branches.get(str(value))
        if choices is None and isinstance(value, (str, int, float, bool)):
            choices = branches.get(value)
    if not isinstance(choices, list):
        return None
    result = []
    for row in choices:
        value = row.get("value") if isinstance(row, dict) else row
        if type(value) not in (str, int, float, bool) or (type(value) is float and not math.isfinite(value)):
            return None
        if value not in result:
            result.append(value)
    return result


def _path(account):
    import machine_lib as ml

    return ml.metadata_cache_path("simulation-capabilities", {"account": account.get("fingerprint"), "scope": _SCOPE})


def _unknown(account, reasons, **extra):
    return {"state": "unknown", "account": account, "scope": dict(_SCOPE),
            "source": None, "as_of": None, "age_seconds": None, "capability_id": None,
            "settings_schema": {}, "concurrency": {"state": "unknown", "limit": None,
            "source": None}, "reasons": reasons, **extra}


def _cached(account, max_age_seconds=None):
    import machine_lib as ml

    policy = ml.metadata_cache_policy()
    maximum = policy["max_age"] if max_age_seconds is None else float(max_age_seconds)
    if account.get("state") != "known":
        return _unknown(account, ["account_unknown"])
    if not policy.get("enabled", True):
        return _unknown(account, ["cache_disabled"])
    entry = cache_store.read_entry(_path(account))
    payload = entry.get("payload")
    if (not isinstance(payload, dict) or entry.get("kind") != "simulation-capabilities"
            or payload.get("version") != 1 or not isinstance(payload.get("account"), dict)
            or payload["account"].get("fingerprint") != account.get("fingerprint")
            or payload.get("scope") != _SCOPE or not isinstance(payload.get("settings_schema"), dict)):
        return _unknown(account, ["snapshot_missing_or_invalid"])
    age = entry.get("age_seconds")
    fresh = bool(entry.get("fresh") and age is not None and 0 <= age <= maximum)
    result = copy.deepcopy(payload)
    result.update(source="private_snapshot", origin_source=entry.get("source"),
                  as_of=entry.get("created_at"), age_seconds=age,
                  state="verified" if fresh and payload.get("state") == "verified" else "unknown",
                  reasons=[] if fresh else ["freshness_unverified"])
    result["concurrency"] = {"state": "unknown", "limit": None, "source": None}
    return result


def cached(max_age_seconds=None):
    """Read local evidence only; do not load credentials or establish a session."""
    return _cached(account_provenance(), max_age_seconds)


def fetch(session, refresh=False, max_age_seconds=None):
    """Read one native settings schema, caching only account-bound live evidence."""
    import machine_lib as ml

    account = account_provenance(session)
    policy = ml.metadata_cache_policy()
    if not refresh and not policy.get("refresh", False):
        previous = _cached(account, max_age_seconds)
        if previous["state"] == "verified":
            return previous
    try:
        options = ml.request_json(session, "OPTIONS", "/simulations")
    except ml.AuthError:
        raise
    except ml.BrainError as exc:
        return _unknown(account, ["options_unavailable"], source=_SOURCE,
                        error={"code": exc.code, "message": str(exc)})
    fields = options
    for key in ("actions", "POST", "settings", "children"):
        fields = fields.get(key) if isinstance(fields, dict) else None
    if not isinstance(fields, dict) or not fields:
        return _unknown(account, ["settings_schema_unavailable"], source=_SOURCE)
    fields = copy.deepcopy(fields)
    now = cache_store.utc_now().isoformat()
    snapshot = {"version": 1, "state": "verified" if account["state"] == "known" else "unknown",
                "account": account, "scope": dict(_SCOPE), "source": _SOURCE,
                "as_of": now, "age_seconds": 0.0, "settings_schema": fields,
                "capability_id": _digest({"account": account.get("fingerprint"), "scope": _SCOPE, "settings_schema": fields}),
                "concurrency": {"state": "unknown", "limit": None, "source": None},
                "reasons": [] if account["state"] == "known" else ["account_unknown"]}
    ttl = policy["max_age"]
    if snapshot["state"] == "verified" and ttl > 0 and policy.get("enabled", True):
        path = _path(account)
        try:
            with cache_store.file_lock(str(path) + ".lock"):
                cache_store.write_json_atomic(path, cache_store.build_entry(
                    "simulation-capabilities", snapshot, ttl=ttl, source=_SOURCE), mode=0o600)
        except (OSError, cache_store.CacheError) as exc:
            snapshot["cache_error"] = type(exc).__name__
    return snapshot


def settings_evidence(settings, snapshot):
    """Check only published constraints, retaining unknown extension settings.

    `checked` names settings whose applicable advertised constraints were checked;
    `unsupported` contains authoritative contradictions and `unknown` explains
    absent/unresolvable metadata. A partial check never claims full verification.
    """
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    schema = snapshot.get("settings_schema")
    schema = schema if isinstance(schema, dict) and snapshot.get("state") == "verified" else {}
    checked, unsupported, unknown = [], [], []
    for key, value in settings.items():
        spec = schema.get(key)
        if not isinstance(spec, dict):
            unknown.append({"setting": key, "reason": "not_advertised"})
            continue
        constraints = 0
        unresolved = False
        failures = []
        if value is None:
            if spec.get("allow_null") is True:
                constraints += 1
            elif spec.get("allow_null") is False:
                constraints += 1
                failures.append("null_not_allowed")
            else:
                unresolved = True
        else:
            kind = spec.get("type")
            type_checks = {"integer": type(value) is int,
                           "float": type(value) is int or (type(value) is float and math.isfinite(value)),
                           "number": type(value) is int or (type(value) is float and math.isfinite(value)),
                           "boolean": type(value) is bool,
                           "string": isinstance(value, str),
                           "object": isinstance(value, dict), "list": isinstance(value, list)}
            if isinstance(kind, str) and kind in type_checks:
                constraints += 1
                if not type_checks[kind]:
                    failures.append("type_mismatch")
            if "choices" in spec:
                values = choice_values(spec, settings)
                if values is None:
                    unresolved = True
                else:
                    constraints += 1
                    # Native integer choices are also published as numeric strings.
                    matches = value in values or (type(value) is int and str(value) in values)
                    if not matches:
                        failures.append("choice_not_advertised")
            for bound, compare in (("min_value", lambda a, b: a < b),
                                   ("max_value", lambda a, b: a > b)):
                limit = spec.get(bound)
                if type(limit) is int or (type(limit) is float and math.isfinite(limit)):
                    if type(value) in (int, float):
                        constraints += 1
                        if (type(value) is float and not math.isfinite(value)) or compare(value, limit):
                            failures.append(bound)
                    else:
                        unresolved = True
            for bound, compare in (("min_length", lambda a, b: a < b),
                                   ("max_length", lambda a, b: a > b)):
                limit = spec.get(bound)
                if type(limit) is int and limit >= 0:
                    if isinstance(value, (str, list, dict)):
                        constraints += 1
                        if compare(len(value), limit):
                            failures.append(bound)
                    else:
                        unresolved = True
        if constraints:
            checked.append(key)
        if failures:
            unsupported.append({"setting": key, "value": copy.deepcopy(value), "reasons": failures})
        elif unresolved or not constraints:
            unknown.append({"setting": key, "reason": "constraints_unresolved"})
    for key, spec in schema.items():
        if key not in settings and isinstance(spec, dict) and spec.get("required") is True:
            unsupported.append({"setting": key, "reasons": ["required_setting_missing"]})
    return {"state": "unsupported" if unsupported else "unknown" if unknown or not schema else "verified",
            "checked": checked, "unsupported": unsupported, "unknown": unknown,
            "capability_id": snapshot.get("capability_id"), "account": snapshot.get("account"),
            "scope": snapshot.get("scope"), "source": snapshot.get("source"),
            "as_of": snapshot.get("as_of"), "age_seconds": snapshot.get("age_seconds")}
