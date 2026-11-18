"""Account-scoped cache for validated read-only reference responses."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import cache_store
import machine_lib as ml
import research_paths

from .runtime import CliError, protect_output


_KIND_PREFIX = "reference-"


def _result(data, entry=None, *, hit=False):
    entry = entry or {}
    return {**data, "cache": {"hit": hit, "age_seconds": entry.get("age_seconds"),
                              "created_at": entry.get("created_at")}}


def _content_digest(data):
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _protect_cache_output(args):
    saved = getattr(args, "save", None)
    if not saved or saved == "-":
        return
    target = Path(saved).expanduser().resolve()
    root = Path(research_paths.metadata_root()).resolve()
    if target.is_relative_to(root):
        parts = target.relative_to(root).parts
        if parts and parts[0].startswith(_KIND_PREFIX):
            protect_output(args, target)


def _hit(path, kind, params, max_age):
    entry = cache_store.read_entry(path)
    payload = entry["payload"]
    age = entry["age_seconds"]
    if (entry["present"] and entry["fresh"] and entry["kind"] == kind
            and entry["expires_in"] is not None
            and age is not None and age <= max_age and isinstance(payload, dict)
            and payload.get("request") == params and isinstance(payload.get("data"), dict)
            and payload.get("digest") == _content_digest(payload["data"])):
        return _result(payload["data"], entry, hit=True)
    return None


def read_cached(args, kind, key, fetch):
    """Serve an exact fresh response before calling the authenticated loader."""
    _protect_cache_output(args)
    offline = getattr(args, "offline", False)
    bypass = getattr(args, "no_cache", False)
    refresh = getattr(args, "refresh_meta", False) or getattr(args, "relogin", False)
    if offline and (refresh or bypass):
        raise CliError("INVALID_INPUT", "--offline conflicts with --refresh-meta, --relogin and --no-cache")

    policy = ml.metadata_cache_policy()
    max_age = getattr(args, "meta_max_age", None)
    if max_age is None:
        max_age = policy["max_age"]
    username = ml.configured_username().casefold()
    path = None
    if username and not bypass and policy["enabled"] and max_age > 0:
        account = hashlib.sha256(username.encode()).hexdigest()
        kind = _KIND_PREFIX + kind
        params = {"account": account, "key": key}
        path = ml.metadata_cache_path(kind, params)
        if not refresh:
            cached = _hit(path, kind, params, max_age)
            if cached is not None:
                return cached
    if offline:
        raise CliError("NOT_FOUND", "no fresh cached reference response for this account and request",
                       "run the command online once to populate the cache")
    if path is None:
        return _result(fetch())

    with ExitStack() as stack:
        try:
            stack.enter_context(cache_store.file_lock(path.with_suffix(".lock")))
        except (OSError, cache_store.CacheError):
            return _result(fetch())
        if not refresh:
            cached = _hit(path, kind, params, max_age)
            if cached is not None:
                return cached
        data = fetch()
        entry = cache_store.build_entry(
            kind, {"request": params, "data": data, "digest": _content_digest(data)},
            ttl=max_age, source="brain-reference")
        try:
            cache_store.write_json_atomic(path, entry)
        except OSError:
            return _result(data)
        return _result(data, {**entry["_cache"], "age_seconds": 0.0})
