"""Account access and local session/catalog cache commands."""

from __future__ import annotations

import cache_store
import machine_lib as ml
import os
import research_paths
import simulation_capabilities

from .runtime import with_session





def _equity_scope_samples(options):
    """Choose one native-advertised universe/delay per equity region."""
    fields = options
    for key in ("actions", "POST", "settings", "children"):
        fields = fields.get(key) if isinstance(fields, dict) else None
    evidence = {"state": "unknown", "source": "live OPTIONS /simulations",
                "instrument_type": "EQUITY", "advertised_regions": [], "issues": []}
    instruments = simulation_capabilities.choice_values(fields.get("instrumentType"), {}) if isinstance(fields, dict) else None
    if instruments is None:
        evidence["issues"].append("instrument_type_choices_unavailable")
        return [], evidence
    if "EQUITY" not in instruments:
        evidence.update(state="verified", equity_supported=False)
        return [], evidence
    regions = simulation_capabilities.choice_values(fields.get("region"), {"instrumentType": "EQUITY"})
    if regions == []:
        evidence.update(state="verified", equity_supported=True)
        return [], evidence
    regions = [value for value in (regions or []) if isinstance(value, str) and value]
    evidence["advertised_regions"] = regions
    if not regions:
        evidence["issues"].append("equity_region_choices_unavailable")
    scopes = []
    for region in regions:
        context = {"instrumentType": "EQUITY", "region": region}
        advertised_delays = simulation_capabilities.choice_values(fields.get("delay"), context)
        delays = [int(value) for value in (advertised_delays or [])
                  if (type(value) is int and value >= 0)
                  or (isinstance(value, str) and value.isdecimal())]
        if not delays:
            evidence["issues"].append({"region": region, "reason": "delay_choices_unavailable"})
            continue
        delay = ml.DEFAULT_SETTINGS["delay"]
        context["delay"] = delay if delay in delays else delays[0]
        universes = [value for value in (simulation_capabilities.choice_values(fields.get("universe"), context) or [])
                     if isinstance(value, str) and value]
        if not universes:
            evidence["issues"].append({"region": region, "reason": "universe_choices_unavailable"})
            continue
        universe = ml.DEFAULT_SETTINGS["universe"]
        scopes.append({**context, "universe": universe if universe in universes else universes[0]})
    evidence["state"] = "verified" if not evidence["issues"] else "unknown"
    return scopes, evidence


def _data_access(session):
    """Keep live sampled access, empty catalogs and unavailable evidence distinct."""
    result = {"state": "unknown", "regions": [], "probes": [],
              "coverage": "one advertised equity universe/delay per advertised region; not all combinations"}
    try:
        snapshot = simulation_capabilities.fetch(session, refresh=True)
    except ml.AuthError:
        raise
    except ml.BrainError as exc:
        result["settings"] = {"state": "unknown", "error": {"code": exc.code, "message": str(exc)}}
        return result
    result["capabilities"] = snapshot
    if snapshot.get("error"):
        result["settings"] = {"state": "unknown", "error": snapshot["error"]}
        return result
    scopes, evidence = _equity_scope_samples(
        {"actions": {"POST": {"settings": {"children": snapshot["settings_schema"]}}}})
    result["settings"] = evidence
    for scope in scopes:
        cache = {}
        row = {"scope": scope, "state": "unknown"}
        try:
            records = ml.get_datasets(session, instrument_type=scope["instrumentType"],
                                      region=scope["region"], universe=scope["universe"],
                                      delay=scope["delay"], limit=1, refresh=True, report=cache)
            fresh = cache.get("hit") is False and not any(
                cache.get(key) for key in ("stale", "degraded", "error"))
            valid = isinstance(records, list) and all(
                isinstance(record, dict) and isinstance(record.get("id"), str) and record["id"]
                for record in records)
            # One live dataset proves existence even if limit=1 stops paging.
            # An empty prefix only proves absence when enumeration completed.
            if fresh and valid and (records or cache.get("complete") is True):
                row["state"] = "available" if records else "empty"
                row["dataset_count_lower_bound"] = len(records or [])
                if records:
                    result["regions"].append(scope["region"])
        except ml.AuthError:
            raise
        except ml.BrainError as exc:
            row["error"] = {"code": exc.code, "message": str(exc)}
        row["metadata"] = cache
        result["probes"].append(row)
    if evidence["state"] == "verified" and all(row["state"] != "unknown" for row in result["probes"]):
        result["state"] = "verified"
    return result


def cmd_auth_check(args, _sink):
    """Verify BRAIN credentials, report permissions, session cache state and accessible data regions."""
    meta = {}
    session = ml.load_session(use_cache=not args.no_cache, refresh=args.relogin,
                              meta=meta)
    info = ml.session_status()
    access = _data_access(session)
    return {
        "authenticated": True,
        "user_id": info.get("user_id"),
        "permissions": info.get("permissions"),
        "token_lifetime_seconds": info.get("expires_in_seconds"),
        "data_regions": access["regions"],
        "data_scope_evidence": access,
        "session": {
            "cached": meta.get("cached"),
            "reason": meta.get("reason"),
            "cache_state": meta.get("cache_state"),
            "source": info.get("source"),
            "path": info.get("path"),
            "expires_at": info.get("expires_at"),
            "expires_in_seconds": info.get("expires_in_seconds"),
            "age_seconds": info.get("age_seconds"),
        },
        "metadata_cache": ml.metadata_cache_policy(),
        "note": "data_regions confirms only sampled live equity scopes; empty, unknown and unprobed scopes are distinct",
    }


def cmd_auth_logout(args, _sink):
    """Forget the cached session. Does not revoke it server-side."""
    removed = ml.invalidate_session()
    return {
        "logged_out": removed,
        "session_path": str(ml.session_cache_path()),
        "note": ("Removed only the local cookie cache; the BRAIN session itself "
                 "stays valid until it expires server-side."),
    }


#: Endpoints worth knowing the status of, and why. Probing them turns "the platform
#: probably does not expose X" into a measured fact - the difference between a
#: remembered constraint and an invented one.
PROBE_PATHS = (
    ("/users/self/alphas", "own alpha list"),
    ("/alphas/{alpha}", "one alpha record (metrics, checks, grade)"),
    ("/alphas", "global alpha enumeration (expected 405)"),
    ("/leaderboards", "leaderboard (expected 404)"),
    ("/users/self/competitions", "own competitions, where your rank lives"),
    ("/alphas/{alpha}/correlations/prod", "production correlation detail (level-gated)"),
    ("/alphas/{alpha}/recordsets/daily-pnl", "daily PnL recordset"),
    ("/users/self/consultant", "consultant surface (level-gated)"),
)


def cmd_api_probe(args, _sink):
    """Probe a fixed list of BRAIN endpoints and report their HTTP status.

    The point is not to discover features but to retire assumptions: a 403 and a 404
    mean different things (level-gated versus absent), and both get mistaken for
    "the platform cannot do this" in a write-up.
    """
    import requests

    alpha_id = getattr(args, "alpha", None)

    def work(session):
        rows = []
        for path, purpose in PROBE_PATHS:
            target = path.replace("{alpha}", alpha_id) if "{alpha}" in path else path
            if "{alpha}" in path and not alpha_id:
                rows.append({"path": path, "purpose": purpose, "status": None,
                             "note": "no --alpha given"})
                continue
            try:
                response = session.request(
                    "GET", ml.BRAIN_API_URL + target, timeout=30)
                status = response.status_code
            except requests.RequestException as exc:
                rows.append({"path": target, "purpose": purpose, "status": None,
                             "note": str(exc)[:80]})
                continue
            note = {
                200: "available",
                403: "level-gated (exists, your account may not read it)",
                404: "absent",
                405: "exists but rejects GET",
            }.get(status, "")
            rows.append({"path": target, "purpose": purpose, "status": status, "note": note})
        return {"probed": len(rows), "alpha": alpha_id, "endpoints": rows,
                "note": ("403 and 404 are different findings: one is a level gate, the "
                         "other is an endpoint that does not exist.")}

    return with_session(args, work)


def cmd_cache_info(args, _sink):
    """Describe what is cached, how fresh it is, and where it lives."""
    root = research_paths.cache_root()
    stats = cache_store.cache_stats(root)
    catalog = {}
    meta_root = research_paths.metadata_root()
    if os.path.isdir(meta_root):
        for kind in sorted(os.listdir(meta_root)):
            kind_dir = os.path.join(meta_root, kind)
            if not os.path.isdir(kind_dir):
                continue
            entries = []
            for name in sorted(os.listdir(kind_dir)):
                if not name.endswith(".json"):
                    continue
                entry = cache_store.read_entry(os.path.join(kind_dir, name))
                entries.append({
                    "file": name,
                    "fresh": entry["fresh"],
                    "age_seconds": None if entry.get("age_seconds") is None
                    else round(entry["age_seconds"], 1),
                    "expires_in_seconds": None if entry.get("expires_in") is None
                    else round(entry["expires_in"], 1),
                    "error": entry.get("error"),
                })
            catalog[kind] = {
                "entries": len(entries),
                "fresh": sum(1 for e in entries if e["fresh"]),
                "oldest_age_seconds": max(
                    [e["age_seconds"] for e in entries
                     if e["age_seconds"] is not None] or [None])
                if entries else None,
            }
    return {
        "cache_root": root,
        "policy": {
            "metadata": ml.metadata_cache_policy(),
            "session": {
                "single_flight": True,
                "file_mode": "0600",
                "lock_timeout_seconds": float(
                    os.environ.get("BRAIN_LOGIN_LOCK_TIMEOUT", "60") or 60),
            },
        },
        "session": ml.session_status(),
        "metadata": catalog,
        "stats": stats,
    }


def cmd_cache_clear(args, _sink):
    """Drop cached data. Locks are kept so a running process is not disturbed."""
    root = research_paths.cache_root()
    scope = args.kind
    removed = {"session": 0, "metadata": 0}
    if scope in ("all", "session"):
        removed["session"] = 1 if ml.invalidate_session() else 0
    if scope in ("all", "metadata"):
        removed["metadata"] = cache_store.clear_dir(research_paths.metadata_root(),
                                                   keep_locks=True)
    return {
        "cleared": scope,
        "removed": removed,
        "cache_root": root,
        "kept_locks": True,
    }
