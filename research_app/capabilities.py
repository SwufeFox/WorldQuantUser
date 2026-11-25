"""Private, account-bound catalog snapshots for offline capability evidence."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta


import cache_store
import machine_lib as ml
import simulation_capabilities

DISCLOSURE_DATASETS = frozenset({"fundamental2", "fundamental6", "analyst4"})
_REQUIRED_FIELDS_WAIT_SECONDS = 30.0
_FIELD_SEARCH_INTERVAL_SECONDS = 1.0
_FIELD_PROOF_KIND = "required-field-proof"
_FIELD_PROOF_SOURCE = "authenticated_field_search"


def scope_settings(settings):
    settings = settings or {}
    return {key: settings.get(key) if settings.get(key) is not None else ml.DEFAULT_SETTINGS[key]
            for key in ("region", "universe", "delay", "instrumentType")}




def snapshot_path(scope, account):
    return ml.metadata_cache_path("capabilities", {"scope": scope, "account": account.get("fingerprint")})


def _identify_evidence(records, evidence):
    freshness = evidence.get("freshness") or {}
    evidence["as_of"] = freshness.get("as_of")
    evidence["age_seconds"] = freshness.get("age_seconds", 0.0 if freshness.get("state") == "fresh" else None)
    identity = {"account": (evidence.get("account") or {}).get("fingerprint"), "scope": evidence.get("scope"),
                "coverage": evidence.get("coverage"), "records": records}
    evidence["capability_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return evidence


def field_scope(records, evidence):
    verified = evidence.get("state") == "verified"
    evidence = dict(evidence)
    _identify_evidence(records, evidence)
    evidence["type_coverage"] = {
        "known_count": sum(bool(row.get("type")) for row in records) if verified else 0,
        "unknown_fields": sorted(row["id"] for row in records if row.get("id") and not row.get("type")) if verified else None,
        "state": "checked" if verified else "unknown"}
    known = {row["id"] for row in records if row.get("id")} if verified else None
    types = {row["id"]: row["type"] for row in records if row.get("id") and row.get("type")} if verified else None
    market = set()
    classified = verified and bool(records)
    for row in records if verified else []:
        dataset = row.get("dataset")
        dataset = dataset.get("id") if isinstance(dataset, dict) else dataset
        if not isinstance(dataset, str) or not dataset:
            classified = False
        elif row.get("id") and dataset not in DISCLOSURE_DATASETS:
            market.add(row["id"])
    return {"known_fields": known, "field_types": types,
            "market_fields": market if classified else None,
            "records": records if verified else [], "evidence": evidence}


def save_snapshot(session, settings, records, coverage):
    """Persist only a complete live scope; unidentified accounts remain unknown offline."""
    scope, account = scope_settings(settings), simulation_capabilities.account_provenance(session)
    if (coverage.get("complete") is not True or coverage.get("full_scope") is not True
            or not isinstance(records, list) or coverage.get("field_count") != len(records)
            or account["state"] != "known"):
        return None
    payload = {"version": 1, "scope": scope, "account": account,
               "coverage": coverage, "records": records}
    path = snapshot_path(scope, account)
    ttl = ml.metadata_cache_policy()["max_age"]
    if ttl <= 0:
        return None
    with cache_store.file_lock(str(path) + ".lock"):
        cache_store.write_json_atomic(path, cache_store.build_entry(
            "scoped-capabilities", payload, ttl=ttl, source="authenticated-full-catalog"), mode=0o600)
    return str(path)


def fetch_fields(session, region="USA", universe="TOP3000", delay=1,
                 instrument_type="EQUITY", dataset_ids=None, report=None, refresh=True):
    """Capture account-scoped capability evidence at the application boundary."""
    import field_index

    evidence = report if report is not None else {}
    evidence["account"] = simulation_capabilities.account_provenance(session)
    records = field_index.fetch_all_fields(
        session, region=region, universe=universe, delay=delay,
        instrument_type=instrument_type, dataset_ids=dataset_ids, report=evidence, refresh=refresh)
    catalogs = evidence.get("catalogs") or []
    coverage = evidence.get("coverage") or {}
    if (coverage.get("full_scope") is True and catalogs
            and all(row.get("complete") is True and not any(
                row.get(key) for key in ("hit", "stale", "degraded", "error")) for row in catalogs)):
        # Successful full enumeration can authoritatively advertise zero fields.
        coverage["complete"] = True
        evidence["state"] = "verified"
    _identify_evidence(records, evidence)
    if evidence.get("coverage", {}).get("complete"):
        try:
            evidence["snapshot"] = save_snapshot(session, evidence["scope"], records, evidence["coverage"])
        except (OSError, cache_store.CacheError) as exc:
            evidence["snapshot_error"] = type(exc).__name__
    return records


def _field_proof_path(account, scope, field):
    return ml.metadata_cache_path(_FIELD_PROOF_KIND, {
        "account": account.get("fingerprint"), "scope": scope, "field": field})


def _field_proof_freshness(search, maximum, now=None):
    """Apply both the original expiry and the caller's current maximum age."""
    try:
        observed = datetime.fromisoformat(search["as_of"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(search["expires_at"].replace("Z", "+00:00"))
        now = now or cache_store.utc_now()
        age = (now - observed).total_seconds()
        return age, 0 <= age <= maximum and now < expires
    except (KeyError, TypeError, ValueError, AttributeError):
        return None, False


def _cached_field_proof(path, account, scope, field, maximum):
    """Read a positive exact observation, never generic or stale catalog data."""
    try:
        entry = cache_store.read_entry(path)
    except (OSError, cache_store.CacheError):
        return None
    payload = entry.get("payload")
    if (not entry.get("present") or not entry.get("fresh") or entry.get("kind") != _FIELD_PROOF_KIND
            or entry.get("source") != _FIELD_PROOF_SOURCE or not isinstance(payload, dict)
            or payload.get("version") != 1 or payload.get("scope") != scope
            or payload.get("field") != field or not isinstance(payload.get("account"), dict)
            or payload["account"].get("state") != "known"
            or payload["account"].get("fingerprint") != account.get("fingerprint")):
        return None
    search, row = payload.get("search"), payload.get("row")
    if (not isinstance(search, dict) or search.get("kind") != "datafield_search"
            or search.get("field") != field or search.get("state") != "verified"
            or search.get("source") != _FIELD_PROOF_SOURCE
            or type(search.get("exact_count")) is not int or search.get("exact_count") != 1
            or any(search.get(key) for key in ("error", "stale", "degraded"))
            or not isinstance(row, dict) or row.get("id") != field
            or any(row.get(key) is not None and str(row[key]) != str(value)
                   for key, value in scope.items())):
        return None
    try:
        observed = datetime.fromisoformat(search["as_of"].replace("Z", "+00:00"))
        created = datetime.fromisoformat(entry["created_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(entry["expires_at"].replace("Z", "+00:00"))
        if "expires_at" in search and datetime.fromisoformat(
                search["expires_at"].replace("Z", "+00:00")) != expires:
            return None
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    # Older point entries did not duplicate expiry in search metadata. Carry
    # their envelope expiry into memory without renewing the observation.
    search = {**search, "expires_at": entry["expires_at"]}
    age, fresh = _field_proof_freshness(search, maximum)
    if observed != created or not fresh:
        return None
    return ({**search, "reused_from_cache": True, "source": "private_field_proof",
             "origin_source": _FIELD_PROOF_SOURCE, "age_seconds": age}, row, search["as_of"])


def required_field_scope(session, settings, required_fields, *, deadline=None, proof_cache=None):
    """Verify required IDs with fresh exact proofs and bounded live searches.

    ``verified_records`` exposes individually verified, still-fresh records even
    when some required IDs remain unknown. The original fields retain their
    all-required-IDs-verified contract and never imply a complete catalog.
    """
    from requests.exceptions import RequestException

    scope, account = scope_settings(settings), simulation_capabilities.account_provenance(session)
    required = set(required_fields or ())
    catalogs, records, timestamps, budget_deferred = [], [], [], []
    started = time.monotonic()
    field_deadline = started + _REQUIRED_FIELDS_WAIT_SECONDS if deadline is None else deadline
    policy = ml.metadata_cache_policy()
    maximum = policy["max_age"]
    fingerprint = account.get("fingerprint")
    identified = account.get("state") == "known" and isinstance(fingerprint, str) and bool(fingerprint)
    durable = identified and policy.get("enabled", True) and maximum > 0
    cache_scope = ("field", fingerprint, tuple(scope.items())) if fingerprint and proof_cache is not None else None
    pace_key = ("field_search_next", fingerprint)
    next_search_at = proof_cache.get(pace_key, 0.0) if proof_cache is not None else 0.0
    for field in sorted(required):
        cache_key = (*cache_scope, field) if cache_scope is not None else None
        previous = proof_cache.get(cache_key) if cache_key is not None else None
        if previous is not None:
            search, row, stamp = previous
            age, fresh = _field_proof_freshness(search, maximum)
            refreshed_here = search.get("forced_refresh") and not search.get("reused_from_cache")
            if (not durable or not fresh
                    or (policy.get("refresh", False) and not refreshed_here)):
                previous = None
                proof_cache.pop(cache_key, None)
            else:
                previous = ({**search, "age_seconds": age}, row, stamp)
        path = _field_proof_path(account, scope, field) if identified else None
        reused_in_invocation = previous is not None
        if previous is None and durable and not policy.get("refresh", False):
            previous = _cached_field_proof(path, account, scope, field, maximum)
        if previous is not None:
            search, row, stamp = previous
            catalogs.append({**search, **({"reused_in_invocation": True} if reused_in_invocation else {})})
            records.append(row)
            timestamps.append(stamp)
            if cache_key is not None:
                proof_cache[cache_key] = previous
            continue
        now = time.monotonic()
        search_at = max(now, next_search_at)
        if search_at >= field_deadline:
            budget_deferred.append(field)
            continue
        # Local policy spaces adjacent field requests by at least one second.
        if search_at > now:
            time.sleep(search_at - now)
        next_search_at = search_at + _FIELD_SEARCH_INTERVAL_SECONDS
        if proof_cache is not None:
            proof_cache[pace_key] = next_search_at
        search = {"kind": "datafield_search", "field": field, "state": "unknown"}
        catalogs.append(search)
        if path is not None and policy.get("refresh", False):
            # Once a refresh starts, a failed observation must not resurrect the
            # previous proof on a later invocation. Budget deferral leaves it alone.
            try:
                cache_store.remove(path)
            except (OSError, cache_store.CacheError) as exc:
                search.update(error="SERVICE_ERROR", cache_error=type(exc).__name__, cache_operation="invalidate")
                continue
        try:
            result = ml.search_datafield(session, field, scope, deadline=field_deadline)
        except ml.AuthError:
            raise
        except ml.RateLimited as exc:
            search["error"] = exc.code
            if exc.retry_after is not None:
                search["retry_at"] = time.time() + exc.retry_after
            break
        except (ml.BrainError, RequestException) as exc:
            search["error"] = getattr(exc, "code", type(exc).__name__)
            continue
        matches = result["matches"]
        search.update({key: result[key] for key in ("returned_count", "reported_count", "complete")},
                      exact_count=len(matches))
        if len(matches) != 1:
            # Even a miss in a complete fuzzy search is no proof of absence.
            search["reason"] = "ambiguous_exact_match" if matches else "no_exact_match"
            continue
        row = matches[0]
        if not isinstance(row, dict) or row.get("id") != field:
            search["reason"] = "invalid_exact_match"
            continue
        if any(row.get(key) is not None and str(row[key]) != str(value) for key, value in scope.items()):
            search["reason"] = "scope_mismatch"
            continue
        search["state"] = "verified"
        records.append(row)
        observed = cache_store.utc_now()
        stamp = observed.isoformat()
        search.update(as_of=stamp, age_seconds=0.0, source=_FIELD_PROOF_SOURCE)
        if maximum > 0:
            search["expires_at"] = (observed + timedelta(seconds=maximum)).isoformat()
        if policy.get("refresh", False):
            search["forced_refresh"] = True
        timestamps.append(stamp)
        if durable:
            payload = {"version": 1, "account": account, "scope": scope,
                       "field": field, "search": dict(search), "row": row}
            try:
                cache_store.write_json_atomic(path, cache_store.build_entry(
                    _FIELD_PROOF_KIND, payload, ttl=maximum, source=_FIELD_PROOF_SOURCE, now=observed), mode=0o600)
            except (OSError, cache_store.CacheError) as exc:
                search["cache_error"] = type(exc).__name__
        if cache_key is not None:
            proof_cache[cache_key] = (dict(search), row, stamp)
    # A later slow lookup can outlive a proof accepted earlier in this call.
    # Fail closed at the return boundary; the next invocation can refresh only
    # the expired fields instead of adding another request loop here.
    expired = set()
    finished = cache_store.utc_now()
    for search in catalogs:
        if search.get("state") != "verified" or "expires_at" not in search:
            continue
        age, fresh = _field_proof_freshness(search, maximum, finished)
        search["age_seconds"] = age
        if not fresh:
            field = search["field"]
            expired.add(field)
            search.update(state="unknown", error="SERVICE_ERROR", reason="proof_expired_during_preflight")
            if cache_scope is not None:
                proof_cache.pop((*cache_scope, field), None)
    if expired:
        records = [row for row in records if row["id"] not in expired]
        timestamps = [search["as_of"] for search in catalogs if search.get("state") == "verified"]
    available = {row["id"] for row in records}
    state = "verified" if required and required <= available else "unknown"
    stamp = min(timestamps, default=None)
    age = (cache_store.utc_now() - datetime.fromisoformat(stamp)).total_seconds() if stamp else None
    cached_only = bool(catalogs) and all(row.get("reused_from_cache") for row in catalogs)
    evidence = {"state": state, "scope": scope, "account": account,
                "source": "private_field_proof" if cached_only else _FIELD_PROOF_SOURCE, "catalogs": catalogs,
                "verification": "required_fields_only",
                "coverage": {"complete": False, "full_scope": False, "field_count": len(records)},
                "freshness": {"state": "fresh" if stamp else "unknown", "as_of": stamp,
                              "age_seconds": age},
                "required_fields": {"state": state, "ids": sorted(required), "verified_ids": sorted(available),
                                    "unverified_ids": sorted(required - available),
                                    "routing_source": "scoped_field_proofs" if any(
                                        row.get("reused_from_cache") for row in catalogs) else "scoped_live_search"}}
    if cached_only:
        evidence["origin_source"] = _FIELD_PROOF_SOURCE
    if budget_deferred:
        evidence["required_fields"]["budget_deferred_ids"] = budget_deferred
    if expired:
        evidence["required_fields"]["expired_ids"] = sorted(expired)
    return {**field_scope(records, evidence), "verified_records": records}


def cached_field_scope(settings, max_age_seconds=None, *, account=None):
    """Read exactly one scope/account snapshot; unknown metadata is never verified.

    Callers with an authenticated session can supply its account provenance.
    Without it, retain the configured offline account lookup.
    """
    scope = scope_settings(settings)
    account = simulation_capabilities.account_provenance() if account is None else account
    if not isinstance(account, dict):
        account = {"state": "unknown", "fingerprint": None}
    maximum = ml.metadata_cache_policy()["max_age"] if max_age_seconds is None else float(max_age_seconds)
    entry = cache_store.read_entry(snapshot_path(scope, account))
    payload = entry.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    age = entry.get("age_seconds")
    fresh = bool(entry.get("fresh") and age is not None and 0 <= age <= maximum)
    records = payload.get("records")
    records_valid = isinstance(records, list) and all(
        isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"] for row in records)
    coverage = payload.get("coverage") or {}
    reasons = []
    if (account.get("state") != "known" or not isinstance(account.get("fingerprint"), str)
            or not account["fingerprint"]):
        reasons.append("account_unknown")
    if not entry.get("present"):
        reasons.append("snapshot_missing_or_invalid")
    if (payload.get("scope") != scope or not isinstance(payload.get("account"), dict)
            or payload["account"].get("fingerprint") != account.get("fingerprint")):
        reasons.append("scope_or_account_mismatch")
    if not fresh:
        reasons.append("freshness_unverified")
    if (payload.get("version") != 1 or entry.get("kind") != "scoped-capabilities"
            or not isinstance(coverage, dict) or coverage.get("complete") is not True
            or coverage.get("full_scope") is not True or not isinstance(coverage.get("dataset_ids"), list)
            or not records_valid or coverage.get("field_count") != len(records)):
        reasons.append("coverage_incomplete")
    evidence = {"state": "unknown" if reasons else "verified", "source": "private_snapshot",
                "scope": scope, "account": account, "coverage": coverage,
                "freshness": {"state": "fresh" if fresh else "unknown", "as_of": entry.get("created_at"),
                              "age_seconds": age, "max_age_seconds": maximum}, "reasons": reasons}
    return field_scope(records if records_valid else [], evidence)
