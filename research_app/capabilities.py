"""Private, account-bound catalog snapshots for offline capability evidence."""

from __future__ import annotations

import hashlib
import json
import os

import cache_store
import machine_lib as ml

DISCLOSURE_DATASETS = frozenset({"fundamental2", "fundamental6", "analyst4"})


def scope_settings(settings):
    settings = settings or {}
    return {key: settings.get(key) if settings.get(key) is not None else ml.DEFAULT_SETTINGS[key]
            for key in ("region", "universe", "delay", "instrumentType")}


def account_provenance(session=None):
    """Hash local identity evidence without logging in or storing credentials."""
    auth = getattr(session, "auth", None)
    username = auth[0] if isinstance(auth, (tuple, list)) and auth else os.environ.get("BRAIN_USERNAME")
    identity = {}
    if isinstance(username, str) and username.strip():
        identity["username"] = username.strip().casefold()
    entry = cache_store.read_entry(ml.session_cache_path())
    payload = entry.get("payload")
    if entry.get("fresh") and isinstance(payload, dict) and payload.get("user_id"):
        matches = session is None
        if session is not None:
            cookies = getattr(session, "cookies", None)
            matches = hasattr(cookies, "get_dict") and cookies.get_dict() == payload.get("cookies")
        if matches:
            identity["user_id"] = str(payload["user_id"])
    if not identity:
        return {"state": "unknown", "fingerprint": None, "sources": []}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"state": "known", "fingerprint": digest, "sources": sorted(identity)}


def snapshot_path(scope, account):
    return ml.metadata_cache_path("capabilities", {"scope": scope, "account": account.get("fingerprint")})


def field_scope(records, evidence):
    verified = evidence.get("state") == "verified"
    evidence = dict(evidence)
    evidence["type_coverage"] = {
        "known_count": sum(bool(row.get("type")) for row in records) if verified else 0,
        "unknown_fields": sorted(row["id"] for row in records if row.get("id") and not row.get("type")) if verified else None,
        "state": "checked" if verified else "unknown"}
    known = {row["id"] for row in records if row.get("id")} if verified else None
    types = {row["id"]: row["type"] for row in records if row.get("id") and row.get("type")} if verified else None
    market = set()
    for row in records if verified else []:
        dataset = row.get("dataset")
        dataset = dataset.get("id") if isinstance(dataset, dict) else dataset
        if row.get("id") and dataset and dataset not in DISCLOSURE_DATASETS:
            market.add(row["id"])
    return {"known_fields": known, "field_types": types,
            "market_fields": market if verified else None,
            "records": records if verified else [], "evidence": evidence}


def save_snapshot(session, settings, records, coverage):
    """Persist only a complete live scope; unidentified accounts remain unknown offline."""
    scope, account = scope_settings(settings), account_provenance(session)
    if not coverage.get("complete") or not records or account["state"] != "known":
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
    evidence["account"] = account_provenance(session)
    records = field_index.fetch_all_fields(
        session, region=region, universe=universe, delay=delay,
        instrument_type=instrument_type, dataset_ids=dataset_ids, report=evidence, refresh=refresh)
    if evidence.get("coverage", {}).get("complete"):
        try:
            evidence["snapshot"] = save_snapshot(session, evidence["scope"], records, evidence["coverage"])
        except (OSError, cache_store.CacheError) as exc:
            evidence["snapshot_error"] = type(exc).__name__
    return records


def cached_field_scope(settings, max_age_seconds=None):
    """Read exactly one scope/account snapshot; unknown metadata is never verified."""
    scope, account = scope_settings(settings), account_provenance()
    maximum = ml.metadata_cache_policy()["max_age"] if max_age_seconds is None else float(max_age_seconds)
    entry = cache_store.read_entry(snapshot_path(scope, account))
    payload = entry.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    age = entry.get("age_seconds")
    fresh = bool(entry.get("fresh") and age is not None and 0 <= age <= maximum)
    records = payload.get("records")
    records_valid = isinstance(records, list) and bool(records) and all(
        isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"] for row in records)
    coverage = payload.get("coverage") or {}
    reasons = []
    if account["state"] != "known":
        reasons.append("account_unknown")
    if not entry.get("present"):
        reasons.append("snapshot_missing_or_invalid")
    if payload.get("scope") != scope or payload.get("account") != account:
        reasons.append("scope_or_account_mismatch")
    if not fresh:
        reasons.append("freshness_unverified")
    if (payload.get("version") != 1 or entry.get("kind") != "scoped-capabilities"
            or not isinstance(coverage, dict) or coverage.get("complete") is not True
            or coverage.get("full_scope") is not True or not coverage.get("dataset_ids")
            or not records_valid or coverage.get("field_count") != len(records)):
        reasons.append("coverage_incomplete")
    evidence = {"state": "unknown" if reasons else "verified", "source": "private_snapshot",
                "scope": scope, "account": account, "coverage": coverage,
                "freshness": {"state": "fresh" if fresh else "unknown", "as_of": entry.get("created_at"),
                              "age_seconds": age, "max_age_seconds": maximum}, "reasons": reasons}
    return field_scope(records if records_valid else [], evidence)
