"""Private, account-bound catalog snapshots for offline capability evidence."""

from __future__ import annotations

import hashlib
import json


import cache_store
import machine_lib as ml
import simulation_capabilities

DISCLOSURE_DATASETS = frozenset({"fundamental2", "fundamental6", "analyst4"})


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


def required_field_scope(session, settings, required_fields):
    """Revalidate used datasets live; cached routes never prove field access.

    A complete private snapshot can locate datasets even after it expires. Each
    used catalog must still be fetched in full by the current authenticated
    session. Unrelated catalog failures cannot invalidate that narrower proof,
    and a partial proof is never saved as a complete offline capability scope.
    """
    scope, account = scope_settings(settings), simulation_capabilities.account_provenance(session)
    required = set(required_fields or ())
    entry = cache_store.read_entry(snapshot_path(scope, account))
    payload = entry.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    coverage = payload.get("coverage") or {}
    records = payload.get("records")
    routes = {}
    if (required and account.get("state") == "known"
            and entry.get("kind") == "scoped-capabilities" and payload.get("version") == 1
            and payload.get("scope") == scope and isinstance(payload.get("account"), dict)
            and payload["account"].get("fingerprint") == account.get("fingerprint")
            and isinstance(coverage, dict) and coverage.get("complete") is True
            and coverage.get("full_scope") is True and isinstance(records, list)
            and coverage.get("field_count") == len(records)):
        for row in records:
            if not isinstance(row, dict):
                continue
            dataset = row.get("dataset")
            dataset = dataset.get("id") if isinstance(dataset, dict) else dataset
            if row.get("id") in required and isinstance(dataset, str) and dataset:
                routes.setdefault(row["id"], set()).add(dataset)
    dataset_ids = (sorted({dataset for values in routes.values() for dataset in values})
                   if required and required <= routes.keys() else None)

    def fetch(datasets):
        evidence = {}
        rows = fetch_fields(session, region=scope["region"], universe=scope["universe"],
                            delay=scope["delay"], instrument_type=scope["instrumentType"],
                            dataset_ids=datasets, refresh=True, report=evidence)
        return rows, evidence

    def fresh_datasets(evidence):
        return {row["dataset"] for row in evidence.get("catalogs", [])
                if row.get("kind") == "datafields" and row.get("dataset")
                and row.get("complete") is True and not any(
                    row.get(key) for key in ("hit", "stale", "degraded", "error"))}

    rows, evidence = fetch(dataset_ids)
    # A previously mapped field can move. A successful live catalog that no
    # longer contains it triggers discovery rather than a false absence claim.
    if (dataset_ids and not required <= {row.get("id") for row in rows}
            and set(dataset_ids) <= fresh_datasets(evidence)):
        previous = evidence
        rows, evidence = fetch(None)
        evidence["routing_retry"] = previous
    fresh = fresh_datasets(evidence)
    verified_rows = []
    for row in rows:
        dataset = row.get("dataset")
        dataset = dataset.get("id") if isinstance(dataset, dict) else dataset
        if dataset in fresh:
            verified_rows.append(row)
    available = {row.get("id") for row in verified_rows}
    verified = bool(required) and required <= available
    evidence["required_fields"] = {
        "state": "verified" if verified else "unknown", "ids": sorted(required),
        "verified_ids": sorted(required & available),
        "unverified_ids": sorted(required - available),
        "routing_source": "private_snapshot_hint" if dataset_ids else "full_discovery"}
    if verified and evidence.get("state") != "verified":
        evidence["state"] = "verified"
        evidence["verification"] = "required_fields_only"
        return field_scope(verified_rows, evidence)
    return field_scope(rows, evidence)


def cached_field_scope(settings, max_age_seconds=None):
    """Read exactly one scope/account snapshot; unknown metadata is never verified."""
    scope, account = scope_settings(settings), simulation_capabilities.account_provenance()
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
    if account["state"] != "known":
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
