"""Batch admission may use positive field proofs without approving unknown fields."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import cache_store
import machine_lib as ml
import simulation_capabilities
from research_app import capabilities


SETTINGS = {"region": "USA", "universe": "TOP3000", "delay": 1, "instrumentType": "EQUITY"}
FIELDS = [{"id": "field_a", "type": "MATRIX", "dataset": "dataset_a"},
          {"id": "field_b", "type": "VECTOR", "dataset": {"id": "dataset_b"}},
          {"id": "field_c", "type": "MATRIX", "dataset": "dataset_c"}]
COVERAGE = {"complete": True, "full_scope": True,
            "dataset_ids": ["dataset_a", "dataset_b", "dataset_c"], "field_count": len(FIELDS)}


def lookup(_session, field, _scope, *, deadline):
    matches = [dict(row) for row in FIELDS if row["id"] == field]
    return {"matches": matches, "returned_count": len(matches),
            "reported_count": 1000, "complete": False}


@pytest.fixture
def context(tmp_path, monkeypatch):
    elapsed = [0.0]
    started = datetime(2026, 11, 25, tzinfo=timezone.utc)
    policy = {"enabled": True, "refresh": False, "max_age": 3600}
    session = SimpleNamespace(auth=("live-session", "unused"), _brain_account_id="live-principal")
    monkeypatch.setenv("BRAIN_USERNAME", "different-configured-account")
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    monkeypatch.setattr(ml, "session_cache_path", lambda: tmp_path / "session.json")
    monkeypatch.setattr(ml, "metadata_cache_policy", lambda: dict(policy))
    monkeypatch.setattr(capabilities.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(capabilities.time, "sleep", lambda delay: elapsed.__setitem__(0, elapsed[0] + delay))
    monkeypatch.setattr(cache_store, "utc_now", lambda: started + timedelta(seconds=elapsed[0]))
    monkeypatch.setattr(ml, "search_datafield", lookup)
    monkeypatch.setattr(ml, "request_json", lambda *_a, **_k: pytest.fail("unexpected network"))
    return SimpleNamespace(session=session, account=simulation_capabilities.account_provenance(session),
                           elapsed=elapsed, policy=policy)


def assert_unknown(result):
    assert result["evidence"]["state"] == "unknown"
    assert result["records"] == []
    assert result["known_fields"] is None
    assert result["field_types"] is None
    assert result["market_fields"] is None


def test_partial_records_expose_only_positive_proofs_without_loosening_old_interface(context):
    required = {row["id"] for row in FIELDS} | {"unknown_field"}
    result = capabilities.required_field_scope(context.session, SETTINGS, required)

    assert_unknown(result)
    assert result["verified_records"] == FIELDS
    assert result["evidence"]["required_fields"]["unverified_ids"] == ["unknown_field"]
    assert result["evidence"]["coverage"] == {"complete": False, "full_scope": False, "field_count": 3}


@pytest.mark.parametrize("memory_cache", [False, True])
def test_expired_budget_only_reads_fresh_positive_proofs(context, monkeypatch, memory_cache):
    required = {row["id"] for row in FIELDS}
    cache = {} if memory_cache else None
    first = capabilities.required_field_scope(context.session, SETTINGS, required, proof_cache=cache)
    assert first["known_fields"] == required
    paths = [capabilities._field_proof_path(context.account, SETTINGS, field) for field in sorted(required)]
    before = [path.read_bytes() for path in paths]
    context.elapsed[0] = 10
    monkeypatch.setattr(ml, "search_datafield", lambda *_a, **_k: pytest.fail("budget exhausted"))

    result = capabilities.required_field_scope(
        context.session, SETTINGS, required | {"absent_field"}, deadline=10, proof_cache=cache)

    assert_unknown(result)
    assert result["verified_records"] == FIELDS
    assert result["evidence"]["required_fields"]["budget_deferred_ids"] == ["absent_field"]
    assert result["evidence"]["as_of"] == first["evidence"]["as_of"]
    assert [path.read_bytes() for path in paths] == before


@pytest.mark.parametrize("elapsed,maximum", [(3601, 7200), (120, 60)])
def test_expired_budget_cannot_reuse_expired_or_overage_proofs(context, monkeypatch, elapsed, maximum):
    cache = {}
    capabilities.required_field_scope(context.session, SETTINGS, {"field_a"}, proof_cache=cache)
    path = capabilities._field_proof_path(context.account, SETTINGS, "field_a")
    before = path.read_bytes()
    context.elapsed[0] = elapsed
    context.policy["max_age"] = maximum
    monkeypatch.setattr(ml, "search_datafield", lambda *_a, **_k: pytest.fail("budget exhausted"))

    result = capabilities.required_field_scope(
        context.session, SETTINGS, {"field_a"}, deadline=elapsed, proof_cache=cache)

    assert_unknown(result)
    assert result["verified_records"] == []
    assert result["evidence"]["required_fields"]["budget_deferred_ids"] == ["field_a"]
    assert path.read_bytes() == before


def test_partial_records_exclude_proofs_expiring_during_a_slow_later_lookup(context, monkeypatch):
    context.policy["max_age"] = 60
    cache = {}
    capabilities.required_field_scope(context.session, SETTINGS, {"field_a"}, proof_cache=cache)
    context.elapsed[0] = 50
    reads = []

    def slow_lookup(session, field, scope, *, deadline):
        reads.append((field, deadline))
        context.elapsed[0] += 27.4
        return lookup(session, field, scope, deadline=deadline)

    monkeypatch.setattr(ml, "search_datafield", slow_lookup)
    result = capabilities.required_field_scope(
        context.session, SETTINGS, {"field_a", "field_b"}, proof_cache=cache)

    assert_unknown(result)
    assert result["verified_records"] == [FIELDS[1]]
    assert result["evidence"]["required_fields"]["expired_ids"] == ["field_a"]
    assert reads == [("field_b", 80)]


@pytest.mark.parametrize("change", ["account", "unknown_account", "region", "universe", "delay", "instrumentType"])
def test_positive_proofs_never_cross_accounts_or_scopes_at_expired_budget(context, monkeypatch, change):
    cache = {}
    capabilities.required_field_scope(context.session, SETTINGS, {"field_a"}, proof_cache=cache)
    session, settings = context.session, dict(SETTINGS)
    if change == "account":
        session = SimpleNamespace(_brain_account_id="another-principal")
    elif change == "unknown_account":
        session = SimpleNamespace()
    else:
        settings[change] = {"region": "EUR", "universe": "TOP500", "delay": 0,
                            "instrumentType": "OTHER"}[change]
    monkeypatch.setattr(ml, "search_datafield", lambda *_a, **_k: pytest.fail("budget exhausted"))

    result = capabilities.required_field_scope(session, settings, {"field_a"}, deadline=0, proof_cache=cache)

    assert_unknown(result)
    assert result["verified_records"] == []


def test_full_snapshot_uses_explicit_authenticated_account_without_changing_offline_default(context):
    capabilities.save_snapshot(context.session, SETTINGS, FIELDS, COVERAGE)
    configured = simulation_capabilities.account_provenance()
    assert configured["fingerprint"] != context.account["fingerprint"]

    assert_unknown(capabilities.cached_field_scope(SETTINGS))
    current = capabilities.cached_field_scope(SETTINGS, account=context.account)

    assert current["evidence"]["state"] == "verified"
    assert current["records"] == FIELDS
    assert current["evidence"]["account"] == context.account
    for account in ({}, {"state": "unknown", "fingerprint": context.account["fingerprint"]},
                    {"state": "known", "fingerprint": None}):
        assert_unknown(capabilities.cached_field_scope(SETTINGS, account=account))


@pytest.mark.parametrize("change", ["account", "scope"])
def test_explicit_account_does_not_weaken_snapshot_payload_validation(context, change):
    path = Path(capabilities.save_snapshot(context.session, SETTINGS, FIELDS, COVERAGE))
    account, settings = context.account, dict(SETTINGS)
    if change == "account":
        account = simulation_capabilities.account_provenance(SimpleNamespace(_brain_account_id="another-principal"))
    else:
        settings["universe"] = "TOP500"
    target = capabilities.snapshot_path(settings, account)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())

    result = capabilities.cached_field_scope(settings, account=account)

    assert_unknown(result)
    assert "scope_or_account_mismatch" in result["evidence"]["reasons"]


@pytest.mark.parametrize("elapsed,maximum", [(3601, 7200), (120, 60)])
def test_explicit_account_keeps_original_expiry_and_callers_stricter_maximum(context, elapsed, maximum):
    capabilities.save_snapshot(context.session, SETTINGS, FIELDS, COVERAGE)
    context.elapsed[0] = elapsed

    result = capabilities.cached_field_scope(SETTINGS, maximum, account=context.account)

    assert_unknown(result)
    assert "freshness_unverified" in result["evidence"]["reasons"]
