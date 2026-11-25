"""Evidence survives ordinary writes and legacy replay without becoming a pass."""

import json
from unittest.mock import patch

import pytest

import alpha_registry as registry
import registry_index


T0 = "2026-11-20T10:00:00+00:00"
T1 = "2026-11-21T10:00:00+00:00"
T2 = "2026-11-22T10:00:00+00:00"
T3 = "2026-11-23T10:00:00+00:00"
CHECKS = {
    "LOW_SHARPE": {"result": "PASS", "value": 2.0, "limit": 1.25},
    "LOW_GLB_EMEA_SHARPE": {"result": "FAIL", "value": 0.2, "limit": 1.0},
}


def _record(**changes):
    return {
        "alpha_id": "Alpha1", "expression": "rank(close)", "status": "COMPLETE",
        "settings": {"region": "GLB", "universe": "TOP3000", "simulationMode": "FULL"},
        "metrics": {"sharpe": 2.0, "fitness": 1.5}, "checks": CHECKS,
        **changes,
    }


def _write(path, at, record=None, **kwargs):
    with patch.object(registry, "_now", return_value=at):
        return registry.record_simulation([record or _record()], {}, registry_path=path, **kwargs)


def _latest(path):
    rows = registry.load_entries(path)
    with registry.indexed_registry(path) as index:
        assert index is not None
        assert index.entry("Alpha1") == rows[0]
    return rows[0]


def _raw(path, *rows):
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_metrics_only_refresh_retains_one_prior_snapshot_without_a_current_pass(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _write(path, T0, batch="campaign-1")
    first = _latest(path)
    assert first["checks_state"] == "observed"
    assert first["checks_observed_at"] == T0
    assert first["verdict"] == "rejected"

    refreshed = _record()
    refreshed.pop("checks")
    _write(path, T1, refreshed)
    row = _latest(path)
    assert row["checks"] == {} and row["failing_checks"] == []
    assert row["verdict"] == "watch"
    assert row["checks_state"] == "stale"
    assert row["checks_observed_at"] is None
    assert row["last_check_snapshot"] == {
        "checks": CHECKS, "observed_at": T0, "source": "simulate", "simulation_mode": "FULL",
        "request_identity": registry.evidence_state(first)["request_identity"],
    }
    assert row["first_checks_observed_at"] == T0
    assert row["measurement_observed_at"] == T0
    assert row["batch"] == "campaign-1"
    assert len(path.read_text().splitlines()) == 2
    assert _write(path, T2, refreshed)["appended"] == 0
    _write(path, T2, {**refreshed, "metrics": {"sharpe": 1.8, "fitness": 1.2}})
    changed = _latest(path)
    assert changed["last_check_snapshot"] == row["last_check_snapshot"]
    assert changed["measurement_observed_at"] == T2
    assert changed["first_measurement_observed_at"] == T0


def test_new_check_value_has_its_own_observation_time(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _write(path, T0)
    passed = {name: {**body, "result": "PASS", "value": 2.0} for name, body in CHECKS.items()}
    _write(path, T2, _record(checks=passed))
    row = _latest(path)
    assert row["first_checks_observed_at"] == T0
    assert row["checks_observed_at"] == T2
    assert row["verdict"] == "candidate"
    assert "last_check_snapshot" not in row
    # A repeated sync is not when this target value first became available.
    assert _write(path, T3, _record(checks=passed), source="sync")["appended"] == 0


def test_partial_payload_does_not_hide_a_previously_observed_check(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _write(path, T0)
    _write(path, T1, _record(checks={"LOW_SHARPE": CHECKS["LOW_SHARPE"]}))
    row = _latest(path)
    assert row["checks_state"] == "partial"
    assert row["verdict"] == "watch"
    assert row["checks_observed_at"] == T1
    assert row["last_check_snapshot"]["checks"] == CHECKS
    projection = registry_index.summary_fields(row)
    assert json.loads(projection["unresolved_checks"]) == ["LOW_GLB_EMEA_SHARPE"]
    assert _write(path, T2, _record(checks={"LOW_SHARPE": CHECKS["LOW_SHARPE"]}))["appended"] == 0


@pytest.mark.parametrize("changes", [
    {"settings": {"region": "GLB", "universe": "TOP3000", "simulationMode": "QUICK"}},
    {"expression": "rank(volume)"},
])
def test_prior_checks_do_not_cross_request_identity(tmp_path, changes):
    path = tmp_path / "ledger.jsonl"
    _write(path, T0)
    _write(path, T1, _record(checks={}, **changes))
    row = _latest(path)
    assert row["checks_state"] == "missing"
    assert "last_check_snapshot" not in row
    assert row["checks_observed_at"] is None
    assert row["measurement_observed_at"] == T1


@pytest.mark.parametrize("checks, expected", [
    (None, "watch"), ({}, "watch"),
    ({"LOW_SHARPE": {"result": "PENDING"}}, "watch"),
    ({"LOW_SHARPE": {"result": "UNKNOWN"}}, "watch"),
    ({"LOW_SHARPE": {"result": "FAIL"}}, "rejected"),
    ({"LOW_SHARPE": {"result": "ERROR"}}, "rejected"),
    ({"UNITS": {"result": "WARNING"}}, "candidate"),
    ({"LOW_SHARPE": {"result": " pass "}}, "candidate"),
])
def test_automatic_verdict_requires_observed_resolved_checks(checks, expected):
    assert registry.auto_verdict(_record(checks=checks)) == expected


def test_manual_provisional_label_remains_distinct_from_measured_quality(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _write(path, T0, _record(checks={}), verdict="candidate")
    _write(path, T1, _record(checks={}))
    row = _latest(path)
    assert row["verdict"] == "candidate" and row["verdict_manual"] is True
    assert row["checks_state"] == "missing"
    assert registry.auto_verdict(row) == "watch"


def test_relabel_and_dedupe_do_not_move_evidence_times(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _write(path, T0)
    with patch.object(registry, "_now", return_value=T1):
        labelled = registry.relabel(_latest(path), mechanism="reviewed mechanism")
        registry.record_entry(labelled, path)
    with patch.object(registry, "_now", return_value=T2):
        registry.record_entry(registry.mark_duplicate(_latest(path), "Alpha0"), path)
    row = _latest(path)
    assert row["recorded_at"] == T2
    assert row["first_recorded_at"] == T0
    assert row["measurement_observed_at"] == T0
    assert row["checks_observed_at"] == T0
    assert row["checks_source"] == "simulate"


def test_legacy_replay_reconstructs_provenance_without_rewriting_log(tmp_path):
    path = tmp_path / "ledger.jsonl"
    first = {**_record(), "key": "Alpha1", "recorded_at": T0, "source": "simulate"}
    label = {**first, "recorded_at": T1, "source": "relabel:derived", "mechanism": "M"}
    _raw(path, first, label)
    before = path.read_bytes()
    row = _latest(path)
    assert row["first_recorded_at"] == T0
    assert row["checks_observed_at"] == T0
    assert row["measurement_observed_at"] == T0
    assert path.read_bytes() == before
    # Rebuild and incremental replay must agree with the legacy JSONL reader.
    with registry.indexed_registry(path) as index:
        index.connection.execute("PRAGMA user_version = 0")
    assert _latest(path) == row
    missing = {**label, "recorded_at": T2, "source": "sync", "checks": {}}
    _raw(path, missing)
    row = _latest(path)
    assert row["checks_state"] == "stale"
    assert row["last_check_snapshot"]["observed_at"] == T0


def test_first_metadata_event_does_not_invent_measurement_or_label_time(tmp_path):
    path = tmp_path / "ledger.jsonl"
    label = {**_record(), "key": "Alpha1", "recorded_at": T0, "source": "relabel"}
    _raw(path, label)
    row = _latest(path)
    assert row["first_recorded_at"] == T0
    assert row["first_measurement_observed_at"] is None
    assert row["measurement_observed_at"] is None
    assert row["first_checks_observed_at"] is None
    assert row["checks_observed_at"] is None
    _raw(path, {**label, "source": "simulate", "recorded_at": T2})
    row = _latest(path)
    assert row["first_recorded_at"] == T0
    assert row["first_measurement_observed_at"] == T2
    assert row["first_checks_observed_at"] == T2


def test_index_failure_uses_same_folded_legacy_provenance(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _write(path, T0)
    with patch.object(registry_index, "open_index") as opening:
        opening.return_value.__enter__.return_value = None
        _write(path, T1, _record(checks={}))
    row = _latest(path)
    assert row["first_recorded_at"] == T0
    assert row["last_check_snapshot"]["observed_at"] == T0


def test_latest_row_export_preserves_known_provenance(tmp_path):
    original, exported = tmp_path / "original.jsonl", tmp_path / "exported.jsonl"
    _write(original, T0)
    _write(original, T1, _record(checks={}))
    row = _latest(original)
    _raw(exported, row)
    assert _latest(exported) == row


def test_legacy_automatic_candidate_is_not_a_pass_with_missing_checks(tmp_path):
    path = tmp_path / "ledger.jsonl"
    event = {**_record(checks={}), "key": "Alpha1", "source": "simulate",
             "verdict": "candidate", "recorded_at": T0}
    _raw(path, event)
    assert _latest(path)["verdict"] == "watch"
    assert json.loads(path.read_text())["verdict"] == "candidate"
    _raw(path, {**event, "source": "relabel", "verdict_manual": True, "recorded_at": T1})
    assert _latest(path)["verdict"] == "candidate"


def test_legacy_ledger_record_nondefault_verdict_is_explicit_manual_judgement(tmp_path):
    path = tmp_path / "ledger.jsonl"
    event = {**_record(checks={}), "key": "Alpha1", "source": "record",
             "verdict": "candidate", "recorded_at": T0}
    _raw(path, event)
    row = _latest(path)
    assert row["verdict"] == "candidate"
    assert row["verdict_manual"] is True
    assert row["checks_state"] == "missing"
    assert registry.auto_verdict(row) == "watch"
    assert "verdict_manual" not in json.loads(path.read_text())
