"""Offline regressions for fail-closed autonomous validation."""

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import alpha_registry
from research_app import pnl_snapshot, selection, sm_workers


def _full_doc(tmp_path):
    settings = {"simulationMode": "QUICK", "testPeriod": "P1Y"}
    arm = {"label": "a", "alpha_id": "Q1", "expression": "rank(close)", "settings": settings}
    path = tmp_path / "run-selection.json"
    manifest = selection.freeze(path, source="triage", rows=[arm], inputs={"run_id": "run"},
                                window={"train_only": True})
    return {"run_id": "run", "state": "AUTO_FULL_VALIDATE", "selection_frozen": True,
            "selection_path": str(path), "selection_id": manifest["selection_id"],
            "approved_arms": [{"label": "a", "alpha_id": "Q1"}], "arms": [arm]}


@pytest.mark.parametrize("bad_identity", ["extra_approval", "wrong_receipt", "wrong_result", "failed_result"])
def test_full_receipt_never_validates_a_different_arm(tmp_path, monkeypatch, bad_identity):
    """A FULL receipt must identify the one frozen arm and have succeeded."""
    doc = _full_doc(tmp_path)
    result = {"label": "a", "status": "COMPLETE", "alpha_id": "F1"}
    if bad_identity == "extra_approval":
        doc["approved_arms"].append({"label": "b", "alpha_id": "Q2"})
    elif bad_identity == "wrong_receipt":
        doc["full_records"] = [{"label": "b", "status": "COMPLETE", "alpha_id": "F1"}]
    elif bad_identity == "wrong_result":
        result["label"] = "b"
    else:
        result["status"] = "FAILED"
    simulate = Mock(return_value=[result])
    fetch = Mock()
    monkeypatch.setattr(sm_workers.ml, "simulate_alphas", simulate)
    monkeypatch.setattr(sm_workers.ml, "fetch_alpha_metrics", fetch)
    monkeypatch.setattr(sm_workers.simulation_capabilities, "account_provenance", lambda _: {})
    monkeypatch.setattr(sm_workers, "_persist_worker_progress", Mock())
    sm_workers._step_full_validate(SimpleNamespace(root=tmp_path), doc, object(), {})
    assert doc["state"] == "FAILED"
    fetch.assert_not_called()
    if bad_identity in {"extra_approval", "wrong_receipt"}:
        simulate.assert_not_called()


@pytest.mark.parametrize("test_sharpe", [None, float("nan"), float("inf"), True, "1.2", 0.0])
def test_full_validation_rejects_unproven_test_performance(tmp_path, monkeypatch, test_sharpe):
    """The platform must report a finite, positive Test Sharpe."""
    doc = _full_doc(tmp_path)
    monkeypatch.setattr(sm_workers.ml, "simulate_alphas", lambda *_, **__: [
        {"label": "a", "status": "COMPLETE", "alpha_id": "F1"},
    ])
    monkeypatch.setattr(sm_workers.ml, "fetch_alpha_metrics", lambda *_: {
        "id": "F1", "is": {"sharpe": 2.0}, "test": {"sharpe": test_sharpe},
    })
    monkeypatch.setattr(sm_workers.ml, "fetch_alpha_checks", lambda *_, **__: {
        "complete": True, "checks": {}, "self_correlation": {"result": "PASS", "value": 0.25},
    })
    monkeypatch.setattr(sm_workers.ml, "fetch_self_correlation", lambda *_: {"records": []})
    monkeypatch.setattr(sm_workers.ml, "fetch_alpha_recordsets", lambda *_, **__: {})
    monkeypatch.setattr(sm_workers.simulation_capabilities, "account_provenance", lambda _: {})
    monkeypatch.setattr(sm_workers, "_persist_worker_progress", Mock())
    sm_workers._step_full_validate(SimpleNamespace(root=tmp_path), doc, object(), {})
    assert doc["state"] == "REJECTED_AT_VALIDATION"
    assert doc["validation"]["failed_reason"] == (
        "nonpositive_test_sharpe" if type(test_sharpe) is float and test_sharpe == 0.0
        else "test_sharpe_unavailable"
    )


@pytest.mark.parametrize("yearly_state", ["pass", "fail", "unavailable", "insufficient_full_train_years", None])
def test_hydration_requires_explicit_yearly_train_pass(tmp_path, monkeypatch, yearly_state):
    """Missing and inconclusive yearly evidence must not become an approval."""
    doc = {"selected_arms": [{"label": "a", "alpha_id": "Q1"}]}
    monkeypatch.setattr(alpha_registry, "load_entries", lambda _: [{
        "alpha_id": "Q1", "out_of_sample": {
            "train": {"startDate": "2019-01-01"}, "test": {"startDate": "2023-01-01"},
        },
    }])
    monkeypatch.setattr(sm_workers.ml, "fetch_daily_pnl", Mock())
    monkeypatch.setattr(pnl_snapshot, "yearly_train_batch", lambda *_: [{"state": yearly_state}] if yearly_state else [])
    sm_workers._step_hydrate_pnl(SimpleNamespace(root=tmp_path), doc, object(), {})
    assert doc["state"] == "GATE_APPROVE_FULL"
    assert doc["hydration"]["by_label"]["a"]["gate_pass"] is (yearly_state == "pass")


def test_quick_never_uses_aggregate_metric_for_train(tmp_path, monkeypatch):
    """Aggregate metrics can coexist with a missing train split."""
    arm = {"label": "a", "expression": "rank(close)", "settings": {"testPeriod": "P1Y"}}
    doc = {"run_id": "run", "state": "AUTO_QUICK_SIM", "arms": [arm], "records": [{"label": "a"}]}
    monkeypatch.setattr(sm_workers.simulation_capabilities, "account_provenance", lambda _: {})
    monkeypatch.setattr(sm_workers.ml, "simulate_alphas", lambda *_, **__: [{
        "status": "COMPLETE", "alpha_id": "Q1", "metrics": {"sharpe": 2.5},
        "out_of_sample": {"test": {"sharpe": 5.0}},
    }])
    sm_workers._step_quick_sim(SimpleNamespace(root=tmp_path), doc, object(), {})
    assert doc["state"] == "FAILED"
    assert doc["rows"][0]["train_sharpe"] is None


def test_audit_uses_named_turnover_field_and_handles_flat_net_pnl():
    """Cost estimates use overlap, schema names and the observed turnover series."""
    days = [(date(2020, 1, 1) + timedelta(days=i)).isoformat() for i in range(20)]
    recordsets = {"series": {
        "daily-pnl": {"records": [{"date": day, "pnl": 100.0} for day in days]},
        "turnover": {"schema": {"properties": [{"name": "turnover"}, {"name": "date"}]},
                     "records": [[0.1, day] for day in days]},
    }}
    audit = sm_workers._compute_cost_audit({"is": {"bookSize": 1000.0}}, recordsets)
    assert audit["joint_sessions"] == 20
    assert audit["scenarios"][0] == {"cost_bps": 5, "net_sharpe": None, "net_margin_bps": 9995.0}
    recordsets["series"]["turnover"]["complete"] = False
    assert sm_workers._compute_cost_audit({"is": {"bookSize": 1000.0}}, recordsets) is None


def test_quick_and_full_use_current_simulator_signature(tmp_path, monkeypatch):
    """Both paths can enter the real dependency with no credentials."""
    simulate = sm_workers.ml.simulate_alphas
    modes = []

    def offline(session, expressions, **kwargs):
        modes.append(kwargs["settings"]["simulationMode"])
        return simulate(session, expressions, dry_run=True, **kwargs)

    monkeypatch.setattr(sm_workers.ml, "simulate_alphas", offline)
    monkeypatch.setattr(sm_workers.simulation_capabilities, "account_provenance", lambda _: {})
    monkeypatch.setattr(sm_workers, "_persist_worker_progress", Mock())
    engine = SimpleNamespace(root=tmp_path)
    doc = _full_doc(tmp_path)
    quick = {**doc, "state": "AUTO_QUICK_SIM", "records": [{"label": "a", "status": "QUEUED"}]}
    sm_workers._step_quick_sim(engine, quick, None, {})
    sm_workers._step_full_validate(engine, doc, None, {})
    assert modes == ["QUICK", "FULL"]


def test_checkpoint_keeps_native_and_single_transport_receipts(monkeypatch):
    """One callback captures complete native batches and single polling results."""
    doc = {"state": "AUTO_QUICK_SIM", "records": [{"label": "a"}, {"label": "b"}]}
    persist = Mock()
    monkeypatch.setattr(sm_workers, "_persist_worker_progress", persist)
    checkpoint = sm_workers._simulation_checkpoint(None, doc, "records")
    checkpoint("multi_submitting", {"index": 0, "records": [
        {"index": 0, "label": "a", "multi_request_id": "request"},
        {"index": 1, "label": "b", "multi_request_id": "request"},
    ]})
    assert [row["multi_request_id"] for row in doc["records"]] == ["request", "request"]
    assert persist.call_args.kwargs == {"merge_on_cancel": False}
    checkpoint("submitted", {"index": 0, "label": "a", "progress_url": "simulation-a"})
    assert doc["records"][0]["progress_url"] == "simulation-a"
    assert persist.call_args.kwargs == {"merge_on_cancel": True}
    checkpoint("completed", {"index": 0, "label": "a", "alpha_id": "Q1"})
    assert doc["records"][0] == {"label": "a", "alpha_id": "Q1"}


def test_quick_retries_do_not_duplicate_ledger_rows(tmp_path, monkeypatch):
    """Replaying one completed arm while another polls produces one ledger entry."""
    arms = [{"label": x, "expression": f"rank({field})", "settings": {"testPeriod": "P1Y"}}
            for x, field in [("a", "close"), ("b", "volume")]]
    doc = {"run_id": "run", "state": "AUTO_QUICK_SIM", "arms": arms,
           "records": [{"label": "a", "status": "QUEUED"}, {"label": "b", "status": "QUEUED"}]}
    monkeypatch.setattr(sm_workers.simulation_capabilities, "account_provenance", lambda _: {})
    monkeypatch.setattr(sm_workers.ml, "simulate_alphas", lambda *_, **__: [
        {"label": "a", "status": "COMPLETE", "alpha_id": "Q1", "metrics": {"sharpe": 1.2}},
        {"label": "b", "status": "PENDING", "submission_attempted": True},
    ])
    engine = SimpleNamespace(root=tmp_path)
    sm_workers._step_quick_sim(engine, doc, None, {})
    sm_workers._step_quick_sim(engine, doc, None, {})
    raw_ledger = tmp_path / "alpha-registry.jsonl"
    assert len(raw_ledger.read_text(encoding="utf-8").splitlines()) == 1
