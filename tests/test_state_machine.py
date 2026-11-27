"""Offline tests for the v4 4-gate persistent state machine and workers."""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import requests

import alpha_registry
import cache_store
import machine_lib as ml
import simulation_capabilities
from research_app import buffer_cmd, experiments, sm_workers, state_machine, validation
from research_app.runtime import CliError

OFFICIAL_DESCRIPTION = (
    "Idea:\n- Cross-sectional rank of closing price captures short-term mean reversion.\n\n"
    "Rationale for data used:\n- close: split-adjusted daily close price.\n\n"
    "Rationale for operators used:\n- rank: normalizes values across the stock universe."
)


def _explore_spec():
    return {
        "name": "rev-explore",
        "track": "REGULAR",
        "design": "explore",
        "idea": "Short-term reversal rank test",
        "mechanism": "mean_reversion",
        "settings": {"simulationMode": "QUICK"},
        "arms": [
            {"label": "c1", "expression": "rank(close)", "idea": "Close rank control"},
            {"label": "d1", "expression": "rank(volume)", "role": "diagnostic", "idea": "Volume diagnostic"},
        ],
    }


def _dummy_session():
    s = SimpleNamespace(_brain_account_id="ACC_TEST", close=lambda: None)
    return s


def _make_recordsets():
    start = date(2020, 1, 1)
    days = [(start + timedelta(days=i)).isoformat() for i in range(30)]
    pnl_rows = [[d, 100.0 + (i % 5) * 15.0] for i, d in enumerate(days)]
    to_rows = [[d, 0.05] for d in days]
    return {
        "series": {
            "daily-pnl": {"schema": {"properties": [{"name": "date"}, {"name": "pnl"}]}, "records": pnl_rows},
            "turnover": {"schema": {"properties": [{"name": "date"}, {"name": "turnover"}]}, "records": to_rows},
        }
    }


def test_full_four_gate_happy_path_and_train_test_firewall(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    engine = state_machine.StateMachineEngine(tmp_path)
    proposed = engine.propose(_explore_spec())
    run_id = proposed["run_id"]
    assert proposed["state"] == "AUTO_PREFLIGHT"
    assert proposed["metrics_scope"] == "train"
    assert proposed["test_view_open"] is False

    acc = {"fingerprint": "fp-1", "quota": {"remaining": 200, "blocked": False}}
    quick_res = [
        {"index": 0, "label": "c1", "status": "COMPLETE", "alpha_id": "Q1", "submission_attempted": True,
         "metrics": {"sharpe": 2.4, "fitness": 1.6, "turnover": 0.15},
         "out_of_sample": {"train": {"sharpe": 2.5, "fitness": 1.7, "startDate": "2019-01-01"},
                           "test": {"sharpe": 9.9, "startDate": "2023-01-01"}}},
        {"index": 1, "label": "d1", "status": "COMPLETE", "alpha_id": "Q2", "submission_attempted": True,
         "metrics": {"sharpe": 1.1, "fitness": 0.8, "turnover": 0.2},
         "out_of_sample": {"train": {"sharpe": 1.2, "fitness": 0.9, "startDate": "2019-01-01"},
                           "test": {"sharpe": -5.0, "startDate": "2023-01-01"}}},
    ]
    full_res = [{"index": 0, "label": "c1", "status": "COMPLETE", "alpha_id": "F1", "submission_attempted": True}]
    full_payload = {
        "id": "F1",
        "status": "COMPLETE",
        "regular": {"code": "rank(close)", "description": OFFICIAL_DESCRIPTION},
        "settings": {"region": "USA", "universe": "TOP3000", "simulationMode": "FULL"},
        "is": {"sharpe": 2.3, "fitness": 1.6, "turnover": 0.15, "bookSize": 20000000},
        "train": {"sharpe": 2.5, "fitness": 1.7, "turnover": 0.14},
        "test": {"sharpe": 1.4, "fitness": 1.0, "turnover": 0.16},
    }
    checks_ok = {
        "alpha_id": "F1",
        "complete": True,
        "self_correlation": {"result": "PASS", "value": 0.35},
        "checks": {"LOW_SHARPE": {"result": "PASS", "value": 2.3, "limit": 1.58},
                   "SELF_CORRELATION": {"result": "PASS", "value": 0.35, "limit": 0.7}},
    }

    days_pnl = {f"2020-01-{i + 1:02d}": 10.0 + (i % 3) for i in range(25)}

    with mock.patch.object(simulation_capabilities, "account_provenance", return_value=acc), \
            mock.patch.object(validation, "_preflight_lint", return_value={"ok": True}), \
            mock.patch.object(ml, "simulate_alphas", side_effect=[quick_res, full_res]), \
            mock.patch.object(ml, "fetch_daily_pnl", return_value=days_pnl), \
            mock.patch.object(ml, "fetch_alpha_metrics", side_effect=[
                full_payload,
                {**full_payload, "regular": {"code": "rank(close)", "description": OFFICIAL_DESCRIPTION}},
            ]), \
            mock.patch.object(ml, "fetch_alpha_checks", return_value=checks_ok), \
            mock.patch.object(ml, "fetch_self_correlation", return_value={"records": []}), \
            mock.patch.object(ml, "fetch_alpha_recordsets", return_value=_make_recordsets()), \
            mock.patch.object(ml, "set_alpha_metadata", return_value=full_payload), \
            mock.patch.object(ml, "submit_alpha", return_value={"submitted": True, "submission": "CONFIRMED", "date_submitted": "2026-11-27T00:00:00Z"}):

        # AUTO_PREFLIGHT -> AUTO_QUICK_SIM -> GATE_TRIAGE
        d1 = sm_workers.advance_run_once(engine, run_id, _dummy_session)
        assert d1["state"] == "AUTO_QUICK_SIM"
        d2 = sm_workers.advance_run_once(engine, run_id, _dummy_session)
        assert d2["state"] == "GATE_TRIAGE"

        # Protocol Train/Test firewall holds before freeze.
        view_pre = engine.inspect_run(run_id)
        assert view_pre["metrics_scope"] == "train"
        assert view_pre["test_view_open"] is False
        assert "validation" not in view_pre
        assert all("test" not in row and "out_of_sample" not in row for row in view_pre["rows"])
        assert len(alpha_registry.load_entries(str(tmp_path / "alpha-registry.jsonl"))) == 2

        # Gate 1: Reject hydrating diagnostic arm; then hydrate candidate arm.
        with pytest.raises(CliError) as bad_diag:
            engine.act(run_id, "hydrate-pnl", {"arms": ["d1"]})
        assert bad_diag.value.code == "INVALID_INPUT"

        g1 = engine.act(run_id, "hydrate-pnl", {"arms": ["c1"]})
        assert g1["state"] == "AUTO_HYDRATE_PNL"
        d3 = sm_workers.advance_run_once(engine, run_id, _dummy_session)
        assert d3["state"] == "GATE_APPROVE_FULL"
        assert engine.inspect_run(run_id)["test_view_open"] is False

        # Gate 2: Approve FULL (freezes selection manifest & opens test view) -> AUTO_FULL_VALIDATE -> GATE_DESCRIBE
        g2 = engine.act(run_id, "approve-full")
        assert g2["state"] == "AUTO_FULL_VALIDATE"
        assert g2["selection_frozen"] is True
        assert Path(g2["selection_path"]).is_file()

        d4 = sm_workers.advance_run_once(engine, run_id, _dummy_session)
        assert d4["state"] == "GATE_DESCRIBE"
        view_post = engine.inspect_run(run_id)
        assert view_post["test_view_open"] is True
        assert view_post["metrics_scope"] == "train+test"
        assert view_post["validation"]["test_sharpe"] == 1.4
        assert len(view_post["cost_audit"]["scenarios"]) == 3

        # Gate 3: Reject short prose; accept official Power Pool template -> AUTO_BUFFER -> GATE_CONFIRM_SUBMIT
        with pytest.raises(CliError) as bad_meta:
            engine.act(run_id, "describe", {"name": "s", "category": "PRICE_VOLUME", "tags": ["rev"], "description": "too short"})
        assert bad_meta.value.code == "INVALID_INPUT"

        g3 = engine.act(run_id, "describe", {
            "name": "RevRank_v1",
            "category": "PRICE_VOLUME",
            "tags": ["reversion"],
            "description": OFFICIAL_DESCRIPTION,
        })
        assert g3["state"] == "AUTO_BUFFER"
        d5 = sm_workers.advance_run_once(engine, run_id, _dummy_session)
        assert d5["state"] == "GATE_CONFIRM_SUBMIT"
        assert "F1" in buffer_cmd.read_seeds(str(tmp_path / buffer_cmd.SEEDS_FILENAME))
        assert len(buffer_cmd.read_events(str(tmp_path / buffer_cmd.BUFFER_FILENAME))) == 1

        # Gate 4: Requires --yes and BRAIN_ALLOW_AGENT_SUBMIT=true
        with pytest.raises(CliError) as no_yes:
            engine.act(run_id, "submit", {})
        assert no_yes.value.code == "CONFIRM_REQUIRED"

        monkeypatch.delenv(ml.SUBMIT_UNLOCK_ENV, raising=False)
        with mock.patch.object(ml, "agent_submit_allowed", return_value=False):
            with pytest.raises(CliError) as disabled:
                engine.act(run_id, "submit", {"yes": True})
            assert disabled.value.code == "SUBMIT_DISABLED"

        with mock.patch.object(ml, "agent_submit_allowed", return_value=True):
            g4 = engine.act(run_id, "submit", {"yes": True})
        assert g4["state"] == "AUTO_SUBMITTING"

        d6 = sm_workers.advance_run_once(engine, run_id, _dummy_session)
        assert d6["state"] == "SUBMITTED"


def test_quota_floor_pauses_quick_sim_but_not_full_validate(tmp_path, monkeypatch):
    import simulation_slots

    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    engine = state_machine.StateMachineEngine(tmp_path)
    engine.update_config({"quota_floor": 10})
    proposed = engine.propose(_explore_spec())
    run_id = proposed["run_id"]

    # Force into AUTO_QUICK_SIM
    doc = engine.load_run_raw(run_id)
    doc["state"] = "AUTO_QUICK_SIM"
    engine.save_run(doc)

    low_quota_acc = {"fingerprint": "fp-1"}
    simulation_slots.observe_quota(low_quota_acc, {
        "x-ratelimit-limit": "200",
        "x-ratelimit-remaining": "8",
        "x-ratelimit-reset": "3600",
    })
    with mock.patch.object(simulation_capabilities, "account_provenance", return_value=low_quota_acc), \
            mock.patch.object(ml, "simulate_alphas", side_effect=AssertionError("quota floor must pause quick sim")):
        after_q = sm_workers.advance_run_once(engine, run_id, _dummy_session)
    assert after_q["state"] == "AUTO_QUICK_SIM"
    assert after_q["next_wake_at"] is not None

    # AUTO_FULL_VALIDATE ignores soft quota_floor unless blocked=True.
    doc = engine.load_run_raw(run_id)
    doc["state"] = "AUTO_FULL_VALIDATE"
    doc["approved_arms"] = [{"label": "c1", "alpha_id": "Q1"}]
    engine.save_run(doc)

    full_res = [{"index": 0, "label": "c1", "status": "COMPLETE", "alpha_id": "F1", "submission_attempted": True}]
    full_payload = {
        "id": "F1", "status": "COMPLETE", "regular": {"code": "rank(close)", "description": OFFICIAL_DESCRIPTION},
        "settings": {"region": "USA", "universe": "TOP3000", "simulationMode": "FULL"},
        "is": {"sharpe": 2.0, "fitness": 1.3}, "train": {"sharpe": 2.0}, "test": {"sharpe": 1.2},
    }
    checks_ok = {"alpha_id": "F1", "complete": True, "self_correlation": {"result": "PASS", "value": 0.2},
                 "checks": {"LOW_SHARPE": {"result": "PASS"}}}
    with mock.patch.object(simulation_capabilities, "account_provenance", return_value=low_quota_acc), \
            mock.patch.object(ml, "simulate_alphas", return_value=full_res) as sim_mock, \
            mock.patch.object(ml, "fetch_alpha_metrics", return_value=full_payload), \
            mock.patch.object(ml, "fetch_alpha_checks", return_value=checks_ok), \
            mock.patch.object(ml, "fetch_self_correlation", return_value={"records": []}), \
            mock.patch.object(ml, "fetch_alpha_recordsets", return_value={}):
        after_f = sm_workers.advance_run_once(engine, run_id, _dummy_session)
    sim_mock.assert_called_once()
    assert after_f["state"] == "GATE_DESCRIBE"


def test_three_unchanged_defers_trips_preflight_blocked_and_recovers(tmp_path):
    engine = state_machine.StateMachineEngine(tmp_path)
    engine.update_config({"max_unchanged_defers": 3})
    run_id = engine.propose(_explore_spec())["run_id"]

    with mock.patch.object(simulation_capabilities, "account_provenance", side_effect=ml.BrainError("transient")):
        for _ in range(4):
            doc = sm_workers.advance_run_once(engine, run_id, _dummy_session)
    assert doc["state"] == "GATE_PREFLIGHT_BLOCKED"
    assert doc["preflight_block"]["defers"] >= 3

    recovered = engine.act(run_id, "retry-preflight")
    assert recovered["state"] == "AUTO_PREFLIGHT"
    assert recovered["unchanged_defers"] == 0
    assert recovered["preflight_block"] is None


@pytest.mark.parametrize("case", ["fail_check", "self_corr", "negative_test_sharpe", "pp_budget"])
def test_validation_auto_rejects_failing_candidates(tmp_path, case):
    engine = state_machine.StateMachineEngine(tmp_path)
    spec = _explore_spec()
    if case == "pp_budget":
        spec["track"] = "POWER_POOL"
        spec["arms"] = [{"label": "c1", "expression": "rank(a) + rank(b) + rank(c) + rank(d)", "idea": "4 fields"}]
    run_id = engine.propose(spec)["run_id"]

    doc = engine.load_run_raw(run_id)
    doc["state"] = "AUTO_FULL_VALIDATE"
    doc["selection_frozen"] = True
    doc["approved_arms"] = [{"label": "c1", "alpha_id": "Q1"}]
    engine.save_run(doc)

    full_res = [{"index": 0, "label": "c1", "status": "COMPLETE", "alpha_id": "F1", "submission_attempted": True}]
    test_sharpe = -0.2 if case == "negative_test_sharpe" else 1.5
    full_payload = {
        "id": "F1", "status": "COMPLETE", "regular": {"code": spec["arms"][0]["expression"]},
        "settings": {"region": "USA", "universe": "TOP3000", "simulationMode": "FULL"},
        "is": {"sharpe": 2.0, "fitness": 0.63 if case == "fail_check" else 1.5},
        "train": {"sharpe": 2.1}, "test": {"sharpe": test_sharpe},
    }
    checks = {"LOW_SHARPE": {"result": "PASS"}}
    if case == "fail_check":
        checks["LOW_FITNESS"] = {"result": "FAIL", "value": 0.63, "limit": 1.0}
    sc_payload = {"records": [["X", "X", 0, 0, 0, 0.85, 1.5]]} if case == "self_corr" else {"records": []}
    acc = {"fingerprint": "fp-1", "quota": {"remaining": 50, "blocked": False}}

    with mock.patch.object(simulation_capabilities, "account_provenance", return_value=acc), \
            mock.patch.object(ml, "simulate_alphas", return_value=full_res), \
            mock.patch.object(ml, "fetch_alpha_metrics", return_value=full_payload), \
            mock.patch.object(ml, "fetch_alpha_checks", return_value={"alpha_id": "F1", "complete": True, "checks": checks}), \
            mock.patch.object(ml, "fetch_self_correlation", return_value=sc_payload), \
            mock.patch.object(ml, "fetch_alpha_recordsets", return_value={}):
        after = sm_workers.advance_run_once(engine, run_id, _dummy_session)
    assert after["state"] == "REJECTED_AT_VALIDATION"


def test_uncertain_submit_reconciles_without_reposting(tmp_path):
    engine = state_machine.StateMachineEngine(tmp_path)
    run_id = engine.propose(_explore_spec())["run_id"]
    doc = engine.load_run_raw(run_id)
    doc.update({"state": "AUTO_SUBMITTING", "full_alpha_id": "F1", "selection_frozen": True})
    engine.save_run(doc)

    def timeout_after_post(_session, _aid, **kwargs):
        kwargs["on_event"]({"event": "submit_posting"})
        raise requests.Timeout("lost response")

    with mock.patch.object(ml, "submit_alpha", side_effect=timeout_after_post) as post:
        mid = sm_workers.advance_run_once(engine, run_id, _dummy_session)
    assert mid["state"] == "AUTO_SUBMITTING"
    assert mid["submit_attempt"]["state"] == "UNCERTAIN"
    assert post.call_count == 1

    # Reconcile readback finds ACTIVE + dateSubmitted without ever calling submit_alpha again.
    with mock.patch.object(ml, "submit_alpha", side_effect=AssertionError("must never double-POST")), \
            mock.patch.object(ml, "fetch_alpha_metrics", return_value={"id": "F1", "status": "ACTIVE", "dateSubmitted": "2026-11-27T01:02:03Z"}):
        done = sm_workers.advance_run_once(engine, run_id, _dummy_session)
    assert done["state"] == "SUBMITTED"
    assert done["submit_attempt"]["state"] == "CONFIRMED"

    # Separate run: read_alpha_submission_status -> rejected
    run2 = engine.propose({**_explore_spec(), "name": "uncertain-reject", "arms": [{"label": "c1", "expression": "rank(open)", "idea": "x"}]})["run_id"]
    d2 = engine.load_run_raw(run2)
    d2.update({"state": "AUTO_SUBMITTING", "full_alpha_id": "F2", "submit_attempt": {"state": "UNCERTAIN", "post_attempted": True}})
    engine.save_run(d2)
    with mock.patch.object(ml, "submit_alpha", side_effect=AssertionError("must never double-POST")), \
            mock.patch.object(ml, "fetch_alpha_metrics", return_value={"id": "F2", "status": "UNSUBMITTED"}), \
            mock.patch.object(ml, "read_alpha_submission_status", return_value={"state": "rejected", "failed_checks": ["LOW_FITNESS"]}):
        rej = sm_workers.advance_run_once(engine, run2, _dummy_session)
    assert rej["state"] == "REJECTED_AT_SUBMIT"


def test_reconcile_orphan_slots_harvests_completed_alpha(tmp_path, monkeypatch):
    import simulation_slots

    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    engine = state_machine.StateMachineEngine(tmp_path)
    journal_path, _lock_path = simulation_slots._paths("fp-orphan")
    Path(journal_path).parent.mkdir(parents=True, exist_ok=True)
    cache_store.write_json_atomic(journal_path, {
        "version": 1,
        "account_fingerprint": "fp-orphan",
        "reservations": {
            "slot-1": {
                "state": "accepted",
                "progress_url": "https://api.worldquantbrain.com/simulations/orphan1",
                "created_at": 1.0,
            }
        },
    })
    alpha_payload = {
        "id": "ORPHAN1",
        "status": "COMPLETE",
        "regular": {"code": "rank(close)"},
        "settings": {"region": "USA", "universe": "TOP3000"},
        "is": {"sharpe": 2.0, "fitness": 1.4},
    }
    with mock.patch.object(ml, "poll_simulation", return_value={"status": "COMPLETE", "alpha": "ORPHAN1"}), \
            mock.patch.object(ml, "fetch_alpha_metrics", return_value=alpha_payload):
        res = sm_workers.reconcile_orphan_slots(engine, _dummy_session, max_age_min=0.0)
    assert res["harvested"] == ["ORPHAN1"]
    entries = alpha_registry.load_entries(str(tmp_path / "alpha-registry.jsonl"))
    assert [e["alpha_id"] for e in entries] == ["ORPHAN1"]


def test_cancel_and_migrate_workspace(tmp_path):
    engine = state_machine.StateMachineEngine(tmp_path)
    run_id = engine.propose(_explore_spec())["run_id"]
    doc = engine.load_run_raw(run_id)
    doc["state"] = "AUTO_QUICK_SIM"
    doc["records"][0].update({"status": "PENDING", "progress_url": "https://api.worldquantbrain.com/simulations/live1"})
    engine.save_run(doc)

    cancelling = engine.cancel(run_id, reason="superseded")
    assert cancelling["state"] == "AUTO_CANCELLING"
    with mock.patch.object(ml, "cancel_simulation") as cancel_mock:
        cancelled = sm_workers.advance_run_once(engine, run_id, _dummy_session)
    cancel_mock.assert_called_once_with(mock.ANY, "https://api.worldquantbrain.com/simulations/live1")
    assert cancelled["state"] == "CANCELLED"

    # Legacy workspace migration (--dry-run and live archive)
    backlog = tmp_path / "backlog"
    results = tmp_path / "backlog-results"
    backlog.mkdir()
    results.mkdir()
    (results / "old-batch.json").write_text(json.dumps({
        "name": "legacy-probe",
        "track": "REGULAR",
        "complete": True,
        "arms": [{"label": "a1", "expression": "rank(vwap)", "settings": {}}],
        "records": [{"label": "a1", "status": "COMPLETE", "alpha_id": "LEG1"}],
        "rows": [{"label": "a1", "alpha_id": "LEG1", "train_sharpe": 1.8}],
        "execution": {"total": 1, "accepted": 1, "finished": 1},
    }), encoding="utf-8")
    (backlog / "queued.json").write_text(json.dumps({
        "name": "legacy-queued",
        "idea": "Queued idea",
        "design": "explore",
        "arms": [{"label": "q1", "expression": "rank(high)"}],
    }), encoding="utf-8")

    dry = engine.migrate_workspace(dry_run=True)
    assert dry["dry_run"] is True
    assert len(dry["migrated"]) == 2
    assert backlog.exists() and results.exists()

    live = engine.migrate_workspace(dry_run=False)
    assert len(live["migrated"]) == 2
    assert not backlog.exists() and not results.exists()
    assert (tmp_path / "legacy-v3-archive").is_dir()
