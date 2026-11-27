"""Offline tests for the v4 4-gate persistent state machine and workers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import json
from pathlib import Path
from threading import Event
import time
from types import SimpleNamespace
from unittest import mock

import pytest
import requests

import alpha_registry
import cache_store
import machine_lib as ml
import simulation_capabilities
import workspace_db
from research_app import buffer_cmd, experiments, pnl_snapshot, sm_workers, state_machine, validation
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
            mock.patch.object(pnl_snapshot, "yearly_train_batch", return_value=[{"state": "pass"}]), \
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
        views = [json.loads(line) for line in (tmp_path / "views.jsonl").read_text().splitlines()]
        assert [event["tool"] for event in views] == ["approve-full", "inspect"]

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


def _approve_fixture(engine, run_id):
    doc = engine.load_run_raw(run_id)
    doc.update(
        state="GATE_APPROVE_FULL",
        selected_arms=[{"label": "c1", "alpha_id": "Q1"}],
        hydration={"by_label": {"c1": {"alpha_id": "Q1", "gate_pass": True}}},
    )
    engine.save_run(doc)
    engine.act(run_id, "approve-full")


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
    _approve_fixture(engine, run_id)

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

    _approve_fixture(engine, run_id)

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

    engine = state_machine.StateMachineEngine(tmp_path)
    account = {"fingerprint": "fp-orphan"}
    journal_path, _lock_path = simulation_slots._paths(account["fingerprint"], tmp_path)
    simulation_slots._save(journal_path, {
        "version": 1,
        "account_fingerprint": account["fingerprint"],
        "reservations": {
            "slot-1": {
                "state": "accepted",
                "progress_url": "https://api.worldquantbrain.com/simulations/orphan1",
                "created_at": 1.0,
            }
        },
    }, research_dir=tmp_path)
    assert not Path(journal_path).exists()
    other = {"fingerprint": "fp-other"}
    other_journal, _ = simulation_slots._paths(other["fingerprint"], tmp_path)
    simulation_slots._save(other_journal, {
        "version": 1,
        "account_fingerprint": other["fingerprint"],
        "reservations": {"slot-2": {"state": "accepted", "progress_url":
                                     "https://api.worldquantbrain.com/simulations/other1"}},
    }, research_dir=tmp_path)
    alpha_payload = {
        "id": "ORPHAN1",
        "status": "COMPLETE",
        "regular": {"code": "rank(close)"},
        "settings": {"region": "USA", "universe": "TOP3000"},
        "is": {"sharpe": 2.0, "fitness": 1.4},
    }
    with mock.patch.object(simulation_capabilities, "account_provenance", return_value=account), \
            mock.patch.object(ml, "_read_simulation_snapshot",
                              return_value=({"status": "COMPLETE", "alpha": "ORPHAN1"}, None)) as poll, \
            mock.patch.object(ml, "fetch_alpha_metrics", return_value=alpha_payload):
        res = sm_workers.reconcile_orphan_slots(engine, _dummy_session, max_age_min=0.0)
    poll.assert_called_once_with(mock.ANY, "https://api.worldquantbrain.com/simulations/orphan1", "orphan1", deadline=mock.ANY)
    assert res["harvested"] == ["ORPHAN1"]
    entries = alpha_registry.load_entries(str(tmp_path / "alpha-registry.jsonl"))
    assert [e["alpha_id"] for e in entries] == ["ORPHAN1"]
    assert "slot-2" in simulation_slots._load(other_journal, other["fingerprint"], research_dir=tmp_path)["reservations"]


def test_slot_migration_preserves_newer_sqlite_journal(tmp_path):
    import simulation_slots

    path, _ = simulation_slots._paths("fp-migration", tmp_path)
    legacy = {"version": 1, "account_fingerprint": "fp-migration",
              "reservations": {"old": {"state": "accepted"}}}
    cache_store.write_json_atomic(path, legacy)
    assert simulation_slots._load(path, "fp-migration", research_dir=tmp_path) == legacy

    current = {**legacy, "reservations": {"current": {"state": "accepted"}}}
    simulation_slots._save(path, current, research_dir=tmp_path)
    assert Path(path).is_file()
    assert simulation_slots._load(path, "fp-migration", research_dir=tmp_path) == current
    archive = tmp_path / "archived-slots.json"
    assert simulation_slots.import_legacy_journal(path, research_dir=tmp_path, archive_path=archive)
    assert archive.is_file()
    assert simulation_slots._load(path, "fp-migration", research_dir=tmp_path) == current


def test_latest_quota_uses_legacy_accounts_until_migrated(tmp_path):
    import simulation_slots
    import workspace_db

    path, _ = simulation_slots._paths("current", tmp_path)
    current = {"version": 1, "account_fingerprint": "current", "reservations": {},
               "quota": {"observed_at": 10}}
    simulation_slots._save(path, current, research_dir=tmp_path)
    cache_store.write_json_atomic(path, {**current, "quota": {"observed_at": 30}})
    other_path, _ = simulation_slots._paths("legacy", tmp_path)
    cache_store.write_json_atomic(other_path, {**current, "account_fingerprint": "legacy",
                                               "quota": {"observed_at": 20}})
    assert workspace_db.latest_quota_snapshot(research_dir=tmp_path) == {"observed_at": 20}


def test_cancel_and_migrate_workspace(tmp_path):
    engine = state_machine.StateMachineEngine(tmp_path)
    run_id = engine.propose(_explore_spec())["run_id"]
    doc = engine.load_run_raw(run_id)
    doc["state"] = "AUTO_QUICK_SIM"
    doc["records"][0].update({"status": "PENDING", "progress_url": "https://api.worldquantbrain.com/simulations/live1"})
    engine.save_run(doc)

    cancelling = engine.cancel(run_id, reason="superseded")
    assert cancelling["state"] == "AUTO_CANCELLING"
    with mock.patch.object(ml, "cancel_simulation", return_value={"state": "cancelled", "http_status": 204}) as cancel_mock:
        cancelled = sm_workers.advance_run_once(engine, run_id, _dummy_session)
    cancel_mock.assert_called_once_with(mock.ANY, "https://api.worldquantbrain.com/simulations/live1", deadline=mock.ANY)
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


def test_platform_wait_does_not_block_cli_or_overwrite_cancel(tmp_path, monkeypatch):
    engine = state_machine.StateMachineEngine(tmp_path)
    run_id = engine.propose(_explore_spec())["run_id"]
    doc = engine.load_run_raw(run_id)
    doc["state"] = "AUTO_QUICK_SIM"
    engine.save_run(doc)
    started = Event()
    release = Event()
    url = "https://api.worldquantbrain.com/simulations/live1"

    def step(engine, doc, _session, _cfg):
        started.set()
        assert release.wait(timeout=45)
        doc["records"][0].update(progress_url=url, status="PENDING")
        sm_workers._persist_worker_progress(engine, doc, "AUTO_QUICK_SIM", ("records",))
        raise AssertionError("superseded step must exit before advancing")

    monkeypatch.setitem(sm_workers._HANDLERS, "AUTO_QUICK_SIM", step)
    with ThreadPoolExecutor(2) as workers:
        advance = workers.submit(sm_workers.advance_run_once, engine, run_id, _dummy_session)
        try:
            assert started.wait(timeout=12)
            assert workers.submit(engine.update_config, {"concurrency": 2}).result(timeout=12)["concurrency"] == 2
            proposed = workers.submit(engine.propose, {**_explore_spec(), "arms": [
                {"label": "other", "expression": "rank(open)"},
            ]}).result(timeout=12)
            assert proposed["state"] == "AUTO_PREFLIGHT"
            cancelled = workers.submit(engine.cancel, run_id).result(timeout=12)
            assert cancelled["state"] == "AUTO_CANCELLING"
        finally:
            release.set()
        assert advance.result(timeout=12)["records"][0]["progress_url"] == url
    pending = ml.RateLimited("wait before cancel", retry_after=60)
    with mock.patch.object(ml, "cancel_simulation", side_effect=pending):
        after = sm_workers.advance_run_once(engine, run_id, _dummy_session)
    assert after["state"] == "AUTO_CANCELLING"
    assert after["next_wake_at"] > time.time() + 50
    assert after["last_error"]["code"] == "RATE_LIMITED"


def test_approval_rejects_failed_and_multiple_arms(tmp_path):
    engine = state_machine.StateMachineEngine(tmp_path)
    spec = _explore_spec()
    spec["arms"][1].pop("role")
    run_id = engine.propose(spec)["run_id"]
    doc = engine.load_run_raw(run_id)
    doc.update(state="GATE_APPROVE_FULL", selected_arms=[
        {"label": "c1", "alpha_id": "Q1"}, {"label": "d1", "alpha_id": "Q2"},
    ], hydration={"by_label": {"c1": {"gate_pass": False}}})
    engine.save_run(doc)
    for params in ({}, {"arms": ["c1"]}, {"arms": ["c1", "d1"]}):
        with pytest.raises(CliError, match="passed|exactly one"):
            engine.act(run_id, "approve-full", params)
    assert not list(engine.runs_dir.glob("*-selection.json"))
    doc["hydration"]["by_label"]["d1"] = {"alpha_id": "Q2", "gate_pass": True}
    engine.save_run(doc)
    assert engine.act(run_id, "approve-full")["approved_arms"] == [{"label": "d1", "alpha_id": "Q2"}]


def test_propose_rejects_different_settings_without_writing_a_run(tmp_path):
    engine = state_machine.StateMachineEngine(tmp_path)
    spec = _explore_spec()
    spec["arms"][1]["settings"] = {"decay": 17}
    with pytest.raises(CliError, match="share simulation settings"):
        engine.propose(spec)
    assert engine.list_runs()["count"] == 0


def test_run_and_event_roll_back_together(tmp_path):
    engine = state_machine.StateMachineEngine(tmp_path)
    run_id = engine.propose(_explore_spec())["run_id"]
    doc = engine.load_run_raw(run_id)
    doc["state"] = "AUTO_QUICK_SIM"
    with pytest.raises(TypeError):
        engine.save_run(doc, from_state="AUTO_PREFLIGHT", trigger="test", detail=object())
    assert engine.load_run_raw(run_id)["state"] == "AUTO_PREFLIGHT"
    with workspace_db.connect(tmp_path) as conn:
        assert [tuple(row) for row in conn.execute("SELECT to_state, trigger FROM run_events")] == [("AUTO_PREFLIGHT", "propose")]


def test_spc_runs_off_request_thread_without_a_platform_session(tmp_path, monkeypatch):
    engine = state_machine.StateMachineEngine(tmp_path)
    calls = []
    monkeypatch.setattr(sm_workers.spc_experiment, "experiment", lambda *args, **kw: calls.append(args) or {"ok": True})
    predictions = [{"company_name": "Apple Inc.", "ticker": "AAPL", "isin": "US0378331005",
                    "mic": "XNAS", "confidence_score": 0.82, "investment_horizon": 22,
                    "return_prediction": 0.045}]
    spec = {"track": "SPC", "name": "Offline test", "prompt": "Predict return",
            "sampleOutput": json.dumps(predictions), "modelFamily": "local",
            "weight": 1.0, "updateFrequency": "daily", "runs": ["model.json"]}
    proposed = engine.propose(spec, source_path=tmp_path / "spec.json")
    assert proposed["state"] == "AUTO_SPC"
    assert calls == []
    done = sm_workers.advance_run_once(engine, proposed["run_id"], lambda: pytest.fail("SPC is local"))
    assert done["state"] == "COMPLETED"
    assert len(calls) == 1


def test_dry_run_leaves_sqlite_absent_and_archives_unfrozen_legacy_full(tmp_path):
    engine = state_machine.StateMachineEngine(tmp_path)
    existing = engine.runs_dir / "old.json"
    existing.write_text(json.dumps({"run_id": "old", "state": "COMPLETED"}), encoding="utf-8")
    results = tmp_path / "backlog-results"
    results.mkdir()
    (results / "full-batch.json").write_text(json.dumps({
        "complete": True, "records": [{"settings": {"simulationMode": "FULL"}, "alpha_id": "F1"}],
        "rows": [{"alpha_id": "F1", "metrics": {"sharpe": 2.0}}],
    }), encoding="utf-8")
    preview = engine.migrate_workspace(dry_run=True)
    assert preview["runs_json_migrated"] == 1
    assert not workspace_db.db_path(tmp_path).exists()
    migrated = engine.migrate_workspace()["migrated"][0]
    assert migrated["state"] == "ARCHIVED"
    assert engine.inspect_run(migrated["run_id"])["test_view_open"] is False
    assert existing.is_file()


def test_cancellation_recovers_accepted_url_from_slot_journal(tmp_path, monkeypatch):
    import simulation_slots

    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    engine = state_machine.StateMachineEngine(tmp_path)
    run_id = engine.propose(_explore_spec())["run_id"]
    url = "https://api.worldquantbrain.com/simulations/accepted1"
    fingerprint = "account-test"
    doc = engine.load_run_raw(run_id)
    doc["state"] = "AUTO_CANCELLING"
    doc["records"][0].update(status="UNKNOWN_SUBMISSION", slot_reservation={
        "workspace": simulation_slots._workspace(), "account_fingerprint": fingerprint, "id": "slot-1",
    })
    engine.save_run(doc)
    workspace_db.save_slot_journal(fingerprint, {"version": 1, "account_fingerprint": fingerprint,
        "reservations": {"slot-1": {"state": "accepted", "progress_url": url}}}, research_dir=tmp_path)
    with mock.patch.object(simulation_capabilities, "account_provenance", return_value={"fingerprint": fingerprint}), \
            mock.patch.object(ml, "cancel_simulation", return_value={"state": "cancelled"}) as cancel:
        cancelled = sm_workers.advance_run_once(engine, run_id, _dummy_session)
    cancel.assert_called_once_with(mock.ANY, url, deadline=mock.ANY)
    assert cancelled["state"] == "CANCELLED"
