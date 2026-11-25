"""Quick and regular simulations share one durable evidence policy."""

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from unittest import mock

import pytest
import requests

import alpha_registry
import machine_lib
import simulation_capabilities
from tools.pipeline import batch_quick


ACCOUNT = {"state": "known", "fingerprint": "seed-recording-account"}
PROGRESS_URL = f"{machine_lib.BRAIN_API_URL}/simulations/seed-test"
ALPHA_ID = "SeedAlpha1"
ARM = {"label": "cap_rank", "expr": "rank(cap)",
       "settings": {"region": "USA", "universe": "TOP3000", "delay": 1}}


def observed(alpha_id="Alpha1", **changes):
    return {"alpha_id": alpha_id, "expression": "rank(close)",
            "settings": {"region": "USA", "universe": "TOP3000", "simulationMode": "QUICK"},
            "metrics": {"sharpe": 2.1, "fitness": 1.4},
            "checks": {"QUALITY": {"result": "PASS", "value": 1.4, "limit": 1.0}},
            "status": "COMPLETE", **changes}


def response(code, payload=None, **headers):
    result = requests.Response()
    result.status_code = code
    result.headers.update(headers)
    result._content = json.dumps(payload or {}).encode()
    return result


def test_single_arm_wrapper_keeps_observations_consistent_and_replays_idempotently(tmp_path):
    path = tmp_path / "ledger.jsonl"
    first = alpha_registry.build_entry(observed(), idea="initial hypothesis")
    alpha_registry.record_arm(first, path)
    worse = observed(metrics={"sharpe": 0.2, "fitness": 0.0}, status="WARNING",
                     checks={"QUALITY": {"result": "FAIL", "value": 0.0, "limit": 1.0}})
    entry = alpha_registry.build_entry(worse)
    current, written = alpha_registry.record_arm(entry, path)
    assert written is True
    assert (current["metrics"], current["checks"], current["status"]) == (
        worse["metrics"], worse["checks"], "WARNING")
    assert current["failing_checks"] == ["QUALITY"]
    assert current["verdict"] == "rejected"
    assert current["idea"] == "initial hypothesis"
    before = path.read_bytes()
    same, written = alpha_registry.record_arm(entry, path)
    assert written is False and same == current
    assert path.read_bytes() == before
    assert "attempts" not in same

    other = alpha_registry.build_entry(observed("Alpha2", metrics={"sharpe": -0.3, "fitness": -0.1}))
    alpha_registry.record_arm(other, path)
    stored = {item["key"]: item for item in alpha_registry.load_entries(path)}
    assert set(stored) == {"Alpha1", "Alpha2"}
    assert stored["Alpha2"]["metrics"] == other["metrics"]
    assert stored["Alpha1"]["metrics"] == worse["metrics"]


def test_missing_idea_failure_preserves_success_and_full_request_identity(tmp_path):
    path = tmp_path / "ledger.jsonl"
    success = observed()
    alpha_registry.record_simulation([success], {}, registry_path=path)
    original = alpha_registry.load_entries(path)[0]
    failed = {**success, "status": "ERROR", "error": "simulation failed",
              "provenance": {"run_id": "job-1", "arm_label": "arm-1"}}
    first = alpha_registry.record_simulation([failed], {}, registry_path=path)
    assert first["recorded_failures"] == first["missing_idea_count"] == 1
    replay = alpha_registry.record_simulation([failed], {}, registry_path=path)
    assert replay["appended"] == 0
    full = {**failed, "settings": {**failed["settings"], "simulationMode": "FULL"}}
    alpha_registry.record_simulation([full], {}, registry_path=path)
    stored = {entry["key"]: entry for entry in alpha_registry.load_entries(path)}
    assert stored["Alpha1"] == original
    failures = [entry for entry in stored.values() if entry.get("execution_outcome") == "failed"]
    assert len(failures) == 2
    assert {entry["settings"]["simulationMode"] for entry in failures} == {"QUICK", "FULL"}
    for entry in failures:
        assert entry["key"].startswith("failure:")
        assert entry["alpha_id"] is None and entry["related_alpha_id"] == "Alpha1"
        assert entry["metrics"] == entry["checks"] == {}
        assert entry["verdict"] == "watch" and entry["checks_state"] == "missing"


@pytest.mark.parametrize("changes", [
    {"status": "PENDING"},
    {"status": "UNKNOWN_SUBMISSION"},
    {"status": "UNVERIFIED_RESULT"},
    {"status": "RUNNER_ERROR"},
    {"status": "COMPLETE", "alpha_id": None},
    {"status": "COMPLETE", "metrics": {"sharpe": None, "fitness": None}},
    {"status": "COMPLETE", "metrics": {"sharpe": float("nan")}},
    {"status": "COMPLETE", "metrics_pending": True},
    {"status": "COMPLETE", "submission_state": "unknown_submission"},
])
def test_incomplete_or_ambiguous_observations_do_not_become_scientific_success(tmp_path, changes):
    path = tmp_path / "ledger.jsonl"
    record = observed(**changes)
    summary = alpha_registry.record_simulation([record], {}, registry_path=path)
    assert summary["deferred_count"] == 1 and summary["recorded"] == 0
    assert alpha_registry.record_arm(record, path) == (None, False)
    assert alpha_registry.load_entries(path) == []


@pytest.mark.parametrize("outcome", ["COMPLETE", "ERROR"])
def test_quick_checkpoint_and_result_replay_repair_seed_gap_without_another_post(
    tmp_path, monkeypatch, capsys, outcome
):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    monkeypatch.setattr(simulation_capabilities, "account_provenance", lambda _session: ACCOUNT)
    monkeypatch.setattr(machine_lib, "concurrency_plan", lambda *_args, **_kwargs: {
        "account": ACCOUNT, "effective": 1})
    arms, out = tmp_path / "arms.json", tmp_path / "quick.jsonl"
    arms.write_text(json.dumps([ARM]))
    args = Namespace(arms=str(arms), account_slots=1, poll_timeout=5, limit=0, concurrency=1)
    checks = [{"name": "REGIONAL_BREADTH", "result": "FAIL", "value": 0.2, "limit": 1.3,
               "details": {"region": "EMEA"}, "message": "too narrow"},
              {"name": "LOW_SHARPE", "result": "PASS", "value": 2.0, "limit": 1.25}]
    session = mock.MagicMock()
    session.__enter__.return_value = session
    session.post.return_value = response(201, Location=PROGRESS_URL)

    def get(url, **_kwargs):
        if url == PROGRESS_URL:
            return response(200, {"status": outcome, "alpha": ALPHA_ID if outcome == "COMPLETE" else None})
        assert url == f"{machine_lib.BRAIN_API_URL}/alphas/{ALPHA_ID}"
        return response(200, {"is": {"sharpe": 2.0, "fitness": 1.2, "margin": 0.003,
                                      "checks": deepcopy(checks)}})

    session.get.side_effect = get
    monkeypatch.setattr(machine_lib, "load_session", lambda **_kwargs: session)
    original = alpha_registry.record_simulation
    with mock.patch.object(alpha_registry, "record_simulation", side_effect=RuntimeError("seed write interrupted")):
        with pytest.raises(RuntimeError, match="seed write interrupted"):
            batch_quick._run_batch(args, out)
    session.post.assert_called_once()
    row = json.loads(out.read_text())
    checkpoint_path = out.with_suffix(".jsonl.state") / f"{row['simulation_job_key']}.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["simulation_status"] == outcome
    if outcome == "COMPLETE":
        assert checkpoint["checks"]["REGIONAL_BREADTH"] == {key: value for key, value in checks[0].items()
                                                             if key != "name"}
        assert row["checks"] == checkpoint["checks"]
    monkeypatch.setattr(alpha_registry, "record_simulation", original)
    session.reset_mock()
    assert batch_quick._run_batch(args, out) == 0
    session.post.assert_not_called()
    session.get.assert_not_called()
    registry = Path(alpha_registry.default_registry_path())
    seed, = alpha_registry.load_entries(registry)
    assert seed["provenance"]["run_id"] == row["simulation_job_key"]
    assert seed["provenance"]["request_fingerprint"] == row["simulation_key"]
    assert seed["provenance"]["source_spec"] == str(arms)
    assert seed["provenance"]["arm_label"] == ARM["label"]
    assert seed["settings"]["simulationMode"] == "QUICK"
    assert "seed recorded without idea" in capsys.readouterr().err
    if outcome == "COMPLETE":
        assert seed["checks"] == row["checks"]
        assert seed["failing_checks"] == ["REGIONAL_BREADTH"]
        assert seed["metrics"]["margin"] == 0.003
        assert seed["verdict"] == "rejected"
    else:
        assert seed["execution_outcome"] == "failed" and seed["alpha_id"] is None
        assert seed["metrics"] == seed["checks"] == {}
    before = registry.read_bytes()
    assert batch_quick._run_batch(args, out) == 0
    assert registry.read_bytes() == before
    session.post.assert_not_called()
    session.get.assert_not_called()


def test_legacy_compact_quick_checks_do_not_claim_a_full_check_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    payload = batch_quick._payload(ARM)
    row = {"status": "COMPLETE", "alpha_id": ALPHA_ID, "sharpe": 2.0, "fitness": 1.2,
           "LOW_SHARPE": {"result": "PASS", "value": 2.0}, "fails": [],
           "simulation_job_key": "known-job", "simulation_key": batch_quick._simulation_key(payload)}
    summary = batch_quick._record_seed({**ARM, "idea": "declared hypothesis"}, row, payload,
                                     tmp_path / "arms.json", tmp_path / "results.jsonl")
    assert summary["recorded"] == 1
    seed, = alpha_registry.load_entries()
    assert seed["checks_state"] == "missing" and seed["checks"] == {}
    assert seed["verdict"] == "watch"
    assert seed["provenance"]["checks_payload_present"] is False


def test_quick_partial_check_refresh_cannot_promote_a_missing_failure_to_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    arm = {**ARM, "idea": "declared hypothesis"}
    payload = batch_quick._payload(arm)
    checks = {"LOW_SHARPE": {"result": "PASS"}, "REGIONAL_BREADTH": {"result": "FAIL"}}
    row = {"status": "COMPLETE", "alpha_id": ALPHA_ID, "sharpe": 2.0, "fitness": 1.2,
           "checks": checks, "simulation_job_key": "known-job"}
    args = (arm, row, payload, tmp_path / "arms.json", tmp_path / "results.jsonl")
    batch_quick._record_seed(*args)
    row["checks"] = {"LOW_SHARPE": {"result": "PASS"}}
    batch_quick._record_seed(*args)
    seed, = alpha_registry.load_entries()
    assert seed["checks_state"] == "partial"
    assert seed["verdict"] == "watch"
    assert seed["last_check_snapshot"]["checks"]["REGIONAL_BREADTH"]["result"] == "FAIL"
