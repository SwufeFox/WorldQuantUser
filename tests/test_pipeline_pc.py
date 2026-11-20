"""Offline coverage for Full simulation and production-correlation handoff."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import requests

from tools.pipeline import batch_quick, pipeline_pc


def arm(label, decay=0):
    return {"label": label, "expr": f"rank({label})",
            "settings": {"region": "USA", "universe": "TOP3000", "delay": 1, "decay": decay}}


def simulation(option, *, sharpe=1.3):
    return {"label": option["label"], "expr": option["expr"], "status": "COMPLETE",
            "stage": "simulate", "alpha_id": option["label"] + "-alpha",
            "region": "USA", "universe": "TOP3000", "sharpe": sharpe, "turnover": 0.2}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def configure(tmp_path, monkeypatch, arms, *, jsonl=False):
    inputs = tmp_path / "arms.json"
    if jsonl:
        write_rows(inputs, arms)
    else:
        inputs.write_text(json.dumps(arms))
    output = tmp_path / "ledger.jsonl"
    args = SimpleNamespace(arms=inputs, concurrency=2, account_slots=5, limit=0,
                           min_sharpe=0.9, max_turnover=0.7, pc_max_wait=2)
    runner = Mock()
    runner_factory = Mock(return_value=runner)
    login = Mock(return_value=MagicMock())
    monkeypatch.setattr(pipeline_pc, "Runner", runner_factory)
    monkeypatch.setattr(pipeline_pc.ml, "load_session", login)
    monkeypatch.setattr(pipeline_pc.simulation_capabilities, "account_provenance",
                        lambda session: {"fingerprint": "test"})
    return args, output, runner, login


def test_resumes_unresolved_and_skipped_checks_without_repeating_simulations(monkeypatch, tmp_path):
    unresolved, skipped, below_gate = [arm(label) for label in ("unresolved", "skipped", "below_gate")]
    args, output, runner, login = configure(tmp_path, monkeypatch, [unresolved, skipped, below_gate],
                                           jsonl=True)
    write_rows(output, [
        {**simulation(unresolved), "stage": "checked", "pc_result": "UNRESOLVED"},
        {**simulation(skipped), "stage": "checked", "pc_result": "SKIPPED"},
        simulation(below_gate, sharpe=0.6),
    ])
    check = Mock(return_value={"pc_result": "PASS", "prod_correlation": 0.3})
    monkeypatch.setattr(pipeline_pc, "check_pc", check)

    assert pipeline_pc._run_batch(args, output) == 0

    runner.run.assert_not_called()
    assert {call.args[1] for call in check.call_args_list} == {"unresolved-alpha", "skipped-alpha"}
    assert {row["pc_result"] for row in read_rows(output)[3:]} == {"PASS"}

    assert pipeline_pc._run_batch(args, output) == 0
    login.assert_called_once()
    assert len(read_rows(output)) == 5


def test_full_simulation_is_durable_before_check_and_other_results_keep_writing(monkeypatch, tmp_path):
    first, second = arm("first"), arm("second")
    args, output, runner, _ = configure(tmp_path, monkeypatch, [first, second])
    check_started, second_written = threading.Event(), threading.Event()
    append = pipeline_pc._append

    def record(stream, row):
        append(stream, row)
        if row["label"] == "second":
            second_written.set()

    def run(candidate):
        assert candidate["settings"]["simulationMode"] == "FULL"
        assert candidate["settings"]["visualization"] is False
        if candidate["label"] == "second":
            assert check_started.wait(timeout=2)
        return simulation(candidate, sharpe=1.3 if candidate["label"] == "first" else 0.5)

    def check(_runner, alpha_id, _max_wait):
        assert alpha_id == "first-alpha"
        assert read_rows(output)[0]["stage"] == "simulate"
        check_started.set()
        assert second_written.wait(timeout=2)
        return {"pc_result": "PASS"}

    runner.run.side_effect = run
    monkeypatch.setattr(pipeline_pc, "_append", record)
    monkeypatch.setattr(pipeline_pc, "check_pc", check)

    assert pipeline_pc._run_batch(args, output) == 0
    assert [(row["label"], row["stage"]) for row in read_rows(output)] == [
        ("first", "simulate"), ("second", "simulate"), ("first", "checked")]


def test_uses_native_check_payload_and_keeps_missing_pc_retryable():
    runner = Mock()
    client = runner._session.return_value
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({"is": {"checks": [
        {"name": "PROD_CORRELATION", "result": "FAIL", "value": 0.85, "limit": 0.7},
        {"name": "SELF_CORRELATION", "result": "PASS", "value": 0.2, "limit": 0.7},
    ]}}).encode()
    client.get.return_value = response

    assert pipeline_pc.check_pc(runner, "candidate-alpha", 2) == {
        "pc_result": "FAIL", "prod_correlation": 0.85,
        "self_correlation": 0.2, "failed": ["PROD_CORRELATION"]}
    assert client.get.call_args.args[0].endswith("/alphas/candidate-alpha/check")

    response._content = b'{"is":{"checks":[{"name":"PROD_CORRELATION","result":"PENDING"}]}}'
    assert pipeline_pc.check_pc(runner, "candidate-alpha", 0.01)["pc_result"] == "UNRESOLVED"


def test_changed_full_settings_do_not_reuse_old_row(monkeypatch, tmp_path):
    candidate = arm("candidate", decay=1)
    args, output, runner, _ = configure(tmp_path, monkeypatch, [candidate])
    args.pc_max_wait = 0
    old = arm("candidate")
    write_rows(output, [{**simulation(old), "settings": {
        **old["settings"], "simulationMode": "FULL", "visualization": False}}])
    runner.run.side_effect = simulation

    assert pipeline_pc._run_batch(args, output) == 0
    runner.run.assert_called_once()
    assert read_rows(output)[-1]["settings"]["decay"] == 1


def test_quick_ledger_row_does_not_satisfy_full_simulation(monkeypatch, tmp_path):
    candidate = arm("candidate")
    args, output, runner, _ = configure(tmp_path, monkeypatch, [candidate])
    args.pc_max_wait = 0
    row = simulation(candidate)
    row.pop("stage")
    row["simulation_key"] = batch_quick._simulation_key(batch_quick._payload(candidate))
    write_rows(output, [row])
    runner.run.side_effect = simulation

    assert pipeline_pc._run_batch(args, output) == 0
    runner.run.assert_called_once()
    assert read_rows(output)[-1]["settings"]["simulationMode"] == "FULL"
