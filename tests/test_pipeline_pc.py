"""Offline coverage for Full simulation and production-correlation handoff."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import requests

from tools.pipeline import batch_quick, pipeline_pc


def arm(label, decay=0):
    return {"label": label, "expr": f"rank({label})",
            "settings": {"region": "USA", "universe": "TOP3000", "delay": 1, "decay": decay}}


def simulation(option, *, sharpe=1.3):
    return {"label": option["label"], "expr": option["expr"], "status": "COMPLETE",
            "stage": "simulate", "alpha_id": option["label"] + "-alpha",
            "region": "USA", "universe": "TOP3000", "sharpe": sharpe, "turnover": 0.2}


def full_settings(option):
    return {**option["settings"], "simulationMode": "FULL", "visualization": False}


def snapshot(option, **settings):
    return {"id": option["label"] + "-alpha", "type": "REGULAR", "regular": {"code": option["expr"]},
            "settings": {**full_settings(option), "language": "FASTEXPR", **settings}}


def response(status, body=None, *, retry_after=None):
    receipt = requests.Response()
    receipt.status_code = status
    receipt._content = json.dumps(body).encode()
    if retry_after is not None:
        receipt.headers["Retry-After"] = str(retry_after)
    return receipt


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
    runner.account = {"fingerprint": "test"}
    runner.jobs = {}
    runner.shared_jobs = set()
    runner.stop_requested = threading.Event()
    runner._job_key.side_effect = lambda label, payload: label + ":" + batch_quick._simulation_key(payload)
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
        {**simulation(unresolved), "stage": "checked", "pc_result": "UNRESOLVED",
         "settings": full_settings(unresolved), "account_fingerprint": "test"},
        {**simulation(skipped), "stage": "checked", "pc_result": "SKIPPED",
         "settings": full_settings(skipped), "account_fingerprint": "test"},
        {**simulation(below_gate, sharpe=0.6), "settings": full_settings(below_gate), "account_fingerprint": "test"},
    ])
    check = Mock(return_value={"pc_result": "PASS", "prod_correlation": 0.3})
    monkeypatch.setattr(pipeline_pc, "check_pc", check)

    assert pipeline_pc._run_batch(args, output) == 0

    runner.run.assert_not_called()
    assert {call.args[1] for call in check.call_args_list} == {"unresolved-alpha", "skipped-alpha"}
    assert {row["pc_result"] for row in read_rows(output)[3:]} == {"PASS"}

    assert pipeline_pc._run_batch(args, output) == 0
    assert login.call_count == 2
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

    def check(_runner, alpha_id, _max_wait, **kwargs):
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
    runner.stop_requested = threading.Event()
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
    write_rows(output, [{**simulation(old), "settings": full_settings(old), "account_fingerprint": "test"}])
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


def test_verify_legacy_before_attaching_full_settings(monkeypatch, tmp_path):
    candidate = arm("candidate", decay=19)
    args, output, runner, _ = configure(tmp_path, monkeypatch, [candidate])
    old = arm("candidate", decay=7)
    write_rows(output, [simulation(old)])
    runner._session.return_value.get.return_value = response(200, snapshot(old))
    runner.run.side_effect = lambda option: {**simulation(option), "alpha_id": "NewAlpha"}
    check = Mock(return_value={"pc_result": "PASS", "prod_correlation": 0.3})
    monkeypatch.setattr(pipeline_pc, "check_pc", check)

    assert pipeline_pc._run_batch(args, output) == 0

    assert runner.run.call_args.args[0]["settings"]["decay"] == 19
    assert check.call_args.args[1] == "NewAlpha"
    assert read_rows(output)[0].get("settings") is None
    assert read_rows(output)[-1]["settings"]["decay"] == 19


def test_missing_legacy_settings_leave_reuse_unresolved(monkeypatch, tmp_path):
    candidate = arm("candidate", decay=19)
    args, output, runner, _ = configure(tmp_path, monkeypatch, [candidate])
    write_rows(output, [{**simulation(candidate), "settings": full_settings(candidate)}])
    partial = snapshot(candidate)
    partial["settings"].pop("decay")
    runner._session.return_value.get.return_value = response(200, partial)
    check = Mock()
    monkeypatch.setattr(pipeline_pc, "check_pc", check)

    assert pipeline_pc._run_batch(args, output) == 0

    runner.run.assert_not_called()
    check.assert_not_called()
    pending = read_rows(output)[-1]
    assert pending["status"] == "REUSE_UNRESOLVED"
    assert "settings" not in pending and "account_fingerprint" not in pending

    runner._session.return_value.get.return_value = response(200, snapshot(candidate))
    check.return_value = {"pc_result": "PASS", "prod_correlation": 0.3}
    assert pipeline_pc._run_batch(args, output) == 0
    runner.run.assert_not_called()
    assert check.call_args.args[1] == "candidate-alpha"
    assert read_rows(output)[-1]["account_fingerprint"] == "test"


def test_stale_legacy_row_leaves_next_candidate_its_own_read_and_persists_proof(monkeypatch, tmp_path):
    stale, valid = arm("stale"), arm("valid")
    args, output, runner, _ = configure(tmp_path, monkeypatch, [stale, valid])
    args.pc_max_wait = 0
    write_rows(output, [simulation(stale), simulation(valid)])
    now = 0

    def get(url, *, timeout, **kwargs):
        nonlocal now
        assert timeout >= 5
        if url.endswith("/stale-alpha"):
            now += 29
            return response(404)
        assert url.endswith("/valid-alpha")
        now += 5
        return response(200, snapshot(valid))

    monkeypatch.setattr(pipeline_pc.time, "monotonic", lambda: now)
    client = runner._session.return_value
    client.get.side_effect = get

    assert pipeline_pc._run_batch(args, output) == 0
    assert client.get.call_count == 2
    assert read_rows(output)[-2]["status"] == "REUSE_UNRESOLVED"
    verified = read_rows(output)[-1]
    assert verified["alpha_id"] == "valid-alpha"
    assert verified["account_fingerprint"] == "test"
    assert verified["settings"] == full_settings(valid)

    client.get.reset_mock()
    assert pipeline_pc._run_batch(args, output) == 0
    client.get.assert_called_once()
    assert client.get.call_args.args[0].endswith("/stale-alpha")
    runner.run.assert_not_called()


def test_batch_limit_avoids_legacy_reads_after_filling_fresh_work(monkeypatch, tmp_path):
    fresh, old = arm("fresh"), arm("old")
    args, output, runner, _ = configure(tmp_path, monkeypatch, [fresh, old])
    args.limit = 1
    args.pc_max_wait = 0
    write_rows(output, [simulation(old)])
    runner.run.side_effect = simulation

    assert pipeline_pc._run_batch(args, output) == 0
    assert runner.run.call_args.args[0]["label"] == "fresh"
    runner._session.return_value.get.assert_not_called()


def test_saved_nonnumeric_turnover_keeps_result_without_blocking_next_arm(monkeypatch, tmp_path):
    old, fresh = arm("old"), arm("fresh")
    args, output, runner, _ = configure(tmp_path, monkeypatch, [old, fresh])
    args.limit = 1
    saved = {**simulation(old), "settings": full_settings(old), "account_fingerprint": "test",
             "turnover": "0.2"}
    write_rows(output, [saved])
    runner.run.side_effect = lambda option: simulation(option, sharpe=0)

    assert pipeline_pc._run_batch(args, output) == 0
    assert runner.run.call_args.args[0]["label"] == "fresh"
    runner._session.return_value.get.assert_not_called()
    assert read_rows(output)[0] == saved


def test_no_work_leaves_ledger_unchanged(monkeypatch, tmp_path):
    candidate = arm("candidate")
    args, output, runner, _ = configure(tmp_path, monkeypatch, [candidate])
    write_rows(output, [{**simulation(candidate), "settings": full_settings(candidate),
                         "account_fingerprint": "test", "pc_result": "PASS"}])
    original = output.read_bytes() + b'{"stage":'
    output.write_bytes(original)

    assert pipeline_pc._run_batch(args, output) == 0
    assert output.read_bytes() == original
    runner.run.assert_not_called()
    runner._session.return_value.get.assert_not_called()


def test_other_accounts_and_terminal_failures_do_not_block_next_simulation(monkeypatch, tmp_path):
    first, second = arm("first"), arm("second")
    args, output, runner, _ = configure(tmp_path, monkeypatch, [first, second])
    args.limit = 1
    args.pc_max_wait = 0
    write_rows(output, [{**simulation(first), "settings": full_settings(first), "account_fingerprint": "other"}])

    def run(candidate):
        if candidate["label"] == "first":
            return {**simulation(candidate), "alpha_id": None, "status": "FAIL", "simulation_status": "FAIL"}
        return simulation(candidate)

    runner.run.side_effect = run
    for _ in range(2):
        assert pipeline_pc._run_batch(args, output) == 0
    assert [call.args[0]["label"] for call in runner.run.call_args_list] == ["first", "second"]
    assert [(row["status"], row["account_fingerprint"]) for row in read_rows(output)] == [
        ("COMPLETE", "other"), ("FAIL", "test"), ("COMPLETE", "test")]


def test_pc_respects_retry_after_and_the_remaining_time(monkeypatch):
    now = 0.0

    def wait(seconds):
        nonlocal now
        now += seconds
        return False

    monkeypatch.setattr(pipeline_pc.time, "monotonic", lambda: now)
    runner = Mock()
    runner.stop_requested.is_set.return_value = False
    runner.stop_requested.wait.side_effect = wait
    session = runner._session.return_value
    limited, passed = requests.Response(), requests.Response()
    limited.status_code = 429
    limited.headers["Retry-After"] = "10"
    passed.status_code = 200
    passed._content = b'{"is":{"checks":[{"name":"PROD_CORRELATION","result":"PASS","value":0.3}]}}'
    session.get.side_effect = [limited, passed]

    assert pipeline_pc.check_pc(runner, "OfflineAlpha", 15)["pc_result"] == "PASS"
    assert session.get.call_count == 2
    assert session.get.call_args_list[1].kwargs["timeout"] == 5
    runner.stop_requested.wait.assert_called_once_with(10)

    session.get.reset_mock(side_effect=True)
    session.get.return_value = limited
    assert pipeline_pc.check_pc(runner, "OfflineAlpha", 5)["pc_error"] == "RateLimited"
    session.get.assert_called_once()
    assert now == 10


def test_pc_retry_after_survives_restart_and_leaves_room_for_other_arms(monkeypatch, tmp_path):
    candidate, fresh = arm("candidate"), arm("fresh")
    args, output, runner, _ = configure(tmp_path, monkeypatch, [candidate, fresh])
    args.limit = 1
    args.pc_max_wait = 15
    write_rows(output, [{**simulation(candidate), "settings": full_settings(candidate), "account_fingerprint": "test"}])
    now = 10
    monkeypatch.setattr(pipeline_pc.time, "time", lambda: now)
    monkeypatch.setattr(pipeline_pc.time, "monotonic", lambda: now)
    runner.run.side_effect = lambda option: simulation(option, sharpe=0)
    client = runner._session.return_value
    client.get.return_value = response(429, retry_after=60)

    assert pipeline_pc._run_batch(args, output) == 0
    assert read_rows(output)[-1]["next_pc_poll_at"] == 70
    assert read_rows(output)[-1]["pc_error"] == "RateLimited"
    runner.run.assert_not_called()

    assert pipeline_pc._run_batch(args, output) == 0
    client.get.assert_called_once()
    assert runner.run.call_args.args[0]["label"] == "fresh"

    now = 70
    client.get.return_value = response(200, {"is": {"checks": [{"name": "PROD_CORRELATION", "result": "PASS"}]}})
    assert pipeline_pc._run_batch(args, output) == 0
    latest = read_rows(output)[-1]
    assert latest["pc_result"] == "PASS"
    assert "next_pc_poll_at" not in latest
    assert "pc_error" not in latest


def test_pc_saves_retry_after_before_waiting(monkeypatch, tmp_path):
    candidate = arm("candidate")
    args, output, runner, _ = configure(tmp_path, monkeypatch, [candidate])
    args.pc_max_wait = 15
    write_rows(output, [{**simulation(candidate), "settings": full_settings(candidate), "account_fingerprint": "test"}])
    runner.stop_requested = Mock()
    runner.stop_requested.is_set.return_value = False
    runner._session.return_value.get.return_value = response(429, retry_after=10)
    monkeypatch.setattr(pipeline_pc.time, "time", lambda: 100)

    def wait(delay):
        assert delay == 10
        assert read_rows(output)[-1]["next_pc_poll_at"] == 110
        return True

    runner.stop_requested.wait.side_effect = wait
    assert pipeline_pc._run_batch(args, output) == 0
    runner.stop_requested.wait.assert_called_once()
    assert read_rows(output)[-1]["next_pc_poll_at"] == 110


def test_pc_does_not_start_http_after_session_setup_exhausts_its_budget(monkeypatch):
    now = 0.0
    session = Mock()
    runner = Mock()
    runner.stop_requested = threading.Event()

    def slow_session(*, refresh, deadline):
        nonlocal now
        assert not refresh
        assert deadline == 2
        now = 2
        return session

    monkeypatch.setattr(pipeline_pc.time, "monotonic", lambda: now)
    runner._session.side_effect = slow_session

    assert pipeline_pc.check_pc(runner, "OfflineAlpha", 2)["pc_result"] == "UNRESOLVED"
    session.get.assert_not_called()


def test_pc_retries_transient_session_refresh_within_check_budget(monkeypatch):
    now = 0
    runner = Mock()
    runner.stop_requested.is_set.return_value = False
    client = Mock()
    runner._session.side_effect = [client, pipeline_pc.ml.AuthenticationUnavailable("offline"), client]
    client.get.side_effect = [response(401), response(200, {"is": {"checks": [
        {"name": "PROD_CORRELATION", "result": "PASS"},
    ]}})]

    def wait(delay):
        nonlocal now
        now += delay
        return False

    runner.stop_requested.wait.side_effect = wait
    monkeypatch.setattr(pipeline_pc.time, "monotonic", lambda: now)

    assert pipeline_pc.check_pc(runner, "OfflineAlpha", 15)["pc_result"] == "PASS"
    assert runner._session.call_count == 3
    assert [call.kwargs.get("refresh", False) for call in runner._session.call_args_list] == [False, True, True]
    assert client.get.call_args.kwargs["timeout"] == 7


def test_interrupt_does_not_start_unqueued_simulations(monkeypatch, tmp_path):
    args, output, runner, _ = configure(tmp_path, monkeypatch, [arm("first"), arm("second")])
    args.concurrency = 1
    started = threading.Event()

    def run(candidate):
        started.set()
        assert runner.stop_requested.wait(2)
        return {**simulation(candidate), "status": "PENDING"}

    def interrupted(_pending):
        assert started.wait(2)
        raise KeyboardInterrupt

    runner.run.side_effect = run
    monkeypatch.setattr(pipeline_pc, "_finished", interrupted)
    with pytest.raises(KeyboardInterrupt):
        pipeline_pc._run_batch(args, output)
    runner.run.assert_called_once()
    runner.close.assert_called_once()
