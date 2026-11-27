"""Offline tests for the v4 DaemonServer loopback HTTP IPC, auth, wakeup, and CLI lifecycle."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request
from unittest import mock

import pytest

import machine_lib as ml
import simulation_capabilities
from research_app import daemon, experiments, sm_workers, state_machine, validation

ROOT = Path(__file__).resolve().parent.parent


def _http_raw(url, method="GET", token=None, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req, timeout=5.0)


def test_loopback_http_auth_and_wait_wakeup(tmp_path):
    engine = state_machine.StateMachineEngine(tmp_path)
    srv = daemon.DaemonServer(engine, session_factory=lambda: type("S", (), {"close": lambda self: None})())

    acc = {"fingerprint": "fp-ipc", "quota": {"remaining": 100, "blocked": False}}
    quick_res = [{"index": 0, "label": "c1", "status": "COMPLETE", "alpha_id": "IPC1", "submission_attempted": True,
                  "metrics": {"sharpe": 2.0, "fitness": 1.4, "turnover": 0.15},
                  "out_of_sample": {"train": {"sharpe": 2.1, "fitness": 1.5}, "test": {"sharpe": 1.3}}}]

    with mock.patch.object(simulation_capabilities, "account_provenance", return_value=acc), \
            mock.patch.object(validation, "_preflight_lint", return_value={"ok": True}), \
            mock.patch.object(ml, "simulate_alphas", return_value=quick_res):
        ep = srv.start()
        try:
            url = f"http://{ep['host']}:{ep['port']}/v1/status"
            # Missing or invalid bearer token -> 403 ACCESS_DENIED
            for bad_token in (None, "wrong-token"):
                with pytest.raises(urllib.error.HTTPError) as denied:
                    _http_raw(url, token=bad_token)
                assert denied.value.code == 403
                err_body = json.loads(denied.value.read().decode("utf-8"))
                assert err_body["error"]["code"] == "ACCESS_DENIED"

            status = daemon._http_call(ep, "GET", "/v1/status")
            assert status["running"] is True

            spec = {"name": "ipc-test", "design": "explore", "idea": "IPC test idea",
                    "arms": [{"label": "c1", "expression": "rank(close)"}]}
            proposed = daemon._http_call(ep, "POST", "/v1/propose", body={"spec": spec})
            assert proposed["state"] == "AUTO_PREFLIGHT"

            waited = daemon._http_call(ep, "POST", "/v1/wait",
                                       body={"run_id": proposed["run_id"], "max_wait_seconds": 10.0})
            assert waited["woken"] is True
            assert waited["run"]["state"] == "GATE_TRIAGE"
            assert waited["run"]["test_view_open"] is False
        finally:
            srv.stop()

    assert not (tmp_path / "daemon" / "endpoint.json").exists()


def test_crash_restart_recovery_resumes_auto_quick_sim(tmp_path):
    engine = state_machine.StateMachineEngine(tmp_path)
    run_id = engine.propose({
        "name": "restart-recovery",
        "design": "explore",
        "idea": "Survives daemon restart",
        "arms": [{"label": "c1", "expression": "rank(volume)"}],
    })["run_id"]
    doc = engine.load_run_raw(run_id)
    doc["state"] = "AUTO_QUICK_SIM"
    engine.save_run(doc)

    acc = {"fingerprint": "fp-rec", "quota": {"remaining": 50, "blocked": False}}
    quick_res = [{"index": 0, "label": "c1", "status": "COMPLETE", "alpha_id": "REC1", "submission_attempted": True,
                  "metrics": {"sharpe": 1.9, "fitness": 1.3},
                  "out_of_sample": {"train": {"sharpe": 1.9, "fitness": 1.3}}}]

    with mock.patch.object(simulation_capabilities, "account_provenance", return_value=acc), \
            mock.patch.object(ml, "simulate_alphas", return_value=quick_res):
        srv = daemon.DaemonServer(engine, session_factory=lambda: type("S", (), {"close": lambda self: None})())
        srv.start()
        try:
            waited = srv.wait_for_gate(run_id=run_id, max_wait_seconds=10.0)
            assert waited["woken"] is True
            assert waited["run"]["state"] == "GATE_TRIAGE"
        finally:
            srv.stop()


def test_subprocess_cli_daemon_start_propose_inbox_stop(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("BRAIN_")}
    spec = tmp_path / "cli-daemon.json"
    spec.write_text(json.dumps({
        "name": "cli-subproc",
        "idea": "Cross-sectional liquidity",
        "design": "explore",
        "arms": [{"label": "c1", "expression": "rank(close)"}],
    }), encoding="utf-8")

    def cli(*args):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "research_cli.py"), "--research-dir", str(tmp_path), *args],
            capture_output=True, encoding="utf-8", cwd=str(ROOT), env=env, timeout=45,
        )
        return proc.returncode, json.loads(proc.stdout), proc.stderr

    try:
        rc, started, stderr = cli("daemon", "start")
        assert rc == 0, stderr
        assert started["data"]["running"] is True

        rc, status, stderr = cli("daemon", "status")
        assert rc == 0, stderr
        assert status["data"]["running"] is True
        assert status["data"]["pid"] == started["data"]["pid"]

        rc, proposed, stderr = cli("propose", str(spec))
        assert rc == 0, stderr
        run_id = proposed["data"]["run_id"]

        rc, runs_doc, stderr = cli("runs")
        assert rc == 0, stderr
        assert any(r["run_id"] == run_id for r in runs_doc["data"]["runs"])

        rc, inbox_doc, stderr = cli("inbox")
        assert rc == 0, stderr
        assert "counts_by_state" in inbox_doc["data"]
    finally:
        subprocess.run(
            [sys.executable, str(ROOT / "research_cli.py"), "--research-dir", str(tmp_path), "daemon", "stop"],
            capture_output=True, encoding="utf-8", cwd=str(ROOT), env=env, timeout=30,
        )
