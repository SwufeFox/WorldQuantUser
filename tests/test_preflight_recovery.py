"""Slow live field proofs make durable progress before one simulation launch."""

from argparse import Namespace
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock

import pytest
import requests

import cache_store
import machine_lib as ml
import simulation_capabilities
from research_app import experiments, validation


class Clock:
    def __init__(self):
        self.elapsed = 0.0
        self.epoch = datetime(2026, 11, 24, 12, tzinfo=timezone.utc)

    def monotonic(self):
        return 1000.0 + self.elapsed

    def time(self):
        return self.epoch.timestamp() + self.elapsed

    def utc_now(self):
        return self.epoch + timedelta(seconds=self.elapsed)

    def advance(self, seconds):
        assert seconds >= 0
        self.elapsed += seconds


def arguments(**values):
    return Namespace(**{"spec": None, "name": None, "dry_run": False, "show_test": False,
                        "force": False, "no_record": False, "registry": None,
                        "max_wait_seconds": 30, **values})


def test_slow_field_proofs_survive_cooldown_and_fresh_resume_without_duplicate_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    monkeypatch.setattr(ml, "metadata_cache_policy", lambda: {
        "enabled": True, "refresh": False, "max_age": 3600})
    clock = Clock()
    monkeypatch.setattr(experiments.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(experiments.time, "time", clock.time)
    monkeypatch.setattr(experiments.time, "sleep", clock.advance)
    monkeypatch.setattr(cache_store, "utc_now", clock.utc_now)
    session = requests.Session()
    session._brain_account_id = "offline-slow-fields"
    monkeypatch.setattr(experiments, "with_session", lambda _args, work, **_kwargs: work(session))
    monkeypatch.setattr(experiments, "_novelty_gate", lambda arms, **_kwargs: [{} for _ in arms])
    monkeypatch.setattr(experiments, "_preflight_lint", validation._preflight_lint)
    account = simulation_capabilities.account_provenance(session)
    capabilities = Mock(return_value={"state": "unknown", "account": account,
                                     "concurrency": {"state": "unknown", "limit": None}})
    monkeypatch.setattr(simulation_capabilities, "fetch", capabilities)

    def operators(_session, **kwargs):
        kwargs["report"].update(complete=True)
        return [{"name": "rank"}]

    operator_fetch = Mock(side_effect=operators)
    monkeypatch.setattr(ml, "fetch_operators", operator_fetch)
    for name in ("get_datasets", "get_datafields"):
        monkeypatch.setattr(ml, name, Mock(side_effect=AssertionError("no generic catalog fallback")))

    reads = []

    def request(method, url, *, params, json, timeout):
        assert method == "GET" and url == ml.BRAIN_API_URL + "/data-fields"
        field = params["search"]
        reads.append((field, timeout))
        # Exercise the real request_json timeout, not an always-successful
        # search mock that could hide a short per-query deadline.
        clock.advance(min(timeout, 27.4))
        if timeout < 27.4:
            raise requests.exceptions.ReadTimeout("offline delayed first byte")
        response = requests.Response()
        response.status_code = 200
        response.json = lambda: {"results": [{"id": field, "type": "MATRIX", "dataset": "synthetic",
            **{key: params[key] for key in ("region", "universe", "delay", "instrumentType")}}], "count": 1}
        return response

    monkeypatch.setattr(session, "request", request)

    def simulate(expressions, settings, **kwargs):
        assert expressions == ["rank(a_slow + b_slow)"]
        assert kwargs["preflight"]["ok"] is True
        kwargs["on_event"]("submitting", {"index": 0, "status": "UNKNOWN_SUBMISSION",
                                          "submission_attempted": True})
        row = {"alpha_id": "FinishedAlpha", "status": "COMPLETE", "submission_attempted": True,
               "metrics": {"sharpe": 1.8, "fitness": 1.2},
               "out_of_sample": {"train": {"sharpe": 1.6, "fitness": 1.1}}, "checks": {}}
        kwargs["on_event"]("completed", {"index": 0, **row})
        return [row]

    launch = Mock(side_effect=simulate)
    monkeypatch.setattr(experiments.guarded_sim, "simulate", launch)
    spec, batch = tmp_path / "spec.json", tmp_path / "batch.json"
    spec.write_text(json.dumps({"name": "slow-fields", "design": "explore",
        "arms": [{"label": "candidate", "expression": "rank(a_slow + b_slow)"}]}), encoding="utf-8")
    try:
        first = experiments.cmd_probe(arguments(spec=str(spec), batch_path=str(batch), no_record=True), None)
        assert first["execution"]["not_submitted"] == 1
        assert first["simulated"] == 0
        launch.assert_not_called()
        deferred = first["preflight_deferred"]["candidate"]
        assert deferred["code"] == "SERVICE_ERROR"
        assert deferred["required_fields"]["verified_ids"] == ["a_slow"]
        assert deferred["required_fields"]["unverified_ids"] == ["b_slow"]
        assert reads == [("a_slow", pytest.approx(30)), ("b_slow", pytest.approx(2.6))]
        saved = json.loads(batch.read_text(encoding="utf-8"))
        assert saved["records"][0]["submission_attempted"] is False
        assert deferred["retry_at"] > clock.time()

        immediate = experiments.cmd_probe(arguments(resume=str(batch)), None)
        assert immediate["execution"]["not_submitted"] == 1
        assert immediate["preflight_deferred"]["candidate"] == deferred
        assert len(reads) == 2 and operator_fetch.call_count == 1
        capabilities.assert_called_once()
        launch.assert_not_called()

        clock.advance(deferred["retry_at"] - clock.time() + 0.1)
        resumed = experiments.cmd_probe(arguments(resume=str(batch)), None)
        assert resumed["complete"] is True and not resumed["preflight_deferred"]
        assert [field for field, _timeout in reads] == ["a_slow", "b_slow", "b_slow"]
        assert reads[-1][1] == pytest.approx(30)
        launch.assert_called_once()
        proof = resumed["preflight"][-1]["field_catalog"]
        assert proof["verification"] == "required_fields_only"
        assert proof["required_fields"]["verified_ids"] == ["a_slow", "b_slow"]
        reused = next(row for row in proof["catalogs"] if row["field"] == "a_slow")
        assert reused["reused_from_cache"] is True
        assert reused["as_of"] == (clock.epoch + timedelta(seconds=27.4)).isoformat()
        assert reused["age_seconds"] > 0

        terminal = experiments.cmd_probe(arguments(resume=str(batch)), None)
        assert terminal["complete"] is True and terminal["rows"][0]["alpha_id"] == "FinishedAlpha"
        assert len(reads) == 3 and operator_fetch.call_count == 2
        launch.assert_called_once()
    finally:
        session.close()
