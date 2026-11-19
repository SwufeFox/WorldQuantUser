"""Exercise the Multi API, checkpoint and ledger together without BRAIN I/O."""

import json
import time
from unittest import mock

import requests

import alpha_registry
import machine_lib as ml
from research_app import app, experiments


def response(payload=None, *, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers or {})
    result._content = json.dumps(payload).encode("utf-8") if payload is not None else b""
    return result


def test_cli_resumes_partial_multi_parent_without_reposting(capsys, tmp_path):
    checkpoint = tmp_path / "multi.json"
    registry = tmp_path / "registry.jsonl"
    inputs = ["rank(close)", "rank(volume)"]
    session = mock.Mock(spec=requests.Session)
    session._brain_account_id = "multi-workflow-test"
    session._brain_permissions = ["MULTI_SIMULATION"]
    sent = []
    parent_reads = []
    alpha_reads = []

    def post(url, *, json, **_kwargs):
        assert url == ml.BRAIN_API_URL + "/simulations"
        assert [payload["regular"] for payload in json] == inputs
        saved = checkpoint_record()
        assert all(row["submission_attempted"] for row in saved["records"])
        sent.extend(json)
        return response(status=201, headers={"Location": "/simulations/Parent"})

    def checkpoint_record():
        return json.loads(checkpoint.read_text(encoding="utf-8"))

    def get(url, **_kwargs):
        endpoint = url.removeprefix(ml.BRAIN_API_URL)
        if endpoint == "/simulations/Parent":
            parent_reads.append(url)
            if len(parent_reads) == 1:
                return response({"progress": 0.5}, headers={"Retry-After": "120"})
            return response({"id": "Parent", "status": "ERROR",
                             "children": ["Bad", "Good"]}, status=400)
        if endpoint == "/simulations/Good":
            return response({**sent[0], "id": "Good", "parent": "Parent", "status": "COMPLETE",
                             "alpha": "Alpha1"})
        if endpoint == "/simulations/Bad":
            return response({**sent[1], "id": "Bad", "parent": "Parent", "status": "FAIL"})
        if endpoint == "/alphas/Alpha1":
            alpha_reads.append(url)
            if len(alpha_reads) == 1:
                return response(status=404)
            if len(alpha_reads) == 2:
                return response({})
            return response({"id": "Alpha1", "settings": sent[0]["settings"],
                             "is": {"sharpe": 1.5, "fitness": 1.1}, "train": {"fitness": 1.0}})
        raise AssertionError(f"unexpected GET {endpoint}")

    session.post.side_effect = post
    session.get.side_effect = get
    with mock.patch.object(ml, "load_session", return_value=session), \
         mock.patch.object(experiments, "_invocation_capabilities", return_value={}), \
         mock.patch.object(experiments, "_preflight_lint", return_value={"ok": True}):
        assert app.main(["simulate", "--expr", inputs[0], "--expr", inputs[1], "--multi-size", "2",
                         "--checkpoint", str(checkpoint), "--registry", str(registry),
                         "--max-wait-seconds", "5", "--quiet"]) == 0
        first = json.loads(capsys.readouterr().out)["data"]
        assert first["status"] == "PENDING"
        assert checkpoint_record()["multi_size"] == 2
        assert app.main(["resume", str(checkpoint), "--max-wait-seconds", "1", "--quiet"]) == 0
        assert json.loads(capsys.readouterr().out)["data"]["status"] == "PENDING"
        assert len(parent_reads) == 1
        clock = time.time
        with mock.patch("time.time", side_effect=lambda: clock() + 121):
            for _ in range(2):
                assert app.main(["resume", str(checkpoint), "--max-wait-seconds", "5", "--quiet"]) == 0
                deferred = json.loads(capsys.readouterr().out)["data"]
                assert deferred["status"] == "PENDING"
                good = checkpoint_record()["records"][0]
                assert good["alpha_id"] == "Alpha1" and good["metrics_pending"]
            assert app.main(["resume", str(checkpoint), "--max-wait-seconds", "5", "--quiet"]) == 0
        resumed = json.loads(capsys.readouterr().out)["data"]

    session.post.assert_called_once()
    assert len(parent_reads) == 2
    assert len(alpha_reads) == 3
    assert resumed["status"] == "PARTIAL"
    good, bad = resumed["results"]
    assert (good["expression"], good["alpha_id"], good["metrics"]["fitness"]) == (inputs[0], "Alpha1", 1.1)
    assert (bad["expression"], bad["alpha_id"], bad["status"]) == (inputs[1], None, "FAIL")
    assert any(row["alpha_id"] == "Alpha1" for row in alpha_registry.load_entries(registry))
