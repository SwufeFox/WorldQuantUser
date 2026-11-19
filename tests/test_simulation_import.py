"""Offline regressions for reading existing BRAIN simulations from Probe."""

import json
from unittest import mock

import pytest
import requests

import machine_lib as ml
from research_app import app


def response(payload=None, *, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers or {})
    result._content = json.dumps(payload).encode("utf-8") if payload is not None else b""
    return result


def run_import(capsys, reference, responses, *options):
    session = mock.Mock(spec=requests.Session)
    session.get.side_effect = responses
    with mock.patch.object(ml, "load_session", return_value=session):
        status = app.main(["probe", "--from-simulation", reference, *options])
    session.post.assert_not_called()
    session.request.assert_not_called()
    output = json.loads(capsys.readouterr().out)
    return status, output, session


def test_imports_multi_simulation_in_child_order_without_fetching_metrics(capsys, tmp_path):
    saved = tmp_path / "simulation-import.json"
    status, output, session = run_import(
        capsys, "https://api.worldquantbrain.com/simulations/Parent-0", [
            response({"id": "Parent-0", "status": "COMPLETE", "children": [
                "Child-1", "Child-2", "Child-3", "Child-4", "Child-5"]}),
            response({"id": "Child-1", "parent": "Parent-0", "status": "COMPLETE",
                      "alpha": "Alpha1", "test": {"sharpe": 100}, "regular": "secret expression"}),
            response({"id": "Child-2", "parent": "Parent-0", "status": "ERROR", "alpha": "Ignore2"}),
            response({"id": "Child-3", "parent": "Parent-0", "status": "SIMULATING"}),
            response(status=404),
            response({"id": "Child-5", "parent": "Parent-0", "status": "WARNING", "alpha": "Alpha1"}),
        ], "--save", str(saved))

    assert status == 0
    assert json.loads(saved.read_text()) == output
    report = output["data"]
    assert report["kind"] == "multi"
    assert report["children_state"] == "available"
    assert report["alpha_ids"] == ["Alpha1"]
    assert report["summary"] == {"total": 5, "complete": 2, "failed": 1, "pending": 1, "unknown": 1}
    assert [row["state"] for row in report["rows"]] == ["complete", "failed", "pending", "unknown", "complete"]
    assert report["rows"][3]["reason"] == "NOT_FOUND"
    assert "secret expression" not in json.dumps(report)
    assert "sharpe" not in json.dumps(report)
    assert [call.args[0] for call in session.get.call_args_list] == [
        f"{ml.BRAIN_API_URL}/simulations/{simulation}" for simulation in
        ["Parent-0", "Child-1", "Child-2", "Child-3", "Child-4", "Child-5"]]
    assert all(call.kwargs["allow_redirects"] is False and call.kwargs["timeout"] > 0
               for call in session.get.call_args_list)


def test_retry_after_is_reported_without_waiting_or_reading_children(capsys):
    status, output, session = run_import(capsys, "Parent", [
        response({"status": "COMPLETE", "children": ["NotReady"], "alpha": "Ignore"},
                 headers={"Retry-After": "60"}),
    ])
    assert status == 0
    assert output["data"]["state"] == "pending"
    assert output["data"]["retry_after_seconds"] == 60
    assert output["data"]["children_state"] == "unavailable"
    assert output["data"]["rows"] == output["data"]["alpha_ids"] == []
    assert session.get.call_count == 1


def test_failed_parent_with_http_400_preserves_successful_child(capsys):
    status, output, session = run_import(capsys, "Parent", [
        response({"id": "Parent", "status": "ERROR", "children": ["Good", "Bad"]}, status=400),
        response({"id": "Good", "parent": "Parent", "status": "COMPLETE", "alpha": "Alpha1"}),
        response({"id": "Bad", "parent": "Parent", "status": "FAIL"}, status=400),
    ])
    report = output["data"]
    assert status == 0
    assert report["state"] == "failed"
    assert report["alpha_ids"] == ["Alpha1"]
    assert report["summary"] == {"total": 2, "complete": 1, "failed": 1, "pending": 0, "unknown": 0}
    assert session.get.call_count == 3


@pytest.mark.parametrize("payload", [
    {"detail": "bad request"},
    {"status": "ERROR"},
    {"id": "Parent", "status": "SIMULATING", "children": ["Good"]},
    {"id": "Other", "status": "ERROR", "children": ["Good"]},
])
def test_http_400_requires_failed_parent_evidence(capsys, payload):
    status, output, session = run_import(capsys, "Parent", [response(payload, status=400)])
    assert status == 4
    assert output["error"]["code"] == "SERVICE_ERROR"
    assert session.get.call_count == 1


def test_child_failures_preserve_provenance_and_stop_after_rate_limit(capsys):
    status, output, session = run_import(capsys, "Parent", [
        response({"status": "COMPLETE", "children": [
            "Child", "Child", "https://untrusted.invalid/child", "Throttle", "Unread"]}),
        response({"id": "Child", "parent": "Other", "status": "COMPLETE", "alpha": "Ignore"}),
        response(status=429),
    ])
    assert status == 0
    assert output["data"]["alpha_ids"] == []
    assert [row["reason"] for row in output["data"]["rows"]] == [
        "parent_mismatch", "duplicate_child_id", "invalid_child_id", "RATE_LIMITED", "RATE_LIMITED"]
    assert session.get.call_count == 3


def test_completed_single_simulation_returns_one_alpha(capsys):
    status, output, _ = run_import(capsys, "Single1", [
        response({"id": "Single1", "status": "WARNING", "alpha": "Alpha1"}),
    ])
    assert status == 0
    assert output["data"]["kind"] == "single"
    assert output["data"]["alpha_ids"] == ["Alpha1"]


@pytest.mark.parametrize("reference", [
    "https://untrusted.invalid/simulations/Parent",
    "https://user:pass@api.worldquantbrain.com/simulations/Parent",
    "http://api.worldquantbrain.com/simulations/Parent",
    "https://api.worldquantbrain.com:444/simulations/Parent",
    "https://api.worldquantbrain.com/simulations/Parent?token=not-in-output",
    "https://api.worldquantbrain.com/simulations/Parent%2FOther",
    "https://api.worldquantbrain.com/alphas/Parent",
    "Parent\n",
])
def test_reference_validation_precedes_authentication(capsys, reference):
    with mock.patch.object(ml, "load_session") as load:
        status = app.main(["probe", "--from-simulation", reference])
    load.assert_not_called()
    assert status == 2
    output = json.loads(capsys.readouterr().out)
    assert output["error"]["code"] == "INVALID_INPUT"
    assert reference not in json.dumps(output)


@pytest.mark.parametrize("options", [
    ["spec.json"], ["--batch-path", "batch.json"], ["--dry-run"],
    ["--show-test"], ["--region", "USA"], ["--recover", "batch.json"],
])
def test_import_refuses_other_probe_modes_and_settings(capsys, options):
    with mock.patch.object(ml, "load_session") as load:
        status = app.main(["probe", "--from-simulation", "Parent", *options])
    load.assert_not_called()
    assert status == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "USAGE"


def test_redirects_are_refused_without_following_location(capsys):
    status, output, session = run_import(
        capsys, "Parent", [response(status=302, headers={"Location": "https://outside.invalid"})])
    assert status == 4
    assert output["error"]["code"] == "SERVICE_ERROR"
    assert session.get.call_count == 1
