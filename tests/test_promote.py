"""Promotion orchestration and evidence retention, entirely offline."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

import machine_lib as ml
from research_app import alphas, app, buffer_cmd, selection, triage
from research_app.runtime import CliError


@pytest.fixture
def flow(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    monkeypatch.setattr(ml, "load_session", lambda **kwargs: pytest.fail("unexpected authentication"))
    monkeypatch.setattr(ml, "invalidate_session", lambda: False)
    row = {"alpha_id": "A1", "train_sharpe": 2.0, "test_sharpe": 1.0,
           "retention": 0.5, "rho_book": 0.12, "max_member_rho": 0.3}
    result = alphas._check_result("A1", {
        "complete": True, "checks": {"SELF_CORRELATION": {"result": "PASS", "value": 0.3},
                                    "LOW_SHARPE": {"result": "PASS"}},
        "self_correlation": {"result": "PASS", "value": 0.3}})
    state = {"row": row, "checked": {"complete": True, "results": [result]},
             "calls": [], "args": {}, "failures": {}}

    def called(name, args):
        state["calls"].append(name)
        state["args"][name] = args
        if name in state["failures"]:
            raise state["failures"][name]

    def score(args, sink):
        name = "validate" if args.show_test else "freeze"
        called(name, args)
        if name == "freeze":
            assert args.ids == ["A1"]
            frozen_row = {"alpha_id": "A1", "train_sharpe": 2.0}
            manifest = selection.freeze(args.freeze, source="triage", rows=[frozen_row],
                                        criteria={"retention_floor": args.retention_floor})
            return {"selection_id": manifest["selection_id"], "rows": [frozen_row]}
        assert args.freeze is None
        selection.load(args.selection, source="triage")
        return {"rows": [deepcopy(state["row"])]}

    def check(args, sink):
        called("checks", args)
        assert args.alpha_ids == ["A1"]
        assert args.workers == 1
        return deepcopy(state["checked"])

    def add(args, sink, *, checked=None):
        called("buffer", args)
        assert checked == state["checked"]["results"][0]
        selection.load(args.selection, source="triage")
        return {"written": True, "path": str(buffer_cmd.buffer_path(args)),
                "entry": {"alpha_id": "A1", "note": args.note}}

    monkeypatch.setattr(triage, "cmd_triage", score)
    monkeypatch.setattr(alphas, "cmd_alpha_check", check)
    state["real_add"] = buffer_cmd.cmd_buffer_add
    monkeypatch.setattr(buffer_cmd, "cmd_buffer_add", add)
    return state


def invoke(capsys, *extra):
    code = app.main(["promote", "A1", *map(str, extra)])
    output = capsys.readouterr()
    return code, json.loads(output.out)


def test_all_pass_buffers_with_forwarded_options(flow, capsys, tmp_path):
    freeze, shelf, saved = (tmp_path / name for name in ("choice.json", "shelf.jsonl", "saved.json"))
    code, output = invoke(capsys, "--freeze", freeze, "--buffer", shelf, "--save", saved,
                          "--note", "chosen candidate", "--cost-bps", 5, 10, 20,
                          "--retention-floor", 0.4, "--compact", "--quiet")
    assert code == 0 and output["ok"] and output["command"] == "promote"
    data = output["data"]
    assert data["outcome"] == "buffered" and data["buffered"] is True
    assert data["reasons"] == []
    assert data["validate"] == {key: flow["row"][key] for key in (
        "train_sharpe", "test_sharpe", "retention", "rho_book", "max_member_rho")}
    assert flow["calls"] == ["freeze", "validate", "checks", "buffer"]
    for step in ("freeze", "validate", "buffer"):
        assert flow["args"][step].cost_bps == [5.0, 10.0, 20.0]
    assert flow["args"]["freeze"].retention_floor == flow["args"]["validate"].retention_floor == 0.4
    assert flow["args"]["buffer"].note == "chosen candidate"
    assert flow["args"]["buffer"].buffer == str(shelf)
    assert data["freeze_path"] == str(freeze) and freeze.exists()
    assert json.loads(saved.read_text()) == output


@pytest.mark.parametrize("sharpe", [0, -0.01])
def test_nonpositive_test_sharpe_is_rejected(flow, capsys, sharpe):
    flow["row"]["test_sharpe"] = sharpe
    code, output = invoke(capsys)
    assert code == 0 and output["ok"]
    assert output["data"]["outcome"] == "rejected"
    assert output["data"]["buffered"] is False
    assert "test_sharpe <= 0" in output["data"]["reasons"]
    assert flow["calls"] == ["freeze", "validate", "checks"]


def test_low_retention_is_rejected(flow, capsys):
    flow["row"]["retention"] = 0.19
    _, output = invoke(capsys)
    assert output["data"]["outcome"] == "rejected"
    assert any("retention < retention_floor" in reason for reason in output["data"]["reasons"])
    assert "buffer" not in flow["calls"]


@pytest.mark.parametrize("verdict", ["FAIL", "ERROR", "PENDING"])
def test_failed_or_unresolved_checks_reject(flow, capsys, verdict):
    result = flow["checked"]["results"][0]
    result["checks"]["LOW_SHARPE"]["result"] = verdict
    if verdict != "FAIL":
        result["checks_resolved"] = False
    _, output = invoke(capsys)
    assert output["data"]["outcome"] == "rejected"
    assert output["data"]["checks"] == flow["checked"]
    assert any("platform_checks_" in reason for reason in output["data"]["reasons"])
    assert "buffer" not in flow["calls"]


def test_warnings_are_allowed_and_preserved(flow, capsys):
    warning = {"result": "WARNING", "message": "units note", "value": 7, "limit": 5}
    flow["checked"]["results"][0]["checks"]["UNITS"] = warning
    _, output = invoke(capsys)
    assert output["data"]["outcome"] == "buffered"
    assert output["data"]["checks"]["results"][0]["checks"]["UNITS"] == warning


def test_dry_run_freezes_validates_checks_but_never_buffers(flow, capsys, tmp_path):
    _, output = invoke(capsys, "--dry-run")
    data = output["data"]
    assert data["outcome"] == "would_buffer" and data["buffered"] is False
    assert data["steps"]["buffer"]["status"] == "would_buffer"
    assert flow["calls"] == ["freeze", "validate", "checks"]
    assert Path(data["freeze_path"]).parent == tmp_path
    assert Path(data["freeze_path"]).name.startswith("promote-freeze-A1-")
    assert Path(data["freeze_path"]).exists()
    assert not buffer_cmd.buffer_path().exists()


def test_dry_run_does_not_label_rejections_would_buffer(flow, capsys):
    flow["row"]["retention"] = 0.1
    _, output = invoke(capsys, "--dry-run")
    assert output["data"]["outcome"] == "rejected"
    assert "buffer" not in flow["calls"]


@pytest.mark.parametrize("field", ["test_sharpe", "retention"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_missing_or_nonfinite_validation_never_buffers(flow, capsys, field, value):
    flow["row"][field] = value
    _, output = invoke(capsys)
    assert output["data"]["outcome"] == "rejected"
    assert field + "_missing_or_nonfinite" in output["data"]["reasons"]
    assert "buffer" not in flow["calls"]


def test_retention_equal_to_floor_is_accepted(flow, capsys):
    flow["row"]["retention"] = 0.2
    _, output = invoke(capsys)
    assert output["data"]["outcome"] == "buffered"


@pytest.mark.parametrize("step,code,error", [
    ("freeze", 2, CliError("INVALID_INPUT", "immutable selection")),
    ("validate", 2, ValueError("invalid frozen choice")),
    ("checks", 4, ml.RateLimited("retry later")),
    ("buffer", 4, OSError("shelf write failed")),
    ("checks", 130, KeyboardInterrupt()),
])
def test_failures_stop_immediately_and_save_prior_evidence(flow, capsys, tmp_path, step, code, error):
    flow["failures"][step] = error
    saved = tmp_path / "failure.json"
    exit_code, output = invoke(capsys, "--save", saved)
    assert exit_code == code and output["ok"] is False
    data = output["data"]
    assert data["failed_step"] == step
    assert data["outcome"] == "rejected" and data["buffered"] is False
    assert data["reasons"]
    order = ["freeze", "validate", "checks", "buffer"]
    assert flow["calls"] == order[:order.index(step) + 1]
    for completed in order[:order.index(step)]:
        assert data["steps"][completed]["status"] == "completed"
        assert (data["checks"] if completed == "checks" else data["steps"][completed]["evidence"])
    assert json.loads(saved.read_text()) == output


def test_check_endpoint_errors_keep_typed_error_and_check_evidence(flow, capsys):
    flow["checked"]["results"][0] = alphas._check_result("A1", error={
        "code": "NOT_FOUND", "message": "unknown alpha", "hint": None})
    flow["checked"]["complete"] = False
    code, output = invoke(capsys)
    assert code == 5 and output["error"]["code"] == "NOT_FOUND"
    assert output["data"]["checks"] == flow["checked"]
    assert output["data"]["steps"]["validate"]["status"] == "completed"
    assert "buffer" not in flow["calls"]


def test_real_buffer_add_reuses_checks_without_second_request(flow, capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(buffer_cmd, "cmd_buffer_add", flow["real_add"])
    monkeypatch.setattr(buffer_cmd, "with_session", lambda args, work: work(object()))
    monkeypatch.setattr(ml, "fetch_alpha_checks", lambda *args, **kwargs: pytest.fail("duplicate checks"))
    monkeypatch.setattr(ml, "fetch_alpha_metrics", lambda *args: {
        "id": "A1", "status": "COMPLETE", "regular": {"code": "rank(close)"},
        "is": {}, "oos": {"train": {"sharpe": 2.0}, "test": {"sharpe": 1.0}}})
    code, output = invoke(capsys, "--note", "keep", "--cost-bps", 5, 10, 20)
    assert code == 0 and output["data"]["buffered"] is True
    entries = buffer_cmd.read_events(buffer_cmd.buffer_path())
    assert len(entries) == 1 and entries[0]["note"] == "keep"
    assert entries[0]["eligible"] is True
    assert entries[0]["selection_id"] == selection.load(output["data"]["freeze_path"])["selection_id"]
    assert entries[0]["costs"]["scope"] == "train"


@pytest.mark.parametrize("extra", [[], ["--not-an-option"]])
def test_save_cannot_overwrite_shelf_even_on_parse_error(flow, capsys, tmp_path, extra):
    shelf = tmp_path / "shelf.jsonl"
    shelf.write_text('test shelf evidence\n')
    code, output = invoke(capsys, "--buffer", shelf, "--save", shelf, *extra)
    assert code == 2 and output["error"]["code"] == "INVALID_INPUT"
    assert shelf.read_text() == 'test shelf evidence\n'
    assert flow["calls"] == []


def test_freeze_and_shelf_paths_must_differ(flow, capsys, tmp_path):
    path = tmp_path / "same.json"
    code, output = invoke(capsys, "--freeze", path, "--buffer", path)
    assert code == 2 and output["error"]["code"] == "INVALID_INPUT"
    assert flow["calls"] == [] and not path.exists()
