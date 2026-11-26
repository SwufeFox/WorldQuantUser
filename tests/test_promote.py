"""Promotion orchestration and evidence retention, entirely offline."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

import machine_lib as ml
from research_app import alphas, app, buffer_cmd, promote, selection, triage
from research_app.runtime import CliError
from tools import campaign_runner

SEED_DESCRIPTION = (
    "Idea:\n- Higher quality firms should earn higher subsequent returns.\n\n"
    "Rationale for data used:\n- assets: total reported corporate assets.\n\n"
    "Rationale for operators used:\n- rank: compares firms cross-sectionally."
)


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
             "alpha": {"settings": {"simulationMode": "FULL"}},
             "calls": [], "args": {}, "failures": {}}
    monkeypatch.setattr(promote, "with_session", lambda args, work: work(object()))
    monkeypatch.setattr(ml, "fetch_alpha_metrics", lambda session, alpha_id: state["alpha"])

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


def invoke(capsys, *extra, metadata=True):
    seed = (["--name", "Seed A1", "--category", "FUNDAMENTAL", "--tags", "quality",
             "--description", SEED_DESCRIPTION] if metadata else [])
    code = app.main(["promote", "A1", *seed, *map(str, extra)])
    output = capsys.readouterr()
    return code, json.loads(output.out)


def test_all_pass_buffers_with_forwarded_options(flow, capsys, tmp_path):
    freeze, shelf, saved = (tmp_path / name for name in ("choice.json", "shelf.jsonl", "saved.json"))
    seeds = tmp_path / "custom-seeds.jsonl"
    code, output = invoke(capsys, "--freeze", freeze, "--buffer", shelf, "--save", saved, "--seeds", seeds,
                          "--note", "chosen candidate", "--cost-bps", 5, 10, 20,
                          "--retention-floor", 0.4, "--compact", "--quiet",
                          "--name", "seed A1", "--category", "FUNDAMENTAL",
                          "--tags", "quality", "--description", SEED_DESCRIPTION)
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
    assert flow["args"]["buffer"].seeds == str(seeds)
    assert data["freeze_path"] == str(freeze) and freeze.exists()
    assert json.loads(saved.read_text()) == output


@pytest.mark.parametrize("dry_run", [False, True])
def test_eligible_promotion_without_metadata_returns_a_stable_action(flow, capsys, tmp_path, dry_run):
    extra = ["--max-self-correlation", "0.6"] + (["--dry-run"] if dry_run else [])
    code, output = invoke(capsys, *extra, metadata=False)
    data = output["data"]
    assert code == 0 and output["ok"]
    assert data["outcome"] == "metadata_required" and data["buffered"] is False
    assert len(data["metadata_required"]) == 4
    assert data["steps"]["buffer"] == {"status": "blocked", "reason": "seed_metadata_required"}
    assert flow["calls"] == ["freeze", "validate", "checks"]
    action = data["next_action"]
    assert action["action"] == "complete_seed_metadata"
    assert action["required_options"] == ["--name", "--category", "--tags", "--description"]
    assert data["freeze_path"] in action["argv"] and "--max-self-correlation" in action["argv"]
    assert ("--dry-run" in action["argv"]) is dry_run
    assert not buffer_cmd.buffer_path().exists() and not buffer_cmd.seeds_path().exists()


@pytest.mark.parametrize("saved_choice", [False, True])
def test_quick_mode_stops_before_freeze_or_validation(flow, capsys, tmp_path, saved_choice):
    flow["alpha"]["settings"]["simulationMode"] = "QUICK"
    choice, shelf = tmp_path / "choice.json", tmp_path / "shelf.jsonl"
    if saved_choice:
        selection.freeze(choice, source="triage", rows=[{"alpha_id": "A1"}])
    code, output = invoke(capsys, "--selection" if saved_choice else "--freeze", choice,
                          "--buffer", shelf)
    assert code == 2 and output["error"]["code"] == "INVALID_INPUT"
    assert "Quick Mode" in output["error"]["message"]
    assert output["data"]["steps"]["validate"]["status"] == "pending"
    assert flow["calls"] == []
    assert choice.exists() is saved_choice and not shelf.exists()


@pytest.mark.parametrize("sharpe", [0, -0.01])
def test_nonpositive_test_sharpe_is_rejected(flow, capsys, sharpe):
    flow["row"]["test_sharpe"] = sharpe
    code, output = invoke(capsys)
    assert code == 0 and output["ok"]
    assert output["data"]["outcome"] == "rejected"
    assert output["data"]["buffered"] is False
    assert "test_sharpe <= 0" in output["data"]["reasons"]
    assert output["data"]["checks"] == {}
    assert output["data"]["steps"]["checks"]["status"] == "skipped"
    assert output["data"]["steps"]["buffer"]["status"] == "skipped"
    assert flow["calls"] == ["freeze", "validate"]


def test_low_retention_is_rejected(flow, capsys):
    flow["row"]["retention"] = 0.19
    _, output = invoke(capsys)
    assert output["data"]["outcome"] == "rejected"
    assert any("retention < retention_floor" in reason for reason in output["data"]["reasons"])
    assert flow["calls"] == ["freeze", "validate"]


def test_existing_selection_reuses_frozen_criteria_without_refreezing(flow, capsys, tmp_path):
    choice = tmp_path / "existing.json"
    manifest = selection.freeze(choice, source="triage", rows=[{"alpha_id": "A1"}],
                                criteria={"retention_floor": 0.45, "cost_bps": [5, 10]})
    original = choice.read_bytes()
    code, output = invoke(capsys, "--selection", choice)
    assert code == 0 and output["data"]["buffered"]
    assert flow["calls"] == ["validate", "checks", "buffer"]
    assert flow["args"]["buffer"].cost_bps == [5, 10]
    assert output["data"]["steps"]["freeze"]["evidence"]["selection_id"] == manifest["selection_id"]
    assert choice.read_bytes() == original


def test_saved_candidate_and_criteria_are_checked_before_validation(flow, capsys, tmp_path):
    choice = tmp_path / "existing.json"
    selection.freeze(choice, source="triage", rows=[{"alpha_id": "OTHER"}])
    code, output = invoke(capsys, "--selection", choice)
    assert code == 2 and "exactly this candidate" in output["error"]["message"]
    choice = tmp_path / "different-criteria.json"
    selection.freeze(choice, source="triage", rows=[{"alpha_id": "A1"}],
                     criteria={"retention_floor": 0.45, "cost_bps": [5]})
    code, output = invoke(capsys, "--selection", choice, "--retention-floor", 0.3)
    assert code == 2 and "conflicts with the frozen selection" in output["error"]["message"]
    assert not flow["calls"]


def test_scoped_selection_rejects_a_different_shelf_before_validation(flow, capsys, tmp_path):
    choice = tmp_path / "existing.json"
    shelf = tmp_path / "scored-shelf.jsonl"
    selection.freeze(choice, source="triage", rows=[{"alpha_id": "A1"}],
                     inputs={"shelf": {"path": str(shelf)}})
    code, output = invoke(capsys, "--selection", choice, "--buffer", tmp_path / "other-shelf.jsonl")
    assert code == 2 and "conflicts with the frozen shelf" in output["error"]["message"]
    assert not flow["calls"]


@pytest.mark.parametrize("value,reason", [
    (None, "self_correlation_missing_or_nonfinite"),
    (0.7, "self_correlation_at_or_above_limit"),
])
def test_optional_live_correlation_cap_rejects_before_buffering(flow, capsys, value, reason):
    flow["checked"]["results"][0]["self_correlation"]["value"] = value
    code, output = invoke(capsys, "--max-self-correlation", 0.7)
    assert code == 0 and reason in output["data"]["reasons"]
    assert flow["calls"] == ["freeze", "validate", "checks"]


@pytest.mark.parametrize("verdict,retryable", [
    ("FAIL", False), ("ERROR", True), ("PENDING", True), ("UNKNOWN", True),
])
def test_failed_or_unresolved_checks_preserve_campaign_retry_semantics(flow, capsys, verdict, retryable):
    checks = deepcopy(flow["checked"]["results"][0]["checks"])
    checks["LOW_SHARPE"]["result"] = verdict
    result = alphas._check_result("A1", {"complete": True, "checks": checks})
    flow["checked"] = {"complete": result["complete"], "results": [result]}
    _, output = invoke(capsys)
    assert output["data"]["outcome"] == "rejected"
    assert output["data"]["checks"] == flow["checked"]
    expected = "platform_checks_not_resolved" if retryable else "platform_checks_failed: LOW_SHARPE"
    assert output["data"]["reasons"] == [expected]
    # Campaigns must retry an operational check error instead of caching it as
    # a terminal rejection of the candidate's signal.
    assert campaign_runner._promotion_pending(output["data"]) is retryable
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
    assert flow["calls"] == ["freeze", "validate"]


@pytest.mark.parametrize("field", ["test_sharpe", "retention"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_missing_or_nonfinite_validation_skips_live_checks(flow, capsys, field, value):
    flow["row"][field] = value
    _, output = invoke(capsys)
    assert output["data"]["outcome"] == "rejected"
    assert field + "_missing_or_nonfinite" in output["data"]["reasons"]
    assert output["data"]["steps"]["checks"]["status"] == "skipped"
    assert flow["calls"] == ["freeze", "validate"]


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
        "id": "A1", "status": "COMPLETE",
        "regular": {"code": "rank(close)", "description": SEED_DESCRIPTION},
        "is": {}, "oos": {"train": {"sharpe": 2.0}, "test": {"sharpe": 1.0}}})
    code, output = invoke(capsys, "--note", "keep", "--cost-bps", 5, 10, 20,
                          "--name", "seed A1", "--category", "FUNDAMENTAL", "--tags", "quality",
                          "--description", SEED_DESCRIPTION)
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


@pytest.fixture
def saved_promotion(flow, tmp_path):
    choice = tmp_path / "choice.json"
    row = {**flow["row"], "source_expression": "rank(close)",
           "source_settings": {"simulationMode": "FULL"},
           "costs": {"scope": "train", "scenarios": [{"cost_bps": 20, "net_margin_bps": 2.0}]}}
    manifest = selection.freeze(choice, source="triage", rows=[{
        key: row[key] for key in ("alpha_id", "train_sharpe", "source_expression", "source_settings")
    }], criteria={"retention_floor": 0.4}, window={"mode": "train"})
    check = deepcopy(flow["checked"])
    receipt = {"ok": True, "command": "promote", "data": {
        "alpha_id": "A1", "outcome": "buffered", "checks": check, "steps": {
            "freeze": {"evidence": {"selection_id": manifest["selection_id"]}},
            "validate": {"evidence": {"selection_id": manifest["selection_id"],
                                      "frozen_window": manifest["window"], "rows": [row]}},
        }}}
    return choice, tmp_path / "receipt.json", receipt


@pytest.mark.parametrize("pc,quality,expected", [
    (None, "PASS", "deferred"),
    ("PENDING", "PASS", "deferred"),
    ("PENDING", "FAIL", "blocked"),
    ("PASS", "FAIL", "blocked"),
    ("PASS", "PASS", "passes_review"),
])
def test_review_derives_offline_verdict_from_named_checks(flow, capsys, saved_promotion, pc, quality, expected):
    choice, saved, receipt = saved_promotion
    checks = receipt["data"]["checks"]["results"][0]["checks"]
    checks["LOW_SHARPE"]["result"] = quality
    if pc is not None:
        checks["PROD_CORRELATION"] = {"result": pc}
    saved.write_text(json.dumps(receipt))
    code, response = invoke(capsys, "--review-receipt", saved, "--selection", choice)
    data = response["data"]
    assert code == 0 and response["ok"]
    assert data["recorded_outcome"] == "buffered" and not data["live_verified"]
    assert data["review_outcome"] == expected
    assert data["production_correlation"] == (None if pc is None else {"result": pc})
    assert data["failed_checks"] == (["LOW_SHARPE"] if quality == "FAIL" else [])
    assert data["validation"]["costs"]["scenarios"][0]["net_margin_bps"] == 2.0
    assert not flow["calls"]


@pytest.mark.parametrize("pc,pc_value,histogram_max,outcome", [
    ("PENDING", 0.63, 0.68, "deferred"),
    ("PASS", 0.728, 0.73, "passes_review"),
])
def test_review_keeps_saved_production_histogram_diagnostic(
        flow, capsys, saved_promotion, tmp_path, pc, pc_value, histogram_max, outcome):
    choice, saved, receipt = saved_promotion
    row = receipt["data"]["checks"]["results"][0]
    row["checks"]["PROD_CORRELATION"] = {"result": pc, "value": pc_value, "limit": 0.7}
    row["production_correlation_evidence"] = {"formal": {"captured_at": "2026-11-20T00:00:00+00:00"}}
    saved.write_text(json.dumps(receipt))
    hist = tmp_path / "histogram.json"
    archive = {"source": "GET /alphas/{id}/correlations/prod",
               "requested_ids": ["A1"], "results": {"A1": [{
                   "captured_at": "2026-11-21T19:15:00Z", "http_status": 200, "retry_after": "0.00",
                   "raw_json_response": {"schema": {"name": "prodCorrelation", "properties": [
                                             {"name": "min"}, {"name": "max"}, {"name": "alphas"}]},
                                         "max": histogram_max,
                                         "records": [[0.65, 0.7, 3], [0.7, 0.8, 1 if pc == "PASS" else 0]]},
               }, {"captured_at": "2026-11-21T21:14:00+02:00", "http_status": 429}]}}
    hist.write_text(json.dumps(archive))
    code, response = invoke(capsys, "--review-receipt", saved, "--selection", choice, "--pc-distribution", hist)
    data = response["data"]
    evidence = data["production_correlation_evidence"]
    assert code == 0 and data["review_outcome"] == outcome
    assert evidence["formal"]["result"] == pc and evidence["formal"]["value"] == pc_value
    assert evidence["formal"]["basis"] == "saved_receipt"
    assert evidence["formal"]["captured_at"] == "2026-11-20T00:00:00+00:00"
    assert evidence["distribution"]["max"] == histogram_max
    assert evidence["distribution"]["source"] == "GET /alphas/A1/correlations/prod"
    assert evidence["distribution"]["captured_at"] == "2026-11-21T19:15:00+00:00"
    assert evidence["distribution"]["count_in_bins_at_or_above_formal_limit"] == (1 if pc == "PASS" else 0)
    assert evidence["sharpe_exception"]["state"] == "unavailable"
    assert not flow["calls"]

    archive["results"]["A1"][0]["retry_after"] = "4"
    hist.write_text(json.dumps(archive))
    code, response = invoke(capsys, "--review-receipt", saved, "--selection", choice, "--pc-distribution", hist)
    assert code == 0 and response["data"]["review_outcome"] == outcome
    distribution = response["data"]["production_correlation_evidence"]["distribution"]
    assert distribution["state"] == "unavailable" and distribution["max"] is None
    assert not flow["calls"]

    archive["requested_ids"] = ["B2"]
    hist.write_text(json.dumps(archive))
    code, response = invoke(capsys, "--review-receipt", saved, "--selection", choice, "--pc-distribution", hist)
    assert code == 2 and response["error"]["code"] == "INVALID_INPUT"
    assert not flow["calls"]

    content = hist.read_text()
    code, response = invoke(capsys, "--pc-distribution", hist, "--save", hist, "--not-an-option")
    assert code == 2 and response["error"]["code"] == "INVALID_INPUT"
    assert hist.read_text() == content


def test_review_keeps_operational_failure_separate(flow, capsys, saved_promotion):
    choice, saved, receipt = saved_promotion
    receipt.update(ok=False, error={"code": "RATE_LIMITED"})
    receipt["data"]["checks"] = {}
    saved.write_text(json.dumps(receipt))
    code, response = invoke(capsys, "--review-receipt", saved, "--selection", choice)
    assert code == 0 and response["data"]["review_outcome"] == "deferred"
    assert response["data"]["recorded_error"]["code"] == "RATE_LIMITED"
    assert response["data"]["failed_checks"] == []
    assert not flow["calls"]


def test_review_rejects_different_frozen_window(flow, capsys, saved_promotion):
    choice, saved, receipt = saved_promotion
    receipt["data"]["steps"]["validate"]["evidence"]["frozen_window"] = {}
    saved.write_text(json.dumps(receipt))
    code, response = invoke(capsys, "--review-receipt", saved, "--selection", choice)
    assert code == 2 and "do not match" in response["error"]["message"]
    assert not flow["calls"]


def test_review_protects_input_receipt_even_on_parse_error(flow, capsys, saved_promotion):
    choice, saved, receipt = saved_promotion
    content = json.dumps(receipt)
    saved.write_text(content)
    code, response = invoke(capsys, "--review-receipt", saved, "--selection", choice,
                            "--save", saved, "--not-an-option")
    assert code == 2 and response["error"]["code"] == "INVALID_INPUT"
    assert saved.read_text() == content
