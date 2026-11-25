"""One check-state interpretation across live checks, triage and the shelf."""

from argparse import Namespace
from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

import alpha_registry
from check_evidence import CLIENT_PARSE_ERROR, normalize_checks, platform_check_names, summarize_checks
import machine_lib as ml
import registry_index
from research_app import alphas, buffer_cmd, selection, triage
from research_app.runtime import CliError


@pytest.mark.parametrize("shape", ["mapping", "list", "legacy"])
@pytest.mark.parametrize("verdict,passed,failed,pending,warned,errored,resolved", [
    ("PASS", True, False, False, False, False, True),
    ("WARNING", True, False, False, True, False, True),
    ("FAIL", False, True, False, False, False, True),
    ("ERROR", False, True, False, False, True, False),
    ("PENDING", False, False, True, False, False, False),
    (None, False, False, True, False, False, False),
    ("", False, False, True, False, False, False),
    ("RUNNING", False, False, True, False, False, False),
    (True, False, False, True, False, False, False),
    (" warning ", True, False, False, True, False, True),
])
def test_named_state_matrix(shape, verdict, passed, failed, pending, warned, errored, resolved):
    if shape == "mapping":
        checks = {"QUALITY": {"result": verdict}}
    elif shape == "list":
        checks = [{"name": "QUALITY", "result": verdict}]
    else:
        checks = {"QUALITY": verdict}
    summary = summarize_checks(checks)
    expected = dict(passed=passed, failed=failed, pending=pending, warned=warned, errored=errored)
    assert {key: summary[key] for key in expected} == {
        key: ["QUALITY"] if value else [] for key, value in expected.items()
    }
    assert summary["available"] is True
    assert summary["resolved"] is resolved


@pytest.mark.parametrize("checks", [None, {}, []])
def test_absent_checks_never_resolve(checks):
    summary = summarize_checks(checks, required=("QUALITY",))
    assert summary["available"] is False
    assert summary["resolved"] is False
    assert summary["pending"] == ["QUALITY"]
    assert summary["passed"] == []


@pytest.mark.parametrize("checks", ["PASS", [None], [{"result": "PASS"}]])
def test_malformed_only_checks_are_retained_without_establishing_a_platform_check(checks):
    normalized = normalize_checks(checks)
    assert normalized[CLIENT_PARSE_ERROR]["source"] == "client_parser"
    assert normalized[CLIENT_PARSE_ERROR]["result"] is None
    assert normalized[CLIENT_PARSE_ERROR]["raw_records"]
    summary = summarize_checks(normalized, required=("QUALITY",))
    assert summary["available"] is False and summary["resolved"] is False
    assert summary["pending"] == sorted([CLIENT_PARSE_ERROR, "QUALITY"])
    assert platform_check_names(normalized) == set()


AMBIGUOUS_CHECKS = [
    ([{"name": "QUALITY", "result": "FAIL"}, {"name": "QUALITY", "result": "PASS"}], "QUALITY"),
    ([{"name": "QUALITY", "result": "PASS"}, {"name": "QUALITY", "result": "FAIL"}], "QUALITY"),
    ([{"result": "FAIL"}, {"name": "QUALITY", "result": "PASS"}], CLIENT_PARSE_ERROR),
    ([None, {"name": "QUALITY", "result": "PASS"}], CLIENT_PARSE_ERROR),
]


@pytest.mark.parametrize("raw,diagnostic_name", AMBIGUOUS_CHECKS)
def test_ambiguous_checks_survive_json_roundtrips_without_resolving_or_growing(raw, diagnostic_name):
    before = deepcopy(raw)
    normalized = normalize_checks(raw)
    diagnostic = normalized[diagnostic_name]
    assert diagnostic["source"] == "client_parser" and diagnostic["result"] is None
    assert diagnostic["raw_records"] == (raw if diagnostic_name == "QUALITY" else raw[:1])
    assert summarize_checks(normalized)["resolved"] is False
    encoded = json.dumps(normalized, sort_keys=True)
    for _ in range(3):
        normalized = normalize_checks(json.loads(encoded))
        assert json.dumps(normalized, sort_keys=True) == encoded
    assert raw == before


@pytest.mark.parametrize("raw,diagnostic_name", AMBIGUOUS_CHECKS)
def test_http_check_extraction_keeps_ambiguous_evidence_pending(raw, diagnostic_name):
    # A valid independent platform check keeps both its formal PASS and its
    # contradictory numeric diagnostic while the whole payload stays pending.
    raw = deepcopy(raw) + [{"name": "PROD_CORRELATION", "result": "PASS", "value": 0.73, "limit": 0.7}]
    response = Mock(status_code=200, headers={}, **{"json.return_value": {"is": {"checks": raw}}})
    session = Mock(get=Mock(return_value=response))
    payload = ml.fetch_alpha_checks(session, "A", max_wait_seconds=0)
    assert payload["complete"] is True and payload["payload_received"] is True
    live = alphas._check_result("A", payload)
    assert live["complete"] is False and live["checks_resolved"] is False
    assert live["outcome"] == "pending"
    diagnostic = live["checks"][diagnostic_name]
    assert diagnostic["source"] == "client_parser" and diagnostic["raw_records"]
    correlation = live["checks"]["PROD_CORRELATION"]
    assert correlation["result"] == "PASS"
    assert correlation["expected_result"] == "FAIL" and correlation["value_consistent"] is False
    session.get.assert_called_once()


@pytest.mark.parametrize("raw,diagnostic_name", AMBIGUOUS_CHECKS)
def test_ambiguous_checks_remain_unknown_after_ledger_and_index_replay(tmp_path, raw, diagnostic_name):
    path = tmp_path / "ledger.jsonl"
    record = {"alpha_id": "A", "expression": "rank(close)", "status": "COMPLETE",
              "settings": {"simulationMode": "FULL"},
              "metrics": {"sharpe": 2.0, "fitness": 1.5}, "checks": raw}
    alpha_registry.record_simulation([record], {}, registry_path=path)
    written = json.loads(path.read_text())
    row, = alpha_registry.load_entries(path)
    with alpha_registry.indexed_registry(path) as index:
        assert index.entry("A") == row
    assert row["checks"] == written["checks"] == normalize_checks(raw)
    assert row["checks"][diagnostic_name]["raw_records"]
    assert alpha_registry.auto_verdict(row) == row["verdict"] == "watch"
    assert diagnostic_name in json.loads(registry_index.summary_fields(row)["unresolved_checks"])
    # Projection of a raw legacy list must not silently pre-filter its bad row.
    assert diagnostic_name in json.loads(registry_index.summary_fields(record)["unresolved_checks"])


def test_valid_refresh_recovers_after_client_parse_diagnostic(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    passed = [{"name": name, "result": "PASS"}
              for name in ("QUALITY", "SELF_CORRELATION", "MATCHES_COMPETITION")]
    record = {"alpha_id": "A", "expression": "rank(close)", "status": "COMPLETE",
              "settings": {"simulationMode": "FULL"},
              "metrics": {"sharpe": 2.0, "fitness": 1.5}, "checks": passed + [{"result": "FAIL"}]}
    monkeypatch.setattr(alpha_registry, "_now", lambda: "2026-11-20T00:00:00+00:00")
    alpha_registry.record_simulation([record], {}, registry_path=path)
    first, = alpha_registry.load_entries(path)
    assert first["verdict"] == "watch" and CLIENT_PARSE_ERROR in first["checks"]
    monkeypatch.setattr(alpha_registry, "_now", lambda: "2026-11-21T00:00:00+00:00")
    alpha_registry.record_simulation([{**record, "checks": passed}], {}, registry_path=path)
    latest, = alpha_registry.load_entries(path)
    assert latest["verdict"] == "candidate" and latest["checks_state"] == "observed"
    assert "last_check_snapshot" not in latest and CLIENT_PARSE_ERROR not in latest["checks"]
    assert triage._unresolved(latest) == []
    projection = registry_index.summary_fields(latest)
    assert projection["check_count"] == 3 and json.loads(projection["unresolved_checks"]) == []
    with alpha_registry.indexed_registry(path) as index:
        assert index.entry("A") == latest


def test_required_coverage_and_formal_metadata_are_independent_of_numeric_diagnostics():
    checks = [{"name": "PROD_CORRELATION", "result": "PASS", "value": 0.73,
               "limit": 0.7, "message": "platform exception", "details": {"x": 1}}]
    before = deepcopy(checks)
    normalized = normalize_checks(checks)
    assert normalized["PROD_CORRELATION"] == {key: value for key, value in checks[0].items() if key != "name"}
    assert summarize_checks(normalized)["passed"] == ["PROD_CORRELATION"]
    required = summarize_checks(normalized, required=("PROD_CORRELATION", "SELF_CORRELATION"))
    assert required["pending"] == ["SELF_CORRELATION"]
    assert required["resolved"] is False
    assert checks == before
    # The live-check view must retain the actual platform verdict even when the
    # displayed value exceeds the displayed limit.
    result = alphas._check_result("A", {"complete": True, "checks": normalized})
    formal = result["production_correlation_evidence"]["formal"]
    assert result["outcome"] == "passed"
    assert formal["result"] == "PASS" and formal["value_vs_limit"] == "above"


def test_warnings_are_resolved_consistently_in_live_checks_triage_and_shelf_summary():
    checks = {
        "SELF_CORRELATION": {"result": "PASS"},
        "MATCHES_COMPETITION": {"result": "PASS"},
        "UNITS": {"result": "WARNING", "message": "unit inference note"},
    }
    live = alphas._check_result("A", {"complete": True, "checks": checks})
    assert live["complete"] is True and live["outcome"] == "warning"
    assert live["checks"]["UNITS"] == checks["UNITS"]
    shelf = buffer_cmd._checks_summary(checks)
    assert set(shelf) == {"passed", "failed", "pending", "warned"}
    assert shelf["warned"] == ["UNITS"] and shelf["pending"] == shelf["failed"] == []
    row = {"delta": 0.1, "overlap_days": 200,
           "failing": triage._failing({"checks": checks}),
           "unresolved": triage._unresolved({"checks": checks})}
    verdict, _ = triage._verdict(row, {"retention_floor": 0.4, "min_delta": 0.01, "show_test": True})
    assert verdict == "shelve"


@pytest.mark.parametrize("checks", [
    {},
    {"SELF_CORRELATION": {"result": "PENDING"}},
    {"SELF_CORRELATION": {"result": "NEW_UNRECOGNIZED_STATE"}},
    {"SELF_CORRELATION": {"result": "ERROR"}},
])
def test_live_and_triage_do_not_treat_unknown_or_error_as_pass(checks):
    live = alphas._check_result("A", {"complete": True, "checks": checks})
    assert live["complete"] is False and live["checks_resolved"] is False
    assert live["outcome"] == "pending"
    assert triage._unresolved({"checks": checks})
    shelf = buffer_cmd._checks_summary(checks)
    assert not shelf["passed"]


@pytest.mark.parametrize("prior_verdict", ["PASS", "FAIL"])
def test_partial_ledger_refresh_stays_unverified_until_prior_check_returns(tmp_path, prior_verdict):
    path = tmp_path / "ledger.jsonl"
    checks = {"SELF_CORRELATION": {"result": "PASS"},
              "MATCHES_COMPETITION": {"result": "PASS"},
              "QUALITY": {"result": prior_verdict}}
    record = {"alpha_id": "A", "expression": "rank(close)", "status": "COMPLETE",
              "settings": {"simulationMode": "FULL"},
              "metrics": {"sharpe": 2.0, "fitness": 1.2}, "checks": checks}
    alpha_registry.record_simulation([record], {}, registry_path=path)
    partial = {name: body for name, body in checks.items() if name != "QUALITY"}
    alpha_registry.record_simulation([{**record, "checks": partial}], {}, registry_path=path)
    entry, = alpha_registry.load_entries(path)
    assert entry["checks_state"] == "partial"
    assert triage._failing(entry) == []
    assert triage._unresolved(entry) == ["QUALITY"]
    row = {"delta": 0.1, "overlap_days": 200,
           "failing": triage._failing(entry), "unresolved": triage._unresolved(entry)}
    criteria = {"retention_floor": 0.4, "min_delta": 0.01, "show_test": True}
    verdict, reason = triage._verdict(row, criteria)
    assert verdict == "unverified" and "QUALITY" in reason

    restored = {**partial, "QUALITY": {"result": "PASS"}}
    alpha_registry.record_simulation([{**record, "checks": restored}], {}, registry_path=path)
    entry, = alpha_registry.load_entries(path)
    assert triage._unresolved(entry) == []
    row.update(failing=triage._failing(entry), unresolved=triage._unresolved(entry))
    assert triage._verdict(row, criteria)[0] == "shelve"


@pytest.fixture
def shelf_context(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    chosen = tmp_path / "choice.json"
    selection.freeze(chosen, source="triage", rows=[{"alpha_id": "A"}])
    args = Namespace(alpha_id="A", buffer=str(tmp_path / "shelf.jsonl"),
                     registry=str(tmp_path / "ledger.jsonl"), selection=str(chosen),
                     name="Synthetic checked candidate", category="PRICE_VOLUME", tags=["synthetic"],
                     description="Synthetic offline metadata for the shelf eligibility regression. " * 2,
                     seeds=str(tmp_path / "seeds.jsonl"),
                     note=None, max_wait=None, yes=False, force=False)
    monkeypatch.setattr(buffer_cmd, "with_session", lambda _args, work: work(object()))
    monkeypatch.setattr(ml, "fetch_alpha_metrics", lambda *_args: {
        "id": "A", "status": "COMPLETE", "regular": {"code": "rank(close)"},
        "settings": {"simulationMode": "FULL"}, "is": {"sharpe": 2.0, "fitness": 1.2},
    })
    return args


@pytest.mark.parametrize("verdict,complete,eligible", [
    ("PASS", True, True),
    ("WARNING", True, True),
    ("PASS", False, False),
    ("PENDING", True, False),
    ("UNKNOWN", True, False),
    (None, True, False),
])
def test_actual_shelf_eligibility_requires_resolved_checks_and_completed_endpoint(
    shelf_context, verdict, complete, eligible
):
    checks = {"SELF_CORRELATION": {"result": verdict, "value": 0.4}}
    if verdict is None:
        checks = {}
    computed = {"complete": complete, "checks": checks,
                "self_correlation": {"result": verdict, "value": 0.4}}
    entry = buffer_cmd.cmd_buffer_add(shelf_context, None, checked=computed)["entry"]
    assert entry["eligible"] is eligible


@pytest.mark.parametrize("verdict", ["FAIL", "ERROR"])
def test_failed_or_errored_checks_cannot_enter_the_shelf(shelf_context, verdict):
    computed = {"complete": True, "checks": {"QUALITY": {"result": verdict}},
                "self_correlation": {"result": "PASS", "value": 0.4}}
    with pytest.raises(CliError, match="fails QUALITY"):
        buffer_cmd.cmd_buffer_add(shelf_context, None, checked=computed)
