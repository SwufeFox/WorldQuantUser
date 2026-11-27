"""Bulk admission controls new POSTs while preserving durable recovery evidence."""

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from unittest import mock

import pytest
import requests

import alpha_registry
import machine_lib as ml
import simulation_capabilities
import simulation_slots
from tools.pipeline import batch_quick as quick


ACCOUNT = {"state": "known", "fingerprint": "quick-admission-account"}
FIELDS = [
    {"id": "close", "type": "MATRIX", "dataset": {"id": "prices"}},
    {"id": "analyst_signal", "type": "MATRIX", "dataset": "analyst"},
    {"id": "sentiment_signal", "type": "MATRIX", "dataset": {"id": "sentiment"}},
]


def test_worker_handoff_parks_fresh_spec_without_changing_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    inbox = tmp_path / "scope" / "backlog"
    claims = inbox.parent / "backlog-claimed"
    inbox.mkdir(parents=True)
    claims.mkdir()
    claim = claims / "w3-fresh.json"
    claim.write_text('{"name":"fresh"}')
    switch = tmp_path / "quick-worker-handoff.json"
    switch.write_text(json.dumps({"worker": "w3", "inbox": str(inbox)}))
    monkeypatch.setattr(quick.time, "sleep", lambda _seconds: switch.unlink())
    args = Namespace(arms=str(tmp_path / "arms-w3-fresh.json"),
                     out=str(tmp_path / "out-w3-fresh.jsonl"),
                     recover_only=False, admission_only=False, recover_job=None)
    assert quick._handoff_watcher(args)
    assert claim.samefile(inbox / "fresh.json")
    assert (inbox.parent / ".w3-parked.json").is_file()
    assert not Path(args.out).exists()


def response(status, body=None, **headers):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers)
    result._content = json.dumps(body or {}).encode()
    return result


def arms():
    return [{"label": row["id"], "expr": "rank(%s)" % row["id"],
             "idea": "Test whether this measured input predicts subsequent returns.",
             "settings": {"region": "USA", "universe": "TOP3000", "delay": 1}}
            for row in FIELDS]


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    monkeypatch.setattr(simulation_capabilities, "account_provenance", lambda session=None: ACCOUNT)
    monkeypatch.setattr(ml, "concurrency_plan", lambda *args, **kwargs: {"account": ACCOUNT, "effective": 3})
    session = mock.MagicMock()
    session.__enter__.return_value = session

    def post(_url, **kwargs):
        field = kwargs["json"]["regular"].split("(")[1].split(")")[0]
        return response(201, Location=ml.BRAIN_API_URL + "/simulations/" + field)

    def get(url, **kwargs):
        if "/simulations/" in url:
            return response(200, {"status": "COMPLETE", "alpha": "Alpha_" + url.rsplit("/", 1)[-1]})
        assert "/alphas/" in url
        return response(200, {"is": {"sharpe": 1.2, "fitness": 0.7, "checks": []}})

    session.post.side_effect = post
    session.get.side_effect = get
    monkeypatch.setattr(ml, "load_session", lambda **kwargs: session)
    cached = mock.Mock(side_effect=lambda settings, **kwargs: {
        "records": deepcopy(FIELDS), "evidence": {"state": "verified", "scope": settings,
                                                 "source": "private_snapshot"}})
    required = mock.Mock(side_effect=AssertionError("a fresh full-scope cache must avoid catalog reads"))
    monkeypatch.setattr(quick.capabilities, "cached_field_scope", cached)
    monkeypatch.setattr(quick.capabilities, "required_field_scope", required)
    path, out = tmp_path / "arms.json", tmp_path / "quick.jsonl"
    args = Namespace(arms=str(path), account_slots=3, poll_timeout=2, limit=0, concurrency=2)

    def run(rows, limit=0):
        path.write_text(json.dumps(rows), encoding="utf-8")
        args.limit = limit
        return quick._run_batch(args, out)

    return Namespace(run=run, session=session, cached=cached, required=required, out=out, args=args)


@pytest.mark.parametrize("idea", [None, "", " \n\t ", 42])
def test_bad_idea_anywhere_in_input_blocks_even_a_limited_run(workflow, idea):
    rows = arms()
    rows[-1]["idea"] = idea
    assert workflow.run(rows, limit=1) == 2
    workflow.session.post.assert_not_called()
    workflow.session.get.assert_not_called()
    workflow.cached.assert_not_called()
    workflow.required.assert_not_called()
    result = json.loads(workflow.out.read_text())
    assert result["status"] == "BLOCKED"
    assert result["admission_codes"] == ["idea_required"]
    receipt = json.loads(Path(result["admission_path"]).read_text())
    assert len(receipt["arms"]) == 3
    assert receipt["blocking"][0]["label"] == rows[-1]["label"]
    assert receipt["new_posts_allowed"] is False and receipt["recovery_allowed"] is True
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 0
    assert alpha_registry.load_entries() == []


def test_limit_does_not_hide_ninth_skeleton_variant(workflow):
    rows = [{**arms()[index % 3], "label": "arm%d" % index,
             "expr": "ts_mean(%s, %d)" % (FIELDS[index % 3]["id"], index + 2)} for index in range(9)]
    assert workflow.run(rows, limit=1) == 2
    result = json.loads(workflow.out.read_text())
    assert "skeleton_limit_exceeded" in result["admission_codes"]
    workflow.session.post.assert_not_called()
    workflow.cached.assert_not_called()
    receipt = json.loads(Path(result["admission_path"]).read_text())
    assert receipt["skeleton_groups"][0]["count"] == 9


def test_verified_batch_posts_and_reuses_scope_evidence_once(workflow):
    assert workflow.run(arms()) == 0
    assert workflow.session.post.call_count == 3
    workflow.cached.assert_called_once()
    assert workflow.cached.call_args.kwargs["account"] == ACCOUNT
    workflow.required.assert_not_called()
    receipt = json.loads(workflow.out.with_suffix(".jsonl.admission.json").read_text())
    assert receipt["diversity"]["verified_count"] == 3
    assert receipt["new_posts_allowed"] is True
    assert {row["status"] for row in quick._jsonl_rows(workflow.out)} == {"COMPLETE"}
    assert len(alpha_registry.load_entries()) == 3
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 0


@pytest.mark.parametrize("valid", [True, False])
def test_admission_only_leaves_submission_and_recording_untouched(workflow, valid):
    rows = arms()
    if not valid:
        rows[-1]["idea"] = ""
    workflow.args.admission_only = True

    assert workflow.run(rows) == (0 if valid else 2)
    receipt = json.loads(workflow.out.with_suffix(".jsonl.admission.json").read_text())
    assert receipt["new_posts_allowed"] is valid
    workflow.session.post.assert_not_called()
    assert not workflow.out.exists()
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 0


@pytest.mark.parametrize("unknown, code", [(False, "dataset_diversity_insufficient"),
                                          (True, "dataset_diversity_unverified")])
def test_proven_shortfall_and_missing_evidence_are_distinct(workflow, unknown, code):
    records = deepcopy(FIELDS)
    if unknown:
        records.pop()
    else:
        records[-1]["dataset"] = records[0]["dataset"]
    workflow.cached.side_effect = lambda settings, **kwargs: {
        "records": records, "evidence": {"state": "verified", "scope": settings}}
    assert workflow.run(arms()) == 2
    workflow.session.post.assert_not_called()
    assert all(code in row["admission_codes"] for row in quick._jsonl_rows(workflow.out))


def test_recover_only_records_accepted_arm_without_starting_other_arms(workflow):
    rows = arms()
    accepted = rows[1]
    runner = quick.Runner(workflow.out, ACCOUNT, 3, 2)
    key = runner._job_key(accepted["label"], quick._payload(accepted))
    slot = simulation_slots.quick_acquire(
        ACCOUNT, 3, key, existing_url=ml.BRAIN_API_URL + "/simulations/prior",
    )
    simulation_slots.quick_update(ACCOUNT, slot["id"], "parked")
    runner.close()
    workflow.args.recover_only = True

    assert workflow.run(rows) == 0
    workflow.session.post.assert_not_called()
    workflow.cached.assert_not_called()
    workflow.required.assert_not_called()
    results = quick._jsonl_rows(workflow.out)
    assert [(row["label"], row["alpha_id"]) for row in results] == [(accepted["label"], "Alpha_prior")]
    assert [entry["alpha_id"] for entry in alpha_registry.load_entries()] == ["Alpha_prior"]


@pytest.mark.parametrize("kind", ["accepted", "unknown", "preparing"])
def test_blocked_batch_recovers_existing_jobs_without_new_posts(workflow, kind):
    rows = arms()[:2]
    rows[0].pop("idea")  # Historical evidence must remain recordable.
    runner = quick.Runner(workflow.out, ACCOUNT, 3, 2)
    key = runner._job_key(rows[0]["label"], quick._payload(rows[0]))
    url = ml.BRAIN_API_URL + "/simulations/prior"
    slot = simulation_slots.quick_acquire(ACCOUNT, 3, key,
        existing_url=url if kind == "accepted" else None, prepare=kind == "preparing")
    if kind == "accepted":
        simulation_slots.quick_update(ACCOUNT, slot["id"], "parked")
    elif kind == "unknown":
        simulation_slots.quick_update(ACCOUNT, slot["id"], "unknown")
    runner.close()

    assert workflow.run(rows) == 2
    workflow.session.post.assert_not_called()
    workflow.required.assert_not_called()
    results = {row["label"]: row for row in quick._jsonl_rows(workflow.out)}
    assert results[rows[1]["label"]]["status"] == "BLOCKED"
    assert results[rows[0]["label"]]["status"] == {
        "accepted": "COMPLETE", "unknown": "UNKNOWN_SUBMISSION", "preparing": "BLOCKED"}[kind]
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == (1 if kind == "unknown" else 0)
    recorded = alpha_registry.load_entries()
    assert len(recorded) == (1 if kind == "accepted" else 0)
    if kind == "accepted":
        assert recorded[0]["alpha_id"] == "Alpha_prior"
    else:
        workflow.session.get.assert_not_called()


@pytest.mark.parametrize("state", ["accepted", "unknown"])
def test_shared_journal_only_recovery_does_not_query_catalog(workflow, state):
    row = arms()[0]
    runner = quick.Runner(workflow.out, ACCOUNT, 3, 2)
    key = runner._job_key(row["label"], quick._payload(row))
    slot = simulation_slots.quick_acquire(ACCOUNT, 3, key,
        existing_url=ml.BRAIN_API_URL + "/simulations/prior" if state == "accepted" else None)
    simulation_slots.quick_update(ACCOUNT, slot["id"], "parked" if state == "accepted" else "unknown")
    runner.close()
    assert workflow.run([row]) == 0
    workflow.cached.assert_not_called()
    workflow.required.assert_not_called()
    workflow.session.post.assert_not_called()
    assert json.loads(workflow.out.read_text())["status"] == (
        "COMPLETE" if state == "accepted" else "UNKNOWN_SUBMISSION")


@pytest.mark.parametrize("state", ["UNKNOWN_SUBMISSION", "UNVERIFIED_RESULT"])
@pytest.mark.parametrize("raw_key", [False, True, "rerendered"])
@pytest.mark.parametrize("stale_retry", [False, True])
def test_append_only_uncertain_result_survives_missing_sidecars(workflow, state, raw_key, stale_retry):
    arm = arms()[0]
    runner = quick.Runner(workflow.out, ACCOUNT, 3, 2)
    payload = quick._payload(arm)
    key = runner._raw_key(arm["label"], payload) if raw_key else runner._job_key(arm["label"], payload)
    previous = {"label": arm["label"], "expr": arm["expr"], "status": state,
                "simulation_job_key": key, "simulation_key": quick._simulation_key(payload),
                "account_fingerprint": ACCOUNT["fingerprint"],
                "observed_alpha_id": "Unverified1", "detail": "original identity evidence missing"}
    workflow.out.write_text(json.dumps(previous) + "\n")
    assert not runner.state_path.exists()
    if stale_retry:
        runner._checkpoint(key, {"label": arm["label"], "expr": arm["expr"], "next_retry_at": 1})
    if raw_key == "rerendered":
        arm["expr"] = "(( rank( close ) ))"
    assert workflow.run([arm]) == 0
    workflow.session.post.assert_not_called()
    workflow.session.get.assert_not_called()
    workflow.cached.assert_not_called()
    latest = list(quick._jsonl_rows(workflow.out))[-1]
    assert latest["status"] == state and latest["simulation_job_key"] == key
    assert latest["observed_alpha_id"] == previous["observed_alpha_id"]
    assert latest["detail"] == previous["detail"]
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 0


def test_ledger_only_unverified_url_is_read_again_and_its_payload_verified(workflow):
    arm = arms()[0]
    runner = quick.Runner(workflow.out, ACCOUNT, 3, 2)
    payload = quick._payload(arm)
    key = runner._job_key(arm["label"], payload)
    url = ml.BRAIN_API_URL + "/simulations/prior"
    previous = {"label": arm["label"], "expr": arm["expr"], "status": "UNVERIFIED_RESULT",
                "simulation_status": "COMPLETE", "alpha_id": None, "observed_alpha_id": "PriorAlpha",
                "progress_url": url, "simulation_job_key": key, "simulation_key": quick._simulation_key(payload),
                "account_fingerprint": ACCOUNT["fingerprint"]}
    workflow.out.write_text(json.dumps(previous) + "\n")

    def get(target, **kwargs):
        if target == url:
            return response(200, {"status": "COMPLETE", "alpha": "PriorAlpha"})
        assert target == ml.BRAIN_API_URL + "/alphas/PriorAlpha"
        return response(200, {**payload, "settings": {**payload["settings"], "language": "FASTEXPR"},
                              "id": "PriorAlpha", "is": {"sharpe": 1.2, "checks": []}})

    workflow.session.get.side_effect = get
    assert workflow.run([arm]) == 0
    workflow.session.post.assert_not_called()
    workflow.cached.assert_not_called()
    assert workflow.session.get.call_count == 2
    latest = list(quick._jsonl_rows(workflow.out))[-1]
    assert latest["status"] == "COMPLETE" and latest["identity_verified"] is True
    assert latest["alpha_id"] == "PriorAlpha"
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 0


def test_catalog_budget_is_shared_and_rate_limit_stops_live_reads(workflow, monkeypatch):
    rows = arms()
    rows[1]["settings"]["region"] = "EUR"
    rows[2]["settings"]["region"] = "ASI"
    clock = [100.0]
    monkeypatch.setattr(quick.time, "monotonic", lambda: clock[0])
    workflow.cached.side_effect = lambda settings, **kwargs: {"evidence": {"state": "unknown"}}
    calls = []

    def verify(session, scope, required, *, deadline, proof_cache):
        calls.append((scope, required, deadline, proof_cache))
        clock[0] += 1
        if len(calls) == 1:
            evidence = {"state": "unknown", "catalogs": [{"error": "RATE_LIMITED", "retry_at": 999.0}]}
            records = FIELDS[:1]
        else:
            # required_field_scope must receive an exhausted deadline, so only
            # its existing positive-cache path can supply later observations.
            assert deadline <= clock[0]
            evidence = {"state": "verified", "catalogs": []}
            records = [FIELDS[len(calls) - 1]]
        return {"verified_records": records, "evidence": evidence}

    workflow.required.side_effect = verify
    result = quick._build_admission(rows, workflow.session, ACCOUNT)
    assert calls[0][2] == 130.0
    assert calls[1][2] == calls[2][2] == 101.0
    assert calls[0][3] is calls[1][3] is calls[2][3]
    assert result["next_retry_at"] == 999.0
    # Three independent positive observations prove the count even though the
    # first scope's full verification could not finish within its budget.
    assert result["ok"] is True and result["diversity"]["verified_count"] == 3
