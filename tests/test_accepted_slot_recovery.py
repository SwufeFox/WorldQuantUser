"""Accepted reservations recover from terminal platform evidence, not age."""

import json
from pathlib import Path

import pytest
import requests

import cache_store
import machine_lib as ml
import simulation_capabilities
import simulation_slots as slots


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    session = requests.Session()
    session._brain_account_id = "offline-accepted-recovery"
    yield session, simulation_capabilities.account_provenance(session)
    session.close()


def accept(account, simulation_id="Accepted", limit=4):
    identity, _ = slots.reserve(account, limit)
    row = {"status": "PENDING", "submission_attempted": True, "slot_reservation": identity,
           "progress_url": ml.BRAIN_API_URL + "/simulations/" + simulation_id}
    slots.reconcile_record(row, account, outcome="accepted")
    return row


def response(status, body, headers=None):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(body).encode()
    result.headers.update(headers or {})
    return result


def test_terminal_sweep_releases_slot_and_restores_stale_checkpoint_without_post(context, monkeypatch):
    session, account = context
    record = accept(account)
    reads = []

    def get(url, **kwargs):
        reads.append(url)
        assert kwargs["allow_redirects"] is False
        return response(200, {"id": "Accepted", "status": "COMPLETE", "alpha": "FinishedAlpha"})

    monkeypatch.setattr(session, "get", get)
    monkeypatch.setattr(session, "post", lambda *_a, **_k: pytest.fail("recovery must never POST"))
    assert slots.snapshot(account, 1)["active"] == 1
    report = ml.recover_accepted_slots(session, account)
    assert len(report["released"]) == 1 and report["checked"] == 1
    assert slots.snapshot(account, 1)["active"] == 0
    slots.reconcile_record(record, account)
    assert record["alpha_id"] == "FinishedAlpha"
    assert record["simulation_status"] == "COMPLETE"
    assert record["metrics_pending"] is True
    assert record["slot_reservation_state"] == "released"
    assert "progress_url" not in record
    slots.reconcile_record(record, account)
    assert slots.snapshot(account, 1)["active"] == 0
    assert len(reads) == 1


def test_native_parent_completion_releases_one_slot_without_inventing_child_results(context, monkeypatch):
    session, account = context
    parent = accept(account, "Parent")
    monkeypatch.setattr(session, "get", lambda *_a, **_k: response(
        200, {"id": "Parent", "status": "COMPLETE", "children": ["ChildOne", "ChildTwo"]}))
    report = ml.recover_accepted_slots(session, account)
    assert len(report["released"]) == 1
    slots.reconcile_record(parent, account)
    assert parent["status"] == "COMPLETE"
    assert not parent.get("alpha_id") and not parent.get("metrics_pending")
    assert slots.snapshot(account, 4)["active"] == 0


@pytest.mark.parametrize("code,body,headers", [
    (503, {}, {}),
    (200, {"id": "Accepted", "status": "RUNNING"}, {}),
    (200, {"id": "Other", "status": "COMPLETE", "alpha": "FinishedAlpha"}, {}),
    (200, {"id": "Accepted", "status": "COMPLETE", "alpha": "FinishedAlpha"}, {"Retry-After": "300"}),
    (200, {"id": "Accepted", "status": "COMPLETE", "alpha": "FinishedAlpha"}, {"Retry-After": "nan"}),
])
def test_missing_pending_mismatched_and_unready_evidence_cannot_release(context, monkeypatch, code, body, headers):
    session, account = context
    accept(account)
    monkeypatch.setattr(session, "get", lambda *_a, **_k: response(code, body, headers))
    report = ml.recover_accepted_slots(session, account)
    assert report["released"] == []
    assert slots.snapshot(account, 1)["active"] == 1


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_preserves_reservation_and_reaches_caller(context, monkeypatch, status):
    session, account = context
    accept(account)
    monkeypatch.setattr(session, "get", lambda *_a, **_k: response(status, {}))
    with pytest.raises(ml.AuthError):
        ml.recover_accepted_slots(session, account)
    assert slots.snapshot(account, 1)["active"] == 1


def test_sweep_has_one_budget_and_a_slow_first_read_leaves_time_for_other_jobs(context, monkeypatch):
    session, account = context
    first, second = accept(account, "Slow"), accept(account, "Done")
    elapsed, timeouts = [100.0], []
    monkeypatch.setattr(ml.time, "monotonic", lambda: elapsed[0])

    def get(url, **kwargs):
        timeouts.append(kwargs["timeout"])
        if url.endswith("Slow"):
            elapsed[0] += kwargs["timeout"]
            raise requests.Timeout("slow first read")
        return response(200, {"id": "Done", "status": "COMPLETE", "alpha": "FinishedAlpha"})

    monkeypatch.setattr(session, "get", get)
    report = ml.recover_accepted_slots(session, account, deadline=104)
    assert timeouts == [2.0, 2.0]
    assert [item["id"] for item in report["released"]] == [second["slot_reservation"]["id"]]
    assert [item["id"] for item in report["kept"]] == [first["slot_reservation"]["id"]]
    assert slots.snapshot(account, 4)["active"] == 1


def test_sweep_does_not_release_a_reservation_changed_during_read(context):
    _session, account = context
    record = accept(account)
    path = Path(slots._paths(account["fingerprint"])[0])

    def changed(_url):
        journal, _ = cache_store.load_json(path)
        journal["reservations"][record["slot_reservation"]["id"]]["progress_url"] = (
            ml.BRAIN_API_URL + "/simulations/NewLocation")
        cache_store.write_json_atomic(path, journal)
        return {"status": "COMPLETE", "alpha": "FinishedAlpha"}

    report = slots.reconcile_orphans(account, changed)
    assert report["released"] == []
    assert report["kept"][0]["reason"] == "reservation_changed"
    assert slots.snapshot(account, 1)["active"] == 1


def test_unknown_post_without_location_is_not_released_by_age(context):
    _session, account = context
    slots.reserve(account, 1)
    report = slots.reconcile_orphans(account, lambda _url: pytest.fail("unknown POST has no poll target"),
                                     now=10**12, max_age_min=0, hard_max_age_min=0)
    assert report["released"] == []
    assert report["kept"][0]["overdue"] is True
    assert slots.snapshot(account, 1)["active"] == 1


def test_unknown_or_other_account_cannot_poll_or_release(context, monkeypatch):
    session, account = context
    accept(account)
    monkeypatch.setattr(session, "get", lambda *_a, **_k: pytest.fail("wrong account"))
    for other in ({}, {"fingerprint": "other"}):
        assert ml.recover_accepted_slots(session, other)["state"] == "account_unverified"
    assert slots.snapshot(account, 1)["active"] == 1


@pytest.mark.parametrize("delete_status,state", [(200, "cancelled"), (204, "cancelled"),
                                                  (404, "already_gone")])
def test_unchanged_progress_past_threshold_cancels_with_evidence(
        context, monkeypatch, delete_status, state):
    session, account = context
    clock = [1000.0]
    monkeypatch.setattr(slots.time, "time", lambda: clock[0])
    record = accept(account)
    monkeypatch.setattr(session, "get", lambda *_a, **_k: response(
        200, {"id": "Accepted", "status": "RUNNING", "progress": 0.10}))

    first = ml.recover_accepted_slots(session, account)
    assert first["released"] == []
    assert slots.snapshot(account, 1)["active"] == 1

    monkeypatch.setattr(session, "delete", lambda *_a, **_k: response(delete_status, {}))
    clock[0] += 46 * 60
    second = ml.recover_accepted_slots(session, account)
    assert len(second["cancelled"]) == 1
    assert second["released"][0]["state"] == state
    assert second["released"][0]["status_code"] == delete_status
    assert slots.snapshot(account, 1)["active"] == 0

    path, _ = slots._paths(account["fingerprint"])
    journal = slots._load(path, account["fingerprint"])
    evidence = journal["cancelled"][record["slot_reservation"]["id"]]
    assert evidence["state"] == state
    assert evidence["progress_url"] == record["progress_url"]
    assert evidence["progress"] == 0.10
    assert evidence["last_progress_at"] == 1000.0
    assert evidence["cancel_evidence"]["minutes_unchanged"] == 46.0
    assert evidence["cancel_evidence"]["status_code"] == delete_status
    assert evidence["cancel_evidence"]["cancelled_at"] == 1000.0 + 46 * 60

    slots.reconcile_record(record, account)
    assert record["slot_reservation_state"] == state
    assert record["slot_cancel_evidence"]["progress_url"] == record["progress_url"]
    assert slots.snapshot(account, 1)["active"] == 0


def test_progress_that_advances_is_never_cancelled(context, monkeypatch):
    session, account = context
    clock, progress = [1000.0], [0.10]
    monkeypatch.setattr(slots.time, "time", lambda: clock[0])
    accept(account)
    monkeypatch.setattr(session, "get", lambda *_a, **_k: response(
        200, {"id": "Accepted", "status": "RUNNING", "progress": progress[0]}))
    monkeypatch.setattr(session, "delete", lambda *_a, **_k: pytest.fail(
        "a progressing simulation must never be cancelled"))

    ml.recover_accepted_slots(session, account)
    clock[0] += 46 * 60
    progress[0] = 0.20
    report = ml.recover_accepted_slots(session, account)
    assert report["cancelled"] == []
    assert slots.snapshot(account, 1)["active"] == 1
    path, _ = slots._paths(account["fingerprint"])
    entry = next(iter(slots._load(path, account["fingerprint"])["reservations"].values()))
    assert entry["progress"] == 0.20
    assert entry["last_progress_at"] == clock[0]


def test_get_404_records_already_gone_and_releases_local_slot(context, monkeypatch):
    session, account = context
    record = accept(account)
    monkeypatch.setattr(session, "get", lambda *_a, **_k: response(404, {}))
    monkeypatch.setattr(session, "delete", lambda *_a, **_k: pytest.fail(
        "an already cancelled simulation must not be deleted again"))

    report = ml.recover_accepted_slots(session, account)
    assert report["released"][0]["state"] == "already_gone"
    assert report["released"][0]["status_code"] == 404
    assert slots.snapshot(account, 1)["active"] == 0
    path, _ = slots._paths(account["fingerprint"])
    evidence = slots._load(path, account["fingerprint"])["cancelled"][
        record["slot_reservation"]["id"]]
    assert evidence["state"] == "already_gone"
    assert evidence["progress_url"] == record["progress_url"]


def test_single_simulation_admission_recovers_completed_external_reservation(context, monkeypatch):
    session, account = context
    accept(account, "External", limit=1)
    monkeypatch.setattr(session, "get", lambda *_a, **_k: response(
        200, {"id": "External", "status": "COMPLETE", "alpha": "ExternalAlpha"}))
    posts = []

    def submit(_session, payload, **_kwargs):
        posts.append(payload)
        return ml.BRAIN_API_URL + "/simulations/NewTask"

    monkeypatch.setattr(ml, "submit_single_simulation", submit)
    monkeypatch.setattr(ml, "poll_simulation", lambda *_a, **_k: {"status": "PENDING"})
    scheduling = {}
    records = ml.simulate_alphas(session, ["rank(close)"], concurrency=1, capability_snapshot={},
                                scheduling_report=scheduling, max_wait_seconds=1)
    assert len(posts) == 1
    assert records[0]["progress_url"].endswith("/NewTask")
    assert len(scheduling["accepted_slot_recovery"]["released"]) == 1
    assert slots.snapshot(account, 1)["active"] == 1


def test_native_resume_does_not_resurrect_a_swept_parent_while_children_are_unknown(context, monkeypatch):
    session, account = context
    session._brain_permissions = ["MULTI_SIMULATION"]
    expressions = ["rank(close)", "rank(open)"]
    monkeypatch.setattr(ml, "submit_multi_simulation", lambda *_a, **_k: ml.BRAIN_API_URL + "/simulations/Parent")
    original_read = ml.read_simulation_snapshot
    monkeypatch.setattr(ml, "read_simulation_snapshot", lambda *_a, **_k: ({"status": "PENDING"}, None))
    records = ml.simulate_alphas(session, expressions, multi_size=2, capability_snapshot={}, max_wait_seconds=1)
    assert slots.snapshot(account, 1)["active"] == 1
    monkeypatch.setattr(ml, "read_simulation_snapshot", original_read)
    monkeypatch.setattr(session, "get", lambda *_a, **_k: response(
        200, {"id": "Parent", "status": "COMPLETE", "children": ["ChildOne", "ChildTwo"]}))
    assert len(ml.recover_accepted_slots(session, account)["released"]) == 1
    monkeypatch.setattr(session, "get", lambda *_a, **_k: response(503, {}))
    monkeypatch.setattr(ml, "submit_multi_simulation", lambda *_a, **_k: pytest.fail("resume must not POST"))
    restored = ml.simulate_alphas(session, expressions, multi_size=2, capability_snapshot={},
                                 initial_records=records, recover_only=True, max_wait_seconds=1)
    assert slots.snapshot(account, 1)["active"] == 0
    assert all(not row.get("alpha_id") and row["status"] == "PENDING" for row in restored)
