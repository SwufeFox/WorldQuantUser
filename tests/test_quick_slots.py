import multiprocessing
from pathlib import Path

import pytest

import simulation_slots as slots


ACCOUNT = {"fingerprint": "quick-test"}
URL = "https://api.worldquantbrain.com/simulations/first"


def _hold_slot(ready, release, location, prepare=False):
    reservation = slots.quick_acquire(ACCOUNT, 1, "first", prepare=prepare)
    if location:
        slots.quick_update(ACCOUNT, reservation["id"], "accepted", progress_url=location)
    ready.set()
    release.wait(10)


@pytest.mark.parametrize("location", [URL, None])
def test_cross_process_capacity_and_recovery_after_holder_exits(tmp_path, monkeypatch, location):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(target=_hold_slot, args=(ready, release, location))
    process.start()
    try:
        assert ready.wait(5)
        assert slots.quick_orphans(ACCOUNT) == []
        assert slots.quick_acquire(ACCOUNT, 2, "another") is None
        assert slots.quick_acquire(ACCOUNT, 1, "first") is None
        assert slots.reserve(ACCOUNT, 2)[0] is None
        assert slots.snapshot(ACCOUNT, 2)["available"] == 0
    finally:
        release.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert process.exitcode == 0
    assert slots.quick_acquire(ACCOUNT, 1, "another") is None
    if not location:
        assert slots.quick_orphans(ACCOUNT) == []
        assert slots.quick_acquire(ACCOUNT, 1, "first")["state"] == "unknown"
        return
    orphan, = slots.quick_orphans(ACCOUNT)
    assert orphan["job_key"] == "first"
    assert orphan["progress_url"] == URL
    assert slots.quick_update(ACCOUNT, orphan["id"], "terminal", retain=True,
                              terminal_result={"status": "COMPLETE", "alpha_id": "SomeAlpha",
                                               "ignored_metadata": 0})
    replacement = slots.quick_acquire(ACCOUNT, 1, "another")
    assert replacement["state"] == "submitting"
    receipt = slots.quick_acquire(ACCOUNT, 1, "first")
    assert receipt["resolved"] and receipt["progress_url"] == URL
    assert receipt["terminal_result"] == {"status": "COMPLETE", "alpha_id": "SomeAlpha"}
    assert slots.quick_update(ACCOUNT, receipt["id"], "terminal")
    assert slots.snapshot(ACCOUNT, 1)["active"] == 1


def test_parking_preserves_shared_occupancy_and_location(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    job = slots.quick_acquire(ACCOUNT, 1, "first", existing_url=URL)
    assert job["state"] == "accepted"
    assert slots.quick_update(ACCOUNT, job["id"], "parked")
    assert slots.quick_acquire(ACCOUNT, 1, "another") is None
    orphan, = slots.quick_orphans(ACCOUNT)
    assert orphan["progress_url"] == URL
    assert slots.quick_orphans(ACCOUNT, exclude_job="first") == []
    assert slots.quick_acquire(ACCOUNT, 1, "first")["id"] == job["id"]
    assert slots.quick_orphans(ACCOUNT) == []


def test_killed_preparation_frees_slot_before_any_post(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(target=_hold_slot, args=(ready, release, None, True))
    process.start()
    try:
        assert ready.wait(5)
        assert slots.quick_acquire(ACCOUNT, 1, "another", prepare=True) is None
    finally:
        process.kill()
        process.join(5)
    replacement = slots.quick_acquire(ACCOUNT, 1, "another", prepare=True)
    assert replacement["state"] == "quick_preparing"
    assert slots.quick_known_jobs(ACCOUNT) == {"another"}
    assert slots.quick_update(ACCOUNT, replacement["id"], "submitting")
    assert not slots.quick_update(ACCOUNT, replacement["id"], "submitting")


def test_submission_backoff_keeps_the_latest_retry_time_in_the_same_scope(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(workspace))
    monkeypatch.setattr(slots.time, "time", lambda: 100)
    assert slots.defer_submission(ACCOUNT, 30) == 130
    assert slots.defer_submission(ACCOUNT, 5) == 130
    assert slots.submission_retry_at(ACCOUNT) == 130
    assert slots.submission_retry_at({"fingerprint": "another"}) is None

    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path / "another-workspace"))
    assert slots.submission_retry_at(ACCOUNT) is None
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(workspace))
    monkeypatch.setattr(slots.time, "time", lambda: 130)
    assert slots.submission_retry_at(ACCOUNT) is None


def test_admission_evidence_counts_only_distinct_durable_ordinary_acceptances(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    path = Path(slots._paths(ACCOUNT["fingerprint"])[0])
    accepted = {"state": "accepted", "progress_url": URL}
    slots.cache_store.write_json_atomic(path, {
        "version": 1, "account_fingerprint": ACCOUNT["fingerprint"],
        "reservations": {
            "ordinary": accepted,
            "alias": {**accepted, "progress_url": URL.replace(".com/", ".com:443/")},
            "legacy_quick": {**accepted, "quick_job_key": "legacy"},
            "preparing": {**accepted, "state": "preparing"},
            "unknown": {**accepted, "state": "unknown"},
            "invalid": {**accepted, "progress_url": "https://example.invalid/simulations/first"},
            "without_location": {"state": "accepted"},
        },
    })
    before = path.read_bytes()
    assert slots.confirmed_accepted_count(ACCOUNT) == 1
    assert slots.confirmed_accepted_count({"fingerprint": "another"}) == 0
    assert slots.confirmed_accepted_count({}) is None
    assert path.read_bytes() == before
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path / "another-workspace"))
    assert slots.confirmed_accepted_count(ACCOUNT) == 0
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    path.write_text("unreadable", encoding="utf-8")
    assert slots.confirmed_accepted_count(ACCOUNT) is None


def test_dead_quick_owner_does_not_cap_the_workspace(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    account = {"fingerprint": "dead-owner-test"}
    path, _lock = slots._paths(slots._fingerprint(account))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    journal = {
        "version": 1,
        "account_fingerprint": slots._fingerprint(account),
        "reservations": {
            "stale": {"state": "accepted", "quick_limit": 1,
                      "quick_owner": {"pid": 999999, "start": "0"}},
        },
    }
    Path(path).write_text(json.dumps(journal))
    identity, state = slots.reserve(account, 4)
    assert identity is not None
    assert state["active"] == 2


def test_reconcile_releases_terminal_missing_and_old_orphans(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    first = slots.quick_acquire(ACCOUNT, 4, "done")
    slots.quick_update(ACCOUNT, first["id"], "accepted", progress_url="https://api.worldquantbrain.com/simulations/done")
    second = slots.quick_acquire(ACCOUNT, 4, "gone")
    slots.quick_update(ACCOUNT, second["id"], "accepted", progress_url="https://api.worldquantbrain.com/simulations/gone")
    third = slots.quick_acquire(ACCOUNT, 4, "healthy")
    slots.quick_update(ACCOUNT, third["id"], "accepted", progress_url="https://api.worldquantbrain.com/simulations/live")

    statuses = {"https://api.worldquantbrain.com/simulations/done":
                    {"status": "COMPLETE", "alpha": "ALPHA1"},
                "https://api.worldquantbrain.com/simulations/gone": "NOT_FOUND",
                "https://api.worldquantbrain.com/simulations/live": "RUNNING"}
    result = slots.reconcile_orphans(ACCOUNT, statuses.get)
    released = {item["id"] for item in result["released"]}
    assert {item["alpha"] for item in result["released"] if item["alpha"]} == {"ALPHA1"}
    assert released == {first["id"], second["id"]}
    assert [item["id"] for item in result["kept"]] == [third["id"]]
    assert slots.snapshot(ACCOUNT, 10)["active"] == 1


def test_reconcile_releases_unreadable_only_past_max_age(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    old = slots.quick_acquire(ACCOUNT, 1, "old")
    slots.quick_update(ACCOUNT, old["id"], "accepted", progress_url="https://api.worldquantbrain.com/simulations/old")
    young = slots.reconcile_orphans(ACCOUNT, lambda url: None, max_age_min=45)
    assert young["released"] == []
    aged = slots.reconcile_orphans(ACCOUNT, lambda url: None, max_age_min=0, now=10**12)
    assert [item["id"] for item in aged["released"]] == [old["id"]]
