import multiprocessing

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
