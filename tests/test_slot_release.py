"""A missing receipt cannot free a slot; terminal evidence stays account-scoped."""

from pathlib import Path

import cache_store
import simulation_slots as slots

ACCOUNT = {"fingerprint": "slot-release-test"}
OTHER_ACCOUNT = {"fingerprint": "other-slot-release-test"}
URL = "https://api.worldquantbrain.com/simulations/gone"


def _accepted(account, url):
    identity, _ = slots.reserve(account, 2)
    row = {"status": "PENDING", "submission_attempted": True,
           "slot_reservation": identity, "progress_url": url}
    slots.reconcile_record(row, account, outcome="accepted")
    return identity


def test_missing_receipt_keeps_actual_reservations_in_both_accounts(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    missing = _accepted(ACCOUNT, URL)
    live = _accepted(ACCOUNT, URL + "-alive")
    _accepted(OTHER_ACCOUNT, URL)
    paths = [Path(slots._paths(account["fingerprint"])[0])
             for account in (ACCOUNT, OTHER_ACCOUNT)]
    before = [path.read_bytes() for path in paths]
    reads = []

    def poll(url):
        reads.append(url)
        return "NOT_FOUND" if url == URL else "RUNNING"

    report = slots.reconcile_orphans(ACCOUNT, poll)

    assert report["released"] == []
    assert {item["id"] for item in report["kept"]} == {missing["id"], live["id"]}
    assert sorted(reads) == [URL, URL + "-alive"]
    assert [path.read_bytes() for path in paths] == before
    assert slots.snapshot(ACCOUNT, 2)["active"] == 2
    assert slots.snapshot(OTHER_ACCOUNT, 2)["active"] == 1


def test_terminal_release_reads_outside_lock_and_changes_only_the_observed_account(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    accepted = _accepted(ACCOUNT, URL)
    _accepted(OTHER_ACCOUNT, URL)
    _path, lock = slots._paths(ACCOUNT["fingerprint"])
    other_path = Path(slots._paths(OTHER_ACCOUNT["fingerprint"])[0])
    other_before = other_path.read_bytes()

    def poll(url):
        assert url == URL
        with cache_store.file_lock(lock, timeout=0, poll=0):
            return {"status": "COMPLETE", "alpha": "FinishedAlpha"}

    report = slots.reconcile_orphans(ACCOUNT, poll)

    assert [item["id"] for item in report["released"]] == [accepted["id"]]
    assert slots.snapshot(ACCOUNT, 2)["active"] == 0
    assert slots.snapshot(OTHER_ACCOUNT, 2)["active"] == 1
    assert other_path.read_bytes() == other_before
