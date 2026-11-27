"""A missing receipt cannot free a slot; terminal evidence stays account-scoped."""

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
    fingerprints = [account["fingerprint"] for account in (ACCOUNT, OTHER_ACCOUNT)]
    paths = [slots._paths(fingerprint)[0] for fingerprint in fingerprints]
    before = [slots._load(path, fingerprint) for path, fingerprint in zip(paths, fingerprints)]
    reads = []

    def poll(url):
        reads.append(url)
        return "NOT_FOUND" if url == URL else "RUNNING"

    report = slots.reconcile_orphans(ACCOUNT, poll)

    assert report["released"] == []
    assert {item["id"] for item in report["kept"]} == {missing["id"], live["id"]}
    assert sorted(reads) == [URL, URL + "-alive"]
    assert [slots._load(path, fingerprint) for path, fingerprint in zip(paths, fingerprints)] == before
    assert slots.snapshot(ACCOUNT, 2)["active"] == 2
    assert slots.snapshot(OTHER_ACCOUNT, 2)["active"] == 1


def test_terminal_release_reads_outside_lock_and_changes_only_the_observed_account(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    accepted = _accepted(ACCOUNT, URL)
    _accepted(OTHER_ACCOUNT, URL)
    _path, lock = slots._paths(ACCOUNT["fingerprint"])
    fingerprint = OTHER_ACCOUNT["fingerprint"]
    other_path, _ = slots._paths(fingerprint)
    other_before = slots._load(other_path, fingerprint)

    def poll(url):
        assert url == URL
        with cache_store.file_lock(lock, timeout=0, poll=0):
            return {"status": "COMPLETE", "alpha": "FinishedAlpha"}

    report = slots.reconcile_orphans(ACCOUNT, poll)

    assert [item["id"] for item in report["released"]] == [accepted["id"]]
    assert slots.snapshot(ACCOUNT, 2)["active"] == 0
    assert slots.snapshot(OTHER_ACCOUNT, 2)["active"] == 1
    assert slots._load(other_path, fingerprint) == other_before
