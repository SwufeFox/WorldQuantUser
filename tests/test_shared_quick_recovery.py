"""Offline interoperability between Quick leases and the native simulator."""

import json
from unittest.mock import Mock

import pytest
import requests

import cache_store
import machine_lib as ml
import simulation_capabilities
import simulation_slots as slots


ACCOUNT = {"fingerprint": "quick-interop-account"}
ORPHAN_URL = f"{ml.BRAIN_API_URL}/simulations/Orphan"


def response(status, payload=None, **headers):
    reply = requests.Response()
    reply.status_code = status
    reply._content = json.dumps(payload or {}).encode()
    reply.headers.update(headers)
    return reply


def parked_orphan():
    slot = slots.quick_acquire(ACCOUNT, 1, "old-quick-job", existing_url=ORPHAN_URL)
    assert slots.quick_update(ACCOUNT, slot["id"], "parked")
    return slot


@pytest.mark.parametrize("native_multi", [False, True])
def test_native_can_launch_after_reclaiming_a_completed_quick_slot(tmp_path, monkeypatch, native_multi):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    monkeypatch.setattr(simulation_capabilities, "account_provenance", lambda _session=None: ACCOUNT)
    orphan = parked_orphan()
    session = Mock(spec=requests.Session)
    session._brain_permissions = ["MULTI_SIMULATION"]
    sent = []

    def post(_url, *, json, **_options):
        assert session.get.call_args_list[0].args == (ORPHAN_URL,)
        sent.extend(json if isinstance(json, list) else [json])
        return response(201, Location="/simulations/Native")

    def get(url, **_options):
        if url == ORPHAN_URL:
            return response(200, {"id": "Orphan", "status": "COMPLETE", "alpha": "OldAlpha"})
        sim_id = url.rsplit("/", 1)[-1]
        if sim_id == "Native":
            if native_multi:
                return response(200, {"id": sim_id, "status": "COMPLETE", "children": ["Child0", "Child1"]})
            return response(200, {"id": sim_id, "status": "COMPLETE", "alpha": "NativeAlpha"})
        index = int(sim_id[-1])
        return response(200, {**sent[index], "id": sim_id, "parent": "Native",
                             "status": "COMPLETE", "alpha": f"NativeAlpha{index}"})

    session.post.side_effect = post
    session.get.side_effect = get
    expressions = ["rank(close)", "rank(volume)"] if native_multi else ["rank(close)"]
    results = ml.simulate_alphas(session, expressions, concurrency=1,
                                 multi_size=2 if native_multi else None,
                                 fetch_metrics=False, max_wait_seconds=5)
    assert [row["regular"] for row in sent] == expressions
    assert [row["alpha_id"] for row in results] == (
        ["NativeAlpha0", "NativeAlpha1"] if native_multi else ["NativeAlpha"])
    assert slots.snapshot(ACCOUNT, 1)["active"] == 0
    resumed = slots.quick_acquire(ACCOUNT, 1, "old-quick-job")
    assert resumed["id"] == orphan["id"]
    assert resumed["resolved"]
    assert resumed["terminal_result"] == {"status": "COMPLETE", "alpha_id": "OldAlpha"}


def test_slow_first_orphan_does_not_block_native_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    monkeypatch.setattr(simulation_capabilities, "account_provenance", lambda _session=None: ACCOUNT)
    now = 100.0
    monkeypatch.setattr(ml.time, "monotonic", lambda: now)
    slow_url = f"{ml.BRAIN_API_URL}/simulations/Slow"
    for key, url in (("slow", slow_url), ("ready", ORPHAN_URL)):
        slot = slots.quick_acquire(ACCOUNT, 2, key, existing_url=url)
        assert slots.quick_update(ACCOUNT, slot["id"], "parked")

    session = Mock(spec=requests.Session)
    session.post.return_value = response(201, Location="/simulations/Native")

    def get(url, *, timeout, **_options):
        nonlocal now
        if url == slow_url:
            now += timeout
            raise requests.ReadTimeout()
        if url == ORPHAN_URL:
            return response(200, {"id": "Orphan", "status": "COMPLETE", "alpha": "OldAlpha"})
        return response(200, {"id": "Native", "status": "COMPLETE", "alpha": "NativeAlpha"})

    session.get.side_effect = get
    result, = ml.simulate_alphas(session, ["rank(close)"], concurrency=2, fetch_metrics=False,
                                 max_wait_seconds=30)

    assert result["alpha_id"] == "NativeAlpha"
    session.post.assert_called_once()
    assert [call.args[0] for call in session.get.call_args_list[:2]] == [slow_url, ORPHAN_URL]
    assert slots.snapshot(ACCOUNT, 2)["active"] == 1


def test_optional_recovery_does_not_wait_past_deadline_for_journal_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    parked_orphan()
    now = 100.0
    monkeypatch.setattr(ml.time, "monotonic", lambda: now)
    session = Mock()

    def get(*_args, **_options):
        nonlocal now
        now = 101.0
        return response(200, {"id": "Orphan", "status": "COMPLETE", "alpha": "OldAlpha"})

    session.get.side_effect = get
    _, lock_path = slots._paths(ACCOUNT["fingerprint"])
    with cache_store.file_lock(lock_path):
        assert ml.recover_quick_orphans(session, ACCOUNT, deadline=101.0) == 0
    session.get.assert_called_once()
    assert slots.snapshot(ACCOUNT, 1)["active"] == 1
    assert ml.recover_quick_orphans(session, ACCOUNT) == 1


def test_orphan_recovery_leaves_job_resumed_by_owner_during_read(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    parked_orphan()
    session = Mock()

    def get(*_args, **_options):
        assert slots.quick_acquire(ACCOUNT, 1, "old-quick-job")["state"] == "accepted"
        return response(200, {"id": "Orphan", "status": "COMPLETE", "alpha": "OldAlpha"})

    session.get.side_effect = get
    assert ml.recover_quick_orphans(session, ACCOUNT) == 0
    assert slots.snapshot(ACCOUNT, 1)["active"] == 1


@pytest.mark.parametrize(("http_status", "body"), [
    (404, {"status": "FAIL"}),
    (400, {"id": "Different", "status": "FAIL"}),
    (400, {"status": "FAIL"}),
    (200, {"id": "Different", "status": "COMPLETE", "alpha": "OtherAlpha"}),
    (200, {"status": "COMPLETE"}),
    (200, {"progress": 0.5}),
])
def test_absent_or_unverified_receipt_keeps_occupancy(tmp_path, monkeypatch, http_status, body):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    parked_orphan()
    session = Mock()
    session.get.return_value = response(http_status, body)
    assert ml.recover_quick_orphans(session, ACCOUNT) == 0
    assert slots.snapshot(ACCOUNT, 1)["active"] == 1
    assert slots.quick_known_jobs(ACCOUNT) == {"old-quick-job"}


def test_verified_terminal_failure_frees_slot_and_retains_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    parked_orphan()
    session = Mock()
    session.get.return_value = response(400, {"id": "Orphan", "status": "FAIL"})
    assert ml.recover_quick_orphans(session, ACCOUNT) == 1
    assert slots.snapshot(ACCOUNT, 1)["active"] == 0
    assert slots.quick_acquire(ACCOUNT, 1, "old-quick-job")["terminal_result"] == {
        "status": "FAIL", "alpha_id": None}


@pytest.mark.parametrize("http_status", [401, 403])
def test_orphan_auth_expiry_reaches_caller_without_releasing(tmp_path, monkeypatch, http_status):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    parked_orphan()
    session = Mock()
    session.get.return_value = response(http_status)
    with pytest.raises(ml.AuthError):
        ml.recover_quick_orphans(session, ACCOUNT)
    assert slots.snapshot(ACCOUNT, 1)["active"] == 1


@pytest.mark.parametrize("http_status", [200, 429])
def test_retry_after_survives_restarts_for_all_callers(tmp_path, monkeypatch, http_status):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    now = 100.0
    monkeypatch.setattr(slots.time, "time", lambda: now)
    orphan = parked_orphan()
    session = Mock()
    session.get.return_value = response(http_status, {"status": "COMPLETE", "alpha": "OldAlpha"},
                                        **{"Retry-After": "120"})
    assert ml.recover_quick_orphans(session, ACCOUNT) == 0
    assert slots.snapshot(ACCOUNT, 1)["active"] == 1
    assert ml.recover_quick_orphans(session, ACCOUNT) == 0
    session.get.assert_called_once()
    restored = slots.quick_acquire(ACCOUNT, 1, "old-quick-job")
    assert restored["next_poll_at"] == 220
    assert slots.quick_defer(ACCOUNT, orphan["id"], 110)
    assert slots.quick_update(ACCOUNT, orphan["id"], "parked")
    now = 220.0
    session.get.return_value = response(200, {"status": "COMPLETE", "alpha": "OldAlpha"})
    assert ml.recover_quick_orphans(session, ACCOUNT) == 1
    assert session.get.call_count == 2


def test_manually_recovered_location_is_held_until_payload_is_verified(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    monkeypatch.setattr(simulation_capabilities, "account_provenance", lambda _: ACCOUNT)
    initial = slots.quick_acquire(ACCOUNT, 1, "manual-job")
    assert slots.quick_update(ACCOUNT, initial["id"], "unknown")
    session = Mock()
    session.get.return_value = response(200, {"id": "Orphan", "progress": 0.5},
                                        **{"Retry-After": "120"})
    restored = ml.attach_quick_location(session, ACCOUNT, "manual-job", "/simulations/Orphan")
    assert restored["id"] == initial["id"]
    assert restored["progress_url"] == ORPHAN_URL
    assert restored["identity_unverified"]
    assert restored["manual_location"] and restored["next_poll_at"] > 0
    assert slots.snapshot(ACCOUNT, 1)["active"] == 1
    assert slots.quick_acquire(ACCOUNT, 1, "manual-job")["identity_unverified"]
    assert slots.quick_update(ACCOUNT, restored["id"], "parked")
    monkeypatch.setattr(slots.time, "time", lambda: restored["next_poll_at"] + 1)
    assert ml.recover_quick_orphans(session, ACCOUNT) == 0
    session.get.assert_called_once()
    session.post.assert_not_called()


def test_legacy_import_can_free_slot_but_keeps_unverified_result_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    lease = slots.quick_acquire(ACCOUNT, 1, "legacy-job", existing_url=ORPHAN_URL,
                                identity_unverified=True)
    assert slots.quick_update(ACCOUNT, lease["id"], "parked")
    session = Mock()
    session.get.return_value = response(200, {"status": "COMPLETE", "alpha": "OldAlpha"})
    assert ml.recover_quick_orphans(session, ACCOUNT) == 1
    receipt = slots.quick_acquire(ACCOUNT, 1, "legacy-job")
    assert receipt["resolved"] and receipt["identity_unverified"]
    assert receipt["terminal_result"]["alpha_id"] == "OldAlpha"


def test_recovery_rejects_wrong_account_active_post_wrong_receipt_and_existing_address(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    initial = slots.quick_acquire(ACCOUNT, 2, "manual-job")
    account = {"fingerprint": "other"}
    monkeypatch.setattr(simulation_capabilities, "account_provenance", lambda _: account)
    session = Mock()
    with pytest.raises(ValueError, match="authenticated account"):
        ml.attach_quick_location(session, ACCOUNT, "manual-job", ORPHAN_URL)
    session.get.assert_not_called()
    account = ACCOUNT
    session.get.return_value = response(200, {"id": "Orphan", "progress": 0.5})
    with pytest.raises(ValueError, match="still be posting"):
        ml.attach_quick_location(session, ACCOUNT, "manual-job", ORPHAN_URL)

    assert slots.quick_update(ACCOUNT, initial["id"], "unknown")
    assert slots.quick_acquire(ACCOUNT, 2, "manual-job", existing_url=ORPHAN_URL,
                               identity_unverified=True)["state"] == "unknown"
    session.get.return_value = response(200, {"id": "Different", "progress": 0.5})
    with pytest.raises(ml.ServiceError, match="identity"):
        ml.attach_quick_location(session, ACCOUNT, "manual-job", ORPHAN_URL)

    parked_orphan()
    session.get.return_value = response(200, {"id": "Orphan", "progress": 0.5})
    with pytest.raises(ValueError, match="another simulation"):
        ml.attach_quick_location(session, ACCOUNT, "manual-job", ORPHAN_URL + "/?token=test")
    assert slots.snapshot(ACCOUNT, 2)["active"] == 2
    assert slots.quick_acquire(ACCOUNT, 1, "manual-job")["state"] == "unknown"
    session.post.assert_not_called()
