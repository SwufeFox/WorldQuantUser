"""Offline regressions for Quick retry state and authorization boundaries."""

import json
from unittest.mock import Mock

import pytest
import requests

import machine_lib
import simulation_slots
from tools.pipeline import batch_quick


ACCOUNT = {"fingerprint": "offline-quick-recovery"}
ARM = {"label": "candidate", "expr": "rank(close)",
       "settings": {"region": "USA", "universe": "TOP3000", "delay": 1}}
PROGRESS = machine_lib.BRAIN_API_URL + "/simulations/op-test"
ALPHA = machine_lib.BRAIN_API_URL + "/alphas/AlphaTest"


def response(status, payload=None, **headers):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(payload or {}).encode()
    result.headers.update(headers)
    return result


def finished(url, **kwargs):
    if url == PROGRESS:
        return response(200, {"status": "COMPLETE", "alpha": "AlphaTest"})
    assert url == ALPHA
    return response(200, {"is": {"sharpe": 2.5, "checks": []}})


@pytest.fixture
def offline(tmp_path, monkeypatch):
    now = 0.0

    def sleep(seconds):
        nonlocal now
        now += seconds

    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    monkeypatch.setattr(batch_quick.simulation_capabilities, "account_provenance", lambda session: ACCOUNT)
    monkeypatch.setattr(batch_quick.Runner, "_session_generation", lambda self: "cached")
    monkeypatch.setattr(batch_quick.time, "monotonic", lambda: now)
    monkeypatch.setattr(batch_quick.time, "time", lambda: 1000 + now)
    monkeypatch.setattr(batch_quick.time, "sleep", sleep)

    def runner(client=None, *, timeout=2, name="run"):
        if client is not None:
            monkeypatch.setattr(machine_lib, "load_session", lambda refresh=False: client)
        return batch_quick.Runner(tmp_path / f"{name}.jsonl", ACCOUNT, 1, timeout)

    return runner


@pytest.mark.parametrize("code", [401, 403])
def test_orphan_auth_refresh_can_free_the_only_slot(offline, monkeypatch, code):
    old_url = machine_lib.BRAIN_API_URL + "/simulations/orphan"
    old = simulation_slots.quick_acquire(ACCOUNT, 1, "old-job", existing_url=old_url)
    simulation_slots.quick_update(ACCOUNT, old["id"], "parked")
    stale, fresh = Mock(), Mock()
    stale.get.return_value = response(code)
    fresh.post.return_value = response(201, Location=PROGRESS)

    def read(url, **kwargs):
        if url == old_url:
            return response(200, {"status": "COMPLETE", "alpha": "OldAlpha", "id": "orphan"})
        return finished(url, **kwargs)

    fresh.get.side_effect = read
    login = Mock(side_effect=[stale, fresh])
    monkeypatch.setattr(machine_lib, "load_session", login)

    runner = offline()
    assert runner.run(ARM)["status"] == "COMPLETE"
    assert [call.kwargs["refresh"] for call in login.call_args_list] == [False, True]
    stale.post.assert_not_called()
    fresh.post.assert_called_once()


def test_accepted_location_persisted_before_quota_observation(offline, monkeypatch):
    client = Mock()
    client.post.return_value = response(201, Location=PROGRESS)
    client.get.side_effect = finished
    runner = offline(client)
    job_key = runner._job_key(ARM["label"], batch_quick._payload(ARM))

    def inspect(account, headers):
        slot = simulation_slots.quick_acquire(account, 1, job_key)
        assert slot["progress_url"] == PROGRESS
        return None

    monkeypatch.setattr(simulation_slots, "observe_quota", inspect)
    assert runner.run(ARM)["status"] == "COMPLETE"


@pytest.mark.parametrize("first_code", [201, 429])
def test_zero_quota_without_reset_stops_other_arms_in_same_run(offline, first_code):
    client = Mock()
    first = (response(201, Location=PROGRESS, **{"X-Ratelimit-Remaining": "0"})
             if first_code == 201 else response(429, {"detail": "DAILY_SIMULATION_LIMIT_EXCEEDED"}))
    client.post.return_value = first
    client.get.side_effect = finished
    runner = offline(client)

    assert runner.run(ARM)["status"] == ("COMPLETE" if first_code == 201 else "QUEUED")
    outcome = runner.run({**ARM, "label": "later", "expr": "rank(volume)"})
    assert outcome["status"] == "QUEUED"
    assert outcome["detail"] == "daily simulation quota exhausted"
    client.post.assert_called_once()


def test_invalid_json_releases_the_definitely_unposted_slot(offline):
    class NeverSend(requests.Session):
        def prepare_request(self, request):
            raise requests.exceptions.InvalidJSONError("local serialization failed")

        def send(self, *args, **kwargs):
            raise AssertionError("invalid payload must not make an HTTP request")

    assert offline(NeverSend()).run(ARM)["status"] == "POST_FAIL"
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 0


def test_post_retry_after_survives_runner_restart(offline):
    client = Mock()
    client.post.return_value = response(429, **{"Retry-After": "60"})

    assert offline(client).run(ARM)["status"] == "QUEUED"
    result = offline(client).run(ARM)
    assert result["status"] == "QUEUED"
    assert result["next_retry_at"] == pytest.approx(1060)
    other = offline(client).run({**ARM, "label": "other", "expr": "rank(volume)"})
    assert other["status"] == "QUEUED"
    assert other["next_retry_at"] == pytest.approx(1060)
    client.post.assert_called_once()


def test_poll_retry_after_survives_restart_and_suppresses_orphan_poll(offline):
    client = Mock()
    client.post.return_value = response(201, Location=PROGRESS)
    client.get.return_value = response(200, **{"Retry-After": "60"})

    assert offline(client).run(ARM)["status"] == "PENDING"
    assert not simulation_slots.quick_orphans(ACCOUNT)
    assert offline(client).run(ARM)["status"] == "PENDING"
    assert offline(client, name="other").run({**ARM, "label": "other"})["status"] == "QUEUED"
    client.post.assert_called_once()
    client.get.assert_called_once()


def test_metrics_retry_after_survives_runner_restart(offline):
    client = Mock()
    client.post.return_value = response(201, Location=PROGRESS)
    client.get.side_effect = [finished(PROGRESS), response(200, **{"Retry-After": "60"})]

    first = offline(client).run(ARM)
    assert first["status"] == "PENDING"
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 0
    resumed = offline(client).run(ARM)
    assert resumed["status"] == "PENDING"
    assert resumed["next_metrics_at"] == pytest.approx(first["next_metrics_at"])
    client.post.assert_called_once()
    assert client.get.call_count == 2


def test_interrupt_during_session_setup_does_not_post_or_pin_slot(offline, monkeypatch):
    client = Mock()
    runner = offline()

    def interrupt(refresh=False):
        runner.stop_requested.set()
        return client

    monkeypatch.setattr(machine_lib, "load_session", interrupt)
    assert runner.run(ARM)["status"] == "QUEUED"
    client.post.assert_not_called()
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 0


@pytest.mark.parametrize("matches", [True, False])
def test_attached_location_retains_original_slot_until_alpha_matches(offline, matches):
    client = Mock()
    runner = offline(client)
    payload = batch_quick._payload(ARM)
    key = runner._job_key(ARM["label"], payload)
    slot = simulation_slots.quick_acquire(ACCOUNT, 1, key)
    simulation_slots.quick_update(ACCOUNT, slot["id"], "unknown")
    simulation_slots.quick_attach_location(ACCOUNT, key, PROGRESS)
    metadata = {**payload, "id": "AlphaTest", "is": {"sharpe": 2.5, "checks": []},
                "settings": {**payload["settings"], "language": "FASTEXPR"}}
    if not matches:
        metadata["regular"] = "rank(volume)"
    client.get.side_effect = [finished(PROGRESS), response(200, metadata)]

    result = runner.run(ARM)
    assert result["status"] == ("COMPLETE" if matches else "UNVERIFIED_RESULT")
    assert result["alpha_id"] == ("AlphaTest" if matches else None)
    assert bool(result.get("identity_verified")) is matches
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == (0 if matches else 1)
    client.post.assert_not_called()
