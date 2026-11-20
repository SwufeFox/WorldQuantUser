"""Offline regressions for Quick retry state and authorization boundaries."""

import json
from unittest.mock import MagicMock, Mock

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
    monkeypatch.setattr(batch_quick.time, "monotonic", lambda: now)
    monkeypatch.setattr(batch_quick.time, "time", lambda: 1000 + now)
    monkeypatch.setattr(batch_quick.time, "sleep", sleep)

    def runner(client=None, *, timeout=2, name="run"):
        if client is not None:
            monkeypatch.setattr(machine_lib, "load_session", lambda **kwargs: client)
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


@pytest.mark.parametrize("location", [PROGRESS, PROGRESS + "/?token=example"])
def test_accepted_location_persisted_before_quota_observation(offline, monkeypatch, location):
    client = Mock()
    client.post.return_value = response(201, Location=location)
    client.get.side_effect = lambda url, **kw: finished(PROGRESS if url == location else url, **kw)
    runner = offline(client)
    job_key = runner._job_key(ARM["label"], batch_quick._payload(ARM))

    def inspect(account, headers):
        slot = simulation_slots.quick_acquire(account, 1, job_key)
        assert slot["progress_url"] == location
        return None

    monkeypatch.setattr(simulation_slots, "observe_quota", inspect)
    assert runner.run(ARM)["status"] == "COMPLETE"
    assert client.get.call_args_list[0].args == (location,)


@pytest.mark.parametrize("location", ["/simulations/", "/simulations/nested/op-test"])
def test_accepted_post_with_invalid_location_stays_unknown_without_reposting(offline, location):
    client = Mock()
    client.post.return_value = response(201, Location=location)

    assert offline(client).run(ARM)["status"] == "UNKNOWN_SUBMISSION"
    assert offline(client).run(ARM)["status"] == "UNKNOWN_SUBMISSION"
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 1
    client.post.assert_called_once()
    client.get.assert_not_called()


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


def test_short_retry_after_takes_precedence_over_progress_interval(offline):
    client = Mock()
    client.post.return_value = response(201, Location=PROGRESS)
    client.get.side_effect = [response(200, {"progress": 0.9}, **{"Retry-After": "0.25"}),
                              finished(PROGRESS), finished(ALPHA)]

    assert offline(client, timeout=1).run(ARM)["status"] == "COMPLETE"
    assert batch_quick.time.monotonic() == pytest.approx(0.75)


def test_resumed_progress_uses_remaining_retry_after(offline):
    client = Mock()
    client.post.return_value = response(201, Location=PROGRESS)
    client.get.side_effect = [response(200, **{"Retry-After": "0.75"}), finished(PROGRESS), finished(ALPHA)]

    first = offline(client, timeout=1).run(ARM)
    assert first["status"] == "PENDING"
    assert first["next_poll_at"] == pytest.approx(1001.25)

    batch_quick.time.sleep(0.5)
    assert offline(client, timeout=0.5).run(ARM)["status"] == "COMPLETE"
    assert batch_quick.time.monotonic() == pytest.approx(1.25)
    client.post.assert_called_once()


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

    def interrupt(**kwargs):
        runner.stop_requested.set()
        return client

    monkeypatch.setattr(machine_lib, "load_session", interrupt)
    assert runner.run(ARM)["status"] == "QUEUED"
    client.post.assert_not_called()
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 0


@pytest.mark.parametrize("already_observed", [True, False])
def test_confirmed_original_failure_releases_slot_and_survives_restart(offline, tmp_path, monkeypatch,
                                                                    already_observed):
    client = MagicMock()
    client.__enter__.return_value = client
    runner = offline(client)
    key = runner._job_key(ARM["label"], batch_quick._payload(ARM))
    slot = simulation_slots.quick_acquire(ACCOUNT, 1, key)
    simulation_slots.quick_update(ACCOUNT, slot["id"], "unknown")
    location = PROGRESS + "/?token=example"
    client.get.return_value = response(400, {"id": "op-test", "status": "FAIL"})

    if already_observed:
        machine_lib.attach_quick_location(client, ACCOUNT, key, location)
        assert runner.run(ARM)["status"] == "UNVERIFIED_RESULT"
        assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 1

    arms = tmp_path / "arms.json"
    arms.write_text(json.dumps([ARM]), encoding="utf-8")
    monkeypatch.setattr(batch_quick.sys, "argv", ["batch_quick.py", str(arms), "--out", str(runner.out),
                                                "--recover-job", key, "--recover-location", location,
                                                "--confirm-original-failure"])
    assert batch_quick.main() == 0
    client.get.assert_called_with(location, timeout=60, allow_redirects=False)
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 0
    client.get.reset_mock()

    for _ in range(2):
        result = offline(client).run(ARM)
        assert result["status"] == "FAIL"
        assert result["identity_verified"] and result["alpha_id"] is None
    client.get.assert_not_called()
    client.post.assert_not_called()


@pytest.mark.parametrize("payload, headers", [
    ({"id": "op-test", "progress": 0.5}, {}),
    ({"status": "FAIL"}, {}),
    ({"id": "op-test", "status": "FAIL"}, {"Retry-After": "15"}),
    ({"id": "op-test", "status": "FAIL", "alpha": "AlphaTest"}, {}),
])
def test_manual_failure_requires_an_identified_terminal_failure(offline, payload, headers):
    client = Mock()
    runner = offline(client)
    key = runner._job_key(ARM["label"], batch_quick._payload(ARM))
    slot = simulation_slots.quick_acquire(ACCOUNT, 1, key)
    simulation_slots.quick_update(ACCOUNT, slot["id"], "unknown")
    client.get.return_value = response(200, payload, **headers)

    with pytest.raises(machine_lib.ServiceError, match="terminal failure"):
        machine_lib.attach_quick_location(client, ACCOUNT, key, PROGRESS, confirm_failure=True)
    assert simulation_slots.quick_acquire(ACCOUNT, 1, key)["state"] == "unknown"
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 1


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


@pytest.mark.parametrize("login_delay", [0.75, 2.0])
def test_read_budget_includes_login_before_dispatch(offline, monkeypatch, login_delay):
    client = Mock()
    client.get.return_value = finished(ALPHA)

    def login(*, deadline, **kwargs):
        assert deadline == 2
        batch_quick.time.sleep(login_delay)
        return client

    monkeypatch.setattr(machine_lib, "load_session", login)
    outcome = offline()._read(ALPHA, 2)
    if login_delay < 2:
        assert outcome["is"]["sharpe"] == 2.5
        client.get.assert_called_once_with(ALPHA, timeout=1.25, allow_redirects=False)
    else:
        assert outcome is None
        client.get.assert_not_called()


def test_quick_retries_a_wrapped_auth_transport_timeout(offline, monkeypatch):
    client, auth = Mock(), Mock()
    client.get.return_value = finished(ALPHA)
    auth.post.side_effect = requests.exceptions.Timeout()
    monkeypatch.setattr(machine_lib, "get_credentials", lambda: ("offline", "test"))
    monkeypatch.setattr(requests, "Session", lambda: auth)

    def load(*, deadline, **kwargs):
        if not auth.post.called:
            return machine_lib.authenticate(deadline=deadline)[0]
        return client

    monkeypatch.setattr(machine_lib, "load_session", load)
    assert offline()._read(ALPHA, 2)["is"]["sharpe"] == 2.5
    auth.post.assert_called_once()
    client.get.assert_called_once()


def test_quick_refresh_rejects_the_generation_returned_by_login(offline, monkeypatch):
    stale, fresh = Mock(), Mock()
    stale.get.return_value = response(401)
    fresh.get.return_value = finished(ALPHA)

    def login(*, refresh, meta, deadline, rejected_generation):
        assert deadline == 2
        if refresh:
            assert rejected_generation == "first-published"
            return fresh
        assert rejected_generation is None
        meta["generation"] = "first-published"
        return stale

    monkeypatch.setattr(machine_lib, "load_session", login)
    assert offline()._read(ALPHA, 2)["is"]["sharpe"] == 2.5
    stale.get.assert_called_once()
    fresh.get.assert_called_once()


def test_dispatched_post_keeps_request_timeout_when_phase_almost_over(offline, monkeypatch):
    client = Mock()

    def login(*, deadline, **kwargs):
        assert deadline == 2
        batch_quick.time.sleep(1.99)
        return client

    def post(url, *, timeout, **kwargs):
        assert timeout == 90
        batch_quick.time.sleep(0.5)
        return response(201, Location=PROGRESS)

    client.post.side_effect = post
    client.get.side_effect = finished
    monkeypatch.setattr(machine_lib, "load_session", login)
    assert offline().run(ARM)["status"] == "COMPLETE"
    client.post.assert_called_once()
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 0


@pytest.mark.parametrize("failure", [machine_lib.AuthenticationDeadline, machine_lib.RateLimited])
def test_auth_budget_or_rate_limit_defers_without_pinning_slot(offline, monkeypatch, failure):
    login = Mock(side_effect=failure("offline auth deferred"))
    monkeypatch.setattr(machine_lib, "load_session", login)
    assert offline().run(ARM)["status"] == "QUEUED"
    login.assert_called_once()
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 0
