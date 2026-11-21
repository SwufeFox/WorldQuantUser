"""Offline contracts for sending and recovering native multi-simulation requests."""

import os
from unittest import mock

import pytest
import requests

import machine_lib as ml
import simulation_capabilities
import simulation_multi
import simulation_slots


ACCOUNT = {"state": "known", "fingerprint": "native-multi-offline-account"}
EXPRESSIONS = ["rank(close)", "rank(open)", "rank(volume)"]
SETTINGS = {"simulationMode": "QUICK", "testPeriod": "P1Y"}
PARENT_URL = ml.BRAIN_API_URL + "/simulations/NativeParent"


@pytest.fixture
def simulator(tmp_path):
    class Session:
        _brain_permissions = ["MULTI_SIMULATION"]

    session = Session()
    with mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": str(tmp_path)}), \
            mock.patch.object(simulation_capabilities, "account_provenance", return_value=ACCOUNT):
        def run(expressions=EXPRESSIONS[:2], **kwargs):
            return ml.simulate_alphas(session, expressions, settings=SETTINGS, capability_snapshot={},
                                      multi_size=2, max_wait_seconds=1, **kwargs)

        yield run


def child(number, *, status="COMPLETE", parent="NativeParent", expression=None):
    body = ml._single_simulation_data(
        expression or EXPRESSIONS[number - 1], 6, ml.resolve_settings(SETTINGS))
    body["settings"]["startDate"] = "2019-01-01"
    return {"id": "Child%d" % number, "parent": parent, "status": status,
            "alpha": "Alpha%d" % number, **body}


def test_native_preview_keeps_child_order_and_singleton_tail(simulator):
    scheduling = {}
    records = simulator(EXPRESSIONS, dry_run=True, scheduling_report=scheduling)
    first, tail = scheduling["multi_requests"]
    assert first == {"indices": [0, 1], "payload": [row["payload"] for row in records[:2]]}
    assert tail == {"indices": [2], "payload": records[2]["payload"]}
    assert all(row["payload"]["settings"]["simulationMode"] == "QUICK" for row in records)
    duplicate = {}
    simulator([EXPRESSIONS[0], EXPRESSIONS[0], EXPRESSIONS[1]], dry_run=True, scheduling_report=duplicate)
    assert [request["indices"] for request in duplicate["multi_requests"]] == [[0], [1, 2]]


def test_native_matches_child_echo_without_assuming_order_and_preserves_failure(simulator):
    events = []
    replies = {
        "NativeParent": {"id": "NativeParent", "status": "ERROR", "children": ["Child2", "Child1"]},
        "Child1": child(1),
        "Child2": child(2, status="ERROR"),
    }

    def inspect(session, identity, **kwargs):
        return replies[identity], None

    with mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL) as post, \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=inspect), \
            mock.patch.object(ml, "fetch_alpha_metrics", return_value={
                "is": {"fitness": 1}, "train": {"fitness": 1}}) as metrics:
        records = simulator(on_event=lambda event, payload: events.append((event, payload)))

    assert [row["status"] for row in records] == ["COMPLETE", "ERROR"]
    assert [row["alpha_id"] for row in records] == ["Alpha1", None]
    assert [row["multi_child_id"] for row in records] == ["Child1", "Child2"]
    assert records[0]["out_of_sample"]["train"]["fitness"] == 1
    metrics.assert_called_once()
    assert len(post.call_args.args[1]) == 2
    first, second = events[:2]
    assert first[0] == "multi_submitting"
    assert [row["status"] for row in first[1]["records"]] == ["UNKNOWN_SUBMISSION"] * 2
    assert second[0] == "multi_submitted"
    assert [row["multi_parent_url"] for row in second[1]["records"]] == [PARENT_URL] * 2
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 0


def test_native_matches_child_echo_that_omits_request_only_settings(simulator):
    """BRAIN echoes executed settings: testPeriod and simulationMode do not come back."""
    replies = {"NativeParent": {"id": "NativeParent", "status": "COMPLETE",
                                "children": ["Child1", "Child2"]}}
    for number in (1, 2):
        body = child(number)
        body["settings"].pop("testPeriod", None)
        body["settings"].pop("simulationMode", None)
        replies["Child%d" % number] = body

    with mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL), \
            mock.patch.object(ml, "read_simulation_snapshot",
                              side_effect=lambda s, identity, **kw: (replies[identity], None)), \
            mock.patch.object(ml, "fetch_alpha_metrics",
                              return_value={"is": {"fitness": 1}, "train": {"fitness": 1}}):
        records = simulator()

    assert [row["alpha_id"] for row in records] == ["Alpha1", "Alpha2"]
    assert [row["multi_child_id"] for row in records] == ["Child1", "Child2"]
    assert not [row for row in records if row.get("error")]


@pytest.mark.parametrize("status", ["COMPLETE", "ERROR"])
def test_native_ambiguous_echo_does_not_attribute_an_alpha(simulator, status):
    replies = {
        "NativeParent": {"id": "NativeParent", "status": status, "children": ["Child1", "Child2"]},
        "Child1": child(1, status=status),
        "Child2": child(2, status=status, expression=EXPRESSIONS[0]),
    }
    if status == "ERROR":
        for child_id in ("Child1", "Child2"):
            replies[child_id].pop("alpha")
    with mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL), \
            mock.patch.object(ml, "read_simulation_snapshot",
                              side_effect=lambda s, identity, **kw: (replies[identity], None)), \
            mock.patch.object(ml, "fetch_alpha_metrics") as metrics:
        records = simulator()
    assert all(row["status"] == "PENDING" and not row["alpha_id"] for row in records)
    assert "ambiguous" in records[0]["error"]
    metrics.assert_not_called()


def anonymous_failure(child_id, message="resource limit"):
    return {"id": child_id, "parent": "NativeParent", "status": "ERROR", "message": message}


def test_native_complete_anonymous_failures_finish_without_allocating_child_ids(simulator):
    parent = {"status": "ERROR", "children": ["Child2", "Child1"]}
    with mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL) as post, \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=[
                (parent, None), (anonymous_failure("Child2", "second error"), None),
                (anonymous_failure("Child1", "first error"), None),
            ]):
        records = simulator(fetch_metrics=False)
    post.assert_called_once()
    assert [row["status"] for row in records] == ["ERROR", "ERROR"]
    assert all(row.get("multi_child_id") is None and row.get("alpha_id") is None for row in records)
    assert len({row["error"] for row in records}) == 1
    assert all("all remaining multi-simulation children" in row["error"] for row in records)
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 0

    with mock.patch.object(ml, "submit_multi_simulation") as post, \
            mock.patch.object(ml, "read_simulation_snapshot") as read:
        resumed = simulator(initial_records=records, fetch_metrics=False, recover_only=True)
    post.assert_not_called()
    read.assert_not_called()
    assert [row["status"] for row in resumed] == ["ERROR", "ERROR"]


@pytest.mark.parametrize("second_read", [
    ml.NotFound("simulation not found"),
    (anonymous_failure("Child2"), 120),
    ({"status": "COMPLETE"}, None),
    ({**anonymous_failure("Child2"), "alpha": "AnAlpha"}, None),
    (anonymous_failure("UnexpectedChild"), None),
], ids=["missing", "retry", "success-without-echo", "has-alpha", "wrong-identity"])
def test_native_anonymous_failure_keeps_unproven_siblings_recoverable(simulator, second_read):
    parent = {"status": "ERROR", "children": ["Child1", "Child2"]}
    with mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL), \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=[
                (parent, None), (anonymous_failure("Child1"), None), second_read,
            ]):
        records = simulator(fetch_metrics=False)
    assert [row["status"] for row in records] == ["PENDING", "PENDING"]
    assert all(row.get("multi_child_id") is None and row.get("alpha_id") is None for row in records)


def test_native_partial_child_lists_do_not_prove_complete_anonymous_failure(simulator):
    first_parent = {"status": "ERROR", "children": ["Child1"]}
    with mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL), \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=[
                (first_parent, None), (anonymous_failure("Child1"), None),
            ]):
        records = simulator(fetch_metrics=False)
    assert [row["status"] for row in records] == ["PENDING", "PENDING"]

    second_parent = {"status": "ERROR", "children": ["Child2"]}
    with mock.patch.object(ml, "submit_multi_simulation") as post, \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=[
                (second_parent, None), (anonymous_failure("Child1"), None),
                (anonymous_failure("Child2"), None),
            ]):
        records = simulator(initial_records=records, fetch_metrics=False, recover_only=True)
    post.assert_not_called()
    assert records[0]["multi_children"] == ["Child1", "Child2"]
    assert [row["status"] for row in records] == ["PENDING", "PENDING"]


@pytest.mark.parametrize("has_resolved_identity", [True, False])
def test_native_anonymous_failure_requires_an_identity_for_each_resolved_child(simulator, has_resolved_identity):
    partial = {"status": "ERROR", "children": ["Child1"]}
    with mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL), \
            mock.patch.object(ml, "read_simulation_snapshot",
                              side_effect=[(partial, None), (child(1), None)]):
        first = simulator(fetch_metrics=False)
    assert [row["alpha_id"] for row in first] == ["Alpha1", None]
    assert first[1]["status"] == "PENDING"

    complete = {"status": "ERROR", "children": ["Child2", "Child1"]}
    reads = [(complete, None)]
    if not has_resolved_identity:
        first[0].pop("multi_child_id")
        reads.append((anonymous_failure("Child1"), None))
    reads.append((anonymous_failure("Child2"), None))
    with mock.patch.object(ml, "submit_multi_simulation") as post, \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=reads):
        resumed = simulator(initial_records=first, fetch_metrics=False, recover_only=True)
    post.assert_not_called()
    assert resumed[0]["alpha_id"] == "Alpha1"
    assert resumed[1]["status"] == ("ERROR" if has_resolved_identity else "PENDING")
    assert resumed[1].get("multi_child_id") is None


def test_native_resume_restores_parent_location_before_releasing_reservation(simulator):
    durable = []

    def interrupted(event, payload):
        if event == "multi_submitting":
            durable.extend(payload["records"])
        if event == "multi_submitted":
            raise KeyboardInterrupt

    with mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL) as post:
        with pytest.raises(KeyboardInterrupt):
            simulator(on_event=interrupted)
    post.assert_called_once()
    assert not any(row.get("multi_parent_url") for row in durable)
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 1
    simulation_slots.defer_submission(ACCOUNT, 60)
    events = []

    def observe(event, payload):
        if event in {"multi_submitted", "multi_pending"}:
            events.append((event, simulation_slots.snapshot(ACCOUNT, 3)["active"], payload["records"]))

    parent_complete = {"status": "COMPLETE", "children": ["Child1", "Child2"]}
    with mock.patch.object(ml, "submit_multi_simulation") as resumed_post, \
            mock.patch.object(ml, "read_simulation_snapshot",
                              side_effect=[(parent_complete, None), (child(1), None), (child(2), None)]):
        records = simulator(initial_records=durable, recover_only=True, fetch_metrics=False, on_event=observe)
    resumed_post.assert_not_called()
    assert events[0][0:2] == ("multi_submitted", 1)
    assert events[0][2][0]["multi_parent_url"] == PARENT_URL
    assert events[1][0:2] == ("multi_pending", 1)
    assert events[1][2][0]["multi_parent_terminal"]
    assert [row["alpha_id"] for row in records] == ["Alpha1", "Alpha2"]
    assert all(row["metrics_pending"] for row in records)
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 0


def test_native_lost_post_response_does_not_replay_members(simulator):
    with mock.patch.object(ml, "submit_multi_simulation", side_effect=requests.Timeout()):
        records = simulator()
    assert all(row["status"] == "UNKNOWN_SUBMISSION" and row["submission_attempted"] for row in records)
    assert records[0]["multi_request_id"] == records[1]["multi_request_id"]
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 1
    with mock.patch.object(ml, "submit_multi_simulation") as post, \
            mock.patch.object(ml, "submit_single_simulation") as single:
        resumed = simulator(initial_records=records)
    post.assert_not_called()
    single.assert_not_called()
    assert all(row["status"] == "UNKNOWN_SUBMISSION" for row in resumed)


def test_native_partial_parent_receipt_can_recover_a_later_child(simulator):
    partial = {"status": "ERROR", "children": ["Child1"]}
    complete = {"status": "ERROR", "children": ["Child2", "Child1"]}
    with mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL), \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=[(partial, None), (child(1), None)]):
        first = simulator(fetch_metrics=False)
    assert [row["alpha_id"] for row in first] == ["Alpha1", None]
    with mock.patch.object(ml, "submit_multi_simulation") as post, \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=[(complete, None), (child(2), None)]):
        second = simulator(initial_records=first, fetch_metrics=False, recover_only=True)
    post.assert_not_called()
    assert [row["alpha_id"] for row in second] == ["Alpha1", "Alpha2"]


def test_native_parent_and_child_retry_after_survive_resume(simulator):
    incomplete = {"id": "NativeParent", "status": "SIMULATING"}
    with mock.patch.object(simulation_multi.time, "time", return_value=100), \
            mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL), \
            mock.patch.object(ml, "read_simulation_snapshot", return_value=(incomplete, 120)):
        records = simulator(fetch_metrics=False)
    assert records[0]["multi_next_parent_poll_at"] == 220

    with mock.patch.object(simulation_multi.time, "time", return_value=110), \
            mock.patch.object(ml, "read_simulation_snapshot") as read:
        records = simulator(initial_records=records, fetch_metrics=False, recover_only=True)
    read.assert_not_called()

    parent = {"status": "COMPLETE", "children": ["Child1", "Child2"]}
    with mock.patch.object(simulation_multi.time, "time", return_value=221), \
            mock.patch.object(ml, "read_simulation_snapshot",
                              side_effect=[(parent, None), (child(1), 30), (child(2), None)]):
        records = simulator(initial_records=records, fetch_metrics=False, recover_only=True)
    assert records[0]["multi_child_retry"] == {"Child1": 251}
    assert [row["alpha_id"] for row in records] == [None, "Alpha2"]

    with mock.patch.object(simulation_multi.time, "time", return_value=230), \
            mock.patch.object(ml, "read_simulation_snapshot") as read:
        records = simulator(initial_records=records, fetch_metrics=False, recover_only=True)
    read.assert_not_called()
    assert "pending" in records[0]["error"]

    with mock.patch.object(simulation_multi.time, "time", return_value=252), \
            mock.patch.object(ml, "read_simulation_snapshot", return_value=(child(1), None)) as read:
        records = simulator(initial_records=records, fetch_metrics=False, recover_only=True)
    assert read.call_args.args[1] == "Child1"
    assert [row["alpha_id"] for row in records] == ["Alpha1", "Alpha2"]


def test_native_retry_after_and_redirect_avoid_array_replay():
    session = mock.Mock()
    permissions = ["MULTI_SIMULATION"]
    session._brain_permissions = permissions
    array = [ml._single_simulation_data(expression, 6, ml.DEFAULT_SETTINGS) for expression in EXPRESSIONS[:2]]
    throttled = requests.Response()
    throttled.status_code = 429
    throttled.headers["Retry-After"] = "1"
    created = requests.Response()
    created.status_code = 201
    created.headers["Location"] = "/simulations/NativeParent"
    events = []
    with mock.patch.object(ml, "sleep") as sleep, \
            mock.patch.object(simulation_slots, "quota_snapshot", return_value={"blocked": False}), \
            mock.patch.object(simulation_capabilities, "account_provenance", return_value={}):
        session.post.side_effect = [throttled, created]
        assert ml.submit_multi_simulation(session, array, on_event=lambda k, p: events.append((k, p))) == PARENT_URL
        sleep.assert_called_once_with(1)
        assert events == [("rate_limited", {"attempt": 1, "wait_seconds": 1, "child_count": 2})]
        assert session.post.call_args.kwargs["json"] == array
        assert session.post.call_args.kwargs["allow_redirects"] is False

        for code, location in [(307, "/simulations/NativeParent"),
                               (201, "/simulations/NativeParent?query=invalid")]:
            forged = requests.Response()
            forged.status_code = code
            forged.headers["Location"] = location
            session.post.reset_mock()
            session.post.side_effect = [forged]
            with pytest.raises(ml.SimulationError):
                ml.submit_multi_simulation(session, array)
            session.post.assert_called_once()


def test_native_transient_429_defers_new_requests_from_another_invocation(simulator, monkeypatch):
    monkeypatch.setattr(simulation_slots.time, "time", lambda: 100)
    session = mock.Mock()
    session._brain_permissions = ["MULTI_SIMULATION"]
    throttled = requests.Response()
    throttled.status_code = 429
    throttled.headers["Retry-After"] = "120"
    session.post.return_value = throttled
    payloads = [ml._single_simulation_data(expression, 6, ml.DEFAULT_SETTINGS) for expression in EXPRESSIONS[:2]]

    with pytest.raises(ml.RateLimited, match="after 1 attempts"):
        ml.submit_multi_simulation(session, payloads, max_retries=1)

    assert simulation_slots.submission_retry_at(ACCOUNT) == 220
    scheduling = {}
    with mock.patch.object(ml, "submit_multi_simulation") as post:
        rows = simulator(scheduling_report=scheduling)
    post.assert_not_called()
    assert all(row["status"] == "QUEUED" and not row.get("submission_attempted") for row in rows)
    assert scheduling["submission_retry_at"] == 220
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 0
