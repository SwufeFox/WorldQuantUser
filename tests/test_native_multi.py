"""Offline contracts for sending and recovering native multi-simulation requests."""

import os
from threading import Event
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
        def run(expressions=EXPRESSIONS[:2], *, max_wait_seconds=1, **kwargs):
            return ml.simulate_alphas(session, expressions, settings=SETTINGS, capability_snapshot={},
                                      multi_size=2, max_wait_seconds=max_wait_seconds, **kwargs)

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


def test_workspace_quick_ceiling_leaves_a_slot_for_full(tmp_path, simulator):
    policy = tmp_path / "simulation-slots" / "quick-slot-limit"
    policy.parent.mkdir()
    policy.write_text("6\n")
    for index in range(6):
        assert simulation_slots.quick_acquire(ACCOUNT, 7, f"quick-{index}")
    assert simulation_slots.quick_acquire(ACCOUNT, 7, "quick-6") is None
    full_reservation, shared = simulation_slots.reserve(ACCOUNT, 7)
    assert full_reservation and shared["active"] == 7

    quick = {}
    simulator(dry_run=True, concurrency=7, scheduling_report=quick)
    full = {}
    ml.simulate_alphas(None, EXPRESSIONS[:2], settings={"simulationMode": "FULL"},
                       concurrency=7, dry_run=True, scheduling_report=full)
    assert quick["effective"] == 6
    assert full["effective"] == 7


def test_native_partition_preserves_legacy_and_full_capacity(tmp_path, simulator):
    policies = tmp_path / "simulation-slots"
    policies.mkdir()
    (policies / "quick-slot-limit").write_text("7\n")
    (policies / "quick-single-slot-limit").write_text("2\n")
    (policies / "quick-native-slot-limit").write_text("4\n")
    checkpoint = mock.Mock()
    for _ in range(4):
        identity, _ = simulation_slots.prepare_submission(ACCOUNT, 7, checkpoint, quick_native=True)
        assert identity
    identity, state = simulation_slots.prepare_submission(ACCOUNT, 7, checkpoint, quick_native=True)
    assert identity is None and state["available"] == 0 and checkpoint.call_count == 4
    for index in range(2):
        assert simulation_slots.quick_acquire(ACCOUNT, 7, f"legacy-{index}")
    assert simulation_slots.quick_acquire(ACCOUNT, 7, "legacy-extra") is None
    full, state = simulation_slots.prepare_submission(ACCOUNT, 8, checkpoint)
    assert full and state["active"] == 7


def test_legacy_single_ceiling_preserves_native_and_full_slots(tmp_path, simulator):
    policies = tmp_path / "simulation-slots"
    policies.mkdir()
    (policies / "quick-slot-limit").write_text("6\n")
    (policies / "quick-single-slot-limit").write_text("2\n")
    for index in range(2):
        assert simulation_slots.quick_acquire(ACCOUNT, 7, f"legacy-{index}")
    assert simulation_slots.quick_acquire(ACCOUNT, 7, "legacy-2") is None
    for index in range(4):
        reservation, shared = simulation_slots.reserve(ACCOUNT, 6)
        assert reservation and shared["active"] == index + 3
    assert simulation_slots.reserve(ACCOUNT, 6)[0] is None
    full, shared = simulation_slots.reserve(ACCOUNT, 7)
    assert full and shared["active"] == 7


def test_native_batch_waits_for_shared_capacity_before_one_post(simulator):
    occupied, _ = simulation_slots.reserve(ACCOUNT, 1)
    replies = {
        "NativeParent": {"id": "NativeParent", "status": "COMPLETE", "children": ["Child1", "Child2"]},
        "Child1": child(1),
        "Child2": child(2),
    }

    def release_other_runner(_seconds):
        simulation_slots.reconcile_record({"slot_reservation": occupied}, ACCOUNT, outcome="refused")

    with mock.patch.object(ml, "sleep", side_effect=release_other_runner) as wait, \
            mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL) as post, \
            mock.patch.object(ml, "read_simulation_snapshot",
                              side_effect=lambda _session, identity, **_kwargs: (replies[identity], None)), \
            mock.patch.object(ml, "fetch_alpha_metrics", return_value={"is": {"fitness": 1}}):
        records = simulator(concurrency=1)

    wait.assert_called_once()
    post.assert_called_once()
    assert [record["alpha_id"] for record in records] == ["Alpha1", "Alpha2"]
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 0


def test_native_batch_fills_slot_from_peer_while_its_first_parent_is_pending(simulator):
    occupied, _ = simulation_slots.reserve(ACCOUNT, 2)
    next_submitted = Event()
    expressions = [*EXPRESSIONS, "rank(high)"]
    second_url = PARENT_URL + "Second"
    replies = {
        "NativeParent": {"id": "NativeParent", "status": "COMPLETE", "children": ["Child1", "Child2"]},
        "NativeParentSecond": {"id": "NativeParentSecond", "status": "COMPLETE", "children": ["Child3", "Child4"]},
        **{f"Child{index}": child(index, parent="NativeParent" if index < 3 else "NativeParentSecond",
                                 expression=expression)
           for index, expression in enumerate(expressions, 1)},
    }

    def submit(_session, requests, **_kwargs):
        if requests[0]["regular"] == expressions[0]:
            return PARENT_URL
        next_submitted.set()
        return second_url

    def inspect(_session, identity, **_kwargs):
        if identity == "NativeParent":
            assert next_submitted.wait(2)
        return replies[identity], None

    def release_peer(*_args, **_kwargs):
        simulation_slots.reconcile_record({"slot_reservation": occupied}, ACCOUNT, outcome="refused")
        return {"released": []}

    with mock.patch.object(ml, "submit_multi_simulation", side_effect=submit) as post, \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=inspect), \
            mock.patch.object(ml, "recover_accepted_slots", side_effect=release_peer), \
            mock.patch.object(ml, "recover_quick_orphans", return_value=False), \
            mock.patch.object(ml, "fetch_alpha_metrics", return_value={"is": {"fitness": 1}}):
        records = simulator(expressions, concurrency=2, max_wait_seconds=4)

    assert post.call_count == 2
    assert [record["alpha_id"] for record in records] == ["Alpha1", "Alpha2", "Alpha3", "Alpha4"]
    assert simulation_slots.snapshot(ACCOUNT, 2)["active"] == 0


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
    parent = {"status": "ERROR", "children": ["Child2", "Child1"],
              "error": {"message": "Parent execution failed", "body": "private raw body"}}
    second_child = {**anonymous_failure("Child2", "second error"), "status": "FAIL"}
    first_child = {**anonymous_failure("Child1", "Authorization: Bearer synthetic-secret"),
                   "detail": "first child\n execution timed out", "status": "TIMEOUT"}
    events = []
    with mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL) as post, \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=[
                (parent, None), (second_child, None), (first_child, None),
            ]):
        records = simulator(fetch_metrics=False, on_event=lambda kind, data: events.append((kind, data)))
    post.assert_called_once()
    assert [row["status"] for row in records] == ["ERROR", "ERROR"]
    assert all(row.get("multi_child_id") is None and row.get("alpha_id") is None for row in records)
    assert [row["multi_failure_attribution"] for row in records] == ["unattributed", "unattributed"]
    assert len({row["error"] for row in records}) == 1
    assert all("all remaining multi-simulation children" in row["error"] for row in records)
    evidence = records[0]["multi_terminal_evidence"]
    assert evidence == {
        "parent": {"status": "ERROR", "message": "Parent execution failed"},
        "children": {
            "Child2": {"status": "FAIL", "message": "second error"},
            "Child1": {"status": "TIMEOUT", "message": "first child execution timed out"},
        },
    }
    parent_checkpoint = next(data["records"][0] for event, data in events
                             if event == "multi_pending" and data["records"][0].get("multi_parent_terminal"))
    assert parent_checkpoint["multi_terminal_evidence"] == {"parent": evidence["parent"]}
    assert records[1].get("multi_terminal_evidence") is None
    assert any(event == "multi_pending" and data["records"][0].get("multi_terminal_evidence") == evidence
               for event, data in events)
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 0

    with mock.patch.object(ml, "submit_multi_simulation") as post, \
            mock.patch.object(ml, "read_simulation_snapshot") as read:
        resumed = simulator(initial_records=records, fetch_metrics=False, recover_only=True)
    post.assert_not_called()
    read.assert_not_called()
    assert [row["status"] for row in resumed] == ["ERROR", "ERROR"]
    assert resumed[0]["multi_terminal_evidence"] == evidence


def test_native_recovery_keeps_safe_details_and_rejects_mismatched_child_evidence(simulator):
    parent = {"status": "ERROR", "children": ["Child1", "Child2"], "message": "parent detail"}
    first_child = anonymous_failure("Child1", "one " + "x" * 1024)
    wrong_child = {**anonymous_failure("Child2"), "parent": "OtherParent"}
    with mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL), \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=[
                (parent, None), (first_child, None), (wrong_child, None),
            ]):
        first = simulator(fetch_metrics=False)
    children = first[0]["multi_terminal_evidence"]["children"]
    assert children == {"Child1": {"status": "ERROR", "message": "one " + "x" * 508}}
    assert [row["status"] for row in first] == ["PENDING", "PENDING"]
    first[0]["multi_terminal_evidence"]["children"] = {**children, "UnknownChild": {"status": "ERROR"}}

    no_message = anonymous_failure("Child1", "")
    private_message = anonymous_failure("Child2", "See https://example.test/?key=synthetic-secret")
    with mock.patch.object(ml, "submit_multi_simulation") as post, \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=[
                (no_message, None), (private_message, None),
            ]):
        resumed = simulator(initial_records=first, fetch_metrics=False, recover_only=True)
    post.assert_not_called()
    assert resumed[0]["multi_terminal_evidence"] == {
        "parent": {"status": "ERROR", "message": "parent detail"},
        "children": {**children, "Child2": {"status": "ERROR"}},
    }
    assert all(row["multi_failure_attribution"] == "unattributed" for row in resumed)


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
    assert "multi_parent_accepted_at" not in events[0][2][0]
    assert events[1][0:2] == ("multi_pending", 1)
    assert events[1][2][0]["multi_parent_terminal"]
    assert events[1][2][0]["multi_parent_terminal_observed_at"] > 0
    assert "multi_parent_accepted_at" not in records[0]
    assert not any(key in records[1] for key in simulation_multi.PARENT_TIMING_FIELDS)
    assert [row["alpha_id"] for row in records] == ["Alpha1", "Alpha2"]
    assert all(row["metrics_pending"] for row in records)
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 0


@pytest.mark.parametrize("legacy", [True, False], ids=["old-raw", "new-formatted"])
@pytest.mark.parametrize("contradictory_echo", [False, True], ids=["exact-echo", "changed-echo"])
def test_native_resume_matches_the_verified_wire_payload(legacy, contradictory_echo):
    raw = ["rank( close)", "rank(close )" if legacy else "rank(open )"]
    formatted = ["rank(close)", "rank(close)" if legacy else "rank(open)"]
    settings = ml.regular_wire_settings(ml.resolve_settings(SETTINGS))

    def payload(expression):
        return {"type": "REGULAR", "regular": expression, "settings": settings}

    submitted = raw if legacy else formatted
    records = [{"status": "PENDING", "submission_attempted": True, "multi_request_id": "SavedParent",
                "multi_member": index, "multi_size": 2, "multi_parent_url": PARENT_URL,
                "multi_parent_terminal": True, "multi_children": ["Child2", "Child1"],
                "multi_payload_sha": simulation_multi.payload_digest(payload(expression))}
               for index, expression in enumerate(submitted)]
    runs = simulation_multi.MultiRuns(None, records, lambda index: payload(formatted[index]), {}, 2,
                                      None, lambda *args: None, lambda *args: None,
                                      original_payload_for=lambda index: payload(raw[index]))
    assert list(runs.restore()) == [(0, 1)]
    echoed = list(submitted)
    if contradictory_echo:
        echoed[0] = formatted[0] if legacy else raw[0]
    snapshots = {f"Child{number}": {**payload(expression), "id": f"Child{number}",
                                      "status": "COMPLETE", "alpha": f"Alpha{number}"}
                 for number, expression in enumerate(echoed, 1)}
    with mock.patch.object(ml, "read_simulation_snapshot",
                           side_effect=lambda _, child_id, **kwargs: (snapshots[child_id], None)):
        runs.collect_children((0, 1))

    assert [row.get("alpha_id") for row in records] == [None if contradictory_echo else "Alpha1", "Alpha2"]
    records[0]["multi_payload_sha"] = "unverified"
    with pytest.raises(ValueError, match="does not match"):
        list(runs.restore())


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


@pytest.mark.parametrize("legacy", [False, True])
def test_native_partial_parent_receipt_can_recover_a_later_child(simulator, legacy):
    partial = {"status": "ERROR", "children": ["Child1"]}
    complete = {"status": "ERROR", "children": ["Child2", "Child1"]}
    with mock.patch.object(simulation_multi.time, "time", return_value=100), \
            mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL), \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=[(partial, None), (child(1), None)]):
        first = simulator(fetch_metrics=False)
    assert [row["alpha_id"] for row in first] == ["Alpha1", None]
    assert first[0]["multi_parent_terminal_observed_at"] == 100
    if legacy:
        for key in simulation_multi.PARENT_TIMING_FIELDS:
            first[0].pop(key, None)
    with mock.patch.object(simulation_multi.time, "time", return_value=101), \
            mock.patch.object(ml, "submit_multi_simulation") as post, \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=[(complete, None), (child(2), None)]):
        second = simulator(initial_records=first, fetch_metrics=False, recover_only=True)
    post.assert_not_called()
    assert [row["alpha_id"] for row in second] == ["Alpha1", "Alpha2"]
    assert second[0].get("multi_parent_terminal_observed_at") == (None if legacy else 100)
    assert second[0].get("multi_parent_accepted_at") == (None if legacy else 100)


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


def test_native_pending_parent_without_retry_after_saves_a_fallback_poll(simulator):
    pending = {"id": "NativeParent", "status": "PENDING"}
    with mock.patch.object(simulation_multi.time, "time", return_value=100), \
            mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL), \
            mock.patch.object(ml, "read_simulation_snapshot", return_value=(pending, None)):
        records = simulator(fetch_metrics=False)
    assert records[0]["multi_next_parent_poll_at"] == (
        100 + simulation_multi.PARENT_POLL_FALLBACK_SECONDS)

    with mock.patch.object(simulation_multi.time, "time", return_value=129), \
            mock.patch.object(ml, "read_simulation_snapshot") as read:
        simulator(initial_records=records, fetch_metrics=False, recover_only=True)
    read.assert_not_called()


@pytest.mark.parametrize("status", [200, 429])
def test_native_watch_parent_reads_once_and_retains_retry_before_terminal(simulator, status):
    pending = {"id": "NativeParent", "status": "SIMULATING"}
    with mock.patch.object(simulation_multi.time, "time", return_value=100), \
            mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL), \
            mock.patch.object(ml, "read_simulation_snapshot", return_value=(pending, 5)):
        saved = simulator(fetch_metrics=False)
    assert saved[0]["multi_next_parent_poll_at"] == 105
    assert saved[0]["multi_parent_accepted_at"] == 100
    assert saved[0]["multi_last_pending_observed_at"] == 100
    assert "multi_parent_terminal_observed_at" not in saved[0]

    response = mock.Mock(status_code=status, headers={"Retry-After": "5"})
    response.json.return_value = pending
    session = mock.Mock()
    session.get.return_value = response
    inspect = ml.read_simulation_snapshot
    recovery = {"fetch_metrics": False, "recover_only": True, "native_poll_once": True,
                "max_wait_seconds": 60}
    progress = []
    with mock.patch.object(ml, "sleep", side_effect=AssertionError("watch recovery must not wait")), \
            mock.patch.object(ml, "submit_multi_simulation") as post:
        with mock.patch.object(simulation_multi.time, "time", return_value=105), \
                mock.patch.object(ml, "read_simulation_snapshot", side_effect=lambda _, identity, **kw:
                                  inspect(session, identity, **kw)):
            saved = simulator(initial_records=saved, on_event=lambda kind, payload:
                              progress.append((kind, payload)), **recovery)
        session.get.assert_called_once()
        assert saved[0]["multi_next_parent_poll_at"] == 110
        last_pending = 105 if status == 200 else 100
        assert saved[0]["multi_last_pending_observed_at"] == last_pending
        assert [(kind, payload["records"][0]["multi_next_parent_poll_at"])
                for kind, payload in progress if kind == "multi_pending"] == [("multi_pending", 110)]
        assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 1

        with mock.patch.object(simulation_multi.time, "time", return_value=109), \
                mock.patch.object(ml, "read_simulation_snapshot") as read:
            saved = simulator(initial_records=saved, **recovery)
        read.assert_not_called()
        assert saved[0]["multi_next_parent_poll_at"] == 110

        completed = {"id": "NativeParent", "status": "COMPLETE", "children": ["Child1", "Child2"]}
        with mock.patch.object(simulation_multi.time, "time", return_value=110), \
                mock.patch.object(ml, "read_simulation_snapshot", side_effect=[
                    (completed, None), (child(1), None), (child(2), None)]) as read:
            saved = simulator(initial_records=saved, **recovery)
        assert [call.args[1] for call in read.call_args_list] == ["NativeParent", "Child1", "Child2"]
    post.assert_not_called()
    assert [row["alpha_id"] for row in saved] == ["Alpha1", "Alpha2"]
    assert saved[0]["multi_next_parent_poll_at"] is None
    assert saved[0]["multi_parent_accepted_at"] == 100
    assert saved[0]["multi_last_pending_observed_at"] == last_pending
    assert saved[0]["multi_parent_terminal_observed_at"] == 110
    assert not any(key in saved[1] for key in simulation_multi.PARENT_TIMING_FIELDS)
    assert simulation_slots.snapshot(ACCOUNT, 3)["active"] == 0


def test_direct_native_recovery_waits_within_its_budget(simulator):
    pending = {"id": "NativeParent", "status": "SIMULATING"}
    with mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL), \
            mock.patch.object(ml, "read_simulation_snapshot", return_value=(pending, 60)):
        saved = simulator(fetch_metrics=False)
    terminal = {"id": "NativeParent", "status": "COMPLETE", "children": ["Child1", "Child2"]}
    seen = []
    with mock.patch.object(ml, "sleep", side_effect=lambda _delay: seen.append("sleep")) as sleep, \
            mock.patch.object(ml, "read_simulation_snapshot", side_effect=[
                (pending, 5), (terminal, None), (child(1), None), (child(2), None)]) as read:
        saved = simulator(initial_records=saved, fetch_metrics=False, recover_only=True, max_wait_seconds=120,
                          on_event=lambda kind, _: seen.append(kind))
    assert [call.args[1] for call in read.call_args_list] == ["NativeParent", "NativeParent", "Child1", "Child2"]
    assert sleep.call_count == 2
    assert sleep.call_args.args == (5,)
    assert seen[:4] == ["sleep", "multi_pending", "polling", "sleep"]
    assert [row["alpha_id"] for row in saved] == ["Alpha1", "Alpha2"]


@pytest.mark.parametrize("payload,expected", [({"id": "NativeParent", "status": "RUNNING"}, 101),
                                             ({"id": "Other", "status": "PENDING"}, 100),
                                             ({"id": "NativeParent", "status": "UNRECOGNIZED"}, 100)])
def test_native_pending_observation_requires_valid_pending_snapshot(simulator, payload, expected):
    with mock.patch.object(simulation_multi.time, "time", return_value=100), \
            mock.patch.object(ml, "submit_multi_simulation", return_value=PARENT_URL), \
            mock.patch.object(ml, "read_simulation_snapshot", return_value=({"progress": 0.25}, 1)):
        saved = simulator(native_poll_once=True, fetch_metrics=False)
    with mock.patch.object(simulation_multi.time, "time", return_value=101), \
            mock.patch.object(ml, "read_simulation_snapshot", return_value=(payload, None)):
        saved = simulator(initial_records=saved, recover_only=True, native_poll_once=True, fetch_metrics=False)
    assert saved[0]["multi_last_pending_observed_at"] == expected
    assert "multi_parent_terminal_observed_at" not in saved[0]


def test_native_watch_flag_keeps_singleton_on_ordinary_polling_route(simulator):
    url = ml.BRAIN_API_URL + "/simulations/Single"
    with (mock.patch.object(ml, "submit_single_simulation", return_value=url) as post,
          mock.patch.object(ml, "poll_simulation", return_value={
              "status": "PENDING", "next_poll_after_seconds": 5}) as poll,
          mock.patch.object(ml, "read_simulation_snapshot") as read_native):
        rows = simulator(EXPRESSIONS[:1], native_poll_once=True, fetch_metrics=False)
    post.assert_called_once()
    poll.assert_called_once()
    assert poll.call_args.args[1] == url
    read_native.assert_not_called()
    assert rows[0]["progress_url"] == url
    assert rows[0]["poll_after_seconds"] == 5
    assert not rows[0].get("multi_request_id")


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


def test_request_groups_split_on_settings_change():
    payloads = {
        0: {"settings": {"decay": 4}, "regular": "rank(close)"},
        1: {"settings": {"decay": 8}, "regular": "rank(open)"},
        2: {"settings": {"decay": 8}, "regular": "rank(volume)"},
        3: {"settings": {"decay": 8}, "regular": "rank(close)"},
    }
    groups = list(simulation_multi.request_groups([0, 1, 2, 3], payloads.__getitem__, 4))
    assert groups == [(0,), (1, 2, 3)]


def test_request_groups_split_on_repeat_but_keep_size():
    payloads = {
        0: {"settings": {"decay": 4}, "regular": "rank(close)"},
        1: {"settings": {"decay": 4}, "regular": "rank(close)"},
        2: {"settings": {"decay": 4}, "regular": "rank(open)"},
    }
    groups = list(simulation_multi.request_groups([0, 1, 2], payloads.__getitem__, 2))
    assert groups == [(0,), (1, 2)]
