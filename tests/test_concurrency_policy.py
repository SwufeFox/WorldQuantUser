"""Legacy runners share account evidence, not a universal request-burst limit."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

import machine_lib as ml
import simulation_capabilities
import simulation_multi
import simulation_slots
from tools.pipeline import batch_quick, pipeline_pc


ARM = {"label": "close_rank", "expr": "rank(close)",
       "settings": {"region": "USA", "universe": "TOP3000", "delay": 1}}


def evidence(account, limit=None, *, state="verified"):
    return {"state": state, "account": account,
            "concurrency": {"state": "verified" if limit is not None else "unknown",
                            "limit": limit, "source": "offline-authoritative"}}


@pytest.fixture
def offline(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    session = requests.Session()
    session._brain_account_id = "offline-concurrency-account"
    account = simulation_capabilities.account_provenance(session)
    no_network = Mock(side_effect=AssertionError("concurrency policy must not discover or simulate online"))
    monkeypatch.setattr(requests.Session, "request", no_network)
    monkeypatch.setattr(simulation_capabilities, "fetch", no_network)
    monkeypatch.setattr(ml, "load_session", lambda **_kwargs: session)
    monkeypatch.setattr(simulation_capabilities, "cached", lambda: evidence(account))
    yield session, account
    session.close()


def test_argument_parsing_preserves_separate_worker_and_account_budgets(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["batch_quick.py", "arms.json", "--concurrency", "13",
                                     "--account-slots", "11"])
    monkeypatch.setattr(ml, "concurrency_plan", Mock(side_effect=AssertionError("not authenticated yet")))
    args = batch_quick._batch_args(batch_quick._batch_parser("offline", "results.jsonl", 5))
    assert (args.concurrency, args.account_slots) == (13, 11)


@pytest.mark.parametrize("module", [batch_quick, pipeline_pc], ids=["quick", "full-pc"])
@pytest.mark.parametrize("kind,expected", [
    ("unknown", 11), ("verified", 3), ("larger", 11), ("stale", 11), ("other_account", 11), ("zero", 0),
])
def test_legacy_runners_pass_effective_capacity_to_shared_slots(
        tmp_path, monkeypatch, capsys, offline, module, kind, expected):
    _, account = offline
    snapshot = evidence(account, {"unknown": None, "larger": 20, "zero": 0}.get(kind, 3))
    if kind == "stale":
        snapshot["state"] = "unknown"
        snapshot["reasons"] = ["freshness_unverified"]
    elif kind == "other_account":
        snapshot["account"] = {**account, "fingerprint": "different-account"}
    monkeypatch.setattr(simulation_capabilities, "cached", lambda: snapshot)
    source = tmp_path / "arms.json"
    source.write_text(json.dumps([ARM]), encoding="utf-8")
    args = SimpleNamespace(arms=source, concurrency=13, account_slots=11, limit=0, poll_timeout=1,
                           min_sharpe=0.9, max_turnover=0.7, pc_max_wait=0)
    created = []
    real_runner = module.Runner

    def create(*args, **kwargs):
        runner = real_runner(*args, **kwargs)
        created.append(runner)
        return runner

    monkeypatch.setattr(module, "Runner", create)
    if module is batch_quick:
        monkeypatch.setattr(module, "_select_arms", lambda *_args: [])
    else:
        monkeypatch.setattr(module, "_process", lambda *_args: None)
    assert module._run_batch(args, tmp_path / "results.jsonl") == 0
    runner, = created
    assert runner.account == account
    assert runner.slots == expected
    assert (args.concurrency, args.account_slots) == (13, 11)
    assert ("verified capacity" in capsys.readouterr().err) == (expected < 11)

    # The effective limit must reach the shared reservation journal, so a second
    # process cannot interpret a caller's original larger budget as capacity.
    for index in range(expected):
        assert simulation_slots.quick_acquire(account, runner.slots, f"job-{index}") is not None
    assert simulation_slots.quick_acquire(account, runner.slots, "over-budget") is None


def test_zero_capacity_defers_new_work_but_keeps_accepted_job_recoverable(tmp_path, offline):
    session, account = offline
    url = ml.BRAIN_API_URL + "/simulations/ExistingJob"
    reservation = simulation_slots.quick_acquire(account, 2, "existing", existing_url=url)
    runner = batch_quick.Runner(tmp_path / "zero.jsonl", account, 0, poll_timeout=1)
    try:
        slot, deferred = runner._post(batch_quick._payload(ARM), "new", ARM, existing_url=None)
        assert slot is None
        assert deferred["status"] == "QUEUED"
        assert deferred["detail"] == "account slot budget is zero"
        slot, deferred = runner._post(batch_quick._payload(ARM), "existing", ARM, existing_url=url)
        assert deferred is None
        assert slot["id"] == reservation["id"]
        assert slot["progress_url"] == url
        assert simulation_slots.snapshot(account, 0)["active"] == 1
    finally:
        runner.close()


def test_zero_capacity_revokes_a_prior_pre_post_preparation(tmp_path, offline):
    _, account = offline
    reservation = simulation_slots.quick_acquire(account, 2, "prepared", prepare=True)
    assert reservation["state"] == "quick_preparing"
    runner = batch_quick.Runner(tmp_path / "zero-prepared.jsonl", account, 0, poll_timeout=1)
    try:
        slot, deferred = runner._post(batch_quick._payload(ARM), "prepared", ARM, existing_url=None)
        assert slot is None and deferred["status"] == "QUEUED"
        assert not simulation_slots.quick_update(account, reservation["id"], "submitting")
        assert simulation_slots.snapshot(account, 0)["active"] == 0
    finally:
        runner.close()


def test_429_sets_shared_cooldown_without_becoming_a_capacity_bound(tmp_path, monkeypatch, offline):
    session, account = offline
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "60"
    response._content = b"{}"
    post = Mock(return_value=response)
    monkeypatch.setattr(session, "post", post)
    plan = batch_quick._account_slot_plan(session, 11)
    runner = batch_quick.Runner(tmp_path / "throttled.jsonl", account, plan["effective"], poll_timeout=1)
    try:
        assert runner.run(ARM)["status"] == "QUEUED"
        assert simulation_slots.submission_retry_at(account) is not None
        assert simulation_slots.snapshot(account, 11)["active"] == 0
        later = runner.run({**ARM, "label": "later", "expr": "rank(volume)"})
        assert later["status"] == "QUEUED"
        post.assert_called_once()
        after = batch_quick._account_slot_plan(session, 11)
        assert (after["effective"], after["platform_limit"]) == (11, None)
        assert runner.slots == 11
    finally:
        runner.close()


def test_native_parent_limit_stays_separate_from_local_workers_and_pack_size():
    scheduling = {}
    ml.simulate_alphas(None, ["rank(close)", "rank(open)", "rank(volume)"], dry_run=True,
                       concurrency=11, multi_size=2, capability_snapshot={}, scheduling_report=scheduling)
    assert scheduling["requested"] == 11
    assert scheduling["effective"] == simulation_multi.CONSULTANT_PARENT_LIMIT
    assert scheduling["platform_limit"] is None
    assert [row["indices"] for row in scheduling["multi_requests"]] == [[0, 1], [2]]
