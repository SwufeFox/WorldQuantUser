"""An unverified original Location cannot release its job's shared reservation."""

from types import SimpleNamespace
import json
from unittest.mock import Mock

import pytest
import requests

import alpha_registry
import machine_lib as ml
import simulation_capabilities
import simulation_slots
from tools.pipeline import batch_quick as quick


ACCOUNT = {"state": "known", "fingerprint": "quick-identity-lease-account"}
ARM = {"label": "original", "expr": "rank(close)", "idea": "Test the original price signal.",
       "settings": {"region": "USA", "universe": "TOP3000", "delay": 1}}
PROGRESS = ml.BRAIN_API_URL + "/simulations/identity-original"
ALPHA_ID = "IdentityAlpha1"
ALPHA_URL = ml.BRAIN_API_URL + "/alphas/" + ALPHA_ID


def response(status, body=None, **headers):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers)
    result._content = json.dumps(body or {}).encode()
    return result


@pytest.fixture(params=["UNKNOWN_SUBMISSION", "UNVERIFIED_RESULT"])
def original_job(request, tmp_path, monkeypatch):
    now = [0.0]

    def advance(seconds):
        assert seconds >= 0
        now[0] += seconds

    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    monkeypatch.setattr(simulation_capabilities, "account_provenance", lambda session=None: ACCOUNT)
    monkeypatch.setattr(quick.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(quick.time, "time", lambda: 1000 + now[0])
    monkeypatch.setattr(quick.time, "sleep", advance)
    client = Mock()
    client.post.side_effect = AssertionError("identity recovery must never POST")
    monkeypatch.setattr(ml, "load_session", lambda **kwargs: client)
    out = tmp_path / "quick.jsonl"
    payload = quick._payload(ARM)
    runners = []

    def runner():
        current = quick.Runner(out, ACCOUNT, 1, 2, admission={
            "ok": False, "path": str(tmp_path / "admission.json"),
            "blocking": [{"code": "recovery_only"}]})
        runners.append(current)
        return current

    initial = runner()
    key = initial._job_key(ARM["label"], payload)
    previous = {"label": ARM["label"], "expr": ARM["expr"], "status": request.param,
                "simulation_status": "COMPLETE", "alpha_id": None, "observed_alpha_id": ALPHA_ID,
                "progress_url": PROGRESS, "simulation_job_key": key,
                "simulation_key": quick._simulation_key(payload),
                "account_fingerprint": ACCOUNT["fingerprint"]}
    out.write_text(json.dumps(previous) + "\n", encoding="utf-8")
    assert not initial.state_path.exists()
    assert simulation_slots.quick_known_jobs(ACCOUNT) == set()

    def recover():
        current = runner()
        assert quick._select_arms([ARM], current, quick.load_done(out, ACCOUNT)) == [ARM]
        assert current.needs_new_submission(ARM) is False
        return current.run(ARM)

    def matching(target, **kwargs):
        if target == PROGRESS:
            return response(200, {"status": "COMPLETE", "alpha": ALPHA_ID})
        assert target == ALPHA_URL
        return response(200, {**payload, "id": ALPHA_ID,
                              "settings": {**payload["settings"], "language": "FASTEXPR"},
                              "is": {"sharpe": 1.2, "fitness": 0.7, "checks": []}})

    yield SimpleNamespace(client=client, out=out, payload=payload, recover=recover,
                          matching=matching, advance=advance, key=key)
    for current in runners:
        current.close()


def test_mismatched_alpha_keeps_the_unverified_identity_and_slot(original_job):
    def wrong_payload(target, **kwargs):
        result = original_job.matching(target, **kwargs)
        if target == ALPHA_URL:
            body = result.json()
            body["regular"] = "rank(volume)"
            return response(200, body)
        return result

    original_job.client.get.side_effect = wrong_payload
    result = original_job.recover()
    assert result["status"] == "UNVERIFIED_RESULT"
    assert result["progress_url"] == PROGRESS
    assert result["observed_alpha_id"] == ALPHA_ID
    assert result["alpha_id"] is None
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 1
    original_job.client.post.assert_not_called()


def test_pending_identity_metrics_keep_the_slot_until_a_later_match(original_job):
    def pending_metrics(target, **kwargs):
        if target == PROGRESS:
            return original_job.matching(target, **kwargs)
        assert target == ALPHA_URL
        return response(429, **{"Retry-After": "30"})

    original_job.client.get.side_effect = pending_metrics
    pending = original_job.recover()
    assert pending["status"] == "PENDING"
    assert pending["metrics_pending"] is True
    assert pending["progress_url"] == PROGRESS
    assert pending["observed_alpha_id"] == ALPHA_ID
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 1

    original_job.advance(31)
    original_job.client.get.side_effect = original_job.matching
    completed = original_job.recover()
    assert completed["status"] == "COMPLETE"
    assert completed["identity_verified"] is True
    assert completed["alpha_id"] == ALPHA_ID
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 0
    original_job.client.post.assert_not_called()


def test_unconfirmed_failed_location_neither_releases_nor_records_a_failure(original_job):
    def failed_operation(target, **kwargs):
        assert target == PROGRESS
        return response(200, {"id": "identity-original", "status": "ERROR"})

    original_job.client.get.side_effect = failed_operation
    result = original_job.recover()
    assert result["status"] == "UNVERIFIED_RESULT"
    assert result["progress_url"] == PROGRESS
    assert result["alpha_id"] is None
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 1
    quick._record_seed(ARM, result, original_job.payload,
                       str(original_job.out.with_suffix(".arms.json")), original_job.out)
    assert alpha_registry.load_entries() == []
    original_job.client.post.assert_not_called()

    # An independently confirmed original failure must have an exit: the
    # existing explicit reconciliation path releases it and records the error.
    ml.attach_quick_location(original_job.client, ACCOUNT, original_job.key, PROGRESS, confirm_failure=True)
    assert simulation_slots.snapshot(ACCOUNT, 1)["active"] == 0
    confirmed = original_job.recover()
    assert confirmed["status"] == "ERROR" and confirmed["identity_verified"] is True
    summary = quick._record_seed(ARM, confirmed, original_job.payload,
                                 str(original_job.out.with_suffix(".arms.json")), original_job.out)
    assert summary["recorded_failures"] == 1
    original_job.client.post.assert_not_called()
