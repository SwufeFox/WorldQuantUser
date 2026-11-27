"""Regression coverage for strict publication and offline queue diagnosis."""

import argparse
import hashlib
import json

import pytest

import cache_store
from research_app import campaign_delivery as delivery, experiments
from research_app.runtime import CliError
from tools import _runner
from tools.pipeline import batch_quick


@pytest.fixture
def delivery_args(tmp_path):
    spec = tmp_path / "pilot.json"
    arms = [
        {"label": f"arm{index}", "expression": expression,
         "idea": f"Scoped hypothesis {index}", "mechanism": "reporting-revisions"}
        for index, expression in enumerate(
            [f"rank(signal{i})" for i in range(8)]
            + [f"ts_delta(signal{i}, 40)" for i in range(8, 16)])
    ]
    spec.write_text(json.dumps({
        "name": "test delivery", "track": "REGULAR", "design": "explore",
        "settings": {"region": "KOR", "universe": "TOP600", "simulationMode": "QUICK"},
        "arms": arms,
    }))
    return argparse.Namespace(spec=str(spec), research_dir=tmp_path, campaign_dir=tmp_path / "campaign",
                              phase="pilot", pilot_batch=None, max_wait_seconds=30, multi_size=10,
                              meta_max_age=None, limit=10, save=None, subcommand="publish")


@pytest.fixture
def platform_evidence(monkeypatch):
    calls = []
    account = {"fingerprint": "test-account"}
    field_evidence = {"state": "verified"}
    records = [{"id": f"signal{i}", "type": "MATRIX", "dataset": {"id": f"ds{i % 3}"}}
               for i in range(16)]

    def check(root, *command, input_text, timeout):
        calls.append(command)
        assert command[:4] == ("probe", "-", "--check", "--strict-gate")
        identity = experiments.probe_spec_identity("-", document=json.loads(input_text))
        return _runner.CliResult(True, {
            "preflight_complete": True, "capability_snapshot": {"account": account},
            "request_fingerprint": identity["request_fingerprint"],
        })

    monkeypatch.setattr(delivery, "run_cli", check)
    monkeypatch.setattr(delivery, "with_session", lambda args, work: work(object()))
    monkeypatch.setattr(delivery.simulation_capabilities, "account_provenance", lambda session: account)
    monkeypatch.setattr(batch_quick.capabilities, "cached_field_scope",
                        lambda settings, account: {"evidence": field_evidence, "records": records})
    return calls, records


def test_strict_preflight_publishes_identical_bytes_to_top_level(delivery_args, platform_evidence):
    result = delivery.cmd_campaign_publish(delivery_args, None)
    destination = delivery_args.campaign_dir / "backlog" / "pilot.json"
    assert result["state"] == "queued"
    assert result["top_level_ls"] == str(destination)
    assert result["arms"] == 16
    assert result["max_skeleton_count"] == 8
    assert result["datasets"] == ["ds0", "ds1", "ds2"]
    assert result["spec_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert platform_evidence[0] == [
        ("probe", "-", "--check", "--strict-gate", "--max-wait-seconds", "30", "--multi-size", "10")]
    with pytest.raises(CliError, match="spec name already"):
        delivery.cmd_campaign_publish(delivery_args, None)


def test_incomplete_check_and_unverified_diversity_never_publish(delivery_args, platform_evidence, monkeypatch):
    _, records = platform_evidence
    records[:] = [row for row in records if row["dataset"]["id"] != "ds2"]
    with pytest.raises(CliError, match="verified dataset admission blocked"):
        delivery.cmd_campaign_publish(delivery_args, None)
    monkeypatch.setattr(delivery, "run_cli",
                        lambda *args, **kwargs: _runner.CliResult(True, {"preflight_complete": False}))
    with pytest.raises(CliError, match="exact probe preflight did not pass"):
        delivery.cmd_campaign_publish(delivery_args, None)
    assert not (delivery_args.campaign_dir / "backlog").exists()


def test_expansion_requires_same_scope_qualifying_mechanism(delivery_args, platform_evidence, monkeypatch):
    delivery_args.phase = "expansion"
    delivery_args.pilot_batch = delivery_args.campaign_dir / "backlog-results" / "pilot-batch.json"
    with open(delivery_args.spec) as handle:
        identity = experiments.probe_spec_identity("-", document=json.load(handle))
    arms = identity["results"]
    pilot = {"complete": True, "track": identity["track"], "arms": arms,
             "rows": [{"status": "COMPLETE", "alpha_id": "Aa12345z",
                       "train_sharpe": 1.2, "train_turnover": 0.45, "train_fitness": 0.7}
                      for arm in arms]}
    monkeypatch.setattr(experiments, "load_probe_batch", lambda *args, **kwargs: (args[0], pilot))
    with pytest.raises(CliError, match="same-scope pilot evidence"):
        delivery.cmd_campaign_publish(delivery_args, None)
    assert not platform_evidence[0]
    pilot["rows"][0]["train_turnover"] = 0.2
    result = delivery.cmd_campaign_publish(delivery_args, None)
    assert result["phase"] == "expansion"
    assert result["pilot"]["qualifying_alpha_ids_by_mechanism"] == {"reporting-revisions": ["Aa12345z"]}


def test_diagnose_recovers_exact_archived_failure_without_log(tmp_path):
    inbox = tmp_path / "backlog"
    inbox.mkdir()
    spec = inbox / "path.json"
    spec.write_text("{}")
    code = _runner.run_queue(
        tmp_path, inbox, lambda path: (tmp_path / "batch.json", tmp_path / "receipt.json"),
        lambda _event: None, max_wait=1, interval=1, once=True,
        execute=lambda *args, **kwargs: _runner.ProbeRun(
            "failed", {"diagnosis": {"reason": "invalid_spec"}}, "fixture reason"),
    )
    assert code == 1
    manifest = inbox / "failed" / ".reasons" / "path.json"
    assert cache_store.load_json(manifest)[0]["detail"] == "fixture reason"
    output = delivery.cmd_campaign_diagnose(argparse.Namespace(
        research_dir=tmp_path, campaign_dir=None, subcommand="diagnose", limit=10, save=None), None)
    assert output["approved_count"] == output["ready_count"] == 0
    assert "approved_reserve_empty" in output["issues"]
    assert output["recent_failures"][0]["diagnosis"]["reason"] == "invalid_spec"
