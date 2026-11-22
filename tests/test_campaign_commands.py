"""Exercise campaign routing and recovery through the public CLI."""

import json
from pathlib import Path

from research_app import app
from test_python_preflight import SOURCE
from test_spc_specs import SAMPLE, spec as spc_spec


def invoke(capsys, root, *arguments):
    code = app.main(["--research-dir", str(root), "--quiet", *map(str, arguments)])
    return code, json.loads(capsys.readouterr().out)


def test_preview_keeps_research_track_out_of_native_request(tmp_path, capsys):
    source = tmp_path / "probe.json"
    spec = {"track": "Power Pool", "design": "explore",
            "settings": {"simulation_mode": "quick"},
            "arms": [{"expression": "rank(close)"}]}
    source.write_text(json.dumps(spec), encoding="utf-8")

    code, envelope = invoke(capsys, tmp_path, "campaign", "preview", source)
    assert code == 0
    preview = envelope["data"]
    assert preview["track"] == "POWER_POOL"
    assert preview["routes"] == [{"track": "POWER_POOL", "protocol": "brain_simulation",
                                   "simulation_type": "REGULAR", "language": "FASTEXPR",
                                   "simulation_mode": "QUICK"}]
    assert preview["results"][0]["payload"]["type"] == "REGULAR"
    assert preview["results"][0]["payload"]["settings"]["simulationMode"] == "QUICK"
    assert "track" not in preview["results"][0]["payload"]
    assert "track" not in preview["results"][0]["payload"]["settings"]

    spec["track"] = "ATOM"
    source.write_text(json.dumps(spec), encoding="utf-8")
    code, atom = invoke(capsys, tmp_path, "campaign", "preview", source)
    assert code == 0
    assert atom["data"]["track"] == "ATOM"
    assert atom["data"]["request_fingerprint"] == preview["request_fingerprint"]


def test_campaign_runs_spc_offline_and_archives_terminal_spec(tmp_path, capsys):
    inbox = tmp_path / "backlog"
    inbox.mkdir()
    capture = inbox / "captures" / "record.json"
    capture.parent.mkdir()
    capture.write_text(SAMPLE, encoding="utf-8")
    source = inbox / "prediction.json"
    source.write_text(json.dumps(spc_spec(track="SPC", runs=["captures/record.json"])), encoding="utf-8")

    code, status = invoke(capsys, tmp_path, "campaign", "status")
    assert code == 0
    assert status["data"]["jobs"][0]["track"] == "SPC"
    assert not (tmp_path / "backlog-results").exists()
    code, result = invoke(capsys, tmp_path, "campaign", "run")
    assert code == 0
    assert result["data"]["state"] == "done_or_empty"
    assert not source.exists()
    assert (inbox / "done" / source.name).exists()
    receipt = json.loads((tmp_path / "backlog-results" / "prediction-result.json").read_text())
    assert receipt["data"]["experiment"]["ok"] is True
    assert receipt["data"]["simulated"] == 0
    assert not (tmp_path / "backlog-results" / "prediction-batch.json").exists()
    assert capture.read_text() == SAMPLE
    code, status = invoke(capsys, tmp_path, "campaign", "status")
    assert code == 0
    assert status["data"]["archived_quick_pnl"]["unreadable_batches"] == 0


def test_campaign_routes_python_source_on_its_effective_arm_settings(tmp_path, capsys):
    (tmp_path / "alpha.py").write_text(SOURCE, encoding="utf-8")
    source = tmp_path / "comparison.json"
    source.write_text(json.dumps({"design": "explore", "arms": [
        {"expression": "rank(close)", "settings": {"simulationMode": "QUICK"}},
        {"source_file": "alpha.py", "settings": {"language": "PYTHON", "lookback": 2,
                                                "simulationMode": "FULL"}},
    ]}), encoding="utf-8")
    code, envelope = invoke(capsys, tmp_path, "campaign", "preview", source)
    assert code == 0
    data = envelope["data"]
    assert data["lint"]["ok"]
    assert [(route["language"], route["simulation_mode"]) for route in data["routes"]] == [
        ("FASTEXPR", "QUICK"), ("PYTHON", "FULL")]
    assert data["results"][1]["payload"]["regular"] == SOURCE


def test_recover_only_does_not_create_a_batch_for_an_unstarted_spec(tmp_path, capsys):
    inbox = tmp_path / "backlog"
    inbox.mkdir()
    source = inbox / "research.json"
    source.write_text(json.dumps({"design": "explore", "arms": [{"expression": "rank(close)"}]}))

    code, result = invoke(capsys, tmp_path, "campaign", "run", "--recover-only")
    assert code == 0
    assert result["data"]["state"] == "deferred"
    assert source.exists()
    assert not list((tmp_path / "backlog-results").glob("*.json"))


def test_spc_preview_protects_recorded_output(tmp_path, capsys):
    capture = tmp_path / "capture.json"
    capture.write_text(SAMPLE, encoding="utf-8")
    source = tmp_path / "prediction.json"
    source.write_text(json.dumps(spc_spec(runs=[capture.name])), encoding="utf-8")

    code, result = invoke(capsys, tmp_path, "campaign", "preview", source, "--save", capture)
    assert code != 0
    assert result["error"]["code"] == "INVALID_INPUT"
    assert capture.read_text() == SAMPLE


def test_preview_protects_the_probe_spec_even_if_input_is_invalid(tmp_path, capsys):
    source = tmp_path / "broken.json"
    source.write_text("{unclosed", encoding="utf-8")
    code, result = invoke(capsys, tmp_path, "campaign", "preview", source, "--save", source)
    assert code != 0
    assert result["error"]["code"] == "INVALID_INPUT"
    assert source.read_text() == "{unclosed"


def quick_arm(label, expression, **extra):
    return {"label": label, "expression": expression,
            "settings": {"simulationMode": "QUICK", "testPeriod": "P1Y"}, **extra}


def quick_record(alpha_id, train_sharpe=None, *, full=6.0, status="COMPLETE"):
    train = ({"sharpe": train_sharpe, "fitness": train_sharpe / 2}
             if train_sharpe is not None else {})
    return {"alpha_id": alpha_id, "status": status, "metrics": {"sharpe": full, "fitness": full},
            "out_of_sample": {"train": train, "test": {"sharpe": full}}}


def write_quick_batch(path, arms, records, **extra):
    batch = {"format": "probe-batch", "version": 2, "run_id": "quick-run",
             "track": "ATOM", "design": {"mode": "explore"}, "mechanism": "hypothesis A",
             "arms": arms, "records": records, "rows": [{} for _ in arms], **extra}
    path.write_text(json.dumps(batch), encoding="utf-8")


def test_campaign_breed_freezes_balanced_train_only_offspring(tmp_path, capsys, monkeypatch):
    import machine_lib as ml
    from research_app import experiments

    def unexpected_session(*args, **kwargs):
        raise AssertionError("offline breeding opened a session")

    monkeypatch.setattr(experiments, "with_session", unexpected_session)
    batch = tmp_path / "parents.json"
    output = tmp_path / "offspring.spec.json"
    settings = {**ml.DEFAULT_SETTINGS, "simulationMode": "QUICK", "testPeriod": "P1Y"}
    arms = [quick_arm("winner", "rank(ts_mean(close,5)/ts_mean(volume,10))", settings=settings),
            quick_arm("other", "rank(ts_mean(close,20)/ts_mean(volume,40))", settings=settings),
            quick_arm("different-window", "rank(ts_mean(close,30)/ts_mean(volume,60))", settings=settings),
            quick_arm("no-train", "rank(ts_mean(close,50)/ts_mean(volume,80))", settings=settings)]
    records = [quick_record("A1", 2, full=0.1), quick_record("A2", 1, full=6),
               quick_record("A3", 7), quick_record("A4", full=10)]
    records[2]["out_of_sample"]["test"]["startDate"] = "2022-12-31"
    write_quick_batch(batch, arms, records)
    command = ("campaign", "breed", batch, "--out", output, "--seed", 0, "--limit", 1)

    code, response = invoke(capsys, tmp_path, *command)

    assert code == 0
    spec = json.loads(output.read_text(encoding="utf-8"))
    evidence = spec["breeding_source"]
    assert evidence == response["data"]["breeding_source"]
    assert evidence["scope"] == "train"
    assert evidence["proposals_per_policy"] == evidence["contrasting_pairs"] == 1
    [group] = evidence["comparisons"]
    assert [row["alpha_id"] for row in group["eligible"]] == ["A1", "A2"]
    assert group["selected"]["parent_alpha_id"] == "A1"
    assert group["lottery"]["parent_alpha_id"] == "A2"
    assert {arm["expression"] for arm in spec["arms"]} == {
        "rank(ts_mean(close, 5) / ts_mean(volume, 40))",
        "rank(ts_mean(close, 20) / ts_mean(volume, 10))",
    }
    assert all(arm["settings"] == settings for arm in spec["arms"])
    assert experiments.probe_spec_identity(output)["breeding_source"] == evidence

    original = output.read_bytes()
    for record in records:
        record["metrics"]["fitness"] += 100
        record["out_of_sample"]["test"]["sharpe"] += 100
    write_quick_batch(batch, arms, records)
    code, _ = invoke(capsys, tmp_path, *command)
    assert code == 0 and output.read_bytes() == original


def campaign_pnl_fixture(root, name, archive_state, *, split=False, collected=False):
    inbox = root / "backlog"
    archive = inbox / archive_state
    archive.mkdir(parents=True, exist_ok=True)
    spec = archive / f"{name}.json"
    arms = [quick_arm("candidate", "rank(close)")]
    record = quick_record("A1", 1.2)
    record["out_of_sample"]["test"]["startDate"] = "2021-01-01"
    pending = [] if collected else ["candidate"]
    if collected:
        record["pnl_evidence"] = {"complete": True}
        record["half_sharpes"] = [0.8, 1.1]
    records = [record]
    if split:
        arms.append(quick_arm("failed", "rank(open)"))
        records.append({"status": "ERROR", "submission_attempted": True})
    spec.write_text(json.dumps({"track": "ATOM", "design": "explore", "arms": arms}), encoding="utf-8")
    results = root / "backlog-results"
    results.mkdir(exist_ok=True)
    batch = results / f"{name}-batch.json"
    receipt = results / f"{name}-result.json"
    extra = {"quick_pnl": {"pending_labels": pending}} if pending else {}
    write_quick_batch(batch, arms, records, spec=str(inbox / spec.name), **extra)
    receipt.write_text(json.dumps({"ok": True, "command": "probe", "data": {
        "path": str(batch), "quick_pnl": {"pending_labels": ["stale receipt"]},
    }}), encoding="utf-8")
    return spec, batch, receipt


def test_campaign_status_discovers_pending_pnl_from_canonical_archive_batches(tmp_path, capsys, monkeypatch):
    from tools import _runner

    def unexpected_cli(*args, **kwargs):
        raise AssertionError("campaign status called a subprocess")

    monkeypatch.setattr(_runner, "run_cli", unexpected_cli)
    done, done_batch, _ = campaign_pnl_fixture(tmp_path, "done-research", "done")
    failed, failed_batch, _ = campaign_pnl_fixture(tmp_path, "mixed-research", "failed", split=True)
    campaign_pnl_fixture(tmp_path, "fulfilled", "done", collected=True)
    live, _, _ = campaign_pnl_fixture(tmp_path, "still-live", "")

    code, result = invoke(capsys, tmp_path, "campaign", "status")
    assert code == 0
    data = result["data"]
    assert [job["spec"] for job in data["jobs"]] == [str(live)]
    pending = data["archived_quick_pnl"]
    assert (pending["pending_batches"], pending["pending_arms"], pending["omitted"]) == (2, 2, 0)
    jobs = {job["spec"]: job for job in pending["jobs"]}
    assert set(jobs) == {str(done), str(failed)}
    for spec, batch, archive_state in ((done, done_batch, "done"), (failed, failed_batch, "failed")):
        job = jobs[str(spec)]
        assert (job["batch"], job["archive_state"]) == (str(batch), archive_state)
        assert job["quick_pnl"]["pending_labels"] == ["candidate"]

    code, result = invoke(capsys, tmp_path, "campaign", "status", "--limit", 1)
    assert code == 0
    pending = result["data"]["archived_quick_pnl"]
    assert pending["pending_batches"] == 2
    assert len(pending["jobs"]) == pending["omitted"] == 1


def test_campaign_recover_pnl_uses_only_probe_recovery_and_keeps_deferred_arms(tmp_path, capsys, monkeypatch):
    import cache_store
    from tools import _runner

    completed, completed_batch, receipt = campaign_pnl_fixture(tmp_path, "can-complete", "done")
    deferred, deferred_batch, _ = campaign_pnl_fixture(tmp_path, "still-deferred", "failed", split=True)
    before = receipt.read_bytes()
    archived = {spec: spec.read_bytes() for spec in (completed, deferred)}
    calls = []

    def run_cli(root, *args, **kwargs):
        assert root == tmp_path
        assert args[:2] == ("probe", "--recover")
        assert "--resume" not in args
        assert "--show-test" not in args
        batch = Path(args[2])
        calls.append(batch)
        if batch == deferred_batch:
            return _runner.CliResult(True, {"quick_pnl": {"pending_labels": ["candidate"]}})
        assert batch == completed_batch
        checkpoint = json.loads(batch.read_bytes())
        checkpoint["records"][0].update(pnl_evidence={"complete": True}, half_sharpes=[0.8, 1.1])
        checkpoint.pop("quick_pnl", None)
        batch.write_text(json.dumps(checkpoint), encoding="utf-8")
        return _runner.CliResult(True, {})

    monkeypatch.setattr(_runner, "run_cli", run_cli)
    with cache_store.file_lock(tmp_path / "backlog" / ".runner.lock", timeout=0):
        code, result = invoke(capsys, tmp_path, "campaign", "recover-pnl", "--limit", 2, "--max-wait-seconds", 8)
    assert code == 0
    assert set(calls) == {completed_batch, deferred_batch}
    jobs = {job["batch"]: job for job in result["data"]["jobs"]}
    assert set(jobs) == {str(completed_batch), str(deferred_batch)}
    for batch, state, remaining in ((completed_batch, "recovered", 0), (deferred_batch, "deferred", 1)):
        job = jobs[str(batch)]
        assert (job["state"], job["pending_before"], job["pending_after"]) == (state, 1, remaining)
    assert all(spec.read_bytes() == content for spec, content in archived.items())
    assert receipt.read_bytes() == before

    calls.clear()
    code, result = invoke(capsys, tmp_path, "campaign", "recover-pnl", "--limit", 1)
    assert code == 0
    assert calls == [deferred_batch]
    [job] = result["data"]["jobs"]
    assert job["state"] == "deferred"
    assert job["pending_after"] == 1


def test_campaign_prepares_limited_full_spec_from_train_only(tmp_path, capsys, monkeypatch):
    from research_app import experiments

    def reject_network(*args, **kwargs):
        raise AssertionError("offline preparation opened a session")

    monkeypatch.setattr(experiments, "with_session", reject_network)
    batch = tmp_path / "quick-batch.json"
    receipt = tmp_path / "quick-result.json"
    output = tmp_path / "full.spec.json"
    arms = [quick_arm("win", "rank(close)"), quick_arm("second", "rank(open)"),
            quick_arm("test-only", "rank(high)"), quick_arm("pending", "rank(low)"),
            quick_arm("diagnostic", "rank(volume)", role="diagnostic"),
            quick_arm("already-full", "rank(vwap)", settings={"simulationMode": "FULL"})]
    records = [quick_record("A1", 2.5, full=0.4), quick_record("A2", 2, full=5),
               quick_record("A3", full=9), quick_record("A4", 8, status="PENDING"),
               quick_record("A5", 7), quick_record("A6", 6)]
    write_quick_batch(batch, arms, records)
    receipt.write_text(json.dumps({"data": {"path": batch.name}}), encoding="utf-8")

    code, result = invoke(capsys, tmp_path, "campaign", "prepare-full", receipt, "--out", output, "--limit", 1)
    assert code == 0
    spec = json.loads(output.read_text(encoding="utf-8"))
    assert spec["track"] == "ATOM"
    assert spec["mechanism"] == "hypothesis A"
    assert spec["arms"] == [{**arms[0], "settings": {**arms[0]["settings"], "simulationMode": "FULL"}}]
    source = spec["quick_source"]
    assert source["batch_path"] == str(batch)
    assert source["run_id"] == "quick-run"
    assert source["request_fingerprint"] == experiments._request_fingerprint(arms)
    assert source["arms"] == [{"label": "win", "alpha_id": "A1", "train": {"sharpe": 2.5, "fitness": 1.25}}]
    assert experiments.probe_spec_identity(output)["quick_source"] == source
    assert result["data"]["omitted"] == {"limit": 1, "unfinished": 1, "diagnostic": 1,
                                          "not_quick": 1, "missing_train_metrics": 1}
    stamp = output.stat().st_mtime_ns
    code, _ = invoke(capsys, tmp_path, "campaign", "prepare-full", batch, "--out", output, "--limit", 1)
    assert code == 0
    assert output.stat().st_mtime_ns == stamp
    code, changed = invoke(capsys, tmp_path, "campaign", "prepare-full", batch, "--out", output, "--limit", 2)
    assert code != 0
    assert changed["error"]["code"] == "INVALID_INPUT"
    assert output.stat().st_mtime_ns == stamp


def test_campaign_full_draft_keeps_the_control_and_its_budget(tmp_path, capsys):
    batch = tmp_path / "quick-batch.json"
    output = tmp_path / "full.spec.json"
    arms = [quick_arm("baseline", "rank(close)", role="diagnostic", design="control"),
            quick_arm("rescue", "rank(open)", design="rescue", proxy_of="baseline"),
            quick_arm("add", "rank(close) + rank(volume)", design="add")]
    records = [quick_record("A1", -0.2), quick_record("A2", 2), quick_record("A3", 1)]
    design = {"mode": "controlled", "control": "baseline"}
    write_quick_batch(batch, arms, records, design=design)

    code, result = invoke(capsys, tmp_path, "campaign", "prepare-full", batch, "--out", output, "--limit", 1)
    assert code != 0
    assert result["error"]["code"] == "INVALID_INPUT"
    assert not output.exists()
    code, _ = invoke(capsys, tmp_path, "campaign", "prepare-full", batch, "--out", output, "--limit", 2)
    assert code == 0
    code, preview = invoke(capsys, tmp_path, "campaign", "preview", output)
    assert code == 0
    assert preview["data"]["design"]["control"] == "baseline"
    assert preview["data"]["design"]["designs"] == {"baseline": "control", "rescue": "rescue"}
    control, candidate = preview["data"]["results"]
    assert control["role"] == "diagnostic"
    assert candidate["proxy_of"] == "baseline"
    assert preview["data"]["quick_source"]["arms"][0]["train"]["sharpe"] == -0.2

    records[0]["status"] = "PENDING"
    write_quick_batch(batch, arms, records, design=design)
    code, rejected = invoke(capsys, tmp_path, "campaign", "prepare-full", batch, "--out", output, "--limit", 2)
    assert code != 0
    assert rejected["error"]["details"] == {"control": "baseline", "reason": "unfinished"}


def test_campaign_full_draft_embeds_captured_python_without_reopening_source(tmp_path, capsys):
    batch = tmp_path / "quick-batch.json"
    output = tmp_path / "full.spec.json"
    source = tmp_path / "original.py"
    arm = quick_arm("python", SOURCE, source_file=str(source),
                    settings={"simulation_mode": "quick", "language": "PYTHON", "lookback": 2, "testPeriod": "P1Y"})
    write_quick_batch(batch, [arm], [quick_record("A1", 1.2)], track="POWER_POOL")

    code, _ = invoke(capsys, tmp_path, "campaign", "prepare-full", batch, "--out", output, "--limit", 2)
    assert code == 0
    code, preview = invoke(capsys, tmp_path, "campaign", "preview", output)
    assert code == 0
    assert preview["data"]["track"] == "POWER_POOL"
    [result] = preview["data"]["results"]
    assert result["payload"]["regular"] == SOURCE
    assert result["payload"]["settings"]["simulationMode"] == "FULL"
    assert result["payload"]["settings"]["lookback"] == 2
    assert not source.exists()


def test_campaign_full_draft_protects_canonical_batch_and_receipt_alias(tmp_path, capsys):
    batch = tmp_path / "quick-batch.json"
    receipt = tmp_path / "quick-result.json"
    alias = tmp_path / "quick-alias.json"
    output = tmp_path / "full.spec.json"
    write_quick_batch(batch, [quick_arm("one", "rank(close)")], [quick_record("A1", 1.2)])
    receipt.write_text(json.dumps({"data": {"path": batch.name}}), encoding="utf-8")
    alias.write_text(json.dumps({"data": {"path": receipt.name}}), encoding="utf-8")
    original = batch.read_bytes()
    code, _ = invoke(capsys, tmp_path, "campaign", "prepare-full", receipt, "--out", batch, "--limit", 2)
    assert code != 0
    code, _ = invoke(capsys, tmp_path, "campaign", "prepare-full", receipt, "--out", output,
                     "--limit", 2, "--save", batch)
    assert code != 0
    assert batch.read_bytes() == original
    original_receipt = receipt.read_bytes()
    code, _ = invoke(capsys, tmp_path, "campaign", "prepare-full", alias, "--out", output,
                     "--limit", 2, "--save", receipt)
    assert code != 0
    assert receipt.read_bytes() == original_receipt
    assert not output.exists()
