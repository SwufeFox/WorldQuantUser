"""Exercise campaign routing and recovery through the public CLI."""

import json

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
