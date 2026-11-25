"""Offline recovery receipts retain actionable commands and complete-evidence paths."""

from copy import deepcopy
import json

import pytest

import machine_lib as ml
from research_app import buffer_cmd, experiments, parser


def test_probe_actions_preserve_the_workspace_when_copied(monkeypatch, tmp_path):
    research_dir = tmp_path / "research with spaces"
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(research_dir))
    target = research_dir / "batches" / "saved experiment.json"
    settings = ml.resolve_settings({})
    document = {
        "arms": [{"label": label, "expression": expression, "settings": settings, "ops": 1}
                 for label, expression in (("queued", "rank(close)"), ("accepted", "rank(volume)"))],
        "records": [
            {"status": "QUEUED", "submission_attempted": False, "settings": settings},
            {"status": "PENDING", "submission_attempted": True, "settings": settings,
             "progress_url": "https://example.invalid/simulations/synthetic"},
        ],
    }
    experiments._probe_document_update(document, target)
    assert {action["action"] for action in document["next_actions"]} == {"recover", "resume"}

    # A later shell may have a different default; the copied command must keep its own workspace.
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path / "different workspace"))
    cli = parser.build_parser()
    for action in document["next_actions"]:
        parsed = cli.parse_args(action["argv"][2:])
        assert parsed.command == "probe"
        assert parsed.research_dir == str(research_dir)
        assert getattr(parsed, action["action"]) == str(target)
        assert action["batch_path"] == str(target)


def test_probe_summary_reports_omitted_items_and_existing_complete_evidence(tmp_path):
    checkpoint = tmp_path / "batch.json"
    response = {
        "path": str(checkpoint),
        "rows": [{"label": f"arm{i}", "status": "QUEUED"} for i in range(25)],
        "preflight_deferred": {
            f"scope{i}": {"code": "SERVICE_ERROR", "message": "catalog pending",
                           "hint": "recover later", "scope": {"region": "USA"}, "retry_at": None,
                           "evidence": {"required_field_ids": [f"field{i}"]}}
            for i in range(25)
        },
    }
    complete = deepcopy(response)
    checkpoint.write_text(json.dumps(complete), encoding="utf-8")
    experiments._summarize_probe_response(response)

    assert response["truncated"] is response["summary"]["truncated"] is True
    assert len(response["rows"]) == len(response["preflight_deferred"]) == 20
    assert response["rows_total"] == response["preflight_deferred_total"] == 25
    assert response["rows_omitted"] == response["preflight_deferred_omitted"] == 5
    assert response["rows_truncated"] is response["preflight_deferred_truncated"] is True
    assert response["summary"]["full_evidence"] == {
        "available": True, "source": "checkpoint", "path": str(checkpoint)}
    assert json.loads(checkpoint.read_text(encoding="utf-8")) == complete


@pytest.mark.parametrize("mode", ({"dry_run": True}, {"check": True}, {}))
def test_probe_summary_does_not_invent_a_checkpoint(tmp_path, mode):
    checkpoint = tmp_path / "batch.json"
    if mode:
        # A preview can name an existing batch without having saved its current evidence there.
        checkpoint.write_text("{}", encoding="utf-8")
    response = {"path": str(checkpoint), "rows": [], **mode}
    experiments._summarize_probe_response(response)
    assert response["summary"]["full_evidence"] == {
        "available": False, "source": None, "path": None}
    assert response["truncated"] is False


def test_probe_summary_flags_compacted_evidence_even_when_all_rows_fit():
    response = {"rows": [{"label": "one"}], "preflight": [{
        "ok": True, "checked_requests": [{"expression": "rank(close)"}], "results": []}]}
    experiments._summarize_probe_response(response)
    assert response["rows_omitted"] == response["preflight_omitted"] == 0
    assert response["truncated"] is True
    assert response["preflight"][0]["truncated"] is True
    assert response["preflight"][0]["checked_requests_omitted"] == 1


@pytest.mark.parametrize("description, actual", ((None, 0), ("  \n  ", 0),
                                                (" " + "x" * 99 + "\n", 99), ("字" * 99, 99)))
def test_seed_rationale_reports_the_trimmed_character_count(description, actual):
    problems = buffer_cmd.validate_seed("Seed", "SENTIMENT", ["signal"], description)
    assert len(problems) == 1
    assert "description" in problems[0]
    assert f"actual={actual} characters" in problems[0]
    assert "minimum=100" in problems[0]


@pytest.mark.parametrize("description", ("x" * 100, " \n" + "字" * 100 + " \n"))
def test_seed_rationale_accepts_the_existing_hundred_character_boundary(description):
    assert buffer_cmd.validate_seed("Seed", "SENTIMENT", ["signal"], description) == []
