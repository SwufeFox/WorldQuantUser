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


def _official_description(total=100):
    lead = "Idea:\n- "
    tail = ("\n\nRationale for data used:\n- x"
            "\n\nRationale for operators used:\n- y")
    return lead + "x" * (total - len(lead) - len(tail)) + tail


@pytest.mark.parametrize("description, actual", ((None, 0), ("  \n  ", 0),
                                                (" " + "x" * 99 + "\n", 99), ("字" * 99, 99)))
def test_seed_rationale_reports_the_trimmed_character_count(description, actual):
    problems = buffer_cmd.validate_seed("Seed", "SENTIMENT", ["signal"], description)
    assert len(problems) == 1
    assert "description" in problems[0]
    assert f"actual={actual} characters" in problems[0]
    assert "minimum=100" in problems[0]


@pytest.mark.parametrize("description", (_official_description(100),
                                        " \n" + _official_description(100) + " \n"))
def test_seed_rationale_accepts_the_existing_hundred_character_boundary(description):
    assert buffer_cmd.validate_seed("Seed", "SENTIMENT", ["signal"], description) == []


def test_seed_rationale_rejects_plain_prose_even_when_long_enough():
    problems = buffer_cmd.validate_seed("Seed", "SENTIMENT", ["signal"], "x" * 100)
    assert len(problems) == 1
    assert "official Power Pool template" in problems[0]


def _invoke_cli_in_process(monkeypatch, tmp_path, argv):
    from io import StringIO
    from research_app import app

    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path / "research"))
    buf = StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    code = app.main(argv)
    return code, json.loads(buf.getvalue())


def test_bare_cli_invocation_returns_top_level_help(monkeypatch, tmp_path):
    code, payload = _invoke_cli_in_process(monkeypatch, tmp_path, [])
    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "help"
    assert "commands" in payload["data"]
    assert "campaign" in payload["data"]["commands"]
    assert "schema" in payload["data"]["commands"]


@pytest.mark.parametrize("argv", (
    ["pairs"],
    ["legs"],
    ["benchmark"],
    ["marginal", "--target", "close"],
    ["independence", "--target", "close", "--against", "open"],
    ["prune", "--set", "unknown_key=1"],
))
def test_validation_commands_emit_structured_errors_with_hints(monkeypatch, tmp_path, argv):
    code, payload = _invoke_cli_in_process(monkeypatch, tmp_path, argv)
    assert code == 2
    assert payload["ok"] is False
    error = payload["error"]
    assert error["code"] in {"INVALID_INPUT", "USAGE"}
    assert error["hint"]
    assert isinstance(error["hint"], str) and error["hint"].strip()


def test_every_cli_error_and_actionable_brain_error_provides_a_hint():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    missing = []
    for path in sorted((root / "research_app").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "CliError":
                if len(node.args) < 3 and not any(kw.arg == "hint" for kw in node.keywords):
                    missing.append((path.name, node.lineno))
    assert missing == []

    for err in (
        ml.MissingCredentials("missing"),
        ml.AuthError("rejected"),
        ml.RateLimited("slow down"),
        ml.SimulationError("failed"),
        ml.ReadInterrupted("interrupted"),
        ml.AuthenticationUnavailable("unavailable"),
        ml.AuthenticationDeadline("deadline"),
        ml.PermissionDenied("denied"),
    ):
        assert err.hint and err.hint.strip()
    assert ml.NotFound("unknown").hint is None
    assert ml.ServiceError("bad gateway").hint is None


def test_workspace_skill_and_docs_links_resolve():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    skill = root / ".agents" / "skills" / "worldquant-research-cli" / "SKILL.md"
    assert skill.is_file()
    skill_text = skill.read_text(encoding="utf-8")
    assert skill_text.startswith("---\nname: worldquant-research-cli\ndescription:")
    assert "docs/errors.md" in skill_text

    doc_files = [skill, root / "README.md", root / "AGENTS.md", *sorted((root / "docs").glob("*.md"))]
    link_re = re.compile(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]+)?\)")
    broken = []
    for doc in doc_files:
        text = doc.read_text(encoding="utf-8")
        for target in link_re.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (doc.parent / target).resolve()
            if not resolved.exists():
                broken.append((str(doc.relative_to(root)), target))
    assert broken == []


def test_layered_architecture_has_no_reverse_imports():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent

    def imported_roots(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.append((node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.append((node.lineno, node.module))
        return found

    core_violations = []
    for path in sorted(root.glob("*.py")) + sorted((root / "brain_core").glob("*.py")):
        if path.name == "research_cli.py":
            continue
        for lineno, target in imported_roots(path):
            top = target.split(".")[0]
            if top in {"research_app", "tools"}:
                core_violations.append((str(path.relative_to(root)), lineno, target))
    assert core_violations == []

    app_violations = []
    for path in sorted((root / "research_app").glob("*.py")):
        for lineno, target in imported_roots(path):
            if target.split(".")[0] == "tools":
                app_violations.append((str(path.relative_to(root)), lineno, target))
    assert app_violations == []


def test_compatibility_tool_aliases_resolve_to_canonical_modules():
    import batch_policy
    import spc_experiment
    from research_app import campaign_runner, guarded_sim, queue_runner
    from tools import _runner, campaign_runner as legacy_campaign_runner, guarded_sim as legacy_guarded_sim
    from tools import spc_experiment as legacy_spc_experiment
    from tools.pipeline import batch_policy as legacy_batch_policy

    assert _runner is queue_runner
    assert legacy_campaign_runner is campaign_runner
    assert legacy_guarded_sim is guarded_sim
    assert legacy_spc_experiment is spc_experiment
    assert legacy_batch_policy is batch_policy
