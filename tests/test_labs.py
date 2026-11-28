"""Offline tests for Labs ticket validation, scoped warnings, and evidence summaries."""

import json

import pytest

import workspace_db
from research_app import labs, state_machine
from research_app.runtime import CliError


def _ticket(**changes):
    return {
        "trigger_rule": "distribution", "region": "usa", "universe": "top3000", "delay": 1,
        "fields": ["close"], "questions": ["What fraction of values change daily?"],
        "expected_metrics": ["median daily update rate"], "branches": ["rate < 0.1 -> use update indicators"],
        **changes,
    }


def _spec(expr="rank(close)", **changes):
    return {"name": "labs-test", "idea": "Test field behavior", "design": "explore",
            "arms": [{"label": "lead", "expression": expr}], **changes}


@pytest.mark.parametrize("payload", [[], _ticket(questions="not a list"), _ticket(fields=["close", 4]),
                                     _ticket(branches=[{"action": "missing condition"}]), _ticket(delay=True),
                                     _ticket(region="USA\nOTHER"), _ticket(dataset={})])
def test_invalid_inputs_are_structured_and_have_no_side_effect(tmp_path, payload):
    with pytest.raises(CliError) as exc:
        labs.open_ticket(payload, research_dir=tmp_path)
    assert exc.value.code == "INVALID_INPUT"
    assert not workspace_db.db_path(tmp_path).exists()


def test_proposals_warn_on_open_tickets_in_the_same_scope(tmp_path):
    engine = state_machine.StateMachineEngine(tmp_path)
    ticket = labs.open_ticket(_ticket(), research_dir=tmp_path)
    assert engine.propose(_spec("rank(volume)"))["warnings"] == []
    assert engine.propose(_spec(settings={"region": "EUR", "universe": "TOP1200"}))["warnings"] == []
    proposed = engine.propose(_spec())
    assert proposed["state"] == "AUTO_PREFLIGHT"
    assert proposed["warnings"] == [{
        "code": "LABS_TICKET_OPEN", "ticket_id": ticket["ticket_id"], "scope": "USA/TOP3000/D1",
        "overlapping_fields": ["close"],
    }]
    assert engine.load_run_raw(proposed["run_id"])["warnings"] == proposed["warnings"]
    assert engine.inspect_run(proposed["run_id"])["warnings"] == proposed["warnings"]
    assert engine.inbox(limit=1)["open_labs_total"] == 1
    with pytest.raises(CliError):
        labs.resolve_ticket(ticket["ticket_id"], summary="Invalid numeric summary", metrics='{"rate": NaN}',
                            research_dir=tmp_path)
    assert labs.open_ticket_summary(tmp_path)["count"] == 1
    labs.resolve_ticket(ticket["ticket_id"], summary="Median rate was 0.03.", metrics={"median_rate": 0.03},
                        research_dir=tmp_path)
    assert engine.propose(_spec("ts_rank(close, 5)"))["warnings"] == []


def test_multiple_arms_and_repeated_proposals_warn_once_per_ticket_and_scope(tmp_path):
    engine = state_machine.StateMachineEngine(tmp_path)
    ticket = labs.open_ticket(_ticket(fields=["close", "volume"]), research_dir=tmp_path)
    expected = [{"code": "LABS_TICKET_OPEN", "ticket_id": ticket["ticket_id"],
                 "scope": "USA/TOP3000/D1", "overlapping_fields": ["close", "volume"]}]

    for day_count in (5, 10):
        arms = [{"label": field, "expression": f"ts_rank({field}, {day_count})"}
                for field in ("close", "volume", "close + volume")]
        proposed = engine.propose(_spec(arms=arms))
        assert proposed["state"] == "AUTO_PREFLIGHT"
        assert proposed["warnings"] == expected


def test_python_fields_and_effective_delay_are_reported(tmp_path):
    ticket = labs.open_ticket(_ticket(), research_dir=tmp_path)
    source = 'from brain.alphas import alpha as declare\n@declare(data=["close"])\ndef factor(data, settings):\n    return data.close[-1]\n'
    arms = [{"expression": source, "lookback": 2, "settings": {"language": "python", "delay": None}}]
    warnings = labs.open_ticket_warnings_for_arms(arms, research_dir=tmp_path)
    assert warnings == [{"code": "LABS_TICKET_OPEN", "ticket_id": ticket["ticket_id"],
                         "scope": "USA/TOP3000/D1", "overlapping_fields": ["close"]}]


def test_linked_ticket_restores_exact_gate_and_other_references_cannot_resume_it(tmp_path):
    engine = state_machine.StateMachineEngine(tmp_path)
    run_id = engine.propose(_spec())["run_id"]
    doc = engine.load_run_raw(run_id)
    doc["state"] = "GATE_APPROVE_FULL"
    engine.save_run(doc)
    params = {"trigger": "distribution", "question": ["How often does close update?"],
              "expect": ["median update fraction"], "branch": ["low -> run an event study"]}
    paused = engine.act(run_id, "request-labs", params)
    assert paused["state"] == "GATE_LABS"
    assert paused["test_view_open"] is False
    assert paused["pre_labs_state"] == "GATE_APPROVE_FULL"
    unrelated = labs.open_ticket(_ticket(source_run_id=run_id, fields=["volume"]), research_dir=tmp_path)
    unrelated_result = labs.resolve_ticket(unrelated["ticket_id"], summary="Volume updates daily.", research_dir=tmp_path)
    assert "run_transition" not in unrelated_result
    with pytest.raises(CliError):
        engine.act(run_id, "resume-from-labs")
    result = labs.resolve_ticket(paused["labs_ticket_id"], summary="Update fraction 0.94.", research_dir=tmp_path)
    assert result["run_transition"] == {"run_id": run_id, "state": "GATE_APPROVE_FULL"}
    assert engine.inspect_run(run_id)["test_view_open"] is False
    with pytest.raises(CliError):
        labs.resolve_ticket(paused["labs_ticket_id"], summary="Attempt to overwrite.", research_dir=tmp_path)
    assert engine.load_run_raw(run_id)["labs_resolution"]["resolution"]["summary"] == "Update fraction 0.94."


def test_ticket_insert_rolls_back_when_gate_write_fails(tmp_path, monkeypatch):
    engine = state_machine.StateMachineEngine(tmp_path)
    run_id = engine.propose(_spec())["run_id"]
    doc = engine.load_run_raw(run_id)
    doc["state"] = "GATE_TRIAGE"
    engine.save_run(doc)

    def fail(*args, **kwargs):
        raise RuntimeError("gate write failed")

    monkeypatch.setattr(workspace_db, "_write_run", fail)
    with pytest.raises(RuntimeError, match="gate write failed"):
        engine.act(run_id, "request-labs", {"question": ["What is the cadence?"], "expect": ["update rate"],
                                          "branch": ["slow -> inspect events"]})
    assert engine.load_run_raw(run_id)["state"] == "GATE_TRIAGE"
    assert labs.open_ticket_summary(tmp_path)["count"] == 0


def test_suggestions_require_recurring_train_or_check_evidence(tmp_path):
    entries = []
    for idx in range(4):
        for field in ("close", "volume", "open", "low"):
            entry = {
                "status": "COMPLETE", "settings": {"region": "USA", "universe": "TOP3000", "delay": 1},
                "expression": f"ts_rank({field}, {idx + 2})", "fields": [field],
                "metrics": {"turnover": 0.91},
                "out_of_sample": {"train": {"turnover": 0.9 if field == "close" and idx < 2 else 0.4},
                                  "test": {"turnover": 0.94}},
            }
            if field == "open" and idx < 2:
                entry["checks"] = [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"}]
            if field == "low":
                entry.update(execution_outcome="failed", checks={"CONCENTRATED_WEIGHT": "FAIL"})
            entries.append(entry)
    entries += [dict(entries[0], alpha_id="replay")]
    suggestions = labs.suggest_delegations(tmp_path, entries=entries)
    assert [(item["field"], item["attempts"], item["anomalies"]) for item in suggestions] == [
        ("close", 4, ["high_turnover(2)"]), ("open", 4, ["low_sub_universe(2)"]),
    ]


def test_snippet_emits_json_aggregates_for_sparse_matrices(tmp_path, capsys):
    pd = pytest.importorskip("pandas")
    ticket = labs.open_ticket(_ticket(fields=["close", "volume"]), research_dir=tmp_path)
    matrices = {"close": pd.DataFrame([[1, 2], [1, 3]], index=[1, 2]),
                "volume": pd.DataFrame([[float("inf"), 2], [None, 3]], index=[2, 3])}
    exec(ticket["notebook_snippet"], {"load_is_matrix": matrices.__getitem__})
    summary = json.loads(capsys.readouterr().out)
    assert summary["fields"]["close"]["daily_update_rate"] == 0.5
    assert summary["fields"]["volume"]["mean_coverage"] == 0.5
    assert summary["pairwise_rank_corr"] == {}


def test_suggest_delegations_reads_the_workspace_registry(tmp_path):
    """`labs suggest` used to crash on a missing alpha_registry constant."""
    from research_app import labs

    (tmp_path / "alpha-registry.jsonl").write_text("", encoding="utf-8")
    assert labs.suggest_delegations(str(tmp_path)) == []


def test_suggest_delegations_skips_entries_with_conflicting_settings():
    """One legacy entry with both spellings must not abort the whole report."""
    from research_app import labs

    conflicting = {"expression": "rank(close)", "status": "COMPLETE",
                   "settings": {"nanHandling": "ON", "nan_handling": "OFF",
                                "region": "HKG", "universe": "TOP500", "delay": 1}}
    assert labs.suggest_delegations("/nonexistent", entries=[conflicting]) == []
