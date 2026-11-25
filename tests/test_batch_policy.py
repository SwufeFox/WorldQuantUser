"""Offline evidence and AST boundaries for new bulk-exploration admission."""

from copy import deepcopy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import machine_lib as ml
from tools.pipeline import batch_policy as policy


def arm(expression="rank(close)", **extra):
    return {"label": "candidate", "expr": expression, "settings": {},
            "idea": "Test whether this measured input persists.", **extra}


def field(name, dataset=None, kind="MATRIX"):
    row = {"id": name, "type": kind}
    if dataset is not None:
        row["dataset"] = dataset
    return row


def codes(report):
    return {item["code"] for item in report["blocking"]}


@pytest.mark.parametrize("idea", [None, "", " \t\n ", 42, False, {"reason": "not a string"}])
def test_idea_must_be_a_nonempty_string(idea):
    report = policy.analyze_arms([arm(idea=idea)])
    assert not report["ok"]
    assert "idea_required" in codes(report)
    assert report["arms"][0]["skeleton"] == "rank(F)"


def test_missing_idea_and_missing_arm_are_structured_blockers():
    candidate = arm()
    del candidate["idea"]
    report = policy.analyze_arms([candidate, None])
    assert codes(report) == {"idea_required", "invalid_arm"}
    assert [item["index"] for item in report["arms"]] == [0, 1]
    json.dumps(report, allow_nan=False)


def test_scope_uses_machine_lib_defaults_and_key_normalization(monkeypatch):
    monkeypatch.setitem(ml.DEFAULT_SETTINGS, "delay", 2)
    report = policy.analyze_arms([arm(settings={"instrument_type": "EQUITY", "Region": "EUR"})])
    assert report["ok"]
    expected = ml.resolve_settings({"instrument_type": "EQUITY", "Region": "EUR"})
    assert report["arms"][0]["scope"] == {name: expected[name]
                                            for name in ("region", "universe", "delay", "instrumentType")}


@pytest.mark.parametrize("settings", [[], {"delay": -1}, {"region": "USA", "Region": "EUR"}])
def test_bad_settings_do_not_throw_or_hide_other_arms(settings):
    report = policy.analyze_arms([arm(settings=settings), arm("rank(volume)")])
    assert not report["ok"]
    assert "invalid_settings" in codes(report)
    assert report["arms"][0]["scope"] is None
    assert report["arms"][1]["skeleton"] == "rank(F)"


@pytest.mark.parametrize("count,allowed", [(8, True), (9, False)])
def test_skeleton_cap_counts_whole_input_despite_fields_windows_locals_and_scopes(count, allowed):
    candidates = []
    for index in range(count):
        expression = "rank(ts_mean(field_%d, %d))" % (index, index + 1)
        if index % 3 == 1:
            expression = "local_%d = ts_mean(field_%d, %d); rank(local_%d)" % (index, index, index + 1, index)
        elif index % 3 == 2:
            expression = "unused = rank(unused_%d); " % index + expression
        candidates.append(arm(expression, label="arm-%d" % index,
                              settings={"region": "USA" if index % 2 else "EUR"}))
    report = policy.analyze_arms(candidates)
    assert report["ok"] is allowed
    assert len(report["skeleton_groups"]) == 1
    assert report["skeleton_groups"][0] == {
        "skeleton": "rank(ts_mean(F,L))", "count": count,
        "indices": list(range(count)), "labels": ["arm-%d" % index for index in range(count)],
    }
    assert ("skeleton_limit_exceeded" in codes(report)) is not allowed
    assert [item["fields"] for item in report["arms"]] == [["field_%d" % index] for index in range(count)]


def test_repeated_identical_rows_count_as_separate_arms():
    report = policy.analyze_arms([arm()] * 9)
    assert report["skeleton_groups"][0]["count"] == 9
    assert "skeleton_limit_exceeded" in codes(report)


def test_only_final_inlined_signal_fields_and_skeleton_are_used():
    expression = "@note(unrelated_field); unused=rank(fake_field); x=ts_mean(close,5); y=rank(x); y"
    report = policy.analyze_arms([arm(expression), arm("rank(ts_mean(close,5))")])
    assert report["ok"]
    assert [entry["fields"] for entry in report["arms"]] == [["close"], ["close"]]
    assert report["skeleton_groups"][0]["count"] == 2


def test_nested_ast_forms_and_named_literals_preserve_real_fields():
    source = "x=close-low; y=-x; y>0 ? group_rank(y,industry,constant=true) : ts_mean(volume,5)"
    report = policy.analyze_arms([arm(source)])
    assert report["ok"]
    assert set(report["arms"][0]["fields"]) == {"close", "low", "industry", "volume"}
    assert "x" not in report["arms"][0]["skeleton"]


@pytest.mark.parametrize("expression,reason", [
    ("x=close; x=volume; rank(x)", "duplicate_assignment"),
    ("x=x+close; rank(x)", "cyclic_assignment"),
    ("x=y; y=x; rank(x)", "cyclic_assignment"),
    ("x=y; y=close; rank(x)", "forward_reference"),
    ("rank(close); x=volume", "assignment_after_output"),
    ("rank(close); rank(volume)", "multiple_outputs"),
    ("true=close; rank(true)", "literal_assignment"),
    ("x=close", "missing_signal"),
    ("rank(close", "parse_error"),
])
def test_ambiguous_or_unsupported_programs_are_explicit(expression, reason):
    report = policy.analyze_arms([arm(expression)])
    assert not report["ok"]
    assert report["arms"][0]["skeleton"] is None
    assert report["blocking"][0]["reason"] == reason


def test_unused_but_ambiguous_local_assignments_are_rejected():
    report = policy.analyze_arms([arm("x=y; y=x; rank(close)")])
    assert report["blocking"][0]["reason"] == "cyclic_assignment"


def test_expansion_is_bounded_without_claiming_a_skeleton():
    source = "x0=close; " + "; ".join("x%d=x%d+x%d" % (i, i-1, i-1) for i in range(1, 15)) + "; rank(x14)"
    report = policy.analyze_arms([arm(source)])
    assert report["blocking"][0]["reason"] == "expansion_limit"
    assert report["arms"][0]["skeleton"] is None


@pytest.mark.parametrize("language", ["PYTHON", "python"])
def test_python_is_explicitly_unsupported_even_if_source_looks_like_fastexpr(language):
    report = policy.analyze_arms([arm(settings={"language": language})])
    assert report["arms"][0]["fields"] == []
    assert report["blocking"][0]["reason"] == "python"


def test_expression_alias_is_supported_but_conflicts_are_not_guessed():
    candidate = arm()
    candidate["expression"] = candidate.pop("expr")
    assert policy.analyze_arms([candidate])["ok"]
    candidate["expr"] = "rank(volume)"
    report = policy.analyze_arms([candidate])
    assert report["blocking"][0]["reason"] == "conflicting_expression_aliases"


def test_three_verified_datasets_pass_and_report_is_json_safe_without_mutation():
    candidates = [arm("rank(a)"), arm("rank(b)"), arm("rank(c)")]
    analysis = policy.analyze_arms(candidates)
    before = deepcopy(analysis)
    records = [field("a", {"id": "d1"}), field("b", "d2", "VECTOR"), field("c", "d3")]
    report = policy.assess_datasets(analysis, [records] * 3)
    assert report["ok"]
    assert report["diversity"]["verified_datasets"] == ["d1", "d2", "d3"]
    assert report["diversity"]["state"] == "verified"
    assert analysis == before
    json.dumps(report, allow_nan=False)


def test_declared_family_dataset_and_metadata_for_unused_fields_are_not_evidence():
    candidates = [arm("unused=fake_%d; rank(close)" % i, family="family-%d" % i, dataset="dataset-%d" % i)
                  for i in range(3)]
    records = [field("close", "pv1"), *(field("fake_%d" % i, "d%d" % i) for i in range(3))]
    report = policy.assess_datasets(policy.analyze_arms(candidates), [records] * 3)
    assert not report["ok"]
    assert report["diversity"]["state"] == "insufficient"
    assert report["diversity"]["verified_datasets"] == ["pv1"]
    assert "dataset_diversity_insufficient" in codes(report)


def test_group_and_universe_do_not_inflate_dataset_count():
    records = [field("a", "d1"), field("b", "d2"), field("industry", "essential", "GROUP"),
               field("my_universe", kind="UNIVERSE")]
    report = policy.assess_datasets(policy.analyze_arms([arm("group_rank(a+b,industry)+my_universe")]), [records])
    assert report["diversity"]["verified_count"] == 2
    assert report["diversity"]["state"] == "insufficient"
    assert report["diversity"]["excluded_fields"] == ["industry", "my_universe"]
    assert report["diversity"]["unknown_fields"] == []


@pytest.mark.parametrize("records,reason,ambiguous", [
    (None, "no_verified_record", False),
    ([], "no_verified_record", False),
    ([{"id": "a", "dataset": "d1"}], "missing_field_type", False),
    ([field("a", "d1", "FUTURE_TYPE")], "unrecognized_field_type", False),
    ([field("a")], "missing_dataset_id", False),
    ([field("a", "d1"), field("a", "d2")], "conflicting_dataset_ids", True),
    ([field("a", "d1"), field("a", "d1", "GROUP")], "conflicting_field_types", True),
    ([field("a", "d1"), field("a", "d1", "VECTOR")], "conflicting_field_types", True),
])
def test_missing_or_conflicting_field_evidence_stays_unknown(records, reason, ambiguous):
    report = policy.assess_datasets(policy.analyze_arms([arm("rank(a)")]), [records])
    assert report["diversity"]["state"] == "unknown"
    assert report["diversity"]["verified_count"] == 0
    assert report["diversity"]["unknown_fields"] == ["a"]
    assert report["diversity"]["ambiguous_fields"] == (["a"] if ambiguous else [])
    assert report["diversity"]["by_arm"][0]["fields"][0]["reason"] == reason
    assert "dataset_diversity_unverified" in codes(report)


def test_duplicate_consistent_rows_are_one_dataset():
    records = [field("a", {"id": "d1"}), field("a", "d1"), field("b", "d2"), field("c", "d3")]
    report = policy.assess_datasets(policy.analyze_arms([arm("a+b+c")]), [records])
    assert report["ok"]
    assert report["diversity"]["verified_count"] == 3


def test_unknown_extra_field_does_not_disprove_three_verified_datasets():
    records = [field("a", "d1"), field("b", "d2"), field("c", "d3")]
    report = policy.assess_datasets(policy.analyze_arms([arm("a+b+c+unknown")]), [records])
    assert report["ok"]
    assert report["diversity"]["state"] == "verified"
    assert report["diversity"]["unknown_fields"] == ["unknown"]


def test_conflicting_record_across_arms_in_same_scope_cannot_create_extra_dataset():
    analysis = policy.analyze_arms([arm("a"), arm("a"), arm("b")])
    report = policy.assess_datasets(analysis, [[field("a", "d1")], [field("a", "d2")], [field("b", "d3")]])
    assert report["diversity"]["verified_datasets"] == ["d3"]
    assert report["diversity"]["ambiguous_fields"] == ["a"]
    assert report["diversity"]["state"] == "unknown"


def test_same_field_in_another_scope_cannot_fill_missing_evidence():
    candidates = [arm("a+b+c", settings={"region": "USA"}), arm("a", settings={"region": "EUR"})]
    report = policy.assess_datasets(policy.analyze_arms(candidates),
                                    [[field("b", "d2"), field("c", "d3")], [field("a", "d1")]])
    assert report["ok"]  # Three actual datasets across the complete batch.
    assert report["diversity"]["by_arm"][0]["unknown_fields"] == ["a"]
    assert report["diversity"]["by_arm"][1]["unknown_fields"] == []


def test_three_datasets_do_not_clear_an_idea_or_skeleton_blocker():
    candidates = [arm("rank(a+b+c)", idea="")] * 9
    records = [field("a", "d1"), field("b", "d2"), field("c", "d3")]
    report = policy.assess_datasets(policy.analyze_arms(candidates), [records] * len(candidates))
    assert report["diversity"]["state"] == "verified"
    assert not report["ok"]
    assert {"idea_required", "skeleton_limit_exceeded"} <= codes(report)


def test_reassessment_replaces_only_prior_diversity_blockers():
    analysis = policy.analyze_arms([arm("a+b+c")])
    unknown = policy.assess_datasets(analysis, [None])
    verified = policy.assess_datasets(unknown, [[field("a", "d1"), field("b", "d2"), field("c", "d3")]])
    assert verified["ok"]
    assert unknown["diversity"]["state"] == "unknown"


def test_metadata_alignment_errors_are_explicit_caller_errors():
    analysis = policy.analyze_arms([arm()])
    with pytest.raises(ValueError, match="one-to-one"):
        policy.assess_datasets(analysis, [])
    with pytest.raises(ValueError, match="list or None"):
        policy.assess_datasets(analysis, [{"close": "pv1"}])
