#!/usr/bin/env python3
"""Offline tests for field contracts, expression lint and the alpha registry.

Everything here runs without a BRAIN account. The sample catalog records are
trimmed copies of real ``GET /data-fields`` and ``GET /operators`` responses, so
the tests exercise the shapes the modules actually meet.

Run with::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import alpha_registry  # noqa: E402
import expression_lint  # noqa: E402
import field_contract  # noqa: E402
import machine_lib  # noqa: E402
import research_paths  # noqa: E402


def datafield(field_id, description, dataset="fundamental6", delay=1,
              coverage=0.5, date_coverage=1.0, region="USA", type_="MATRIX"):
    return {
        "id": field_id,
        "description": description,
        "dataset": {"id": dataset},
        "category": {"id": "fundamental"},
        "subcategory": {"id": "fundamental-data"},
        "region": region,
        "delay": delay,
        "type": type_,
        "coverage": coverage,
        "dateCoverage": date_coverage,
    }


# A small slice of the real operator catalog.
OPERATORS = [
    {"name": "rank", "category": "Cross Sectional", "definition": "rank(x, rate=2)"},
    {"name": "ts_delta", "category": "Time Series", "definition": "ts_delta(x, d)"},
    {"name": "ts_backfill", "category": "Time Series",
     "definition": "ts_backfill(x, lookback = d, k = 1, ignore = \"NAN\")"},
    {"name": "add", "category": "Arithmetic", "definition": "add(x, y, filter = false)"},
    {"name": "subtract", "category": "Arithmetic", "definition": "subtract(x, y, filter = false)"},
    {"name": "vec_sum", "category": "Vector", "definition": "vec_sum(x)"},
    {"name": "bucket", "category": "Group",
     "definition": 'bucket(rank(x), range="0, 1, 0.1", skipBoth = false, NaNGroup = false)'},
    # ``max`` spells its variadic marker with TWO dots, not three. A lint that
    # only recognised ``...`` counted the marker as a required parameter and
    # rejected the legal two-argument call ``max(a, b)``.
    {"name": "max", "category": "Arithmetic", "definition": "max(x, y, ..)"},
]


class FieldContractTests(unittest.TestCase):
    def test_period_inferred_from_family_prefix(self):
        record = datafield("fnd6_newa1v1300_capx", "Capital Expenditures")
        contract = field_contract.audit_field(record)
        period = contract["attributes"]["period"]
        self.assertEqual(period["state"], "inferred")
        self.assertEqual(period["value"], "annual")
        self.assertEqual(period["rule"], "id_family_newa")

    def test_period_handles_version_segment(self):
        # ``newqv1300`` must not need the trailing-letter fallback.
        record = datafield("fnd6_newqv1300_cshoq", "Common Shares Outstanding - Quarter")
        contract = field_contract.audit_field(record)
        self.assertEqual(contract["attributes"]["period"]["value"], "quarterly")
        self.assertEqual(contract["attributes"]["period"]["rule"], "id_family_newq")

    def test_period_inferred_from_analyst4_frequency_token(self):
        # ``analyst4`` writes the frequency inline (``af`` annual, ``qf``
        # quarterly) instead of using a family prefix. Missing this rule made
        # every analyst4 field audit as "period unknown", which blocked the
        # cash-flow-forecast coverage ratio for the wrong reason.
        annual = datafield(
            "anl4_fs_detail_estimates_basic_af_v4_nd_cfps_number",
            "Cash Flow Per Share - number of estimates for the annual period",
        )
        contract = field_contract.audit_field(annual)
        period = contract["attributes"]["period"]
        self.assertEqual(period["state"], "inferred")
        self.assertEqual(period["value"], "annual")

        quarterly = datafield(
            "anl4_fs_detail_estimate_1qf_v4_nd_cfo_mean",
            "Cash Flow From Operations - mean estimate for the quarterly period",
        )
        contract = field_contract.audit_field(quarterly)
        self.assertEqual(contract["attributes"]["period"]["value"], "quarterly")

    def test_analyst4_frequency_token_is_not_applied_to_other_datasets(self):
        # A stray ``_af_`` outside analyst4 must not silently become "annual".
        contract = field_contract.audit_field(datafield("pv1_af_demo", "demo"))
        self.assertNotEqual(contract["attributes"]["period"].get("rule"), "id_anl4_af")

    def test_unit_and_currency_inferred(self):
        record = datafield("fnd6_newa1v1300_at", "Total Assets")
        contract = field_contract.audit_field(record)
        self.assertEqual(contract["attributes"]["unit"]["value"], "currency_amount")
        self.assertEqual(contract["attributes"]["currency"]["value"], "USD")

    def test_declared_versus_inferred_is_distinguishable(self):
        record = datafield("fnd6_newa1v1300_at", "Total Assets", delay=1)
        contract = field_contract.audit_field(record)
        # delay comes straight from the catalog...
        self.assertEqual(contract["delay"]["state"], "declared")
        # ...while period and unit are inferences, and are labelled as such.
        self.assertEqual(contract["attributes"]["period"]["state"], "inferred")

    def test_missing_description_blocks_unit(self):
        record = datafield("mystery_field", "")
        contract = field_contract.audit_field(record)
        self.assertEqual(contract["attributes"]["unit"]["state"], "unknown")
        self.assertIn("unit", contract["blocking_unknowns"])
        self.assertEqual(contract["verdict"], "blocked")

    def test_share_count_differencing_is_blocked(self):
        """Idea 47's blocker, reproduced by rule instead of prose."""
        records = [
            datafield("fnd6_newqv1300_cshopq", "Total Shares Repurchased - Quarter"),
            datafield("fnd6_newqv1300_cshoq", "Common Shares Outstanding"),
        ]
        document = field_contract.audit_fields(records, operation="ts_difference")
        self.assertEqual(document["verdict"], "blocked")
        self.assertIn("corporate_action_adjustment",
                      document["operation"]["blocking_unknowns"])

    def test_cross_field_ratio_keeps_fiscal_alignment_warning(self):
        """Idea 50's caveat, reproduced by rule instead of prose."""
        records = [
            datafield("fnd6_newa1v1300_capx", "Capital Expenditures"),
            datafield("fnd6_newa2v1300_oancf", "Operating Cash Flow"),
            datafield("fnd6_newa1v1300_at", "Total Assets"),
        ]
        document = field_contract.audit_fields(records, operation="ratio")
        self.assertEqual(document["verdict"], "needs_review")
        self.assertIn("fiscal_period_alignment", document["operation"]["warnings"])
        self.assertIn("sign_convention", document["operation"]["warnings"])

    def test_sign_convention_blocks_difference(self):
        records = [
            datafield("fnd6_newa1v1300_capx", "Capital Expenditures"),
            datafield("fnd6_newa2v1300_oancf", "Operating Cash Flow"),
        ]
        document = field_contract.audit_fields(records, operation="difference")
        self.assertEqual(document["verdict"], "blocked")
        self.assertIn("sign_convention", document["operation"]["blocking_unknowns"])

    def test_mixed_periods_block_combination(self):
        records = [
            datafield("fnd6_newqv1300_saleq", "Sales - Quarter"),
            datafield("fnd6_newa2v1300_sale", "Sales - Annual"),
        ]
        document = field_contract.audit_fields(records, operation="ratio")
        self.assertEqual(document["verdict"], "blocked")
        self.assertIn("period", document["operation"]["blocking_unknowns"])

    def test_mixed_units_block_combination(self):
        records = [
            datafield("fnd6_newqv1300_cshoq", "Common Shares Outstanding"),
            datafield("fnd6_newqv1300_atq", "Total Assets"),
        ]
        document = field_contract.audit_fields(records, operation="ratio")
        self.assertEqual(document["verdict"], "blocked")

    def test_same_field_ratio_is_ready(self):
        records = [datafield("fnd6_newa1v1300_at", "Total Assets")]
        document = field_contract.audit_fields(records, operation="ts_difference")
        self.assertEqual(document["verdict"], "ready")

    def test_unknown_operation_rejected(self):
        records = [datafield("fnd6_newa1v1300_at", "Total Assets")]
        with self.assertRaises(ValueError):
            field_contract.audit_fields(records, operation="multiply")


class ExpressionLintTests(unittest.TestCase):
    def setUp(self):
        self.index = expression_lint.build_operator_index(OPERATORS)

    def test_extract_calls_counts_nested_arguments(self):
        calls = expression_lint.extract_calls("rank(ts_delta(close, 5))")
        names = [call["name"] for call in calls]
        self.assertEqual(names, ["rank", "ts_delta"])
        self.assertEqual(calls[0]["arg_count"], 1)
        self.assertEqual(calls[1]["arg_count"], 2)

    def test_named_arguments_are_not_calls(self):
        calls = expression_lint.extract_calls('ts_backfill(close, 20, k=2, ignore="NAN")')
        self.assertEqual([call["name"] for call in calls], ["ts_backfill"])
        self.assertEqual(calls[0]["arg_count"], 4)

    def test_reserved_identifier_is_reported(self):
        # The simulator rejects ``eps=...`` with "Variable eps is reserved and
        # cannot be assigned". Nothing in the catalog documents this, so the
        # offline lint carries the list; missing it burned a simulation round.
        report = expression_lint.lint_expression("eps=ts_backfill(close,20);rank(eps)")
        codes = [error["code"] for error in report["errors"]]
        self.assertIn("reserved_identifier", codes)
        offending = [e for e in report["errors"]
                     if e["code"] == "reserved_identifier"]
        self.assertEqual(offending[0]["identifier"], "eps")

    def test_reserved_check_does_not_fire_on_comparisons(self):
        report = expression_lint.lint_expression(
            "up = close > open; dn = up != 0; rank(up)")
        self.assertEqual(
            [e["code"] for e in report["errors"] if e["code"] == "reserved_identifier"],
            [])

    def test_commas_inside_strings_are_ignored(self):
        calls = expression_lint.extract_calls(
            'group_neutralize(x, bucket(rank(cap), range="0.1,1,0.1"))')
        self.assertEqual([call["name"] for call in calls],
                         ["group_neutralize", "bucket", "rank"])

    def test_keyword_argument_does_not_trigger_arity_warning(self):
        # bucket's documented signature lists optional parameters, so passing
        # range= as a keyword legitimately leaves only two positional
        # arguments. Flagging that as a missing argument is a false positive.
        report = expression_lint.lint_expression(
            'group_neutralize(rank(cap), bucket(rank(cap), range="0.1,1,0.1"))',
            operator_index=self.index)
        codes = [w["code"] for w in report["warnings"]]
        self.assertNotIn("arity_suspicious", codes)

    def test_positional_arity_warning_still_fires(self):
        report = expression_lint.lint_expression(
            "rank(ts_delta(close))",
            operator_index=self.index)
        codes = [w["code"] for w in report["warnings"]]
        self.assertIn("arity_suspicious", codes)

    def test_two_dot_variadic_marker_is_not_a_parameter(self):
        # ``max(x, y, ..)`` documents "at least 2 inputs". Counting the ``..``
        # marker as a required parameter made the legal call max(a, b) warn.
        report = expression_lint.lint_expression(
            "rank(max(close, open))", operator_index=self.index)
        codes = [w["code"] for w in report["warnings"]]
        self.assertNotIn("arity_suspicious", codes)
        self.assertEqual(self.index["max"]["min_arity"], 2)
        self.assertIsNone(self.index["max"]["max_arity"])

    def test_three_dot_variadic_marker_also_accepted(self):
        index = expression_lint.build_operator_index(
            [{"name": "max", "definition": "max(x, y, ...)"}])
        self.assertEqual(index["max"]["min_arity"], 2)
        self.assertIsNone(index["max"]["max_arity"])

    def test_signature_parsing_marks_optional_parameters(self):
        index = expression_lint.build_operator_index(
            [{"name": "ts_delta", "definition": "ts_delta(x, d)"},
             {"name": "rank", "definition": "rank(x, rate=2)"}])
        self.assertEqual(index["ts_delta"]["min_arity"], 2)
        self.assertEqual(index["ts_delta"]["max_arity"], 2)
        self.assertEqual(index["rank"]["min_arity"], 1)
        self.assertEqual(index["rank"]["max_arity"], 2)

    def test_unknown_operator_is_an_error(self):
        """`regression_neut` has a docs page but is not a real operator."""
        report = expression_lint.lint_expression(
            "rank(regression_neut(close, volume))", operator_index=self.index)
        self.assertFalse(report["ok"])
        self.assertEqual([error["code"] for error in report["errors"]],
                         ["unknown_operator"])
        self.assertEqual(report["errors"][0]["operator"], "regression_neut")

    def test_valid_expression_passes(self):
        report = expression_lint.lint_expression(
            "rank(ts_delta(close, 5))", operator_index=self.index)
        self.assertTrue(report["ok"])
        self.assertEqual(report["errors"], [])

    def test_unbalanced_parens_is_an_error(self):
        report = expression_lint.lint_expression(
            "rank(ts_delta(close, 5)", operator_index=self.index)
        self.assertFalse(report["ok"])
        self.assertIn("unbalanced_parens",
                      [error["code"] for error in report["errors"]])

    def test_unterminated_string_is_an_error(self):
        report = expression_lint.lint_expression(
            'bucket(rank(cap), range="0.1,1,0.1)', operator_index=self.index)
        self.assertFalse(report["ok"])
        self.assertIn("unbalanced_string",
                      [error["code"] for error in report["errors"]])

    def test_empty_expression_is_an_error(self):
        report = expression_lint.lint_expression("   ", operator_index=self.index)
        self.assertFalse(report["ok"])
        self.assertEqual(report["errors"][0]["code"], "empty")

    def test_arity_below_minimum_warns(self):
        report = expression_lint.lint_expression(
            "ts_delta(close)", operator_index=self.index)
        self.assertTrue(report["ok"])  # a warning, not a hard failure
        self.assertIn("arity_suspicious",
                      [warning["code"] for warning in report["warnings"]])

    def test_unknown_identifier_warns_unless_strict(self):
        report = expression_lint.lint_expression(
            "rank(ts_delta(not_a_real_field, 5))",
            operator_index=self.index,
            known_fields={"close", "volume"})
        self.assertTrue(report["ok"])
        self.assertIn("unknown_identifier",
                      [warning["code"] for warning in report["warnings"]])

        strict = expression_lint.lint_expression(
            "rank(ts_delta(not_a_real_field, 5))",
            operator_index=self.index,
            known_fields={"close", "volume"},
            strict_fields=True)
        self.assertFalse(strict["ok"])

    def test_offline_mode_skips_operator_check(self):
        report = expression_lint.lint_expression(
            "rank(regression_neut(close, volume))", operator_index=None)
        self.assertTrue(report["ok"])
        self.assertEqual(report["calls"][0]["name"], "rank")

    def test_denylist_block_severity_fails_lint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "deny.json")
            expression_lint.record_rejection("subtract", "Operator subtract boom",
                                             path=path, severity="block")
            denylist = expression_lint.load_denylist(path)
            report = expression_lint.lint_expression(
                "subtract(close, open)", operator_index=self.index,
                denylist=denylist)
            self.assertFalse(report["ok"])
            self.assertEqual(report["errors"][0]["code"], "operator_rejected")

    def test_denylist_warn_severity_only_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "deny.json")
            expression_lint.record_rejection(
                "subtract", "Operator subtract does not support event inputs",
                path=path, severity="warn")
            denylist = expression_lint.load_denylist(path)
            report = expression_lint.lint_expression(
                "subtract(close, open)", operator_index=self.index,
                denylist=denylist)
            self.assertTrue(report["ok"])
            self.assertEqual(report["warnings"][0]["code"], "operator_rejected")

    def test_rejection_count_increments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "deny.json")
            expression_lint.record_rejection("nope", "unknown operator nope", path=path)
            expression_lint.record_rejection("nope", "unknown operator nope", path=path)
            entries = expression_lint.load_denylist(path)
            self.assertEqual(entries["nope"]["count"], 2)

    def test_corrupt_denylist_degrades_to_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "deny.json")
            Path(path).write_text("{not json", encoding="utf-8")
            self.assertEqual(expression_lint.load_denylist(path), {})

    def test_lint_all_flattens_errors(self):
        report = expression_lint.lint_all(
            [("rank(close)", None), ("rank(bogus_op(close))", 4)],
            operator_index=self.index)
        self.assertFalse(report["ok"])
        self.assertEqual(report["expressions"], 2)
        self.assertEqual(len(report["errors"]), 1)
        self.assertIn("expression", report["errors"][0])


class AlphaRegistryTests(unittest.TestCase):
    def _alpha(self, alpha_id="ABC123", expression="rank(close)", sharpe=1.4,
               fitness=1.1, checks=None, status="COMPLETE"):
        return {
            "alpha_id": alpha_id,
            "expression": expression,
            "decay": 4,
            "settings": {"region": "USA", "universe": "TOP3000", "delay": 1,
                         "neutralization": "SUBINDUSTRY"},
            "metrics": {"sharpe": sharpe, "fitness": fitness, "turnover": 0.12},
            "checks": checks if checks is not None else [
                {"name": "LOW_SHARPE", "result": "PASS"},
                {"name": "CONCENTRATED_WEIGHT", "result": "PASS"},
            ],
            "status": status,
        }

    def test_record_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reg.jsonl")
            entry = alpha_registry.build_entry(
                self._alpha(), idea="Idea 47", mechanism="share retirement",
                verdict="candidate", note="kept")
            alpha_registry.record_entry(entry, path)
            loaded = alpha_registry.load_entries(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["alpha_id"], "ABC123")
            self.assertEqual(loaded[0]["idea"], "Idea 47")
            self.assertEqual(loaded[0]["metrics"]["sharpe"], 1.4)
            # decay is folded into settings so an entry is self-contained.
            self.assertEqual(loaded[0]["settings"]["decay"], 4)

    def test_newest_entry_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reg.jsonl")
            first = alpha_registry.build_entry(self._alpha(sharpe=0.5),
                                              verdict="rejected")
            alpha_registry.record_entry(first, path)
            second = alpha_registry.build_entry(self._alpha(sharpe=1.9),
                                               verdict="candidate")
            alpha_registry.record_entry(second, path)
            loaded = alpha_registry.load_entries(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["metrics"]["sharpe"], 1.9)
            self.assertEqual(loaded[0]["verdict"], "candidate")

    def test_skip_unchanged_avoids_duplicate_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reg.jsonl")
            entry = alpha_registry.build_entry(self._alpha(), verdict="candidate")
            _, first = alpha_registry.record_entry(entry, path)
            again = alpha_registry.build_entry(self._alpha(), verdict="candidate")
            _, second = alpha_registry.record_entry(again, path)
            self.assertTrue(first)
            self.assertFalse(second)
            self.assertEqual(len(Path(path).read_text(encoding="utf-8").splitlines()), 1)

    def test_failing_checks_are_extracted(self):
        checks = [{"name": "LOW_SHARPE", "result": "PASS"},
                  {"name": "CONCENTRATED_WEIGHT", "result": "FAIL"}]
        entry = alpha_registry.build_entry(self._alpha(checks=checks))
        self.assertEqual(entry["failing_checks"], ["CONCENTRATED_WEIGHT"])

    def test_entry_without_alpha_id_is_keyed_by_expression(self):
        alpha = self._alpha(alpha_id=None)
        entry = alpha_registry.build_entry(alpha, verdict="rejected")
        self.assertTrue(entry["key"].startswith("expr:"))
        self.assertEqual(entry["key"], entry["expression_hash"])

    def test_invalid_verdict_rejected(self):
        with self.assertRaises(ValueError):
            alpha_registry.build_entry(self._alpha(), verdict="amazing")

    def test_summarize_counts_and_ranks(self):
        entries = [
            alpha_registry.build_entry(self._alpha("A1", sharpe=0.4), verdict="rejected"),
            alpha_registry.build_entry(self._alpha("A2", sharpe=1.8,
                                                   expression="rank(open)"),
                                       verdict="candidate"),
        ]
        summary = alpha_registry.summarize(entries)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["by_verdict"]["candidate"], 1)
        self.assertEqual(summary["best"][0]["alpha_id"], "A2")

    def test_markdown_render_includes_every_alpha(self):
        entries = [alpha_registry.build_entry(self._alpha("A1"))]
        markdown = alpha_registry.render_markdown(entries)
        self.assertIn("| alpha |", markdown)
        self.assertIn("A1", markdown)

    def test_corrupt_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reg.jsonl")
            entry = alpha_registry.build_entry(self._alpha(), verdict="watch")
            alpha_registry.record_entry(entry, path)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("this is not json\n\n")
            loaded = alpha_registry.load_entries(path)
            self.assertEqual(len(loaded), 1)

    def test_missing_registry_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                alpha_registry.load_entries(os.path.join(tmp, "nope.jsonl")), [])


class ResearchPathTests(unittest.TestCase):
    """Results must land outside the git working tree, on a persistent volume."""

    def test_env_var_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "custom-results")
            old = os.environ.get(research_paths.ENV_VAR)
            os.environ[research_paths.ENV_VAR] = target
            try:
                self.assertEqual(research_paths.research_dir(), target)
                self.assertEqual(research_paths.source(), "env:" + research_paths.ENV_VAR)
                self.assertEqual(research_paths.resolve("x.json"),
                                 os.path.join(target, "x.json"))
            finally:
                if old is None:
                    del os.environ[research_paths.ENV_VAR]
                else:
                    os.environ[research_paths.ENV_VAR] = old

    def test_default_is_outside_the_repository(self):
        old = os.environ.pop(research_paths.ENV_VAR, None)
        try:
            path = os.path.abspath(research_paths.research_dir())
        finally:
            if old is not None:
                os.environ[research_paths.ENV_VAR] = old
        repo = str(ROOT)
        self.assertFalse(
            path == repo or path.startswith(repo + os.sep),
            "research output must not live inside the git working tree: " + path)

    def test_ensure_dir_creates_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "nested", "results")
            old = os.environ.get(research_paths.ENV_VAR)
            os.environ[research_paths.ENV_VAR] = target
            try:
                self.assertFalse(os.path.isdir(target))
                self.assertEqual(research_paths.ensure_dir(), target)
                self.assertTrue(os.path.isdir(target))
            finally:
                if old is None:
                    del os.environ[research_paths.ENV_VAR]
                else:
                    os.environ[research_paths.ENV_VAR] = old

    def test_default_paths_follow_the_shared_root(self):
        old = os.environ.pop(research_paths.ENV_VAR, None)
        try:
            base = research_paths.research_dir()
        finally:
            if old is not None:
                os.environ[research_paths.ENV_VAR] = old
        self.assertTrue(alpha_registry.default_registry_path().startswith(base))
        self.assertTrue(expression_lint.default_denylist_path().startswith(base))


if __name__ == "__main__":
    unittest.main()


class OutOfSampleSplitTests(unittest.TestCase):
    """BRAIN's train/test split is the only OOS evidence the API exposes."""

    def test_extract_split_reads_train_and_test_blocks(self):
        import machine_lib as ml

        payload = {
            "settings": {"testPeriod": "P1Y"},
            "is": {"sharpe": 1.65},
            "train": {"sharpe": 1.94, "fitness": 1.71, "startDate": "2019-01-01"},
            "test": {"sharpe": 0.43, "fitness": 0.16, "turnover": 0.043,
                     "startDate": "2022-12-31", "endDate": "2023-12-31"},
        }
        split = ml.extract_split(payload)
        self.assertEqual(split["test_period"], "P1Y")
        self.assertAlmostEqual(split["train"]["sharpe"], 1.94)
        self.assertAlmostEqual(split["test"]["sharpe"], 0.43)
        self.assertEqual(split["test"]["startDate"], "2022-12-31")
        # Keys outside the metric allow-list must not leak through.
        self.assertNotIn("is", split)

    def test_extract_split_returns_none_without_a_split(self):
        import machine_lib as ml

        # testPeriod P0Y leaves no train/test blocks: there is nothing to report.
        self.assertIsNone(ml.extract_split({"settings": {"testPeriod": "P0Y"}, "is": {}}))
        self.assertIsNone(ml.extract_split(None))

    def test_registry_keeps_out_of_sample(self):
        split = {"test_period": "P1Y", "train": {"sharpe": 1.94}, "test": {"sharpe": 0.43}}
        entry = alpha_registry.build_entry(
            {"alpha_id": "abc123", "expression": "rank(close)", "status": "UNSUBMITTED",
             "metrics": {"sharpe": 1.65, "fitness": 1.27}, "checks": [], "out_of_sample": split},
            idea="Idea 57", mechanism="stability", verdict="candidate",
        )
        self.assertEqual(entry["out_of_sample"], split)


class CheckReconciliationTests(unittest.TestCase):
    """BRAIN can serve a cached verdict beside a refreshed value.

    Observed live: SELF_CORRELATION with value 0.7526, limit 0.7 and result
    "PASS". A decision must follow the number, not the stale label.
    """

    def test_ceiling_check_contradiction_is_flagged(self):
        verdict = machine_lib.reconcile_check(
            "SELF_CORRELATION",
            {"value": 0.7526, "limit": 0.7, "result": "PASS"},
        )
        self.assertEqual(verdict["expected"], "FAIL")
        self.assertFalse(verdict["consistent"])

    def test_ceiling_check_agreement(self):
        verdict = machine_lib.reconcile_check(
            "SELF_CORRELATION",
            {"value": 0.6212, "limit": 0.7, "result": "PASS"},
        )
        self.assertEqual(verdict["expected"], "PASS")
        self.assertTrue(verdict["consistent"])

    def test_floor_check_uses_the_opposite_direction(self):
        low = machine_lib.reconcile_check(
            "LOW_SHARPE", {"value": 0.5, "limit": 1.25, "result": "PASS"})
        self.assertEqual(low["expected"], "FAIL")
        good = machine_lib.reconcile_check(
            "LOW_SHARPE", {"value": 2.1, "limit": 1.25, "result": "PASS"})
        self.assertTrue(good["consistent"])

    def test_checks_without_value_are_not_reconciled(self):
        self.assertIsNone(machine_lib.reconcile_check(
            "CONCENTRATED_WEIGHT", {"value": None, "limit": None, "result": "PASS"}))
        self.assertIsNone(machine_lib.reconcile_check(
            "SELF_CORRELATION", {"value": None, "limit": None, "result": "PENDING"}))


class RegistryQueryTests(unittest.TestCase):
    """`alpha find` exists so a field is not re-tested by accident."""

    ENTRIES = [
        {"alpha_id": "A1", "verdict": "candidate", "idea": "Idea 70",
         "mechanism": "value_quality", "expression": "rank(cashflow_op/cap)",
         "metrics": {"sharpe": 1.5, "fitness": 1.4}},
        {"alpha_id": "A2", "verdict": "rejected", "idea": "Idea 71",
         "mechanism": "short_interest", "expression": "rank(-news_short_interest)",
         "metrics": {"sharpe": -0.9, "fitness": -0.3}},
        {"alpha_id": "A3", "verdict": "rejected", "idea": "Idea 72",
         "mechanism": "revision_gap",
         "expression": "rank(ts_delta(anl4_cfo_number,63))",
         "metrics": {"fitness": 0.2}},
    ]

    def test_field_match_finds_prior_use(self):
        found = alpha_registry.find_entries(self.ENTRIES, field="anl4_cfo_number")
        self.assertEqual([e["alpha_id"] for e in found], ["A3"])

    def test_field_match_is_case_insensitive(self):
        self.assertEqual(
            len(alpha_registry.find_entries(self.ENTRIES, field="CASHFLOW_OP")), 1)

    def test_verdict_and_fitness_filters_stack(self):
        rejected = alpha_registry.find_entries(self.ENTRIES, verdicts=["rejected"])
        self.assertEqual(len(rejected), 2)
        strong = alpha_registry.find_entries(self.ENTRIES, min_fitness=1.0)
        self.assertEqual([e["alpha_id"] for e in strong], ["A1"])
        both = alpha_registry.find_entries(
            self.ENTRIES, verdicts=["rejected"], min_fitness=1.0)
        self.assertEqual(both, [])

    def test_mechanism_substring(self):
        found = alpha_registry.find_entries(self.ENTRIES, mechanism="interest")
        self.assertEqual([e["alpha_id"] for e in found], ["A2"])

    def test_no_filters_returns_everything(self):
        self.assertEqual(len(alpha_registry.find_entries(self.ENTRIES)), 3)

    def test_family_is_declared_and_validated(self):
        # ``family`` is the number to quote when the question is "how many
        # independent bets", so it must be a closed vocabulary: mechanism
        # labels overstate diversity (13 book alphas, 11 labels, 4 families).
        self.assertIn("oi", alpha_registry.FAMILIES)
        self.assertIn("quality", alpha_registry.FAMILIES)
        entry = alpha_registry.build_entry(
            {"alpha_id": "Z1", "expression": "rank(cashflow_op/cap)"},
            verdict="candidate", mechanism="value_quality", family="quality")
        self.assertEqual(entry["family"], "quality")
        with self.assertRaises(ValueError):
            alpha_registry.build_entry(
                {"alpha_id": "Z2", "expression": "x"}, family="value-ish")

    def test_find_filters_by_family(self):
        entries = [
            dict(self.ENTRIES[0], family="quality"),
            dict(self.ENTRIES[1], family="attention"),
            dict(self.ENTRIES[2], family="analyst"),
        ]
        found = alpha_registry.find_entries(entries, family="quality")
        self.assertEqual([e["alpha_id"] for e in found], ["A1"])

    def test_summarize_counts_families_and_unclassified(self):
        entries = [
            dict(self.ENTRIES[0], family="quality"),
            dict(self.ENTRIES[1], family="quality"),
            dict(self.ENTRIES[2]),
        ]
        summary = alpha_registry.summarize(entries)
        self.assertEqual(summary["by_family"], {"quality": 2})
        self.assertEqual(summary["family_count"], 1)
        self.assertEqual(summary["unclassified_family"], 1)

    def test_falsified_is_a_distinct_verdict(self):
        # A refuted pre-registration is a result, not a failure: it must stay
        # distinguishable from "the expression underperformed" so a later round
        # does not re-open the mechanism by accident.
        self.assertIn("falsified", alpha_registry.VERDICTS)
        self.assertIn("rejected", alpha_registry.VERDICTS)
        found = alpha_registry.find_entries(self.ENTRIES, verdicts=["falsified"])
        self.assertEqual(found, [])
