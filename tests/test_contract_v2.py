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
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import alpha_registry  # noqa: E402
import expression_lint
import expression_types
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import probe_units  # noqa: E402
import expression_macros  # noqa: E402
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


class AlphaEntryValidationTests(unittest.TestCase):
    def test_unknown_per_alpha_key_raises(self):
        # A per-alpha key that the loader does not read must not be ignored: a
        # config claiming to override neutralisation would otherwise test
        # nothing while looking like it did.
        with self.assertRaises(ValueError) as ctx:
            machine_lib.normalize_alphas(
                [{"expression": "rank(close)", "neutralization": "SUBINDUSTRY"}],
                {"decay": 4})
        self.assertIn("neutralization", str(ctx.exception))

    def test_known_per_alpha_keys_still_accepted(self):
        pairs = machine_lib.normalize_alphas(
            [{"expression": "rank(close)", "decay": 7, "mechanism": "m",
              "family": "quality", "candidate_id": "C1", "hypothesis": "h"}],
            {"decay": 4})
        self.assertEqual(pairs, [("rank(close)", 7)])


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

    def test_assigned_variables_are_not_treated_as_fields(self):
        # A composite names its intermediate legs; those names are locals, not
        # datafields. Flagging them made every readable multi-line expression
        # fail strict field checking.
        expression = (
            "lev = liabilities / assets;"
            "news = ts_mean(mean_news_impact_projection, 21);"
            "group_rank(lev, subindustry) + group_rank(news, subindustry)"
        )
        report = expression_lint.lint_expression(
            expression,
            operator_index=self.index,
            known_fields={"liabilities", "assets", "mean_news_impact_projection",
                          "subindustry"},
            strict_fields=True)
        flagged = [item.get("identifier") for item in report["errors"]
                   if item.get("code") == "unknown_identifier"]
        self.assertEqual(flagged, [])
        # The stub operator index does not know every call in the snippet, so the
        # assertion stays on what this test is about: locals must not be treated
        # as datafields, while a genuinely missing field still is.
        typo = expression_lint.lint_expression(
            expression + "; rank(not_a_real_field)",
            operator_index=self.index,
            known_fields={"liabilities", "assets", "mean_news_impact_projection",
                          "subindustry"},
            strict_fields=True)
        self.assertIn("not_a_real_field",
                      [item.get("identifier") for item in typo["errors"]
                       if item.get("code") == "unknown_identifier"])

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

    def test_entry_without_alpha_id_is_keyed_by_expression_and_settings(self):
        # Keying on the expression alone conflated one formula run under two
        # universes, so the second record silently replaced the first.
        alpha = self._alpha(alpha_id=None)
        entry = alpha_registry.build_entry(alpha, verdict="rejected")
        self.assertTrue(entry["key"].startswith("expr:"))
        self.assertTrue(entry["key"].startswith(entry["expression_hash"]))
        self.assertIn("@", entry["key"])
        other = alpha_registry.build_entry(
            dict(alpha, settings={"universe": "TOP500", "decay": 12}),
            verdict="rejected")
        self.assertNotEqual(entry["key"], other["key"])

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


class OperatorBudgetTests(unittest.TestCase):
    """The platform limits operator *instances*, not distinct operators.

    Two composites in this campaign were rejected after spending quota with
    "Expression with 78 operators exceeds limit of 64" and the same message at 95.
    The estimate here reproduces both numbers exactly, which is why it is trusted to
    reject locally instead of at the simulator.
    """

    def test_known_rejections_are_predicted(self):
        block = ("onv = open / ts_delay(close, 1) - 1; intr = close / open - 1; "
                 "oiv = rank(ts_sum(onv, 10)) - rank(ts_sum(intr, 10)); ")
        # Ten operator instances per block; eight blocks is comfortably over the wall.
        self.assertGreater(expression_lint.estimate_operator_count(block * 8),
                           expression_lint.OPERATOR_LIMIT)

    def test_small_expression_has_no_finding(self):
        self.assertEqual(
            expression_lint.operator_budget_findings("rank(close) - 0.5"), [])

    def test_limit_is_an_error_and_warn_band_is_a_warning(self):
        self.assertEqual(
            expression_lint.operator_budget_findings("rank(a)", limit=0, warn=0)[0]["severity"],
            "error")
        self.assertEqual(
            expression_lint.operator_budget_findings("rank(a) + rank(b)",
                                                     limit=10, warn=1)[0]["severity"],
            "warning")

    def test_lint_surfaces_the_finding(self):
        # Distinct-operator count stays small; only the instance count is over.
        expression = " + ".join(["rank(close)"] * 70)
        result = expression_lint.lint_expression(expression)
        codes = {item.get("code") for item in result["errors"]}
        self.assertIn("operator_budget", codes)
        self.assertGreater(result["operator_estimate"], expression_lint.OPERATOR_LIMIT)


class UnitProbeTests(unittest.TestCase):
    """Probe interpretation, pinned to the simulator's own wording.

    The first version read the "found" side of
    ``expected "Unit[TSPrice:1]", found "Unit[]"`` and therefore recorded every
    price-denominated field as dimensionless. Only printing the raw error showed it.
    """

    def test_expected_side_is_the_field_unit(self):
        record = {"status": "WARNING",
                  "error": 'Incompatible unit for input of "add" at index 1, '
                           'expected "Unit[TSPrice:1]", found "Unit[]".'}
        unit, _note = probe_units.interpret(record)
        self.assertEqual(unit, "Unit[TSPrice:1]")

    def test_clean_probe_means_dimensionless(self):
        self.assertEqual(probe_units.interpret({"status": "COMPLETE", "error": None})[0],
                         "dimensionless")

    def test_unknown_field_is_not_a_unit(self):
        unit, note = probe_units.interpret(
            {"status": "ERROR", "error": "Attempted to use unknown variable foo"})
        self.assertIsNone(unit)
        self.assertIn("does not exist", note)

    def test_raw_error_is_kept_for_other_failures(self):
        unit, note = probe_units.interpret(
            {"status": "ERROR", "error": "some other rejection"})
        self.assertIsNone(unit)
        self.assertIn("some other rejection", note)


class TrailingTokenTests(unittest.TestCase):
    """PDF extraction glued page numbers onto eight of the 101 formulas."""

    def test_trailing_digit_is_an_error(self):
        report = expression_lint.lint_expression("(if_else((adv20 < volume), 1, -1)) 9")
        codes = {item.get("code") for item in report["errors"]}
        self.assertIn("trailing_token", codes)

    def test_clean_expression_has_no_finding(self):
        self.assertEqual(
            expression_lint.trailing_token_findings("(if_else((adv20 < volume), 1, -1))"),
            [])

    def test_plain_expression_without_parens_is_fine(self):
        self.assertEqual(expression_lint.trailing_token_findings("rank(close) - 0.5"), [])


class VectorFieldTests(unittest.TestCase):
    """A third of the catalogue is VECTOR, and most operators cannot read it.

    A forum report documents rank() and ts_backfill() failing with "event input" on
    such a field while ts_sum() works; this account hit "Operator multiply does not
    support event inputs" independently. Batch templating without this check fails
    en masse.
    """

    TYPES = {"mdl109_eps_sur_decay": "VECTOR", "close": "MATRIX",
             "five_minute_price_change_pct": "VECTOR"}

    def test_rank_on_a_vector_field_is_flagged(self):
        findings = expression_lint.vector_findings(
            "rank(mdl109_eps_sur_decay)", self.TYPES)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["operator"], "rank")

    def test_aggregation_is_the_documented_workaround(self):
        self.assertEqual(
            expression_lint.vector_findings("ts_sum(mdl109_eps_sur_decay, 90)", self.TYPES),
            [])

    def test_matrix_field_is_not_flagged(self):
        self.assertEqual(
            expression_lint.vector_findings("rank(close)", self.TYPES), [])

    def test_aggregation_inside_another_call_is_accepted(self):
        # rank(vec_avg(vector_field)) is the documented workaround. The first version
        # scanned the outer call's whole argument text and flagged it, blocking four
        # legitimate expressions.
        self.assertEqual(
            expression_lint.vector_findings(
                "rank(vec_avg(five_minute_price_change_pct))", self.TYPES), [])

    def test_direct_use_inside_a_rejecting_operator_is_flagged(self):
        findings = expression_lint.vector_findings(
            "rank(five_minute_price_change_pct)", self.TYPES)
        self.assertEqual(findings[0]["operator"], "rank")

    def test_strict_mode_escalates_to_error(self):
        finding = expression_lint.vector_findings(
            "rank(mdl109_eps_sur_decay)", self.TYPES, strict=True)[0]
        self.assertEqual(finding["severity"], "error")

    def test_no_type_information_means_no_finding(self):
        self.assertEqual(expression_lint.vector_findings("rank(anything)", None), [])

    def test_lint_surfaces_it(self):
        # The handover: this used to be reported by the hand-built VECTOR denylist as
        # `vector_field_input`. The signature table now derives the same verdict as
        # `type_mismatch`, and the denylist only covers operators the table lacks, so the
        # problem is reported once and with expected/actual types attached.
        report = expression_lint.lint_expression(
            "rank(mdl109_eps_sur_decay)", field_types=self.TYPES)
        codes = {item.get("code") for item in report["warnings"] + report["errors"]}
        self.assertIn("type_mismatch", codes)
        self.assertNotIn("vector_field_input", codes)


class AlphaMetadataTests(unittest.TestCase):
    """name and category are writable; description does not exist.

    Eighteen candidate spellings for a description all return "Unexpected property", so
    none is invented: the rule is name and category, and tags stay available as the
    platform's only free-text list.
    """

    class _Session:
        def __init__(self, status=200, payload=None, echo=None):
            self.status_code = status
            self.payload = payload or {}
            self.echo = echo
            self.sent = None

        def patch(self, url, json=None, timeout=None):
            self.sent = json
            outer = self

            class _R:
                status_code = outer.status_code
                text = ""

            return _R()

    def test_invalid_category_is_refused_before_the_call(self):
        with self.assertRaises(ValueError) as ctx:
            machine_lib.set_alpha_metadata(AlphaMetadataTests._Session(),
                                           "A", category="NOT_A_CATEGORY")
        self.assertIn("category must be one of", str(ctx.exception))

    def test_tags_are_sent_as_a_list(self):
        session = AlphaMetadataTests._Session()
        with mock.patch.object(machine_lib, "fetch_alpha_metrics",
                               return_value={"id": "A", "name": "n",
                                             "category": "VOLUME",
                                             "tags": ["t"]}):
            machine_lib.set_alpha_metadata(session, "A", name="n",
                                           category="VOLUME", tags=["t"])
        self.assertIsInstance(session.sent["tags"], list)

    def test_metadata_that_does_not_stick_raises(self):
        # The API answers 200 for updates it has only accepted; the write is verified
        # by re-reading, the same way a submission is verified by dateSubmitted.
        session = AlphaMetadataTests._Session()
        with mock.patch.object(machine_lib, "fetch_alpha_metrics",
                               return_value={"id": "A", "name": None,
                                             "category": None, "tags": []}):
            with self.assertRaises(machine_lib.ServiceError) as ctx:
                machine_lib.set_alpha_metadata(session, "A", name="n")
        self.assertIn("did not stick", str(ctx.exception))

    def test_nothing_to_update_is_refused(self):
        with self.assertRaises(ValueError):
            machine_lib.set_alpha_metadata(AlphaMetadataTests._Session(), "A")


class SelfCorrelationTests(unittest.TestCase):
    """The platform's self-correlation view, and the substitute route's arithmetic.

    A forum thread documents two things this pins: the endpoint answers 200 with an
    empty body plus Retry-After when it is not ready (reading that as "no data" reports
    an alpha as uncorrelated), and the rule allows a correlated alpha through when its
    Sharpe exceeds, by the margin, the Sharpe of *every* correlated alpha.
    """

    def test_no_counterparts_above_the_limit_is_clear(self):
        verdict = machine_lib.self_correlation_verdict(
            {"records": [["A", None, "EQUITY", "USA", "TOP3000", 0.31, 2.0, 0.1, 0.1, 1.5, 0.001]]})
        self.assertFalse(verdict["blocked"])

    def test_bar_uses_the_highest_counterpart_sharpe_not_the_correlated_one(self):
        # The tempting shortcut is to take the 0.90 pair's Sharpe; a second pair at
        # 0.84 with a higher Sharpe raises the bar and the shortcut would pass it.
        records = [
            ["low_sharpe", None, "EQUITY", "USA", "TOP3000", 0.90, 1.50, 0.1, 0.1, 1.2, 0.001],
            ["high_sharpe", None, "EQUITY", "USA", "TOP3000", 0.84, 2.54, 0.1, 0.1, 2.3, 0.001],
        ]
        verdict = machine_lib.self_correlation_verdict({"records": records})
        self.assertTrue(verdict["blocked"])
        self.assertAlmostEqual(verdict["sharpe_needed"], 2.794, places=3)

    def test_pairs_below_the_limit_do_not_raise_the_bar(self):
        records = [
            ["faint", None, "EQUITY", "USA", "TOP3000", 0.20, 4.00, 0.1, 0.1, 3.0, 0.001],
            ["real", None, "EQUITY", "USA", "TOP3000", 0.85, 2.00, 0.1, 0.1, 1.8, 0.001],
        ]
        verdict = machine_lib.self_correlation_verdict({"records": records})
        self.assertAlmostEqual(verdict["sharpe_needed"], 2.2, places=3)


class ExpressionMacroTests(unittest.TestCase):
    """Named templates, so a shape is written once and instantiated in bulk.

    The campaign kept retyping the same accounting-growth shape with a new field; each
    transcription risked a typo that only lint would sometimes catch.
    """

    SOURCE = """
# a comment between definitions belongs to neither body
def growth(f, w=63):
    ts_delta(ts_backfill({f}, 252), {w}) / (abs(ts_delay(ts_backfill({f}, 252), {w})) + 1)

def position():
    (close - low) / (high - low) - 0.5

def carrier(f, fast, fw=0.5, w=63):
    -rank(growth({f}, {w})) - {fw}*rank({fast})

def loop_a(x):
    loop_b({x})

def loop_b(x):
    loop_a({x})
"""

    def setUp(self):
        self.macros = expression_macros.parse_macros(self.SOURCE)

    def test_parameters_and_defaults_parse(self):
        self.assertEqual([p for p, _ in self.macros["growth"]["params"]], ["f", "w"])
        self.assertEqual(self.macros["growth"]["params"][1][1], "63")

    def test_comment_between_definitions_is_not_absorbed(self):
        # Without skipping comment lines the trailing prose is glued onto the previous
        # body and appears inside every expansion.
        self.assertNotIn("#", self.macros["growth"]["body"])
        self.assertNotIn("belongs to neither", self.macros["growth"]["body"])

    def test_nested_macro_expands(self):
        out = expression_macros.expand(self.macros, "carrier", ["fnd6_x", "position()"])
        self.assertIn("ts_delta(ts_backfill(fnd6_x, 252), 63)", out)
        self.assertIn("rank((close - low) / (high - low) - 0.5)", out)
        self.assertNotIn("{", out)

    def test_cycle_is_refused(self):
        with self.assertRaises(expression_macros.MacroError) as ctx:
            expression_macros.expand(self.macros, "loop_a", ["x"])
        self.assertIn("cycle", str(ctx.exception))

    def test_missing_argument_is_refused(self):
        with self.assertRaises(expression_macros.MacroError) as ctx:
            expression_macros.expand(self.macros, "growth", [])
        self.assertIn("missing argument", str(ctx.exception))

    def test_duplicate_expressions_are_skipped_not_resubmitted(self):
        plan = [
            {"macro": "position", "args": [], "decay": 4, "candidate_id": "A"},
            {"macro": "position", "args": [], "decay": 4, "candidate_id": "B"},
        ]
        alphas, skipped = expression_macros.build(self.macros, plan, built=set())
        self.assertEqual(len(alphas), 1)
        self.assertEqual(skipped[0]["candidate_id"], "B")
        self.assertIn("duplicate", skipped[0]["reason"])

    def test_expansion_is_deterministic(self):
        first = expression_macros.expand(self.macros, "carrier", ["f", "position()"])
        second = expression_macros.expand(self.macros, "carrier", ["f", "position()"])
        self.assertEqual(first, second)


class ExpressionTypeTests(unittest.TestCase):
    """Five field types, operator signatures, and the mismatches they imply.

    Before this, the VECTOR rule was a hand-built denylist assembled from error messages.
    A signature table derives it: only the vector reducers accept a VECTOR, so `rank` on
    one is a type error rather than a special case.
    """

    TYPES = {"five_minute_price_change_pct": "VECTOR", "close": "MATRIX",
             "subindustry": "GROUP", "top3000": "UNIVERSE", "ticker": "SYMBOL"}

    def codes(self, expression):
        return [f["code"] for f in expression_types.type_findings(
            expression, self.TYPES, strict=True)]

    def test_vector_into_rank_is_a_type_error(self):
        self.assertIn("type_mismatch", self.codes("rank(five_minute_price_change_pct)"))

    def test_vector_reduced_first_is_accepted(self):
        self.assertEqual(self.codes("rank(vec_avg(five_minute_price_change_pct))"), [])

    def test_arithmetic_on_a_vector_is_a_type_error(self):
        self.assertIn("type_mismatch",
                      self.codes("multiply(five_minute_price_change_pct, 2)"))

    def test_group_slot_rejects_a_matrix(self):
        self.assertIn("type_mismatch", self.codes("group_rank(close, close)"))

    def test_group_slot_accepts_a_group(self):
        self.assertEqual(self.codes("group_rank(close, subindustry)"), [])

    def test_bucket_produces_a_group(self):
        self.assertEqual(self.codes("group_rank(close, bucket(rank(close)))"), [])
        self.assertIn("type_mismatch", self.codes("densify(close)"))

    def test_documented_but_unavailable_operator_is_named(self):
        findings = expression_types.type_findings(
            "rank(vec_choose(five_minute_price_change_pct, -1))", self.TYPES, strict=True)
        self.assertIn("operator_documented_unavailable", [f["code"] for f in findings])

    def test_unknown_types_are_not_guessed(self):
        self.assertEqual(self.codes("rank(some_assigned_variable)"), [])

    def test_window_arguments_are_not_type_checked(self):
        self.assertEqual(self.codes("ts_delta(close, 5)"), [])
