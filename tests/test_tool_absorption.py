"""Tests for the selectively absorbed tooling: PP budget, planner, description, rules."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from research_app import description, pp_budget, rules_snapshot, submission_planner


class PpBudgetTests(unittest.TestCase):
    def test_backfill_is_exempt_from_the_charged_count(self):
        budget = pp_budget.pp_budget(
            "group_rank(ts_backfill(fnd6_assets, 21), industry) - 0.5",
            field_datasets={"fnd6_assets": "fundamental6"})
        self.assertEqual(budget["free_operators"], 1)
        # ts_backfill (free) + group_rank + the binary minus
        self.assertEqual(budget["operators_total"], 3)
        self.assertEqual(budget["operators"], 2)
        self.assertEqual(budget["n_fields"], 1)  # industry is a grouping key
        self.assertTrue(budget["single_dataset"])
        self.assertTrue(budget["ok"])

    def test_unary_minus_and_symbols_are_charged(self):
        budget = pp_budget.pp_budget("-rank(close) + rank(open)",
                                     field_datasets={"close": "pv1", "open": "pv1"})
        # rank, unary minus, rank, binary plus
        self.assertEqual(budget["operators_total"], 4)
        self.assertEqual(budget["n_fields"], 2)

    def test_missing_dataset_evidence_is_not_a_pass(self):
        budget = pp_budget.pp_budget("rank(mystery_field)", field_datasets={})
        self.assertFalse(budget["ok"])
        self.assertIn("no dataset evidence", " ".join(budget["issues"]))
        self.assertFalse(budget["single_dataset"])

    def test_field_limit_and_platform_operator_count(self):
        expression = "rank(a) + rank(b) + rank(c) + rank(d)"
        budget = pp_budget.pp_budget(
            expression, field_datasets={k: "pv1" for k in "abcd"},
            platform_operator_count=9)
        self.assertEqual(budget["operator_count_source"], "platform")
        self.assertEqual(budget["operators"], 9)
        self.assertIn("4 data fields > 3", " ".join(budget["issues"]))

    def test_gate_summary_treats_absent_checks_as_unknown(self):
        budget = pp_budget.pp_budget("rank(a)", field_datasets={"a": "pv1"})
        summary = pp_budget.gate_summary(budget, check=[])
        self.assertIn("power_pool_correlation", summary["unknown"])
        self.assertFalse(summary["ok"])

    def test_gate_summary_blocks_on_measured_correlation(self):
        budget = pp_budget.pp_budget("rank(a)", field_datasets={"a": "pv1"})
        check = [{"name": "POWER_POOL_CORRELATION", "result": "PASS", "value": 0.62, "limit": 0.5},
                 {"name": "LOW_SHARPE", "result": "PASS", "value": 1.4, "limit": 1.0}]
        summary = pp_budget.gate_summary(budget, check)
        self.assertIn("power_pool_correlation", summary["blocking"])


class SubmissionPlannerTests(unittest.TestCase):
    def test_conflict_drops_the_lighter_candidate(self):
        result = submission_planner.plan(
            [{"id": "a", "weight": 2.0}, {"id": "b", "weight": 1.0}, {"id": "c", "weight": 1.0}],
            {("a", "b"): 0.8, ("a", "c"): 0.1, ("b", "c"): 0.0})
        self.assertTrue(result["ok"])
        self.assertEqual(result["selected"], ["a", "c"])
        self.assertEqual(result["order"], ["a", "c"])
        self.assertTrue(result["optimal"])
        self.assertEqual([x["id"] for x in result["excluded"]], ["b"])

    def test_negative_correlation_is_a_conflict(self):
        result = submission_planner.plan(
            [{"id": "a", "weight": 1.0}, {"id": "b", "weight": 1.0}],
            {("a", "b"): -0.9})
        self.assertEqual(len(result["selected"]), 1)

    def test_missing_weight_refuses_instead_of_guessing(self):
        result = submission_planner.plan([{"id": "a", "weight": 1.0}, {"id": "b"}], {})
        self.assertFalse(result["ok"])
        self.assertIn("no weight for b", result["reason"])

    def test_unmeasured_pairs_are_reported_not_assumed(self):
        result = submission_planner.plan(
            [{"id": "a", "weight": 1.0}, {"id": "b", "weight": 1.0}, {"id": "c", "weight": 1.0}],
            {("a", "b"): 0.1})
        self.assertEqual(len(result["unknown_pairs"]), 2)  # a-c and b-c
        self.assertIn("never assumed", result["evidence"]["correlations"])


class DescriptionTests(unittest.TestCase):
    def test_compose_refuses_missing_field_description(self):
        draft = description.compose("Leverage predicts underperformance.", ["liabilities"],
                                    field_notes={})
        self.assertFalse(draft["ok"])
        self.assertIn("liabilities", draft["missing"])

    def test_compose_builds_a_checked_description(self):
        draft = description.compose(
            "Within a subindustry, higher leverage predicts lower forward returns.",
            ["liabilities", "assets"],
            field_notes={"liabilities": "total liabilities", "assets": "total assets"})
        self.assertTrue(draft["ok"])
        self.assertGreaterEqual(draft["check"]["length"], 100)
        self.assertTrue(draft["check"]["ok"], draft["check"]["errors"])

    def test_check_rejects_short_text_and_missing_sections(self):
        result = description.check("too short")
        self.assertFalse(result["ok"])
        self.assertTrue(any("length" in e for e in result["errors"]))
        self.assertTrue(any("RATIONALE" in e for e in result["errors"]))

    def test_check_flags_a_near_duplicate(self):
        text = ("IDEA: " + "leverage predicts returns. " * 6 +
                "\n\nRATIONALE: " + "ranked inside subindustry. " * 4)
        result = description.check(text, others=[text])
        self.assertFalse(result["ok"])
        self.assertTrue(any("near-duplicate" in e for e in result["errors"]))


class RulesSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "rules.jsonl"

    def test_extract_keeps_quotas_themes_and_limits(self):
        payload = [
            {"name": "POWER_POOL_CORRELATION", "result": "PASS", "value": 0.31, "limit": 0.5},
            {"name": "REGULAR_SUBMISSION", "result": "PASS", "value": 0, "limit": 4},
            {"name": "MATCHES_THEMES", "result": "WARNING",
             "themes": [{"id": "t1", "name": "ASI/D1 Power Pool Dec`26", "multiplier": 1.0}]},
        ]
        records = rules_snapshot.extract_check_rules(
            payload, scope={"region": "EUR", "delay": 1}, observed_at="2026-11-25T10:00:00Z")
        kinds = {r["name"]: r["kind"] for r in records}
        self.assertEqual(kinds["REGULAR_SUBMISSION"], "quota")
        self.assertEqual(kinds["MATCHES_THEMES"], "themes")
        theme = next(r for r in records if r["name"] == "MATCHES_THEMES")
        self.assertEqual(theme["themes"][0]["multiplier"], 1.0)
        self.assertEqual(theme["scope"], {"region": "EUR", "delay": 1})

    def test_lookup_reports_staleness_and_absence(self):
        rules_snapshot.append(self.path, rules_snapshot.extract_check_rules(
            [{"name": "LOW_SHARPE", "result": "PASS", "value": 1.7, "limit": 1.58}],
            scope={"region": "USA"}, observed_at="2026-11-20T00:00:00Z"))
        found = rules_snapshot.lookup(rules_snapshot.load(self.path), "LOW_SHARPE",
                                      {"region": "USA"}, max_age_seconds=86400)
        self.assertIsNotNone(found)
        self.assertTrue(found["stale"])
        self.assertIsNone(rules_snapshot.lookup(rules_snapshot.load(self.path), "LOW_SHARPE",
                                                {"region": "EUR"}))

    def test_official_rule_carries_its_source(self):
        rule = rules_snapshot.official_rule(
            "pp_max_operators", 8, source_id="power-pool", source_url="https://support.worldquantbrain.com/...",
            statement="operators including repeats <= 8")
        self.assertEqual(rule["source"], "official_doc")
        self.assertEqual(rule["source_id"], "power-pool")


if __name__ == "__main__":
    unittest.main()