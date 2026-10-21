#!/usr/bin/env python3
"""Tests for ablation variant generation and the contribution table.

The refusals matter as much as the successes. A product or a gated combination
cannot be reduced syntactically, and guessing would emit a variant that is not the
experiment anyone intended — one that spends quota and returns a number that looks
like evidence.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ablation  # noqa: E402

COMPOSITE = "a = rank(cap); n = rank(volume); c = rank(close); a + n + c"


class ParseTests(unittest.TestCase):
    def test_named_legs_and_plain_sum(self):
        parsed = ablation.parse(COMPOSITE)
        self.assertEqual(parsed["order"], ["a", "n", "c"])
        self.assertEqual(parsed["used"], ["a", "n", "c"])
        self.assertEqual(parsed["legs"]["a"], "rank(cap)")

    def test_product_combination_is_refused(self):
        # Removing a leg from a product is not a subtraction; refuse rather than
        # invent one.
        with self.assertRaises(ablation.AblationError) as ctx:
            ablation.parse("a = rank(cap); b = rank(close); a * b")
        self.assertIn("sum of legs", str(ctx.exception))

    def test_function_call_in_combination_is_refused(self):
        with self.assertRaises(ablation.AblationError):
            ablation.parse("a = rank(cap); b = rank(close); a + rank(b)")

    def test_undeclared_token_is_refused(self):
        with self.assertRaises(ablation.AblationError) as ctx:
            ablation.parse("a = rank(cap); a + zzz")
        self.assertIn("zzz", str(ctx.exception))

    def test_non_assignment_head_is_refused(self):
        with self.assertRaises(ablation.AblationError):
            ablation.parse("rank(cap); a = rank(close); a")

    def test_missing_legs_is_refused(self):
        with self.assertRaises(ablation.AblationError) as ctx:
            ablation.parse("rank(cap)")
        self.assertIn("no legs", str(ctx.exception))

    def test_duplicate_leg_in_combination_is_refused(self):
        with self.assertRaises(ablation.AblationError):
            ablation.parse("a = rank(cap); b = rank(close); a + a + b")

    def test_declared_but_unused_leg_is_reported(self):
        parsed = ablation.parse("a = rank(cap); b = rank(close); c = rank(open); a + b")
        self.assertEqual(parsed["unused"], ["c"])


class VariantTests(unittest.TestCase):
    def test_three_leg_composite_yields_seven_variants(self):
        plan = ablation.variants(COMPOSITE)
        ids = [item["candidate_id"] for item in plan]
        self.assertEqual(ids, ["full", "loo:a", "loo:n", "loo:c",
                               "solo:a", "solo:n", "solo:c"])

    def test_full_is_first_and_unchanged_in_meaning(self):
        plan = ablation.variants(COMPOSITE)
        self.assertEqual(plan[0]["candidate_id"], "full")
        self.assertTrue(plan[0]["expression"].endswith("a + n + c"))

    def test_leave_one_out_drops_exactly_one_leg(self):
        plan = {item["candidate_id"]: item for item in ablation.variants(COMPOSITE)}
        self.assertTrue(plan["loo:n"]["expression"].endswith("a + c"))
        self.assertEqual(plan["loo:n"]["removed"], "n")

    def test_solo_keeps_the_assignments_and_selects_one_leg(self):
        plan = {item["candidate_id"]: item for item in ablation.variants(COMPOSITE)}
        self.assertTrue(plan["solo:c"]["expression"].endswith("c"))
        self.assertIn("a = rank(cap)", plan["solo:c"]["expression"])

    def test_single_leg_has_no_leave_one_out(self):
        plan = ablation.variants("a = rank(cap); b = rank(close); a + b")
        ids = [item["candidate_id"] for item in plan]
        self.assertIn("loo:a", ids)
        plan_one = ablation.variants("a = rank(cap); b = rank(close); a + b",
                                     modes=["loo"])
        self.assertEqual(len(plan_one), 2)

    def test_mode_selection_is_honoured(self):
        plan = ablation.variants(COMPOSITE, modes=["full"])
        self.assertEqual([item["candidate_id"] for item in plan], ["full"])

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ablation.AblationError):
            ablation.variants(COMPOSITE, modes=["bogus"])


class ReportTests(unittest.TestCase):
    def _records(self, full, loo, solo):
        records = [{"candidate_id": "full", "alpha_id": "A", "metrics": {"fitness": full}}]
        for leg, value in loo.items():
            records.append({"candidate_id": "loo:" + leg, "alpha_id": leg,
                            "metrics": {"fitness": value}})
        for leg, value in solo.items():
            records.append({"candidate_id": "solo:" + leg, "alpha_id": leg,
                            "metrics": {"fitness": value}})
        return records

    def test_contribution_is_full_minus_without(self):
        records = self._records(1.60, {"a": 1.40, "n": 1.20, "c": 1.62},
                               {"a": 0.9, "n": 0.4, "c": 0.5})
        result = ablation.report(records)
        by_leg = {row["leg"]: row for row in result["legs"]}
        self.assertAlmostEqual(by_leg["a"]["contribution"], 0.20)
        self.assertAlmostEqual(by_leg["c"]["contribution"], -0.02)

    def test_verdicts_separate_adds_hurts_and_neutral(self):
        records = self._records(1.60, {"a": 1.20, "n": 1.59, "c": 1.90},
                               {"a": 0.9, "n": 0.4, "c": 0.5})
        by_leg = {row["leg"]: row for row in ablation.report(records)["legs"]}
        self.assertEqual(by_leg["a"]["verdict"], "adds")
        self.assertEqual(by_leg["n"]["verdict"], "neutral")
        self.assertEqual(by_leg["c"]["verdict"], "hurts")

    def test_hurting_legs_are_listed(self):
        records = self._records(1.49, {"i": 1.59}, {"i": 0.8})
        result = ablation.report(records)
        self.assertIn("i", result["hurts"])

    def test_missing_leave_one_out_arm_is_unmeasured_not_zero(self):
        # A missing arm and a zero contribution are different facts; reporting the
        # first as the second would hide an unfinished run.
        records = [{"candidate_id": "full", "metrics": {"fitness": 1.5}},
                   {"candidate_id": "solo:a", "metrics": {"fitness": 0.4}}]
        row = ablation.report(records)["legs"][0]
        self.assertIsNone(row["contribution"])
        self.assertEqual(row["verdict"], "unmeasured")

    def test_records_without_labels_are_ignored(self):
        result = ablation.report([{"alpha_id": "X", "metrics": {"fitness": 2.0}}])
        self.assertIsNone(result["full_fitness"])
        self.assertEqual(result["legs"], [])


if __name__ == "__main__":
    unittest.main()