#!/usr/bin/env python3
"""Tests for offline lint rules that catch what the simulator would reject.

Local lint sees operator names, arity and field ids - it cannot see units. Every rule
here exists because a unit or typing error cost a simulation: the point is to move the
failure forward to the editor.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import expression_lint


class GroupProductTests(unittest.TestCase):
    """`bucket` returns a group label, not a number - offline lint can only see the name."""

    def test_arithmetic_on_a_bucket_is_flagged(self):
        # 2026-11-08: three calendar arms multiplied a bucket by 21 and the simulator
        # answered "expected Unit[], found Unit[Group:1]" for each of them.
        report = expression_lint.lint_expression(
            'rank(ts_step(1) - 21*bucket(ts_step(1), range="0,10000,21"))')
        codes = [w["code"] for w in report["warnings"]]
        self.assertIn("group_value_misused", codes)

    def test_a_bucket_fed_into_group_rank_is_fine(self):
        report = expression_lint.lint_expression(
            'group_rank(close, bucket(rank(cap), range="0,1,0.5"))')
        codes = [w["code"] for w in report["warnings"]]
        self.assertNotIn("group_value_misused", codes)

    def test_densify_is_treated_the_same_way(self):
        report = expression_lint.lint_expression("rank(2*densify(bucket(rank(cap))))")
        codes = [w["code"] for w in report["warnings"]]
        self.assertIn("group_value_misused", codes)


if __name__ == "__main__":
    unittest.main()

class FormatTests(unittest.TestCase):
    """`fmt`: the book stores multi-line expressions; everything written here did not."""

    SAMPLE = ("leverage = liabilities / assets; gp = (revenue - cogs) / assets; "
              "core = rank( group_rank(gp, subindustry) + 0.5 * group_rank(leverage, subindustry) ); "
              "core + rank(leverage)")

    def test_one_statement_per_line_and_semicolons_kept(self):
        out = expression_lint.pretty_expression(self.SAMPLE)
        self.assertEqual(out.count(";"), self.SAMPLE.count(";"))
        # three statements -> at least three lines, and the closing call is on its own
        self.assertGreaterEqual(len(out.splitlines()), 4)
        self.assertTrue(out.startswith("leverage ="))

    def test_formatting_preserves_the_expression(self):
        import re
        out = expression_lint.pretty_expression(self.SAMPLE)
        strip = lambda text: re.sub(r"\s+", "", text)
        self.assertEqual(strip(out), strip(self.SAMPLE))

    def test_formatting_is_idempotent(self):
        once = expression_lint.pretty_expression(self.SAMPLE)
        self.assertEqual(expression_lint.pretty_expression(once), once)

    def test_a_short_expression_is_left_alone(self):
        self.assertEqual(expression_lint.pretty_expression("rank(close)"), "rank(close)")
