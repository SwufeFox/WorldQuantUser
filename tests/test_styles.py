#!/usr/bin/env python3
"""Tests for `ledger styles`: what an alpha compares against.

Comparison sets must remain distinct even when they use the same input. Each bucket
is exercised with a representative expression.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research_app import styles


class StylesTests(unittest.TestCase):
    def test_a_static_ratio_is_static(self):
        verdict = styles.classify("group_rank(fnd6_x / assets, subindustry) - 0.5")
        self.assertEqual(verdict["comparison"], "static")

    def test_a_time_series_percentile_is_history(self):
        verdict = styles.classify("ts_rank(eps / close, 252)")
        self.assertEqual(verdict["comparison"], "history")

    def test_a_window_without_a_history_comparison_is_window(self):
        # ts_sum(returns, 5) is a five-day move: it uses history without comparing
        # against it, and lumping it with ts_rank would hide the distinction the
        # screen exists for.
        verdict = styles.classify("rank(ts_sum(returns, 5))")
        self.assertEqual(verdict["comparison"], "window")

    def test_pure_arithmetic_is_level(self):
        self.assertEqual(styles.classify("liabilities / assets")["comparison"], "level")

    def test_backfill_is_transparent(self):
        verdict = styles.classify("group_rank(ts_backfill(x, 252) / y, subindustry)")
        self.assertEqual(verdict["comparison"], "static")

    def test_summary_counts_and_sorts_static_first(self):
        entries = [
            {"alpha_id": "A", "expression": "rank(close)"},
            {"alpha_id": "B", "expression": "ts_rank(close, 252)", "metrics": {"fitness": 2.0}},
            {"alpha_id": "C", "expression": ""},
        ]
        summary = styles.summarize(entries)
        self.assertEqual(summary["counts"], {"history": 1, "window": 0, "static": 1, "level": 0})
        self.assertEqual(summary["skipped"], ["C"])
        self.assertEqual(summary["rows"][0]["alpha_id"], "A")

    def test_training_fitness_alone_orders_each_comparison_group(self):
        entries = [{"alpha_id": alpha_id, "expression": "rank(close)",
                    "metrics": {"fitness": full},
                    "out_of_sample": {"train": {"fitness": train}}}
                   for alpha_id, train, full in (("A", 2, -5), ("B", 0, 10),
                                                 ("C", -1, 20), ("D", None, 99))]
        self.assertEqual([row["alpha_id"] for row in styles.summarize(list(reversed(entries)))["rows"]],
                         ["A", "B", "C", "D"])
        for entry in entries:
            entry["metrics"]["fitness"] *= -10
        self.assertEqual([row["alpha_id"] for row in styles.summarize(entries)["rows"]],
                         ["A", "B", "C", "D"])


if __name__ == "__main__":
    unittest.main()
