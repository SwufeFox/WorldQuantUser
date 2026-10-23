#!/usr/bin/env python3
"""Identical results from different inputs mean the differing part was thrown away.

Measured 2026-10-23: six alphas over six different Core Earnings fields returned Sharpe
2.09-2.11 and turnover 0.81 because the fields were unpopulated and every expression collapsed
to the fast leg. Two diagnostic runs settled it - the field alone scored 0.0 with zero
turnover, the fast leg alone scored -2.09 with 0.8111.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import degenerate_batch  # noqa: E402


def record(candidate, expression, sharpe, turnover):
    return {"candidate_id": candidate, "expression": expression,
            "metrics": {"sharpe": sharpe, "turnover": turnover}}


class DegenerateBatchTests(unittest.TestCase):
    def test_identical_metrics_from_different_fields_are_flagged(self):
        records = [record("A", "rank(field_a)", 2.09, 0.81),
                   record("B", "rank(field_b)", 2.09, 0.81),
                   record("C", "rank(field_c)", 2.09, 0.81)]
        report = degenerate_batch.report(records)
        self.assertEqual(len(report["identical_clusters"]), 1)
        self.assertIn("degenerate", report["verdict"])
        self.assertEqual(len(report["identical_clusters"][0]["candidates"]), 3)

    def test_one_record_is_not_a_cluster(self):
        report = degenerate_batch.report([record("A", "rank(field_a)", 1.2, 0.2)])
        self.assertEqual(report["identical_clusters"], [])
        self.assertEqual(report["verdict"], "ok")

    def test_the_same_expression_twice_is_a_duplicate_not_a_degeneracy(self):
        records = [record("A", "rank(field_a)", 1.2, 0.2),
                   record("B", "rank(field_a)", 1.2, 0.2)]
        self.assertEqual(degenerate_batch.report(records)["identical_clusters"], [])

    def test_zero_turnover_and_zero_sharpe_is_reported_as_empty(self):
        report = degenerate_batch.report([record("A", "rank(dead_field)", 0.0, 0.0)])
        self.assertEqual(report["empty_results"], ["A"])


if __name__ == "__main__":
    unittest.main()
