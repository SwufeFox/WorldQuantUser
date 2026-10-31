#!/usr/bin/env python3
"""Tests for rejection attribution.

The point of these tests is the distinction the ledger could not previously make:
`LOW_FITNESS` alone does not say whether to change the trading, the amplitude, or the
idea, and those are three different next moves.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import rejection_reasons as rr  # noqa: E402


def entry(checks, **metrics):
    return {"verdict": "rejected", "failing_checks": list(checks), "metrics": metrics}


class NamedGates(unittest.TestCase):
    def test_concentration_wins_over_fitness(self):
        code, _ = rr.classify(entry(["CONCENTRATED_WEIGHT", "LOW_FITNESS"],
                                    sharpe=2.5, fitness=3.3, turnover=0.2, returns=0.3))
        self.assertEqual(code, "weight_concentration")

    def test_turnover_floor_is_named_not_guessed(self):
        code, detail = rr.classify(entry(["LOW_TURNOVER"], sharpe=1.5, fitness=1.2,
                                         turnover=0.006, returns=0.05))
        self.assertEqual(code, "turnover_floor")
        self.assertIn("cannot be traded", detail)

    def test_self_correlation_is_crowded(self):
        self.assertEqual(rr.classify(entry(["SELF_CORRELATION"]))[0], "crowded")

    def test_fragile_subuniverse_only_when_other_gates_pass(self):
        self.assertEqual(
            rr.classify(entry(["LOW_SUB_UNIVERSE_SHARPE"], sharpe=1.6, fitness=1.4,
                              turnover=0.1, returns=0.09))[0],
            "fragile_subuniverse")
        self.assertEqual(
            rr.classify(entry(["LOW_SUB_UNIVERSE_SHARPE", "LOW_SHARPE"],
                              sharpe=0.8, fitness=0.4, turnover=0.1, returns=0.05))[0],
            "weak_signal")


class FitnessDecomposition(unittest.TestCase):
    def test_weak_sharpe_is_not_blamed_on_amplitude(self):
        code, _ = rr.classify(entry(["LOW_SHARPE", "LOW_FITNESS"],
                                    sharpe=0.93, fitness=0.49, turnover=0.096, returns=0.034))
        self.assertEqual(code, "weak_signal")

    def test_turnover_excess_when_capping_would_clear(self):
        # F = 1.3 * sqrt(0.12/0.40) = 0.71 -> fails; at the 0.125 floor it is 1.27.
        code, detail = rr.classify(entry(["LOW_FITNESS"], sharpe=1.3, fitness=0.71,
                                         turnover=0.40, returns=0.12))
        self.assertEqual(code, "turnover_excess")
        self.assertIn("turnover floor", detail)

    def test_low_amplitude_when_turnover_is_already_slow(self):
        # F = 1.15 * sqrt(0.0457/0.125) = 0.70, and turnover is below the floor.
        code, detail = rr.classify(entry(["LOW_FITNESS"], sharpe=1.15, fitness=0.70,
                                         turnover=0.0627, returns=0.0457))
        self.assertEqual(code, "low_amplitude")
        self.assertIn("needs returns", detail)

    def test_missing_metrics_is_unclassified_rather_than_guessed(self):
        self.assertEqual(rr.classify(entry(["LOW_FITNESS"]))[0], "unclassified")

    def test_no_checks_is_unclassified(self):
        self.assertEqual(rr.classify({"verdict": "rejected"})[0], "unclassified")

    def test_unmodelled_gate_is_reported_as_itself(self):
        code, detail = rr.classify(entry(["SOMETHING_NEW"], sharpe=1.5, fitness=1.2))
        self.assertEqual(code, "other_gate")
        self.assertIn("SOMETHING_NEW", detail)


if __name__ == "__main__":
    unittest.main()