"""One Sharpe, one place: the convention must not fork between callers."""

import argparse
import unittest
from unittest.mock import patch

from research_app import alphas, experiments, legs, pairs, stats


class StatsTests(unittest.TestCase):
    def test_alpha_explain_rebuilds_negative_returns_with_the_turnover_floor(self):
        candidate = {"alpha_id": "NEGATIVE", "expression": "rank(close)",
                     "metrics": {"sharpe": -1.2, "returns": -0.04, "turnover": 0.10, "fitness": -0.679}}
        args = argparse.Namespace(alpha_id="NEGATIVE", no_corr=True)
        with patch("alpha_registry.load_entries", return_value=[candidate]), \
                patch("alpha_unit.load_field_datasets", return_value={}), \
                patch.object(alphas, "_closed_reason", return_value=None):
            result = alphas.cmd_alpha_explain(args, None)["fitness_arithmetic"]
        self.assertTrue(result["below_turnover_floor"])
        self.assertEqual(result["implied_fitness"], -0.679)
        self.assertEqual(result["gap"], 0)

    def test_the_canonical_sharpe_matches_the_legacy_helpers(self):
        values = [0.01, -0.02, 0.015, 0.0, 0.005] * 40
        self.assertAlmostEqual(stats.sharpe(values), legs.sharpe_of(values), places=9)

    def test_a_short_series_is_not_measured(self):
        self.assertIsNone(stats.sharpe([0.01] * 5))
        self.assertIsNone(legs.sharpe_of([0.01] * 5))

    def test_a_flat_series_has_no_sharpe(self):
        self.assertIsNone(stats.sharpe([0.0] * 50))

    def test_pairs_and_legs_agree_on_one_row(self):
        values = [0.02, -0.01, 0.03, -0.02, 0.01] * 30
        matrix = __import__("numpy").array([values])
        self.assertAlmostEqual(pairs._sharpe(matrix)[0], legs.sharpe_of(values), places=9)

    def test_effective_bets_uses_the_mean_pairwise_correlation(self):
        self.assertAlmostEqual(stats.effective_bets([0.25] * 10, 10), 10 / (1 + 9 * 0.25), places=2)
        self.assertIsNone(stats.effective_bets([], 5))
