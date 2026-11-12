"""The internal bar: grounded factors, and the two that were rejected."""

import unittest

from research_app import benchmark


class BenchmarkTests(unittest.TestCase):
    def test_the_shrink_factor_follows_the_measured_hump(self):
        # Retention by train-Sharpe bucket was 0.41 / 0.77 / 0.85 / 0.64 / 0.49.
        self.assertAlmostEqual(benchmark.shrink_factor(0.8), 0.48)
        self.assertAlmostEqual(benchmark.shrink_factor(1.2), 0.91)
        self.assertAlmostEqual(benchmark.shrink_factor(1.7), 1.00)
        self.assertAlmostEqual(benchmark.shrink_factor(2.2), 0.75)
        self.assertAlmostEqual(benchmark.shrink_factor(3.0), 0.58)

    def test_a_dilutive_alpha_scores_zero_however_good_its_own_fitness(self):
        self.assertEqual(benchmark.contribution_factor(-0.05, 0.05), 0.0)
        self.assertEqual(benchmark.contribution_factor(0.10, 0.05), 1.0)
        self.assertAlmostEqual(benchmark.contribution_factor(0.025, 0.05), 0.5)

    def test_a_weak_block_is_discounted_but_never_below_half(self):
        self.assertLess(benchmark.retention_factor(0.5), benchmark.retention_factor(1.0))
        self.assertGreaterEqual(benchmark.retention_factor(0.1), 0.5)

    def test_long_expressions_warn_without_scoring(self):
        self.assertIsNone(benchmark.ops_warning(4, [8, 16, 32]))
        self.assertIsNotNone(benchmark.ops_warning(20, [8, 16, 32]))
        self.assertIn("unauditable", benchmark.ops_warning(50, [8, 16, 32]))

    def test_the_block_factor_is_opt_in_and_defaults_to_neutral(self):
        # Off by default, so every stored score keeps its old meaning.
        self.assertEqual(benchmark.block_factor("value_quality", {}), 1.0)
        self.assertEqual(benchmark.block_factor("value_quality", {"block_ratio": 0}), 1.0)

    def test_the_block_factor_reads_the_measured_ratios(self):
        on = {"block_ratio": 1}
        # value_quality median test/train is 0.21 against a 0.75 baseline -> 0.28.
        self.assertAlmostEqual(benchmark.block_factor("value_quality", on), 0.28, places=2)
        # analyst_dispersion is 1.74 / 0.75 = 2.32, clipped to 2.0.
        self.assertAlmostEqual(benchmark.block_factor("analyst_dispersion", on), 2.0)
        # A block that is absent from the table is not evidence of decay.
        self.assertEqual(benchmark.block_factor("invented_block", on), 1.0)
        self.assertEqual(benchmark.block_factor(None, on), 1.0)
