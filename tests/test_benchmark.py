"""Benchmark contribution, warning and unknown-evidence behavior."""

import unittest

from research_app import benchmark


class BenchmarkTests(unittest.TestCase):

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

    def test_the_block_factor_is_opt_in_and_defaults_to_neutral(self):
        # Off by default, so every stored score keeps its old meaning.
        self.assertEqual(benchmark.block_factor("value_quality", {}), 1.0)
        self.assertEqual(benchmark.block_factor("value_quality", {"block_ratio": 0}), 1.0)

    def test_an_unknown_block_receives_no_penalty(self):
        on = {"block_ratio": 1}
        self.assertEqual(benchmark.block_factor("invented_block", on), 1.0)
        self.assertEqual(benchmark.block_factor(None, on), 1.0)
