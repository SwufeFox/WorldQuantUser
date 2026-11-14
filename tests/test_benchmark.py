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


class FrozenBenchmarkTests(unittest.TestCase):
    def setUp(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from research_app import selection

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.registry = self.root / "ledger.jsonl"
        self.pnl = self.root / "pnl"
        self.pnl.mkdir()
        self.frozen = self.root / "selection.json"
        self.rows = [{"alpha_id": "first", "train_fitness": 2},
                     {"alpha_id": "second", "train_fitness": 1}]
        self.manifest = selection.freeze(
            self.frozen, source="triage", rows=self.rows, criteria=benchmark.DEFAULTS,
            inputs={"registry_path": str(self.registry), "pnl_dir": str(self.pnl)},
            window={"mode": "train", "end": "2021-01-01"})
        self.registry.write_text("\n".join(json.dumps(row) for row in [
            {"alpha_id": "second", "metrics": {"fitness": 100},
             "out_of_sample": {"train": {"sharpe": 2}}},
            {"alpha_id": "first", "metrics": {"fitness": -5},
             "out_of_sample": {"train": {"sharpe": 2}}},
            {"alpha_id": "extra", "metrics": {"fitness": 999}},
        ]), encoding="utf-8")
        patcher = patch("research_paths.log_view")
        self.log = patcher.start()
        self.addCleanup(patcher.stop)

    def test_validation_requires_a_manifest(self):
        with self.assertRaisesRegex(ValueError, "requires --selection"):
            benchmark.benchmark_rows(["first"])

    def test_validation_annotates_training_order_and_records_bound_access(self):
        result = benchmark.benchmark_rows(["second", "first"], selection_path=self.frozen)
        self.assertEqual([row["alpha_id"] for row in result["rows"]], ["first", "second"])
        self.assertLess(result["rows"][0]["internal_fitness"], result["rows"][1]["internal_fitness"])
        self.assertEqual(result["frozen_window"], self.manifest["window"])
        self.assertEqual(self.log.call_args.kwargs["note"]["selection_id"], self.manifest["selection_id"])

    def test_absent_candidate_is_not_filled_from_training_snapshot(self):
        self.registry.write_text('{"alpha_id":"second","metrics":{"fitness":7}}\n', encoding="utf-8")
        result = benchmark.benchmark_rows(selection_path=self.frozen)
        self.assertTrue(result["rows"][0]["missing"])
        self.assertNotIn("train_fitness", result["rows"][0])
        self.assertNotIn("internal_fitness", result["rows"][0])
        self.assertEqual(result["rows"][1]["platform_fitness"], 7)

    def test_inputs_and_criteria_cannot_override_frozen_selection(self):
        for kwargs in ({"ids": ["extra"]}, {"pnl_dir": str(self.root / "other")},
                       {"registry_path": str(self.root / "other.jsonl")},
                       {"criteria": {"delta_full_credit": 100}}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "conflict"):
                benchmark.benchmark_rows(selection_path=self.frozen, **kwargs)
