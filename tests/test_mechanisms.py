#!/usr/bin/env python3
"""Tests for the mechanism registry and the convergence monitor.

The campaign counts candidates by independent mechanism. These tests pin the key
(`(block, skeleton)`, so a field swap or a window change is still one mechanism) and the
trigger (outcome-based, with a Sharpe floor - a dead arm with no correlation posts a
positive marginal and must not count as progress).
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research_app import mechanisms


def _entry(alpha_id, expression, status="ACTIVE", fitness=1.5, train_sharpe=1.5,
           idea="H-1", family="quality"):
    return {"alpha_id": alpha_id, "expression": expression, "status": status,
            "family": family, "idea": idea, "recorded_at": "2026-11-09T00:00:00Z",
            "metrics": {"fitness": fitness},
            "out_of_sample": {"train": {"sharpe": train_sharpe}}}


class MechanismKeyTests(unittest.TestCase):
    def test_a_window_change_is_the_same_mechanism(self):
        left = mechanisms.mechanism_key("rank(ts_delta(close, 5))")
        right = mechanisms.mechanism_key("rank(ts_delta(close, 21))")
        self.assertEqual(left, right)

    def test_a_different_field_in_the_same_block_is_the_same_mechanism(self):
        left = mechanisms.mechanism_key("rank(revenue / assets)")
        right = mechanisms.mechanism_key("rank(cogs / assets)")
        self.assertEqual(left[0], right[0])          # same economic block
        self.assertEqual(left, right)                # and same skeleton

    def test_a_different_construction_is_a_different_mechanism(self):
        left = mechanisms.mechanism_key("rank(revenue / assets)")
        right = mechanisms.mechanism_key("ts_corr(abs(returns), ts_delay(abs(returns), 1))")
        self.assertNotEqual(left, right)


class SummariseTests(unittest.TestCase):
    def test_count_is_by_mechanism_not_by_expression(self):
        entries = [_entry("A1", "rank(revenue / assets)"),
                   _entry("A2", "rank(cogs / assets)"),          # same mechanism
                   _entry("A3", "ts_rank(close, 252)")]          # different
        data = mechanisms.summarise(entries)
        self.assertEqual(data["members"], 3)
        self.assertEqual(data["mechanisms"], 2)
        self.assertEqual(data["rows"][0]["members"], 2)


class ConvergenceTests(unittest.TestCase):
    def test_a_dead_arm_with_low_correlation_does_not_count_as_progress(self):
        entries = [_entry("B1", "rank(revenue / assets)")]
        # train Sharpe 0.4 and no correlation -> marginal 0.4 >= 0.3, but useless.
        candidate = _entry("C1", "ts_rank(close, 252)", status="COMPLETE", train_sharpe=0.4)
        marginals = {"B1": 0.5, "C1": 0.5}
        data = mechanisms.convergence(entries + [candidate], marginals, recent_batches=5)
        self.assertEqual(data["recent_additive"], 1)      # only the book member

    def test_a_strong_independent_candidate_counts(self):
        entries = [_entry("B1", "rank(revenue / assets)")]
        candidate = _entry("C1", "ts_rank(close, 252)", status="COMPLETE", train_sharpe=1.8)
        data = mechanisms.convergence(entries + [candidate], {"B1": 0.5, "C1": 0.6},
                                      recent_batches=5)
        self.assertEqual(data["recent_additive"], 2)


if __name__ == "__main__":
    unittest.main()


class SnapshotTests(unittest.TestCase):
    def test_brief_reads_the_snapshot_instead_of_computing(self):
        import tempfile
        from unittest import mock
        import research_paths
        from research_app import campaign as cli_campaign

        tmp = tempfile.mkdtemp()
        with mock.patch.object(research_paths, "ensure_dir", lambda: tmp):
            payload = cli_campaign._convergence_snapshot()
        self.assertTrue(payload["stale"])
        self.assertIn("hint", payload)

    def test_a_fresh_snapshot_is_not_stale(self):
        import json, os, tempfile
        from datetime import datetime, timezone
        from unittest import mock
        import research_paths
        from research_app import campaign as cli_campaign

        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "convergence.json"), "w") as handle:
            json.dump({"as_of": datetime.now(timezone.utc).isoformat(),
                       "recent_productive": 3, "recent_additive": 5,
                       "book_mechanisms": 36}, handle)
        with mock.patch.object(research_paths, "ensure_dir", lambda: tmp):
            payload = cli_campaign._convergence_snapshot()
        self.assertFalse(payload["stale"])
        self.assertEqual(payload["recent_productive"], 3)


class NoveltyGateTests(unittest.TestCase):
    def test_an_expression_already_in_the_ledger_is_flagged(self):
        import tempfile
        from unittest import mock
        import research_paths
        from research_app import experiments as cli_experiments

        tmp = tempfile.mkdtemp()
        registry = os.path.join(tmp, "r.jsonl")
        import alpha_registry
        alpha_registry.record_entry(
            alpha_registry.build_entry({"alpha_id": "A1", "expression": "rank(close)",
                                        "metrics": {}, "status": "ACTIVE"}), registry)
        with mock.patch.object(alpha_registry, "default_registry_path",
                               return_value=registry), \
                mock.patch.object(research_paths, "ensure_dir", return_value=tmp):
            verdicts = cli_experiments._novelty_gate(["rank( close )", "rank(volume)"])
        self.assertTrue(verdicts[0]["already_simulated"])
        self.assertFalse(verdicts[1]["already_simulated"])
