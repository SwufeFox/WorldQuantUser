"""Train-to-full decay: the ratio the selection rule never measured."""

import json
import os
import tempfile
import unittest

from research_app import ledger


class DecayTests(unittest.TestCase):
    def _registry(self, rows):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "r.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def _entry(self, key, expression, train, full, status="COMPLETE"):
        return {"alpha_id": key, "expression": expression, "status": status,
                "metrics": {"sharpe": full}, "out_of_sample": {"train": {"sharpe": train}}}

    def _args(self, registry, **kwargs):
        class Args:
            min_train_sharpe = 0.8
            min_arms = 2
        args = Args()
        for name, value in kwargs.items():
            setattr(args, name, value)
        args.registry = registry
        return args

    def test_a_block_that_loses_half_its_sharpe_shows_ratio_near_half(self):
        rows = [self._entry("A", "rank(assets)", 2.0, 1.0),
                self._entry("B", "rank(atq)", 2.0, 1.0)]
        data = ledger.cmd_ledger_decay(self._args(self._registry(rows)), None)
        block = data["blocks"][0]
        self.assertAlmostEqual(block["median_ratio"], 0.5, places=3)
        self.assertEqual(block["share_below_80pct"], 1.0)

    def test_weak_train_arms_are_skipped(self):
        rows = [self._entry("A", "rank(assets)", 0.3, 0.2),
                self._entry("B", "rank(atq)", 0.3, 0.2)]
        data = ledger.cmd_ledger_decay(self._args(self._registry(rows)), None)
        self.assertEqual(data["blocks"], [])

    def test_the_book_ratio_is_reported_separately(self):
        rows = [self._entry("A", "rank(assets)", 2.0, 1.8, status="ACTIVE"),
                self._entry("B", "rank(atq)", 2.0, 1.0)]
        data = ledger.cmd_ledger_decay(self._args(self._registry(rows)), None)
        block = data["blocks"][0]
        self.assertAlmostEqual(block["book_ratio"], 0.9, places=3)
        self.assertAlmostEqual(block["median_ratio"], 0.7, places=3)
