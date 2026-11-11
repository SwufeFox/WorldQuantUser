"""Pruning the cache against a bar that lives in a file."""

import json
import os
import tempfile
import unittest
from unittest import mock

import research_paths
from research_app import prune


class PruneTests(unittest.TestCase):
    def _workspace(self, rows, cached):
        tmp = tempfile.mkdtemp()
        registry = os.path.join(tmp, "registry.jsonl")
        with open(registry, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        pnl = os.path.join(tmp, "pnl")
        os.makedirs(pnl, exist_ok=True)
        for alpha_id in cached:
            with open(os.path.join(pnl, alpha_id + ".json"), "w") as handle:
                handle.write('{"2019-01-02": 0.0}')
        return tmp, registry, pnl

    def _args(self, **kwargs):
        class Args:
            apply = kwargs.get("apply", False)
            criteria = kwargs.get("criteria")
            set = kwargs.get("set")
            show_criteria = False
            limit = 10
            registry = kwargs.get("registry")
        return Args()

    def test_a_dry_run_deletes_nothing(self):
        rows = [{"alpha_id": "A", "status": "COMPLETE", "expression": "rank(close)",
                 "metrics": {"fitness": 0.4, "sharpe": 0.6, "turnover": 0.1}}]
        tmp, registry, pnl = self._workspace(rows, ["A"])
        with mock.patch.object(research_paths, "ensure_dir", return_value=tmp):
            data = prune.cmd_prune(self._args(registry=registry), None)
        self.assertEqual(data["dropped"], 1)
        self.assertEqual(data["removed"], 0)
        self.assertTrue(os.path.exists(os.path.join(pnl, "A.json")))

    def test_apply_removes_the_failing_entry(self):
        rows = [{"alpha_id": "A", "status": "COMPLETE", "expression": "rank(close)",
                 "metrics": {"fitness": 0.4, "sharpe": 0.6, "turnover": 0.1}}]
        tmp, registry, pnl = self._workspace(rows, ["A"])
        with mock.patch.object(research_paths, "ensure_dir", return_value=tmp):
            data = prune.cmd_prune(self._args(registry=registry, apply=True), None)
        self.assertEqual(data["removed"], 1)
        self.assertFalse(os.path.exists(os.path.join(pnl, "A.json")))

    def test_a_book_member_is_never_dropped(self):
        rows = [{"alpha_id": "B", "status": "ACTIVE", "expression": "rank(close)",
                 "metrics": {"fitness": 0.4, "sharpe": 0.6, "turnover": 0.1}}]
        tmp, registry, pnl = self._workspace(rows, ["B"])
        with mock.patch.object(research_paths, "ensure_dir", return_value=tmp):
            data = prune.cmd_prune(self._args(registry=registry, apply=True), None)
        self.assertEqual(data["dropped"], 0)
        self.assertTrue(os.path.exists(os.path.join(pnl, "B.json")))

    def test_a_criterion_file_moves_the_bar(self):
        rows = [{"alpha_id": "C", "status": "COMPLETE", "expression": "rank(close)",
                 "metrics": {"fitness": 1.1, "sharpe": 1.3, "turnover": 0.1}}]
        tmp, registry, pnl = self._workspace(rows, ["C"])
        criteria = os.path.join(tmp, "criteria.json")
        with open(criteria, "w", encoding="utf-8") as handle:
            json.dump({"min_fitness": 1.2}, handle)
        with mock.patch.object(research_paths, "ensure_dir", return_value=tmp):
            loose = prune.cmd_prune(self._args(registry=registry), None)
            strict = prune.cmd_prune(self._args(registry=registry, criteria=criteria), None)
        self.assertEqual(loose["dropped"], 0)
        self.assertEqual(strict["dropped"], 1)

    def test_a_closed_block_is_dropped_whatever_its_score(self):
        rows = [{"alpha_id": "D", "status": "COMPLETE",
                 "expression": "rank(ts_mean(vec_avg(composite_sentiment_score_2), 20))",
                 "metrics": {"fitness": 1.5, "sharpe": 1.6, "turnover": 0.1}}]
        tmp, registry, pnl = self._workspace(rows, ["D"])
        with mock.patch.object(research_paths, "ensure_dir", return_value=tmp):
            data = prune.cmd_prune(
                self._args(registry=registry, set=["drop_blocks=[\"news_sentiment\"]"]), None)
        self.assertEqual(data["dropped"], 1)


class PruneSafetyTests(unittest.TestCase):
    """A broken criteria file must refuse, not fall back to defaults."""

    def test_unreadable_criteria_refuses_instead_of_using_defaults(self):
        import tempfile
        from research_app import prune
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "criteria.json")
        with open(path, "w") as handle:
            handle.write("{not json")
        with self.assertRaises(ValueError):
            prune.load_criteria(path)

    def test_a_missing_file_still_uses_defaults(self):
        from research_app import prune
        criteria, source = prune.load_criteria("/nonexistent/criteria.json")
        self.assertEqual(source, "defaults")
        self.assertEqual(criteria["min_sharpe"], prune.DEFAULTS["min_sharpe"])
