"""Level vs change, measured rather than argued."""

import json
import os
import tempfile
import unittest

from research_app import ledger


class FormTests(unittest.TestCase):
    def _registry(self, rows):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "r.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def _entry(self, expression, sharpe, status="COMPLETE"):
        return {"alpha_id": "A%d" % (abs(hash(expression)) % 10000),
                "expression": expression, "status": status,
                "metrics": {"sharpe": sharpe}}

    def _args(self, **kwargs):
        class Args:
            min_arms = 4
            min_per_form = 2
            include_book = kwargs.get("include_book", False)
            registry = kwargs.get("registry")
        return Args()

    def test_change_wins_when_its_arms_are_more_reliably_positive(self):
        # Same information block on purpose: the comparison is within a block, so arms
        # from two blocks would each fall below the floor and be dropped.
        rows = [self._entry("rank(assets)", 1.0), self._entry("rank(atq)", 0.9),
                self._entry("rank(ts_delta(assets, 63))", 1.8),
                self._entry("rank(ts_delta(atq, 63))", 1.7)]
        data = ledger.cmd_ledger_form(self._args(registry=self._registry(rows)), None)
        block = data["blocks"][0]
        self.assertEqual(block["verdict"], "change")
        self.assertEqual(block["change"]["n"], 2)

    def test_book_members_are_excluded_by_default(self):
        rows = [self._entry("rank(assets)", 2.0, status="ACTIVE"),
                self._entry("rank(ts_delta(assets, 63))", 1.8)]
        data = ledger.cmd_ledger_form(self._args(registry=self._registry(rows)), None)
        self.assertTrue(data["book_excluded"])
        for block in data["blocks"]:
            self.assertLess(block["level"]["n"] + block["change"]["n"], 2)

    def test_a_block_below_the_floor_is_dropped(self):
        rows = [self._entry("rank(close)", 1.0)]
        data = ledger.cmd_ledger_form(self._args(registry=self._registry(rows)), None)
        self.assertEqual(data["blocks"], [])
