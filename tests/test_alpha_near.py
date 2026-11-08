#!/usr/bin/env python3
"""Tests for `alpha near`: structural nearest neighbours, before simulating.

PnL correlation needs a simulation; reading 55 expressions by eye is unreliable. The
AST skeleton is the offline signal: an identical skeleton means "same construction,
different grid point", and these tests pin that the search finds those and does not
claim it when the skeleton differs.
"""

import argparse
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import alpha_registry
from research_app import alphas as cli_alphas


def _entry(alpha_id, expression, status="ACTIVE", fitness=1.5, family="quality"):
    return alpha_registry.build_entry(
        {"alpha_id": alpha_id, "expression": expression,
         "metrics": {"fitness": fitness}, "status": status},
        family=family, verdict="candidate")


class AlphaNearTests(unittest.TestCase):
    def _run(self, expression, entries, **overrides):
        tmp = tempfile.mkdtemp()
        registry = os.path.join(tmp, "r.jsonl")
        for item in entries:
            alpha_registry.record_entry(item, registry)
        args = argparse.Namespace(alpha_id=None, expression=expression, status="ACTIVE",
                                  limit=5, registry=registry)
        for key, value in overrides.items():
            setattr(args, key, value)
        return cli_alphas.cmd_alpha_near(args, mock.MagicMock())

    def test_the_same_construction_with_another_field_is_flagged(self):
        # Same skeleton, different field: the classic "same construction, new grid point".
        # Different field, same shape. `cap` is shared, so the field overlap is only
        # 1/3 - the skeleton is what identifies the duplicate, not the fields.
        data = self._run("rank(ebitda / cap)",
                         [_entry("A1", "rank(revenue / cap)")])
        self.assertEqual(data["same_skeleton_count"], 1)
        self.assertTrue(data["near"][0]["same_skeleton"])
        self.assertLess(data["near"][0]["field_jaccard"], 0.5)

    def test_a_different_construction_is_not_flagged_as_same(self):
        data = self._run("rank(ebitda / cap)",
                         [_entry("A1", "rank(ts_delta(close, 5))")])
        self.assertEqual(data["same_skeleton_count"], 0)
        self.assertFalse(data["near"][0]["same_skeleton"])

    def test_only_requested_statuses_are_compared(self):
        data = self._run("rank(ebitda / cap)",
                         [_entry("A1", "rank(revenue / cap)", status="UNSUBMITTED")])
        self.assertEqual(data["count"], 0)


if __name__ == "__main__":
    unittest.main()
