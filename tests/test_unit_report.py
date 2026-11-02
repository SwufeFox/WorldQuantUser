#!/usr/bin/env python3
"""Tests for the dataset view of the unit layer.

The view exists because a book-level question was mistaken for a ledger-level one:
"the book has no model51" was read as "model51 is unexplored", and six probes were
simulated for shapes the ledger already held - one of them byte-identical. These
tests pin the two things that make the view usable: a dataset id resolves to the
family the units are grouped by, and a family nobody has touched says zero instead
of crashing.
"""

from __future__ import annotations

import io
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import alpha_registry  # noqa: E402
import alpha_unit as au  # noqa: E402
import unit_report  # noqa: E402


def entry(alpha_id, expression, status="COMPLETE", fitness=0.5):
    return alpha_registry.build_entry(
        {"alpha_id": alpha_id, "expression": expression,
         "settings": {"region": "USA"},
         "metrics": {"fitness": fitness, "sharpe": 0.4}, "status": status})


class DatasetViewTests(unittest.TestCase):
    def _corpus(self, entries, datasets):
        return au.build(entries, datasets=datasets)

    def test_a_dataset_id_resolves_to_its_family(self):
        # model51 is the dataset; "model" is the family a unit is grouped by. Asking
        # for the id has to answer, or the report silently returns nothing and looks
        # like an empty space.
        corpus = self._corpus([entry("M1", "rank(-beta_last_360_days_spy)")],
                              {"beta_last_360_days_spy": "model51"})
        buffer = io.StringIO()
        result = unit_report.dataset_view(corpus, "model51", stream=buffer)
        self.assertEqual(result["dataset"], "model")
        self.assertEqual(result["expressions"], 1)
        self.assertEqual(result["best_fitness"], 0.5)
        self.assertIn("model51 → family model", buffer.getvalue())

    def test_an_untouched_family_reports_zero_instead_of_failing(self):
        corpus = self._corpus([entry("M1", "rank(-beta_last_360_days_spy)")],
                              {"beta_last_360_days_spy": "model51"})
        result = unit_report.dataset_view(corpus, "social", stream=io.StringIO())
        self.assertEqual(result["expressions"], 0)
        self.assertIsNone(result["best_fitness"])

    def test_book_membership_is_counted_separately(self):
        corpus = self._corpus(
            [entry("M1", "rank(-beta_last_360_days_spy)", status="ACTIVE"),
             entry("M2", "rank(correlation_last_360_days_spy)")],
            {"beta_last_360_days_spy": "model51",
             "correlation_last_360_days_spy": "model51"})
        result = unit_report.dataset_view(corpus, "model", stream=io.StringIO())
        self.assertEqual(result["expressions"], 2)
        self.assertEqual(result["in_book"], 1)


if __name__ == "__main__":
    unittest.main()