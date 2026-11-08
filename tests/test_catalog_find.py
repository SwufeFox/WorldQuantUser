#!/usr/bin/env python3
"""Tests for `catalog find`: one search across every dataset.

`catalog fields` takes a single dataset id, so asking "is there a short-term debt field
anywhere?" meant paging fourteen datasets by hand. The search itself is the same local
ranking; what this adds is the loop plus a tie-break that reflects what can actually
carry an alpha (coverage), which these tests pin.
"""

import argparse
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import machine_lib as ml
from research_app import catalog as cli_catalog


def _field(field_id, coverage, description="", kind="MATRIX"):
    return {"id": field_id, "description": description, "type": kind,
            "coverage": coverage}


class CatalogFindTests(unittest.TestCase):
    def setUp(self):
        self.datasets = [
            {"id": "fundamental6", "category": {"name": "Fundamental"}},
            {"id": "fundamental2", "category": {"name": "Fundamental"}},
        ]
        self.fields = {
            "fundamental6": [_field("debt", 0.50, "Total debt"),
                             _field("assets", 0.98, "Total assets")],
            "fundamental2": [_field("debt_repayments_total", 0.79, "Repayments on debt")],
        }

    def _args(self, **overrides):
        base = dict(query="debt", dataset=None, min_coverage=None, fuzzy=False,
                    any_term=False, region="USA", universe="TOP3000", delay=1,
                    instrument_type="EQUITY", limit=10)
        base.update(overrides)
        return argparse.Namespace(**base)

    def _run(self, **overrides):
        with mock.patch.object(cli_catalog, "with_session", lambda a, work: work(object())):
            with mock.patch.object(ml, "get_datasets", return_value=self.datasets), \
                    mock.patch.object(ml, "get_datafields",
                                      side_effect=lambda s, **kw: self.fields[kw["dataset_id"]]):
                return cli_catalog.cmd_catalog_find(self._args(**overrides), mock.MagicMock())

    def test_it_searches_every_dataset_and_reports_the_source(self):
        data = self._run()
        self.assertEqual(data["datasets_scanned"], 2)
        self.assertEqual(data["fields_scanned"], 3)
        datasets = {row["dataset"] for row in data["fields"]}
        self.assertEqual(datasets, {"fundamental6", "fundamental2"})

    def test_low_coverage_matches_are_dropped_on_request(self):
        data = self._run(min_coverage=0.6)
        ids = [row["id"] for row in data["fields"]]
        self.assertNotIn("debt", ids)          # 0.50 < 0.6
        self.assertIn("debt_repayments_total", ids)

    def test_ranking_prefers_an_exact_id_then_coverage(self):
        data = self._run()
        self.assertEqual(data["fields"][0]["id"], "debt")

    def test_a_dataset_filter_limits_the_scan(self):
        data = self._run(dataset="fundamental2")
        self.assertEqual(data["datasets_scanned"], 1)
        self.assertEqual({row["dataset"] for row in data["fields"]}, {"fundamental2"})


if __name__ == "__main__":
    unittest.main()
