#!/usr/bin/env python3
"""Tests for `health`: per-alpha marginal contribution to the book.

Equal weighting pays every member the same, so the only question that decides whether a
slot earns its keep is `S_i - rho_i * S_book`. These tests pin the arithmetic and the
direction: a duplicate of the book must show a negative marginal, an independent alpha a
positive one.
"""

import argparse
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import alpha_registry
import research_paths
from research_app import campaign as cli_campaign


def _series(kind, days=600):
    out = {}
    for index in range(days):
        day = "2020-%04d" % index
        if kind == "steady":
            out[day] = 1.0 if index % 2 else 0.6
        elif kind == "twin":       # same shape, slightly scaled -> duplicate
            out[day] = 1.05 if index % 2 else 0.63
        else:                      # alternating, orthogonal to the steady pattern
            out[day] = -0.5 if index % 2 else 0.9
    return out


class HealthTests(unittest.TestCase):
    def test_annual_sharpe_matches_the_definition(self):
        values = [1.0, -0.2] * 100
        value = cli_campaign._annual_sharpe(values)
        self.assertGreater(value, 0)
        self.assertIsNone(cli_campaign._annual_sharpe([1.0, 1.0]))

    def test_a_duplicate_of_the_book_has_a_negative_marginal(self):
        tmp = tempfile.mkdtemp()
        research = tempfile.mkdtemp()
        os.makedirs(os.path.join(research, "pnl"))
        registry = os.path.join(tmp, "r.jsonl")
        for name, kind in (("S1", "steady"), ("S2", "steady"), ("S3", "steady")):
            alpha_registry.record_entry(
                alpha_registry.build_entry({"alpha_id": name, "expression": "rank(close)",
                                            "metrics": {}, "status": "ACTIVE"}), registry)
            with open(os.path.join(research, "pnl", "%s.json" % name), "w") as handle:
                json.dump(_series(kind), handle)
        args = argparse.Namespace(limit=8, min_overlap=100, registry=registry)
        with mock.patch.object(research_paths, "ensure_dir", lambda: research):
            data = cli_campaign.cmd_health(args, mock.MagicMock())
        # Three copies of one series: every member's rho against the rest is 1, so the
        # marginal is exactly S - S = 0 for all of them, never positive.
        self.assertEqual(data["count"], 3)
        self.assertTrue(all(row["marginal"] <= 0.01 for row in data["weakest"]))

    def test_the_default_window_stops_before_the_test_year(self):
        # Ranking a pool on the full window would be ranking on the test year.
        tmp = tempfile.mkdtemp()
        research = tempfile.mkdtemp()
        os.makedirs(os.path.join(research, "pnl"))
        registry = os.path.join(tmp, "r.jsonl")
        import datetime
        def _dates(year, count):
            start = datetime.date(year, 1, 1)
            return [(start + datetime.timedelta(days=k)).isoformat() for k in range(count)]
        series = {}
        for index, day in enumerate(_dates(2021, 400)):
            series[day] = 1.0 if index % 2 else 0.4
        for day in _dates(2023, 100):
            series[day] = -5.0
        alpha_registry.record_entry(
            alpha_registry.build_entry({"alpha_id": "W1", "expression": "rank(close)",
                                        "metrics": {}, "status": "ACTIVE"}), registry)
        with open(os.path.join(research, "pnl", "W1.json"), "w") as handle:
            json.dump(series, handle)
        args = argparse.Namespace(limit=4, min_overlap=100, registry=registry,
                                  window="train", train_end="2023-01-01", pool=False)
        with mock.patch.object(research_paths, "ensure_dir", lambda: research):
            data = cli_campaign.cmd_health(args, mock.MagicMock())
        self.assertEqual(data["days"], 400)
        self.assertEqual(data["window"], "train")


if __name__ == "__main__":
    unittest.main()
