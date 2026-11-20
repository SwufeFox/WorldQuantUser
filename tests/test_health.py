#!/usr/bin/env python3
"""Tests for `health`: per-alpha marginal contribution to the book.

Equal weighting pays every member the same, so the only question that decides whether a
slot earns its keep is `S_i - rho_i * S_book`. These tests pin the arithmetic and the
direction: a duplicate of the book must show a negative marginal, an independent alpha a
positive one.
"""

import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path
from random import Random
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
    def _health(self, rows, **options):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pnl_dir = root / "pnl"
            pnl_dir.mkdir()
            registry = root / "registry.jsonl"
            entries = []
            for name, status, series, test_start in rows:
                entry = {"alpha_id": name, "status": status}
                if test_start is not None:
                    entry["out_of_sample"] = {"test": {"startDate": test_start}}
                entries.append(entry)
                (pnl_dir / f"{name}.json").write_text(json.dumps(series), encoding="utf-8")
            registry.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
            args = argparse.Namespace(registry=str(registry), limit=8, min_overlap=120, pool=True,
                                      gate=0.7, min_marginal=-100, min_sharpe=-100, **options)
            with mock.patch.object(research_paths, "ensure_dir", return_value=directory):
                return cli_campaign.cmd_health(args, None)

    def test_health_uses_recorded_test_starts_for_book_and_candidate(self):
        days = [(date(2020, 1, 1) + timedelta(days=index)).isoformat() for index in range(600)]
        values = {day: float(index % 3) for index, day in enumerate(days)}
        candidate = {day: 2 * value if day < days[360] else -10000.0 for day, value in values.items()}
        data = self._health([
            ("book", "ACTIVE", values, days[500]),
            ("candidate", "UNSUBMITTED", candidate, days[360]),
        ], window="train", train_end="2023-01-01")
        self.assertEqual(data["train_end"], days[500])
        self.assertEqual(data["days"], 500)
        row = data["pool"][0]
        self.assertEqual((row["train_end"], row["overlap_days"]), (days[360], 360))
        self.assertAlmostEqual(row["max_single"], 1.0)

    def test_rest_keeps_small_members_beside_a_large_one(self):
        days = [(date(2020, 1, 1) + timedelta(days=index)).isoformat() for index in range(120)]
        large = {day: 1e16 * (index % 3 + 1) for index, day in enumerate(days)}
        small = {day: float((-1) ** index) for index, day in enumerate(days)}
        data = self._health([
            ("large", "ACTIVE", large, None), ("small", "ACTIVE", small, None),
        ])
        self.assertEqual(data["count"], 2)
        self.assertTrue(all(row["delta"] is not None for row in data["strongest"]))

    def test_pool_gate_uses_the_same_daily_correlation_as_the_report(self):
        rng = Random(184)
        book, candidate = {}, {}
        for index in range(600):
            day = (date(2020, 1, 1) + timedelta(days=index)).isoformat()
            value = rng.uniform(-4, 4) if index % 2 else rng.uniform(-1, 1)
            book[day] = value
            candidate[day] = 2 + (value if index % 2 else rng.uniform(-1, 1))
        with mock.patch.object(cli_campaign.pnl_snapshot, "read_series",
                               wraps=cli_campaign.pnl_snapshot.read_series) as read:
            data = self._health([
                ("book", "ACTIVE", book, None), ("candidate", "UNSUBMITTED", candidate, None),
            ])
        self.assertEqual([call.args[0].stem for call in read.call_args_list], ["book", "candidate"])
        self.assertGreater(data["pool"][0]["max_single"], 0.9)
        self.assertFalse(data["pool"][0]["passes"])
        self.assertEqual(data["pool_passing"], 0)

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
