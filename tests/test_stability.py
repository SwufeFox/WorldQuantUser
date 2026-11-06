#!/usr/bin/env python3
"""Tests for `ledger stability`: does a train Sharpe live in both halves?

The verb exists because a single train number hid the fact that a candidate's edge was
one regime. These tests pin the two states the book is judged on: a stable alpha and a
sign-flipping one, plus the late-bloomer shape that neither call should label a flip.
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

import research_paths
from research_app import ledger as cli_ledger


def _series(days, first, second):
    out = {}
    for index, day in enumerate(days):
        out[day] = first if index % 2 == 0 else second
    return out


class StabilityTests(unittest.TestCase):
    def _run(self, series, min_move=0.3):
        tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp, "pnl"), exist_ok=True)
        entries = []
        for name, pnl in series.items():
            with open(os.path.join(tmp, "pnl", "%s.json" % name), "w") as handle:
                json.dump(pnl, handle)
            entries.append({"alpha_id": name, "status": "ACTIVE",
                            "out_of_sample": {"test": {"startDate": "2023-01-01"}}})
        args = argparse.Namespace(where=None, status="ACTIVE", min_move=min_move,
                                  limit=10, list=True, registry=None)
        with mock.patch.object(cli_ledger, "_ledger", lambda a: (None, entries)), \
                mock.patch.object(research_paths, "ensure_dir", return_value=tmp):
            return cli_ledger.cmd_ledger_stability(args, mock.MagicMock())

    def test_a_steady_alpha_is_not_a_flip(self):
        days = ["2020-%03d" % day for day in range(1, 200)]
        data = self._run({"STEADY": _series(days, 1.0, 0.5)})
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["flips"], 0)

    def test_a_sign_flip_between_halves_is_reported(self):
        first = ["2020-%03d" % day for day in range(1, 100)]
        second = ["2021-%03d" % day for day in range(1, 100)]
        flipping = {}
        for day in first:
            flipping[day] = 9.0 if int(day[-3:]) % 2 else 3.0
        for day in second:
            flipping[day] = -9.0 if int(day[-3:]) % 2 else -3.0
        data = self._run({"FLIP": flipping})
        self.assertEqual(data["flips"], 1)
        self.assertLess(data["flip_list"][0]["h1"] * data["flip_list"][0]["h2"], 0)

    def test_a_quiet_first_half_is_reported_as_a_gap_not_a_flip(self):
        first = ["2020-%03d" % day for day in range(1, 100)]
        second = ["2021-%03d" % day for day in range(1, 100)]
        late = {day: (0.1 if int(day[-3:]) % 2 else -0.1) for day in first}
        late.update({day: (12.0 if int(day[-3:]) % 2 else 4.0) for day in second})
        data = self._run({"LATE": late})
        self.assertEqual(data["flips"], 0)
        self.assertGreater(data["worst"][0]["gap"], 1.0)


if __name__ == "__main__":
    unittest.main()
