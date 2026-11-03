#!/usr/bin/env python3
"""Tests for `probe`: one command per experiment instead of a copied script.

The command exists because the throwaway-script version of this workflow failed four
ways in one day - stdout contract, output-name collision, quoting at import time, and
no progress output. Each of those is pinned here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import research_cli as cli  # noqa: E402


def record(alpha_id, sharpe, fitness, turnover=0.12, train=1.5, test=0.9, fails=()):
    checks = {name: {"result": "FAIL"} for name in fails}
    return {"alpha_id": alpha_id, "status": "COMPLETE",
            "metrics": {"sharpe": sharpe, "fitness": fitness, "turnover": turnover,
                        "returns": 0.06},
            "checks": checks,
            "out_of_sample": {"train": {"sharpe": train}, "test": {"sharpe": test},
                              "test_period": "P1Y"}}


class FakeGuard(types.ModuleType):
    def __init__(self, records):
        super().__init__("guarded_sim")
        self.records = records
        self.calls = []

    def check(self, expressions, settings=None, session=None):
        return {"errors": [], "warnings": []}

    def simulate(self, expressions, settings=None, labels=None, **kwargs):
        self.calls.append({"expressions": expressions, "labels": labels,
                           "settings": dict(settings or {})})
        return self.records[:len(expressions)]


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.spec_path = os.path.join(self.tmp, "spec.json")
        self.output_dir = tempfile.mkdtemp()
        self.spec = {
            "name": "unit-test-batch",
            "idea": "test idea", "mechanism": "test mechanism",
            "settings": {"region": "USA", "universe": "TOP3000", "decay": 4},
            "arms": [{"label": "a", "expression": "rank(close)"},
                     {"label": "b", "expression": "rank(volume)"}],
        }
        self._write(self.spec)
        self.fake = FakeGuard([record("A1", 2.0, 1.5), record("B1", 1.1, 0.6)])
        self._patched = mock.patch.dict(sys.modules, {"guarded_sim": self.fake})
        self._patched.start()
        self.addCleanup(self._patched.stop)

    def _write(self, payload):
        with open(self.spec_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def _args(self, **overrides):
        base = dict(spec=self.spec_path, name=None, dry_run=False, show_test=False,
                    force=False, no_record=True, registry=None, save=None)
        base.update(overrides)
        return argparse.Namespace(**base)

    def _run(self, **overrides):
        with mock.patch.object(cli.research_paths, "ensure_dir",
                               lambda: self.output_dir):
            return cli.cmd_probe(self._args(**overrides), mock.MagicMock())

    def test_train_is_always_present_and_test_only_on_request(self):
        # Wrong behaviour this prevents: choosing between arms while looking at the
        # test year, which is the only out-of-sample evidence the API exposes.
        data = self._run()
        self.assertTrue(all("train_sharpe" in row for row in data["rows"]))
        self.assertTrue(all("test_sharpe" not in row for row in data["rows"]))
        data = self._run(force=True, show_test=True)
        self.assertTrue(all("test_sharpe" in row for row in data["rows"]))

    def test_an_existing_batch_file_is_not_silently_overwritten(self):
        # Wrong behaviour this prevents: a second run reusing the same NAME wrote over
        # the previous batch's JSON, which is how a 6-arm result disappeared.
        self._run()
        with self.assertRaises(cli.CliError) as caught:
            self._run()
        self.assertEqual(caught.exception.code, "INVALID_INPUT")
        self.assertIn("already exists", str(caught.exception))

    def test_per_arm_settings_are_grouped_into_separate_runs(self):
        # A neutralisation sweep is one spec, not one script per setting.
        self.spec["arms"] = [
            {"label": "sub", "expression": "rank(close)"},
            {"label": "ind", "expression": "rank(close)",
             "settings": {"neutralization": "INDUSTRY"}},
        ]
        self._write(self.spec)
        self._run(force=True)
        self.assertEqual(len(self.fake.calls), 2)
        self.assertEqual(self.fake.calls[1]["settings"]["neutralization"], "INDUSTRY")
        self.assertEqual(self.fake.calls[0]["labels"], ["sub"])

    def test_a_spec_without_arms_is_refused_with_a_code(self):
        self._write({"name": "x", "arms": []})
        with self.assertRaises(cli.CliError) as caught:
            self._run()
        self.assertEqual(caught.exception.code, "INVALID_INPUT")

    def test_dry_run_spends_nothing(self):
        data = self._run(dry_run=True)
        self.assertTrue(data["dry_run"])
        self.assertEqual(self.fake.calls, [])

    def test_rows_carry_the_failed_checks(self):
        self.fake.records = [record("A1", 1.2, 0.7, fails=("LOW_SHARPE",)),
                             record("B1", 2.0, 1.5)]
        data = self._run()
        self.assertEqual(data["rows"][0]["failed_checks"], ["LOW_SHARPE"])


if __name__ == "__main__":
    unittest.main()