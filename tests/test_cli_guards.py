#!/usr/bin/env python3
"""Tests for two unreachable-but-real code paths found by static checking.

Both are the same species: a name that was renamed somewhere and left behind at one
call site. Nothing imports them, so nothing fails until a user does, and then the
error is a ``NameError`` instead of the documented envelope with a stable
``error.code`` - which is exactly what AGENTS.md promises will not happen.

The rule this pins: every failure a command can reach has to leave as a ``CliError``
carrying a code, because callers branch on the code and never on the message.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import research_cli as cli  # noqa: E402


class PaperSaveTests(unittest.TestCase):
    def test_save_writes_the_file_instead_of_raising_name_error(self):
        # Wrong behaviour this prevents: `paper "..." --save out.json` died with
        # NameError: _write_json, after the network call had already been paid for.
        target = os.path.join(tempfile.mkdtemp(), "papers.json")
        args = argparse.Namespace(query="accruals", limit=5, from_year=None,
                                  min_citations=0, save=target)
        payload = {"results": [{"title": "Anomalies", "year": 1996}]}
        with mock.patch.object(cli.paper_search, "search_papers", return_value=payload):
            data = cli.cmd_paper(args, mock.MagicMock())
        self.assertEqual(data["saved_to"], target)
        self.assertTrue(os.path.exists(target))
        with open(target, encoding="utf-8") as handle:
            self.assertIn("Anomalies", handle.read())


class NegativeMaxWaitTests(unittest.TestCase):
    def test_alpha_check_refuses_negative_wait_with_a_code(self):
        args = argparse.Namespace(max_wait=-1, alpha_id="X", save=None)
        with self.assertRaises(cli.CliError) as caught:
            cli.cmd_alpha_check(args, mock.MagicMock())
        self.assertEqual(caught.exception.code, "USAGE")

    def test_alpha_submit_refuses_negative_wait_with_a_code(self):
        args = argparse.Namespace(max_wait=-5, alpha_id="X", save=None)
        with self.assertRaises(cli.CliError) as caught:
            cli.cmd_alpha_submit(args, mock.MagicMock())
        self.assertEqual(caught.exception.code, "USAGE")


def result_record(alpha_id="A1", sharpe=2.0, test_sharpe=1.4):
    return {"alpha_id": alpha_id, "expression": "rank(close)",
            "metrics": {"sharpe": sharpe, "fitness": 1.5, "turnover": 0.15},
            "checks": {"LOW_SHARPE": {"result": "PASS"}},
            "out_of_sample": {"train": {"sharpe": sharpe + 0.1},
                              "test": {"sharpe": test_sharpe}}}


class CompareShapeTests(unittest.TestCase):
    """`compare` has to read what this repository actually writes."""

    def _run(self, *payloads):
        paths = []
        for payload in payloads:
            path = os.path.join(tempfile.mkdtemp(), "batch.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            paths.append(path)
        args = argparse.Namespace(files=paths, min_fitness=None, sort="sharpe",
                                  contributions=False, save=None, frozen=True)
        return cli.cmd_compare(args, mock.MagicMock())

    def test_a_bare_list_is_accepted(self):
        # Wrong behaviour this prevents: the research template writes a bare list of
        # records, and `compare` refused its own repository's output.
        data = self._run([result_record()])
        self.assertEqual(data["count"], 1)
        self.assertAlmostEqual(data["rows"][0]["test_sharpe"], 1.4)

    def test_the_envelope_shape_is_still_accepted(self):
        data = self._run({"data": {"results": [result_record("A1"),
                                              result_record("A2", sharpe=1.1)]}})
        self.assertEqual([row["alpha_id"] for row in data["rows"]], ["A1", "A2"])

    def test_an_already_tabulated_file_is_refused_with_a_reason(self):
        # `compare`'s own rows carry `sharpe` at the top level. Reading them as
        # records would print a table of nulls instead of saying what is wrong.
        rows = {"files": ["x.json"], "count": 1,
                "rows": [{"alpha_id": "A1", "sharpe": 2.0, "fitness": 1.5}]}
        with self.assertRaises(cli.CliError) as caught:
            self._run(rows)
        self.assertEqual(caught.exception.code, "INVALID_INPUT")
        self.assertIn("not a simulate result document", str(caught.exception))


class AlphaGetMetadataTests(unittest.TestCase):
    def test_name_category_and_tags_are_visible(self):
        # Wrong behaviour this prevents: `alpha submit` refuses an alpha that has no
        # name or category, and `alpha get` used to show neither - the only way to see
        # why the submission stopped was to fetch the raw payload by hand.
        payload = {"id": "A1", "name": "qf dispersion change", "category": "ANALYST",
                   "tags": ["dispersion"], "regular": {"code": "rank(close)",
                                                       "description": "", "operatorCount": 1},
                   "settings": {"region": "USA"}, "status": "UNSUBMITTED",
                   "is": {"sharpe": 2.0}, "checks": []}
        args = argparse.Namespace(alpha_id="A1", save=None)
        with mock.patch.object(cli, "with_session", lambda a, fn: payload):
            data = cli.cmd_alpha_get(args, mock.MagicMock())
        self.assertEqual(data["name"], "qf dispersion change")
        self.assertEqual(data["category"], "ANALYST")
        self.assertEqual(data["tags"], ["dispersion"])


class CompareMinFitnessTests(unittest.TestCase):
    def test_min_fitness_applies_to_the_view_it_reports_on(self):
        # The train view filters on train_fitness; --frozen filters on the full-window
        # number. Mixing them would let a row pass a bar it never cleared.
        path = os.path.join(tempfile.mkdtemp(), "batch.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([{"alpha_id": "A1", "metrics": {"sharpe": 2.0, "fitness": 1.6},
                        "checks": {},
                        "out_of_sample": {"train": {"sharpe": 1.0, "fitness": 0.6},
                                          "test": {"sharpe": 0.8}}}], handle)
        args = argparse.Namespace(files=[path], min_fitness=1.0, sort="fitness",
                                  contributions=False, save=None, frozen=False)
        self.assertEqual(cli.cmd_compare(args, mock.MagicMock())["count"], 0)
        args.frozen = True
        self.assertEqual(cli.cmd_compare(args, mock.MagicMock())["count"], 1)


if __name__ == "__main__":
    unittest.main()
