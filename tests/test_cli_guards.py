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


if __name__ == "__main__":
    unittest.main()