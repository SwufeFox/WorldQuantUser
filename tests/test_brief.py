#!/usr/bin/env python3
"""Tests for `brief`, the session-start digest.

The value of this command is that it is *small*: it exists so a session can stop
reading a multi-megabyte ledger, a directory of write-ups and last week's note
before it knows what it is looking at. A digest that grows with the archive would
reproduce the problem it was built to solve, so the size bound is part of the
contract and is asserted here.
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

import alpha_registry  # noqa: E402
import research_cli as cli  # noqa: E402


def entry(alpha_id, status, sharpe, source="manual", family="quality"):
    return alpha_registry.build_entry(
        {"alpha_id": alpha_id, "expression": "rank(close)",
         "settings": {"region": "USA", "universe": "TOP3000"},
         "metrics": {"sharpe": sharpe, "fitness": sharpe / 2.0, "turnover": 0.15},
         "status": status},
        idea="probe", mechanism="test", verdict="candidate", family=family,
        source=source)


class BriefTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.registry = os.path.join(self.tmp, "registry.jsonl")
        self.research = tempfile.mkdtemp()
        alpha_registry.record_entry(entry("BOOK1", "ACTIVE", 2.0), self.registry)
        alpha_registry.record_entry(entry("BOOK2", "ACTIVE", 1.6, family="event"),
                                    self.registry)
        alpha_registry.record_entry(entry("NEW1", "UNSUBMITTED", 2.4, source="simulate"),
                                    self.registry)
        with open(os.path.join(self.research, "open-threads.md"), "w",
                  encoding="utf-8") as handle:
            handle.write("# 未决\n\n- [open] socialmedia12 buzz\n- [open] news12 vectors\n"
                         "- [closed] turnover repair\n- a line with no tag\n")
        with open(os.path.join(self.research, "x-20261103-results.md"), "w",
                  encoding="utf-8") as handle:
            handle.write("# 某次实验 · 2026-11-03\n\n正文\n")

    def _run(self, **overrides):
        args = argparse.Namespace(days=7.0, top=5, registry=self.registry, save=None)
        for key, value in overrides.items():
            setattr(args, key, value)
        with mock.patch.object(cli.research_paths, "ensure_dir", lambda: self.research):
            return cli.cmd_brief(args, mock.MagicMock())

    def test_book_is_counted_by_family_and_universe(self):
        data = self._run()
        self.assertEqual(data["book"]["count"], 2)
        self.assertEqual(data["book"]["by_family"], {"quality": 1, "event": 1})
        self.assertEqual(data["book"]["by_universe"], {"TOP3000": 2})

    def test_recent_runs_exclude_hand_recorded_rows(self):
        data = self._run()
        self.assertEqual(data["recent"]["runs"], 1)
        self.assertEqual(data["recent"]["best"][0]["alpha_id"], "NEW1")

    def test_threads_are_parsed_and_untagged_lines_ignored(self):
        data = self._run()
        self.assertEqual(data["threads"]["open"], ["socialmedia12 buzz", "news12 vectors"])
        self.assertEqual(data["threads"]["closed"], ["turnover repair"])

    def test_write_ups_are_listed_with_their_title(self):
        data = self._run()
        self.assertEqual(data["documents"][0]["file"], "x-20261103-results.md")
        self.assertEqual(data["documents"][0]["title"], "某次实验 · 2026-11-03")

    def test_the_window_is_respected(self):
        data = self._run(days=0)
        self.assertEqual(data["recent"]["runs"], 0)
        self.assertEqual(data["documents"], [])

    def test_the_document_stays_small(self):
        # The bound is the point: a digest is useful only while it is cheaper to read
        # than the archive it replaces.
        data = self._run()
        self.assertLess(len(json.dumps(data, ensure_ascii=False)), 2000)


if __name__ == "__main__":
    unittest.main()