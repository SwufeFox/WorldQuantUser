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
import research_paths
from research_app import campaign as cli_campaign


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
        with mock.patch.object(research_paths, "ensure_dir", lambda: self.research):
            return cli_campaign.cmd_brief(args, mock.MagicMock())

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

    def test_ledger_health_counts_label_debt_and_recording_gaps(self):
        # Both numbers were invisible until counted by hand: an ACTIVE row with no
        # family/mechanism breaks every "how many mechanisms" question, and a row with
        # metrics but no checks cannot be re-audited for why it failed.
        alpha_registry.record_entry(entry("BOOK3", "ACTIVE", 1.2, family="event"),
                                    self.registry)
        rows = [json.loads(line) for line in open(self.registry, encoding="utf-8")]
        rows[-1]["family"] = None
        rows[-1]["mechanism"] = None
        rows[-1]["checks"] = {}
        with open(self.registry, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        data = self._run()
        self.assertEqual(data["ledger"]["book_rows_missing_labels"], 1)
        self.assertGreaterEqual(data["ledger"]["rows_with_metrics_but_no_checks"], 1)

    def test_the_book_is_screened_for_same_bet_pairs(self):
        # The platform compares a submission against a periodically refreshed snapshot,
        # so two alphas 0.88 correlated can both be in the book. Nothing was scanning
        # the book after the fact; `brief` now does it on monthly PnL.
        import math
        days = ["2026-%02d-%02d" % (month, day) for month in range(1, 25)
                for day in range(1, 28)]
        base = {day: math.sin(index * 0.4) for index, day in enumerate(days)}
        twin = {day: value * 2 + 0.001 for day, value in base.items()}
        for name, series in (("BOOK1", base), ("BOOK2", twin)):
            folder = os.path.join(self.research, "pnl")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "%s.json" % name), "w",
                      encoding="utf-8") as handle:
                json.dump(series, handle)
            alpha_registry.record_entry(entry(name, "ACTIVE", 2.0), self.registry)
        data = self._run()
        screen = data["book"]["self_duplicates"]
        self.assertGreaterEqual(screen["count"], 1)
        pair = screen["pairs"][0]
        self.assertEqual({pair["a"], pair["b"]}, {"BOOK1", "BOOK2"})


    def test_stability_ignores_the_test_year(self):
        # The halves must stay inside train: a violent test-year tail must not turn a
        # steady book alpha into a "flip", because that would smuggle test data into a
        # selection-grade number.
        folder = os.path.join(self.research, "pnl")
        os.makedirs(folder, exist_ok=True)
        # ISO dates, because the split compares date strings: a "%03d" fixture is not
        # a date and would quietly defeat the test-start filter.
        import datetime
        def _dates(year, count):
            start = datetime.date(year, 1, 1)
            return [(start + datetime.timedelta(days=offset)).isoformat()
                    for offset in range(count)]
        series = {}
        for index, day in enumerate(_dates(2020, 199)):
            series[day] = 1.0 if index % 2 else 0.6
        for index, day in enumerate(_dates(2021, 59)):
            series[day] = 1.2 if index % 2 else 0.8
        for day in _dates(2023, 59):
            series[day] = -40.0
        with open(os.path.join(folder, "STAB1.json"), "w", encoding="utf-8") as handle:
            json.dump(series, handle)
        payload = entry("STAB1", "ACTIVE", 2.0)
        payload["out_of_sample"] = {"test": {"startDate": "2023-01-01"}}
        alpha_registry.record_entry(payload, self.registry)
        data = self._run()
        stability = data["book"]["stability"]
        self.assertEqual(stability["flips"], 0)
        self.assertEqual(stability["measured"], 1)

    def test_the_document_stays_small(self):
        # The bound is the point: a digest is useful only while it is cheaper to read
        # than the archive it replaces.
        data = self._run()
        self.assertLess(len(json.dumps(data, ensure_ascii=False)), 2000)


if __name__ == "__main__":
    unittest.main()

class DriftTests(unittest.TestCase):
    def test_a_submitted_alpha_missing_from_the_ledger_shows_as_drift(self):
        # 2026-11-07: `P0o5QGZK` was ACTIVE on the platform while the ledger still said
        # COMPLETE, so the reported book was one short.
        book = {"A": {"status": "ACTIVE"}, "B": {"status": "COMPLETE"}}
        drift = cli_campaign._book_drift(book, {"A": "ACTIVE", "B": "ACTIVE"})
        self.assertFalse(drift["in_sync"])
        self.assertEqual(drift["on_platform_not_in_ledger"], ["B"])
        self.assertEqual(drift["platform_active"], 2)
        self.assertEqual(drift["ledger_active"], 1)

    def test_a_matching_set_is_in_sync(self):
        book = {"A": {"status": "ACTIVE"}}
        drift = cli_campaign._book_drift(book, {"A": "ACTIVE"})
        self.assertTrue(drift["in_sync"])
