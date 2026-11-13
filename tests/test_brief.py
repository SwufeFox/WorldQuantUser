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
         "out_of_sample": {"train": {"sharpe": sharpe, "fitness": sharpe / 2.0}},
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
        alpha_registry.record_entry(
            alpha_registry.relabel(entry("OLD", "COMPLETE", -0.5), verdict="rejected"),
            self.registry)
        data = self._run(days=0)
        self.assertEqual(data["recent"]["runs"], 0)
        self.assertEqual(data["recent"]["setbacks"], {"count": 0, "items": []})
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
        # Highly correlated train PnL should be visible even with a stale platform snapshot.
        import math
        from datetime import date, timedelta

        days = [(date(2020, 1, 1) + timedelta(days=index)).isoformat() for index in range(648)]
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

    def test_test_period_co_movement_does_not_change_startup_duplicates(self):
        import math
        from datetime import date, timedelta

        folder = os.path.join(self.research, "pnl")
        os.makedirs(folder, exist_ok=True)
        train = [(date(2020, 1, 1) + timedelta(days=index)).isoformat() for index in range(300)]
        holdout = [(date(2021, 1, 1) + timedelta(days=index)).isoformat() for index in range(300)]
        for name, shift in (("BOOK1", 0.0), ("BOOK2", math.pi / 2)):
            series = {day: math.sin(index * 0.4 + shift) for index, day in enumerate(train)}
            series.update({day: 1000 * math.sin(index * 0.4) for index, day in enumerate(holdout)})
            with open(os.path.join(folder, name + ".json"), "w", encoding="utf-8") as handle:
                json.dump(series, handle)
            payload = entry(name, "ACTIVE", 2.0)
            payload["out_of_sample"]["test"] = {"startDate": "2021-01-01"}
            alpha_registry.record_entry(payload, self.registry)
        screen = self._run()["book"]["self_duplicates"]
        self.assertEqual(screen["window"], "train")
        self.assertEqual(screen["count"], 0)


    def test_stability_ignores_the_test_year_without_a_recorded_boundary(self):
        # An old ledger row may have PnL but no out-of-sample split. The default boundary
        # still keeps the halves in train; otherwise a test-year tail becomes selection data.
        import datetime

        folder = os.path.join(self.research, "pnl")
        os.makedirs(folder, exist_ok=True)
        series = {}
        start = datetime.date(2020, 1, 1)
        for index in range(258):
            day = (start + datetime.timedelta(days=index)).isoformat()
            series[day] = 1.0 if index % 2 else 0.6
        for index in range(59):
            series[(datetime.date(2023, 1, 1) + datetime.timedelta(days=index)).isoformat()] = -40.0
        with open(os.path.join(folder, "OLD1.json"), "w", encoding="utf-8") as handle:
            json.dump(series, handle)
        alpha_registry.record_entry(entry("OLD1", "ACTIVE", 2.0), self.registry)
        stability = self._run()["book"]["stability"]
        self.assertEqual(stability["measured"], 1)
        self.assertEqual(stability["flips"], 0)

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

    def test_score_leads_use_train_and_do_not_fill_gaps_from_full_window(self):
        strong = entry("TRAIN", "COMPLETE", 0.1, source="simulate")
        strong["out_of_sample"] = {"train": {"sharpe": 3.0, "fitness": 1.5},
                                   "test": {"sharpe": -10.0}}
        missing = entry("FULL_ONLY", "COMPLETE", 100.0, source="simulate")
        missing.pop("out_of_sample")
        for payload in (strong, missing):
            alpha_registry.record_entry(payload, self.registry)
        recent = self._run()["recent"]
        self.assertEqual(recent["metrics_scope"], "train")
        self.assertEqual([item["alpha_id"] for item in recent["best"]], ["TRAIN", "NEW1"])
        self.assertEqual(recent["best"][0]["sharpe"], 3.0)
        self.assertEqual(recent["missing_train_metrics"], 1)

    def test_setbacks_include_relabels_and_preserve_scope_across_mechanisms(self):
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        cases = [("R1", "rejected", "revision", 1),
                 ("R2", "rejected", "revision", 2),
                 ("F1", "falsified", "revision", 3),
                 ("N1", "no_go", "value", 4),
                 ("W1", "watch", "value", 5)]
        for alpha_id, verdict, mechanism, minutes in cases:
            payload = alpha_registry.relabel(entry(alpha_id, "COMPLETE", -0.5),
                                             verdict=verdict, mechanism=mechanism,
                                             note="Control and scope for " + alpha_id)
            payload["recorded_at"] = (now - timedelta(minutes=minutes)).isoformat()
            alpha_registry.record_entry(payload, self.registry)

        data = self._run(top=3)
        self.assertEqual(data["recent"]["runs"], 1)
        setbacks = data["recent"]["setbacks"]
        self.assertEqual(setbacks["count"], 4)
        self.assertEqual([(item["alpha_id"], item["verdict"]) for item in setbacks["items"]],
                         [("R1", "rejected"), ("F1", "falsified"), ("N1", "no_go")])
        self.assertEqual(setbacks["items"][1]["note"], "Control and scope for F1")
        self.assertEqual(self._run(top=1)["recent"]["setbacks"]["items"],
                         setbacks["items"][:1])

    def test_thread_previews_are_bounded_and_full_evidence_remains_reachable(self):
        path = os.path.join(self.research, "open-threads.md")
        question = "一个需要保留适用条件的研究问题" * 100
        with open(path, "w", encoding="utf-8") as handle:
            for index in range(30):
                handle.write("- [open] %s %s\n" % (index, question))
            for index in range(10):
                handle.write("- [closed] resolved %s\n" % index)
        data = self._run(top=2)
        threads = data["threads"]
        self.assertEqual(threads["counts"], {"open": 30, "closed": 10})
        self.assertEqual(threads["closed"], ["resolved 0", "resolved 1"])
        self.assertEqual(len(threads["open"]), 2)
        self.assertTrue(all(len(text) <= 240 and text.endswith("…")
                            for text in threads["open"]))
        with open(threads["path"], encoding="utf-8") as handle:
            self.assertIn(question, handle.read())
        self.assertTrue(threads["exists"])
        self.assertEqual(data["research_dir"], self.research)

    def test_missing_sources_are_distinct_from_present_but_empty_sources(self):
        os.remove(self.registry)
        path = os.path.join(self.research, "open-threads.md")
        os.remove(path)
        missing = self._run()
        self.assertFalse(missing["ledger"]["exists"])
        self.assertFalse(missing["threads"]["exists"])
        for source in (self.registry, path):
            with open(source, "w", encoding="utf-8"):
                pass
        empty = self._run()
        self.assertTrue(empty["ledger"]["exists"])
        self.assertTrue(empty["threads"]["exists"])
        self.assertEqual(empty["ledger"]["rows"], 0)
        self.assertEqual(empty["threads"]["counts"], {"open": 0, "closed": 0})

    def test_setback_note_is_a_preview_with_a_recoverable_ledger_key(self):
        note = "Control evidence with an important qualification. " * 100
        payload = alpha_registry.relabel(entry("LONG", "COMPLETE", -0.5),
                                         verdict="falsified", note=note)
        alpha_registry.record_entry(payload, self.registry)
        data = self._run()
        preview = data["recent"]["setbacks"]["items"][0]
        self.assertLessEqual(len(preview["note"]), 240)
        self.assertTrue(preview["note"].endswith("…"))
        rows = alpha_registry.load_entries(data["ledger"]["path"])
        self.assertEqual(alpha_registry.resolve(rows, preview["key"])["note"], note)

    def test_write_up_list_respects_the_same_requested_limit(self):
        for index in range(6):
            with open(os.path.join(self.research, "%s-results.md" % index), "w",
                      encoding="utf-8") as handle:
                handle.write("# Result %s\n" % index)
        self.assertEqual(len(self._run(top=2)["documents"]), 2)


if __name__ == "__main__":
    unittest.main()

class DriftTests(unittest.TestCase):
    def test_a_submitted_alpha_missing_from_the_ledger_shows_as_drift(self):
        # A platform ACTIVE alpha still marked COMPLETE locally understates the book.
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
