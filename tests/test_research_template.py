#!/usr/bin/env python3
"""Tests for the experiment template and the shared ledger recording.

Two things are pinned here, both of them about the path of least resistance being
the correct one.

The template is the file every experiment is copied from. It used to support only a
hand-written list of expressions, so the shape that actually dominates this campaign
- one proven parent with one token swapped N times - was re-implemented by hand in
every script, and those scripts did not record to the ledger at all. Now the swap
and the recording are the template's job, and they are tested without a network.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import alpha_registry  # noqa: E402
import research_template  # noqa: E402

PARENT = ("x = ts_backfill(accrued_liabilities_total_2, 252); "
          "group_rank(x/(ts_backfill(fnd6_cptnewqv1300_atq, 252) + 1), subindustry) - 0.5")


def stored_entry(alpha_id="PARENT1", expression=PARENT, settings=None,
                 sharpe=2.5, fitness=1.7):
    return alpha_registry.build_entry(
        {"alpha_id": alpha_id, "expression": expression,
         "settings": settings or {"region": "USA", "universe": "TOP3000", "delay": 1,
                                  "decay": 4, "neutralization": "SUBINDUSTRY",
                                  "startDate": "2019-01-01", "endDate": "2023-12-31",
                                  "simulationMode": "FULL"},
         "metrics": {"sharpe": sharpe, "fitness": fitness, "turnover": 0.18},
         "checks": {"LOW_SHARPE": {"result": "PASS", "value": sharpe, "limit": 1.25}},
         "status": "COMPLETE"})


def record(sharpe=2.0, fitness=1.5, status="COMPLETE", checks=None, mechanism=None,
           alpha_id="A1"):
    body = {"alpha_id": alpha_id,
            "expression": "rank(close)",
            "settings": {"region": "USA", "universe": "TOP3000"},
            "metrics": {"sharpe": sharpe, "fitness": fitness, "turnover": 0.15},
            "checks": checks if checks is not None else
            {"LOW_SHARPE": {"result": "PASS", "value": sharpe, "limit": 1.25}},
            "out_of_sample": {"train": {"sharpe": sharpe + 0.1},
                              "test": {"sharpe": sharpe - 0.4},
                              "test_period": "P1Y"},
            "status": status}
    if mechanism:
        body["mechanism"] = mechanism
    return body


class ExpandTests(unittest.TestCase):
    def test_one_pair_per_value_and_only_the_target_changes(self):
        pairs = research_template.expand(
            "a = ts_backfill(liab_2, 252); a",
            {"liab_2": ["taxes_due", "senior_notes"]})
        self.assertEqual([label for label, _ in pairs],
                         ["liab_2=taxes_due", "liab_2=senior_notes"])
        for _label, expression in pairs:
            self.assertNotIn("liab_2", expression)
            self.assertIn("ts_backfill(taxes_due, 252)", pairs[0][1])

    def test_two_targets_expand_as_a_product(self):
        pairs = research_template.expand("x = A + B",
                                         {"A": ["a1", "a2"], "B": ["b1"]})
        self.assertEqual([label for label, _ in pairs],
                         ["A=a1,B=b1", "A=a2,B=b1"])
        self.assertEqual(pairs[1][1], "x = a2 + b1")

    def test_a_missing_target_is_an_error_not_a_silent_no_op(self):
        # Wrong behaviour this prevents: 20 identical expressions submitted as
        # "variants", which reads exactly like 20 variants that all failed.
        with self.assertRaises(SystemExit):
            research_template.expand("x = rank(close)", {"liab_total_2": ["a"]})


class HandwrittenTests(unittest.TestCase):
    def test_a_tuple_carries_its_label_into_the_batch(self):
        # The label becomes the ledger entry's mechanism; a truncated first line of
        # the expression is not a mechanism, and that is what the old code used.
        pairs = research_template.handwritten([("beta_low", "rank(close)"),
                                               "rank(volume)"])
        self.assertEqual([label for label, _ in pairs], ["beta_low", "expr2"])
        self.assertEqual(pairs[1][1], "rank(volume)")


class ParentExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.registry = os.path.join(self.tmp, "registry.jsonl")
        alpha_registry.record_entry(stored_entry(), self.registry)
        self.saved = dict(research_template.SUBSTITUTIONS)

    def tearDown(self):
        research_template.SUBSTITUTIONS.clear()
        research_template.SUBSTITUTIONS.update(self.saved)

    def test_settings_are_inherited_without_the_platform_assigned_fields(self):
        research_template.SUBSTITUTIONS.update({"accrued_liabilities_total_2": ["tax_a"]})
        settings, pairs = research_template.parent_experiment("PARENT1", self.registry)
        self.assertEqual(settings["universe"], "TOP3000")
        self.assertEqual(settings["decay"], 4)
        self.assertNotIn("startDate", settings)
        self.assertNotIn("simulationMode", settings)
        self.assertEqual(len(pairs), 1)
        self.assertIn("ts_backfill(tax_a, 252)", pairs[0][1])
        # The parent is already compiled; the sugar pass must not have re-rendered it.
        self.assertTrue(pairs[0][1].startswith("x = ts_backfill(tax_a, 252);"))

    def test_unknown_parent_says_so(self):
        research_template.SUBSTITUTIONS.update({"accrued_liabilities_total_2": ["tax_a"]})
        with self.assertRaises(SystemExit):
            research_template.parent_experiment("NOPE", self.registry)

    def test_empty_substitutions_is_refused(self):
        with self.assertRaises(SystemExit):
            research_template.parent_experiment("PARENT1", self.registry)


class BatchFileCollisionTests(unittest.TestCase):
    def test_an_existing_batch_file_stops_the_run_before_simulating(self):
        # The template opened its output with mode "w" after a finished simulation, so
        # a same-day re-run both spent quota and silently replaced the earlier file.
        tmp = tempfile.mkdtemp()
        target = os.path.join(tmp, "template-2026-11-04.json")
        with open(target, "w", encoding="utf-8") as handle:
            json.dump([{"alpha_id": "OLD"}], handle)
        called = []

        def fake_simulate(*args, **kwargs):
            called.append(args)
            return []

        with mock.patch.object(research_template, "EXPRESSIONS", ["rank(close)"]), \
                mock.patch.object(research_template, "PARENT", ""), \
                mock.patch.object(research_template, "NAME", "template"), \
                mock.patch.object(research_template.research_paths, "batch_path",
                                  lambda name: target), \
                mock.patch.object(research_template.guarded_sim, "simulate", fake_simulate), \
                mock.patch("sys.argv", ["research_template.py"]):
            with self.assertRaises(SystemExit):
                research_template.main()
        self.assertEqual(called, [])


class AutoVerdictTests(unittest.TestCase):
    def test_clear_bar_is_a_candidate(self):
        self.assertEqual(alpha_registry.auto_verdict(record(sharpe=1.30, fitness=1.02)),
                         "candidate")

    def test_failed_check_is_rejected(self):
        body = record(sharpe=2.4, fitness=1.9,
                      checks={"SELF_CORRELATION": {"result": "FAIL", "value": 0.71}})
        self.assertEqual(alpha_registry.auto_verdict(body), "rejected")

    def test_unfinished_is_rejected(self):
        self.assertEqual(alpha_registry.auto_verdict(record(status="PENDING")), "rejected")

    def test_a_warning_run_is_classified_by_its_checks(self):
        # WARNING means "finished, one check warned without failing" (UNITS is the
        # usual one). Treating it as unfinished filed a Sharpe 2.37 / Fitness 1.68
        # alpha as `rejected` with reason `unrecorded_check`.
        self.assertEqual(alpha_registry.auto_verdict(record(status="WARNING")), "candidate")
        body = record(status="WARNING", checks={"UNITS": {"result": "WARNING"}})
        self.assertEqual(alpha_registry.auto_verdict(body), "candidate")

    def test_weak_but_clean_stays_watch(self):
        self.assertEqual(alpha_registry.auto_verdict(record(sharpe=1.1, fitness=0.8)),
                         "watch")


class RecordSimulationTests(unittest.TestCase):
    def test_one_entry_per_record_with_per_record_labels(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "registry.jsonl")
        summary = alpha_registry.record_simulation(
            [record(alpha_id="A1", mechanism="carry"),
             record(alpha_id="A2", mechanism="skew")],
            {"region": "USA"}, idea="probe", registry_path=path)
        self.assertEqual(summary["recorded"], 2)
        self.assertEqual(summary["appended"], 2)
        self.assertEqual(summary["by_verdict"]["candidate"], 2)
        entries = {e["alpha_id"]: e for e in alpha_registry.load_entries(path)}
        self.assertEqual(entries["A1"]["mechanism"], "carry")
        self.assertEqual(entries["A2"]["mechanism"], "skew")
        self.assertEqual(entries["A1"]["idea"], "probe")
        # The split travels with the entry: it is the only out-of-sample evidence
        # the API exposes, and losing it makes every later review in-sample only.
        self.assertAlmostEqual(entries["A1"]["out_of_sample"]["test"]["sharpe"], 1.6)

    def test_rerunning_the_same_records_does_not_duplicate(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "registry.jsonl")
        alpha_registry.record_simulation([record()], {}, registry_path=path)
        summary = alpha_registry.record_simulation([record()], {}, registry_path=path)
        self.assertEqual(summary["appended"], 0)
        self.assertEqual(len(alpha_registry.load_entries(path)), 1)

    def test_an_explicit_verdict_overrides_the_automatic_one(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "registry.jsonl")
        alpha_registry.record_simulation([record(sharpe=2.4, fitness=1.9)], {},
                                        verdict="watch", registry_path=path)
        self.assertEqual(alpha_registry.load_entries(path)[0]["verdict"], "watch")

    def test_a_rerun_cannot_demote_a_book_alpha(self):
        # BRAIN returns the existing id when an identical expression is simulated
        # again, so a re-run of a book member arrives looking like a new record. On
        # 2026-11-02 that rewrote an ACTIVE entry to COMPLETE/rejected and silently
        # shrank the book by one.
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "registry.jsonl")
        alpha_registry.record_entry(
            alpha_registry.build_entry(
                {"alpha_id": "BOOK1", "expression": "rank(close)",
                 "settings": {"region": "USA"}, "metrics": {"sharpe": 1.9, "fitness": 1.2},
                 "status": "ACTIVE"}, verdict="candidate"),
            path)
        rerun = record(alpha_id="BOOK1", sharpe=1.89, fitness=1.15)
        rerun["status"] = "COMPLETE"
        alpha_registry.record_simulation([rerun], {}, registry_path=path)
        newest = alpha_registry.load_entries(path)[0]
        self.assertEqual(newest["status"], "ACTIVE")
        self.assertEqual(newest["verdict"], "candidate")
        # The metrics still update: the run is recorded, it just cannot move the book.
        self.assertAlmostEqual(newest["metrics"]["sharpe"], 1.89)


if __name__ == "__main__":
    unittest.main()
