#!/usr/bin/env python3
"""Regression tests for the evidence-chain defects found in review (2026-11-04).

Five of the six findings are about the same failure mode: the tooling quietly
dropping or overwriting something a human would need later - a finished run counted
as failed, arms outside `--top` never recorded, manual labels erased by a re-run,
clusters only half merged, a "single-factor" swap that changed two fields. A crash is
loud; an archive that is missing rows is not.
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
sys.path.insert(0, os.path.join(ROOT, "tools"))

import alpha_registry  # noqa: E402
import placebo_controls  # noqa: E402
import ablation
from research_app import experiments as cli_experiments
from research_app import ledger as cli_ledger
import research_template  # noqa: E402


def record(alpha_id="A1", status="COMPLETE", sharpe=2.0, fitness=1.5):
    return {"alpha_id": alpha_id, "expression": "rank(close)", "status": status,
            "metrics": {"sharpe": sharpe, "fitness": fitness, "turnover": 0.15},
            "checks": {}, "out_of_sample": {"train": {"sharpe": sharpe - 0.3}}}


class WarningStatusTests(unittest.TestCase):
    def test_warning_counts_as_completed(self):
        # Wrong behaviour this prevents: a batch of WARNING records (a UNITS warning
        # is enough) summarised as completed=0, so `simulate` raised
        # SIMULATION_FAILED on a run that had every number it needed.
        summary = cli_experiments.summarize([record(status="WARNING", sharpe=2.37, fitness=1.68)])
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["best"]["sharpe"], 2.37)

    def test_only_an_unfinished_status_is_rejected(self):
        self.assertEqual(alpha_registry.auto_verdict(record(status="WARNING")), "candidate")
        self.assertEqual(alpha_registry.auto_verdict(record(status="PENDING")), "rejected")

    def test_an_all_warning_batch_does_not_raise(self):
        tmp = tempfile.mkdtemp()
        checkpoint_path = os.path.join(tmp, "cp.json")
        args = argparse.Namespace(sort="none", top=None, record=False)
        records = [record(alpha_id="A1", status="WARNING"),
                   record(alpha_id="A2", status="WARNING", sharpe=1.9)]
        result = cli_experiments._finalize_simulation(records, {"region": "USA"}, args,
                                          checkpoint_path, {"records": []}, explicit=True)
        self.assertEqual(result["summary"]["completed"], 2)


class TopTruncationTests(unittest.TestCase):
    def test_every_arm_is_recorded_even_beyond_top(self):
        # Wrong behaviour this prevents: `--top 5` truncated the list *before*
        # recording, so 15 simulated arms vanished from the ledger and the checkpoint
        # was deleted at the end - paid-for experiments with no trace.
        tmp = tempfile.mkdtemp()
        checkpoint_path = os.path.join(tmp, "cp.json")
        registry = os.path.join(tmp, "registry.jsonl")
        captured = {}

        def fake_record(records, settings, args):
            captured["ids"] = [r["alpha_id"] for r in records]
            return {}

        args = argparse.Namespace(sort="sharpe", top=2, record=True, registry=registry)
        records = [record(alpha_id="A%d" % i, sharpe=2.0 - i * 0.1) for i in range(5)]
        with mock.patch.object(cli_experiments, "_record_simulation", fake_record):
            result = cli_experiments._finalize_simulation(records, {"region": "USA"}, args,
                                              checkpoint_path, {"records": []},
                                              explicit=True)
        self.assertEqual(len(captured["ids"]), 5)
        self.assertEqual(result["summary"]["total"], 5)
        self.assertEqual(len(result["results"]), 2)          # display view keeps --top
        self.assertEqual(result["results_total"], 5)


class JudgementInheritanceTests(unittest.TestCase):
    def test_a_rerun_keeps_family_mechanism_and_lineage(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "registry.jsonl")
        first = alpha_registry.build_entry(
            {"alpha_id": "A1", "expression": "rank(close)", "settings": {"region": "USA"},
             "metrics": {"sharpe": 1.2, "fitness": 0.9}, "status": "COMPLETE",
             "parent": "PARENT", "op": "drop_leg:b"},
            idea="idea", mechanism="mech", family="quality", verdict="watch")
        alpha_registry.record_entry(first, path)
        # Same alpha, re-simulated with no labels at all.
        alpha_registry.record_simulation([record(alpha_id="A1", sharpe=1.4, fitness=1.0)],
                                         {}, registry_path=path)
        newest = alpha_registry.load_entries(path)[0]
        self.assertEqual(newest["family"], "quality")
        self.assertEqual(newest["mechanism"], "mech")
        self.assertEqual(newest["idea"], "idea")
        self.assertEqual(newest["parent"], "PARENT")
        self.assertEqual(newest["op"], "drop_leg:b")
        self.assertAlmostEqual(newest["metrics"]["sharpe"], 1.4)   # metrics do update

    def test_a_manual_verdict_survives_an_automatic_re_record(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "registry.jsonl")
        base = alpha_registry.build_entry(
            {"alpha_id": "A1", "expression": "rank(close)", "settings": {"region": "USA"},
             "metrics": {"sharpe": 1.2, "fitness": 0.9}, "status": "COMPLETE"},
            verdict="watch")
        alpha_registry.record_entry(alpha_registry.relabel(base, verdict="candidate"), path)
        alpha_registry.record_simulation([record(alpha_id="A1", sharpe=0.4, fitness=0.2)],
                                         {}, registry_path=path)
        self.assertEqual(alpha_registry.load_entries(path)[0]["verdict"], "candidate")


class LabelResponseTests(unittest.TestCase):
    def test_label_returns_the_stored_entry(self):
        # `ledger record` returns the row it wrote; `ledger label` returned only a
        # summary, so `data["entry"]` came back None and the caller had to reload the
        # ledger to see what had actually been stored.
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "registry.jsonl")
        alpha_registry.record_entry(
            alpha_registry.build_entry(
                {"alpha_id": "A1", "expression": "rank(close)",
                 "settings": {"region": "USA"}, "metrics": {"sharpe": 1.0},
                 "status": "COMPLETE"}, family="quality"), path)
        args = argparse.Namespace(alpha_id="A1", family="event", verdict=None,
                                  mechanism="m", idea=None, note=None,
                                  registry=path, save=None)
        data = cli_ledger.cmd_ledger_label(args, mock.MagicMock())
        self.assertTrue(data["written"])
        self.assertEqual(data["registry"], path)
        self.assertEqual(data["entry"]["family"], "event")
        self.assertEqual(data["entry"]["mechanism"], "m")
        self.assertEqual(data["before"]["family"], "quality")


class PlaceboClusterMergeTests(unittest.TestCase):
    def test_merging_one_member_merges_the_whole_cluster(self):
        pooled = {"clusters": {"A": ["a1", "a2"], "B": ["b1", "b2"]}}
        merged = placebo_controls.apply_review(pooled, {"merges": [["a1", "b1"]]})
        self.assertEqual(list(merged["clusters"].values()), [["a1", "a2", "b1", "b2"]])

    def test_chained_merges_land_in_one_cluster(self):
        pooled = {"clusters": {"A": ["a1"], "B": ["b1"], "C": ["c1"]}}
        merged = placebo_controls.apply_review(pooled, {"merges": [["a1", "b1"], ["b1", "c1"]]})
        self.assertEqual(len(merged["clusters"]), 1)
        self.assertEqual(list(merged["clusters"].values())[0], ["a1", "b1", "c1"])

    def test_exclusions_still_apply_after_a_merge(self):
        pooled = {"clusters": {"A": ["a1", "a2"], "B": ["b1"]}}
        merged = placebo_controls.apply_review(
            pooled, {"merges": [["a1", "b1"]], "exclusions": {"a2": "sparse"}})
        self.assertEqual(list(merged["clusters"].values()), [["a1", "b1"]])


class AblationViewTests(unittest.TestCase):
    def test_contributions_can_be_computed_from_a_chosen_metric_block(self):
        # `compare --contributions` used to read the full window even in the train
        # view, so a "drop this leg" verdict could rest on the test year.
        records = [
            {"candidate_id": "full", "metrics": {"fitness": 2.0},
             "out_of_sample": {"train": {"fitness": 1.0}}},
            {"candidate_id": "loo:a", "metrics": {"fitness": 1.9},
             "out_of_sample": {"train": {"fitness": 1.4}}},
        ]
        frozen = ablation.report(records)
        train = ablation.report(
            records, metrics_of=lambda r: (r.get("out_of_sample") or {}).get("train", {}))
        self.assertEqual(frozen["legs"][0]["verdict"], "adds")
        self.assertEqual(train["legs"][0]["verdict"], "hurts")


class AblationPendingTests(unittest.TestCase):
    def test_a_pending_loo_is_not_called_unmeasured(self):
        # A PENDING arm is a number that has not arrived, not a number that cannot
        # exist; collapsing both into "unmeasured" hid a `resume` that had to run
        # before the leg could be judged.
        records = [
            {"candidate_id": "full", "status": "COMPLETE", "metrics": {"fitness": 2.0}},
            {"candidate_id": "loo:a", "status": "PENDING", "metrics": None},
            {"candidate_id": "loo:b", "status": "COMPLETE", "metrics": {"fitness": 1.9}},
        ]
        report = ablation.report(records)
        legs = {row["leg"]: row for row in report["legs"]}
        self.assertEqual(legs["a"]["verdict"], "pending")
        self.assertEqual(legs["a"]["status"], "PENDING")
        self.assertEqual(legs["b"]["verdict"], "adds")
        self.assertEqual(report["pending"], ["a"])


class IdentifierSubstitutionTests(unittest.TestCase):
    def test_a_token_that_is_a_prefix_of_another_field_is_not_rewritten(self):
        # Wrong behaviour this prevents: swapping `sales` also rewrote `sales_growth`,
        # so a "single-factor" experiment changed two inputs and its lint still passed.
        pairs = research_template.expand("rank(sales) + rank(sales_growth)",
                                        {"sales": ["revenue"]})
        self.assertEqual(pairs[0][1], "rank(revenue) + rank(sales_growth)")

    def test_substitution_happens_at_every_whole_token(self):
        pairs = research_template.expand("sales / sales", {"sales": ["revenue"]})
        self.assertEqual(pairs[0][1], "revenue / revenue")


if __name__ == "__main__":
    unittest.main()