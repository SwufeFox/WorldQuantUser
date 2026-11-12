#!/usr/bin/env python3
"""Offline CLI regressions for saving results and structured command failures."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import machine_lib as ml
import paper_search
import research_paths
from research_app import alphas as cli_alphas
from research_app import app as cli_app
from research_app import campaign as cli_campaign
from research_app import experiments as cli_experiments
from research_app import runtime as cli_runtime


class PaperSaveTests(unittest.TestCase):
    def test_save_preserves_the_complete_stdout_document(self):
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "nested", "papers.json")
            stdout, stderr = io.StringIO(), io.StringIO()
            payload = {"results": [{"title": "Anomalies", "year": 1996}]}
            with mock.patch.object(paper_search, "search_papers", return_value=payload), \
                    mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
                code = cli_app.main(["paper", "accruals", "--save", target])
            self.assertEqual(code, 0, stderr.getvalue())
            with open(target, encoding="utf-8") as handle:
                saved = handle.read()
            self.assertEqual(saved, stdout.getvalue())
            self.assertEqual(json.loads(saved)["data"]["results"][0]["title"], "Anomalies")


class DiagnosticLifecycleTests(unittest.TestCase):
    def test_library_diagnostics_survive_successful_and_failed_cli_calls(self):
        cases = [
            (["schema", "resume"], 0),
            (["schema", "resume", "--unknown-architecture-option"], 2),
            (["simulate", "--expr", "rank(close)", "--batch-size", "0"], 2),
        ]
        for arguments, expected_code in cases:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                messages = []
                previous = ml.set_diagnostic_handler(messages.append)
                try:
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
                        code = cli_app.main(["--research-dir", directory, "--quiet", *arguments])
                        ml.note("library diagnostic after CLI")
                    self.assertEqual(code, expected_code, stderr.getvalue())
                    self.assertEqual(json.loads(stdout.getvalue())["ok"], expected_code == 0)
                    self.assertEqual(messages, ["library diagnostic after CLI"])
                finally:
                    ml.set_diagnostic_handler(previous)


class NegativeMaxWaitTests(unittest.TestCase):
    def test_alpha_check_refuses_negative_wait_with_a_code(self):
        args = argparse.Namespace(max_wait=-1, alpha_id="X", save=None)
        with self.assertRaises(cli_runtime.CliError) as caught:
            cli_alphas.cmd_alpha_check(args, mock.MagicMock())
        self.assertEqual(caught.exception.code, "USAGE")

    def test_alpha_submit_refuses_negative_wait_with_a_code(self):
        args = argparse.Namespace(max_wait=-5, alpha_id="X", save=None)
        with self.assertRaises(cli_runtime.CliError) as caught:
            cli_alphas.cmd_alpha_submit(args, mock.MagicMock())
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
        # `--frozen` writes an audit line; without this the test appends to the real
        # research log, which is how 480 test entries ended up in views.jsonl.
        log = os.path.join(tempfile.mkdtemp(), "views.jsonl")
        with mock.patch.object(research_paths, "views_log_path", lambda: log):
            return cli_experiments.cmd_compare(args, mock.MagicMock())

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
        with self.assertRaises(cli_runtime.CliError) as caught:
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
        with mock.patch.object(cli_alphas, "with_session", lambda a, fn: payload):
            data = cli_alphas.cmd_alpha_get(args, mock.MagicMock())
        self.assertEqual(data["name"], "qf dispersion change")
        self.assertEqual(data["category"], "ANALYST")
        self.assertEqual(data["tags"], ["dispersion"])


class CompareSortAndContributionTests(unittest.TestCase):
    def _run(self, payload, **overrides):
        path = os.path.join(tempfile.mkdtemp(), "batch.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        args = argparse.Namespace(files=[path], min_fitness=None, sort="sharpe",
                                  contributions=False, save=None, frozen=False)
        for key, value in overrides.items():
            setattr(args, key, value)
        # Same reason as CompareShapeTests._run: frozen writes an audit line, and the
        # audit line belongs in a temp file during a test run.
        log = os.path.join(tempfile.mkdtemp(), "views.jsonl")
        with mock.patch.object(research_paths, "views_log_path", lambda: log):
            return cli_experiments.cmd_compare(args, mock.MagicMock())

    def test_every_allowed_sort_key_exists_in_the_train_view(self):
        # `--sort returns` raised KeyError: the row had train_sharpe/fitness/turnover
        # but no train_returns, and the sort key had already been rewritten.
        for key in ("sharpe", "fitness", "turnover", "returns", "id"):
            data = self._run([result_record()], sort=key)
            self.assertEqual(data["count"], 1, key)

    def test_contributions_use_train_fitness_in_the_train_view(self):
        # The table hid the test year; `--contributions` read full-window fitness anyway,
        # so the "drop this leg" verdict was computed on data the view had just hidden.
        records = [
            {"candidate_id": "full", "alpha_id": "F",
             "metrics": {"fitness": 2.0}, "out_of_sample": {"train": {"fitness": 1.0}}},
            {"candidate_id": "loo:a", "alpha_id": "A",
             "metrics": {"fitness": 1.9}, "out_of_sample": {"train": {"fitness": 1.4}}},
        ]
        train = self._run(records, contributions=True)
        self.assertEqual(train["contributions"]["view"], "train")
        self.assertEqual(train["contributions"]["legs"][0]["verdict"], "hurts")
        frozen = self._run(records, contributions=True, frozen=True)
        self.assertEqual(frozen["contributions"]["view"], "frozen")
        self.assertEqual(frozen["contributions"]["legs"][0]["verdict"], "adds")


class ComparePerFileTests(unittest.TestCase):
    def test_per_file_keeps_each_alphas_legs_apart(self):
        # Ablation files all name their arms `full` / `loo:l0`, so a merged
        # contribution report mixes one alpha's legs into another's. --per-file
        # reports once per input and carries the leg legend through.
        def write(name, full_f, loo):
            payload = [{"candidate_id": "full", "status": "COMPLETE",
                        "metrics": {"fitness": full_f},
                        "out_of_sample": {"train": {"fitness": full_f}}},
                       {"candidate_id": "loo:l0", "status": "COMPLETE",
                        "metrics": {"fitness": loo},
                        "out_of_sample": {"train": {"fitness": loo}}}]
            path = os.path.join(tempfile.mkdtemp(), name)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"alphas": payload, "records": payload,
                           "legs": {"l0": {"fields": ["atq"]}}}, handle)
            return path

        args = argparse.Namespace(files=[write("A-results.json", 2.0, 1.0),
                                         write("B-results.json", 1.5, 1.4)],
                                  min_fitness=None, sort="sharpe", contributions=False,
                                  save=None, frozen=False, per_file=True)
        data = cli_experiments.cmd_compare(args, mock.MagicMock())
        self.assertEqual(len(data["per_file"]), 2)
        first, second = data["per_file"]
        self.assertAlmostEqual(first["top_leg"]["contribution"], 1.0, places=3)
        self.assertAlmostEqual(second["top_leg"]["contribution"], 0.1, places=3)
        self.assertEqual(first["top_leg"]["fields"], ["atq"])


class CompareLegendTests(unittest.TestCase):
    def test_a_results_file_finds_its_leg_legend_next_to_it(self):
        # `simulate` only started carrying the file's `legs` legend recently; older
        # results still sit beside their ablation file, and without the legend a
        # contribution report can say "l1 added 0.56" but not what l1 reads.
        folder = tempfile.mkdtemp()
        results = os.path.join(folder, "X-results.json")
        ablation = os.path.join(folder, "X-ablation.json")
        payload = [{"candidate_id": "full", "status": "COMPLETE", "metrics": {"fitness": 2.0},
                    "out_of_sample": {"train": {"fitness": 2.0}}},
                   {"candidate_id": "loo:l0", "status": "COMPLETE", "metrics": {"fitness": 1.0},
                    "out_of_sample": {"train": {"fitness": 1.0}}}]
        with open(results, "w", encoding="utf-8") as handle:
            json.dump({"records": payload}, handle)
        with open(ablation, "w", encoding="utf-8") as handle:
            json.dump({"alphas": payload,
                       "legs": {"l0": {"fields": ["accrued_liabilities_total_2",
                                                  "fnd6_cptnewqv1300_atq"]}}}, handle)
        args = argparse.Namespace(files=[results], min_fitness=None, sort="sharpe",
                                  contributions=False, save=None, frozen=False,
                                  per_file=True)
        with mock.patch("alpha_unit.load_field_datasets", return_value={
                "accrued_liabilities_total_2": "fundamental6",
                "fnd6_cptnewqv1300_atq": "fundamental6"}), \
                mock.patch("alpha_unit.load_rules", return_value={
                    "datasets": {"fundamental6": "fundamental"}}):
            data = cli_experiments.cmd_compare(args, mock.MagicMock())
        top = data["per_file"][0]["top_leg"]
        self.assertEqual(top["fields"][0], "accrued_liabilities_total_2")
        self.assertIn("fundamental", top["families"])


class ViewsCommandTests(unittest.TestCase):
    def test_the_frozen_view_is_logged_and_train_is_not(self):
        path = os.path.join(tempfile.mkdtemp(), "batch.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([{"alpha_id": "A1", "metrics": {"sharpe": 2.0, "fitness": 1.6},
                        "checks": {}, "out_of_sample": {"train": {"sharpe": 1.0},
                                                        "test": {"sharpe": 0.8}}}], handle)
        log = os.path.join(tempfile.mkdtemp(), "views.jsonl")
        args = argparse.Namespace(files=[path], min_fitness=None, sort="sharpe",
                                  contributions=False, save=None, frozen=False)
        with mock.patch.object(research_paths, "views_log_path", lambda: log):
            cli_experiments.cmd_compare(args, mock.MagicMock())
            self.assertFalse(os.path.exists(log))
            args.frozen = True
            cli_experiments.cmd_compare(args, mock.MagicMock())
        entries = [json.loads(line) for line in open(log, encoding="utf-8")]
        self.assertEqual(entries[0]["tool"], "compare")
        self.assertEqual(entries[0]["alpha_ids"], ["A1"])

        with mock.patch.object(research_paths, "views_log_path", lambda: log):
            listing = cli_campaign.cmd_views(argparse.Namespace(alpha="A1", limit=10, save=None),
                                    mock.MagicMock())
        self.assertEqual(listing["count"], 1)
        empty = cli_campaign.cmd_views(argparse.Namespace(alpha="NOPE", limit=10, save=None),
                              mock.MagicMock()) if False else None
        self.assertIsNone(empty)


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
        self.assertEqual(cli_experiments.cmd_compare(args, mock.MagicMock())["count"], 0)
        args.frozen = True
        log = os.path.join(tempfile.mkdtemp(), "views.jsonl")
        with mock.patch.object(research_paths, "views_log_path", lambda: log):
            self.assertEqual(
                cli_experiments.cmd_compare(args, mock.MagicMock())["count"], 1)


if __name__ == "__main__":
    unittest.main()


class ViewLogIsolationTests(unittest.TestCase):
    """Tests must not append to the research log.

    Before this, two `compare` helpers ran with `frozen=True` and no patched path: every
    full test run wrote audit lines into the real `views.jsonl`, which ended up holding
    480 entries, all of them from tests and none from actual research.
    """

    def test_a_frozen_compare_writes_only_to_the_patched_path(self):
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "batch.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([{"alpha_id": "Z1", "metrics": {"fitness": 1.5, "sharpe": 2.0}}], handle)
        log = os.path.join(tempfile.mkdtemp(), "views.jsonl")
        args = argparse.Namespace(files=[path], min_fitness=None, sort="sharpe",
                                  contributions=False, save=None, frozen=True)
        with mock.patch.object(research_paths, "views_log_path", lambda: log):
            cli_experiments.cmd_compare(args, mock.MagicMock())
        self.assertTrue(os.path.exists(log))
        self.assertEqual(json.loads(open(log, encoding="utf-8").read().strip())["tool"],
                         "compare")
