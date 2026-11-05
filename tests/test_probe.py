#!/usr/bin/env python3
"""Tests for `probe`: one command per experiment instead of a copied script.

The command exists because the throwaway-script version of this workflow failed four
ways in one day - stdout contract, output-name collision, quoting at import time, and
no progress output. Each of those is pinned here.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import alpha_registry
import expression_lint
import machine_lib as ml
import research_paths
from research_app import experiments as cli_experiments
from research_app import parser as cli_parser
from research_app import runtime as cli_runtime


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

    def simulate(self, expressions, settings=None, labels=None, session=None, **kwargs):
        preview = ml.simulate_alphas(
            None, expressions, settings=settings, labels=labels, dry_run=True)
        self.calls.append({"expressions": expressions, "labels": labels,
                           "settings": dict(settings or {}), "session": session,
                           "payloads": [item["payload"] for item in preview]})
        return [{**fixture, "expression": item["expression"], "decay": item["decay"],
                 "mechanism": item["mechanism"]}
                for fixture, item in zip(self.records, preview)]


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
        self._patched = mock.patch.object(cli_experiments, "guarded_sim", self.fake)
        self._patched.start()
        self.addCleanup(self._patched.stop)
        self.session = object()
        session_patch = mock.patch.object(ml, "load_session", return_value=self.session)
        self.load_session = session_patch.start()
        self.addCleanup(session_patch.stop)

    def _write(self, payload):
        with open(self.spec_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def _args(self, **overrides):
        base = dict(spec=self.spec_path, name=None, dry_run=False, show_test=False,
                    force=False, no_record=True, registry=None, save=None)
        base.update(overrides)
        return argparse.Namespace(**base)

    def _run(self, **overrides):
        with mock.patch.object(research_paths, "ensure_dir",
                               lambda: self.output_dir):
            return cli_experiments.cmd_probe(self._args(**overrides), mock.MagicMock())

    def test_train_is_always_present_and_test_only_on_request(self):
        # Wrong behaviour this prevents: choosing between arms while looking at the
        # test year, which is the only out-of-sample evidence the API exposes.
        data = self._run()
        self.assertTrue(all("train_sharpe" in row for row in data["rows"]))
        self.assertTrue(all("train_fitness" in row for row in data["rows"]))
        # The full-window numbers contain the test year, so they are protected too.
        self.assertTrue(all("sharpe" not in row for row in data["rows"]))
        self.assertTrue(all("test_sharpe" not in row for row in data["rows"]))
        data = self._run(force=True, show_test=True)
        self.assertTrue(all("test_sharpe" in row for row in data["rows"]))

    def test_a_name_collision_is_detected_before_anything_is_spent(self):
        # Wrong behaviour this prevents: the same-day re-run paid for the whole batch
        # and only then refused to save it, because the collision was checked after
        # `guarded_sim.simulate` had already been called.
        self._run()
        self.fake.calls.clear()
        with self.assertRaises(cli_runtime.CliError):
            self._run()
        self.assertEqual(self.fake.calls, [])

    def test_an_existing_batch_file_is_not_silently_overwritten(self):
        # Wrong behaviour this prevents: a second run reusing the same NAME wrote over
        # the previous batch's JSON, which is how a 6-arm result disappeared.
        self._run()
        with self.assertRaises(cli_runtime.CliError) as caught:
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

    def test_cli_overrides_all_arm_settings_but_preserves_explicit_arm_decay(self):
        self.spec["settings"].update(region="EUR", universe="TOP600", delay=0)
        self.spec["arms"] = [
            {"label": "default", "expression": "rank(close)"},
            {"label": "explicit", "expression": "rank(volume)", "decay": 2,
             "settings": {"region": "GLB", "universe": "TOP1000", "decay": 3}},
            {"label": "settings", "expression": "rank(cap)", "settings": {"decay": 1}},
        ]
        self._write(self.spec)
        args = cli_parser.build_parser().parse_args([
            "probe", self.spec_path, "--dry-run", "--region", "usa", "--universe", "top500",
            "--neutralization", "industry", "--delay", "1", "--decay", "7",
            "--pasteurization", "off", "--nan-handling", "verify", "--test-period", "p1y",
        ])
        data = self._run(**vars(args))
        self.assertEqual(data["groups"], 2)
        by_label = {record["label"]: record for record in data["results"]}
        self.assertEqual({label: record["decay"] for label, record in by_label.items()},
                         {"default": 7, "explicit": 2, "settings": 7})
        for label, item in by_label.items():
            expected = {**ml.DEFAULT_SETTINGS, "universe": "TOP500", "neutralization": "INDUSTRY",
                        "pasteurization": "OFF", "nanHandling": "VERIFY", "testPeriod": "P1Y",
                        "decay": 2 if label == "explicit" else 7}
            self.assertEqual(item["payload"]["settings"], expected)
        self.load_session.assert_not_called()

    def test_stdin_spec_supports_utf8_bom_and_real_preview_payloads(self):
        self.spec["arms"][0]["label"] = "研究"
        stream = io.TextIOWrapper(io.BytesIO(
            ("\ufeff" + json.dumps(self.spec, ensure_ascii=False)).encode("utf-8")), encoding="utf-8")
        with stream, mock.patch.object(sys, "stdin", stream):
            data = self._run(spec="-", dry_run=True)
        self.assertEqual(data["arms"], 2)
        self.assertEqual(data["results"][0]["label"], "研究")
        self.assertEqual(data["results"][0]["payload"]["regular"], "rank(close)")
        self.load_session.assert_not_called()

    def test_multiple_groups_share_the_requested_session_policy(self):
        self.spec["arms"][1]["settings"] = {"neutralization": "INDUSTRY"}
        self._write(self.spec)
        args = cli_parser.build_parser().parse_args([
            "probe", self.spec_path, "--no-cache", "--relogin", "--no-record"])
        data = self._run(**vars(args))
        self.assertEqual(data["simulated"], 2)
        self.assertEqual(len(self.fake.calls), 2)
        self.assertTrue(all(call["session"] is self.session for call in self.fake.calls))
        self.load_session.assert_called_once_with(use_cache=False, refresh=True)

    def test_a_spec_without_arms_is_refused_with_a_code(self):
        self._write({"name": "x", "arms": []})
        with self.assertRaises(cli_runtime.CliError) as caught:
            self._run()
        self.assertEqual(caught.exception.code, "INVALID_INPUT")

    def test_a_typo_in_a_top_level_key_is_reported(self):
        # `"setting"` instead of `"settings"` would otherwise run the batch on the
        # defaults and read as a failed hypothesis rather than a typo.
        self.spec["setting"] = {"universe": "TOP500"}
        self._write(self.spec)
        with mock.patch.object(cli_experiments, "note") as spy:
            self._run()
        self.assertTrue(any("unknown key" in call.args[0] for call in spy.call_args_list))

    def test_dry_run_is_offline_and_does_not_load_a_session(self):
        with mock.patch.object(ml, "login", side_effect=AssertionError("login")), \
                mock.patch.object(ml, "get_credentials", side_effect=AssertionError("credentials")), \
                mock.patch.object(ml, "fetch_operators", side_effect=AssertionError("catalog")), \
                mock.patch.object(self.fake, "check", side_effect=AssertionError("live lint")), \
                mock.patch.object(ml.requests.sessions.Session, "request",
                                  side_effect=AssertionError("network")):
            data = self._run(dry_run=True)
        self.load_session.assert_not_called()
        self.assertTrue(data["dry_run"])
        self.assertEqual(data["arms"], 2)
        self.assertEqual(data["groups"], 1)
        self.assertEqual(self.fake.calls, [])
        self.assertFalse(data["lint"]["operator_catalog_checked"])
        self.assertFalse(data["lint"]["field_catalog_checked"])
        self.assertIn("not checked", data["lint"]["hint"])
        self.assertEqual(os.listdir(self.output_dir), [])

    def test_dry_run_contains_exact_payloads_and_resolved_settings(self):
        self.spec["arms"] = [
            {"label": "spaced", "expression": "  rank(close)  "},
            {"label": "fast", "expression": "rank(volume)", "decay": 2,
             "settings": {"neutralization": "INDUSTRY", "universe": "TOP500"}},
        ]
        self._write(self.spec)
        data = self._run(dry_run=True)
        default = {**ml.DEFAULT_SETTINGS, "decay": 4}
        fast = {**default, "decay": 2, "neutralization": "INDUSTRY", "universe": "TOP500"}
        self.assertEqual(data["settings"], [default, fast])
        self.assertEqual(data["groups"], 2)
        self.assertEqual([item["label"] for item in data["results"]], ["spaced", "fast"])
        self.assertEqual([item["mechanism"] for item in data["results"]], ["spaced", "fast"])
        self.assertEqual([item["decay"] for item in data["results"]], [4, 2])
        self.assertEqual([item["payload"] for item in data["results"]], [
            {"type": "REGULAR", "regular": "rank(close)", "settings": default},
            {"type": "REGULAR", "regular": "rank(volume)", "settings": fast},
        ])
        self.assertEqual([item["settings"] for item in data["results"]], [default, fast])

    def test_dry_run_reuses_structural_lint_and_learned_denylist(self):
        self.spec["arms"] = [
            {"label": "denied", "expression": "rank(close)"},
            {"label": "unbalanced", "expression": "rank(close"},
        ]
        self._write(self.spec)
        with mock.patch.object(expression_lint, "load_denylist", return_value={
                "rank": {"severity": "block", "message": "observed rejection"}}):
            data = self._run(dry_run=True)
        self.assertFalse(data["lint"]["ok"])
        self.assertTrue(any(item["code"] == "operator_rejected" for item in data["blocking"]))
        self.assertTrue(any(item["code"] == "unbalanced_parens" for item in data["blocking"]))
        self.assertEqual(data["blocking"], data["lint"]["errors"])
        self.assertEqual(len(data["results"]), 2)
        self.assertEqual(self.fake.calls, [])

    def test_equivalent_defaults_share_one_group_and_session(self):
        self.spec["arms"] = [
            {"label": "implicit", "expression": "rank(close)"},
            {"label": "explicit", "expression": "rank(volume)",
             "settings": {"neutralization": "SUBINDUSTRY", "delay": 1, "decay": 4}},
        ]
        self._write(self.spec)
        preview = self._run(dry_run=True)
        self.assertEqual(preview["groups"], 1)
        self._run()
        self.load_session.assert_called_once_with(use_cache=True, refresh=False)
        self.assertEqual(len(self.fake.calls), 1)
        self.assertIs(self.fake.calls[0]["session"], self.session)
        self.assertEqual(self.fake.calls[0]["labels"], ["implicit", "explicit"])
        self.assertEqual(self.fake.calls[0]["payloads"],
                         [item["payload"] for item in preview["results"]])

    def test_invalid_later_group_never_authenticates_or_spends_quota(self):
        invalid_arms = [
            {"expression": "rank(volume)", "settings": {"region": "INVALID"}},
            {"expression": "rank(volume)", "settings": {"neutralization": "INVALID"}},
            {"expression": "rank(volume)", "settings": {"delay": 2}},
            {"expression": "rank(volume)", "settings": {"decay": -1}},
            {"expression": "rank(volume)", "decay": True},
            {"expression": "rank(volume)", "decay": 1.5},
        ]
        for dry_run in (False, True):
            for invalid in invalid_arms:
                with self.subTest(dry_run=dry_run, arm=invalid):
                    self.spec["arms"] = [{"label": "valid", "expression": "rank(close)"},
                                         {"label": "invalid", **invalid}]
                    self._write(self.spec)
                    with self.assertRaises(cli_runtime.CliError) as caught:
                        self._run(dry_run=dry_run)
                    self.assertEqual(caught.exception.code, "INVALID_INPUT")
                    self.load_session.assert_not_called()
                    self.assertEqual(self.fake.calls, [])
                    self.assertEqual(os.listdir(self.output_dir), [])

    def test_settings_must_be_objects(self):
        for owner in (self.spec, self.spec["arms"][1]):
            for invalid in ([], "USA", None):
                with self.subTest(scope="spec" if owner is self.spec else "arm", value=invalid):
                    original = owner.get("settings", {})
                    owner["settings"] = invalid
                    self._write(self.spec)
                    with self.assertRaises(cli_runtime.CliError) as caught:
                        self._run(dry_run=True)
                    self.assertEqual(caught.exception.code, "INVALID_INPUT")
                    self.load_session.assert_not_called()
                    owner["settings"] = original

    def test_arm_decay_and_labels_match_preview_results_and_ledger(self):
        self.spec["arms"] = [
            {"label": "slow", "expression": "  rank(close)  "},
            {"label": "fast", "expression": "rank(volume)", "decay": 2},
            {"label": "same-fast", "expression": "rank(cap)", "settings": {"decay": 2}},
        ]
        self._write(self.spec)
        preview = self._run(dry_run=True)
        with mock.patch.object(alpha_registry, "record_simulation", return_value={}) as ledger:
            data = self._run(no_record=False)
        self.assertEqual(len(self.fake.calls), 2)
        self.assertEqual([call["settings"]["decay"] for call in self.fake.calls], [4, 2])
        self.assertEqual([call["labels"] for call in self.fake.calls], [["slow"], ["fast", "same-fast"]])
        self.assertEqual([payload for call in self.fake.calls for payload in call["payloads"]],
                         [item["payload"] for item in preview["results"]])
        self.assertEqual([row["label"] for row in data["rows"]], ["slow", "fast", "same-fast"])
        self.assertEqual([row["expression"] for row in data["rows"]],
                         [item["expression"] for item in preview["results"]])
        self.assertEqual([row["decay"] for row in data["rows"]], [4, 2, 2])
        records = ledger.call_args.args[0]
        self.assertEqual([item["mechanism"] for item in records], ["slow", "fast", "same-fast"])
        self.assertEqual([item["settings"] for item in records],
                         [item["payload"]["settings"] for item in preview["results"]])
        self.assertEqual([row["settings"] for row in data["rows"]],
                         [item["settings"] for item in records])
        self.load_session.assert_called_once_with(use_cache=True, refresh=False)
        self.assertTrue(all(call["session"] is self.session for call in self.fake.calls))

    def test_each_record_carries_the_settings_it_ran_under(self):
        # The ledger used to receive the spec-level settings for every arm, so an
        # INDUSTRY arm was archived as SUBINDUSTRY and the comparison could not be
        # reproduced.
        self.spec["arms"] = [{"label": "ind", "expression": "rank(close)",
                              "settings": {"neutralization": "INDUSTRY"}}]
        self._write(self.spec)
        data = self._run(force=True)
        written = json.load(open(data["path"], encoding="utf-8"))
        self.assertEqual(written["records"][0]["settings"]["neutralization"], "INDUSTRY")

    def test_the_batch_file_can_be_read_by_compare(self):
        data = self._run(force=True)
        args = argparse.Namespace(files=[data["path"]], min_fitness=None,
                                  sort="sharpe", contributions=False, save=None,
                                  frozen=False)
        compared = cli_experiments.cmd_compare(args, mock.MagicMock())
        self.assertEqual(compared["count"], 2)

    def test_showing_the_test_period_is_written_to_the_views_log(self):
        # Hiding a column does not stop the leak; the log is what tells a frozen
        # validation apart from an iterative peek three rounds later.
        log = os.path.join(self.output_dir, "views.jsonl")
        with mock.patch.object(research_paths, "views_log_path", lambda: log):
            self._run(force=True, show_test=True)
            self._run(force=True)
        entries = [json.loads(line) for line in open(log, encoding="utf-8")]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["tool"], "probe")
        self.assertEqual(entries[0]["view"], "test")
        self.assertEqual(sorted(entries[0]["alpha_ids"]), ["A1", "B1"])

    def test_rows_carry_the_failed_checks(self):
        self.fake.records = [record("A1", 1.2, 0.7, fails=("LOW_SHARPE",)),
                             record("B1", 2.0, 1.5)]
        data = self._run()
        self.assertEqual(data["rows"][0]["failed_checks"], ["LOW_SHARPE"])


if __name__ == "__main__":
    unittest.main()