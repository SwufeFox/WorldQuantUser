"""Deterministic offline contracts for durable probe and frozen validation."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import machine_lib as ml
from research_app import experiments as app
from research_app.runtime import CliError


def arguments(**values):
    return argparse.Namespace(**{"spec": None, "name": None, "dry_run": False,
        "show_test": False, "force": False, "no_record": False, "registry": None,
        "max_wait_seconds": 30, **values})


def finished(alpha_id):
    return {"alpha_id": alpha_id, "status": "COMPLETE", "submission_attempted": True,
        "metrics": {"fitness": 1.1, "sharpe": 1.5, "turnover": .1, "margin": .002},
        "out_of_sample": {"train": {"fitness": 1.0, "sharpe": 1.4, "turnover": .1,
                                    "margin": .0015}}, "checks": {}}


class ProbeWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = self.root / "spec.json"
        self.batch = self.root / "batch.json"
        self.registry = self.root / "registry.jsonl"
        self.spec.write_text(json.dumps({"name": "synthetic", "design": "explore",
            "arms": [{"label": "first", "expression": "rank(close)"},
                     {"label": "second", "expression": "rank(volume)", "settings": {"decay": 7}},
                     {"label": "third", "expression": "rank(cap)"}]}), encoding="utf-8")
        for target, value in [("_novelty_gate", lambda arms, **kw: [{} for _ in arms]),
                              ("_preflight_lint", lambda *a: {"ok": True}),
                              ("with_session", lambda args, fn: fn(object()))]:
            patch = mock.patch.object(app, target, value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(app.research_paths, "views_log_path", return_value=str(self.root / "views.jsonl"))
        patch.start()
        self.addCleanup(patch.stop)

    def run_probe(self, **extra):
        return app.cmd_probe(arguments(spec=str(self.spec), batch_path=str(self.batch),
                                       registry=str(self.registry), **extra), None)

    def simulator(self, expressions, settings, **kw):
        records = []
        for index, label in enumerate(kw["labels"]):
            pending = {"index": index, "status": "UNKNOWN_SUBMISSION", "submission_attempted": True}
            kw["on_event"]("submitting", pending)
            result = finished("SYNTHETIC_" + label)
            kw["on_event"]("completed", {"index": index, **result})
            records.append(result)
        return records

    def test_preview_fingerprint_ignores_clock_and_output_path(self):
        with mock.patch.object(app, "cached_field_scope", return_value={"evidence": {"state": "unknown"}}), \
                mock.patch.object(app, "_turnover_band", return_value={}), \
                mock.patch.object(app, "with_session", side_effect=AssertionError("offline")):
            first = self.run_probe(dry_run=True)
            moved = app.cmd_probe(arguments(spec=str(self.spec), dry_run=True,
                batch_path=str(self.root / "2099-12-31-other.json")), None)
        self.assertEqual(first["request_fingerprint"], moved["request_fingerprint"])
        self.assertEqual([row["label"] for row in first["results"]], ["first", "second", "third"])
        self.assertFalse(first["preflight_complete"])
        self.assertFalse(self.batch.exists())

    def test_check_reports_incomplete_without_simulating(self):
        with mock.patch.object(app, "_preflight_lint", return_value={"ok": False,
                "coverage": {"fields": "unknown"}}), mock.patch.object(app.guarded_sim, "simulate") as simulate:
            result = self.run_probe(check=True, allow_lint_errors=True)
        self.assertFalse(result["preflight_complete"])
        simulate.assert_not_called()
        self.assertFalse(self.batch.exists())

    def test_groups_preserve_original_order_and_recover_deduplicates_recording(self):
        with mock.patch.object(app.guarded_sim, "simulate", side_effect=self.simulator):
            result = self.run_probe()
        self.assertEqual([row["label"] for row in result["rows"]], ["first", "second", "third"])
        self.assertEqual(result["execution"]["accepted"], 3)
        self.assertEqual(result["recording"]["appended"], 3)
        before = self.registry.read_bytes()
        envelope = self.root / "response.json"
        envelope.write_text(json.dumps({"ok": True, "data": result}), encoding="utf-8")
        with mock.patch.object(app, "with_session", side_effect=AssertionError("no platform work")), \
                mock.patch.object(ml, "fetch_daily_pnl", side_effect=AssertionError("no boundary")):
            recovered = app.cmd_probe(arguments(recover=str(envelope)), None)
        self.assertEqual(recovered["path"], str(self.batch.resolve()))
        self.assertEqual(recovered["recording"]["appended"], 0)
        self.assertEqual(self.registry.read_bytes(), before)
        self.assertTrue(all(row["train_sharpe_h1"] is None for row in recovered["rows"]))

    def test_unknown_post_is_saved_and_never_reposted(self):
        def interrupted(expressions, settings, **kw):
            kw["on_event"]("submitting", {"index": 0, "status": "UNKNOWN_SUBMISSION",
                                         "submission_attempted": True})
            raise KeyboardInterrupt
        with mock.patch.object(app.guarded_sim, "simulate", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_probe()
        saved = json.loads(self.batch.read_text(encoding="utf-8"))
        self.assertEqual(saved["execution"]["unknown_submission"], 1)
        self.assertEqual(saved["execution"]["not_submitted"], 2)
        with mock.patch.object(app, "with_session", side_effect=AssertionError("offline")):
            preview = app.cmd_probe(arguments(resume=str(self.batch), dry_run=True), None)
        self.assertEqual(preview["blocked_unknown"], ["first"])
        seen = []
        def continuing(expressions, settings, **kw):
            seen.extend(kw["labels"])
            return self.simulator(expressions, settings, **kw)
        with mock.patch.object(app.guarded_sim, "simulate", side_effect=continuing):
            result = app.cmd_probe(arguments(resume=str(self.batch)), None)
        self.assertEqual(set(seen), {"second", "third"})
        self.assertNotIn("first", seen)
        self.assertEqual(result["execution"]["unknown_submission"], 1)
        self.assertEqual(result["incomplete"], ["first"])

    def test_resume_rejects_request_changes(self):
        with mock.patch.object(app.guarded_sim, "simulate", side_effect=self.simulator):
            self.run_probe()
        for changes in ({"decay": 4}, {"spec": str(self.spec)}, {"registry": "other.jsonl"}):
            with self.subTest(changes=changes), self.assertRaises(CliError):
                app.cmd_probe(arguments(resume=str(self.batch), dry_run=True, **changes), None)

    def test_partial_and_all_empty_are_structured(self):
        for partial in (False, True):
            def pending(expressions, settings, **kw):
                records = []
                for index, label in enumerate(kw["labels"]):
                    record = finished("SYNTHETIC_" + label) if partial and index == 0 else {
                        "status": "QUEUED", "submission_attempted": False}
                    records.append(record)
                return records
            with mock.patch.object(app.guarded_sim, "simulate", side_effect=pending):
                result = self.run_probe(force=True)
            self.assertEqual(result["all_empty"], not partial)
            self.assertFalse(result["complete"])
            self.assertTrue(any(action["action"] == "resume" for action in result["next_actions"]))

    def test_legacy_unknown_batch_does_not_assume_it_was_unsubmitted(self):
        self.batch.write_text(json.dumps({"rows": [{"label": "legacy", "expression": "rank(close)",
            "settings": {"decay": 4}, "status": "QUEUED"}]}), encoding="utf-8")
        with mock.patch.object(app, "with_session", side_effect=AssertionError("offline")):
            result = app.cmd_probe(arguments(resume=str(self.batch), dry_run=True), None)
        self.assertEqual(result["blocked_unknown"], ["legacy"])
        self.assertEqual(result["continuing"], [])

    def test_show_test_cannot_create_new_simulation(self):
        with mock.patch.object(app.guarded_sim, "simulate") as simulate:
            with self.assertRaises(CliError):
                self.run_probe(show_test=True, selection="choice.json")
        simulate.assert_not_called()

    def test_frozen_test_changes_cannot_filter_or_reorder_candidates(self):
        source = self.root / "results.json"
        records = [finished("SYNTHETIC_A"), finished("SYNTHETIC_B")]
        records[1]["out_of_sample"]["train"]["fitness"] = 2
        for row in records:
            row["out_of_sample"]["train"].update(startDate="2019-01-01", endDate="2020-12-31")
        source.write_text(json.dumps(records), encoding="utf-8")
        choice = self.root / "choice.json"
        train = argparse.Namespace(files=[str(source)], min_fitness=None, sort="fitness",
                                   contributions=False, frozen=False, freeze=str(choice), cost_bps=[7, 15])
        selected = app.cmd_compare(train, None)
        records[0]["metrics"]["fitness"] = 999
        records[1]["metrics"]["fitness"] = -999
        source.write_text(json.dumps(records), encoding="utf-8")
        for override in ({"sort": "id"}, {"min_fitness": 9999}, {"cost_bps": [0]}):
            with self.subTest(override=override), self.assertRaises(CliError):
                app.cmd_compare(argparse.Namespace(files=[], contributions=False, frozen=True,
                    selection=str(choice), **override), None)
        validated = app.cmd_compare(argparse.Namespace(files=[], min_fitness=None, sort=None,
            contributions=False, frozen=True, selection=str(choice), summary=True), None)
        self.assertEqual([row["alpha_id"] for row in validated["rows"]],
                         [row["alpha_id"] for row in selected["rows"]])
        self.assertIn("missing_evidence", validated)
        self.assertIn("selection_id", validated)
        evidence = validated["rows"][0]["cost_evidence"]
        self.assertEqual([scenario["cost_bps"] for scenario in evidence["scenarios"]], [7, 15])
        self.assertEqual(evidence["period"], {"start": "2019-01-01", "end": "2020-12-31"})

    def test_halves_without_boundary_remain_unknown(self):
        pnl = {"2020-%03d" % day: (-1 if day % 2 else 2) for day in range(200)}
        self.assertEqual(app.half_split_sharpes(pnl), (None, None))

    def test_resume_keeps_accepted_job_identity(self):
        def accepted(expressions, settings, **kw):
            record = {"status": "PENDING", "submission_attempted": True,
                      "progress_url": "https://example.invalid/simulations/synthetic"}
            kw["on_event"]("submitted", {"index": 0, **record})
            raise KeyboardInterrupt
        with mock.patch.object(app.guarded_sim, "simulate", side_effect=accepted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_probe()
        captured = []
        def resume(expressions, settings, **kw):
            captured.extend(kw["initial_records"])
            return self.simulator(expressions, settings, **kw)
        with mock.patch.object(app.guarded_sim, "simulate", side_effect=resume):
            result = app.cmd_probe(arguments(resume=str(self.batch)), None)
        first = next(record for record in captured if record.get("mechanism") == "first")
        self.assertEqual(first["progress_url"], "https://example.invalid/simulations/synthetic")
        self.assertTrue(result["complete"])

    def test_incomplete_pnl_does_not_produce_train_halves(self):
        record = finished("SYNTHETIC_A")
        record["out_of_sample"]["test"] = {"startDate": "2022-01-01"}
        def incomplete(*args, **kwargs):
            kwargs["report"].update(complete=False, error="page missing")
            return {"2020-%03d" % day: day % 3 for day in range(200)}
        with mock.patch.object(ml, "fetch_daily_pnl", side_effect=incomplete), \
                mock.patch.object(app.research_paths, "ensure_dir", return_value=str(self.root)):
            app._probe_halves(object(), [record])
        self.assertEqual(record["half_sharpes"], (None, None))

    def test_selection_from_response_envelope_can_validate_canonical_batch(self):
        with mock.patch.object(app.guarded_sim, "simulate", side_effect=self.simulator):
            result = self.run_probe()
        response = self.root / "response.json"
        response.write_text(json.dumps({"data": result}), encoding="utf-8")
        choice = self.root / "choice.json"
        app.cmd_compare(argparse.Namespace(files=[str(response)], min_fitness=None, sort="none",
            contributions=False, frozen=False, freeze=str(choice)), None)
        validated = app.cmd_probe(arguments(recover=str(response), show_test=True, selection=str(choice)), None)
        self.assertEqual([row["alpha_id"] for row in validated["rows"]],
                         [row["alpha_id"] for row in result["rows"]])
        self.assertFalse(any(row.get("missing") for row in validated["rows"]))
