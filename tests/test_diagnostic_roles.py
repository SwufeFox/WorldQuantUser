"""Offline intent, selection, presentation and scheduling contracts."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import alpha_registry
import machine_lib as ml
from research_app import alphas, buffer_cmd, experiments, promote, selection
from research_app.runtime import CliError


class DiagnosticRoleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = str(self.root / "registry.jsonl")
        patch = mock.patch.object(alpha_registry, "default_registry_path", return_value=self.registry)
        patch.start()
        self.addCleanup(patch.stop)

    def store(self, role=None):
        entry = alpha_registry.build_entry({"alpha_id": "SYNTHETIC_ROLE", "role": role,
            "expression": "rank(close)", "metrics": {"fitness": 0}, "status": "COMPLETE"})
        return alpha_registry.record_entry(entry, self.registry)[0]

    def test_role_is_explicit_and_survives_rerecord_and_refresh(self):
        self.assertIsNone(alpha_registry.evidence_role({"label": "candidate", "note": "diagnostic only"}))
        self.store("diagnostic")
        refreshed = self.store()
        self.assertEqual(refreshed["role"], "diagnostic")
        self.assertEqual(self.store("candidate")["role"], "diagnostic")
        platform = alphas._platform_entry({"regular": {"code": "rank(close)"}, "is": {"fitness": 0}},
                                         "SYNTHETIC_ROLE", refreshed)
        self.assertEqual(platform["role"], "diagnostic")
        with self.assertRaisesRegex(ValueError, "role"):
            alpha_registry.evidence_role({"role": "control"})

    def test_diagnostic_freeze_and_legacy_manifest_are_guarded(self):
        path = self.root / "selection.json"
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            selection.freeze(path, source="compare", rows=[{"alpha_id": "SYNTHETIC_ROLE", "role": "diagnostic"}])
        manifest = selection.freeze(path, source="compare", rows=[{"alpha_id": "SYNTHETIC_ROLE"}])
        joined = selection.ordered_rows(manifest, [{"alpha_id": "SYNTHETIC_ROLE"}])
        self.assertIsNone(joined[0]["role"])
        self.store("diagnostic")
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            selection.load(path)

    def test_cross_source_diagnostic_cannot_reenter_frozen_order(self):
        manifest = selection.freeze(self.root / "selection.json", source="compare", rows=[
            {"alpha_id": "SYNTHETIC_ROLE", "source_file": "first.json", "role": "candidate"}])
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            selection.ordered_rows(manifest, [
                {"alpha_id": "SYNTHETIC_ROLE", "source_file": "first.json", "role": "candidate"},
                {"alpha_id": "SYNTHETIC_ROLE", "source_file": "other.json", "role": "diagnostic"}])

    def test_saved_source_intent_invalidates_stale_manifest(self):
        source = self.root / "source.json"
        source.write_text(json.dumps({"results": [{"alpha_id": "SYNTHETIC_ROLE", "role": "candidate"}]}))
        path = self.root / "selection.json"
        selection.freeze(path, source="compare", rows=[{"alpha_id": "SYNTHETIC_ROLE", "source_file": str(source)}],
                         inputs={"files": [str(source)]})
        source.write_text(json.dumps({"results": [{"alpha_id": "SYNTHETIC_ROLE", "role": "diagnostic"}]}))
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            selection.load(path)

    def test_shelf_promotion_and_new_submission_reject_diagnostic(self):
        self.store("diagnostic")
        args = argparse.Namespace(alpha_id="SYNTHETIC_ROLE", registry=self.registry, selection=None)
        for command in (buffer_cmd.cmd_buffer_add, promote.cmd_promote):
            with self.subTest(command=command.__name__), self.assertRaisesRegex(CliError, "diagnostic"):
                command(args, None)
        with self.assertRaisesRegex(CliError, "diagnostic"):
            alphas._candidate_role(args, {"alpha_id": args.alpha_id, "role": "candidate"})

    def test_probe_summary_bounds_detail_and_keeps_zero_distinct(self):
        rows = [{"label": "arm%d" % i, "role": "diagnostic" if i == 0 else None,
                 "status": "COMPLETE", "train_sharpe": 0, "train_fitness": None,
                 "cost_evidence": {"margin_bps": 0}, "sharpe": 999} for i in range(25)]
        response = {"path": str(self.root / "batch.json"), "request_fingerprint": "synthetic-request",
                    "rows": rows, "execution": {"total": 25}, "recording": {"state": "reconciled"},
                    "preflight": [{"ok": True, "results": [{}] * 100, "checked_requests": [{}] * 100}]}
        experiments._summarize_probe_response(response)
        self.assertEqual(len(response["rows"]), 20)
        self.assertEqual(len(rows), 25)
        self.assertEqual(response["summary"]["rows_omitted"], 5)
        self.assertEqual(response["summary"]["roles"]["unspecified"], 24)
        self.assertEqual(response["summary"]["missing_evidence_count"], 25)
        self.assertIn("| 0 | - |", response["summary"]["table"])
        self.assertNotIn("999", response["summary"]["table"])
        self.assertEqual(response["preflight"][0]["checked_requests_omitted"], 100)
        self.assertNotIn("results", response["preflight"][0])

    def test_compare_markdown_preserves_train_only_rows_and_missing(self):
        source = self.root / "results.json"
        source.write_text(json.dumps({"results": [{"alpha_id": "SYNTHETIC_ROLE", "role": "diagnostic",
            "mechanism": "attribution", "status": "COMPLETE", "metrics": {"fitness": 999},
            "out_of_sample": {"train": {"fitness": 0, "sharpe": 0, "turnover": 0, "margin": 0}}}]}))
        args = argparse.Namespace(files=[str(source)], sort="none", min_fitness=None,
                                  contributions=False, format="markdown", frozen=False)
        result = experiments.cmd_compare(args, None)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["rows"][0]["role"], "diagnostic")
        self.assertIn("| 0 | 0 | 0 | 0.0 |", result["table"])
        self.assertNotIn("999", result["table"])
        args.freeze = str(self.root / "freeze.json")
        with self.assertRaisesRegex(CliError, "diagnostic"):
            experiments.cmd_compare(args, None)

    def test_probe_resume_changes_policy_without_request_or_role_drift(self):
        settings = ml.resolve_settings({})
        arms = [{"label": "control", "expression": "rank(close)", "settings": settings,
                 "role": "diagnostic", "ops": 1, "arm_index": 0}]
        document = {"format": "probe-batch", "version": 2, "name": "synthetic", "arms": arms,
                    "request_fingerprint": experiments._request_fingerprint(arms), "concurrency": 2,
                    "recording_policy": {"enabled": False}, "records": [{"expression": "rank(close)",
                    "settings": settings, "status": "QUEUED", "role": "diagnostic",
                    "submission_attempted": False}]}
        path = self.root / "batch.json"
        experiments._probe_document_update(document, path)
        path.write_text(json.dumps(document))
        snapshot = {"state": "unknown", "account": {"state": "unknown"},
                    "concurrency": {"state": "unknown", "limit": None}}
        report = {"ok": True, "checked_requests": [{"settings": settings}]}
        def simulate(expressions, resolved, **kwargs):
            self.assertIs(kwargs["preflight"], report)
            self.assertIs(kwargs["capability_snapshot"], snapshot)
            self.assertEqual(kwargs["concurrency"], 7)
            self.assertEqual(kwargs["initial_records"][0]["role"], "diagnostic")
            kwargs["scheduling_report"].update(requested=7, effective=7, source="explicit_local_policy")
            return [{"status": "QUEUED", "submission_attempted": False}]
        args = argparse.Namespace(resume=str(path), concurrency=7, summary=True, max_wait_seconds=10)
        with mock.patch.object(experiments, "with_session", side_effect=lambda args, fn: fn(object())), \
                mock.patch.object(experiments.simulation_capabilities, "fetch", return_value=snapshot) as fetch, \
                mock.patch.object(experiments, "_preflight_lint", return_value=report), \
                mock.patch.object(experiments.guarded_sim, "simulate", side_effect=simulate):
            result = experiments.cmd_probe(args, None)
        fetch.assert_called_once_with(mock.ANY, refresh=True)
        saved = json.loads(path.read_text())
        self.assertEqual(saved["request_fingerprint"], document["request_fingerprint"])
        self.assertEqual(saved["recording_policy"], document["recording_policy"])
        self.assertEqual(saved["concurrency"], 7)
        self.assertEqual(saved["scheduling"][0]["effective"], 7)
        self.assertEqual(saved["records"][0]["role"], "diagnostic")
        self.assertEqual(result["rows"][0]["role"], "diagnostic")
        self.assertEqual(result["next_actions"][0]["argv"][-2:], ["--resume", str(path)])
        with self.assertRaises(CliError):
            experiments.cmd_probe(argparse.Namespace(resume=str(path), setting=["extension=true"]), None)

    def test_checkpoint_event_and_recovery_preserve_role(self):
        path = self.root / "checkpoint.json"
        checkpoint = {"records": [{"role": "diagnostic", "status": "QUEUED"}]}
        callback = experiments._simulation_checkpoint_callback(path, checkpoint, None)
        callback("submitted", {"index": 0, "status": "PENDING", "alpha_id": "SYNTHETIC_ROLE"})
        self.assertEqual(json.loads(path.read_text())["records"][0]["role"], "diagnostic")
        batch = self.root / "legacy.json"
        batch.write_text(json.dumps({"rows": [{"label": "C1", "role": "diagnostic", "expression": "rank(close)",
                                               "settings": ml.resolve_settings({}), "status": "QUEUED"}]}))
        _, recovered = experiments._load_probe_batch(batch)
        self.assertEqual(recovered["arms"][0]["role"], "diagnostic")
        self.assertEqual(recovered["records"][0]["role"], "diagnostic")
        self.assertEqual(recovered["rows"][0]["role"], "diagnostic")

    def test_new_submission_cannot_use_manifest_predating_diagnostic_intent(self):
        choice = self.root / "choice.json"
        selection.freeze(choice, source="compare", rows=[{"alpha_id": "SYNTHETIC_ROLE"}])
        self.store("diagnostic")
        args = argparse.Namespace(alpha_id="SYNTHETIC_ROLE", registry=self.registry,
                                  selection=str(choice), yes=True, max_wait=1)
        with mock.patch.object(alphas, "_submission_path", return_value=self.root / "submission.json"), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value={"status": "UNSUBMITTED"}), \
                mock.patch.object(buffer_cmd, "_book_evidence", return_value={}), \
                mock.patch.object(ml, "submit_alpha") as submit:
            with self.assertRaisesRegex(CliError, "diagnostic"):
                alphas.submit_workflow(object(), args)
        submit.assert_not_called()

    def test_recovery_refresh_keeps_explicit_diagnostic_intent(self):
        batch = self.root / "recover.json"
        batch.write_text(json.dumps({"rows": [{"alpha_id": "SYNTHETIC_ROLE", "label": "control",
            "role": "diagnostic", "expression": "rank(close)", "settings": ml.resolve_settings({}),
            "status": "PENDING"}]}))
        payload = {"status": "COMPLETE", "is": {"fitness": 0, "sharpe": 0},
                   "train": {"fitness": 0, "sharpe": 0}}
        with mock.patch.object(experiments, "with_session", side_effect=lambda args, fn: fn(object())), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value=payload), \
                mock.patch.object(ml, "extract_split", return_value={"train": payload["train"]}):
            result = experiments.cmd_probe(argparse.Namespace(recover=str(batch), summary=True), None)
        saved = json.loads(batch.read_text())
        self.assertEqual(saved["records"][0]["role"], "diagnostic")
        self.assertEqual(result["rows"][0]["role"], "diagnostic")
        self.assertEqual(result["rows"][0]["train_fitness"], 0)

    def test_polling_saved_jobs_does_not_recheck_reduced_launch_permissions(self):
        settings = ml.resolve_settings({})
        arm = {"label": "accepted", "expression": "rank(close)", "settings": settings,
               "role": "diagnostic", "arm_index": 0, "ops": 1}
        document = {"arms": [arm], "name": "synthetic", "records": [{"expression": arm["expression"],
            "settings": settings, "role": "diagnostic", "status": "PENDING", "submission_attempted": True,
            "progress_url": "https://example.invalid/simulations/synthetic"}],
            "recording_policy": {"enabled": False}}
        path = self.root / "accepted.json"
        experiments._probe_document_update(document, path)
        path.write_text(json.dumps(document))
        snapshot = {"state": "verified", "account": {"state": "unknown"}, "settings_schema": {}}
        with mock.patch.object(experiments, "with_session", side_effect=lambda args, fn: fn(object())), \
                mock.patch.object(experiments.simulation_capabilities, "fetch", return_value=snapshot), \
                mock.patch.object(experiments, "_preflight_lint", side_effect=AssertionError("launch-only")), \
                mock.patch.object(experiments.guarded_sim, "simulate", side_effect=AssertionError("launch-only")), \
                mock.patch.object(ml, "simulate_alphas", return_value=[document["records"][0]]) as poll:
            result = experiments.cmd_probe(argparse.Namespace(resume=str(path), concurrency=1), None)
        poll.assert_called_once()
        self.assertEqual(result["execution"]["accepted"], 1)
        self.assertEqual(result["rows"][0]["role"], "diagnostic")
