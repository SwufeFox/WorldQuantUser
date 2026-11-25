"""Deterministic offline contracts for durable probe and frozen validation."""

import argparse
import json
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import requests

import machine_lib as ml
from research_app import campaign
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
                              ("_preflight_lint", lambda *a, **kw: {"ok": True}),
                              ("with_session", lambda args, fn, **kw: fn(object()))]:
            patch = mock.patch.object(app, target, value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(app.simulation_capabilities, "fetch", return_value={
            "state": "unknown", "account": {"state": "unknown"},
            "concurrency": {"state": "unknown", "limit": None}})
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

    def test_saved_spec_identity_does_not_repeat_history_scan(self):
        preview = self.run_probe(dry_run=True)
        with mock.patch.object(app, "_novelty_gate", side_effect=AssertionError("history scan")):
            identity = app.probe_spec_identity(self.spec)
        self.assertEqual(identity["request_fingerprint"], preview["request_fingerprint"])
        self.assertEqual(identity["design"], preview["design"])
        self.assertEqual(identity["track"], preview["track"])

    def test_preview_formats_the_native_request_without_rewriting_the_spec_or_its_identity(self):
        original = "x = rank(close); rank(x)"
        self.spec.write_text(json.dumps({"design": "explore", "arms": [{"expression": original}]}),
                             encoding="utf-8")
        before = self.spec.read_bytes()
        preview = self.run_probe(dry_run=True)
        wire = preview["results"][0]["payload"]["regular"]
        self.assertEqual(wire, "x = rank(close);\nrank(x)")
        self.assertEqual(self.spec.read_bytes(), before)
        self.assertEqual(app.probe_spec_identity(self.spec)["request_fingerprint"], preview["request_fingerprint"])
        self.assertEqual(app._recovery_expression_key(original, "FASTEXPR"),
                         app._recovery_expression_key(wire, "FASTEXPR"))
        self.assertNotEqual(app._recovery_expression_key('bucket(close, range="0  1")', "FASTEXPR"),
                            app._recovery_expression_key('bucket(close, range="0 1")', "FASTEXPR"))

    def test_preview_loads_turnover_history_once_across_fast_expression_groups(self):
        from research_app import mine

        catalog = {"close": {"dataset": "pv1"}}
        history = {"pv1": {"median": .15}}
        with mock.patch.object(mine, "load_catalog", return_value=catalog) as load, \
                mock.patch.object(mine, "dataset_turnover", return_value=history) as stats:
            preview = self.run_probe(dry_run=True)
        load.assert_called_once_with()
        stats.assert_called_once_with(catalog=catalog, datasets={"pv1"}, registry_path=str(self.registry))
        self.assertEqual(preview["groups"], 2)
        self.assertEqual(preview["results"][0]["turnover_band"]["low"], .15)

    def test_native_multi_preview_groups_compatible_settings_in_arm_order(self):
        with mock.patch.object(app, "cached_field_scope", return_value={"evidence": {"state": "unknown"}}), \
                mock.patch.object(app, "_turnover_band", return_value={}), \
                mock.patch.object(app, "with_session", side_effect=AssertionError("offline")):
            result = self.run_probe(dry_run=True, multi_size=2)
        self.assertEqual(result["arm_order"], ["first", "second", "third"])
        self.assertEqual(result["multi_size"], 2)
        self.assertEqual([(request["indices"], request["labels"]) for request in result["multi_requests"]],
                         [([0, 2], ["first", "third"]), ([1], ["second"])])
        payloads = [row["payload"] for row in result["results"]]
        self.assertEqual([request["payload"] for request in result["multi_requests"]],
                         [[payloads[0], payloads[2]], payloads[1]])

    def test_submit_cooldown_defers_fresh_arms_before_field_preflight(self):
        retry_at = 4000000000.0
        with mock.patch.object(app.simulation_slots, "submission_retry_at", return_value=retry_at), \
                mock.patch.object(app.simulation_slots, "confirmed_accepted_count") as admission, \
                mock.patch.object(app, "_preflight_lint") as preflight, \
                mock.patch.object(app.guarded_sim, "simulate") as launch, \
                mock.patch.object(app.ml, "simulate_alphas") as recover:
            result = self.run_probe(no_record=True)
        self.assertEqual(result["execution"]["not_submitted"], 3)
        self.assertEqual(result["execution"]["unknown_submission"], 0)
        self.assertTrue(all(item["submission_retry_at"] == retry_at
                            for item in result["scheduling"]))
        admission.assert_not_called()
        preflight.assert_not_called()
        launch.assert_not_called()
        recover.assert_not_called()

    def test_full_native_slots_defer_preflight_and_resume_when_evidence_is_unavailable(self):
        limit = app.simulation_multi.CONSULTANT_PARENT_LIMIT
        account = {"fingerprint": "probe-admission-test"}
        with mock.patch.object(app.simulation_capabilities, "account_provenance", return_value=account), \
                mock.patch.object(app.simulation_slots, "submission_retry_at", return_value=None), \
                mock.patch.object(app.simulation_slots, "confirmed_accepted_count", return_value=limit) as occupied, \
                mock.patch.object(app, "_preflight_lint", return_value={"ok": True}) as preflight, \
                mock.patch.object(app.guarded_sim, "simulate", side_effect=self.simulator) as launch:
            deferred = self.run_probe(multi_size=2, concurrency=limit + 2, no_record=True)
            preflight.assert_not_called()
            launch.assert_not_called()
            occupied.assert_called_with(account)
            self.assertEqual(deferred["execution"]["not_submitted"], 3)
            self.assertEqual(deferred["execution"]["unknown_submission"], 0)
            self.assertTrue(all(item == {
                "state": "deferred", "reason": "shared_slots_occupied",
                "confirmed_accepted_slots": limit, "effective_limit": limit,
            } for item in deferred["scheduling"]))

            checked = self.run_probe(multi_size=2, concurrency=limit + 2, check=True, no_record=True)
            self.assertTrue(checked["preflight_complete"])
            launch.assert_not_called()
            preflight.reset_mock()

            occupied.return_value = None
            resumed = app.cmd_probe(arguments(resume=str(self.batch)), None)
            self.assertTrue(resumed["complete"])
            self.assertEqual(preflight.call_count, 2)
            self.assertEqual(launch.call_count, 2)

    def test_field_service_failure_defers_only_the_affected_group_and_resumes_exactly_once(self):
        evidence = {"scope": {"region": "USA"}, "required_fields": {"unverified_ids": ["close"]},
                    "catalogs": [{"field": "close", "error": "SERVICE_ERROR"}]}

        def preflight(_session, pairs, _settings, _args, **kwargs):
            if pairs[0][0] == "rank(close)":
                raise app.FieldEvidenceRefreshError("SERVICE_ERROR", "field unavailable",
                                                    details={"field_catalog": evidence})
            return {"ok": True}

        with mock.patch.object(app, "_preflight_lint", side_effect=preflight), \
                mock.patch.object(app.guarded_sim, "simulate", side_effect=self.simulator) as launch:
            first = self.run_probe(multi_size=2)
            self.assertEqual(first["execution"]["not_submitted"], 2)
            self.assertEqual(first["recording"]["appended"], 1)
            saved = json.loads(self.batch.read_text(encoding="utf-8"))
            self.assertEqual(saved["preflight_deferred"], first["preflight_deferred"])
            deferred = saved["preflight_deferred"]["first"]
            self.assertEqual(deferred["labels"], ["first", "third"])
            self.assertEqual(deferred["catalogs"], evidence["catalogs"])
            self.assertFalse(saved["records"][0]["submission_attempted"])
            summary = app._summarize_probe_response(dict(first))
            self.assertEqual(summary["preflight_deferred_total"], 1)
            self.assertEqual(summary["preflight_deferred"]["first"]["code"], "SERVICE_ERROR")

            with mock.patch.object(app, "_preflight_lint", return_value={"ok": True}):
                resumed = app.cmd_probe(arguments(resume=str(self.batch)), None)

        self.assertTrue(resumed["complete"])
        self.assertFalse(resumed["preflight_deferred"])
        self.assertEqual(resumed["rows"][1]["alpha_id"], first["rows"][1]["alpha_id"])
        self.assertEqual(resumed["recording"]["appended"], 2)
        self.assertEqual([call.kwargs["labels"] for call in launch.call_args_list],
                         [["second"], ["first", "third"]])

    def test_field_rate_limit_defers_new_preflight_until_saved_retry_time(self):
        failure = app.FieldEvidenceRefreshError("RATE_LIMITED", "catalog rate limit", details={
            "field_catalog": {"catalogs": [{"field": "close", "error": "RATE_LIMITED", "retry_at": 160}]}})
        with (mock.patch.object(app.time, "time", return_value=100) as clock,
              mock.patch.object(app, "_preflight_lint", side_effect=failure) as preflight,
              mock.patch.object(app.guarded_sim, "simulate", side_effect=self.simulator) as launch):
            self.run_probe(multi_size=2, no_record=True)
            app.cmd_probe(arguments(resume=str(self.batch)), None)
            preflight.assert_called_once()
            launch.assert_not_called()

            clock.return_value = 160
            preflight.side_effect = None
            preflight.return_value = {"ok": True}
            result = app.cmd_probe(arguments(resume=str(self.batch)), None)

        self.assertTrue(result["complete"])
        self.assertFalse(result["preflight_deferred"])
        self.assertEqual([call.kwargs["labels"] for call in launch.call_args_list],
                         [["second"], ["first", "third"]])

    def test_platform_classification_stays_in_saved_validation_evidence(self):
        def simulate(expressions, settings, **kwargs):
            return [{**finished(label), "classification_ids": ["POWER_POOL_ELIGIBLE"]}
                    for label in kwargs["labels"]]

        with mock.patch.object(app.guarded_sim, "simulate", side_effect=simulate):
            result = self.run_probe(no_record=True)
        self.assertNotIn("classification_ids", result["rows"][0])
        saved = json.loads(self.batch.read_text(encoding="utf-8"))
        self.assertEqual(saved["rows"][0]["classification_ids"], ["POWER_POOL_ELIGIBLE"])

    def test_failed_native_progress_write_keeps_published_records_and_rows(self):
        settings = ml.resolve_settings()
        arms = [{"label": label, "expression": "rank(close)", "settings": settings, "ops": []}
                for label in ("first", "second", "third")]
        records = [{"status": "QUEUED", "settings": settings, "submission_attempted": False} for _ in arms]
        document = {"arms": arms, "records": records}
        app._save_probe_checkpoint(document, self.batch)
        original = self.batch.read_bytes()
        published_rows = document["rows"]
        payload = {"records": [{"index": index, "multi_request_id": "synthetic-request"}
                               for index in range(2)]}

        def fail_write(path, draft, **kwargs):
            self.assertEqual(path, self.batch)
            self.assertEqual([row["submission_state"] for row in draft["rows"]],
                             ["unknown_submission", "not_submitted", "unknown_submission"])
            self.assertEqual([row["status"] for row in records], ["QUEUED"] * len(arms))
            raise OSError("synthetic disk full")

        with mock.patch.object(app.cache_store, "write_json_atomic", side_effect=fail_write):
            with self.assertRaisesRegex(OSError, "synthetic disk full"):
                app._save_probe_progress(document, self.batch, [0, 2], "multi_submitting", payload)
        self.assertIs(document["rows"], published_rows)
        self.assertEqual(self.batch.read_bytes(), original)

        app._save_probe_progress(document, self.batch, [0, 2], "multi_submitting", payload)
        self.assertIs(document["records"], records)
        self.assertEqual([row["status"] for row in records], ["UNKNOWN_SUBMISSION", "QUEUED", "UNKNOWN_SUBMISSION"])
        self.assertEqual(json.loads(self.batch.read_text()), document)

    def test_native_batch_preserves_terminal_evidence_in_train_only_recovery(self):
        self.spec.write_text(json.dumps({"design": "explore", "settings": {"simulationMode": "QUICK"},
            "arms": [{"label": "first", "expression": "rank(close)"},
                     {"label": "second", "expression": "rank(volume)"}]}), encoding="utf-8")
        parent_url = ml.BRAIN_API_URL + "/simulations/NativeParent"
        payloads = {
            "NativeParent": {"status": "ERROR", "children": ["Child2", "Child1"], "message": "batch failed"},
            "Child2": {"id": "Child2", "parent": "NativeParent", "status": "TIMEOUT", "test_sharpe": 91},
            "Child1": {"id": "Child1", "parent": "NativeParent", "status": "FAIL", "detail": "resource limit"},
        }
        evidence = {"parent": {"status": "ERROR", "message": "batch failed"},
                    "children": {"Child2": {"status": "TIMEOUT"},
                                 "Child1": {"status": "FAIL", "message": "resource limit"}}}
        account = {"state": "known", "fingerprint": "probe-native-evidence-offline"}
        session = mock.Mock(_brain_permissions=["MULTI_SIMULATION"])

        def preflight(_session, pairs, settings, _args, **kwargs):
            return {"checked_requests": {"settings": settings, "pairs": [list(pair) for pair in pairs]},
                    "errors": []}

        with mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": str(self.root)}), \
                mock.patch.object(app, "with_session", side_effect=lambda args, fn, **kw: fn(session)), \
                mock.patch.object(app, "_preflight_lint", side_effect=preflight), \
                mock.patch.object(app.simulation_capabilities, "account_provenance", return_value=account), \
                mock.patch.object(ml, "submit_multi_simulation", return_value=parent_url) as post, \
                mock.patch.object(ml, "submit_single_simulation", side_effect=AssertionError("single POST")), \
                mock.patch.object(ml, "read_simulation_snapshot",
                                  side_effect=lambda s, identity, **kw: (payloads[identity], None)):
            result = self.run_probe(multi_size=2, no_record=True)
        post.assert_called_once()
        self.assertEqual(result["execution"]["terminal"], 2)
        self.assertEqual(result["rows"][0]["multi_terminal_evidence"], evidence)
        saved = json.loads(self.batch.read_text(encoding="utf-8"))
        self.assertEqual(saved["records"][0]["multi_terminal_evidence"], evidence)
        self.assertEqual(saved["rows"][0]["multi_terminal_evidence"], evidence)

        with mock.patch.object(app, "with_session", side_effect=AssertionError("already terminal")):
            recovered = app.cmd_probe(arguments(recover=str(self.batch)), None)
        self.assertFalse(recovered["show_test"])
        self.assertEqual(recovered["rows"][0]["multi_terminal_evidence"], evidence)
        self.assertIsNone(recovered["rows"][1].get("multi_terminal_evidence"))
        for row in recovered["rows"]:
            self.assertEqual(row["submission_state"], "terminal")
            self.assertEqual(row["multi_failure_attribution"], "unattributed")
            self.assertIsNone(row.get("multi_child_id"))
            self.assertIsNone(row["alpha_id"])
            self.assertNotIn("test_sharpe", row)

    def test_native_parent_checkpoint_recovers_entire_group_without_launching(self):
        parent_url = "https://api.worldquantbrain.com/simulations/synthetic-parent"

        def interrupted(expressions, settings, **kwargs):
            self.assertEqual(kwargs["multi_size"], 2)
            rows = [{"index": index, "multi_request_id": "synthetic-request", "multi_member": index,
                     "multi_size": 2, "submission_attempted": True, "status": "UNKNOWN_SUBMISSION"}
                    for index in range(2)]
            kwargs["on_event"]("multi_submitting", {"records": rows})
            saved = json.loads(self.batch.read_text())["records"]
            self.assertEqual([row.get("multi_request_id") for row in saved],
                             ["synthetic-request", None, "synthetic-request"])
            rows = [{**row, "status": "PENDING", "multi_parent_url": parent_url} for row in rows]
            kwargs["on_event"]("multi_submitted", {"records": rows})
            raise KeyboardInterrupt

        with mock.patch.object(app.guarded_sim, "simulate", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_probe(multi_size=2, no_record=True)
        with self.assertRaises(CliError):
            app.cmd_probe(arguments(resume=str(self.batch), dry_run=True, multi_size=3), None)
        preview = app.cmd_probe(arguments(resume=str(self.batch), dry_run=True), None)
        self.assertEqual([(request["indices"], request["labels"]) for request in preview["multi_requests"]],
                         [([1], ["second"])])

        with mock.patch.object(ml, "simulate_alphas", side_effect=lambda *a, **kw: kw["initial_records"]) as watch:
            app.cmd_probe(arguments(recover=str(self.batch), native_poll_once=True), None)
        watch.assert_called_once()
        self.assertIs(watch.call_args.kwargs["native_poll_once"], True)
        for overrides in ({"resume": str(self.batch), "dry_run": True},
                          {"spec": str(self.spec), "check": True}, {"from_simulation": "NativeParent"}):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(CliError, "requires a live probe run"):
                app.cmd_probe(arguments(native_poll_once=True, **overrides), None)

        def read_parent(session, expressions, **kwargs):
            self.assertIs(kwargs["recover_only"], True)
            self.assertNotIn("native_poll_once", kwargs)
            self.assertEqual(kwargs["multi_size"], 2)
            self.assertEqual(kwargs["labels"], ["first", "third"])
            records = kwargs["initial_records"]
            self.assertTrue(all(row["multi_parent_url"] == parent_url for row in records))
            return [{**row, **finished(label), "multi_parent_terminal": True}
                    for row, label in zip(records, ("SYNTHETIC_FIRST", "SYNTHETIC_THIRD"))]

        with mock.patch.object(ml, "simulate_alphas", side_effect=read_parent) as recover, \
                mock.patch.object(app.simulation_slots, "confirmed_accepted_count") as admission, \
                mock.patch.object(ml, "poll_simulation", side_effect=AssertionError("parent is not a single job")), \
                mock.patch.object(ml, "list_alphas", side_effect=AssertionError("no heuristic recovery")):
            result = app.cmd_probe(arguments(recover=str(self.batch)), None)
        recover.assert_called_once()
        admission.assert_not_called()
        self.assertEqual(result["multi_size"], 2)
        self.assertEqual(result["matched"], {"first": "SYNTHETIC_FIRST", "third": "SYNTHETIC_THIRD"})
        self.assertEqual(result["not_submitted"], ["second"])

        stored = json.loads(self.batch.read_text())
        stored["capability_snapshot"]["account"]["fingerprint"] = "original-account"
        self.batch.write_text(json.dumps(stored))
        with mock.patch.object(app.simulation_capabilities, "account_provenance",
                               return_value={"fingerprint": "other-account"}), \
                mock.patch.object(app, "_invocation_capabilities") as capabilities, \
                mock.patch.object(app.guarded_sim, "simulate") as launch:
            with self.assertRaisesRegex(CliError, "belongs to another"):
                app.cmd_probe(arguments(resume=str(self.batch)), None)
        capabilities.assert_not_called()
        launch.assert_not_called()

    def test_watch_native_launch_publishes_acceptance_before_poll_and_resumes_without_post(self):
        from tools._runner import probe_poll_at

        self.spec.write_text(json.dumps({"design": "explore", "arms": [
            {"label": "first", "expression": "rank(close)"},
            {"label": "second", "expression": "rank(volume)"}]}), encoding="utf-8")
        parent_url = ml.BRAIN_API_URL + "/simulations/NativeParent"
        account = {"state": "known", "fingerprint": "probe-watch-launch-offline"}
        accepted = requests.Response()
        accepted.status_code = 201
        accepted.headers["Location"] = parent_url
        pending = mock.Mock(status_code=200, headers={"Retry-After": "5"})
        pending.json.return_value = {"id": "NativeParent", "status": "SIMULATING"}
        session = mock.Mock(_brain_permissions=["MULTI_SIMULATION"])
        session.post.return_value = accepted

        def preflight(_session, pairs, settings, _args, **kwargs):
            return {"checked_requests": {"settings": settings, "pairs": [list(pair) for pair in pairs]},
                    "errors": []}

        def inspect(url, **kwargs):
            self.assertEqual(url, parent_url)
            saved = json.loads(self.batch.read_text(encoding="utf-8"))
            self.assertEqual([row["submission_state"] for row in saved["rows"]], ["accepted"] * 2)
            self.assertEqual(saved["records"][0]["slot_reservation_state"], "accepted")
            self.assertEqual(saved["records"][0]["multi_parent_accepted_at"], 100)
            self.assertNotIn("multi_parent_accepted_at", saved["records"][1])
            self.assertEqual(app.simulation_slots.confirmed_accepted_count(account), 1)
            return pending

        session.get.side_effect = inspect
        with (mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": str(self.root)}),
              mock.patch.object(app, "with_session", side_effect=lambda args, fn, **kw: fn(session)),
              mock.patch.object(app, "_preflight_lint", side_effect=preflight) as check,
              mock.patch.object(app.simulation_capabilities, "account_provenance", return_value=account),
              mock.patch.object(ml.time, "time", return_value=100) as wall,
              mock.patch.object(ml, "sleep", side_effect=AssertionError("no native polling delay"))):
            result = self.run_probe(multi_size=2, native_poll_once=True, no_record=True)
            self.assertEqual(result["execution"]["accepted"], 2)
            self.assertEqual(result["execution"]["unknown_submission"], 0)
            session.get.assert_called_once()
            saved = json.loads(self.batch.read_text(encoding="utf-8"))
            self.assertEqual(probe_poll_at(saved), 105)
            self.assertEqual(saved["records"][0]["multi_last_pending_observed_at"], 100)
            self.assertEqual(result["rows"][0]["multi_parent_accepted_at"], 100)
            self.assertNotIn("multi_parent_accepted_at", result["rows"][1])
            self.assertNotIn("native_poll_once", saved)

            app.cmd_probe(arguments(resume=str(self.batch), native_poll_once=True), None)
            session.get.assert_called_once()
            wall.return_value = 105
            app.cmd_probe(arguments(recover=str(self.batch), native_poll_once=True), None)
            self.assertEqual(session.get.call_count, 2)
            recovered = json.loads(self.batch.read_text(encoding="utf-8"))
            self.assertEqual(probe_poll_at(recovered), 110)
            self.assertEqual(recovered["records"][0]["multi_parent_accepted_at"], 100)
            self.assertEqual(recovered["records"][0]["multi_last_pending_observed_at"], 105)
        session.post.assert_called_once()
        check.assert_called_once()

    def test_check_reports_incomplete_without_simulating(self):
        with mock.patch.object(app, "_preflight_lint", return_value={"ok": False,
                "coverage": {"fields": "unknown"}}), mock.patch.object(app.guarded_sim, "simulate") as simulate:
            result = self.run_probe(check=True, allow_lint_errors=True)
        self.assertFalse(result["preflight_complete"])
        simulate.assert_not_called()
        self.assertFalse(self.batch.exists())

    def test_preflight_spends_time_from_the_probe_deadline(self):
        clock = mock.Mock(wraps=app.time)
        clock.monotonic.return_value = 0

        def preflight(*_, deadline, proof_cache):
            self.assertEqual(deadline, 30)
            clock.monotonic.return_value = 31
            return {"ok": True}

        with mock.patch.object(app, "time", clock), \
                mock.patch.object(app, "_preflight_lint", side_effect=preflight), \
                mock.patch.object(app.guarded_sim, "simulate") as simulate:
            result = self.run_probe(no_record=True)
        simulate.assert_not_called()
        self.assertEqual(result["execution"]["not_submitted"], 3)
        stored = json.loads(self.batch.read_text(encoding="utf-8"))
        self.assertEqual(stored["preflight"], [{"ok": True}])
        self.assertTrue(any(action["action"] == "resume" for action in result["next_actions"]))

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

    def test_incentive_evidence_survives_summary_and_offline_recovery(self):
        def simulate(expressions, settings, **kwargs):
            records = []
            for index, label in enumerate(kwargs["labels"]):
                record = finished("SYNTHETIC_" + label)
                record["checks"] = ml._extract_checks({"is": {"checks": [
                    {"name": "MATCHES_PYRAMID", "result": "PASS", "multiplier": 2.25,
                     "effective": 0, "pyramids": [{"name": "synthetic-%d" % n, "multiplier": 2.25}
                                                  for n in range(8)]},
                    {"name": "MATCHES_THEMES", "result": "PENDING"},
                ]}})
                kwargs["on_event"]("completed", {"index": index, **record})
                records.append(record)
            return records

        with mock.patch.object(app.guarded_sim, "simulate", side_effect=simulate):
            result = self.run_probe(summary=True)
        summary = result["summary"]
        self.assertEqual(summary["view"], "train")
        self.assertEqual(summary["incentives_reported_count"], 3)
        self.assertEqual(summary["incentives_unknown_count"], 0)
        reported = summary["incentives"][0]["checks"]
        self.assertEqual(reported["MATCHES_PYRAMID"]["pyramids"]["omitted"], 3)
        self.assertEqual(reported["MATCHES_PYRAMID"]["effective"], 0)
        self.assertEqual(reported["MATCHES_THEMES"]["result"], "PENDING")
        self.assertEqual(result["rows"][0]["train_sharpe"], 1.4)
        self.assertNotIn("sharpe", result["rows"][0])
        stored = json.loads(self.batch.read_text(encoding="utf-8"))
        self.assertEqual(len(stored["records"][0]["checks"]["MATCHES_PYRAMID"]["pyramids"]), 8)
        with mock.patch.object(app, "with_session", side_effect=AssertionError("offline recovery")):
            recovered = app.cmd_probe(arguments(recover=str(self.batch), summary=True), None)
        self.assertEqual(recovered["summary"]["incentives"], summary["incentives"])

    def test_summary_preserves_daily_quota_pause_and_recovery_instruction(self):
        session = requests.Session()
        session._brain_account_id = "offline-quota-summary"
        accepted = requests.Response()
        accepted.status_code = 201
        accepted.headers.update({"Location": "/simulations/one", "X-Ratelimit-Limit": "100",
                                 "X-Ratelimit-Remaining": "0", "X-Ratelimit-Reset": "3600"})
        accepted._content = b"{}"

        def simulate(expressions, settings, session=None, preflight=None, force=False, **kwargs):
            return ml.simulate_alphas(session, expressions, settings=settings, **kwargs)

        with mock.patch.object(app.research_paths, "research_dir", return_value=str(self.root)), \
                mock.patch.object(app, "with_session", side_effect=lambda args, work, **kw: work(session)), \
                mock.patch.object(app.guarded_sim, "simulate", side_effect=simulate), \
                mock.patch.object(session, "post", return_value=accepted) as post, \
                mock.patch.object(ml, "poll_simulation", return_value={"status": "COMPLETE", "alpha": "A1"}), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value={"is": {"sharpe": 1.0}}):
            result = self.run_probe(summary=True)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result["execution"]["not_submitted"], 2)
        self.assertEqual(result["summary"]["quota"]["remaining"], 0)
        self.assertTrue(result["summary"]["quota"]["blocked"])
        self.assertGreater(result["summary"]["quota"]["reset_after_seconds"], 0)
        self.assertTrue(any(action["action"] == "resume" for action in result["next_actions"]))
        stored = json.loads(self.batch.read_text(encoding="utf-8"))
        self.assertEqual(stored["records"][0]["alpha_id"], "A1")
        self.assertEqual(stored["scheduling"][-1]["quota"]["remaining"], 0)

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

    def test_saved_alpha_id_recovers_metrics_without_repolling_the_simulation(self):
        def interrupted(_expressions, _settings, **kwargs):
            emit = kwargs["on_event"]
            emit("submitted", {"index": 0, "status": "PENDING", "progress_url": "/simulations/finished"})
            emit("pending", {"index": 0, "status": "PENDING", "alpha_id": "KNOWN",
                             "metrics_pending": True, "simulation_status": "COMPLETE"})
            raise KeyboardInterrupt

        with mock.patch.object(app.guarded_sim, "simulate", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_probe()
        pending = json.loads(self.batch.read_text(encoding="utf-8"))["records"][0]
        self.assertTrue(pending["metrics_pending"])
        self.assertNotIn("progress_url", pending)

        payload = {"is": {"fitness": 1.3}, "train": {"fitness": 1.1}}
        with mock.patch.object(ml, "poll_simulation") as poll, mock.patch.object(ml, "list_alphas") as listing, \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value=payload) as fetch:
            result = app.cmd_probe(arguments(recover=str(self.batch)), None)
        poll.assert_not_called()
        listing.assert_not_called()
        fetch.assert_called_once_with(mock.ANY, "KNOWN", deadline=mock.ANY)
        self.assertEqual(result["matched"], {"first": "KNOWN"})
        self.assertEqual(result["rows"][0]["status"], "COMPLETE")
        self.assertEqual(result["recording"]["appended"], 1)
        saved = json.loads(self.batch.read_text(encoding="utf-8"))["records"][0]
        self.assertNotIn("metrics_pending", saved)
        self.assertNotIn("simulation_status", saved)

    def test_probe_launches_later_settings_groups_before_waiting_for_earlier_metrics(self):
        for resume in (False, True):
            with self.subTest(resume=resume):
                self.batch.unlink(missing_ok=True)
                if resume:
                    def interrupted(_expressions, _settings, **kwargs):
                        kwargs["on_event"]("pending", {"index": 0, "status": "PENDING",
                            "alpha_id": "close", "metrics_pending": True, "simulation_status": "COMPLETE"})
                        raise KeyboardInterrupt

                    with mock.patch.object(app.guarded_sim, "simulate", side_effect=interrupted), \
                         self.assertRaises(KeyboardInterrupt):
                        self.run_probe(no_record=True)

                posted = []
                posted_when_read = {}

                def submit(_session, payload, **_kwargs):
                    field = payload["regular"].removeprefix("rank(").removesuffix(")")
                    posted.append(field)
                    return "/simulations/" + field

                def poll(_session, location, **_kwargs):
                    return {"status": "COMPLETE", "alpha": location.rsplit("/", 1)[-1]}

                def fetch(_session, alpha_id, **_kwargs):
                    posted_when_read[alpha_id] = tuple(posted)
                    if alpha_id == "close":
                        return {"_deferred": True, "next_poll_after_seconds": 120}
                    return {"is": {"fitness": 1.3}, "train": {"fitness": 1.1}}

                def simulate(expressions, settings, session, preflight, force, **kwargs):
                    return ml.simulate_alphas(session, expressions, settings=settings, **kwargs)

                with mock.patch.object(app.guarded_sim, "simulate", side_effect=simulate), \
                     mock.patch.object(ml, "submit_single_simulation", side_effect=submit), \
                     mock.patch.object(ml, "poll_simulation", side_effect=poll), \
                     mock.patch.object(ml, "fetch_alpha_metrics", side_effect=fetch) as metrics:
                    if resume:
                        result = app.cmd_probe(arguments(resume=str(self.batch), batch_size=1), None)
                    else:
                        result = self.run_probe(no_record=True, batch_size=1)

                self.assertEqual(posted, ["volume", "cap"] if resume else ["close", "cap", "volume"])
                self.assertIn("volume", posted_when_read["close"])
                self.assertEqual(result["execution"]["accepted"], 3)
                self.assertEqual(metrics.call_count, 3)
                saved = json.loads(self.batch.read_text(encoding="utf-8"))["records"]
                self.assertTrue(saved[0]["metrics_pending"])
                self.assertEqual(saved[0]["poll_after_seconds"], 120)
                self.assertEqual(saved[1]["status"], "COMPLETE")

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

    def test_deferred_quick_pnl_keeps_results_and_recovers_without_new_simulations(self):
        self.spec = self.root / "backlog" / "done" / "selective.json"
        self.spec.parent.mkdir(parents=True)
        self.batch = self.root / "backlog-results" / "selective-batch.json"
        self.batch.parent.mkdir()
        status_args = argparse.Namespace(research_dir=str(self.root), limit=20, save=None)
        self.spec.write_text(json.dumps({"design": "explore",
            "settings": {"simulationMode": "QUICK", "testPeriod": "P1Y"},
            "arms": [{"label": "quick", "expression": "rank(close)"},
                     {"label": "quick-other", "expression": "rank(open)"},
                     {"label": "full", "expression": "rank(volume)",
                      "settings": {"simulationMode": "FULL"}}]}), encoding="utf-8")
        boundary = "2021-01-01"
        train_pnl = {(date(2020, 1, 1) + timedelta(days=day)).isoformat(): -1 if day % 2 else 2
                     for day in range(120)}
        expected_halves = app.train_half_sharpes(list(train_pnl.values()))

        def simulate(expressions, settings, **kwargs):
            records = [finished("SYNTHETIC_" + label) for label in kwargs["labels"]]
            for record in records:
                record["out_of_sample"]["test"] = {"startDate": boundary}
            return records

        def fetch_pnl(session, alpha_id, *, report, **kwargs):
            report["complete"] = True
            return {**train_pnl, boundary: -1000}

        with mock.patch.object(app.guarded_sim, "simulate", side_effect=simulate) as launch, \
                mock.patch.dict(os.environ, {app.research_paths.ENV_VAR: str(self.root)}), \
                mock.patch.object(app.research_paths, "ensure_dir", return_value=str(self.root)), \
                mock.patch.object(ml, "fetch_daily_pnl", side_effect=fetch_pnl) as fetch:
            result = self.run_probe(defer_quick_pnl=True, no_record=True)
            self.assertEqual(result["execution"]["finished"], 3)
            self.assertEqual(result["execution"]["train_evidence"], 3)
            self.assertEqual([row["alpha_id"] for row in result["rows"]],
                             ["SYNTHETIC_quick", "SYNTHETIC_quick-other", "SYNTHETIC_full"])
            self.assertEqual([row["train_sharpe"] for row in result["rows"]], [1.4, 1.4, 1.4])
            self.assertIsNone(result["rows"][0]["train_sharpe_h1"])
            self.assertEqual(result["rows"][2]["train_sharpe_h1"], expected_halves[0])
            fetch.assert_called_once()
            self.assertEqual(fetch.call_args.args[1], "SYNTHETIC_full")
            saved = json.loads(self.batch.read_text(encoding="utf-8"))
            pending = result["quick_pnl"]
            self.assertEqual(pending, saved["quick_pnl"])
            self.assertEqual(pending["pending_labels"], ["quick", "quick-other"])
            self.assertEqual(pending["recover_argv"][2:],
                             ["--research-dir", str(self.root), "probe", "--recover", str(self.batch)])
            self.assertIs(pending["platform_read_only"], True)
            status = campaign.cmd_campaign_status(status_args, None)["archived_quick_pnl"]
            self.assertEqual(status["pending_arms"], 2)

            with mock.patch.object(app, "with_session", side_effect=AssertionError("only deferred PnL remains")):
                still_pending = app.cmd_probe(arguments(recover=str(self.batch), defer_quick_pnl=True), None)
                for unknown in ("SYNTHETIC_full", "not-in-this-batch"):
                    with self.subTest(unknown=unknown), self.assertRaises(CliError) as invalid:
                        app.cmd_probe(arguments(recover=str(self.batch), pnl_alpha_id=[unknown]), None)
                    self.assertEqual(invalid.exception.code, "INVALID_INPUT")
            self.assertEqual(still_pending["quick_pnl"], pending)

            with mock.patch.object(ml, "submit_single_simulation") as single_post, \
                    mock.patch.object(ml, "submit_multi_simulation") as multi_post:
                chosen = arguments(recover=str(self.batch), pnl_alpha_id=["SYNTHETIC_quick"])
                selective = app.cmd_probe(chosen, None)
                self.assertEqual(selective["quick_pnl"]["pending_labels"], ["quick-other"])
                status = campaign.cmd_campaign_status(status_args, None)["archived_quick_pnl"]
                self.assertEqual(status["jobs"][0]["quick_pnl"], selective["quick_pnl"])
                self.assertEqual(selective["quick_pnl"],
                                 json.loads(self.batch.read_text(encoding="utf-8"))["quick_pnl"])
                self.assertEqual(selective["rows"][0]["train_sharpe_h1"], expected_halves[0])
                self.assertIsNone(selective["rows"][1]["train_sharpe_h1"])
                self.assertEqual([call.args[1] for call in fetch.call_args_list],
                                 ["SYNTHETIC_full", "SYNTHETIC_quick"])
                with mock.patch.object(app, "with_session", side_effect=AssertionError("chosen PnL is complete")):
                    repeated = app.cmd_probe(chosen, None)
                self.assertEqual(repeated["quick_pnl"], selective["quick_pnl"])
                recovered = app.cmd_probe(arguments(recover=str(self.batch)), None)
            single_post.assert_not_called()
            multi_post.assert_not_called()
            self.assertEqual(launch.call_count, 2)
            self.assertEqual([call.args[1] for call in fetch.call_args_list],
                             ["SYNTHETIC_full", "SYNTHETIC_quick", "SYNTHETIC_quick-other"])
            status = campaign.cmd_campaign_status(status_args, None)["archived_quick_pnl"]
            self.assertEqual(status["pending_batches"], 0)

        self.assertEqual(recovered["request_fingerprint"], result["request_fingerprint"])
        self.assertEqual(recovered["execution"], result["execution"])
        quick = recovered["rows"][0]
        self.assertEqual((quick["train_sharpe_h1"], quick["train_sharpe_h2"]), expected_halves)
        self.assertNotIn("quick_pnl", recovered)
        self.assertNotIn("quick_pnl", json.loads(self.batch.read_text(encoding="utf-8")))

    def test_probe_halves_stop_at_shared_deadline(self):
        records = [finished("FIRST"), finished("SECOND")]
        for record in records:
            record["out_of_sample"]["test"] = {"startDate": "2021-01-01"}

        clock = mock.Mock(wraps=app.time)
        clock.monotonic.side_effect = [8, 10]
        with mock.patch.object(app, "time", clock), \
                mock.patch.object(ml, "fetch_daily_pnl", return_value={}) as fetch:
            app._probe_halves(object(), records, deadline=10)
        fetch.assert_called_once()
        self.assertEqual(fetch.call_args.args[1], "FIRST")
        self.assertEqual(fetch.call_args.kwargs["deadline"], 10)
        self.assertNotIn("half_sharpes", records[1])

    def test_resume_keeps_accepted_job_identity(self):
        def accepted(expressions, settings, **kw):
            record = {"status": "PENDING", "submission_attempted": True,
                      "progress_url": "https://api.worldquantbrain.com/simulations/synthetic"}
            kw["on_event"]("submitted", {"index": 0, **record})
            raise KeyboardInterrupt
        with mock.patch.object(app.guarded_sim, "simulate", side_effect=accepted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_probe()
        captured = []
        def resume(expressions, settings, **kw):
            captured.extend(kw["initial_records"])
            return self.simulator(expressions, settings, **kw)
        with mock.patch.object(app.guarded_sim, "simulate", side_effect=resume), \
                mock.patch.object(ml, "poll_simulation", return_value={
                    "status": "COMPLETE", "alpha": "SYNTHETIC_first"}) as poll, \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value={
                    "is": {"fitness": 1.1}, "train": {"fitness": 1.0}}), \
                mock.patch.object(ml, "submit_single_simulation", side_effect=AssertionError("accepted job reposted")):
            result = app.cmd_probe(arguments(resume=str(self.batch)), None)
        self.assertEqual(poll.call_args.args[1], "https://api.worldquantbrain.com/simulations/synthetic")
        self.assertFalse(any(record.get("mechanism") == "first" for record in captured))
        first = next(row for row in result["rows"] if row["label"] == "first")
        self.assertEqual(first["alpha_id"], "SYNTHETIC_first")
        self.assertTrue(result["complete"])

    def test_terminal_recovery_releases_the_slot_and_retires_the_poll_route(self):
        self.spec.write_text(json.dumps({"name": "synthetic", "design": "explore",
            "arms": [{"label": "first", "expression": "rank(close)"}]}), encoding="utf-8")
        account = {"fingerprint": "synthetic-terminal-recovery"}
        url = "https://api.worldquantbrain.com/simulations/synthetic"

        def accepted(expressions, settings, **kwargs):
            identity, _ = app.simulation_slots.reserve(account, 1)
            record = {"index": 0, "status": "PENDING", "submission_attempted": True,
                      "slot_reservation": identity, "progress_url": url}
            app.simulation_slots.reconcile_record(record, account, outcome="accepted")
            kwargs["on_event"]("submitted", record)
            raise KeyboardInterrupt

        with mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": str(self.root)}):
            with mock.patch.object(app.guarded_sim, "simulate", side_effect=accepted):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_probe(no_record=True)
            self.assertEqual(app.simulation_slots.snapshot(account, 1)["active"], 1)
            with mock.patch.object(app.simulation_capabilities, "account_provenance", return_value=account), \
                    mock.patch.object(ml, "poll_simulation", return_value={
                        "status": "ERROR", "message": "synthetic failure"}) as poll, \
                    mock.patch.object(ml, "list_alphas") as listing:
                result = app.cmd_probe(arguments(recover=str(self.batch)), None)
            poll.assert_called_once()
            listing.assert_not_called()
            self.assertEqual(result["rows"][0]["status"], "ERROR")
            self.assertEqual(result["rows"][0]["error"], "synthetic failure")
            self.assertEqual(result["execution"]["terminal"], 1)
            self.assertEqual(result["execution"]["unknown_submission"], 0)
            self.assertEqual(result["still_missing"], ["first"])
            self.assertEqual(result["next_actions"], [])
            self.assertEqual(app.simulation_slots.snapshot(account, 1)["active"], 0)
            saved = json.loads(self.batch.read_text(encoding="utf-8"))
            self.assertNotIn("progress_url", saved["records"][0])

            # Earlier checkpoints could retain a stale route after terminal polling.
            saved["records"][0].update(progress_url=url, next_poll_at=1000)
            self.batch.write_text(json.dumps(saved), encoding="utf-8")
            with mock.patch.object(app, "with_session", side_effect=AssertionError("terminal arm went online")), \
                    mock.patch.object(ml, "submit_single_simulation") as post:
                for mode in ("recover", "resume"):
                    repeated = app.cmd_probe(arguments(**{mode: str(self.batch)}), None)
                    self.assertEqual(repeated["next_actions"], [])
                    self.assertNotIn("progress_url", repeated["rows"][0])
            post.assert_not_called()

    def test_unposted_failure_is_not_a_terminal_platform_task(self):
        record = {"status": "FAILED", "submission_attempted": False}
        self.assertEqual(app._submission_state(record), "not_submitted")

    def test_probe_restores_progress_saved_only_in_slot_journal(self):
        self.spec.write_text(json.dumps({"name": "synthetic", "design": "explore",
            "arms": [{"label": "first", "expression": "rank(close)"}]}), encoding="utf-8")
        account = {"fingerprint": "synthetic-probe-recovery"}
        progress_url = "https://api.worldquantbrain.com/simulations/synthetic"

        def interrupted(expressions, settings, **kwargs):
            reservation, _ = app.simulation_slots.reserve(account, 1)
            attempt = {"index": 0, "status": "UNKNOWN_SUBMISSION", "submission_attempted": True,
                       "slot_reservation": reservation}
            kwargs["on_event"]("submitting", attempt)
            app.simulation_slots.reconcile_record(attempt, account, outcome="accepted",
                                                  progress_url=progress_url)
            raise KeyboardInterrupt

        for mode in ("resume", "recover"):
            with self.subTest(mode=mode), mock.patch.dict(
                    os.environ, {"BRAIN_RESEARCH_DIR": str(self.root)}):
                self.batch = self.root / (mode + ".json")
                with mock.patch.object(app.guarded_sim, "simulate", side_effect=interrupted):
                    with self.assertRaises(KeyboardInterrupt):
                        self.run_probe(no_record=True)
                saved = json.loads(self.batch.read_text(encoding="utf-8"))
                self.assertEqual(saved["records"][0]["status"], "UNKNOWN_SUBMISSION")
                self.assertNotIn("progress_url", saved["records"][0])
                if mode == "resume":
                    with mock.patch.object(app, "with_session", side_effect=AssertionError("offline")):
                        preview = app.cmd_probe(arguments(resume=str(self.batch), dry_run=True), None)
                    self.assertEqual(preview["continuing"], ["first"])
                    self.assertEqual(preview["rows"][0]["progress_url"], progress_url)
                    self.assertEqual(json.loads(self.batch.read_text(encoding="utf-8")), saved)
                else:
                    with mock.patch.object(app.simulation_capabilities, "account_provenance",
                                           return_value={"fingerprint": "another-account"}), \
                            mock.patch.object(ml, "poll_simulation") as wrong_account_poll:
                        with self.assertRaisesRegex(CliError, "another or unidentified account"):
                            app.cmd_probe(arguments(recover=str(self.batch)), None)
                    wrong_account_poll.assert_not_called()
                    self.assertEqual(json.loads(self.batch.read_text(encoding="utf-8")), saved)

                with mock.patch.object(app.simulation_capabilities, "account_provenance",
                                       return_value=account), \
                        mock.patch.object(ml, "poll_simulation", return_value={
                            "status": "COMPLETE", "alpha": "SYNTHETIC_first"}) as poll, \
                        mock.patch.object(ml, "fetch_alpha_metrics", return_value={
                            "status": "COMPLETE", "is": {"fitness": 1.1}, "train": {"fitness": 1.0}}), \
                        mock.patch.object(ml, "submit_single_simulation") as post, \
                        mock.patch.object(ml, "list_alphas") as search:
                    result = app.cmd_probe(arguments(**{mode: str(self.batch)}), None)
                self.assertEqual(poll.call_args.args[1], progress_url)
                post.assert_not_called()
                search.assert_not_called()
                self.assertEqual(result["rows"][0]["alpha_id"], "SYNTHETIC_first")
                self.assertEqual(app.simulation_slots.snapshot(account, 1)["active"], 0)

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


class UnsubmittedTerminalTests(unittest.TestCase):
    def test_unsubmitted_row_needs_no_recovery(self):
        from research_app.experiments import _probe_needs_recovery

        row = {"status": "UNSUBMITTED", "alpha_id": "abc", "train_fitness": None}
        self.assertFalse(_probe_needs_recovery(row))

    def test_guard_refusal_does_not_keep_batch_incomplete(self):
        from research_app.experiments import _batch_incomplete_labels

        rows = [{"label": "refused", "train_fitness": None},
                {"label": "done", "train_fitness": 1.2}]
        records = [{"status": "UNSUBMITTED"}, {"status": "COMPLETE"}]
        self.assertEqual(_batch_incomplete_labels(rows, records), [])
