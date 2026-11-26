"""Offline contracts for durable submission and independently repairable effects."""

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

import alpha_registry
import cache_store
import machine_lib as ml
from research_app import alphas, buffer_cmd, confirm, selection
from research_app.runtime import CliError

SEED_DESCRIPTION = (
    "Idea:\n- Higher quality firms should earn higher subsequent returns.\n\n"
    "Rationale for data used:\n- assets: total reported corporate assets.\n\n"
    "Rationale for operators used:\n- rank: compares firms cross-sectionally."
)


def platform_alpha(alpha_id="CANDIDATE", status="COMPLETE"):
    return {"id": alpha_id, "status": status, "name": "Synthetic candidate",
            "category": "QUALITY",
            "regular": {"code": "rank(close)", "description": SEED_DESCRIPTION},
            "is": {"sharpe": 2.0, "fitness": 1.3, "turnover": 0.2, "margin": 0.001,
                   "checks": [{"name": "SELF_CORRELATION", "result": "PASS", "value": 0.2}]},
            "train": {"sharpe": 2.0, "margin": 0.001, "turnover": 0.2},
            "test": {"sharpe": 1.0},
            "dateSubmitted": "2026-01-02T03:04:05Z" if status == "ACTIVE" else None}


def complete_checks():
    return {"complete": True, "checks": {"SELF_CORRELATION": {"result": "PASS", "value": 0.2}},
            "self_correlation": {"result": "PASS", "value": 0.2}}


def complete_pnl(_session, _alpha_id, **kwargs):
    kwargs["report"].update(complete=True, source="fresh", pages=1)
    return {"2025-01-01": 1.0, "2025-01-02": 2.0}


class SubmissionWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": temporary.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        for module in (alphas, buffer_cmd):
            patcher = mock.patch.object(module, "with_session", lambda args, work: work(object()))
            patcher.start()
            self.addCleanup(patcher.stop)
        self.registry = str(self.root / "ledger.jsonl")
        self.shelf = str(self.root / "shelf.jsonl")
        self.manifest_path = self.root / "selection.json"
        for aid in ("CANDIDATE", "OTHER"):
            buffer_cmd._write_seed(buffer_cmd.seeds_path(),
                                   {"alpha_id": aid, "name": "seed " + aid,
                                    "category": "FUNDAMENTAL", "tags": ["test"],
                                    "description": SEED_DESCRIPTION})
        self.manifest = selection.freeze(self.manifest_path, source="triage",
                                         rows=[{"alpha_id": "CANDIDATE"}, {"alpha_id": "OTHER"}])

    def args(self, **overrides):
        values = {"alpha_id": "CANDIDATE", "registry": self.registry,
                  "name": "seed CANDIDATE", "category": "FUNDAMENTAL", "tags": ["test"],
                  "description": SEED_DESCRIPTION, "seeds": str(buffer_cmd.seeds_path()),
                  "buffer": self.shelf, "selection": str(self.manifest_path),
                  "yes": True, "max_wait": 0, "force": False, "note": None, "cost_bps": [5]}
        values.update(overrides)
        return argparse.Namespace(**values)

    def shelve(self, alpha_id="CANDIDATE"):
        buffer_cmd._append(self.shelf, {"event": "shelved", "alpha_id": alpha_id,
                                        "at": "2026-01-01", "eligible": True})

    def files(self):
        return {str(path.relative_to(self.root)): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def stage(self, command, args, post):
        with self.assertRaises(CliError) as caught:
            command(args, None)
        self.assertEqual(caught.exception.code, "CONFIRM_REQUIRED")
        post.assert_not_called()

    def test_quick_mode_blocks_both_entrypoints_before_confirmation(self):
        self.shelve()
        quick = {**platform_alpha(), "settings": {"simulationMode": "QUICK"}}
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=quick), \
                mock.patch.object(ml, "fetch_alpha_checks") as checks, \
                mock.patch.object(ml, "submit_alpha") as post, \
                mock.patch.object(confirm, "require_double_call") as confirm_intent:
            for command in (alphas.cmd_alpha_submit, buffer_cmd.cmd_buffer_submit):
                for yes in (False, True):
                    with self.subTest(command=command.__name__, yes=yes):
                        with self.assertRaises(CliError) as caught:
                            command(self.args(yes=yes, force=True), None)
                        self.assertEqual(caught.exception.code, "INVALID_INPUT")
                        self.assertIn("Full Mode", caught.exception.hint)
        checks.assert_not_called()
        post.assert_not_called()
        confirm_intent.assert_not_called()
        self.assertFalse(alphas._submission_path("CANDIDATE").exists())

    def test_mode_omitted_from_platform_uses_exact_saved_quick_evidence(self):
        alpha_registry.record_entry(alpha_registry.build_entry({
            "alpha_id": "CANDIDATE", "expression": "rank(close)",
            "settings": {"simulationMode": "QUICK"}}), path=self.registry)
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha()), \
                mock.patch.object(ml, "submit_alpha") as post:
            with self.assertRaises(CliError) as caught:
                alphas.cmd_alpha_submit(self.args(), None)
        self.assertIn("Quick Mode", caught.exception.message)
        post.assert_not_called()

        full_mode = {**platform_alpha(), "settings": {"simulationMode": "FULL"}}
        with mock.patch.object(ml, "fetch_alpha_metrics", side_effect=[platform_alpha("OTHER"), full_mode]), \
                mock.patch.object(ml, "submit_alpha") as post:
            self.stage(alphas.cmd_alpha_submit, self.args(alpha_id="OTHER"), post)
            self.stage(alphas.cmd_alpha_submit, self.args(), post)

    def test_quick_mode_found_on_final_read_records_no_post(self):
        session = mock.Mock()
        legacy = platform_alpha()
        quick = {**legacy, "settings": {"simulationMode": "QUICK"}}
        with mock.patch.object(alphas, "with_session", lambda args, work: work(session)), \
                mock.patch.object(ml, "agent_submit_allowed", return_value=True), \
                mock.patch.object(ml, "fetch_alpha_metrics", side_effect=[legacy, legacy, quick, quick]):
            self.stage(alphas.cmd_alpha_submit, self.args(), session.post)
            result = alphas.cmd_alpha_submit(self.args(), None)
        session.post.assert_not_called()
        self.assertFalse(result["post_attempted"])
        self.assertEqual(result["submission"], "NOT_SENT")
        journal, error = cache_store.load_json(alphas._submission_path("CANDIDATE"))
        self.assertIsNone(error)
        self.assertEqual(journal["state"], "NOT_SENT")

    def test_both_entrypoints_reconcile_same_facts_without_record_flag(self):
        records = []
        for command, alpha_id in ((alphas.cmd_alpha_submit, "CANDIDATE"),
                                  (buffer_cmd.cmd_buffer_submit, "OTHER")):
            self.shelve(alpha_id)
            state = {"active": False}

            def fetch(_session, requested):
                return platform_alpha(requested, "ACTIVE" if state["active"] else "COMPLETE")

            def submit(_session, requested, **kwargs):
                journal, error = cache_store.load_json(alphas._submission_path(requested))
                self.assertIsNone(error)
                self.assertEqual(journal["state"], "INTENT")
                self.assertTrue(kwargs["confirm"])
                state["active"] = True
                return {"submitted": True, "submission": "CONFIRMED", "status": "ACTIVE",
                        "date_submitted": "2026-01-02T03:04:05Z"}

            with mock.patch.object(ml, "fetch_alpha_metrics", side_effect=fetch), \
                    mock.patch.object(ml, "fetch_alpha_checks", return_value=complete_checks()), \
                    mock.patch.object(ml, "submit_alpha", side_effect=submit) as post, \
                    mock.patch.object(ml, "fetch_daily_pnl", side_effect=complete_pnl):
                args = self.args(alpha_id=alpha_id)
                self.stage(command, args, post)
                result = command(args, None)
            post.assert_called_once()
            self.assertTrue(result["reconciled"])
            self.assertTrue(result["reconciliation"]["shelf"]["consumed"])
            self.assertTrue(result["reconciliation"]["pnl"]["complete"])
            records.append(next(row for row in alpha_registry.load_entries(self.registry)
                                if row["alpha_id"] == alpha_id))
        for key in ("status", "metrics", "extra", "source", "expression"):
            self.assertEqual(records[0][key], records[1][key])
        self.assertEqual(records[0]["extra"]["dateSubmitted"], "2026-01-02T03:04:05Z")
        self.assertEqual(records[0]["extra"]["selection_id"], self.manifest["selection_id"])

    def test_ambiguous_transport_is_read_back_and_never_reposted(self):
        def timeout_after_post_boundary(_session, _alpha_id, **kwargs):
            kwargs["on_event"]({"event": "submit_posting"})
            raise requests.Timeout("unknown delivery")

        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha()) as reads, \
                mock.patch.object(ml, "submit_alpha", side_effect=timeout_after_post_boundary) as post:
            self.stage(alphas.cmd_alpha_submit, self.args(), post)
            first = alphas.cmd_alpha_submit(self.args(), None)
            second = alphas.cmd_alpha_submit(self.args(selection=None), None)
        self.assertEqual(first["submission"], "UNKNOWN")
        self.assertEqual(second["submission"], "UNKNOWN")
        self.assertFalse(second["post_attempted"])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(reads.call_count, 4)

    def test_new_attempt_requires_valid_seed_in_read_only_and_write_modes(self):
        path = buffer_cmd.seeds_path()
        path.write_text(json.dumps({"alpha_id": "CANDIDATE"}) + "\n", encoding="utf-8")
        for yes in (False, True):
            with self.subTest(yes=yes), \
                    mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha()), \
                    mock.patch.object(ml, "submit_alpha") as post:
                with self.assertRaises(CliError) as caught:
                    alphas.cmd_alpha_submit(self.args(yes=yes), None)
                self.assertIn("stored seed missing", caught.exception.message)
                post.assert_not_called()

    def test_new_attempt_requires_alpha_describe_to_match_the_seed_description(self):
        current = platform_alpha()
        current["regular"]["description"] = "Idea:\n- Different platform description."
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=current), \
                mock.patch.object(ml, "submit_alpha") as post:
            with self.assertRaises(CliError) as caught:
                alphas.cmd_alpha_submit(self.args(yes=False), None)
        self.assertIn("alpha describe", caught.exception.hint)
        post.assert_not_called()

    def test_pending_and_confirmed_attempts_do_not_depend_on_seed_store_health(self):
        buffer_cmd.seeds_path().write_text("broken seed JSON\n", encoding="utf-8")
        path = alphas._submission_path("CANDIDATE")
        cache_store.write_json_atomic(path, {"alpha_id": "CANDIDATE", "state": "UNKNOWN"})
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha()), \
                mock.patch.object(ml, "submit_alpha") as post:
            unknown = alphas.cmd_alpha_submit(self.args(yes=False, selection=None), None)
        self.assertEqual(unknown["submission"], "UNKNOWN")
        post.assert_not_called()
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha(status="ACTIVE")), \
                mock.patch.object(alphas, "_reconcile_submission", return_value={}), \
                mock.patch.object(ml, "submit_alpha") as post:
            confirmed = alphas.cmd_alpha_submit(self.args(selection=None), None)
        self.assertTrue(confirmed["submitted"])
        post.assert_not_called()

    def test_custom_seed_store_is_used_and_wrong_request_seed_is_rejected(self):
        path = self.root / "custom-seeds.jsonl"
        seed = {"alpha_id": "CANDIDATE", "name": "Seed", "category": "FUNDAMENTAL",
                "tags": ["quality"], "description": SEED_DESCRIPTION,
                "expression": "rank(close)", "settings": {"region": "USA"}}
        buffer_cmd._write_seed(path, seed)
        current = {**platform_alpha(), "settings": {"region": "USA"}}
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=current), \
                mock.patch.object(ml, "submit_alpha", return_value={"submission": "NOT_SENT"}) as precheck:
            result = alphas.cmd_alpha_submit(self.args(yes=False, seeds=str(path)), None)
        self.assertEqual(result["submission"], "NOT_SENT")
        precheck.assert_called_once()
        for changes in ({"expression": "rank(open)"}, {"settings": {"region": "EUR"}}):
            buffer_cmd._write_seed(path, {**seed, **changes})
            with self.subTest(changes=changes), \
                    mock.patch.object(ml, "fetch_alpha_metrics", return_value=current), \
                    mock.patch.object(ml, "submit_alpha") as post:
                with self.assertRaises(CliError):
                    alphas.cmd_alpha_submit(self.args(yes=False, seeds=str(path)), None)
                post.assert_not_called()

    def test_pre_post_metrics_failure_is_not_sent_and_does_not_poison_the_journal(self):
        session = mock.Mock()
        with mock.patch.object(alphas, "with_session", lambda args, work: work(session)), \
                mock.patch.object(confirm, "require_double_call"), \
                mock.patch.object(ml, "agent_submit_allowed", return_value=True), \
                mock.patch.object(ml, "fetch_alpha_metrics", side_effect=[
                    platform_alpha(), ml.ServiceError("pre-POST GET unavailable"), platform_alpha()]):
            result = alphas.cmd_alpha_submit(self.args(), None)
        session.post.assert_not_called()
        self.assertEqual(result["submission"], "NOT_SENT")
        self.assertFalse(result["post_attempted"])
        journal, _ = cache_store.load_json(alphas._submission_path("CANDIDATE"))
        self.assertEqual(journal["state"], "NOT_SENT")
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha()), \
                mock.patch.object(ml, "fetch_alpha_checks", return_value=complete_checks()) as check:
            precheck = alphas.cmd_alpha_submit(self.args(yes=False, selection=None), None)
        check.assert_called_once()
        self.assertEqual(precheck["submission"], "NOT_SENT")
        self.assertNotIn("submission_recovery", precheck)

    def test_actual_post_boundary_is_durable_before_transport_and_unsubmitted_does_not_clear_it(self):
        session = mock.Mock()

        def interrupted(*_args, **_kwargs):
            journal, _ = cache_store.load_json(alphas._submission_path("CANDIDATE"))
            self.assertEqual(journal["post_phase"], "STARTED")
            self.assertEqual(journal["state"], "INTENT")
            raise requests.Timeout("response lost after POST started")

        session.post.side_effect = interrupted
        unsubmitted = platform_alpha(status="UNSUBMITTED")
        with mock.patch.object(alphas, "with_session", lambda args, work: work(session)), \
                mock.patch.object(confirm, "require_double_call"), \
                mock.patch.object(ml, "agent_submit_allowed", return_value=True), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value=unsubmitted):
            first = alphas.cmd_alpha_submit(self.args(), None)
            second = alphas.cmd_alpha_submit(self.args(selection=None), None)
        self.assertEqual(first["submission"], "UNKNOWN")
        self.assertTrue(first["post_attempted"])
        self.assertEqual(second["submission"], "UNKNOWN")
        self.assertFalse(second["post_attempted"])
        session.post.assert_called_once()

    def test_interrupted_explicit_preparation_can_be_resolved_without_another_post(self):
        path = alphas._submission_path("CANDIDATE")
        cache_store.write_json_atomic(path, {"alpha_id": "CANDIDATE", "state": "INTENT",
                                            "post_phase": "PREPARING"})
        before = path.read_bytes()
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha()), \
                mock.patch.object(ml, "submit_alpha") as post:
            dry = alphas.cmd_alpha_submit(self.args(yes=False, selection=None), None)
            self.assertEqual(dry["submission"], "INTENT")
            self.assertEqual(path.read_bytes(), before)
            result = alphas.cmd_alpha_submit(self.args(selection=None), None)
        self.assertEqual(result["submission"], "NOT_SENT")
        self.assertTrue(result["reconciled"])
        self.assertFalse(result["post_attempted"])
        post.assert_not_called()

    def test_completed_submission_failure_resolves_uncertainty_without_reposting(self):
        path = alphas._submission_path("CANDIDATE")
        cache_store.write_json_atomic(path, {"alpha_id": "CANDIDATE", "state": "UNCERTAIN"})
        before = path.read_bytes()
        session = mock.Mock()
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"id": "CANDIDATE", "is": {"checks": [
            {"name": "LOW_FITNESS", "result": "FAIL", "value": 0.8, "limit": 1.0}]}}).encode()
        session.get.return_value = response
        with mock.patch.object(alphas, "with_session", lambda args, work: work(session)), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha(status="UNSUBMITTED")):
            dry = alphas.cmd_alpha_submit(self.args(yes=False, selection=None, max_wait=5), None)
            self.assertEqual(dry["submission"], "REJECTED")
            self.assertEqual(path.read_bytes(), before)
            result = alphas.cmd_alpha_submit(self.args(selection=None, max_wait=5), None)
        self.assertEqual(result["submission"], "REJECTED")
        self.assertTrue(result["reconciled"])
        self.assertFalse(result["post_attempted"])
        self.assertEqual(result["failed_checks"], ["LOW_FITNESS"])
        session.post.assert_not_called()
        self.assertEqual(session.get.call_count, 2)
        journal, _ = cache_store.load_json(path)
        self.assertEqual(journal["state"], "REJECTED")

    def test_submission_recovery_requires_ready_identified_failure_evidence(self):
        failed = {"id": "CANDIDATE", "is": {"checks": [{"name": "LOW_FITNESS", "result": "FAIL"}]}}
        cases = [(404, {}, {}, "unknown"), (200, {}, {}, "unknown"),
                 (200, {**failed, "id": "OTHER"}, {}, "unknown"),
                 (200, {**failed, "status": "PENDING"}, {}, "pending"),
                 (200, failed, {"Retry-After": "300"}, "pending"),
                 (200, {"is": {"checks": [{"name": "FITNESS", "result": "ERROR"}]}}, {}, "unknown"),
                 (200, {"is": "invalid"}, {}, "unknown")]
        for code, body, headers, state in cases:
            with self.subTest(code=code, body=body, headers=headers):
                session = mock.Mock()
                response = requests.Response()
                response.status_code, response._content = code, json.dumps(body).encode()
                response.headers.update(headers)
                session.get.return_value = response
                result = ml.read_alpha_submission_status(session, "CANDIDATE", max_wait_seconds=5)
                self.assertEqual(result["state"], state)
                session.post.assert_not_called()

    def test_failed_post_boundary_checkpoint_prevents_transport(self):
        session = mock.Mock()
        write = cache_store.write_json_atomic

        def fail_boundary(path, document, **kwargs):
            if document.get("state") == "INTENT" and document.get("post_phase") == "STARTED":
                raise OSError("submission journal write failed")
            return write(path, document, **kwargs)

        with mock.patch.object(alphas, "with_session", lambda args, work: work(session)), \
                mock.patch.object(confirm, "require_double_call"), \
                mock.patch.object(ml, "agent_submit_allowed", return_value=True), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha()), \
                mock.patch.object(cache_store, "write_json_atomic", side_effect=fail_boundary):
            result = alphas.cmd_alpha_submit(self.args(), None)
        session.post.assert_not_called()
        self.assertEqual(result["submission"], "NOT_SENT")
        self.assertFalse(result["post_attempted"])

    def test_gateway_timeout_response_keeps_the_journal_unresolved(self):
        session = mock.Mock()
        response = requests.Response()
        response.status_code = 504
        response.headers["Retry-After"] = "1"
        response._content = b"{}"
        session.post.return_value = response
        args = self.args(max_wait=10)
        with mock.patch.object(alphas, "with_session", lambda args, work: work(session)), \
                mock.patch.object(ml, "agent_submit_allowed", return_value=True), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha()):
            self.stage(alphas.cmd_alpha_submit, args, session.post)
            first = alphas.cmd_alpha_submit(args, None)
            second = alphas.cmd_alpha_submit(self.args(selection=None), None)
        self.assertEqual(first["submission"], "UNCERTAIN")
        self.assertEqual(first["status_code"], 504)
        self.assertEqual(second["submission"], "UNCERTAIN")
        self.assertFalse(second["post_attempted"])
        session.post.assert_called_once()
        session.get.assert_not_called()

    def test_status_read_failure_does_not_turn_an_accepted_post_into_a_refusal(self):
        session = mock.Mock()
        accepted = requests.Response()
        accepted.status_code = 201
        accepted.headers["Retry-After"] = "1"
        accepted._content = b"{}"
        missing = requests.Response()
        missing.status_code = 404
        missing._content = b"{}"
        session.post.return_value = accepted
        session.get.return_value = missing
        with mock.patch.object(ml, "agent_submit_allowed", return_value=True), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha()), \
                mock.patch.object(ml.time, "sleep"):
            result = ml.submit_alpha(session, "CANDIDATE", confirm=True, max_wait_seconds=10)
        self.assertEqual(result["submission"], "UNCERTAIN")
        self.assertEqual(result["status_code"], 404)
        session.post.assert_called_once()
        session.get.assert_called_once()

    def test_accepted_and_uncertain_are_not_reposted(self):
        quick = {**platform_alpha(), "settings": {"simulationMode": "QUICK"}}
        for state in ("INTENT", "ACCEPTED", "UNCERTAIN"):
            with self.subTest(state=state):
                cache_store.write_json_atomic(alphas._submission_path("CANDIDATE"),
                                              {"alpha_id": "CANDIDATE", "state": state})
                with mock.patch.object(ml, "fetch_alpha_metrics", return_value=quick), \
                        mock.patch.object(ml, "submit_alpha") as post:
                    result = alphas.cmd_alpha_submit(self.args(selection=None), None)
                post.assert_not_called()
                self.assertEqual(result["submission"], state)

    def test_confirmed_local_failure_is_repaired_idempotently_without_post(self):
        self.shelve()
        active = {**platform_alpha(status="ACTIVE"), "settings": {"simulationMode": "QUICK"}}
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=active), \
                mock.patch.object(ml, "submit_alpha") as post, \
                mock.patch.object(ml, "fetch_daily_pnl", side_effect=complete_pnl):
            with mock.patch.object(alpha_registry, "append_batch", side_effect=OSError("disk unavailable")):
                first = buffer_cmd.cmd_buffer_submit(self.args(selection=None), None)
            self.assertEqual(first["submission"], "CONFIRMED")
            self.assertIn("repair ledger", first["repair_actions"])
            journal, _ = cache_store.load_json(alphas._submission_path("CANDIDATE"))
            self.assertEqual(journal["state"], "CONFIRMED")
            second = buffer_cmd.cmd_buffer_submit(self.args(selection=None), None)
            ledger_bytes = Path(self.registry).read_bytes()
            third = alphas.cmd_alpha_submit(self.args(selection=None), None)
        post.assert_not_called()
        self.assertTrue(second["reconciled"])
        self.assertTrue(third["reconciled"])
        self.assertEqual(Path(self.registry).read_bytes(), ledger_bytes)
        self.assertEqual(sum(row["event"] == "consumed" for row in buffer_cmd.read_events(self.shelf)), 1)

    def test_dry_existing_active_never_changes_any_local_file(self):
        self.shelve()
        before = self.files()
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha(status="ACTIVE")), \
                mock.patch.object(ml, "submit_alpha") as post, \
                mock.patch.object(ml, "fetch_daily_pnl") as pnl:
            for command in (alphas.cmd_alpha_submit, buffer_cmd.cmd_buffer_submit):
                result = command(self.args(yes=False, selection=None), None)
                self.assertTrue(result["dry_run"])
                self.assertFalse(result["reconciled"])
        post.assert_not_called()
        pnl.assert_not_called()
        self.assertEqual(before, self.files())

    def test_precheck_needs_no_selection_but_new_submission_does(self):
        before = self.files()
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha()), \
                mock.patch.object(ml, "submit_alpha", return_value={"submission": "NOT_SENT", "submitted": False}) as post:
            alphas.cmd_alpha_submit(self.args(yes=False, selection=None), None)
            self.assertFalse(post.call_args.kwargs["confirm"])
            self.assertEqual(before, self.files())
            with self.assertRaises(CliError):
                alphas.cmd_alpha_submit(self.args(selection=None), None)
            with self.assertRaises(CliError):
                alphas.cmd_alpha_submit(self.args(alpha_id="ABSENT"), None)
        self.assertEqual(post.call_count, 1)

    def test_pnl_partial_does_not_erase_remote_success(self):
        def partial(_session, _alpha_id, **kwargs):
            kwargs["report"].update(complete=False, pages=2, reason="page bound")
            return {"2025-01-01": 1.0}
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha(status="ACTIVE")), \
                mock.patch.object(ml, "fetch_daily_pnl", side_effect=partial):
            result = alphas.cmd_alpha_submit(self.args(selection=None), None)
        self.assertEqual(result["submission"], "CONFIRMED")
        self.assertTrue(result["reconciliation"]["ledger"]["complete"])
        self.assertFalse(result["reconciliation"]["pnl"]["complete"])
        self.assertIn("repair pnl", result["repair_actions"])

    def test_active_rejected_by_shelf_and_unknown_checks_are_not_eligible(self):
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha(status="ACTIVE")):
            with self.assertRaises(CliError):
                buffer_cmd.cmd_buffer_add(self.args(), None)
        self.assertFalse(Path(self.shelf).exists())
        for checks in ({"complete": False, "checks": {}},
                       {"complete": True, "checks": {"SELF_CORRELATION": {"result": "PENDING"}}}):
            with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha()), \
                    mock.patch.object(ml, "fetch_alpha_checks", return_value=checks):
                result = buffer_cmd.cmd_buffer_add(self.args(selection=None), None)
            self.assertFalse(result["entry"]["eligible"])
            self.assertEqual(result["entry"]["eligibility"], "unknown")

    def test_shelf_events_are_idempotent_and_replace_failure_is_atomic(self):
        event = {"event": "shelved", "alpha_id": "CANDIDATE", "at": "first"}
        self.assertTrue(buffer_cmd._append(self.shelf, event))
        self.assertFalse(buffer_cmd._append(self.shelf, {**event, "at": "later"}))
        before = Path(self.shelf).read_bytes()
        with mock.patch.object(buffer_cmd.os, "replace", side_effect=OSError("rename refused")):
            with self.assertRaises(OSError):
                buffer_cmd.consume_submission(self.args())
        self.assertEqual(before, Path(self.shelf).read_bytes())
        self.assertTrue(buffer_cmd.consume_submission(self.args())["consumed"])
        self.assertFalse(buffer_cmd.consume_submission(self.args())["consumed"])
        self.assertEqual(len(buffer_cmd.read_events(self.shelf)), 2)


    def test_per_alpha_lock_serializes_two_real_submit_calls(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        first_post = Event()
        second_started = Event()
        state = {"active": False}

        def fetch(_session, alpha_id):
            return platform_alpha(alpha_id, "ACTIVE" if state["active"] else "COMPLETE")

        def submit(_session, alpha_id, **kwargs):
            first_post.set()
            self.assertTrue(second_started.wait(5), "second caller never started")
            state["active"] = True
            return {"submission": "CONFIRMED", "submitted": True, "status": "ACTIVE",
                    "date_submitted": "2026-01-02T03:04:05Z"}

        def second():
            self.assertTrue(first_post.wait(5), "first caller never submitted")
            second_started.set()
            return alphas.cmd_alpha_submit(self.args(), None)

        with mock.patch.object(ml, "fetch_alpha_metrics", side_effect=fetch), \
                mock.patch.object(ml, "submit_alpha", side_effect=submit) as post, \
                mock.patch.object(ml, "fetch_daily_pnl", side_effect=complete_pnl), \
                ThreadPoolExecutor(max_workers=2) as pool:
            self.stage(alphas.cmd_alpha_submit, self.args(), post)
            one = pool.submit(alphas.cmd_alpha_submit, self.args(), None)
            two = pool.submit(second)
            self.assertTrue(one.result(timeout=10)["submitted"])
            self.assertTrue(two.result(timeout=10)["submitted"])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(len(alpha_registry.load_entries(self.registry)), 1)

    def test_confirmed_journal_failure_still_reports_remote_success(self):
        path = alphas._submission_path("CANDIDATE")
        cache_store.write_json_atomic(path, {"alpha_id": "CANDIDATE", "state": "ACCEPTED"})
        atomic_write = cache_store.write_json_atomic

        def write(target, payload, **kwargs):
            if Path(target) == path:
                raise OSError("journal volume unavailable")
            return atomic_write(target, payload, **kwargs)

        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha(status="ACTIVE")), \
                mock.patch.object(ml, "submit_alpha") as post, \
                mock.patch.object(ml, "fetch_daily_pnl", side_effect=complete_pnl), \
                mock.patch.object(cache_store, "write_json_atomic", side_effect=write):
            result = alphas.cmd_alpha_submit(self.args(selection=None), None)
        post.assert_not_called()
        self.assertEqual(result["submission"], "CONFIRMED")
        self.assertTrue(result["reconciliation"]["ledger"]["complete"])
        self.assertIn("repair submission journal", result["repair_actions"])
        self.assertFalse(result["reconciled"])


if __name__ == "__main__":
    unittest.main()
