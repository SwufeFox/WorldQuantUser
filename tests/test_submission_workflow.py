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
from research_app import alphas, buffer_cmd, selection
from research_app.runtime import CliError


def platform_alpha(alpha_id="CANDIDATE", status="COMPLETE"):
    return {"id": alpha_id, "status": status, "name": "Synthetic candidate",
            "category": "QUALITY", "regular": {"code": "rank(close)"},
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
        self.manifest = selection.freeze(self.manifest_path, source="triage",
                                         rows=[{"alpha_id": "CANDIDATE"}, {"alpha_id": "OTHER"}])

    def args(self, **overrides):
        values = {"alpha_id": "CANDIDATE", "registry": self.registry,
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
                result = command(self.args(alpha_id=alpha_id), None)
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
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha()) as reads, \
                mock.patch.object(ml, "submit_alpha", side_effect=requests.Timeout("unknown delivery")) as post:
            first = alphas.cmd_alpha_submit(self.args(), None)
            second = alphas.cmd_alpha_submit(self.args(selection=None), None)
        self.assertEqual(first["submission"], "UNKNOWN")
        self.assertEqual(second["submission"], "UNKNOWN")
        self.assertFalse(second["post_attempted"])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(reads.call_count, 3)

    def test_accepted_and_uncertain_are_not_reposted(self):
        for state in ("INTENT", "ACCEPTED", "UNCERTAIN"):
            with self.subTest(state=state):
                cache_store.write_json_atomic(alphas._submission_path("CANDIDATE"),
                                              {"alpha_id": "CANDIDATE", "state": state})
                with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha()), \
                        mock.patch.object(ml, "submit_alpha") as post:
                    result = alphas.cmd_alpha_submit(self.args(selection=None), None)
                post.assert_not_called()
                self.assertEqual(result["submission"], state)

    def test_confirmed_local_failure_is_repaired_idempotently_without_post(self):
        self.shelve()
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform_alpha(status="ACTIVE")), \
                mock.patch.object(ml, "submit_alpha") as post, \
                mock.patch.object(ml, "fetch_daily_pnl", side_effect=complete_pnl):
            with mock.patch.object(alpha_registry, "record_entry", side_effect=OSError("disk unavailable")):
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
