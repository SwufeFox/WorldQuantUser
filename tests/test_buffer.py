#!/usr/bin/env python3
"""Tests for `buffer`: the shelf of checked-but-unsubmitted alphas.

The shelf exists because submitting is irreversible and the platform's self-correlation
decays as the book grows. So the tests pin three things: an entry stores the evidence it
was judged on, a failing check never reaches the shelf, and submission refuses when the
fresh correlation has crossed the gate since.
"""

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import machine_lib as ml
from research_app import buffer_cmd, selection
from research_app.runtime import CliError


OFFICIAL_DESCRIPTION = (
    "Idea:\n- Higher quality firms should earn higher subsequent returns.\n\n"
    "Rationale for data used:\n- assets: total reported corporate assets.\n\n"
    "Rationale for operators used:\n- rank: compares firms cross-sectionally."
)


def _payload(name="A1", checks=None, self_corr=0.45, description=OFFICIAL_DESCRIPTION):
    return {"id": name, "status": "COMPLETE", "name": "x", "category": "QUALITY",
            "regular": {"code": "rank(close)", "description": description},
            "settings": {"region": "USA", "universe": "TOP3000", "simulationMode": "FULL"},
            "is": {"sharpe": 2.4, "fitness": 1.7, "turnover": 0.2},
            "oos": {"train": {"sharpe": 2.6, "fitness": 2.0, "turnover": 0.22,
                              "returns": 0.12}},
            "checks": checks if checks is not None else
            {"LOW_SHARPE": {"result": "PASS"}, "SELF_CORRELATION": {"result": "PASS"}},
            "_self_corr": self_corr}


def _checks(self_corr=0.45, result="PASS"):
    return {"alpha_id": "A1", "complete": True,
            "self_correlation": {"result": result, "value": self_corr},
            "checks": {"LOW_SHARPE": {"result": "PASS"},
                       "SELF_CORRELATION": {"result": result, "value": self_corr}}}


class BufferTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = temporary.name
        patcher = mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": self.tmp})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.selection_path = os.path.join(self.tmp, "selection.json")
        selection.freeze(self.selection_path, source="triage", rows=[{"alpha_id": "A1"}])
        self.shelf = os.path.join(self.tmp, "shelf.jsonl")
        patcher = mock.patch.object(buffer_cmd, "with_session",
                                    lambda args, work: work(object()))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _args(self, **overrides):
        base = dict(alpha_id="A1", buffer=self.shelf, note=None,
                    name="seed A1", category="FUNDAMENTAL", tags=["quality"],
                    description=OFFICIAL_DESCRIPTION,
                    seeds=str(Path(self.shelf).with_name("seeds.jsonl")),
                    registry=os.path.join(self.tmp, "ledger.jsonl"),
                    selection=self.selection_path, max_wait=None, yes=False, force=False)
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_add_records_the_evidence_it_was_judged_on(self):
        split = {"train": {"sharpe": 2.6, "fitness": 2.0, "turnover": 0.22,
                           "returns": 0.12}, "test": {"sharpe": 1.1}}
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=_payload()), \
                mock.patch.object(ml, "fetch_alpha_checks", return_value=_checks()), \
                mock.patch.object(ml, "extract_split", return_value=split), \
                mock.patch.object(buffer_cmd, "_book_size", return_value=53):
            data = buffer_cmd.cmd_buffer_add(self._args(), mock.MagicMock())
        entry = data["entry"]
        self.assertEqual(entry["self_correlation"], 0.45)
        self.assertEqual(entry["book_size"], 53)
        self.assertEqual(entry["train"]["fitness"], 2.0)
        self.assertEqual(data["seed"]["settings"], _payload()["settings"])
        self.assertEqual(data["seed"]["name"], "seed A1")
        self.assertEqual(data["seed"]["category"], "FUNDAMENTAL")
        self.assertEqual(buffer_cmd.read_seeds(data["seeds"])["A1"], data["seed"])

    def test_seed_write_failure_does_not_leave_a_shelved_entry(self):
        with mock.patch.object(buffer_cmd, "_write_seed", side_effect=OSError("disk full")), \
                mock.patch.object(buffer_cmd, "_append") as append:
            with self.assertRaises(CliError) as caught:
                self._shelve()
        append.assert_not_called()
        self.assertFalse(caught.exception.details["partial"])
        self.assertFalse(caught.exception.details["seed"]["complete"])
        self.assertFalse(Path(self.shelf).exists())

    def test_shelf_write_failure_reports_seed_progress_and_retry_is_idempotent(self):
        with mock.patch.object(buffer_cmd, "_append", side_effect=OSError("shelf unavailable")):
            with self.assertRaises(CliError) as caught:
                self._shelve()
        details = caught.exception.details
        self.assertTrue(details["partial"] and details["seed"]["complete"])
        self.assertFalse(details["shelf"]["complete"])
        self.assertIn("A1", buffer_cmd.read_seeds(self._args().seeds))
        self.assertFalse(Path(self.shelf).exists())
        result = self._shelve()
        self.assertTrue(result["written"])
        self.assertFalse(result["seed_written"])
        self.assertFalse(self._shelve()["written"])
        self.assertEqual(len(buffer_cmd.read_events(self.shelf)), 1)

    def test_damaged_seed_store_is_preserved_and_does_not_create_a_shelf(self):
        seed_path = Path(self._args().seeds)
        original = '{"alpha_id":"existing"}\n{"incomplete":\n'
        seed_path.write_text(original, encoding="utf-8")
        with self.assertRaises(CliError):
            self._shelve()
        self.assertEqual(seed_path.read_text(encoding="utf-8"), original)
        self.assertFalse(Path(self.shelf).exists())

    def test_seed_targets_cannot_overwrite_shelf_selection_or_registry(self):
        for target in (self.shelf, self.selection_path, self._args().registry):
            with self.subTest(target=target), mock.patch.object(ml, "fetch_alpha_metrics") as read:
                with self.assertRaises(CliError):
                    buffer_cmd.cmd_buffer_add(self._args(seeds=target), None)
                read.assert_not_called()
        selection.load(self.selection_path)

    def test_non_text_tags_are_rejected_before_any_read(self):
        for tags in ([None], ["quality", 4], [{"tag": "quality"}]):
            with self.subTest(tags=tags), mock.patch.object(ml, "fetch_alpha_metrics") as read:
                with self.assertRaises(CliError) as caught:
                    buffer_cmd.cmd_buffer_add(self._args(tags=tags), None)
                self.assertIn("tags", caught.exception.message)
                read.assert_not_called()

    def test_mismatched_platform_or_check_identity_never_writes_seed_or_shelf(self):
        for payload, checks in ((_payload(name="OTHER"), _checks()),
                                (_payload(), {**_checks(), "alpha_id": "OTHER"})):
            with self.subTest(payload=payload["id"], checks=checks["alpha_id"]), \
                    mock.patch.object(ml, "fetch_alpha_metrics", return_value=payload), \
                    mock.patch.object(ml, "fetch_alpha_checks", return_value=checks):
                with self.assertRaises(CliError):
                    buffer_cmd.cmd_buffer_add(self._args(), None)
        self.assertFalse(Path(self.shelf).exists())
        self.assertFalse(Path(self._args().seeds).exists())

    def test_a_failing_check_is_refused(self):
        failing = {"alpha_id": "A1", "complete": True,
                   "self_correlation": {"result": "PASS", "value": 0.4},
                   "checks": {"LOW_FITNESS": {"result": "FAIL", "value": 0.63}}}
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=_payload()), \
                mock.patch.object(ml, "fetch_alpha_checks", return_value=failing):
            with self.assertRaises(CliError) as caught:
                buffer_cmd.cmd_buffer_add(self._args(), mock.MagicMock())
        self.assertIn("LOW_FITNESS", caught.exception.message)
        self.assertFalse(os.path.exists(self.shelf))
        self.assertFalse(Path(self._args().seeds).exists())

    def test_old_template_is_refused_before_any_platform_read(self):
        with mock.patch.object(ml, "fetch_alpha_metrics") as read:
            with self.assertRaises(CliError) as caught:
                buffer_cmd.cmd_buffer_add(
                    self._args(description="IDEA:\n- " + "quality " * 20 +
                               "\nRATIONALE:\n- " + "rank " * 20), None)
        self.assertIn("official Power Pool template", caught.exception.message)
        read.assert_not_called()
        self.assertFalse(Path(self.shelf).exists())

    def test_platform_description_must_be_written_before_buffer_add(self):
        with mock.patch.object(ml, "fetch_alpha_metrics",
                               return_value=_payload(description="old description")):
            with self.assertRaises(CliError) as caught:
                buffer_cmd.cmd_buffer_add(self._args(), None)
        self.assertIn("alpha describe", caught.exception.hint)
        self.assertFalse(Path(self.shelf).exists())
        self.assertFalse(Path(self._args().seeds).exists())

    def test_quick_mode_is_refused_before_checks_or_shelving(self):
        payload = {**_payload(), "settings": {"simulationMode": "QUICK"}}
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=payload), \
                mock.patch.object(ml, "fetch_alpha_checks") as checks:
            with self.assertRaises(CliError) as caught:
                buffer_cmd.cmd_buffer_add(self._args(), None)
        self.assertEqual(caught.exception.code, "INVALID_INPUT")
        self.assertIn("Full Mode", caught.exception.hint)
        checks.assert_not_called()
        self.assertFalse(os.path.exists(self.shelf))

    def test_pending_is_not_a_failure(self):
        pending = {"alpha_id": "A1", "complete": True,
                   "self_correlation": {"result": "PENDING", "value": None},
                   "checks": {"SELF_CORRELATION": {"result": "PENDING"}}}
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=_payload()), \
                mock.patch.object(ml, "fetch_alpha_checks", return_value=pending), \
                mock.patch.object(buffer_cmd, "_book_size", return_value=53):
            data = buffer_cmd.cmd_buffer_add(self._args(), mock.MagicMock())
        self.assertEqual(data["entry"]["checks"]["failed"], [])
        self.assertIn("SELF_CORRELATION", data["entry"]["checks"]["pending"])
        self.assertFalse(data["entry"]["eligible"])

    def _shelve(self):
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=_payload()), \
                mock.patch.object(ml, "fetch_alpha_checks", return_value=_checks()), \
                mock.patch.object(buffer_cmd, "_book_size", return_value=53):
            return buffer_cmd.cmd_buffer_add(self._args(), mock.MagicMock())

    def test_list_reports_staleness_and_hides_closed_entries(self):
        self._shelve()
        with mock.patch.object(buffer_cmd, "_book_size", return_value=57):
            data = buffer_cmd.cmd_buffer_list(self._args(), mock.MagicMock())
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["rows"][0]["new_alphas_since"], 4)
        buffer_cmd.cmd_buffer_drop(self._args(), mock.MagicMock())
        data = buffer_cmd.cmd_buffer_list(self._args(), mock.MagicMock())
        self.assertEqual(data["count"], 0)

    def test_submit_refuses_when_the_fresh_correlation_crossed_the_gate(self):
        self._shelve()
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=_payload()), \
                mock.patch.object(ml, "fetch_alpha_checks",
                                  return_value=_checks(self_corr=0.78, result="FAIL")):
            with self.assertRaises(Exception):
                buffer_cmd.cmd_buffer_submit(self._args(yes=True), mock.MagicMock())

    def test_submit_marks_the_entry_consumed(self):
        self._shelve()
        payload = {**_payload(), "status": "ACTIVE", "dateSubmitted": "2026-01-02T00:00:00Z"}

        def pnl(_session, _alpha_id, **kwargs):
            kwargs["report"].update(complete=True)
            return {"2025-01-01": 1.0}

        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=payload), \
                mock.patch.object(ml, "fetch_daily_pnl", side_effect=pnl), \
                mock.patch.object(ml, "submit_alpha") as post:
            result = buffer_cmd.cmd_buffer_submit(self._args(yes=True), mock.MagicMock())
        post.assert_not_called()
        self.assertTrue(result["submitted"])
        data = buffer_cmd.cmd_buffer_list(self._args(), mock.MagicMock())
        self.assertEqual(data["count"], 0)


if __name__ == "__main__":
    unittest.main()


class SyncDriftTests(unittest.TestCase):
    """`ledger sync` must catch status drift, not just unknown ids."""

    def test_a_submitted_alpha_whose_row_says_complete_is_a_drift(self):
        import tempfile, os
        from research_app import ledger as cli_ledger

        tmp = tempfile.mkdtemp()
        registry = os.path.join(tmp, "r.jsonl")
        import alpha_registry
        alpha_registry.record_entry(
            alpha_registry.build_entry({"alpha_id": "DRIFT1", "expression": "rank(close)",
                                        "metrics": {}, "status": "COMPLETE"}),
            registry)
        platform = {"DRIFT1": {"id": "DRIFT1", "status": "ACTIVE"}}
        args = argparse.Namespace(platform_status="ACTIVE", status=None, limit=100,
                                  pages=1, order="-dateCreated", limit_record=10,
                                  dry_run=True, registry=registry)
        with mock.patch.object(cli_ledger, "with_session",
                               lambda a, work, **kw: work(object())), \
                mock.patch.object(ml, "list_alphas", return_value=list(platform.values())):
            data = cli_ledger.cmd_ledger_sync(args, mock.MagicMock())
        self.assertEqual(data["missing"], [])
        self.assertEqual(data["drifted"], ["DRIFT1"])


class StaleShelfTests(unittest.TestCase):
    """The recorded self-correlation is true only at shelving time."""

    def test_an_entry_shelved_before_a_new_alpha_is_marked_stale(self):
        import tempfile
        from research_app import buffer_cmd

        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "buffer.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "shelved", "alpha_id": "A1",
                                     "train": {"fitness": 2.0, "sharpe": 2.0},
                                     "turnover": 0.1, "self_correlation": 0.44,
                                     "book_size": 55, "at": "2026-11-01T00:00:00"}) + "\n")

        class Args:
            buffer = path
            registry = None
            limit = 20

        with mock.patch.object(buffer_cmd, "_book_size", return_value=61):
            data = buffer_cmd.cmd_buffer_list(Args(), None)
        row = data["rows"][0]
        self.assertTrue(row["stale"])
        self.assertEqual(row["new_alphas_since"], 6)
