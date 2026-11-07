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
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import machine_lib as ml
from research_app import buffer_cmd


def _payload(name="cand", checks=None, self_corr=0.45):
    return {"id": name, "status": "COMPLETE", "name": "x", "category": "QUALITY",
            "regular": {"code": "rank(close)"},
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
        self.tmp = tempfile.mkdtemp()
        self.shelf = os.path.join(self.tmp, "shelf.jsonl")
        patcher = mock.patch.object(buffer_cmd, "with_session",
                                    lambda args, work: work(object()))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _args(self, **overrides):
        base = dict(alpha_id="A1", buffer=self.shelf, note=None, registry=None,
                    max_wait=None, yes=False, force=False)
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

    def test_a_failing_check_is_refused(self):
        failing = {"alpha_id": "A1", "complete": True,
                   "self_correlation": {"result": "PASS", "value": 0.4},
                   "checks": {"LOW_SHARPE": {"result": "FAIL"}}}
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=_payload()), \
                mock.patch.object(ml, "fetch_alpha_checks", return_value=failing):
            with self.assertRaises(Exception):
                buffer_cmd.cmd_buffer_add(self._args(), mock.MagicMock())
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
                buffer_cmd.cmd_buffer_submit(self._args(), mock.MagicMock())

    def test_submit_marks_the_entry_consumed(self):
        self._shelve()
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=_payload()), \
                mock.patch.object(ml, "fetch_alpha_checks", return_value=_checks(0.44)), \
                mock.patch.object(ml, "submit_alpha",
                                  return_value={"submitted": True}):
            result = buffer_cmd.cmd_buffer_submit(self._args(yes=True), mock.MagicMock())
        self.assertTrue(result["submitted"])
        data = buffer_cmd.cmd_buffer_list(self._args(), mock.MagicMock())
        self.assertEqual(data["count"], 0)


if __name__ == "__main__":
    unittest.main()
