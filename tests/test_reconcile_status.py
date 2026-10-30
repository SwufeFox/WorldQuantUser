#!/usr/bin/env python3
"""Tests for ledger status reconciliation.

The rule being pinned: a row is promoted when the platform lists it as ACTIVE, and a
row that the ledger calls ACTIVE but the platform does not list is *marked*, never
silently downgraded -- "it left the book" and "it was never in it" are different
facts and only one is knowable from here.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import reconcile_status  # noqa: E402


class Reconcile(unittest.TestCase):
    def test_row_promoted_when_platform_lists_it(self):
        rows = [{"alpha_id": "A", "status": "COMPLETE"}]
        promoted, demoted = reconcile_status.reconcile(rows, {"A": {}}, "T")
        self.assertEqual(promoted, ["A"])
        self.assertEqual(demoted, [])
        self.assertEqual(rows[0]["status"], "ACTIVE")
        self.assertEqual(rows[0]["status_source"], "reconciled:api")

    def test_submission_date_is_carried_over(self):
        rows = [{"alpha_id": "A", "status": "UNSUBMITTED"}]
        reconcile_status.reconcile(rows, {"A": {"dateSubmitted": "2026-10-01"}}, "T")
        self.assertEqual(rows[0]["date_submitted"], "2026-10-01")

    def test_row_already_active_is_left_alone(self):
        rows = [{"alpha_id": "A", "status": "ACTIVE"}]
        promoted, demoted = reconcile_status.reconcile(rows, {"A": {}}, "T")
        self.assertEqual((promoted, demoted), ([], []))
        self.assertNotIn("status_source", rows[0])

    def test_missing_from_platform_is_marked_not_downgraded(self):
        rows = [{"alpha_id": "B", "status": "ACTIVE"}]
        promoted, demoted = reconcile_status.reconcile(rows, {}, "T")
        self.assertEqual(demoted, ["B"])
        self.assertEqual(rows[0]["status"], "ACTIVE")
        self.assertTrue(rows[0]["not_in_book"])

    def test_rows_without_an_alpha_id_are_ignored(self):
        rows = [{"alpha_id": None, "status": "QUEUED"}]
        promote, demote = reconcile_status.reconcile(rows, {"A": {}}, "T")
        self.assertEqual((promote, demote), ([], []))
        self.assertEqual(rows[0]["status"], "QUEUED")

    def test_one_alpha_with_several_rows_is_promoted_in_each(self):
        # The ledger keeps several rows per alpha; rows therefore outnumber alphas.
        rows = [{"alpha_id": "A", "status": "COMPLETE"},
                {"alpha_id": "A", "status": "UNSUBMITTED"}]
        promoted, _ = reconcile_status.reconcile(rows, {"A": {}}, "T")
        self.assertEqual(promoted, ["A", "A"])
        self.assertEqual([r["status"] for r in rows], ["ACTIVE", "ACTIVE"])


if __name__ == "__main__":
    unittest.main()