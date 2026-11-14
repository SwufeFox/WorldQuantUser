"""Offline account-capture and honest book/PnL evidence contracts."""

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import alpha_registry
import cache_store
import machine_lib as ml
from research_app import book_state, buffer_cmd, ledger


def alpha(alpha_id, status="ACTIVE"):
    return {"id": alpha_id, "status": status, "regular": {"code": "rank(close)"},
            "is": {"sharpe": 2.0, "fitness": 1.2}, "settings": {},
            "dateSubmitted": "2026-01-01T00:00:00Z" if status == "ACTIVE" else None}


class BookStateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": temporary.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.registry = str(self.root / "ledger.jsonl")
        patcher = mock.patch.object(ledger, "with_session", lambda args, work: work(object()))
        patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, **overrides):
        values = {"registry": self.registry, "status": None, "platform_status": "ACTIVE",
                  "limit": 100, "pages": 2, "order": "-dateCreated", "limit_record": 25,
                  "dry_run": False, "pnl": False, "hidden": False, "alpha_type_not": None}
        values.update(overrides)
        return argparse.Namespace(**values)

    def record(self, alpha_id, status="ACTIVE"):
        alpha_registry.record_entry(alpha_registry.build_entry(
            {"alpha_id": alpha_id, "status": status, "expression": "rank(close)",
             "metrics": {"sharpe": 2.0, "fitness": 1.2}}), self.registry)

    def test_same_count_different_members_changes_digest_and_shelf_staleness(self):
        self.record("MEMBER_A")
        first = book_state.describe_book(alpha_registry.load_entries(self.registry), registry_path=self.registry)
        self.record("MEMBER_A", "COMPLETE")
        self.record("MEMBER_B")
        second = book_state.describe_book(alpha_registry.load_entries(self.registry), registry_path=self.registry)
        self.assertEqual(first["local"]["member_count"], second["local"]["member_count"])
        self.assertNotEqual(first["local"]["member_digest"], second["local"]["member_digest"])
        self.assertFalse(second["local"]["platform_confirmed"])
        self.assertFalse(second["platform"]["available"])
        shelf = self.root / "shelf.jsonl"
        buffer_cmd._append(shelf, {"event": "shelved", "alpha_id": "CANDIDATE",
                                    "book": first, "book_size": 1, "eligible": True})
        result = buffer_cmd.cmd_buffer_list(argparse.Namespace(buffer=str(shelf), registry=self.registry), None)
        self.assertTrue(result["rows"][0]["stale"])
        self.assertFalse(result["rows"][0]["eligible"])
        self.assertEqual(result["rows"][0]["new_alphas_since"], 0)

    def test_legacy_and_unattributed_pnl_are_not_complete(self):
        entries = [{"alpha_id": "MEMBER_A", "status": "ACTIVE"}]
        pnl_dir = self.root / "pnl"
        cache_store.write_json_atomic(pnl_dir / "MEMBER_A.json", {"2025-01-01": 1.0})
        evidence = book_state.describe_book(entries, registry_path=self.registry)
        self.assertEqual(evidence["pnl"]["unknown_ids"], ["MEMBER_A"])
        self.assertFalse(evidence["pnl"]["complete"])
        import hashlib
        import json
        series = {"2025-01-01": 1.0}
        digest = hashlib.sha256(json.dumps(series, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode("utf-8")).hexdigest()
        cache_store.write_json_atomic(pnl_dir / ".meta" / "MEMBER_A.json",
                                      {"format": "worldquantuser.pnl-evidence", "version": 1,
                                       "alpha_id": "MEMBER_A", "data_sha256": digest,
                                       "complete": True, "as_of": "2026-01-01"})
        evidence = book_state.describe_book(entries, series={"MEMBER_A": {"2024-01-01": 3.0}},
                                            registry_path=self.registry)
        self.assertFalse(evidence["pnl"]["complete"])
        evidence = book_state.describe_book(entries, registry_path=self.registry, pnl_dir=pnl_dir)
        self.assertTrue(evidence["pnl"]["complete"])
        other = self.root / "other-pnl"
        evidence = book_state.describe_book(entries, series={"MEMBER_A": {"2024-01-01": 3.0}},
                                            registry_path=self.registry, pnl_dir=other)
        self.assertFalse(evidence["pnl"]["complete"])
        cache_store.write_json_atomic(pnl_dir / "MEMBER_A.json", {"2025-01-01": 99.0})
        evidence = book_state.describe_book(entries, registry_path=self.registry, pnl_dir=pnl_dir)
        self.assertFalse(evidence["pnl"]["complete"])
        self.assertEqual(evidence["pnl"]["incomplete_ids"], ["MEMBER_A"])
        self.assertEqual(evidence["pnl"]["coverage"]["MEMBER_A"]["error"], "pnl_metadata_mismatch")

    def test_effective_page_size_advances_without_skipping_ids(self):
        rows = [alpha("MEMBER_%03d" % index) for index in range(101)]
        with mock.patch.object(ml, "list_alphas", side_effect=[rows[:100], rows[100:]]) as listing:
            result = ledger.cmd_ledger_sync(self.args(limit=250, dry_run=True), None)
        self.assertEqual([call.kwargs["offset"] for call in listing.call_args_list], [0, 100])
        self.assertEqual([call.kwargs["limit"] for call in listing.call_args_list], [100, 100])
        self.assertEqual(result["platform"], 101)
        self.assertTrue(result["snapshot"]["complete"])
        self.assertEqual(len(result["snapshot"]["member_ids"]), 101)
        self.assertFalse(book_state.snapshot_path(self.registry).exists())

    def test_truncated_listing_does_not_infer_reverse_drift(self):
        self.record("OLD_MEMBER")
        with mock.patch.object(ml, "list_alphas", return_value=[alpha("NEW_MEMBER")]), \
                mock.patch.object(ml, "fetch_alpha_metrics", side_effect=lambda session, aid: alpha(aid)) as reads:
            result = ledger.cmd_ledger_sync(self.args(limit=1, pages=1), None)
        self.assertFalse(result["snapshot"]["complete"])
        self.assertEqual(result["absent_from_scope"], [])
        self.assertEqual(result["reverse_drift"], [])
        self.assertEqual([call.args[1] for call in reads.call_args_list], ["NEW_MEMBER"])
        old = next(row for row in alpha_registry.load_entries(self.registry) if row["alpha_id"] == "OLD_MEMBER")
        self.assertEqual(old["status"], "ACTIVE")
        self.assertFalse(result["membership"]["complete"])

    def test_complete_listing_reads_missing_member_before_reverse_update(self):
        self.record("OLD_MEMBER")
        with mock.patch.object(ml, "list_alphas", return_value=[]), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value=alpha("OLD_MEMBER", "COMPLETE")) as reads:
            result = ledger.cmd_ledger_sync(self.args(), None)
        reads.assert_called_once()
        self.assertEqual(result["reverse_drift"], ["OLD_MEMBER"])
        self.assertTrue(result["membership"]["complete"])
        self.assertEqual(alpha_registry.load_entries(self.registry)[0]["status"], "COMPLETE")
        evidence = book_state.describe_book(alpha_registry.load_entries(self.registry), registry_path=self.registry)
        self.assertTrue(evidence["platform"]["current_capture"])
        self.assertTrue(evidence["platform"]["matches_local_members"])
        self.assertIsNotNone(evidence["platform"]["as_of"])

    def test_missing_from_filtered_scope_but_still_active_is_preserved(self):
        self.record("HIDDEN_MEMBER")
        with mock.patch.object(ml, "list_alphas", return_value=[]), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value=alpha("HIDDEN_MEMBER")):
            result = ledger.cmd_ledger_sync(self.args(hidden=False), None)
        self.assertEqual(result["reverse_drift"], [])
        self.assertEqual(alpha_registry.load_entries(self.registry)[0]["status"], "ACTIVE")
        self.assertEqual(result["snapshot"]["scope"]["hidden"], False)
        self.assertEqual(result["snapshot"]["coverage"], "selected status/hidden/type scope only")
        evidence = book_state.describe_book(alpha_registry.load_entries(self.registry), registry_path=self.registry)
        self.assertFalse(evidence["platform"]["matches_local_members"])

    def test_record_cap_and_partial_pnl_cannot_claim_complete_reconciliation(self):
        rows = [alpha("MEMBER_A"), alpha("MEMBER_B")]

        def partial(_session, _alpha_id, **kwargs):
            kwargs["report"].update(complete=False, pages=1, error="next page unavailable")
            return {"2025-01-01": 1.0}

        with mock.patch.object(ml, "list_alphas", return_value=rows), \
                mock.patch.object(ml, "fetch_alpha_metrics", side_effect=lambda session, aid: alpha(aid)), \
                mock.patch.object(ml, "fetch_daily_pnl", side_effect=partial):
            result = ledger.cmd_ledger_sync(self.args(limit_record=1, pnl=True), None)
        self.assertTrue(result["snapshot"]["complete"])
        self.assertFalse(result["fully_reconciled"])
        self.assertFalse(result["membership"]["complete"])
        self.assertFalse(result["metadata"]["complete"])
        self.assertFalse(result["pnl"]["complete"])
        self.assertEqual(result["skipped_beyond_limit"], 1)
        self.assertEqual(result["pnl"]["skipped_beyond_limit"], 1)
        self.assertEqual(result["pnl"]["rows"][0]["error"], "next page unavailable")

    def test_duplicate_pagination_is_incomplete_even_with_terminal_page(self):
        with mock.patch.object(ml, "list_alphas", side_effect=[[alpha("MEMBER_A")],
                                                              [alpha("MEMBER_A")], []]):
            result = ledger.cmd_ledger_sync(self.args(limit=1, pages=3, dry_run=True), None)
        self.assertFalse(result["snapshot"]["complete"])
        self.assertTrue(result["snapshot"]["pages"][1]["duplicate_ids"])

    def test_submission_invalidates_platform_capture_without_erasing_history(self):
        snapshot = {"complete": True, "as_of": "2026-01-01", "scope": {"statuses": ["ACTIVE"]},
                    "member_ids": ["MEMBER_A"], "member_digest": book_state.member_digest(["MEMBER_A"])}
        book_state.persist_snapshot(snapshot, self.registry)
        book_state.invalidate_snapshot("MEMBER_B", self.registry)
        evidence = book_state.describe_book([], registry_path=self.registry)
        self.assertEqual(evidence["platform"]["as_of"], "2026-01-01")
        self.assertTrue(evidence["platform"]["complete"])
        self.assertFalse(evidence["platform"]["current_capture"])


if __name__ == "__main__":
    unittest.main()
