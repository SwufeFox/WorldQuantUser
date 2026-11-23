"""Offline account-capture and honest book/PnL evidence contracts."""

import argparse
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import alpha_registry
import cache_store
import machine_lib as ml
from research_app import alphas, book_state, buffer_cmd, ledger


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
        patcher = mock.patch.object(ledger, "with_session", lambda args, work, **kw: work(object()))
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

    def test_income_readiness_counts_full_live_book_and_eastern_submission_days(self):
        def active(alpha_id, stamp, region="USA"):
            row = alpha(alpha_id)
            row.update(dateSubmitted=stamp, settings={"region": region, "delay": 1})
            return row

        picked = active("SAME_DAY", "2026-10-01T15:30:00+00:00")
        first = [active(f"OLDER_{index}", "2026-09-25T12:00:00+00:00") for index in range(97)]
        first += [active("BEFORE_QUARTER", "2026-10-01T03:30:00+00:00"),
                  active("FIRST_DAY", "2026-10-01T04:10:00+00:00", "EUR"), picked]
        visible_next = [active("MISSING", None, "EUR")]
        hidden = [picked, active("SECOND_DAY", "2026-10-02T21:00:00+00:00", "EUR"),
                  active("UNSCOPED", "2026-12-01T12:00:00+00:00", None)]
        as_of = datetime.fromisoformat("2026-11-23T12:00:00+00:00")
        with mock.patch.object(alphas, "with_session",
                               lambda args, read: read(object(), as_of=as_of)), \
                mock.patch.object(ml, "list_alphas", side_effect=[first, visible_next, hidden]) as listing:
            result = alphas.cmd_alpha_book(self.args(income=True, limit=1), None)

        self.assertEqual([(c.kwargs["hidden"], c.kwargs["offset"]) for c in listing.call_args_list],
                         [(False, 0), (False, 100), (True, 0)])
        self.assertTrue(all(c.kwargs["status"] == "ACTIVE" and c.kwargs["alpha_type_not"] is None
                            for c in listing.call_args_list))
        self.assertEqual(result["active_alphas"], 103)
        quarterly = result["quarterly"]
        self.assertEqual(quarterly["quarter_start"], "2026-10-01")
        self.assertEqual(quarterly["observed_active_submission_days"], 2)
        self.assertEqual(quarterly["observed_submission_day_gap"], 18)
        self.assertEqual(quarterly["unverified_active_submission_dates"], 2)
        osmosis = result["osmosis"]
        self.assertEqual(osmosis["unscoped_active_alphas"], 1)
        self.assertEqual(osmosis["observed_active_scope_gap"], 2)
        self.assertEqual(osmosis["scopes"], [
            {"region": "EUR", "delay": 1, "active_alphas": 3, "active_count_gap": 7},
            {"region": "USA", "delay": 1, "active_alphas": 99, "active_count_gap": 0},
        ])

    def test_truncated_listing_does_not_infer_reverse_drift(self):
        self.record("OLD_MEMBER")
        with mock.patch.object(ml, "list_alphas", return_value=[alpha("NEW_MEMBER")]), \
                mock.patch.object(ml, "fetch_alpha_metrics", side_effect=lambda _session, aid, **kw: alpha(aid)) as reads:
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
                mock.patch.object(ml, "fetch_alpha_metrics", side_effect=lambda _session, aid, **kw: alpha(aid)), \
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

    def test_sync_batches_readbacks_and_reports_one_refusal_without_losing_other_rows(self):
        self.record("MEMBER_A")
        rows = [alpha("MEMBER_A"), alpha("MEMBER_B"), alpha("MEMBER_C")]
        readbacks = {row["id"]: row for row in rows}
        readbacks["MEMBER_A"] = {**rows[0], "is": {}}
        with mock.patch.object(ml, "list_alphas", return_value=rows), \
                mock.patch.object(ml, "fetch_alpha_metrics", side_effect=lambda _session, aid, **kw: readbacks[aid]), \
                mock.patch.object(alpha_registry, "load_entries", wraps=alpha_registry.load_entries) as loads:
            result = ledger.cmd_ledger_sync(self.args(), None)
        self.assertEqual(loads.call_count, 2)  # initial listing comparison and locked write batch
        by_id = {row["alpha_id"]: row for row in result["rows"]}
        self.assertFalse(by_id["MEMBER_A"]["complete"])
        self.assertIn("stored row has metrics", by_id["MEMBER_A"]["error"])
        self.assertTrue(by_id["MEMBER_B"]["written"])
        self.assertTrue(by_id["MEMBER_C"]["written"])
        self.assertEqual(result["recorded"], 2)

    def test_sync_retains_activation_written_while_the_readback_is_in_flight(self):
        self.record("MEMBER_A", "COMPLETE")
        captured = alpha("MEMBER_A", "COMPLETE")

        def readback(_session, _alpha_id, **kw):
            self.record("MEMBER_A", "ACTIVE")
            return captured

        with mock.patch.object(ml, "list_alphas", return_value=[captured]), \
                mock.patch.object(ml, "fetch_alpha_metrics", side_effect=readback):
            result = ledger.cmd_ledger_sync(self.args(platform_status="ALL"), None)
        self.assertEqual(result["recorded"], 0)
        self.assertIn("local status changed", result["rows"][0]["error"])
        self.assertTrue(result["snapshot"]["invalidated"])
        self.assertFalse(result["membership"]["complete"])
        self.assertEqual(alpha_registry.load_entries(self.registry)[0]["status"], "ACTIVE")

    def test_sync_persists_finished_batches_while_more_network_reads_are_pending(self):
        rows = [alpha("MEMBER_" + key) for key in "ABCD"]
        committed_before_next_read = []

        def readback(_session, alpha_id, **kw):
            if alpha_id in {"MEMBER_C", "MEMBER_D"}:
                committed_before_next_read.append(
                    [row["key"] for row in alpha_registry.load_entries(self.registry)])
            if alpha_id == "MEMBER_D":
                raise KeyboardInterrupt
            return alpha(alpha_id)

        with mock.patch.object(ledger, "_SYNC_WRITE_BATCH_SIZE", 2), \
                mock.patch.object(ml, "list_alphas", return_value=rows), \
                mock.patch.object(ml, "fetch_alpha_metrics", side_effect=readback):
            with self.assertRaises(KeyboardInterrupt):
                ledger.cmd_ledger_sync(self.args(), None)
        self.assertEqual(committed_before_next_read, [["MEMBER_A", "MEMBER_B"]] * 2)
        self.assertEqual([row["key"] for row in alpha_registry.load_entries(self.registry)],
                         ["MEMBER_A", "MEMBER_B", "MEMBER_C"])
        self.assertFalse(book_state.snapshot_path(self.registry).exists())

    def test_sync_keeps_finished_rows_when_readiness_wait_exhausts_the_budget(self):
        rows = [alpha("MEMBER_" + key) for key in "ABC"]
        clock = mock.Mock()
        clock.monotonic.return_value = 100.0

        def readback(_session, alpha_id, *, deadline):
            self.assertEqual(deadline, 105.0)
            if alpha_id == "MEMBER_B":
                clock.monotonic.return_value = 105.0
                return {"_deferred": True, "next_poll_after_seconds": 15.0}
            return alpha(alpha_id)

        with mock.patch.object(ledger, "time", clock), \
                mock.patch.object(ml, "list_alphas", return_value=rows) as listing, \
                mock.patch.object(ml, "fetch_alpha_metrics", side_effect=readback) as metrics, \
                mock.patch.object(ml, "fetch_daily_pnl") as pnl:
            result = ledger.cmd_ledger_sync(self.args(max_wait_seconds=5.0, pnl=True), None)
        self.assertEqual(listing.call_args.kwargs["deadline"], 105.0)
        self.assertEqual([call.args[1] for call in metrics.call_args_list], ["MEMBER_A", "MEMBER_B"])
        pnl.assert_not_called()
        self.assertTrue(result["wait_budget_exhausted"])
        self.assertFalse(result["fully_reconciled"])
        self.assertEqual(result["metadata"]["unrepaired_ids"], ["MEMBER_B", "MEMBER_C"])
        self.assertEqual(result["rows"][1]["retry_after_seconds"], 15.0)
        self.assertEqual([row["key"] for row in alpha_registry.load_entries(self.registry)], ["MEMBER_A"])

    def test_submission_invalidates_platform_capture_without_erasing_history(self):
        snapshot = {"complete": True, "as_of": "2026-01-01", "scope": {"statuses": ["ACTIVE"]},
                    "member_ids": ["MEMBER_A"], "member_digest": book_state.member_digest(["MEMBER_A"])}
        book_state.persist_snapshot(snapshot, self.registry)
        book_state.invalidate_snapshot("MEMBER_B", self.registry)
        evidence = book_state.describe_book([], registry_path=self.registry)
        self.assertEqual(evidence["platform"]["as_of"], "2026-01-01")
        self.assertTrue(evidence["platform"]["complete"])
        self.assertFalse(evidence["platform"]["current_capture"])

    def test_submission_ledger_refresh_preserves_manually_selected_verdict(self):
        self.record("MEMBER_A", "COMPLETE")
        previous = alpha_registry.load_entries(self.registry)[0]
        alpha_registry.record_entry(alpha_registry.relabel(previous, verdict="watch", note="reviewed"),
                                    self.registry)

        with mock.patch.object(buffer_cmd, "consume_submission"), mock.patch.object(ml, "fetch_daily_pnl"):
            report = alphas._reconcile_submission(object(), self.args(alpha_id="MEMBER_A"),
                                                  alpha("MEMBER_A"), "selection")
        self.assertTrue(report["ledger"]["complete"])
        stored = alpha_registry.load_entries(self.registry)[0]
        self.assertEqual(stored["status"], "ACTIVE")
        self.assertEqual(stored["verdict"], "watch")
        self.assertEqual(stored["note"], "reviewed")


if __name__ == "__main__":
    unittest.main()
