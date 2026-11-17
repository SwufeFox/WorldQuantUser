"""Offline BRAIN protocol regressions: quota pauses and recordset readiness."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

import machine_lib as ml
import simulation_capabilities
import simulation_slots


ACCOUNT = {"state": "known", "fingerprint": "quota-fixture-account", "sources": ["user_id"]}


def response(status=201, headers=None, payload=None):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers or {})
    result._content = json.dumps(payload or {}).encode("utf-8")
    return result


def quota_headers(remaining, reset="60"):
    headers = {"Location": "/simulations/fixture", "x-ratelimit-limit": "100",
               "X-Ratelimit-Remaining": str(remaining)}
    if reset is not None:
        headers["X-RATELIMIT-RESET"] = reset
    return headers


class SimulationQuotaTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for patcher in (
            mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": self.directory.name}),
            mock.patch.object(simulation_capabilities, "account_provenance", return_value=ACCOUNT),
            mock.patch.object(simulation_capabilities, "cached", return_value={
                "state": "unknown", "account": ACCOUNT,
                "concurrency": {"state": "unknown", "limit": None}}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_batch(self, session, **options):
        report = {}
        arguments = {"concurrency": 1, "max_wait_seconds": 10, "scheduling_report": report}
        arguments.update(options)
        with mock.patch.object(ml, "poll_simulation", return_value={"status": "COMPLETE", "alpha": "A1"}), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value={"is": {}}):
            records = ml.simulate_alphas(session, ["rank(cap)", "rank(volume)", "rank(sales)"], **arguments)
        return records, report

    def test_success_exhausts_quota_without_losing_accepted_job_or_posting_later_waves(self):
        session = mock.Mock()
        session.post.return_value = response(headers=quota_headers(0))
        records, report = self.run_batch(session)
        self.assertEqual(session.post.call_count, 1)
        self.assertEqual([row["status"] for row in records], ["COMPLETE", "QUEUED", "QUEUED"])
        self.assertEqual(records[0]["alpha_id"], "A1")
        self.assertFalse(records[1]["submission_attempted"])
        self.assertEqual(report["quota"]["remaining"], 0)
        self.assertTrue(report["quota"]["blocked"])
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 0)

    def test_resume_polls_accepted_job_at_zero_then_only_posts_queued_work_after_reset(self):
        with mock.patch.object(simulation_slots.time, "time", return_value=100):
            simulation_slots.observe_quota(ACCOUNT, quota_headers(0))
        records = [
            {"expression": "rank(cap)", "decay": 0, "status": "PENDING", "submission_attempted": True,
             "progress_url": "https://api.worldquantbrain.com/simulations/accepted"},
            {"expression": "rank(volume)", "decay": 0, "status": "QUEUED"},
            {"expression": "rank(sales)", "decay": 0, "status": "QUEUED"},
        ]
        blocked = mock.Mock()
        with mock.patch.object(simulation_slots.time, "time", return_value=110):
            saved, _ = self.run_batch(blocked, initial_records=records)
        blocked.post.assert_not_called()
        self.assertEqual(saved[0]["alpha_id"], "A1")
        renewed = mock.Mock()
        renewed.post.return_value = response(headers=quota_headers(5))
        with mock.patch.object(simulation_slots.time, "time", return_value=161):
            expired = simulation_slots.quota_snapshot(ACCOUNT)
            self.assertEqual(expired["state"], "reset_due")
            self.assertEqual(expired["remaining"], 0)
            self.assertFalse(expired["blocked"])
            completed, report = self.run_batch(renewed, initial_records=saved)
        self.assertEqual(renewed.post.call_count, 2)
        self.assertEqual([row["status"] for row in completed], ["COMPLETE"] * 3)
        self.assertEqual(report["quota"]["remaining"], 5)

    def test_daily_429_stops_immediately_without_sleep_or_later_batch_posts(self):
        session = mock.Mock()
        session.post.return_value = response(429, {**quota_headers(0), "Retry-After": "3600"})
        with mock.patch.object(ml, "sleep") as wait:
            records, report = self.run_batch(session)
        self.assertEqual(session.post.call_count, 1)
        wait.assert_not_called()
        self.assertTrue(all(row["status"] == "QUEUED" for row in records))
        self.assertFalse(records[0]["submission_attempted"])
        self.assertEqual(report["quota"]["remaining"], 0)
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 0)

    def test_transient_throttle_honors_backoff_and_preserves_positive_daily_remaining(self):
        session = mock.Mock()
        session.post.side_effect = [response(429, {**quota_headers(5), "Retry-After": "2"}),
                                    response(headers=quota_headers(4))]
        with mock.patch.object(ml, "sleep") as wait:
            progress = ml.submit_single_simulation(session, {"regular": "rank(cap)"})
        wait.assert_called_once_with(2)
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(progress, "https://api.worldquantbrain.com/simulations/fixture")
        self.assertEqual(simulation_slots.quota_snapshot(ACCOUNT)["remaining"], 4)

    def test_throttle_beyond_wait_budget_does_not_skip_into_later_waves(self):
        session = mock.Mock()
        session.post.return_value = response(429, {**quota_headers(5), "Retry-After": "3600"})
        with mock.patch.object(ml, "sleep") as wait:
            records, _ = self.run_batch(session)
        self.assertEqual(session.post.call_count, 1)
        self.assertTrue(all(row["status"] == "QUEUED" for row in records))
        wait.assert_not_called()

    def test_missing_reset_stops_current_run_without_permanently_blocking_later_runs(self):
        session = mock.Mock()
        session.post.return_value = response(headers=quota_headers(0, reset=None))
        saved, report = self.run_batch(session)
        self.assertEqual(session.post.call_count, 1)
        self.assertTrue(report["quota"]["blocked"])
        self.assertIsNone(report["quota"]["reset_at"])
        later = mock.Mock()
        later.post.return_value = response(headers=quota_headers(5))
        completed, _ = self.run_batch(later, initial_records=saved)
        self.assertEqual(later.post.call_count, 2)
        self.assertTrue(all(row["status"] == "COMPLETE" for row in completed))

    def test_missing_or_invalid_headers_never_invent_zero_or_nonfinite_output(self):
        for headers in ({}, {"X-Ratelimit-Limit": "100"},
                        {"X-Ratelimit-Remaining": "1.5", "X-Ratelimit-Reset": "nan"},
                        {"X-Ratelimit-Limit": "5", "X-Ratelimit-Remaining": "8", "X-Ratelimit-Reset": "inf"}):
            with self.subTest(headers=headers):
                session = mock.Mock()
                session.post.return_value = response(headers={"Location": "/simulations/fixture", **headers})
                records, report = self.run_batch(session)
                self.assertEqual(session.post.call_count, 3)
                self.assertTrue(all(row["status"] == "COMPLETE" for row in records))
                self.assertIsNone(report["quota"]["remaining"])
                self.assertFalse(report["quota"]["blocked"])
                json.dumps(report, allow_nan=False)

    def test_quota_is_scoped_to_account_and_unrelated_headers_are_not_saved(self):
        session = mock.Mock()
        session.post.return_value = response(headers={**quota_headers(0), "Set-Cookie": "unrelated-secret"})
        ml.submit_single_simulation(session, {"regular": "rank(cap)"})
        blocked = mock.Mock()
        with self.assertRaises(ml.RateLimited):
            ml.submit_single_simulation(blocked, {"regular": "rank(volume)"})
        blocked.post.assert_not_called()
        other = {**ACCOUNT, "fingerprint": "another-fixture-account"}
        with mock.patch.object(simulation_capabilities, "account_provenance", return_value=other):
            session.post.return_value = response(headers={"Location": "/simulations/other"})
            self.assertTrue(ml.submit_single_simulation(session, {"regular": "rank(volume)"}).endswith("/other"))
        journal, _ = simulation_slots._paths(ACCOUNT["fingerprint"])
        self.assertNotIn("unrelated-secret", Path(journal).read_text(encoding="utf-8"))

    def test_quota_storage_failure_does_not_discard_accepted_progress_identity(self):
        session = mock.Mock()
        session.post.return_value = response(headers=quota_headers(0))
        events = []
        with mock.patch.object(simulation_slots.cache_store, "write_json_atomic", side_effect=OSError("disk")):
            progress = ml.submit_single_simulation(session, {"regular": "rank(cap)"},
                on_event=lambda kind, payload: events.append((kind, payload)))
        self.assertEqual(progress, "https://api.worldquantbrain.com/simulations/fixture")
        self.assertEqual(events[0][1]["remaining"], 0)
        self.assertEqual(events[0][1]["storage_error"], "OSError")


    def test_documented_terminal_states_release_slots_and_classify_outcomes(self):
        from research_app import experiments

        for status in ("TIMEOUT", "FAIL", "ERROR", "CANCELLED", "WARNING"):
            with self.subTest(status=status):
                session = mock.Mock()
                session.post.return_value = response(headers={"Location": "/simulations/fixture"})
                with mock.patch.object(ml, "poll_simulation", return_value={"status": status}):
                    records = ml.simulate_alphas(session, ["rank(cap)", "rank(volume)"], concurrency=1)
                self.assertEqual(session.post.call_count, 2)
                self.assertEqual([row["status"] for row in records], [status, status])
                self.assertTrue(all("progress_url" not in row for row in records))
                self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 0)
                summary = experiments.summarize(records)
                self.assertEqual(summary["failed"], 0 if status == "WARNING" else 2)
                self.assertEqual(summary["completed"], 2 if status == "WARNING" else 0)


class RecordsetReadinessTests(unittest.TestCase):
    def test_retry_after_overrides_nonempty_body_and_only_ready_values_are_cached(self):
        session = mock.Mock()
        session.get.side_effect = [
            response(200, {"Retry-After": "2"}, {"records": [["2020-01-01", 999]], "count": 1}),
            response(200, payload={"records": [["2020-01-01", 2]], "count": 1}),
        ]
        report = {}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(ml.time, "sleep") as wait:
            series = ml.fetch_daily_pnl(session, "A", cache_dir=directory, report=report)
            self.assertEqual(series, {"2020-01-01": 2.0})
            self.assertEqual(json.loads((Path(directory) / "A.json").read_text()), series)
        wait.assert_called_once_with(2)
        self.assertEqual(session.get.call_count, 2)
        self.assertTrue(report["complete"])
        self.assertEqual(report["pages"], 1)

    def test_long_delay_keeps_existing_cache_and_reports_incomplete_evidence(self):
        for status in (200, 429):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "A.json"
                original = b'{"2020-01-01": 7}'
                path.write_bytes(original)
                session = mock.Mock()
                session.get.return_value = response(status, {"Retry-After": "7200"}, {"records": []})
                report = {}
                with mock.patch.object(ml.time, "sleep") as wait, self.assertRaises(ml.RateLimited):
                    ml.fetch_daily_pnl(session, "A", cache_dir=directory, max_age_hours=0,
                                       max_wait_seconds=1, report=report)
                wait.assert_not_called()
                self.assertEqual(session.get.call_count, 1)
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse(report["complete"])
                self.assertEqual(report["retry_after_seconds"], 7200)
                self.assertEqual(report["error"], "RATE_LIMITED")

    def test_attempt_limit_never_turns_a_pending_body_into_complete_data(self):
        session = mock.Mock()
        session.get.return_value = response(200, {"Retry-After": "1"}, {"records": [["2020-01-01", 999]]})
        report = {}
        with mock.patch.object(ml.time, "sleep") as wait, self.assertRaises(ml.RateLimited):
            ml.fetch_daily_pnl(session, "A", attempts=1, report=report)
        wait.assert_not_called()
        self.assertFalse(report["complete"])
        self.assertEqual(report["pages"], 0)


if __name__ == "__main__":
    unittest.main()
