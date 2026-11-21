"""Offline regressions for the QUICK runner's durable HTTP workflow."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from unittest import TestCase, mock

import requests

import machine_lib
import simulation_capabilities
import simulation_slots
from tools.pipeline import batch_quick


ACCOUNT = {"state": "known", "fingerprint": "quick-runner-test-account"}
PROGRESS_URL = f"{machine_lib.BRAIN_API_URL}/simulations/op-test"
ALPHA_ID = "AlphaTest1"
ALPHA_URL = f"{machine_lib.BRAIN_API_URL}/alphas/{ALPHA_ID}"
ARM = {
    "label": "cap_rank",
    "expr": "rank(cap)",
    "settings": {"region": "USA", "universe": "TOP3000", "delay": 1},
}
REAL_MONOTONIC = batch_quick.time.monotonic
REAL_SLEEP = batch_quick.time.sleep


def response(code, payload=None, **headers):
    result = requests.Response()
    result.status_code = code
    result.headers.update(headers)
    result._content = json.dumps(payload or {}).encode()
    return result


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def time(self):
        return 100 + self.now

    def sleep(self, seconds):
        if seconds < 0:
            raise AssertionError(f"negative sleep: {seconds}")
        self.now += seconds


class QuickRunnerTests(TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.out = Path(directory.name) / "results.jsonl"
        self.clock = Clock()
        for patcher in (
            mock.patch.dict("os.environ", {"BRAIN_RESEARCH_DIR": directory.name}),
            mock.patch.object(simulation_capabilities, "account_provenance", return_value=ACCOUNT),
            mock.patch.object(batch_quick.time, "monotonic", self.clock.monotonic),
            mock.patch.object(batch_quick.time, "time", self.clock.time),
            mock.patch.object(batch_quick.time, "sleep", self.clock.sleep),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def runner(self, timeout=60, slots=1):
        runner = batch_quick.Runner(self.out, ACCOUNT, slots, timeout)
        self.addCleanup(runner.close)
        return runner

    def complete(self, url, **_kwargs):
        if url == PROGRESS_URL:
            return response(200, {"status": "COMPLETE", "alpha": ALPHA_ID})
        self.assertEqual(url, ALPHA_URL)
        return response(200, {"is": {"sharpe": 2.3, "fitness": 1.5, "checks": []}})

    def test_quick_and_full_posts_format_source_without_changing_request_identity(self):
        for mode in ("QUICK", "FULL"):
            with self.subTest(mode=mode):
                settings = dict(ARM["settings"])
                if mode == "FULL":
                    settings["simulationMode"] = mode
                arm = {"label": mode, "expr": "x = rank(cap); rank(x)", "settings": settings}
                raw = batch_quick._payload(arm)
                session = mock.Mock()
                session.post.return_value = response(201, Location=PROGRESS_URL)
                session.get.side_effect = self.complete
                with mock.patch.object(machine_lib, "load_session", return_value=session):
                    result = self.runner().run(arm)

                self.assertEqual(result["status"], "COMPLETE")
                self.assertEqual(result["expr"], "x = rank(cap); rank(x)")
                session.post.assert_called_once()
                wire = session.post.call_args.kwargs["json"]
                self.assertEqual(wire, {
                    "type": "REGULAR",
                    "settings": {**settings, "simulationMode": mode, "visualization": False},
                    "regular": "x = rank(cap);\nrank(x)",
                })
                self.assertEqual(batch_quick._simulation_key(wire), batch_quick._simulation_key(raw))

    def test_resumes_shared_legacy_raw_job_without_reposting(self):
        arm = {**ARM, "expr": "x = rank(cap); rank(x)"}
        old_payload = {"type": "REGULAR", "regular": arm["expr"],
                       "settings": {**ARM["settings"], "simulationMode": "QUICK", "visualization": False}}
        legacy_key = self.runner()._raw_key(arm["label"], old_payload)
        slot = simulation_slots.quick_acquire(ACCOUNT, 1, legacy_key)
        simulation_slots.quick_update(ACCOUNT, slot["id"], "accepted", progress_url=PROGRESS_URL)

        session = mock.Mock()
        session.get.side_effect = self.complete
        with mock.patch.object(machine_lib, "load_session", return_value=session):
            result = self.runner().run(arm)

        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["simulation_job_key"], legacy_key)
        self.assertEqual(result["alpha_id"], ALPHA_ID)
        session.post.assert_not_called()

    def test_post_refreshes_auth_and_429_remains_retryable_after_four_attempts(self):
        stale, current = mock.Mock(), mock.Mock()
        stale.post.return_value = response(401)
        attempt_times = []

        def post(_url, **_kwargs):
            attempt_times.append(self.clock.monotonic())
            if len(attempt_times) <= 6:
                return response(429, {"detail": "CONCURRENT_SIMULATION_LIMIT_EXCEEDED"},
                                **{"Retry-After": "2"})
            return response(201, Location="/simulations/op-test")

        current.post.side_effect = post
        current.get.side_effect = self.complete
        with mock.patch.object(machine_lib, "load_session", side_effect=(stale, current)) as login:
            result = self.runner().run(ARM)

        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["alpha_id"], ALPHA_ID)
        self.assertEqual(result["sharpe"], 2.3)
        self.assertEqual(len(attempt_times), 7)
        self.assertTrue(all(after - before >= 2 - 1e-9
                            for before, after in zip(attempt_times, attempt_times[1:])))
        self.assertTrue(login.call_args.kwargs.get("refresh"))
        stale.post.assert_called_once()
        stale.get.assert_not_called()
        for call in current.post.call_args_list:
            self.assertEqual(call.args[0], f"{machine_lib.BRAIN_API_URL}/simulations")
            self.assertEqual(call.kwargs["json"]["regular"], ARM["expr"])
            self.assertEqual(call.kwargs["json"]["settings"]["simulationMode"], "QUICK")
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 0)

    def test_timeout_resumes_accepted_location_without_reposting(self):
        first_session = mock.Mock()
        first_session.post.return_value = response(201, Location="/simulations/op-test")

        def unfinished(url, **_kwargs):
            self.assertEqual(url, PROGRESS_URL)
            state_dir = self.out.with_suffix(".jsonl.state")
            stored, = state_dir.glob("*.json")
            self.assertIn(PROGRESS_URL, stored.read_text())
            self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 1)
            return response(200, {"progress": 0.3})

        first_session.get.side_effect = unfinished
        with mock.patch.object(machine_lib, "load_session", return_value=first_session):
            pending = self.runner(timeout=2).run(ARM)

        first_session.post.assert_called_once()
        self.assertEqual(pending["status"], "PENDING")
        self.assertEqual(pending["progress_url"], PROGRESS_URL)
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 1)

        simulation_slots.defer_submission(ACCOUNT, 60)
        resumed_session = mock.Mock()
        resumed_session.get.side_effect = self.complete
        with mock.patch.object(machine_lib, "load_session", return_value=resumed_session):
            result = self.runner().run(ARM)

        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["alpha_id"], ALPHA_ID)
        self.assertEqual(result["fitness"], 1.5)
        resumed_session.post.assert_not_called()
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 0)

    def test_separate_quick_runner_defers_new_post_until_shared_retry_time(self):
        session = mock.Mock()
        session.post.return_value = response(429, **{"Retry-After": "5"})
        with mock.patch.object(machine_lib, "load_session", return_value=session):
            queued = self.runner(timeout=1).run(ARM)
        session.post.assert_called_once()
        self.assertEqual(queued["status"], "QUEUED")
        self.assertEqual(simulation_slots.submission_retry_at(ACCOUNT), 105)

        other = {**ARM, "label": "other", "expr": "rank(volume)"}
        resumed = mock.Mock()
        resumed.post.return_value = response(201, Location="/simulations/op-test")
        resumed.get.side_effect = self.complete
        with mock.patch.object(machine_lib, "load_session", return_value=resumed):
            deferred = self.runner(timeout=1).run(other)
            self.assertEqual(deferred["next_retry_at"], 105)
            resumed.post.assert_not_called()
            self.clock.sleep(5)
            self.assertEqual(self.runner().run(other)["alpha_id"], ALPHA_ID)
        resumed.post.assert_called_once()

    def test_daily_exhaustion_defers_posts_but_allows_recovery_from_shared_journal(self):
        first_session = mock.Mock()
        first_session.post.return_value = response(201, Location="/simulations/op-test",
                                                  **{"X-Ratelimit-Limit": "5000",
                                                     "X-Ratelimit-Remaining": "0",
                                                     "X-Ratelimit-Reset": "60"})
        with mock.patch.object(machine_lib, "load_session", return_value=first_session):
            first = self.runner()
            with mock.patch.object(first, "_checkpoint", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    first.run(ARM)
        first_session.post.assert_called_once()
        first_session.get.assert_not_called()
        self.assertTrue(simulation_slots.quota_snapshot(ACCOUNT)["blocked"])
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 1)

        resumed_session = mock.Mock()
        resumed_session.get.side_effect = self.complete
        other_arm = {**ARM, "label": "other", "expr": "rank(volume)"}
        with mock.patch.object(machine_lib, "load_session", return_value=resumed_session):
            runner = self.runner()
            queued = runner.run(other_arm)
            result = runner.run(ARM)

        self.assertEqual(queued["status"], "QUEUED")
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["alpha_id"], ALPHA_ID)
        resumed_session.post.assert_not_called()
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 0)

    def test_ambiguous_post_is_not_repeated(self):
        interrupted = mock.Mock()
        interrupted.post.side_effect = requests.exceptions.ReadTimeout()
        with mock.patch.object(machine_lib, "load_session", return_value=interrupted):
            result = self.runner().run(ARM)
        self.assertEqual(result["status"], "UNKNOWN_SUBMISSION")
        interrupted.post.assert_called_once()
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 1)

        resumed = mock.Mock()
        with mock.patch.object(machine_lib, "load_session", return_value=resumed):
            result = self.runner().run(ARM)
        self.assertEqual(result["status"], "UNKNOWN_SUBMISSION")
        resumed.post.assert_not_called()

    def test_terminal_operation_resumes_metrics_only(self):
        first = mock.Mock()
        first.post.return_value = response(201, Location="/simulations/op-test")

        def finish_then_defer_metrics(url, **_kwargs):
            if url == PROGRESS_URL:
                return self.complete(url)
            self.assertEqual(url, ALPHA_URL)
            return response(200, **{"Retry-After": "60"})

        first.get.side_effect = finish_then_defer_metrics
        with mock.patch.object(machine_lib, "load_session", return_value=first):
            pending = self.runner(timeout=2).run(ARM)
        self.assertEqual(pending["status"], "PENDING")
        self.assertEqual(pending["alpha_id"], ALPHA_ID)
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 0)

        self.clock.sleep(60)
        resumed = mock.Mock()
        resumed.get.return_value = self.complete(ALPHA_URL)
        with mock.patch.object(machine_lib, "load_session", return_value=resumed):
            result = self.runner().run(ARM)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["sharpe"], 2.3)
        resumed.post.assert_not_called()
        resumed.get.assert_called_once()
        self.assertEqual(resumed.get.call_args.args[0], ALPHA_URL)

    def test_poll_adapts_and_refreshes_auth_for_progress_and_metrics(self):
        first, second, third = mock.Mock(), mock.Mock(), mock.Mock()
        first.post.return_value = response(201, Location="/simulations/op-test")
        first.get.return_value = response(401)
        poll_times = []

        def poll_and_reject_metrics(url, **_kwargs):
            if url == ALPHA_URL:
                self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 0)
                return response(401)
            self.assertEqual(url, PROGRESS_URL)
            poll_times.append(self.clock.monotonic())
            if len(poll_times) <= 8:
                return response(200, {"progress": 0.25})
            return self.complete(url)

        second.get.side_effect = poll_and_reject_metrics
        third.get.side_effect = self.complete
        with mock.patch.object(machine_lib, "load_session", side_effect=(first, second, third)) as login:
            result = self.runner().run(ARM)

        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["sharpe"], 2.3)
        self.assertEqual(len(poll_times), 9)
        gaps = [after - before for before, after in zip(poll_times, poll_times[1:])]
        self.assertAlmostEqual(max(gaps), 4)
        self.assertLess(min(gaps), max(gaps))
        self.assertEqual(sum(bool(call.kwargs.get("refresh")) for call in login.call_args_list), 2)
        first.post.assert_called_once()
        second.post.assert_not_called()
        third.post.assert_not_called()

    def test_workers_do_not_share_http_sessions(self):
        posts_in_flight = Barrier(2)
        sessions = [mock.Mock(), mock.Mock()]

        def post(url, **kwargs):
            self.assertEqual(url, f"{machine_lib.BRAIN_API_URL}/simulations")
            expr = kwargs["json"]["regular"]
            operation = "cap" if expr == ARM["expr"] else "volume"
            posts_in_flight.wait(timeout=5)
            return response(201, Location=f"/simulations/{operation}")

        def get(url, **_kwargs):
            if "/simulations/" in url:
                return response(200, {"status": "COMPLETE", "alpha": url.rsplit("/", 1)[-1]})
            return response(200, {"is": {"sharpe": 2.3}})

        for session in sessions:
            session.post.side_effect = post
            session.get.side_effect = get
        work = [ARM, {**ARM, "label": "volume_rank", "expr": "rank(volume)"}]
        with mock.patch.object(batch_quick.time, "monotonic", REAL_MONOTONIC), \
                mock.patch.object(batch_quick.time, "sleep", REAL_SLEEP), \
                mock.patch.object(machine_lib, "load_session", side_effect=sessions) as login, \
                ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(self.runner(slots=2).run, work))

        self.assertEqual([result["alpha_id"] for result in results], ["cap", "volume"])
        self.assertEqual(login.call_count, 2)
        for session in sessions:
            session.post.assert_called_once()
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 2)["active"], 0)

    def test_worker_exception_keeps_label_and_expression_in_ledger(self):
        inputs = self.out.parent / "arms.json"
        inputs.write_text(json.dumps([ARM]))
        with mock.patch.object(machine_lib, "load_session", return_value=mock.MagicMock()), \
                mock.patch.object(batch_quick.Runner, "run", side_effect=RuntimeError("broken worker")), \
                mock.patch.object(sys, "argv", ["batch_quick.py", str(inputs), "--out", str(self.out)]):
            self.assertEqual(batch_quick.main(), 0)

        result = json.loads(self.out.read_text())
        self.assertEqual(result["status"], "RUNNER_ERROR")
        self.assertEqual(result["label"], ARM["label"])
        self.assertEqual(result["expr"], ARM["expr"])
