"""Offline live-check batch contracts; no BRAIN requests or authentication."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

import machine_lib as ml
from research_app import alphas
from research_app import app
from research_app.runtime import CliError


def checked(alpha_id, verdict="PASS", **extra):
    check = {"result": verdict, "value": 0.31, "limit": 0.7}
    return {"alpha_id": alpha_id, "complete": True, "payload_received": True,
            "checks": {"SELF_CORRELATION": check, **extra}, "self_correlation": check}


def arguments(ids=("A",), workers=1, max_wait=0):
    return argparse.Namespace(alpha_ids=list(ids), workers=workers, max_wait=max_wait,
                              no_cache=False, relogin=False)


class BatchCheckTests(unittest.TestCase):
    def setUp(self):
        self.session = requests.Session()
        self.addCleanup(self.session.close)
        self.login = mock.patch.object(ml, "load_session", return_value=self.session).start()
        self.log_view = mock.patch.object(alphas.research_paths, "log_view").start()
        self.addCleanup(mock.patch.stopall)

    def test_single_and_multiple_use_same_complete_shape_in_unique_requested_order(self):
        shapes = []
        for ids in (("B",), ("B", "A", "B", "C", "A")):
            with self.subTest(ids=ids), mock.patch.object(ml, "fetch_alpha_checks",
                                                        side_effect=lambda s, alpha_id, **kw: checked(alpha_id)) as fetch:
                self.login.reset_mock()
                data = alphas.cmd_alpha_check(arguments(ids), None)
                unique = list(dict.fromkeys(ids))
                self.assertEqual([row["alpha_id"] for row in data["results"]], unique)
                self.assertEqual([call.args[1] for call in fetch.call_args_list], unique)
                self.assertTrue(all(call.args[0] is self.session for call in fetch.call_args_list))
                self.assertEqual(data["count"], len(unique))
                self.assertTrue(data["complete"])
                self.assertEqual(data["summary"], {"complete": len(unique), "pending": 0,
                                                  "error": 0, "passed": len(unique), "failed": 0, "warning": 0})
                self.assertTrue(all(row["error"] is None for row in data["results"]))
                self.login.assert_called_once_with(use_cache=True, refresh=False)
                self.log_view.assert_called_with("alpha check", "validation", unique)
                shapes.append((set(data), set(data["results"][0])))
        self.assertEqual(shapes[0], shapes[1])

    def test_pending_warning_missing_and_failed_checks_are_not_hidden(self):
        payloads = {
            "pass": checked("pass"),
            "pending": checked("pending", "PENDING", LOW_SHARPE={"result": "PASS", "value": 2.3}),
            "warning": checked("warning", "WARNING"),
            "missing": {"complete": False, "payload_received": True, "checks": None},
            "no-result": checked("no-result", None),
            "unknown": checked("unknown", "RUNNING"),
            "fail": checked("fail", "FAIL"),
        }
        with mock.patch.object(ml, "fetch_alpha_checks", side_effect=lambda s, alpha_id, **kw: payloads[alpha_id]):
            data = alphas.cmd_alpha_check(arguments(payloads), None)
        rows = {row["alpha_id"]: row for row in data["results"]}
        self.assertFalse(data["complete"])
        self.assertEqual(data["summary"], {"complete": 3, "pending": 4, "error": 0,
                                           "passed": 1, "failed": 1, "warning": 1})
        for alpha_id in ("pending", "missing", "no-result", "unknown"):
            self.assertEqual(rows[alpha_id]["status"], "pending")
            self.assertTrue(rows[alpha_id]["payload_received"])
            self.assertFalse(rows[alpha_id]["checks_resolved"])
        self.assertEqual(rows["pending"]["checks"], payloads["pending"]["checks"])
        self.assertEqual(rows["warning"]["outcome"], "warning")
        self.assertEqual(rows["fail"]["status"], "complete")
        self.assertEqual(rows["fail"]["outcome"], "failed")
        self.assertEqual(rows["fail"]["self_correlation"]["result"], "FAIL")
        self.assertEqual(rows["missing"]["checks"], {})

    def test_timeout_and_per_id_service_errors_preserve_successful_peers(self):
        def fetch(session, alpha_id, **kwargs):
            self.assertEqual(kwargs["max_wait_seconds"], 7)
            if alpha_id == "timeout":
                return {"complete": False, "payload_received": False, "checks": None}
            if alpha_id == "missing":
                raise ml.NotFound("unknown alpha")
            if alpha_id == "service":
                raise ml.ServiceError("HTTP 503 fetching checks")
            if alpha_id == "network":
                raise requests.ConnectionError("private transport detail")
            return checked(alpha_id)
        with mock.patch.object(ml, "fetch_alpha_checks", side_effect=fetch):
            data = alphas.cmd_alpha_check(arguments(("ok", "timeout", "missing", "service", "network"), max_wait=7), None)
        rows = data["results"]
        self.assertEqual([row["status"] for row in rows], ["complete", "pending", "error", "error", "error"])
        self.assertEqual(rows[2]["error"], {"code": "NOT_FOUND", "message": "unknown alpha", "hint": None})
        self.assertEqual(rows[3]["error"]["code"], "SERVICE_ERROR")
        self.assertEqual(rows[4]["error"]["code"], "SERVICE_ERROR")
        self.assertNotIn("private transport detail", json.dumps(data))
        self.assertFalse(rows[1]["payload_received"])

    def test_auth_refresh_retries_only_auth_failed_ids(self):
        refreshed = requests.Session()
        self.addCleanup(refreshed.close)
        self.login.side_effect = [self.session, refreshed]
        calls = []
        def fetch(session, alpha_id, **kwargs):
            calls.append((session, alpha_id))
            if session is self.session and alpha_id in {"expired", "rejected"}:
                raise ml.AuthError("expired")
            if alpha_id == "rejected":
                raise ml.AuthError("still rejected")
            if alpha_id == "pending":
                return {"complete": False, "checks": None}
            if alpha_id == "missing":
                raise ml.NotFound("unknown alpha")
            return checked(alpha_id)
        ids = ("ok", "expired", "pending", "missing", "rejected", "last")
        with mock.patch.object(ml, "fetch_alpha_checks", side_effect=fetch):
            data = alphas.cmd_alpha_check(arguments(ids), None)
        self.assertEqual([alpha_id for _, alpha_id in calls], [*ids, "expired", "rejected"])
        self.assertTrue(all(session is refreshed for session, _ in calls[-2:]))
        self.assertEqual([row["status"] for row in data["results"]],
                         ["complete", "complete", "pending", "error", "error", "complete"])
        self.assertEqual(data["results"][4]["error"]["code"], "AUTH_FAILED")
        self.assertEqual(self.login.call_args_list,
                         [mock.call(use_cache=True, refresh=False), mock.call(use_cache=False, refresh=True)])

    def test_concurrent_refresh_reuses_new_authentication_only_for_rejected_ids(self):
        refreshed = requests.Session()
        self.addCleanup(refreshed.close)
        self.session.cookies.set("token", "old")
        refreshed.cookies.set("token", "new")
        self.login.side_effect = [self.session, refreshed]
        calls, lock = [], threading.Lock()
        def fetch(session, alpha_id, **kwargs):
            token = session.cookies.get("token")
            with lock:
                calls.append((alpha_id, token))
            self.assertIsNot(session, self.session)
            self.assertIsNot(session, refreshed)
            if token == "old" and alpha_id == "expired":
                raise ml.AuthError("expired")
            return checked(alpha_id)
        with mock.patch.object(ml, "fetch_alpha_checks", side_effect=fetch):
            data = alphas.cmd_alpha_check(arguments(("ok", "expired", "last"), workers=2), None)
        self.assertCountEqual(calls, [("ok", "old"), ("expired", "old"),
                                      ("last", "old"), ("expired", "new")])
        self.assertTrue(data["complete"])
        self.assertEqual([row["alpha_id"] for row in data["results"]], ["ok", "expired", "last"])

    def test_failed_refresh_preserves_finished_evidence(self):
        self.login.side_effect = [self.session, ml.CaptchaRequired("captcha needed")]
        def fetch(session, alpha_id, **kwargs):
            if alpha_id == "expired":
                raise ml.AuthError("expired")
            return checked(alpha_id)
        with mock.patch.object(ml, "fetch_alpha_checks", side_effect=fetch):
            data = alphas.cmd_alpha_check(arguments(("ok", "expired")), None)
        self.assertEqual(data["results"][0]["status"], "complete")
        self.assertEqual(data["results"][1]["error"]["code"], "AUTH_CAPTCHA")
        self.assertEqual(data["results"][1]["error"]["hint"], ml.CaptchaRequired.hint)

    def test_initial_auth_failure_remains_normal_typed_error_envelope(self):
        self.login.side_effect = ml.MissingCredentials("credentials missing")
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
            code = app.main(["alpha", "check", "A"])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 3)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "AUTH_MISSING")
        self.log_view.assert_not_called()

    def test_partial_batch_uses_success_envelope(self):
        def fetch(session, alpha_id, **kwargs):
            if alpha_id == "missing":
                raise ml.NotFound("unknown alpha")
            return checked(alpha_id)
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(ml, "fetch_alpha_checks", side_effect=fetch), \
                mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
            code = app.main(["alpha", "check", "ok", "missing", "--max-wait", "0"])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["data"]["complete"])
        self.assertEqual(payload["data"]["summary"]["error"], 1)

    def test_invalid_wait_and_worker_limits_fail_before_authentication(self):
        for value in (-1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(max_wait=value), self.assertRaises(CliError) as caught:
                alphas.cmd_alpha_check(arguments(max_wait=value), None)
            self.assertEqual(caught.exception.code, "USAGE")
        for workers in (0, 4):
            with self.subTest(workers=workers), self.assertRaises(CliError):
                alphas.cmd_alpha_check(arguments(workers=workers), None)
        self.login.assert_not_called()

    def test_concurrent_workers_are_bounded_isolated_and_closed(self):
        self.session.auth = ("account", "secret")
        self.session.cookies.set("token", "authenticated")
        self.session.headers["X-Session"] = "original"
        lock, ready = threading.Lock(), threading.Event()
        created, observed = [], []
        active = peak = 0
        real_session = requests.Session
        def create():
            session = real_session()
            session.close = mock.Mock(wraps=session.close)
            created.append(session)
            return session
        def fetch(session, alpha_id, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 3:
                    ready.set()
                observed.append((session.cookies.get("token"), session.headers["X-Session"], session.auth))
            try:
                self.assertTrue(ready.wait(3), "three independent workers must be able to enter")
                session.cookies.set("token", alpha_id)
                session.headers["X-Session"] = alpha_id
                if alpha_id == "C":
                    raise ml.ServiceError("HTTP 503")
                return checked(alpha_id)
            finally:
                with lock:
                    active -= 1
        with mock.patch.object(alphas.requests, "Session", side_effect=create), \
                mock.patch.object(ml, "fetch_alpha_checks", side_effect=fetch):
            data = alphas.cmd_alpha_check(arguments(("F", "E", "D", "C", "B", "A"), workers=3), None)
        self.assertEqual(peak, 3)
        self.assertEqual([row["alpha_id"] for row in data["results"]], ["F", "E", "D", "C", "B", "A"])
        self.assertEqual(data["summary"]["complete"], 5)
        self.assertEqual(data["results"][3]["status"], "error")
        self.assertEqual(observed, [("authenticated", "original", ("account", "secret"))] * 6)
        self.assertEqual(self.session.cookies.get("token"), "authenticated")
        self.assertEqual(self.session.headers["X-Session"], "original")
        self.assertEqual(len({id(session.cookies) for session in created}), 6)
        self.assertTrue(all(session.adapters["https://"] is not self.session.adapters["https://"] for session in created))
        for session in created:
            session.close.assert_called_once_with()
        self.login.assert_called_once_with(use_cache=True, refresh=False)


class FetchCheckHttpTests(unittest.TestCase):
    def test_http_failures_are_typed_without_polling_or_retry(self):
        for status, error in ((401, ml.AuthError), (403, ml.AuthError), (404, ml.NotFound),
                              (429, ml.RateLimited), (500, ml.ServiceError), (503, ml.ServiceError)):
            with self.subTest(status=status):
                response = mock.Mock(status_code=status, headers={"Retry-After": "10"})
                session = mock.Mock()
                session.get.return_value = response
                with mock.patch.object(ml.time, "sleep") as sleep, self.assertRaises(error):
                    ml.fetch_alpha_checks(session, "A", max_wait_seconds=0)
                session.get.assert_called_once()
                sleep.assert_not_called()

    def test_payload_completion_is_distinct_from_pending_check_resolution(self):
        response = mock.Mock(status_code=200, headers={})
        response.json.return_value = {"is": {"checks": [
            {"name": "SELF_CORRELATION", "result": "PENDING"},
            {"name": "LOW_SHARPE", "result": "WARNING", "value": 1.2},
        ]}}
        session = mock.Mock()
        session.get.return_value = response
        result = ml.fetch_alpha_checks(session, "A", max_wait_seconds=0)
        self.assertTrue(result["complete"])
        self.assertTrue(result["payload_received"])
        row = alphas._check_result("A", result)
        self.assertFalse(row["complete"])
        self.assertFalse(row["checks_resolved"])
        self.assertEqual(row["checks"]["LOW_SHARPE"]["result"], "WARNING")
        self.assertEqual(row["self_correlation"]["result"], "PENDING")

    def test_empty_successful_payload_and_unfinished_job_remain_explicit(self):
        for headers, payload_received in (({}, True), ({"Retry-After": "10"}, False)):
            with self.subTest(headers=headers):
                response = mock.Mock(status_code=200, headers=headers)
                response.json.return_value = {}
                session = mock.Mock()
                session.get.return_value = response
                result = ml.fetch_alpha_checks(session, "A", max_wait_seconds=0)
                self.assertFalse(result["complete"])
                self.assertEqual(result["payload_received"], payload_received)
                self.assertIsNone(result["checks"])


if __name__ == "__main__":
    unittest.main()
