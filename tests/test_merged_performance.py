"""Read-only merged-performance protocol and CLI contracts, without BRAIN credentials."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

import machine_lib as ml
import research_paths
from research_app import app


def response(payload=None, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers or {})
    result._content = json.dumps(payload if payload is not None else {}).encode("utf-8")
    return result


def comparison():
    return {"partition": "synthetic", "partitionName": "Synthetic partition",
            "stats": {"before": {"sharpe": 0.0, "turnover": 0.5, "returns": None},
                      "after": {"sharpe": 0.75, "turnover": 0.25, "returns": 0.1}},
            "yearlyStats": {"records": [["synthetic-year", 2.0]]},
            "pnl": {"records": [["synthetic-date", 2.0]]},
            "score": {"before": None, "after": 7}}


class MergedPerformanceProtocolTests(unittest.TestCase):
    def test_user_team_and_competition_scopes_use_read_only_encoded_paths(self):
        for options, expected in (({}, "/users/self/"),
                                  ({"team_id": "team/with space"}, "/teams/team%2Fwith%20space/"),
                                  ({"competition_id": "contest"}, "/competitions/contest/")):
            with self.subTest(options=options):
                session = mock.Mock()
                session.get.return_value = response(comparison())
                document = ml.fetch_merged_performance(session, "A/1", **options)
                url = session.get.call_args.args[0]
                self.assertIn(expected, url)
                self.assertTrue(url.endswith("/alphas/A%2F1/before-and-after-performance"))
                self.assertTrue(document["complete"])
                self.assertEqual(document["payload"]["stats"]["before"]["sharpe"], 0.0)
                session.post.assert_not_called()
                session.patch.assert_not_called()
                self.assertGreater(session.get.call_args.kwargs["timeout"], 0)

    def test_retry_after_overrides_a_nonempty_ready_looking_body(self):
        session = mock.Mock()
        session.get.return_value = response(comparison(), headers={"Retry-After": "7200"})
        with mock.patch.object(ml, "sleep") as wait:
            pending = ml.fetch_merged_performance(session, "A1", max_wait_seconds=1)
        self.assertFalse(pending["complete"])
        self.assertIsNone(pending["payload"])
        self.assertEqual(pending["next_poll_after_seconds"], 7200)
        self.assertEqual(session.get.call_count, 1)
        wait.assert_not_called()

    def test_transient_pending_response_is_polled_without_another_method(self):
        session = mock.Mock()
        session.get.side_effect = [response(status=503, headers={"Retry-After": "1"}), response(comparison())]
        with mock.patch.object(ml, "sleep") as wait:
            document = ml.fetch_merged_performance(session, "A1", max_wait_seconds=10)
        self.assertTrue(document["complete"])
        self.assertEqual(session.get.call_count, 2)
        wait.assert_called_once_with(1)
        session.post.assert_not_called()

    def test_wait_budget_overrun_does_not_send_an_extra_get(self):
        session = mock.Mock()
        session.get.return_value = response(headers={"Retry-After": "1"})
        with mock.patch.object(ml.time, "monotonic", side_effect=[0, 0, 0.1, 5]), \
                mock.patch.object(ml, "sleep"):
            pending = ml.fetch_merged_performance(session, "A1", max_wait_seconds=2)
        self.assertFalse(pending["complete"])
        self.assertEqual(pending["next_poll_after_seconds"], 0)
        self.assertEqual(session.get.call_count, 1)

    def test_permission_missing_and_service_errors_are_not_empty_comparisons(self):
        for status, error_type in ((401, ml.AuthError), (403, ml.AuthError),
                                   (404, ml.NotFound), (429, ml.RateLimited), (500, ml.ServiceError)):
            with self.subTest(status=status):
                session = mock.Mock()
                session.get.return_value = response({"detail": "fixture error"}, status=status)
                with self.assertRaises(error_type):
                    ml.fetch_merged_performance(session, "A1")
                self.assertEqual(session.get.call_count, 1)

    def test_invalid_or_nonfinite_payload_is_not_a_successful_empty_result(self):
        for payload in ({}, [], {"stats": []}, {"stats": {}}, {"stats": {"before": {}}},
                        {"stats": {"before": 4}},
                        {"stats": {"before": {"sharpe": float("nan")}, "after": {}}}):
            with self.subTest(payload=payload):
                session = mock.Mock()
                session.get.return_value = response(payload)
                with self.assertRaises(ml.ServiceError):
                    ml.fetch_merged_performance(session, "A1")

    def test_conflicting_scopes_and_invalid_budgets_fail_before_io(self):
        for options in ({"team_id": "team", "competition_id": "contest"}, {"team_id": ""},
                        {"max_wait_seconds": -1}, {"max_wait_seconds": float("nan")},
                        {"max_wait_seconds": True}):
            with self.subTest(options=options):
                session = mock.Mock()
                with self.assertRaises(ValueError):
                    ml.fetch_merged_performance(session, "A1", **options)
                session.get.assert_not_called()


class MergedPerformanceCLITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        patcher = mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": str(self.root)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.session = requests.Session()
        patcher = mock.patch.object(ml, "load_session", return_value=self.session)
        self.login = patcher.start()
        self.addCleanup(patcher.stop)
        self.alpha = {"id": "A1", "regular": {"code": "rank(close)"}, "settings": {"region": "TEST_REGION"},
                      "is": {"sharpe": 0.4, "checks": [{"name": "MATCHES_PYRAMID", "result": "PASS",
                          "effective": 1.5, "multiplier": 2.25,
                          "pyramids": [{"name": "Synthetic pyramid", "multiplier": 2.25}]}]}}

    def invoke(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()):
            code = app.main(["alpha", "get", "A1", *arguments, "--research-dir", str(self.root), "--quiet"])
        return code, json.loads(stdout.getvalue())

    def test_default_get_preserves_incentives_without_extra_performance_request(self):
        with mock.patch.object(self.session, "get", return_value=response(self.alpha)) as get:
            code, document = self.invoke()
        self.assertEqual(code, 0, document)
        self.assertEqual(get.call_count, 1)
        self.assertNotIn("merged_performance", document["data"])
        self.assertEqual(document["data"]["metrics"]["sharpe"], 0.4)
        self.assertEqual(document["data"]["checks"]["MATCHES_PYRAMID"]["pyramids"][0]["name"], "Synthetic pyramid")
        self.assertEqual(document["data"]["incentives"]["basis"], "alpha_snapshot")

    def test_summary_keeps_native_stats_and_zero_based_deltas_but_omits_series(self):
        with mock.patch.object(self.session, "get", side_effect=[response(self.alpha), response(comparison())]):
            code, document = self.invoke("--performance")
        self.assertEqual(code, 0, document)
        result = document["data"]["merged_performance"]
        self.assertEqual(result["delta_after_minus_before"], {"sharpe": 0.75, "turnover": -0.25})
        self.assertEqual(result["unavailable_deltas"], ["returns"])
        self.assertEqual(result["data"]["stats"]["before"]["sharpe"], 0)
        self.assertNotIn("pnl", result["data"])
        self.assertEqual(set(result["omitted_sections"]), {"pnl", "yearlyStats"})
        self.assertFalse(result["train_only_verified"])
        self.assertEqual(document["data"]["metrics"]["sharpe"], 0.4)
        views = [json.loads(line) for line in Path(research_paths.views_log_path()).read_text().splitlines()]
        self.assertEqual(views[-1]["tool"], "alpha get --performance")
        self.assertEqual(views[-1]["view"], "validation")

    def test_full_view_preserves_series_partition_and_competition_scope(self):
        native = comparison()
        with mock.patch.object(self.session, "get", side_effect=[response(self.alpha), response(native)]) as get:
            code, document = self.invoke("--performance", "full", "--competition", "synthetic-contest")
        self.assertEqual(code, 0, document)
        result = document["data"]["merged_performance"]
        self.assertEqual(result["data"], native)
        self.assertEqual(result["omitted_sections"], [])
        self.assertEqual(result["scope"], {"kind": "competition", "id": "synthetic-contest"})
        self.assertIn("/competitions/synthetic-contest/", get.call_args_list[-1].args[0])

    def test_pending_comparison_keeps_alpha_data_and_explicit_retry_timing(self):
        pending = response(comparison(), headers={"Retry-After": "7200"})
        with mock.patch.object(self.session, "get", side_effect=[response(self.alpha), pending]), \
                mock.patch.object(ml, "sleep") as wait:
            code, document = self.invoke("--performance", "--max-wait", "1")
        self.assertEqual(code, 0, document)
        self.assertEqual(document["data"]["alpha_id"], "A1")
        result = document["data"]["merged_performance"]
        self.assertFalse(result["complete"])
        self.assertIsNone(result["data"])
        self.assertEqual(result["next_poll_after_seconds"], 7200)
        self.assertEqual(result["delta_after_minus_before"], {})
        wait.assert_not_called()
        self.assertFalse(Path(research_paths.views_log_path()).exists())

    def test_scope_flags_require_performance_and_are_mutually_exclusive(self):
        for arguments in (("--team", "team"), ("--performance", "--team", "team", "--competition", "contest"),
                          ("--performance", "--competition", ""), ("--performance", "--max-wait", "nan")):
            with self.subTest(arguments=arguments):
                code, document = self.invoke(*arguments)
                self.assertEqual(code, 2, document)
        self.login.assert_not_called()


if __name__ == "__main__":
    unittest.main()
