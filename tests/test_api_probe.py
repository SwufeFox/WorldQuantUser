"""Endpoint probe: 403 and 404 are different findings."""

import unittest
from unittest import mock

import machine_lib as ml
from research_app import account


class _Response:
    def __init__(self, status):
        self.status_code = status


class _Session:
    def __init__(self, mapping):
        self.mapping = mapping

    def request(self, method, url, timeout=30, allow_redirects=True):
        for suffix, status in self.mapping.items():
            if url.endswith(suffix):
                return _Response(status)
        return _Response(404)


class ApiProbeTests(unittest.TestCase):
    def test_statuses_and_notes_are_reported_per_endpoint(self):
        session = _Session({
            "/users/self/alphas": 200,
            "/alphas": 405,
            "/leaderboards": 404,
            "/correlations/prod": 403,
        })

        class Args:
            alpha = "A1"

        with mock.patch.object(ml, "load_session", return_value=session):
            data = account.cmd_api_probe(Args(), None)
        by_path = {row["path"]: row for row in data["endpoints"]}
        self.assertEqual(by_path["/users/self/alphas"]["status"], 200)
        self.assertEqual(by_path["/alphas"]["status"], 405)
        self.assertEqual(by_path["/leaderboards"]["note"], "absent")
        gated = [row for row in data["endpoints"] if "correlations/prod" in row["path"]]
        self.assertEqual(gated[0]["status"], 403)
        self.assertIn("level-gated", gated[0]["note"])

    def test_probe_retains_results_and_skips_endpoints_after_its_budget(self):
        clock = [100.0]
        session = mock.Mock()

        def get(*_args, **_kwargs):
            clock[0] += account.DIAGNOSTIC_SECONDS + 1
            return _Response(200)

        session.request.side_effect = get
        with mock.patch.object(account.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(ml, "load_session", return_value=session) as load:
            data = account.cmd_api_probe(mock.Mock(alpha="A1", no_cache=False, relogin=False), None)
        self.assertEqual(load.call_args.kwargs["deadline"], 100.0 + account.DIAGNOSTIC_SECONDS)
        session.request.assert_called_once_with("GET", ml.BRAIN_API_URL + "/users/self/alphas",
                                                timeout=account.PROBE_SECONDS, allow_redirects=False)
        self.assertEqual(data["endpoints"][0]["status"], 200)
        self.assertTrue(all(row["note"] == "diagnostic budget expired" for row in data["endpoints"][1:]))
