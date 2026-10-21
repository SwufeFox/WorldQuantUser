#!/usr/bin/env python3
"""Tests for the DeepSeek literature search wrapper.

All offline: the HTTP layer is stubbed, so these run without a key and without
spending a single request. The behaviour worth pinning is the error mapping —
each status code has to become a distinct exception so the CLI can report a
useful next step instead of one opaque failure.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import deepseek_search as ds  # noqa: E402


class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class ApiKeyTests(unittest.TestCase):
    def test_missing_key_raises_with_actionable_message(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(ds.machine_lib, "_load_dotenv", lambda: None):
            with self.assertRaises(ds.MissingSearchKey) as ctx:
                ds.api_key()
            self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))

    def test_env_key_is_used(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=True):
            self.assertEqual(ds.api_key(), "sk-test")

    def test_alias_key_is_accepted(self):
        with mock.patch.dict(os.environ, {"WEBSEARCH_API_KEY": "sk-alias"}, clear=True):
            self.assertEqual(ds.api_key(), "sk-alias")

    def test_available_reports_without_network(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(ds.machine_lib, "_load_dotenv", lambda: None):
            self.assertFalse(ds.available())
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=True):
            self.assertTrue(ds.available())


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"},
                                   clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_payload_carries_query_and_count(self):
        captured = {}

        def fake_post(url, payload, timeout=60):
            captured["url"] = url
            captured["payload"] = payload
            return {"results": [{"title": "t", "url": "u"}]}

        with mock.patch.object(ds, "_post", fake_post):
            results = ds.search("alpha decay", count=5)
        self.assertEqual(captured["url"], ds.SEARCH_URL)
        self.assertEqual(captured["payload"]["query"], "alpha decay")
        self.assertEqual(captured["payload"]["count"], 5)
        self.assertEqual(len(results), 1)

    def test_count_is_clamped_to_the_documented_range(self):
        seen = {}

        def fake_post(url, payload, timeout=60):
            seen.update(payload)
            return {"results": []}

        with mock.patch.object(ds, "_post", fake_post):
            ds.search("q", count=999)
        self.assertEqual(seen["count"], 50)

    def test_overlong_query_is_rejected_locally(self):
        with self.assertRaises(ds.SearchServiceError) as ctx:
            ds.search("x" * (ds.MAX_QUERY_CHARS + 1))
        self.assertIn("128", str(ctx.exception))

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ds.SearchServiceError):
            ds.search("   ")

    def test_fetch_requires_http_scheme(self):
        with self.assertRaises(ds.SearchServiceError):
            ds.fetch("file:///etc/passwd")


class ErrorMappingTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"},
                                   clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def _raise_for(self, response):
        with mock.patch.object(ds.requests, "post", return_value=response):
            with self.assertRaises(ds.DeepSeekSearchError) as ctx:
                ds.search("q")
        return ctx.exception

    def test_401_is_auth_error(self):
        self.assertIsInstance(
            self._raise_for(_Response(401, {"error": {"message": "bad key"}})),
            ds.SearchAuthError)

    def test_402_is_insufficient_balance(self):
        self.assertIsInstance(
            self._raise_for(_Response(402, {"error": {"message": "no funds"}})),
            ds.SearchInsufficientBalance)

    def test_429_is_rate_limited(self):
        self.assertIsInstance(
            self._raise_for(_Response(429, {"error": {"message": "slow"}})),
            ds.SearchRateLimited)

    def test_404_is_blocked(self):
        self.assertIsInstance(
            self._raise_for(_Response(404, {"error": {"message": "gone"}})),
            ds.SearchBlocked)

    def test_500_is_service_error(self):
        self.assertIsInstance(
            self._raise_for(_Response(500, text="boom")), ds.SearchServiceError)


class QualityFilterTests(unittest.TestCase):
    def test_low_rerank_score_is_dropped(self):
        results = [{"rerank_score": 2.0}, {"rerank_score": 0.5}]
        self.assertEqual(len(ds.high_quality(results)), 1)

    def test_authority_floor_is_applied_when_asked(self):
        results = [{"rerank_score": 2.0, "authority": 4},
                   {"rerank_score": 2.0, "authority": 1}]
        self.assertEqual(len(ds.high_quality(results, min_authority=3)), 1)

    def test_missing_score_is_treated_as_low(self):
        self.assertEqual(ds.high_quality([{"title": "x"}]), [])


if __name__ == "__main__":
    unittest.main()