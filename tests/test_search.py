#!/usr/bin/env python3
"""Tests for the DeepSeek literature search wrapper.

All offline: the HTTP layer is stubbed, so these run without a key and without
spending a single request. The behaviour worth pinning is the error mapping —
each status code has to become a distinct exception so the CLI can report a
useful next step instead of one opaque failure.
"""

import contextlib
import json
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
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
        # Clearing os.environ is not enough: api_key() falls back to .env, so once a real
        # key exists on disk this test reads it and passes for the wrong reason. Measured
        # exactly that when the key was configured - the assertion saw the live key.
        with mock.patch.dict(os.environ, {"WEBSEARCH_API_KEY": "sk-alias"}, clear=True), \
                mock.patch.object(ds.machine_lib, "_load_dotenv", lambda: None):
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


class SearchCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = temporary.name
        self.rows = [{"title": "Research", "url": "https://example.org/paper"}]
        transport = mock.patch.object(ds, "_post", return_value={"results": self.rows})
        self.post = transport.start()
        self.addCleanup(transport.stop)

    def test_reuses_effective_request_and_reports_cache_truthfully(self):
        first = ds.search_cached(self.directory, "  alpha decay  ", count=999,
                                 need_content=1, ori_question="x" * 130)
        second = ds.search_cached(self.directory, "alpha decay", count=50,
                                  need_content=True, ori_question="x" * 128)
        self.assertEqual(first, (self.rows, False))
        self.assertEqual(second, (self.rows, True))
        self.post.assert_called_once()
        self.assertEqual(self.post.call_args.args[1], {
            "query": "alpha decay", "count": 50, "need_content": True,
            "ori_question": "x" * 128})

    def test_every_effective_option_gets_its_own_cache(self):
        options = [{}, {"count": 3}, {"need_content": True},
                   {"ori_question": "mechanism"}, {"ori_question": "evidence"}]
        for option in options:
            self.assertFalse(ds.search_cached(self.directory, "alpha", **option)[1])
        self.assertFalse(ds.search_cached(self.directory, "beta")[1])
        for option in options:
            self.assertTrue(ds.search_cached(self.directory, "alpha", **option)[1])
        self.assertEqual(self.post.call_count, 6)

    def test_empty_results_are_reusable(self):
        self.post.return_value = {"results": []}
        self.assertEqual(ds.search_cached(self.directory, "empty"), ([], False))
        self.assertEqual(ds.search_cached(self.directory, "empty"), ([], True))
        self.post.assert_called_once()

    def test_expiry_and_stricter_caller_age_refetch(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with mock.patch.object(ds.cache_store, "utc_now", return_value=start) as clock:
            ds.search_cached(self.directory, "alpha", max_age_hours=10)
            clock.return_value = start + timedelta(hours=2)
            self.assertTrue(ds.search_cached(self.directory, "alpha", max_age_hours=10)[1])
            self.assertFalse(ds.search_cached(self.directory, "alpha", max_age_hours=1)[1])
            clock.return_value = start + timedelta(hours=3)
            self.assertFalse(ds.search_cached(self.directory, "alpha", max_age_hours=10)[1])
        self.assertEqual(self.post.call_count, 3)

    def test_corruption_and_legacy_entries_are_refetched(self):
        path = ds.cache_path(self.directory, "alpha")
        path.parent.mkdir(parents=True)
        invalid = ["{broken", "[]", json.dumps({"query": "alpha", "count": 10,
                                                "results": self.rows}),
                   json.dumps({"_cache": ["bad"], "payload": {}}),
                   json.dumps(ds.cache_store.build_entry("deepseek-search", [], ttl=60)),
                   json.dumps(ds.cache_store.build_entry("deepseek-search", {
                       "request": ds._search_payload("alpha", 10, False, None),
                       "results": [None]}, ttl=60))]
        for content in invalid:
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                self.assertEqual(ds.search_cached(self.directory, "alpha"), (self.rows, False))
                self.assertTrue(ds.search_cached(self.directory, "alpha")[1])
        self.assertEqual(self.post.call_count, len(invalid))

    def test_concurrent_identical_requests_fetch_and_write_once(self):
        barrier = threading.Barrier(4)
        original_lock = ds.cache_store.file_lock

        @contextlib.contextmanager
        def coordinated_lock(path):
            barrier.wait(timeout=10)
            with original_lock(path):
                yield

        with mock.patch.object(ds.cache_store, "file_lock", coordinated_lock), \
                mock.patch.object(ds.cache_store, "write_json_atomic",
                                  wraps=ds.cache_store.write_json_atomic) as write:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: ds.search_cached(self.directory, "alpha"),
                                        range(4)))
        self.assertEqual(sum(cached for _, cached in results), 3)
        self.assertTrue(all(rows == self.rows for rows, _ in results))
        self.post.assert_called_once()
        write.assert_called_once()

    def test_cache_write_failure_preserves_one_successful_request(self):
        with mock.patch.object(ds.cache_store, "write_json_atomic",
                               side_effect=PermissionError("read only")):
            self.assertEqual(ds.search_cached(self.directory, "alpha"), (self.rows, False))
        self.post.assert_called_once()
        self.assertFalse(ds.cache_path(self.directory, "alpha").exists())

    def test_unwritable_cache_directory_preserves_one_successful_request(self):
        directory = Path(self.directory) / "file"
        directory.write_text("not a directory", encoding="utf-8")
        self.assertEqual(ds.search_cached(directory, "alpha"), (self.rows, False))
        self.post.assert_called_once()

    def test_transport_failure_is_not_cached_or_retried(self):
        self.post.side_effect = ds.SearchServiceError("unavailable")
        with self.assertRaises(ds.SearchServiceError):
            ds.search_cached(self.directory, "alpha")
        self.post.assert_called_once()
        self.assertFalse(ds.cache_path(self.directory, "alpha").exists())
        self.post.side_effect = None
        self.assertEqual(ds.search_cached(self.directory, "alpha"), (self.rows, False))


if __name__ == "__main__":
    unittest.main()