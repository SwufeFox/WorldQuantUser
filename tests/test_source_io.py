"""Offline contracts for source transport, publication time and secret handling."""

import io
import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import source_io


class SourceIOTests(unittest.TestCase):
    def test_unknown_or_coarse_publication_time_is_not_collection_time(self):
        for value, precision in [(None, "unknown"), ("6个月前", "unknown"),
                                 ("2026", "year"), ("2026-11-17", "date"),
                                 ("2026-11-17T09:00:00", "unknown")]:
            with self.subTest(value=value):
                result = source_io.parse_timestamp(value)
                self.assertIsNone(result["value"])
                self.assertEqual(result["precision"], precision)
        result = source_io.parse_timestamp("Tue, 17 Nov 2026 09:30:00 -0500")
        self.assertEqual(result["value"], "2026-11-17T14:30:00+00:00")

    def test_canonical_urls_keep_content_identity_and_redact_private_parameters(self):
        first = source_io.canonical_url("https://EXAMPLE.org/story?id=1&utm_source=test#top")
        self.assertEqual(first, "https://example.org/story?id=1")
        self.assertNotEqual(first, source_io.canonical_url("https://example.org/story?id=2"))
        reported = source_io.public_url("https://example.org/feed?token=fixture-secret&id=7")
        self.assertNotIn("fixture-secret", reported)
        self.assertIn("id=7", reported)
        with self.assertRaises(source_io.SourceError):
            source_io.validate_url("https://user:password@example.org/feed")

    def test_headers_are_not_forwarded_to_redirected_storage(self):
        seen = []

        def transport(request, timeout):
            seen.append(request)
            return io.BytesIO(b'{"result": 1}')

        with mock.patch.object(source_io.urllib.request, "urlopen", side_effect=transport):
            payload, evidence = source_io.request_json("https://example.org/data", timeout=3,
                headers={"Authorization": "Bearer fixture-secret", "APCA-API-KEY-ID": "fixture-key"})
        self.assertEqual(payload, {"result": 1})
        redirected = urllib.request.HTTPRedirectHandler().redirect_request(
            seen[0], None, 302, "Moved", {}, "https://storage.example.org/data")
        self.assertIsNone(redirected.get_header("Authorization"))
        self.assertIsNone(redirected.get_header("Apca-api-key-id"))
        self.assertNotIn("fixture-secret", json.dumps(evidence))
        self.assertNotIn("fixture-key", json.dumps(evidence))

    def test_http_errors_keep_retry_timing_without_exposing_error_urls(self):
        error = urllib.error.HTTPError("https://example.org/?token=fixture-secret", 429,
                                       "echoed fixture-secret", {"Retry-After": "17"}, None)
        with mock.patch.object(source_io.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(source_io.SourceError) as caught:
                source_io.request_json("https://example.org/data")
        report = caught.exception.public()
        self.assertEqual(report["code"], "RATE_LIMITED")
        self.assertEqual(report["retry_after_seconds"], 17)
        self.assertNotIn("fixture-secret", json.dumps(report))

    def test_invalid_or_oversized_json_is_not_empty_success(self):
        for body in (b'{broken', b'{"price": NaN}', b'{"price": 1e999}', b'not-json'):
            with self.subTest(body=body), mock.patch.object(source_io.urllib.request, "urlopen",
                    return_value=io.BytesIO(body)):
                with self.assertRaises(source_io.SourceError) as caught:
                    source_io.request_json("https://example.org/data")
                self.assertEqual(caught.exception.code, "SERVICE_ERROR")
        with mock.patch.object(source_io.urllib.request, "urlopen", return_value=io.BytesIO(b'012345')):
            with self.assertRaises(source_io.SourceError):
                source_io.request_bytes("https://example.org/data", max_bytes=5)


if __name__ == "__main__":
    unittest.main()
