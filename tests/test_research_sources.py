"""Offline source parsing, access, provenance, and cache regression tests."""

import json
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cache_store
import research_sources as sources
from source_io import SourceError

COLLECTED = "2026-11-17T12:00:00+00:00"
RSS_SOURCE = {"id": "niche", "kind": "forum", "adapter": "rss", "url": "https://example.org/feed"}
RSS = b'''<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel><title>Forum</title>
 <item><guid>native-1</guid><title>Claim &amp; context</title><link>https://example.org/a?utm_source=rss</link>
 <content:encoded><![CDATA[<p>First <b>claim</b>.</p><p>Second &amp; detail.</p><script>hidden</script>]]></content:encoded>
 <dc:creator>Observer</dc:creator><pubDate>Tue, 17 Nov 2026 09:00:00 GMT</pubDate></item>
 <item><guid>native-2</guid><title>No date</title><description>Unconfirmed</description></item>
 </channel></rss>'''


def meta(url="https://example.org/feed"):
    return {"url": url, "collected_at": COLLECTED, "content_type": "application/json"}


def hn_row(identifier="42"):
    return {"objectID": identifier, "title": "A market claim", "url": "https://example.org/article",
            "story_text": "<p>Discussion</p>", "author": "observer", "points": 12,
            "num_comments": 3, "created_at": "2026-11-16T09:15:00Z"}


class SourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        patch = mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": temporary.name})
        patch.start()
        self.addCleanup(patch.stop)

    def config(self, definitions, bom=False):
        path = self.directory / "sources.json"
        path.write_text(json.dumps({"sources": definitions}), encoding="utf-8-sig" if bom else "utf-8")
        return path

    def assert_code(self, code, operation):
        with self.assertRaises(SourceError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_registry_bom_custom_and_scheduling(self):
        targeted = dict(RSS_SOURCE, id="targeted", url="https://example.org/feed?q={query}")
        registry = sources.load_sources(self.config([RSS_SOURCE, targeted], bom=True))
        self.assertEqual(set(sources.BUILTIN_SOURCES), {"paper", "news", "social", "forum", "trends"})
        self.assertEqual(registry["targeted"]["url"], targeted["url"])
        self.assertFalse(sources.requires_query(registry["niche"]))
        self.assertFalse(sources.requires_query(registry["trends"]))
        self.assertTrue(sources.requires_query(registry["news"]))
        self.assertTrue(sources.requires_query(registry["targeted"]))
        registry["news"]["url"] = "changed"
        self.assertNotEqual(sources.BUILTIN_SOURCES["news"]["url"], "changed")

    def test_bad_definitions_fail_before_network_without_leaking_values(self):
        bad = [dict(RSS_SOURCE, token="private-value"), dict(RSS_SOURCE, url="https://example.org/f?api_key=private-value"),
               dict(RSS_SOURCE, url="https://user:private-value@example.org/f"), dict(RSS_SOURCE, url="https://example.org/{topic}"),
               dict(RSS_SOURCE, id="news"), dict(RSS_SOURCE, kind="paper"),
               {"id": "x", "adapter": "web", "kind": "forum", "sites": ["example.org/forum"]},
               {"id": "x", "adapter": "bluesky", "kind": "social", "sort": "top"}]
        with mock.patch.object(sources, "request_bytes") as network:
            for definition in bad:
                with self.subTest(definition=definition):
                    error = self.assert_code("INVALID_INPUT", lambda: sources.load_sources(self.config([definition])))
                    self.assertNotIn("private-value", json.dumps(error.public()))
            self.assert_code("INVALID_INPUT", lambda: sources.load_sources(self.config([RSS_SOURCE, RSS_SOURCE])))
        network.assert_not_called()

    def test_rss_namespaces_html_and_unknown_time(self):
        with mock.patch.object(sources, "request_bytes", return_value=(RSS, meta())):
            packet = sources.fetch_source(RSS_SOURCE)
        first, second = packet["items"]
        self.assertEqual(first["id"], "native-1")
        self.assertEqual(first["author"], "Observer")
        self.assertEqual(first["url"], "https://example.org/a")
        self.assertEqual(first["excerpt"], "First claim. Second & detail.")
        self.assertEqual(first["published_at"], "2026-11-17T09:00:00+00:00")
        self.assertEqual(first["evidence"]["claim_state"], "unverified_claim")
        self.assertIsNone(second["published_at"])
        self.assertIsNone(second["published_at_raw"])
        self.assertEqual(second["time_precision"], "unknown")
        self.assertEqual(second["collected_at"], COLLECTED)
        self.assertEqual(packet["coverage"]["ordering"], "provider_feed_order")

    def test_atom_xhtml_relative_link_and_date_precision(self):
        atom = b'''<feed xmlns="http://www.w3.org/2005/Atom" xml:base="https://example.org/base/">
        <entry><id>urn:entry:1</id><title>Observation</title><link rel="self" href="ignored"/>
        <link href="post"/><content type="xhtml"><div xmlns="http://www.w3.org/1999/xhtml">A <b>claim</b></div></content>
        <author><name>Author</name></author><published>2026-11-16</published><updated>2026-11-17T00:00:00Z</updated></entry>
        <entry><id>urn:entry:2</id><title>Updated only</title><updated>2026-11-17T00:00:00Z</updated></entry>
        </feed>'''
        with mock.patch.object(sources, "request_bytes", return_value=(atom, meta())):
            packet = sources.fetch_source(RSS_SOURCE)
        first, second = packet["items"]
        self.assertEqual(first["url"], "https://example.org/base/post")
        self.assertEqual(first["excerpt"], "A claim")
        self.assertEqual(first["author"], "Author")
        self.assertEqual(first["time_precision"], "date")
        self.assertEqual(first["published_at_raw"], "2026-11-16")
        self.assertIsNone(first["published_at"])
        self.assertIsNone(second["published_at"])
        self.assertEqual(second["raw"]["updated"], "2026-11-17T00:00:00Z")

    def test_query_template_replaces_only_literal_placeholder(self):
        source = dict(RSS_SOURCE, url="https://example.org/search?q={query}&literal=%7Bquery%7D")
        with mock.patch.object(sources, "request_bytes", return_value=(RSS, meta())) as network:
            packet = sources.fetch_source(source, "rates & China/中国", limit=1)
        params = parse_qs(urlsplit(network.call_args.args[0]).query)
        self.assertEqual(params, {"q": ["rates & China/中国"], "literal": ["{query}"]})
        self.assertEqual(packet["query"], "rates & China/中国")
        self.assertEqual(packet["coverage"]["omitted_count"], 1)
        self.assertTrue(packet["coverage"]["has_more"])
        self.assertIn("{query}", sources.describe_source(source)["url"])
        with mock.patch.object(sources, "request_bytes") as network:
            self.assert_code("QUERY_REQUIRED", lambda: sources.fetch_source(source))
        network.assert_not_called()

    def test_google_news_is_explicitly_an_aggregator(self):
        with mock.patch.object(sources, "request_bytes", return_value=(RSS, meta())):
            packet = sources.fetch_source(sources.BUILTIN_SOURCES["news"], "rates")
        self.assertEqual(packet["source"], "google_news")
        self.assertEqual(packet["items"][0]["evidence"]["basis"], "aggregator_headline")
        self.assertIn("non-commercial", " ".join(packet["coverage"]["limitations"]))

    def test_bad_xml_and_truncated_response_are_typed_and_not_cached(self):
        for body in [b"<rss><channel>", b"<html>Challenge</html>", b'<!DOCTYPE rss><rss><channel/></rss>']:
            with self.subTest(body=body), mock.patch.object(sources, "request_bytes", return_value=(body, meta())):
                self.assert_code("FORMAT_ERROR", lambda: sources.fetch_source(RSS_SOURCE))
        with mock.patch.object(sources, "request_bytes", side_effect=SourceError("RESPONSE_TOO_LARGE", "Bound exceeded.")):
            self.assert_code("RESPONSE_TOO_LARGE", lambda: sources.fetch_source(RSS_SOURCE))
        self.assertEqual(list(self.directory.rglob("*.json")), [])

    def test_partial_rows_keep_valid_evidence_and_do_not_cache(self):
        payload = {"hits": [None, hn_row(), {"title": "Missing id"}], "nbPages": 1}
        with mock.patch.object(sources, "request_json", return_value=(payload, meta())) as network:
            packet = sources.fetch_source(sources.BUILTIN_SOURCES["forum"], "market")
            sources.fetch_source(sources.BUILTIN_SOURCES["forum"], "market")
        self.assertEqual(network.call_count, 2)
        self.assertEqual(packet["coverage"]["invalid_count"], 2)
        self.assertTrue(packet["coverage"]["partial"])
        self.assertEqual(len(packet["items"]), 1)
        with mock.patch.object(sources, "request_json", return_value=({"hits": [None]}, meta())):
            self.assert_code("FORMAT_ERROR", lambda: sources.fetch_source(sources.BUILTIN_SOURCES["forum"], "other"))

    def test_hn_preserves_discussion_external_link_and_native_counts(self):
        with mock.patch.object(sources, "request_json", return_value=({"hits": [hn_row()], "nbPages": 3, "nbHits": 25}, meta())) as network:
            packet = sources.fetch_source(sources.BUILTIN_SOURCES["forum"], "market")
        item = packet["items"][0]
        self.assertEqual(item["url"], "https://news.ycombinator.com/item?id=42")
        self.assertEqual(item["external_url"], "https://example.org/article")
        self.assertEqual(item["engagement"]["points"], 12)
        self.assertEqual(item["published_at"], "2026-11-16T09:15:00+00:00")
        self.assertTrue(packet["coverage"]["has_more"])
        self.assertEqual(parse_qs(urlsplit(network.call_args.args[0]).query)["tags"], ["story"])

    def test_front_page_is_fixed_and_does_not_claim_hn_rank(self):
        with mock.patch.object(sources, "request_json", return_value=({"hits": [hn_row()], "nbPages": 1}, meta())) as network:
            packet = sources.fetch_source(sources.BUILTIN_SOURCES["trends"], "ignored topic")
        self.assertIsNone(packet["query"])
        params = parse_qs(urlsplit(network.call_args.args[0]).query)
        self.assertEqual(params["tags"], ["front_page"])
        self.assertNotIn("query", params)
        self.assertEqual(packet["items"][0]["kind"], "trend")
        self.assertIn("not the current HN", " ".join(packet["coverage"]["limitations"]))

    def test_bluesky_preserves_native_identity_date_and_unknown_counts(self):
        row = {"uri": "at://did:plc:abc/app.bsky.feed.post/xyz", "author": {"did": "did:plc:abc", "handle": "writer.test"},
               "record": {"text": "A <b>literal</b> social claim", "createdAt": "2026-11-16T10:00:00Z"},
               "likeCount": 0, "replyCount": 4, "indexedAt": "2026-11-17T11:00:00Z"}
        with mock.patch.object(sources, "request_json", return_value=({"posts": [row], "cursor": "next", "hitsTotal": 100}, meta())) as network:
            packet = sources.fetch_source(sources.BUILTIN_SOURCES["social"], "market")
        item = packet["items"][0]
        self.assertEqual(item["id"], row["uri"])
        self.assertEqual(item["url"], "https://bsky.app/profile/did:plc:abc/post/xyz")
        self.assertEqual(item["author"], "writer.test")
        self.assertEqual(item["excerpt"], "A <b>literal</b> social claim")
        self.assertEqual(item["published_at"], "2026-11-16T10:00:00+00:00")
        self.assertEqual(item["engagement"]["likeCount"], 0)
        self.assertIsNone(item["engagement"]["repostCount"])
        self.assertEqual(parse_qs(urlsplit(network.call_args.args[0]).query)["sort"], ["latest"])
        self.assertTrue(packet["coverage"]["has_more"])

    def test_access_failures_propagate_without_retry_or_fake_empty_success(self):
        with mock.patch.object(sources, "request_json", side_effect=SourceError("ACCESS_BLOCKED", "Source access denied.")) as network:
            error = self.assert_code("ACCESS_BLOCKED", lambda: sources.fetch_source(sources.BUILTIN_SOURCES["social"], "market"))
        network.assert_called_once()
        self.assertEqual(error.public()["message"], "Source access denied.")
        for payload in [{}, [], {"posts": {}}, {"posts": [None]}]:
            with mock.patch.object(sources, "request_json", return_value=(payload, meta())):
                self.assert_code("FORMAT_ERROR", lambda: sources.fetch_source(sources.BUILTIN_SOURCES["social"], "market"))

    def test_source_urls_and_raw_links_redact_known_credentials(self):
        row = hn_row()
        row["url"] = "https://example.org/article?token=private-value&utm_source=x"
        with mock.patch.object(sources, "request_json", return_value=({"hits": [row]}, meta())):
            packet = sources.fetch_source(sources.BUILTIN_SOURCES["forum"], "privacy")
        self.assertNotIn("private-value", json.dumps(packet))
        self.assertIn("redacted", packet["items"][0]["external_url"])
        for path in self.directory.rglob("*.json"):
            self.assertNotIn("private-value", path.read_text())

    def test_cache_identity_hits_preserve_collection_and_expiry_refetches(self):
        with mock.patch.object(sources, "request_bytes", return_value=(RSS, meta())) as network:
            first = sources.fetch_source(RSS_SOURCE, "ignored", limit=1)
            hit = sources.fetch_source(RSS_SOURCE, "another ignored", limit=1)
            self.assertFalse(first["cached"])
            self.assertTrue(hit["cached"])
            self.assertEqual(hit["collected_at"], first["collected_at"])
            self.assertEqual(hit["coverage"]["as_of"], COLLECTED)
            self.assertGreaterEqual(hit["coverage"]["cache_age_seconds"], 0)
            network.assert_called_once()
            sources.fetch_source(RSS_SOURCE, limit=2)
            self.assertEqual(network.call_count, 2)
            for path in self.directory.rglob("*.json"):
                entry = json.loads(path.read_text())
                entry["_cache"]["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
                cache_store.write_json_atomic(path, entry)
            self.assertFalse(sources.fetch_source(RSS_SOURCE, limit=1)["cached"])
            self.assertEqual(network.call_count, 3)

    def test_cache_key_includes_query_and_source_definition(self):
        with mock.patch.object(sources, "request_bytes", return_value=(RSS, meta())) as network:
            source = dict(RSS_SOURCE, url="https://example.org/search?q={query}")
            sources.fetch_source(source, "one")
            sources.fetch_source(source, "two")
            sources.fetch_source(dict(source, kind="news"), "one")
        self.assertEqual(network.call_count, 3)

    def test_identical_misses_coalesce_under_lock(self):
        entered, release = threading.Event(), threading.Event()

        def transport(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test did not release transport")
            return RSS, meta()

        with mock.patch.object(sources, "request_bytes", side_effect=transport) as network, ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(sources.fetch_source, RSS_SOURCE)
            self.assertTrue(entered.wait(timeout=5))
            second = pool.submit(sources.fetch_source, RSS_SOURCE)
            release.set()
            packets = [first.result(timeout=10), second.result(timeout=10)]
        network.assert_called_once()
        self.assertEqual(sorted(packet["cached"] for packet in packets), [False, True])

    def test_valid_empty_result_is_distinct_from_malformed_payload(self):
        with mock.patch.object(sources, "request_json", return_value=({"hits": [], "nbPages": 0}, meta())) as network:
            packet = sources.fetch_source(sources.BUILTIN_SOURCES["forum"], "no match")
            hit = sources.fetch_source(sources.BUILTIN_SOURCES["forum"], "no match")
        self.assertEqual(packet["items"], [])
        self.assertFalse(packet["coverage"]["partial"])
        self.assertTrue(hit["cached"])
        network.assert_called_once()

    def test_provider_limit_is_reported_and_invalid_budgets_fail_before_io(self):
        with mock.patch.object(sources, "request_json", return_value=({"posts": []}, meta())) as network:
            packet = sources.fetch_source(sources.BUILTIN_SOURCES["social"], "market", limit=101)
        self.assertEqual(packet["coverage"]["requested_limit"], 101)
        self.assertEqual(packet["coverage"]["provider_request_limit"], 100)
        self.assertEqual(parse_qs(urlsplit(network.call_args.args[0]).query)["limit"], ["100"])
        with mock.patch.object(sources, "request_bytes") as network:
            for kwargs in ({"limit": True}, {"limit": 0}, {"timeout": float("nan")}, {"timeout": -1}):
                self.assert_code("INVALID_INPUT", lambda: sources.fetch_source(RSS_SOURCE, **kwargs))
        network.assert_not_called()

    def test_year_and_unparseable_dates_remain_native_unknowns(self):
        body = b'<rss version="2.0"><channel><item><title>Year</title><pubDate>2025</pubDate></item><item><title>Unparsed</title><pubDate>yesterday-ish</pubDate></item></channel></rss>'
        with mock.patch.object(sources, "request_bytes", return_value=(body, meta())):
            packet = sources.fetch_source(RSS_SOURCE)
        year, unknown = packet["items"]
        self.assertEqual(year["time_precision"], "year")
        self.assertEqual(year["published_at_raw"], "2025")
        self.assertIsNone(year["published_at"])
        self.assertEqual(unknown["published_at_raw"], "yesterday-ish")
        self.assertIsNone(unknown["published_at"])

    def test_file_config_resolves_relative_path_and_requires_no_topic(self):
        export = self.directory / "captured.jsonl"
        export.write_text('{"id":"sample-1","title":"An observation","text":"Captured claim"}\n', encoding="utf-8")
        config = self.config([{"id": "captured", "kind": "forum", "adapter": "file", "path": "captured.jsonl"}])
        source = sources.load_sources(config)["captured"]
        self.assertEqual(source["path"], str(export.resolve()))
        self.assertFalse(sources.requires_query(source))
        self.assertEqual(sources.describe_source(source)["cache_ttl_seconds"], 0)
        with mock.patch.object(sources, "request_bytes") as http, mock.patch.object(sources, "request_json") as api, \
                mock.patch.object(sources, "_cache_hit") as cached:
            packet = sources.fetch_source(source)
        http.assert_not_called()
        api.assert_not_called()
        cached.assert_not_called()
        self.assertEqual(packet["items"][0]["id"], "sample-1")
        self.assertIsNone(packet["coverage"]["endpoint"])

    def test_jsonl_import_preserves_thread_provenance_and_unknown_relative_dates(self):
        export = self.directory / "thread.jsonl"
        main = {"id": "synthetic-main", "title": "Synthetic discussion", "text": "An anecdotal metric: 1.23",
                "url": "https://example.org/discussion", "author": "participant",
                "published_at_raw": "6个月前", "content_scope": "main_post", "metric": 1.23}
        comment = {"id": "synthetic-comment", "title": "Reply", "content": "<b>Literal captured text</b>",
                   "parent_id": "synthetic-main", "published_at": "2026-10-01T09:00:00Z",
                   "published_at_raw": "2026-10-01 09:00 UTC", "content_scope": "comment"}
        export.write_text(json.dumps(main, ensure_ascii=False) + "\n" + json.dumps(comment) + "\n", encoding="utf-8-sig")
        source = {"id": "captured", "kind": "forum", "adapter": "file", "path": str(export)}
        with mock.patch.object(sources, "utc_now", return_value=COLLECTED):
            packet = sources.fetch_source(source, "ignored topic")
        first, second = packet["items"]
        self.assertEqual(first["raw"], main)
        self.assertEqual(second["raw"], comment)
        self.assertEqual(first["content_scope"], "main_post")
        self.assertEqual(second["content_scope"], "comment")
        self.assertEqual(second["parent_id"], "synthetic-main")
        self.assertEqual(second["source_line"], 2)
        self.assertEqual(second["source_path"], str(export.resolve()))
        self.assertEqual(first["published_at_raw"], "6个月前")
        self.assertEqual(first["time_precision"], "unknown")
        self.assertIsNone(first["published_at"])
        self.assertEqual(second["published_at"], "2026-10-01T09:00:00+00:00")
        self.assertEqual(second["published_at_raw"], "2026-10-01 09:00 UTC")
        self.assertEqual(second["excerpt"], "<b>Literal captured text</b>")
        self.assertEqual(first["collected_at"], COLLECTED)
        self.assertEqual(first["evidence"]["claim_state"], "unverified_claim")
        self.assertIsNone(packet["query"])
        self.assertEqual(packet["coverage"]["imported_at"], COLLECTED)

    def test_jsonl_bad_rows_preserve_valid_peers_and_report_omissions(self):
        export = self.directory / "partial.jsonl"
        export.write_bytes(b'{"id":"one","text":"One"}\nnot JSON\n\xff\n[]\n'
                           b'{"id":"two","content":"Two"}\n{"text":"Bad date","published_at":42}\n')
        source = {"id": "captured", "kind": "forum", "adapter": "file", "path": str(export)}
        packet = sources.fetch_source(source, limit=1)
        self.assertEqual(packet["items"][0]["id"], "one")
        self.assertEqual(packet["coverage"]["provider_returned_count"], 6)
        self.assertEqual(packet["coverage"]["invalid_count"], 4)
        self.assertEqual(packet["coverage"]["omitted_count"], 1)
        self.assertTrue(packet["coverage"]["partial"])
        self.assertTrue(packet["coverage"]["has_more"])
        export.write_text('{"text":42}\nnot JSON\n', encoding="utf-8")
        self.assert_code("FORMAT_ERROR", lambda: sources.fetch_source(source))

    def test_plain_exports_preserve_code_and_reflect_edits_without_cache(self):
        for suffix in (".txt", ".md"):
            with self.subTest(suffix=suffix):
                export = self.directory / ("captured" + suffix)
                text = '# Captured note\n```python\nraise RuntimeError("never execute")\n```\n'
                export.write_bytes(text.encode("utf-8-sig"))
                source = {"id": "captured", "kind": "forum", "adapter": "file", "path": str(export)}
                with mock.patch.object(sources, "utc_now", side_effect=[COLLECTED, "2026-11-17T12:01:00+00:00"]), \
                        mock.patch.object(sources, "_cache_hit") as cached:
                    first = sources.fetch_source(source)
                    export.write_text("Changed captured text", encoding="utf-8")
                    second = sources.fetch_source(source)
                cached.assert_not_called()
                item = first["items"][0]
                self.assertEqual(item["excerpt"], text)
                self.assertEqual(item["raw"], {"text": text})
                self.assertEqual(item["content_scope"], "captured_text")
                self.assertIsNone(item["source_line"])
                self.assertIsNone(item["published_at"])
                self.assertIsNone(item["parent_id"])
                self.assertFalse(first["cached"])
                self.assertFalse(second["cached"])
                self.assertNotEqual(item["id"], second["items"][0]["id"])
                self.assertNotEqual(first["collected_at"], second["collected_at"])
                self.assertEqual(second["items"][0]["excerpt"], "Changed captured text")

    def test_file_access_size_encoding_and_configuration_failures_are_typed(self):
        source = {"id": "captured", "kind": "forum", "adapter": "file", "path": str(self.directory / "missing.txt")}
        self.assert_code("FILE_ACCESS_ERROR", lambda: sources.fetch_source(source))
        self.assert_code("INVALID_INPUT", lambda: sources.load_sources(self.config([dict(source, path="export.py")])))
        self.assert_code("INVALID_INPUT", lambda: sources.load_sources(self.config([dict(source, encoding="latin-1")])))
        export = self.directory / "captured.txt"
        source["path"] = str(export)
        export.write_bytes(b"\xff\xfe")
        self.assert_code("FORMAT_ERROR", lambda: sources.fetch_source(source))
        export.write_bytes(b"x" * 17)
        with mock.patch.object(sources, "MAX_FILE_BYTES", 16):
            self.assert_code("RESPONSE_TOO_LARGE", lambda: sources.fetch_source(source))

    def test_market_delegation_never_uses_text_snapshot_cache(self):
        market = SimpleNamespace(validate_market=mock.Mock(return_value={"venue": "coinbase", "symbol": "BTC-USD", "depth": 2, "feed": "iex"}),
                                 fetch_market=mock.Mock(side_effect=[{"snapshot": 1}, {"snapshot": 2}]))
        source = {"id": "book", "kind": "market", "adapter": "market", "venue": "coinbase", "symbol": "BTC-USD", "depth": 2}
        with mock.patch.dict(sys.modules, {"market_sources": market}), mock.patch.object(sources, "_cache_hit") as cached:
            self.assertEqual(sources.fetch_source(source, timeout=4), {"snapshot": 1})
            self.assertEqual(sources.fetch_source(source, timeout=4), {"snapshot": 2})
        cached.assert_not_called()
        self.assertEqual(market.fetch_market.call_count, 2)
        market.fetch_market.assert_called_with("coinbase", "BTC-USD", depth=2, feed="iex", timeout=4)


if __name__ == "__main__":
    unittest.main()
