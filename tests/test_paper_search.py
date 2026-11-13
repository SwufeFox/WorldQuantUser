#!/usr/bin/env python3
"""Offline OpenAlex retrieval, source-link and cache regressions."""

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

import paper_search as ps


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class SourceLinkTests(unittest.TestCase):
    def test_primary_landing_page_precedes_doi_and_openalex(self):
        paper = ps.normalise({
            "id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/paper",
            "primary_location": {"landing_page_url": "https://publisher.org/paper"},
            "best_oa_location": {"landing_page_url": "https://repository.org/paper",
                                 "pdf_url": "https://repository.org/paper.pdf"},
            "open_access": {"is_oa": True}})
        self.assertEqual(paper["url"], "https://publisher.org/paper")
        self.assertEqual(paper["landing_page_url"], "https://publisher.org/paper")
        self.assertEqual(paper["pdf_url"], "https://repository.org/paper.pdf")
        self.assertTrue(paper["open_access"])

    def test_other_locations_supply_available_source_links(self):
        paper = ps.normalise({"id": "https://openalex.org/W1", "locations": [
            {"landing_page_url": "https://repository.org/paper",
             "pdf_url": "https://repository.org/paper.pdf"}]})
        self.assertEqual(paper["url"], "https://repository.org/paper")
        self.assertEqual(paper["pdf_url"], "https://repository.org/paper.pdf")
        self.assertFalse(paper["open_access"])

    def test_doi_then_openalex_fallback_without_invented_availability(self):
        work = {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/paper"}
        self.assertEqual(ps.normalise(work)["url"], work["doi"])
        del work["doi"]
        paper = ps.normalise(work)
        self.assertEqual(paper["url"], work["id"])
        self.assertNotIn("landing_page_url", paper)
        self.assertNotIn("pdf_url", paper)
        self.assertFalse(paper["open_access"])


class PaperCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = temporary.name
        environment = mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": self.directory})
        environment.start()
        self.addCleanup(environment.stop)
        self.payload = {"meta": {"count": 2}, "results": [
            {"title": "Research", "cited_by_count": 20,
             "primary_location": {"landing_page_url": "https://publisher.org/paper"}},
            {"title": "Low cited", "cited_by_count": 1}]}
        transport = mock.patch.object(ps.urllib.request, "urlopen",
                                      return_value=_Response(self.payload))
        self.urlopen = transport.start()
        self.addCleanup(transport.stop)

    def path(self):
        return Path(ps._cache_path("alpha", 10, None, None))

    def test_effective_request_reuses_cache_and_preserves_source_links(self):
        first = ps.search_papers(" alpha ", limit=999, from_year="0", min_citations="0")
        second = ps.search_papers("alpha", limit=50)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertFalse(first["cached"])
        self.assertEqual(first["papers"], second["papers"])
        self.assertEqual(second["papers"][0]["url"], "https://publisher.org/paper")
        self.urlopen.assert_called_once()
        query = ps.urllib.parse.parse_qs(ps.urllib.parse.urlparse(
            self.urlopen.call_args.args[0].full_url).query)
        self.assertEqual(query, {"search": ["alpha"], "per-page": ["50"]})

    def test_all_filters_and_complete_query_distinguish_requests(self):
        options = [{}, {"limit": 2}, {"from_year": 2020}, {"min_citations": 5}]
        for option in options:
            self.assertFalse(ps.search_papers("alpha", **option)["cached"])
        prefix = "long query " * 10
        self.assertFalse(ps.search_papers(prefix + "one")["cached"])
        self.assertFalse(ps.search_papers(prefix + "two")["cached"])
        for option in options:
            self.assertTrue(ps.search_papers("alpha", **option)["cached"])
        self.assertEqual(self.urlopen.call_count, 6)

    def test_filters_keep_relevance_order_and_request_year_server_side(self):
        result = ps.search_papers("alpha", limit=2, from_year=2020, min_citations=5)
        self.assertEqual([paper["title"] for paper in result["papers"]], ["Research"])
        self.assertEqual(result["total_available"], 2)
        self.assertEqual(result["returned"], 1)
        query = ps.urllib.parse.parse_qs(ps.urllib.parse.urlparse(
            self.urlopen.call_args.args[0].full_url).query)
        self.assertEqual(query["filter"], ["from_publication_date:2020-01-01"])
        self.assertEqual(query["per-page"], ["6"])

    def test_empty_results_are_reusable(self):
        self.urlopen.return_value = _Response({"results": [], "meta": {"count": 0}})
        self.assertFalse(ps.search_papers("alpha")["cached"])
        self.assertTrue(ps.search_papers("alpha")["cached"])
        self.urlopen.assert_called_once()

    def test_expiry_is_relative_to_creation_not_time_buckets(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with mock.patch.object(ps.cache_store, "utc_now", return_value=start) as clock:
            ps.search_papers("alpha")
            clock.return_value = start + timedelta(seconds=ps.CACHE_TTL_SECONDS - 1)
            self.assertTrue(ps.search_papers("alpha")["cached"])
            clock.return_value = start + timedelta(seconds=ps.CACHE_TTL_SECONDS)
            self.assertFalse(ps.search_papers("alpha")["cached"])
        self.assertEqual(self.urlopen.call_count, 2)
        self.assertEqual(len(list(self.path().parent.glob("*.json"))), 1)

    def test_corrupt_and_legacy_entries_refetch(self):
        self.path().parent.mkdir(parents=True)
        invalid = ["{broken", "[]", json.dumps({"papers": [], "cached": False}),
                   json.dumps({"_cache": ["bad"], "payload": {}}),
                   json.dumps(ps.cache_store.build_entry("papers", [], ttl=60)),
                   json.dumps(ps.cache_store.build_entry("papers", {
                       "query": "alpha", "source": "openalex", "returned": 1,
                       "papers": [None]}, ttl=60))]
        for content in invalid:
            with self.subTest(content=content):
                self.path().write_text(content, encoding="utf-8")
                result = ps.search_papers("alpha")
                self.assertFalse(result["cached"])
                self.assertEqual(result["returned"], 2)
                self.assertTrue(ps.search_papers("alpha")["cached"])
        self.assertEqual(self.urlopen.call_count, len(invalid))

    def test_concurrent_identical_requests_fetch_and_write_once(self):
        barrier = threading.Barrier(4)
        original_lock = ps.cache_store.file_lock

        @contextlib.contextmanager
        def coordinated_lock(path):
            barrier.wait(timeout=10)
            with original_lock(path):
                yield

        with mock.patch.object(ps.cache_store, "file_lock", coordinated_lock), \
                mock.patch.object(ps.cache_store, "write_json_atomic",
                                  wraps=ps.cache_store.write_json_atomic) as write:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: ps.search_papers("alpha"), range(4)))
        self.assertEqual(sum(result["cached"] for result in results), 3)
        self.assertTrue(all(result["returned"] == 2 for result in results))
        self.urlopen.assert_called_once()
        write.assert_called_once()

    def test_write_failure_returns_success_without_refetching(self):
        with mock.patch.object(ps.cache_store, "write_json_atomic",
                               side_effect=PermissionError("read only")):
            result = ps.search_papers("alpha")
        self.assertEqual(result["returned"], 2)
        self.assertFalse(result["cached"])
        self.urlopen.assert_called_once()
        self.assertFalse(self.path().exists())

    def test_unwritable_cache_directory_returns_success_without_refetching(self):
        blocked = Path(self.directory) / "cache"
        blocked.write_text("not a directory", encoding="utf-8")
        result = ps.search_papers("alpha")
        self.assertEqual(result["returned"], 2)
        self.assertFalse(result["cached"])
        self.urlopen.assert_called_once()

    def test_transport_failure_neither_retries_nor_poisons_cache(self):
        self.urlopen.side_effect = OSError("offline")
        with self.assertRaises(OSError):
            ps.search_papers("alpha")
        self.urlopen.assert_called_once()
        self.assertFalse(self.path().exists())
        self.urlopen.side_effect = None
        self.assertFalse(ps.search_papers("alpha")["cached"])

    def test_invalid_json_is_not_cached(self):
        self.urlopen.return_value = mock.MagicMock()
        self.urlopen.return_value.__enter__.return_value.read.return_value = b"{broken"
        with self.assertRaises(ValueError):
            ps.search_papers("alpha")
        self.assertFalse(self.path().exists())
        self.urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
