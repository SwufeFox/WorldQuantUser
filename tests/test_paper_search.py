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


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patch = mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": directory.name})
        patch.start()
        self.addCleanup(patch.stop)
        self.key = mock.patch.object(ps, "_api_key", return_value="test-secret")
        self.key_mock = self.key.start()
        self.addCleanup(self.key.stop)
        transport = mock.patch.object(ps.urllib.request, "urlopen")
        self.urlopen = transport.start()
        self.addCleanup(transport.stop)

    def work(self, key="W1", **extra):
        return {"id": "https://openalex.org/" + key, "title": "A paper " + key,
                "doi": "https://doi.org/10.1234/" + key.lower(),
                "publication_year": 2025, "cited_by_count": 0,
                "referenced_works": [], "has_content": {"pdf": True, "grobid_xml": True},
                "abstract_inverted_index": {"Full": [0], "abstract": [1]}, **extra}

    def response(self, value):
        if not isinstance(value, bytes):
            return _Response(value)
        response = mock.MagicMock()
        response.__enter__.return_value.read.side_effect = lambda size=-1: value if size < 0 else value[:size]
        return response

    def test_identifier_aliases_and_invalid_inputs(self):
        for value in ("w123", "W123", "https://openalex.org/W123"):
            self.assertEqual(ps.normalize_identifier(value), "W123")
        for value in ("10.1234/ABC", "doi:10.1234/ABC", "https://doi.org/10.1234/ABC"):
            self.assertEqual(ps.normalize_identifier(value), "10.1234/abc")
        for value in ("A1", "W1?api_key=secret", "https://evil.org/W1",
                      "https://openalex.org/W1?api_key=secret", "https://secret@doi.org/10.1234/a",
                      "https://doi.org/W1", "10.1/a", "https://openalex.org/10.1234/a"):
            with self.subTest(value=value), self.assertRaises(ps.PaperError):
                ps.normalize_identifier(value)
        self.urlopen.assert_not_called()

    def test_singleton_cache_reuses_canonical_and_doi_identifiers(self):
        self.urlopen.return_value = self.response(self.work())
        first = ps.get_paper("doi:10.1234/W1")
        self.assertFalse(first["cached"])
        self.assertEqual(first["paper"]["abstract"], "Full abstract")
        self.assertTrue(ps.get_paper("https://openalex.org/W1")["cached"])
        self.assertTrue(ps.get_paper("https://doi.org/10.1234/w1")["cached"])
        self.urlopen.assert_called_once()

    def test_wrong_work_identity_is_not_cached_as_alias(self):
        self.urlopen.return_value = self.response(self.work("W2"))
        for identifier in ("W1", "10.1234/w1"):
            with self.assertRaises(ps.PaperError) as failure:
                ps.get_paper(identifier)
            self.assertEqual(failure.exception.code, "SERVICE_ERROR")
            self.assertIsNone(ps._lookup_record(identifier))

    def test_secondary_doi_requires_location_evidence(self):
        work = self.work(locations=[{"doi": "https://doi.org/10.1234/preprint"}])
        self.urlopen.return_value = self.response(work)
        ps.get_paper("10.1234/preprint")
        self.assertTrue(ps.get_paper("10.1234/preprint")["cached"])
        self.assertTrue(ps.get_paper("W1")["cached"])

    def test_search_seeds_record_cache_without_content_fetch(self):
        self.urlopen.return_value = self.response({"results": [self.work()], "meta": {"count": 1}})
        ps.search_papers("a paper")
        self.assertTrue(ps.get_paper("W1")["cached"])
        self.urlopen.assert_called_once()

    def test_missing_metadata_remains_unknown(self):
        self.urlopen.return_value = self.response({"id": "https://openalex.org/W1"})
        paper = ps.get_paper("W1")["paper"]
        self.assertIsNone(paper["open_access"])
        self.assertIsNone(paper["referenced_works"])
        self.assertIsNone(paper["reading"]["grobid_xml"])
        self.assertEqual(ps.read_content("W1")["content"]["state"], "unknown")
        result = ps.related_papers("W1")
        self.assertIsNone(result["total_available"])
        self.assertFalse(result["coverage"]["references_known"])
        self.urlopen.assert_called_once()

    def test_legacy_search_cache_does_not_claim_singleton_metadata(self):
        result = {"query": "old", "source": "openalex", "papers": [{"title": "old"}], "returned": 1}
        ps._save_cache(ps._cache_path("old", 10, None, None), "papers", result)
        self.urlopen.return_value = self.response(self.work())
        self.assertTrue(ps.search_papers("old")["cached"])
        self.assertFalse(ps.get_paper("W1")["cached"])
        self.urlopen.assert_called_once()

    def test_record_write_failure_keeps_search_success(self):
        self.urlopen.return_value = self.response({"results": [self.work()]})
        with mock.patch.object(ps.cache_store, "write_json_atomic", side_effect=PermissionError):
            result = ps.search_papers("query")
        self.assertEqual(result["returned"], 1)

    def test_references_are_bounded_and_reordered_to_seed_order(self):
        references = ["https://openalex.org/W%d" % number for number in range(2, 110)]
        self.urlopen.side_effect = [self.response(self.work(referenced_works=references)),
                                  self.response({"results": [self.work("W3"), self.work("W2")]})]
        result = ps.related_papers("W1", limit=1)
        self.assertEqual([paper["openalex_id"] for paper in result["papers"]], ["W2"])
        self.assertEqual(result["total_available"], 108)
        self.assertEqual(result["coverage"]["requested_candidates"], 100)
        self.assertEqual(result["coverage"]["unfetched_candidates"], 8)
        self.assertTrue(result["coverage"]["truncated"])
        self.assertEqual(result["coverage"]["ordering"], "referenced_works_order")
        self.assertTrue(ps.related_papers("W1", limit=1)["cached"])
        self.assertEqual(self.urlopen.call_count, 2)
        query = ps.urllib.parse.parse_qs(ps.urllib.parse.urlsplit(self.urlopen.call_args.args[0].full_url).query)
        self.assertEqual(len(query["filter"][0].split("|")), 100)

    def test_citations_preserve_provider_order_and_filters(self):
        self.urlopen.side_effect = [self.response(self.work()), self.response({
            "results": [self.work("W9", cited_by_count=20), self.work("W2", cited_by_count=30)],
            "meta": {"count": 23}})]
        result = ps.related_papers("W1", "citations", limit=2, from_year=2020, min_citations=10)
        self.assertEqual([paper["openalex_id"] for paper in result["papers"]], ["W9", "W2"])
        self.assertEqual(result["coverage"]["ordering"], "provider_order")
        query = ps.urllib.parse.parse_qs(ps.urllib.parse.urlsplit(self.urlopen.call_args.args[0].full_url).query)
        self.assertEqual(query["filter"], ["cites:W1,from_publication_date:2020-01-01,cited_by_count:>9"])
        self.assertEqual(query["per-page"], ["2"])

    def test_auth_headers_and_errors_never_expose_key(self):
        for status, code in ((401, "AUTH_FAILED"), (403, "AUTH_FAILED"), (429, "RATE_LIMITED"),
                             (404, "NOT_FOUND"), (500, "SERVICE_ERROR")):
            self.urlopen.side_effect = ps.urllib.error.HTTPError(
                "https://api.openalex.org/?api_key=test-secret", status, "test-secret", {}, None)
            with self.assertRaises(ps.PaperError) as failure:
                ps.get_paper("W1")
            self.assertEqual(failure.exception.code, code)
            self.assertNotIn("test-secret", str(failure.exception))
            request = self.urlopen.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
            self.assertNotIn("test-secret", request.full_url)

    def test_content_windows_reuse_full_text_without_key(self):
        xml = b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div><head>Results</head><p>A <hi>real</hi> result.</p></div></body></text></TEI>'
        self.urlopen.side_effect = [self.response(self.work()), self.response(ps.gzip.compress(xml))]
        first = ps.read_content("W1", max_chars=7)["content"]
        self.assertEqual(first["state"], "read")
        self.assertEqual(first["text"], "Results")
        self.assertEqual(first["next_offset"], 7)
        self.key_mock.return_value = ""
        second = ps.read_content("10.1234/w1", offset=7, max_chars=100)["content"]
        self.assertEqual(second["text"], "\n\nA real result.")
        self.assertTrue(second["cached"])
        self.assertIsNone(second["next_offset"])
        self.assertEqual(self.urlopen.call_count, 2)
        cached = Path(first["cache_path"]).read_text(encoding="utf-8")
        self.assertNotIn("test-secret", cached)

    def test_legacy_tei_shape_and_malformed_content(self):
        legacy = b'<html><body><tei><text><div><head>Legacy</head><p>Actual text</p></div></text></tei></body></html>'
        self.urlopen.side_effect = [self.response(self.work()), self.response(legacy)]
        self.assertEqual(ps.read_content("W1")["content"]["text"], "Legacy\n\nActual text")
        for number, data in enumerate((b"<html>Not scholarly text</html>", b"<TEI", b'<!DOCTYPE TEI><TEI/>'), start=2):
            key = "W%d" % number
            self.urlopen.side_effect = [self.response(self.work(key)), self.response(data)]
            with self.assertRaises(ps.PaperError):
                ps.read_content(key)
            self.assertIsNone(ps._read_cache(ps._record_path(key, "content"), "paper_content"))

    def test_content_unavailable_or_missing_key_is_explicit(self):
        self.urlopen.return_value = self.response(self.work(has_content={"pdf": True, "grobid_xml": False}))
        result = ps.read_content("W1")
        self.assertEqual(result["content"]["state"], "unavailable")
        self.assertIsNone(result["content"]["text"])
        self.urlopen.assert_called_once()
        self.urlopen.return_value = self.response(self.work("W2"))
        self.key_mock.return_value = ""
        with self.assertRaises(ps.PaperError) as failure:
            ps.read_content("W2")
        self.assertEqual(failure.exception.code, "AUTH_MISSING")

    def test_content_size_window_and_wrong_host_bounds(self):
        work = self.work(content_urls={"grobid_xml": "https://evil.org/file?api_key=test-secret"})
        self.urlopen.side_effect = [self.response(work), self.response(b"x" * 101)]
        with mock.patch.object(ps, "MAX_CONTENT_BYTES", 100), self.assertRaises(ps.PaperError):
            ps.read_content("W1")
        self.assertEqual(self.urlopen.call_args.args[0].full_url, ps.CONTENT_BASE + "W1.grobid-xml")
        for kwargs in ({"offset": -1}, {"max_chars": 0}, {"max_chars": 100001}):
            with self.assertRaises(ps.PaperError):
                ps.read_content("W1", **kwargs)


    def test_empty_tei_and_missing_download_do_not_fabricate_text(self):
        self.urlopen.side_effect = [self.response(self.work()), self.response(b"<TEI><text/></TEI>")]
        content = ps.read_content("W1")["content"]
        self.assertEqual(content["state"], "unavailable")
        self.assertIsNone(content["text"])
        self.urlopen.side_effect = [self.response(self.work("W2")),
            ps.urllib.error.HTTPError("https://content.openalex.org/secret", 404, "secret", {}, None)]
        content = ps.read_content("W2")["content"]
        self.assertEqual(content["state"], "unavailable")
        self.assertNotIn("secret", json.dumps(content))

    def test_compressed_expansion_is_bounded(self):
        self.urlopen.side_effect = [self.response(self.work()),
            self.response(ps.gzip.compress(b"<TEI><text><p>" + b"a" * 1000 + b"</p></text></TEI>"))]
        with mock.patch.object(ps, "MAX_CONTENT_BYTES", 100), self.assertRaises(ps.PaperError):
            ps.read_content("W1")
        self.assertIsNone(ps._read_cache(ps._record_path("W1", "content"), "paper_content"))

    def test_alias_cache_cannot_return_different_work(self):
        ps._store_record(self.work("W2"))
        ps._save_cache(ps._record_path("10.1234/w1"), "paper_alias", {
            "identifier": "10.1234/w1", "openalex_id": "W2"})
        self.urlopen.return_value = self.response(self.work())
        result = ps.get_paper("10.1234/w1")
        self.assertEqual(result["paper"]["openalex_id"], "W1")
        self.assertFalse(result["cached"])


if __name__ == "__main__":
    unittest.main()
