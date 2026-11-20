#!/usr/bin/env python3
"""Tests for `catalog find`: one search across every dataset.

`catalog fields` takes a single dataset id, so asking "is there a short-term debt field
anywhere?" meant paging fourteen datasets by hand. The search itself is the same local
ranking; what this adds is the loop plus a tie-break that reflects what can actually
carry an alpha (coverage), which these tests pin.
"""

import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cache_store
import machine_lib as ml
import research_paths
from research_app import app
from research_app import catalog as cli_catalog


def _field(field_id, coverage, description="", kind="MATRIX"):
    return {"id": field_id, "description": description, "type": kind,
            "coverage": coverage}


class CatalogFindTests(unittest.TestCase):
    def setUp(self):
        self.datasets = [
            {"id": "fundamental6", "category": {"name": "Fundamental"}},
            {"id": "fundamental2", "category": {"name": "Fundamental"}},
        ]
        self.fields = {
            "fundamental6": [_field("debt", 0.50, "Total debt"),
                             _field("assets", 0.98, "Total assets")],
            "fundamental2": [_field("debt_repayments_total", 0.79, "Repayments on debt")],
        }

    def _args(self, **overrides):
        base = dict(query="debt", dataset=None, min_coverage=None, fuzzy=False,
                    any_term=False, region="USA", universe="TOP3000", delay=1,
                    instrument_type="EQUITY", limit=10)
        base.update(overrides)
        return argparse.Namespace(**base)

    def _run(self, **overrides):
        with mock.patch.object(ml, "get_datasets", return_value=self.datasets), \
                mock.patch.object(ml, "get_datafields",
                                  side_effect=lambda s, **kw: self.fields[kw["dataset_id"]]):
            return cli_catalog.cmd_catalog_find(self._args(**overrides), None)

    def test_it_searches_every_dataset_and_reports_the_source(self):
        data = self._run()
        self.assertEqual(data["datasets_scanned"], 2)
        self.assertEqual(data["fields_scanned"], 3)
        datasets = {row["dataset"] for row in data["fields"]}
        self.assertEqual(datasets, {"fundamental6", "fundamental2"})

    def test_low_coverage_matches_are_dropped_on_request(self):
        data = self._run(min_coverage=0.6)
        ids = [row["id"] for row in data["fields"]]
        self.assertNotIn("debt", ids)          # 0.50 < 0.6
        self.assertIn("debt_repayments_total", ids)

    def test_ranking_prefers_an_exact_id_then_coverage(self):
        data = self._run()
        self.assertEqual(data["fields"][0]["id"], "debt")

    def test_date_coverage_preserves_native_value_and_missing(self):
        self.fields["fundamental6"][0]["dateCoverage"] = 0
        self.fields["fundamental2"].append(
            {**_field("debt_issued", 0.85), "dateCoverage": 0.76}
        )
        rows = self._run()["fields"]
        self.assertEqual([row["id"] for row in rows],
                         ["debt", "debt_issued", "debt_repayments_total"])
        self.assertEqual([row["date_coverage"] for row in rows], [0, 0.76, None])

    def test_a_dataset_filter_limits_the_scan(self):
        data = self._run(dataset="fundamental2")
        self.assertEqual(data["datasets_scanned"], 1)
        self.assertEqual({row["dataset"] for row in data["fields"]}, {"fundamental2"})


class CatalogCacheTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        environment = mock.patch.dict(os.environ, {research_paths.ENV_VAR: workspace.name,
                                                   "BRAIN_USERNAME": "synthetic-first"})
        environment.start()
        self.addCleanup(environment.stop)
        self.session = SimpleNamespace(auth=("Synthetic-First", "placeholder"))
        self.datasets = [{"id": "first"}, {"id": "second"}]

    def invoke(self, *args):
        return self.invoke_with_progress(*args)[0]

    def invoke_with_progress(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
            code = app.main(["catalog", *args, "--compact"])
        document = json.loads(stdout.getvalue())
        self.assertEqual(code, 0, document)
        return document["data"], stderr.getvalue()

    def test_warm_datasets_fields_and_find_use_disk_before_login(self):
        with mock.patch.object(ml, "_get_datasets", return_value=self.datasets), \
                mock.patch.object(ml, "_get_datafields", return_value=[_field("first_signal", 0.7)]):
            ml.get_datasets(self.session)
            for row in self.datasets:
                ml.get_datafields(self.session, dataset_id=row["id"])

        with mock.patch.object(ml, "load_session", side_effect=AssertionError("warm cache should not log in")):
            datasets = self.invoke("datasets")
            fields = self.invoke("fields", "--dataset", "first")
            search = self.invoke("find", "signal")
        self.assertTrue(datasets["cache"]["hit"])
        self.assertTrue(fields["cache"]["hit"])
        self.assertEqual(search["count"], len(self.datasets))
        self.assertTrue(all(row["metadata"]["hit"] for row in search["discovery"]["catalogs"]))
        self.assertEqual(search["discovery"]["state"], "unknown")

    def test_find_reuses_session_across_catalog_misses_and_relogin(self):
        with mock.patch.object(ml, "load_session", return_value=self.session) as load, \
                mock.patch.object(ml, "_get_datasets", return_value=self.datasets) as datasets, \
                mock.patch.object(ml, "_get_datafields", return_value=[]) as fields:
            self.invoke("find", "signal", "--relogin")
        load.assert_called_once_with(use_cache=True, refresh=True)
        self.assertIs(datasets.call_args.args[0], self.session)
        self.assertEqual([call.args[0] for call in fields.call_args_list], [self.session] * len(self.datasets))

    def test_find_scans_past_first_page_and_ranks_filtered_fields_globally(self):
        first_page = [
            {**_field("first_qualified", 0.99), "alphaCount": 0, "pyramidMultiplier": 2},
            {**_field("low_coverage", 0.7), "alphaCount": 0, "pyramidMultiplier": 9},
            {**_field("vector", 0.99, kind="VECTOR"), "alphaCount": 0, "pyramidMultiplier": 9},
            {**_field("unknown_count", 0.99), "pyramidMultiplier": 9},
            {**_field("boolean_count", 0.99), "alphaCount": False, "pyramidMultiplier": 9},
            {**_field("string_count", 0.99), "alphaCount": "0", "pyramidMultiplier": 9},
            {**_field("negative_count", 0.99), "alphaCount": -1, "pyramidMultiplier": 9},
        ]
        first_page += [{**_field(f"used_{index}", 0.99), "alphaCount": 20}
                       for index in range(51 - len(first_page))]
        fields = {
            "analyst69": first_page + [{**_field("found_on_second_page", 0.8),
                                        "alphaCount": 0, "userCount": 0, "pyramidMultiplier": 3}],
            "analyst15": [{**_field("other_dataset", 0.82), "alphaCount": 0,
                           "userCount": 3, "pyramidMultiplier": 4},
                          {**_field("missing_multiplier", 0.99), "alphaCount": 0}],
        }

        def request_json(session, method, path, params=None, **_kwargs):
            self.assertIs(session, self.session)
            self.assertEqual(method, "GET")
            self.assertEqual({key: params[key] for key in ("region", "universe", "delay", "instrumentType")},
                             {"region": "EUR", "universe": "TOP2500", "delay": 1, "instrumentType": "EQUITY"})
            if path == "/data-sets":
                return {"count": len(fields), "results": [{"id": dataset} for dataset in fields]}
            self.assertEqual(path, "/data-fields")
            self.assertEqual(params["limit"], 50)
            self.assertEqual(params["type"], "MATRIX")
            rows = [row for row in fields[params["dataset.id"]] if row["type"] == params["type"]]
            offset = params["offset"]
            return {"count": len(rows), "results": rows[offset:offset + params["limit"]]}

        arguments = ("find", "--dataset", "analyst69,analyst15", "--region", "EUR", "--universe", "TOP2500",
                     "--type", "MATRIX", "--max-alpha-count", "0", "--min-coverage", "0.8",
                     "--sort", "pyramid-multiplier")
        with mock.patch.object(ml, "load_session", return_value=self.session) as load, \
                mock.patch.object(ml, "request_json", side_effect=request_json) as request:
            result, progress = self.invoke_with_progress(*arguments, "--limit", "2", "--stream")
        load.assert_called_once()
        offsets = [(call.kwargs["params"]["dataset.id"], call.kwargs["params"]["offset"])
                   for call in request.call_args_list if call.args[2] == "/data-fields"]
        self.assertCountEqual(offsets, [("analyst69", 0), ("analyst69", 50), ("analyst15", 0)])
        self.assertTrue(result["discovery"]["complete"])
        self.assertEqual(result["count"], 4)
        self.assertEqual([row["id"] for row in result["fields"]], ["other_dataset", "found_on_second_page"])
        self.assertEqual([(row["alpha_count"], row["user_count"], row["pyramid_multiplier"])
                          for row in result["fields"]], [(0, 3, 4), (0, 0, 3)])
        catalogs = {row["dataset"]: row for row in result["discovery"]["catalogs"]}
        self.assertEqual({key: row["matches"] for key, row in catalogs.items()}, {"analyst69": 2, "analyst15": 2})
        events = [json.loads(line) for line in progress.splitlines() if line.startswith("{")]
        self.assertEqual({event["dataset"] for event in events if event.get("dataset")}, set(fields))

        with mock.patch.object(ml, "load_session", side_effect=AssertionError("warm cache should not log in")), \
                mock.patch.object(ml, "request_json", side_effect=AssertionError("warm cache should not query")):
            warm = self.invoke(*arguments, "--limit", "4")
        self.assertTrue(warm["cache"]["hit"])
        self.assertTrue(warm["discovery"]["complete"])
        self.assertEqual(warm["fields"][:2], result["fields"])
        self.assertEqual([row["id"] for row in warm["fields"][2:]], ["first_qualified", "missing_multiplier"])
        self.assertIsNone(warm["fields"][-1]["pyramid_multiplier"])

    def test_find_distinguishes_no_match_from_limited_and_unlisted_datasets(self):
        selection = ["available", "bad_page", "no_match", "not_listed", "rate_limited", "after_rate_limit"]
        datasets = [dataset for dataset in selection if dataset != "not_listed"]

        def request_json(_session, _method, path, params=None, **_kwargs):
            if path == "/data-sets":
                return {"count": len(datasets), "results": [{"id": dataset} for dataset in datasets]}
            self.assertEqual(path, "/data-fields")
            dataset = params["dataset.id"]
            self.assertNotIn(dataset, {"not_listed", "after_rate_limit"})
            if dataset == "bad_page":
                return {"count": 1, "results": []}
            if dataset == "rate_limited":
                raise ml.RateLimited("server limited this read")
            row = {**_field(dataset, 0.92), "alphaCount": 7 if dataset == "no_match" else 0}
            return {"count": 1, "results": [row]}

        with mock.patch.object(ml, "load_session", return_value=self.session), \
                mock.patch.object(ml, "request_json", side_effect=request_json):
            data = self.invoke("find", "--dataset", ",".join(selection), "--max-alpha-count", "0")
        self.assertEqual([row["id"] for row in data["fields"]], ["available"])
        self.assertFalse(data["discovery"]["complete"])
        evidence = {row["dataset"]: row for row in data["discovery"]["catalogs"]}
        self.assertEqual(evidence["available"]["matches"], 1)
        self.assertEqual(evidence["available"]["fields_scanned"], 1)
        self.assertIsNone(evidence["bad_page"]["matches"])
        self.assertEqual(evidence["bad_page"]["error_code"], "SERVICE_ERROR")
        self.assertEqual(evidence["no_match"]["matches"], 0)
        self.assertEqual(evidence["no_match"]["fields_scanned"], 1)
        self.assertIsNone(evidence["rate_limited"]["matches"])
        self.assertEqual(evidence["rate_limited"]["error_code"], "RATE_LIMITED")
        self.assertIsNone(evidence["not_listed"]["matches"])
        self.assertEqual(evidence["not_listed"]["state"], "unknown")
        self.assertEqual(evidence["after_rate_limit"]["state"], "not_scanned")
        self.assertIsNone(evidence["after_rate_limit"]["matches"])

    def test_malformed_field_id_does_not_discard_other_datasets(self):
        fields = {
            "before": [_field("first_signal", 0.9)],
            "malformed": [_field("also_signal", 0.9), _field(42, 0.9)],
            "after": [_field("last_signal", 0.9)],
        }

        def request_json(_session, _method, path, params=None, **_kwargs):
            if path == "/data-sets":
                rows = [{"id": dataset} for dataset in fields]
            else:
                self.assertEqual(path, "/data-fields")
                rows = fields[params["dataset.id"]]
            return {"count": len(rows), "results": rows}

        with mock.patch.object(ml, "load_session", return_value=self.session), \
                mock.patch.object(ml, "request_json", side_effect=request_json) as request:
            result = self.invoke("find", "--min-coverage", "0.8")
            request.reset_mock()
            fields["malformed"] = [_field("repaired_signal", 0.9)]
            retried = self.invoke("find", "--min-coverage", "0.8")

        self.assertFalse(result["discovery"]["complete"])
        self.assertEqual([row["id"] for row in result["fields"]], ["first_signal", "last_signal"])
        catalogs = {row["dataset"]: row for row in result["discovery"]["catalogs"]}
        self.assertEqual(catalogs["malformed"]["error_code"], "SERVICE_ERROR")
        self.assertIsNone(catalogs["malformed"]["matches"])
        self.assertEqual(catalogs["before"]["matches"], 1)
        self.assertEqual(catalogs["after"]["matches"], 1)
        request.assert_called_once()
        self.assertEqual(request.call_args.kwargs["params"]["dataset.id"], "malformed")
        self.assertTrue(retried["discovery"]["complete"])
        self.assertEqual([row["id"] for row in retried["fields"]],
                         ["first_signal", "last_signal", "repaired_signal"])

    def test_stricter_max_age_refetches_and_labels_fallback(self):
        expired_by_policy = cache_store.utc_now() - timedelta(minutes=2)
        previous = [_field("old_signal", 0.7)]
        with mock.patch.object(ml, "_get_datafields", return_value=previous), \
                mock.patch.object(cache_store, "utc_now", return_value=expired_by_policy):
            ml.get_datafields(self.session, dataset_id="first")

        with mock.patch.object(ml, "load_session", return_value=self.session), \
                mock.patch.object(ml, "_get_datafields", side_effect=ml.ServiceError("unavailable")) as fetch:
            fallback = self.invoke("fields", "--dataset", "first", "--meta-max-age", "60")
        fetch.assert_called_once()
        self.assertEqual(fallback["cache"]["error"], "SERVICE_ERROR")
        self.assertTrue(fallback["cache"]["stale"])
        self.assertTrue(fallback["cache"]["degraded"])

        with mock.patch.object(ml, "load_session", return_value=self.session), \
                mock.patch.object(ml, "_get_datafields", return_value=[_field("new_signal", 0.7)]) as fetch:
            fresh = self.invoke("fields", "--dataset", "first", "--meta-max-age", "60")
        fetch.assert_called_once()
        self.assertEqual(fresh["fields"][0]["id"], "new_signal")
        self.assertFalse(fresh["cache"]["stale"])

    def test_account_switch_and_unknown_account_cannot_reuse_other_snapshot(self):
        details = {}
        with mock.patch.object(ml, "_get_datafields", return_value=[_field("a_signal", 0.7)]):
            ml.get_datafields(self.session, dataset_id="first", report=details)
        second = SimpleNamespace(auth=("synthetic-second", "placeholder"))
        with mock.patch.dict(os.environ, {"BRAIN_USERNAME": "synthetic-second"}), \
                mock.patch.object(ml, "load_session", return_value=second) as load, \
                mock.patch.object(ml, "_get_datafields", return_value=[_field("b_signal", 0.7)]):
            result = self.invoke("fields", "--dataset", "first")
            visible_paths = {str(path) for path, bound in ml.catalog_field_cache_paths() if bound}
        load.assert_called_once()
        self.assertEqual(result["fields"][0]["id"], "b_signal")
        self.assertNotEqual(result["cache"]["path"], details["path"])
        self.assertEqual(visible_paths, {result["cache"]["path"]})

        unknown = {}
        with mock.patch.object(ml, "_get_datafields", return_value=[]) as fetch:
            ml.get_datafields(SimpleNamespace(), dataset_id="first", report=unknown)
        fetch.assert_called_once()
        self.assertTrue(unknown["disabled"])
        self.assertIsNone(unknown["path"])


if __name__ == "__main__":
    unittest.main()
