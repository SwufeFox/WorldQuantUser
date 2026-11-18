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
        stdout = io.StringIO()
        with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", io.StringIO()):
            code = app.main(["catalog", *args, "--compact"])
        document = json.loads(stdout.getvalue())
        self.assertEqual(code, 0, document)
        return document["data"]

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
