"""Consumer-visible incentive metadata, cache separation and bounded catalog views."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cache_store
import machine_lib as ml
from research_app import app, incentives


class CatalogIncentiveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        patcher = mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": str(self.root),
                                               "BRAIN_USERNAME": "synthetic-catalog"})
        patcher.start()
        self.addCleanup(patcher.stop)
        policy = ml.metadata_cache_policy()
        self.addCleanup(lambda: ml.configure_metadata_cache(max_age=policy["max_age"],
            refresh=policy["refresh"], enabled=policy["enabled"]))
        ml.configure_metadata_cache(max_age=3600, refresh=False, enabled=True)
        self.session = ml.requests.Session()
        self.session.auth = ("synthetic-catalog", "placeholder")
        patcher = mock.patch.object(ml, "load_session", return_value=self.session)
        patcher.start()
        self.addCleanup(patcher.stop)

    def invoke(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()):
            code = app.main([*arguments, "--research-dir", str(self.root), "--quiet"])
        return code, json.loads(stdout.getvalue())

    def test_theme_filter_has_separate_caches_and_never_filters_by_multiplier(self):
        calls = []

        def request(session, method, path, params=None, **kwargs):
            self.assertEqual(method, "GET")
            self.assertEqual(path, "/data-sets")
            theme = params.get("theme")
            calls.append(theme)
            return {"count": 1, "results": [{"id": str(theme), "pyramidMultiplier": 2.25, "themes": []}]}

        with mock.patch.object(ml, "request_json", side_effect=request):
            for theme in ("true", "false", None, "true", "false", None):
                arguments = ["catalog", "datasets"] + (["--theme", theme] if theme else [])
                code, document = self.invoke(*arguments)
                self.assertEqual(code, 0, document)
                self.assertEqual(document["data"]["datasets"][0]["id"], str(theme))
        self.assertEqual(calls, ["true", "false", None])

    def test_catalog_summary_bounds_rows_but_retains_full_cached_evidence(self):
        rows = [{"id": "d%d" % index, "name": "Synthetic dataset", "description": "text " * 300,
                 "category": {"id": "synthetic-category", "name": "Synthetic category"},
                 "pyramidMultiplier": 2.25,
                 "themes": [{"id": "theme-%d" % member, "name": "Synthetic theme", "multiplier": 3.5}
                            for member in range(9)]}
                for index in range(25)]
        with mock.patch.object(ml, "request_json", return_value={"count": len(rows), "results": rows}):
            code, document = self.invoke("catalog", "datasets", "--summary", "--region", "TEST_REGION")
        self.assertEqual(code, 0, document)
        data = document["data"]
        self.assertEqual(data["count"], 25)
        self.assertEqual(data["displayed_count"], 20)
        self.assertEqual(data["rows_omitted"], 5)
        self.assertEqual(data["datasets"][0]["pyramid"]["region"], "TEST_REGION")
        self.assertEqual(data["datasets"][0]["themes"]["count"], 9)
        self.assertEqual(data["datasets"][0]["themes"]["omitted"], 4)
        self.assertNotIn("theme-8", json.dumps(data["datasets"]))
        stored = cache_store.read_entry(data["cache"]["path"])["payload"]
        self.assertEqual(stored[-1]["id"], "d24")
        self.assertEqual(stored[0]["themes"][-1]["id"], "theme-8")
        self.assertEqual(stored[0]["description"], rows[0]["description"])

    def test_missing_metadata_stays_unknown_and_explicit_zero_and_empty_remain_visible(self):
        rows = [{"id": "unknown"}, {"id": "zero", "pyramidMultiplier": 0, "themes": []},
                {"id": "signed", "pyramidMultiplier": -0.25},
                {"id": "bad", "pyramidMultiplier": True, "themes": None}]
        with mock.patch.object(ml, "request_json", return_value={"count": 4, "results": rows}):
            code, document = self.invoke("catalog", "datasets", "--summary")
        self.assertEqual(code, 0, document)
        unknown, zero, signed, bad = document["data"]["datasets"]
        self.assertEqual(unknown["pyramid"]["state"], "unknown")
        self.assertIsNone(unknown["pyramid"]["multiplier"])
        self.assertIsNone(unknown["pyramid"]["category"])
        self.assertIsNone(unknown["themes"]["count"])
        self.assertEqual(zero["pyramid"]["state"], "reported")
        self.assertEqual(zero["pyramid"]["multiplier"], 0)
        self.assertEqual(zero["themes"]["items"], [])
        self.assertEqual(signed["pyramid"]["multiplier"], -0.25)
        self.assertIsNone(bad["pyramid"]["multiplier"])

    def test_field_summary_does_not_leak_unbounded_expression_lists(self):
        rows = [{"id": "synthetic_%02d" % index, "type": "MATRIX", "description": "Fixture",
                 "category": {"id": "synthetic-category"}, "pyramidMultiplier": 2.25}
                for index in range(25)]
        with mock.patch.object(ml, "get_datafields", return_value=rows):
            code, document = self.invoke("catalog", "fields", "--dataset", "synthetic",
                                          "--summary", "--expressions")
        self.assertEqual(code, 0, document)
        data = document["data"]
        self.assertEqual(data["scanned"], 25)
        self.assertEqual(data["rows_omitted"], 5)
        self.assertEqual(data["expressions_scope"], "displayed_fields")
        self.assertIn("synthetic_00", json.dumps(data["expressions"]))
        self.assertNotIn("synthetic_24", json.dumps(data["expressions"]))

    def test_unknown_check_state_and_native_effective_value_are_not_inferred(self):
        checks = ml._extract_checks({"is": {"checks": [
            {"name": "MATCHES_PYRAMID", "result": "PENDING", "effective": 1.5, "multiplier": 2.25,
             "pyramids": [{"name": "synthetic-%d" % index, "multiplier": 2.25} for index in range(8)]},
            {"name": "MATCHES_THEMES", "result": "WARNING", "themes": []},
        ]}})
        view = incentives.check_summary(checks)
        pyramid = view["checks"]["MATCHES_PYRAMID"]
        self.assertEqual(pyramid["result"], "PENDING")
        self.assertEqual(pyramid["effective"], 1.5)
        self.assertEqual(pyramid["pyramids"]["omitted"], 3)
        self.assertEqual(checks["MATCHES_PYRAMID"]["pyramids"][-1]["name"], "synthetic-7")
        self.assertEqual(view["checks"]["MATCHES_THEMES"]["result"], "WARNING")
        self.assertIsNone(view["checks"]["MATCHES_THEMES"]["multiplier"])
        self.assertEqual(incentives.check_summary(None)["state"], "unknown")

    def test_non_boolean_theme_is_rejected_before_network_or_cache(self):
        with mock.patch.object(ml, "cached_metadata", side_effect=AssertionError("cache reached")):
            with self.assertRaises(ValueError):
                ml.get_datasets(self.session, theme="false")


if __name__ == "__main__":
    unittest.main()
