"""Offline synthetic checks for scoped metadata, type order and economic evidence."""

import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cache_store
import expression_lint
import field_contract
import field_index
import machine_lib as ml
from research_app import capabilities, feed, validation
from research_app.runtime import CliError


SETTINGS = {"region": "USA", "universe": "TOP3000", "delay": 1, "instrumentType": "EQUITY"}
FIELDS = [{"id": "sample_matrix", "type": "MATRIX", "dataset": "sample_dataset", "description": "Sample value"},
          {"id": "sample_vector", "type": "VECTOR", "dataset": "sample_dataset"},
          {"id": "sample_universe", "type": "UNIVERSE", "dataset": "sample_dataset"}]
COVERAGE = {"complete": True, "full_scope": True, "dataset_ids": ["sample_dataset"], "field_count": len(FIELDS)}


class ScopedCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.session = SimpleNamespace(auth=("synthetic-account", "placeholder"))
        for patch in (mock.patch.dict(os.environ, {"BRAIN_USERNAME": "synthetic-account"}),
                      mock.patch.object(ml.research_paths, "metadata_root", return_value=str(self.root / "metadata")),
                      mock.patch.object(ml, "session_cache_path", return_value=self.root / "session.json"),
                      mock.patch.object(ml, "metadata_cache_policy", return_value={"max_age": 3600})):
            patch.start()
            self.addCleanup(patch.stop)

    def capture(self):
        return capabilities.save_snapshot(self.session, SETTINGS, FIELDS, COVERAGE)

    def test_complete_scope_is_available_without_network(self):
        path = self.capture()
        self.assertTrue(Path(path).is_file())
        with mock.patch.object(ml, "get_datafields", side_effect=AssertionError("network")):
            scope = validation.cached_field_scope(SETTINGS)
        self.assertEqual(scope["evidence"]["state"], "verified")
        self.assertEqual(scope["field_types"]["sample_vector"], "VECTOR")
        for key, other in (("region", "EUR"), ("universe", "TOP1000"), ("delay", 0), ("instrumentType", "OTHER")):
            self.assertIsNone(validation.cached_field_scope({**SETTINGS, key: other})["known_fields"])

    def test_account_switch_and_missing_identity_cannot_verify(self):
        self.capture()
        with mock.patch.dict(os.environ, {"BRAIN_USERNAME": "different-account"}):
            self.assertIsNone(validation.cached_field_scope(SETTINGS)["known_fields"])
        with mock.patch.dict(os.environ, {"BRAIN_USERNAME": ""}):
            result = validation.cached_field_scope(SETTINGS)
            self.assertEqual(result["evidence"]["account"]["state"], "unknown")
            self.assertIsNone(result["field_types"])

    def test_stale_and_incomplete_snapshots_are_unknown(self):
        path = self.capture()
        payload = cache_store.read_entry(path)["payload"]
        old = cache_store.utc_now() - timedelta(hours=2)
        cache_store.write_json_atomic(path, cache_store.build_entry("scoped-capabilities", payload, ttl=3600, now=old))
        self.assertIsNone(validation.cached_field_scope(SETTINGS)["known_fields"])
        payload["coverage"] = {**COVERAGE, "complete": False}
        cache_store.write_json_atomic(path, cache_store.build_entry("scoped-capabilities", payload, ttl=3600))
        self.assertIsNone(validation.cached_field_scope(SETTINGS)["known_fields"])

    def test_maximum_age_is_stricter_than_stored_expiry(self):
        path = self.capture()
        payload = cache_store.read_entry(path)["payload"]
        old = cache_store.utc_now() - timedelta(minutes=2)
        cache_store.write_json_atomic(path, cache_store.build_entry("scoped-capabilities", payload, ttl=3600, now=old))
        self.assertIsNone(validation.cached_field_scope(SETTINGS, max_age_seconds=60)["known_fields"])

    def test_explicit_requirements_use_only_verified_catalog(self):
        paper = {"title": "Momentum with a proxy", "required_fields": ["sample_matrix", "absent_field"]}
        self.capture()
        result = feed.data_fit(paper, validation.cached_field_scope(SETTINGS))
        self.assertEqual(result["verdict"], "required_fields_missing")
        self.assertEqual(result["missing_fields"], ["absent_field"])
        available = feed.data_fit({"required_fields": ["sample_matrix"]}, validation.cached_field_scope(SETTINGS))
        self.assertEqual(available["verdict"], "fields_available")
        unknown = feed.data_fit(paper, validation.cached_field_scope({**SETTINGS, "delay": 0}))
        self.assertEqual(unknown["verdict"], "unknown")
        self.assertEqual(unknown["missing_fields"], [])
        self.assertEqual(len(unknown["unknown_fields"]), 2)

    def test_full_online_capture_is_saved_but_stale_fallback_is_not_verified(self):
        def datasets(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"id": "sample_dataset"}]
        def fields(session, **kwargs):
            kwargs["report"].update(complete=True)
            return FIELDS
        with mock.patch.object(ml, "get_datasets", side_effect=datasets), mock.patch.object(ml, "get_datafields", side_effect=fields):
            evidence = {}
            capabilities.fetch_fields(self.session, report=evidence)
        self.assertEqual(evidence["state"], "verified")
        self.assertEqual(validation.cached_field_scope(SETTINGS)["evidence"]["state"], "verified")
        def stale(session, **kwargs):
            kwargs["report"].update(complete=True, degraded=True, stale=True)
            return FIELDS
        with mock.patch.object(ml, "get_datasets", side_effect=datasets), mock.patch.object(ml, "get_datafields", side_effect=stale):
            evidence = {}
            capabilities.fetch_fields(self.session, report=evidence)
        self.assertEqual(evidence["state"], "unknown")


class PreflightEvidenceTests(unittest.TestCase):
    def test_catalog_failure_blocks_before_submission(self):
        session = mock.Mock()
        with mock.patch.object(ml, "fetch_operators", side_effect=RuntimeError("metadata unavailable")), \
                mock.patch.object(field_index, "fetch_all_fields", side_effect=RuntimeError("metadata unavailable")), \
                mock.patch.object(capabilities, "account_provenance", return_value={"state": "unknown"}):
            with self.assertRaises(CliError) as caught:
                validation._preflight_lint(session, [("rank(sample_matrix)", 0)], SETTINGS, SimpleNamespace(allow_lint_errors=False))
            report = validation._preflight_lint(session, [("rank(sample_matrix)", 0)], SETTINGS, SimpleNamespace(allow_lint_errors=True))
        session.post.assert_not_called()
        self.assertFalse(report["ok"])
        self.assertEqual(report["coverage"], {"operators": "unknown", "fields": "unknown"})
        self.assertEqual(report["field_catalog"]["scope"], SETTINGS)
        self.assertIn("nothing was submitted", str(caught.exception))

    def test_empty_scope_cannot_pass_preflight(self):
        def empty_fields(session, **kwargs):
            kwargs["report"].update(state="unknown", scope=SETTINGS, coverage={"complete": False}, freshness={"state": "fresh"})
            return []
        with mock.patch.object(ml, "fetch_operators", return_value=[{"name": "rank"}]), \
                mock.patch.object(field_index, "fetch_all_fields", side_effect=empty_fields), \
                mock.patch.object(capabilities, "account_provenance", return_value={"state": "unknown"}):
            with self.assertRaises(CliError):
                validation._preflight_lint(mock.Mock(), [("rank(sample_matrix)", 0)], SETTINGS, SimpleNamespace())


class ASTContractTests(unittest.TestCase):
    def test_vector_reduction_must_precede_matrix_operation(self):
        types = {row["id"]: row["type"] for row in FIELDS}
        good = expression_lint.lint_expression("rank(vec_avg(sample_vector))", field_types=types)
        bad = expression_lint.lint_expression("vec_avg(rank(sample_vector))", field_types=types)
        self.assertTrue(good["ok"])
        self.assertTrue(any(row["code"] == "type_mismatch" for row in bad["errors"]))
        unknown = expression_lint.lint_expression("rank(sample_vector)", field_types=None)
        self.assertFalse(any("suggestion" in row for row in unknown["errors"] + unknown["warnings"]))

    def test_universe_is_not_matrix_in_infix_or_final_signal(self):
        types = {row["id"]: row["type"] for row in FIELDS}
        for expression in ("sample_universe", "sample_matrix * sample_universe", "u = sample_universe; rank(u)"):
            self.assertFalse(expression_lint.lint_expression(expression, field_types=types)["ok"])

    def test_nested_economic_operation_keeps_unknown_evidence(self):
        result = field_contract.audit_expression("ts_delta(rank(sample_matrix), 20) / sample_matrix", FIELDS)
        self.assertEqual(result["state"], "unknown")
        self.assertTrue(any(row.get("reason") == "derived_or_unknown_operand" for row in result["operations"]))
        scoped = capabilities.field_scope(FIELDS, {"state": "verified"})
        report = expression_lint.lint_all([("ts_delta(rank(sample_matrix), 20)", 0)], field_types=scoped["field_types"])
        validation.attach_field_contracts(report, scoped)
        self.assertTrue(report["ok"])
        self.assertEqual(report["field_contracts"][0]["state"], "unknown")
        self.assertTrue(any(row["code"] == "field_contract_unknown" for row in report["warnings"]))


class VerifiedPreflightTests(unittest.TestCase):
    def test_declared_type_errors_block_but_unknown_economics_only_warn(self):
        def operators(session, **kwargs):
            kwargs["report"].update(complete=True, age_seconds=0)
            return [{"name": "rank"}, {"name": "ts_delta"}]

        def fields(session, **kwargs):
            kwargs["report"].update(state="verified", scope=SETTINGS, coverage=COVERAGE,
                                      freshness={"state": "fresh"})
            return FIELDS

        with mock.patch.object(ml, "fetch_operators", side_effect=operators), \
                mock.patch.object(field_index, "fetch_all_fields", side_effect=fields), \
                mock.patch.object(capabilities, "account_provenance", return_value={"state": "unknown"}):
            args = SimpleNamespace(allow_lint_errors=False)
            with self.assertRaises(CliError):
                validation._preflight_lint(mock.Mock(), [("rank(sample_vector)", 0)], SETTINGS, args)
            report = validation._preflight_lint(mock.Mock(), [("ts_delta(rank(sample_matrix), 20)", 0)], SETTINGS, args)
        self.assertTrue(report["ok"])
        self.assertEqual(report["coverage"], {"operators": "verified", "fields": "verified"})
        self.assertEqual(report["field_contracts"][0]["state"], "unknown")
