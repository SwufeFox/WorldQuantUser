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
import simulation_capabilities
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

    def test_preflight_routes_ast_fields_and_refreshes_only_used_datasets(self):
        path = self.capture()
        before = Path(path).read_bytes()

        def fields(session, **kwargs):
            self.assertIs(session, self.session)
            self.assertEqual(kwargs["dataset_id"], "sample_dataset")
            self.assertTrue(kwargs["refresh"])
            self.assertEqual(kwargs["universe"], SETTINGS["universe"])
            kwargs["report"].update(complete=True)
            return FIELDS

        def operators(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"name": "rank"}]

        with mock.patch.object(ml, "get_datasets", side_effect=AssertionError("unrelated discovery")), \
                mock.patch.object(ml, "get_datafields", side_effect=fields) as fetched, \
                mock.patch.object(ml, "fetch_operators", side_effect=operators):
            report = validation._preflight_lint(
                self.session, [("x = sample_matrix; rank(x) /* unrelated_news_field */", 0)],
                SETTINGS, SimpleNamespace(allow_lint_errors=False))
        self.assertTrue(report["ok"])
        self.assertEqual(fetched.call_count, 1)
        evidence = report["field_catalog"]
        self.assertEqual(evidence["required_fields"]["ids"], ["sample_matrix"])
        self.assertEqual(evidence["verification"], "required_fields_only")
        self.assertFalse(evidence["coverage"]["complete"])
        self.assertFalse(evidence["coverage"]["full_scope"])
        self.assertTrue(report["field_contracts"][0]["fields"])
        self.assertEqual(Path(path).read_bytes(), before)

    def test_stale_routes_are_hints_and_never_replace_live_required_evidence(self):
        path = self.capture()
        payload = cache_store.read_entry(path)["payload"]
        cache_store.write_json_atomic(path, cache_store.build_entry(
            "scoped-capabilities", payload, ttl=3600,
            now=cache_store.utc_now() - timedelta(hours=2)))
        for flags in ({"hit": True}, {"stale": True}, {"degraded": True},
                      {"error": "RATE_LIMITED"}, {"complete": False}):
            with self.subTest(flags=flags):
                def fields(session, **kwargs):
                    kwargs["report"].update({"complete": True, **flags})
                    return FIELDS
                with mock.patch.object(ml, "get_datafields", side_effect=fields), \
                        mock.patch.object(ml, "get_datasets", side_effect=AssertionError("discovery")):
                    result = capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"})
                self.assertEqual(result["evidence"]["state"], "unknown")
                self.assertEqual(result["evidence"]["required_fields"]["unverified_ids"], ["sample_matrix"])
                self.assertIsNone(result["known_fields"])

    def test_discovery_failure_in_unused_dataset_keeps_required_live_proof(self):
        def datasets(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"id": "sample_dataset"}, {"id": "unrelated_news"}]

        def fields(session, **kwargs):
            kwargs["report"].update(complete=True)
            if kwargs["dataset_id"] == "unrelated_news":
                kwargs["report"].update(hit=True, stale=True, degraded=True, error="RATE_LIMITED")
                return [{"id": "news_field", "dataset": "unrelated_news", "type": "VECTOR"}]
            return FIELDS

        with mock.patch.object(ml, "get_datasets", side_effect=datasets), \
                mock.patch.object(ml, "get_datafields", side_effect=fields):
            result = capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"})
        self.assertEqual(result["evidence"]["state"], "verified")
        self.assertEqual(result["evidence"]["freshness"]["state"], "unknown")
        self.assertFalse(result["evidence"]["coverage"]["complete"])
        self.assertNotIn("news_field", result["known_fields"])
        self.assertIsNone(validation.cached_field_scope(SETTINGS)["known_fields"])

    def test_unverified_used_dataset_blocks_preflight_despite_cached_field(self):
        self.capture()

        def fields(session, **kwargs):
            kwargs["report"].update(complete=True, stale=True, error="RATE_LIMITED")
            return FIELDS

        def operators(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"name": "rank"}]

        with mock.patch.object(ml, "get_datafields", side_effect=fields), \
                mock.patch.object(ml, "fetch_operators", side_effect=operators):
            with self.assertRaises(CliError) as caught:
                validation._preflight_lint(self.session, [("rank(sample_matrix)", 0)],
                                           SETTINGS, SimpleNamespace(allow_lint_errors=False))
        self.assertEqual(caught.exception.details["coverage"]["fields"], "unknown")

    def test_account_and_scope_switch_do_not_reuse_routes(self):
        self.capture()

        def datasets(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"id": "new_dataset"}]

        def fields(session, **kwargs):
            self.assertEqual(kwargs["dataset_id"], "new_dataset")
            kwargs["report"].update(complete=True)
            return [{"id": "sample_matrix", "dataset": "new_dataset", "type": "MATRIX"}]

        cases = [(SimpleNamespace(auth=("other-account", "placeholder")), SETTINGS),
                 (self.session, {**SETTINGS, "universe": "TOP500"})]
        for session, settings in cases:
            with self.subTest(settings=settings, account=session.auth[0]), \
                    mock.patch.object(ml, "get_datasets", side_effect=datasets) as discovery, \
                    mock.patch.object(ml, "get_datafields", side_effect=fields):
                result = capabilities.required_field_scope(session, settings, {"sample_matrix"})
            self.assertTrue(discovery.called)
            self.assertEqual(result["evidence"]["required_fields"]["routing_source"], "full_discovery")
            self.assertEqual(result["evidence"]["scope"], settings)

    def test_field_moving_from_cached_dataset_triggers_live_discovery(self):
        self.capture()

        def datasets(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"id": "new_dataset"}]

        def fields(session, **kwargs):
            kwargs["report"].update(complete=True)
            if kwargs["dataset_id"] == "sample_dataset":
                return []
            return [{"id": "sample_matrix", "dataset": "new_dataset", "type": "MATRIX"}]

        with mock.patch.object(ml, "get_datasets", side_effect=datasets) as discovery, \
                mock.patch.object(ml, "get_datafields", side_effect=fields):
            result = capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"})
        discovery.assert_called_once()
        self.assertEqual(result["evidence"]["state"], "verified")
        self.assertIn("routing_retry", result["evidence"])

    def test_unknown_id_in_partial_discovery_is_not_admitted_or_declared_absent(self):
        def datasets(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"id": "sample_dataset"}]

        def fields(session, **kwargs):
            kwargs["report"].update(complete=False)
            return FIELDS

        with mock.patch.object(ml, "get_datasets", side_effect=datasets), \
                mock.patch.object(ml, "get_datafields", side_effect=fields):
            result = capabilities.required_field_scope(self.session, SETTINGS, {"unknown_field"})
        self.assertEqual(result["evidence"]["state"], "unknown")
        self.assertEqual(result["evidence"]["required_fields"]["unverified_ids"], ["unknown_field"])
        self.assertIsNone(result["known_fields"])

    def test_new_field_discovery_replaces_cached_catalog_without_historical_gate(self):
        self.capture()

        def datasets(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"id": "new_dataset"}]

        def fields(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"id": "new_field", "type": "NEW_TYPE", "dataset": "new_dataset"}]

        with mock.patch.object(ml, "get_datasets", side_effect=datasets), \
                mock.patch.object(ml, "get_datafields", side_effect=fields):
            result = capabilities.required_field_scope(self.session, SETTINGS, {"new_field"})
        self.assertEqual(result["evidence"]["state"], "verified")
        self.assertEqual(result["field_types"], {"new_field": "NEW_TYPE"})
        self.assertEqual(validation.cached_field_scope(SETTINGS)["known_fields"], {"new_field"})
        self.assertNotIn("sample_matrix", result["known_fields"])
        for key in ("account", "scope", "source", "as_of", "age_seconds", "capability_id"):
            self.assertIsNotNone(result["evidence"][key])

    def test_complete_empty_discovery_replaces_removed_field_snapshot(self):
        self.capture()

        def datasets(session, **kwargs):
            kwargs["report"].update(complete=True)
            return []

        def fields(session, **kwargs):
            kwargs["report"].update(complete=True)
            return []

        with mock.patch.object(ml, "get_datasets", side_effect=datasets), \
                mock.patch.object(ml, "get_datafields", side_effect=fields):
            result = capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"})
        self.assertEqual(result["known_fields"], set())
        self.assertEqual(result["evidence"]["state"], "verified")
        self.assertEqual(result["evidence"]["required_fields"]["state"], "unknown")
        cached = validation.cached_field_scope(SETTINGS)
        self.assertEqual(cached["known_fields"], set())
        self.assertTrue(cached["evidence"]["coverage"]["complete"])

    def test_field_snapshot_matches_principal_despite_source_description_change(self):
        path = self.capture()
        payload = cache_store.read_entry(path)["payload"]
        payload["account"]["sources"] = ["other-confirmation-of-same-principal"]
        cache_store.write_json_atomic(path, cache_store.build_entry("scoped-capabilities", payload, ttl=3600))
        self.assertEqual(validation.cached_field_scope(SETTINGS)["evidence"]["state"], "verified")


class PreflightEvidenceTests(unittest.TestCase):
    def test_catalog_failure_blocks_before_submission(self):
        session = mock.Mock()
        with mock.patch.object(ml, "fetch_operators", side_effect=RuntimeError("metadata unavailable")), \
                mock.patch.object(field_index, "fetch_all_fields", side_effect=RuntimeError("metadata unavailable")), \
                mock.patch.object(simulation_capabilities, "account_provenance", return_value={"state": "unknown"}):
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
                mock.patch.object(simulation_capabilities, "account_provenance", return_value={"state": "unknown"}):
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
                mock.patch.object(simulation_capabilities, "account_provenance", return_value={"state": "unknown"}):
            args = SimpleNamespace(allow_lint_errors=False)
            with self.assertRaises(CliError):
                validation._preflight_lint(mock.Mock(), [("rank(sample_vector)", 0)], SETTINGS, args)
            report = validation._preflight_lint(mock.Mock(), [("ts_delta(rank(sample_matrix), 20)", 0)], SETTINGS, args)
        self.assertTrue(report["ok"])
        self.assertEqual(report["coverage"], {"operators": "verified", "fields": "verified"})
        self.assertEqual(report["field_contracts"][0]["state"], "unknown")
