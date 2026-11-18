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

    def test_preflight_searches_only_fields_extracted_from_ast(self):
        path = self.capture()
        before = Path(path).read_bytes()
        expression = "x = sample_matrix; if_else(true, rank(x), NaN) /* unrelated_news_field */"

        def operators(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"name": "rank"}, {"name": "if_else"}]

        with mock.patch.object(ml, "get_datasets", side_effect=AssertionError("full discovery")), \
                mock.patch.object(ml, "request_json", return_value={"results": FIELDS, "count": len(FIELDS)}) as fetched, \
                mock.patch.object(ml, "fetch_operators", side_effect=operators):
            report = validation._preflight_lint(
                self.session, [(expression, 0)],
                SETTINGS, SimpleNamespace(allow_lint_errors=False))
        self.assertTrue(report["ok"])
        fetched.assert_called_once()
        self.assertEqual(fetched.call_args.args, (self.session, "GET", "/data-fields"))
        self.assertEqual(fetched.call_args.kwargs["params"],
                         {**SETTINGS, "search": "sample_matrix", "limit": 50, "offset": 0})
        self.assertEqual(fetched.call_args.kwargs["max_retries"], 1)
        evidence = report["field_catalog"]
        self.assertEqual(evidence["required_fields"]["ids"], ["sample_matrix"])
        self.assertEqual(evidence["verification"], "required_fields_only")
        self.assertFalse(evidence["coverage"]["complete"])
        self.assertFalse(evidence["coverage"]["full_scope"])
        self.assertTrue(report["field_contracts"][0]["fields"])
        self.assertEqual(Path(path).read_bytes(), before)

    def test_search_miss_ambiguity_or_scope_mismatch_cannot_reuse_captured_field(self):
        self.capture()
        requested = FIELDS[0]
        cases = [([{"id": "sample_matrix_adj"}], "no_exact_match"),
                 ([requested, {**requested, "type": "VECTOR"}], "ambiguous_exact_match"),
                 ([{**requested, "region": "AMR"}], "scope_mismatch")]
        for rows, reason in cases:
            with self.subTest(reason=reason), \
                    mock.patch.object(ml, "request_json", return_value={"results": rows, "count": 1000}), \
                    mock.patch.object(ml, "get_datasets", side_effect=AssertionError("full discovery")):
                result = capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"})
            evidence = result["evidence"]
            self.assertEqual(evidence["state"], "unknown")
            self.assertEqual(evidence["catalogs"][0]["reason"], reason)
            self.assertEqual(evidence["required_fields"]["unverified_ids"], ["sample_matrix"])
            self.assertIsNone(result["known_fields"])
        self.assertIn("sample_matrix", validation.cached_field_scope(SETTINGS)["known_fields"])

    def test_preflight_reuses_verified_proofs_across_decay_groups_only_within_scope(self):
        def operators(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"name": "rank"}]

        args = SimpleNamespace(allow_lint_errors=False)
        cache = {}
        with mock.patch.object(ml, "fetch_operators", side_effect=operators) as operator_fetch, \
                mock.patch.object(ml, "request_json", return_value={"results": FIELDS, "count": len(FIELDS)}) as field_fetch:
            first = validation._preflight_lint(self.session, [("rank(sample_matrix)", 0)], SETTINGS, args,
                                               proof_cache=cache)
            second = validation._preflight_lint(self.session, [("rank(sample_matrix)", 6)], SETTINGS, args,
                                                proof_cache=cache)
            validation._preflight_lint(self.session, [("rank(sample_matrix)", 6)],
                                       {**SETTINGS, "universe": "TOP500"}, args, proof_cache=cache)
            validation._preflight_lint(SimpleNamespace(auth=("other-account", "unused")),
                                       [("rank(sample_matrix)", 6)], SETTINGS, args, proof_cache=cache)
        self.assertEqual(operator_fetch.call_count, 2)
        self.assertEqual(field_fetch.call_count, 3)
        self.assertTrue(second["operator_catalog"]["reused_in_invocation"])
        self.assertTrue(second["field_catalog"]["catalogs"][0]["reused_in_invocation"])
        self.assertEqual(first["field_catalog"]["as_of"], second["field_catalog"]["as_of"])

    def test_rate_limited_search_stops_without_preflight_approval_or_full_scan(self):
        self.capture()

        def operators(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"name": "rank"}]

        with mock.patch.object(ml, "request_json", side_effect=ml.RateLimited("slow down")) as fetch, \
                mock.patch.object(ml, "get_datasets", side_effect=AssertionError("full discovery")), \
                mock.patch.object(ml, "fetch_operators", side_effect=operators):
            with self.assertRaises(CliError) as caught:
                validation._preflight_lint(self.session, [("rank(sample_matrix + sample_vector)", 0)],
                                           SETTINGS, SimpleNamespace(allow_lint_errors=False))
        fetch.assert_called_once()
        evidence = caught.exception.details["field_catalog"]
        self.assertEqual(evidence["state"], "unknown")
        self.assertEqual(evidence["catalogs"][0]["error"], "RATE_LIMITED")
        self.assertEqual(evidence["required_fields"]["unverified_ids"], ["sample_matrix", "sample_vector"])

    def test_preflight_does_not_read_fields_for_constants_or_invalid_syntax(self):
        def operators(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"name": "rank"}]

        args = SimpleNamespace(allow_lint_errors=False)
        with mock.patch.object(ml, "get_datasets", side_effect=AssertionError("full discovery")), \
                mock.patch.object(ml, "search_datafield", side_effect=AssertionError("field search")), \
                mock.patch.object(ml, "fetch_operators", side_effect=operators):
            report = validation._preflight_lint(self.session, [("rank(1)", 0)], SETTINGS, args)
            with self.assertRaises(CliError) as caught:
                validation._preflight_lint(self.session, [("rank(", 0)], SETTINGS, args)
        self.assertEqual(report["coverage"]["fields"], "not_applicable")
        self.assertEqual(caught.exception.details["coverage"]["fields"], "not_applicable")

    def test_account_and_scope_switch_use_current_session_and_scope(self):
        self.capture()
        cases = [(SimpleNamespace(auth=("other-account", "placeholder")), SETTINGS),
                 (self.session, {**SETTINGS, "universe": "TOP500"})]
        for session, settings in cases:
            with self.subTest(settings=settings, is_other_session=session is not self.session), \
                    mock.patch.object(ml, "request_json", return_value={"results": FIELDS, "count": len(FIELDS)}) as fetch:
                result = capabilities.required_field_scope(session, settings, {"sample_matrix"})
            self.assertIs(fetch.call_args.args[0], session)
            params = fetch.call_args.kwargs["params"]
            self.assertEqual({key: params[key] for key in SETTINGS}, settings)
            self.assertEqual(result["evidence"]["scope"], settings)
            self.assertEqual(result["evidence"]["account"], simulation_capabilities.account_provenance(session))

    def test_new_exact_fields_use_one_live_page_each_and_keep_full_snapshot(self):
        snapshot_path = self.capture()
        before = Path(snapshot_path).read_bytes()
        new_fields = {"total_visit_count_today", "employee_inflow_count"}

        def search(session, method, path, *, params, **kwargs):
            field = params["search"]
            return {"results": [{"id": field, "type": "MATRIX", "dataset": "new_dataset", **SETTINGS},
                                {"id": field + "_fuzzy", "type": "VECTOR"}], "count": 4800}

        with mock.patch.object(ml, "request_json", side_effect=search) as fetch, \
                mock.patch.object(ml, "get_datasets", side_effect=AssertionError("full discovery")):
            result = capabilities.required_field_scope(self.session, SETTINGS, new_fields)
        self.assertEqual([call.kwargs["params"]["search"] for call in fetch.call_args_list], sorted(new_fields))
        self.assertEqual(result["evidence"]["state"], "verified")
        self.assertEqual(result["known_fields"], new_fields)
        self.assertFalse(result["evidence"]["coverage"]["complete"])
        self.assertTrue(all(not row["complete"] for row in result["evidence"]["catalogs"]))
        self.assertEqual(Path(snapshot_path).read_bytes(), before)
        for key in ("account", "scope", "source", "as_of", "age_seconds", "capability_id"):
            self.assertIsNotNone(result["evidence"][key])

    def test_required_field_search_honors_per_scope_budget(self):
        field = FIELDS[0]
        matches = {"matches": [field], "returned_count": 1, "reported_count": 1, "complete": True}
        with mock.patch.object(capabilities.time, "monotonic", side_effect=(0, 24, 31)), \
                mock.patch.object(ml, "search_datafield", return_value=matches) as search:
            result = capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix", "sample_vector"})
        search.assert_called_once_with(self.session, "sample_matrix", SETTINGS, deadline=30)
        evidence = result["evidence"]
        self.assertEqual(evidence["state"], "unknown")
        self.assertEqual(evidence["required_fields"]["verified_ids"], ["sample_matrix"])
        self.assertEqual(evidence["required_fields"]["unverified_ids"], ["sample_vector"])

    def test_explicit_complete_empty_capture_replaces_removed_field_snapshot(self):
        self.capture()

        def datasets(session, **kwargs):
            kwargs["report"].update(complete=True)
            return []

        with mock.patch.object(ml, "get_datasets", side_effect=datasets):
            report = {}
            self.assertEqual(capabilities.fetch_fields(self.session, report=report), [])
        self.assertEqual(report["state"], "verified")
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
                mock.patch.object(ml, "request_json", side_effect=ml.ServiceError("metadata unavailable")), \
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
        with mock.patch.object(ml, "fetch_operators", return_value=[{"name": "rank"}]), \
                mock.patch.object(ml, "request_json", return_value={"results": [], "count": 0}), \
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

        with mock.patch.object(ml, "fetch_operators", side_effect=operators), \
                mock.patch.object(ml, "request_json", return_value={"results": FIELDS, "count": len(FIELDS)}), \
                mock.patch.object(simulation_capabilities, "account_provenance", return_value={"state": "unknown"}):
            args = SimpleNamespace(allow_lint_errors=False)
            with self.assertRaises(CliError):
                validation._preflight_lint(mock.Mock(), [("rank(sample_vector)", 0)], SETTINGS, args)
            report = validation._preflight_lint(mock.Mock(), [("ts_delta(rank(sample_matrix), 20)", 0)], SETTINGS, args)
        self.assertTrue(report["ok"])
        self.assertEqual(report["coverage"], {"operators": "verified", "fields": "verified"})
        self.assertEqual(report["field_contracts"][0]["state"], "unknown")
