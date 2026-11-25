"""Offline synthetic checks for scoped metadata, type order and economic evidence."""

import copy
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

    def test_verified_field_proof_is_reused_across_preflight_calls(self):
        self.session._brain_account_id = "synthetic-account-id"
        field = "sample_matrix"
        matches = {"matches": [FIELDS[0]], "returned_count": 1, "reported_count": 1, "complete": True}
        with mock.patch.object(ml, "search_datafield", return_value=matches) as search:
            first = capabilities.required_field_scope(self.session, SETTINGS, {field})
            second = capabilities.required_field_scope(self.session, SETTINGS, {field})
        search.assert_called_once()
        self.assertEqual(first["evidence"]["state"], "verified")
        self.assertEqual(second["evidence"]["state"], "verified")
        self.assertTrue(second["evidence"]["catalogs"][0]["reused_from_cache"])

    def test_field_proof_cache_never_stores_misses(self):
        self.session._brain_account_id = "synthetic-account-id"
        misses = {"matches": [], "returned_count": 0, "reported_count": 4000, "complete": False}
        with mock.patch.object(ml, "search_datafield", return_value=misses) as search:
            first = capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"})
            second = capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"})
        self.assertEqual(first["evidence"]["state"], "unknown")
        self.assertEqual(second["evidence"]["state"], "unknown")
        self.assertEqual(search.call_count, 2)

    def test_expired_field_proof_is_researched(self):
        self.session._brain_account_id = "synthetic-account-id"
        matches = {"matches": [FIELDS[0]], "returned_count": 1, "reported_count": 1, "complete": True}
        with mock.patch.object(ml, "search_datafield", return_value=matches) as search:
            capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"})
            later = cache_store.utc_now() + timedelta(hours=2)
            with mock.patch.object(cache_store, "utc_now", return_value=later):
                capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"})
        self.assertEqual(search.call_count, 2)

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
        self.assertEqual(evidence["required_fields"]["budget_deferred_ids"], ["sample_vector"])

    def test_wide_field_group_can_use_more_than_the_default_scope_budget(self):
        required = {f"field_{index:02}" for index in range(36)}

        def lookup(_session, field, _scope, *, deadline):
            self.assertLessEqual(deadline, 45)
            return {"matches": [{"id": field}], "returned_count": 1, "reported_count": 1, "complete": True}

        with mock.patch.object(capabilities.time, "monotonic", side_effect=[0, *range(36)]), \
                mock.patch.object(ml, "search_datafield", side_effect=lookup) as search:
            result = capabilities.required_field_scope(self.session, SETTINGS, required, deadline=45)

        self.assertEqual(search.call_count, len(required))
        self.assertEqual(result["known_fields"], required)

    def test_required_field_search_paces_adjacent_requests(self):
        fields = {row["id"]: row for row in FIELDS}

        def lookup(session, field, scope, *, deadline):
            return {"matches": [fields[field]], "returned_count": 1, "reported_count": 1, "complete": True}

        with mock.patch.object(capabilities.time, "monotonic", side_effect=(0, 0, 0.25)), \
                mock.patch.object(capabilities.time, "sleep") as sleep, \
                mock.patch.object(ml, "search_datafield", side_effect=lookup) as search:
            result = capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix", "sample_vector"})

        sleep.assert_called_once_with(0.75)
        self.assertEqual([item.kwargs["deadline"] for item in search.call_args_list], [30.0, 30.0])
        self.assertEqual(result["evidence"]["state"], "verified")

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


class RequiredFieldProofTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.session = SimpleNamespace(auth=("synthetic-account", "unused"))
        self.policy = {"enabled": True, "max_age": 3600, "refresh": False}
        for patch in (
            mock.patch.object(ml.research_paths, "metadata_root", return_value=str(self.root / "metadata")),
            mock.patch.object(ml, "session_cache_path", return_value=self.root / "session.json"),
            mock.patch.object(ml, "metadata_cache_policy", return_value=self.policy),
            mock.patch.object(ml, "request_json", side_effect=AssertionError("unexpected network")),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def lookup(self, _session, field, _scope, *, deadline):
        # A positive exact match does not require a complete fuzzy search page.
        return {"matches": [{"id": field, "type": "MATRIX", "dataset": "sample_dataset"}],
                "returned_count": 1, "reported_count": 1000, "complete": False}

    def proof_path(self, field="sample_matrix"):
        account = simulation_capabilities.account_provenance(self.session)
        return capabilities._field_proof_path(account, SETTINGS, field)

    def prove(self, fields=("sample_matrix",), **kwargs):
        return capabilities.required_field_scope(self.session, SETTINGS, fields, proof_cache={}, **kwargs)

    def test_saved_exact_proof_survives_fresh_invocation_without_retiming(self):
        observed = cache_store.utc_now()
        with mock.patch.object(cache_store, "utc_now", return_value=observed), \
                mock.patch.object(ml, "search_datafield", side_effect=self.lookup):
            first = self.prove()
        before = self.proof_path().read_bytes()
        with mock.patch.object(cache_store, "utc_now", return_value=observed + timedelta(minutes=20)), \
                mock.patch.object(ml, "search_datafield", side_effect=AssertionError("cached exact proof")):
            second = self.prove()
        self.assertEqual(second["known_fields"], {"sample_matrix"})
        self.assertEqual(second["evidence"]["as_of"], first["evidence"]["as_of"])
        self.assertEqual(second["evidence"]["age_seconds"], 1200)
        self.assertEqual(second["evidence"]["source"], "private_field_proof")
        self.assertTrue(second["evidence"]["catalogs"][0]["reused_from_cache"])
        self.assertFalse(second["evidence"]["coverage"]["complete"])
        self.assertFalse(second["evidence"]["coverage"]["full_scope"])
        self.assertEqual(self.proof_path().read_bytes(), before)

    def test_shared_memory_cache_cannot_extend_expiry_or_override_refresh(self):
        observed = cache_store.utc_now()
        cache = {}
        self.policy["max_age"] = 1
        with mock.patch.object(cache_store, "utc_now", return_value=observed), \
                mock.patch.object(ml, "search_datafield", side_effect=self.lookup):
            capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"}, proof_cache=cache)
        with mock.patch.object(cache_store, "utc_now", return_value=observed + timedelta(seconds=2)), \
                mock.patch.object(ml, "search_datafield", side_effect=ml.ServiceError("unavailable")) as expired:
            result = capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"}, proof_cache=cache)
        expired.assert_called_once()
        self.assertEqual(result["evidence"]["state"], "unknown")
        self.policy["max_age"] = 3600
        with mock.patch.object(ml, "search_datafield", side_effect=self.lookup):
            capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"}, proof_cache=cache)
        self.policy["refresh"] = True
        with mock.patch.object(ml, "search_datafield", side_effect=ml.ServiceError("unavailable")) as refreshed:
            result = capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"}, proof_cache=cache)
        refreshed.assert_called_once()
        self.assertEqual(result["evidence"]["state"], "unknown")
        self.assertFalse(self.proof_path().exists())

    def test_live_refresh_proof_is_reusable_across_groups_in_same_invocation(self):
        cache = {}
        self.policy["refresh"] = True
        with mock.patch.object(ml, "search_datafield", side_effect=self.lookup) as search:
            first = capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"}, proof_cache=cache)
            second = capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"}, proof_cache=cache)
        search.assert_called_once()
        self.assertTrue(second["evidence"]["catalogs"][0]["reused_in_invocation"])
        self.assertEqual(first["evidence"]["as_of"], second["evidence"]["as_of"])

    def test_proof_expiring_during_later_lookup_defers_preflight_without_retry_loop(self):
        elapsed = [0.0]
        started = cache_store.utc_now()
        self.policy["max_age"] = 60

        def operators(_session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"name": "rank"}]

        def slow_lookup(session, field, scope, *, deadline):
            self.assertEqual(field, "sample_vector")
            elapsed[0] += 27.4
            return self.lookup(session, field, scope, deadline=deadline)

        with mock.patch.object(capabilities.time, "monotonic", side_effect=lambda: elapsed[0]), \
                mock.patch.object(cache_store, "utc_now", side_effect=lambda: started + timedelta(seconds=elapsed[0])):
            with mock.patch.object(ml, "search_datafield", side_effect=self.lookup):
                first = self.prove()
            elapsed[0] = 50.0
            with mock.patch.object(ml, "search_datafield", side_effect=slow_lookup) as search, \
                    mock.patch.object(ml, "fetch_operators", side_effect=operators), \
                    self.assertRaises(validation.FieldEvidenceRefreshError) as caught:
                validation._preflight_lint(
                    self.session, [("rank(sample_matrix + sample_vector)", 0)], SETTINGS,
                    SimpleNamespace(allow_lint_errors=False), deadline=90, proof_cache={})
        search.assert_called_once()
        self.assertEqual(caught.exception.code, "SERVICE_ERROR")
        evidence = caught.exception.details["field_catalog"]
        self.assertEqual(evidence["state"], "unknown")
        self.assertEqual(evidence["required_fields"]["verified_ids"], ["sample_vector"])
        self.assertEqual(evidence["required_fields"]["expired_ids"], ["sample_matrix"])
        expired = evidence["catalogs"][0]
        self.assertEqual(expired["state"], "unknown")
        self.assertEqual(expired["reason"], "proof_expired_during_preflight")
        self.assertEqual(expired["as_of"], first["evidence"]["as_of"])
        self.assertAlmostEqual(expired["age_seconds"], 77.4)

    def test_increasing_maximum_age_cannot_extend_original_memory_proof_expiry(self):
        observed = cache_store.utc_now()
        cache = {}
        self.policy["max_age"] = 10
        with mock.patch.object(cache_store, "utc_now", return_value=observed), \
                mock.patch.object(ml, "search_datafield", side_effect=self.lookup):
            capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"}, proof_cache=cache)
        self.policy["max_age"] = 3600
        with mock.patch.object(cache_store, "utc_now", return_value=observed + timedelta(seconds=11)), \
                mock.patch.object(ml, "search_datafield", side_effect=ml.ServiceError("unavailable")) as search:
            result = capabilities.required_field_scope(self.session, SETTINGS, {"sample_matrix"}, proof_cache=cache)
        search.assert_called_once()
        self.assertEqual(result["evidence"]["state"], "unknown")
        self.assertIsNone(result["known_fields"])

    def test_slow_search_uses_caller_budget_and_saves_partial_progress(self):
        elapsed = [0.0]
        started = cache_store.utc_now()

        def lookup(session, field, scope, *, deadline):
            remaining = deadline - elapsed[0]
            if remaining < 27.4:
                elapsed[0] = deadline
                raise ml.ServiceError("response exceeded remaining wait budget")
            elapsed[0] += 27.4
            return self.lookup(session, field, scope, deadline=deadline)

        with mock.patch.object(capabilities.time, "monotonic", side_effect=lambda: elapsed[0]), \
                mock.patch.object(cache_store, "utc_now", side_effect=lambda: started + timedelta(seconds=elapsed[0])), \
                mock.patch.object(ml, "search_datafield", side_effect=lookup) as search:
            first = self.prove(("sample_matrix", "sample_vector"))
            before = self.proof_path().read_bytes()
            second = self.prove(("sample_matrix", "sample_vector"), deadline=90)
        self.assertEqual(first["evidence"]["state"], "unknown")
        self.assertEqual(first["evidence"]["required_fields"]["verified_ids"], ["sample_matrix"])
        self.assertEqual([call.args[1] for call in search.call_args_list],
                         ["sample_matrix", "sample_vector", "sample_vector"])
        self.assertEqual([call.kwargs["deadline"] for call in search.call_args_list], [30, 30, 90])
        self.assertEqual(second["known_fields"], {"sample_matrix", "sample_vector"})
        self.assertEqual(second["evidence"]["as_of"], first["evidence"]["as_of"])
        self.assertAlmostEqual(second["evidence"]["age_seconds"], 30.0)
        self.assertEqual(self.proof_path().read_bytes(), before)

    def test_account_and_every_scope_component_prevent_cross_scope_reuse(self):
        with mock.patch.object(ml, "search_datafield", side_effect=self.lookup):
            self.prove()
        cases = [(SimpleNamespace(auth=("other-account", "unused")), SETTINGS),
                 (SimpleNamespace(), SETTINGS)]
        cases.extend((self.session, {**SETTINGS, key: value}) for key, value in (
            ("region", "EUR"), ("universe", "TOP1000"), ("delay", 0), ("instrumentType", "OTHER")))
        for session, settings in cases:
            with self.subTest(settings=settings, account=getattr(session, "auth", None)), \
                    mock.patch.object(ml, "search_datafield", side_effect=self.lookup) as search:
                result = capabilities.required_field_scope(session, settings, {"sample_matrix"}, proof_cache={})
            search.assert_called_once()
            self.assertNotIn("reused_from_cache", result["evidence"]["catalogs"][0])

    def test_expiry_and_stricter_maximum_age_reject_saved_proofs(self):
        observed = cache_store.utc_now()
        with mock.patch.object(cache_store, "utc_now", return_value=observed), \
                mock.patch.object(ml, "search_datafield", side_effect=self.lookup):
            self.prove()
        for age, maximum in ((3601, 7200), (120, 60)):
            self.policy["max_age"] = maximum
            with self.subTest(age=age, maximum=maximum), \
                    mock.patch.object(cache_store, "utc_now", return_value=observed + timedelta(seconds=age)), \
                    mock.patch.object(ml, "search_datafield", side_effect=ml.ServiceError("unavailable")) as search:
                result = self.prove()
            search.assert_called_once()
            self.assertEqual(result["evidence"]["state"], "unknown")

    def test_disabled_or_zero_age_cache_neither_reuses_nor_renews_proofs(self):
        with mock.patch.object(ml, "search_datafield", side_effect=self.lookup):
            self.prove()
        before = self.proof_path().read_bytes()
        for enabled, maximum in ((False, 3600), (True, 0)):
            self.policy.update(enabled=enabled, max_age=maximum)
            with self.subTest(enabled=enabled, maximum=maximum), \
                    mock.patch.object(ml, "search_datafield", side_effect=self.lookup) as search, \
                    mock.patch.object(cache_store, "write_json_atomic", side_effect=AssertionError("disabled cache")):
                result = self.prove()
            search.assert_called_once()
            self.assertEqual(result["evidence"]["state"], "verified")
            self.assertEqual(self.proof_path().read_bytes(), before)

    def test_refresh_failure_cannot_resurrect_an_earlier_positive_proof(self):
        failures = [ml.ServiceError("unavailable"),
                    {"matches": [], "returned_count": 0, "reported_count": 0, "complete": True},
                    {"matches": [FIELDS[0], FIELDS[0]], "returned_count": 2, "reported_count": 2, "complete": True},
                    {"matches": [{**FIELDS[0], "region": "EUR"}], "returned_count": 1,
                     "reported_count": 1, "complete": True}]
        for failure in failures:
            self.policy["refresh"] = False
            with mock.patch.object(ml, "search_datafield", side_effect=self.lookup):
                self.prove()
            self.policy["refresh"] = True
            with self.subTest(failure=failure), \
                    mock.patch.object(ml, "search_datafield", side_effect=[failure]) as refresh:
                result = self.prove()
            refresh.assert_called_once()
            self.assertEqual(result["evidence"]["state"], "unknown")
            self.assertFalse(self.proof_path().exists())
            self.policy["refresh"] = False
            with mock.patch.object(ml, "search_datafield", side_effect=ml.ServiceError("still unavailable")) as search:
                self.assertEqual(self.prove()["evidence"]["state"], "unknown")
            search.assert_called_once()

    def test_deferred_refresh_does_not_destroy_unattempted_proof(self):
        with mock.patch.object(ml, "search_datafield", side_effect=self.lookup):
            self.prove()
        before = self.proof_path().read_bytes()
        self.policy["refresh"] = True
        with mock.patch.object(capabilities.time, "monotonic", return_value=10), \
                mock.patch.object(ml, "search_datafield", side_effect=AssertionError("past deadline")):
            result = self.prove(deadline=10)
        self.assertEqual(result["evidence"]["required_fields"]["budget_deferred_ids"], ["sample_matrix"])
        self.assertEqual(self.proof_path().read_bytes(), before)

    def test_malformed_or_degraded_proofs_do_not_authorize_fields(self):
        with mock.patch.object(ml, "search_datafield", side_effect=self.lookup):
            self.prove()
        original, error = cache_store.load_json(self.proof_path())
        self.assertIsNone(error)
        changes = [(("_cache", "source"), "untrusted"),
                   (("payload", "version"), 999),
                   (("payload", "account", "fingerprint"), "other-account"),
                   (("payload", "scope", "delay"), 0),
                   (("payload", "field"), "other_field"),
                   (("payload", "row", "id"), "other_field"),
                   (("payload", "row", "universe"), "TOP500"),
                   (("payload", "search", "state"), "unknown"),
                   (("payload", "search", "exact_count"), 2),
                   (("payload", "search", "degraded"), True),
                   (("payload", "search", "stale"), True),
                   (("payload", "search", "as_of"), "invalid")]
        for keys, value in changes:
            mutated = copy.deepcopy(original)
            target = mutated
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            cache_store.write_json_atomic(self.proof_path(), mutated)
            with self.subTest(keys=keys), \
                    mock.patch.object(ml, "search_datafield", side_effect=ml.ServiceError("unavailable")) as search:
                result = self.prove()
            search.assert_called_once()
            self.assertEqual(result["evidence"]["state"], "unknown")
            self.assertIsNone(result["known_fields"])

    def test_cache_write_failure_keeps_live_proof_but_creates_no_durable_success(self):
        with mock.patch.object(cache_store, "write_json_atomic", side_effect=OSError("read-only cache")), \
                mock.patch.object(ml, "search_datafield", side_effect=self.lookup):
            live = self.prove()
        self.assertEqual(live["evidence"]["state"], "verified")
        self.assertEqual(live["evidence"]["catalogs"][0]["cache_error"], "OSError")
        self.assertFalse(self.proof_path().exists())
        with mock.patch.object(ml, "search_datafield", side_effect=ml.ServiceError("unavailable")) as search:
            self.assertEqual(self.prove()["evidence"]["state"], "unknown")
        search.assert_called_once()

    def test_cache_read_failure_cannot_supply_a_proof(self):
        with mock.patch.object(ml, "search_datafield", side_effect=self.lookup):
            self.prove()
        path = self.proof_path()
        read = cache_store.read_entry

        def unreadable(candidate, **kwargs):
            if Path(candidate) == path:
                raise OSError("unreadable field proof")
            return read(candidate, **kwargs)

        with mock.patch.object(cache_store, "read_entry", side_effect=unreadable), \
                mock.patch.object(ml, "search_datafield", side_effect=ml.ServiceError("unavailable")) as search:
            self.assertEqual(self.prove()["evidence"]["state"], "unknown")
        search.assert_called_once()

    def test_failed_invalidation_blocks_refresh_before_an_old_proof_can_be_contradicted(self):
        with mock.patch.object(ml, "search_datafield", side_effect=self.lookup):
            self.prove()
        before = self.proof_path().read_bytes()
        self.policy["refresh"] = True
        with mock.patch.object(cache_store, "remove", side_effect=OSError("read-only cache")), \
                mock.patch.object(ml, "search_datafield", side_effect=AssertionError("invalidation failed")):
            result = self.prove()
        self.assertEqual(result["evidence"]["state"], "unknown")
        self.assertEqual(result["evidence"]["catalogs"][0]["error"], "SERVICE_ERROR")
        self.assertEqual(result["evidence"]["catalogs"][0]["cache_operation"], "invalidate")
        self.assertEqual(self.proof_path().read_bytes(), before)


class PreflightEvidenceTests(unittest.TestCase):
    def test_field_budget_exhaustion_reports_a_retryable_error(self):
        session = mock.Mock()
        fields = capabilities.field_scope([], {"state": "unknown", "scope": SETTINGS,
            "required_fields": {"budget_deferred_ids": ["sample_matrix"]}})

        def operators(_session, **kwargs):
            kwargs["report"]["complete"] = True
            return [{"name": "rank"}]

        with mock.patch.object(validation, "_field_scope", return_value=fields), \
                mock.patch.object(ml, "fetch_operators", side_effect=operators):
            with self.assertRaises(validation.FieldEvidenceRefreshError) as caught:
                validation._preflight_lint(session, [("rank(sample_matrix)", 0)],
                                           SETTINGS, SimpleNamespace())
        self.assertEqual(caught.exception.code, "SERVICE_ERROR")
        self.assertIn("budget expired", caught.exception.message)
        session.post.assert_not_called()

    def test_field_rate_limit_has_retryable_code_before_submission(self):
        session = mock.Mock()

        def operators(_session, **kwargs):
            kwargs["report"]["complete"] = True
            return [{"name": "rank"}]

        session.request.return_value.status_code = 429
        session.request.return_value.headers = {"Retry-After": "150"}
        with mock.patch.object(ml, "fetch_operators", side_effect=operators), \
                mock.patch.object(capabilities.time, "time", return_value=100), \
                mock.patch.object(simulation_capabilities, "account_provenance",
                                  return_value={"state": "unknown"}):
            with self.assertRaises(validation.FieldEvidenceRefreshError) as caught:
                validation._preflight_lint(session, [("rank(sample_matrix)", 0)],
                                           SETTINGS, SimpleNamespace())
        self.assertEqual(caught.exception.code, "RATE_LIMITED")
        self.assertEqual(caught.exception.details["field_catalog"]["catalogs"][0]["retry_at"], 250)
        session.request.assert_called_once()
        session.post.assert_not_called()

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
