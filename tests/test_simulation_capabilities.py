"""Synthetic native capability expansion, contraction, identity and cache proofs."""

import copy
import json
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cache_store
import machine_lib as ml
import simulation_capabilities as capabilities
from research_app import validation


def options(schema):
    return {"actions": {"POST": {"settings": {"children": schema}}}}


def native_schema():
    return {
        "instrumentType": {"type": "choice", "choices": ["EQUITY", "FUTURE"]},
        "region": {"choices": {"instrumentType": {"FUTURE": ["NEWREGION"]}}},
        "delay": {"type": "integer", "min_value": 0, "choices": {"region": {"NEWREGION": [5]}}},
        "universe": {"choices": {"region": {"NEWREGION": {"delay": {"5": ["NEWUNIVERSE"]}}}}},
        "newSetting": {"type": "integer", "min_value": 2, "max_value": 9},
    }


SETTINGS = {"instrumentType": "FUTURE", "region": "NEWREGION", "delay": 5,
            "universe": "NEWUNIVERSE", "newSetting": 7}


class SimulationCapabilityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.session = SimpleNamespace(auth=("synthetic-account", "not-a-real-password"))
        for patch in (
            mock.patch.dict(os.environ, {"BRAIN_USERNAME": "synthetic-account"}),
            mock.patch.object(ml, "session_cache_path", return_value=self.root / "session.json"),
            mock.patch.object(ml.research_paths, "metadata_root", return_value=str(self.root / "metadata")),
            mock.patch.object(ml, "metadata_cache_policy", return_value={"enabled": True, "max_age": 3600}),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def fetch(self, schema=None):
        with mock.patch.object(ml, "request_json", return_value=options(native_schema() if schema is None else schema)):
            return capabilities.fetch(self.session, refresh=True)

    def test_expanded_native_context_and_new_setting_are_supported(self):
        snapshot = self.fetch()
        settings = copy.deepcopy(SETTINGS)
        evidence = capabilities.settings_evidence(settings, snapshot)
        self.assertEqual(evidence["state"], "verified")
        self.assertEqual(set(evidence["checked"]), set(settings))
        self.assertEqual(settings, SETTINGS)
        self.assertEqual(snapshot["concurrency"], {"state": "unknown", "limit": None, "source": None})
        for key in ("account", "scope", "source", "as_of", "age_seconds", "capability_id"):
            self.assertIsNotNone(snapshot[key])

    def test_contracted_choices_and_bounds_produce_authoritative_negatives(self):
        before = self.fetch()
        schema = native_schema()
        schema["newSetting"]["max_value"] = 3
        schema["universe"]["choices"]["region"]["NEWREGION"]["delay"]["5"] = []
        after = self.fetch(schema)
        evidence = capabilities.settings_evidence(SETTINGS, after)
        self.assertEqual(evidence["state"], "unsupported")
        self.assertEqual({row["setting"] for row in evidence["unsupported"]}, {"newSetting", "universe"})
        self.assertNotEqual(before["capability_id"], after["capability_id"])
        self.assertEqual(capabilities.cached()["capability_id"], after["capability_id"])

    def test_removed_metadata_and_unpublished_extensions_are_unknown(self):
        schema = native_schema()
        del schema["newSetting"]
        settings = {**SETTINGS, "futureOption": None}
        evidence = capabilities.settings_evidence(settings, self.fetch(schema))
        self.assertEqual(evidence["state"], "unknown")
        self.assertEqual(evidence["unsupported"], [])
        self.assertEqual({row["setting"] for row in evidence["unknown"]}, {"newSetting", "futureOption"})
        self.assertIn("futureOption", settings)
        self.assertIsNone(settings["futureOption"])

    def test_dependent_choices_distinguish_empty_unknown_and_malformed(self):
        spec = native_schema()["universe"]
        self.assertEqual(capabilities.choice_values(spec, SETTINGS), ["NEWUNIVERSE"])
        self.assertIsNone(capabilities.choice_values(spec, {"region": "NEWREGION"}))
        self.assertIsNone(capabilities.choice_values(spec, {**SETTINGS, "delay": 6}))
        self.assertEqual(capabilities.choice_values({"choices": []}, {}), [])
        self.assertIsNone(capabilities.choice_values({"choices": [{"label": "missing value"}]}, {}))
        self.assertIsNone(capabilities.choice_values({}, {}))

    def test_unresolved_choices_cannot_be_verified_by_type_alone(self):
        snapshot = self.fetch({"newSetting": {"type": "integer", "choices": {"missingContext": {"x": [7]}}}})
        evidence = capabilities.settings_evidence({"newSetting": 7}, snapshot)
        self.assertEqual(evidence["state"], "unknown")
        self.assertEqual(evidence["unsupported"], [])

    def test_published_types_nullability_and_required_settings(self):
        snapshot = self.fetch({"newSetting": {"type": "integer", "allow_null": False},
                               "requiredExtension": {"type": "string", "required": True}})
        for value, reason in ((None, "null_not_allowed"), (True, "type_mismatch"), ("7", "type_mismatch")):
            with self.subTest(value=value):
                report = capabilities.settings_evidence({"newSetting": value}, snapshot)
                self.assertIn(reason, report["unsupported"][0]["reasons"])
                self.assertEqual(report["unsupported"][1]["reasons"], ["required_setting_missing"])

    def test_cache_reuse_is_offline_and_bound_to_account(self):
        live = self.fetch()
        with mock.patch.object(ml, "request_json", side_effect=AssertionError("network")), \
                mock.patch.object(ml, "load_session", side_effect=AssertionError("login")):
            cached = capabilities.cached()
            fetched = capabilities.fetch(self.session)
        self.assertEqual(cached["state"], "verified")
        self.assertEqual(cached["source"], "private_snapshot")
        self.assertEqual(fetched["capability_id"], live["capability_id"])
        with mock.patch.dict(os.environ, {"BRAIN_USERNAME": "other-account"}):
            self.assertEqual(capabilities.cached()["state"], "unknown")
        with mock.patch.dict(os.environ, {"BRAIN_USERNAME": ""}):
            self.assertEqual(capabilities.cached()["reasons"], ["account_unknown"])

    def test_stale_and_maximum_age_evidence_cannot_block_settings(self):
        live = self.fetch()
        path = ml.metadata_cache_path("simulation-capabilities", {
            "account": live["account"]["fingerprint"], "scope": live["scope"]})
        payload = cache_store.read_entry(path)["payload"]
        cache_store.write_json_atomic(path, cache_store.build_entry(
            "simulation-capabilities", payload, ttl=3600,
            now=cache_store.utc_now() - timedelta(minutes=2)))
        cached = capabilities.cached(max_age_seconds=60)
        self.assertEqual(cached["state"], "unknown")
        self.assertEqual(cached["capability_id"], live["capability_id"])
        self.assertEqual(capabilities.settings_evidence({"delay": 99}, cached)["unsupported"], [])
        cache_store.write_json_atomic(path, cache_store.build_entry(
            "simulation-capabilities", payload, ttl=1,
            now=cache_store.utc_now() - timedelta(hours=2)))
        self.assertEqual(capabilities.cached()["state"], "unknown")

    def test_options_failure_does_not_reuse_previous_verified_authority(self):
        self.fetch()
        with mock.patch.object(ml, "request_json", side_effect=ml.RateLimited("synthetic")):
            snapshot = capabilities.fetch(self.session, refresh=True)
        self.assertEqual(snapshot["state"], "unknown")
        self.assertEqual(snapshot["error"]["code"], "RATE_LIMITED")
        with mock.patch.object(ml, "request_json", side_effect=ml.AuthError("synthetic")):
            with self.assertRaises(ml.AuthError):
                capabilities.fetch(self.session, refresh=True)

    def test_unpublished_concurrency_aliases_are_not_interpreted(self):
        response = options(native_schema())
        response.update(concurrency=20, maxConcurrentSimulations=30, permissions=["EXPERT"])
        with mock.patch.object(ml, "request_json", return_value=response):
            snapshot = capabilities.fetch(self.session, refresh=True)
        self.assertIsNone(snapshot["concurrency"]["limit"])
        self.assertEqual(snapshot["concurrency"]["state"], "unknown")

    def test_identity_uses_confirmed_principal_across_auth_modes_and_permissions(self):
        self.session._brain_account_id = "synthetic-user-id"
        first = capabilities.account_provenance(self.session)
        cache_store.write_json_atomic(ml.session_cache_path(), cache_store.build_entry("session", {
            "user_id": "synthetic-user-id", "username": "synthetic-account",
            "cookies": {"session": "synthetic-cookie"}, "permissions": ["EXPANDED"]}, ttl=3600))
        cookies = SimpleNamespace(get_dict=lambda: {"session": "synthetic-cookie"})
        cookie_session = SimpleNamespace(cookies=cookies, auth=None)
        self.assertEqual(capabilities.account_provenance(cookie_session)["fingerprint"], first["fingerprint"])
        self.assertEqual(capabilities.account_provenance()["fingerprint"], first["fingerprint"])
        self.session.auth = ("renamed-user", "new-password")
        self.session.permissions = ["REDUCED"]
        self.assertEqual(capabilities.account_provenance(self.session)["fingerprint"], first["fingerprint"])
        serialized = json.dumps(first)
        for secret in ("synthetic-user-id", "synthetic-account", "synthetic-cookie", "new-password"):
            self.assertNotIn(secret, serialized)
        with mock.patch.dict(os.environ, {"BRAIN_USERNAME": "other-account"}):
            self.assertNotEqual(capabilities.account_provenance()["fingerprint"], first["fingerprint"])

    def test_cookie_material_is_never_a_fallback_identity(self):
        anonymous = SimpleNamespace(auth=None, cookies=SimpleNamespace(get_dict=lambda: {"token": "opaque"}))
        self.assertEqual(capabilities.account_provenance(anonymous)["state"], "unknown")

    def test_preflight_reuses_supplied_snapshot_and_checks_actual_pair_decay(self):
        snapshot = self.fetch({"decay": {"type": "integer", "min_value": 0, "max_value": 2}})
        settings = ml.resolve_settings({"decay": 8, "futureOption": None})
        scope = {"known_fields": {"sample_field"}, "field_types": {"sample_field": "MATRIX"},
                 "market_fields": {"sample_field"}, "records": [{"id": "sample_field", "type": "MATRIX"}],
                 "evidence": {"state": "verified", "scope": {}, "coverage": {"complete": True}}}

        def operators(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"name": "rank"}]

        args = SimpleNamespace(allow_lint_errors=False, capability_snapshot=snapshot)
        with mock.patch.object(ml, "request_json", side_effect=AssertionError("OPTIONS repeated")), \
                mock.patch.object(ml, "fetch_operators", side_effect=operators), \
                mock.patch.object(validation, "_field_scope", return_value=scope):
            report = validation._preflight_lint(self.session, [("rank(sample_field)", 1)], settings, args)
        self.assertTrue(report["ok"])
        self.assertEqual(report["checked_requests"], {"settings": settings, "pairs": [["rank(sample_field)", 1]]})
        self.assertEqual(report["capability_snapshot"], snapshot)
        self.assertEqual(report["settings_capabilities"]["state"], "unknown")
        self.assertEqual(report["settings_capabilities"]["unsupported"], [])
        self.assertIsNone(report["checked_requests"]["settings"]["futureOption"])

    def test_regular_preflight_and_dispatch_check_the_wire_settings(self):
        snapshot = self.fetch({
            "decay": {"type": "integer", "max_value": 2},
            "selectionHandling": {"type": "string", "required": True},
            "selection_limit": {"type": "integer", "required": True},
        })
        settings = ml.resolve_settings({"selection_handling": "NON_NAN", "selectionLimit": 1000})
        pairs = [("rank(sample_field)", 1)]
        args = SimpleNamespace(allow_lint_errors=True, capability_snapshot=snapshot)
        self.session.post = mock.Mock(return_value=SimpleNamespace(
            status_code=201, headers={"Location": "/simulations/accepted"}, raise_for_status=lambda: None))

        with mock.patch.object(ml, "fetch_operators", side_effect=RuntimeError("metadata unavailable")), \
                mock.patch.object(validation, "_field_scope", side_effect=RuntimeError("metadata unavailable")):
            preflight = validation._preflight_lint(self.session, pairs, settings, args)
        self.assertEqual(preflight["settings_capabilities"]["unsupported"], [])
        self.assertEqual(preflight["settings_capabilities"]["omitted_from_payload"],
                         ["selection_handling", "selectionLimit"])
        self.assertEqual(preflight["checked_requests"]["settings"], settings)
        self.assertIn("selectionHandling", preflight["capability_snapshot"]["settings_schema"])

        with mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": str(self.root)}), \
                mock.patch.object(ml, "poll_simulation", return_value={"status": "COMPLETE", "alpha": "A1"}), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value={"is": {}}):
            records = ml.simulate_alphas(self.session, pairs + [("rank(close)", 3)], settings=settings,
                                         capability_snapshot=snapshot, concurrency=1)
        self.session.post.assert_called_once()
        payload = self.session.post.call_args.kwargs["json"]
        self.assertEqual(payload, {"type": "REGULAR", "regular": "rank(sample_field)",
                                   "settings": {**ml.DEFAULT_SETTINGS, "decay": 1}})
        self.assertEqual(records[0]["alpha_id"], "A1")
        self.assertEqual(records[1]["settings_capabilities"]["unsupported"],
                         [{"setting": "decay", "value": 3, "reasons": ["max_value"]}])

    def test_snapshot_accepts_changed_provenance_sources_for_same_principal(self):
        live = self.fetch()
        path = ml.metadata_cache_path("simulation-capabilities", {
            "account": live["account"]["fingerprint"], "scope": live["scope"]})
        payload = cache_store.read_entry(path)["payload"]
        payload["account"]["sources"] = ["new-confirmation"]
        cache_store.write_json_atomic(path, cache_store.build_entry("simulation-capabilities", payload, ttl=3600))
        self.assertEqual(capabilities.cached()["state"], "verified")

    def test_confirmed_identity_cache_is_reused_after_permission_change(self):
        self.session._brain_account_id = "synthetic-user-id"
        live = self.fetch()
        cache_store.write_json_atomic(ml.session_cache_path(), cache_store.build_entry("session", {
            "user_id": "synthetic-user-id", "username": "synthetic-account",
            "permissions": ["CHANGED"], "cookies": {"session": "changed-cookie"}}, ttl=3600))
        self.assertEqual(capabilities.cached()["capability_id"], live["capability_id"])
        self.assertEqual(capabilities.cached()["state"], "verified")

    def test_lint_override_cannot_override_authoritative_settings_rejection(self):
        snapshot = self.fetch({"decay": {"type": "integer", "max_value": 2}})
        args = SimpleNamespace(allow_lint_errors=True, capability_snapshot=snapshot)
        with mock.patch.object(ml, "fetch_operators", side_effect=RuntimeError("synthetic metadata outage")), \
                mock.patch.object(validation, "_field_scope", side_effect=RuntimeError("synthetic metadata outage")):
            with self.assertRaises(validation.CliError) as caught:
                validation._preflight_lint(self.session, [("rank(sample_field)", 9)], {}, args)
        evidence = caught.exception.details["settings_capabilities"]
        self.assertEqual(evidence["state"], "unsupported")
        self.assertEqual(evidence["unsupported"][0]["setting"], "decay")
        self.assertEqual(evidence["unsupported"][0]["pair_index"], 0)


if __name__ == "__main__":
    unittest.main()
