"""Python Alpha checks parse source and verify account evidence without executing code."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase, mock

import machine_lib as ml
import simulation_capabilities
from research_app import capabilities, experiments, validation
from research_app.runtime import CliError
from tools import guarded_sim


SOURCE = """from brain.alphas import alpha as signal
would_raise_if_executed()

@signal(data=["close"], store=[])
def example(data, store):
    return data.close[-1]
"""


class PythonPreflightTests(TestCase):
    def setUp(self):
        self.session = SimpleNamespace(_brain_account_id="account-with-python")
        self.snapshot = {
            "state": "verified",
            "source": "test-options",
            "account": simulation_capabilities.account_provenance(self.session),
            "settings_schema": {"language": {"type": "string", "choices": ["FASTEXPR", "PYTHON"]}},
        }
        self.settings = ml.resolve_settings({"language": "PYTHON", "lookback": 2})
        self.fields = capabilities.field_scope([{"id": "close", "type": "MATRIX"}],
                                               {"state": "verified"})

    def test_static_check_keeps_source_and_does_not_execute_it(self):
        result = validation.lint_python_source(SOURCE)
        self.assertTrue(result["ok"])
        self.assertEqual(result["expression"], SOURCE)
        self.assertEqual(result["declared_fields"], ["close"])
        self.assertEqual(result["fields_state"], "unknown")
        self.assertEqual([item["code"] for item in result["warnings"]], ["python_fields_unverified"])
        self.assertIsNone(result["operator_count"])

    def test_history_warnings_distinguish_index_from_truncated_slice(self):
        source = SOURCE.replace("data, store", "frame, store").replace(
            "data.close[-1]", "frame.close[-4], frame.close[-5:]")
        result = validation.lint_python_source(source, known_fields={"close"}, lookback=2)
        self.assertTrue(result["ok"])
        self.assertEqual(
            [(item["code"], item["required_rows"], item["available_rows"])
             for item in result["warnings"]],
            [("python_history_index_out_of_range", 4, 3),
             ("python_history_slice_truncated", 5, 3)],
        )
        self.assertEqual({item["line"] for item in result["warnings"]}, {6})
        self.assertEqual({item["identifier"] for item in result["warnings"]}, {"close"})

    def test_history_warning_stays_with_direct_literal_declared_data_access(self):
        source = SOURCE.replace("return data.close[-1]", """
    def helper(data):
        return data.close[-99]
    alias = data
    indices = 99
    return (data.close[-3], data.close[-3:], data.close[-indices],
            data.close[-99:1], data.close[-99::2], data.undeclared[-99],
            store.close[-99], alias.close[-99],
            [data.close[-99] for data in []])
""")
        result = validation.lint_python_source(source, known_fields={"close"}, lookback=2)
        self.assertTrue(result["ok"])
        self.assertEqual(result["warnings"], [])
        rebound = SOURCE.replace("return data.close[-1]", "data = store\n    return data.close[-99]")
        self.assertEqual(validation.lint_python_source(
            rebound, known_fields={"close"}, lookback=2)["warnings"], [])

    def test_bad_static_sources_fail_before_field_verification(self):
        cases = [
            (SOURCE.replace("def example(data, store):", "def example(data, store)"), "python_syntax"),
            (SOURCE.replace('data=["close"]', "data=FIELDS"), "python_data_declaration"),
            (SOURCE.replace('data=["close"]', 'data=["universe"]'), "python_implicit_field"),
            (SOURCE.replace("@signal", "@unrelated"), "python_alpha_entrypoint"),
        ]
        for source, error in cases:
            with self.subTest(error=error):
                report = validation.lint_python_source(source)
                self.assertFalse(report["ok"])
                self.assertEqual(report["errors"][0]["code"], error)

    def test_online_checks_fields_and_skips_fastexpr_operator_catalog(self):
        self.snapshot["settings_schema"].update(
            selectionHandling={"type": "string", "required": True},
            selectionLimit={"type": "integer", "required": True},
            lookback={"type": "integer", "min_value": 0, "required": True},
        )
        with (
            mock.patch.object(ml, "fetch_operators") as operators,
            mock.patch.object(validation, "_field_scope", return_value=self.fields) as fields,
        ):
            report = validation._preflight_lint(
                self.session, [(SOURCE, 6)], self.settings, SimpleNamespace(),
                capability_snapshot=self.snapshot)
        operators.assert_not_called()
        fields.assert_called_once_with(self.session, self.settings, required_fields={"close"})
        self.assertEqual(report["coverage"], {"operators": "not_applicable", "fields": "verified"})
        self.assertEqual(report["language_capability"]["state"], "verified")
        self.assertEqual(report["checked_requests"]["pairs"], [[SOURCE, 6]])
        self.assertEqual(report["results"][0]["fields_state"], "verified")
        self.assertIn("selectionLimit", report["capability_snapshot"]["settings_schema"])
        self.assertIn("lookback", report["settings_capabilities"]["checked"])

    def test_language_when_unverified_or_unsupported_cannot_be_forced(self):
        cases = [
            ({**self.snapshot, "settings_schema": {"language": {"type": "string"}}},
             "python_language_unverified"),
            ({**self.snapshot, "settings_schema": {"language": {"choices": ["FASTEXPR"]}}},
             "unsupported_setting"),
            ({**self.snapshot, "account": {"fingerprint": "other"}}, "python_language_unverified"),
        ]
        for snapshot, code in cases:
            with (
                self.subTest(code=code),
                mock.patch.object(validation, "_field_scope", return_value=self.fields),
                self.assertRaises(CliError) as caught,
            ):
                validation._preflight_lint(
                    self.session, [(SOURCE, 6)], self.settings, SimpleNamespace(allow_lint_errors=True),
                    capability_snapshot=snapshot)
            self.assertIn(code, [item["code"] for item in caught.exception.details["errors"]])

    def test_available_language_does_not_hide_an_unknown_field(self):
        fields = capabilities.field_scope([], {"state": "verified"})
        with (
            mock.patch.object(validation, "_field_scope", return_value=fields),
            self.assertRaises(CliError) as caught,
        ):
            validation._preflight_lint(
                self.session, [(SOURCE, 6)], self.settings, SimpleNamespace(),
                capability_snapshot=self.snapshot)
        self.assertEqual(caught.exception.details["errors"][0]["identifier"], "close")

    def test_implicit_universe_does_not_require_a_field_request(self):
        source = SOURCE.replace('data=["close"]', "data=[]").replace("data.close", "data.universe")
        with mock.patch.object(validation, "_field_scope") as fields:
            report = validation._preflight_lint(
                self.session, [(source, 6)], self.settings, SimpleNamespace(),
                capability_snapshot=self.snapshot)
        fields.assert_not_called()
        self.assertEqual(report["coverage"]["fields"], "not_applicable")

    def test_guarded_sim_checks_both_source_and_account_before_dispatch(self):
        with mock.patch.object(validation, "_field_scope", return_value=self.fields):
            preflight = validation._preflight_lint(
                self.session, [(SOURCE, 6)], self.settings, SimpleNamespace(),
                capability_snapshot=self.snapshot)
        with mock.patch.object(ml, "simulate_alphas") as dispatch:
            other = SimpleNamespace(_brain_account_id="other")
            with self.assertRaisesRegex(guarded_sim.UnknownNameError, "this account"):
                guarded_sim.simulate([SOURCE], self.settings, session=other, preflight=preflight)
            with self.assertRaisesRegex(guarded_sim.UnknownNameError, "exact simulation requests"):
                guarded_sim.simulate([SOURCE + "\n"], self.settings, session=self.session, preflight=preflight)
            dispatch.assert_not_called()
            guarded_sim.simulate([SOURCE], self.settings, session=self.session, preflight=preflight)
            dispatch.assert_called_once()
            self.assertEqual(dispatch.call_args.args, (self.session, [SOURCE]))

    def test_guarded_sim_without_report_does_not_force_an_unverified_language(self):
        snapshot = {**self.snapshot, "state": "unknown"}
        with (
            mock.patch.object(validation, "_field_scope", return_value=self.fields),
            mock.patch.object(simulation_capabilities, "fetch", return_value=snapshot),
            mock.patch.object(ml, "simulate_alphas") as dispatch,
            self.assertRaisesRegex(guarded_sim.UnknownNameError, "Python language availability"),
        ):
            guarded_sim.simulate([SOURCE], self.settings, session=self.session, force=True)
        dispatch.assert_not_called()

    def test_offline_lint_never_opens_a_session(self):
        source = SOURCE.replace("data.close[-1]", "data.close[-4:]")
        args = SimpleNamespace(expr=[source], setting=[("language", "PYTHON"), ("lookback", 2)], offline=True)
        with mock.patch.object(validation, "with_session") as session:
            result = validation.cmd_lint(args, None)
        session.assert_not_called()
        self.assertTrue(result["ok"])
        self.assertFalse(result["catalog_checked"])
        self.assertEqual(result["language_capability"]["state"], "unknown")
        self.assertIn("python_history_slice_truncated", [item["code"] for item in result["warnings"]])

    def test_probe_preview_and_preflight_use_each_arms_resolved_lookback(self):
        source = SOURCE.replace("data.close[-1]", "data.close[-4]")
        spec = {"design": "explore", "settings": {"language": "PYTHON", "lookback": 2}, "arms": [
            {"label": "short", "expression": source},
            {"label": "long", "expression": source, "settings": {"lookback": 3}},
        ]}
        evidence = capabilities.field_scope([], {"state": "unknown"})
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": directory}),
            mock.patch.object(experiments, "cached_field_scope", return_value=evidence),
            mock.patch.object(experiments, "with_session") as session,
        ):
            path = Path(directory) / "windows.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            result = experiments.cmd_probe(SimpleNamespace(spec=str(path), dry_run=True), None)
        session.assert_not_called()
        self.assertTrue(result["lint"]["ok"])
        self.assertEqual(
            [item["lookback"] for item in result["lint"]["warnings"] if "lookback" in item], [2],
        )
        with mock.patch.object(validation, "_field_scope", return_value=self.fields):
            report = validation._preflight_lint(
                self.session, [(source, 6)], self.settings, SimpleNamespace(),
                capability_snapshot=self.snapshot)
        self.assertTrue(report["ok"])
        self.assertEqual([item["code"] for item in report["warnings"]],
                         ["python_history_index_out_of_range"])
