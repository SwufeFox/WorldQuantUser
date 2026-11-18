"""SPC specs are checked against the live schema before anything is submitted."""

from contextlib import redirect_stdout
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from requests import Response
from requests.exceptions import ReadTimeout

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import spc_client  # noqa: E402

SCHEMA = {
    "name": {"type": "string", "required": True, "maxLength": 200},
    "prompt": {"type": "string", "required": True, "maxLength": 10000},
    "sampleOutput": {"type": "string", "required": True, "maxLength": 100000},
    "modelFamily": {
        "type": "string",
        "required": True,
        "maxLength": 100,
        "choices": [
            {"value": "deepseek", "items": [{"value": "deepseek.v3.2"}]},
        ],
    },
    "weight": {"type": "decimal", "required": True},
    "updateFrequency": {
        "type": "choice",
        "required": True,
        "choices": [{"value": "daily"}, {"value": "weekly"}],
    },
}

SAMPLE = json.dumps([{
    "company_name": "Apple Inc.", "ticker": "AAPL", "isin": "US0378331005",
    "mic": "XNAS", "confidence_score": 0.82, "investment_horizon": 22,
    "return_prediction": 0.045,
}])


def spec(**overrides):
    base = {"name": "demo", "prompt": "rank and predict", "sampleOutput": SAMPLE,
            "modelFamily": "deepseek.v3.2", "weight": 1.0, "updateFrequency": "daily"}
    base.update(overrides)
    return base


class SpcSpecTests(unittest.TestCase):
    def spec_file(self, data):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "spec.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    @staticmethod
    def platform_response(status, body):
        response = Response()
        response.status_code = status
        response._content = json.dumps(body).encode()
        return response

    def test_submit_requires_consent_before_session(self):
        with mock.patch.object(spc_client, "session") as session, redirect_stdout(io.StringIO()):
            self.assertEqual(spc_client.cmd_submit("unused.json", False), 2)
        session.assert_not_called()

    def test_invalid_and_http_refused_submissions_exit_nonzero(self):
        for invalid in (True, False):
            with self.subTest(invalid=invalid):
                path = self.spec_file(spec(weight=5) if invalid else spec())
                session = mock.Mock()
                session.post.return_value = self.platform_response(400, {"modelFamily": "rejected"})
                stdout = io.StringIO()
                with mock.patch.object(spc_client, "session", return_value=session), \
                        mock.patch.object(spc_client, "schema", return_value=SCHEMA), redirect_stdout(stdout):
                    self.assertEqual(spc_client.cmd_submit(path, True), 1)
                self.assertEqual(session.post.call_count, 0 if invalid else 1)
                self.assertIn("FAIL demo", stdout.getvalue())

    def test_unknown_post_outcome_stops_without_repeating_or_submitting_remaining_specs(self):
        path = self.spec_file([spec(), spec(name="second")])
        for outcome in (ReadTimeout(), self.platform_response(502, {}), self.platform_response(201, {}),
                        self.platform_response(307, {})):
            with self.subTest(outcome=outcome):
                session = mock.Mock()
                if isinstance(outcome, Exception):
                    session.post.side_effect = outcome
                else:
                    session.post.return_value = outcome
                stdout = io.StringIO()
                with mock.patch.object(spc_client, "session", return_value=session), \
                        mock.patch.object(spc_client, "schema", return_value=SCHEMA), redirect_stdout(stdout):
                    self.assertEqual(spc_client.cmd_submit(path, True), 1)
                session.post.assert_called_once_with(spc_client.SPC, json=spec(), timeout=30,
                                                     allow_redirects=False)
                self.assertIn("check recon before retrying", stdout.getvalue())

    def test_submit_success_identifies_server_record(self):
        path = self.spec_file(spec())
        session = mock.Mock()
        session.post.return_value = self.platform_response(201, {"id": "spc-1"})
        stdout = io.StringIO()
        with mock.patch.object(spc_client, "session", return_value=session), \
                mock.patch.object(spc_client, "schema", return_value=SCHEMA), redirect_stdout(stdout):
            self.assertEqual(spc_client.cmd_submit(path, True), 0)
        self.assertIn("id=spc-1", stdout.getvalue())

    def test_recon_prioritizes_submissions_and_stops_optional_probes_at_deadline(self):
        clock = [100.0]
        session = mock.Mock()
        session.options.return_value = self.platform_response(200, {"actions": {"POST": SCHEMA}})

        def submissions(*_args, **_kwargs):
            clock[0] += spc_client.DIAGNOSTIC_SECONDS + 1
            return self.platform_response(200, {"count": 1, "results": [{"id": "spc-1"}]})

        session.get.side_effect = submissions
        stdout = io.StringIO()
        with mock.patch.object(spc_client.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(spc_client, "session", return_value=session), redirect_stdout(stdout):
            self.assertEqual(spc_client.cmd_recon(), 0)
        session.options.assert_called_once_with(spc_client.SPC, timeout=spc_client.PROBE_SECONDS,
                                                allow_redirects=False)
        session.get.assert_called_once_with(f"{spc_client.SPC}?limit=50", timeout=spc_client.PROBE_SECONDS,
                                            allow_redirects=False)
        self.assertIn("== submissions: 1 ==", stdout.getvalue())
        self.assertIn("diagnostic budget expired", stdout.getvalue())

    def test_valid_spec_passes(self):
        self.assertEqual(spc_client.validate_spec(spec(), SCHEMA), [])

    def test_missing_required_field_is_named(self):
        bad = spec()
        del bad["weight"]
        self.assertTrue(any("weight" in p for p in spc_client.validate_spec(bad, SCHEMA)))

    def test_unknown_model_is_rejected(self):
        problems = spc_client.validate_spec(spec(modelFamily="gpt-5-ultra"), SCHEMA)
        self.assertTrue(any("modelFamily" in p for p in problems))

    def test_sample_output_must_be_json_array(self):
        problems = spc_client.validate_spec(spec(sampleOutput='{"a": 1}'), SCHEMA)
        self.assertTrue(any("JSON array" in p for p in problems))

    def test_length_limits_are_enforced(self):
        problems = spc_client.validate_spec(spec(prompt="x" * 10001), SCHEMA)
        self.assertTrue(any("prompt too long" in p for p in problems))

    def test_frequency_must_come_from_the_schema(self):
        problems = spc_client.validate_spec(spec(updateFrequency="hourly"), SCHEMA)
        self.assertTrue(any("updateFrequency" in p for p in problems))

    def test_rejects_malformed_predictions_and_weight(self):
        original = json.loads(SAMPLE)[0]
        for field, value in (("isin", "US0378331006"), ("confidence_score", True),
                             ("investment_horizon", 2.5), ("return_prediction", -0.1)):
            with self.subTest(field=field):
                output = json.dumps([{**original, field: value}])
                self.assertTrue(any(field in problem for problem in
                                    spc_client.validate_spec(spec(sampleOutput=output), SCHEMA)))
        for invalid in (True, float("nan"), 1.01, -0.01, 0.001):
            with self.subTest(weight=invalid):
                self.assertTrue(any("weight" in problem for problem in
                                    spc_client.validate_spec(spec(weight=invalid), SCHEMA)))

    def test_rejects_empty_duplicate_and_nonfinite_output(self):
        row = json.loads(SAMPLE)[0]
        for output, finding in (("[]", "non-empty JSON array"),
                                (json.dumps([row, row]), "duplicate ISIN and MIC pair"),
                                (SAMPLE.replace("0.045", "1e999"), "non-finite"),
                                (SAMPLE.replace('"mic":', '"mic": "XNAS", "mic":'), "duplicate JSON object key")):
            with self.subTest(finding=finding):
                self.assertTrue(any(finding in problem for problem in
                                    spc_client.validate_spec(spec(sampleOutput=output), SCHEMA)))

    def test_experiment_runs_offline_and_keeps_failed_captures(self):
        apple = json.loads(SAMPLE)[0]
        microsoft = {**apple, "company_name": "Microsoft Corporation", "ticker": "MSFT",
                     "isin": "US5949181045", "confidence_score": 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "experiment.json"
            capture = root / "repeat.json"
            failed = root / "failed.json"
            reference = root / "universe.json"
            manifest.write_text(json.dumps(spec(sampleOutput=json.dumps([apple, microsoft]))), encoding="utf-8")
            capture.write_text(json.dumps([microsoft, {**apple, "confidence_score": -0.7}]), encoding="utf-8-sig")
            failed.write_text("```json\n[]\n```", encoding="utf-8")
            reference.write_text(json.dumps([{"isin": apple["isin"], "mic": "XNAS", "region": "americas",
                                              "sector": "Information Technology"}]), encoding="utf-8")

            stdout = io.StringIO()
            with mock.patch.object(spc_client, "session", side_effect=AssertionError("network")), redirect_stdout(stdout):
                status = spc_client.main(["experiment", str(manifest), "--run", str(capture),
                                          "--run", str(failed), "--reference", str(reference)])

        report = json.loads(stdout.getvalue())
        self.assertEqual(status, 2)
        self.assertFalse(report["ok"])
        self.assertEqual([row["ok"] for row in report["runs"]], [True, False])
        self.assertEqual(len(report["comparisons"]), 1)
        comparison = report["comparisons"][0]
        self.assertEqual((comparison["shared"], comparison["jaccard"], comparison["side_changes"]), (2, 1, 1))
        self.assertEqual(report["sample"]["coverage"]["denominator"], 1)
        self.assertEqual(report["sample"]["coverage"]["region"]["qualifying"], ["Americas"])


if __name__ == "__main__":
    unittest.main()
