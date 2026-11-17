"""SPC specs are checked against the live schema before anything is submitted."""

from contextlib import redirect_stdout
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

# spc_client imports machine_lib at module scope; stub it so the pure validation
# function can be tested without credentials or network.
if "machine_lib" not in sys.modules:
    stub = types.ModuleType("machine_lib")
    stub.load_session = lambda *a, **k: None
    sys.modules["machine_lib"] = stub

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
