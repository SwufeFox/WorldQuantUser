"""SPC specs are checked against the live schema before anything is submitted."""

import json
import sys
import types
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()