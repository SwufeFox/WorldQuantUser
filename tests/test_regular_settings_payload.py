"""A REGULAR simulation must not carry the selection settings.

``OPTIONS /simulations`` lists ``selectionHandling`` and ``selectionLimit`` as
required, so the capability preflight asks every spec for them. The POST
contract then answers 400 - "Selection handling NON_NAN is not allowed for
simulation type REGULAR" - which left the tool unable to submit anything at
all. The payload builder drops them where the simulation type is known.
"""

from __future__ import annotations

import unittest

import machine_lib as ml
import research_config


class RegularPayloadDropsSelectionSettings(unittest.TestCase):
    def test_selection_settings_are_not_sent(self):
        settings = {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 4,
            "neutralization": "SUBINDUSTRY",
            "truncation": 0.08,
            "pasteurization": "ON",
            "testPeriod": "P0Y",
            "unitHandling": "VERIFY",
            "nanHandling": "ON",
            "selectionHandling": "NON_NAN",
            "selectionLimit": 1000,
            "language": "FASTEXPR",
            "visualization": False,
        }

        payload = ml._single_simulation_data("rank(close)", 4, settings)

        self.assertEqual(payload["type"], "REGULAR")
        self.assertNotIn("selectionHandling", payload["settings"])
        self.assertNotIn("selectionLimit", payload["settings"])
        # Everything else must survive untouched.
        for key in ("region", "universe", "delay", "neutralization",
                    "unitHandling", "nanHandling", "language"):
            self.assertEqual(payload["settings"][key], settings[key])
        self.assertEqual(payload["settings"]["decay"], 4)
        self.assertEqual(payload["regular"], "rank(close)")

    def test_other_settings_are_not_dropped(self):
        payload = ml._single_simulation_data("rank(close)", 6,
                                             {"region": "USA", "decay": 99})
        self.assertEqual(payload["settings"]["region"], "USA")
        self.assertEqual(payload["settings"]["decay"], 6)

    def test_python_file_and_platform_readback_preserve_source(self):
        source = '\n@alpha(data=["returns"], store=[])\r\ndef sample(data, store):\r\n    return data.returns[-1]\r\n'
        expected = source.replace("\r\n", "\n")
        settings, entries = research_config.parse_research_text(source, "sample.py")
        resolved, [(code, decay)] = research_config.build_config({**settings, "lookback": 4}, entries)
        payload = ml._single_simulation_data(code, decay, resolved)
        self.assertEqual(payload["regular"], expected)
        self.assertEqual(payload["settings"]["language"], "PYTHON")
        self.assertEqual(payload["settings"]["lookback"], 4)
        self.assertNotIn("unitHandling", payload["settings"])
        self.assertNotIn("nanHandling", payload["settings"])
        self.assertEqual(ml.regular_expression({"regular": {"code": source, "language": "PYTHON"}})[0], expected)
        self.assertEqual(ml.normalize_alphas(["  rank(close)\n"], ml.DEFAULT_SETTINGS)[0][0], "rank(close)")

    def test_python_regular_settings_check_omits_super_only_required_fields(self):
        settings = ml.resolve_settings({"language": "python", "lookback": 0})
        snapshot = {"state": "verified", "settings_schema": {
            "language": {"required": True, "choices": [{"value": "PYTHON"}]},
            "lookback": {"required": True, "type": "integer", "minValue": 0, "maxValue": 5},
            "selectionHandling": {"required": True},
            "selectionLimit": {"required": True},
            "unitHandling": {"required": True},
            "nanHandling": {"required": True},
        }}
        report = ml.regular_settings_evidence(settings, snapshot)
        self.assertEqual(report["unsupported"], [])
        self.assertIn("language", report["checked"])
        self.assertEqual(report["omitted_from_payload"], ["unitHandling", "nanHandling"])
        self.assertIn("selectionLimit", snapshot["settings_schema"])
        outside = ml.regular_settings_evidence({**settings, "lookback": 6}, snapshot)
        self.assertEqual(outside["unsupported"], [
            {"setting": "lookback", "value": 6, "reasons": ["max_value"]},
        ])


if __name__ == "__main__":
    unittest.main()
