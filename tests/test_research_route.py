import copy
import json
import unittest
from unittest import mock

from research_app.research_route import ATOM_RUNNER_MECHANISM, normalize_route


def _spc_spec(**kwargs):
    sample = [{
        "company_name": "Apple Inc.",
        "ticker": "AAPL",
        "isin": "US0378331005",
        "mic": "XNAS",
        "confidence_score": 0.82,
        "investment_horizon": 22,
        "return_prediction": 0.045,
    }]
    return {
        "name": "Example",
        "prompt": "Estimate the next return.",
        "sampleOutput": json.dumps(sample),
        "modelFamily": "model-to-verify-with-provider",
        "weight": 0.25,
        "updateFrequency": "frequency-to-verify-with-provider",
        **kwargs,
    }


class ResearchRouteTests(unittest.TestCase):
    def test_legacy_probe_keeps_native_defaults(self):
        spec = {"arms": [{"expression": "rank(close)"}]}
        self.assertEqual(normalize_route(spec), {
            "track": "REGULAR",
            "protocol": "brain_simulation",
            "simulation_type": "REGULAR",
            "language": "FASTEXPR",
            "simulation_mode": None,
        })
        self.assertNotIn("settings", spec)

    def test_regular_tracks_do_not_change_native_type(self):
        for track, expected in (("power-pool", "POWER_POOL"),
                                ("PP", "POWER_POOL"), ("atom", "ATOM")):
            with self.subTest(track=track):
                spec = {"track": track, "settings": {"simulationMode": "quick"}}
                original = copy.deepcopy(spec)
                route = normalize_route(spec)
                self.assertEqual(route["track"], expected)
                self.assertEqual(route["protocol"], "brain_simulation")
                self.assertEqual(route["simulation_type"], "REGULAR")
                self.assertEqual(route["simulation_mode"], "QUICK")
                self.assertNotIn("eligible", route)
                self.assertEqual(spec, original)

    def test_legacy_atom_scan_keeps_its_intended_track(self):
        route = normalize_route({"mechanism": ATOM_RUNNER_MECHANISM, "arms": []})
        self.assertEqual(route["track"], "ATOM")

    def test_python_is_language_and_uses_effective_arm_settings(self):
        spec = {"track": "ATOM", "settings": {"simulationMode": "QUICK"}}
        effective = {"language": "python", "simulationMode": "FULL"}
        route = normalize_route(spec, settings=effective)
        self.assertEqual(route["track"], "ATOM")
        self.assertEqual(route["simulation_type"], "REGULAR")
        self.assertEqual(route["language"], "PYTHON")
        self.assertEqual(route["simulation_mode"], "FULL")
        self.assertEqual(effective["language"], "python")

    def test_mode_aliases_and_provider_extensions(self):
        spec = {"settings": {"simulation_mode": "full", "simulationMode": "FULL"}}
        self.assertEqual(normalize_route(spec)["simulation_mode"], "FULL")
        spec["settings"]["simulation_mode"] = "QUICK"
        with self.assertRaisesRegex(ValueError, "conflicting settings"):
            normalize_route(spec)
        self.assertEqual(normalize_route({
            "settings": {"simulationMode": "NATIVE_EXTENSION"}
        })["simulation_mode"], "NATIVE_EXTENSION")

    def test_spc_has_its_own_protocol_and_validates_offline(self):
        for spec in (_spc_spec(), _spc_spec(track="spc")):
            with self.subTest(spec=spec):
                with mock.patch("machine_lib.load_session", side_effect=AssertionError("network")):
                    self.assertEqual(normalize_route(spec), {
                        "track": "SPC",
                        "protocol": "spc_submission",
                        "simulation_type": None,
                        "language": None,
                        "simulation_mode": None,
                    })

    def test_spc_reuses_sample_and_spec_shape_checks(self):
        for changes, error in (({"sampleOutput": "[]"}, "sampleOutput"),
                               ({"prompt": ""}, "prompt"),
                               ({"weight": 0.123}, "weight"),
                               ({"weight": True}, "weight")):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, error):
                    normalize_route(_spc_spec(track="SPC", **changes))

    def test_rejects_contradictory_and_malformed_routes(self):
        for spec, error in (({"track": "typo"}, "track"),
                            (_spc_spec(track="REGULAR"), "SPC track"),
                            (_spc_spec(track="SPC", arms=[]), "simulation arms"),
                            ({"settings": 6}, "settings must be an object"),
                            ({"settings": {"language": 6}}, "language")):
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(ValueError, error):
                    normalize_route(spec)
