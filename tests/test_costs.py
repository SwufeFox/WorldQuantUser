"""Cost scenarios use the exact supplied period and explicit units."""

import unittest

from research_app import costs


class CostEvidenceTests(unittest.TestCase):
    def test_raw_margin_conversion_and_one_way_scenarios(self):
        row = costs.evidence({"margin": 0.0015, "turnover": 0.2,
                              "startDate": "2020-01-01", "endDate": "2020-12-31"},
                             "train", [0, 5, 20])
        self.assertEqual(row["raw_margin"], 0.0015)
        self.assertEqual(row["margin_bps"], 15)
        self.assertEqual([item["net_margin_bps"] for item in row["scenarios"]], [15, 10, -5])
        self.assertEqual(row["turnover"], 0.2)
        self.assertEqual(row["turnover_period"], "unknown")
        self.assertEqual(row["period"], {"start": "2020-01-01", "end": "2020-12-31"})
        self.assertEqual(row["cost_basis"], "one_way_per_dollar_traded")
        self.assertIsNone(row["net_sharpe"])
        self.assertIsNone(row["daily_costs"])

    def test_unknown_margin_and_turnover_do_not_invent_values(self):
        row = costs.evidence({}, "train")
        self.assertEqual(row["state"], "unknown_margin")
        self.assertIsNone(row["margin_bps"])
        self.assertIsNone(row["turnover"])
        self.assertTrue(all(item["net_margin_bps"] is None for item in row["scenarios"]))
        invalid = costs.evidence({"margin": float("nan"), "turnover": -1}, "train")
        self.assertIsNone(invalid["raw_margin"])
        self.assertIsNone(invalid["turnover"])

    def test_scopes_are_isolated_without_full_period_fallback(self):
        entry = {"metrics": {"margin": 0.005},
                 "out_of_sample": {"test": {"margin": -0.001}}}
        train = costs.evidence(entry["out_of_sample"].get("train"), "train")
        test = costs.evidence(entry["out_of_sample"]["test"], "test")
        full = costs.evidence(entry["metrics"], "full")
        self.assertIsNone(train["margin_bps"])
        self.assertEqual(test["margin_bps"], -10)
        self.assertEqual(full["margin_bps"], 50)

    def test_invalid_costs_are_rejected(self):
        for value in (-1, float("nan"), float("inf"), -float("inf"), True, "5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                costs.evidence({}, "train", [value])
