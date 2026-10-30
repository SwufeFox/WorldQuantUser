#!/usr/bin/env python3
"""Tests for annotations (syntax, checks, ablation derivation) and the intent rewrite."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import fastexpr_ast as fa  # noqa: E402
import intent  # noqa: E402

class Intent(unittest.TestCase):
    def test_bare_level_is_rewritten(self):
        plans = intent.plan(fa.parse("group_rank(fnd6_newa2v1300_sale, subindustry)"))
        self.assertEqual(len(plans), 1)
        self.assertIn("fnd6_newa2v1300_sale / ts_backfill(fnd6_cptnewqv1300_atq, 252)",
                      plans[0]["after"])

    def test_already_normalised_is_left_alone(self):
        source = "group_rank(fnd6_newa2v1300_sale / fnd6_cptnewqv1300_atq, subindustry)"
        self.assertEqual(intent.plan(fa.parse(source)), [])

    def test_field_outside_the_curated_list_is_left_alone(self):
        self.assertEqual(intent.plan(fa.parse("group_rank(close, subindustry)")), [])

    def test_only_the_offending_leg_is_rewritten(self):
        source = "a = rank(fnd6_cptnewqv1300_ltq);\nb = rank(close / vwap);\na + b"
        after = intent.plan(fa.parse(source))[0]["after"]
        self.assertIn("a = rank(fnd6_cptnewqv1300_ltq / ts_backfill(", after)
        self.assertIn("b = rank(close / vwap)", after)




class Derivation(unittest.TestCase):
    """Levels from units + dataset, so the curated list is an override, not the source."""

    UNITS = {"assets_total": "Unit[TSPrice:1]", "close": "Unit[TSPrice:1]",
             "atq": "Unit[TSPrice:1]", "noise": "dimensionless"}
    DATASETS = {"assets_total": "fnd6", "close": "pv13", "atq": "fnd6",
                "noise": "fnd6"}

    def test_money_in_a_fundamental_dataset_is_a_level(self):
        self.assertIn("assets_total",
                      intent.derive_levels(self.UNITS, self.DATASETS, ("atq",)))

    def test_price_series_is_excluded_by_dataset(self):
        self.assertNotIn("close", intent.derive_levels(self.UNITS, self.DATASETS, ("atq",)))

    def test_size_measures_are_excluded(self):
        self.assertNotIn("atq", intent.derive_levels(self.UNITS, self.DATASETS, ("atq",)))

    def test_dimensionless_field_is_not_a_level(self):
        self.assertNotIn("noise", intent.derive_levels(self.UNITS, self.DATASETS, ()))

    def test_unmeasured_field_is_absent_rather_than_guessed(self):
        self.assertEqual(intent.derive_levels({}, {"whatever": "fnd6"}), [])

    def test_explicit_level_list_overrides_the_curated_file(self):
        program = fa.parse("group_rank(weird_field, subindustry)")
        self.assertEqual(intent.plan(program), [])
        plans = intent.plan(program, level_fields=["weird_field"])
        self.assertEqual(len(plans), 1)


if __name__ == "__main__":
    unittest.main()
