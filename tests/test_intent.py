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




if __name__ == "__main__":
    unittest.main()
