#!/usr/bin/env python3
"""Tests for the sugar layer: it must compile away, and stay stable when re-parsed."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alpha_sugar  # noqa: E402
import fastexpr_ast as fa  # noqa: E402


class Sugar(unittest.TestCase):
    def setUp(self):
        alpha_sugar.load()

    def test_band_is_the_inverted_u(self):
        self.assertEqual(alpha_sugar.fs("band(r)"), "-(abs(r) - 0.25)")

    def test_band_width_is_cosmetic_after_neutralisation(self):
        # The constant differs, the shape does not: this is the documented trap.
        left = alpha_sugar.fs("band(r, w=0.15)")
        right = alpha_sugar.fs("band(r, w=0.35)")
        self.assertNotEqual(left, right)
        self.assertEqual(fa.skeleton(fa.parse(left)), fa.skeleton(fa.parse(right)))

    def test_rrank_centres(self):
        self.assertEqual(alpha_sugar.fs("rrank(x, g)"), "group_rank(x, g) - 0.5")

    def test_combo_is_equal_weight(self):
        out = alpha_sugar.fs("combo(a, b)")
        self.assertEqual(out, "0.5 * (a + b)")

    def test_sugar_fully_disappears(self):
        out = alpha_sugar.fs("gate(close > avg(close, 60), rrank(-chg(close, 21), subindustry))")
        for name in alpha_sugar.load():
            self.assertNotIn(name + "(", out)

    def test_ratio_does_not_double_abs(self):
        self.assertEqual(alpha_sugar.fs("ratio(x, abs(y))"), "x / abs(y)")
        self.assertEqual(alpha_sugar.fs("ratio(x, y)"), "x / abs(y)")

    def test_expansion_output_is_stable(self):
        out = alpha_sugar.fs("deadband(band(rrank(close, subindustry)), cut=0.05)")
        self.assertEqual(fa.parse(out).render(), out)



if __name__ == "__main__":
    unittest.main()
