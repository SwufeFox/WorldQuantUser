#!/usr/bin/env python3
"""Tests for the sugar recogniser.

The recogniser rewrites expressions back into sugar syntax, so any mistake prints a
*different formula* and reports it as the same one. Two such mistakes were made while
building it, both about parentheses, and both are pinned here:

* `(close - low) / (high - low)` printed as `close - low - high - close / high - low`;
* `-(a - b)` printed as `-a - b`, which flips the sign of the second term.

The faithfulness check in the tool compares canonical forms, which is what caught them.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import alpha_sugar  # noqa: E402,F401  (registers the macros)
import fastexpr_ast as fa  # noqa: E402
import sugar_rewrite as sr  # noqa: E402


def rewrite(expression):
    program = fa.parse(expression)
    parts = []
    for name, value in program.assignments:
        parts.append("%s = %s" % (name, sr.sugar_of(value)[0]))
    if program.signal is not None:
        parts.append(sr.sugar_of(program.signal)[0])
    return ";\n".join(parts)


class Faithfulness(unittest.TestCase):
    def assertSame(self, expression):
        self.assertEqual(fa.canonical(rewrite(expression)), fa.canonical(expression),
                         "rewrite changed the meaning: %s" % rewrite(expression))

    def test_group_rank_minus_half_becomes_rrank(self):
        out = rewrite("group_rank(close, subindustry) - 0.5")
        self.assertIn("rrank(close, subindustry)", out)
        self.assertSame("group_rank(close, subindustry) - 0.5")

    def test_band_shape_is_recognised(self):
        expression = "-(abs(group_rank(historical_volatility_60, subindustry) - 0.5) - 0.25)"
        self.assertIn("band(", rewrite(expression))
        self.assertSame(expression)

    def test_ratio_shape_is_recognised(self):
        expression = "eps_high - eps_low / (abs(eps_mean) + 0.05)"
        self.assertSame(expression)

    def test_unary_minus_keeps_parentheses(self):
        # regression: `-(a - b)` must not print as `-a - b`
        out = rewrite("rank(-(news_a - news_b))")
        self.assertIn("-(", out)
        self.assertSame("rank(-(news_a - news_b))")

    def test_division_keeps_parentheses(self):
        # regression: `(close - low) / (high - low)` must not print as a flat expression
        expression = "((close - low) / (high - low)) - 0.5"
        self.assertSame(expression)

    def test_lag_and_gate_are_recognised(self):
        self.assertIn("lag(", rewrite("ts_delay(close, 1)"))
        self.assertIn("gate(", rewrite("if_else(close > open, close, 0)"))


class Counting(unittest.TestCase):
    def test_hits_count_recognised_forms(self):
        _text, hits = sr.sugar_of(fa.parse("group_rank(close, subindustry) - 0.5").signal)
        self.assertEqual(hits, 1)

    def test_plain_expression_has_no_hits(self):
        _text, hits = sr.sugar_of(fa.parse("close").signal)
        self.assertEqual(hits, 0)


if __name__ == "__main__":
    unittest.main()