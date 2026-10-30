#!/usr/bin/env python3
"""Tests for the FASTEXPR parser, the sugar layer and the skeleton.

The parser is the one component every other tool now depends on, so these tests
pin the properties that matter: it must round-trip real expressions exactly, it
must not confuse a local variable with a field, and the sugar must vanish
completely by the time the string reaches the simulator.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fastexpr_ast as fa  # noqa: E402


class RoundTrip(unittest.TestCase):
    def test_simple_field(self):
        self.assertEqual(fa.parse("close").render(), "close")

    def test_assignment_and_signal(self):
        program = fa.parse("a = rank(close);\na - 0.5")
        self.assertEqual(len(program.assignments), 1)
        self.assertEqual(program.render(), "a = rank(close);\na - 0.5")

    def test_kwargs_survive(self):
        source = 'bucket(rank(close), range="0, 1, 0.1")'
        self.assertEqual(fa.parse(source).render(), source)

    def test_precedence_is_preserved(self):
        for source in ["(a + b) * c", "a + b * c", "a - (b - c)", "-(a - 0.5)",
                       "a / (b + 1)", "(a - b) / (abs(c) + 0.05)"]:
            self.assertEqual(fa.parse(source).render(), source, source)

    def test_ternary(self):
        source = "a > b ? a : b"
        self.assertEqual(fa.parse(source).render(), source)

    def test_comments_are_ignored(self):
        source = "// note\n/* block */\na = 1;\n# tail\na"
        self.assertEqual(fa.parse(source).render(), "a = 1;\na")

    def test_unbalanced_raises_with_offset(self):
        with self.assertRaises(fa.ParseError) as caught:
            fa.parse("rank(close")
        self.assertGreater(caught.exception.offset, 0)

    def test_trailing_junk_raises(self):
        with self.assertRaises(fa.ParseError):
            fa.parse("rank(close) 13")


class Fields(unittest.TestCase):
    def test_locals_are_not_fields(self):
        program = fa.parse("hv = group_rank(historical_volatility_60, subindustry) - 0.5;\n"
                           "-(abs(hv) - 0.25)")
        self.assertEqual(fa.field_names(program),
                         ["historical_volatility_60", "subindustry"])

    def test_operators_unique_in_order(self):
        program = fa.parse("a = ts_mean(close, 20);\nrank(a) + rank(open)")
        self.assertEqual(fa.operator_names(program), ["ts_mean", "rank"])


class Skeletons(unittest.TestCase):
    def test_same_shape_same_skeleton(self):
        left = fa.parse("group_rank(close, subindustry) - 0.5")
        right = fa.parse("group_rank(open, subindustry) - 0.5")
        self.assertEqual(fa.skeleton(left), fa.skeleton(right))

    def test_shape_change_changes_skeleton(self):
        left = fa.parse("rank(close)")
        right = fa.parse("-rank(close)")
        self.assertNotEqual(fa.skeleton(left), fa.skeleton(right))

    def test_literals_can_be_kept(self):
        left = fa.parse("ts_delta(close, 5)")
        right = fa.parse("ts_delta(close, 63)")
        self.assertEqual(fa.skeleton(left), fa.skeleton(right))
        self.assertNotEqual(fa.skeleton(left, keep_literals=True),
                            fa.skeleton(right, keep_literals=True))


class Transforms(unittest.TestCase):
    def test_substitute_inlines_a_local(self):
        program = fa.parse("x = rank(close);\nx - 0.5")
        replaced = fa.substitute(program, "x", fa.parse("rank(open)").signal)
        self.assertEqual(replaced.render(), "rank(open) - 0.5")

    def test_map_fields_renames(self):
        program = fa.parse("group_rank(close, subindustry)")
        renamed = fa.map_fields(program, {"close": "vwap"})
        self.assertEqual(renamed.render(), "group_rank(vwap, subindustry)")


if __name__ == "__main__":
    unittest.main()
