#!/usr/bin/env python3
"""Tests for the tree-backed macro expander, including the two text defects it fixes.

The point of these tests is not that the new backend works -- it is that the old
one was wrong in two specific, silent ways, and that the new one is right on
exactly those cases while staying identical everywhere else.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import ast_macros  # noqa: E402
import expression_macros as text_macros  # noqa: E402
import fastexpr_ast as fa  # noqa: E402

SOURCE = """
def growth(f, w=63, b=252):
    ts_delta(ts_backfill({f}, {b}), {w}) / (abs(ts_delay(ts_backfill({f}, {b}), {w})) + 1)

def position():
    (close - low) / (high - low) - 0.5

def carrier(f, fast, fw=1.0, w=63):
    -rank(growth({f}, {w})) - {fw}*rank({fast})

def scaled(fast, k=2):
    {fast} * {k}

def share(f, fast, fw=0.5, v=x):
    {v} = ts_backfill({f}, 252); group_rank({v}/(ts_backfill(atq, 252) + 1), subindustry) - 0.5 - {fw}*rank({fast})

def loop_a(x):
    loop_b({x})

def loop_b(x):
    loop_a({x})
"""


class Parity(unittest.TestCase):
    """Where the old backend was already right, both must agree."""

    def setUp(self):
        self.ast = ast_macros.parse_macros(SOURCE)
        self.text = text_macros.parse_macros(SOURCE)

    def assert_same(self, name, args):
        old = self.text_expand(name, args)
        new = ast_macros.expand(self.ast, name, args)
        self.assertEqual(fa.parse(old).render(), new, "%s%s" % (name, tuple(args)))

    def text_expand(self, name, args):
        return text_macros.expand(self.text, name, args)

    def test_plain_field_argument(self):
        self.assert_same("growth", ["fnd6_x"])

    def test_nested_zero_arg_macro(self):
        self.assert_same("carrier", ["fnd6_x", "position()"])

    def test_numeric_defaults(self):
        self.assert_same("growth", ["fnd6_x", "21", "126"])


class TextDefects(unittest.TestCase):
    """The two failures the tree backend exists to remove."""

    def setUp(self):
        self.ast = ast_macros.parse_macros(SOURCE)
        self.text = text_macros.parse_macros(SOURCE)

    def test_text_leaks_a_nested_macro_when_the_argument_has_parentheses(self):
        # The call regex cannot match '(...)' inside an argument list, so the
        # nested call survives into the output and the simulator rejects the name.
        old = text_macros.expand(self.text, "carrier", ["(a + b) / 2", "position()"])
        self.assertIn("growth(", old)

        new = ast_macros.expand(self.ast, "carrier", ["(a + b) / 2", "position()"])
        self.assertNotIn("growth(", new)

    def test_text_loses_parentheses_around_a_compound_argument(self):
        # body is '{fast} * {k}', argument is 'a - b'
        old = text_macros.expand(self.text, "scaled", ["a - b"])
        new = ast_macros.expand(self.ast, "scaled", ["a - b"])
        self.assertEqual(old, "a - b * 2")
        self.assertEqual(new, "(a - b) * 2")

    def test_tree_output_is_always_parseable(self):
        for args in (["a - b"], ["(a + b) / 2"], ["close"]):
            out = ast_macros.expand(self.ast, "scaled", args)
            self.assertEqual(fa.parse(out).render(), out)


class Acceptance(unittest.TestCase):
    """The gate for flipping build() onto the tree backend."""

    def test_no_unexplained_difference_on_the_shipped_macros(self):
        import tools.macro_parity as parity

        text = ast_macros.parse_macros(SOURCE)
        reference = text_macros.parse_macros(SOURCE)
        risky = parity.non_atomic_params(text["carrier"]["ast"])
        self.assertIn("fw", risky)  # {fw} sits directly against an operator
        for args in (["fnd6_x", "position()"], ["fnd6_x", "a - b"],
                     ["(a + b) / 2", "position()"]):
            kind, _detail = parity.classify(text, reference, "carrier", text["carrier"], args)
            # A leak is an old-backend defect too, so it is acceptable here; what
            # must never appear is a difference nobody can explain.
            self.assertIn(kind, {"equal", "diff_explained", "old_leaks_macro"},
                          "carrier%s -> %s" % (args, kind))


class Structure(unittest.TestCase):
    def setUp(self):
        self.ast = ast_macros.parse_macros(SOURCE)

    def test_parameter_used_as_a_variable_is_renamed(self):
        out = ast_macros.expand(self.ast, "share", ["fnd6_x", "position()", "0.5", "v"])
        self.assertTrue(out.startswith("v = ts_backfill(fnd6_x, 252);"))
        self.assertIn("group_rank(v /", out)

    def test_non_identifier_rejected_only_for_variable_parameters(self):
        with self.assertRaises(ast_macros.AstMacroError):
            ast_macros.expand(self.ast, "share", ["fnd6_x", "position()", "0.5", "a - b"])
        # 'fast' is not used as a variable, so a compound value is fine there.
        ast_macros.expand(self.ast, "share", ["fnd6_x", "a - b", "0.5", "v"])

    def test_cycle_is_reported(self):
        with self.assertRaises(ast_macros.AstMacroError):
            ast_macros.expand(self.ast, "loop_a", ["x"])

    def test_missing_argument_is_reported(self):
        with self.assertRaises(ast_macros.AstMacroError):
            ast_macros.expand(self.ast, "growth", [])


if __name__ == "__main__":
    unittest.main()