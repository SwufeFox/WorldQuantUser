#!/usr/bin/env python3
"""Tests for expression identity.

Identity by text is wrong in a direction that costs simulations: an expression rewritten
with redundant parentheses is the same experiment, and re-running it proves nothing.
Identity by tree fixes that, but only if it stops short of algebra -- `a + b` and
`b + a` are the same value and two different constructions, and merging them would
destroy the ledger's ability to tell experiments apart.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import fastexpr_ast as fa  # noqa: E402


class Canonical(unittest.TestCase):
    def test_redundant_parentheses_collapse(self):
        self.assertEqual(fa.canonical("((rank(close)))"), fa.canonical("rank(close)"))

    def test_whitespace_collapses(self):
        self.assertEqual(fa.canonical("  rank( close )  "), fa.canonical("rank(close)"))

    def test_comments_collapse(self):
        self.assertEqual(fa.canonical("// note\nrank(close)"), fa.canonical("rank(close)"))

    def test_literal_spelling_collapses(self):
        self.assertEqual(fa.canonical("rank(close) - 0.50"), fa.canonical("rank(close) - 0.5"))

    def test_algebraic_reorder_is_not_collapsed(self):
        self.assertNotEqual(fa.canonical("a + b"), fa.canonical("b + a"))

    def test_different_shape_is_not_collapsed(self):
        self.assertNotEqual(fa.canonical("rank(close)"), fa.canonical("-rank(close)"))

    def test_unparseable_text_falls_back_to_whitespace(self):
        self.assertEqual(fa.canonical("rank(close)   13"), "rank(close) 13")


class Hash(unittest.TestCase):
    def test_same_tree_same_hash(self):
        self.assertEqual(alpha_registry.expression_hash("rank(close)"),
                         alpha_registry.expression_hash("((rank(close)))"))

    def test_different_tree_different_hash(self):
        self.assertNotEqual(alpha_registry.expression_hash("rank(close)"),
                            alpha_registry.expression_hash("rank(open)"))

    def test_parseable_hashes_carry_the_version_prefix(self):
        self.assertTrue(alpha_registry.expression_hash("rank(close)").startswith("expr2:"))

    def test_damaged_text_keeps_the_old_prefix(self):
        # Historical rows used `expr:`; a damaged expression must not collide with a
        # parseable one that happens to canonicalise to the empty string.
        self.assertTrue(alpha_registry.expression_hash("rank(close) 13").startswith("expr:"))

    def test_same_expression_helper(self):
        self.assertTrue(alpha_registry.same_expression("a = rank(close);\na",
                                                       "a=rank(close);\r\na"))
        self.assertFalse(alpha_registry.same_expression("rank(close)", "rank(open)"))


class BuildDedupe(unittest.TestCase):
    """The macro builder must not re-simulate a rewrapped expression."""

    def test_parenthesised_variant_counts_as_already_built(self):
        import expression_macros

        macros = expression_macros.parse_macros("def pos():\n    ((close - low) / (high - low) - 0.5)\n")
        plan = [{"macro": "pos", "args": [], "decay": 4, "candidate_id": "A"}]
        built = {(fa.canonical("(close - low) / (high - low) - 0.5"), 4, "TOP3000")}
        alphas, skipped = expression_macros.build(macros, plan, built=built, universe="TOP3000")
        self.assertEqual(alphas, [])
        self.assertIn("duplicate", skipped[0]["reason"])


if __name__ == "__main__":
    unittest.main()