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
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import fastexpr_ast as fa  # noqa: E402
from research_app import alphas, ledger, mechanisms  # noqa: E402


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
        self.assertTrue(alpha_registry.expression_hash("rank(close) 13").startswith("text:"))

    def test_same_expression_helper(self):
        self.assertTrue(alpha_registry.same_expression("a = rank(close);\na",
                                                       "a=rank(close);\r\na"))
        self.assertFalse(alpha_registry.same_expression("rank(close)", "rank(open)"))

    def test_python_hash_preserves_indentation_and_string_whitespace(self):
        source = "if cond:\n    out = 'a b'\n"
        equal = alpha_registry.expression_hash(source.replace("\n", "\r\n"), "PYTHON")
        digest = alpha_registry.expression_hash(source, "python")
        self.assertEqual(digest, equal)
        self.assertTrue(digest.startswith("py:"))
        for different in (source.replace("    ", "  "), source.replace("a b", "a  b"), source.rstrip()):
            self.assertNotEqual(alpha_registry.expression_hash(different, "PYTHON"), digest)
            self.assertFalse(alpha_registry.same_expression(different, source, "PYTHON"))

    def test_identical_source_in_different_languages_has_distinct_placeholder_key(self):
        source = "rank(close)"
        fast = alpha_registry.build_entry({"expression": source})
        python = alpha_registry.build_entry({"expression": source, "settings": {"language": "PYTHON"}})
        self.assertNotEqual(fast["key"], python["key"])
        self.assertNotEqual(fast["expression_hash"], python["expression_hash"])
        self.assertIsNone(mechanisms.mechanism_key(source, "PYTHON"))

    def test_old_python_hash_kept_on_disk_but_new_exact_identity_is_available(self):
        source = "rank(close)"
        old = alpha_registry.expression_hash(source)
        row = {"expression": source, "expression_hash": old, "settings": {"language": "PYTHON"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            path.write_text(json.dumps(row) + "\n")
            entries = alpha_registry.load_entries(path)
            self.assertEqual(entries[0]["expression_hash"], old)
            self.assertTrue(entries[0]["key"].startswith("py:"))
            self.assertEqual(alpha_registry.resolve(entries, alpha_registry.expression_hash(source, "PYTHON")),
                             entries[0])
            self.assertEqual(json.loads(path.read_text()), row)


class PythonLedgerSource(unittest.TestCase):
    def test_record_preserves_native_python_source_and_skip_fast_expr_format(self):
        source = "\nif cond:\n    out = 'a  b'\n"
        payload = {"id": "PYALPHA", "regular": {"language": "PYTHON", "code": source},
                   "settings": {"region": "USA"}, "is": {"sharpe": 1.5}, "status": "ACTIVE"}
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(alpha_id="PYALPHA", registry=str(Path(tmp) / "ledger.jsonl"), dry_run=True)
            with mock.patch.object(ledger, "with_session", return_value=payload):
                result = ledger.cmd_ledger_record(args, None)
            self.assertEqual(result["entry"]["expression"], source)
            self.assertEqual(result["entry"]["settings"]["language"], "PYTHON")
            self.assertTrue(result["entry"]["expression_hash"].startswith("py:"))
            with mock.patch("expression_lint.pretty_expression") as fmt:
                report = ledger.cmd_ledger_format(args, None)
            fmt.assert_not_called()
            self.assertEqual(report["changed"], 0)

    def test_alpha_book_exact_match_uses_proven_source_language(self):
        code = "rank(close)"
        fast = alpha_registry.build_entry({"alpha_id": "FAST", "expression": code}, mechanism="fast")
        python = alpha_registry.build_entry({"alpha_id": "PY", "expression": code, "language": "PYTHON"},
                                            mechanism="python")
        payload = {"id": "NEW", "regular": {"language": "PYTHON", "code": code}, "status": "ACTIVE"}
        args = Namespace(registry=None, limit=50, check=None)
        with mock.patch.object(alpha_registry, "load_entries", return_value=[fast, python]), \
                mock.patch.object(alphas, "with_session", return_value=[payload]):
            row = alphas.cmd_alpha_book(args, None)["alphas"][0]
        self.assertEqual(row["language"], "PYTHON")
        self.assertEqual(row["mechanism"], "python")


if __name__ == "__main__":
    unittest.main()
