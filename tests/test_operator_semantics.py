#!/usr/bin/env python3
"""Tests for the curated invariants the operator catalog does not publish."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import operator_semantics as semantics  # noqa: E402

RULES = {
    "returns": {"bucket": "Group"},
    "constraints": [
        {"operator": "kth_element", "kind": "le", "left": "k", "right": "d"},
        {"operator": "bucket", "kind": "exactly_one", "keys": ["range", "buckets"]},
        {"operator": "bucket", "kind": "increasing", "key": "buckets"},
    ],
}

INDEX = {
    "kth_element": {"definition": "kth_element(x, d, k, ignore=\"NaN\")"},
    "bucket": {"definition": "bucket(rank(x), range=\"0,1,0.1\", buckets=\"2,5\")"},
}


def codes(expression):
    found = semantics.rule_findings(expression, INDEX, RULES)
    found += semantics.return_type_findings(expression, INDEX, RULES)
    return sorted({item["code"] for item in found})


class Constraints(unittest.TestCase):
    def test_le_violation_is_caught(self):
        self.assertEqual(codes("kth_element(close, 5, k=7)"),
                         ["operator_arg_constraint"])

    def test_le_boundary_is_allowed(self):
        self.assertEqual(codes("kth_element(close, 5, k=5)"), [])

    def test_both_range_and_buckets_rejected(self):
        # bucket() at top level also trips signal_type, which is correct: it
        # returns a Group. The constraint is what this test is about.
        self.assertIn("operator_arg_constraint",
                      codes('group_rank(close, bucket(rank(close), range="0,1,0.1", buckets="0,0.5,1"))'))

    def test_non_increasing_buckets_rejected(self):
        self.assertIn("operator_arg_constraint",
                      codes('group_rank(close, bucket(rank(close), buckets="0,0.5,0.4"))'))

    def test_increasing_buckets_allowed(self):
        self.assertEqual(
            codes('group_rank(close, bucket(rank(close), buckets="0,0.5,1"))'), [])

    def test_non_matrix_signal_rejected(self):
        self.assertEqual(codes('bucket(rank(close), range="0,1,0.1")'), ["signal_type"])

    def test_unknown_parameter_name_skips_instead_of_misfiring(self):
        # Definition cannot be indexed, so the rule must stay silent.
        self.assertEqual(semantics.rule_findings("kth_element(close, 5, k=7)", {}, RULES), [])


class CommentProse(unittest.TestCase):
    """The regex identifier scan cannot tell a comment from code; the tree can."""

    def test_comment_words_are_not_fields(self):
        import expression_lint

        expression = ("//leverage = liabilities / assets;\r\n\r\n"
                      "FLEV = -rank(equity / assets);\r\nFLEV")
        report = expression_lint.lint_expression(
            expression, known_fields=set(), strict_fields=True)
        unknown = {item["identifier"] for item in report["errors"]
                   if item["code"] == "unknown_identifier"}
        self.assertNotIn("leverage", unknown)
        self.assertIn("equity", unknown)

    def test_crlf_assignment_is_a_local_not_a_field(self):
        import expression_lint

        expression = "# day1\r\nd1_mean = ts_mean(close, 20);\r\nd1_mean"
        report = expression_lint.lint_expression(
            expression, known_fields=set(), strict_fields=True)
        unknown = {item["identifier"] for item in report["errors"]
                   if item["code"] == "unknown_identifier"}
        self.assertNotIn("d1_mean", unknown)


class PackagedRules(unittest.TestCase):
    def test_packaged_default_exists(self):
        self.assertTrue(os.path.exists(semantics.packaged_default_path()))

    def test_load_prefers_an_explicit_path(self):
        rules = semantics.load_rules(semantics.packaged_default_path())
        self.assertTrue(rules["present"])
        self.assertTrue(rules["constraints"])


if __name__ == "__main__":
    unittest.main()
