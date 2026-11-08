#!/usr/bin/env python3
"""Tests for the ledger search extensions: regex and multi-field matching.

`--field` is a substring test, which answers "has this id been used". Two questions it
cannot answer came up on 2026-11-08: "any ts_corr of an absolute series" (needs a
pattern) and "expressions that use both debt_st and debt_lt" (needs all-of).
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import alpha_registry


def _entry(alpha_id, expression):
    return alpha_registry.build_entry(
        {"alpha_id": alpha_id, "expression": expression, "metrics": {},
         "status": "COMPLETE"}, verdict="candidate")


class LedgerSearchTests(unittest.TestCase):
    def setUp(self):
        self.entries = [
            _entry("A1", "rank((debt_st + debt_lt) / assets)"),
            _entry("A2", "rank(debt_st / assets)"),
            _entry("A3", "ts_corr(abs(returns), ts_delay(abs(returns), 1))"),
        ]

    def test_all_fields_requires_every_id(self):
        matched = alpha_registry.find_entries(self.entries, all_fields="debt_st,debt_lt")
        self.assertEqual([e["alpha_id"] for e in matched], ["A1"])

    def test_regex_matches_a_shape_not_a_token(self):
        matched = alpha_registry.find_entries(self.entries, expression=r"ts_corr\(abs\(",
                                              regex=True)
        self.assertEqual([e["alpha_id"] for e in matched], ["A3"])

    def test_a_bad_regex_is_reported(self):
        with self.assertRaises(ValueError):
            alpha_registry.find_entries(self.entries, expression="([", regex=True)

    def test_substring_still_works_when_regex_is_off(self):
        matched = alpha_registry.find_entries(self.entries, expression="debt_lt")
        self.assertEqual([e["alpha_id"] for e in matched], ["A1"])


if __name__ == "__main__":
    unittest.main()
