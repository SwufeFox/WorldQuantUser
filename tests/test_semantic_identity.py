#!/usr/bin/env python3
"""Tests for the conservative semantic identity used to count distinct ideas."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import semantic_identity as si


class IdentityTests(unittest.TestCase):
    def test_positive_scale_is_not_signal(self):
        self.assertTrue(si.same_identity("2 * rank(close)", "rank(close)"))
        self.assertTrue(si.same_identity("0.5 * rank(close)", "rank(close)"))
        self.assertTrue(si.same_identity("rank(close) / 2", "rank(close)"))

    def test_sign_is_direction(self):
        self.assertTrue(si.same_identity("-2 * rank(close)", "-rank(close)"))
        self.assertFalse(si.same_identity("-rank(close)", "rank(close)"))

    def test_constant_offset_is_identity_only_without_demeaning(self):
        self.assertTrue(si.same_identity("rank(close) - 0.5", "rank(close)"))
        self.assertFalse(si.same_identity("rank(close) - 0.5", "rank(close)", neutralization="NONE"))

    def test_commutative_arguments_compare_unordered(self):
        self.assertTrue(si.same_identity("rank(close) + rank(volume)", "rank(volume) + rank(close)"))
        self.assertTrue(si.same_identity("max(rank(close), rank(volume))", "max(rank(volume), rank(close))"))

    def test_nested_offset_is_not_dropped(self):
        self.assertFalse(si.same_identity("rank(close - 0.5)", "rank(close)"))

    def test_fingerprint_is_stable_and_distinguishes_direction(self):
        first = si.fingerprint("2 * rank(close)")
        second = si.fingerprint("rank(close)")
        opposite = si.fingerprint("-rank(close)")
        self.assertEqual(first, second)
        self.assertNotEqual(first, opposite)


if __name__ == "__main__":
    unittest.main()
