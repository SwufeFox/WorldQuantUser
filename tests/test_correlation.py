#!/usr/bin/env python3
"""Tests for the correlation maths.

These are the calculations every independence claim rests on, so they get
deterministic fixtures rather than live data: a wrong sign or a silently short
overlap would invalidate the conclusion while looking entirely plausible.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import correlation  # noqa: E402


def series(values, start=1):
    """Build ``{date: value}`` with zero-padded dates, easy to eyeball."""
    return {"2020-01-%02d" % (start + i): float(v) for i, v in enumerate(values)}


class AlignTests(unittest.TestCase):
    def test_common_dates_only(self):
        left = series([1, 2, 3])
        right = series([1, 2, 3], start=2)
        x, y, dates = correlation.align(left, right, min_overlap=1)
        self.assertEqual(dates, ["2020-01-02", "2020-01-03"])
        self.assertEqual(x, [2.0, 3.0])

    def test_short_overlap_is_refused_rather_than_averaged(self):
        left = series([1, 2, 3])
        right = series([1, 2, 3], start=2)
        self.assertEqual(correlation.align(left, right, min_overlap=200),
                         ([], [], []))


class PearsonTests(unittest.TestCase):
    def test_perfect_positive(self):
        self.assertAlmostEqual(correlation.pearson([1, 2, 3], [2, 4, 6]), 1.0)

    def test_perfect_negative(self):
        self.assertAlmostEqual(correlation.pearson([1, 2, 3], [6, 4, 2]), -1.0)

    def test_constant_series_is_undefined_not_zero(self):
        # A flat PnL series has no correlation with anything. Reporting 0.0 would
        # read as "measured uncorrelated" and quietly pass a duplicate through.
        self.assertIsNone(correlation.pearson([1, 1, 1], [1, 2, 3]))

    def test_length_mismatch_is_undefined(self):
        self.assertIsNone(correlation.pearson([1, 2, 3], [1, 2]))


class MatrixTests(unittest.TestCase):
    def test_every_pair_is_reported(self):
        pnls = {"a": series([1, 2, 3, 4]), "b": series([2, 4, 6, 8]),
                "c": series([4, 3, 2, 1])}
        result = correlation.matrix(pnls, min_overlap=1)
        pairs = {(p["a"], p["b"]): p["r"] for p in result["pairs"]}
        self.assertEqual(len(pairs), 3)
        self.assertAlmostEqual(pairs[("a", "b")], 1.0)
        self.assertAlmostEqual(pairs[("a", "c")], -1.0)

    def test_short_pairs_are_listed_not_dropped(self):
        # A missing pair and a zero pair mean different things; the caller has to
        # be able to tell them apart.
        pnls = {"a": series([1, 2, 3, 4]), "b": series([1, 2], start=3)}
        result = correlation.matrix(pnls, min_overlap=4)
        self.assertEqual(result["measured"], 0)
        self.assertEqual(len(result["skipped"]), 1)


class MaxAgainstTests(unittest.TestCase):
    def test_highest_absolute_wins_and_order_is_stable(self):
        target = series([1, 2, 3, 4])
        refs = {"same": series([2, 4, 6, 8]), "mirror": series([4, 3, 2, 1]),
                "flat": series([0, 0, 0, 0])}
        result = correlation.max_against(target, refs, min_overlap=1)
        self.assertEqual(result["max"]["with"], "same")
        self.assertAlmostEqual(result["max"]["r"], 1.0)
        self.assertEqual(len(result["all"]), 2)  # the flat series has no correlation


class ResidualizeTests(unittest.TestCase):
    def test_shared_factor_is_removed(self):
        control = series([1, 2, 3, 4, 5, 6, 7, 8])
        # Two series dominated by one shared factor, with unrelated small noise.
        # The noise must not be the mirror image of itself, or the residuals come
        # out perfectly anti-correlated and the fixture proves nothing.
        noise_a = [0.31, -0.72, 0.45, 0.11, -0.28, 0.66, -0.53, 0.19]
        noise_b = [-0.44, 0.27, 0.61, -0.35, 0.52, -0.18, 0.24, -0.63]
        a = {"2020-01-%02d" % (i + 1): 3.0 * (i + 1) + noise_a[i]
             for i in range(8)}
        b = {"2020-01-%02d" % (i + 1): 2.0 * (i + 1) + noise_b[i]
             for i in range(8)}
        raw = correlation.pearson([a[d] for d in sorted(a)],
                                 [b[d] for d in sorted(b)])
        self.assertGreater(raw, 0.9)
        ra = correlation.residualize(a, control, min_overlap=4)
        rb = correlation.residualize(b, control, min_overlap=4)
        x, y, _ = correlation.align(ra, rb, min_overlap=4)
        self.assertLess(abs(correlation.pearson(x, y)), 0.5)

    def test_residual_keeps_the_date_index(self):
        control = series([1, 2, 3, 4])
        target = series([2, 4, 6, 8])
        residual = correlation.residualize(target, control, min_overlap=2)
        self.assertEqual(sorted(residual), sorted(target))


if __name__ == "__main__":
    unittest.main()
