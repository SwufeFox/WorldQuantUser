#!/usr/bin/env python3
"""Tests for the correlation maths.

These are the calculations every independence claim rests on, so they get
deterministic fixtures rather than live data: a wrong sign or a silently short
overlap would invalidate the conclusion while looking entirely plausible.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

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

    def test_research_correlation_handles_extreme_finite_scales(self):
        from research_app import stats

        for scale in (1e-100, 1e100):
            with self.subTest(scale=scale):
                values = [scale, 2 * scale, 3 * scale]
                self.assertAlmostEqual(stats.correlation(values, values, min_points=3), 1.0)


def test_book_correlator_reuses_equal_date_series_and_aligns_different_dates():
    from research_app.brief import _book_correlator

    a = series([2, -1, 3, 0, 7])
    b = series([1, 4, 2, -1, 6])
    c = series([4, 1, 6, 2, 3])
    sparse = series([-3, 5, 1], start=2)
    query = _book_correlator({"a": a, "b": b, "c": c, "sparse": sparse})
    with mock.patch.object(correlation, "center_for_pearson", wraps=correlation.center_for_pearson) as prepare:
        pairs = [query("a", "b", 3), query("a", "c", 3), query("b", "c", 3)]
        assert prepare.call_count == 3
    assert pairs == [(correlation.pearson(list(left.values()), list(right.values())), len(left))
                     for left, right in [(a, b), (a, c), (b, c)]]
    x, y, days = correlation.align(a, sparse, min_overlap=3)
    assert query("a", "sparse", 3) == (correlation.pearson(x, y), len(days))
    assert query("a", "sparse", 4) == (None, 0)


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

    def test_skip_reason_separates_short_overlap_from_a_flat_series(self):
        # Waiting fixes a short overlap; nothing fixes a flat PnL. A caller that
        # retries the first and waits on the second wastes a run either way.
        pnls = {"a": series([1, 2, 3, 4]), "b": series([1, 2], start=3),
                "c": series([2, 2, 2, 2])}
        result = correlation.matrix(pnls, min_overlap=4)
        reasons = {(p["a"], p["b"]): p["reason"] for p in result["skipped"]}
        self.assertEqual(reasons[("a", "b")], "short_overlap")
        self.assertEqual(reasons[("a", "c")], "undefined")


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
