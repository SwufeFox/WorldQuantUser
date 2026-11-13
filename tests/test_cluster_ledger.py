#!/usr/bin/env python3
"""Tests for the greedy independent set in the cluster ledger.

The set is the answer to "how many separate bets do I actually hold", and it is a
lower bound by construction, so its two failure modes cost in opposite directions.
Taking two near-duplicates overstates the count; refusing an alpha because of a
pair that never enters the set understates it for no evidence at all. Review found
the second: one unmeasurable pair removed *both* of its alphas, so a single gap
could cost two members.

What the set claims is narrower than what it used to check - its own members are
pairwise distinct - so the requirement is local too.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cluster_ledger as cl  # noqa: E402


def pair(a, b, r):
    return {"a": a, "b": b, "r": r, "n": 500}


class IndependentSetTests(unittest.TestCase):
    def test_gap_between_two_outsiders_keeps_both(self):
        # A-B 0.10 and A-C 0.20 are measured; B-C never was. B's open question is
        # with C, which the set does not contain, so B stays and only C is held out.
        chosen, refused = cl.independent_set(
            ["A", "B", "C"], [pair("A", "B", 0.10), pair("A", "C", 0.20)],
            0.7, {"A": 2.0, "B": 1.8, "C": 1.5})
        self.assertEqual(chosen, ["A", "B"])
        self.assertEqual([row["alpha_id"] for row in refused], ["C"])
        self.assertEqual(refused[0]["reason"], "unmeasured")
        self.assertEqual(refused[0]["with"], "B")
        self.assertIsNone(refused[0]["r"])

    def test_one_gap_does_not_cost_two_members(self):
        # The old rule held out every alpha named in any skipped pair, so this
        # input returned just ["A"]. B is measured against A and that is all the
        # set needs to say about it.
        chosen, refused = cl.independent_set(
            ["A", "B"], [pair("A", "B", 0.10)], 0.7, {"A": 2.0, "B": 1.8})
        self.assertEqual(chosen, ["A", "B"])
        self.assertEqual(refused, [])

    def test_candidate_facing_a_gap_with_a_member_is_refused(self):
        # Nothing measured at all: A is taken on priority, C cannot be shown
        # distinct from it, so C is refused rather than assumed independent.
        chosen, refused = cl.independent_set(["A", "C"], [], 0.7, {"A": 2.0, "C": 1.0})
        self.assertEqual(chosen, ["A"])
        self.assertEqual(refused[0]["reason"], "unmeasured")
        self.assertEqual(refused[0]["with"], "A")

    def test_near_duplicate_is_refused_with_its_correlation(self):
        chosen, refused = cl.independent_set(
            ["A", "B"], [pair("A", "B", -0.85)], 0.7, {"A": 2.0, "B": 1.9})
        self.assertEqual(chosen, ["A"])
        self.assertEqual(refused[0]["reason"], "dependent")
        self.assertAlmostEqual(refused[0]["r"], -0.85)
        self.assertEqual(refused[0]["with"], "A")

    def test_priority_decides_which_of_two_duplicates_survives(self):
        chosen, refused = cl.independent_set(
            ["A", "B"], [pair("A", "B", 0.95)], 0.7, {"A": 1.0, "B": 2.0})
        self.assertEqual(chosen, ["B"])
        self.assertEqual(refused[0]["alpha_id"], "A")

    def test_a_gap_mid_pass_does_not_cascade(self):
        # Every pair is below the line except B-D, which has no measurement. C-D
        # is fine, so D is refused on B alone and C is unaffected.
        chosen, refused = cl.independent_set(
            ["A", "B", "C", "D"],
            [pair("A", "B", 0.10), pair("A", "C", 0.20), pair("A", "D", 0.30),
             pair("B", "C", 0.15), pair("C", "D", 0.25)],
            0.7, {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
        self.assertEqual(chosen, ["A", "B", "C"])
        self.assertEqual([(row["alpha_id"], row["reason"]) for row in refused],
                         [("D", "unmeasured")])

    def test_a_fully_measured_clean_pool_keeps_everyone(self):
        chosen, refused = cl.independent_set(
            ["A", "B", "C"],
            [pair("A", "B", 0.10), pair("A", "C", 0.20), pair("B", "C", 0.30)],
            0.7, {"A": 3.0, "B": 2.0, "C": 1.0})
        self.assertEqual(chosen, ["A", "B", "C"])
        self.assertEqual(refused, [])


class SpectralSummaryTests(unittest.TestCase):
    def test_an_equicorrelated_pool_is_fewer_bets_than_names(self):
        # Diffuse correlation can reduce effective bets even when no pair crosses
        # the clustering threshold.
        ids = ["A", "B", "C", "D"]
        pairs = [pair(a, b, 0.3) for a, b in
                 ("AB", "AC", "AD", "BC", "BD", "CD")]
        summary = cl.cluster_ledger_spectral(ids, pairs) if hasattr(cl, "cluster_ledger_spectral") \
            else cl.spectral_summary(ids, pairs)
        self.assertLess(summary["effective_bets"], len(ids))
        self.assertGreater(summary["pc1_share"], 1.0 / len(ids))

    def test_perfectly_independent_names_are_their_own_bets(self):
        ids = ["A", "B", "C"]
        summary = cl.spectral_summary(ids, [pair("A", "B", 0.0), pair("A", "C", 0.0),
                                            pair("B", "C", 0.0)])
        self.assertAlmostEqual(summary["effective_bets"], 3.0, places=1)
        self.assertAlmostEqual(summary["pc1_share"], 1.0 / 3, places=2)

    def test_a_single_name_has_no_spread(self):
        summary = cl.spectral_summary(["A"], [])
        self.assertIsNone(summary["effective_bets"])


if __name__ == "__main__":
    unittest.main()