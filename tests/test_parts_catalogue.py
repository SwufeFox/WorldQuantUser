#!/usr/bin/env python3
"""Tests for the parts catalogue and seed screening.

Two properties matter and both were got wrong once while building this:

* a part must be a construction, not a fragment — `ts_mean(F, L)` appears in hundreds of
  expressions and says nothing;
* a seed must be measured by how many *book families* it serves, not by how many
  partners it has co-occurred with. Most book alphas are one expression with no named
  leg, so partner counts are zero by construction, and with a few dozen proven parts
  almost every pair is untried anyway.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import parts_catalogue as pc  # noqa: E402

TWO_LEG = ("x = ts_backfill(fnd6_newqv1300_ivltq, 252); "
           "group_rank(x/(ts_backfill(fnd6_cptnewqv1300_atq, 252) + 1), subindustry) "
           "- 0.5 - 0.5*rank((close - low) / (high - low) - 0.5)")


def entry(expression, fitness=None, active=False):
    return {"expression": expression, "status": "ACTIVE" if active else "UNSUBMITTED",
            "metrics": {"fitness": fitness} if fitness is not None else {}}


class Extraction(unittest.TestCase):
    def test_leg_and_combiner_are_both_extracted(self):
        parts = pc.parts_of(TWO_LEG, min_nodes=1)
        kinds = {kind for _s, _f, kind in parts}
        self.assertEqual(kinds, {"leg", "combiner"})

    def test_trivial_fragment_is_not_a_part(self):
        # `x = ts_mean(close, 20)` alone is vocabulary, not a construction.
        parts = pc.parts_of("x = ts_mean(close, 20); rank(x)", min_nodes=6)
        self.assertNotIn("leg", {kind for _s, _f, kind in parts})

    def test_fields_are_recorded_per_part(self):
        parts = pc.parts_of(TWO_LEG, min_nodes=1)
        fields = {f for _s, fs, _k in parts for f in fs}
        self.assertIn("fnd6_newqv1300_ivltq", fields)


class Catalogue(unittest.TestCase):
    def test_book_count_and_best_fitness(self):
        catalogue = pc.build([entry(TWO_LEG, fitness=1.7, active=True),
                              entry(TWO_LEG, fitness=1.1, active=False)])
        combiner = max(catalogue.values(), key=lambda s: s["n_expressions"])
        self.assertEqual(combiner["n_book"], 1)
        self.assertEqual(combiner["best_fitness"], 1.7)

    def test_same_shape_different_field_is_one_part(self):
        other = TWO_LEG.replace("fnd6_newqv1300_ivltq", "fnd6_newqv1300_ppegtq")
        catalogue = pc.build([entry(TWO_LEG, fitness=1.5), entry(other, fitness=1.4)])
        combiner = max(catalogue.values(), key=lambda s: s["n_expressions"])
        self.assertEqual(combiner["n_expressions"], 2)


class Screening(unittest.TestCase):
    def test_proven_untapped_dead_are_separated(self):
        catalogue = pc.build([
            entry(TWO_LEG, fitness=1.8, active=True),
            entry("y = ts_delta(ts_backfill(fnd6_newqv1300_ivltq, 252), 63) / "
                  "(abs(ts_delay(ts_backfill(fnd6_newqv1300_ivltq, 252), 63)) + 1); "
                  "rank(y)", fitness=1.9),
        ])
        proven, untapped, _dead = pc.screen(catalogue)
        self.assertTrue(proven)
        self.assertTrue(untapped)

    def test_portable_seed_requires_a_single_book_family(self):
        catalogue = pc.build([entry(TWO_LEG, fitness=1.7, active=True)])
        seeds = pc.portable_seeds(catalogue)
        self.assertTrue(seeds)
        self.assertTrue(all(slot["n_book"] == 1 for slot in seeds))

    def test_low_fitness_part_is_not_a_seed(self):
        catalogue = pc.build([entry(TWO_LEG, fitness=0.6, active=True)])
        self.assertEqual(pc.portable_seeds(catalogue), [])


if __name__ == "__main__":
    unittest.main()