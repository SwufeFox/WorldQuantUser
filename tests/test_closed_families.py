#!/usr/bin/env python3
"""Tests for the closed-direction list and the places that consume it.

The list exists because a tool recommended the option family first: it ranks top by
best-ever Fitness (F3.38, F3.89) and it is closed for concentration. Two properties
matter, and both were got wrong by hand before:

* a missing or damaged file must **not** stop a tool -- it degrades to "nothing is
  closed", and the absence is visible in the returned payload;
* a closed family must be **labelled and sorted last**, never silently dropped.
  Hiding the row is how it got recommended in the first place.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import closed_families as cf  # noqa: E402
import parts_catalogue as pc  # noqa: E402

#: A structurally different second construction: same idea, one leg fewer. Changing a
#: field would not do -- parts are indexed by shape, so two fields are one slot.
PLAIN = ("x = ts_backfill(fnd6_newqv1300_ivltq, 252); "
         "group_rank(x/(ts_backfill(fnd6_cptnewqv1300_atq, 252) + 1), subindustry) - 0.5")

LADDER = ("x = ts_backfill(fnd6_newqv1300_ivltq, 252); "
          "group_rank(x/(ts_backfill(fnd6_cptnewqv1300_atq, 252) + 1), subindustry) "
          "- 0.5 - 0.5*rank((close - low) / (high - low) - 0.5)")


def write_spec(root, closed):
    path = os.path.join(root, "closed.json")
    with open(path, "w") as handle:
        json.dump({"version": 1, "closed": closed}, handle)
    return path


class Loading(unittest.TestCase):
    def test_a_missing_file_is_an_empty_list_not_an_exception(self):
        payload = cf.load("/definitely/not/here.json")
        self.assertEqual(payload["closed"], [])
        self.assertIn("error", payload)

    def test_a_damaged_file_is_reported_and_survivable(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "closed.json")
        with open(path, "w") as handle:
            handle.write("{not json")
        self.assertEqual(cf.load(path)["closed"], [])
        self.assertEqual(cf.closed_families(path), set())

    def test_only_entries_with_a_family_match_count(self):
        directory = tempfile.mkdtemp()
        path = write_spec(directory, [
            {"id": "a", "match": {"family": "option"}, "reason": "concentration"},
            {"id": "b", "match": None, "reason": "a technique, not a family"},
        ])
        self.assertEqual(cf.closed_families(path), {"option"})
        self.assertEqual(cf.reason("option", path), "concentration")
        self.assertIsNone(cf.reason("quality", path))


class SeedsRespectIt(unittest.TestCase):
    def entry(self, family, fitness, expression=None):
        return {"expression": expression or LADDER, "status": "ACTIVE",
                "family": family, "alpha_id": family + "-1",
                "metrics": {"fitness": fitness}}

    def test_a_seed_from_a_closed_family_is_labelled_and_sorted_last(self):
        directory = tempfile.mkdtemp()
        cf_path = write_spec(directory, [{"id": "x", "match": {"family": "option"},
                                          "reason": "concentration"}])
        catalogue = pc.build([self.entry("option", 3.9),
                              self.entry("quality", 1.4, PLAIN)])
        rows = pc.portable_seeds(catalogue, min_fitness=0.0, spec=cf_path)
        self.assertGreaterEqual(len(rows), 2)
        self.assertIsNotNone(rows[-1].get("closed_reason"))
        self.assertIsNone(rows[0].get("closed_reason"))

    def test_without_the_file_nothing_is_marked_closed(self):
        catalogue = pc.build([self.entry("option", 3.9)])
        rows = pc.portable_seeds(catalogue, min_fitness=0.0, spec="/definitely/not/here")
        self.assertTrue(rows)
        self.assertTrue(all(row.get("closed_reason") is None for row in rows))


if __name__ == "__main__":
    unittest.main()