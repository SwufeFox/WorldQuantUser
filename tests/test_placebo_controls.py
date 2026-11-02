"""Tests for the frozen control-pool builder.

The stakes here are that a bad pool silently inflates every percentile computed
against it, so the tests pin down the two ways it went wrong in practice:
a near-neighbour pair counted as two independent controls, and a sample that spent
its whole budget inside one accounting family.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import placebo_controls as pc


def field(name, description, dataset="fundamental2", type_="MATRIX", delay=1,
          coverage=0.5, alpha_count=0):
    return {"id": name, "description": description,
            "dataset": {"id": dataset}, "type": type_, "delay": delay,
            "coverage": coverage, "alphaCount": alpha_count, "userCount": 0}


class TokenTests(unittest.TestCase):
    def test_stopwords_and_numbers_are_dropped(self):
        self.assertEqual(pc.tokens("Total amount of 2024 liabilities"),
                         frozenset({"liability"}))

    def test_plural_is_folded(self):
        self.assertEqual(pc.tokens("accrued liabilities"), pc.tokens("accrued liability"))

    def test_morphology_folds_an_acquisition_family(self):
        for word in ("acquire", "acquired", "acquisition", "acquiring"):
            self.assertEqual(pc.tokens(word), frozenset({"acquisition"}))


class NearTests(unittest.TestCase):
    def test_qualifier_only_difference_is_near(self):
        verdict, why = pc.is_near("Net deferred tax asset", "Gross deferred tax asset")
        self.assertTrue(verdict, why)

    def test_shared_concept_across_wording_is_near(self):
        verdict, why = pc.is_near(
            "Fair value of assets measured on a recurring basis, level 2",
            "Fair value of liabilities measured on a recurring basis, level 2")
        self.assertTrue(verdict, why)

    def test_unrelated_fields_are_not_near(self):
        verdict, _ = pc.is_near("Order backlog", "Lease minimum payments due in two years")
        self.assertFalse(verdict)


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.fields = [
            field("target", "Total liabilities incurred", coverage=0.50),
            field("same_band", "Order backlog", coverage=0.52),
            field("other_dataset", "Order backlog", dataset="fundamental6", coverage=0.50),
            field("other_type", "Order backlog", type_="VECTOR", coverage=0.50),
            field("other_delay", "Order backlog", delay=0, coverage=0.50),
            field("far_coverage", "Order backlog", coverage=0.90),
            field("twin", "Net liabilities incurred", coverage=0.50),
        ]

    def test_hard_filters_report_a_reason_each(self):
        pooled = pc.build_pool(self.fields, "target")
        reasons = {row["id"]: row["reason"].split(":")[0] for row in pooled["rejected"]}
        self.assertEqual(reasons["other_dataset"], "dataset_mismatch")
        self.assertEqual(reasons["other_type"], "type_mismatch")
        self.assertEqual(reasons["other_delay"], "delay_mismatch")
        self.assertEqual(reasons["far_coverage"], "coverage_out_of_band")

    def test_near_competitor_is_separated_from_the_null_pool(self):
        pooled = pc.build_pool(self.fields, "target")
        self.assertIn("twin", [row["id"] for row in pooled["near_competitors"]])
        self.assertNotIn("twin", pooled["pool"])

    def test_unassessed_requirements_are_named(self):
        pooled = pc.build_pool(self.fields, "target")
        self.assertIn("persistence", pooled["unassessed"])
        self.assertIn("missing_pattern", pooled["unassessed"])

    def test_usage_is_recorded_but_never_a_filter(self):
        fields = self.fields + [field("popular", "Order backlog", coverage=0.50,
                                      alpha_count=900)]
        pooled = pc.build_pool(fields, "target")
        self.assertIn("popular", pooled["pool"])


class SampleTests(unittest.TestCase):
    def test_one_field_per_cluster(self):
        pooled = {"clusters": {"a": ["a1", "a2"], "b": ["b1"], "c": ["c1"]}}
        picked = pc.sample(pooled, draws=10, seed=1)
        chosen = picked["chosen"]
        self.assertEqual(len(chosen), 3)
        self.assertEqual(sum(1 for name in chosen if name in ("a1", "a2")), 1)

    def test_shortfall_is_reported_not_padded(self):
        pooled = {"clusters": {"a": ["a1"], "b": ["b1"]}}
        picked = pc.sample(pooled, draws=20, seed=1)
        self.assertEqual(len(picked["chosen"]), 2)
        self.assertTrue(any("LOW-CONTROL-COUNT" in flag for flag in picked["flags"]))

    def test_same_seed_same_draw(self):
        pooled = {"clusters": {"a": ["a1"], "b": ["b1"], "c": ["c1"]}}
        self.assertEqual(pc.sample(pooled, 2, 7)["chosen"],
                         pc.sample(pooled, 2, 7)["chosen"])


class ReviewTests(unittest.TestCase):
    def test_merges_collapse_two_clusters(self):
        pooled = {"clusters": {"a": ["x1"], "b": ["x2"], "c": ["y1"]}}
        merged = pc.apply_review(pooled, {"merges": [["x1", "x2"]]})
        self.assertEqual(len(merged["clusters"]), 2)

    def test_exclusions_leave_the_pool(self):
        pooled = {"clusters": {"a": ["x1"], "b": ["y1"]}}
        merged = pc.apply_review(pooled, {"exclusions": {"x1": "leaks future info"}})
        self.assertNotIn("x1", [name for group in merged["clusters"].values()
                                for name in group])

    def test_review_is_recorded_on_the_pool(self):
        pooled = {"clusters": {"a": ["x1"]}}
        merged = pc.apply_review(pooled, {"merges": [], "reviewer": "me",
                                          "date": "2026-11-02"})
        self.assertEqual(merged["review"]["reviewer"], "me")


class ManifestTests(unittest.TestCase):
    def test_pool_json_round_trips(self):
        fields = [field("target", "Total liabilities"), field("a", "Order backlog")]
        pooled = pc.build_pool(fields, "target")
        payload = dict(pooled, sample=pc.sample(pooled, 5, 1))
        text = json.dumps(payload, default=list)
        self.assertEqual(json.loads(text)["target"]["id"], "target")


if __name__ == "__main__":
    unittest.main()