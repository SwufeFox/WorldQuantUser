#!/usr/bin/env python3
"""Tests for field_index: mechanical grouping with no baked vocabulary.

The design constraint these tests protect is that the module carries **no
knowledge of any particular catalog's naming**. A curated prefix/alias table
inside the tool would encode one snapshot, rot silently when families are
renamed, and mis-group exactly the fields that matter. So the tests assert both
that grouping works when the caller supplies ignore words, and that nothing
collapses on its own when the caller supplies none.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import field_index as fi  # noqa: E402


def field(fid, coverage, alpha_count=0, field_type="MATRIX", dataset="analyst4"):
    return {
        "id": fid,
        "coverage": coverage,
        "alphaCount": alpha_count,
        "type": field_type,
        "dataset": dataset,
    }


class TokenizeTests(unittest.TestCase):
    def test_splits_on_non_alphanumeric(self):
        self.assertEqual(fi.tokenize("anl4_afv4_cfps_number"),
                         ["anl4", "afv4", "cfps", "number"])

    def test_keeps_digits_because_they_may_be_meaningful(self):
        self.assertEqual(fi.tokenize("price_change_pct_2"),
                         ["price", "change", "pct", "2"])

    def test_empty_input_is_empty_list(self):
        self.assertEqual(fi.tokenize(""), [])
        self.assertEqual(fi.tokenize(None), [])


class ConceptKeyTests(unittest.TestCase):
    def test_key_is_order_insensitive(self):
        self.assertEqual(fi.concept_key("cfps_estimate_count"),
                         fi.concept_key("count_cfps_estimate"))

    def test_ignore_words_are_removed(self):
        key = fi.concept_key("anl4_afv4_cfps_number", ignore_words=["anl4", "afv4"])
        self.assertEqual(key, "cfps number")

    def test_ignore_list_is_caller_supplied_not_built_in(self):
        # With no ignore words the two families must stay distinct. This is the
        # regression guard for the design rule: the tool must not quietly own a
        # vocabulary that makes them look the same.
        left = fi.concept_key("cashflow_per_share_estimate_count")
        right = fi.concept_key("anl4_afv4_cfps_number")
        self.assertNotEqual(left, right)

    def test_identical_ids_produce_identical_keys(self):
        self.assertEqual(fi.concept_key("close"), fi.concept_key("close"))

    def test_numeric_tokens_kept_unless_drop_numeric_requested(self):
        self.assertNotEqual(fi.concept_key("cfps_number"),
                            fi.concept_key("cfps_number_2"))
        self.assertEqual(fi.concept_key("cfps_number", drop_numeric=True),
                         fi.concept_key("cfps_number_2", drop_numeric=True))

    def test_all_ignored_words_yield_empty_key(self):
        self.assertEqual(fi.concept_key("anl4_afv4", ignore_words=["anl4", "afv4"]), "")

    def test_tail_tokens_treats_leading_words_as_boilerplate(self):
        # Mechanical positional rule, no vocabulary: the trailing words are the
        # concept, everything before them is family noise.
        left = fi.concept_key("anl4_afv4_cfps_mean", tail_tokens=2)
        right = fi.concept_key(
            "anl4_fs_detail_estimates_basic_af_v4_nd_cfps_mean", tail_tokens=2)
        self.assertEqual(left, "cfps mean")
        self.assertEqual(left, right)

    def test_tail_tokens_zero_uses_every_token(self):
        key = fi.concept_key("a_b_c", tail_tokens=0)
        self.assertEqual(key, "a b c")


class GroupingTests(unittest.TestCase):
    def test_group_sorts_best_coverage_first(self):
        records = [
            field("cfps_number", 0.3977),
            field("anl4_afv4_cfps_number", 0.6751),
        ]
        groups = fi.group_by_concept(records, ignore_words=["anl4", "afv4"])
        group = groups["cfps number"]
        self.assertEqual([r["id"] for r in group],
                         ["anl4_afv4_cfps_number", "cfps_number"])

    def test_records_without_id_are_skipped(self):
        self.assertEqual(fi.group_by_concept([{"coverage": 1.0}]), {})

    def test_coverage_none_sorts_last_and_does_not_crash(self):
        records = [field("x_a", None), field("x_a_2", 0.5)]
        groups = fi.group_by_concept(records, drop_numeric=True)
        ordered = [r["id"] for r in list(groups.values())[0]]
        self.assertEqual(ordered[0], "x_a_2")


class DuplicateConceptTests(unittest.TestCase):
    def test_spread_is_best_minus_worst(self):
        records = [field("cfps_number", 0.6751), field("cfps_number_2", 0.3977)]
        out = fi.duplicate_concepts(records, drop_numeric=True)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["coverage_spread"], 0.2774, places=4)
        self.assertEqual(out[0]["best"], "cfps_number")

    def test_min_spread_filters_small_differences(self):
        records = [field("cfps_number", 0.60), field("cfps_number_2", 0.50)]
        self.assertEqual(
            fi.duplicate_concepts(records, min_spread=0.5, drop_numeric=True), [])

    def test_min_members_respected(self):
        records = [field("cfps_number", 0.60), field("cfps_number_2", 0.50)]
        self.assertEqual(
            fi.duplicate_concepts(records, min_members=3, drop_numeric=True), [])

    def test_field_type_filter(self):
        records = [
            field("cfps_number", 0.60, field_type="MATRIX"),
            field("cfps_number_2", 0.50, field_type="VECTOR"),
        ]
        self.assertEqual(
            fi.duplicate_concepts(records, field_type="MATRIX", drop_numeric=True), [])

    def test_sorted_by_spread_descending(self):
        records = [
            field("a_one", 0.9), field("a_one_2", 0.8),        # spread 0.1
            field("b_two", 1.0), field("b_two_2", 0.4),        # spread 0.6
        ]
        out = fi.duplicate_concepts(records, drop_numeric=True)
        self.assertEqual([item["coverage_spread"] for item in out], [0.6, 0.1])


class FieldIdTests(unittest.TestCase):
    def test_collects_ids(self):
        records = [field("a", 0.1), field("b", 0.2), {"coverage": 0.3}]
        self.assertEqual(fi.field_ids(records), {"a", "b"})

    def test_empty_is_empty_set(self):
        self.assertEqual(fi.field_ids([]), set())
        self.assertEqual(fi.field_ids(None), set())


if __name__ == "__main__":
    unittest.main()
