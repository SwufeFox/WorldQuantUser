#!/usr/bin/env python3
"""Tests for field_search: ranked, vocabulary-free catalog search.

What these protect: the search must be forgiving about form (word order,
prefixes, typos) while staying honest about meaning. It never assumes two words
are synonyms, and when nothing matches it says which term was absent instead of
silently relaxing the query.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import field_search as fs  # noqa: E402


def record(fid, coverage=0.5, description=""):
    return {"id": fid, "coverage": coverage, "description": description}


class ParseQueryTests(unittest.TestCase):
    def test_splits_words(self):
        required, excluded = fs.parse_query("cfps count")
        self.assertEqual(required, ["cfps", "count"])
        self.assertEqual(excluded, [])

    def test_negation_moves_term_to_excluded(self):
        required, excluded = fs.parse_query("cfps -median")
        self.assertEqual(required, ["cfps"])
        self.assertEqual(excluded, ["median"])

    def test_punctuation_is_a_separator(self):
        required, _ = fs.parse_query("cash-flow,per_share")
        self.assertEqual(required, ["cash", "flow", "per", "share"])

    def test_empty_query(self):
        self.assertEqual(fs.parse_query(""), ([], []))
        self.assertEqual(fs.parse_query(None), ([], []))


class ScoringTests(unittest.TestCase):
    def test_exact_token_beats_prefix_beats_substring(self):
        exact, _ = fs.score_record(record("cfps_number"), ["cfps"], [])
        prefix, _ = fs.score_record(record("cfpsx_number"), ["cfps"], [])
        substring, _ = fs.score_record(record("xxcfpsxx").copy(), ["cfps"], [])
        self.assertGreater(exact, prefix)
        self.assertGreater(prefix, substring)

    def test_id_match_beats_description_match(self):
        in_id, _ = fs.score_record(record("cfps_number"), ["cfps"], [])
        in_desc, _ = fs.score_record(
            record("unrelated", description="cash flow per share cfps"),
            ["cfps"], [])
        self.assertGreater(in_id, in_desc)

    def test_all_terms_required_by_default(self):
        score, _ = fs.score_record(record("cfps_number"), ["cfps", "count"], [])
        self.assertEqual(score, 0)

    def test_any_term_mode_scores_partial(self):
        score, matched = fs.score_record(
            record("cfps_number"), ["cfps", "count"], [], match_all=False)
        self.assertGreater(score, 0)
        self.assertEqual(matched, ["cfps"])

    def test_exclusion_removes_match(self):
        self.assertEqual(fs.score_record(record("cfps_median"), ["cfps"], ["median"]),
                         (0, []))

    def test_reverse_prefix_requires_substantial_token(self):
        # "perc" should not match the fragment "per": that invents hits.
        score, _ = fs.score_record(record("per_share"), ["perc"], [])
        self.assertEqual(score, 0)

    def test_reverse_prefix_matches_spelled_out_abbreviation(self):
        # Query spells out what the id abbreviates.
        score, _ = fs.score_record(record("cash_value"), ["cashflow"], [])
        self.assertGreater(score, 0)

    def test_fuzzy_tolerates_one_edit(self):
        score, _ = fs.score_record(record("cfps_number"), ["cfpps"], [], fuzzy=True)
        self.assertGreater(score, 0)

    def test_fuzzy_off_by_default(self):
        score, _ = fs.score_record(record("cfps_number"), ["cfpps"], [])
        self.assertEqual(score, 0)

    def test_fuzzy_rejects_two_edits(self):
        score, _ = fs.score_record(record("cfps_number"), ["cxxxs"], [], fuzzy=True)
        self.assertEqual(score, 0)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            record("anl4_afv4_cfps_number", 0.6751),
            record("anl4_qf_az_cfps_number", 0.3273),
            record("anl4_afv4_eps_number", 1.0),
            record("unrelated_field", 1.0),
        ]

    def test_hits_carry_score_and_matched_terms(self):
        hits = fs.search_records(self.records, "cfps")
        self.assertTrue(hits)
        self.assertIn("score", hits[0])
        self.assertEqual(hits[0]["matched_terms"], ["cfps"])

    def test_ties_break_on_higher_coverage(self):
        hits = fs.search_records(self.records, "cfps")
        self.assertEqual([h["id"] for h in hits],
                         ["anl4_afv4_cfps_number", "anl4_qf_az_cfps_number"])

    def test_limit_applies_after_ranking(self):
        hits = fs.search_records(self.records, "cfps", limit=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "anl4_afv4_cfps_number")

    def test_empty_query_preserves_input_order(self):
        hits = fs.search_records(self.records, "")
        self.assertEqual([h["id"] for h in hits], [r["id"] for r in self.records])
        self.assertEqual({h["score"] for h in hits}, {0})

    def test_no_match_returns_nothing(self):
        self.assertEqual(fs.search_records(self.records, "zzzz"), [])

    def test_exclusion_only_query_filters_without_scoring(self):
        hits = fs.search_records(self.records, "-eps")
        ids = [h["id"] for h in hits]
        self.assertNotIn("anl4_afv4_eps_number", ids)
        self.assertIn("anl4_afv4_cfps_number", ids)


class AbsentTermTests(unittest.TestCase):
    def test_reports_word_the_catalog_never_uses(self):
        records = [record("cfps_number")]
        self.assertEqual(fs.absent_terms(records, "cfps volume"), ["volume"])

    def test_empty_when_every_word_appears(self):
        records = [record("cfps_number")]
        self.assertEqual(fs.absent_terms(records, "cfps number"), [])


if __name__ == "__main__":
    unittest.main()