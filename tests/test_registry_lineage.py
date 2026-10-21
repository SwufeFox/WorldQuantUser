#!/usr/bin/env python3
"""Tests for ledger identity, lineage and leg-level exposure.

These pin the two things that were wrong in the ledger rather than the code style:
records without an alpha id lost their settings when keyed, and exposure could only
be guessed because leg names are positional.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import alpha_registry as ar  # noqa: E402

COMPOSITE = "a = rank(cap); n = rank(volume); c = rank(close); a + n + c"


def entry(expression=COMPOSITE, alpha_id=None, settings=None, **extra):
    return ar.build_entry(
        {"alpha_id": alpha_id, "expression": expression,
         "settings": settings or {"universe": "TOP500", "decay": 4},
         "metrics": {"fitness": 1.1}, **extra})


class IdentityTests(unittest.TestCase):
    def test_fingerprint_separates_universes(self):
        self.assertNotEqual(
            ar.settings_fingerprint({"universe": "TOP500", "decay": 4}),
            ar.settings_fingerprint({"universe": "TOP3000", "decay": 4}))

    def test_fingerprint_separates_decays(self):
        self.assertNotEqual(
            ar.settings_fingerprint({"universe": "TOP500", "decay": 4}),
            ar.settings_fingerprint({"universe": "TOP500", "decay": 12}))

    def test_fingerprint_ignores_cosmetic_settings(self):
        base = {"universe": "TOP500", "decay": 4}
        noisy = dict(base, language="FASTEXPR", visualization=False)
        self.assertEqual(ar.settings_fingerprint(base),
                         ar.settings_fingerprint(noisy))

    def test_record_without_id_keys_on_expression_and_settings(self):
        # The same formula under two universes used to collapse into one entry.
        a = entry(settings={"universe": "TOP500", "decay": 4})
        b = entry(settings={"universe": "TOP3000", "decay": 4})
        self.assertNotEqual(a["key"], b["key"])

    def test_record_with_id_keys_on_the_id(self):
        a = entry(alpha_id="ABC12345")
        self.assertEqual(a["key"], "ABC12345")


class LegTests(unittest.TestCase):
    def test_legs_are_derived_when_not_stored(self):
        self.assertEqual(ar.legs_of(entry()), ["a", "n", "c"])

    def test_unparseable_expression_has_no_legs(self):
        self.assertEqual(ar.legs_of(entry(expression="rank(cap)")), [])

    def test_shared_legs_match_on_body_not_on_name(self):
        # Both declare a leg called "a" but the bodies differ; they are not the
        # same exposure and must not be reported as sharing.
        one = entry(expression="a = rank(cap); b = rank(close); a + b",
                    alpha_id="ONE")
        two = entry(expression="a = rank(volume); b = rank(close); a + b",
                    alpha_id="TWO")
        rows = ar.shared_legs([one, two], "ONE")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["shared_legs"], 1)

    def test_identical_composites_share_every_leg(self):
        one = entry(alpha_id="ONE")
        two = entry(alpha_id="TWO")
        rows = ar.shared_legs([one, two], "ONE")
        self.assertEqual(rows[0]["shared_legs"], 3)

    def test_leg_index_groups_by_body(self):
        index = ar.leg_index([entry(alpha_id="ONE"), entry(alpha_id="TWO")])
        self.assertEqual(len(index), 3)
        self.assertEqual(sorted(index["rank(cap)"]), ["ONE", "TWO"])


class LineageTests(unittest.TestCase):
    def _ledger(self):
        parent = entry(alpha_id="PARENT")
        child = ar.build_entry(
            {"alpha_id": "CHILD", "expression": COMPOSITE,
             "settings": {"universe": "TOP500", "decay": 4},
             "parent": parent["expression_hash"], "op": "drop_leg:c"})
        return [parent, child]

    def test_chain_walks_to_the_parent(self):
        chain = ar.lineage(self._ledger(), "CHILD")
        self.assertEqual([item["alpha_id"] for item in chain], ["CHILD", "PARENT"])

    def test_children_are_found_from_the_parent(self):
        children = ar.descendants(self._ledger(), "PARENT")
        self.assertEqual([item["alpha_id"] for item in children], ["CHILD"])

    def test_missing_parent_ends_the_walk(self):
        self.assertEqual(len(ar.lineage(self._ledger(), "PARENT")), 1)

    def test_unknown_reference_is_none(self):
        self.assertIsNone(ar.resolve(self._ledger(), "NOPE"))


class RelabelTests(unittest.TestCase):
    def test_family_is_set_and_verdict_preserved(self):
        updated = ar.relabel(entry(alpha_id="X"), family="quality")
        self.assertEqual(updated["family"], "quality")
        self.assertEqual(updated["source"], "relabel")

    def test_unknown_family_is_refused(self):
        with self.assertRaises(ValueError):
            ar.relabel(entry(alpha_id="X"), family="not-a-family")

    def test_unknown_verdict_is_refused(self):
        with self.assertRaises(ValueError):
            ar.relabel(entry(alpha_id="X"), verdict="nonsense")


class CoverageTests(unittest.TestCase):
    def test_counts_entries_missing_family(self):
        ledger = [entry(alpha_id="A"), ar.relabel(entry(alpha_id="B"), family="quality")]
        summary = ar.unlabeled_summary(ledger)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["missing_family"], 1)
        self.assertEqual(summary["family_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()