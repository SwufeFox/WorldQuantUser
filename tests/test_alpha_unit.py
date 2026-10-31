#!/usr/bin/env python3
"""Tests for the unit layer: split, template round-trip, grouping and derivation.

The load-bearing property is the round trip. A compression that cannot re-instantiate
its own input is a summary, and a summary that looks like a compression is worse than
no compression at all -- it makes coverage numbers say 100% when nothing was checked.
So most of this file is about slots: that they are found, that they are shared by
value rather than by position, and that refilling them reproduces the expression.

The second property is that derivation stays on the evidence. The grid rule may only
move a slot to a value the corpus already put there; inventing values is a separate,
opt-in call, and it has its own guards.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_unit as au  # noqa: E402
import fastexpr_ast as fa  # noqa: E402

TWO_LEG = ("x = ts_backfill(fnd6_newqv1300_ivltq, 252); "
           "group_rank(x/(ts_backfill(fnd6_cptnewqv1300_atq, 252) + 1), subindustry) "
           "- 0.5 - 0.5*rank((close - low) / (high - low) - 0.5)")

DATASETS = {
    "fnd6_newqv1300_ivltq": "fundamental6",
    "fnd6_newqv1300_ppentq": "fundamental6",
    "fnd6_newqv1300_invtq": "fundamental6",
    "fnd6_cptnewqv1300_atq": "fundamental6",
    "fnd6_cptnewqv1300_ltq": "fundamental6",
    "close": "pv1",
    "low": "pv1",
    "high": "pv1",
    "subindustry": "essential",
}


def entry(expression, fitness=None, active=False, status=None, **extra):
    row = {"expression": expression,
           "status": status or ("ACTIVE" if active else "UNSUBMITTED"),
           "metrics": {"fitness": fitness} if fitness is not None else {}}
    row.update(extra)
    return row


class Split(unittest.TestCase):
    def test_assembly_reads_the_signs_and_the_offset(self):
        point = au.split(TWO_LEG, datasets=DATASETS)
        self.assertEqual(point.assembly, "G -L -G")

    def test_legs_carry_weights_and_roles(self):
        point = au.split(TWO_LEG, datasets=DATASETS)
        self.assertEqual(len(point.legs), 2)
        self.assertEqual([leg.weight for leg in point.legs], [1.0, -0.5])
        self.assertEqual(point.legs[0].role, "rank(ratio(F,mix(F,L)))")
        self.assertEqual(point.legs[1].role,
                         "rank(mix(ratio(mix(F,F),mix(F,F)),L))")

    def test_weights_are_not_part_of_the_shape(self):
        light = au.split(TWO_LEG.replace("- 0.5*rank", "- 1.5*rank"), datasets=DATASETS)
        heavy = au.split(TWO_LEG, datasets=DATASETS)
        self.assertEqual(light.template.render(), heavy.template.render())
        self.assertNotEqual(light.filling, heavy.filling)

    def test_local_variable_naming_is_not_content(self):
        inlined = TWO_LEG.replace(
            "x = ts_backfill(fnd6_newqv1300_ivltq, 252); ",
            "").replace("x/", "ts_backfill(fnd6_newqv1300_ivltq, 252)/")
        self.assertEqual(au.split(TWO_LEG, datasets=DATASETS).canonical,
                         au.split(inlined, datasets=DATASETS).canonical)

    def test_a_damaged_expression_is_skipped_not_raised(self):
        self.assertIsNone(au.split("this is not an expression (", datasets=DATASETS))


class Templates(unittest.TestCase):
    def test_refilling_the_template_reproduces_the_expression(self):
        point = au.split(TWO_LEG, datasets=DATASETS)
        self.assertEqual(fa.canonical(point.render()), point.canonical)

    def test_a_field_used_twice_is_one_slot(self):
        # `close - low` and `close + low` are one slot for close: substituting it has
        # to move both, or the template stops meaning "the same construction".
        point = au.split("rank(close - low) + rank(close + low)", datasets=DATASETS)
        names = [name for name in point.filling if name.startswith("f")]
        self.assertEqual(len(names), 2)
        filling = dict(point.filling)
        index = [name for name, value in filling.items() if value == "close"][0]
        filling[index] = "high"
        rendered = point.render(filling)
        self.assertEqual(rendered.count("high"), 2)

    def test_filling_a_missing_slot_fails_instead_of_inventing(self):
        template, filling = au.template_of(fa.parse("rank(close)").signal)
        with self.assertRaises(KeyError):
            au.fill(template, {})


class Grouping(unittest.TestCase):
    def test_points_sharing_a_template_land_in_one_unit(self):
        other = TWO_LEG.replace("fnd6_newqv1300_ivltq", "fnd6_newqv1300_invtq")
        corpus = au.build([entry(TWO_LEG), entry(other)], datasets=DATASETS)
        self.assertEqual(len(corpus.units), 1)
        self.assertEqual(corpus.units[next(iter(corpus.units))].size, 2)

    def test_a_different_assembly_is_a_different_kind(self):
        flipped = TWO_LEG.replace("- 0.5 - 0.5*rank", "+ 0.5 - 0.5*rank")
        corpus = au.build([entry(TWO_LEG), entry(flipped)], datasets=DATASETS)
        self.assertEqual(len(corpus.kinds), 2)

    def test_duplicate_expressions_are_one_point(self):
        corpus = au.build([entry(TWO_LEG), entry("  " + TWO_LEG + " ")],
                          datasets=DATASETS)
        self.assertEqual(len(corpus.points), 1)

    def test_abandoned_rows_do_not_enter_the_corpus(self):
        corpus = au.build([entry(TWO_LEG, abandoned=True)], datasets=DATASETS)
        self.assertEqual(len(corpus.points), 0)

    def test_summary_counts_match_the_corpus(self):
        other = TWO_LEG.replace("fnd6_newqv1300_ivltq", "fnd6_newqv1300_invtq")
        corpus = au.build([entry(TWO_LEG), entry(other),
                           entry("rank(close)")], datasets=DATASETS)
        counts = au.summary(corpus)
        self.assertEqual(counts["expressions"], 3)
        self.assertEqual(counts["units"], 2)
        self.assertEqual(counts["unit_singletons"], 1)


class RoundTrip(unittest.TestCase):
    def test_every_point_refills(self):
        other = TWO_LEG.replace("fnd6_newqv1300_ivltq", "fnd6_newqv1300_invtq")
        corpus = au.build([entry(TWO_LEG), entry(other), entry("rank(close)")],
                          datasets=DATASETS)
        checked, failures = au.verify(corpus)
        self.assertEqual(checked, 3)
        self.assertEqual(failures, [])

    def test_verify_reports_a_broken_template(self):
        corpus = au.build([entry(TWO_LEG)], datasets=DATASETS)
        point = corpus.points[0]
        point.filling = {"f0": "close"}  # drop most slots
        _checked, failures = au.verify(corpus)
        self.assertEqual(len(failures), 1)


class Derivation(unittest.TestCase):
    def setUp(self):
        self.point = au.split(TWO_LEG, datasets=DATASETS)

    def test_ablation_asks_about_every_leg(self):
        variants = au.derive(self.point)
        rules = {variant.label for variant in variants}
        self.assertIn("zero:l0", rules)
        self.assertIn("zero:l1", rules)
        self.assertIn("drop:l0", rules)
        self.assertIn("drop:l1", rules)

    def test_ablations_keep_the_offset(self):
        # Deleting the `- 0.5` centring alongside the leg would measure a different
        # construction while looking like an ablation.
        dropped = [variant for variant in au.derive(self.point)
                   if variant.label == "drop:l1"][0]
        self.assertIn("0.5", dropped.expression)

    def test_grid_rule_only_uses_values_the_corpus_used(self):
        grid = {"f0": ["fnd6_newqv1300_ivltq", "fnd6_newqv1300_invtq"]}
        variants = [variant for variant in au.derive(self.point, grid=grid)
                    if variant.rule == "data"]
        self.assertEqual(len(variants), 1)
        self.assertIn("fnd6_newqv1300_invtq", variants[0].expression)
        self.assertEqual(variants[0].status, "novel")

    def test_domain_rule_needs_to_be_asked_for(self):
        without = [variant for variant in au.derive(self.point) if variant.rule == "domain"]
        self.assertEqual(without, [])
        with_domain = [variant for variant in
                       au.derive(self.point, domains={"f0": ["fnd6_newqv1300_ppentq"]})
                       if variant.rule == "domain"]
        self.assertEqual(len(with_domain), 1)

    def test_observed_variants_are_labelled_observed(self):
        sibling = au.split(TWO_LEG.replace("fnd6_newqv1300_ivltq", "fnd6_newqv1300_invtq"),
                           datasets=DATASETS)
        variants = au.derive(self.point, grid={"f0": ["fnd6_newqv1300_invtq"]},
                             observed={sibling.canonical})
        data = [variant for variant in variants if variant.rule == "data"]
        self.assertEqual(data[0].status, "observed")


class DomainExpansion(unittest.TestCase):
    def test_only_a_varied_slot_is_expanded(self):
        other = TWO_LEG.replace("fnd6_newqv1300_ivltq", "fnd6_newqv1300_invtq")
        corpus = au.build([entry(TWO_LEG), entry(other)], datasets=DATASETS)
        unit = next(iter(corpus.units.values()))
        domains = au.expand_domain(unit, DATASETS)
        self.assertEqual(list(domains), ["f0"])
        self.assertIn("fnd6_newqv1300_ppentq", domains["f0"])
        self.assertNotIn("close", domains["f0"])

    def test_the_group_key_is_never_offered(self):
        other = TWO_LEG.replace("subindustry", "sector")
        corpus = au.build([entry(TWO_LEG), entry(other)],
                          datasets=dict(DATASETS, sector="essential"))
        unit = next(iter(corpus.units.values()))
        domains = au.expand_domain(unit, dict(DATASETS, sector="essential"))
        self.assertNotIn("f4", domains)

    def test_a_slot_mixing_families_is_not_offered(self):
        # `rank(F)` swept across model and fundamental fields has no carrier family,
        # so there is nothing to expand it *within*.
        corpus = au.build([entry("rank(cashflow_efficiency_rank_derivative)"),
                           entry("rank(fnd6_newqv1300_ivltq)")],
                          datasets={"cashflow_efficiency_rank_derivative": "model16",
                                    "fnd6_newqv1300_ivltq": "fundamental6"})
        unit = next(iter(corpus.units.values()))
        self.assertEqual(au.expand_domain(unit, {"cashflow_efficiency_rank_derivative":
                                                 "model16",
                                                 "fnd6_newqv1300_ivltq":
                                                 "fundamental6"}), {})


class ProgramBridge(unittest.TestCase):
    def test_a_point_renders_into_an_annotated_program(self):
        point = au.split(TWO_LEG, datasets=DATASETS)
        program = au.to_program(point, "carrier plus stabiliser")
        self.assertEqual([name for name, _value in program.assignments], ["l0", "l1"])
        names = [note.name for note in program.annotations]
        self.assertEqual(names, ["mechanism", "leg", "leg"])

    def test_the_program_passes_the_annotation_check(self):
        import annotations

        point = au.split(TWO_LEG, datasets=DATASETS)
        self.assertEqual(annotations.check(au.to_program(point, "test")), [])

    def test_the_program_measures_the_same_thing(self):
        point = au.split(TWO_LEG, datasets=DATASETS)
        program = au.to_program(point, "test")
        self.assertEqual(fa.canonical(au._flatten(program.render())), point.canonical)


if __name__ == "__main__":
    unittest.main()