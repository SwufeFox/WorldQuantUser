#!/usr/bin/env python3
"""Tests for annotations (syntax, checks, ablation derivation) and the intent rewrite."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import annotations  # noqa: E402
import fastexpr_ast as fa  # noqa: E402

TWO_LEGS = """@mechanism("carrier plus stabiliser")
@leg(name="a", role="carrier")
@leg(name="b", role="stabiliser")
a = group_rank(close / vwap, subindustry) - 0.5;
b = group_rank(-ts_sum(returns, 5), subindustry) - 0.5;
0.7 * a + 0.3 * b"""


class AnnotationSyntax(unittest.TestCase):
    def test_annotations_are_parsed_with_args_and_kwargs(self):
        program = fa.parse(TWO_LEGS)
        parsed = annotations.annotations_of(program)
        self.assertEqual(parsed[0], ("mechanism", ['"carrier plus stabiliser"'], {}))
        self.assertEqual(parsed[1][0], "leg")
        self.assertEqual(parsed[1][2]["name"], '"a"')

    def test_annotations_never_reach_the_compiled_expression(self):
        compiled = fa.parse(TWO_LEGS).render()
        self.assertNotIn("@", compiled)
        self.assertNotIn("mechanism", compiled)

    def test_source_round_trips(self):
        program = fa.parse(TWO_LEGS)
        again = fa.parse(program.render_source())
        self.assertEqual(again.render(), program.render())
        self.assertEqual(len(again.annotations), len(program.annotations))

    def test_annotation_only_input_is_not_empty(self):
        program = fa.parse('@mechanism("x")')
        self.assertEqual(program.assignments, ())
        self.assertIsNone(program.signal)


class Checks(unittest.TestCase):
    def test_valid_program_has_no_findings(self):
        self.assertEqual(annotations.check(fa.parse(TWO_LEGS)), [])

    def test_unknown_annotation_is_reported(self):
        findings = annotations.check(fa.parse('@wat("x")\nrank(close)'))
        self.assertEqual([f["code"] for f in findings], ["unknown_annotation"])

    def test_stale_leg_reference_is_reported(self):
        findings = annotations.check(fa.parse('@leg(name="c")\na = rank(close);\na'))
        self.assertEqual([f["code"] for f in findings], ["stale_annotation"])
        self.assertIn("does not define", findings[0]["message"])

    def test_unknown_field_in_expectation_is_reported(self):
        findings = annotations.check(fa.parse('@expect(field="ghost")\nrank(close)'),
                                     known_fields={"close"})
        self.assertEqual([f["code"] for f in findings], ["stale_field"])


class Ablations(unittest.TestCase):
    def test_every_leg_is_ablated_when_nothing_is_requested(self):
        variants = annotations.derive_ablations(fa.parse(TWO_LEGS))
        labels = {v["label"] for v in variants}
        self.assertIn("zero:a", labels)
        self.assertIn("drop:b", labels)

    def test_explicit_request_narrows_the_set(self):
        source = TWO_LEGS.replace('@leg(name="a", role="carrier")',
                                  '@leg(name="a", role="carrier")\n@ablate(leg="b")')
        variants = annotations.derive_ablations(fa.parse(source))
        self.assertEqual({v["leg"] for v in variants}, {"b"})

    def test_zero_leg_keeps_the_weights(self):
        variants = {v["label"]: v for v in annotations.derive_ablations(fa.parse(TWO_LEGS))}
        expression = variants["zero:a"]["expression"]
        self.assertIn("a = 0;", expression)
        self.assertIn("0.7 * a", expression)

    def test_drop_term_renormalises_the_remaining_coefficients(self):
        variants = {v["label"]: v for v in annotations.derive_ablations(fa.parse(TWO_LEGS))}
        expression = variants["drop:a"]["expression"]
        self.assertIn("1.0 * b", expression)
        self.assertNotIn("0.7 * a", expression)

    def test_drop_is_not_invented_for_a_single_term(self):
        variants = annotations.derive_ablations(fa.parse("a = rank(close);\na"))
        self.assertEqual([v["kind"] for v in variants], ["zero_leg"])



if __name__ == "__main__":
    unittest.main()
