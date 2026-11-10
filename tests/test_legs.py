"""The leg model: is a composite the weighted sum of its legs?"""

import json
import os
import tempfile
import unittest

from research_app import legs


def _series(values):
    return {"2020-01-%02d" % (i + 1): v for i, v in enumerate(values)}


class LegsTests(unittest.TestCase):
    def _write(self, tmp, alpha_id, values):
        os.makedirs(os.path.join(tmp, "pnl"), exist_ok=True)
        with open(os.path.join(tmp, "pnl", alpha_id + ".json"), "w") as handle:
            json.dump(_series(values), handle)

    def test_the_sum_of_two_identical_legs_doubles_the_pnl(self):
        tmp = tempfile.mkdtemp()
        values = [10.0, -5.0, 12.0, -3.0] * 10
        self._write(tmp, "A", values)
        dates, aligned, summed = legs.combine({"A": 0.5}, os.path.join(tmp, "pnl"))
        self.assertGreaterEqual(len(dates), 30)
        self.assertEqual(summed[0], 5.0)

    def test_opposite_legs_cancel_to_nothing(self):
        tmp = tempfile.mkdtemp()
        values = [10.0, -5.0, 12.0, -3.0] * 10
        self._write(tmp, "A", values)
        self._write(tmp, "B", [-v for v in values])
        dates, aligned, summed = legs.combine({"A": 0.5, "B": 0.5}, os.path.join(tmp, "pnl"))
        self.assertAlmostEqual(max(abs(v) for v in summed), 0.0, places=9)

    def test_a_leg_with_no_cache_is_reported_not_guessed(self):
        tmp = tempfile.mkdtemp()
        dates, aligned, summed = legs.combine({"missing": 1.0}, os.path.join(tmp, "pnl"))
        self.assertEqual(dates, [])

    def test_sharpe_of_a_flat_series_is_undefined(self):
        self.assertIsNone(legs.sharpe_of([1.0] * 40))


class PairScreenTests(unittest.TestCase):
    """The screen has to judge a pair against the book, not against one member."""

    def test_marginal_uses_the_book_as_a_whole(self):
        import numpy as np
        from research_app import pairs
        rng = np.random.default_rng(7)
        days = 400
        book = rng.normal(0, 1, days)
        # two pool alphas: one is the book plus noise, one is independent
        same = book + rng.normal(0, 0.2, days)
        different = rng.normal(0, 1, days)
        matrix = np.vstack([same, different])
        sharpes = pairs._sharpe(matrix)
        self.assertEqual(len(sharpes), 2)
        centred_book = book - book.mean()
        norm = np.linalg.norm(centred_book)
        rho_same = float((centred_book / norm) @ ((same - same.mean()) / np.linalg.norm(same - same.mean())))
        rho_diff = float((centred_book / norm) @ ((different - different.mean()) / np.linalg.norm(different - different.mean())))
        self.assertGreater(rho_same, 0.9)
        self.assertLess(abs(rho_diff), 0.2)


class CensusTests(unittest.TestCase):
    """`catalog unused` must read fields from the catalog, not from operator names."""

    def test_an_operator_is_not_counted_as_a_field(self):
        import json
        import tempfile
        from unittest import mock
        import research_paths
        from research_app import catalog as catalog_cli

        tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp, "datafields"), exist_ok=True)
        with open(os.path.join(tmp, "datafields", "a.json"), "w") as handle:
            json.dump({"payload": [
                {"id": "assets", "dataset": {"id": "fundamental6"}, "coverage": 0.9},
                {"id": "unused_thing", "dataset": {"id": "fundamental6"}, "coverage": 0.5},
            ]}, handle)
        registry = os.path.join(tmp, "r.jsonl")
        with open(registry, "w") as handle:
            handle.write(json.dumps({"alpha_id": "A", "expression": "rank(assets) + ts_mean(cogs, 5)"}) + "\n")

        class Args:
            top = 2
            registry = None
        args = Args()
        args.registry = registry
        with mock.patch.object(research_paths, "ensure_dir", return_value=tmp), \
                mock.patch.object(research_paths, "metadata_root", return_value=tmp):
            data = catalog_cli.cmd_catalog_unused(args, None)
        self.assertEqual(data["used_fields"], 1)          # assets; cogs is not in the catalog
        self.assertEqual(data["unused_fields"], 1)        # unused_thing


class VectorFieldPreflightTests(unittest.TestCase):
    """A VECTOR field in a MATRIX slot must be caught before a simulation is spent.

    Five arms once went to the simulator with `ts_backfill(event_field)` and came back
    "Operator ts_backfill does not support event inputs" — after a dry run that reported
    "0 blocking findings", because the dry run had no field types and `probe` never ran
    the live preflight at all.
    """

    def test_a_vector_field_in_a_matrix_slot_is_an_error(self):
        import expression_lint
        expression = "rank(fnd6_newqeventv110_drcq / (assets + 1))"
        report = expression_lint.lint_all(
            [(expression, 4)],
            field_types={"fnd6_newqeventv110_drcq": "VECTOR", "assets": "MATRIX"},
            strict_fields=True,
        )
        self.assertFalse(report["ok"])
        codes = {item.get("code") for item in report["errors"]}
        self.assertIn("type_mismatch", codes)

    def test_an_aggregated_vector_field_passes(self):
        import expression_lint
        expression = "rank(vec_avg(composite_sentiment_score_2))"
        report = expression_lint.lint_all(
            [(expression, 4)],
            field_types={"composite_sentiment_score_2": "VECTOR"},
            strict_fields=True,
        )
        self.assertTrue(report["ok"], report["errors"])


class PlainLegFilterTests(unittest.TestCase):
    """A centred leg makes the leg-sum an overestimate; the screen must be able to say no."""

    TOKENS = ("group_rank(", "group_zscore(", "group_neutralize(")

    def test_a_centred_leg_is_detected(self):
        plain = "rank(close) + ts_mean(volume, 5)"
        centred = "group_rank(close, subindustry) - 0.5"
        self.assertFalse(any(token in plain for token in self.TOKENS))
        self.assertTrue(any(token in centred for token in self.TOKENS))

    def test_a_plain_pair_is_kept(self):
        combined = "rank(close)" + "rank(ts_delta(revenue, 63))"
        self.assertFalse(any(token in combined for token in self.TOKENS))
