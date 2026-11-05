"""Tests for the `ask` verb: filter, group, aggregate, panel, matrix.

The point of these is the one the verb exists for: an analysis that lives in a test
file is an analysis that cannot silently change its mind between runs. The first
real use of `ask` already corrected two hand-rolled numbers, so the arithmetic here
is checked against hand-computable cases rather than against itself.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_app import ledger as cli, runtime  # noqa: E402


def entry(alpha, *, status="COMPLETE", neutralization="SUBINDUSTRY", decay=4,
          turnover=0.2, sharpe=1.0, fitness=1.0, checks=None,
          verdict="rejected", family="analyst", expression_hash="expr:one",
          recorded_at="2026-11-01T00:00:00Z", metrics=True):
    return {
        "key": alpha,
        "alpha_id": alpha,
        "status": status,
        "verdict": verdict,
        "family": family,
        "expression_hash": expression_hash,
        "recorded_at": recorded_at,
        "settings": {"neutralization": neutralization, "decay": decay,
                     "region": "USA", "universe": "TOP3000", "truncation": 0.08,
                     "delay": 1},
        "metrics": ({"turnover": turnover, "sharpe": sharpe, "fitness": fitness}
                    if metrics else {}),
        "checks": checks or {},
    }


class FilterTests(unittest.TestCase):
    def test_equality_and_negation(self):
        row = entry("A1", status="ACTIVE", neutralization="MARKET")
        self.assertTrue(cli._ask_passes(row, cli._ask_filters("status=ACTIVE")))
        self.assertFalse(cli._ask_passes(row, cli._ask_filters("status=COMPLETE")))
        self.assertTrue(cli._ask_passes(row, cli._ask_filters("neutralization!=SUBINDUSTRY")))
        self.assertFalse(cli._ask_passes(row, cli._ask_filters("neutralization!=MARKET")))

    def test_a_missing_field_never_matches_an_equality(self):
        # "never recorded" and "recorded as this value" are different facts; an
        # empty family must not answer `--where family=` by accident.
        row = entry("A1")
        row["family"] = None
        self.assertFalse(cli._ask_passes(row, cli._ask_filters("family=analyst")))
        self.assertTrue(cli._ask_passes(row, cli._ask_filters("family!=analyst")))

    def test_check_filter_reads_the_recorded_result(self):
        row = entry("A1", checks={"SELF_CORRELATION": {"result": "PENDING"}})
        self.assertTrue(cli._ask_passes(
            row, cli._ask_filters("check=SELF_CORRELATION:PENDING")))
        self.assertFalse(cli._ask_passes(
            row, cli._ask_filters("check=SELF_CORRELATION:PASS")))

    def test_has_metrics_rejects_an_empty_metrics_block(self):
        self.assertTrue(cli._ask_passes(entry("A1"), cli._ask_filters("has=metrics")))
        self.assertFalse(cli._ask_passes(entry("A1", metrics=False),
                                        cli._ask_filters("has=metrics")))

    def test_unknown_field_is_a_usage_error(self):
        with self.assertRaises(runtime.CliError):
            cli._ask_filters("colour=blue")


class StatTests(unittest.TestCase):
    def test_median_and_quantiles(self):
        rows = [entry("A%d" % i, sharpe=value) for i, value in enumerate([1, 2, 3, 4])]
        self.assertEqual(cli._ask_stat(rows, "median:sharpe"), 2.5)
        self.assertEqual(cli._ask_stat(rows, "p25:sharpe"), 1.75)
        self.assertEqual(cli._ask_stat(rows, "p75:sharpe"), 3.25)

    def test_floor_share_counts_turnover_below_the_tax_threshold(self):
        rows = [entry("A1", turnover=0.05), entry("A2", turnover=0.2),
                entry("A3", turnover=0.125), entry("A4", turnover=0.3)]
        # 0.125 is *at* the floor, so only one of four pays the tax.
        self.assertEqual(cli._ask_stat(rows, "floor_share"), 0.25)

    def test_a_group_without_the_metric_reports_none_not_zero(self):
        self.assertIsNone(cli._ask_stat([entry("A1", metrics=False)], "median:fitness"))

    def test_unknown_stat_raises(self):
        with self.assertRaises(runtime.CliError):
            cli._ask_stat([entry("A1")], "variance:sharpe")


class PanelTests(unittest.TestCase):
    def test_slope_is_measured_inside_one_construction(self):
        rows = [entry("A%d" % decay, decay=decay, turnover=turnover)
                for decay, turnover in ((2, 0.4), (4, 0.2), (8, 0.1))]
        report = cli._ask_panel(rows, "decay", "turnover")
        self.assertEqual(report["turnover"]["panels"], 1)
        self.assertLess(report["turnover"]["median_r"], -0.9)

    def test_a_flat_metric_is_undefined_rather_than_zero(self):
        rows = [entry("A%d" % decay, decay=decay, fitness=1.0) for decay in (2, 4, 8)]
        report = cli._ask_panel(rows, "decay", "fitness")
        self.assertEqual(report["fitness"]["panels"], 0)
        self.assertIsNone(report["fitness"]["median_r"])

    def test_two_constructions_that_trend_oppositely_are_kept_apart(self):
        # Pooling the corpus would average these to zero. Panels keep both signs,
        # so "median 0" here means "two opposite slopes", not "no effect".
        up = [entry("U%d" % decay, decay=decay, sharpe=1.0 + decay * 0.1,
                    expression_hash="expr:up") for decay in (2, 4, 8)]
        down = [entry("D%d" % decay, decay=decay, sharpe=2.0 - decay * 0.1,
                      expression_hash="expr:down") for decay in (2, 4, 8)]
        report = cli._ask_panel(up + down, "decay", "sharpe")
        self.assertEqual(report["sharpe"]["panels"], 2)
        self.assertEqual(sorted(round(item["r"]) for item in report["sharpe"]["detail"]),
                         [-1, 1])


class MatrixTests(unittest.TestCase):
    def _write(self, root, alpha, values):
        path = Path(root) / "pnl" / ("%s.json" % alpha)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values))

    def test_pairs_are_restricted_to_one_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            series = {("2026-01-%02d" % day): float(day) for day in range(1, 29)}
            flipped = {key: -value for key, value in series.items()}
            self._write(tmp, "A1", series)
            self._write(tmp, "A2", series)
            self._write(tmp, "B1", flipped)
            entries = [entry("A1", expression_hash="expr:x"),
                       entry("A2", expression_hash="expr:x"),
                       entry("B1", expression_hash="expr:y")]
            with mock.patch.object(cli.research_paths, "ensure_dir",
                                   return_value=tmp):
                # 28 shared days: below the default 200, so the overlap guard has
                # to be raised rather than relied on.
                block = cli._ask_matrix(entries, "expr", min_overlap=10)
            self.assertEqual(block["pairs_measured"], 1)
            self.assertAlmostEqual(block["pairs"][0]["r"], 1.0, places=6)
            self.assertEqual(block["pairs_skipped"], 0)

    def test_a_factor_probe_names_the_variants_of_one_leg(self):
        # The question the cluster count cannot answer: if a dozen alphas are the same
        # leg re-parameterised, they correlate ~1 with a factor built from any one of
        # them - and the book holds one bet, not a dozen.
        with tempfile.TemporaryDirectory() as tmp:
            base = {("2026-01-%02d" % day): float((day * 7) % 11) for day in range(1, 41)}
            scaled = {key: value * 3.0 for key, value in base.items()}
            other = {key: float((day * 3) % 5) for day, key in
                     enumerate(base, start=1)}
            self._write(tmp, "F1", base)
            self._write(tmp, "COPY", scaled)
            self._write(tmp, "OTHER", other)
            entries = [entry("COPY"), entry("OTHER")]
            with mock.patch.object(cli.research_paths, "ensure_dir",
                                   return_value=tmp):
                block = cli._ask_factor(entries, ["F1"], min_overlap=10)
            self.assertEqual(block["n"], 1)
            listed = {row["alpha_id"]: row["r"] for row in block["correlations"]}
            self.assertAlmostEqual(abs(listed["COPY"]), 1.0, places=4)
            self.assertLess(abs(listed["OTHER"]), 0.9)
            self.assertEqual(block["at_or_above_threshold"], ["COPY"])

    def test_the_matrix_reports_effective_bets_not_just_pair_counts(self):
        # The count of clusters answers "how many groups", not "how many bets": a pool
        # can split into many clusters and still move as one. The spectrum is the
        # number a portfolio decision needs, so it has to come out of `ask` too.
        with tempfile.TemporaryDirectory() as tmp:
            base = {("2026-01-%02d" % day): float(day % 7) for day in range(1, 29)}
            other = {key: float(value + index) for index, (key, value) in
                     enumerate(base.items())}
            self._write(tmp, "A1", base)
            self._write(tmp, "A2", other)
            entries = [entry("A1"), entry("A2")]
            with mock.patch.object(cli.research_paths, "ensure_dir",
                                   return_value=tmp):
                block = cli._ask_matrix(entries, "none", min_overlap=10)
            self.assertEqual(block["pairs_measured"], 1)
            self.assertIsNotNone(block["effective_bets"])
            self.assertIsNotNone(block["pc1_share"])
            self.assertEqual(block["groups"][0]["within"], None)

    def test_a_pnl_outside_the_overlap_guard_is_skipped_not_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            series = {("2026-01-%02d" % day): float(day) for day in range(1, 29)}
            self._write(tmp, "A1", series)
            self._write(tmp, "A2", series)
            entries = [entry("A1"), entry("A2")]
            with mock.patch.object(cli.research_paths, "ensure_dir",
                                   return_value=tmp):
                block = cli._ask_matrix(entries, "expr", min_overlap=200)
            self.assertEqual(block["pairs_measured"], 0)
            self.assertEqual(block["pairs_skipped"], 1)
            self.assertEqual(block["skipped"][0]["reason"], "short_overlap")


if __name__ == "__main__":
    unittest.main()
