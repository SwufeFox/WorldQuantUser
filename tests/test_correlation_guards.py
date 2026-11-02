#!/usr/bin/env python3
"""Regression tests for the correlation guards.

These pin down five defects found in review, all of the same species: the toolkit
reporting a *clear* result where it had simply not measured anything, or measuring
the wrong thing. A crash is loud; a confident wrong answer is not, which is why
each test states the wrong answer it is preventing.

Everything runs offline. Session-dependent commands are driven by patching
``with_session`` so the decision logic is exercised through the real command body.
"""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ablation  # noqa: E402
import correlation  # noqa: E402
import research_cli as cli  # noqa: E402


class PeerRankingTests(unittest.TestCase):
    def test_mirror_outranks_a_weak_positive_peer(self):
        # Wrong answer this prevents: sorting on the signed value picks +0.20 and
        # calls a -0.95 mirror `clear`.
        scored = [{"alpha_id": "A", "correlation": -0.95},
                  {"alpha_id": "B", "correlation": 0.20},
                  {"alpha_id": "C", "correlation": 0.12}]
        self.assertEqual(cli._rank_peers(scored)[0]["alpha_id"], "A")

    def test_ordering_is_descending_by_absolute_value(self):
        scored = [{"alpha_id": "x", "correlation": 0.3},
                  {"alpha_id": "y", "correlation": -0.8},
                  {"alpha_id": "z", "correlation": 0.6}]
        self.assertEqual([row["alpha_id"] for row in cli._rank_peers(scored)],
                         ["y", "z", "x"])


class ClassifyTests(unittest.TestCase):
    def test_mirror_blocks(self):
        verdict, estimate = correlation.classify(-0.95)
        self.assertEqual(verdict, "blocked")
        self.assertGreaterEqual(estimate, 0.7)

    def test_missing_measurement_is_unknown_not_clear(self):
        # Wrong answer this prevents: an empty peer set defaulting to 0.0 -> `clear`.
        verdict, estimate = correlation.classify(None)
        self.assertEqual(verdict, "unknown")
        self.assertIsNone(estimate)

    def test_ordinary_peer_still_clears(self):
        verdict, _ = correlation.classify(0.10)
        self.assertEqual(verdict, "clear")


class IndependenceStateTests(unittest.TestCase):
    def test_skipped_pair_is_unknown(self):
        # Wrong answer this prevents: no measured pair above the line -> `independent`
        # even though the unmeasurable pair is the likeliest duplicate.
        self.assertEqual(cli._independence_state([], [{"a": "x", "b": "y", "n": 100}], 0),
                         "unknown")

    def test_no_pairs_at_all_is_unknown(self):
        self.assertEqual(cli._independence_state([], [], 0), "unknown")

    def test_flag_wins_over_skipped(self):
        self.assertEqual(cli._independence_state([{"r": 0.9}], [{"a": "x"}], 1),
                         "dependent")

    def test_fully_measured_and_clean_is_independent(self):
        self.assertEqual(cli._independence_state([], [], 3), "independent")


def _series(days, seed, shape):
    """Deterministic synthetic daily PnL."""
    import math
    return {("2020-01-%02d" % (i % 28 + 1)) if False else ("d%05d" % i):
            shape(seed, i) for i in range(days)}


class CorrCommandTests(unittest.TestCase):
    """Drive `alpha corr` through its real body with a patched session."""

    def _run(self, payload, extra=None):
        args = argparse.Namespace(
            alpha_id=list(payload), vs_book=False, control=None, min_overlap=200,
            threshold=0.7, max_age_hours=24, json=None, top=5, book=False,
            label=False, group_by=None, drop_common=0.5, as_of=None)
        for key, value in (extra or {}).items():
            setattr(args, key, value)
        sink = mock.MagicMock()

        def fake_fetch(session, alpha_id, cache_dir=None, max_age_hours=None):
            return payload[alpha_id]

        with mock.patch.object(cli, "with_session", lambda a, fn: fn(None)), \
                mock.patch.object(cli.ml, "fetch_daily_pnl", fake_fetch), \
                mock.patch.object(cli.research_paths, "ensure_dir", lambda: "/tmp"):
            return cli.cmd_alpha_corr(args, sink)

    def test_short_overlap_reports_unknown_not_independent(self):
        import math
        short = {"d%05d" % i: math.sin(i * 0.7) for i in range(100)}
        other = {"d%05d" % i: math.cos(i * 0.9) for i in range(100)}
        data = self._run({"a": short, "b": other})
        self.assertEqual(data["independence"], "unknown")
        self.assertFalse(data["independent"])
        self.assertGreater(data["coverage"]["skipped_pairs"], 0)

    def test_mirror_pair_is_dependent(self):
        import math
        base = {"d%05d" % i: math.sin(i * 0.3) + 0.01 * (i % 7) for i in range(400)}
        mirror = {key: -value for key, value in base.items()}
        data = self._run({"a": base, "b": mirror})
        self.assertEqual(data["independence"], "dependent")
        self.assertTrue(data["flagged"])

    def test_clean_pair_is_independent(self):
        import math
        left = {"d%05d" % i: math.sin(i * 0.3) for i in range(400)}
        right = {"d%05d" % i: math.cos(i * 1.7) + 0.5 * math.sin(i * 0.11)
                 for i in range(400)}
        data = self._run({"a": left, "b": right})
        self.assertEqual(data["independence"], "independent")


class PreflightSettingsTests(unittest.TestCase):
    def test_field_catalog_uses_resolved_settings(self):
        # Wrong answer this prevents: a research file asking for TOP500 is linted
        # against TOP3000, so the field check runs on the wrong catalogue.
        captured = {}

        def fake_fetch(session, **kwargs):
            captured.update(kwargs)
            return []

        args = argparse.Namespace(allow_lint_errors=True, region=None, universe=None,
                                  delay=None, instrument_type=None)
        settings = {"region": "USA", "universe": "TOP500", "delay": 1,
                    "instrumentType": "EQUITY"}
        with mock.patch.object(cli, "_operator_index_from_catalog", return_value={}), \
                mock.patch.object(cli.field_index, "fetch_all_fields", fake_fetch):
            cli._preflight_lint(None, [("x", "rank(close)")], settings, args)
        self.assertEqual(captured.get("universe"), "TOP500")


class AblationReportTests(unittest.TestCase):
    def test_full_and_loo_only_still_reports_contributions(self):
        # Wrong answer this prevents: `--modes full,loo` reported zero legs because
        # the leg list was built from `solo:` records alone.
        records = [
            {"candidate_id": "full", "metrics": {"fitness": 2.0}},
            {"candidate_id": "loo:alpha", "metrics": {"fitness": 1.2}},
            {"candidate_id": "loo:beta", "metrics": {"fitness": 2.1}},
        ]
        report = ablation.report(records)
        legs = {row["leg"]: row for row in report["legs"]}
        self.assertEqual(set(legs), {"alpha", "beta"})
        self.assertAlmostEqual(legs["alpha"]["contribution"], 0.8, places=4)
        self.assertEqual(legs["alpha"]["verdict"], "adds")
        self.assertEqual(legs["beta"]["verdict"], "hurts")
        self.assertEqual(report["measured"], 2)
        self.assertIsNone(legs["alpha"]["solo"])


if __name__ == "__main__":
    unittest.main()