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
import machine_lib as ml
import research_paths
from research_app import alphas as cli_alphas
from research_app import validation as cli_validation


class PeerRankingTests(unittest.TestCase):
    def test_mirror_outranks_a_weak_positive_peer(self):
        # Wrong answer this prevents: sorting on the signed value picks +0.20 and
        # calls a -0.95 mirror `clear`.
        scored = [{"alpha_id": "A", "correlation": -0.95},
                  {"alpha_id": "B", "correlation": 0.20},
                  {"alpha_id": "C", "correlation": 0.12}]
        self.assertEqual(cli_alphas._rank_peers(scored)[0]["alpha_id"], "A")

    def test_ordering_is_descending_by_absolute_value(self):
        scored = [{"alpha_id": "x", "correlation": 0.3},
                  {"alpha_id": "y", "correlation": -0.8},
                  {"alpha_id": "z", "correlation": 0.6}]
        self.assertEqual([row["alpha_id"] for row in cli_alphas._rank_peers(scored)],
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
        self.assertEqual(cli_alphas._independence_state([], [{"a": "x", "b": "y", "n": 100}], 0),
                         "unknown")

    def test_no_pairs_at_all_is_unknown(self):
        self.assertEqual(cli_alphas._independence_state([], [], 0), "unknown")

    def test_flag_wins_over_skipped(self):
        self.assertEqual(cli_alphas._independence_state([{"r": 0.9}], [{"a": "x"}], 1),
                         "dependent")

    def test_fully_measured_and_clean_is_independent(self):
        self.assertEqual(cli_alphas._independence_state([], [], 3), "independent")


def _series(days, seed, shape):
    """Deterministic synthetic daily PnL."""
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

        with mock.patch.object(cli_alphas, "with_session", lambda a, fn: fn(None)), \
                mock.patch.object(ml, "fetch_daily_pnl", fake_fetch), \
                mock.patch.object(research_paths, "ensure_dir", lambda: "/tmp"):
            return cli_alphas.cmd_alpha_corr(args, sink)

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
    def test_preflight_checks_fields_in_the_requested_universe(self):
        def fields(session, **scope):
            scope["report"].update(state="verified", coverage={"complete": True})
            return ([{"id": "scoped_field", "type": "MATRIX"}]
                    if scope["universe"] == "TOP500" else [])

        def operators(session, **kwargs):
            kwargs["report"].update(complete=True)
            return [{"name": "rank"}]

        settings = {"region": "USA", "universe": "TOP500", "delay": 1,
                    "instrumentType": "EQUITY"}
        with mock.patch.object(cli_validation.capabilities, "fetch_fields", side_effect=fields), \
                mock.patch.object(ml, "fetch_operators", side_effect=operators), \
                mock.patch.object(cli_validation.expression_lint, "load_denylist", return_value={}):
            report = cli_validation._preflight_lint(
                None, [("rank(scoped_field)", 4)], settings, argparse.Namespace(allow_lint_errors=False))
        self.assertTrue(report["ok"])


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


class VerdictGapTests(unittest.TestCase):
    """`clear` is a claim about every peer, so a gap has to weaken it."""

    def test_no_gaps_leaves_the_verdict_alone(self):
        self.assertEqual(cli_alphas._verdict_with_gaps("clear", []), ("clear", None))

    def test_a_gap_turns_clear_into_unknown_and_says_why(self):
        gaps = [{"alpha_id": "X", "reason": "undefined", "n": 400},
                {"alpha_id": "Y", "reason": "no_pnl", "n": 0}]
        verdict, note = cli_alphas._verdict_with_gaps("clear", gaps)
        self.assertEqual(verdict, "unknown")
        self.assertIn("2 peer(s)", note)
        self.assertIn("no_pnl 1", note)
        self.assertIn("undefined 1", note)

    def test_a_gap_cannot_rescue_a_block(self):
        # Missing evidence can only add a reason to refuse, never remove one.
        gaps = [{"alpha_id": "X", "reason": "undefined", "n": 400}]
        self.assertEqual(cli_alphas._verdict_with_gaps("blocked", gaps)[0], "blocked")
        self.assertEqual(cli_alphas._verdict_with_gaps("borderline", gaps)[0], "borderline")


class SelfCheckGapTests(unittest.TestCase):
    """Drive `alpha self-check` through its real body with a patched session.

    These cover the two ways a peer with no usable number used to vanish: a flat
    PnL series crashed `round(None, 4)`, and a peer that returned no PnL was
    skipped silently so the reported maximum was taken over survivors only.
    """

    DAYS = 400

    def _run(self, series, peers, extra=None):
        import math
        candidate = {"d%05d" % i: math.sin(i * 0.3) + 0.01 * (i % 7)
                     for i in range(self.DAYS)}
        entries = [{"id": alpha_id, "status": "ACTIVE",
                    "dateCreated": "2026-10-01T00:00:00Z",
                    "dateSubmitted": "2026-10-01T00:00:00Z"} for alpha_id in peers]
        args = argparse.Namespace(alpha_id="CAND", as_of=None, include_unsent_hours=24.0,
                                  threshold=0.7, platform_bias=0.015, top=3,
                                  min_overlap=200, peer_limit=300,
                                  max_age_hours=168.0, json=None)
        for key, value in (extra or {}).items():
            setattr(args, key, value)

        def fake_list(session, limit=None, status=None, alpha_type_not=None):
            return entries if status == "ACTIVE" else []

        def fake_pnl(session, alpha_id, cache_dir=None, max_age_hours=None):
            return candidate if alpha_id == "CAND" else series.get(alpha_id)

        with mock.patch.object(cli_alphas, "with_session", lambda a, fn: fn(None)), \
                mock.patch.object(ml, "list_alphas", fake_list), \
                mock.patch.object(ml, "fetch_daily_pnl", fake_pnl), \
                mock.patch.object(ml, "fetch_alpha_metrics",
                                  lambda s, i: {"dateCreated": "2026-10-01T00:00:00Z",
                                                "dateSubmitted": None}), \
                mock.patch.object(research_paths, "ensure_dir", lambda: "/tmp"):
            return cli_alphas.cmd_alpha_self_check(args, mock.MagicMock())

    def test_flat_peer_is_unmeasured_not_a_crash_and_not_a_zero(self):
        flat = {"d%05d" % i: 0.25 for i in range(self.DAYS)}
        data = self._run({"FLAT": flat}, ["FLAT"])
        self.assertEqual(data["verdict"], "unknown")
        self.assertIsNone(data["max_daily"])
        self.assertEqual([gap["reason"] for gap in data["peers_unmeasured"]], ["undefined"])
        self.assertIn("could not be measured", data["verdict_note"])

    def test_peer_without_pnl_is_reported_rather_than_dropped(self):
        data = self._run({}, ["NOPNL"])
        self.assertEqual(data["verdict"], "unknown")
        self.assertEqual(data["peers_considered"], 1)
        self.assertEqual(data["peers_measured"], 0)
        self.assertEqual([gap["reason"] for gap in data["peers_unmeasured"]], ["no_pnl"])

    def test_short_overlap_is_unmeasured(self):
        import math
        short = {"d%05d" % i: math.cos(i * 0.9) for i in range(100)}
        data = self._run({"SHORT": short}, ["SHORT"])
        self.assertEqual(data["verdict"], "unknown")
        self.assertEqual([gap["reason"] for gap in data["peers_unmeasured"]],
                         ["short_overlap"])

    def test_a_measured_weak_peer_still_clears(self):
        import math
        weak = {"d%05d" % i: math.cos(i * 1.7) + 0.5 * math.sin(i * 0.11)
                for i in range(self.DAYS)}
        data = self._run({"WEAK": weak}, ["WEAK"])
        self.assertEqual(data["verdict"], "clear")
        self.assertEqual(data["peers_measured"], 1)
        self.assertEqual(data["peers_unmeasured"], [])
        self.assertIsNone(data["verdict_note"])

    def test_a_gap_does_not_hide_a_measured_block(self):
        import math
        flat = {"d%05d" % i: 0.25 for i in range(self.DAYS)}
        mirror = {"d%05d" % i: -(math.sin(i * 0.3) + 0.01 * (i % 7))
                  for i in range(self.DAYS)}
        data = self._run({"FLAT": flat, "MIRROR": mirror}, ["FLAT", "MIRROR"])
        self.assertEqual(data["verdict"], "blocked")
        self.assertEqual(data["counterpart"], "MIRROR")
        self.assertEqual(len(data["peers_unmeasured"]), 1)


if __name__ == "__main__":
    unittest.main()
