#!/usr/bin/env python3
"""Tests for the mechanism registry and the convergence monitor.

The campaign counts candidates by independent mechanism. These tests pin the key
(`(block, skeleton)`, so a field swap or a window change is still one mechanism) and the
trigger (outcome-based, with a Sharpe floor - a dead arm with no correlation posts a
positive marginal and must not count as progress).
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research_app import mechanisms


def _entry(alpha_id, expression, status="ACTIVE", fitness=1.5, train_sharpe=1.5,
           idea="H-1", family="quality"):
    return {"alpha_id": alpha_id, "expression": expression, "status": status,
            "family": family, "idea": idea, "recorded_at": "2026-11-09T00:00:00Z",
            "metrics": {"fitness": fitness},
            "out_of_sample": {"train": {"sharpe": train_sharpe}}}


class MechanismKeyTests(unittest.TestCase):
    def test_a_window_change_is_the_same_mechanism(self):
        left = mechanisms.mechanism_key("rank(ts_delta(close, 5))")
        right = mechanisms.mechanism_key("rank(ts_delta(close, 21))")
        self.assertEqual(left, right)

    def test_a_different_field_in_the_same_block_is_the_same_mechanism(self):
        left = mechanisms.mechanism_key("rank(revenue / assets)")
        right = mechanisms.mechanism_key("rank(cogs / assets)")
        self.assertEqual(left[0], right[0])          # same economic block
        self.assertEqual(left, right)                # and same skeleton

    def test_a_different_construction_is_a_different_mechanism(self):
        left = mechanisms.mechanism_key("rank(revenue / assets)")
        right = mechanisms.mechanism_key("ts_corr(abs(returns), ts_delay(abs(returns), 1))")
        self.assertNotEqual(left, right)


class SummariseTests(unittest.TestCase):
    def test_count_is_by_mechanism_not_by_expression(self):
        entries = [_entry("A1", "rank(revenue / assets)"),
                   _entry("A2", "rank(cogs / assets)"),          # same mechanism
                   _entry("A3", "ts_rank(close, 252)")]          # different
        data = mechanisms.summarise(entries)
        self.assertEqual(data["members"], 3)
        self.assertEqual(data["mechanisms"], 2)
        self.assertEqual(data["rows"][0]["members"], 2)


class ConvergenceTests(unittest.TestCase):
    def test_a_dead_arm_with_low_correlation_does_not_count_as_progress(self):
        entries = [_entry("B1", "rank(revenue / assets)")]
        # train Sharpe 0.4 and no correlation -> marginal 0.4 >= 0.3, but useless.
        candidate = _entry("C1", "ts_rank(close, 252)", status="COMPLETE", train_sharpe=0.4)
        marginals = {"B1": 0.5, "C1": 0.5}
        data = mechanisms.convergence(entries + [candidate], marginals, recent_batches=5)
        self.assertEqual(data["recent_additive"], 1)      # only the book member

    def test_a_strong_independent_candidate_counts(self):
        entries = [_entry("B1", "rank(revenue / assets)")]
        candidate = _entry("C1", "ts_rank(close, 252)", status="COMPLETE", train_sharpe=1.8)
        data = mechanisms.convergence(entries + [candidate], {"B1": 0.5, "C1": 0.6},
                                      recent_batches=5)
        self.assertEqual(data["recent_additive"], 2)


if __name__ == "__main__":
    unittest.main()


class SnapshotTests(unittest.TestCase):
    def test_brief_reads_the_snapshot_instead_of_computing(self):
        import tempfile
        from unittest import mock
        import research_paths
        from research_app import campaign as cli_campaign

        tmp = tempfile.mkdtemp()
        with mock.patch.object(research_paths, "ensure_dir", lambda: tmp):
            payload = cli_campaign._convergence_snapshot()
        self.assertTrue(payload["stale"])
        self.assertIn("hint", payload)

    def test_a_fresh_snapshot_is_not_stale(self):
        import json, os, tempfile
        from datetime import datetime, timezone
        from unittest import mock
        import research_paths
        from research_app import campaign as cli_campaign

        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "convergence.json"), "w") as handle:
            json.dump({"as_of": datetime.now(timezone.utc).isoformat(),
                       "recent_productive": 3, "recent_additive": 5,
                       "book_mechanisms": 36}, handle)
        with mock.patch.object(research_paths, "ensure_dir", lambda: tmp):
            payload = cli_campaign._convergence_snapshot()
        self.assertFalse(payload["stale"])
        self.assertEqual(payload["recent_productive"], 3)


class NoveltyGateTests(unittest.TestCase):
    def _gate(self, entries, settings=None, expression="rank( close )"):
        import tempfile
        from unittest import mock
        import alpha_registry
        import machine_lib as ml
        import research_paths
        from research_app import experiments

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(alpha_registry, "load_entries", return_value=entries) as load, \
                mock.patch.object(research_paths, "ensure_dir", return_value=tmp):
            result = experiments._novelty_gate(
                [{"expression": expression, "settings": settings or dict(ml.DEFAULT_SETTINGS)}],
                registry_path="chosen-registry.jsonl")[0]
        load.assert_called_once_with("chosen-registry.jsonl")
        return result

    def _entry(self, **overrides):
        import machine_lib as ml
        return {"key": "evidence-1", "alpha_id": "A1", "expression": "rank(close)",
                "status": "COMPLETE", "settings": dict(ml.DEFAULT_SETTINGS),
                "metrics": {"fitness": 1.2}, **overrides}

    def test_exact_repeat_exposes_finished_evidence_keys(self):
        verdict = self._gate([self._entry()])
        self.assertTrue(verdict["already_simulated"])
        self.assertEqual(verdict["exact_repeat_keys"], ["evidence-1"])
        self.assertEqual(verdict["structural_similarity_keys"], [])

    def test_every_resolved_setting_is_part_of_identity(self):
        import machine_lib as ml
        changes = {"region": "EUR", "universe": "TOP500", "delay": 0, "decay": 3,
                   "neutralization": "INDUSTRY", "truncation": 0.05,
                   "pasteurization": "OFF", "nanHandling": "VERIFY", "testPeriod": "P2Y",
                   "unitHandling": "OFF", "language": "OTHER", "visualization": True,
                   "instrumentType": "OTHER"}
        self.assertEqual(set(changes), set(ml.DEFAULT_SETTINGS))
        for key, value in changes.items():
            with self.subTest(setting=key):
                requested = dict(ml.DEFAULT_SETTINGS, **{key: value})
                self.assertFalse(self._gate([self._entry()], requested)["already_simulated"])

    def test_partial_historical_settings_never_inherit_defaults(self):
        import machine_lib as ml
        for missing in ml.DEFAULT_SETTINGS:
            with self.subTest(missing=missing):
                partial = {key: value for key, value in ml.DEFAULT_SETTINGS.items() if key != missing}
                self.assertFalse(self._gate([self._entry(settings=partial)])["already_simulated"])

    def test_unfinished_or_empty_results_are_not_repeats(self):
        for status in ("FAILED", "QUEUED", "PENDING", "DRY_RUN"):
            with self.subTest(status=status):
                self.assertFalse(self._gate([self._entry(status=status)])["already_simulated"])
        for metrics in ({}, {"fitness": None}, {"fitness": float("nan")}, {"fitness": True}):
            with self.subTest(metrics=metrics):
                self.assertFalse(self._gate([self._entry(metrics=metrics)])["already_simulated"])
        self.assertTrue(self._gate([self._entry(status="WARNING")])["already_simulated"])

    def test_structural_similarity_does_not_claim_an_exact_repeat(self):
        verdict = self._gate([self._entry(status="ACTIVE", expression="ts_rank(close, 20)")],
                             expression="ts_rank(close, 60)")
        self.assertFalse(verdict["already_simulated"])
        self.assertEqual(verdict["exact_repeat_keys"], [])
        self.assertEqual(verdict["structural_similarity_keys"], ["evidence-1"])


class CardTests(unittest.TestCase):
    _BASE = {
        "mechanism_id": "M-001",
        "source": {"title": "paper"},
        "hypothesis": {"economic_mechanism": "why", "falsification": "what would refute"},
        "novelty": {"closest_existing_family": "volatility_structure"},
        "implementation": {"verified_brain_fields": []},
        "status": "UNTESTED",
    }

    def test_a_complete_card_validates(self):
        ok, problems = mechanisms.validate_card(dict(self._BASE))
        self.assertTrue(ok, problems)

    def test_an_invented_field_is_rejected(self):
        card = dict(self._BASE)
        card["implementation"] = {"verified_brain_fields": ["not_a_real_field"]}
        ok, problems = mechanisms.validate_card(card, known_fields={"assets"})
        self.assertFalse(ok)
        self.assertIn("unverified field", problems[0])

    def test_a_missing_falsification_is_rejected(self):
        card = dict(self._BASE)
        card["hypothesis"] = {"economic_mechanism": "why"}
        ok, problems = mechanisms.validate_card(card)
        self.assertFalse(ok)
        self.assertIn("hypothesis.falsification", problems[0])

    def test_an_unknown_status_is_rejected(self):
        card = dict(self._BASE)
        card["status"] = "MAYBE"
        ok, problems = mechanisms.validate_card(card)
        self.assertFalse(ok)


class BlockTests(unittest.TestCase):
    """Block inference must read *fields*, never operator names.

    `_std` was meant to catch `eps_std`-style analyst columns, but it also matched the
    operator `ts_std_dev`, so eight book alphas built on realised volatility were counted
    as analyst dispersion - which is how a "saturated analyst block" reading got reported.
    """

    def test_an_operator_does_not_decide_the_block(self):
        block = mechanisms.block_of("-rank(ts_std_dev(returns, 120))")
        self.assertNotEqual(block, "analyst_dispersion")
        self.assertEqual(block, "volatility_structure")

    def test_a_field_does_decide_the_block(self):
        self.assertEqual(mechanisms.block_of("rank(eps_high - eps_low)"), "analyst_dispersion")

    def test_a_reversal_on_close_is_a_price_path(self):
        self.assertEqual(mechanisms.block_of("-rank(ts_delta(close, 5))"), "price_path")
