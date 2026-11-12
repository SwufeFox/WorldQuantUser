"""Recovery must not attach another experiment's result to a row.

The audit flagged that `probe --recover` matched on expression text alone, so an arm
re-simulated under different settings could borrow an older alpha's metrics. These
tests pin the three cases: exact match, settings mismatch, and genuine ambiguity.
"""

import unittest
from unittest import mock

import machine_lib as ml
from research_app import experiments


class _Alpha:
    def __init__(self, alpha_id, expression, settings):
        self.alpha_id = alpha_id
        self.expression = expression
        self.settings = settings

    def get(self, key, default=None):
        if key == "id":
            return self.alpha_id
        if key == "settings":
            return self.settings
        return default


SETTINGS_A = {"region": "USA", "universe": "TOP3000", "delay": 1,
              "neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08,
              "instrumentType": "EQUITY"}
SETTINGS_B = dict(SETTINGS_A, neutralization="INDUSTRY")


def _payload(fitness, sharpe=1.5):
    # `extract_split` reads the top-level train/test blocks, not `out_of_sample`.
    return {"status": "UNSUBMITTED", "settings": SETTINGS_A,
            "is": {"fitness": fitness, "sharpe": sharpe, "turnover": 0.1, "returns": 0.05},
            "regular": {"code": "rank(close)"},
            "train": {"fitness": fitness, "sharpe": sharpe, "turnover": 0.1,
                      "returns": 0.05},
            "test": {"fitness": fitness, "sharpe": sharpe}}


class RecoverTests(unittest.TestCase):
    def _rows(self, settings):
        return [{"label": "a1", "expression": "rank(close)", "train_fitness": None,
                 "settings": settings,
                 "settings_signature": experiments._settings_signature(settings)}]

    def _recover(self, alphas, rows):
        with mock.patch.object(ml, "list_alphas", return_value=alphas), \
                mock.patch.object(ml, "regular_expression",
                                  side_effect=lambda a: (a.expression, "", 1)), \
                mock.patch.object(ml, "fetch_alpha_metrics",
                                  return_value=_payload(2.0)):
            return experiments._recover_probe_rows(object(), rows)

    def test_an_exact_expression_and_settings_match_is_recovered(self):
        rows = self._rows(SETTINGS_A)
        outcome = self._recover([_Alpha("A1", "rank(close)", SETTINGS_A)], rows)
        self.assertEqual(outcome["matched"], {"a1": "A1"})
        self.assertEqual(rows[0]["alpha_id"], "A1")

    def test_the_same_expression_under_other_settings_is_not_borrowed(self):
        rows = self._rows(SETTINGS_A)
        outcome = self._recover([_Alpha("B1", "rank(close)", SETTINGS_B)], rows)
        self.assertEqual(outcome["matched"], {})
        self.assertIsNone(rows[0]["train_fitness"])
        self.assertIn("a1", outcome["still_missing"])

    def test_two_identical_candidates_are_left_ambiguous(self):
        rows = self._rows(SETTINGS_A)
        outcome = self._recover(
            [_Alpha("A1", "rank(close)", SETTINGS_A), _Alpha("A2", "rank(close)", SETTINGS_A)],
            rows)
        self.assertEqual(outcome["matched"], {})
        self.assertEqual(len(outcome["ambiguous"]), 1)
        self.assertIsNone(rows[0]["train_fitness"])

    def test_the_signature_covers_the_settings_that_change_the_experiment(self):
        self.assertNotEqual(experiments._settings_signature(SETTINGS_A),
                            experiments._settings_signature(SETTINGS_B))

    def test_recovery_does_not_cross_other_resolved_setting_boundaries(self):
        for key, value in {"pasteurization": "OFF", "nanHandling": "VERIFY",
                           "testPeriod": "P1Y", "unitHandling": "OFF"}.items():
            with self.subTest(setting=key):
                rows = self._rows(SETTINGS_A)
                outcome = self._recover([_Alpha("A1", "rank(close)", dict(SETTINGS_A, **{key: value}))], rows)
                self.assertEqual(outcome["recovered"], 0)
                self.assertIsNone(rows[0]["train_fitness"])

    def test_recovery_recomputes_train_comparison_and_preserves_context(self):
        import argparse
        import json
        import tempfile
        from pathlib import Path

        control = {"label": "control", "design": "control", "status": "COMPLETE",
                   "train_fitness": 1.0, "train_sharpe": 1.0, "train_turnover": 0.2,
                   "train_returns": 0.02, "failed_checks": ["LOW_SHARPE"],
                   "sharpe": 100, "test_sharpe": 200}
        rescued = {**self._rows(SETTINGS_A)[0], "design": "rescue", "proxy_of": "close"}
        document = {"name": "recover-context", "idea": "idea", "mechanism": "economic quantity",
                    "note": "why the proxy matters", "design": {"mode": "controlled", "control": "control"},
                    "rows": [control, rescued], "comparisons": {"stale": True}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            args = argparse.Namespace(recover=str(path), show_test=False)
            with mock.patch.object(experiments, "with_session", side_effect=lambda args, fn: fn(object())), \
                    mock.patch.object(ml, "list_alphas", return_value=[_Alpha("A1", "rank(close)", SETTINGS_A)]), \
                    mock.patch.object(ml, "regular_expression", side_effect=lambda a: (a.expression, "", 1)), \
                    mock.patch.object(ml, "fetch_alpha_metrics", return_value=_payload(2.0)):
                result = experiments._cmd_probe_recover(args)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(result["complete"])
        self.assertFalse(result["all_empty"])
        self.assertEqual(result["comparisons"], saved["comparisons"])
        comparison = result["comparisons"]["rows"][0]
        self.assertEqual(comparison["deltas"]["fitness"], 1.0)
        self.assertEqual(comparison["deltas"]["sharpe"], 0.5)
        self.assertEqual(comparison["proxy_of"], "close")
        self.assertEqual(comparison["unavailable"], {})
        for key in ("idea", "mechanism", "note", "design"):
            self.assertEqual(result[key], document[key])
            self.assertEqual(saved[key], document[key])
        self.assertNotIn("failed_checks", result["rows"][0])
        self.assertNotIn("sharpe", result["rows"][0])
        self.assertNotIn("test_sharpe", result["rows"][0])

    def test_ambiguous_recovery_cannot_create_a_comparison(self):
        rows = self._rows(SETTINGS_A)
        rows[0].update(design="add", status="QUEUED")
        control = {"label": "control", "design": "control", "status": "COMPLETE", "train_fitness": 1.0}
        outcome = self._recover([_Alpha("A1", "rank(close)", SETTINGS_A),
                                 _Alpha("A2", "rank(close)", SETTINGS_A)], rows)
        comparison = experiments._probe_comparisons(
            [control, *rows], {"mode": "controlled", "control": "control"})["rows"][0]
        self.assertEqual(outcome["recovered"], 0)
        self.assertEqual(comparison["deltas"], {})
        self.assertIn("arm_train_metric_unavailable", comparison["unavailable"]["fitness"])


class BatchProvenanceTests(unittest.TestCase):
    """The ledger must say which batch produced a row, not leave it to recollection."""

    def test_recording_carries_the_batch_name(self):
        import json
        import os
        import tempfile
        import alpha_registry

        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "registry.jsonl")
        records = [{"alpha_id": "A1", "expression": "rank(close)", "status": "COMPLETE",
                    "settings": SETTINGS_A, "metrics": {"fitness": 1.2, "sharpe": 1.4},
                    "checks": {}, "out_of_sample": {}}]
        alpha_registry.record_simulation(records, SETTINGS_A, registry_path=path,
                                         batch="h-999-test")
        entry = json.loads(open(path).read().strip().split("\n")[-1])
        self.assertEqual(entry["batch"], "h-999-test")

    def test_a_re_record_keeps_the_original_batch_unless_one_is_given(self):
        import json
        import os
        import tempfile
        import alpha_registry

        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "registry.jsonl")
        record = {"alpha_id": "A1", "expression": "rank(close)", "status": "COMPLETE",
                  "settings": SETTINGS_A, "metrics": {"fitness": 1.2, "sharpe": 1.4},
                  "checks": {}, "out_of_sample": {}}
        alpha_registry.record_simulation([record], SETTINGS_A, registry_path=path,
                                         batch="first-batch")
        alpha_registry.record_simulation([record], SETTINGS_A, registry_path=path)
        entries = [json.loads(line) for line in open(path) if line.strip()]
        latest = {}
        for entry in entries:
            latest[entry.get("key")] = entry
        self.assertEqual(latest["A1"]["batch"], "first-batch")
