"""Assertions for the submission-chain fixes (uncertainty, metadata, evidence, n_ops)."""

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import alpha_registry
import cache_store
import machine_lib as ml
from research_app import alphas, confirm, promote, selection
from research_app.runtime import CliError


def platform(alpha_id="mLKwo5Nx", status="UNSUBMITTED", named=True):
    return {"id": alpha_id, "status": status,
            "name": "n" if named else None, "category": "FUNDAMENTAL" if named else None,
            "regular": {"code": "rank(close)", "operatorCount": 8},
            "is": {"sharpe": 1.65, "fitness": 0.9, "turnover": 0.2},
            "train": {"sharpe": 1.65}, "test": {"sharpe": 0.93},
            "dateSubmitted": None}


class SubmitReconcileTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for patch in (mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": temp.name}),
                      mock.patch.object(alphas, "with_session", lambda args, work: work(object()))):
            patch.start(); self.addCleanup(patch.stop)

    def args(self, **kw):
        base = {"alpha_id": "mLKwo5Nx", "yes": False, "max_wait": 0, "selection": None,
                "force": False, "note": None, "cost_bps": [5]}
        base.update(kw)
        return argparse.Namespace(**base)

    def test_uncertain_attempt_is_reconciled_to_not_sent_when_platform_is_unsubmitted(self):
        path = alphas._submission_path("mLKwo5Nx")
        cache_store.write_json_atomic(path, {"alpha_id": "mLKwo5Nx", "state": "UNCERTAIN"})
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform()), \
                mock.patch.object(ml, "submit_alpha",
                                  return_value={"submission": "NOT_SENT", "checks": {}}) as post:
            result = alphas.cmd_alpha_submit(self.args(), None)
        self.assertEqual(result["submission"], "NOT_SENT")
        self.assertEqual(result["reconciled_from"], "UNCERTAIN")
        self.assertTrue(result["reconciled"])
        post.assert_called_once()
        self.assertEqual(json.loads(Path(path).read_text())["state"], "NOT_SENT")

    def test_missing_metadata_blocks_the_read_only_path(self):
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform(named=False)), \
                mock.patch.object(ml, "submit_alpha") as post:
            with self.assertRaises(CliError) as caught:
                alphas.cmd_alpha_submit(self.args(), None)
        self.assertEqual(caught.exception.code, "INVALID_INPUT")
        self.assertIn("no name, category", caught.exception.message)
        post.assert_not_called()


class PromoteEvidenceTests(unittest.TestCase):
    def test_current_evidence_is_recorded_before_validation(self):
        with mock.patch.object(alpha_registry, "record_entry", return_value=({}, True)) as record:
            promote._record_current_evidence(platform(), "mLKwo5Nx")
        entry, path = record.call_args.args
        self.assertEqual(entry["alpha_id"], "mLKwo5Nx")
        self.assertEqual(entry["operator_count"], 8)
        self.assertEqual(entry["operator_count_basis"], "platform")
        self.assertEqual(entry["out_of_sample"]["test"]["sharpe"], 0.93)


class OperatorCountBasisTests(unittest.TestCase):
    def test_platform_count_is_marked_and_not_confused_with_distinct_estimate(self):
        platform_entry = alpha_registry.build_entry(
            {"alpha_id": "A", "expression": "rank(ts_mean(close,5))",
             "metrics": {}, "checks": {}, "operator_count": 8})
        self.assertEqual(platform_entry["operator_count"], 8)
        self.assertEqual(platform_entry["operator_count_basis"], "platform")

        estimated = alpha_registry.build_entry(
            {"alpha_id": "B", "expression": "rank(ts_mean(close,5))",
             "metrics": {}, "checks": {}})
        self.assertEqual(estimated["operator_count_basis"], "estimate_total")
        self.assertGreaterEqual(estimated["operator_count"], 2)


class BatchArmAliasTests(unittest.TestCase):
    def test_expression_field_is_accepted_as_an_alias(self):
        from tools.pipeline.batch_quick import _load_arms
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "arms.jsonl"
            path.write_text(json.dumps({"label": "a", "expression": "rank(close)",
                                        "settings": {}}) + "\n")
            arms = _load_arms(path)
        self.assertEqual(arms[0]["expr"], "rank(close)")
