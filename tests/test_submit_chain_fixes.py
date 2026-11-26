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

SEED_DESCRIPTION = (
    "Idea:\n- Higher quality firms should earn higher subsequent returns.\n\n"
    "Rationale for data used:\n- assets: total reported corporate assets.\n\n"
    "Rationale for operators used:\n- rank: compares firms cross-sectionally."
)


def platform(alpha_id="mLKwo5Nx", status="UNSUBMITTED", named=True):
    return {"id": alpha_id, "status": status,
            "name": "n" if named else None, "category": "FUNDAMENTAL" if named else None,
            "regular": {"code": "rank(close)", "operatorCount": 8,
                        "description": SEED_DESCRIPTION},
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
        from research_app import buffer_cmd
        buffer_cmd._write_seed(buffer_cmd.seeds_path(),
                               {"alpha_id": "mLKwo5Nx", "name": "seed",
                                "category": "FUNDAMENTAL", "tags": ["test"],
                                "description": SEED_DESCRIPTION})

    def args(self, **kw):
        base = {"alpha_id": "mLKwo5Nx", "yes": False, "max_wait": 0, "selection": None,
                "force": False, "note": None, "cost_bps": [5]}
        base.update(kw)
        return argparse.Namespace(**base)

    def test_unsubmitted_snapshot_cannot_prove_uncertain_post_was_not_sent(self):
        path = alphas._submission_path("mLKwo5Nx")
        cache_store.write_json_atomic(path, {"alpha_id": "mLKwo5Nx", "state": "UNCERTAIN"})
        with mock.patch.object(ml, "fetch_alpha_metrics", return_value=platform()), \
                mock.patch.object(ml, "submit_alpha",
                                  return_value={"submission": "NOT_SENT", "checks": {}}) as post:
            result = alphas.cmd_alpha_submit(self.args(), None)
        self.assertEqual(result["submission"], "UNCERTAIN")
        self.assertFalse(result["reconciled"])
        self.assertFalse(result["post_attempted"])
        post.assert_not_called()
        self.assertEqual(json.loads(Path(path).read_text())["state"], "UNCERTAIN")

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


class SeedStoreTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        patcher = mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": temp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.root = Path(temp.name)
        from research_app import buffer_cmd
        self.buffer = buffer_cmd

    def args(self, **overrides):
        base = {"alpha_id": "A1", "buffer": str(self.root / "buffer.jsonl"),
                "seeds": str(self.root / "seeds.jsonl"), "note": None,
                "name": "seed", "category": "FUNDAMENTAL", "tags": ["quality"],
                "description": SEED_DESCRIPTION, "max_wait": 0, "force": False,
                "selection": None, "cost_bps": [5]}
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_missing_fields_are_named_and_rejected(self):
        from research_app.runtime import CliError
        with self.assertRaises(CliError) as caught:
            self.buffer.cmd_buffer_add(self.args(name=None, category=None, tags=None,
                                                 description="short"), None)
        message = caught.exception.message
        for field in ("name", "category", "tags", "description"):
            self.assertIn(field, message)


class SeedArmLedgerTests(unittest.TestCase):
    def test_record_arm_preserves_unexplained_failure_without_fabricating_metrics(self):
        import alpha_registry
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "registry.jsonl")
            base = {"alpha_id": None, "expression": "rank(ts_mean(fnd6_x,20))",
                    "settings": {"decay": 8, "region": "USA", "universe": "TOP3000"},
                    "metrics": {}, "checks": {}, "status": "ERROR", "error": "failed execution"}
            first, written = alpha_registry.record_arm(alpha_registry.build_entry(base, idea=None), path)
            self.assertTrue(written)
            _, written_again = alpha_registry.record_arm(alpha_registry.build_entry(base, idea=None), path)
            self.assertFalse(written_again)
            complete = {**base, "alpha_id": "RealAlpha", "metrics": {"sharpe": 2.0, "fitness": 1.5},
                        "status": "WARNING", "error": None}
            alpha_registry.record_arm(alpha_registry.build_entry(complete, idea="momentum"), path)
            stored = alpha_registry.load_entries(path)
            row = next(r for r in stored if r.get("key") == first["key"])
            self.assertEqual(len(stored), 2)
            self.assertEqual(row["metrics"], {})
            self.assertIsNone(row["idea"])
            self.assertEqual(next(r for r in stored if r.get("alpha_id") == "RealAlpha")["metrics"]["sharpe"], 2.0)
