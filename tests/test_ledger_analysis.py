"""Offline regression cases for the bounded ledger evidence report."""

import contextlib
from datetime import date, timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import alpha_registry
import machine_lib as ml
from research_app.app import main


def _row(alpha_id, *, mechanism="M1", region="ASI", score=1.8, checks=None, **overrides):
    if checks is None:
        checks = {"LOW_SHARPE": "PASS", "PROD_CORRELATION": "PASS"}
    return {
        "key": alpha_id, "alpha_id": alpha_id, "expression": "rank(close)",
        "status": "COMPLETE", "role": "candidate", "mechanism": mechanism,
        "family": "price", "recorded_at": "2025-03-15T12:00:00",
        "settings": {"region": region, "universe": "TOP3000", "delay": 1, "simulationMode": "FULL"},
        "metrics": {"sharpe": 99.0}, "checks": checks,
        "out_of_sample": {
            "train": {"startDate": "2019-01-01", "sharpe": score, "fitness": 1.2,
                      "turnover": 0.11, "margin": 0.003, "longCount": 160, "shortCount": 180},
            "test": {"startDate": "2022-01-01", "sharpe": 99.0},
        },
        **overrides,
    }


class LedgerAnalysisTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name)
        self.ledger = self.root / "alpha-registry.jsonl"

    def _write(self, *rows):
        with self.ledger.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def _run(self, *flags):
        with patch.object(ml, "load_session", side_effect=AssertionError("unexpected BRAIN login")), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = main(["--research-dir", str(self.root), "ledger", "analyze", *flags])
        return code, json.loads(stdout.getvalue())

    def test_report_separates_evidence_gaps_and_native_failures(self):
        self._write(
            _row("STALE", score=3.2), _row("A1", score=2.1),
            _row("A2", score=1.9, checks={"LOW_SHARPE": "PASS", "CLUSTER_TEST": "WARNING"}),
            _row("A3", mechanism="M2", checks={"PROD_CORRELATION": "PENDING"}),
            _row("NO_CHECKS", checks={}),
            _row("FAIL_PC", mechanism="M3", checks={"PROD_CORRELATION": "FAIL"}),
            _row("FAIL_BOTH", checks={"PROD_CORRELATION": "FAIL", "LOW_FITNESS": "ERROR"}),
            _row("DIAG", score=8.0, role="diagnostic"),
            _row("ACTIVE", score=8.0, status="ACTIVE"),
            _row("NO_SPLIT", out_of_sample={}),
            _row("OTHER", region="USA"),
            _row("STALE", score=0.7),
        )
        code, result = self._run("--region", "asi", "--mode", "FULL", "--group-by", "mechanism", "--yearly")
        self.assertEqual(code, 0, result)
        report = result["data"]
        self.assertEqual(report["counts"]["entries"], 10)
        self.assertEqual(sum(report["funnel"].values()), 10)
        self.assertEqual(report["funnel"]["hard_failure"], 2)
        self.assertEqual(report["funnel"]["weak_training"], 1)
        self.assertEqual(report["funnel"]["training_unknown"], 1)
        self.assertEqual(report["funnel"]["checks_missing"], 1)
        self.assertEqual(report["funnel"]["research_screen"], 3)
        self.assertEqual(report["funnel"]["diagnostic"], 1)
        self.assertEqual(report["funnel"]["active_or_submitted"], 1)
        self.assertEqual(report["failure_drivers"][0], {
            "check_name": "PROD_CORRELATION", "entries": 2, "sole_failure_entries": 1,
            "otherwise_screened_entries": 1,
        })
        self.assertEqual([item["alpha_id"] for item in report["representatives"]], ["A1", "A3"])
        self.assertEqual(report["representatives"][1]["unresolved_checks"], ["PROD_CORRELATION"])
        self.assertEqual(report["representatives"][0]["net_margin_bps"], 20.0)
        self.assertEqual(report["yearly_train"]["representative_counts"], {"unavailable": 2})
        self.assertEqual([item["alpha_id"] for item in report["single_failure_leads"]], ["FAIL_PC"])
        self.assertNotIn("test_sharpe", json.dumps(report))

    def test_diagnostic_alpha_alias_is_excluded_even_outside_the_cohort(self):
        self._write(
            _row("A1", score=9.9),
            _row("A2", score=9.8, checks={"PROD_CORRELATION": "FAIL"}),
            _row("A1", key="alias-1", region="USA", role="diagnostic"),
            _row("A2", key="alias-2", region="USA", role="diagnostic"),
            _row("A3"),
        )
        code, result = self._run("--region", "ASI")
        self.assertEqual(code, 0, result)
        data = result["data"]
        self.assertEqual(data["funnel"]["diagnostic"], 2)
        self.assertEqual([row["alpha_id"] for row in data["representatives"]], ["A3"])
        self.assertEqual(data["single_failure_leads"], [])

    def test_yearly_uses_only_verified_full_training_years(self):
        self._write(_row("A1"))
        pnl = self.root / "pnl"
        (pnl / ".meta").mkdir(parents=True)
        current, last = date(2019, 1, 1), date(2023, 1, 1)
        values = {}
        while current < last:
            if current.weekday() < 5:
                values[current.isoformat()] = ((-1) ** current.toordinal() + 0.15
                                                if current.year < 2022 else -1000)
            current += timedelta(days=1)
        (pnl / "A1.json").write_text(json.dumps(values))
        metadata = {"format": ml._PNL_EVIDENCE_FORMAT, "alpha_id": "A1", "complete": True,
                    "as_of": "2025-03-15", "data_sha256": ml._pnl_digest(values)}
        (pnl / ".meta" / "A1.json").write_text(json.dumps(metadata))

        code, result = self._run("--yearly", "--limit", "1")
        self.assertEqual(code, 0, result)
        annual = result["data"]["representatives"][0]["yearly_train"]
        self.assertEqual(annual["state"], "pass")
        self.assertEqual([item["year"] for item in annual["years"]], [2019, 2020, 2021])
        self.assertGreaterEqual(annual["recent_three_mean"], 1.0)

        values["2022-03-01"] = 1000
        (pnl / "A1.json").write_text(json.dumps(values))
        _, result = self._run("--yearly")
        self.assertEqual(result["data"]["representatives"][0]["yearly_train"]["reason"],
                         "cache_completeness_unverified")

    def test_incremental_analysis_keeps_authority_and_protects_outputs(self):
        self._write(_row("A1"))
        _, result = self._run()
        self.assertEqual(result["data"]["counts"]["entries"], 1)
        self._write(_row("A2", mechanism="M2"))
        with patch.object(alpha_registry, "load_entries", side_effect=AssertionError("raw ledger scan")):
            code, result = self._run("--min-train-sharpe", "2.0")
        self.assertEqual(code, 0, result)
        self.assertEqual(result["data"]["counts"]["entries"], 2)
        self.assertEqual(result["data"]["funnel"]["weak_training"], 2)
        previous = self.ledger.read_bytes()
        code, result = self._run("--save", str(self.ledger))
        self.assertNotEqual(code, 0, result)
        self.assertEqual(self.ledger.read_bytes(), previous)

    def test_unknown_cohort_and_invalid_options(self):
        self._write(_row("A1"))
        _, result = self._run("--region", "EUR")
        self.assertEqual(result["data"]["counts"]["entries"], 0)
        self.assertEqual(result["data"]["representatives"], [])
        for flags in (("--limit", "101"), ("--since", "2025-02-30"), ("--cost-bps", "-1")):
            with self.subTest(flags=flags):
                code, result = self._run(*flags)
                self.assertEqual(code, 2, result)
