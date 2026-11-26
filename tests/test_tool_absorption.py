"""Tests for candidate gates, submission planning, descriptions, rule snapshots and wiring."""

import argparse
from datetime import date, timedelta
import io
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import machine_lib as ml
from research_app import (
    alphas,
    app,
    buffer_cmd,
    capabilities,
    description,
    pp_budget,
    rules_snapshot,
    selection,
    submission_planner,
    triage,
)
from tools import _runner, atom_runner


class PpBudgetTests(unittest.TestCase):
    def test_backfill_is_exempt_from_the_charged_count(self):
        budget = pp_budget.pp_budget(
            "group_rank(ts_backfill(fnd6_assets, 21), industry) - 0.5",
            field_datasets={"fnd6_assets": "fundamental6"})
        self.assertEqual(budget["free_operators"], 1)
        # ts_backfill (free) + group_rank + the binary minus
        self.assertEqual(budget["operators_total"], 3)
        self.assertEqual(budget["operators"], 2)
        self.assertEqual(budget["n_fields"], 1)  # industry is a grouping key
        self.assertTrue(budget["single_dataset"])
        self.assertTrue(budget["ok"])

    def test_unary_minus_and_symbols_are_charged(self):
        budget = pp_budget.pp_budget("-rank(close) + rank(open)",
                                     field_datasets={"close": "pv1", "open": "pv1"})
        # rank, unary minus, rank, binary plus
        self.assertEqual(budget["operators_total"], 4)
        self.assertEqual(budget["n_fields"], 2)

    def test_missing_dataset_evidence_is_not_a_pass(self):
        budget = pp_budget.pp_budget("rank(mystery_field)", field_datasets={})
        self.assertFalse(budget["ok"])
        self.assertIn("no dataset evidence", " ".join(budget["issues"]))
        self.assertFalse(budget["single_dataset"])

    def test_field_limit_and_platform_operator_count(self):
        expression = "rank(a) + rank(b) + rank(c) + rank(d)"
        budget = pp_budget.pp_budget(
            expression, field_datasets={k: "pv1" for k in "abcd"},
            platform_operator_count=9)
        self.assertEqual(budget["operator_count_source"], "platform")
        self.assertEqual(budget["operators"], 9)
        self.assertIn("4 data fields > 3", " ".join(budget["issues"]))

    def test_bool_and_negative_platform_operator_counts_are_ignored(self):
        for invalid in (True, False, -1, 2.5):
            budget = pp_budget.pp_budget("rank(close)", field_datasets={"close": "pv1"},
                                         platform_operator_count=invalid)
            self.assertEqual(budget["operator_count_source"], "ast")
            self.assertEqual(budget["operators"], 1)

    def test_gate_summary_treats_absent_checks_as_unknown(self):
        budget = pp_budget.pp_budget("rank(a)", field_datasets={"a": "pv1"})
        summary = pp_budget.gate_summary(budget, check=[])
        self.assertIn("power_pool_correlation", summary["unknown"])
        self.assertFalse(summary["ok"])

    def test_gate_summary_blocks_on_measured_correlation_and_exact_half_boundary(self):
        budget = pp_budget.pp_budget("rank(a)", field_datasets={"a": "pv1"})
        for corr in (0.62, 0.5):
            check = [{"name": "POWER_POOL_CORRELATION", "result": "PASS", "value": corr, "limit": 0.5},
                     {"name": "LOW_SHARPE", "result": "PASS", "value": 1.4, "limit": 1.0}]
            summary = pp_budget.gate_summary(budget, check)
            self.assertIn("power_pool_correlation", summary["blocking"])

    def test_gate_summary_respects_platform_fail_pending_and_nonfinite_values(self):
        budget = pp_budget.pp_budget("rank(a)", field_datasets={"a": "pv1"})
        # Formal FAIL must never be overridden by a seemingly good numeric value.
        failed = pp_budget.gate_summary(budget, [
            {"name": "POWER_POOL_CORRELATION", "result": "FAIL", "value": 0.35, "limit": 0.5},
            {"name": "LOW_SHARPE", "result": "PENDING", "value": 1.8, "limit": 1.0},
        ])
        self.assertIn("power_pool_correlation", failed["blocking"])
        self.assertIn("sharpe", failed["unknown"])
        # Non-finite or boolean values remain unknown rather than passing.
        malformed = pp_budget.gate_summary(budget, [
            {"name": "POWER_POOL_CORRELATION", "result": "PASS", "value": True, "limit": 0.5},
            {"name": "LOW_SHARPE", "result": "PASS", "value": float("nan"), "limit": 1.0},
        ])
        self.assertIn("power_pool_correlation", malformed["unknown"])
        self.assertIn("sharpe", malformed["unknown"])


class SubmissionPlannerTests(unittest.TestCase):
    def test_conflict_drops_the_lighter_candidate(self):
        result = submission_planner.plan(
            [{"id": "a", "weight": 2.0}, {"id": "b", "weight": 1.0}, {"id": "c", "weight": 1.0}],
            {("a", "b"): 0.8, ("a", "c"): 0.1, ("b", "c"): 0.0})
        self.assertTrue(result["ok"])
        self.assertEqual(result["selected"], ["a", "c"])
        self.assertEqual(result["order"], ["a", "c"])
        self.assertTrue(result["optimal"])
        self.assertEqual([x["id"] for x in result["excluded"]], ["b"])

    def test_negative_correlation_is_a_conflict(self):
        result = submission_planner.plan(
            [{"id": "a", "weight": 1.0}, {"id": "b", "weight": 1.0}],
            {("a", "b"): -0.9})
        self.assertEqual(len(result["selected"]), 1)

    def test_missing_or_boolean_weight_refuses_instead_of_guessing(self):
        for bad in ([{"id": "a", "weight": 1.0}, {"id": "b"}],
                    [{"id": "a", "weight": True}]):
            result = submission_planner.plan(bad, {})
            self.assertFalse(result["ok"])

    def test_unmeasured_pairs_are_reported_and_never_co_selected(self):
        result = submission_planner.plan(
            [{"id": "a", "weight": 1.0}, {"id": "b", "weight": 1.0}, {"id": "c", "weight": 1.0}],
            {("a", "b"): 0.1})
        self.assertEqual(len(result["unknown_pairs"]), 2)  # a-c and b-c
        self.assertEqual(result["selected"], ["a", "b"])
        self.assertEqual(result["excluded"][0]["id"], "c")
        self.assertEqual(result["excluded"][0]["unmeasured_with"], ["a", "b"])
        self.assertIn("never assumed", result["evidence"]["correlations"])


class DescriptionTests(unittest.TestCase):
    def test_compose_refuses_missing_field_description_or_empty_fields(self):
        draft = description.compose("Leverage predicts underperformance.", ["liabilities"],
                                    field_notes={})
        self.assertFalse(draft["ok"])
        self.assertIn("liabilities", draft["missing"])
        empty_fields = description.compose("Idea text.", [], field_notes={})
        self.assertFalse(empty_fields["ok"])
        self.assertIn("fields", empty_fields["missing"])

    def test_compose_builds_a_checked_description_without_fabricating_neutralization(self):
        draft = description.compose(
            "Within a subindustry, higher leverage predicts lower forward returns.",
            ["liabilities", "assets"],
            field_notes={"liabilities": "total liabilities", "assets": "total assets"})
        self.assertTrue(draft["ok"])
        self.assertGreaterEqual(draft["check"]["length"], 100)
        self.assertTrue(draft["check"]["ok"], draft["check"]["errors"])
        self.assertEqual(
            [line for line in draft["text"].splitlines()
             if line in description.SECTION_TITLES],
            list(description.SECTION_TITLES),
        )
        for title in description.SECTION_TITLES:
            start = draft["text"].index(title) + len(title)
            end = min((draft["text"].index(other, start)
                       for other in description.SECTION_TITLES
                       if other in draft["text"][start:]), default=len(draft["text"]))
            self.assertTrue(any(line.startswith("- ")
                                for line in draft["text"][start:end].splitlines()))
        self.assertNotIn("negative rank", draft["text"].lower())
        self.assertNotIn("market beta", draft["text"].lower())

    def test_check_rejects_short_text_and_missing_sections(self):
        result = description.check("too short")
        self.assertFalse(result["ok"])
        self.assertTrue(any("length" in e for e in result["errors"]))
        self.assertTrue(any(description.SECTION_OPERATORS in e for e in result["errors"]))

    def test_check_rejects_the_old_uppercase_template(self):
        text = ("IDEA:\n- " + "leverage predicts returns. " * 5 +
                "\n\nRATIONALE:\n- " + "ranked inside subindustry. " * 4)
        result = description.check(text)
        self.assertFalse(result["ok"])
        self.assertFalse(result["template_ok"])
        self.assertTrue(any(description.SECTION_IDEA in e
                            for e in result["template_errors"]))

    def test_check_flags_a_near_duplicate(self):
        text = (
            "Idea:\n- " + "leverage predicts returns. " * 5 +
            "\n\nRationale for data used:\n- " + "assets and liabilities are documented inputs. " * 3 +
            "\n\nRationale for operators used:\n- rank compares firms inside subindustry. " * 3
        )
        result = description.check(text, others=[text])
        self.assertFalse(result["ok"])
        self.assertTrue(any("near-duplicate" in e for e in result["errors"]))


class RulesSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "rules.jsonl"

    def test_extract_keeps_quotas_themes_and_limits(self):
        payload = [
            {"name": "POWER_POOL_CORRELATION", "result": "PASS", "value": 0.31, "limit": 0.5},
            {"name": "REGULAR_SUBMISSION", "result": "PASS", "value": 0, "limit": 4},
            {"name": "MATCHES_THEMES", "result": "WARNING",
             "themes": [{"id": "t1", "name": "ASI/D1 Power Pool Dec`26", "multiplier": 1.0}]},
        ]
        records = rules_snapshot.extract_check_rules(
            payload, scope={"region": "EUR", "delay": 1}, observed_at="2026-11-25T10:00:00Z")
        kinds = {r["name"]: r["kind"] for r in records}
        self.assertEqual(kinds["REGULAR_SUBMISSION"], "quota")
        self.assertEqual(kinds["MATCHES_THEMES"], "themes")
        theme = next(r for r in records if r["name"] == "MATCHES_THEMES")
        self.assertEqual(theme["themes"][0]["multiplier"], 1.0)
        self.assertEqual(theme["scope"], {"region": "EUR", "delay": 1})

    def test_lookup_uses_true_utc_across_iso_formats_and_handles_invalid_stamps(self):
        from datetime import datetime, timezone

        t0 = datetime(2026, 11, 25, 10, 0, 0, tzinfo=timezone.utc).timestamp()
        rules_snapshot.append(self.path, [
            {"name": "LOW_SHARPE", "limit": 1.25, "scope": {"region": "USA"},
             "observed_at": "not-a-timestamp"},
            {"name": "LOW_SHARPE", "limit": 1.58, "scope": {"region": "USA"},
             "observed_at": "2026-11-25T10:00:00.500000+00:00"},
        ])
        # Inject a stray non-dict JSON line to verify load() skips it safely.
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write("[1, 2, 3]\n")
        records = rules_snapshot.load(self.path)
        found = rules_snapshot.lookup(records, "LOW_SHARPE", {"region": "USA"},
                                      max_age_seconds=60, now=t0 + 10.5)
        self.assertIsNotNone(found)
        self.assertEqual(found["rule"]["limit"], 1.58)
        self.assertAlmostEqual(found["age_seconds"], 10.0, places=3)
        self.assertFalse(found["stale"])
        self.assertIsNone(rules_snapshot.lookup(records, "LOW_SHARPE", {"region": "EUR"}))

        unparsed = rules_snapshot.lookup(
            [{"name": "X", "scope": {}, "observed_at": "corrupt"}],
            "X", max_age_seconds=60, now=t0)
        self.assertIsNone(unparsed["age_seconds"])
        self.assertIsNone(unparsed["stale"])

    def test_record_observed_checks_deduplicates_unchanged_rules(self):
        checks = {"LOW_SHARPE": {"result": "PASS", "value": 1.9, "limit": 1.58}}
        first = rules_snapshot.record_observed_checks(
            checks, scope={"region": "USA"}, path=self.path, observed_at="2026-11-25T10:00:00Z")
        second = rules_snapshot.record_observed_checks(
            checks, scope={"region": "USA"}, path=self.path, observed_at="2026-11-25T11:00:00Z")
        changed = rules_snapshot.record_observed_checks(
            {"LOW_SHARPE": {"result": "PASS", "value": 1.9, "limit": 1.65}},
            scope={"region": "USA"}, path=self.path, observed_at="2026-11-25T12:00:00Z")
        self.assertEqual(first["appended"], 1)
        self.assertEqual(second["appended"], 0)
        self.assertEqual(changed["appended"], 1)
        self.assertEqual(len(rules_snapshot.load(self.path)), 2)

    def test_official_rule_carries_its_source(self):
        rule = rules_snapshot.official_rule(
            "pp_max_operators", 8, source_id="power-pool",
            source_url="https://support.worldquantbrain.com/...",
            statement="operators including repeats <= 8")
        self.assertEqual(rule["source"], "official_doc")
        self.assertEqual(rule["source_id"], "power-pool")


class ModuleWiringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": str(self.root)})
        env.start()
        self.addCleanup(env.stop)

    def test_campaign_preview_runs_in_process_without_spawning_cli(self):
        spec = self.root / "preview.spec.json"
        spec.write_text(json.dumps({
            "track": "POWER_POOL",
            "design": "explore",
            "settings": {"simulationMode": "QUICK"},
            "arms": [{"label": "one", "expression": "rank(close)"}],
        }), encoding="utf-8")
        stdout = io.StringIO()
        with mock.patch.object(_runner, "run_cli", side_effect=AssertionError("spawned subprocess")), \
                mock.patch("sys.stdout", stdout):
            code = app.main(["--research-dir", str(self.root), "--quiet", "campaign", "preview", str(spec)])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["track"], "POWER_POOL")

    def test_alpha_get_and_buffer_add_attach_budget_description_and_rule_snapshots(self):
        alpha_payload = {
            "id": "A100",
            "status": "UNSUBMITTED",
            "name": "lev",
            "category": "FUNDAMENTAL",
            "regular": {"code": "group_rank(ts_backfill(fnd6_assets, 21), industry)", "operatorCount": 2},
            "settings": {"region": "USA", "universe": "TOP3000", "delay": 1, "simulationMode": "FULL"},
            "is": {"sharpe": 2.1, "fitness": 1.5, "turnover": 0.15,
                   "checks": [{"name": "LOW_SHARPE", "result": "PASS", "value": 2.1, "limit": 1.58},
                              {"name": "POWER_POOL_CORRELATION", "result": "PASS", "value": 0.32, "limit": 0.5}]},
            "oos": {"train": {"sharpe": 2.2, "fitness": 1.6, "turnover": 0.15, "returns": 0.1},
                    "test": {"sharpe": 1.8, "fitness": 1.2, "turnover": 0.15}},
        }
        scoped = {
            "records": [{"id": "fnd6_assets", "dataset": {"id": "fundamental6"}, "type": "MATRIX"}],
            "evidence": {"state": "verified"},
        }
        with mock.patch.object(alphas, "with_session", lambda _args, fn: fn(object())), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value=alpha_payload), \
                mock.patch.object(capabilities, "cached_field_scope", return_value=scoped):
            got = alphas.cmd_alpha_get(argparse.Namespace(alpha_id="A100"), None)
        self.assertIn("power_pool_budget", got)
        self.assertEqual(got["power_pool_budget"]["budget"]["operators"], 1)
        self.assertTrue(got["power_pool_budget"]["budget"]["single_dataset"])
        self.assertEqual(got["power_pool_budget"]["gates"]["gates"]["power_pool_correlation"]["state"], "pass")

        frozen = self.root / "selection.json"
        selection.freeze(frozen, source="triage", rows=[{"alpha_id": "A100"}])
        draft = description.compose(
            "Asset growth within industry predicts cross-sectional returns.",
            ["fnd6_assets"],
            field_notes={"fnd6_assets": "total book assets"},
        )
        alpha_payload["regular"]["description"] = draft["text"]
        check_payload = {
            "alpha_id": "A100",
            "complete": True,
            "self_correlation": {"result": "PASS", "value": 0.35},
            "checks": {"LOW_SHARPE": {"result": "PASS", "value": 2.1, "limit": 1.58},
                       "SELF_CORRELATION": {"result": "PASS", "value": 0.35, "limit": 0.7}},
        }
        args = argparse.Namespace(
            alpha_id="A100", buffer=str(self.root / "shelf.jsonl"),
            seeds=str(self.root / "seeds.jsonl"), registry=str(self.root / "ledger.jsonl"),
            selection=str(frozen), name="lev_a100", category="FUNDAMENTAL",
            tags=["leverage"], description=draft["text"], note=None, max_wait=None,
        )
        with mock.patch.object(buffer_cmd, "with_session", lambda _args, fn: fn(object())), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value=alpha_payload), \
                mock.patch.object(ml, "fetch_alpha_checks", return_value=check_payload):
            shelved = buffer_cmd.cmd_buffer_add(args, None)
        self.assertTrue(shelved["description_review"]["ok"])
        snap = rules_snapshot.load(rules_snapshot.default_snapshot_path())
        self.assertTrue(any(r["name"] == "LOW_SHARPE" and r.get("limit") == 1.58 for r in snap))

    def test_triage_builds_train_only_submission_plan_for_multiple_candidates(self):
        pnl_dir = self.root / "pnl"
        pnl_dir.mkdir()
        train_days = [(date(2020, 1, 1) + timedelta(days=i)).isoformat() for i in range(180)]
        test_days = [(date(2021, 1, 1) + timedelta(days=i)).isoformat() for i in range(140)]
        rng = random.Random(42)
        book_series = {d: rng.gauss(0.02, 1.0) for d in train_days + test_days}
        c1_base = [rng.gauss(0.4, 1.0) for _ in train_days]
        c3_base = [rng.gauss(0.35, 1.0) for _ in train_days]
        series_map = {
            "B1": book_series,
            "C1": {**dict(zip(train_days, c1_base)), **{d: 1.0 for d in test_days}},
            # C2 is identical to C1 on train days -> conflicts with C1 and has lower/equal weight
            "C2": {**dict(zip(train_days, [v * 0.99 for v in c1_base])), **{d: 1.0 for d in test_days}},
            # C3 is independent on train days
            "C3": {**dict(zip(train_days, c3_base)), **{d: 1.0 for d in test_days}},
        }
        rows = []
        for aid, series in series_map.items():
            (pnl_dir / f"{aid}.json").write_text(json.dumps(series), encoding="utf-8")
            rows.append({
                "alpha_id": aid,
                "status": "ACTIVE" if aid == "B1" else "COMPLETE",
                "metrics": {"sharpe": 2.0, "fitness": 1.5},
                "out_of_sample": {"train": {"sharpe": 2.0, "fitness": 1.5},
                                  "test": {"startDate": "2021-01-01", "sharpe": 1.5}},
            })
        registry = self.root / "ledger.jsonl"
        registry.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        first = triage.triage(["C1", "C2", "C3"], registry_path=str(registry), pnl_dir=str(pnl_dir))
        self.assertIn("submission_plan", first)
        plan = first["submission_plan"]
        self.assertTrue(plan["ok"])
        self.assertIn("C3", plan["selected"])
        self.assertEqual(len(plan["selected"]), 2)

        # Mutating test-period PnL must not alter the train submission_plan at all.
        for aid in ("C1", "C2", "C3"):
            p = pnl_dir / f"{aid}.json"
            updated = json.loads(p.read_text(encoding="utf-8"))
            for day in test_days:
                updated[day] = -999.0 if aid == "C3" else 999.0
            p.write_text(json.dumps(updated), encoding="utf-8")
        second = triage.triage(["C1", "C2", "C3"], registry_path=str(registry), pnl_dir=str(pnl_dir))
        self.assertEqual(second["submission_plan"], plan)

    def test_atom_runner_attaches_pp_budget_to_projected_rows(self):
        row = atom_runner._result_row(
            {"run_id": "r1"},
            0,
            {"label": "1: demo", "expression": "group_rank(ts_backfill(close, 5), industry)"},
            {"alpha_id": "A1", "status": "COMPLETE", "metrics": {"sharpe": 1.5, "turnover": 0.2},
             "checks": {"LOW_SHARPE": {"result": "PASS", "value": 1.5}}},
        )
        self.assertIn("pp_budget", row)
        self.assertEqual(row["pp_budget"]["budget"]["operators"], 1)
        self.assertEqual(row["pp_budget"]["budget"]["free_operators"], 1)
        self.assertEqual(row["pp_budget"]["gates"]["gates"]["sharpe"]["state"], "pass")


if __name__ == "__main__":
    unittest.main()
