"""One table per batch: the four questions, and two of them are vetoes."""

import json
import os
import random
import tempfile
import unittest

from research_app import triage


class TriageTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        patcher = patch("research_paths.log_view")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _validate(self, tmp, directory, rows, alpha_id):
        registry = self._registry(tmp, rows)
        frozen = os.path.join(tmp, "selection.json")
        triage.triage([alpha_id], registry_path=registry, pnl_dir=directory, freeze_path=frozen)
        return triage.triage(show_test=True, selection_path=frozen)

    def _workspace(self, book_size=4, days=300):
        tmp = tempfile.mkdtemp()
        directory = os.path.join(tmp, "pnl")
        os.makedirs(directory)
        rng = random.Random(7)
        dates = ["2020-01-%03d" % (i + 1) for i in range(days)]
        rows = []
        for index in range(book_size):
            values = {d: rng.gauss(0, 1) for d in dates}
            with open(os.path.join(directory, "B%d.json" % index), "w") as handle:
                json.dump(values, handle)
            rows.append({"alpha_id": "B%d" % index, "status": "ACTIVE",
                         "metrics": {"sharpe": 1.0, "fitness": 1.0}})
        return tmp, directory, dates, rows

    def _write(self, directory, dates, alpha_id, values):
        with open(os.path.join(directory, alpha_id + ".json"), "w") as handle:
            json.dump({d: v for d, v in zip(dates, values)}, handle)

    def _registry(self, tmp, rows):
        path = os.path.join(tmp, "r.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def test_a_blocked_arm_is_reported_as_blocked(self):
        tmp, directory, dates, rows = self._workspace()
        rng = random.Random(3)
        self._write(directory, dates, "U1", [rng.gauss(2, 1) for _ in dates])
        rows.append({"alpha_id": "U1", "status": "COMPLETE",
                     "metrics": {"sharpe": 2.0, "fitness": 2.0},
                     "failing_checks": ["CONCENTRATED_WEIGHT"],
                     "out_of_sample": {"train": {"sharpe": 2.0}, "test": {"sharpe": 1.5}}})
        data = self._validate(tmp, directory, rows, "U1")
        self.assertEqual(data["rows"][0]["verdict"], "blocked")

    def test_poor_test_retention_vetoes_even_when_the_book_rises(self):
        tmp, directory, dates, rows = self._workspace()
        rng = random.Random(5)
        self._write(directory, dates, "U2", [rng.gauss(3, 1) for _ in dates])
        rows.append({"alpha_id": "U2", "status": "COMPLETE",
                     "metrics": {"sharpe": 2.5, "fitness": 3.0}, "failing_checks": [],
                     "out_of_sample": {"train": {"sharpe": 3.0}, "test": {"sharpe": 0.2}}})
        data = self._validate(tmp, directory, rows, "U2")
        row = data["rows"][0]
        self.assertEqual(row["retention"], 0.07)
        self.assertEqual(row["verdict"], "overfit-risk")

    def test_a_correlated_arm_is_dilutive(self):
        tmp, directory, dates, rows = self._workspace()
        # an arm that is a copy of the book itself adds nothing
        book_values = [sum(json.load(open(os.path.join(directory, "B%d.json" % i)))[d]
                           for i in range(4)) for d in dates]
        self._write(directory, dates, "U3", book_values)
        rows.append({"alpha_id": "U3", "status": "COMPLETE",
                     "metrics": {"sharpe": 2.0, "fitness": 2.0}, "failing_checks": [],
                     "out_of_sample": {"train": {"sharpe": 2.0}, "test": {"sharpe": 1.5}}})
        data = triage.triage(["U3"], registry_path=self._registry(tmp, rows),
                             pnl_dir=directory)
        self.assertEqual(data["rows"][0]["verdict"], "dilutive")


class TriageEvidenceBoundaryTests(unittest.TestCase):
    def setUp(self):
        from datetime import date, timedelta
        from pathlib import Path
        from unittest.mock import patch
        patcher = patch("research_paths.log_view")
        self.view_log = patcher.start()
        self.addCleanup(patcher.stop)

        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.pnl = self.root / "pnl"
        self.pnl.mkdir()
        self.registry = self.root / "alpha-registry.jsonl"
        self.train_days = [(date(2020, 1, 1) + timedelta(days=index)).isoformat()
                           for index in range(200)]
        self.test_days = [(date(2021, 1, 1) + timedelta(days=index)).isoformat()
                          for index in range(140)]
        rng = random.Random(73)
        self.rows = []
        for alpha_id, status, mean in (("B1", "ACTIVE", 0.05), ("B2", "ACTIVE", 0.05),
                                       ("C1", "COMPLETE", 0.5)):
            series = {day: rng.gauss(mean, 1) for day in self.train_days + self.test_days}
            (self.pnl / (alpha_id + ".json")).write_text(json.dumps(series), encoding="utf-8")
            self.rows.append({"key": alpha_id, "alpha_id": alpha_id, "status": status,
                              "metrics": {"sharpe": 9.0, "fitness": 8.0},
                              "failing_checks": [],
                              "out_of_sample": {"train": {"sharpe": 2.0, "fitness": 1.0},
                                                "test": {"startDate": "2021-01-01",
                                                         "sharpe": 1.8}}})
        self.save_registry()

    def save_registry(self):
        self.registry.write_text("\n".join(json.dumps(row) for row in self.rows), encoding="utf-8")

    def score(self, **kwargs):
        return triage.triage(["C1"], registry_path=str(self.registry),
                             pnl_dir=str(self.pnl), **kwargs)

    def test_holdout_changes_cannot_change_selection_or_leak_through_criteria(self):
        selected = self.score()
        frozen_path = self.root / "selection.json"
        self.score(freeze_path=frozen_path)
        row = selected["rows"][0]
        self.assertEqual(row["sharpe"], 2.0)
        self.assertEqual(row["fitness"], 1.0)
        self.assertEqual(row["overlap_days"], 200)
        self.assertNotIn("test_sharpe", row)
        self.assertNotIn("failing", row)
        self.assertNotIn("retention", row)
        for entry in self.rows:
            entry["metrics"] = {"sharpe": -99.0, "fitness": -88.0}
            entry["out_of_sample"]["test"]["sharpe"] = -7.0
            entry["failing_checks"] = ["LOW_SHARPE"]
            path = self.pnl / (entry["alpha_id"] + ".json")
            series = json.loads(path.read_text(encoding="utf-8"))
            series.update({day: -1000.0 for day in self.test_days})
            path.write_text(json.dumps(series), encoding="utf-8")
        self.save_registry()
        self.assertEqual(self.score(criteria={"show_test": True, "internal": True}), selected)
        validation = self.score(show_test=True, selection_path=frozen_path)
        validated = validation["rows"][0]
        self.assertEqual(validation["window"]["mode"], "full")
        self.assertEqual(validated["overlap_days"], 340)
        self.assertEqual(validated["test_sharpe"], -7.0)
        self.assertEqual(validated["verdict"], "blocked")

    def test_explicit_end_cannot_extend_past_a_recorded_test_boundary(self):
        data = self.score(train_end="2025-01-01")
        self.assertEqual(data["window"]["end"], "2021-01-01")
        self.assertEqual(data["rows"][0]["overlap_days"], 200)
        earlier = self.score(train_end="2020-06-01")
        self.assertEqual(earlier["rows"][0]["overlap_days"], 152)

    def test_absent_train_metrics_stay_unknown_despite_headline_values(self):
        self.rows[-1]["out_of_sample"].pop("train")
        self.save_registry()
        selected = self.score()["rows"][0]
        self.assertIsNone(selected["sharpe"])
        self.assertIsNone(selected["fitness"])
        self.assertEqual(selected["overlap_days"], 200)

    def test_only_explicit_cli_validation_records_a_test_view(self):
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        command = [sys.executable, str(root / "research_cli.py"), "triage", "--ids", "C1",
                   "--research-dir", str(self.root)]
        frozen_path = self.root / "cli-selection.json"
        selected = subprocess.run(command + ["--freeze", str(frozen_path)], capture_output=True, encoding="utf-8")
        self.assertEqual(selected.returncode, 0, selected.stdout + selected.stderr)
        self.assertFalse((self.root / "views.jsonl").exists())
        validated = subprocess.run(command + ["--show-test", "--selection", str(frozen_path)], capture_output=True, encoding="utf-8")
        self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)
        self.assertIn("test_sharpe", json.loads(validated.stdout)["data"]["rows"][0])
        logged = [json.loads(line) for line in (self.root / "views.jsonl").read_text(
            encoding="utf-8").splitlines()]
        self.assertEqual([(item["tool"], item["view"], item["alpha_ids"]) for item in logged],
                         [("triage", "test", ["C1"])])
        self.assertEqual(logged[0]["note"]["selection_id"], json.loads(validated.stdout)["data"]["selection_id"])

    def test_validation_requires_selection_and_rejects_input_scope_conflicts(self):
        with self.assertRaisesRegex(ValueError, "requires --selection"):
            self.score(show_test=True)
        frozen = self.root / "selection.json"
        self.score(freeze_path=frozen, min_delta=0.07, retention_floor=0.4)
        for kwargs in ({"ids": ["extra"]}, {"registry_path": str(self.root / "other.jsonl")},
                       {"pnl_dir": str(self.root / "other-pnl")}, {"train_end": "2025-01-01"},
                       {"min_delta": 0.02}, {"retention_floor": 0.2},
                       {"criteria": {"delta_full_credit": 9}}, {"cost_bps": [99]},
                       {"whole_book": True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "conflict"):
                triage.triage(show_test=True, selection_path=frozen, **kwargs)
        validated = triage.triage(show_test=True, selection_path=frozen)
        self.assertEqual(validated["criteria"]["min_delta"], 0.07)
        self.assertEqual(validated["criteria"]["retention_floor"], 0.4)

    def test_reversed_validation_ranking_keeps_frozen_rows_and_reports_missing(self):
        rng = random.Random(91)
        second = {day: rng.gauss(-0.5 if day in self.train_days else 20, 1)
                  for day in self.train_days + self.test_days}
        (self.pnl / "C2.json").write_text(json.dumps(second), encoding="utf-8")
        self.rows.append({"alpha_id": "C2", "status": "COMPLETE",
                          "metrics": {"fitness": 99},
                          "out_of_sample": {"train": {"fitness": -1},
                                            "test": {"startDate": "2021-01-01"}}})
        self.save_registry()
        frozen = self.root / "selection.json"
        selected = triage.triage(["C2", "C1"], registry_path=str(self.registry),
                                 pnl_dir=str(self.pnl), freeze_path=frozen)
        expected = [row["alpha_id"] for row in selected["rows"]]
        self.assertEqual(expected, ["C1", "C2"])
        path = self.pnl / "C1.json"
        values = json.loads(path.read_text(encoding="utf-8"))
        values.update({day: -20 + rng.gauss(0, 1) for day in self.test_days})
        path.write_text(json.dumps(values), encoding="utf-8")
        self.rows.append({"alpha_id": "extra", "metrics": {"fitness": 10000}})
        self.rows.reverse()
        self.save_registry()
        validated = triage.triage(["C2", "C1"], selection_path=frozen, show_test=True)
        self.assertEqual([row["alpha_id"] for row in validated["rows"]], expected)
        self.assertLess(validated["rows"][0]["delta"], validated["rows"][1]["delta"])
        self.assertEqual(validated["criteria"], selected["criteria"])
        self.assertEqual(validated["frozen_window"], selected["window"])
        self.rows = [row for row in self.rows if row["alpha_id"] != "C1"]
        self.save_registry()
        missing = triage.triage(selection_path=frozen, show_test=True)["rows"][0]
        self.assertTrue(missing["missing"])
        self.assertNotIn("sharpe", missing)
        self.assertNotIn("delta", missing)
        self.assertEqual(self.view_log.call_args.kwargs["note"]["selection_id"], selected["selection_id"])

    def test_equal_count_book_replacement_and_settings_change_remain_visible(self):
        frozen = self.root / "selection.json"
        selected = self.score(freeze_path=frozen)
        self.rows[0]["alpha_id"] = self.rows[0]["key"] = "B3"
        (self.pnl / "B3.json").write_bytes((self.pnl / "B1.json").read_bytes())
        self.rows[-1]["settings"] = {"universe": "CHANGED"}
        self.save_registry()
        validated = self.score(show_test=True, selection_path=frozen)
        self.assertEqual(validated["book_members"], selected["book_members"])
        self.assertTrue(validated["book_changed"])
        self.assertNotEqual(validated["book_state"]["local"]["member_digest"],
                            validated["frozen_book_state"]["local"]["member_digest"])
        change = validated["rows"][0]["source_changes"]["source_settings"]
        self.assertEqual(change, {"frozen": {}, "current": {"universe": "CHANGED"}})

    def test_costs_are_period_scoped_and_do_not_filter_training_rows(self):
        candidate = self.rows[-1]
        candidate["metrics"]["margin"] = 0.02
        candidate["out_of_sample"]["test"]["margin"] = -0.01
        self.save_registry()
        frozen = self.root / "selection.json"
        selected = self.score(freeze_path=frozen, cost_bps=[999])
        self.assertIsNone(selected["rows"][0]["costs"]["margin_bps"])
        validated = self.score(show_test=True, selection_path=frozen)["rows"][0]
        self.assertEqual(validated["costs"]["margin_bps"], 200)
        self.assertIsNone(validated["costs_by_period"]["train"]["margin_bps"])
        self.assertEqual(validated["costs_by_period"]["test"]["margin_bps"], -100)

    def test_zero_contribution_sorts_before_negative_and_unknown(self):
        from unittest.mock import patch
        self.rows.extend({"alpha_id": alpha_id} for alpha_id in ("C2", "C3"))
        self.save_registry()
        with patch("research_app.triage._score", side_effect=lambda alpha_id, *_:
                   {"alpha_id": alpha_id, "delta": {"C1": 0, "C2": -0.1, "C3": None}[alpha_id]}):
            result = triage.triage(["C2", "C3", "C1"], registry_path=str(self.registry), pnl_dir=str(self.pnl))
        self.assertEqual([row["alpha_id"] for row in result["rows"]], ["C1", "C2", "C3"])

    def test_platform_thresholds_are_preserved_without_local_pass_inference(self):
        self.rows[-1]["checks"] = {
            "SELF_CORRELATION": {"result": "FAIL", "value": 0.8, "limit": 0.7},
            "MATCHES_COMPETITION": {"result": "PASS"},
        }
        self.rows[-1]["failing_checks"] = []
        self.save_registry()
        frozen = self.root / "selection.json"
        self.score(freeze_path=frozen)
        row = self.score(show_test=True, selection_path=frozen)["rows"][0]
        self.assertEqual(row["verdict"], "blocked")
        self.assertEqual(row["platform_checks"]["SELF_CORRELATION"]["limit"], 0.7)
        self.assertFalse(row["correlation_policy"]["platform_pass_inferred"])
        self.assertEqual(row["correlation_policy"]["local_duplicate_threshold"], 0.9)
