"""One table per batch: the four questions, and two of them are vetoes."""

import json
import os
import random
import tempfile
import unittest

from research_app import triage


class TriageTests(unittest.TestCase):
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
        data = triage.triage(["U1"], registry_path=self._registry(tmp, rows),
                             pnl_dir=directory, show_test=True)
        self.assertEqual(data["rows"][0]["verdict"], "blocked")

    def test_poor_test_retention_vetoes_even_when_the_book_rises(self):
        tmp, directory, dates, rows = self._workspace()
        rng = random.Random(5)
        self._write(directory, dates, "U2", [rng.gauss(3, 1) for _ in dates])
        rows.append({"alpha_id": "U2", "status": "COMPLETE",
                     "metrics": {"sharpe": 2.5, "fitness": 3.0}, "failing_checks": [],
                     "out_of_sample": {"train": {"sharpe": 3.0}, "test": {"sharpe": 0.2}}})
        data = triage.triage(["U2"], registry_path=self._registry(tmp, rows),
                             pnl_dir=directory, show_test=True)
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
        validation = self.score(show_test=True)
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
        selected = subprocess.run(command, capture_output=True, encoding="utf-8")
        self.assertEqual(selected.returncode, 0, selected.stdout + selected.stderr)
        self.assertFalse((self.root / "views.jsonl").exists())
        validated = subprocess.run(command + ["--show-test"], capture_output=True, encoding="utf-8")
        self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)
        self.assertIn("test_sharpe", json.loads(validated.stdout)["data"]["rows"][0])
        logged = [json.loads(line) for line in (self.root / "views.jsonl").read_text(
            encoding="utf-8").splitlines()]
        self.assertEqual([(item["tool"], item["view"], item["alpha_ids"]) for item in logged],
                         [("triage", "test", ["C1"])])
