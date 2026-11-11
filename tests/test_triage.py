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
                             pnl_dir=directory)
        self.assertEqual(data["rows"][0]["verdict"], "blocked")

    def test_poor_test_retention_vetoes_even_when_the_book_rises(self):
        tmp, directory, dates, rows = self._workspace()
        rng = random.Random(5)
        self._write(directory, dates, "U2", [rng.gauss(3, 1) for _ in dates])
        rows.append({"alpha_id": "U2", "status": "COMPLETE",
                     "metrics": {"sharpe": 2.5, "fitness": 3.0}, "failing_checks": [],
                     "out_of_sample": {"train": {"sharpe": 3.0}, "test": {"sharpe": 0.2}}})
        data = triage.triage(["U2"], registry_path=self._registry(tmp, rows),
                             pnl_dir=directory)
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
