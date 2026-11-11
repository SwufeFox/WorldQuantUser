"""Triage absorbed health, marginal and benchmark: one scorer, three entry points."""

import json
import os
import random
import tempfile
import unittest

from research_app import triage


class TriageModeTests(unittest.TestCase):
    def _workspace(self, book_size=5, days=300):
        tmp = tempfile.mkdtemp()
        directory = os.path.join(tmp, "pnl")
        os.makedirs(directory)
        rng = random.Random(11)
        dates = ["2020-02-%03d" % (i + 1) for i in range(days)]
        rows = []
        for index in range(book_size):
            values = {d: rng.gauss(0, 1) for d in dates}
            with open(os.path.join(directory, "B%d.json" % index), "w") as handle:
                json.dump(values, handle)
            rows.append({"alpha_id": "B%d" % index, "status": "ACTIVE",
                         "metrics": {"sharpe": 1.0, "fitness": 1.0}})
        return tmp, directory, dates, rows

    def _registry(self, tmp, rows):
        path = os.path.join(tmp, "r.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def test_book_mode_measures_what_removing_a_member_costs(self):
        tmp, directory, dates, rows = self._workspace()
        data = triage.triage(registry_path=self._registry(tmp, rows), pnl_dir=directory,
                             whole_book=True)
        self.assertEqual(data["mode"], "book")
        self.assertEqual(len(data["rows"]), 5)
        for row in data["rows"]:
            self.assertTrue(row.get("member"))
            self.assertIn(row["verdict"], ("drag", "carries", "member"))

    def test_internal_mode_adds_the_discounted_fitness(self):
        tmp, directory, dates, rows = self._workspace()
        rng = random.Random(21)
        with open(os.path.join(directory, "C1.json"), "w") as handle:
            json.dump({d: rng.gauss(1, 1) for d in dates}, handle)
        rows.append({"alpha_id": "C1", "status": "COMPLETE",
                     "metrics": {"sharpe": 2.0, "fitness": 3.0},
                     "out_of_sample": {"train": {"sharpe": 1.7}, "test": {"sharpe": 1.4}}})
        data = triage.triage(["C1"], registry_path=self._registry(tmp, rows),
                             pnl_dir=directory, internal=True)
        row = data["rows"][0]
        self.assertIn("internal_fitness", row)
        self.assertLessEqual(row["internal_fitness"], row["fitness"])
        self.assertEqual(row["retention"], 0.82)
