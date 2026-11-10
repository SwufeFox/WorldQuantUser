"""Marginal contribution and independence: the two numbers the campaign kept confusing."""

import json
import os
import tempfile
import unittest
from unittest import mock

import research_paths
from research_app import independence


class MarginalTests(unittest.TestCase):
    def _book(self, tmp, members=6, days=200):
        import random
        rng = random.Random(11)
        directory = os.path.join(tmp, "pnl")
        os.makedirs(directory, exist_ok=True)
        dates = ["2020-01-%03d" % (i + 1) for i in range(days)]
        rows = []
        for index in range(members):
            values = {d: rng.gauss(0, 1) for d in dates}
            with open(os.path.join(directory, "A%d.json" % index), "w") as handle:
                json.dump(values, handle)
            rows.append({"alpha_id": "A%d" % index, "status": "ACTIVE"})
        # a candidate that is pure noise: orthogonal, so the slope and the book move differ
        values = {d: rng.gauss(0, 1) for d in dates}
        with open(os.path.join(directory, "NEW.json"), "w") as handle:
            json.dump(values, handle)
        rows.append({"alpha_id": "NEW", "status": "COMPLETE"})
        registry = os.path.join(tmp, "registry.jsonl")
        with open(registry, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return registry, directory

    def test_the_slope_and_the_book_move_are_reported_separately(self):
        tmp = tempfile.mkdtemp()
        registry, directory = self._book(tmp)
        with mock.patch.object(research_paths, "ensure_dir", return_value=tmp):
            report = independence.marginal_report("NEW", registry_path=registry)
        self.assertIn("theoretical_marginal", report)
        self.assertIn("actual_book_change", report)
        # The book move must be far smaller than the slope: one member of seven.
        self.assertLess(abs(report["actual_book_change"]),
                        abs(report["theoretical_marginal"]) + 0.2)

    def test_a_missing_candidate_is_an_error(self):
        tmp = tempfile.mkdtemp()
        registry, _ = self._book(tmp)
        with mock.patch.object(research_paths, "ensure_dir", return_value=tmp):
            with self.assertRaises(ValueError):
                independence.marginal_report("NOPE", registry_path=registry)


class IndependenceTests(unittest.TestCase):
    def test_independent_members_give_more_effective_bets(self):
        import random
        rng = random.Random(3)
        tmp = tempfile.mkdtemp()
        directory = os.path.join(tmp, "pnl")
        os.makedirs(directory, exist_ok=True)
        dates = ["2020-01-%03d" % (i + 1) for i in range(400)]
        rows = []
        for index in range(5):
            values = {d: rng.gauss(0, 1) for d in dates}
            with open(os.path.join(directory, "I%d.json" % index), "w") as handle:
                json.dump(values, handle)
            rows.append({"alpha_id": "I%d" % index, "status": "ACTIVE"})
        registry = os.path.join(tmp, "r.jsonl")
        with open(registry, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        with mock.patch.object(research_paths, "ensure_dir", return_value=tmp):
            report = independence.independence_report(registry_path=registry)
        self.assertLess(abs(report["full_sample"]["mean"]), 0.25)
        self.assertGreater(report["full_sample"]["effective_bets"], 3.0)
