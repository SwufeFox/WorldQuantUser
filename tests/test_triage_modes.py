"""Triage absorbed health, marginal and benchmark: one scorer, three entry points."""

import json
import os
import random
import tempfile
import unittest

from research_app import triage


class TriageModeTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        patcher = patch("research_paths.log_view")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _validate(self, tmp, directory, rows, alpha_id, **kwargs):
        registry = self._registry(tmp, rows)
        frozen = os.path.join(tmp, "selection.json")
        triage.triage([alpha_id], registry_path=registry, pnl_dir=directory, freeze_path=frozen)
        return triage.triage(show_test=True, selection_path=frozen, **kwargs)

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
        data = self._validate(tmp, directory, rows, "C1", internal=True)
        row = data["rows"][0]
        self.assertIn("internal_fitness", row)
        self.assertLessEqual(row["internal_fitness"], row["fitness"])
        self.assertEqual(row["retention"], 0.82)


class UnresolvedCheckTests(TriageModeTests):
    """A PENDING check is not a pass; it is an unverified record."""

    def test_pending_self_correlation_is_reported_as_unverified(self):
        tmp, directory, dates, rows = self._workspace()
        rng = random.Random(31)
        with open(os.path.join(directory, "P1.json"), "w") as handle:
            json.dump({d: rng.gauss(2, 1) for d in dates}, handle)
        rows.append({"alpha_id": "P1", "status": "COMPLETE",
                     "metrics": {"sharpe": 2.0, "fitness": 2.0},
                     "failing_checks": [],
                     "checks": {"SELF_CORRELATION": {"result": "PENDING"}},
                     "out_of_sample": {"train": {"sharpe": 2.0}, "test": {"sharpe": 1.0}}})
        data = self._validate(tmp, directory, rows, "P1")
        row = data["rows"][0]
        self.assertIn("SELF_CORRELATION", row["unresolved"])
        self.assertEqual(row["verdict"], "unverified")


class SubUniverseMarginTests(TriageModeTests):
    """How much room the sub-universe check has is a friction hint, not a score."""

    def test_the_margin_is_value_over_limit(self):
        tmp, directory, dates, rows = self._workspace()
        rng = random.Random(41)
        with open(os.path.join(directory, "S1.json"), "w") as handle:
            json.dump({d: rng.gauss(2, 1) for d in dates}, handle)
        rows.append({"alpha_id": "S1", "status": "COMPLETE",
                     "metrics": {"sharpe": 2.0, "fitness": 2.0}, "failing_checks": [],
                     "checks": {"LOW_SUB_UNIVERSE_SHARPE": {"result": "PASS",
                                                            "value": 1.14,
                                                            "limit": 1.08}},
                     "out_of_sample": {"train": {"sharpe": 2.0}, "test": {"sharpe": 1.5}}})
        data = self._validate(tmp, directory, rows, "S1")
        self.assertAlmostEqual(data["rows"][0]["sub_universe"]["margin"], 1.056, places=3)


class CohortPercentileTests(TriageModeTests):
    """A retention number means little without its cohort."""

    def test_the_percentile_places_a_candidate_inside_its_cohort(self):
        from research_app import triage as triage_module
        values = sorted([-0.3, 0.0, 0.2, 0.5, 0.9])
        self.assertAlmostEqual(triage_module._percentile(values, -0.3), 0.0)
        self.assertAlmostEqual(triage_module._percentile(values, 0.5), 60.0)
        self.assertAlmostEqual(triage_module._percentile(values, 0.95), 100.0)
        self.assertIsNone(triage_module._percentile([], 0.5))


class PerMemberGateTests(TriageModeTests):
    """The platform gates on the worst single member, not on the book aggregate."""

    def test_an_alpha_that_copies_one_member_is_a_duplicate(self):
        tmp, directory, dates, rows = self._workspace()
        source = json.load(open(os.path.join(directory, "B0.json")))
        with open(os.path.join(directory, "COPY.json"), "w") as handle:
            json.dump(source, handle)
        rows.append({"alpha_id": "COPY", "status": "COMPLETE",
                     "metrics": {"sharpe": 2.0, "fitness": 2.0},
                     "out_of_sample": {"train": {"sharpe": 2.0}, "test": {"sharpe": 1.5}}})
        data = triage.triage(["COPY"], registry_path=self._registry(tmp, rows),
                             pnl_dir=directory)
        row = data["rows"][0]
        self.assertGreater(row["max_member_rho"], 0.99)
        self.assertEqual(row["verdict"], "duplicate")


class MemberCorrelationSignTests(unittest.TestCase):
    """max_member_rho is a magnitude: a -0.75 member is not a safe neighbour."""

    def test_a_negatively_correlated_member_still_reads_as_a_duplicate(self):
        import numpy as np
        from research_app import triage
        days = ["2020-%03d" % (i + 1) for i in range(300)]
        rng = np.random.default_rng(7)
        base = rng.normal(size=300)
        book = {days[i]: base[i] for i in range(300)}
        # Mirror image of the book member: strongly related, opposite sign.
        mirror = {days[i]: -base[i] * 10 for i in range(300)}
        from research_app import legs
        original_load = legs.load_pnl
        legs.load_pnl = lambda alpha_id, directory: (mirror if alpha_id == "cand" else book)
        try:
            series = {"member": book}
            row = {"alpha_id": "cand"}
            row = triage._score("cand", {"expression": "rank(close)", "metrics": {"fitness": 1.0}},
                                series, days, {"pnl_dir": ".", "cohort": [], "min_delta": 0.02,
                                               "retention_floor": 0.2, "internal": False,
                                               "show_test": False})
        finally:
            legs.load_pnl = original_load
        self.assertAlmostEqual(row["max_member_rho"], 1.0, places=2)
        self.assertLess(row["max_member_rho_signed"], 0.0)
        self.assertEqual(row["verdict"], "duplicate")
