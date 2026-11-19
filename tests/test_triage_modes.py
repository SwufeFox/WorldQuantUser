"""Triage absorbed health, marginal and benchmark: one scorer, three entry points."""

import hashlib
import json
import math
import os
import random
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from research_app import stats, triage


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

    def test_matrix_rechecks_one_ulp_winner_with_scalar_reduction(self):
        days = ["2020-%03d" % index for index in range(140)]
        values = [float(index) for index in range(len(days))]
        series = {member: dict(zip(days, values)) for member in ("first", "second")}
        units = stats.unit_correlation_rows([values, values])
        self.assertIsNotNone(units)
        lower = 0.9
        higher = math.nextafter(lower, math.inf)
        with patch.object(triage.legs, "correlation", side_effect=[lower, higher]) as scalar:
            count, winner = triage._member_correlations(values, series, days, units)
        self.assertEqual(scalar.call_count, 2)
        self.assertEqual((count, winner), (2, (higher, "second")))

    def test_candidate_missing_dates_keeps_scalar_date_intersection(self):
        days = ["2020-%03d" % index for index in range(145)]
        rng = random.Random(9)
        series = {member: {day: rng.normalvariate(0, 1) for day in days}
                  for member in ("book1", "book2")}
        subset = [day for index, day in enumerate(days) if index % 11 != 0]
        candidate = {day: -series["book1"][day] + 0.05 for day in subset}
        units = stats.unit_correlation_rows([list(values.values()) for values in series.values()])
        book_daily = {day: sum(values[day] for values in series.values()) for day in days}
        criteria = {"pnl_dir": ".", "show_test": False, "min_delta": 0.02, "retention_floor": 0.2}
        entry = {"expression": "rank(close)"}
        legacy = triage._score("cand", entry, series, days, criteria, candidate)
        measured = triage._score("cand", entry, series, days, criteria, candidate,
                                 book_daily=book_daily, unit_book=units)
        self.assertEqual(measured, legacy)
        self.assertEqual(measured["overlap_days"], len(subset))
        self.assertEqual(measured["max_member_rho_signed"], -1.0)


class ShelfTrainingEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.directory = root / "pnl"
        (self.directory / ".meta").mkdir(parents=True)
        self.registry = root / "ledger.jsonl"
        self.shelf = root / "shelf.jsonl"
        self.frozen = root / "selection.json"
        self.cutoff = "2020-07-01"
        self.days = [(date(2020, 1, 1) + timedelta(days=index)).isoformat() for index in range(240)]
        self.records = [self.entry("book", "2020-08-01", status="ACTIVE"),
                        self.entry("cand", "2020-09-01"),
                        self.entry("peer", self.cutoff)]
        self.write_lines(self.registry, self.records)
        self.write_lines(self.shelf, [{"alpha_id": "peer", "event": "shelved", "eligible": True}])
        self.pnl("book", 1, 0.05)
        self.pnl("cand", 2, 0.2)
        self.pnl("peer", 4, 0.2)

    @staticmethod
    def entry(alpha_id, test_start, status="COMPLETE"):
        return {"alpha_id": alpha_id, "status": status, "expression": "rank(close)",
                "settings": {"region": "USA"},
                "out_of_sample": {"train": {"sharpe": 1.5},
                                  "test": {"startDate": test_start, "sharpe": 3}}}

    @staticmethod
    def write_lines(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def pnl(self, alpha_id, pace, drift):
        # Beyond the shelf peer's boundary, all series have a large shared shock.
        values = {day: (drift + (-1) ** (i // pace) if day < self.cutoff else 100 * i)
                  for i, day in enumerate(self.days)}
        wire = json.dumps(values, sort_keys=True, separators=(",", ":"))
        (self.directory / (alpha_id + ".json")).write_text(wire)
        (self.directory / ".meta" / (alpha_id + ".json")).write_text(json.dumps({
            "format": "worldquantuser.pnl-evidence", "alpha_id": alpha_id, "complete": True,
            "data_sha256": hashlib.sha256(wire.encode()).hexdigest(),
        }))

    def screen(self, **kwargs):
        return triage.triage(["cand"], registry_path=self.registry, pnl_dir=self.directory,
                             against_shelf=True, shelf_path=self.shelf, **kwargs)

    def test_same_train_cutoff_for_shelf_book_and_marginal_halves(self):
        result = self.screen(freeze_path=self.frozen, max_shelf_correlation=0.7)
        row = result["rows"][0]
        self.assertIsNotNone(result.get("selection_id"))
        self.assertEqual(row["training_gate"]["state"], "pass")
        self.assertEqual(row["evidence"]["state"], "measured")
        self.assertEqual(row["evidence"]["window"]["end"], self.cutoff)
        self.assertEqual(row["evidence"]["overlap"]["days"], row["shelf_correlation"]["max_member_overlap_days"])
        self.assertLess(row["shelf_correlation"]["max_rho"], 0.05)
        self.assertEqual(sum(half["days"] for half in row["train_delta_by_half"]), row["overlap_days"])
        self.assertNotIn("test_sharpe", row)
        self.assertEqual(result["criteria"]["max_shelf_correlation"], 0.7)

    def test_missing_pnl_and_unverified_book_boundary_leave_no_freeze(self):
        (self.directory / ".meta" / "peer.json").unlink()
        result = self.screen(freeze_path=self.frozen, max_shelf_correlation=0.7)
        row = result["rows"][0]
        self.assertEqual(row["shelf_correlation"]["missing"][0]["reason"], "unverified_pnl_coverage")
        self.assertEqual(row["training_gate"]["state"], "unknown")
        self.assertFalse(self.frozen.exists())
        self.records[0]["out_of_sample"]["test"].pop("startDate")
        self.write_lines(self.registry, self.records)
        row = self.screen(freeze_path=self.frozen, max_shelf_correlation=0.7)["rows"][0]
        self.assertIsNone(row["delta"])
        self.assertEqual(row["evidence"]["missing_train_boundary_ids"], ["book"])
        self.assertEqual(row["training_gate"]["state"], "unknown")
        self.assertFalse(self.frozen.exists())

    def test_scoped_validation_rejects_shelf_and_source_drift_before_reading_test(self):
        self.screen(freeze_path=self.frozen, max_shelf_correlation=0.7)
        with patch.object(triage, "_cohort_retentions") as read_test, patch("research_paths.log_view") as view:
            self.write_lines(self.shelf, [])
            with self.assertRaisesRegex(ValueError, "membership changed"):
                triage.triage(show_test=True, selection_path=self.frozen)
            self.write_lines(self.shelf, [{"alpha_id": "peer", "event": "shelved", "eligible": True}])
            self.records[1]["expression"] = "rank(open)"
            self.write_lines(self.registry, self.records)
            with self.assertRaisesRegex(ValueError, "expression or settings changed"):
                triage.triage(show_test=True, selection_path=self.frozen)
            read_test.assert_not_called()
            view.assert_not_called()

    def test_rejected_shelf_duplicate_leaves_no_freeze(self):
        self.pnl("peer", 2, 0.2)
        row = self.screen(freeze_path=self.frozen, max_shelf_correlation=0.7)["rows"][0]
        self.assertEqual(row["training_gate"]["reason"], "shelf_duplicate")
        self.assertFalse(self.frozen.exists())

    def test_python_has_no_fast_expr_operator_count_or_warning(self):
        self.records[1]["settings"]["language"] = "PYTHON"
        self.records[1]["expression"] = "def alpha(frame):\n    return frame.rank()\n"
        self.write_lines(self.registry, self.records)
        selected = self.screen(freeze_path=self.frozen)
        self.assertIsNone(selected["rows"][0]["ops"])
        with patch.object(triage.benchmark, "ops_warning") as warning, patch("research_paths.log_view"):
            validated = triage.triage(show_test=True, selection_path=self.frozen, internal=True)
        warning.assert_not_called()
        self.assertIsNone(validated["rows"][0]["ops_warning"])
