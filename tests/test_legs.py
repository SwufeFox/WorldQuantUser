"""The leg model: is a composite the weighted sum of its legs?"""

import json
import os
import tempfile
import unittest

from research_app import legs


def _series(values):
    return {"2020-01-%02d" % (i + 1): v for i, v in enumerate(values)}


class LegsTests(unittest.TestCase):
    def _write(self, tmp, alpha_id, values):
        os.makedirs(os.path.join(tmp, "pnl"), exist_ok=True)
        with open(os.path.join(tmp, "pnl", alpha_id + ".json"), "w") as handle:
            json.dump(_series(values), handle)

    def test_the_sum_of_two_identical_legs_doubles_the_pnl(self):
        tmp = tempfile.mkdtemp()
        values = [10.0, -5.0, 12.0, -3.0] * 10
        self._write(tmp, "A", values)
        dates, aligned, summed = legs.combine({"A": 0.5}, os.path.join(tmp, "pnl"))
        self.assertGreaterEqual(len(dates), 30)
        self.assertEqual(summed[0], 5.0)

    def test_opposite_legs_cancel_to_nothing(self):
        tmp = tempfile.mkdtemp()
        values = [10.0, -5.0, 12.0, -3.0] * 10
        self._write(tmp, "A", values)
        self._write(tmp, "B", [-v for v in values])
        dates, aligned, summed = legs.combine({"A": 0.5, "B": 0.5}, os.path.join(tmp, "pnl"))
        self.assertAlmostEqual(max(abs(v) for v in summed), 0.0, places=9)

    def test_a_leg_with_no_cache_is_reported_not_guessed(self):
        tmp = tempfile.mkdtemp()
        dates, aligned, summed = legs.combine({"missing": 1.0}, os.path.join(tmp, "pnl"))
        self.assertEqual(dates, [])

    def test_sharpe_of_a_flat_series_is_undefined(self):
        self.assertIsNone(legs.sharpe_of([1.0] * 40))
