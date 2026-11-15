"""Visible/hidden union and global pagination; no platform requests."""

import argparse
import unittest
from unittest import mock

from research_app import alphas
from research_app.runtime import CliError


def arguments(**changes):
    values = dict(limit=20, offset=0, order="-is.sharpe", status="ACTIVE",
                  region="USA", universe=None, min_sharpe=None,
                  min_fitness=None, search=None, all=False)
    return argparse.Namespace(**{**values, **changes})


def record(alpha_id, sharpe):
    return {"id": alpha_id, "is": {"sharpe": sharpe}, "regular": {"code": "rank(close)"}}


class AlphaListTests(unittest.TestCase):
    def listing(self, args, visible, hidden):
        def fetch(session, *, limit, offset, hidden, **filters):
            self.assertEqual(filters["status"], "ACTIVE")
            self.assertEqual(filters["region"], "USA")
            self.assertEqual(filters["alpha_type_not"], "" if args.all else "SUPER")
            rows = hidden_rows if hidden else visible
            return rows[offset:offset + limit]

        hidden_rows = hidden
        with mock.patch.object(alphas, "with_session", side_effect=lambda a, work: work(None)), \
                mock.patch.object(alphas.ml, "list_alphas", side_effect=fetch) as calls:
            return alphas.cmd_alpha_list(args, None), calls.call_args_list

    def test_all_keeps_visible_alphas_when_hidden_scope_is_empty(self):
        page, calls = self.listing(arguments(all=True), [record("visible", 2)], [])
        self.assertEqual([row["alpha_id"] for row in page["alphas"]], ["visible"])
        self.assertEqual([call.kwargs["hidden"] for call in calls], [False, True])
        self.assertEqual(page["scope"]["source_prefix_counts"], {"visible": 1, "hidden": 0})
        self.assertFalse(page["scope"]["complete_listing"])

    def test_all_applies_offset_after_union_sort_and_deduplication(self):
        page, _ = self.listing(arguments(all=True, offset=1, limit=2),
                               [record("v1", 5), record("same", 2), record("v2", None)],
                               [record("h1", 4), record("same", 2), record("h2", 1)])
        self.assertEqual([row["alpha_id"] for row in page["alphas"]], ["h1", "same"])

    def test_all_fetches_prefix_pages_for_large_global_offset(self):
        visible = [record("v%03d" % n, 300 - 2 * n) for n in range(130)]
        hidden = [record("h%03d" % n, 299 - 2 * n) for n in range(130)]
        page, calls = self.listing(arguments(all=True, offset=110, limit=3), visible, hidden)
        self.assertEqual([row["alpha_id"] for row in page["alphas"]], ["v055", "h055", "v056"])
        self.assertEqual([(call.kwargs["hidden"], call.kwargs["offset"], call.kwargs["limit"])
                          for call in calls], [(False, 0, 100), (False, 100, 13),
                                               (True, 0, 100), (True, 100, 13)])

    def test_default_still_selects_only_visible_non_super_records(self):
        page, calls = self.listing(arguments(), [record("v", 1)], [record("h", 2)])
        self.assertEqual([row["alpha_id"] for row in page["alphas"]], ["v"])
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0].kwargs["hidden"], False)

    def test_all_handles_ascending_order_nulls_and_ties(self):
        page, _ = self.listing(arguments(all=True, order="is.sharpe"),
                               [record("v", 1), record("z", None)],
                               [record("h", 1), record("n", None)])
        self.assertEqual([row["alpha_id"] for row in page["alphas"]], ["v", "h", "z", "n"])

    def test_all_keeps_source_tie_order_at_prefix_boundary(self):
        page, _ = self.listing(arguments(all=True, offset=1, limit=1),
                               [record("z", 2), record("a", 2), record("b", 1)],
                               [record("h", 2)])
        self.assertEqual([row["alpha_id"] for row in page["alphas"]], ["a"])

    def test_all_rejects_compound_sort_instead_of_misordering_page(self):
        with self.assertRaises(CliError):
            self.listing(arguments(all=True, order="-is.sharpe,is.fitness"), [], [])


if __name__ == "__main__":
    unittest.main()
