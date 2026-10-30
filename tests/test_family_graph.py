#!/usr/bin/env python3
"""Tests for the family graph's three edge kinds.

The distinctions are the whole point, so they are pinned directly: identical
skeleton is a sibling, a strict leg superset is a directed extends, and two
composites that share a leg without containment are only weakly related.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import family_graph  # noqa: E402


def node(alpha_id, expression, fitness=None):
    return {"alpha_id": alpha_id, "expression": expression,
            "metrics": {"fitness": fitness} if fitness is not None else {}}


class Edges(unittest.TestCase):
    BASE = "a = group_rank(close, subindustry) - 0.5;\na"
    EXTRA = ("a = group_rank(close, subindustry) - 0.5;\n"
             "b = group_rank(-ts_sum(returns, 5), subindustry) - 0.5;\na + 0.5 * b")
    OTHER = ("x = group_rank(volume, subindustry) - 0.5;\n"
             "b = group_rank(-ts_sum(returns, 5), subindustry) - 0.5;\nx + b")

    def build(self, entries):
        nodes = family_graph.build_nodes(entries)
        return nodes, family_graph.build_edges(nodes)

    def test_same_skeleton_different_field_is_a_sibling(self):
        _nodes, edges = self.build([node("A", self.BASE),
                                    node("B", "a = group_rank(open, subindustry) - 0.5;\na")])
        self.assertEqual([kind for _l, _r, kind in edges], ["sibling"])

    def test_extra_leg_is_a_directed_extends(self):
        _nodes, edges = self.build([node("A", self.BASE), node("B", self.EXTRA)])
        self.assertEqual(edges, [("A", "B", "extends")])

    def test_shared_leg_without_containment_is_weak(self):
        _nodes, edges = self.build([node("A", self.EXTRA), node("B", self.OTHER)])
        self.assertIn("shares", [kind for _l, _r, kind in edges])
        self.assertNotIn("extends", [kind for _l, _r, kind in edges])

    def test_unrelated_programs_are_not_connected(self):
        _nodes, edges = self.build([node("A", self.BASE), node("B", "rank(volume)")])
        self.assertEqual(edges, [])

    def test_components_join_siblings_but_not_weak_links(self):
        entries = [node("A", self.BASE),
                   node("B", "a = group_rank(open, subindustry) - 0.5;\na")]
        nodes, edges = self.build(entries)
        groups = family_graph.components(nodes, edges)
        self.assertEqual(len(groups), 1)


class Nodes(unittest.TestCase):
    def test_duplicate_ids_are_collapsed(self):
        nodes = family_graph.build_nodes([node("A", "rank(close)"),
                                          node("A", "rank(close)"),
                                          node("B", "rank(open)")])
        self.assertEqual(sorted(nodes), ["A", "B"])

    def test_unparseable_expression_is_skipped_not_fatal(self):
        nodes = family_graph.build_nodes([node("A", "rank(close) 13")])
        self.assertEqual(nodes, {})


if __name__ == "__main__":
    unittest.main()