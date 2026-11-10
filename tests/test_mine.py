"""The mining tool: untouched, tradable cadence, and a type you can actually use."""

import json
import os
import tempfile
import unittest
from unittest import mock

import research_paths
from research_app import mine


class MineTests(unittest.TestCase):
    def _workspace(self, fields, expressions):
        tmp = tempfile.mkdtemp()
        directory = os.path.join(tmp, "datafields")
        os.makedirs(directory)
        with open(os.path.join(directory, "a.json"), "w") as handle:
            json.dump({"payload": fields}, handle)
        registry = os.path.join(tmp, "r.jsonl")
        with open(registry, "w") as handle:
            for expression in expressions:
                handle.write(json.dumps({"alpha_id": "A", "expression": expression,
                                         "metrics": {"turnover": 0.12}}) + "\n")
        return tmp, registry

    def test_a_vector_field_is_wrapped_and_a_label_is_skipped(self):
        fields = [
            {"id": "assets", "dataset": {"id": "fundamental6"}, "type": "MATRIX",
             "coverage": 0.9, "description": "Total assets"},
            {"id": "buzz_vec", "dataset": {"id": "socialmedia12"}, "type": "VECTOR",
             "coverage": 0.9, "description": "sentiment volume"},
            {"id": "sector_code", "dataset": {"id": "pv13"}, "type": "GROUP",
             "coverage": 1.0, "description": "grouping fields"},
        ]
        tmp, registry = self._workspace(fields, ["rank(assets)"])
        emitted = os.path.join(tmp, "spec.json")

        class Args:
            dataset = None
            min_coverage = 0.5
            turnover_band = "0.08,0.25"
            limit = 4
            arms = 4
            smooth = 0
            emit = emitted
            name = "t"
            registry = None
        args = Args()
        args.registry = registry

        with mock.patch.object(research_paths, "metadata_root", return_value=tmp):
            data = mine.cmd_mine(args, None)
            spec = json.load(open(emitted, encoding="utf-8"))

        labels = [arm["label"] for arm in spec["arms"]]
        self.assertEqual(labels, ["buzz_vec"])            # assets used, sector_code is a label
        self.assertEqual(spec["arms"][0]["expression"], "rank(vec_avg(buzz_vec))")
        self.assertTrue(data["spec"]["arms"] >= 1)

    def test_smoothing_is_part_of_the_first_shape(self):
        fields = [{"id": "buzz", "dataset": {"id": "socialmedia12"}, "type": "MATRIX",
                   "coverage": 0.9, "description": "attention"}]
        tmp, registry = self._workspace(fields, [])

        class Args:
            dataset = "socialmedia12"
            min_coverage = 0.5
            turnover_band = "0.08,0.25"
            limit = 2
            arms = 2
            smooth = 21
            emit = None
            name = "t"
            registry = None
        args = Args()
        args.registry = registry
        args.emit = os.path.join(tmp, "s.json")

        with mock.patch.object(research_paths, "metadata_root", return_value=tmp):
            mine.cmd_mine(args, None)
            spec = json.load(open(args.emit, encoding="utf-8"))
        self.assertEqual(spec["arms"][0]["expression"], "rank(ts_mean(buzz, 21))")

    def test_the_turnover_band_comes_from_what_the_dataset_produced(self):
        fields = [{"id": "slow_field", "dataset": {"id": "fundamental2"}, "type": "MATRIX",
                   "coverage": 0.9, "description": "x"}]
        tmp, registry = self._workspace(fields, ["rank(slow_field)"])
        entries = [{"alpha_id": "A", "expression": "rank(slow_field)",
                    "metrics": {"turnover": 0.02}}]
        with mock.patch.object(research_paths, "metadata_root", return_value=tmp):
            stats = mine.dataset_turnover(entries)
        self.assertEqual(stats["fundamental2"]["median"], 0.02)


class ShapeTests(unittest.TestCase):
    """Shape-first mining: keep the construction, change the field."""

    def test_the_known_shapes_apply_to_a_sub_expression(self):
        expression = "deferred_tax_assets_net_value / (assets + 1)"
        self.assertEqual(mine.apply_shape("centre", expression),
                         "-(group_rank(%s, subindustry) - 0.5)" % expression)
        self.assertEqual(mine.apply_shape("midtier", expression),
                         "-(abs(group_rank(%s, subindustry) - 0.5))" % expression)
        self.assertIn("ts_delta(", mine.apply_shape("change", expression))

    def test_an_unknown_shape_is_refused(self):
        with self.assertRaises(ValueError):
            mine.apply_shape("magic", "rank(close)")
