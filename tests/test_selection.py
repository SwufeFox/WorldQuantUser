"""Training manifests are immutable, integrity checked, and ordered by identity."""

import json
import tempfile
import unittest
from pathlib import Path

from research_app import selection


class SelectionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "choice.json"
        self.rows = [{"alpha_id": "synthetic-a", "source_file": "batch-one.json",
                      "label": "first", "train_fitness": 2},
                     {"alpha_id": "synthetic-b", "source_file": "batch-one.json",
                      "label": "second", "train_fitness": 1}]

    def freeze(self, **kwargs):
        return selection.freeze(self.path, source="compare", rows=self.rows,
                                criteria={"minimum": 0}, inputs={"region": "USA"},
                                window={"end": "2020-12-31"}, **kwargs)

    def test_order_survives_reversed_metrics_missing_extra_and_input_order(self):
        manifest = self.freeze()
        fresh = [{**self.rows[1], "test_fitness": 99},
                 {"alpha_id": "extra", "test_fitness": 1000},
                 {**self.rows[0], "test_fitness": -99},
                 {"alpha_id": None, "status": "QUEUED"},
                 {"alpha_id": "extra", "test_fitness": 2000}]
        ordered = selection.ordered_rows(manifest, fresh)
        self.assertEqual([row["alpha_id"] for row in ordered], ["synthetic-a", "synthetic-b"])
        self.assertEqual([row["test_fitness"] for row in ordered], [-99, 99])
        missing = selection.ordered_rows(manifest, fresh[:2])
        self.assertTrue(missing[0]["missing"])
        self.assertNotIn("train_fitness", missing[0])
        self.assertNotIn("test_fitness", missing[0])
        self.assertEqual(missing[1]["test_fitness"], 99)

    def test_identical_freeze_is_idempotent_but_changed_content_cannot_overwrite(self):
        manifest = self.freeze()
        before = self.path.read_bytes()
        self.assertEqual(self.freeze(), manifest)
        self.rows.reverse()
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.freeze()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(selection.load(self.path)["selection_id"], manifest["selection_id"])

    def test_content_and_creation_time_tampering_are_rejected(self):
        manifest = self.freeze()
        for key, value in (("criteria", {"minimum": 5}), ("inputs", {"region": "EUR"}),
                           ("window", {"end": "2030-01-01"}),
                           ("created_at", "2021-01-01T00:00:00+00:00"),
                           ("rows", list(reversed(manifest["rows"]))),
                           ("identities", list(reversed(manifest["identities"])) )):
            with self.subTest(key=key):
                self.path.write_text(json.dumps({**manifest, key: value}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    selection.load(self.path)

    def test_missing_ids_and_ambiguous_duplicates_are_rejected(self):
        for rows in ([{}], [{"alpha_id": ""}], [{"alpha_id": "None"}],
                     [{"alpha_id": "A"}, {"alpha_id": "A", "fitness": 5}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                selection.freeze(self.path, source="triage", rows=rows)
        manifest = self.freeze()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            selection.ordered_rows(manifest, [self.rows[0], self.rows[0]])

    def test_source_specific_identifiers_and_cross_source_membership(self):
        rows = [{"alpha_id": "A", "source_file": name, "candidate_id": "arm"}
                for name in ("one.json", "two.json")]
        manifest = selection.freeze(self.path, source="compare", rows=rows)
        self.assertNotEqual(*manifest["identities"])
        self.assertEqual(selection.load(self.path)["source"], "compare")
        with self.assertRaisesRegex(ValueError, "source"):
            selection.load(self.path, source="triage")
        self.assertEqual(selection.load(self.path, source=("triage", "compare")), manifest)

    def test_hash_is_stable_across_mapping_order_and_changes_with_book_members(self):
        first = selection.freeze(self.path, source="triage", rows=[{"alpha_id": "A"}],
                                 inputs={"book": ["member-a"], "count": 1})
        other = self.path.with_name("other.json")
        same = selection.freeze(other, source="triage", rows=[{"alpha_id": "A"}],
                                inputs={"count": 1, "book": ["member-a"]})
        self.assertEqual(first["selection_id"], same["selection_id"])
        different = selection.freeze(self.path.with_name("different.json"), source="triage",
                                     rows=[{"alpha_id": "A"}], inputs={"count": 1, "book": ["member-b"]})
        self.assertNotEqual(first["selection_id"], different["selection_id"])
