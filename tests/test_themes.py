#!/usr/bin/env python3
"""Tests for folding observed platform theme records into one catalog."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research_app import app, themes


def _write_snapshot(root, records):
    path = os.path.join(root, themes.SNAPSHOT_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _theme_record(observed_at, region, members, result="WARNING"):
    return {"kind": "themes", "name": "MATCHES_THEMES", "result": result,
            "observed_at": observed_at, "source": "platform_check",
            "scope": {"region": region, "universe": "TOP500", "delay": 1},
            "themes": members}


class CatalogTests(unittest.TestCase):
    def test_same_theme_id_folds_across_regions_and_keeps_latest_multiplier(self):
        with tempfile.TemporaryDirectory() as root:
            _write_snapshot(root, [
                _theme_record("2026-11-27T10:00:00+00:00", "TWN",
                              [{"id": "t1", "name": "liquidity", "multiplier": 1.5}]),
                _theme_record("2026-11-28T10:00:00+00:00", "HKG",
                              [{"id": "t1", "name": "liquidity", "multiplier": 2.0},
                               {"id": "t2", "name": "revision", "multiplier": None}]),
            ])
            data = themes.catalog(root)
        rows = {row["id"]: row for row in data["themes"]}
        self.assertEqual(data["scanned_records"], 2)
        self.assertEqual(set(rows), {"t1", "t2"})
        self.assertEqual(rows["t1"]["observations"], 2)
        self.assertEqual(rows["t1"]["regions"], ["HKG", "TWN"])
        self.assertEqual(rows["t1"]["multiplier"], 2.0)
        self.assertEqual(rows["t1"]["first_observed_at"], "2026-11-27T10:00:00+00:00")
        self.assertEqual(data["themes"][0]["id"], "t1")

    def test_malformed_and_foreign_records_are_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, themes.SNAPSHOT_FILENAME)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{not json\n")
                handle.write(json.dumps({"kind": "check", "name": "LOW_SHARPE"}) + "\n")
                handle.write(json.dumps(_theme_record(
                    "2026-11-28T10:00:00+00:00", "KOR",
                    [{"id": "t1", "name": "liquidity", "multiplier": 1.2}])) + "\n")
            data = themes.catalog(root)
        self.assertEqual(data["count"] if "count" in data else len(data["themes"]), 1)
        self.assertEqual(data["themes"][0]["regions"], ["KOR"])

    def test_cli_lists_themes_as_one_json_document(self):
        with tempfile.TemporaryDirectory() as root:
            _write_snapshot(root, [_theme_record(
                "2026-11-28T10:00:00+00:00", "TWN",
                [{"id": "t1", "name": "liquidity", "multiplier": 1.5}])])
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = app.main(["themes", "list", "--research-dir", root, "--compact", "--limit", "1"])
            payload = json.loads(buffer.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["count"], 1)
        self.assertEqual(payload["data"]["themes"][0]["id"], "t1")


if __name__ == "__main__":
    unittest.main()
