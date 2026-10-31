#!/usr/bin/env python3
"""Tests for backup/verify/restore and for replaying the ledger.

The load-bearing properties are the boring ones, and each is here because the
failure would be silent:

* an archive is verified **before** it is reported as written -- an unreadable
  backup that claims success stops you making a second one;
* verification fails on an *extra* member as well as a missing one, because an
  archive that quietly gained a file is as suspect as one that lost it;
* a point-in-time view filters before it deduplicates, so a key that was rewritten
  later keeps the value it had at the cutoff;
* replay diffs fields it can compare and reports the ones it cannot, instead of
  treating a missing metric as equal.
"""

from __future__ import annotations

import json
import os
import sys
import tarfile
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import backup_replay as br  # noqa: E402


def write(root, relative, text):
    path = os.path.join(root, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)
    return path


def ledger_row(key, verdict="rejected", status="COMPLETE", stamp="2026-10-20T00:00:00+00:00",
               expression="rank(close)", **metrics):
    return {"key": key, "alpha_id": key, "expression": expression,
            "verdict": verdict, "status": status, "recorded_at": stamp,
            "metrics": metrics}


class Collection(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        write(self.root, "alpha-registry.jsonl", "")
        write(self.root, "field-units.json", "{}")
        write(self.root, "notes.md", "x")
        write(self.root, "cache/metadata/a.json", "{}")
        write(self.root, "results/r1.json", "{}")
        write(self.root, "pkg/__pycache__/x.pyc", "junk")
        write(self.root, "alpha-registry.jsonl.lock", "lock")

    def test_ledger_profile_takes_only_the_ledger(self):
        files = br.collect(self.root, "ledger")
        self.assertEqual(files, ["alpha-registry.jsonl", "field-units.json"])

    def test_state_profile_drops_the_cache_but_keeps_results(self):
        files = br.collect(self.root, "state")
        self.assertIn("results/r1.json", files)
        self.assertNotIn("cache/metadata/a.json", files)
        self.assertNotIn("pkg/__pycache__/x.pyc", files)
        self.assertNotIn("alpha-registry.jsonl.lock", files)

    def test_full_profile_keeps_the_cache(self):
        self.assertIn("cache/metadata/a.json", br.collect(self.root, "full"))

    def test_a_session_cookie_is_never_archived(self):
        # `cache/session.json` is a live BRAIN cookie; a backup travels.
        write(self.root, "cache/session.json", '{"cookie": "secret"}')
        for profile in ("ledger", "state", "full"):
            self.assertNotIn("cache/session.json", br.collect(self.root, profile))
        manifest = br.build_manifest(self.root, br.collect(self.root, "full"), "full")
        self.assertEqual(manifest["excluded_secrets"], ["cache/session.json"])

    def test_an_unknown_profile_is_an_error_not_a_default(self):
        with self.assertRaises(ValueError):
            br.collect(self.root, "everything")


class Manifest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        rows = [ledger_row("a"), ledger_row("b", verdict="candidate", status="ACTIVE"),
                ledger_row("b", verdict="candidate", status="ACTIVE")]
        write(self.root, "alpha-registry.jsonl",
              "\n".join(json.dumps(row) for row in rows) + "\n")
        write(self.root, "alpha-registry.jsonl".replace(".jsonl", ".jsonl.broken"),
              "not json\n")

    def test_manifest_counts_rows_and_keys_separately(self):
        files = br.collect(self.root, "ledger")
        manifest = br.build_manifest(self.root, files, "ledger")
        ledger = manifest["ledger"]
        self.assertEqual(ledger["rows"], 3)
        self.assertEqual(ledger["keys"], 2)  # the duplicate key counts once
        self.assertEqual(ledger["by_status"]["ACTIVE"], 2)
        # Row level and key level differ: the key `b` was filed twice.
        self.assertEqual(ledger["book"], 1)

    def test_a_damaged_line_is_counted_not_raised(self):
        path = os.path.join(self.root, "alpha-registry.jsonl")
        with open(path, "a") as handle:
            handle.write("{not json\n")
        stats = br.ledger_stats(path)
        self.assertEqual(stats["damaged"], 1)

    def test_every_file_carries_a_hash_and_a_size(self):
        files = br.collect(self.root, "ledger")
        manifest = br.build_manifest(self.root, files, "ledger")
        for item in manifest["files"]:
            self.assertEqual(len(item["sha256"]), 64)
            self.assertGreaterEqual(item["bytes"], 0)


class Archive(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        write(self.root, "alpha-registry.jsonl", json.dumps(ledger_row("a")) + "\n")
        write(self.root, "results/r1.json", '{"x": 1}')
        self.files = br.collect(self.root, "state")
        self.manifest = br.build_manifest(self.root, self.files, "state")
        self.archive = os.path.join(tempfile.mkdtemp(), "snap.tar.gz")
        br.write_archive(self.root, self.files, self.manifest, self.archive)

    def test_a_fresh_archive_verifies(self):
        self.assertEqual(br.verify_archive(self.archive, self.manifest), [])

    def test_a_tampered_member_is_caught(self):
        broken = json.loads(json.dumps(self.manifest))
        broken["files"][0]["sha256"] = "0" * 64
        problems = br.verify_archive(self.archive, broken)
        self.assertTrue(any("hashes differently" in problem for problem in problems))

    def test_the_member_checks_are_about_content_not_tar_metadata(self):
        # A second archive of the same files has a different container hash but must
        # pass every member check: the manifest describes content.
        again = os.path.join(tempfile.mkdtemp(), "again.tar.gz")
        second = br.build_manifest(self.root, self.files, "state")
        br.write_archive(self.root, self.files, second, again)
        self.assertEqual(br.verify_archive(again, second), [])
        self.assertEqual(
            [(item["path"], item["sha256"]) for item in second["files"]],
            [(item["path"], item["sha256"]) for item in self.manifest["files"]])


class Restore(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        write(self.root, "alpha-registry.jsonl", json.dumps(ledger_row("a")) + "\n")
        self.files = br.collect(self.root, "state")
        manifest = br.build_manifest(self.root, self.files, "state")
        self.archive = os.path.join(tempfile.mkdtemp(), "snap.tar.gz")
        br.write_archive(self.root, self.files, manifest, self.archive)
        self.target = tempfile.mkdtemp()

    def test_restore_writes_the_files(self):
        outcome = br.restore(self.archive, self.target)
        self.assertTrue(outcome["ok"])
        self.assertTrue(os.path.exists(os.path.join(self.target, "alpha-registry.jsonl")))

    def test_restore_refuses_to_overwrite_without_force(self):
        existing = os.path.join(self.target, "alpha-registry.jsonl")
        write(self.target, "alpha-registry.jsonl", "newer")
        outcome = br.restore(self.archive, self.target)
        self.assertIn("alpha-registry.jsonl", outcome["skipped"])
        with open(existing) as handle:
            self.assertEqual(handle.read(), "newer")
        forced = br.restore(self.archive, self.target, force=True)
        self.assertIn("alpha-registry.jsonl", forced["written"])

    def test_a_path_that_escapes_the_target_is_refused(self):
        import hashlib
        import io

        payload = b"boom"
        evil = os.path.join(tempfile.mkdtemp(), "evil.tar.gz")
        with tarfile.open(evil, "w:gz") as handle:
            info = tarfile.TarInfo("../escaped.txt")
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
        size, digest = br.fingerprint(evil)
        manifest = {"version": 1, "archive": {"sha256": digest},
                    "files": [{"path": "../escaped.txt", "bytes": len(payload),
                               "sha256": hashlib.sha256(payload).hexdigest()}]}
        outcome = br.restore(evil, self.target, manifest=manifest)
        self.assertEqual(outcome["refused"], ["../escaped.txt"])
        self.assertFalse(os.path.exists(os.path.join(self.target, "..", "escaped.txt")))

    def test_dry_run_changes_nothing(self):
        outcome = br.restore(self.archive, self.target, dry_run=True)
        self.assertTrue(outcome["written"])
        self.assertFalse(os.path.exists(os.path.join(self.target, "alpha-registry.jsonl")))


class Prune(unittest.TestCase):
    def test_only_the_newest_n_of_a_profile_survive(self):
        directory = tempfile.mkdtemp()
        names = []
        for day in ("01", "02", "03"):
            name = os.path.join(
                directory, "research-ledger-x-202610%sT000000Z.tar.gz" % day)
            write(directory, os.path.basename(name), "x")
            write(directory, os.path.basename(name) + ".manifest.json", "{}")
            names.append(name)
        removed = br.prune(directory, "ledger", "x", keep=2)
        self.assertEqual(len(removed), 2)  # the archive and its sidecar
        self.assertFalse(os.path.exists(names[0]))
        self.assertFalse(os.path.exists(names[0] + ".manifest.json"))
        self.assertTrue(os.path.exists(names[2]))

    def test_another_profile_is_left_alone(self):
        directory = tempfile.mkdtemp()
        write(directory, "research-state-x-20261001T000000Z.tar.gz", "x")
        write(directory, "research-ledger-x-20261001T000000Z.tar.gz", "x")
        write(directory, "research-ledger-x-20261002T000000Z.tar.gz", "x")
        br.prune(directory, "ledger", "x", keep=1)
        self.assertTrue(os.path.exists(
            os.path.join(directory, "research-state-x-20261001T000000Z.tar.gz")))


class AsOf(unittest.TestCase):
    ROWS = [
        ledger_row("a", stamp="2026-10-20T00:00:00+00:00", fitness=1.0),
        ledger_row("b", stamp="2026-10-21T00:00:00+00:00", fitness=1.1),
        ledger_row("a", stamp="2026-10-25T00:00:00+00:00", fitness=9.9),
        ledger_row("c", stamp="2026-10-26T00:00:00+00:00", verdict="candidate"),
    ]

    def test_a_key_rewritten_later_keeps_its_earlier_value(self):
        rows = br.rows_asof(self.ROWS, "2026-10-22T00:00:00+00:00")
        by_key = {row["key"]: row for row in rows}
        self.assertEqual(sorted(by_key), ["a", "b"])
        self.assertEqual(by_key["a"]["metrics"]["fitness"], 1.0)

    def test_without_a_cutoff_it_is_the_current_view(self):
        rows = br.rows_asof(self.ROWS)
        self.assertEqual([row["key"] for row in rows], ["a", "b", "c"])
        self.assertEqual(rows[0]["metrics"]["fitness"], 9.9)

    def test_a_row_without_a_timestamp_is_excluded_when_a_cutoff_is_given(self):
        rows = br.rows_asof([{"key": "z", "expression": "rank(close)"}],
                            "2026-10-22T00:00:00+00:00")
        self.assertEqual(rows, [])

    def test_summary_reports_the_book(self):
        state = br.summarize([ledger_row("a", status="ACTIVE")])
        self.assertEqual(state["book"], 1)


class Compare(unittest.TestCase):
    def test_identical_metrics_are_identical(self):
        out = br.compare({"sharpe": 2.5, "turnover": 0.1},
                         {"sharpe": 2.5, "turnover": 0.1})
        self.assertTrue(out["identical"])
        self.assertEqual(out["differences"], {})

    def test_a_drift_is_named_with_both_values(self):
        out = br.compare({"sharpe": 2.5}, {"sharpe": 2.4})
        self.assertFalse(out["identical"])
        self.assertAlmostEqual(out["differences"]["sharpe"]["delta"], 0.1)
        self.assertEqual(out["differences"]["sharpe"]["recorded"], 2.5)

    def test_a_missing_metric_is_not_zero(self):
        out = br.compare({"sharpe": 2.5, "fitness": 1.4}, {"sharpe": 2.5})
        self.assertFalse(out["identical"])
        self.assertEqual(out["missing"], ["fitness"])

    def test_tolerance_is_respected(self):
        self.assertTrue(br.compare({"sharpe": 2.5}, {"sharpe": 2.5000001},
                                   tolerance=1e-3)["identical"])


class Replay(unittest.TestCase):
    ENTRIES = [
        {"alpha_id": "a", "key": "a", "expression": "rank(close)",
         "settings": {"universe": "TOP3000"}, "metrics": {"sharpe": 2.5, "turnover": 0.1}},
        {"alpha_id": "b", "key": "b", "expression": "rank(low)",
         "settings": {"universe": "TOP3000"}, "metrics": {"sharpe": 1.0}},
    ]

    def test_a_matching_replay_reports_no_drift(self):
        def fake(expressions, settings, **kwargs):
            return [{"alpha_id": "a", "status": "COMPLETE",
                     "metrics": {"sharpe": 2.5, "turnover": 0.1}}]
        results = br.replay(self.ENTRIES[:1], simulate=fake)
        self.assertTrue(results[0]["comparison"]["identical"])

    def test_drift_and_a_changed_id_are_both_reported(self):
        def fake(expressions, settings, **kwargs):
            return [{"alpha_id": "zzz", "status": "COMPLETE",
                     "metrics": {"sharpe": 2.4, "turnover": 0.1}}]
        results = br.replay(self.ENTRIES[:1], simulate=fake)
        self.assertTrue(results[0]["identity_changed"])
        self.assertEqual(results[0]["fresh_alpha_id"], "zzz")
        self.assertFalse(results[0]["comparison"]["identical"])

    def test_one_failure_does_not_stop_the_batch(self):
        calls = {"n": 0}

        def fake(expressions, settings, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("captcha")
            return [{"alpha_id": "b", "status": "COMPLETE", "metrics": {"sharpe": 1.0}}]
        results = br.replay(self.ENTRIES, simulate=fake)
        self.assertIn("captcha", results[0]["error"])
        self.assertTrue(results[1]["comparison"]["identical"])


class Selection(unittest.TestCase):
    ENTRIES = [
        {"alpha_id": "a", "status": "ACTIVE", "verdict": "candidate"},
        {"alpha_id": "b", "status": "ACTIVE", "verdict": "rejected"},
        {"alpha_id": "c", "status": "COMPLETE", "verdict": "candidate"},
    ]

    def test_ids_select_exactly(self):
        self.assertEqual([e["alpha_id"] for e in br.select(self.ENTRIES, ids="a,c")],
                         ["a", "c"])

    def test_status_and_verdict_combine(self):
        chosen = br.select(self.ENTRIES, status="ACTIVE", verdict="candidate")
        self.assertEqual([e["alpha_id"] for e in chosen], ["a"])

    def test_limit_truncates_after_filtering(self):
        chosen = br.select(self.ENTRIES, verdict="candidate", limit=1)
        self.assertEqual(len(chosen), 1)


if __name__ == "__main__":
    unittest.main()