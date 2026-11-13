"""Offline tests for the cache layer: atomic writes, TTLs, locks, sessions.

Nothing here touches the network. The BRAIN session is stubbed, and every path
points into a temporary directory, so these run anywhere and fast.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cache_store  # noqa: E402
import machine_lib as ml  # noqa: E402
import research_paths  # noqa: E402


class TempCacheCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name

    def path(self, *parts):
        return os.path.join(self.root, *parts)


class AtomicWriteTests(TempCacheCase):
    def test_roundtrip_and_no_temp_files_left(self):
        target = self.path("nested", "entry.json")
        cache_store.write_json_atomic(target, {"a": 1})
        with open(target, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["a"], 1)
        leftovers = [n for n in os.listdir(os.path.dirname(target)) if n.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    @unittest.skipUnless(os.name == "posix", "POSIX mode bits are unavailable on Windows")
    def test_file_mode_defaults_to_private(self):
        target = self.path("secret.json")
        cache_store.write_json_atomic(target, {"ok": True})
        self.assertEqual(os.stat(target).st_mode & 0o777, 0o600)

    def test_overwrite_is_clean(self):
        target = self.path("e.json")
        cache_store.write_json_atomic(target, {"v": "x" * 5000})
        cache_store.write_json_atomic(target, {"v": "short"})
        with open(target, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), {"v": "short"})

    def test_load_json_reports_instead_of_raising(self):
        self.assertEqual(cache_store.load_json(self.path("absent.json"))[1], "missing")
        bad = self.path("bad.json")
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write("{\"truncated\": ")
        data, error = cache_store.load_json(bad)
        self.assertIsNone(data)
        self.assertIn("malformed", error)


class EnvelopeTests(TempCacheCase):
    def test_fresh_entry(self):
        target = self.path("e.json")
        cache_store.write_json_atomic(
            target, cache_store.build_entry("demo", {"x": 1}, ttl=60))
        entry = cache_store.read_entry(target)
        self.assertTrue(entry["present"])
        self.assertTrue(entry["fresh"])
        self.assertEqual(entry["payload"], {"x": 1})
        self.assertGreater(entry["expires_in"], 50)

    def test_expired_entry_is_present_but_not_fresh(self):
        target = self.path("e.json")
        cache_store.write_json_atomic(
            target, cache_store.build_entry("demo", {"x": 1}, ttl=-1))
        entry = cache_store.read_entry(target)
        self.assertTrue(entry["present"])
        self.assertFalse(entry["fresh"])
        self.assertLess(entry["expires_in"], 0)

    def test_entry_without_ttl_never_expires(self):
        target = self.path("e.json")
        cache_store.write_json_atomic(
            target, cache_store.build_entry("session", {"cookies": {}}))
        entry = cache_store.read_entry(target)
        self.assertTrue(entry["fresh"])
        self.assertIsNone(entry["expires_at"])

    def test_version_mismatch_degrades_to_miss(self):
        target = self.path("e.json")
        stale_shape = {"_cache": {"version": 0, "kind": "demo"}, "payload": {"x": 1}}
        cache_store.write_json_atomic(target, stale_shape)
        entry = cache_store.read_entry(target)
        self.assertFalse(entry["present"])
        self.assertIn("version mismatch", entry["error"])

    def test_non_envelope_file_is_reported(self):
        target = self.path("e.json")
        cache_store.write_json_atomic(target, {"just": "data"})
        self.assertIn("not a cache envelope", cache_store.read_entry(target)["error"])

    def test_quarantine_keeps_a_sample_and_prunes_old_ones(self):
        target = self.path("session.json")
        with open(target, "w") as handle:
            handle.write("broken")
        moved = cache_store.quarantine(target, keep=2)
        self.assertTrue(os.path.isfile(moved))
        self.assertFalse(os.path.exists(target))
        for _ in range(4):
            with open(target, "w") as handle:
                handle.write("broken again")
            cache_store.quarantine(target, keep=2)
        samples = [n for n in os.listdir(self.root) if ".corrupt-" in n]
        self.assertLessEqual(len(samples), 2)


def _hold_lock(path, ready, release):
    with cache_store.file_lock(path, timeout=10, poll=0.02):
        ready.set()
        release.wait()


def _take_lock(path, attempted, acquired):
    attempted.set()
    with cache_store.file_lock(path, timeout=10, poll=0.02):
        acquired.set()


def _update_shared_json(path, lock, start, worker_id, updates):
    start.wait(15)
    for index in range(updates):
        with cache_store.file_lock(lock, timeout=10, poll=0.02):
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["updates"].append([worker_id, index])
            cache_store.write_json_atomic(path, payload)


class LockTests(TempCacheCase):
    def setUp(self):
        super().setUp()
        # Spawn exercises independent descriptors on Windows and POSIX alike.
        self.processes = multiprocessing.get_context("spawn")

    def _stop_worker(self, worker):
        if worker.is_alive():
            worker.terminate()
        worker.join(10)
        if worker.is_alive():
            worker.kill()
            worker.join(10)
        self.assertFalse(worker.is_alive(), "child did not stop")

    def _start_worker(self, target, *args):
        worker = self.processes.Process(target=target, args=args)
        worker.start()
        self.addCleanup(self._stop_worker, worker)
        return worker

    def _start_holder(self, lock):
        ready = self.processes.Event()
        release = self.processes.Event()
        worker = self._start_worker(_hold_lock, lock, ready, release)
        self.assertTrue(ready.wait(15), "child never took the lock")
        return worker, release

    def test_second_holder_times_out(self):
        lock = self.path("locks", "brain-session.lock")
        worker, release = self._start_holder(lock)
        with self.assertRaises(cache_store.LockTimeout):
            with cache_store.file_lock(lock, timeout=0.2, poll=0.02):
                self.fail("entered an already held lock")
        release.set()
        worker.join(10)
        self.assertEqual(worker.exitcode, 0)

    def test_waiter_acquires_after_owner_releases(self):
        lock = self.path("l.lock")
        owner, release = self._start_holder(lock)
        attempted = self.processes.Event()
        acquired = self.processes.Event()
        waiter = self._start_worker(_take_lock, lock, attempted, acquired)
        self.assertTrue(attempted.wait(15), "waiter never attempted the lock")
        self.assertFalse(acquired.wait(0.2), "waiter entered while owner held the lock")
        release.set()
        self.assertTrue(acquired.wait(10), "waiter did not acquire the released lock")
        for worker in (owner, waiter):
            worker.join(10)
            self.assertEqual(worker.exitcode, 0)

    def test_lock_released_after_block(self):
        lock = self.path("l.lock")
        with cache_store.file_lock(lock, timeout=1):
            pass
        with cache_store.file_lock(lock, timeout=1):
            pass

    def test_lock_released_after_exception(self):
        lock = self.path("l.lock")
        with self.assertRaisesRegex(RuntimeError, "interrupted work"):
            with cache_store.file_lock(lock, timeout=1):
                raise RuntimeError("interrupted work")
        attempted = self.processes.Event()
        acquired = self.processes.Event()
        worker = self._start_worker(_take_lock, lock, attempted, acquired)
        self.assertTrue(acquired.wait(15), "exception left the lock held")
        worker.join(10)
        self.assertEqual(worker.exitcode, 0)

    def test_lock_reacquired_after_owner_is_killed(self):
        lock = self.path("l.lock")
        worker, _release = self._start_holder(lock)
        worker.kill()
        worker.join(10)
        self.assertIsNotNone(worker.exitcode, "killed owner did not exit")
        self.assertNotEqual(worker.exitcode, 0)
        with cache_store.file_lock(lock, timeout=1, poll=0.02):
            self.assertTrue(os.path.isfile(lock))

    def test_legacy_pid_sentinel_is_ignored_and_preserved(self):
        lock = self.path("l.lock")
        sentinel = lock + ".pid"
        with open(sentinel, "w", encoding="utf-8") as handle:
            handle.write("legacy owner")
        for _attempt in range(2):
            with cache_store.file_lock(lock, timeout=0, poll=0.02):
                with open(sentinel, encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "legacy owner")
        self.assertTrue(os.path.isfile(sentinel))

    def test_unsupported_lock_backend_fails_clearly(self):
        lock = self.path("l.lock")
        with mock.patch.object(cache_store, "fcntl", None), \
                mock.patch.object(cache_store, "msvcrt", None):
            with self.assertRaisesRegex(cache_store.CacheError, "file locking is unsupported"):
                with cache_store.file_lock(lock):
                    self.fail("entered without a supported lock backend")
        self.assertFalse(os.path.exists(lock))

    def test_processes_preserve_shared_json_updates(self):
        path = self.path("shared.json")
        lock = self.path("shared.lock")
        cache_store.write_json_atomic(path, {"updates": []})
        worker_count, updates = 4, 12
        start = self.processes.Barrier(worker_count + 1)
        workers = [
            self._start_worker(_update_shared_json, path, lock, start, worker_id, updates)
            for worker_id in range(worker_count)
        ]
        start.wait(15)
        for worker in workers:
            worker.join(20)
            self.assertEqual(worker.exitcode, 0, "concurrent JSON update failed")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertCountEqual(
            payload["updates"],
            [[worker_id, index] for worker_id in range(worker_count) for index in range(updates)],
        )


class FakeSession:
    """Minimal stand-in for requests.Session with a real cookie jar."""

    def __init__(self):
        import requests
        self.cookies = requests.cookies.RequestsCookieJar()

    def update(self, values):
        self.cookies.update(values)


class SessionCacheTests(TempCacheCase):
    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.cache_path = os.path.join(self._tmpdir.name, "cache", "session.json")
        self.calls = []

        def fake_authenticate():
            self.calls.append(time.time())
            session = FakeSession()
            session.cookies.set("t", "token-value")
            return session, {"user_id": "U1", "permissions": ["TUTORIAL"],
                                   "token_expiry": 3600}

        originals = {
            "authenticate": ml.authenticate,
            "session_cache_path": ml.session_cache_path,
            "legacy_session_cache_path": ml.legacy_session_cache_path,
            "lock_path": research_paths.lock_path,
            "get_credentials": ml.get_credentials,
        }
        ml.authenticate = fake_authenticate
        ml.session_cache_path = lambda: self.cache_path
        ml.legacy_session_cache_path = lambda: os.path.join(
            self._tmpdir.name, "legacy.json")
        research_paths.lock_path = lambda name: os.path.join(
            self._tmpdir.name, "locks", name + ".lock")
        ml.get_credentials = lambda: ("u", "p")

        def restore():
            ml.authenticate = originals["authenticate"]
            ml.session_cache_path = originals["session_cache_path"]
            ml.legacy_session_cache_path = originals["legacy_session_cache_path"]
            research_paths.lock_path = originals["lock_path"]
            ml.get_credentials = originals["get_credentials"]

        self.addCleanup(restore)

    def test_first_call_logs_in_then_reuses_cache(self):
        meta = {}
        ml.load_session(meta=meta)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(meta["cached"])
        self.assertTrue(os.path.isfile(self.cache_path))

        meta2 = {}
        ml.load_session(meta=meta2)
        self.assertEqual(len(self.calls), 1, "second call must not log in again")
        self.assertTrue(meta2["cached"])
        self.assertEqual(meta2["reason"], "reused-cache")
        self.assertEqual(meta2["cache_state"], "fresh")

    def test_refresh_forces_one_new_login(self):
        ml.load_session()
        ml.load_session(refresh=True)
        self.assertEqual(len(self.calls), 2)

    def test_expired_cache_triggers_login(self):
        cache_store.write_json_atomic(
            self.cache_path,
            cache_store.build_entry("brain-session",
                                    {"cookies": {"s": "old"}, "user_id": "U1"}, ttl=-5))
        meta = {}
        ml.load_session(meta=meta)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(meta["cache_state"], "expired")
        self.assertEqual(meta["reason"], "logged-in")

    def test_corrupt_cache_is_quarantined_then_replaced(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w") as handle:
            handle.write("{\"cookies\": ")
        meta = {}
        ml.load_session(meta=meta)
        self.assertEqual(meta["cache_state"], "corrupt")
        samples = [n for n in os.listdir(os.path.dirname(self.cache_path))
                   if ".corrupt-" in n]
        self.assertTrue(samples, "a corrupt cache must be kept for diagnosis")
        self.assertTrue(cache_store.read_entry(self.cache_path)["present"])

    def test_legacy_file_is_migrated_without_logging_in(self):
        legacy = ml.legacy_session_cache_path()
        os.makedirs(os.path.dirname(legacy), exist_ok=True)
        with open(legacy, "w") as handle:
            json.dump({"cookies": {"s": "legacy"}, "saved_at": time.time(),
                       "expiry_seconds": 3600, "user_id": "U1",
                       "permissions": ["TUTORIAL"]}, handle)
        meta = {}
        session = ml.load_session(meta=meta)
        self.assertEqual(len(self.calls), 0, "migration must not force a login")
        self.assertEqual(meta["migrated_from"], legacy)
        self.assertEqual(session.cookies.get("s"), "legacy")
        self.assertFalse(os.path.exists(legacy), "legacy file should be moved")
        self.assertTrue(os.path.isfile(self.cache_path))

    def test_concurrent_callers_log_in_once(self):
        errors = []

        def worker():
            try:
                ml.load_session()
            except Exception as exc:  # pragma: no cover - surfaced by assert below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)
        self.assertEqual(errors, [])
        self.assertEqual(len(self.calls), 1, "single-flight login was not honoured")

    def test_status_reports_expiry_and_owner(self):
        ml.load_session()
        status = ml.session_status(self.cache_path)
        self.assertTrue(status["fresh"])
        self.assertEqual(status["user_id"], "U1")
        self.assertEqual(status["permissions"], ["TUTORIAL"])
        self.assertGreater(status["expires_in_seconds"], 0)
        self.assertTrue(status["has_cookies"])

    def test_invalidate_removes_the_file(self):
        ml.load_session()
        self.assertTrue(ml.invalidate_session(self.cache_path))
        self.assertFalse(os.path.exists(self.cache_path))
        self.assertFalse(ml.invalidate_session(self.cache_path))


class MetadataCacheTests(TempCacheCase):
    def setUp(self):
        super().setUp()
        originals = {
            "metadata_root": research_paths.metadata_root,
            "policy": dict(ml.metadata_cache_policy()),
        }
        research_paths.metadata_root = lambda: os.path.join(self.root, "metadata")
        ml.configure_metadata_cache(max_age=3600, refresh=False, enabled=True)
        self.addCleanup(self._restore, originals)

    def _restore(self, originals):
        research_paths.metadata_root = originals["metadata_root"]
        ml.configure_metadata_cache(
            max_age=originals["policy"]["max_age"],
            refresh=originals["policy"]["refresh"],
            enabled=originals["policy"]["enabled"],
        )

    def test_hit_after_first_fetch(self):
        calls = []

        def fetch():
            calls.append(1)
            return [{"id": "f1"}]

        first = ml.cached_metadata("datafields", {"dataset": "pv1"}, fetch)
        second = ml.cached_metadata("datafields", {"dataset": "pv1"}, fetch)
        self.assertEqual(len(calls), 1)
        self.assertFalse(first["cache"]["hit"])
        self.assertTrue(second["cache"]["hit"])
        self.assertEqual(second["data"], [{"id": "f1"}])

    def test_different_params_do_not_collide(self):
        ml.cached_metadata("datafields", {"dataset": "pv1"}, lambda: ["a"])
        other = ml.cached_metadata("datafields", {"dataset": "pv13"}, lambda: ["b"])
        self.assertEqual(other["data"], ["b"])
        self.assertFalse(other["cache"]["hit"])

    def test_refresh_bypasses_a_fresh_entry(self):
        ml.cached_metadata("operators", {}, lambda: ["old"])
        fresh = ml.cached_metadata("operators", {}, lambda: ["new"], refresh=True)
        self.assertEqual(fresh["data"], ["new"])
        self.assertFalse(fresh["cache"]["hit"])

    def test_zero_ttl_disables_the_cache(self):
        calls = []
        ml.configure_metadata_cache(max_age=0)
        ml.cached_metadata("operators", {}, lambda: calls.append(1) or ["x"])
        ml.cached_metadata("operators", {}, lambda: calls.append(1) or ["x"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(ml.metadata_cache_policy()["enabled"])

    def test_stale_snapshot_is_served_when_refetch_fails(self):
        ml.configure_metadata_cache(max_age=1)
        ml.cached_metadata("operators", {}, lambda: ["cached"])
        time.sleep(1.1)

        def broken():
            raise ml.ServiceError("upstream is down")

        result = ml.cached_metadata("operators", {}, broken)
        self.assertEqual(result["data"], ["cached"])
        self.assertTrue(result["cache"]["stale"])
        self.assertTrue(result["cache"]["degraded"])

    def test_failure_without_a_snapshot_still_raises(self):
        def broken():
            raise ml.ServiceError("upstream is down")

        with self.assertRaises(ml.ServiceError):
            ml.cached_metadata("operators", {}, broken)

    def test_corrupt_snapshot_is_replaced(self):
        path = ml.metadata_cache_path("operators", {})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write("{not json")
        result = ml.cached_metadata("operators", {}, lambda: ["fresh"])
        self.assertEqual(result["data"], ["fresh"])
        self.assertTrue(cache_store.read_entry(path)["present"])

    def test_report_dict_is_filled_for_callers(self):
        report = {}
        ml.cached_metadata("operators", {}, lambda: ["x"], report=report)
        self.assertIn("hit", report)
        self.assertEqual(report["kind"], "operators")


class CacheMaintenanceTests(TempCacheCase):
    def test_stats_and_clear(self):
        root = os.path.join(self.root, "cache")
        cache_store.write_json_atomic(os.path.join(root, "a.json"), {"x": 1})
        cache_store.write_json_atomic(os.path.join(root, "metadata", "b.json"), {"y": 2})
        stats = cache_store.cache_stats(root)
        self.assertEqual(stats["files"], 2)
        self.assertIn("metadata", stats["by_kind"])

        removed = cache_store.clear_dir(root, keep_locks=True)
        self.assertEqual(removed, 2)
        self.assertEqual(cache_store.cache_stats(root)["files"], 0)


if __name__ == "__main__":
    unittest.main()