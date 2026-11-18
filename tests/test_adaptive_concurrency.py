"""Offline concurrency policy, durable coordination, and interruption contracts."""

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import requests

import cache_store
import machine_lib as ml
import simulation_capabilities
import simulation_slots


ACCOUNT = {"state": "known", "fingerprint": "offline-test-account", "sources": ["user_id"]}
UNKNOWN_ACCOUNT = {"state": "unknown", "fingerprint": None, "sources": []}


def evidence(limit=None, account=None):
    return {
        "state": "verified", "account": account or ACCOUNT,
        "capability_id": "offline-test-evidence",
        "concurrency": {"state": "verified" if limit is not None else "unknown",
                        "limit": limit, "source": "test-authoritative" if limit is not None else None},
    }


class AdaptiveConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.env = mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": self.root.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.identity = mock.patch.object(simulation_capabilities, "account_provenance", return_value=ACCOUNT)
        self.identity.start()
        self.addCleanup(self.identity.stop)

    def test_confirmed_session_identity_survives_cookie_cache(self):
        response = mock.Mock(status_code=201)
        response.json.return_value = {"user": {"id": "offline-user"}, "permissions": ["TEST"]}
        with mock.patch.object(ml, "get_credentials", return_value=(" Offline@Example.COM ", "secret")), \
                mock.patch.object(requests.Session, "post", return_value=response):
            session, info = ml.authenticate()
            payload = ml._session_payload(session, info)
            restored = ml._session_from_entry(payload, session.auth)
        self.assertEqual(session._brain_account_id, "offline-user")
        self.assertEqual(restored._brain_account_id, "offline-user")
        self.assertEqual(payload["username"], "offline@example.com")
        self.assertNotIn("password", payload)

    def test_published_capacity_replaces_unknown_local_default(self):
        plan = ml.concurrency_plan(object(), capability_snapshot=evidence(9))
        self.assertEqual(plan["effective"], 9)
        self.assertEqual(plan["source"], "verified_platform_limit")

    def simulate(self, alphas=None, **kwargs):
        kwargs.setdefault("capability_snapshot", evidence())
        return ml.simulate_alphas(object(), alphas or ["rank(cap)"], **kwargs)

    def test_local_default_is_not_platform_allowance(self):
        plan = ml.concurrency_plan(object(), capability_snapshot=evidence())
        self.assertEqual(plan["effective"], ml.DEFAULT_LOCAL_CONCURRENCY)
        self.assertEqual(plan["source"], "conservative_local_default")
        self.assertIsNone(plan["platform_limit"])
        self.assertEqual(ml.concurrency_plan(object(), 19, evidence())["effective"], 19)

    def test_verified_account_limit_clamps_in_both_directions(self):
        for requested, limit, effective in [(8, 6, 6), (8, 2, 2), (2, 9, 2), (8, 0, 0)]:
            with self.subTest(requested=requested, limit=limit):
                plan = ml.concurrency_plan(object(), requested, evidence(limit))
                self.assertEqual(plan["effective"], effective)
                self.assertEqual(plan["platform_limit"], limit)
        other = {**ACCOUNT, "fingerprint": "different-account"}
        self.assertEqual(ml.concurrency_plan(object(), 8, evidence(1, other))["effective"], 8)
        stale = evidence(1)
        stale["state"] = "unknown"
        self.assertEqual(ml.concurrency_plan(object(), 8, stale)["effective"], 8)

    def test_invalid_budgets_fail(self):
        for value in [0, -1, True, 2.5, "8"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                ml.concurrency_plan(None, value, evidence())

    def test_plan_uses_cache_without_discovery(self):
        with mock.patch.object(simulation_capabilities, "cached", return_value=evidence(7)) as cached, \
                mock.patch.object(simulation_capabilities, "fetch", side_effect=AssertionError("online discovery")):
            self.assertEqual(ml.concurrency_plan(object(), 12)["effective"], 7)
        cached.assert_called_once_with()

    def test_structural_future_settings_and_null_extensions(self):
        resolved = ml.resolve_settings({"region": "FUTURE_REGION", "instrumentType": "FUTURES",
            "universe": "FUTURE_UNIVERSE", "delay": 4, "neutralization": "FUTURE_FACTOR",
            "pasteurization": "FUTURE_MODE", "nanHandling": "FUTURE_MODE", "extension": None})
        self.assertEqual(resolved["delay"], 4)
        self.assertIn("extension", resolved)
        self.assertIsNone(resolved["extension"])
        for settings in [{"delay": True}, {"delay": -1}, {"delay": 1.2},
                         {"region": "bad token"}, {"neutralization": []}, {"universe": "X\n"}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ml.resolve_settings(settings)

    def test_dry_run_has_roles_and_no_io(self):
        report = {}
        with mock.patch.object(simulation_capabilities, "account_provenance", side_effect=AssertionError("identity I/O")), \
                mock.patch.object(simulation_capabilities, "cached", side_effect=AssertionError("cache I/O")), \
                mock.patch.object(cache_store, "load_json", side_effect=AssertionError("file I/O")), \
                mock.patch.object(ml, "submit_single_simulation", side_effect=AssertionError("POST")):
            records = ml.simulate_alphas(None, [{"expression": "rank(cap)", "role": "diagnostic"}],
                                         dry_run=True, concurrency=8, scheduling_report=report)
        self.assertEqual(records[0]["role"], "diagnostic")
        self.assertEqual(report["effective"], 8)
        self.assertEqual(report["shared_slots"]["state"], "dry_run")
        with self.assertRaises(ValueError):
            ml.normalize_alphas([{"expression": "rank(cap)", "role": "other"}], ml.DEFAULT_SETTINGS)

    def test_large_wave_and_ordering_with_terminal_release(self):
        expressions = ["rank(field_%d)" % index for index in range(7)]
        posts = []
        def submit(session, body, **kwargs):
            posts.append(body["regular"])
            return "https://api.worldquantbrain.com/simulations/%d" % len(posts)
        def poll(session, url, **kwargs):
            return {"status": "COMPLETE", "alpha": "offline-" + url.rsplit("/", 1)[-1]}
        with mock.patch.object(ml, "submit_single_simulation", side_effect=submit), \
                mock.patch.object(ml, "poll_simulation", side_effect=poll), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value={"is": {"sharpe": 1}}):
            records = self.simulate(expressions, concurrency=7)
        self.assertEqual(posts, expressions)
        self.assertEqual([record["expression"] for record in records], expressions)
        self.assertEqual([record["alpha_id"] for record in records], ["offline-%d" % i for i in range(1, 8)])
        self.assertTrue(all(record["slot_reservation_state"] == "released" for record in records))
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 7)["active"], 0)

    def test_verified_zero_capacity_does_not_post(self):
        with mock.patch.object(ml, "submit_single_simulation") as submit:
            records = self.simulate(concurrency=8, capability_snapshot=evidence(0))
        submit.assert_not_called()
        self.assertEqual(records[0]["status"], "QUEUED")

    def test_permission_contraction_blocks_new_jobs_but_recovers_accepted_jobs(self):
        snapshot = evidence()
        snapshot["settings_schema"] = {"decay": {"type": "integer", "max_value": 2}}
        accepted = {"expression": "rank(cap)", "decay": 9, "status": "PENDING",
                    "submission_attempted": True,
                    "progress_url": "https://api.worldquantbrain.com/simulations/accepted"}
        queued = {"expression": "rank(close)", "decay": 9, "status": "QUEUED",
                  "submission_attempted": False}
        with mock.patch.object(ml, "submit_single_simulation") as post, \
                mock.patch.object(ml, "poll_simulation", return_value={"status": "COMPLETE", "alpha": "recovered"}), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value={"is": {"sharpe": 1}}):
            rows = self.simulate([("rank(cap)", 9), ("rank(close)", 9)],
                                 initial_records=[accepted, queued], capability_snapshot=snapshot)
        post.assert_not_called()
        self.assertEqual(rows[0]["alpha_id"], "recovered")
        self.assertEqual(rows[1]["status"], "FAILED")
        self.assertFalse(rows[1]["submission_attempted"])
        self.assertEqual(rows[1]["settings_capabilities"]["unsupported"][0]["setting"], "decay")
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 3)["active"], 0)

    def test_simultaneous_callers_share_slots_without_network_lock(self):
        entered, release = threading.Event(), threading.Event()
        def submit(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError("other caller could not observe journal during network wait")
            return "https://api.worldquantbrain.com/simulations/offline"
        with mock.patch.object(ml, "submit_single_simulation", side_effect=submit) as post, \
                mock.patch.object(ml, "poll_simulation", return_value={"status": "PENDING"}), \
                ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.simulate, concurrency=1)
            try:
                self.assertTrue(entered.wait(5))
                second = pool.submit(self.simulate, concurrency=1)
                deferred = second.result(timeout=5)
                self.assertEqual(deferred[0]["status"], "QUEUED")
                self.assertFalse(deferred[0]["submission_attempted"])
            finally:
                release.set()
            first.result(timeout=5)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 1)

    def test_completed_wave_retries_queued_work_after_shared_slot_contention(self):
        external, _ = simulation_slots.reserve(ACCOUNT, 2)
        expressions = ["rank(field_%d)" % index for index in range(4)]
        posts = []

        def submit(session, body, **kwargs):
            posts.append(body["regular"])
            return "https://api.worldquantbrain.com/simulations/%d" % len(posts)

        def poll(session, url, **kwargs):
            return {"status": "COMPLETE", "alpha": "offline-" + url.rsplit("/", 1)[-1]}

        with mock.patch.object(ml, "submit_single_simulation", side_effect=submit), \
                mock.patch.object(ml, "poll_simulation", side_effect=poll), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value={"is": {}}):
            rows = self.simulate(expressions, concurrency=2)

        self.assertEqual(posts, expressions)
        self.assertEqual([row["status"] for row in rows], ["COMPLETE"] * len(expressions))
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 2)["active"], 1)
        simulation_slots.reconcile_record({"slot_reservation": external}, ACCOUNT, outcome="terminal")

    def test_atomic_reservations_across_contenders(self):
        barrier = threading.Barrier(8)
        def contend():
            barrier.wait(timeout=5)
            return simulation_slots.reserve(ACCOUNT, 5)[0]
        with ThreadPoolExecutor(max_workers=8) as pool:
            identities = list(pool.map(lambda _: contend(), range(8)))
        self.assertEqual(sum(identity is not None for identity in identities), 5)
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 5)["active"], 5)

    def test_interrupt_keeps_identity_and_unknown_attempt_never_reposts(self):
        saved = []
        def event(kind, payload):
            if kind == "submitting":
                saved.append(dict(payload))
        with mock.patch.object(ml, "submit_single_simulation", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.simulate(concurrency=1, on_event=event)
        self.assertTrue(saved[0]["slot_reservation"]["id"])
        with mock.patch.object(simulation_slots.time, "time", return_value=10**12):
            self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 1)
        with mock.patch.object(ml, "submit_single_simulation") as post:
            resumed = self.simulate(initial_records=saved, concurrency=1)
        post.assert_not_called()
        self.assertEqual(resumed[0]["status"], "UNKNOWN_SUBMISSION")

    def test_checkpoint_interrupt_releases_slot_before_any_post(self):
        saved = []

        def checkpoint(kind, data):
            if kind == "submitting":
                saved.append(dict(data))
                raise KeyboardInterrupt

        with mock.patch.object(ml, "submit_single_simulation") as post:
            with self.assertRaises(KeyboardInterrupt):
                self.simulate(concurrency=1, on_event=checkpoint)
        post.assert_not_called()
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 0)

        with mock.patch.object(ml, "submit_single_simulation", return_value="https://api.worldquantbrain.com/simulations/offline") as post, \
                mock.patch.object(ml, "poll_simulation", return_value={"status": "PENDING"}):
            resumed = self.simulate(concurrency=1, initial_records=saved)
        post.assert_called_once()
        self.assertEqual(resumed[0]["status"], "PENDING")
        self.assertNotEqual(resumed[0]["slot_reservation"], saved[0]["slot_reservation"])

    def test_abandoned_preparation_is_reclaimed_without_discarding_unknown_posts(self):
        identity, _ = simulation_slots.reserve(ACCOUNT, 2)
        sent, _ = simulation_slots.reserve(ACCOUNT, 2)
        path, lock = simulation_slots._paths(ACCOUNT["fingerprint"])
        journal = simulation_slots._load(path, ACCOUNT["fingerprint"])
        journal["reservations"][identity["id"]]["state"] = "preparing"
        cache_store.write_json_atomic(path, journal)
        saved = {"slot_reservation": identity, "status": "UNKNOWN_SUBMISSION", "submission_attempted": True}
        with cache_store.file_lock(lock + ".prepare"):
            self.assertFalse(simulation_slots.restore_record(saved, ACCOUNT))
        self.assertEqual(saved["status"], "UNKNOWN_SUBMISSION")

        with mock.patch.object(ml, "submit_single_simulation", return_value="https://api.worldquantbrain.com/simulations/offline") as post, \
                mock.patch.object(ml, "poll_simulation", return_value={"status": "PENDING"}):
            rows = self.simulate(concurrency=2)
        post.assert_called_once()
        self.assertEqual(rows[0]["status"], "PENDING")
        self.assertTrue(simulation_slots.restore_record(saved, ACCOUNT))
        self.assertEqual(saved["status"], "QUEUED")
        self.assertFalse(saved["submission_attempted"])
        journal = simulation_slots._load(path, ACCOUNT["fingerprint"])
        self.assertNotIn(identity["id"], journal["reservations"])
        self.assertIn(sent["id"], journal["reservations"])

    def test_recovery_reuses_progress_without_repost_and_preserves_role(self):
        identity, _ = simulation_slots.reserve(ACCOUNT, 1)
        saved = {"expression": "rank(cap)", "decay": 0, "status": "UNKNOWN_SUBMISSION",
                 "submission_attempted": True, "slot_reservation": identity, "role": "diagnostic"}
        progress_url = "https://api.worldquantbrain.com/simulations/recovered"
        simulation_slots.reconcile_record(dict(saved), ACCOUNT, outcome="accepted",
                                          progress_url=progress_url)
        with self.assertRaisesRegex(ValueError, "another or unidentified account"):
            simulation_slots.restore_record(dict(saved), UNKNOWN_ACCOUNT)
        merged = simulation_slots.reconcile_record(dict(saved), ACCOUNT)
        self.assertEqual(merged["progress_url"], progress_url)
        self.assertEqual(merged["slot_reservation_state"], "accepted")
        with mock.patch.object(ml, "submit_single_simulation") as post, \
                mock.patch.object(ml, "poll_simulation", return_value={
                    "status": "COMPLETE", "alpha": "offline-result"}) as poll, \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value={"is": {}}):
            records = self.simulate(initial_records=[saved], concurrency=4)
        post.assert_not_called()
        self.assertEqual(poll.call_args.args[1], progress_url)
        self.assertEqual(records[0]["role"], "diagnostic")
        self.assertEqual(records[0]["alpha_id"], "offline-result")
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 0)

    def test_definite_permission_refusal_releases_reservation(self):
        events = []
        with mock.patch.object(ml, "submit_single_simulation", side_effect=ml.AuthError("forbidden")):
            with self.assertRaises(ml.AuthError):
                self.simulate(concurrency=1, on_event=lambda kind, data: events.append((kind, dict(data))))
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 0)
        self.assertEqual(events[-1][0], "deferred")
        self.assertFalse(events[-1][1]["submission_attempted"])

    def test_rate_limit_refusal_releases_but_transport_uncertainty_retains(self):
        for error, active in [(ml.RateLimited("refused"), 0), (requests.Timeout("uncertain"), 1)]:
            with self.subTest(error=error), mock.patch.object(ml, "submit_single_simulation", side_effect=error):
                records = self.simulate(concurrency=1)
            self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], active)
            self.assertEqual(records[0]["status"], "QUEUED" if active == 0 else "UNKNOWN_SUBMISSION")

    def test_nonterminal_poll_retains_accepted_reservation(self):
        with mock.patch.object(ml, "submit_single_simulation", return_value="https://api.worldquantbrain.com/simulations/offline"), \
                mock.patch.object(ml, "poll_simulation", return_value={"status": "RUNNING"}):
            records = self.simulate(concurrency=1)
        self.assertEqual(records[0]["status"], "PENDING")
        self.assertIn("progress_url", records[0])
        self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 1)

    def test_corrupt_journal_fails_closed_before_post(self):
        path, _ = simulation_slots._paths(ACCOUNT["fingerprint"])
        cache_store.write_json_atomic(path, {"not": "a journal"})
        with mock.patch.object(ml, "submit_single_simulation") as post:
            with self.assertRaises(cache_store.CacheError):
                self.simulate(concurrency=8)
        post.assert_not_called()

    def test_unknown_account_and_cross_workspace_boundaries_are_explicit(self):
        identity, state = simulation_slots.reserve(UNKNOWN_ACCOUNT, 8)
        self.assertIsNone(identity)
        self.assertEqual(state["scope"], "invocation_only")
        self.assertFalse(state["cross_workspace_coordinated"])
        identity, _ = simulation_slots.reserve(ACCOUNT, 1)
        with tempfile.TemporaryDirectory() as other, mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": other}):
            self.assertEqual(simulation_slots.snapshot(ACCOUNT, 1)["active"], 0)
            with self.assertRaisesRegex(ValueError, "another research workspace"):
                simulation_slots.reconcile_record({"slot_reservation": identity}, ACCOUNT, outcome="terminal")


if __name__ == "__main__":
    unittest.main()
