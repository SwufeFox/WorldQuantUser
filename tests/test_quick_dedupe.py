"""Regression coverage for Quick simulation identity and saved-result reuse."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import simulation_slots
from tools.pipeline import batch_quick


ACCOUNT = {"fingerprint": "quick-dedupe-test-account"}
ARM = {"label": "valuation", "expr": "rank(cap)",
       "settings": {"region": "USA", "universe": "TOP3000", "delay": 1}}


def write_rows(path, *rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def completed(arm, **extra):
    return {"label": arm["label"], "expr": arm["expr"], "alpha_id": "alpha1",
            "status": "COMPLETE", **extra}


class QuickDedupeTests(TestCase):
    def setUp(self):
        root = TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.out = Path(root.name) / "new.jsonl"
        env = mock.patch.dict("os.environ", {"BRAIN_RESEARCH_DIR": root.name})
        env.start()
        self.addCleanup(env.stop)

    def select(self, *arms):
        runner = batch_quick.Runner(self.out, ACCOUNT, 2, 1)
        return batch_quick._select_arms(arms, runner, batch_quick.load_done(self.out, ACCOUNT))

    def test_new_output_reuses_exact_sibling_completion(self):
        payload = batch_quick._payload(ARM)
        key = batch_quick._simulation_key(payload)
        write_rows(self.out.with_name("previous.jsonl"), completed(
            ARM, simulation_key=key, account_fingerprint=ACCOUNT["fingerprint"]))
        rerendered = {**ARM, "label": "same-mechanism", "expr": "((rank( cap )))"}
        full = {**ARM, "settings": {**ARM["settings"], "simulationMode": "FULL"}}
        self.assertFalse(self.out.exists())
        self.assertEqual(self.select(rerendered, full), [full])

    def test_partial_legacy_rows_only_skip_their_original_label_and_scope(self):
        old = completed(ARM, region="USA", universe="TOP3000")
        write_rows(self.out.with_name("other.jsonl"), old)
        self.assertEqual(self.select(ARM), [ARM])
        write_rows(self.out, old)
        self.assertEqual(self.select(ARM), [])
        alias = {**ARM, "label": "same-source-new-label"}
        self.assertEqual(self.select(ARM, alias), [alias])
        different_expression = {**ARM, "expr": "rank(volume)"}
        different_region = {**ARM, "settings": {**ARM["settings"], "region": "EUR"}}
        full = {**ARM, "settings": {**ARM["settings"], "simulationMode": "FULL"}}
        for changed in (different_expression, different_region, full):
            with self.subTest(arm=changed):
                self.assertEqual(self.select(changed), [changed])
        write_rows(self.out, {**old, "delay": 0})
        self.assertEqual(self.select(ARM), [ARM])

    def test_same_batch_dedupes_only_new_equal_payloads(self):
        alias = {**ARM, "label": "size", "expr": "rank( cap )"}
        slow = {**ARM, "settings": {**ARM["settings"], "decay": 14}}
        fast = {**ARM, "settings": {**ARM["settings"], "decay": 3}}
        self.assertEqual(self.select(ARM, alias, slow, fast, ARM), [ARM, slow, fast])

    def test_each_already_accepted_alias_remains_recoverable(self):
        alias = {**ARM, "label": "size"}
        runner = batch_quick.Runner(self.out, ACCOUNT, 2, 1)
        first_key = runner._key(ARM["label"], batch_quick._payload(ARM))
        alias_key = runner._key(alias["label"], batch_quick._payload(alias))
        simulation_slots.quick_acquire(ACCOUNT, 2, first_key, existing_url="/simulations/first")
        simulation_slots.quick_acquire(ACCOUNT, 2, alias_key, existing_url="/simulations/alias")
        payload_key = batch_quick._simulation_key(batch_quick._payload(ARM))
        write_rows(self.out.with_name("other.jsonl"), completed(
            ARM, simulation_key=payload_key, account_fingerprint=ACCOUNT["fingerprint"]))
        self.assertEqual(self.select(ARM, alias), [ARM, alias])
        write_rows(self.out, completed(ARM, simulation_key=payload_key,
                                      account_fingerprint=ACCOUNT["fingerprint"],
                                      simulation_job_key=first_key))
        self.assertEqual(self.select(ARM, alias), [alias])

    def test_label_only_legacy_location_needs_an_unambiguous_payload(self):
        write_rows(self.out.with_suffix(".jsonl.loc"),
                   {"label": ARM["label"], "loc": "/simulations/legacy"})
        rerendered = {**ARM, "expr": "rank( cap )"}
        self.assertEqual(self.select(ARM, rerendered), [ARM])
        full = {**ARM, "settings": {**ARM["settings"], "simulationMode": "FULL"}}
        with self.assertRaisesRegex(ValueError, "multiple possible payloads"):
            self.select(ARM, full)

    def test_reads_arm_arrays_and_json_lines(self):
        arms = [ARM, {**ARM, "label": "alias"}]
        path = self.out.with_suffix(".arms")
        path.write_text(json.dumps(arms, indent=2), encoding="utf-8")
        self.assertEqual(batch_quick._load_arms(path), arms)
        write_rows(path, *arms)
        self.assertEqual(batch_quick._load_arms(path), arms)
        write_rows(path, ARM)
        self.assertEqual(batch_quick._load_arms(path), [ARM])

    def test_cross_ledger_completions_require_same_account(self):
        payload = batch_quick._payload(ARM)
        key = batch_quick._simulation_key(payload)
        sibling = self.out.with_name("previous.jsonl")
        write_rows(sibling, completed(ARM, simulation_key=key, account_fingerprint="different-account"))
        self.assertEqual(self.select(ARM), [ARM])
        write_rows(sibling, completed(ARM, simulation_key=key))
        self.assertEqual(self.select(ARM), [ARM])

        previous = batch_quick.Runner(sibling, ACCOUNT, 2, 1)
        old_key = previous._raw_key(ARM["label"], payload)
        previous._checkpoint(old_key, completed(ARM))
        write_rows(sibling, completed(ARM, simulation_key=key, simulation_job_key=old_key))
        self.assertEqual(self.select(ARM), [])

    def test_terminal_failure_does_not_use_next_run_limit(self):
        runner = batch_quick.Runner(self.out, ACCOUNT, 2, 1)
        key = runner._key(ARM["label"], batch_quick._payload(ARM))
        runner._checkpoint(key, {**ARM, "simulation_status": "FAIL", "alpha_id": None})
        write_rows(self.out, {"label": ARM["label"], "status": "FAIL", "simulation_job_key": key,
                              "account_fingerprint": ACCOUNT["fingerprint"]})
        next_arm = {**ARM, "label": "new", "expr": "rank(close)"}
        self.assertEqual(self.select(ARM, next_arm)[:1], [next_arm])

    def test_pending_reconciliation_does_not_starve_fresh_work(self):
        runner = batch_quick.Runner(self.out, ACCOUNT, 2, 1)
        key = runner._key(ARM["label"], batch_quick._payload(ARM))
        simulation_slots.quick_acquire(ACCOUNT, 2, key)
        write_rows(self.out, {"label": ARM["label"], "status": "UNKNOWN_SUBMISSION",
                              "simulation_job_key": key, "account_fingerprint": ACCOUNT["fingerprint"]})
        next_arm = {**ARM, "label": "new", "expr": "rank(close)"}
        self.assertEqual(self.select(ARM, next_arm), [next_arm, ARM])

    def test_existing_result_from_legacy_location_requires_metadata_proof(self):
        runner = batch_quick.Runner(self.out, ACCOUNT, 2, 1)
        payload = batch_quick._payload(ARM)
        key = runner._raw_key(ARM["label"], payload)
        location = "/simulations/unverified"
        write_rows(self.out.with_suffix(".jsonl.loc"), {"label": ARM["label"], "loc": location})
        old = completed(ARM, progress_url=location, simulation_key=batch_quick._simulation_key(payload),
                        simulation_job_key=key, account_fingerprint=ACCOUNT["fingerprint"])
        runner._checkpoint(key, old)
        write_rows(self.out, old)
        self.assertEqual(self.select(ARM), [ARM])
        write_rows(self.out, old, {**old, "identity_verified": True})
        self.assertEqual(self.select(ARM), [])

    def test_reformats_resume_legacy_job_with_proven_request_settings(self):
        runner = batch_quick.Runner(self.out, ACCOUNT, 2, 1)
        key = runner._raw_key(ARM["label"], batch_quick._payload(ARM))
        runner._checkpoint(key, {**ARM, "progress_url": "/simulations/already-accepted"})
        edited = {**ARM, "expr": "rank(  cap  )"}
        runner = batch_quick.Runner(self.out, ACCOUNT, 2, 1)
        self.assertEqual(runner._job_key(edited["label"], batch_quick._payload(edited)), key)
        changed = {**edited, "settings": {**ARM["settings"], "decay": 9}}
        self.assertNotEqual(runner._job_key(changed["label"], batch_quick._payload(changed)), key)

    def test_non_finite_settings_are_rejected_in_both_input_formats(self):
        path = self.out.with_suffix(".arms")
        for value in ("NaN", "Infinity", "-Infinity", "1e999"):
            arm = json.dumps(ARM).replace('"delay": 1', '"delay": ' + value)
            for source in (f"[{arm}]", f"{arm}\n{arm}\n"):
                path.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    batch_quick._load_arms(path)
