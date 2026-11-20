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
        return batch_quick._select_arms(arms, runner, batch_quick.load_done(self.out))

    def test_new_output_reuses_exact_sibling_completion(self):
        payload = batch_quick._payload(ARM)
        key = batch_quick._simulation_key(payload)
        write_rows(self.out.with_name("previous.jsonl"), completed(ARM, simulation_key=key))
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
        write_rows(self.out.with_name("other.jsonl"), completed(ARM, simulation_key=payload_key))
        self.assertEqual(self.select(ARM, alias), [ARM, alias])
        write_rows(self.out, completed(ARM, simulation_key=payload_key,
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
