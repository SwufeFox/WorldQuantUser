"""Background workers depend on saved arm states, not the subprocess exit code."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
import json
import subprocess

from research_app import selection
from tools import _runner as runner
from tools import campaign_runner as campaign


def arm(status, **fields):
    return {"status": status, **fields}


def candidate(**overrides):
    return arm("COMPLETE", **({"alpha_id": "candidate", "label": "label",
                             "train_sharpe": 2.0, "train_fitness": 1.2,
                             "train_sharpe_h1": 1.5, "train_sharpe_h2": 2.0} | overrides))


class ResearchRunnerTests(TestCase):
    def setUp(self):
        root = TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = Path(root.name)
        self.inbox = self.root / "queue"
        self.inbox.mkdir()
        self.spec = self.inbox / "scope.json"
        self.spec.write_text("{}", encoding="utf-8")
        self.batch = self.inbox / "batches" / "scope-batch.json"
        self.receipt = self.root / "scope-result.json"
        self.default_batch = self.root / "scope-2026-11-17.json"
        self.identity = {"request_fingerprint": "fingerprint", "design": {"mode": "explore"}}
        self.preview = runner.CliResult(True, {"path": str(self.default_batch), **self.identity,
                                                  "results": [{"role": "candidate"}]})

    def saved(self, path, rows, *, fingerprint=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.identity, "arms": [{"role": "candidate"}], "rows": rows}
        if fingerprint:
            payload["request_fingerprint"] = fingerprint
        path.write_text(json.dumps(payload), encoding="utf-8")

    def run_probe(self):
        return runner.run_probe(self.root, self.spec, self.batch, self.receipt, max_wait=30)

    def test_root_is_forwarded_to_every_cli_subcommand(self):
        subprocess_result = subprocess.CompletedProcess([], 0, '{"ok":true,"data":{}}', "")
        with mock.patch.object(runner.subprocess, "run", return_value=subprocess_result) as run:
            self.assertTrue(runner.run_cli(self.root, "buffer", "list").ok)
        command = run.call_args.args[0]
        self.assertEqual(command[2:5], ["--research-dir", str(self.root), "--quiet"])

    def test_incomplete_success_envelope_is_deferred(self):
        queued = arm("QUEUED", submission_state="not_submitted")
        reply = runner.CliResult(True, {"rows": [queued]})
        with mock.patch.object(runner, "run_cli", side_effect=[self.preview, reply]) as run:
            result = self.run_probe()
        self.assertEqual(result.state, "deferred")
        self.assertIn("--batch-path", run.call_args.args)

    def test_failed_poll_with_progress_can_resume(self):
        pending = arm("FAILED", progress_url="https://test.invalid/1", submission_state="accepted")
        self.saved(self.batch, [pending])
        finished = runner.CliResult(True, {"rows": [arm("COMPLETE", alpha_id="alpha")]})
        with mock.patch.object(runner, "run_cli", side_effect=[self.preview, finished]) as run:
            result = self.run_probe()
        self.assertEqual(result.state, "done")
        self.assertEqual(run.call_args.args[2:4], ("--resume", self.batch))

    def test_unknown_post_uses_read_only_recovery_even_if_other_arms_are_queued(self):
        rows = [arm("UNKNOWN_SUBMISSION", submission_state="unknown_submission"),
                arm("QUEUED", submission_state="not_submitted")]
        self.saved(self.batch, rows)
        with mock.patch.object(runner, "run_cli",
                               side_effect=[self.preview, runner.CliResult(True, {"rows": rows})]) as run:
            result = self.run_probe()
        self.assertEqual(result.state, "deferred")
        self.assertEqual(run.call_args.args[2:4], ("--recover", self.batch))

    def test_completed_legacy_batch_is_bound_and_read_without_a_second_launch(self):
        legacy = self.root / "scope-2026-11-16.json"
        rows = [arm("WARNING", alpha_id="alpha")]
        self.saved(legacy, rows)
        finished = runner.CliResult(True, {"rows": rows})
        with mock.patch.object(runner, "run_cli", side_effect=[self.preview, finished]) as run:
            result = self.run_probe()
        self.assertEqual(result.state, "done")
        self.assertEqual(run.call_args.args[2:4], ("--recover", legacy))

    def test_changed_spec_cannot_resume_existing_batch(self):
        self.saved(self.batch, [arm("QUEUED", submission_state="not_submitted")],
                   fingerprint="old request")
        with mock.patch.object(runner, "run_cli", return_value=self.preview) as run:
            result = self.run_probe()
        self.assertEqual(result.state, "failed")
        self.assertIn("does not match", result.detail)
        run.assert_called_once()

    def test_queue_archives_only_terminal_outcomes(self):
        with mock.patch.object(runner, "run_probe", side_effect=[
                runner.ProbeRun("deferred", {}), runner.ProbeRun("failed", {})]):
            paths = lambda _spec: (self.batch, self.receipt)
            self.assertEqual(runner.run_queue(self.root, self.inbox, paths, mock.Mock(),
                                              max_wait=30, interval=1, once=True), 3)
            self.assertTrue(self.spec.exists())
            self.assertEqual(runner.run_queue(self.root, self.inbox, paths, mock.Mock(),
                                              max_wait=30, interval=1, once=True), 1)
        self.assertTrue((self.inbox / "failed" / self.spec.name).exists())

    def campaign_reply(self, *, freeze=True, **updates):
        training = {"alpha_id": "candidate", "delta": 0.01, "evidence": {"state": "measured"},
                    "training_gate": {"state": "pass", "reason": None},
                    "shelf_correlation": {"state": "measured", "eligible": 0, "measured": 0}}
        training.update(updates)

        def reply(_root, command, *args, **kwargs):
            if command == "triage":
                if not freeze:
                    return runner.CliResult(True, {"rows": [training]})
                path = args[args.index("--freeze") + 1]
                saved = selection.freeze(path, source="triage", rows=[training])
                return runner.CliResult(True, {"selection_id": saved["selection_id"]})
            if command == "promote":
                return runner.CliResult(True, {"outcome": "buffered"})
            self.fail(f"unexpected Campaign command: {command}")

        return reply

    def test_campaign_promotes_from_the_same_training_selection(self):
        log = mock.Mock()
        with mock.patch.object(campaign, "run_cli", side_effect=self.campaign_reply()) as cli:
            campaign.process(self.root, self.root, log, self.spec, {"rows": [candidate()]})
        triage_call, promote_call = [call.args for call in cli.call_args_list]
        self.assertEqual(triage_call[1:5], ("triage", "--ids", "candidate", "--against-shelf"))
        self.assertEqual(triage_call[triage_call.index("--max-shelf-correlation") + 1], "0.7")
        self.assertNotIn("--show-test", triage_call)
        path = triage_call[triage_call.index("--freeze") + 1]
        self.assertEqual(promote_call[1:3], ("promote", "candidate"))
        self.assertEqual(promote_call[promote_call.index("--selection") + 1], path)
        self.assertIn("--max-self-correlation", promote_call)
        self.assertEqual(log.call_args.args[0]["outcome"], "buffered")

    def test_campaign_skips_dilutive_and_duplicate_candidates_without_validation(self):
        for changes, reason in (
            ({"delta": 0.0}, "insufficient_train_marginal"),
            ({"shelf_correlation": {"state": "measured", "eligible": 1, "measured": 1,
                                     "max_rho": 0.7, "max_rho_signed": -0.7,
                                     "max_member": "other"}}, "shelf_duplicate"),
            ({"max_member_rho": 0.99}, "book_duplicate"),
            ({"shelf_correlation": {"state": "unknown", "already_shelved": True}}, "already_shelved"),
        ):
            changes["training_gate"] = {"state": "rejected", "reason": reason}
            with self.subTest(reason=reason), \
                    mock.patch.object(campaign, "run_cli",
                                      side_effect=self.campaign_reply(freeze=False, **changes)) as cli:
                log = mock.Mock()
                campaign.process(self.root, self.root, log, self.spec, {"rows": [candidate()]})
                cli.assert_called_once()
                self.assertEqual(log.call_args.args[0]["kind"], "training-rejected")
                self.assertEqual(log.call_args.args[0]["training_gate"]["reason"], reason)

    def test_campaign_keeps_pending_evidence_in_queue_without_validation(self):
        changes = [
            {"shelf_correlation": {"state": "unknown", "eligible": 2, "measured": 1,
                                  "max_rho": 0.14},
             "training_gate": {"state": "unknown", "reason": "incomplete_shelf_evidence"}},
            {"evidence": {"state": "unknown"}, "delta": 0.1,
             "training_gate": {"state": "unknown", "reason": "incomplete_book_or_candidate_evidence"}},
            {"delta": None},
            {},  # Plausible training values without a frozen selection must also defer.
        ]
        for evidence in changes:
            with self.subTest(evidence=evidence), \
                    mock.patch.object(runner, "run_probe", return_value=runner.ProbeRun(
                        "done", {"rows": [candidate()]})), \
                    mock.patch.object(campaign, "run_cli",
                                      side_effect=self.campaign_reply(freeze=False, **evidence)) as cli:
                log = mock.Mock()
                consume = lambda spec, data: campaign.process(self.root, self.root, log, spec, data)
                code = runner.run_queue(self.root, self.inbox, lambda _: (self.batch, self.receipt),
                                        log, max_wait=30, interval=1, once=True, consume=consume)
                self.assertEqual(code, 3)
                self.assertTrue(self.spec.exists())
                cli.assert_called_once()
                self.assertEqual(log.call_args.args[0]["kind"], "deferred")

    def test_campaign_needs_halves_only_after_initial_training_gate(self):
        rows = [candidate(train_sharpe=0.5, train_sharpe_h2=None)]
        with mock.patch.object(campaign, "run_cli") as cli:
            campaign.process(self.root, self.root, mock.Mock(), self.spec, {"rows": rows})
            with self.assertRaisesRegex(RuntimeError, "half-window"):
                campaign.process(self.root, self.root, mock.Mock(), self.spec,
                                 {"rows": [candidate(train_sharpe_h2=None)]})
        cli.assert_not_called()

    def test_campaign_does_not_archive_after_an_operational_error(self):
        row = candidate()
        consume = lambda spec, data: campaign.process(self.root, self.root, mock.Mock(), spec, data)
        report = mock.Mock()
        with mock.patch.object(runner, "run_probe", return_value=runner.ProbeRun("done", {"rows": [row]})), \
                mock.patch.object(campaign, "run_cli", return_value=runner.CliResult(False, {}, "connection lost")):
            code = runner.run_queue(self.root, self.inbox, lambda _: (self.batch, self.receipt),
                                    report, max_wait=30, interval=1, once=True, consume=consume)
        self.assertEqual(code, 3)
        self.assertTrue(self.spec.exists())
        self.assertIn("training evidence for candidate: connection lost", report.call_args.args[0]["detail"])
