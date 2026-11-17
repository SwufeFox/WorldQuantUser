"""Background workers depend on saved arm states, not the subprocess exit code."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
import json
import subprocess

from tools import _runner as runner
from tools import campaign_runner as campaign


def arm(status, **fields):
    return {"status": status, **fields}


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

    def test_campaign_checks_only_pairs_with_new_candidate(self):
        existing = runner.CliResult(True, {"rows": [{"alpha_id": "A"}, {"alpha_id": "B"}]})
        correlation = runner.CliResult(True, {"matrix": {"pairs": [
            {"a": "A", "b": "B", "r": 0.98}, {"a": "A", "b": "C", "r": 0.10}
        ]}})
        with mock.patch.object(campaign, "run_cli", side_effect=[existing, correlation]):
            self.assertIsNone(campaign.duplicate_of(self.root, "C"))

    def test_campaign_does_not_archive_after_an_operational_error(self):
        row = arm("COMPLETE", alpha_id="candidate", train_sharpe=2.0, train_fitness=1.2,
                  train_sharpe_h1=1.5, train_sharpe_h2=2.0)
        consume = lambda spec, data: campaign.process(self.root, self.root, mock.Mock(), spec, data)
        report = mock.Mock()
        with mock.patch.object(runner, "run_probe", return_value=runner.ProbeRun("done", {"rows": [row]})), \
                mock.patch.object(campaign, "run_cli", return_value=runner.CliResult(False, {}, "connection lost")):
            code = runner.run_queue(self.root, self.inbox, lambda _: (self.batch, self.receipt),
                                    report, max_wait=30, interval=1, once=True, consume=consume)
        self.assertEqual(code, 3)
        self.assertTrue(self.spec.exists())
        self.assertIn("read shelf: connection lost", report.call_args.args[0]["detail"])
