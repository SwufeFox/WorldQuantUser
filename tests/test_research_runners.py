"""Background workers depend on saved arm states, not the subprocess exit code."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
import json
import subprocess

from research_app import experiments, selection
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
        identity = mock.patch.object(experiments, "probe_spec_identity",
                                     return_value={**self.preview.data, "track": "REGULAR"})
        identity.start()
        self.addCleanup(identity.stop)

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
        with mock.patch.object(runner, "run_cli", return_value=finished) as run:
            result = self.run_probe()
        self.assertEqual(result.state, "done")
        run.assert_called_once()
        self.assertEqual(run.call_args.args[2:4], ("--resume", self.batch))

    def test_terminal_platform_task_without_alpha_does_not_defer_the_queue(self):
        for status in ("WARNING", "COMPLETE"):
            with self.subTest(status=status):
                row = arm(status, submission_state="terminal")
                self.assertEqual(runner._probe_state({"rows": [row], "execution": {"finished": 0}}),
                                 ("failed", None))

    def test_unknown_post_uses_read_only_recovery_even_if_other_arms_are_queued(self):
        rows = [arm("UNKNOWN_SUBMISSION", submission_state="unknown_submission"),
                arm("QUEUED", submission_state="not_submitted")]
        self.saved(self.batch, rows)
        with mock.patch.object(runner, "run_cli",
                               return_value=runner.CliResult(True, {"rows": rows})) as run:
            result = self.run_probe()
        self.assertEqual(result.state, "deferred")
        self.assertEqual(run.call_args.args[2:4], ("--recover", self.batch))

    def test_current_probe_work_actions_use_canonical_state(self):
        rows = [arm("PENDING", simulation_status="RUNNING"), arm("QUEUED")]
        execution = {"total": 2, "finished": 0, "terminal": 0, "unknown_submission": 0}
        data = {"rows": rows, "execution": execution, "next_actions": [{"action": "resume"}]}
        self.assertEqual(runner.probe_state(data), ("deferred", "resume"))
        execution["unknown_submission"] = 1
        self.assertEqual(runner.probe_state(data), ("deferred", "recover"))
        execution.update(unknown_submission=0, finished=1, terminal=1)
        self.assertEqual(runner.probe_state(data), ("failed", None))

    def test_recover_only_never_launches_unposted_work(self):
        with mock.patch.object(runner, "run_cli", return_value=self.preview) as run:
            result = runner.run_probe(self.root, self.spec, self.batch, self.receipt,
                                      max_wait=30, recover_only=True)
            self.assertEqual(result.state, "deferred")
            run.assert_called_once()
        self.saved(self.batch, [arm("QUEUED", submission_state="not_submitted")])
        with mock.patch.object(runner, "run_cli", return_value=self.preview) as run:
            result = runner.run_probe(self.root, self.spec, self.batch, self.receipt,
                                      max_wait=30, recover_only=True)
            self.assertEqual(result.state, "deferred")
            run.assert_not_called()
        pending = arm("PENDING", progress_url="https://test.invalid/1", submission_state="accepted")
        self.saved(self.batch, [pending])
        observed = runner.CliResult(True, {"rows": [pending]})
        with mock.patch.object(runner, "run_cli", return_value=observed) as run:
            runner.run_probe(self.root, self.spec, self.batch, self.receipt,
                             max_wait=30, recover_only=True, multi_size=8, strict_gate=True)
            self.assertEqual(run.call_args.args[2:4], ("--recover", self.batch))
            self.assertNotIn("--multi-size", run.call_args.args)
            self.assertNotIn("--strict-gate", run.call_args.args)

    def test_launch_options_apply_to_fresh_probe(self):
        observed = runner.CliResult(True, {"rows": [arm("COMPLETE", alpha_id="alpha")]})
        with mock.patch.object(runner, "run_cli", side_effect=[self.preview, observed]) as run:
            runner.run_probe(self.root, self.spec, self.batch, self.receipt,
                             max_wait=30, multi_size=8, strict_gate=True)
        args = run.call_args.args
        self.assertEqual(args[args.index("--multi-size") + 1], "8")
        self.assertIn("--strict-gate", args)
        self.assertIn("--strict-gate", run.call_args_list[0].args)
        self.assertIn("--multi-size", run.call_args_list[0].args)

    def test_offline_syntax_errors_stop_before_the_live_probe(self):
        preview = runner.CliResult(True, {**self.preview.data,
            "blocking": [{"code": "reserved_identifier"}, {"code": "unknown_identifier"}]})
        with mock.patch.object(runner, "run_cli", return_value=preview) as run:
            result = self.run_probe()
        run.assert_called_once()
        self.assertEqual(result.state, "failed")
        self.assertIn("reserved_identifier", result.detail)

    def test_cached_field_findings_leave_the_live_probe_available(self):
        preview = runner.CliResult(True, {**self.preview.data,
            "blocking": [{"code": "unknown_identifier"}]})
        finished = runner.CliResult(True, {"rows": [arm("COMPLETE", alpha_id="alpha")]})
        with mock.patch.object(runner, "run_cli", side_effect=[preview, finished]) as run:
            self.assertEqual(self.run_probe().state, "done")
        self.assertEqual(run.call_count, 2)

    def test_same_wire_request_does_not_resume_under_a_new_research_track(self):
        self.saved(self.batch, [arm("QUEUED", submission_state="not_submitted")])
        document = json.loads(self.batch.read_text(encoding="utf-8"))
        document["track"] = "ATOM"
        self.batch.write_text(json.dumps(document), encoding="utf-8")
        with mock.patch.object(runner, "run_cli") as run:
            self.assertEqual(self.run_probe().state, "failed")
            run.assert_not_called()

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
        with mock.patch.object(runner, "run_cli") as run:
            result = self.run_probe()
        self.assertEqual(result.state, "failed")
        self.assertIn("does not match", result.detail)
        run.assert_not_called()

    def test_changed_quick_provenance_cannot_resume_the_same_full_request(self):
        self.saved(self.batch, [arm("PENDING", submission_state="accepted")])
        payload = json.loads(self.batch.read_text(encoding="utf-8"))
        payload["quick_source"] = {"run_id": "original"}
        self.batch.write_text(json.dumps(payload), encoding="utf-8")

        with mock.patch.object(runner, "run_cli") as run:
            result = self.run_probe()
        self.assertEqual(result.state, "failed")
        self.assertIn("does not match", result.detail)
        run.assert_not_called()

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

    def test_campaign_recover_only_prioritizes_known_work_in_once_mode(self):
        backlog = self.root / "backlog"
        backlog.mkdir()
        new, queued, accepted = [backlog / name for name in ("a-new.json", "b-queued.json", "c-accepted.json")]
        for spec in (new, queued, accepted):
            spec.write_text("{}", encoding="utf-8")
        batch, receipt = campaign.campaign_paths(self.root, accepted)
        queued_batch, _ = campaign.campaign_paths(self.root, queued)
        self.saved(queued_batch, [arm("QUEUED", submission_state="not_submitted")])
        self.saved(batch, [arm("PENDING", submission_state="accepted")])
        execute = mock.Mock(return_value=runner.ProbeRun("deferred", {}))
        log = mock.Mock()

        self.assertEqual(campaign.run_campaign(self.root, once=True, execute=execute,
                                              log_event=log, recover_only=True), 3)
        execute.assert_called_once_with(self.root, accepted, batch, receipt, max_wait=600, concurrency=3)
        self.assertTrue(new.is_file())
        self.assertTrue(queued.is_file())
        self.assertEqual(log.call_args.args[0]["spec"], accepted.name)

    def test_campaign_advances_a_ready_spec_while_an_accepted_one_is_waiting(self):
        backlog = self.root / "backlog"
        backlog.mkdir()
        waiting, fresh, ready = [backlog / name for name in ("a-wait.json", "b-new.json", "c-ready.json")]
        for spec in (waiting, fresh, ready):
            spec.write_text("{}", encoding="utf-8")
        for spec, due in ((waiting, 120), (ready, 100)):
            batch, _ = campaign.campaign_paths(self.root, spec)
            self.saved(batch, [arm("PENDING", submission_state="accepted",
                                   progress_url="https://test.invalid/1", next_poll_at=due)])
            payload = json.loads(batch.read_text(encoding="utf-8"))
            payload["execution"] = {"total": 1, "finished": 0}
            batch.write_text(json.dumps(payload), encoding="utf-8")
        execute = mock.Mock(return_value=runner.ProbeRun("done", {}))
        with mock.patch.object(runner.time, "time", return_value=100):
            self.assertEqual(campaign.run_campaign(self.root, once=True, execute=execute,
                                                   log_event=mock.Mock()), 0)
        execute.assert_called_once_with(self.root, ready, *campaign.campaign_paths(self.root, ready),
                                      max_wait=600, concurrency=3)
        self.assertTrue(waiting.is_file())
        self.assertTrue(fresh.is_file())

    def test_campaign_reports_cooling_checkpoints_without_invoking_probe(self):
        backlog = self.root / "backlog"
        backlog.mkdir()
        waiting = backlog / "wait.json"
        waiting.write_text("{}", encoding="utf-8")
        batch, _ = campaign.campaign_paths(self.root, waiting)
        self.saved(batch, [arm("PENDING", submission_state="accepted", alpha_id="alpha",
                               metrics_pending=True, next_poll_at=120)])
        payload = json.loads(batch.read_text(encoding="utf-8"))
        payload["execution"] = {"total": 1, "finished": 0}
        batch.write_text(json.dumps(payload), encoding="utf-8")
        execute, log = mock.Mock(), mock.Mock()
        with mock.patch.object(runner.time, "time", return_value=100):
            self.assertEqual(campaign.run_campaign(self.root, once=True, execute=execute, log_event=log), 3)
        execute.assert_not_called()
        self.assertEqual(log.call_args.args[0]["next_poll_at"], 120)
        self.assertTrue(waiting.is_file())

    def test_campaign_uses_the_native_parent_clock_to_advance_other_work(self):
        backlog = self.root / "backlog"
        backlog.mkdir()
        waiting, fresh = [backlog / name for name in ("a-native.json", "b-ready.json")]
        for spec in (waiting, fresh):
            spec.write_text("{}", encoding="utf-8")
        batch, _ = campaign.campaign_paths(self.root, waiting)
        parent = {"multi_request_id": "saved-native", "multi_parent_url": "https://test.invalid/parent"}
        rows = [arm("PENDING", submission_state="accepted", **parent) for _ in range(2)]
        self.saved(batch, rows)
        document = json.loads(batch.read_text(encoding="utf-8"))
        anchor = {**parent, "multi_member": 0, "multi_next_parent_poll_at": 120,
                  "slot_reservation_state": "accepted", "slot_reservation": {"id": "saved-slot"}}
        document.update(execution={"total": 2, "finished": 0},
                        records=[anchor, {**parent, "multi_member": 1}])
        batch.write_text(json.dumps(document), encoding="utf-8")

        execute = mock.Mock(return_value=runner.ProbeRun("done", {}))
        with mock.patch.object(runner.time, "time", return_value=100):
            self.assertEqual(campaign.run_campaign(self.root, once=True, execute=execute,
                                                   log_event=mock.Mock()), 0)
        execute.assert_called_once_with(self.root, fresh, *campaign.campaign_paths(self.root, fresh),
                                      max_wait=600, concurrency=3)
        self.assertTrue(waiting.is_file())

        anchor.pop("slot_reservation_state")
        self.assertIsNone(runner.probe_poll_at(document))
        anchor["slot_reservation_state"] = "accepted"
        for row in rows:
            row["multi_parent_terminal"] = True
        document["rows"] = rows
        self.assertIsNone(runner.probe_poll_at(document))

    def test_terminal_sibling_does_not_hide_an_explicit_poll_time(self):
        waiting = arm("PENDING", submission_state="accepted",
                      progress_url="https://test.invalid/accepted", next_poll_at=125)
        failed = arm("ERROR", submission_state="terminal")
        document = {"execution": {"total": 2, "finished": 0}, "rows": [waiting, failed]}
        self.assertEqual(runner.probe_poll_at(document), 125)
        document["rows"] = [failed, failed]
        self.assertIsNone(runner.probe_poll_at(document))

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

    def test_campaign_reuses_frozen_selection_and_terminal_promotion_receipt(self):
        def write_receipt(_root, command, *args, **kwargs):
            data = self.campaign_reply()(_root, command, *args, **kwargs)
            if command == "promote":
                path = args[args.index("--save") + 1]
                choice = args[args.index("--selection") + 1]
                payload = {"command": "promote", "ok": True,
                           "data": {"alpha_id": "candidate", "freeze_path": str(choice), **data.data}}
                path.write_text(json.dumps(payload), encoding="utf-8")
            return data

        rows = [candidate(alpha_id="pending", train_sharpe_h2=None), candidate()]
        log = mock.Mock()
        with mock.patch.object(campaign, "run_cli", side_effect=write_receipt) as cli:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "half-window metrics incomplete for pending"):
                    campaign.process(self.root, self.root, log, self.spec, {"rows": rows})
        self.assertEqual([call.args[1] for call in cli.call_args_list], ["triage", "promote"])
        self.assertTrue((self.root / "campaign-freeze-candidate.json").is_file())

    def test_campaign_archives_unresolved_check_only_after_a_retry(self):
        attempted = []

        def reply(root, command, *args, **kwargs):
            if command == "triage":
                return self.campaign_reply()(root, command, *args, **kwargs)
            choice = args[args.index("--selection") + 1]
            attempted.append(choice)
            path = args[args.index("--save") + 1]
            outcome = ({"outcome": "rejected", "reasons": ["platform_checks_not_resolved"]}
                       if len(attempted) == 1 else {"outcome": "buffered", "reasons": []})
            data = {"alpha_id": "candidate", "freeze_path": str(choice), **outcome}
            path.write_text(json.dumps({"command": "promote", "ok": True, "data": data}),
                            encoding="utf-8")
            return runner.CliResult(True, data)

        with mock.patch.object(campaign, "run_cli", side_effect=reply) as cli:
            with self.assertRaisesRegex(RuntimeError, "promotion evidence incomplete"):
                campaign.process(self.root, self.root, mock.Mock(), self.spec, {"rows": [candidate()]})
            campaign.process(self.root, self.root, mock.Mock(), self.spec, {"rows": [candidate()]})
        self.assertEqual([call.args[1] for call in cli.call_args_list], ["triage", "promote", "promote"])
        self.assertEqual(attempted, [self.root / "campaign-freeze-candidate.json"] * 2)

    def test_campaign_keep_train_only_and_non_regular_routes_out_of_validation(self):
        cases = [
            (False, {"track": "REGULAR"}, {"simulationMode": "FULL"}, "training-frozen"),
            (True, {"track": "REGULAR"}, {"simulationMode": "QUICK"}, "research-only"),
            (True, {"track": "POWER_POOL"}, {"simulationMode": "FULL"}, "research-only"),
            (True, {"track": "REGULAR"}, {"type": "SUPER", "simulationMode": "FULL"}, "research-only"),
        ]
        for index, (validate, data, settings, expected) in enumerate(cases):
            with self.subTest(data=data, settings=settings, validate=validate):
                log = mock.Mock()
                directory = self.root / str(index)
                directory.mkdir()
                with mock.patch.object(campaign, "run_cli", side_effect=self.campaign_reply()) as cli:
                    campaign.process(self.root, directory, log, self.spec,
                                     {**data, "rows": [candidate(settings=settings)]}, validate=validate)
                commands = ["triage"] if expected == "training-frozen" else []
                self.assertEqual([call.args[1] for call in cli.call_args_list], commands)
                self.assertEqual(log.call_args.args[0]["kind"], expected)

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
