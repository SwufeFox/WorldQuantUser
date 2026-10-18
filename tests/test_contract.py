#!/usr/bin/env python3
"""Contract tests for the research toolchain.

Everything here runs offline: no BRAIN account, no credentials, no network.
That is deliberate -- the whole point of the CLI contract is that a caller can
discover and validate behaviour without spending quota.

Run with::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import machine_lib as ml  # noqa: E402
import research_cli as cli  # noqa: E402
import research_config as config_io  # noqa: E402


def run_cli(*args, expect_env=True):
    """Run the CLI in a subprocess with BRAIN_* credentials stripped."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("BRAIN_")}
    if not expect_env:
        env.pop("BRAIN_USERNAME", None)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "research_cli.py"), *args],
        capture_output=True, text=True, cwd=str(ROOT), env=env,
    )
    return proc


def parse(proc):
    """Every invocation must leave parseable JSON on stdout."""
    return json.loads(proc.stdout)


class SchemaContractTests(unittest.TestCase):
    def test_schema_is_single_json_document(self):
        proc = run_cli("schema")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = parse(proc)
        self.assertTrue(payload["ok"])
        data = payload["data"]
        for key in ("exit_codes", "error_codes", "research_file",
                    "settings_defaults", "credentials", "commands"):
            self.assertIn(key, data)

    def test_schema_lists_every_command_and_its_options(self):
        commands = parse(run_cli("schema"))["data"]["commands"]["subcommands"]
        self.assertEqual(
            set(commands), {"schema", "auth", "catalog", "simulate", "resume",
                            "alpha", "lint", "cache"},
        )
        simulate = commands["simulate"]
        flags = {flag for option in simulate["options"] for flag in option["flags"]}
        self.assertIn("--dry-run", flags)
        self.assertIn("--sort", flags)
        self.assertIn("--max-wait-seconds", flags)
        catalog = commands["catalog"]["subcommands"]
        self.assertIn("--dataset", {f for o in catalog["fields"]["options"] for f in o["flags"]})
        cache = commands["cache"]["subcommands"]
        self.assertEqual(set(cache), {"info", "clear"})

    def test_schema_can_describe_one_command(self):
        payload = parse(run_cli("schema", "simulate"))["data"]
        self.assertEqual(payload["command"]["command"], "simulate")

    def test_schema_rejects_unknown_command(self):
        proc = run_cli("schema", "nope")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(parse(proc)["error"]["code"], "USAGE")


class ErrorEnvelopeTests(unittest.TestCase):
    def test_unknown_command_still_prints_json_on_stdout(self):
        proc = run_cli("bogus")
        self.assertEqual(proc.returncode, 2)
        payload = parse(proc)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "USAGE")

    def test_command_without_subcommand(self):
        proc = run_cli("catalog")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(parse(proc)["error"]["code"], "USAGE")

    def test_simulate_without_any_alpha(self):
        proc = run_cli("simulate")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(parse(proc)["error"]["code"], "USAGE")

    def test_unknown_region_is_invalid_input(self):
        proc = run_cli("simulate", "--expr", "rank(cap)", "--region", "MARS")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(parse(proc)["error"]["code"], "USAGE")  # argparse choices

    def test_region_validation_comes_from_machine_lib(self):
        with self.assertRaises(ValueError):
            ml.resolve_settings({"region": "MARS"})

    def test_every_error_code_has_an_exit_code(self):
        for code in cli.ERROR_CODES:
            self.assertIn(code, cli.EXIT_CODES, code)


class DryRunTests(unittest.TestCase):
    def test_dry_run_needs_no_credentials_and_no_network(self):
        proc = run_cli("simulate", "--expr", "rank(cap)", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = parse(proc)["data"]
        self.assertTrue(data["dry_run"])
        self.assertEqual(data["summary"]["total"], 1)

    def test_dry_run_payload_is_a_valid_simulation_body(self):
        proc = run_cli("simulate", "--expr", "rank(cap)", "--decay", "4",
                       "--region", "USA", "--neutralization", "SUBINDUSTRY", "--dry-run")
        record = parse(proc)["data"]["results"][0]
        self.assertEqual(record["expression"], "rank(cap)")
        self.assertEqual(record["decay"], 4)
        payload = record["payload"]
        self.assertEqual(payload["type"], "REGULAR")
        self.assertEqual(payload["regular"], "rank(cap)")
        self.assertEqual(payload["settings"]["decay"], 4)
        self.assertEqual(payload["settings"]["region"], "USA")

    def test_cli_flags_override_research_file_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps({
                "settings": {"region": "USA", "decay": 9},
                "alphas": ["rank(cap)"],
            }))
            proc = run_cli("simulate", str(path), "--decay", "2", "--dry-run")
            data = parse(proc)["data"]
            self.assertEqual(data["settings"]["decay"], 2)

    def test_per_alpha_decay_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps([
                {"expression": "rank(cap)", "decay": 7},
                "rank(assets)",
            ]))
            data = parse(run_cli("simulate", str(path), "--decay", "1",
                                 "--dry-run"))["data"]
            decays = [r["decay"] for r in data["results"]]
            self.assertEqual(decays, [7, 1])


class ResearchConfigTests(unittest.TestCase):
    def test_json_array(self):
        settings, alphas = config_io.parse_research_text('["rank(cap)"]')
        self.assertEqual(settings, {})
        self.assertEqual(alphas, ["rank(cap)"])

    def test_json_object_with_settings(self):
        text = json.dumps({"settings": {"decay": 3}, "alphas": ["rank(cap)"]})
        settings, alphas = config_io.parse_research_text(text)
        self.assertEqual(settings["decay"], 3)
        self.assertEqual(alphas, ["rank(cap)"])

    def test_single_expression_object(self):
        settings, alphas = config_io.parse_research_text('{"expression": "rank(cap)"}')
        self.assertEqual(alphas, [{"expression": "rank(cap)"}])

    def test_plain_text_skips_blanks_and_comments(self):
        text = "# a comment\n\nrank(cap)\n  rank(assets)  \n"
        _, alphas = config_io.parse_research_text(text)
        self.assertEqual(alphas, ["rank(cap)", "rank(assets)"])

    def test_invalid_json_is_reported_with_source(self):
        with self.assertRaises(config_io.ResearchError) as ctx:
            config_io.parse_research_text('{"alphas": [}', "demo.json")
        self.assertIn("demo.json", str(ctx.exception))

    def test_missing_file(self):
        with self.assertRaises(config_io.ResearchError):
            config_io.load_research_file("/nonexistent/research.json")

    def test_object_without_alphas_or_expression(self):
        with self.assertRaises(config_io.ResearchError):
            config_io.parse_research_text('{"settings": {}}')

    def test_build_config_normalizes_entries(self):
        settings, pairs = config_io.build_config({"decay": 5}, ["rank(cap)", {"expression": "rank(assets)", "decay": 2}])
        self.assertEqual(pairs, [("rank(cap)", 5), ("rank(assets)", 2)])
        self.assertEqual(settings["region"], ml.DEFAULT_SETTINGS["region"])

    def test_build_config_rejects_bad_decay(self):
        with self.assertRaises(config_io.ResearchError):
            config_io.build_config({}, [{"expression": "rank(cap)", "decay": -1}])

    def test_build_config_rejects_empty_expression(self):
        with self.assertRaises(config_io.ResearchError):
            config_io.build_config({}, ["   "])

    def test_build_config_rejects_nothing_to_do(self):
        with self.assertRaises(config_io.ResearchError):
            config_io.build_config({}, [])


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self._saved_flag = ml._DOTENV_LOADED
        self._saved_env = dict(os.environ)

    def tearDown(self):
        ml._DOTENV_LOADED = self._saved_flag
        for key in list(os.environ):
            if key not in self._saved_env:
                del os.environ[key]
        os.environ.update(self._saved_env)

    def test_missing_credentials_raise_a_typed_error(self):
        ml._DOTENV_LOADED = True  # skip the .env file
        with mock.patch.dict(os.environ, {"BRAIN_USERNAME": "", "BRAIN_PASSWORD": ""}):
            with self.assertRaises(ml.MissingCredentials) as ctx:
                ml.get_credentials()
        self.assertEqual(ctx.exception.code, "AUTH_MISSING")

    def test_dotenv_parsing_handles_quotes_export_and_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "# comment\n"
                "export BRAIN_USERNAME='quoted user'\n"
                'BRAIN_PASSWORD="quoted pass"\n'
                "IGNORED\n"
            )
            ml._DOTENV_LOADED = False
            for key in ("BRAIN_USERNAME", "BRAIN_PASSWORD"):
                os.environ.pop(key, None)
            ml._load_dotenv(path)
        self.assertEqual(os.environ["BRAIN_USERNAME"], "quoted user")
        self.assertEqual(os.environ["BRAIN_PASSWORD"], "quoted pass")

    def test_real_environment_wins_over_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("BRAIN_USERNAME=from-file\nBRAIN_PASSWORD=from-file\n")
            ml._DOTENV_LOADED = False
            with mock.patch.dict(os.environ, {"BRAIN_USERNAME": "from-env"}):
                ml._load_dotenv(path)
                self.assertEqual(os.environ["BRAIN_USERNAME"], "from-env")
                self.assertEqual(os.environ["BRAIN_PASSWORD"], "from-file")


class PrivacyTests(unittest.TestCase):
    """The repo must never carry credentials or session cookies."""

    SECRET_PATHS = (".env", ".brain_session.json")

    def test_no_secret_file_is_tracked(self):
        tracked = subprocess.run(["git", "ls-files"], cwd=str(ROOT),
                                 capture_output=True, text=True).stdout.splitlines()
        for name in tracked:
            self.assertNotIn(name, (".env", ".brain_session.json"))

    def test_secret_files_are_git_ignored(self):
        for name in self.SECRET_PATHS:
            proc = subprocess.run(["git", "check-ignore", name], cwd=str(ROOT),
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, "%s is not git-ignored" % name)

    def test_gitignore_keeps_the_example_template(self):
        proc = subprocess.run(["git", "check-ignore", ".env.example"], cwd=str(ROOT),
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1, ".env.example should be committable")

    def test_tracked_files_never_contain_the_real_credentials(self):
        """The strongest privacy check: the actual .env values are nowhere in git."""
        env_file = ROOT / ".env"
        if not env_file.is_file():
            self.skipTest("no .env present to compare against")
        values = []
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("BRAIN_") and "=" in line:
                value = line.partition("=")[2].strip().strip("'\"")
                if value:
                    values.append(value)
        if not values:
            self.skipTest(".env has no filled-in values")
        listing = subprocess.run(["git", "ls-files"], cwd=str(ROOT),
                                 capture_output=True, text=True).stdout.splitlines()
        for name in listing:
            path = ROOT / name
            if not path.is_file():
                continue
            text = path.read_text(errors="ignore")
            for value in values:
                self.assertNotIn(
                    value, text,
                    "tracked file %s leaks a credential value from .env" % name)


class CheckExtractionTests(unittest.TestCase):
    def test_extract_checks_flattens_platform_checks(self):
        payload = {"is": {"checks": [
            {"name": "LOW_SHARPE", "result": "PASS", "limit": 1.25, "value": 1.46},
            {"name": "SELF_CORRELATION", "result": "PENDING"},
            {"result": "PASS"},  # no name -> skipped
        ]}}
        checks = ml._extract_checks(payload)
        self.assertEqual(sorted(checks), ["LOW_SHARPE", "SELF_CORRELATION"])
        self.assertEqual(checks["LOW_SHARPE"]["value"], 1.46)

    def test_extract_checks_returns_none_when_absent(self):
        self.assertIsNone(ml._extract_checks({}))


class SummaryTests(unittest.TestCase):
    def test_ranking_prefers_sharpe_and_records_failures(self):
        records = [
            {"status": "COMPLETE", "alpha_id": "A", "expression": "x",
             "metrics": {"sharpe": 1.0, "fitness": 0.5}},
            {"status": "COMPLETE", "alpha_id": "B", "expression": "y",
             "metrics": {"sharpe": 2.0, "fitness": 0.9}},
            {"status": "FAILED", "expression": "z", "error": "boom"},
            {"status": "WARNING", "expression": "w"},
        ]
        summary = cli.summarize(records)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["other"], 1)
        self.assertEqual(summary["best"]["alpha_id"], "B")

    def test_records_without_metrics_are_not_ranked(self):
        summary = cli.summarize([
            {"status": "COMPLETE", "alpha_id": "A", "expression": "x", "metrics": None},
        ])
        self.assertEqual(summary["ranking"], [])
        self.assertIsNone(summary["best"])

    def test_failed_record_with_progress_url_is_still_resumable(self):
        summary = cli.summarize([
            {"status": "FAILED", "progress_url": "https://api.worldquantbrain.com/simulations/1"},
        ])
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["failed"], 1)


class SimulationResumeTests(unittest.TestCase):
    class Response:
        def __init__(self, status_code=200, headers=None, payload=None):
            self.status_code = status_code
            self.ok = status_code < 400
            self.headers = headers or {}
            self.payload = payload or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("HTTP %s" % self.status_code)

        def json(self):
            return self.payload

    class Session:
        def __init__(self, get_responses=None, post_responses=None):
            self.get_responses = list(get_responses or [])
            self.post_responses = list(post_responses or [])
            self.get_calls = []
            self.post_calls = []

        def get(self, url, **kwargs):
            self.get_calls.append(url)
            return self.get_responses.pop(0)

        def post(self, url, **kwargs):
            self.post_calls.append((url, kwargs))
            return self.post_responses.pop(0)

    def test_checkpoint_is_atomic_and_private(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.json"
            cli._write_simulation_checkpoint(path, {"records": []})
            self.assertEqual(json.loads(path.read_text()), {"records": []})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(list(path.parent.glob(".checkpoint.json.*.tmp")), [])

    def test_progress_url_rejects_untrusted_hosts_before_authenticated_get(self):
        unsafe_urls = [
            "http://api.worldquantbrain.com/simulations/1",
            "https://evil.example/simulations/1",
            "https://api.worldquantbrain.com.evil.example/simulations/1",
            "https://user:pass@api.worldquantbrain.com/simulations/1",
            "https://api.worldquantbrain.com:444/simulations/1",
            "https://api.worldquantbrain.com/alphas/1",
        ]
        for url in unsafe_urls:
            with self.subTest(url=url):
                session = self.Session()
                with self.assertRaises(ml.SimulationError):
                    ml.poll_simulation(session, url)
                self.assertEqual(session.get_calls, [])

    def test_submit_resolves_relative_location_and_rejects_external_location(self):
        session = self.Session(post_responses=[self.Response(
            status_code=201, headers={"Location": "/simulations/123"},
        )])
        progress_url = ml.submit_single_simulation(session, {"regular": "rank(cap)"})
        self.assertEqual(
            progress_url,
            "https://api.worldquantbrain.com/simulations/123",
        )

        bad_session = self.Session(post_responses=[self.Response(
            status_code=201,
            headers={"Location": "https://evil.example/simulations/123"},
        )])
        with self.assertRaises(ml.SimulationError):
            ml.submit_single_simulation(bad_session, {"regular": "rank(cap)"})

    def test_auth_error_during_submit_stops_batch_for_safe_retry(self):
        session = self.Session(post_responses=[self.Response(status_code=401)])
        with self.assertRaises(ml.AuthError):
            ml.simulate_alphas(
                session,
                [("rank(cap)", 1), ("rank(volume)", 1)],
                batch_size=2,
                max_wait_seconds=10,
            )
        self.assertEqual(len(session.post_calls), 1)

    def test_auth_error_during_poll_preserves_accepted_progress_url(self):
        record = {
            "expression": "rank(cap)",
            "decay": 1,
            "status": "SUBMITTED",
            "progress_url": "https://api.worldquantbrain.com/simulations/1",
        }
        with mock.patch.object(ml, "poll_simulation", side_effect=ml.AuthError("expired")):
            results = ml.simulate_alphas(
                self.Session(),
                [("rank(cap)", 1)],
                max_wait_seconds=10,
                initial_records=[record],
            )
        self.assertEqual(results[0]["status"], "PENDING")
        self.assertEqual(results[0]["progress_url"], record["progress_url"])

    def test_auth_error_fetching_metrics_keeps_checkpoint_retryable(self):
        record = {
            "expression": "rank(cap)",
            "decay": 1,
            "alpha_id": "alpha-1",
            "status": "PENDING",
            "simulation_status": "COMPLETE",
            "metrics_pending": True,
        }
        events = []
        with mock.patch.object(
            ml, "fetch_alpha_metrics", side_effect=ml.AuthError("expired session")
        ):
            results = ml.simulate_alphas(
                self.Session(),
                [("rank(cap)", 1)],
                max_wait_seconds=10,
                initial_records=[record],
                on_event=lambda event, data: events.append(event),
            )
        self.assertEqual(results[0]["status"], "PENDING")
        self.assertTrue(results[0]["metrics_pending"])
        self.assertIn("pending", events)

    def test_long_retry_after_returns_pending_without_sleeping(self):
        session = self.Session([self.Response(headers={"Retry-After": "7200"})])
        with mock.patch.object(ml, "sleep") as sleep:
            result = ml.poll_simulation(
                session,
                "https://api.worldquantbrain.com/simulations/1",
                deadline=time.monotonic() + 10,
            )
        self.assertEqual(result["status"], "PENDING")
        self.assertEqual(
            result["progress_url"],
            "https://api.worldquantbrain.com/simulations/1",
        )
        self.assertEqual(result["next_poll_after_seconds"], 7200)
        sleep.assert_not_called()
        self.assertEqual(len(session.get_calls), 1)

    def test_rate_limited_progress_response_uses_server_delay_without_retrying(self):
        session = self.Session([
            self.Response(status_code=429, headers={"Retry-After": "7200"}),
        ])
        with mock.patch.object(ml, "sleep") as sleep:
            result = ml.poll_simulation(
                session,
                "https://api.worldquantbrain.com/simulations/1",
                deadline=time.monotonic() + 10,
            )
        self.assertEqual(result["status"], "PENDING")
        self.assertEqual(result["next_poll_after_seconds"], 7200)
        sleep.assert_not_called()
        self.assertEqual(len(session.get_calls), 1)

    def test_delayed_metrics_fetch_is_checkpointed_and_resumable(self):
        pairs = [("rank(cap)", 4)]
        checkpoint = [{
            "expression": "rank(cap)",
            "decay": 4,
            "alpha_id": None,
            "status": "PENDING",
            "progress_url": "https://api.worldquantbrain.com/simulations/1",
        }]
        first_session = self.Session([
            self.Response(payload={"status": "COMPLETE", "alpha": "A123"}),
            self.Response(headers={"Retry-After": "7200"}),
        ])
        with mock.patch.object(ml, "sleep") as sleep:
            pending = ml.simulate_alphas(
                first_session,
                pairs,
                max_wait_seconds=10,
                initial_records=checkpoint,
            )
        self.assertEqual(pending[0]["status"], "PENDING")
        self.assertTrue(pending[0]["metrics_pending"])
        self.assertNotIn("progress_url", pending[0])
        sleep.assert_not_called()

        pending[0]["next_poll_at"] = time.time() - 1
        resume_session = self.Session([
            self.Response(payload={"is": {"sharpe": 1.2}}),
        ])
        complete = ml.simulate_alphas(
            resume_session,
            pairs,
            max_wait_seconds=10,
            initial_records=pending,
        )
        self.assertEqual(complete[0]["status"], "COMPLETE")
        self.assertEqual(complete[0]["metrics"], {"sharpe": 1.2})
        self.assertEqual(resume_session.post_calls, [])

    def test_pending_checkpoint_resumes_without_resubmitting_accepted_job(self):
        session = self.Session([
            self.Response(payload={"status": "COMPLETE", "alpha": "A123"}),
        ])
        pairs = [("rank(cap)", 4)]
        checkpoint = [{
            "expression": "rank(cap)",
            "decay": 4,
            "alpha_id": None,
            "status": "PENDING",
            "progress_url": "https://api.worldquantbrain.com/simulations/1",
        }]
        with mock.patch.object(
            ml, "fetch_alpha_metrics", return_value={"is": {"sharpe": 1.2}}
        ):
            records = ml.simulate_alphas(
                session,
                pairs,
                max_wait_seconds=10,
                initial_records=checkpoint,
            )
        self.assertEqual(records[0]["status"], "COMPLETE")
        self.assertEqual(records[0]["alpha_id"], "A123")
        self.assertEqual(session.post_calls, [])

    def test_unresolved_job_prevents_duplicate_or_extra_batch_submissions(self):
        session = self.Session([self.Response(headers={"Retry-After": "7200"})])
        pairs = [("rank(cap)", 4), ("rank(volume)", 5)]
        checkpoint = [
            {
                "expression": expression,
                "decay": decay,
                "status": "PENDING" if index == 0 else "QUEUED",
                "progress_url": "https://api.worldquantbrain.com/simulations/1" if index == 0 else None,
            }
            for index, (expression, decay) in enumerate(pairs)
        ]
        with mock.patch.object(ml, "sleep") as sleep:
            records = ml.simulate_alphas(
                session,
                pairs,
                max_wait_seconds=10,
                initial_records=checkpoint,
            )
        self.assertEqual(records[0]["status"], "PENDING")
        self.assertEqual(records[1]["status"], "QUEUED")
        self.assertEqual(session.post_calls, [])
        sleep.assert_not_called()

    def test_cli_checkpoint_can_be_resumed_without_new_submission(self):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            os.chdir(temporary_directory)
            try:
                parser = cli.build_parser()
                simulate_args = parser.parse_args([
                    "simulate", "--expr", "rank(cap)", "--max-wait-seconds", "1",
                ])
                first_session = self.Session(
                    get_responses=[self.Response(headers={"Retry-After": "7200"})],
                    post_responses=[self.Response(
                        status_code=201,
                        headers={
                            "Location": "https://api.worldquantbrain.com/simulations/1"
                        },
                    )],
                )
                with mock.patch.object(
                    cli, "with_session", side_effect=lambda _args, work: work(first_session)
                ):
                    pending = cli.cmd_simulate(simulate_args, None)
                self.assertEqual(pending["status"], "PENDING")
                checkpoint_path = Path(pending["checkpoint"])
                checkpoint = json.loads(checkpoint_path.read_text())
                self.assertEqual(checkpoint["records"][0]["status"], "PENDING")
                checkpoint["records"][0]["next_poll_at"] = time.time() - 1
                cli._write_simulation_checkpoint(checkpoint_path, checkpoint)

                resume_args = parser.parse_args([
                    "resume", str(checkpoint_path), "--max-wait-seconds", "1",
                ])
                resume_session = self.Session([
                    self.Response(payload={"status": "COMPLETE", "alpha": "A123"}),
                ])
                with (
                    mock.patch.object(
                        cli, "with_session", side_effect=lambda _args, work: work(resume_session)
                    ),
                    mock.patch.object(
                        ml, "fetch_alpha_metrics", return_value={"is": {"sharpe": 1.2}}
                    ),
                ):
                    resumed = cli.cmd_resume(resume_args, None)
                self.assertEqual(resumed["status"], "COMPLETE")
                self.assertEqual(resumed["summary"]["completed"], 1)
                self.assertEqual(resume_session.post_calls, [])
            finally:
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()
