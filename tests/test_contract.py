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
import subprocess
import sys
import tempfile
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
            set(commands), {"schema", "auth", "catalog", "simulate", "alpha"},
        )
        simulate = commands["simulate"]
        flags = {flag for option in simulate["options"] for flag in option["flags"]}
        self.assertIn("--dry-run", flags)
        self.assertIn("--sort", flags)
        catalog = commands["catalog"]["subcommands"]
        self.assertIn("--dataset", {f for o in catalog["fields"]["options"] for f in o["flags"]})

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


if __name__ == "__main__":
    unittest.main()