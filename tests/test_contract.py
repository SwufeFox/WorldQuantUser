#!/usr/bin/env python3
"""v4 Contract tests for the resident-daemon CLI toolchain.

Everything here runs offline without credentials or external network traffic.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import alpha_registry  # noqa: E402
import machine_lib as ml  # noqa: E402
import research_config as config_io  # noqa: E402
from research_app import app as cli_app  # noqa: E402
from research_app import description as cli_description  # noqa: E402
from research_app import parser as cli_parser  # noqa: E402
from research_app import runtime as cli_runtime  # noqa: E402


def run_cli(*args, input_text=None):
    """Run research_cli.py in a subprocess with BRAIN_* credentials stripped."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("BRAIN_")}
    return subprocess.run(
        [sys.executable, str(ROOT / "research_cli.py"), *args],
        capture_output=True, encoding="utf-8", cwd=str(ROOT), env=env, input=input_text,
    )


def invoke(*args):
    """Invoke CLI in-process, returning (exit_code, stdout, stderr)."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = cli_app.main(list(args))
    return code, stdout.getvalue(), stderr.getvalue()


class SchemaAndCliContractTests(unittest.TestCase):
    def test_schema_is_v4_single_json_document(self):
        proc = run_cli("schema")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"])
        data = payload["data"]
        self.assertEqual(data["version"], "4.0.0")
        for key in ("exit_codes", "error_codes", "research_file",
                    "settings_defaults", "credentials", "commands"):
            self.assertIn(key, data)
        self.assertEqual(data["exit_codes"]["CONFIRM_REQUIRED"], 2)
        self.assertEqual(data["exit_codes"]["ACCESS_DENIED"], 3)
        self.assertEqual(data["exit_codes"]["SESSION_BUSY"], 4)

    def test_v4_commands_present_in_schema_and_parser(self):
        parser = cli_parser.build_parser()
        v4_cmds = (
            ("propose",),
            ("inbox",),
            ("runs",),
            ("inspect", "run-1"),
            ("act", "run-1", "approve-full"),
            ("cancel", "run-1"),
            ("wait",),
            ("breed", "run-1", "--out", "offspring.json", "--seed", "1", "--limit", "2"),
            ("migrate-workspace",),
            ("daemon", "status"),
            ("daemon", "config"),
            ("shelf", "list"),
            ("shelf", "plan"),
        )
        schema_doc = json.loads(run_cli("schema").stdout)["data"]
        listed = set(schema_doc["commands"]["subcommands"])
        for name in ("propose", "inbox", "runs", "inspect", "act", "cancel",
                     "wait", "breed", "migrate-workspace", "daemon", "shelf"):
            self.assertIn(name, listed)

        for argv in v4_cmds:
            with self.subTest(argv=argv):
                cmd_schema = json.loads(run_cli("schema", *argv[:1]).stdout)
                self.assertTrue(cmd_schema["ok"])
                parsed = parser.parse_args(argv)
                self.assertEqual(parsed.command, argv[0])

    def test_removed_v3_commands_fail_cleanly_with_usage(self):
        for argv in (
            ("campaign", "status"),
            ("probe", "spec.json"),
            ("simulate", "rank(close)"),
            ("resume", "batch.json"),
            ("triage", "CAND1"),
            ("promote", "batch.json"),
            ("buffer", "list"),
            ("alpha", "submit", "A1"),
            ("alpha", "describe", "A1", "--name", "x"),
            ("alpha", "cancel-sim", "S1"),
        ):
            with self.subTest(argv=argv):
                code, stdout, _stderr = invoke(*argv)
                self.assertEqual(code, 2)
                err = json.loads(stdout)
                self.assertFalse(err["ok"])
                self.assertEqual(err["error"]["code"], "USAGE")

    def test_version_and_wait_aliases(self):
        proc = run_cli("--version")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("4.0.0", proc.stdout)

        parser = cli_parser.build_parser()
        for argv, dest, expected_default in (
            (("wait",), "max_wait", 30.0),
            (("alpha", "get", "SYNTHETIC"), "max_wait", 180.0),
            (("alpha", "check", "SYNTHETIC"), "max_wait", 180.0),
            (("ledger", "sync"), "max_wait_seconds", 300.0),
        ):
            parsed = parser.parse_args(argv)
            self.assertEqual(getattr(parsed, dest), expected_default)
            for flag in ("--max-wait", "--max-wait-seconds"):
                override = parser.parse_args((*argv, flag, "2.5"))
                self.assertEqual(getattr(override, dest), 2.5)

    def test_power_pool_description_headings_contract(self):
        text = (
            "Idea:\n- Higher quality firms should earn higher subsequent returns.\n\n"
            "Rationale for data used:\n- assets: total reported corporate assets.\n\n"
            "Rationale for operators used:\n- rank: compares firms cross-sectionally."
        )
        checked = cli_description.check(text)
        self.assertTrue(checked["required_ok"], checked)
        old = text.replace("Idea:", "IDEA:")
        self.assertFalse(cli_description.check(old)["template_ok"])

    def test_save_protection_covers_state_machine_and_ledger_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rel in ("runs/r1.json", "daemon/endpoint.json",
                        "alpha-registry.jsonl", "submit-buffer.jsonl", "seeds.jsonl"):
                code, stdout, _ = invoke("--research-dir", str(root), "inbox", "--save", str(root / rel))
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(stdout)["error"]["code"], "INVALID_INPUT")

    def test_workspace_override_compact_nonfinite_and_suggestions(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": str(ROOT / "unused")}):
            code, stdout, stderr = invoke("ledger", "list", "--research-dir", directory, "--compact")
            self.assertEqual(code, 0, stderr)
            self.assertEqual(len(stdout.strip().splitlines()), 1)
            self.assertEqual(json.loads(stdout)["data"]["registry"],
                             str(Path(directory) / "alpha-registry.jsonl"))

        code, stdout, _ = invoke("alpha", "list", "--min-sharpe", "nan")
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stdout)["error"]["code"], "USAGE")

        code, stdout, _ = invoke("help", "catalog", "fields")
        self.assertEqual(code, 0)
        self.assertIn("catalog fields", json.loads(stdout)["data"]["usage"])

        code, stdout, _ = invoke("propose", "spec.json", "--stdni")
        self.assertEqual(code, 2)
        self.assertIn("--stdin", json.loads(stdout)["error"]["details"]["suggestions"])


class ResearchConfigTests(unittest.TestCase):
    def test_research_file_formats(self):
        self.assertEqual(config_io.parse_research_text('["rank(cap)"]'), ({}, ["rank(cap)"]))
        settings, alphas = config_io.parse_research_text(json.dumps({"settings": {"decay": 3}, "alphas": ["rank(cap)"]}))
        self.assertEqual(settings["decay"], 3)
        self.assertEqual(alphas, ["rank(cap)"])
        _, plain = config_io.parse_research_text("# comment\n\nrank(cap)\n  rank(assets)  \n")
        self.assertEqual(plain, ["rank(cap)", "rank(assets)"])


if __name__ == "__main__":
    unittest.main()
