#!/usr/bin/env python3
"""Contract tests for the research toolchain.

Everything here runs offline: no BRAIN account, no credentials, no network.
That is deliberate -- the whole point of the CLI contract is that a caller can
discover and validate behaviour without spending quota.

Run with::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
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
import field_index
from research_app import app as cli_app
from research_app import catalog as cli_catalog
from research_app import experiments as cli_experiments
from research_app import parser as cli_parser
from research_app import runtime as cli_runtime
from research_app import validation as cli_validation
import research_config as config_io  # noqa: E402


def run_cli(*args, expect_env=True, input_text=None):
    """Run the CLI in a subprocess with BRAIN_* credentials stripped."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("BRAIN_")}
    if not expect_env:
        env.pop("BRAIN_USERNAME", None)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "research_cli.py"), *args],
        capture_output=True, encoding="utf-8", cwd=str(ROOT), env=env, input=input_text,
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



    def test_schema_can_describe_one_command(self):
        payload = parse(run_cli("schema", "simulate"))["data"]
        self.assertEqual(payload["command"]["command"], "simulate")

    def test_schema_rejects_unknown_command(self):
        proc = run_cli("schema", "nope")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(parse(proc)["error"]["code"], "USAGE")


class BriefContractTests(unittest.TestCase):
    def test_nonpositive_limit_cannot_expand_the_startup_context(self):
        for limit in ("0", "-1"):
            with self.subTest(limit=limit):
                proc = run_cli("brief", "--top", limit)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(parse(proc)["error"]["code"], "INVALID_INPUT")


class TriageContractTests(unittest.TestCase):
    def test_validation_only_options_cannot_silently_change_the_selection_view(self):
        for extra in (["--internal"], ["--train-end", "not-a-date"],
                      ["--show-test", "--train-end", "2020-01-01"]):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                proc = run_cli("triage", "--ids", "ANY", "--research-dir", directory, *extra)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertEqual(parse(proc)["error"]["code"], "INVALID_INPUT")


class FeedContractTests(unittest.TestCase):
    def test_missing_topics_is_a_local_input_error_without_creating_state(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = run_cli("feed", "--research-dir", directory)
            self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
            self.assertEqual(parse(proc)["error"]["code"], "INVALID_INPUT")
            self.assertFalse((Path(directory) / "inspiration").exists())

    def test_concurrency_and_timeout_bounds_fail_before_network(self):
        for option, value in (("--workers", "0"), ("--workers", "9"), ("--timeout", "0"),
                              ("--timeout", "nan")):
            with self.subTest(option=option, value=value), tempfile.TemporaryDirectory() as directory:
                proc = run_cli("feed", "--topic", "asset pricing", option, value,
                               "--research-dir", directory)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertEqual(parse(proc)["error"]["code"], "USAGE")

    def test_explicit_thread_query_reaches_concurrent_feed_and_retry_guidance(self):
        from research_app import feed

        with tempfile.TemporaryDirectory() as directory:
            threads = Path(directory) / "open-threads.md"
            threads.write_text("- [open] **修订与惯性** — 证据；query: analyst revisions momentum\n", encoding="utf-8")
            args = cli_parser.build_parser().parse_args(["feed", "--from-threads", "--dry-run",
                                                        "--workers", "2", "--timeout", "1"])
            with mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": directory}), \
                    mock.patch("paper_search.search_papers", side_effect=TimeoutError):
                result = feed.cmd_feed(args, None)
        self.assertEqual(result["topics"], ["analyst revisions momentum"])
        self.assertEqual(result["searches"][0]["status"], "failure")


class CheckpointConcurrencyTests(unittest.TestCase):
    def test_concurrent_completion_events_preserve_every_record(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            checkpoint = {"records": [{"status": "QUEUED"} for _ in range(8)]}
            callback = cli_experiments._simulation_checkpoint_callback(path, checkpoint, None)
            start = threading.Barrier(8)

            def complete(index):
                start.wait(timeout=10)
                callback("completed", {"index": index, "status": "COMPLETE", "alpha_id": "A%d" % index})

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(complete, range(8)))
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual([row["alpha_id"] for row in saved["records"]], ["A%d" % i for i in range(8)])
        self.assertTrue(all(row["status"] == "COMPLETE" for row in saved["records"]))

    def test_competing_resume_refuses_before_reading_or_submitting(self):
        import cache_store

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            args = cli_parser.build_parser().parse_args(["resume", str(path)])
            with cache_store.file_lock(str(path) + ".lock"), \
                    mock.patch.object(ml, "load_session") as login, \
                    mock.patch.object(ml, "simulate_alphas") as simulate:
                with self.assertRaises(cli_runtime.CliError) as caught:
                    cli_experiments.cmd_resume(args, None)
            self.assertEqual(caught.exception.code, "SERVICE_ERROR")
            login.assert_not_called()
            simulate.assert_not_called()


class OptionalPandasTests(unittest.TestCase):
    fields = [
        {"id": "close", "type": "MATRIX"},
        {"id": "news_vector", "type": "VECTOR"},
        {"id": "sector", "type": "GROUP"},
    ]
    expressions = [
        "winsorize(ts_backfill(close, 120), std=4)",
        "vec_avg(news_vector)",
    ]

    def block_pandas(self, error):
        original_import = __import__

        def import_without_pandas(name, *args, **kwargs):
            if name == "pandas" or name.startswith("pandas."):
                raise error
            return original_import(name, *args, **kwargs)

        return mock.patch("builtins.__import__", side_effect=import_without_pandas)

    def test_cli_offline_commands_work_with_missing_or_broken_pandas(self):
        script = """import builtins, runpy, sys
original_import = builtins.__import__
error_type = getattr(builtins, sys.argv.pop(1))
def import_without_pandas(name, *args, **kwargs):
    if name == "pandas" or name.startswith("pandas."):
        raise error_type("unusable pandas installation")
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_pandas
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
        commands = [
            ("schema",),
            ("brief",),
            ("lint", "rank(close)", "--offline"),
            ("simulate", "--expr", "rank(close)", "--dry-run"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            env = {k: v for k, v in os.environ.items() if not k.startswith("BRAIN_")}
            env["BRAIN_RESEARCH_DIR"] = directory
            for error in ("ImportError", "ValueError"):
                for command in commands:
                    with self.subTest(error=error, command=command):
                        proc = subprocess.run(
                            [sys.executable, "-c", script, error,
                             str(ROOT / "research_cli.py"), *command],
                            capture_output=True, encoding="utf-8", cwd=str(ROOT), env=env,
                        )
                        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                        self.assertTrue(parse(proc)["ok"])

    def test_field_lists_work_with_broken_pandas(self):
        with self.block_pandas(ValueError("broken pandas installation")):
            self.assertEqual(ml.process_datafields(self.fields), self.expressions)

    def test_fields_to_df_converts_real_dataframe_and_processes_records(self):
        try:
            import pandas
        except ImportError:
            self.skipTest("pandas is optional")
        frame = ml.fields_to_df(self.fields)
        self.assertIsInstance(frame, pandas.DataFrame)
        self.assertEqual(ml.process_datafields(frame), self.expressions)

    def test_fields_to_df_requires_optional_dependency(self):
        with self.block_pandas(ImportError("pandas is absent")):
            with self.assertRaises(RuntimeError):
                ml.fields_to_df(self.fields)


class ErrorEnvelopeTests(unittest.TestCase):
    def test_unknown_command_still_prints_json_on_stdout(self):
        proc = run_cli("bogus")
        self.assertEqual(proc.returncode, 2)
        payload = parse(proc)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "USAGE")

    def test_command_without_subcommand_shows_local_choices(self):
        proc = run_cli("catalog")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        document = parse(proc)
        self.assertEqual(document["command"], "catalog")
        self.assertIn("fields", document["data"]["commands"])
        self.assertNotIn("ledger", document["data"]["commands"])

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

    def test_well_formed_unknown_neutralization_is_sent(self):
        # The platform owns this enum. Refusing a value locally made "we never
        # tried STATISTICAL" indistinguishable from "the tool cannot say it"
        # (2026-11-04). A well-formed unknown token now goes out and the
        # platform's own answer is the record.
        for value in ("STATISTICAL", "FAST", "SOME_FUTURE_SCHEME"):
            resolved = ml.resolve_settings({"neutralization": value})
            self.assertEqual(resolved["neutralization"], value)

    def test_malformed_neutralization_is_still_a_local_typo(self):
        for value in ("", "sub industry", "Statistical", "SUB-IND"):
            with self.assertRaises(ValueError):
                ml.resolve_settings({"neutralization": value})

    def test_every_error_code_has_an_exit_code(self):
        for code in cli_runtime.ERROR_CODES:
            self.assertIn(code, cli_runtime.EXIT_CODES, code)


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

    def test_dry_run_accepts_a_neutralization_outside_the_local_list(self):
        # The CLI must not be a second gate: one validation point, in
        # machine_lib.resolve_settings.
        proc = run_cli("simulate", "--expr", "rank(cap)",
                       "--neutralization", "STATISTICAL", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(parse(proc)["data"]["settings"]["neutralization"],
                         "STATISTICAL")

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


class CliWorkflowContractTests(unittest.TestCase):
    def invoke(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
            code = cli_app.main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_common_options_work_before_between_and_after_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, arguments in enumerate([
                ["--compact", "--quiet", "schema", "alpha", "get"],
                ["schema", "--compact", "alpha", "get", "--quiet"],
                ["schema", "alpha", "get", "--compact", "--quiet"],
            ]):
                target = Path(directory) / str(index) / "contract.json"
                with self.subTest(arguments=arguments):
                    code, stdout, stderr = self.invoke(*arguments, "--save", str(target))
                    self.assertEqual(code, 0, stderr)
                    self.assertEqual(stdout.count("\n"), 1)
                    self.assertEqual(target.read_bytes(), stdout.encode("utf-8"))
                    self.assertEqual(json.loads(stdout)["data"]["command"]["command"], "alpha get")
                    self.assertEqual(stderr, "")

    def test_saved_usage_failure_keeps_envelope_and_command_context(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "failure.json"
            code, stdout, stderr = self.invoke("--save", str(target), "catalog", "feilds", "--quiet")
            document = json.loads(stdout)
            self.assertEqual(code, 2)
            self.assertEqual(target.read_bytes(), stdout.encode("utf-8"))
            self.assertEqual(document["error"]["code"], "USAGE")
            self.assertIn("fields", document["error"]["details"]["suggestions"])
            self.assertIn("catalog", document["error"]["details"]["usage"])
            self.assertIn("USAGE", stderr)

    def test_invalid_save_destination_is_rejected_before_command_side_effect(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(ml, "invalidate_session") as logout:
            code, stdout, _stderr = self.invoke("auth", "logout", "--save", directory)
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stdout)["error"]["code"], "INVALID_INPUT")
            logout.assert_not_called()

    def test_workspace_override_is_local_to_each_invocation(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": str(ROOT / "unused-research")}):
            code, stdout, stderr = self.invoke("ledger", "list", "--research-dir", directory)
            self.assertEqual(code, 0, stderr)
            self.assertEqual(json.loads(stdout)["data"]["registry"],
                             str(Path(directory) / "alpha-registry.jsonl"))
            self.assertEqual(os.environ["BRAIN_RESEARCH_DIR"], str(ROOT / "unused-research"))

    def test_nonfinite_thresholds_fail_before_contacting_platform(self):
        with mock.patch.object(ml, "load_session") as login:
            code, stdout, _stderr = self.invoke("alpha", "list", "--min-sharpe", "nan")
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stdout)["error"]["code"], "USAGE")
            login.assert_not_called()

    def test_nested_help_and_misspelled_option_are_actionable(self):
        code, stdout, stderr = self.invoke("help", "catalog", "fields")
        self.assertEqual(code, 0, stderr)
        self.assertIn("catalog fields", json.loads(stdout)["data"]["usage"])
        code, stdout, _stderr = self.invoke("simulate", "--expr", "rank(close)", "--dry-rnu")
        self.assertEqual(code, 2)
        self.assertIn("--dry-run", json.loads(stdout)["error"]["details"]["suggestions"])



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

    def test_utf8_research_file_with_bom_preserves_expression(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "研究.json"
            path.write_text(json.dumps({"alphas": ["rank(close) /* 中文 */"]},
                                       ensure_ascii=False), encoding="utf-8-sig")
            _, alphas = config_io.load_research_file(path)
        self.assertEqual(alphas, ["rank(close) /* 中文 */"])

    def test_stdin_decodes_utf8_bom_without_losing_unicode(self):
        text = json.dumps({"settings": {"decay": 3},
                           "alphas": ["rank(close) /* 研究 */"]}, ensure_ascii=False)
        stream = io.BytesIO(("\ufeff" + text).encode("utf-8"))
        with mock.patch.object(sys, "stdin", stream):
            settings, alphas = config_io.load_research_file("-")
        self.assertEqual(settings, {"decay": 3})
        self.assertEqual(alphas, ["rank(close) /* 研究 */"])
        self.assertEqual(stream.tell(), len(stream.getvalue()))

    def test_stdin_rejects_non_utf8_bytes_as_research_input_error(self):
        with mock.patch.object(sys, "stdin", io.BytesIO(b"\xff")):
            with self.assertRaises(config_io.ResearchError) as caught:
                config_io.load_research_file("-")
        self.assertIn("cannot read -", str(caught.exception))

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


class ExpressionWorkflowTests(unittest.TestCase):
    def test_one_research_document_can_flow_through_all_offline_commands(self):
        expression = "a = rank(close); b = rank(volume); a + b"
        text = "\ufeff" + json.dumps({
            "settings": {"region": "EUR", "universe": "TOP600", "delay": 0, "decay": 3},
            "alphas": [{"expression": expression, "decay": 2}],
        })
        commands = [("lint", "--file", "-", "--offline"),
                    ("simulate", "-", "--dry-run"), ("ablate", "-")]
        for command in commands:
            with self.subTest(command=command):
                proc = run_cli(*command, input_text=text)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                data = parse(proc)["data"]
                self.assertEqual(data["settings"]["region"], "EUR")
                self.assertEqual(data["settings"]["universe"], "TOP600")
                self.assertEqual(data["settings"]["delay"], 0)
                if command[0] == "ablate":
                    self.assertEqual(data["source_expression"], expression)
                else:
                    self.assertEqual(data["results"][0]["expression"], expression)
                    self.assertEqual(data["results"][0]["decay"], 2)

    def test_lint_and_simulate_merge_repeated_files_and_expressions(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "first.json", Path(tmp) / "second.json"
            first.write_text(json.dumps({"settings": {"decay": 8, "universe": "TOP500"},
                                         "alphas": ["rank(close)"]}), encoding="utf-8")
            second.write_text(json.dumps({"settings": {"decay": 4},
                                          "alphas": [{"expression": "rank(volume)", "decay": 2}]}),
                              encoding="utf-8")
            for command in ("lint", "simulate"):
                with self.subTest(command=command):
                    args = [command]
                    args += (["--file", str(first)] if command == "lint" else [str(first)])
                    args += ["--file", str(second), "--expr", "rank(cap)",
                             "--expr", "rank(assets)", "--decay", "3"]
                    args += ["--offline"] if command == "lint" else ["--dry-run"]
                    proc = run_cli(*args)
                    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                    data = parse(proc)["data"]
                    self.assertEqual([r["expression"] for r in data["results"]],
                                     ["rank(close)", "rank(volume)", "rank(cap)", "rank(assets)"])
                    self.assertEqual([r["decay"] for r in data["results"]], [3, 2, 3, 3])
                    self.assertEqual(data["settings"]["universe"], "TOP500")
            proc = run_cli("simulate", "--file", str(first), "--file", str(second), "--dry-run")
            self.assertEqual(parse(proc)["data"]["settings"]["decay"], 4)

    def test_lint_positional_expression_is_merged_with_file_and_flags(self):
        proc = run_cli("lint", "rank(cap)", "--file", "-", "--expr", "rank(volume)",
                       "--offline", input_text="rank(close)")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual([r["expression"] for r in parse(proc)["data"]["results"]],
                         ["rank(close)", "rank(cap)", "rank(volume)"])

    def test_settings_flags_are_case_insensitive_and_emit_canonical_values(self):
        proc = run_cli("simulate", "--expr", "rank(close)", "--region", "eur",
                       "--universe", "top600", "--neutralization", "industry",
                       "--pasteurization", "off", "--nan-handling", "verify",
                       "--test-period", "p1y", "--instrument-type", "equity", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        settings = parse(proc)["data"]["results"][0]["payload"]["settings"]
        for key, expected in {"region": "EUR", "universe": "TOP600", "neutralization": "INDUSTRY",
                              "pasteurization": "OFF", "nanHandling": "VERIFY",
                              "testPeriod": "P1Y", "instrumentType": "EQUITY"}.items():
            self.assertEqual(settings[key], expected)

    def test_lint_catalog_uses_file_scope_and_explicit_overrides(self):
        document = {"settings": {"region": "EUR", "universe": "TOP600", "delay": 0},
                    "alphas": ["rank(scoped_field)"]}
        for flags, expected in [([], ("EUR", "TOP600", 0)),
                                (["--region", "usa", "--universe", "top500", "--delay", "1"],
                                 ("USA", "TOP500", 1))]:
            with self.subTest(flags=flags):
                def fields(_session, **scope):
                    actual = (scope["region"], scope["universe"], scope["delay"])
                    return ([{"id": "scoped_field", "type": "MATRIX", "dataset": "model1"}]
                            if actual == expected else [])

                args = cli_parser.build_parser().parse_args(["lint", "--file", "-", "--strict-fields", *flags])
                with mock.patch.object(sys, "stdin", io.StringIO(json.dumps(document))), \
                        mock.patch.object(ml, "load_session", return_value=object()), \
                        mock.patch.object(ml, "fetch_operators", return_value=[
                            {"name": "rank", "definition": "rank(x)"}]), \
                        mock.patch.object(field_index, "fetch_all_fields", side_effect=fields):
                    data = cli_validation.cmd_lint(args, None)
                self.assertTrue(data["ok"])
                self.assertEqual(data["known_field_count"], 1)
                self.assertEqual(tuple(data["settings"][key] for key in ("region", "universe", "delay")),
                                 expected)

    def test_catalog_and_lint_defaults_follow_machine_library(self):
        defaults = {**ml.DEFAULT_SETTINGS, "region": "EUR", "universe": "TOP600", "delay": 0}
        with mock.patch.object(ml, "DEFAULT_SETTINGS", defaults):
            parser = cli_parser.build_parser()
            args = parser.parse_args(["catalog", "datasets"])
            with mock.patch.object(ml, "load_session", return_value=object()), \
                    mock.patch.object(ml, "get_datasets", return_value=[{"id": "available"}]):
                catalog = cli_catalog.cmd_catalog_datasets(args, None)
            lint = cli_validation.cmd_lint(parser.parse_args(["lint", "rank(close)", "--offline"]), None)
        for key in ("region", "universe", "delay"):
            self.assertEqual(catalog[key], defaults[key])
            self.assertEqual(lint["settings"][key], defaults[key])

    def test_conflicting_ablation_inputs_are_rejected_before_reading_files(self):
        for args in [("ablate", "missing.json", "--expr", "a = rank(close); a"),
                     ("ablate", "missing.json", "--ids", "A1"),
                     ("ablate", "--expr", "a = rank(close); a", "--ids", "A1")]:
            with self.subTest(args=args):
                proc = run_cli(*args)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(parse(proc)["error"]["code"], "INVALID_INPUT")
                self.assertNotIn("cannot read", parse(proc)["error"]["message"])

    def test_multiple_stdin_inputs_are_rejected(self):
        for args in [("simulate", "-", "--file", "-", "--dry-run"),
                     ("lint", "--file", "-", "--file", "-", "--offline"),
                     ("compare", "-", "-")]:
            with self.subTest(args=args):
                proc = run_cli(*args, input_text="rank(close)")
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(parse(proc)["error"]["code"], "INVALID_INPUT")
                self.assertIn("only once", parse(proc)["error"]["message"])

    def test_compare_accepts_stdin_and_regular_files(self):
        first = {"alpha_id": "A", "out_of_sample": {"train": {"fitness": 2}}}
        second = {"alpha_id": "B", "out_of_sample": {"train": {"fitness": 1}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(json.dumps([second]), encoding="utf-8-sig")
            proc = run_cli("compare", "-", str(path),
                           input_text="\ufeff" + json.dumps({"ok": True, "data": {"results": [first]}}))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual([r["alpha_id"] for r in parse(proc)["data"]["rows"]], ["A", "B"])

    def test_malformed_stdin_is_invalid_input(self):
        for args in [("simulate", "-", "--dry-run"), ("lint", "--file", "-", "--offline"),
                     ("ablate", "-"), ("probe", "-", "--dry-run"), ("compare", "-")]:
            with self.subTest(args=args):
                proc = run_cli(*args, input_text="{invalid")
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(parse(proc)["error"]["code"], "INVALID_INPUT")


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

    def test_legacy_submission_check_returns_correlation_or_fail(self):
        for status, expected in (("PASS", 0.24), ("PENDING", 0.24), ("FAIL", "fail")):
            with self.subTest(status=status):
                session = mock.Mock()
                session.get.return_value.headers = {}
                session.get.return_value.json.return_value = {"is": {"checks": [
                    {"name": "SELF_CORRELATION", "value": 0.24, "result": "PASS"},
                    {"name": "LOW_SHARPE", "value": 1.5, "result": status},
                    {"name": "WAITING_CHECK"},
                ]}}
                self.assertEqual(ml.get_check_submission(session, "alpha-1"), expected)

    def test_legacy_submission_check_keeps_missing_correlation_value_as_nan(self):
        session = mock.Mock()
        session.get.return_value.headers = {}
        session.get.return_value.json.return_value = {"is": {"checks": [
            {"name": "SELF_CORRELATION", "result": "PENDING"},
            {"name": "LOW_SHARPE", "value": 1.5, "result": "PASS"},
        ]}}
        value = ml.get_check_submission(session, "alpha-1")
        self.assertIsInstance(value, float)
        self.assertNotEqual(value, value)  # NaN, as in the former DataFrame result.

    def test_legacy_submission_check_preserves_sleep_and_error_results(self):
        for payload, expected in (
                ({}, "sleep"),
                ({"is": {"checks": []}}, "error"),
                ({"is": {"checks": [{"name": "SELF_CORRELATION", "value": 0.24}]}},
                 "error"),
                ({"is": {"checks": [{"name": "SELF_CORRELATION"}]}},
                 "error")):
            with self.subTest(payload=payload):
                session = mock.Mock()
                session.get.return_value.headers = {}
                session.get.return_value.json.return_value = payload
                stdout = io.StringIO()
                with mock.patch.object(sys, "stdout", stdout):
                    self.assertEqual(ml.get_check_submission(session, "alpha-1"), expected)

    def test_legacy_submission_check_keeps_retry_after_polling(self):
        pending, complete = mock.Mock(), mock.Mock()
        pending.headers = ml.requests.structures.CaseInsensitiveDict({"Retry-After": "0.25"})
        complete.headers = {}
        complete.json.return_value = {"is": {"checks": [
            {"name": "SELF_CORRELATION", "value": 0.24, "result": "PASS"},
        ]}}
        session = mock.Mock()
        session.get.side_effect = [pending, complete]
        with mock.patch.object(ml.time, "sleep") as sleep:
            self.assertEqual(ml.get_check_submission(session, "alpha-1"), 0.24)
        sleep.assert_called_once_with(0.25)


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
        summary = cli_experiments.summarize(records)
        self.assertEqual(summary["total"], 4)
        # WARNING is a finished run: the platform writes every metric and only a check
        # warned, so it counts as completed and stays out of `other`.
        self.assertEqual(summary["completed"], 3)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["other"], 0)
        self.assertEqual(summary["best"]["alpha_id"], "B")

    def test_records_without_metrics_are_not_ranked(self):
        summary = cli_experiments.summarize([
            {"status": "COMPLETE", "alpha_id": "A", "expression": "x", "metrics": None},
        ])
        self.assertEqual(summary["ranking"], [])
        self.assertIsNone(summary["best"])

    def test_failed_record_with_progress_url_is_still_resumable(self):
        summary = cli_experiments.summarize([
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
            cli_experiments._write_simulation_checkpoint(path, {"records": []})
            self.assertEqual(json.loads(path.read_text()), {"records": []})
            if os.name == "posix":
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
                parser = cli_parser.build_parser()
                source = {"alphas": [{"expression": "rank(cap)", "candidate_id": "loo:l0"}],
                          "legs": {"l0": {"fields": ["cap"], "role": "value", "weight": 1}}}
                simulate_args = parser.parse_args([
                    "simulate", "-", "--max-wait-seconds", "1",
                    "--registry", str(Path(temporary_directory) / "registry.jsonl"),
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
                with mock.patch.object(sys, "stdin", io.StringIO("\ufeff" + json.dumps(source))), \
                        mock.patch.object(cli_experiments, "with_session", side_effect=lambda _args, work: work(first_session)):
                    pending = cli_experiments.cmd_simulate(simulate_args, None)
                self.assertEqual(pending["status"], "PENDING")
                checkpoint_path = Path(pending["checkpoint"])
                checkpoint = json.loads(checkpoint_path.read_text())
                self.assertEqual(checkpoint["records"][0]["status"], "PENDING")
                checkpoint["records"][0]["next_poll_at"] = time.time() - 1
                cli_experiments._write_simulation_checkpoint(checkpoint_path, checkpoint)

                resume_args = parser.parse_args([
                    "resume", str(checkpoint_path), "--max-wait-seconds", "1",
                ])
                resume_session = self.Session([
                    self.Response(payload={"status": "COMPLETE", "alpha": "A123"}),
                ])
                with (
                    mock.patch.object(cli_experiments, "with_session", side_effect=lambda _args, work: work(resume_session)),
                    mock.patch.object(
                        ml, "fetch_alpha_metrics", return_value={
                            "is": {"sharpe": 1.2, "fitness": 1.0}, "train": {"fitness": 1.0}}
                    ),
                ):
                    resumed = cli_experiments.cmd_resume(resume_args, None)
                self.assertEqual(resumed["status"], "COMPLETE")
                self.assertEqual(resumed["summary"]["completed"], 1)
                self.assertEqual(resume_session.post_calls, [])
                results_path = Path(temporary_directory) / "resumed-results.json"
                full = {"candidate_id": "full", "status": "COMPLETE",
                        "out_of_sample": {"train": {"fitness": 2.0}}}
                results_path.write_text(json.dumps({
                    **resumed, "results": [full, *resumed["results"]]}), encoding="utf-8")
                compare_args = parser.parse_args(["compare", str(results_path), "--per-file"])
                contribution = cli_experiments.cmd_compare(compare_args, None)["per_file"][0]["top_leg"]
                self.assertEqual(contribution["fields"], ["cap"])
                self.assertAlmostEqual(contribution["contribution"], 1.0)
            finally:
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()


class RegistryLabelContractTests(unittest.TestCase):
    """Results must be persisted by default and keep their mechanism labels."""


    def test_per_expression_mechanism_survives_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"alphas": [
                {"expression": "rank(close)", "decay": 4, "mechanism": "alpha-family"},
                {"expression": "rank(open)", "decay": 0, "mechanism": "beta-family"},
            ]}))
            proc = run_cli("simulate", str(path), "--dry-run")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            results = parse(proc)["data"]["results"]
            self.assertEqual([r["mechanism"] for r in results],
                             ["alpha-family", "beta-family"])
            self.assertEqual([r["decay"] for r in results], [4, 0])

    def test_labels_are_dropped_rather_than_misassigned(self):
        # One entry expanding into two expressions cannot be mapped 1:1, so a
        # wrong label would be worse than none.
        labels = ml.alpha_labels(
            [{"expression": "rank(close);rank(open)", "label": "x"}], 2
        )
        self.assertEqual(labels, [None, None])
        self.assertEqual(
            ml.alpha_labels([("rank(close)", 4)], 1), [None]
        )


class ConcurrentPollingTests(unittest.TestCase):
    """A slow arm must not hold the batch.

    A batch of three spent six minutes on one arm while the platform had already reported
    the other two COMPLETE. Serial polling makes the slowest arm set the pace for everything
    behind it, and an arm whose progress endpoint keeps answering with an empty body stops
    the batch entirely.
    """

    def test_three_pending_arms_poll_faster_than_three_round_trips(self):
        delay = 0.25
        checkpoint = [{
            "expression": "rank(cap)",
            "decay": 4,
            "alpha_id": None,
            "status": "PENDING",
            "progress_url": "https://api.worldquantbrain.com/simulations/%d" % index,
        } for index in range(3)]

        def slow_poll(session, progress_url, on_event=None, deadline=None):
            time.sleep(delay)
            return {"status": "COMPLETE", "alpha": progress_url.rsplit("/", 1)[-1],
                    "progress_url": progress_url}

        with mock.patch.object(ml, "poll_simulation", side_effect=slow_poll), \
                mock.patch.object(ml, "fetch_alpha_metrics", return_value={"is": {}}):
            started = time.monotonic()
            records = ml.simulate_alphas(
                object(), [("rank(cap)", 4), ("rank(cap)", 4), ("rank(cap)", 4)],
                max_wait_seconds=30, initial_records=checkpoint,
            )
            elapsed = time.monotonic() - started

        self.assertEqual([r["status"] for r in records], ["COMPLETE"] * 3)
        # Serial polling costs at least 3 x delay; concurrent polling costs about one.
        self.assertLess(elapsed, delay * 2.2,
                        "polling looks serial: %.3fs for three %.2fs polls" % (elapsed, delay))
