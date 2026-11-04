#!/usr/bin/env python3
"""research_cli.py -- machine-readable front end for WorldQuant BRAIN research.

Everything here exists to make one caller happy: a program (or an agent) that
wants to drive alpha research without reading this file.

The contract:

* ``stdout`` carries **exactly one JSON document** per invocation -- success or
  failure. Nothing else is ever written there, so ``json.loads(stdout)`` always
  works.
* Diagnostics, progress and streaming events go to ``stderr``.
* Failures are ``{"ok": false, "command": ..., "error": {"code", "message",
  "hint"}}`` and map onto a documented process exit code (see ``EXIT_CODES``).
* ``research_cli.py schema`` prints the whole contract -- commands, arguments,
  the research-file format, error codes -- derived from the live parser, so it
  cannot drift from the implementation.

Typical use::

    research_cli.py schema
    research_cli.py auth check
    research_cli.py catalog fields --dataset fundamental6 --type MATRIX --limit 20
    research_cli.py simulate --expr "rank(liabilities/assets)" --decay 4 --dry-run
    research_cli.py simulate research.json --sort sharpe --top 5
    research_cli.py alpha get P0Mq0wEM
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import machine_lib as ml  # noqa: E402
import research_config as config_io  # noqa: E402
import alpha_registry  # noqa: E402
import expression_lint  # noqa: E402
import expression_units  # noqa: E402
import field_contract  # noqa: E402
import field_index  # noqa: E402
import field_search  # noqa: E402
import research_paths  # noqa: E402
import cache_store  # noqa: E402
import deepseek_search  # noqa: E402
import paper_search  # noqa: E402
import correlation  # noqa: E402
import ablation  # noqa: E402

VERSION = "3.0.0"

#: Stable process exit codes. Callers branch on these, so the mapping from an
#: error code to an exit code is part of the contract.
EXIT_CODES = {
    "OK": 0,
    "USAGE": 2,
    "INVALID_INPUT": 2,
    "AUTH_MISSING": 3,
    "AUTH_FAILED": 3,
    "AUTH_CAPTCHA": 3,
    "RATE_LIMITED": 4,
    "SERVICE_ERROR": 4,
    "NOT_FOUND": 5,
    "SIMULATION_FAILED": 6,
    "SUBMIT_DISABLED": 7,
    "INTERRUPTED": 130,
}

ERROR_CODES = {
    "USAGE": "Bad command line; see `research_cli.py schema`.",
    "INVALID_INPUT": "A research file, expression or setting failed validation.",
    "AUTH_MISSING": "No BRAIN credentials configured (environment or .env).",
    "AUTH_FAILED": "BRAIN rejected the configured credentials.",
    "AUTH_CAPTCHA": "BRAIN wants a captcha before issuing a session; wait, then retry.",
    "NOT_FOUND": "An alpha id or endpoint does not exist for this account.",
    "RATE_LIMITED": "BRAIN rate limited us and the backoff budget ran out.",
    "SERVICE_ERROR": "Network or HTTP failure while talking to BRAIN.",
    "SIMULATION_FAILED": "Every submitted simulation failed.",
    "SUBMIT_DISABLED": "Real submission is disabled by the submission policy.",
    "INTERRUPTED": "Cancelled by signal.",
}

#: Metric fields copied out of ``is`` into every result record, in report order.
METRIC_KEYS = ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin",
               "longCount", "shortCount", "pnl")


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Readable examples together with command-specific defaults."""

    def _get_help_string(self, action):
        if action.default is None or action.default is False or action.default in ("", argparse.SUPPRESS, []):
            return action.help or ""
        return super()._get_help_string(action)


class JsonArgumentParser(argparse.ArgumentParser):
    """The same common options and contextual errors at every command level."""

    def __init__(self, *args, common=True, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        kwargs.setdefault("formatter_class", HelpFormatter)
        super().__init__(*args, **kwargs)
        self.set_defaults(_parser=self)
        if common:
            _add_common_args(self)

    def add_argument(self, *args, **kwargs):
        action = super().add_argument(*args, **kwargs)
        if action.help is None:
            action.help = action.dest.replace("_", " ").capitalize() + "."
        return action

    def parse_args(self, args=None, namespace=None):
        parsed, remaining = self.parse_known_args(args, namespace)
        if remaining:
            parsed._parser.error("unrecognized arguments: %s" % " ".join(remaining))
        return parsed

    def _check_value(self, action, value):
        if isinstance(action, argparse._SubParsersAction) and value not in action.choices:
            suggestions = difflib.get_close_matches(value, action.choices, n=3, cutoff=0.6)
            self.error("unknown command %r" % value, suggestions=suggestions)
        super()._check_value(action, value)

    def error(self, message, suggestions=None):
        if suggestions is None and message.startswith("unrecognized arguments:"):
            token = message.partition(":")[2].strip().split()[0]
            suggestions = difflib.get_close_matches(token, self._option_string_actions,
                                                   n=3, cutoff=0.6)
        path = self.prog.split()[1:]
        hint = "run `%s --help`" % self.prog
        if suggestions:
            hint = "did you mean %s? %s" % (", ".join(suggestions), hint)
        error = CliError("USAGE", message, hint,
                         {"command_path": path, "usage": self.format_usage().strip(),
                          "suggestions": suggestions or []})
        error.command = path[0] if path else "help"
        raise error


class CliError(Exception):
    """A failure we can explain: stable ``code``, message and optional hint."""

    def __init__(self, code, message, hint=None, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        # Optional structured payload so a caller can act on the failure without
        # re-deriving it from the message text.
        self.details = details

    @property
    def exit_code(self):
        return EXIT_CODES.get(self.code, 2)


# ---------------------------------------------------------------------------
# Output plumbing
# ---------------------------------------------------------------------------
def _error_body(exc):
    """Build the error object, including details when the raiser supplied them."""
    body = {"code": exc.code, "message": exc.message, "hint": exc.hint}
    details = getattr(exc, "details", None)
    if details is not None:
        body["details"] = details
    return body


def emit(payload, compact=False, prepared=None):
    """Serialize once, then save atomically and write exactly the same document."""
    text = json.dumps(payload, ensure_ascii=False, indent=None if compact else 2,
                      separators=(",", ":") if compact else None, default=str) + "\n"
    if prepared:
        target, temporary = prepared
        temporary.write_text(text, encoding="utf-8", newline="")
        os.replace(temporary, target)
    sys.stdout.write(text)
    sys.stdout.flush()
    return text


_QUIET = False


def note(message, *, force=False):
    """Human-facing diagnostics. Quiet mode suppresses progress, never errors."""
    if not _QUIET or force:
        sys.stderr.write(str(message) + "\n")
        sys.stderr.flush()


def make_event_sink(stream, verbose):
    """Build an ``on_event`` callback that emits JSON Lines to stderr."""
    if not stream:
        return None

    def sink(kind, payload):
        if kind == "polling" and not verbose:
            return
        record = {"event": kind}
        record.update(payload or {})
        note(json.dumps(record, ensure_ascii=False, default=str))

    return sink


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def _add_common_args(parser):
    group = parser.add_argument_group("Output and workspace (accepted at every level)")
    group.add_argument("--save", metavar="PATH", default=argparse.SUPPRESS,
                       help="Atomically save the same JSON document as stdout; '-' means stdout only.")
    group.add_argument("--compact", action="store_true", default=argparse.SUPPRESS,
                       help="Write JSON on one line.")
    group.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS,
                       help="Suppress progress on stderr; errors are still reported.")
    group.add_argument("--research-dir", metavar="PATH", default=argparse.SUPPRESS,
                       help="Use this research directory for this invocation, overriding BRAIN_RESEARCH_DIR.")


def _add_session_args(parser):
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore the cached session and re-authenticate.")
    parser.add_argument("--relogin", action="store_true",
                        help="Force a fresh login even if the cache looks valid.")
    parser.add_argument("--meta-max-age", type=_non_negative_seconds, default=None,
                        metavar="SECONDS",
                        help="Accept cached catalog metadata up to this age "
                             "(default 86400; 0 disables the catalog cache).")
    parser.add_argument("--refresh-meta", action="store_true",
                        help="Ignore cached catalog metadata and refetch it.")


def _non_negative_seconds(value):
    """Seconds for TTL-style options, where 0 is meaningful (means "disabled")."""
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number of seconds") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError(
            "must be a finite number of seconds, zero or greater"
        )
    return seconds


def _positive_seconds(value):
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number of seconds") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError(
            "must be a finite number of seconds greater than zero"
        )
    return seconds


def _add_settings_args(parser):
    parser.add_argument("--region", type=str.upper, choices=ml.REGIONS,
                        help="BRAIN region.")
    parser.add_argument("--universe", type=str.upper, help="Universe, e.g. TOP3000.")
    parser.add_argument("--instrument-type", type=str.upper,
                        help="BRAIN instrument type.")
    parser.add_argument("--neutralization", type=str.upper, choices=ml.NEUTRALIZATIONS,
                        help="Neutralization scheme.")
    parser.add_argument("--decay", type=int,
                        help="Default decay for alphas that do not set their own.")
    parser.add_argument("--delay", type=int, choices=[0, 1],
                        help="Data delay (0 or 1).")
    parser.add_argument("--truncation", type=float,
                        help="Truncation, e.g. 0.08.")
    parser.add_argument("--pasteurization", type=str.upper, choices=["ON", "OFF"])
    parser.add_argument("--nan-handling", type=str.upper, choices=["ON", "OFF", "VERIFY"])
    parser.add_argument("--test-period", type=str.upper, help="e.g. P0Y, P1Y, P2Y.")


def build_parser():
    parser = JsonArgumentParser(
        prog="research_cli.py",
        description="WorldQuant research: discover data, check expressions, run experiments, "
                    "and keep the evidence. Results are JSON; progress goes to stderr.",
    )
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", metavar="<command>",
                                parser_class=JsonArgumentParser)

    schema = sub.add_parser(
        "schema", help="Print the full call contract as JSON (commands, "
                       "arguments, research-file format, error codes).")
    schema.add_argument("name", nargs="*", metavar="COMMAND",
                        help="Restrict to a command path, e.g. catalog fields or alpha get.")
    schema.set_defaults(func=cmd_schema)

    help_command = sub.add_parser("help", help="Show a concise guide or help for a command path.")
    help_command.add_argument("path", nargs="*", metavar="COMMAND",
                              help="Command path, e.g. help catalog fields.")
    help_command.set_defaults(func=cmd_help)

    auth = sub.add_parser("auth", help="Credential and session checks.")
    auth_sub = auth.add_subparsers(dest="subcommand", metavar="<check>",
                                   parser_class=JsonArgumentParser)
    auth_check = auth_sub.add_parser(
        "check", help="Verify credentials and report account permissions.")
    _add_session_args(auth_check)
    auth_check.set_defaults(func=cmd_auth_check)

    auth_logout = auth_sub.add_parser(
        "logout", help="Forget the cached session cookie (local only).")
    auth_logout.set_defaults(func=cmd_auth_logout)

    cache = sub.add_parser(
        "cache", help="Inspect or clear local caches (session, catalog metadata).")
    cache_sub = cache.add_subparsers(dest="subcommand", metavar="<action>",
                                     parser_class=JsonArgumentParser)
    cache_info = cache_sub.add_parser(
        "info", help="Where caches live, what they hold, and how stale they are.")
    cache_info.set_defaults(func=cmd_cache_info)
    cache_clear = cache_sub.add_parser(
        "clear", help="Drop cached data. Locks are kept for running processes.")
    cache_clear.add_argument("--kind", choices=["session", "metadata", "all"],
                             default="all", help="What to clear (default: all).")
    cache_clear.set_defaults(func=cmd_cache_clear)

    catalog = sub.add_parser(
        "catalog", help="Quota-free metadata: operators, datasets, fields.")
    cat = catalog.add_subparsers(dest="subcommand", metavar="<kind>",
                                 parser_class=JsonArgumentParser)

    cat_op = cat.add_parser("operators", help="FASTEXPR operators.")
    cat_op.add_argument("--search", default="",
                        help="Ranked local search. Multi-word queries are AND; "
                             "'-word' excludes.")
    cat_op.add_argument("--fuzzy", action="store_true",
                        help="Allow one edit per token, for typos.")
    cat_op.add_argument("--any-term", action="store_true",
                        help="Match any query term instead of all of them.")
    cat_op.add_argument("--limit", type=int, default=None)
    cat_op.add_argument("--group", default="", help="Exact category, e.g. 'Time Series'.")
    cat_op.add_argument("--docs", nargs="*", metavar="NAME",
                        help="Also fetch full documentation for these operators.")

    _add_session_args(cat_op)
    cat_op.set_defaults(func=cmd_catalog_operators)

    cat_ds = cat.add_parser("datasets", help="Datasets for a region/universe/delay.")
    cat_ds.add_argument("--instrument-type", type=str.upper, default=ml.DEFAULT_SETTINGS["instrumentType"])
    cat_ds.add_argument("--region", type=str.upper, choices=ml.REGIONS, default=ml.DEFAULT_SETTINGS["region"])
    cat_ds.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_ds.add_argument("--delay", type=int, default=ml.DEFAULT_SETTINGS["delay"], choices=[0, 1])
    cat_ds.add_argument("--limit", type=int, default=None)

    _add_session_args(cat_ds)
    cat_ds.set_defaults(func=cmd_catalog_datasets)

    cat_f = cat.add_parser("fields", help="Datafields of one dataset.")
    cat_f.add_argument("--dataset", required=True, help="Dataset id, e.g. fundamental6.")
    cat_f.add_argument("--region", type=str.upper, choices=ml.REGIONS, default=ml.DEFAULT_SETTINGS["region"])
    cat_f.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_f.add_argument("--delay", type=int, default=ml.DEFAULT_SETTINGS["delay"], choices=[0, 1])
    cat_f.add_argument("--instrument-type", type=str.upper, default=ml.DEFAULT_SETTINGS["instrumentType"])
    cat_f.add_argument("--type", dest="field_type", type=str.upper, choices=["MATRIX", "VECTOR", "GROUP"],
                       help="Keep only this BRAIN field type.")
    cat_f.add_argument("--search", default="",
                       help="Ranked local search. Multi-word queries are AND; "
                            "prefix '-word' excludes; see --fuzzy and --any-term.")
    cat_f.add_argument("--fuzzy", action="store_true",
                       help="Allow one edit per token, for typos.")
    cat_f.add_argument("--any-term", action="store_true",
                       help="Match any query term instead of all of them.")
    cat_f.add_argument("--limit", type=int, default=None)
    cat_f.add_argument("--expressions", action="store_true",
                       help="Also emit ready-to-use FASTEXPR snippets.")

    _add_session_args(cat_f)
    cat_f.set_defaults(func=cmd_catalog_fields)

    cat_cc = cat.add_parser(
        "concepts",
        help="Group fields by concept and show variants side by side.")
    cat_cc.add_argument("--dataset", default=None,
                        help="Limit to one dataset id, e.g. analyst4.")
    cat_cc.add_argument("--region", type=str.upper, choices=ml.REGIONS, default=ml.DEFAULT_SETTINGS["region"])
    cat_cc.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_cc.add_argument("--delay", type=int, default=ml.DEFAULT_SETTINGS["delay"], choices=[0, 1])
    cat_cc.add_argument("--instrument-type", type=str.upper, default=ml.DEFAULT_SETTINGS["instrumentType"])
    cat_cc.add_argument("--type", dest="field_type", type=str.upper,
                        choices=["MATRIX", "VECTOR", "GROUP"],
                        help="Keep only this BRAIN field type.")
    cat_cc.add_argument("--tail-tokens", type=int, default=2,
                        help="Compare only the trailing N tokens, i.e. treat leading "
                             "words as family boilerplate (0 = use every token).")
    cat_cc.add_argument("--drop-numeric", action="store_true",
                        help="Ignore purely numeric tokens (e.g. vintage suffixes). "
                             "A mechanical rule, not catalog knowledge; off by default.")
    cat_cc.add_argument("--ignore-words", nargs="*", default=[],
                        help="Tokens to ignore when grouping, e.g. dataset family "
                             "words. No vocabulary is built in; supply it here.")
    cat_cc.add_argument("--search", default="",
                        help="Ranked search over concept tokens and member ids.")
    cat_cc.add_argument("--fuzzy", action="store_true",
                        help="Allow one edit per token, for typos.")
    cat_cc.add_argument("--any-term", action="store_true",
                        help="Match any query term instead of all of them.")
    cat_cc.add_argument("--min-members", type=int, default=2,
                        help="Only concepts with at least this many variants.")
    cat_cc.add_argument("--min-spread", type=float, default=0.0,
                        help="Only concepts whose coverage spread is at least this.")
    cat_cc.add_argument("--limit", type=int, default=40)

    _add_session_args(cat_cc)
    cat_cc.set_defaults(func=cmd_catalog_concepts)

    cat_vf = cat.add_parser(
        "verify",
        help="Check field ids against the live catalog (catches typed-from-memory names).")
    cat_vf.add_argument("--fields", nargs="+", default=[],
                        help="Field ids, comma or space separated.")
    cat_vf.add_argument("--region", type=str.upper, choices=ml.REGIONS, default=ml.DEFAULT_SETTINGS["region"])
    cat_vf.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_vf.add_argument("--delay", type=int, default=ml.DEFAULT_SETTINGS["delay"], choices=[0, 1])
    cat_vf.add_argument("--instrument-type", type=str.upper, default=ml.DEFAULT_SETTINGS["instrumentType"])
    cat_vf.add_argument("--strict", action="store_true",
                        help="Exit with INVALID_INPUT when any name is unknown.")

    _add_session_args(cat_vf)
    cat_vf.set_defaults(func=cmd_catalog_verify)

    cat_c = cat.add_parser("categories", help="Data categories with crowding stats.")
    cat_c.add_argument("--region", type=str.upper, choices=ml.REGIONS, default=ml.DEFAULT_SETTINGS["region"])
    cat_c.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_c.add_argument("--delay", type=int, default=ml.DEFAULT_SETTINGS["delay"], choices=[0, 1])

    _add_session_args(cat_c)
    cat_c.set_defaults(func=cmd_catalog_categories)

    cat_audit = cat.add_parser(
        "field-audit",
        help="Audit a field set's period/unit/sign contract before combining.")
    cat_audit.add_argument("--fields", default="",
                           help="Comma-separated datafield ids to audit.")
    cat_audit.add_argument("--dataset", default=None,
                           help="Audit a whole dataset, or resolve --fields "
                                "inside it first.")
    cat_audit.add_argument("--operation", choices=list(field_contract.OPERATIONS),
                           default=None,
                           help="The combination you intend to run; decides the "
                                "verdict.")
    cat_audit.add_argument("--search", default="")
    cat_audit.add_argument("--limit", type=int, default=None)
    cat_audit.add_argument("--region", type=str.upper, choices=ml.REGIONS, default=ml.DEFAULT_SETTINGS["region"])
    cat_audit.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_audit.add_argument("--delay", type=int, default=ml.DEFAULT_SETTINGS["delay"], choices=[0, 1])
    cat_audit.add_argument("--instrument-type", type=str.upper, default=ml.DEFAULT_SETTINGS["instrumentType"])

    _add_session_args(cat_audit)
    cat_audit.set_defaults(func=cmd_catalog_field_audit)

    lint = sub.add_parser(
        "lint",
        help="Pre-flight expressions against the operator catalog (quota-free).")
    lint.add_argument("expression", nargs="?",
                      help="Expression to lint, e.g. 'rank(ts_delta(close, 5))'.")
    lint.add_argument("--file", dest="files", action="append", default=[],
                      help="Research file or - for stdin; repeat to merge expressions. "
                           "Later file settings override earlier files.")
    lint.add_argument("--expr", action="append", default=[], metavar="EXPRESSION",
                      help="Inline expression; repeat and combine with files or the positional expression.")
    lint.add_argument("--offline", action="store_true",
                      help="Skip the operator catalog; check structure and the "
                           "learned denylist only.")
    lint.add_argument("--dataset", default=None,
                      help="Treat this dataset's fields as known so unknown "
                           "identifiers are flagged.")
    lint.add_argument("--strict-fields", action="store_true",
                      help="Treat unknown identifiers as errors, not warnings.")
    _add_settings_args(lint)
    lint.add_argument("--denylist", default=None,
                      help="Override the learned denylist path.")

    _add_session_args(lint)
    lint.set_defaults(func=cmd_lint)

    lit = sub.add_parser(
        "search",
        help="Literature search and page fetch via the DeepSeek Search API.")
    lit.add_argument("query", nargs="?",
                     help="Search query. Supports site:, \"exact phrase\", "
                          "lang:, after:, before: and -exclude.")
    lit.add_argument("--fetch", dest="fetch_url", default=None,
                     help="Fetch one URL as Markdown instead of searching.")
    lit.add_argument("--count", type=int, default=10,
                     help="Number of results, 1-50 (default 10).")
    lit.add_argument("--content", action="store_true",
                     help="Ask the API to include page content in results.")
    lit.add_argument("--min-score", type=float, default=0.0,
                     help="Drop results below this rerank_score (1.5+ is high quality).")
    lit.add_argument("--min-authority", type=int, default=None,
                     help="Drop results below this authority level (3+ is authoritative).")
    lit.add_argument("--max-age-hours", type=float, default=168.0,
                     help="Reuse a cached answer younger than this (default 168).")
    lit.add_argument("--no-cache", action="store_true",
                     help="Ignore and do not write the cache.")

    lit.set_defaults(func=cmd_search)

    papers = sub.add_parser(
        "paper",
        help="Find papers via OpenAlex: journal, DOI, citations and abstract.")
    papers.add_argument("query")
    papers.add_argument("--limit", type=int, default=10,
                        help="Number of papers, 1-50 (default 10).")
    papers.add_argument("--from-year", type=int, default=None,
                        help="Only papers published in or after this year.")
    papers.add_argument("--min-citations", type=int, default=None,
                        help="Drop papers with fewer citations than this.")

    papers.set_defaults(func=cmd_paper)

    sim = sub.add_parser(
        "simulate", help="Simulate alphas (spends simulation quota unless --dry-run).")
    sim.add_argument("file", nargs="?",
                     help="Research file: JSON, or one expression per line. "
                          "Omit when using --expr.")
    sim.add_argument("--expr", action="append", default=[], metavar="EXPRESSION",
                     help="Inline expression; repeat for several.")
    sim.add_argument("--file", dest="files", action="append", default=[],
                     help="Additional research file or - for stdin; repeat to merge expressions. "
                          "Later file settings override earlier files.")
    _add_settings_args(sim)
    sim.add_argument("--batch-size", type=int, default=3,
                     help="Concurrent submissions per batch (BRAIN allows 3).")
    sim.add_argument("--max-wait-seconds", type=_positive_seconds, default=300.0,
                     help="Maximum wall time per invocation; longer Retry-After waits "
                          "are saved as pending work (default: 300).")
    sim.add_argument("--checkpoint", default=None,
                     help="Checkpoint path; pending work is saved automatically if omitted.")
    sim.add_argument("--dry-run", action="store_true",
                     help="Validate and print payloads without contacting BRAIN.")
    sim.add_argument("--record", dest="record", action="store_true", default=True,
                     help="Append every finished result to the alpha registry "
                          "(outside the git work tree) with an automatic verdict. "
                          "On by default so results are never lost when a caller "
                          "forgets the flag.")
    sim.add_argument("--no-record", dest="record", action="store_false",
                     help="Skip writing results to the alpha registry.")
    sim.add_argument("--idea", default=None,
                     help="Idea label recorded in the registry, e.g. 'Idea 47'.")
    sim.add_argument("--mechanism", default=None,
                     help="One-line economic mechanism for the registry.")
    sim.add_argument("--note", default=None,
                     help="Free-text note stored with each registry entry.")
    sim.add_argument("--registry", default=None,
                     help="Override the alpha registry path.")
    sim.add_argument("--allow-lint-errors", action="store_true",
                     help="Submit even when the operator lint fails (the previous "
                          "behaviour). Not recommended: the failure costs quota.")
    sim.add_argument("--sort", choices=["none", *METRIC_KEYS], default="none",
                     help="Sort results by this metric, best first.")
    sim.add_argument("--top", type=int, default=None,
                     help="Keep only the first N results after sorting.")
    sim.add_argument("--stream", action="store_true",
                     help="Emit JSON Lines progress events on stderr.")
    sim.add_argument("--verbose", action="store_true",
                     help="With --stream, also emit per-poll events.")
    _add_session_args(sim)
    sim.set_defaults(func=cmd_simulate)

    resume = sub.add_parser(
        "resume", help="Resume pending/queued simulations from a checkpoint.")
    resume.add_argument("checkpoint", help="Checkpoint JSON produced by simulate/resume.")
    resume.add_argument("--max-wait-seconds", type=_positive_seconds, default=300.0,
                        help="Maximum wall time for this resume invocation (default: 300).")
    resume.add_argument("--sort", choices=["none", *METRIC_KEYS], default="none",
                        help="Sort results by this metric, best first.")
    resume.add_argument("--top", type=int, default=None,
                        help="Keep only the first N results after sorting.")
    resume.add_argument("--stream", action="store_true",
                        help="Emit JSON Lines progress events on stderr.")
    resume.add_argument("--verbose", action="store_true",
                        help="With --stream, also emit per-poll events.")
    _add_session_args(resume)
    resume.set_defaults(func=cmd_resume)

    abl = sub.add_parser(
        "ablate",
        help="Generate leave-one-out and solo variants of a composite (no quota).",
        description="Answer 'does this leg earn its place?' with one batch "
                    "instead of hand-writing each variant. Writes a research "
                    "file; run it with `simulate`, then read it back with "
                    "`compare --contributions`.")
    abl.add_argument("file", nargs="?",
                     help="Research file (JSON) holding the composite. Omit with --expr.")
    abl.add_argument("--expr", default=None,
                     help="Composite expression: 'a = ...; b = ...; a + b'.")
    abl.add_argument("--ids", default=None,
                     help="Comma-separated ledger ids: write one ablation file per "
                          "alpha instead of one for --expr/--file. Expressions that "
                          "are not a sum of named legs are reported, not guessed at.")
    abl.add_argument("--out-dir", default=None,
                     help="With --ids, write <id>.json here (default: the research dir).")
    abl.add_argument("--modes", default="full,loo,solo",
                     help="Comma list from full, loo, solo (default all three).")
    _add_settings_args(abl)
    abl.add_argument("--out", default=None,
                     help="Write the research file here instead of only reporting it.")

    abl.set_defaults(func=cmd_ablate)

    cmp_ = sub.add_parser(
        "compare",
        help="Compare simulation results side by side (no quota).",
        description="Read one or more simulate result documents and print a "
                    "single table, so variants from different batches stop being "
                    "aligned by hand.")
    cmp_.add_argument("files", nargs="+", help="Result JSON files from simulate.")
    cmp_.add_argument("--min-fitness", type=float, default=None)
    cmp_.add_argument("--sort", default="fitness",
                      choices=["fitness", "sharpe", "turnover", "returns", "id"])
    cmp_.add_argument("--frozen", action="store_true",
                      help="Show full-window and test-period columns. Only for a "
                           "design that is already frozen: the full window contains "
                           "the test year.")
    cmp_.add_argument("--contributions", action="store_true",
                      help="Also compute per-leg contributions from ablate labels.")

    cmp_.set_defaults(func=cmd_compare)

    brief = sub.add_parser(
        "brief",
        help="One bounded document describing where the research stands (no quota).",
        description="Book composition, what ran in the last N days and how it scored, "
                    "which write-ups are new, and the standing open questions from "
                    "open-threads.md. Built to be read at the start of a session "
                    "instead of the ledger, the result directory and last week's note.")
    brief.add_argument("--days", type=float, default=7.0,
                       help="How far back 'recent' reaches (default 7).")
    brief.add_argument("--top", type=int, default=5,
                       help="How many best runs and write-ups to list (default 5).")
    brief.add_argument("--registry", default=None)

    brief.set_defaults(func=cmd_brief)

    probe = sub.add_parser(
        "probe",
        help="Run a batch of arms from a spec file (lint, simulate, table, record).",
        description="One command for the experiment shape used every day here: a list "
                    "of labelled expressions, one lint pass, one simulation run, a "
                    "train-only table on stderr, a JSON document on stdout, and a "
                    "ledger entry per arm. The table hides the test period unless "
                    "--show-test is passed, because selecting on it destroys the only "
                    "out-of-sample evidence there is.")
    probe.add_argument("spec", help="JSON spec: {name, settings, arms:[{label, expression, settings?}]}.")
    _add_settings_args(probe)
    _add_session_args(probe)
    probe.add_argument("--name", default=None,
                       help="Batch name for the output file (default: the spec's name).")
    probe.add_argument("--dry-run", action="store_true",
                       help="Offline validation and exact request preview; no login or quota spent.")
    probe.add_argument("--show-test", action="store_true",
                       help="Also print the test period. Do not pass this while "
                            "choosing between arms.")
    probe.add_argument("--force", action="store_true",
                       help="Overwrite an existing batch file of the same name.")
    probe.add_argument("--no-record", action="store_true",
                       help="Do not append the arms to the ledger.")
    probe.add_argument("--registry", default=None)

    probe.set_defaults(func=cmd_probe)

    views = sub.add_parser(
        "views",
        help="Who looked at test-period numbers, when, and for which alphas.",
        description="Reads the append-only views log that `probe --show-test` and "
                    "`compare --frozen` write. Hiding a column does not stop the leak; "
                    "recording when it was opened does.")
    views.add_argument("--alpha", default=None, help="Only entries naming this alpha id.")
    views.add_argument("--limit", type=int, default=50)

    views.set_defaults(func=cmd_views)

    alpha = sub.add_parser("alpha", help="Inspect the account's alphas (quota-free).")
    alpha_sub = alpha.add_subparsers(dest="subcommand", metavar="<action>",
                                     parser_class=JsonArgumentParser)

    a_get = alpha_sub.add_parser("get", help="Fetch one alpha's metrics and checks.")
    a_get.add_argument("alpha_id")
    _add_session_args(a_get)

    a_get.set_defaults(func=cmd_alpha_get)

    a_chk = alpha_sub.add_parser(
        "check",
        help="Compute live checks (incl. SELF_CORRELATION). 'get' only shows cached values.")
    a_chk.add_argument("alpha_id")
    a_chk.add_argument("--max-wait", type=float, default=180.0,
                       help="Seconds to wait for the async check job (default 180).")
    _add_session_args(a_chk)

    a_chk.set_defaults(func=cmd_alpha_check)

    a_sc = alpha_sub.add_parser(
        "self-corr",
        help="The platform's own self-correlation, with the counterpart named.",
        description="A bare maximum says a submission will be refused; naming the "
                    "counterpart says whether the substitute route - Sharpe above the "
                    "margin of every correlated alpha - is arithmetically available.")
    a_sc.add_argument("alpha_id")
    a_sc.add_argument("--limit", type=float, default=0.7)
    a_sc.add_argument("--sharpe-margin", type=float, default=1.10)

    _add_session_args(a_sc)
    a_sc.set_defaults(func=cmd_alpha_self_corr)

    a_desc = alpha_sub.add_parser(
        "describe",
        help="Set an alpha's name, category and tags (the only writable metadata).",
        description="The platform accepts name, category and tags on an alpha. It has "
                    "no description property: eighteen candidate spellings all return "
                    "'Unexpected property', so nothing is invented to hold one.")
    a_desc.add_argument("alpha_id")
    a_desc.add_argument("--name", default=None)
    a_desc.add_argument("--category", default=None, choices=list(ml.ALPHA_CATEGORIES))
    a_desc.add_argument("--tags", default=None,
                        help="Comma-separated tags.")
    a_desc.add_argument("--description", default=None,
                        help="Free-text note. Writable, but nested under `regular` in "
                             "the payload rather than as a top-level property.")

    _add_session_args(a_desc)
    a_desc.set_defaults(func=cmd_alpha_describe)

    a_sub = alpha_sub.add_parser(
        "submit",
        help="Submit one alpha to its competition (dry-run unless --yes).")
    a_sub.add_argument("alpha_id")
    a_sub.add_argument("--yes", action="store_true",
                       help="Actually submit. Without it nothing is written.")
    a_sub.add_argument("--max-wait", type=float, default=180.0)
    a_sub.add_argument("--record", action="store_true",
                       help="after a confirmed submission, read the alpha back and "
                            "append it to the ledger (normalised, not raw)")
    _add_session_args(a_sub)

    a_sub.set_defaults(func=cmd_alpha_submit)

    a_expl = alpha_sub.add_parser(
        "explain",
        help="Explain one alpha: what each leg computes, and what the evidence says.")
    a_expl.add_argument("alpha_id")
    a_expl.add_argument("--no-corr", action="store_true",
                        help="skip the local correlation against the book (reads "
                             "<research>/pnl, which is a few hundred small files)")
    _add_session_args(a_expl)

    a_expl.set_defaults(func=cmd_alpha_explain)

    a_list = alpha_sub.add_parser("list", help="List the account's alphas.")
    a_list.add_argument("--limit", type=int, default=20)
    a_list.add_argument("--offset", type=int, default=0)
    a_list.add_argument("--order", default="-is.sharpe",
                        help="API sort syntax, e.g. -is.sharpe or is.fitness.")
    a_list.add_argument("--status", default="UNSUBMITTED",
                        help="UNSUBMITTED, SUBMITTED, or '' for any.")
    a_list.add_argument("--region")
    a_list.add_argument("--universe")
    a_list.add_argument("--min-sharpe", type=float)
    a_list.add_argument("--min-fitness", type=float)
    a_list.add_argument("--search")
    a_list.add_argument("--all", action="store_true",
                        help="Include hidden and SUPER alphas.")
    _add_session_args(a_list)

    a_list.set_defaults(func=cmd_alpha_list)

    ledger = sub.add_parser(
        "ledger", help="Record and explore the durable local research ledger.",
        description="List, label and relate recorded research offline. Only record "
                    "contacts BRAIN: it logs in and fetches an alpha's current metrics "
                    "before appending them; it never runs a simulation.")
    ledger_sub = ledger.add_subparsers(dest="subcommand", metavar="<action>",
                                        parser_class=JsonArgumentParser)

    l_rec = ledger_sub.add_parser(
        "record", help="Fetch one alpha from BRAIN and append it to the ledger (online).",
        description="Requires a BRAIN session and an existing alpha id. Fetches "
                    "current metrics without spending simulation quota. To change "
                    "judgement labels on a stored entry offline, use ledger label.")
    l_rec.add_argument("alpha_id")
    l_rec.add_argument("--idea", default=None, help="e.g. 'Idea 47'.")
    l_rec.add_argument("--mechanism", default=None)
    l_rec.add_argument("--verdict", choices=list(alpha_registry.VERDICTS),
                       default="watch")
    l_rec.add_argument("--note", default=None)
    l_rec.add_argument("--family", default=None, choices=list(alpha_registry.FAMILIES),
                       help="Return-source grouping; declared, never inferred.")
    l_rec.add_argument("--registry", default=None)
    _add_session_args(l_rec)
    l_rec.set_defaults(func=cmd_ledger_record)

    l_list = ledger_sub.add_parser(
        "list", help="List or search the durable ledger (offline).",
        description="Without filters, list every current ledger entry. Combine "
                    "filters to check prior field or mechanism research before "
                    "spending a simulation. Reads the latest record per key, "
                    "including submitted book alphas. No login or network.")
    l_list.add_argument("--verdict", action="append", default=[],
                        choices=list(alpha_registry.VERDICTS),
                        help="Keep any of these verdicts; repeatable.")
    l_list.add_argument("--idea", default=None,
                        help="Case-insensitive substring of the idea label.")
    l_list.add_argument("--min-sharpe", type=float, default=None)
    l_list.add_argument("--min-fitness", type=float, default=None)
    l_list.add_argument("--family", default=None,
                        choices=list(alpha_registry.FAMILIES),
                        help="Return-source grouping declared at record time.")
    l_list.add_argument("--field", default=None,
                        help="Datafield id substring inside recorded expressions.")
    l_list.add_argument("--mechanism", default=None,
                        help="Case-insensitive substring of the mechanism label.")
    l_list.add_argument("--expression", default=None,
                        help="Case-insensitive substring of a recorded expression.")
    l_list.add_argument("--limit", type=int, default=0,
                        help="Maximum entries to show, ranked by fitness; 0 shows all (default).")
    l_list.add_argument("--format", choices=["json", "markdown"], default="json",
                        help="Markdown adds a table inside the JSON envelope.")
    l_list.add_argument("--registry", default=None)
    l_list.set_defaults(func=cmd_ledger_list)

    l_lbl = ledger_sub.add_parser(
        "label",
        help="Set family / verdict / mechanism on an existing ledger entry (offline).",
        description="Fill in the judgement columns without editing JSONL by hand. "
                    "A new record is appended under the same key, so history is kept.")
    l_lbl.add_argument("alpha_id")
    l_lbl.add_argument("--family", default=None, choices=list(alpha_registry.FAMILIES))
    l_lbl.add_argument("--verdict", default=None, choices=list(alpha_registry.VERDICTS))
    l_lbl.add_argument("--mechanism", default=None)
    l_lbl.add_argument("--idea", default=None)
    l_lbl.add_argument("--note", default=None)
    l_lbl.add_argument("--registry", default=None)
    l_lbl.set_defaults(func=cmd_ledger_label)

    l_imp = ledger_sub.add_parser(
        "impact",
        help="Which other alphas stand on the same legs (offline).",
        description="Leg names are positional, so exposure is matched on the leg "
                    "body: the same sub-expression, not the same letter.")
    l_imp.add_argument("alpha_id")
    l_imp.add_argument("--min-shared", type=int, default=1)
    l_imp.add_argument("--limit", type=int, default=25)
    l_imp.add_argument("--registry", default=None)
    l_imp.set_defaults(func=cmd_ledger_impact)

    l_lin = ledger_sub.add_parser(
        "lineage",
        help="Walk the parent edges of an alpha (offline).",
        description="Shows where a variant came from and by which operation.")
    l_lin.add_argument("alpha_id")
    l_lin.add_argument("--registry", default=None)
    l_lin.set_defaults(func=cmd_ledger_lineage)

    l_cov = ledger_sub.add_parser(
        "coverage",
        help="How much of the ledger still lacks labels (offline).",
        description="Independence is counted on the family column; this reports "
                    "how far that column is from being usable.")
    l_cov.add_argument("--registry", default=None)
    l_cov.add_argument("--list-missing", action="store_true",
                       help="Also list keys missing a family, most recent first.")
    l_cov.add_argument("--limit", type=int, default=40)
    l_cov.set_defaults(func=cmd_ledger_coverage)

    a_self = alpha_sub.add_parser(
        "self-check",
        help="Reproduce the platform's self-correlation check locally (quota-free).",
        description="Daily Pearson against the book, calibrated against measured platform "
                    "values, with --as-of for reproducing a historical check and recent "
                    "submissions included because the platform's snapshot lags.")
    a_self.add_argument("alpha_id")
    a_self.add_argument("--as-of", default=None,
                        help="Only compare against peers submitted at or before this ISO "
                             "timestamp, which is how a historical platform value is "
                             "reproduced.")
    a_self.add_argument("--include-unsent-hours", type=float, default=24.0,
                        help="Also compare against UNSUBMITTED alphas created in the last "
                             "N hours, which the platform's snapshot misses (default 24; "
                             "0 disables).")
    a_self.add_argument("--threshold", type=float, default=0.7,
                        help="The platform's limit (default 0.7).")
    a_self.add_argument("--platform-bias", type=float, default=0.015,
                        help="How far above daily Pearson the platform reads on the same "
                             "book (default 0.015, from two same-moment comparisons).")
    a_self.add_argument("--top", type=int, default=3,
                        help="How many counterparts to list (default 3).")
    a_self.add_argument("--min-overlap", type=int, default=200,
                        help="Minimum shared trading days (default 200).")
    a_self.add_argument("--peer-limit", type=int, default=300,
                        help="How many peers to scan per status (default 300).")
    a_self.add_argument("--max-age-hours", type=float, default=168.0,
                        help="Trust a cached PnL series younger than this (default 168).")
    _add_session_args(a_self)
    a_self.set_defaults(func=cmd_alpha_self_check)

    a_corr = alpha_sub.add_parser(
        "corr",
        help="Daily-PnL correlation between alphas (quota-free).",
        description="Correlation on daily PnL, optionally against the submitted "
                    "book and optionally after removing a shared carrier. The "
                    "platform's SELF_CORRELATION check reports one opaque number "
                    "and never names the counterpart.")
    a_corr.add_argument("alpha_id", nargs="+",
                        help="Two or more alpha ids.")
    a_corr.add_argument("--vs-book", action="store_true",
                        help="Also report each alpha's maximum against ACTIVE alphas.")
    a_corr.add_argument("--control", default=None,
                        help="Regress this alpha's PnL out first, then correlate "
                             "the residuals.")
    a_corr.add_argument("--min-overlap", type=int, default=200,
                        help="Minimum shared trading days (default 200).")
    a_corr.add_argument("--threshold", type=float, default=0.7,
                        help="Flag pairs at or above this absolute value (default 0.7).")
    a_corr.add_argument("--max-age-hours", type=float, default=24.0,
                        help="Reuse cached PnL younger than this (default 24).")

    _add_session_args(a_corr)
    a_corr.set_defaults(func=cmd_alpha_corr)

    a_book = alpha_sub.add_parser(
        "book",
        help="Read what is actually submitted, with mechanism labels.",
        description="Lists ACTIVE alphas (the real book) joined with the "
                    "ledger, optionally reporting how a candidate would "
                    "correlate with it. `alpha list` hides this by default.")
    a_book.add_argument("--limit", type=int, default=50)
    a_book.add_argument("--check", default=None, metavar="ALPHA_ID",
                        help="Also report this candidate's self-correlation "
                             "against the whole book.")
    a_book.add_argument("--registry", default=None)
    _add_session_args(a_book)

    a_book.set_defaults(func=cmd_alpha_book)

    _attach_examples(parser)
    return parser


_COMMAND_EXAMPLES = {
    "": ["brief", "catalog fields --dataset fundamental6 --expressions --limit 10",
         'simulate --expr "rank(close)" --dry-run', "ledger list --family quality"],
    "help": ["help catalog fields"],
    "schema": ["schema alpha get"],
    "auth": ["auth check"],
    "auth check": ["auth check"],
    "auth logout": ["auth logout"],
    "cache": ["cache info"],
    "cache info": ["cache info"],
    "cache clear": ["cache clear --kind metadata"],
    "catalog": ["catalog datasets --limit 10", "catalog fields --dataset fundamental6 --limit 10"],
    "catalog operators": ["catalog operators --search rank --limit 10"],
    "catalog datasets": ["catalog datasets --region USA --limit 10"],
    "catalog fields": ["catalog fields --dataset fundamental6 --search liability --expressions --limit 10"],
    "catalog concepts": ["catalog concepts --dataset fundamental6 --limit 10"],
    "catalog verify": ["catalog verify --fields close volume --strict"],
    "catalog categories": ["catalog categories"],
    "catalog field-audit": ["catalog field-audit --fields close,volume"],
    "lint": ['lint --expr "rank(close)" --offline', "lint --file research.json"],
    "search": ['search "post earnings announcement drift" --min-authority 3'],
    "paper": ['paper "accruals stock returns" --limit 3'],
    "simulate": ['simulate --expr "rank(close)" --dry-run',
                 "simulate research.json --sort fitness --save results.json"],
    "resume": ["resume CHECKPOINT.json --save results.json"],
    "ablate": ['ablate --expr "a = rank(close); b = rank(volume); a + b" --out variants.json'],
    "compare": ["compare results.json --sort fitness"],
    "brief": ["brief", "brief --days 7 --top 5"],
    "probe": ["probe batch.spec.json --dry-run", "probe batch.spec.json"],
    "views": ["views --limit 10"],
    "alpha": ["alpha list --limit 10", "alpha get ALPHA_ID"],
    "alpha get": ["alpha get ALPHA_ID"],
    "alpha check": ["alpha check ALPHA_ID"],
    "alpha self-corr": ["alpha self-corr ALPHA_ID"],
    "alpha describe": ['alpha describe ALPHA_ID --name "quality signal" --category FUNDAMENTAL'],
    "alpha submit": ["alpha submit ALPHA_ID"],
    "alpha explain": ["alpha explain ALPHA_ID --no-corr"],
    "alpha list": ["alpha list --limit 10 --min-fitness 1"],
    "alpha self-check": ["alpha self-check ALPHA_ID"],
    "alpha corr": ["alpha corr ALPHA_A ALPHA_B --vs-book"],
    "alpha book": ["alpha book --limit 10"],
    "ledger": ["ledger list --family quality", "ledger coverage --list-missing"],
    "ledger record": ['ledger record ALPHA_ID --family quality --mechanism "profitability"'],
    "ledger list": ["ledger list --family quality --min-fitness 1", "ledger list --field close"],
    "ledger label": ["ledger label ALPHA_ID --family quality --verdict candidate"],
    "ledger impact": ["ledger impact ALPHA_ID"],
    "ledger lineage": ["ledger lineage ALPHA_ID"],
    "ledger coverage": ["ledger coverage --list-missing"],
}


def _command_children(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            summaries = {child.dest: child.help for child in action._choices_actions}
            return {name: (child, summaries.get(name, ""))
                    for name, child in action.choices.items()}
    return {}


def _examples(parser):
    path = " ".join(parser.prog.split()[1:])
    return ["python research_cli.py " + example for example in _COMMAND_EXAMPLES.get(path, [])]


def _attach_examples(parser):
    examples = _examples(parser)
    if examples:
        parser.epilog = "Examples:\n  " + "\n  ".join(examples)
    for child, summary in _command_children(parser).values():
        if child.description is None:
            child.description = summary
        _attach_examples(child)


def _find_command_parser(path):
    parser = build_parser()
    parts = path.split() if isinstance(path, str) else [word for part in path or [] for word in part.split()]
    for part in parts:
        children = _command_children(parser)
        if part not in children:
            parser.error("unknown command %r" % part,
                         suggestions=difflib.get_close_matches(part, children, n=3, cutoff=0.6))
        parser = children[part][0]
    return parser


def _type_name(action):
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "boolean"
    if action.type is int:
        return "integer"
    if action.type in (float, _positive_seconds, _non_negative_seconds):
        return "number"
    return "string"


def _describe_options(parser):
    options = []
    for action in parser._actions:
        if isinstance(action, (argparse._SubParsersAction, argparse._HelpAction, argparse._VersionAction)):
            continue
        entry = {"flags": action.option_strings or [action.dest], "type": _type_name(action)}
        entry["required"] = action.required
        if action.nargs is not None:
            entry["nargs"] = action.nargs
        if isinstance(action, argparse._AppendAction):
            entry["repeatable"] = True
        if action.choices:
            entry["choices"] = [str(c) for c in action.choices]
        if action.help:
            entry["help"] = action.help
        if action.default not in (None, False, [], argparse.SUPPRESS):
            entry["default"] = action.default
        options.append(entry)
    return options


def describe_parser(parser, name="research_cli.py"):
    """Derive the complete call contract from the live parser."""
    description = {"command": name, "usage": parser.format_usage().strip(),
                   "options": _describe_options(parser), "examples": _examples(parser)}
    children = _command_children(parser)
    if children:
        description["subcommands"] = {name: describe_parser(child, name)
                                      for name, (child, _summary) in children.items()}
    return description


def _help_document(parser):
    children = _command_children(parser)
    document = {"usage": parser.format_usage().strip(), "description": parser.description,
                "examples": _examples(parser)}
    if children:
        document["commands"] = {name: summary for name, (_child, summary) in children.items()}
        document["common_options"] = _describe_options(parser)
    else:
        document["options"] = _describe_options(parser)
    document["next"] = "Use --help for terminal help, or schema COMMAND [SUBCOMMAND] for the JSON contract."
    return document


def cmd_help(args, _sink):
    return _help_document(_find_command_parser(getattr(args, "path", [])))


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------
def with_session(args, work):
    """Run ``work(session)``, re-authenticating once if the session died.

    BRAIN sessions expire, and a cached cookie can go stale between calls.
    Retrying once with a forced login keeps every caller from handling that.
    """
    session = ml.load_session(use_cache=not getattr(args, "no_cache", False),
                              refresh=getattr(args, "relogin", False))
    try:
        return work(session)
    except ml.AuthError:
        session = ml.load_session(use_cache=False, refresh=True)
        return work(session)


def collect_alphas(args):
    """Merge files in order, then positional and repeated inline expressions."""
    file_settings, alphas = {}, []
    paths = ([args.file] if getattr(args, "file", None) else [])
    paths.extend(getattr(args, "files", None) or [])
    if sum(str(path) == "-" for path in paths) > 1:
        raise CliError("INVALID_INPUT", "stdin (-) may be read only once")
    for path in paths:
        try:
            settings, entries = config_io.load_research_file(path)
        except config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc)) from exc
        file_settings.update(settings)
        alphas.extend(entries)
    if getattr(args, "expression", None) is not None:
        alphas.append(args.expression)
    expressions = getattr(args, "expr", None)
    if expressions is not None:
        alphas.extend([expressions] if isinstance(expressions, str) else expressions)
    if not alphas:
        raise CliError("USAGE", "no expressions supplied",
                       "pass a research file, --file, or at least one --expr")
    return file_settings, alphas


def merge_settings(file_settings, args):
    """CLI flags win over the file, which wins over machine_lib defaults."""
    overrides = dict(file_settings)
    for key, value in {
        "region": getattr(args, "region", None),
        "universe": getattr(args, "universe", None),
        "instrumentType": getattr(args, "instrument_type", None),
        "neutralization": getattr(args, "neutralization", None),
        "decay": getattr(args, "decay", None),
        "delay": getattr(args, "delay", None),
        "truncation": getattr(args, "truncation", None),
        "pasteurization": getattr(args, "pasteurization", None),
        "nanHandling": getattr(args, "nan_handling", None),
        "testPeriod": getattr(args, "test_period", None),
    }.items():
        if value is not None:
            overrides[key] = value
    try:
        return ml.resolve_settings(overrides)
    except ValueError as exc:
        raise CliError("INVALID_INPUT", str(exc))


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------
def _train_metrics(record):
    """The metrics a selection decision is allowed to see: the train period only.

    The platform's headline numbers use the full window, which contains the test
    year, so every path that ranks, filters or ablates must come through here unless
    the caller has explicitly frozen the design (``--frozen`` / ``--show-test``).
    """
    return (record.get("out_of_sample") or {}).get("train") or {}


def _finished(record):
    """A run that produced final numbers.

    ``WARNING`` is a finished run: BRAIN marks a simulation WARNING when a check
    warns without failing (UNITS is the usual one) and every metric is final.
    Counting only COMPLETE made a batch of WARNING records look like zero
    completions, so `simulate` reported SIMULATION_FAILED and skipped recording.
    """
    return (record.get("status") or "").upper() in alpha_registry.FINISHED_STATUSES


def summarize(records):
    """Count outcomes and rank the scored alphas so a caller can pick winners."""
    completed = [r for r in records if _finished(r)]
    failed = [r for r in records if (r.get("status") or "").upper() == "FAILED"]
    scored = [r for r in completed if isinstance(r.get("metrics"), dict)]
    ranking = sorted(
        (
            {
                "alpha_id": r.get("alpha_id"),
                "expression": r.get("expression"),
                **{k: r["metrics"].get(k) for k in METRIC_KEYS if k in r["metrics"]},
            }
            for r in scored
        ),
        key=lambda item: item.get("sharpe") if item.get("sharpe") is not None else float("-inf"),
        reverse=True,
    )
    return {
        "total": len(records),
        "completed": len(completed),
        "failed": len(failed),
        "pending": sum(
            (r.get("status") or "").upper() == "PENDING" or bool(r.get("progress_url"))
            for r in records
        ),
        "queued": sum((r.get("status") or "").upper() == "QUEUED" for r in records),
        "other": len(records) - len(completed) - len(failed),
        "ranking": ranking,
        "best": ranking[0] if ranking else None,
    }


def _metric_value(record, key):
    metrics = record.get("metrics") or {}
    value = metrics.get(key)
    return value if value is not None else float("-inf")


def _compact_metrics(record):
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        record.pop("metrics", None)
        return
    record["metrics"] = {k: metrics.get(k) for k in METRIC_KEYS if k in metrics}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_schema(args, _sink):
    if args.name:
        selected = _find_command_parser(args.name)
        return {"tool": "research_cli.py", "version": VERSION,
                "exit_codes": EXIT_CODES, "error_codes": ERROR_CODES,
                "command": describe_parser(selected, " ".join(selected.prog.split()[1:]))}
    described = describe_parser(build_parser())
    contract = {
        "tool": "research_cli.py",
        "version": VERSION,
        "stdout_contract": "exactly one JSON document, success or failure",
        "success_envelope": {"ok": True, "command": "simulate", "data": "..."},
        "error_envelope": {
            "ok": False,
            "command": "simulate",
            "error": {"code": "INVALID_INPUT", "message": "...", "hint": "..."},
        },
        "exit_codes": EXIT_CODES,
        "error_codes": ERROR_CODES,
        "research_file": config_io.RESEARCH_FILE_SPEC,
        "settings_defaults": ml.DEFAULT_SETTINGS,
        "settings_choices": {
            "region": ml.REGIONS,
            "neutralization": ml.NEUTRALIZATIONS,
            "universe": ml.UNIVERSES,
            "delay": [0, 1],
        },
        "credentials": {
            "env": ["BRAIN_USERNAME", "BRAIN_PASSWORD"],
            "file": ".env next to machine_lib.py (git-ignored)",
            "precedence": "real environment variables win over .env",
        },
        "session_cache": {
            "path": str(ml.session_cache_path()),
            "format": "cache envelope, mode 0600",
            "single_flight": "concurrent CLI calls share one login (file lock)",
            "note": ("cookies are reused until just before expiry and the file "
                     "lives outside the git working tree"),
        },
        "cache": {
            "root": research_paths.cache_root(),
            "layout": {
                "session": str(ml.session_cache_path()),
                "metadata": research_paths.metadata_root(),
                "locks": os.path.join(research_paths.cache_root(), "locks"),
            },
            "metadata_policy": ml.metadata_cache_policy(),
            "metadata_ttl_env": "BRAIN_METADATA_TTL (seconds; 0 disables)",
            "metadata_kinds": ["datasets", "datafields", "operators",
                               "operator-doc", "data-categories"],
            "stale_on_error": ("a failed refetch serves the last snapshot with "
                               "cache.stale=true instead of failing the command"),
            "commands": ["cache info", "cache clear --kind session|metadata|all",
                         "auth logout"],
        },
        "research_paths": research_paths.describe(),
        "commands": described,
    }
    return contract


def cmd_auth_check(args, _sink):
    meta = {}
    session = ml.load_session(use_cache=not args.no_cache, refresh=args.relogin,
                              meta=meta)
    info = ml.session_status()
    regions = []
    for region in ml.REGIONS:
        try:
            if ml.get_datasets(session, region=region, delay=1, limit=1):
                regions.append(region)
        except ml.BrainError:
            continue
    return {
        "authenticated": True,
        "user_id": info.get("user_id"),
        "permissions": info.get("permissions"),
        "token_lifetime_seconds": info.get("expires_in_seconds"),
        "data_regions": regions,
        "session": {
            "cached": meta.get("cached"),
            "reason": meta.get("reason"),
            "cache_state": meta.get("cache_state"),
            "source": info.get("source"),
            "path": info.get("path"),
            "expires_at": info.get("expires_at"),
            "expires_in_seconds": info.get("expires_in_seconds"),
            "age_seconds": info.get("age_seconds"),
            "migrated_from": meta.get("migrated_from"),
        },
        "metadata_cache": ml.metadata_cache_policy(),
        "note": "regions outside data_regions return no datasets for this account",
    }


def cmd_auth_logout(args, _sink):
    """Forget the cached session. Does not revoke it server-side."""
    removed = ml.invalidate_session()
    return {
        "logged_out": removed,
        "session_path": str(ml.session_cache_path()),
        "note": ("Removed only the local cookie cache; the BRAIN session itself "
                 "stays valid until it expires server-side."),
    }


def cmd_cache_info(args, _sink):
    """Describe what is cached, how fresh it is, and where it lives."""
    root = research_paths.cache_root()
    stats = cache_store.cache_stats(root)
    catalog = {}
    meta_root = research_paths.metadata_root()
    if os.path.isdir(meta_root):
        for kind in sorted(os.listdir(meta_root)):
            kind_dir = os.path.join(meta_root, kind)
            if not os.path.isdir(kind_dir):
                continue
            entries = []
            for name in sorted(os.listdir(kind_dir)):
                if not name.endswith(".json"):
                    continue
                entry = cache_store.read_entry(os.path.join(kind_dir, name))
                entries.append({
                    "file": name,
                    "fresh": entry["fresh"],
                    "age_seconds": None if entry.get("age_seconds") is None
                    else round(entry["age_seconds"], 1),
                    "expires_in_seconds": None if entry.get("expires_in") is None
                    else round(entry["expires_in"], 1),
                    "error": entry.get("error"),
                })
            catalog[kind] = {
                "entries": len(entries),
                "fresh": sum(1 for e in entries if e["fresh"]),
                "oldest_age_seconds": max(
                    [e["age_seconds"] for e in entries
                     if e["age_seconds"] is not None] or [None])
                if entries else None,
            }
    return {
        "cache_root": root,
        "policy": {
            "metadata": ml.metadata_cache_policy(),
            "session": {
                "single_flight": True,
                "file_mode": "0600",
                "lock_timeout_seconds": float(
                    os.environ.get("BRAIN_LOGIN_LOCK_TIMEOUT", "60") or 60),
            },
        },
        "session": ml.session_status(),
        "metadata": catalog,
        "stats": stats,
    }


def cmd_cache_clear(args, _sink):
    """Drop cached data. Locks are kept so a running process is not disturbed."""
    root = research_paths.cache_root()
    scope = args.kind
    removed = {"session": 0, "metadata": 0}
    if scope in ("all", "session"):
        removed["session"] = 1 if ml.invalidate_session() else 0
    if scope in ("all", "metadata"):
        removed["metadata"] = cache_store.clear_dir(research_paths.metadata_root(),
                                                   keep_locks=True)
    return {
        "cleared": scope,
        "removed": removed,
        "cache_root": root,
        "kept_locks": True,
    }


def _prepare_output(target):
    """Check the destination before a command can spend quota or change state."""
    if not target or target == "-":
        return None
    path = Path(target).expanduser().resolve()
    try:
        if path.is_dir() or (path.exists() and not os.access(path, os.W_OK)):
            raise OSError("destination is not a writable file")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".%s." % path.name,
                                                 suffix=".tmp", dir=str(path.parent))
        os.close(descriptor)
    except OSError as exc:
        raise CliError("INVALID_INPUT", "cannot save output to %s: %s" % (path, exc),
                       "choose a writable --save path, or use --save - for stdout") from exc
    return path, Path(temporary)


def cmd_catalog_operators(args, _sink):
    cache_report = {}

    def work(session):
        # Fetch the whole list and rank locally for the same reason fields are
        # ranked locally: the endpoint matches one token, so a two-word intent
        # could not be expressed at all.
        operators = ml.fetch_operators(session, search="", group=args.group,
                                       report=cache_report)
        scanned = len(operators)
        search_note = None
        if args.search:
            required, excluded = field_search.parse_query(args.search)
            operators = field_search.search_records(
                operators, args.search, limit=args.limit, fuzzy=args.fuzzy,
                match_all=not args.any_term, rank_by_coverage=False)
            absent = field_search.absent_terms(
                operators or ml.fetch_operators(session, search="", group=args.group),
                args.search, fuzzy=args.fuzzy) if not operators else []
            search_note = {
                "query": args.search, "required_terms": required,
                "excluded_terms": excluded, "absent_terms": absent,
                "mode": "any" if args.any_term else "all",
                "fuzzy": bool(args.fuzzy),
            }
            if absent:
                search_note["hint"] = "no operator uses these words"
        elif args.limit:
            operators = operators[: args.limit]
        data = {"count": len(operators), "scanned": scanned,
                "operators": operators, "cache": cache_report}
        if search_note:
            data["search"] = search_note
        if args.docs:
            data["docs"] = {n: ml.fetch_operator_doc(session, n) for n in args.docs}
        return data

    data = with_session(args, work)

    return data


def cmd_catalog_datasets(args, _sink):
    cache_report = {}
    datasets = with_session(args, lambda s: ml.get_datasets(
        s, instrument_type=args.instrument_type, region=args.region,
        delay=args.delay, universe=args.universe, limit=args.limit,
        report=cache_report,
    ))
    data = {
        "region": args.region, "universe": args.universe, "delay": args.delay,
        "count": len(datasets), "datasets": datasets, "cache": cache_report,
    }
    if not datasets:
        data["hint"] = ("no datasets for this combination; the account may lack "
                        "permission (a TUTORIAL account only sees USA)")

    return data


def cmd_catalog_fields(args, _sink):
    """List a dataset's fields, with ranked local search.

    Search runs here rather than on the server. The endpoint takes a single
    token, so ``--search cfps count`` used to be a usage error and any multi-word
    intent was impossible to express. Fetching the dataset (cached) and ranking
    locally supports AND terms, exclusions, prefixes and typos, and reports why
    each hit matched.
    """
    cache_report = {}
    everything = with_session(args, lambda s: ml.get_datafields(
        s, instrument_type=args.instrument_type, region=args.region,
        delay=args.delay, universe=args.universe, dataset_id=args.dataset,
        search="", field_type=args.field_type, limit=None,
        report=cache_report,
    ))

    fields = everything
    search_note = None
    if args.search:
        required, excluded = field_search.parse_query(args.search)
        fields = field_search.search_records(
            everything, args.search, limit=args.limit,
            fuzzy=args.fuzzy, match_all=not args.any_term,
        )
        absent = field_search.absent_terms(
            everything, args.search, fuzzy=args.fuzzy) if not fields else []
        search_note = {
            "query": args.search,
            "required_terms": required,
            "excluded_terms": excluded,
            "absent_terms": absent,
            "mode": "any" if args.any_term else "all",
            "fuzzy": bool(args.fuzzy),
            "note": "ranked locally: exact token > prefix > substring, id beats description",
        }
        if absent:
            search_note["hint"] = (
                "the catalog does not use these words anywhere in this dataset; "
                "try --any-term or different wording")
        elif not fields and len(required) > 1:
            # Every word exists somewhere, just never together in one field. That
            # is the common case when asking for an idea spelled across two
            # naming families, and it deserves a different suggestion than a
            # missing word does.
            search_note["hint"] = (
                "each term appears somewhere in this dataset but no single field "
                "contains all of them; try --any-term, or search the terms one at "
                "a time with `catalog concepts`")
    elif args.limit:
        fields = fields[: args.limit]

    data = {
        "dataset": args.dataset, "region": args.region, "universe": args.universe,
        "delay": args.delay, "count": len(fields), "scanned": len(everything),
        "type_counts": {
            kind: len([f for f in fields if f.get("type") == kind])
            for kind in ("MATRIX", "VECTOR", "GROUP")
        },
        "fields": fields, "cache": cache_report,
    }
    if search_note:
        data["search"] = search_note
    if args.expressions:
        data["expressions"] = ml.process_datafields(fields)

    return data


def cmd_catalog_concepts(args, _sink):
    """Group fields by shared tokens and show coverage side by side.

    The tool brings no vocabulary of its own: pass ``--ignore-words`` for the
    boilerplate tokens of the family you are looking at, and everything left is
    compared literally. Baking catalog knowledge into the source would encode one
    snapshot of the naming and rot silently, so the judgment stays here at the
    command line.
    """
    records = with_session(args, lambda s: field_index.fetch_all_fields(
        s, region=args.region, universe=args.universe, delay=args.delay,
        instrument_type=args.instrument_type,
        dataset_ids=[args.dataset] if args.dataset else None,
    ))
    concepts = field_index.duplicate_concepts(
        records, ignore_words=args.ignore_words,
        min_members=args.min_members, min_spread=args.min_spread,
        field_type=args.field_type, drop_numeric=args.drop_numeric,
        tail_tokens=args.tail_tokens,
    )
    if args.search:
        # Rank concepts, not raw fields: the concept key is the thing being
        # asked about, and the member ids are folded into the description so a
        # query can hit either the grouping or a specific variant.
        haystack = [
            {
                "id": item["concept"],
                "description": " ".join(member["id"] for member in item["fields"]),
                "coverage": item["best_coverage"],
                "_concept": item,
            }
            for item in concepts
        ]
        ranked = field_search.search_records(
            haystack, args.search, limit=args.limit, fuzzy=args.fuzzy,
            match_all=not args.any_term,
        )
        concepts = []
        for hit in ranked:
            item = dict(hit["_concept"])
            item["score"] = hit["score"]
            item["matched_terms"] = hit["matched_terms"]
            concepts.append(item)
    limit = args.limit or len(concepts)
    data = {
        "region": args.region, "universe": args.universe, "delay": args.delay,
        "dataset": args.dataset,
        "ignore_words": list(args.ignore_words or []),
        "drop_numeric": bool(args.drop_numeric),
        "tail_tokens": args.tail_tokens,
        "fields_scanned": len(records),
        "concepts_with_variants": len(concepts),
        "shown": min(limit, len(concepts)),
        "concepts": concepts[:limit],
        "note": ("grouping is literal token matching after --ignore-words; "
                 "coverage_spread is best minus worst member"),
    }

    return data


def cmd_catalog_verify(args, _sink):
    """Check field identifiers against the live catalog, quota-free.

    This is the check that catches a name typed from memory. Such a name is not
    an operator, so the operator catalog cannot see it, and the first symptom is
    a simulator error after the submission has already been spent.
    """
    wanted = []
    for chunk in args.fields or []:
        for piece in chunk.replace(",", " ").split():
            if piece and piece not in wanted:
                wanted.append(piece)
    if not wanted:
        raise CliError("USAGE", "nothing to verify",
                       "pass --fields with one or more field ids, comma or space separated")

    known = with_session(args, lambda s: field_index.field_ids(
        field_index.fetch_all_fields(
            s, region=args.region, universe=args.universe, delay=args.delay,
            instrument_type=args.instrument_type,
        )
    ))
    unknown = [name for name in wanted if name not in known]
    found = [name for name in wanted if name in known]
    data = {
        "region": args.region, "universe": args.universe, "delay": args.delay,
        "catalog_size": len(known),
        "checked": len(wanted),
        "found": found,
        "unknown": unknown,
        "verdict": "ok" if not unknown else "unknown_fields",
    }
    if unknown and args.strict:
        raise CliError(
            "INVALID_INPUT",
            "field name(s) not present in the live catalog: %s" % ", ".join(unknown),
            "run `catalog concepts --search <word>` to see the real variants for that idea",
            details=data,
        )

    return data


def cmd_catalog_categories(args, _sink):
    cache_report = {}
    categories = with_session(args, lambda s: ml.fetch_data_categories(
        s, region=args.region, universe=args.universe, delay=args.delay,
        report=cache_report))
    data = {
        "region": args.region, "universe": args.universe, "delay": args.delay,
        "count": len(categories), "categories": categories, "cache": cache_report,
        "note": "alphaCount is a crowding signal: higher means more mined",
    }

    return data


def _write_simulation_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".%s." % path.name, suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(checkpoint, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _simulation_checkpoint_callback(path, checkpoint, sink):
    def on_event(kind, payload):
        if kind in {"submitted", "pending", "completed", "failed", "deferred"}:
            index = payload.get("index")
            if isinstance(index, int) and 0 <= index < len(checkpoint["records"]):
                checkpoint["records"][index] = {
                    key: value for key, value in payload.items() if key != "index"
                }
                _write_simulation_checkpoint(path, checkpoint)
        if sink:
            sink(kind, payload)

    return on_event


def _sort_simulation_results(records, args):
    for record in records:
        _compact_metrics(record)
    if args.sort != "none":
        records.sort(key=lambda r: _metric_value(r, args.sort), reverse=True)
    if args.top is not None:
        records = records[:args.top]
    return records


def _simulation_result(records, settings, checkpoint_path=None, dry_run=False):
    summary = summarize(records)
    if summary["pending"] or summary["queued"]:
        status = "PENDING"
    elif summary["failed"]:
        status = "PARTIAL" if summary["completed"] else "FAILED"
    else:
        status = "COMPLETE"
    result = {
        "status": status,
        "dry_run": dry_run,
        "settings": settings,
        "summary": summary,
        "results": records,
    }
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        result["checkpoint"] = str(Path(checkpoint_path))
    return result


def _prepare_simulation_run(settings, pairs, batch_size, checkpoint_arg,
                            labels=None, families=None, candidates=None):
    if checkpoint_arg:
        path = Path(checkpoint_arg).expanduser().resolve()
        if path.exists():
            raise CliError(
                "INVALID_INPUT",
                "checkpoint already exists; use `resume` or choose another path",
                str(path),
            )
    else:
        path = Path(".brain_checkpoints") / ("simulate-%s.json" % uuid.uuid4().hex)
        path = path.resolve()
    # The label columns must be written here, not only in the dry-run branch.
    # These records are handed to simulate_alphas as initial_records, so anything
    # omitted from this dict is lost for the whole real run: mechanism tags
    # arrived empty in the ledger for exactly this reason.
    labels = labels or [None] * len(pairs)
    families = families or [None] * len(pairs)
    candidates = candidates or [None] * len(pairs)
    records = [
        {
            "expression": expression,
            "decay": decay,
            "candidate_id": candidates[index],
            "mechanism": labels[index],
            "family": families[index],
            "alpha_id": None,
            "status": "QUEUED",
            "metrics": None,
            "checks": None,
            "error": None,
        }
        for index, (expression, decay) in enumerate(pairs)
    ]
    checkpoint = {
        "format": "worldquantuser.simulation-checkpoint",
        "version": 1,
        "settings": settings,
        "batch_size": batch_size,
        "records": records,
    }
    _write_simulation_checkpoint(path, checkpoint)
    return path, checkpoint


def _finalize_simulation(records, settings, args, checkpoint_path, checkpoint, explicit):
    checkpoint["records"] = [dict(record) for record in records]
    _write_simulation_checkpoint(checkpoint_path, checkpoint)
    has_unfinished = any(
        record.get("progress_url")
        or record.get("status") in {"PENDING", "QUEUED"}
        for record in records
    )
    if not explicit and not has_unfinished:
        Path(checkpoint_path).unlink(missing_ok=True)
        checkpoint_path = None

    # The display view honours --sort/--top; the batch, the summary and the ledger
    # must not. Truncating before recording lost every arm outside the top N - and
    # with the checkpoint deleted at the end, those alphas had been paid for and
    # left no trace, which is both wasted quota and a selection bias in the archive.
    every = list(records)
    display = _sort_simulation_results(records, args)
    summary = summarize(every)
    if summary["completed"] == 0 and summary["total"] and not (
        summary["pending"] or summary["queued"]
    ):
        first_error = next(
            (r.get("error") for r in every if r.get("error")), "unknown error"
        )
        raise CliError(
            "SIMULATION_FAILED",
            "all %d simulations failed" % summary["total"],
            str(first_error),
        )
    result = _simulation_result(every, settings, checkpoint_path)
    result["results"] = display
    result["results_total"] = len(every)
    pending_ids = [record.get("alpha_id") or record.get("candidate_id")
                   for record in every
                   if (record.get("status") or "").upper() in {"PENDING", "QUEUED"}]
    if pending_ids:
        result["pending"] = pending_ids
        result["next"] = ("some arms are still running; the checkpoint was kept - "
                          "run `resume %s` before judging them"
                          % (checkpoint_path or "<checkpoint>"))
    if getattr(args, "record", False):
        result["registry"] = _record_simulation(every, settings, args)
    return result


def cmd_ablate(args, _sink):
    """Write a research file holding full / leave-one-out / solo variants.

    Deliberately does not simulate: generating and spending stay separate, so a
    mis-specified ablation costs nothing and can be inspected first.
    """
    expression = args.expr
    decay = args.decay
    if args.file and expression is not None:
        raise CliError("INVALID_INPUT", "ablate accepts either a file or --expr, not both")
    if args.ids and (args.file or expression is not None):
        raise CliError("INVALID_INPUT", "--ids cannot be combined with a file or --expr")
    settings = {}
    settings_source = "defaults"
    alphas = None
    if args.file:
        try:
            settings, alphas = config_io.load_research_file(args.file)
        except config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc)) from exc
        settings_source = args.file
    resolved = merge_settings(settings, args)
    if any(getattr(args, flag, None) is not None for flag in (
            "region", "universe", "instrument_type", "neutralization", "decay",
            "delay", "truncation", "pasteurization", "nan_handling", "test_period")):
        settings_source = "flags"
    if alphas is not None:
        try:
            _settings, pairs = config_io.build_config(resolved, alphas)
        except config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc)) from exc
        if len(pairs) != 1:
            raise CliError("INVALID_INPUT",
                           "ablate takes exactly one composite expression, found %d" % len(pairs),
                           "split the file, or pass --expr with one composite")
        expression = pairs[0][0]
        if decay is None:
            decay = pairs[0][1]
    elif expression is not None:
        try:
            _settings, pairs = config_io.build_config(resolved, [expression])
        except config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc)) from exc
        expression, decay = pairs[0]

    if args.ids:
        # Batch form. One file per alpha, because `compare --contributions` groups by
        # candidate_id and reads `full` / `loo:<leg>` / `solo:<leg>`; merging six
        # alphas into one file would collide on those names and silently attribute
        # one alpha's leg contributions to another.
        path, entries = _ledger(args)
        out_dir = Path(args.out_dir or research_paths.ensure_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        modes = [part.strip() for part in str(args.modes).split(",") if part.strip()]
        done, skipped = [], []
        for raw in str(args.ids).replace(",", " ").split():
            entry = alpha_registry.resolve(entries, raw)
            if entry is None:
                skipped.append({"alpha_id": raw, "reason": "not in the ledger"})
                continue
            source = entry.get("expression") or ""
            try:
                plan = ablation.variants(source, modes)
            except ablation.AblationError as exc:
                skipped.append({"alpha_id": raw, "reason": str(exc)})
                continue
            alpha_settings = merge_settings(entry.get("settings") or {}, args)
            alphas = []
            for item in plan:
                item_entry = {"expression": item["expression"],
                              "candidate_id": item["candidate_id"],
                              "mechanism": entry.get("mechanism") or raw}
                item_entry["decay"] = alpha_settings["decay"]
                alphas.append(item_entry)
            target = out_dir / ("%s-ablation.json" % raw)
            target.write_text(json.dumps({"settings": alpha_settings, "alphas": alphas},
                                         ensure_ascii=False, indent=2))
            done.append({"alpha_id": raw, "file": str(target), "variants": len(plan)})
            note("wrote %s (%d variants)" % (target, len(plan)))
        return {"mode": "ids", "written": done, "skipped": skipped,
                "count": sum(item["variants"] for item in done),
                "next": "simulate each file, then `compare <results> --contributions`"}

    if not expression:
        raise CliError("USAGE", "nothing to ablate",
                       "pass a research file, --expr, or --ids")

    modes = [part.strip() for part in str(args.modes).split(",") if part.strip()]
    try:
        plan = ablation.variants(expression, modes)
    except ablation.AblationError as exc:
        raise CliError("INVALID_INPUT", str(exc),
                       "ablation needs 'name = ...; name = ...; name + name'; "
                       "declare the legs explicitly and test by hand otherwise")

    alphas = []
    for item in plan:
        entry = {
            "expression": item["expression"],
            "candidate_id": item["candidate_id"],
            "mechanism": item["candidate_id"],
        }
        if decay is not None:
            entry["decay"] = decay
        alphas.append(entry)
    document = {"settings": resolved, "alphas": alphas}

    if args.out:
        Path(args.out).write_text(json.dumps(document, ensure_ascii=False, indent=2))
        note("wrote %s" % args.out)

    data = {
        "source_expression": expression,
        "modes": modes,
        "variants": plan,
        "count": len(plan),
        "settings": resolved,
        "settings_source": settings_source,
        "written_to": args.out,
        "next": ("run with `simulate %s`, then `compare <results> --contributions`"
                 % (args.out or "<file>")),
    }

    return data


def cmd_compare(args, _sink):
    """Merge result documents into one table.

    Variants used to live in separate batch files and were aligned by eye, which
    is how a five-leg version scoring below its four-leg parent went unnoticed.

    Two views, because the default one must not leak the test year. ``train`` ranks
    on the train period and does not print the full-window numbers; ``frozen`` shows
    them for a design that is already fixed.
    """
    frozen = bool(getattr(args, "frozen", False))
    def records_from(path, payload):
        """Records from the two shapes this repository writes.

        `simulate` writes ``{"data": {"results": [...]}}`` and the research template
        writes a bare list, so a batch produced by the template could not be compared
        at all - the file was there, the command refused it, and the reason had
        nothing to do with the batch. Rows already flattened by `compare` are not
        accepted on purpose: they carry ``sharpe`` at the top level, and re-reading
        them as records would fill the table with nulls instead of saying so.
        """
        if isinstance(payload, dict) and "data" in payload:
            return records_from(path, payload.get("data"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            return [item for item in payload["results"] if isinstance(item, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            # `probe` writes the raw simulation records next to its display rows, so a
            # batch can be re-read under --frozen without re-running anything.
            return [item for item in payload["records"] if isinstance(item, dict)]
        raise CliError("INVALID_INPUT",
                       "%s is not a simulate result document" % path,
                       "expected {\"results\": [...]} or a bare list of records; "
                       "`ledger list --format json` rows are not records")

    if sum(str(path) == "-" for path in args.files) > 1:
        raise CliError("INVALID_INPUT", "stdin (-) may be read only once")
    records = []
    for path in args.files:
        try:
            payload = json.loads(config_io.read_input_text(path))
        except (config_io.ResearchError, ValueError) as exc:
            raise CliError("INVALID_INPUT", "cannot read %s: %s" % (path, exc),
                           "point at a file written by `simulate`") from exc
        records.extend(records_from(path, payload))

    if not records:
        raise CliError("NOT_FOUND", "no result records in the given files",
                       "the file must come from `simulate`")

    rows = []
    for record in records:
        metrics = record.get("metrics") or {}
        checks = record.get("checks") or {}
        failed = sorted(name for name, value in checks.items()
                        if isinstance(value, dict)
                        and value.get("result") in ("FAIL", "ERROR"))
        oos = record.get("out_of_sample") or {}
        train = oos.get("train") or {}
        row = {
            "candidate_id": record.get("candidate_id"),
            "alpha_id": record.get("alpha_id"),
            "decay": record.get("decay"),
            # The exploration columns. `sharpe`/`fitness` below are the full window,
            # which contains the test year, so they are only filled in the frozen view.
            "train_sharpe": train.get("sharpe"),
            "train_fitness": train.get("fitness"),
            "train_turnover": train.get("turnover"),
            "train_returns": train.get("returns"),
            "failed_checks": failed,
        }
        if frozen:
            row.update({"sharpe": metrics.get("sharpe"), "fitness": metrics.get("fitness"),
                        "turnover": metrics.get("turnover"),
                        "returns": metrics.get("returns"),
                        "test_sharpe": (oos.get("test") or {}).get("sharpe")})
        rows.append(row)

    fitness_key = "fitness" if frozen else "train_fitness"
    if args.min_fitness is not None:
        rows = [row for row in rows
                if isinstance(row.get(fitness_key), (int, float))
                and row[fitness_key] >= args.min_fitness]

    key = "alpha_id" if args.sort == "id" else args.sort
    if not frozen and key in {"sharpe", "fitness", "turnover", "returns"}:
        key = "train_" + key
    rows.sort(key=lambda row: (row[key] is None, -(row[key] or 0)
                               if isinstance(row[key], (int, float))
                               else str(row[key])))

    data = {"files": args.files, "count": len(rows), "view": "frozen" if frozen else "train",
            "rows": rows}
    if frozen:
        research_paths.log_view("compare", "frozen", [row.get("alpha_id") for row in rows],
                                note=", ".join(str(path) for path in args.files))
    if args.contributions:
        # The contribution is a selection decision ("does this leg earn its place"),
        # so in the exploration view it has to be computed from train-period fitness.
        # It used to read the full-window block regardless, which put the test year
        # back into the answer the table had just hidden.
        data["contributions"] = ablation.report(
            records, metrics_of=None if frozen else _train_metrics)
        data["contributions"]["view"] = "frozen" if frozen else "train"

    return data


def _load_probe_spec(path):
    """Read and validate a probe spec: settings plus a list of labelled arms."""
    try:
        spec = json.loads(config_io.read_input_text(path))
    except (config_io.ResearchError, ValueError) as exc:
        raise CliError("INVALID_INPUT", "cannot read spec %s: %s" % (path, exc),
                       "pass a JSON file shaped like "
                       "{\"name\": ..., \"settings\": {...}, \"arms\": [{\"label\", "
                       "\"expression\"}]}") from exc
    if not isinstance(spec, dict):
        raise CliError("INVALID_INPUT", "spec must be a JSON object",
                       "see `research_cli.py schema probe`")
    arms = spec.get("arms")
    if not isinstance(arms, list) or not arms:
        raise CliError("INVALID_INPUT", "spec has no arms",
                       "each arm needs at least {\"expression\": ...}")
    known = {"name", "idea", "mechanism", "note", "settings", "arms"}
    unknown = sorted(set(spec) - known)
    if unknown:
        # A typo here is silent otherwise: `"setting"` instead of `"settings"` would
        # run the whole batch on the defaults and look like a failed hypothesis.
        note("spec: ignoring unknown key(s) %s (known: %s)"
             % (", ".join(unknown), ", ".join(sorted(known))))
    if "settings" in spec and not isinstance(spec["settings"], dict):
        raise CliError("INVALID_INPUT", "spec settings must be an object")
    for index, arm in enumerate(arms, 1):
        if not isinstance(arm, dict) or not isinstance(arm.get("expression"), str) \
                or not arm["expression"].strip():
            raise CliError("INVALID_INPUT", "arm %d has no expression" % index,
                           "an arm is {\"label\": ..., \"expression\": ..., "
                           "\"settings\": {...} (optional)}")
        if "settings" in arm and not isinstance(arm["settings"], dict):
            raise CliError("INVALID_INPUT", "arm %d settings must be an object" % index)
        arm.setdefault("label", "arm%d" % index)
    return spec


def cmd_probe(args, _sink):
    """Run a batch of arms from one spec file: lint, simulate, table, record.

    Every experiment in this campaign is those four steps, and each was being
    re-written as a throwaway script. The copies drifted in ways that cost real time:
    one printed its table to stdout and broke the one-JSON-document contract, one
    reused its own output filename and silently overwrote the previous batch, one had
    a quoting bug that only surfaced at import, and none logged progress, so a
    four-minute wait looked like a hang.

    A spec file plus this verb removes all four: expressions are JSON strings so
    quoting cannot break, the output name carries the spec name *and* refuses to
    overwrite, progress goes to stderr, and the table shows **train only** unless
    ``--show-test`` is passed - selecting on the test year turns the only
    out-of-sample evidence into training data.
    """
    # Resolve every arm before authenticating or spending quota on the first group.
    spec = _load_probe_spec(args.spec)
    settings = dict(spec.get("settings") or {})
    arms = spec["arms"]
    name = args.name or spec.get("name") or ("stdin" if args.spec == "-" else Path(args.spec).stem)

    raw_groups = {}
    for arm in arms:
        overrides = merge_settings({**settings, **arm.get("settings", {})}, args)
        key = json.dumps(overrides, sort_keys=True)
        raw_groups.setdefault(key, {"settings": overrides, "arms": []})["arms"].append(arm)

    groups = {}
    for group in raw_groups.values():
        entries = [{key: value for key, value in arm.items() if key != "settings"}
                   for arm in group["arms"]]
        try:
            resolved, pairs = config_io.build_config(group["settings"], entries)
        except config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc)) from exc
        for arm, (expression, decay) in zip(group["arms"], pairs):
            # guarded_sim lints strings, so each group must share the effective
            # decay as well as the other request settings. Explicit defaults and
            # per-arm decay overrides then coalesce into identical payload scopes.
            effective = {**resolved, "decay": decay}
            key = json.dumps(effective, sort_keys=True)
            prepared = groups.setdefault(key, {"settings": effective, "arms": [], "pairs": []})
            prepared["arms"].append({**arm, "expression": expression})
            prepared["pairs"].append((expression, decay))

    # The output path and its collision rule are decided before anything is spent:
    # an existing batch file used to raise only after the simulations had run, so a
    # same-day re-run paid full quota and then refused to save the result.
    target = Path(research_paths.batch_path(name))
    if target.exists() and not args.force:
        raise CliError("INVALID_INPUT", "%s already exists" % target.name,
                       "pass --name to label this batch, or --force to overwrite it")

    if args.dry_run:
        results = []
        pairs = []
        for group in groups.values():
            labels = [arm["label"] for arm in group["arms"]]
            preview = ml.simulate_alphas(
                None, group["pairs"], settings=group["settings"],
                dry_run=True, labels=labels)
            for record, label in zip(preview, labels):
                record["label"] = label
                record["settings"] = dict(group["settings"])
            results.extend(preview)
            pairs.extend(group["pairs"])
        report = expression_lint.lint_all(
            pairs, denylist=expression_lint.load_denylist())
        blocking = report["errors"]
        note("dry run: %d arm(s) in %d settings group(s), %d blocking finding(s)"
             % (len(arms), len(groups), len(blocking)))
        return {"dry_run": True, "arms": len(arms), "groups": len(groups),
                "blocking": blocking,
                "settings": [group["settings"] for group in groups.values()],
                "results": results,
                "lint": {"operator_catalog_checked": False,
                         "field_catalog_checked": False,
                         "hint": "Live operator and field catalogs were not checked; "
                                 "run `lint` for authoritative catalog checks.",
                         "ok": report["ok"], "errors": report["errors"],
                         "warnings": report["warnings"]}}

    sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
    import guarded_sim  # noqa: E402 - lives in tools/, imported lazily

    def work(session):
        records = []
        for group in groups.values():
            arm_settings = group["settings"]
            note("simulating %d arm(s) under %s/%s/%s"
                 % (len(group["arms"]), arm_settings.get("region"), arm_settings.get("universe"),
                    arm_settings.get("neutralization")))
            produced = guarded_sim.simulate(
                [expression for expression, _decay in group["pairs"]], arm_settings,
                labels=[arm["label"] for arm in group["arms"]], session=session,
                on_event=lambda kind, payload: note("  %s" % kind))
            for record in produced:
                # The ledger must carry the complete settings this arm actually used.
                record["settings"] = dict(arm_settings)
            records.extend(produced)
        return records

    records = with_session(args, work)

    rows = []
    note("")
    header = "%-28s %7s %6s %7s" % ("label", "trS", "trF", "trT")
    if args.show_test:
        header += " %7s %6s %7s %7s" % ("S", "F", "T", "testS")
    note(header)
    for record, arm in zip(records, [a for group in groups.values() for a in group["arms"]]):
        metrics = record.get("metrics") or {}
        split = record.get("out_of_sample") or {}
        train = split.get("train") or {}
        failed = sorted(name for name, body in (record.get("checks") or {}).items()
                        if isinstance(body, dict) and body.get("result") == "FAIL")
        # Exploration view: train period only. The platform's headline metrics use
        # the full window, which *contains* the test year, so ranking on them is
        # ranking on test - hiding one column would not have fixed that.
        row = {"label": arm["label"], "alpha_id": record.get("alpha_id"),
               "expression": arm["expression"],
               "decay": record.get("decay"), "settings": dict(record["settings"]),
               "status": record.get("status"), "error": record.get("error"),
               "train_sharpe": train.get("sharpe"),
               "train_fitness": train.get("fitness"),
               "train_turnover": train.get("turnover"),
               "train_returns": train.get("returns"),
               "failed_checks": failed}
        line = "%-28s %7s %6s %7s" % (arm["label"], row["train_sharpe"],
                                      row["train_fitness"], row["train_turnover"])
        if args.show_test:
            row.update({"sharpe": metrics.get("sharpe"), "fitness": metrics.get("fitness"),
                        "turnover": metrics.get("turnover"),
                        "test_sharpe": (split.get("test") or {}).get("sharpe")})
            line += " %7s %6s %7s %7s" % (row["sharpe"], row["fitness"],
                                          row["turnover"], row["test_sharpe"])
        rows.append(row)
        note(line + ("  fail=" + ",".join(failed) if failed else ""))

    Path(target).write_text(
        json.dumps({"spec": str(args.spec), "name": name, "rows": rows,
                    "records": records},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    if args.show_test:
        research_paths.log_view("probe", "test", [row.get("alpha_id") for row in rows],
                                note=name)
    note("wrote %s" % target)

    summary = {"name": name, "path": str(target), "rows": rows,
               "simulated": len(rows), "show_test": bool(args.show_test)}
    if not args.no_record:
        summary["registry"] = alpha_registry.record_simulation(
            records, settings, idea=spec.get("idea"), mechanism=spec.get("mechanism"),
            note=spec.get("note"), registry_path=args.registry, source="simulate")

    return summary


def cmd_views(args, _sink):
    """Read the log of every time out-of-sample numbers were looked at.

    The point is auditability, not prevention: a frozen validation and an iterative
    peek look identical in a results table, and only the log tells them apart.
    """
    path = research_paths.views_log_path()
    entries = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if args.alpha and args.alpha not in (entry.get("alpha_ids") or []):
                continue
            entries.append(entry)
    entries = entries[-args.limit:]
    return {"path": path, "count": len(entries),
            "by_tool": {tool: sum(1 for entry in entries if entry.get("tool") == tool)
                        for tool in sorted({entry.get("tool") for entry in entries if entry.get("tool")})},
            "entries": entries}


def _first_heading(path):
    """The document's title, or its first non-empty line."""
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text.startswith("#"):
                    return text.lstrip("#").strip()
                if text:
                    return text[:120]
    except OSError:
        return None
    return None


def cmd_brief(args, _sink):
    """The state of the campaign in one bounded document.

    Every session starts by reconstructing where the research stands, and the
    material for that is scattered: a multi-megabyte ledger, a directory of
    date-stamped write-ups, a hand-written weekly note, and whatever the last
    session remembered. Reading all of it is the expensive part of the work, and it
    grows every week while the answer stays the same size.

    So this emits the small part that moves - what the book is, what ran lately and
    how it scored, which write-ups are new, and the standing open questions - and
    nothing else. It is deliberately not a report; it is the input to one.

    The open-question list is read from ``open-threads.md`` in the research
    directory, one ``- [open] ...`` or ``- [closed] ...`` line each. That file is
    maintained by hand on purpose: a question worth carrying across sessions is a
    judgement, and anything derived automatically would be a summary of the past
    rather than a list of what is still undecided.
    """
    from datetime import datetime, timedelta, timezone

    def moment(entry):
        raw = entry.get("recorded_at") or ""
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return None

    days = max(0.0, args.days)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    registry_path = args.registry or alpha_registry.default_registry_path()
    entries = alpha_registry.load_entries(registry_path)

    def metric(entry, name):
        return (entry.get("metrics") or {}).get(name)

    book = {}
    for entry in entries:
        if (entry.get("status") or "").upper() != "ACTIVE":
            continue
        alpha_id = entry.get("alpha_id")
        if alpha_id and alpha_id not in book:
            book[alpha_id] = entry

    sharpes = [metric(entry, "sharpe") for entry in book.values()
               if isinstance(metric(entry, "sharpe"), (int, float))]
    families, universes = {}, {}
    for entry in book.values():
        key = entry.get("family") or "unlabeled"
        families[key] = families.get(key, 0) + 1
        universe = (entry.get("settings") or {}).get("universe") or "?"
        universes[universe] = universes.get(universe, 0) + 1

    recent = []
    for entry in entries:
        when = moment(entry)
        # A row with no timestamp cannot be said to be recent; including it would
        # make the window meaningless on a ledger that predates `recorded_at`.
        if when is not None and when >= cutoff:
            recent.append(entry)
    runs = [entry for entry in recent
            if (entry.get("source") or "") in {"simulate", "record", "submit"}]
    by_verdict = {}
    for entry in runs:
        verdict = entry.get("verdict") or "unlabeled"
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
    scored = [entry for entry in runs if isinstance(metric(entry, "sharpe"), (int, float))]
    scored.sort(key=lambda entry: -metric(entry, "sharpe"))
    best = [{"alpha_id": entry.get("alpha_id"), "sharpe": metric(entry, "sharpe"),
             "fitness": metric(entry, "fitness"),
             "verdict": entry.get("verdict"), "idea": entry.get("idea"),
             "when": entry.get("recorded_at")}
            for entry in scored[:args.top]]

    research_dir = Path(research_paths.ensure_dir())
    documents = []
    for path in sorted(research_dir.glob("*-results.md"),
                       key=lambda item: item.stat().st_mtime, reverse=True):
        written = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if written < cutoff:
            break
        documents.append({"file": path.name,
                          "when": written.isoformat(timespec="minutes"),
                          "title": _first_heading(path)})
        if len(documents) >= args.top * 2:
            break

    threads = {"open": [], "closed": []}
    thread_file = research_dir / "open-threads.md"
    if thread_file.exists():
        for line in thread_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            for tag, key in (("[open]", "open"), ("[closed]", "closed")):
                if text.startswith("- " + tag):
                    body = text[len(tag) + 2:].strip()
                    if body:
                        threads[key].append(body)

    # Label debt and recording gaps: both were invisible until someone counted by
    # hand, and both make the ledger look complete while it is not.
    unlabeled = sum(1 for entry in book.values()
                    if not entry.get("family") or not entry.get("mechanism"))
    no_checks = sum(1 for entry in entries
                    if (entry.get("metrics") or {}) and not (entry.get("checks") or {}))
    data = {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_days": days,
        "ledger": {"rows": len(entries), "path": registry_path,
                   "book_rows_missing_labels": unlabeled,
                   "rows_with_metrics_but_no_checks": no_checks},
        "book": {
            "count": len(book),
            "sharpe_median": (round(sorted(sharpes)[len(sharpes) // 2], 3)
                              if sharpes else None),
            "by_family": dict(sorted(families.items(), key=lambda item: -item[1])),
            "by_universe": dict(sorted(universes.items(), key=lambda item: -item[1])),
        },
        "recent": {"runs": len(runs), "by_verdict": by_verdict, "best": best},
        "documents": documents,
        "threads": threads,
    }

    return data


def _search_failure(exc):
    """Map a DeepSeek search error onto the CLI's documented codes."""
    if isinstance(exc, deepseek_search.MissingSearchKey):
        return CliError(
            "AUTH_MISSING", str(exc),
            "put DEEPSEEK_API_KEY in .env next to machine_lib.py, or export it; "
            "never paste the key into chat or a commit")
    if isinstance(exc, deepseek_search.SearchAuthError):
        return CliError("AUTH_FAILED", str(exc),
                        "check that the key is valid and the account can use the "
                        "Search API")
    if isinstance(exc, deepseek_search.SearchInsufficientBalance):
        return CliError("SERVICE_ERROR", str(exc),
                        "top up the DeepSeek account; Search and Fetch are free "
                        "until billing starts on 2026-11-01 Beijing time")
    if isinstance(exc, deepseek_search.SearchRateLimited):
        return CliError("RATE_LIMITED", str(exc), "wait a moment and retry")
    if isinstance(exc, deepseek_search.SearchBlocked):
        return CliError("NOT_FOUND", str(exc),
                        "the page was refused or had no usable text; try another "
                        "source")
    return CliError("SERVICE_ERROR", str(exc), "retry, or narrow the query")


def cmd_search(args, _sink):
    """Literature search or single-page fetch, quota-free on the BRAIN side.

    Kept in this CLI rather than a browser because the point is to connect a
    mechanism idea to the paper that tests it, and to leave a record of what was
    read. Results keep the provider's ranking and its ``rerank_score`` /
    ``authority`` / ``date`` fields so a journal article can be preferred over a
    blog restating it.
    """
    if not deepseek_search.available():
        raise _search_failure(deepseek_search.MissingSearchKey(
            "No DeepSeek key configured (DEEPSEEK_API_KEY)"))

    if args.fetch_url:
        try:
            page = deepseek_search.fetch(args.fetch_url)
        except deepseek_search.DeepSeekSearchError as exc:
            raise _search_failure(exc) from exc
        content = page.get("content") or ""
        data = {
            "mode": "fetch",
            "url": args.fetch_url,
            "title": page.get("title"),
            "content": content,
            "chars": len(content),
        }

        return data

    if not args.query:
        raise CliError("USAGE", "nothing to search",
                       "pass a query, or --fetch URL to read one page")

    cached = False
    try:
        if args.no_cache:
            results = deepseek_search.search(
                args.query, count=args.count, need_content=args.content)
        else:
            results, cached = deepseek_search.search_cached(
                research_paths.ensure_dir(), args.query, count=args.count,
                need_content=args.content, max_age_hours=args.max_age_hours)
    except deepseek_search.DeepSeekSearchError as exc:
        raise _search_failure(exc) from exc

    total = len(results)
    kept = deepseek_search.high_quality(
        results, min_score=args.min_score, min_authority=args.min_authority)
    data = {
        "mode": "search",
        "query": args.query,
        "total": total,
        "returned": len(kept),
        "cached": cached,
        "filters": {"min_score": args.min_score,
                    "min_authority": args.min_authority},
        "results": [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "summary": item.get("summary"),
                "site_name": item.get("site_name"),
                "date": item.get("date"),
                "rerank_score": item.get("rerank_score"),
                "authority": item.get("authority"),
                "content": item.get("content") if args.content else None,
            }
            for item in kept
        ],
        "note": ("rerank_score 1.5+ is high quality and authority 3+ is "
                 "authoritative; prefer the paper over posts about it"),
    }

    return data


def _preflight_lint(session, pairs, settings, args):
    """Lint against the operator catalog before spending simulation quota.

    Two catalogs are consulted, not one. The operator index catches
    ``regression_neut``-style names that have docs but cannot be simulated. The
    field index catches identifiers that were typed from memory and never
    existed, which the operator list cannot see because such a name is not an
    operator at all -- one such typo cost a full simulation round trip before
    this check existed.

    Raises ``INVALID_INPUT`` on a hard lint failure unless the caller passed
    ``--allow-lint-errors``; a clean pass is reported on stderr as an event so
    stdout stays a single JSON document.
    """
    operator_index = None
    catalog_note = "checked"
    try:
        operator_index = _operator_index_from_catalog(session)
    except Exception as exc:  # noqa: BLE001
        # The operator list is metadata. A hiccup there must not block a run that
        # is otherwise fine, but the degradation is announced rather than hidden
        # so nobody mistakes a partial lint for a full one.
        catalog_note = "unavailable: %s: %s" % (type(exc).__name__, exc)
        note("lint: operator catalog %s; continuing with the learned denylist only"
             % catalog_note)

    known_fields = None
    field_types = None
    market_fields = None
    field_note = "unavailable"
    try:
        # One fetch, three views. The dataset split feeds the turnover check: a construct
        # with no daily leg turned over 0.006-0.010 against a 0.01 floor, while the same
        # disclosure fields with such a leg cleared it comfortably and entered the book.
        # The list is kept here, visible, rather than buried in the linter.
        slow_datasets = {"fundamental2", "fundamental6", "analyst4"}
        records = field_index.fetch_all_fields(
            session,
            # Lint against the settings the simulation will actually use. Reading the
            # raw CLI flags instead meant a research file asking for TOP500 was still
            # linted against TOP3000: different field catalogue, so the check could
            # reject a valid field or wave an unavailable one through.
            region=settings.get("region") or ml.DEFAULT_SETTINGS["region"],
            universe=settings.get("universe") or ml.DEFAULT_SETTINGS["universe"],
            delay=settings.get("delay") if settings.get("delay") is not None else ml.DEFAULT_SETTINGS["delay"],
            instrument_type=settings.get("instrumentType") or ml.DEFAULT_SETTINGS["instrumentType"],
        )
        known_fields = field_index.field_ids(records)
        field_note = "checked (%d names)" % len(known_fields)
        # Field type decides whether an operator can read the field at all: a third
        # of the catalogue is VECTOR and most operators reject event inputs.
        field_types = {record.get("id"): record.get("type")
                       for record in records if record.get("id")}
        market_fields = set()
        for record in records:
            field_id = record.get("id")
            if not field_id:
                continue
            dataset = record.get("dataset")
            dataset_id = dataset.get("id") if isinstance(dataset, dict) else dataset
            if dataset_id not in slow_datasets:
                market_fields.add(field_id)
    except Exception as exc:  # noqa: BLE001
        field_note = "unavailable: %s: %s" % (type(exc).__name__, exc)
        note("lint: field catalog unavailable (%s); field names are NOT verified "
             "for this run, so a typo will only surface at the simulator" % exc)

    report = expression_lint.lint_all(
        pairs,
        operator_index=operator_index,
        denylist=expression_lint.load_denylist(),
        known_fields=known_fields,
        strict_fields=bool(known_fields),
        field_types=field_types,
        market_fields=market_fields,
        field_units=expression_units.load_field_units(),
    )
    report["operator_catalog_size"] = len(operator_index or {})
    report["operator_catalog"] = catalog_note
    report["field_catalog"] = field_note
    if report["ok"]:
        return report
    if getattr(args, "allow_lint_errors", False):
        return report
    unknown = sorted({item.get("operator") for item in report["errors"]
                      if item.get("code") == "unknown_operator"})
    unknown_fields = sorted({item.get("identifier") for item in report["errors"]
                             if item.get("code") == "unknown_identifier"})
    hint = "fix the expressions, or pass --allow-lint-errors to submit anyway"
    if unknown:
        hint = ("operator(s) %s are not in the BRAIN operator list; a docs page "
                "existing does not make them simulatable" % ", ".join(unknown))
    if unknown_fields:
        hint = ("field name(s) %s do not exist in the live catalog for this scope; "
                "run `catalog concepts --search <word>` to find the real variants "
                "instead of typing a name from memory" % ", ".join(unknown_fields))
    raise CliError(
        "INVALID_INPUT",
        "expression lint failed; nothing was submitted",
        hint,
        details={
            "operator_catalog_size": report["operator_catalog_size"],
            "field_catalog": field_note,
            "unknown_fields": unknown_fields,
            "errors": report["errors"],
            "warnings": report["warnings"],
        },
    )


def cmd_paper(args, _sink):
    """Paper search: the scholarly counterpart to `search`.

    `search` is a web search with reranking and, measured over three mechanism queries,
    returned zero to one academic domain in ten. This one talks to OpenAlex, so every row
    carries a journal, a year and a citation count, which is what makes a result worth
    chasing rather than inferring a formula from a blog post.
    """
    try:
        data = paper_search.search_papers(args.query, limit=args.limit,
                                          from_year=args.from_year,
                                          min_citations=args.min_citations)
    except ValueError as exc:
        raise CliError("INVALID_INPUT", str(exc), "query must not be empty") from exc
    except Exception as exc:  # noqa: BLE001 - network and DNS both land here
        raise CliError("SERVICE_ERROR", "OpenAlex request failed: %s" % exc,
                       "the API is public and needs no key; retry shortly") from exc
    return data


def cmd_simulate(args, sink):
    if not 1 <= args.batch_size <= 3:
        raise CliError("INVALID_INPUT", "--batch-size must be between 1 and 3")
    file_settings, alphas = collect_alphas(args)
    settings = merge_settings(file_settings, args)
    try:
        settings, pairs = config_io.build_config(settings, alphas)
    except config_io.ResearchError as exc:
        raise CliError("INVALID_INPUT", str(exc))
    labels = ml.alpha_labels(alphas, len(pairs))
    families = ml.alpha_families(alphas, len(pairs))
    candidates = ml.alpha_candidate_ids(alphas, len(pairs))

    if args.dry_run:
        records = ml.simulate_alphas(
            None, pairs, settings=settings, dry_run=True, labels=labels,
            families=families, candidates=candidates,
        )
        # Offline lint: catches structural problems and learned rejections. The
        # authoritative operator check needs the catalog, so it runs on `lint`
        # and on a real submission below, not here.
        offline_lint = expression_lint.lint_all(
            pairs,
            denylist=expression_lint.load_denylist(),
        )
        return {
            "status": "DRY_RUN",
            "dry_run": True,
            "settings": settings,
            "summary": {"total": len(records), "payloads": len(records)},
            "lint": {
                "operator_catalog_checked": False,
                "hint": "run `lint` for the authoritative operator check",
                "ok": offline_lint["ok"],
                "errors": offline_lint["errors"],
                "warnings": offline_lint["warnings"],
            },
            "results": records,
        }

    # Real submission: lint against the operator catalog first. An unknown
    # operator would otherwise burn a submission slot and a polling wait before
    # BRAIN rejected it -- exactly how `regression_neut` cost a round trip.
    with_session(args, lambda s: _preflight_lint(s, pairs, settings, args))

    checkpoint_path, checkpoint = _prepare_simulation_run(
        settings, pairs, args.batch_size, args.checkpoint,
        labels=labels, families=families, candidates=candidates,
    )
    on_event = _simulation_checkpoint_callback(checkpoint_path, checkpoint, sink)
    records = with_session(
        args,
        lambda s: ml.simulate_alphas(
            s,
            pairs,
            settings=settings,
            batch_size=args.batch_size,
            on_event=on_event,
            max_wait_seconds=args.max_wait_seconds,
            initial_records=checkpoint["records"],
            labels=labels,
            families=families,
            candidates=candidates,
        ),
    )
    return _finalize_simulation(
        records,
        settings,
        args,
        checkpoint_path,
        checkpoint,
        explicit=bool(args.checkpoint),
    )


def cmd_resume(args, sink):
    path = Path(args.checkpoint).expanduser().resolve()
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError("INVALID_INPUT", "cannot read simulation checkpoint", str(exc))
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != "worldquantuser.simulation-checkpoint"
        or checkpoint.get("version") != 1
        or not isinstance(checkpoint.get("records"), list)
    ):
        raise CliError("INVALID_INPUT", "unsupported or malformed simulation checkpoint")

    records = checkpoint["records"]
    try:
        pairs = [(record["expression"], record.get("decay")) for record in records]
    except (KeyError, TypeError) as exc:
        raise CliError("INVALID_INPUT", "checkpoint is missing alpha inputs", str(exc))
    # Keep the per-expression labels recorded in the checkpoint; a resumed run
    # must not silently drop the mechanism tags gathered before the pause.
    labels = [
        record.get("mechanism") if isinstance(record, dict) else None
        for record in records
    ]
    # Same for the candidate id: an ablation resumed halfway must keep the
    # variant label, or the contribution table comes out unlabelled.
    candidates = [
        record.get("candidate_id") if isinstance(record, dict) else None
        for record in records
    ]
    settings = checkpoint.get("settings") or ml.default_settings()
    batch_size = checkpoint.get("batch_size", 3)
    if not any(
        record.get("progress_url")
        or record.get("metrics_pending")
        or record.get("status") == "QUEUED"
        for record in records
        if isinstance(record, dict)
    ):
        return _finalize_simulation(
            records, settings, args, path, checkpoint, explicit=True
        )
    on_event = _simulation_checkpoint_callback(path, checkpoint, sink)
    records = with_session(
        args,
        lambda s: ml.simulate_alphas(
            s,
            pairs,
            settings=settings,
            batch_size=batch_size,
            on_event=on_event,
            max_wait_seconds=args.max_wait_seconds,
            initial_records=checkpoint["records"],
            labels=labels,
            candidates=candidates,
        ),
    )
    return _finalize_simulation(
        records,
        settings,
        args,
        path,
        checkpoint,
        explicit=True,
    )


def cmd_alpha_get(args, _sink):
    payload = with_session(args, lambda s: ml.fetch_alpha_metrics(s, args.alpha_id))
    in_sample = payload.get("is") or {}
    code, description, operator_count = ml.regular_expression(payload)
    document = {
        "alpha_id": payload.get("id", args.alpha_id),
        # Name, category and tags are what a submission carries and what makes an
        # alpha attributable later. They live at the top of the payload, not under
        # `is`, so they were invisible here - the submit path even refuses an alpha
        # without them, and the only way to see why was to read the raw payload.
        "name": payload.get("name"),
        "category": payload.get("category"),
        "tags": payload.get("tags") or [],
        "expression": code,
        "expression_description": description,
        "operator_count": operator_count,
        "settings": payload.get("settings"),
        "status": payload.get("status"),
        "date_created": payload.get("dateCreated"),
        "grade": payload.get("grade"),
        "metrics": {k: in_sample.get(k) for k in METRIC_KEYS if k in in_sample},
        "checks": ml._extract_checks(payload),
    }
    split = ml.extract_split(payload)
    if split:
        document["out_of_sample"] = split

    return document


def cmd_alpha_check(args, _sink):
    """Force the platform to compute checks and return the real values.

    ``alpha get`` reads the cached snapshot embedded in the alpha body, where
    SELF_CORRELATION is usually PENDING or absent. This command hits
    ``/alphas/{id}/check``, which runs the job (answering with Retry-After) and
    then returns the computed correlation.
    """
    if args.max_wait is not None and args.max_wait < 0:
        raise CliError("USAGE", "--max-wait must be non-negative",
                       "pass a number of seconds, or omit the flag for the default")
    result = with_session(args, lambda s: ml.fetch_alpha_checks(
        s, args.alpha_id, max_wait_seconds=args.max_wait))
    if not result.get("complete"):
        raise CliError(
            "SERVICE_ERROR",
            "platform did not finish computing checks within %.0fs" % (args.max_wait or 0),
            hint="Retry, or raise --max-wait; the job keeps running server-side.",
        )
    document = {
        "alpha_id": result["alpha_id"],
        "self_correlation": result.get("self_correlation"),
        "checks": result.get("checks"),
    }

    return document


def cmd_alpha_submit(args, _sink):
    """Submit an alpha, or show exactly what would be sent.

    Irreversible, so ``--yes`` is required; otherwise this behaves like the
    platform's "test submission", which is simply the check flow.
    """
    if args.max_wait is not None and args.max_wait < 0:
        raise CliError("USAGE", "--max-wait must be non-negative",
                       "pass a number of seconds, or omit the flag for the default")

    def work(session):
        # A submission carries metadata with it: the platform accepts a name, a
        # category and tags, and an alpha filed without them is unattributable later.
        # Checked before the submit call so a missing label cannot cost a submission.
        current = ml.fetch_alpha_metrics(session, args.alpha_id)
        missing = [field for field in ("name", "category") if not current.get(field)]
        if missing:
            raise CliError(
                "INVALID_INPUT",
                "%s has no %s" % (args.alpha_id, ", ".join(missing)),
                "set them first: alpha describe %s --name ... --category ..."
                % args.alpha_id)
        result = ml.submit_alpha(session, args.alpha_id, confirm=args.yes,
                                 max_wait_seconds=args.max_wait)
        if args.record and result.get("submitted"):
            # Read the alpha back rather than trusting the submit response: the
            # endpoint answers 201 before the checks finish, and the ledger wants what
            # the platform ended up with.
            result["recorded"] = _record_submitted_alpha(session, args.alpha_id)
        return result

    result = with_session(args, work)

    return result


def _record_submitted_alpha(session, alpha_id):
    """Append a normalised row for one alpha to the ledger and return what was written.

    Two payload shapes are handled here because getting them wrong is not a cosmetic
    mistake: `regular` is a plain string for some alphas and a `{code, description,
    operatorCount}` object for others, and the metrics live under `is`, not `metrics`.
    Reading either one naively recorded a row with no metrics on 2026-11-01, which the
    ledger then treated as the newest truth.
    """
    current = ml.fetch_alpha_metrics(session, alpha_id)
    code, _description, _count = ml.regular_expression(current)
    checks = ml.fetch_alpha_checks(session, alpha_id, max_wait_seconds=60).get("checks")
    entry = alpha_registry.build_entry(
        {"alpha_id": alpha_id,
         "expression": code,
         "settings": current.get("settings") or {},
         "metrics": current.get("is") or {},
         "checks": checks,
         "status": current.get("status"),
         "out_of_sample": {"train": current.get("train") or {},
                           "test": current.get("test") or {},
                           "test_period": "P1Y"}},
        verdict=alpha_registry.auto_verdict({"status": current.get("status"),
                                             "metrics": current.get("is") or {},
                                             "checks": checks}),
        note="recorded by `alpha submit --record`: dateSubmitted %s"
             % (current.get("dateSubmitted"),),
        source="submit")
    stored, written = alpha_registry.record_entry(entry)
    # Pull the new alpha's daily PnL into the local store as part of recording. Every
    # local correlation scan reads <research>/pnl/, so a submission that is not cached
    # here is invisible to the next scan -- and the scan is the cheap way to see what
    # the submission just displaced.
    cached = None
    try:
        pnl_dir = Path(research_paths.ensure_dir()) / "pnl"
        series = ml.fetch_daily_pnl(session, alpha_id, cache_dir=pnl_dir,
                                    max_age_hours=0)
        cached = len(series or {})
    except Exception as error:  # noqa: BLE001 - a cache miss must not fail the record
        cached = "unavailable: %s" % str(error)[:60]
    return {"key": stored.get("key"), "status": stored.get("status"),
            "written": written, "metrics": bool(stored.get("metrics")),
            "pnl_cached": cached}


def cmd_alpha_list(args, _sink):
    alphas = with_session(args, lambda s: ml.list_alphas(
        s, limit=args.limit, offset=args.offset, order=args.order,
        status=args.status, region=args.region, universe=args.universe,
        min_sharpe=args.min_sharpe, min_fitness=args.min_fitness, search=args.search,
        hidden=args.all, alpha_type_not="" if args.all else "SUPER",
    ))
    compact = []
    for alpha in alphas:
        in_sample = alpha.get("is") or {}
        settings = alpha.get("settings") or {}
        code, description, operator_count = ml.regular_expression(alpha)
        compact.append({
            "alpha_id": alpha.get("id"),
            "expression": code,
            "expression_description": description,
            "operator_count": operator_count,
            "sharpe": in_sample.get("sharpe"),
            "fitness": in_sample.get("fitness"),
            "turnover": in_sample.get("turnover"),
            "region": settings.get("region"),
            "universe": settings.get("universe"),
            "neutralization": settings.get("neutralization"),
            "status": alpha.get("status"),
            "grade": alpha.get("grade"),
            "date_created": alpha.get("dateCreated"),
        })
    document = {"count": len(compact), "offset": args.offset, "alphas": compact}

    return document


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _dispatch(args):
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            args._parser.error("--%s must be finite" % name.replace("_", "-"))
        if name in {"limit", "offset", "top", "days", "tail_tokens", "min_members",
                    "min_shared", "min_overlap", "peer_limit"} and value is not None and value < 0:
            raise CliError("INVALID_INPUT", "--%s must be zero or greater" % name.replace("_", "-"),
                           "use a non-negative value")
    if not getattr(args, "func", None):
        args.command = getattr(args, "command", None) or "help"
        return _help_document(args._parser)
    sink = make_event_sink(getattr(args, "stream", False), getattr(args, "verbose", False))
    ml.set_diagnostic_handler(note)
    ml.configure_metadata_cache(
        max_age=getattr(args, "meta_max_age", None),
        refresh=getattr(args, "refresh_meta", None),
    )
    return args.func(args, sink)


# ---------------------------------------------------------------------------
# lint -- expression pre-flight
# ---------------------------------------------------------------------------

def _collect_lint_pairs(args):
    """Return ``(settings, [(expression, decay), ...])`` for the lint target."""
    file_settings, alphas = collect_alphas(args)
    settings = merge_settings(file_settings, args)
    try:
        return config_io.build_config(settings, alphas)
    except config_io.ResearchError as exc:
        raise CliError("INVALID_INPUT", str(exc)) from exc


def _operator_index_from_catalog(session):
    return expression_lint.build_operator_index(ml.fetch_operators(session))


#: Datasets whose fields update on a disclosure schedule rather than with the market.
#:
#: "Disclosure" describes the release cadence, not the value. Three of the book's entries
#: are built entirely from these fields; what they pair with is a leg that reprices daily.
#: The turnover check uses this list to notice a construct with no such leg - which is a
#: property of the expression, and fixable by adding one, not a reason to avoid the data.
DISCLOSURE_DATASETS = frozenset({"fundamental2", "fundamental6", "analyst4"})


def _dataset_id(record):
    """Dataset id from a catalogue record, which may nest it or not."""
    dataset = record.get("dataset")
    return dataset.get("id") if isinstance(dataset, dict) else dataset


def _field_scope(session, settings):
    """Return ``(known_field_ids, market_field_ids)`` from one catalogue fetch.

    Two views on purpose: the first answers "is this a real field", the second "does
    anything here move with the market". Fetching twice would let them disagree.
    """
    records = field_index.fetch_all_fields(
        session,
        region=settings["region"],
        universe=settings["universe"],
        delay=settings["delay"],
        instrument_type=settings["instrumentType"],
    )
    known = field_index.field_ids(records)
    market = {record.get("id") for record in records
              if record.get("id") and _dataset_id(record) not in DISCLOSURE_DATASETS}
    return known, market


def cmd_lint(args, sink):
    settings, pairs = _collect_lint_pairs(args)
    denylist = expression_lint.load_denylist(getattr(args, "denylist", None))
    operator_index = None
    known_fields = None
    market_fields = None
    catalog_checked = False
    if not getattr(args, "offline", False):
        def work(session):
            known, market = _field_scope(session, settings)
            return _operator_index_from_catalog(session), known, market

        operator_index, known_fields, market_fields = with_session(args, work)
        catalog_checked = operator_index is not None

    results = []
    for expression, decay in pairs:
        report = expression_lint.lint_expression(
            expression,
            operator_index=operator_index,
            denylist=denylist,
            known_fields=known_fields,
            strict_fields=getattr(args, "strict_fields", False),
            market_fields=market_fields,
            field_units=expression_units.load_field_units(),
        )
        report["expression"] = expression
        if decay is not None:
            report["decay"] = decay
        results.append(report)

    failed = [item for item in results if not item["ok"]]
    document = {
        "expressions": len(results),
        "ok": not failed,
        "catalog_checked": catalog_checked,
        "operator_catalog_size": len(operator_index or {}),
        "denylist_size": len(denylist),
        "known_field_count": None if known_fields is None else len(known_fields),
        "settings": settings or None,
        "summary": {
            "errors": sum(len(item["errors"]) for item in results),
            "warnings": sum(len(item["warnings"]) for item in results),
        },
        "results": results,
    }

    if failed:
        raise CliError(
            "INVALID_INPUT",
            "%d of %d expression(s) failed lint" % (len(failed), len(results)),
            "see data.results[].errors; `unknown_operator` means the operator is "
            "not in the BRAIN operator list even when a documentation page exists",
            details=document,
        )
    return document


# ---------------------------------------------------------------------------
# catalog field-audit -- field contract before combining fields
# ---------------------------------------------------------------------------

def _fetch_raw_fields(session, args, dataset=None, search=""):
    return ml.get_datafields(
        session,
        instrument_type=getattr(args, "instrument_type", ml.DEFAULT_SETTINGS["instrumentType"]),
        region=getattr(args, "region", ml.DEFAULT_SETTINGS["region"]),
        delay=getattr(args, "delay", ml.DEFAULT_SETTINGS["delay"]),
        universe=getattr(args, "universe", ml.DEFAULT_SETTINGS["universe"]),
        dataset_id=dataset or "",
        search=search,
        limit=getattr(args, "limit", None),
    )


def _resolve_fields_by_id(session, args, field_ids):
    """Locate raw catalog records for specific field ids.

    Tries ``--dataset`` first, then walks the region's datasets until every id
    is found. Dataset listing and field listing are both quota-free, so a miss
    costs a request rather than simulation quota.
    """
    wanted = list(field_ids)
    found = []
    if getattr(args, "dataset", None):
        records = _fetch_raw_fields(session, args, dataset=args.dataset)
        by_id = {record.get("id"): record for record in records}
        found = [by_id[field_id] for field_id in wanted if field_id in by_id]
        remaining = [field_id for field_id in wanted if field_id not in by_id]
        if not remaining:
            return found, []
        wanted = remaining

    datasets = ml.get_datasets(
        session,
        instrument_type=getattr(args, "instrument_type", ml.DEFAULT_SETTINGS["instrumentType"]),
        region=getattr(args, "region", ml.DEFAULT_SETTINGS["region"]),
        delay=getattr(args, "delay", ml.DEFAULT_SETTINGS["delay"]),
        universe=getattr(args, "universe", ml.DEFAULT_SETTINGS["universe"]),
    )
    remaining = set(wanted)
    for dataset in datasets:
        dataset_id = dataset.get("id")
        if not dataset_id or not remaining:
            continue
        records = _fetch_raw_fields(session, args, dataset=dataset_id)
        for record in records:
            field_id = record.get("id")
            if field_id in remaining:
                found.append(record)
                remaining.discard(field_id)
        if not remaining:
            break
    order = {field_id: index for index, field_id in enumerate(wanted)}
    found.sort(key=lambda record: order.get(record.get("id"), len(order)))
    return found, sorted(remaining)


def cmd_catalog_field_audit(args, sink):
    field_ids = [item.strip()
                 for item in (getattr(args, "fields", "") or "").split(",")
                 if item.strip()]
    if not field_ids and not getattr(args, "dataset", None):
        raise CliError(
            "USAGE", "field-audit needs --fields and/or --dataset",
            "e.g. `catalog field-audit --fields fnd6_newa1v1300_capx,"
            "fnd6_newa2v1300_oancf --operation ratio`")

    def work(session):
        if field_ids:
            return _resolve_fields_by_id(session, args, field_ids)
        return _fetch_raw_fields(session, args), []

    raw_records, missing = with_session(args, work)
    if not raw_records:
        raise CliError(
            "NOT_FOUND", "no matching datafields",
            "check the field ids and the region/universe/delay; field listing is "
            "quota-free, so a wrong id costs a retry rather than a simulation")

    document = field_contract.audit_fields(
        raw_records, operation=getattr(args, "operation", None))
    document["catalog"] = {
        "region": getattr(args, "region", ml.DEFAULT_SETTINGS["region"]),
        "universe": getattr(args, "universe", ml.DEFAULT_SETTINGS["universe"]),
        "delay": getattr(args, "delay", ml.DEFAULT_SETTINGS["delay"]),
        "instrument_type": getattr(args, "instrument_type", ml.DEFAULT_SETTINGS["instrumentType"]),
    }
    if field_ids:
        document["requested_fields"] = field_ids
    if missing:
        document["missing_fields"] = missing

    return document


# ---------------------------------------------------------------------------
# ledger -- the durable local research history
# ---------------------------------------------------------------------------

def _record_simulation(records, settings, args):
    """Append finished simulation records to the registry.

    The mapping lives in :func:`alpha_registry.record_simulation`, because the
    research template records the same records and had been copying this body by
    hand - or, more often, skipping it.
    """
    return alpha_registry.record_simulation(
        records, settings,
        idea=getattr(args, "idea", None),
        mechanism=getattr(args, "mechanism", None),
        family=getattr(args, "family", None),
        note=getattr(args, "note", None),
        registry_path=getattr(args, "registry", None),
        source="simulate")


def cmd_ledger_record(args, sink):
    alpha_id = getattr(args, "alpha_id", None)
    if not alpha_id:
        raise CliError(
            "USAGE", "ledger record needs an alpha id",
            "use `alpha list` to find one; a record without an alpha id has no "
            "metrics to persist")

    def work(session):
        return ml.fetch_alpha_metrics(session, alpha_id)

    payload = with_session(args, work)
    in_sample = payload.get("is") or {}
    expression, _, _ = ml.regular_expression(payload)
    entry = alpha_registry.build_entry(
        {
            "alpha_id": payload.get("id") or alpha_id,
            "expression": expression,
            "settings": payload.get("settings") or {},
            "metrics": in_sample,
            "checks": in_sample.get("checks") or {},
            "status": payload.get("status") or "UNSUBMITTED",
            "out_of_sample": ml.extract_split(payload),
        },
        idea=getattr(args, "idea", None),
        mechanism=getattr(args, "mechanism", None),
        verdict=getattr(args, "verdict", "watch"),
        note=getattr(args, "note", None),
        family=getattr(args, "family", None),
        source="record",
    )
    registry_path = (getattr(args, "registry", None)
                     or alpha_registry.default_registry_path())
    stored, written = alpha_registry.record_entry(entry, registry_path)
    return {
        "registry": registry_path,
        "appended": written,
        "entry": stored,
    }


def cmd_ledger_list(args, _sink):
    registry_path, entries = _ledger(args)
    limit = getattr(args, "limit", 0)
    if limit is not None and limit < 0:
        raise CliError("INVALID_INPUT", "--limit must be zero or positive",
                       "use --limit 0 to show all matching entries")
    matched = alpha_registry.find_entries(
        entries,
        field=getattr(args, "field", None),
        mechanism=getattr(args, "mechanism", None),
        idea=getattr(args, "idea", None),
        expression=getattr(args, "expression", None),
        verdicts=getattr(args, "verdict", None),
        min_fitness=getattr(args, "min_fitness", None),
        family=getattr(args, "family", None),
        min_sharpe=getattr(args, "min_sharpe", None),
    )

    def fitness_key(entry):
        value = (entry.get("metrics") or {}).get("fitness")
        return value if isinstance(value, (int, float)) else float("-inf")

    matched.sort(key=fitness_key, reverse=True)
    shown = matched[:limit] if limit else matched
    verdict_counts, mechanisms, families = {}, {}, {}
    for entry in matched:
        verdict = entry.get("verdict") or "unknown"
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        mechanism = entry.get("mechanism") or "unlabelled"
        mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1
        family = entry.get("family") or "unclassified"
        families[family] = families.get(family, 0) + 1
    document = {
        "registry": registry_path,
        "registry_entries": len(entries),
        "matched": len(matched),
        "shown": len(shown),
        "query": {
            "field": getattr(args, "field", None),
            "mechanism": getattr(args, "mechanism", None),
            "idea": getattr(args, "idea", None),
            "expression": getattr(args, "expression", None),
            "verdict": list(getattr(args, "verdict", None) or []),
            "family": getattr(args, "family", None),
            "min_fitness": getattr(args, "min_fitness", None),
            "min_sharpe": getattr(args, "min_sharpe", None),
        },
        "verdict_counts": verdict_counts,
        "mechanism_counts": mechanisms,
        "family_counts": families,
        "summary": alpha_registry.summarize(matched),
        "already_tested": bool(matched),
        "entries": shown,
    }
    if getattr(args, "format", "json") == "markdown":
        document["markdown"] = alpha_registry.render_markdown(shown, title="Research ledger")
    return document


def _ledger(args):
    path = (getattr(args, "registry", None)
            or alpha_registry.default_registry_path())
    return path, alpha_registry.load_entries(path)


def cmd_ledger_label(args, _sink):
    """Rewrite the judgement columns of one entry.

    Appends rather than edits: the ledger is append-only and the newest record for
    a key wins, so a relabel keeps the earlier judgement visible.

    The response contains the stored ``entry`` and whether a new history row was
    ``written``, alongside the changed judgement columns and their earlier values.
    """
    path, entries = _ledger(args)
    target = alpha_registry.resolve(entries, args.alpha_id)
    if target is None:
        raise CliError("NOT_FOUND", "no ledger entry for %s" % args.alpha_id,
                       "run `ledger list` to see what is recorded")
    if all(getattr(args, name, None) is None
           for name in ("family", "verdict", "mechanism", "idea", "note")):
        raise CliError("USAGE", "nothing to change",
                       "pass at least one of --family/--verdict/--mechanism/"
                       "--idea/--note")
    try:
        updated = alpha_registry.relabel(
            target, family=args.family, verdict=args.verdict,
            mechanism=args.mechanism, idea=args.idea, note=args.note)
    except ValueError as exc:
        raise CliError("INVALID_INPUT", str(exc)) from exc
    stored, written = alpha_registry.record_entry(updated, path)
    data = {
        "registry": path,
        "written": written,
        "entry": stored,
        "key": stored.get("key"),
        "alpha_id": stored.get("alpha_id"),
        "family": stored.get("family"),
        "verdict": stored.get("verdict"),
        "mechanism": stored.get("mechanism"),
        "before": {"family": target.get("family"),
                   "verdict": target.get("verdict"),
                   "mechanism": target.get("mechanism")},
    }

    return data


def cmd_alpha_explain(args, _sink):
    """Explain one alpha: what each leg computes, and what the evidence says.

    Three layers, kept apart on purpose, because mixing them is how a description
    starts to sound like a justification:

    * **construction** -- what the expression does, leg by leg, in words. Read off the
      tree, so it cannot drift from the formula.
    * **evidence** -- the metrics, where the Fitness comes from arithmetically, the
      measured ablations if any were run, and the local correlation with the book.
    * **risk** -- whether the family is on the closed list, and what the leg weights do
      *not* tell you (a weight is a multiplier, not a contribution).
    """
    import alpha_unit as au

    _path, entries = _ledger(args)
    target = alpha_registry.resolve(entries, args.alpha_id)
    if target is None:
        raise CliError("NOT_FOUND", "no ledger entry for %s" % args.alpha_id,
                       "run `alpha list` or `alpha get` to find the id")

    point = au.split(target.get("expression") or "",
                     datasets=au.load_field_datasets())
    if point is None:
        raise CliError("INVALID_INPUT", "%s does not parse" % args.alpha_id,
                       "the ledger row may be damaged; `alpha get` shows the raw text")

    explanation = au.explain(point)
    metrics = target.get("metrics") or {}
    settings = target.get("settings") or {}

    # The arithmetic that decides whether the score is earned or borrowed.
    sharpe, turnover = metrics.get("sharpe"), metrics.get("turnover")
    returns, fitness = metrics.get("returns"), metrics.get("fitness")
    denominator = None
    if isinstance(turnover, (int, float)):
        denominator = max(turnover, 0.125)
    implied = None
    if isinstance(sharpe, (int, float)) and isinstance(returns, (int, float)) and denominator:
        implied = round(sharpe * (returns / denominator) ** 0.5, 3)

    ablations = []
    for entry in entries:
        idea = entry.get("idea") or ""
        if idea.endswith(" of %s" % args.alpha_id) or idea.endswith("of %s" % args.alpha_id):
            m = entry.get("metrics") or {}
            ablations.append({"label": idea.split(":")[-1].split(" of ")[0].strip(),
                              "alpha_id": entry.get("alpha_id"),
                              "sharpe": m.get("sharpe"), "fitness": m.get("fitness")})

    gate = None if args.no_corr else _local_book_gate(args.alpha_id, entries)

    document = {
        "alpha_id": args.alpha_id,
        "status": target.get("status"),
        "family": target.get("family"),
        "settings": {k: settings.get(k) for k in
                     ("universe", "neutralization", "decay", "truncation", "delay")},
        "construction": explanation,
        "metrics": {k: metrics.get(k) for k in
                    ("sharpe", "fitness", "turnover", "returns", "drawdown")},
        "fitness_arithmetic": {
            "formula": "sharpe * sqrt(returns / max(turnover, 0.125))",
            "denominator_used": denominator,
            "below_turnover_floor": bool(isinstance(turnover, (int, float))
                                         and turnover < 0.125),
            "implied_fitness": implied,
            # The gap is the integrity check this block exists for: a row whose
            # stored Fitness cannot be rebuilt from its own Sharpe/returns/turnover
            # is stale or damaged, and every ranking built on it inherits that.
            "reported_fitness": fitness,
            "gap": (round(fitness - implied, 3)
                    if isinstance(fitness, (int, float)) and implied is not None
                    else None),
        },
        "ablations_measured": ablations,
        "book_correlation": gate,
        "closed_family": _closed_reason(target.get("family")),
        "caveat": "a leg weight is a multiplier, not a contribution; which leg carries "
                  "the alpha is only known if an ablation was run",
    }

    return document


def _local_book_gate(alpha_id, entries):
    """Daily-correlation of one alpha against the book, computed from cached PnL.

    Measured against the platform on 2026-11-01: median difference 0.003, maximum
    0.023. That is close enough to screen and not close enough to submit -- anything
    above 0.67 needs the platform's own answer.
    """
    from pathlib import Path

    import correlation

    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"
    own = pnl_dir / ("%s.json" % alpha_id)
    if not own.exists():
        return {"available": False, "reason": "no cached PnL for this alpha"}
    book = {e.get("alpha_id") for e in entries
            if (e.get("status") or "").upper() == "ACTIVE" and e.get("alpha_id")}
    try:
        with open(own) as handle:
            target = json.load(handle)
    except (OSError, ValueError):
        return {"available": False, "reason": "cached PnL is unreadable"}
    scores, missing = [], 0
    for peer in sorted(book):
        if peer == alpha_id:
            continue
        path = pnl_dir / ("%s.json" % peer)
        if not path.exists():
            missing += 1
            continue
        try:
            with open(path) as handle:
                series = json.load(handle)
        except (OSError, ValueError):
            missing += 1
            continue
        x, y, _dates = correlation.align(target, series)
        value = correlation.pearson(x, y)
        if value is not None:
            scores.append((value, peer))
    if not scores:
        return {"available": False, "reason": "no overlapping book PnL"}
    best = max(scores, key=lambda pair: abs(pair[0]))
    return {"available": True, "max": round(best[0], 4), "peer": best[1],
            "peers": len(scores), "missing_pnl": missing,
            "verdict": ("clear" if abs(best[0]) < 0.67 else
                        "verify on the platform (>= 0.67)"),
            "method": "daily Pearson on cached PnL"}


def _closed_reason(family):
    try:
        import closed_families
        reason = closed_families.reason(family) if family else None
    except Exception:  # noqa: BLE001 - an explain must work without the file
        return None
    return reason


def cmd_ledger_impact(args, _sink):
    """Alphas that share a leg body with the given one."""
    _path, entries = _ledger(args)
    target = alpha_registry.resolve(entries, args.alpha_id)
    if target is None:
        raise CliError("NOT_FOUND", "no ledger entry for %s" % args.alpha_id)
    bodies = alpha_registry.leg_bodies(target)
    rows = alpha_registry.shared_legs(entries, args.alpha_id,
                                      min_shared=args.min_shared)
    data = {
        "alpha_id": target.get("alpha_id"),
        "key": target.get("key"),
        "legs": [{"name": name, "body": body}
                 for name, body in bodies.items()],
        "shared_count": len(rows),
        "rows": rows[: args.limit],
        "note": ("legs are matched on the sub-expression body; a shared letter "
                 "means nothing on its own"),
    }

    return data


def cmd_ledger_lineage(args, _sink):
    """Parent chain for one alpha."""
    _path, entries = _ledger(args)
    chain = alpha_registry.lineage(entries, args.alpha_id)
    if not chain:
        raise CliError("NOT_FOUND", "no ledger entry for %s" % args.alpha_id)
    children = alpha_registry.descendants(entries, args.alpha_id)
    data = {
        "alpha_id": args.alpha_id,
        "depth": len(chain),
        "chain": [
            {"key": item.get("key"), "alpha_id": item.get("alpha_id"),
             "op": item.get("op"), "verdict": item.get("verdict"),
             "family": item.get("family"),
             "fitness": (item.get("metrics") or {}).get("fitness")}
            for item in chain
        ],
        "children": [{"key": item.get("key"), "alpha_id": item.get("alpha_id")}
                     for item in children],
        "note": ("lineage is recorded, never inferred; an empty op means the "
                 "record predates lineage capture"),
    }

    return data


def cmd_ledger_coverage(args, _sink):
    """How far the ledger is from having a countable family column."""
    _path, entries = _ledger(args)
    summary = alpha_registry.unlabeled_summary(entries)
    data = {"summary": summary}
    if args.list_missing:
        missing = [entry for entry in entries if not entry.get("family")]
        missing.sort(key=lambda entry: entry.get("recorded_at") or "", reverse=True)
        data["missing"] = [
            {"key": entry.get("key"), "alpha_id": entry.get("alpha_id"),
             "verdict": entry.get("verdict"),
             "fitness": (entry.get("metrics") or {}).get("fitness"),
             "mechanism": entry.get("mechanism")}
            for entry in missing[: args.limit]
        ]

    return data


def cmd_alpha_self_corr(args, _sink):
    """Report the platform's self-correlation and the substitute route's bar."""
    payload = with_session(
        args, lambda s: ml.fetch_self_correlation(s, args.alpha_id))
    verdict = ml.self_correlation_verdict(payload, limit=args.limit,
                                          sharpe_margin=args.sharpe_margin)
    current = with_session(args, lambda s: ml.fetch_alpha_metrics(s, args.alpha_id))
    sharpe = (current.get("is") or {}).get("sharpe")
    needed = verdict.get("sharpe_needed")
    qualifies = None
    if verdict["blocked"] and needed is not None and isinstance(sharpe, (int, float)):
        qualifies = sharpe >= needed
    data = {
        "alpha_id": args.alpha_id,
        "sharpe": sharpe,
        "max_correlation": payload.get("max"),
        "limit": args.limit,
        "sharpe_margin": args.sharpe_margin,
        "sharpe_needed": needed,
        "qualifies_via_sharpe": qualifies,
        "verdict": ("clear" if not verdict["blocked"]
                    else ("sharpe route" if qualifies else "blocked")),
        "counterparts": verdict["counterparts"],
    }

    return data


def cmd_alpha_describe(args, _sink):
    """Write an alpha's metadata and verify it stuck."""
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    if not any((args.name, args.category, tags, args.description)):
        raise CliError("USAGE", "nothing to set",
                       "pass --name, --category, --tags and/or --description")

    def work(session):
        return ml.set_alpha_metadata(session, args.alpha_id, name=args.name,
                                     category=args.category,
                                     tags=tags or None,
                                     description=args.description)

    current = with_session(args, work)
    data = {
        "alpha_id": current.get("id"),
        "name": current.get("name"),
        "category": current.get("category"),
        "tags": current.get("tags"),
        "verified": True,
        "description": (current.get("regular") or {}).get("description"),
        "note": "name, category and tags are top-level; description is nested under "
                "`regular`",
    }

    return data


def _rank_peers(scored):
    """Order peers by absolute correlation, largest first.

    The platform's limit is on ``|r|``: a candidate that mirrors an existing alpha
    has a large negative correlation and is blocked exactly as hard as a positive
    one. Sorting on the signed value put such a peer last and reported the
    runner-up as the counterpart.
    """
    return sorted(scored, key=lambda item: -abs(item.get("correlation") or 0.0))


def _independence_state(flagged, skipped, measured):
    """Three states, because "no measured pair cleared the line" is not independence.

    A pair whose overlap is too short lands in ``skipped``. Treating that as a pass
    is absence of evidence read as evidence of absence, and the unmeasurable pair is
    the one most likely to be a duplicate.
    """
    if flagged:
        return "dependent"
    if skipped or not measured:
        return "unknown"
    return "independent"


def _verdict_with_gaps(verdict, gaps):
    """A gap can only hide a block, never a clear.

    ``clear`` means every peer was measured and none came close. When a peer's
    PnL is missing, too short, or flat, that statement is not supported: the
    pair nobody could measure is the likeliest duplicate, so ``clear`` degrades
    to ``unknown`` instead of surviving on the pairs that did work.

    ``blocked`` and ``borderline`` already mean do not submit, and a missing
    measurement can only add a reason, never remove one, so they stand.

    Returns ``(verdict, note)``; the note names the gaps whenever there are any,
    including when the verdict was already ``unknown`` because nothing at all
    could be measured - that case is a gap too, and the least informative one.
    """
    if not gaps:
        return verdict, None
    counts = {}
    for gap in gaps:
        counts[gap["reason"]] = counts.get(gap["reason"], 0) + 1
    note = ("%d peer(s) could not be measured (%s)"
            % (len(gaps), ", ".join("%s %d" % (reason, count)
                                    for reason, count in sorted(counts.items()))))
    if verdict == "clear":
        return "unknown", (note + "; an unmeasured pair is the one most likely to be a "
                                   "duplicate, so `clear` is not claimed")
    return verdict, note


def cmd_alpha_self_check(args, sink):
    """Reproduce the platform's self-correlation check locally, and better.

    Three measured facts shape this command.

    Daily Pearson matches the platform's SELF_CORRELATION within 0.006 to 0.016 and always
    slightly below it; weekly and monthly overstate it by 0.03 to 0.08. Daily is therefore
    the number that decides, and the others are kept as diagnostics.

    The platform compares against a periodically refreshed snapshot of the book, not a live
    one. Two alphas submitted minutes apart were each accepted while correlating 0.975 with
    the other. A local check that looks only at ACTIVE alphas repeats that blind spot, so
    recent submissions - including ones not submitted yet - are included by default.

    A published platform value is historical. To compare against it, the peer set has to be
    the book as it was, which is what ``--as-of`` does. Without it, an alpha rejected last
    night appears to correlate 0.99 with something that entered afterwards, and the apparent
    disagreement is only a difference in what the platform could see.

    Every peer that yields no number is named with a reason - no PnL, too little overlap,
    or an undefined Pearson because one series is flat. A flat series is the sharp case:
    ``round(None, 4)`` used to be a TypeError, and the tempting fix, dropping the pair,
    would read as a measured zero. ``_verdict_with_gaps`` says why a gap downgrades
    ``clear`` to ``unknown`` and leaves ``blocked`` alone.
    """
    from datetime import datetime, timedelta, timezone

    def parse(moment):
        if not moment:
            return None
        try:
            return datetime.fromisoformat(str(moment).replace("Z", "+00:00"))
        except ValueError:
            return None

    cutoff = parse(args.as_of) if args.as_of else None
    if args.as_of and cutoff is None:
        raise CliError("INVALID_INPUT", "cannot read --as-of %r" % args.as_of,
                       "use an ISO timestamp such as 2026-10-23T00:15:00+08:00")
    horizon = (datetime.now(timezone.utc)
               - timedelta(hours=max(0.0, args.include_unsent_hours)))

    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"

    def collect(session):
        candidate = ml.fetch_daily_pnl(session, args.alpha_id, cache_dir=pnl_dir,
                                       max_age_hours=args.max_age_hours)
        if not candidate:
            raise CliError("NOT_FOUND", "no daily PnL for %s" % args.alpha_id,
                           "the alpha must have finished simulating")
        record = ml.fetch_alpha_metrics(session, args.alpha_id)
        peers, excluded = [], []
        seen = set()
        statuses = ["ACTIVE"]
        if args.include_unsent_hours > 0:
            statuses.append("UNSUBMITTED")
        for status in statuses:
            for entry in ml.list_alphas(session, limit=args.peer_limit, status=status,
                                        alpha_type_not=""):
                peer_id = entry.get("id")
                if not peer_id or peer_id == args.alpha_id:
                    continue
                when = parse(entry.get("dateSubmitted")) or parse(entry.get("dateCreated"))
                if cutoff is not None and when is not None and when > cutoff:
                    excluded.append({"alpha_id": peer_id, "when": entry.get("dateSubmitted")
                                     or entry.get("dateCreated")})
                    continue
                if status == "UNSUBMITTED" and when is not None and when < horizon:
                    continue
                if peer_id in seen:
                    # An alpha submitted minutes ago is ACTIVE and was also created
                    # inside the unsubmitted window, so it arrives twice and would be
                    # counted, correlated and gap-reported twice.
                    continue
                seen.add(peer_id)
                peers.append({"alpha_id": peer_id, "when": when,
                              "status": status})
        series = {}
        for peer in peers:
            data = ml.fetch_daily_pnl(session, peer["alpha_id"], cache_dir=pnl_dir,
                                      max_age_hours=args.max_age_hours)
            if data:
                series[peer["alpha_id"]] = data
        return candidate, record, peers, series, excluded

    candidate, record, peers, series, excluded = with_session(args, collect)

    # Every peer that produced no number is named with a reason. The crash this
    # replaces was `round(None, 4)` on a flat PnL series; the quieter defect was
    # that a peer with no PnL at all was skipped without a word, so the maximum
    # was a maximum over whatever happened to be measurable.
    gaps = []
    usable = []
    for peer in peers:
        if series.get(peer["alpha_id"]):
            usable.append(peer)
        else:
            gaps.append({"alpha_id": peer["alpha_id"], "reason": "no_pnl", "n": 0})

    broadcast = {}
    for frequency in ("d", "w", "m"):
        target = correlation.aggregate(candidate, frequency)
        minimum = args.min_overlap if frequency == "d" else 12
        scored = []
        for peer in usable:
            left, right, _ = correlation.align(
                target, correlation.aggregate(series[peer["alpha_id"]], frequency),
                minimum)
            if len(left) < minimum:
                if frequency == "d":
                    gaps.append({"alpha_id": peer["alpha_id"],
                                 "reason": "short_overlap", "n": len(left)})
                continue
            value = correlation.pearson(left, right)
            if value is None:
                if frequency == "d":
                    gaps.append({"alpha_id": peer["alpha_id"],
                                 "reason": "undefined", "n": len(left)})
                continue
            scored.append({"alpha_id": peer["alpha_id"],
                           "correlation": round(value, 4),
                           "when": peer["when"].isoformat() if peer["when"] else None,
                           "status": peer["status"]})
        scored[:] = _rank_peers(scored)
        broadcast[frequency] = scored

    daily = broadcast["d"]
    best = daily[0] if daily else {"alpha_id": None, "correlation": None}
    # The platform reads a little lower than daily Pearson; the offset is measured, not
    # assumed, and it is what makes `borderline` a useful verdict instead of a shrug.
    verdict, estimate = correlation.classify(best["correlation"], args.threshold,
                                              args.platform_bias)
    verdict, verdict_note = _verdict_with_gaps(verdict, gaps)

    submitted = parse(record.get("dateSubmitted")) or parse(record.get("dateCreated"))
    counterpart_when = parse(best.get("when"))
    newer = bool(submitted and counterpart_when and counterpart_when > submitted)

    return {
        "alpha_id": args.alpha_id,
        "candidate_submitted": record.get("dateSubmitted"),
        "as_of": args.as_of,
        "peers_considered": len(peers),
        "peers_measured": len(daily),
        "peers_unmeasured": gaps,
        "peers_excluded_as_newer": [item["alpha_id"] for item in excluded],
        "threshold": args.threshold,
        "platform_bias": args.platform_bias,
        "max_daily": best["correlation"],
        "estimated_platform": estimate,
        "counterpart": best["alpha_id"],
        "counterpart_submitted": best.get("when"),
        "counterpart_entered_after_candidate": newer,
        "verdict": verdict,
        "verdict_note": verdict_note,
        "frequencies": {name: (items[0]["correlation"] if items else None)
                        for name, items in broadcast.items()},
        "top": daily[:args.top],
        "note": ("the counterpart entered after this alpha, so the platform could not have "
                 "seen it - set --as-of to reproduce the original check"
                 if newer else None),
    }


def cmd_alpha_corr(args, sink):
    """Correlation on daily PnL: the evidence independence claims need.

    Three questions this answers that nothing else in the toolkit could:
    whether a pair is really two bets, whether a candidate matches anything in
    the submitted book, and whether a resemblance survives removing the leg both
    sides share.
    """
    ids = []
    for value in args.alpha_id:
        for part in str(value).replace(",", " ").split():
            if part and part not in ids:
                ids.append(part)
    if len(ids) < 2 and not args.vs_book:
        raise CliError("USAGE", "at least two alpha ids are needed",
                       "pass two or more ids, or one id with --vs-book")

    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"

    def load(session):
        series = {}
        for alpha_id in ids:
            series[alpha_id] = ml.fetch_daily_pnl(
                session, alpha_id, cache_dir=pnl_dir,
                max_age_hours=args.max_age_hours)
        book = {}
        if args.vs_book:
            for entry in ml.list_alphas(session, limit=200, status="ACTIVE",
                                        alpha_type_not=""):
                book_id = entry.get("id")
                if book_id and book_id not in series:
                    book[book_id] = ml.fetch_daily_pnl(
                        session, book_id, cache_dir=pnl_dir,
                        max_age_hours=args.max_age_hours)
        control = None
        if args.control:
            control = ml.fetch_daily_pnl(
                session, args.control, cache_dir=pnl_dir,
                max_age_hours=args.max_age_hours)
        return series, book, control

    series, book, control = with_session(args, load)

    empty = [name for name, data in series.items() if not data]
    if empty:
        raise CliError(
            "NOT_FOUND",
            "no daily PnL for %s" % ", ".join(sorted(empty)),
            "the recordset is only populated for alphas that completed a "
            "simulation; check the id with `alpha get`")

    dates = sorted(set.intersection(*(set(v) for v in series.values())))
    matrix = correlation.matrix(series, min_overlap=args.min_overlap)
    high = [p for p in matrix["pairs"] if abs(p["r"]) >= args.threshold]
    # Absence of evidence is not evidence of absence. A pair whose overlap is too
    # short lands in `skipped`, and an earlier version then reported `independent`
    # because no measured pair cleared the threshold -- exactly backwards: the
    # unmeasurable pair is the one most likely to be a duplicate.
    skipped = matrix.get("skipped") or []
    measured = len(matrix["pairs"])
    total_pairs = measured + len(skipped)
    independence = _independence_state(high, skipped, measured)
    data = {
        "ids": ids,
        "window": {
            "start": dates[0] if dates else None,
            "end": dates[-1] if dates else None,
            "shared_days": len(dates),
        },
        "matrix": matrix,
        "flagged": high,
        "threshold": args.threshold,
        "independence": independence,
        "independent": independence == "independent",
        "coverage": {
            "measured_pairs": measured,
            "skipped_pairs": len(skipped),
            "ratio": round(measured / total_pairs, 4) if total_pairs else None,
            "note": ("unmeasurable pairs make independence unknowable"
                     if skipped else None),
        },
    }

    if book:
        against = {}
        for alpha_id, values in series.items():
            if not values:
                continue
            against[alpha_id] = correlation.max_against(
                values, book, min_overlap=args.min_overlap)
        data["vs_book"] = {
            "book_size": len(book),
            "per_alpha": against,
        }

    if control:
        residuals = {}
        for alpha_id, values in series.items():
            if values:
                residuals[alpha_id] = correlation.residualize(
                    values, control, min_overlap=args.min_overlap)
        residual_matrix = correlation.matrix(residuals,
                                             min_overlap=args.min_overlap)
        data["residual"] = {
            "control": args.control,
            "matrix": residual_matrix,
        }


    return data


def cmd_alpha_book(args, sink):
    registry_path = (getattr(args, "registry", None)
                     or alpha_registry.default_registry_path())
    entries = alpha_registry.load_entries(registry_path)

    by_id = {}
    by_hash = {}
    for entry in entries:
        alpha_id = entry.get("alpha_id")
        if alpha_id:
            by_id.setdefault(alpha_id, entry)
        digest = entry.get("expression_hash")
        if digest:
            by_hash.setdefault(digest, entry)

    def load(session):
        return ml.list_alphas(session, limit=getattr(args, "limit", 50),
                              status="ACTIVE", alpha_type_not="")

    alphas = with_session(args, load)

    items = []
    unknown = []
    mechanism_counts = {}
    family_counts = {}
    unclassified = 0
    for alpha in alphas:
        alpha_id = alpha.get("id")
        metrics = alpha.get("is") or {}
        code, _, _ = ml.regular_expression(alpha)
        entry = by_id.get(alpha_id) or by_hash.get(
            alpha_registry.expression_hash(code)) if code else by_id.get(alpha_id)
        mechanism = (entry or {}).get("mechanism")
        family = (entry or {}).get("family")
        if not mechanism:
            unknown.append(alpha_id)
            mechanism = None
        else:
            mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1
        else:
            unclassified += 1
        items.append({
            "alpha_id": alpha_id,
            "status": alpha.get("status"),
            "sharpe": metrics.get("sharpe"),
            "fitness": metrics.get("fitness"),
            "turnover": metrics.get("turnover"),
            "mechanism": mechanism,
            "family": family,
            "idea": (entry or {}).get("idea"),
            "expression": code,
        })

    document = {
        "registry": registry_path,
        "book_size": len(items),
        "mechanism_count": len(mechanism_counts),
        "mechanism_counts": mechanism_counts,
        "family_count": len(family_counts),
        "family_counts": family_counts,
        "unclassified_family": unclassified,
        "unlabelled": unknown,
        "alphas": items,
    }

    candidate_id = getattr(args, "check", None)
    if candidate_id:
        checks = with_session(args, lambda s: ml.fetch_alpha_checks(
            s, candidate_id,
            max_wait_seconds=getattr(args, "max_wait", 180) or 180))
        self_correlation = checks.get("self_correlation") or {}
        document["candidate"] = {
            "alpha_id": candidate_id,
            "self_correlation": self_correlation,
            "clear": (self_correlation.get("value") is not None
                      and self_correlation.get("limit") is not None
                      and self_correlation["value"] <= self_correlation["limit"]),
        }


    return document




def main(argv=None):
    """One boundary for parsing, workspace selection, saving and error envelopes."""
    global _QUIET
    arguments = list(sys.argv[1:] if argv is None else argv)
    options = argparse.Namespace(save=None, compact=False, quiet=False, research_dir=None,
                                 command="help")
    prepared = None
    previous_quiet = _QUIET
    previous_directory = os.environ.get(research_paths.ENV_VAR)
    previous_policy = ml.metadata_cache_policy()
    code = EXIT_CODES["OK"]
    try:
        try:
            # Read only common options first, so errors honour --save/--compact too.
            common = JsonArgumentParser(prog="research_cli.py", add_help=False, common=False)
            _add_common_args(common)
            common.parse_known_args(arguments, namespace=options)
            _QUIET = options.quiet
            if options.research_dir is not None:
                os.environ[research_paths.ENV_VAR] = str(Path(options.research_dir).expanduser().resolve())
            prepared = _prepare_output(options.save)
            parser = build_parser()
            parser.parse_args(arguments, namespace=options)
            data = _dispatch(options)
            payload = {"ok": True, "command": options.command, "data": data}
        except CliError as exc:
            payload = {"ok": False, "command": getattr(exc, "command", options.command),
                       "error": _error_body(exc)}
            code = exc.exit_code
        except ml.BrainError as exc:
            error_code = getattr(exc, "code", "SERVICE_ERROR")
            if str(error_code).startswith("AUTH") and ml.invalidate_session():
                note("session cache cleared after %s" % error_code)
            payload = {"ok": False, "command": options.command,
                       "error": {"code": error_code, "message": str(exc),
                                 "hint": getattr(exc, "hint", None)}}
            code = EXIT_CODES.get(error_code, EXIT_CODES["SERVICE_ERROR"])
        except KeyboardInterrupt:
            payload = {"ok": False, "command": options.command,
                       "error": {"code": "INTERRUPTED", "message": "cancelled", "hint": None}}
            code = EXIT_CODES["INTERRUPTED"]
        except Exception as exc:
            payload = {"ok": False, "command": options.command,
                       "error": {"code": "SERVICE_ERROR", "message": "%s: %s" % (type(exc).__name__, exc),
                                 "hint": "unexpected failure; see stderr for the traceback"}}
            import traceback
            traceback.print_exc(file=sys.stderr)
            code = EXIT_CODES["SERVICE_ERROR"]

        try:
            emit(payload, compact=options.compact, prepared=prepared)
        except OSError as exc:
            payload = {"ok": False, "command": options.command,
                       "error": {"code": "INVALID_INPUT", "message": "cannot write output: %s" % exc,
                                 "hint": "choose a writable --save path"}}
            emit(payload, compact=options.compact)
            code = EXIT_CODES["INVALID_INPUT"]
        if not payload["ok"]:
            error = payload["error"]
            note("%s: %s" % (error["code"], error["message"]), force=True)
            if error.get("hint"):
                note(error["hint"], force=True)
        return code
    finally:
        if prepared:
            prepared[1].unlink(missing_ok=True)
        _QUIET = previous_quiet
        ml.configure_metadata_cache(**previous_policy)
        if previous_directory is None:
            os.environ.pop(research_paths.ENV_VAR, None)
        else:
            os.environ[research_paths.ENV_VAR] = previous_directory


if __name__ == "__main__":
    sys.exit(main())
