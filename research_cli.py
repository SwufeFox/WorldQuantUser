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
import field_contract  # noqa: E402
import research_paths  # noqa: E402
import cache_store  # noqa: E402

VERSION = "2.2.0"

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
    "INTERRUPTED": "Cancelled by signal.",
}

#: Metric fields copied out of ``is`` into every result record, in report order.
METRIC_KEYS = ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin",
               "longCount", "shortCount", "pnl")


class JsonArgumentParser(argparse.ArgumentParser):
    """argparse that keeps the stdout contract: usage errors are JSON too."""

    def error(self, message):
        emit({"ok": False, "command": _command_of(None),
              "error": {"code": "USAGE", "message": message,
                        "hint": "run `research_cli.py schema` for the full contract"}})
        note("USAGE: %s" % message)
        raise SystemExit(EXIT_CODES["USAGE"])


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


def emit(payload):
    """Write the single JSON document that is this process's stdout."""
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")
    sys.stdout.flush()


def note(message):
    """Human-facing diagnostics. Always stderr, never stdout."""
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
    parser.add_argument("--region", choices=ml.REGIONS,
                        help="BRAIN region.")
    parser.add_argument("--universe", help="Universe, e.g. TOP3000.")
    parser.add_argument("--neutralization", choices=ml.NEUTRALIZATIONS,
                        help="Neutralization scheme.")
    parser.add_argument("--decay", type=int,
                        help="Default decay for alphas that do not set their own.")
    parser.add_argument("--delay", type=int, choices=[0, 1],
                        help="Data delay (0 or 1).")
    parser.add_argument("--truncation", type=float,
                        help="Truncation, e.g. 0.08.")
    parser.add_argument("--pasteurization", choices=["ON", "OFF"])
    parser.add_argument("--nan-handling", choices=["ON", "OFF", "VERIFY"])
    parser.add_argument("--test-period", help="e.g. P0Y, P1Y, P2Y.")


def build_parser():
    parser = JsonArgumentParser(
        prog="research_cli.py",
        description="Machine-readable WorldQuant BRAIN research CLI. "
                    "stdout is always one JSON document; stderr is for humans.",
    )
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", metavar="<command>",
                                parser_class=JsonArgumentParser)

    schema = sub.add_parser(
        "schema", help="Print the full call contract as JSON (commands, "
                       "arguments, research-file format, error codes).")
    schema.add_argument("name", nargs="?", help="Restrict to one command.")
    schema.set_defaults(func=cmd_schema)

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
    cat_op.add_argument("--search", default="", help="Filter on name/description.")
    cat_op.add_argument("--group", default="", help="Exact category, e.g. 'Time Series'.")
    cat_op.add_argument("--docs", nargs="*", metavar="NAME",
                        help="Also fetch full documentation for these operators.")
    cat_op.add_argument("--save", help="Also write the JSON result to this file.")
    _add_session_args(cat_op)
    cat_op.set_defaults(func=cmd_catalog_operators)

    cat_ds = cat.add_parser("datasets", help="Datasets for a region/universe/delay.")
    cat_ds.add_argument("--instrument-type", default="EQUITY")
    cat_ds.add_argument("--region", default="USA")
    cat_ds.add_argument("--universe", default="TOP3000")
    cat_ds.add_argument("--delay", type=int, default=1, choices=[0, 1])
    cat_ds.add_argument("--limit", type=int, default=None)
    cat_ds.add_argument("--save")
    _add_session_args(cat_ds)
    cat_ds.set_defaults(func=cmd_catalog_datasets)

    cat_f = cat.add_parser("fields", help="Datafields of one dataset.")
    cat_f.add_argument("--dataset", required=True, help="Dataset id, e.g. fundamental6.")
    cat_f.add_argument("--region", default="USA")
    cat_f.add_argument("--universe", default="TOP3000")
    cat_f.add_argument("--delay", type=int, default=1, choices=[0, 1])
    cat_f.add_argument("--instrument-type", default="EQUITY")
    cat_f.add_argument("--type", dest="field_type", choices=["MATRIX", "VECTOR", "GROUP"],
                       help="Keep only this BRAIN field type.")
    cat_f.add_argument("--search", default="", help="Filter on id/description.")
    cat_f.add_argument("--limit", type=int, default=None)
    cat_f.add_argument("--expressions", action="store_true",
                       help="Also emit ready-to-use FASTEXPR snippets.")
    cat_f.add_argument("--save")
    _add_session_args(cat_f)
    cat_f.set_defaults(func=cmd_catalog_fields)

    cat_c = cat.add_parser("categories", help="Data categories with crowding stats.")
    cat_c.add_argument("--region", default="USA")
    cat_c.add_argument("--universe", default="TOP3000")
    cat_c.add_argument("--delay", type=int, default=1, choices=[0, 1])
    cat_c.add_argument("--save")
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
    cat_audit.add_argument("--region", default="USA")
    cat_audit.add_argument("--universe", default="TOP3000")
    cat_audit.add_argument("--delay", type=int, default=1, choices=[0, 1])
    cat_audit.add_argument("--instrument-type", default="EQUITY")
    cat_audit.add_argument("--save")
    _add_session_args(cat_audit)
    cat_audit.set_defaults(func=cmd_catalog_field_audit)

    lint = sub.add_parser(
        "lint",
        help="Pre-flight expressions against the operator catalog (quota-free).")
    lint.add_argument("expression", nargs="?",
                      help="Expression to lint, e.g. 'rank(ts_delta(close, 5))'.")
    lint.add_argument("--file", default=None,
                      help="Research file (JSON or one expression per line).")
    lint.add_argument("--offline", action="store_true",
                      help="Skip the operator catalog; check structure and the "
                           "learned denylist only.")
    lint.add_argument("--dataset", default=None,
                      help="Treat this dataset's fields as known so unknown "
                           "identifiers are flagged.")
    lint.add_argument("--strict-fields", action="store_true",
                      help="Treat unknown identifiers as errors, not warnings.")
    lint.add_argument("--region", default="USA")
    lint.add_argument("--universe", default="TOP3000")
    lint.add_argument("--delay", type=int, default=1, choices=[0, 1])
    lint.add_argument("--instrument-type", default="EQUITY")
    lint.add_argument("--denylist", default=None,
                      help="Override the learned denylist path.")
    lint.add_argument("--save")
    _add_session_args(lint)
    lint.set_defaults(func=cmd_lint)

    sim = sub.add_parser(
        "simulate", help="Simulate alphas (spends simulation quota unless --dry-run).")
    sim.add_argument("file", nargs="?",
                     help="Research file: JSON, or one expression per line. "
                          "Omit when using --expr.")
    sim.add_argument("--expr", action="append", default=[], metavar="EXPRESSION",
                     help="Inline expression; repeat for several.")
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

    alpha = sub.add_parser("alpha", help="Inspect the account's alphas (quota-free).")
    alpha_sub = alpha.add_subparsers(dest="subcommand", metavar="<action>",
                                     parser_class=JsonArgumentParser)

    a_get = alpha_sub.add_parser("get", help="Fetch one alpha's metrics and checks.")
    a_get.add_argument("alpha_id")
    _add_session_args(a_get)
    a_get.add_argument("--save")
    a_get.set_defaults(func=cmd_alpha_get)

    a_chk = alpha_sub.add_parser(
        "check",
        help="Compute live checks (incl. SELF_CORRELATION). 'get' only shows cached values.")
    a_chk.add_argument("alpha_id")
    a_chk.add_argument("--max-wait", type=float, default=180.0,
                       help="Seconds to wait for the async check job (default 180).")
    _add_session_args(a_chk)
    a_chk.add_argument("--save")
    a_chk.set_defaults(func=cmd_alpha_check)

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
    a_list.add_argument("--save")
    a_list.set_defaults(func=cmd_alpha_list)

    a_rec = alpha_sub.add_parser(
        "record", help="Persist one alpha into the append-only registry.")
    a_rec.add_argument("alpha_id")
    a_rec.add_argument("--idea", default=None, help="e.g. 'Idea 47'.")
    a_rec.add_argument("--mechanism", default=None)
    a_rec.add_argument("--verdict", choices=list(alpha_registry.VERDICTS),
                       default="watch")
    a_rec.add_argument("--note", default=None)
    a_rec.add_argument("--registry", default=None)
    _add_session_args(a_rec)
    a_rec.set_defaults(func=cmd_alpha_record)

    a_reg = alpha_sub.add_parser(
        "registry", help="Read the durable alpha ledger (no network needed).")
    a_reg.add_argument("--verdict", action="append", default=[],
                       choices=list(alpha_registry.VERDICTS),
                       help="Keep only these verdicts; repeatable.")
    a_reg.add_argument("--idea", default=None)
    a_reg.add_argument("--min-sharpe", type=float, default=None)
    a_reg.add_argument("--format", choices=["json", "markdown"], default="json")
    a_reg.add_argument("--registry", default=None)
    a_reg.add_argument("--save")
    a_reg.set_defaults(func=cmd_alpha_registry)

    return parser


def _type_name(action):
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "boolean"
    if action.type is not None:
        if action.type is _positive_seconds:
            return "number"
        return getattr(action.type, "__name__", str(action.type))
    return "string"


def describe_parser(parser, name="research_cli.py"):
    """Turn an argparse parser (and its subparsers) into a JSON description."""
    options = []
    subcommands = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child_name, child in action.choices.items():
                subcommands[child_name] = describe_parser(child, child_name)
            continue
        if isinstance(action, (argparse._HelpAction, argparse._VersionAction)):
            continue
        entry = {
            "flags": action.option_strings or [action.dest],
            "type": _type_name(action),
        }
        if action.choices:
            entry["choices"] = [str(c) for c in action.choices]
        if action.help:
            entry["help"] = action.help
        if action.default not in (None, False, []):
            entry["default"] = action.default
        options.append(entry)
    description = {
        "command": name,
        "usage": parser.format_usage().strip(),
        "options": options,
    }
    if subcommands:
        description["subcommands"] = subcommands
    return description


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
    """Merge ``--expr`` values and the research file into ``(settings, alphas)``."""
    file_settings, alphas = {}, []
    if args.file:
        try:
            file_settings, alphas = config_io.load_research_file(args.file)
        except config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc))
    if args.expr:
        alphas = alphas + list(args.expr)
    if not alphas:
        raise CliError("USAGE", "no alphas to simulate",
                       "pass a research FILE or at least one --expr")
    return file_settings, alphas


def merge_settings(file_settings, args):
    """CLI flags win over the file, which wins over machine_lib defaults."""
    overrides = dict(file_settings)
    for key, value in {
        "region": args.region,
        "universe": args.universe,
        "neutralization": args.neutralization,
        "decay": args.decay,
        "delay": args.delay,
        "truncation": args.truncation,
        "pasteurization": args.pasteurization,
        "nanHandling": args.nan_handling,
        "testPeriod": args.test_period,
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
def summarize(records):
    """Count outcomes and rank the scored alphas so a caller can pick winners."""
    completed = [r for r in records if r.get("status") == "COMPLETE"]
    failed = [r for r in records if r.get("status") == "FAILED"]
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
            r.get("status") == "PENDING" or bool(r.get("progress_url"))
            for r in records
        ),
        "queued": sum(r.get("status") == "QUEUED" for r in records),
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
    if args.name:
        if args.name not in described["subcommands"]:
            raise CliError("USAGE", "unknown command %r" % args.name,
                           "known: " + ", ".join(sorted(described["subcommands"])))
        contract = {
            "tool": "research_cli.py",
            "version": VERSION,
            "exit_codes": EXIT_CODES,
            "error_codes": ERROR_CODES,
            "command": described["subcommands"][args.name],
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


def _maybe_save(args, data):
    target = getattr(args, "save", None)
    if target:
        Path(target).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        note("wrote %s" % target)


def cmd_catalog_operators(args, _sink):
    cache_report = {}

    def work(session):
        operators = ml.fetch_operators(session, search=args.search, group=args.group,
                                       report=cache_report)
        data = {"count": len(operators), "operators": operators,
                "cache": cache_report}
        if args.docs:
            data["docs"] = {n: ml.fetch_operator_doc(session, n) for n in args.docs}
        return data

    data = with_session(args, work)
    _maybe_save(args, data)
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
    _maybe_save(args, data)
    return data


def cmd_catalog_fields(args, _sink):
    cache_report = {}
    fields = with_session(args, lambda s: ml.get_datafields(
        s, instrument_type=args.instrument_type, region=args.region,
        delay=args.delay, universe=args.universe, dataset_id=args.dataset,
        search=args.search, field_type=args.field_type, limit=args.limit,
        report=cache_report,
    ))
    data = {
        "dataset": args.dataset, "region": args.region, "universe": args.universe,
        "delay": args.delay, "count": len(fields),
        "type_counts": {
            kind: len([f for f in fields if f.get("type") == kind])
            for kind in ("MATRIX", "VECTOR", "GROUP")
        },
        "fields": fields, "cache": cache_report,
    }
    if args.expressions:
        data["expressions"] = ml.process_datafields(fields)
    _maybe_save(args, data)
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
    _maybe_save(args, data)
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


def _prepare_simulation_run(settings, pairs, batch_size, checkpoint_arg):
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
    records = [
        {
            "expression": expression,
            "decay": decay,
            "alpha_id": None,
            "status": "QUEUED",
            "metrics": None,
            "checks": None,
            "error": None,
        }
        for expression, decay in pairs
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

    records = _sort_simulation_results(records, args)
    summary = summarize(records)
    if summary["completed"] == 0 and summary["total"] and not (
        summary["pending"] or summary["queued"]
    ):
        first_error = next(
            (r.get("error") for r in records if r.get("error")), "unknown error"
        )
        raise CliError(
            "SIMULATION_FAILED",
            "all %d simulations failed" % summary["total"],
            str(first_error),
        )
    result = _simulation_result(records, settings, checkpoint_path)
    if getattr(args, "record", False):
        result["registry"] = _record_simulation(records, settings, args)
    return result


def _preflight_lint(session, pairs, args):
    """Lint against the operator catalog before spending simulation quota.

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
    report = expression_lint.lint_all(
        pairs,
        operator_index=operator_index,
        denylist=expression_lint.load_denylist(),
    )
    report["operator_catalog_size"] = len(operator_index or {})
    report["operator_catalog"] = catalog_note
    if report["ok"]:
        return report
    if getattr(args, "allow_lint_errors", False):
        return report
    unknown = sorted({item.get("operator") for item in report["errors"]
                      if item.get("code") == "unknown_operator"})
    hint = "fix the expressions, or pass --allow-lint-errors to submit anyway"
    if unknown:
        hint = ("operator(s) %s are not in the BRAIN operator list; a docs page "
                "existing does not make them simulatable" % ", ".join(unknown))
    raise CliError(
        "INVALID_INPUT",
        "expression lint failed; nothing was submitted",
        hint,
        details={
            "operator_catalog_size": report["operator_catalog_size"],
            "errors": report["errors"],
            "warnings": report["warnings"],
        },
    )


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

    if args.dry_run:
        records = ml.simulate_alphas(
            None, pairs, settings=settings, dry_run=True, labels=labels
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
    with_session(args, lambda s: _preflight_lint(s, pairs, args))

    checkpoint_path, checkpoint = _prepare_simulation_run(
        settings, pairs, args.batch_size, args.checkpoint
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
    _maybe_save(args, document)
    return document


def cmd_alpha_check(args, _sink):
    """Force the platform to compute checks and return the real values.

    ``alpha get`` reads the cached snapshot embedded in the alpha body, where
    SELF_CORRELATION is usually PENDING or absent. This command hits
    ``/alphas/{id}/check``, which runs the job (answering with Retry-After) and
    then returns the computed correlation.
    """
    if args.max_wait is not None and args.max_wait < 0:
        raise UsageError("--max-wait must be non-negative")
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
    _maybe_save(args, document)
    return document


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
    _maybe_save(args, document)
    return document


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        args.command, args.func, args.name = "schema", cmd_schema, None
    elif not getattr(args, "func", None):
        children = sorted(describe_parser(build_parser())["subcommands"])
        raise CliError("USAGE", "`%s` needs a subcommand" % args.command,
                       "known subcommands: %s" % ", ".join(children))
    sink = make_event_sink(getattr(args, "stream", False), getattr(args, "verbose", False))
    # Catalog caching is on by default; the flags only tune it. Library
    # diagnostics are routed through the same stderr channel as our own notes,
    # so stdout stays a single JSON document.
    ml.set_diagnostic_handler(note)
    ml.configure_metadata_cache(
        max_age=getattr(args, "meta_max_age", None),
        refresh=getattr(args, "refresh_meta", None),
    )
    data = args.func(args, sink)
    emit({"ok": True, "command": args.command, "data": data})
    return EXIT_CODES["OK"]


# ---------------------------------------------------------------------------
# lint -- expression pre-flight
# ---------------------------------------------------------------------------

def _collect_lint_pairs(args):
    """Return ``(settings, [(expression, decay), ...])`` for the lint target."""
    if getattr(args, "file", None):
        file_settings, alphas = config_io.load_research_file(args.file)
        return config_io.build_config(file_settings, alphas)
    expression = getattr(args, "expression", None)
    if not expression:
        raise CliError(
            "USAGE", "lint needs an expression or --file",
            "pass it positionally, e.g. `lint 'rank(ts_delta(close, 5))'`")
    return {}, [(expression, None)]


def _operator_index_from_catalog(session):
    return expression_lint.build_operator_index(ml.fetch_operators(session))


def _known_field_ids(session, args):
    """Field ids of ``--dataset``, so unknown identifiers can be flagged."""
    dataset = getattr(args, "dataset", None)
    if not dataset:
        return None
    records = ml.get_datafields(
        session,
        instrument_type=getattr(args, "instrument_type", "EQUITY"),
        region=getattr(args, "region", "USA"),
        delay=getattr(args, "delay", 1),
        universe=getattr(args, "universe", "TOP3000"),
        dataset_id=dataset,
    )
    return {record.get("id") for record in records if record.get("id")}


def cmd_lint(args, sink):
    settings, pairs = _collect_lint_pairs(args)
    denylist = expression_lint.load_denylist(getattr(args, "denylist", None))
    operator_index = None
    known_fields = None
    catalog_checked = False
    if not getattr(args, "offline", False):
        def work(session):
            return (_operator_index_from_catalog(session),
                    _known_field_ids(session, args))

        operator_index, known_fields = with_session(args, work)
        catalog_checked = operator_index is not None

    results = []
    for expression, decay in pairs:
        report = expression_lint.lint_expression(
            expression,
            operator_index=operator_index,
            denylist=denylist,
            known_fields=known_fields,
            strict_fields=getattr(args, "strict_fields", False),
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
    _maybe_save(args, document)
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
        instrument_type=getattr(args, "instrument_type", "EQUITY"),
        region=getattr(args, "region", "USA"),
        delay=getattr(args, "delay", 1),
        universe=getattr(args, "universe", "TOP3000"),
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
        instrument_type=getattr(args, "instrument_type", "EQUITY"),
        region=getattr(args, "region", "USA"),
        delay=getattr(args, "delay", 1),
        universe=getattr(args, "universe", "TOP3000"),
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
        "region": getattr(args, "region", "USA"),
        "universe": getattr(args, "universe", "TOP3000"),
        "delay": getattr(args, "delay", 1),
        "instrument_type": getattr(args, "instrument_type", "EQUITY"),
    }
    if field_ids:
        document["requested_fields"] = field_ids
    if missing:
        document["missing_fields"] = missing
    _maybe_save(args, document)
    return document


# ---------------------------------------------------------------------------
# alpha record / alpha registry -- the durable ledger
# ---------------------------------------------------------------------------

def _auto_verdict(record):
    """Classify a simulation record without pretending to be a researcher.

    Only the default for ``simulate --record``: it keeps the ledger useful when
    nobody has written a judgement yet. Anything that cleared the usual bar is a
    ``candidate``; anything that failed a platform check is ``rejected``; the
    rest stays ``watch`` until someone decides.
    """
    if record.get("status") != "COMPLETE":
        return "rejected"
    checks = record.get("checks") or {}
    if isinstance(checks, dict):
        results = [str((value or {}).get("result") or "").upper()
                   if isinstance(value, dict) else str(value).upper()
                   for value in checks.values()]
    else:
        results = [str((value or {}).get("result") or "").upper()
                   for value in checks if isinstance(value, dict)]
    if any(result in {"FAIL", "ERROR"} for result in results):
        return "rejected"
    metrics = record.get("metrics") or {}
    sharpe = metrics.get("sharpe")
    fitness = metrics.get("fitness")
    if isinstance(sharpe, (int, float)) and isinstance(fitness, (int, float)):
        if sharpe >= 1.25 and fitness >= 1.0:
            return "candidate"
    return "watch"


def _record_simulation(records, settings, args):
    """Append finished simulation records to the registry."""
    registry_path = (getattr(args, "registry", None)
                     or alpha_registry.default_registry_path())
    entries = []
    written = 0
    for record in records:
        entry = alpha_registry.build_entry(
            {
                "alpha_id": record.get("alpha_id"),
                "expression": record.get("expression"),
                "decay": record.get("decay"),
                "settings": settings,
                "metrics": record.get("metrics"),
                "checks": record.get("checks"),
                "out_of_sample": record.get("out_of_sample"),
                "status": record.get("status"),
                "error": record.get("error"),
            },
            idea=getattr(args, "idea", None),
            # A per-expression label beats the run-wide default so a batch that
            # spans several mechanisms still lands labelled per entry.
            mechanism=record.get("mechanism") or getattr(args, "mechanism", None),
            verdict=_auto_verdict(record),
            note=getattr(args, "note", None),
            source="simulate",
        )
        _, did_write = alpha_registry.record_entry(entry, registry_path)
        written += 1 if did_write else 0
        entries.append(entry)
    return {
        "registry": registry_path,
        "recorded": len(entries),
        "appended": written,
        "by_verdict": {
            verdict: sum(1 for entry in entries if entry["verdict"] == verdict)
            for verdict in alpha_registry.VERDICTS
        },
    }


def cmd_alpha_record(args, sink):
    alpha_id = getattr(args, "alpha_id", None)
    if not alpha_id:
        raise CliError(
            "USAGE", "alpha record needs an alpha id",
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


def cmd_alpha_registry(args, sink):
    registry_path = (getattr(args, "registry", None)
                     or alpha_registry.default_registry_path())
    entries = alpha_registry.load_entries(registry_path)
    verdicts = set(getattr(args, "verdict", None) or [])
    idea = getattr(args, "idea", None)
    min_sharpe = getattr(args, "min_sharpe", None)
    selected = []
    for entry in entries:
        if verdicts and entry.get("verdict") not in verdicts:
            continue
        if idea and entry.get("idea") != idea:
            continue
        if min_sharpe is not None:
            sharpe = (entry.get("metrics") or {}).get("sharpe")
            if not isinstance(sharpe, (int, float)) or sharpe < min_sharpe:
                continue
        selected.append(entry)
    document = {
        "registry": registry_path,
        "filtered": len(selected),
        "summary": alpha_registry.summarize(selected),
    }
    if getattr(args, "format", "json") == "markdown":
        document["markdown"] = alpha_registry.render_markdown(selected)
    document["entries"] = selected
    _maybe_save(args, document)
    return document


def main(argv=None):
    try:
        return run(argv)
    except CliError as exc:
        emit({"ok": False, "command": _command_of(argv),
              "error": _error_body(exc)})
        note("%s: %s" % (exc.code, exc.message))
        return exc.exit_code
    except ml.BrainError as exc:
        code = getattr(exc, "code", "SERVICE_ERROR")
        if str(code).startswith("AUTH"):
            # The cookie is known-bad; keeping it would make every later call
            # replay the same rejection instead of logging in again.
            if ml.invalidate_session():
                note("session cache cleared after %s" % code)
        hint = getattr(exc, "hint", None)
        emit({"ok": False, "command": _command_of(argv),
              "error": {"code": code, "message": str(exc), "hint": hint}})
        note("%s: %s" % (code, exc))
        return EXIT_CODES.get(code, EXIT_CODES["SERVICE_ERROR"])
    except KeyboardInterrupt:
        emit({"ok": False, "command": _command_of(argv),
              "error": {"code": "INTERRUPTED", "message": "cancelled", "hint": None}})
        return EXIT_CODES["INTERRUPTED"]
    except Exception as exc:  # unexpected: stdout must stay parseable anyway
        emit({"ok": False, "command": _command_of(argv),
              "error": {"code": "SERVICE_ERROR",
                        "message": "%s: %s" % (type(exc).__name__, exc),
                        "hint": "unexpected failure; see stderr for the traceback"}})
        import traceback
        traceback.print_exc(file=sys.stderr)
        return EXIT_CODES["SERVICE_ERROR"]


def _command_of(argv):
    argv = sys.argv[1:] if argv is None else list(argv)
    return next((a for a in argv if not a.startswith("-")), "schema")


if __name__ == "__main__":
    sys.exit(main())
