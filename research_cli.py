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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import machine_lib as ml  # noqa: E402
import research_config as config_io  # noqa: E402

VERSION = "2.0.0"

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

    def __init__(self, code, message, hint=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    @property
    def exit_code(self):
        return EXIT_CODES.get(self.code, 2)


# ---------------------------------------------------------------------------
# Output plumbing
# ---------------------------------------------------------------------------
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
    sim.add_argument("--dry-run", action="store_true",
                     help="Validate and print payloads without contacting BRAIN.")
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

    alpha = sub.add_parser("alpha", help="Inspect the account's alphas (quota-free).")
    alpha_sub = alpha.add_subparsers(dest="subcommand", metavar="<action>",
                                     parser_class=JsonArgumentParser)

    a_get = alpha_sub.add_parser("get", help="Fetch one alpha's metrics and checks.")
    a_get.add_argument("alpha_id")
    _add_session_args(a_get)
    a_get.set_defaults(func=cmd_alpha_get)

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

    return parser


def _type_name(action):
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "boolean"
    if action.type is not None:
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
            "note": "cookies are reused until shortly before expiry; git-ignored",
        },
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
    session = ml.load_session(use_cache=not args.no_cache, refresh=args.relogin)
    info = {"user_id": None, "permissions": None, "token_expiry": None}
    cache = ml.session_cache_path()
    if cache.is_file():
        try:
            cached = json.loads(cache.read_text())
        except ValueError:
            cached = {}
        info = {
            "user_id": cached.get("user_id"),
            "permissions": cached.get("permissions"),
            "token_expiry": cached.get("expiry_seconds"),
        }
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
        "token_expiry": info.get("token_expiry"),
        "data_regions": regions,
        "note": "regions outside data_regions return no datasets for this account",
    }


def _maybe_save(args, data):
    target = getattr(args, "save", None)
    if target:
        Path(target).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        note("wrote %s" % target)


def cmd_catalog_operators(args, _sink):
    def work(session):
        operators = ml.fetch_operators(session, search=args.search, group=args.group)
        data = {"count": len(operators), "operators": operators}
        if args.docs:
            data["docs"] = {n: ml.fetch_operator_doc(session, n) for n in args.docs}
        return data

    data = with_session(args, work)
    _maybe_save(args, data)
    return data


def cmd_catalog_datasets(args, _sink):
    datasets = with_session(args, lambda s: ml.get_datasets(
        s, instrument_type=args.instrument_type, region=args.region,
        delay=args.delay, universe=args.universe, limit=args.limit,
    ))
    data = {
        "region": args.region, "universe": args.universe, "delay": args.delay,
        "count": len(datasets), "datasets": datasets,
    }
    if not datasets:
        data["hint"] = ("no datasets for this combination; the account may lack "
                        "permission (a TUTORIAL account only sees USA)")
    _maybe_save(args, data)
    return data


def cmd_catalog_fields(args, _sink):
    fields = with_session(args, lambda s: ml.get_datafields(
        s, instrument_type=args.instrument_type, region=args.region,
        delay=args.delay, universe=args.universe, dataset_id=args.dataset,
        search=args.search, field_type=args.field_type, limit=args.limit,
    ))
    data = {
        "dataset": args.dataset, "region": args.region, "universe": args.universe,
        "delay": args.delay, "count": len(fields),
        "type_counts": {
            kind: len([f for f in fields if f.get("type") == kind])
            for kind in ("MATRIX", "VECTOR", "GROUP")
        },
        "fields": fields,
    }
    if args.expressions:
        data["expressions"] = ml.process_datafields(fields)
    _maybe_save(args, data)
    return data


def cmd_catalog_categories(args, _sink):
    categories = with_session(args, lambda s: ml.fetch_data_categories(
        s, region=args.region, universe=args.universe, delay=args.delay))
    data = {
        "region": args.region, "universe": args.universe, "delay": args.delay,
        "count": len(categories), "categories": categories,
        "note": "alphaCount is a crowding signal: higher means more mined",
    }
    _maybe_save(args, data)
    return data


def cmd_simulate(args, sink):
    file_settings, alphas = collect_alphas(args)
    settings = merge_settings(file_settings, args)
    try:
        settings, pairs = config_io.build_config(settings, alphas)
    except config_io.ResearchError as exc:
        raise CliError("INVALID_INPUT", str(exc))

    if args.dry_run:
        records = ml.simulate_alphas(None, pairs, settings=settings, dry_run=True)
        return {
            "dry_run": True,
            "settings": settings,
            "summary": {"total": len(records), "payloads": len(records)},
            "results": records,
        }

    records = with_session(args, lambda s: ml.simulate_alphas(
        s, pairs, settings=settings, batch_size=args.batch_size, on_event=sink))
    for record in records:
        _compact_metrics(record)

    if args.sort != "none":
        records.sort(key=lambda r: _metric_value(r, args.sort), reverse=True)
    if args.top is not None:
        records = records[:args.top]

    summary = summarize(records)
    if summary["completed"] == 0 and summary["total"]:
        first_error = next((r.get("error") for r in records if r.get("error")), "unknown error")
        raise CliError("SIMULATION_FAILED",
                       "all %d simulations failed" % summary["total"], first_error)
    return {"dry_run": False, "settings": settings, "summary": summary, "results": records}


def cmd_alpha_get(args, _sink):
    payload = with_session(args, lambda s: ml.fetch_alpha_metrics(s, args.alpha_id))
    in_sample = payload.get("is") or {}
    code, description, operator_count = ml.regular_expression(payload)
    return {
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
    return {"count": len(compact), "offset": args.offset, "alphas": compact}


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
    data = args.func(args, sink)
    emit({"ok": True, "command": args.command, "data": data})
    return EXIT_CODES["OK"]


def main(argv=None):
    try:
        return run(argv)
    except CliError as exc:
        emit({"ok": False, "command": _command_of(argv),
              "error": {"code": exc.code, "message": exc.message, "hint": exc.hint}})
        note("%s: %s" % (exc.code, exc.message))
        return exc.exit_code
    except ml.BrainError as exc:
        code = getattr(exc, "code", "SERVICE_ERROR")
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