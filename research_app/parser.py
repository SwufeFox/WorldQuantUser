"""Command parsing, terminal help, and the JSON command contract."""

from __future__ import annotations

import argparse
import difflib
import math
import os

import alpha_registry
import field_contract
import machine_lib as ml
import research_config as config_io
import research_paths

from . import account, alphas, buffer_cmd, campaign, catalog, experiments, ledger, literature, validation
from .runtime import VERSION, EXIT_CODES, ERROR_CODES, METRIC_KEYS, CliError


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
    parser.add_argument("--neutralization", type=str.upper,
                        help="Neutralization token; validated by machine_lib, with availability decided by BRAIN.")
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
    """Build the complete CLI argument parser with subcommands, defaults and examples."""
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
    auth_check.set_defaults(func=account.cmd_auth_check)

    auth_logout = auth_sub.add_parser(
        "logout", help="Forget the cached session cookie (local only).")
    auth_logout.set_defaults(func=account.cmd_auth_logout)

    cache = sub.add_parser(
        "cache", help="Inspect or clear local caches (session, catalog metadata).")
    cache_sub = cache.add_subparsers(dest="subcommand", metavar="<action>",
                                     parser_class=JsonArgumentParser)
    cache_info = cache_sub.add_parser(
        "info", help="Where caches live, what they hold, and how stale they are.")
    cache_info.set_defaults(func=account.cmd_cache_info)
    cache_clear = cache_sub.add_parser(
        "clear", help="Drop cached data. Locks are kept for running processes.")
    cache_clear.add_argument("--kind", choices=["session", "metadata", "all"],
                             default="all", help="What to clear (default: all).")
    cache_clear.set_defaults(func=account.cmd_cache_clear)

    catalog_parser = sub.add_parser(
        "catalog", help="Quota-free metadata: operators, datasets, fields.")
    cat = catalog_parser.add_subparsers(dest="subcommand", metavar="<kind>",
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
    cat_op.set_defaults(func=catalog.cmd_catalog_operators)

    cat_ds = cat.add_parser("datasets", help="Datasets for a region/universe/delay.")
    cat_ds.add_argument("--instrument-type", type=str.upper, default=ml.DEFAULT_SETTINGS["instrumentType"])
    cat_ds.add_argument("--region", type=str.upper, choices=ml.REGIONS, default=ml.DEFAULT_SETTINGS["region"])
    cat_ds.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_ds.add_argument("--delay", type=int, default=ml.DEFAULT_SETTINGS["delay"], choices=[0, 1])
    cat_ds.add_argument("--limit", type=int, default=None)

    _add_session_args(cat_ds)
    cat_ds.set_defaults(func=catalog.cmd_catalog_datasets)

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
    cat_f.set_defaults(func=catalog.cmd_catalog_fields)

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
    cat_cc.set_defaults(func=catalog.cmd_catalog_concepts)

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
    cat_vf.set_defaults(func=catalog.cmd_catalog_verify)

    cat_c = cat.add_parser("categories", help="Data categories with crowding stats.")
    cat_c.add_argument("--region", type=str.upper, choices=ml.REGIONS, default=ml.DEFAULT_SETTINGS["region"])
    cat_c.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_c.add_argument("--delay", type=int, default=ml.DEFAULT_SETTINGS["delay"], choices=[0, 1])

    _add_session_args(cat_c)
    cat_c.set_defaults(func=catalog.cmd_catalog_categories)

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
    cat_audit.set_defaults(func=catalog.cmd_catalog_field_audit)

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
    lint.set_defaults(func=validation.cmd_lint)

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

    lit.set_defaults(func=literature.cmd_search)

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

    papers.set_defaults(func=literature.cmd_paper)

    sim = sub.add_parser(
        "simulate", help="Simulate alphas (no per-simulation fee on this account; "
                                "time is the only cost - sweep wide).")
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
                          "behaviour). Not recommended: the failure costs a rerun.")
    sim.add_argument("--sort", choices=["none", *METRIC_KEYS], default="none",
                     help="Sort results by this metric, best first.")
    sim.add_argument("--top", type=int, default=None,
                     help="Keep only the first N results after sorting.")
    sim.add_argument("--stream", action="store_true",
                     help="Emit JSON Lines progress events on stderr.")
    sim.add_argument("--verbose", action="store_true",
                     help="With --stream, also emit per-poll events.")
    _add_session_args(sim)
    sim.set_defaults(func=experiments.cmd_simulate)

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
    resume.set_defaults(func=experiments.cmd_resume)

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
                     help="Comma-separated ledger ids: write one ablation file per alpha. "
                          "Named sums use named legs; other parseable expressions use structural drop-term variants.")
    abl.add_argument("--out-dir", default=None,
                     help="With --ids, write <id>.json here (default: the research dir).")
    abl.add_argument("--modes", default="full,loo,solo",
                     help="Comma list from full, loo, solo (default all three).")
    _add_settings_args(abl)
    abl.add_argument("--out", default=None,
                     help="Write the research file here instead of only reporting it.")

    abl.set_defaults(func=experiments.cmd_ablate)

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
    cmp_.add_argument("--per-file", action="store_true",
                      help="Report contributions per input file so identically named ablation arms stay separate.")

    cmp_.set_defaults(func=experiments.cmd_compare)

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

    brief.set_defaults(func=campaign.cmd_brief)

    ask = sub.add_parser(
        "ask",
        help="Query the local ledger: filter, group, summarise, correlate.",
        description="Filter ledger entries, group statistics, measure within-panel slopes, "
                    "or correlate cached PnL without network requests.")
    ask.add_argument("--where", default="",
                     help="Comma list of field=value / field!=value; fields: " +
                          ", ".join(ledger._ASK_FIELDS) +
                          " plus check=NAME:RESULT and has=metrics|pnl|checks.")
    ask.add_argument("--by", default=None,
                     help="Grouping axis; one of " + ", ".join(ledger._ASK_FIELDS) + ".")
    ask.add_argument("--stat", default="n",
                     help="Comma list: n, floor_share, median:METRIC, mean:METRIC, "
                          "p25:METRIC, p75:METRIC, min:METRIC, max:METRIC.")
    ask.add_argument("--panel", default="",
                     help="Comma list of metrics whose within-panel slope against numeric --by is reported.")
    ask.add_argument("--factor-ids", default=None,
                     help="Comma-separated ids whose average PnL becomes a factor; every "
                          "matched entry is correlated against it. Use it to ask how "
                          "many alphas are re-parameterisations of one measured leg.")
    ask.add_argument("--factor-where", default=None,
                     help="Build the factor from a ledger query instead of --factor-ids, "
                          "e.g. --factor-where status=ACTIVE for 'correlate every match "
                          "against the live book'.")
    ask.add_argument("--factor-threshold", type=float, default=0.8,
                     help="|r| at or above this counts as a variant of the factor "
                          "(default 0.8).")
    ask.add_argument("--matrix", default=None, choices=["corr"],
                     help="Pairwise correlation over cached PnL of matched entries.")
    ask.add_argument("--matrix-within", default="expr",
                     choices=["none", "expr", "family", "neutralization"],
                     help="Restrict matrix pairs to entries sharing this key. The "
                          "default (expr) answers 'same construction, different "
                          "settings' and therefore reports very few pairs when every "
                          "entry is a distinct expression - use --matrix-within none "
                          "for the whole matched set.")
    ask.add_argument("--min-overlap", type=int, default=200,
                     help="Shared trading days required for each correlation pair.")
    ask.add_argument("--list", action="store_true",
                     help="Include one line per matched entry.")
    ask.add_argument("--limit", type=int, default=20,
                     help="Maximum listed entries or correlation pairs.")
    ask.add_argument("--registry", default=None)
    ask.set_defaults(func=ledger.cmd_ask)

    probe = sub.add_parser(
        "probe",
        help="Run a batch of arms from a spec file (lint, simulate, table, record).",
        description="One command for the experiment shape used every day here: a list "
                    "of labelled expressions, one lint pass, one simulation run, a "
                    "train-only table on stderr, a JSON document on stdout, and a "
                    "ledger entry per arm. The table hides the test period unless "
                    "--show-test is passed, because selecting on it destroys the only "
                    "out-of-sample evidence there is.")
    probe.add_argument("--recover", default=None, metavar="BATCH_JSON",
                       help="Fill a batch file's empty arms by matching expressions to "
                            "alphas already on the platform (no simulation). Use after a "
                            "killed run or a rate-limited batch.")
    probe.add_argument("spec", nargs="?", default=None,
                       help="JSON spec: {name, design?, settings, arms:[{label, expression, "
                            "design?, proxy_of?, settings?}]}; use design=explore for scans. "
                            "Omitted when --recover is given.")
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

    probe.set_defaults(func=experiments.cmd_probe)

    views = sub.add_parser(
        "views",
        help="Who looked at test-period numbers, when, and for which alphas.",
        description="Reads the append-only views log that `probe --show-test` and "
                    "`compare --frozen` write. Hiding a column does not stop the leak; "
                    "recording when it was opened does.")
    views.add_argument("--alpha", default=None, help="Only entries naming this alpha id.")
    views.add_argument("--limit", type=int, default=50)

    views.set_defaults(func=campaign.cmd_views)

    alpha = sub.add_parser("alpha", help="Inspect the account's alphas (quota-free).")
    alpha_sub = alpha.add_subparsers(dest="subcommand", metavar="<action>",
                                     parser_class=JsonArgumentParser)

    a_get = alpha_sub.add_parser("get", help="Fetch one alpha's metrics and checks.")
    a_get.add_argument("alpha_id")
    _add_session_args(a_get)

    a_get.set_defaults(func=alphas.cmd_alpha_get)

    a_chk = alpha_sub.add_parser(
        "check",
        help="Compute live checks (incl. SELF_CORRELATION). 'get' only shows cached values.")
    a_chk.add_argument("alpha_id")
    a_chk.add_argument("--max-wait", type=float, default=180.0,
                       help="Seconds to wait for the async check job (default 180).")
    _add_session_args(a_chk)

    a_chk.set_defaults(func=alphas.cmd_alpha_check)

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
    a_sc.set_defaults(func=alphas.cmd_alpha_self_corr)

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
    a_desc.set_defaults(func=alphas.cmd_alpha_describe)

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

    a_sub.set_defaults(func=alphas.cmd_alpha_submit)

    a_expl = alpha_sub.add_parser(
        "explain",
        help="Explain one alpha: what each leg computes, and what the evidence says.")
    a_expl.add_argument("alpha_id")
    a_expl.add_argument("--no-corr", action="store_true",
                        help="skip the local correlation against the book (reads "
                             "<research>/pnl, which is a few hundred small files)")
    _add_session_args(a_expl)

    a_expl.set_defaults(func=alphas.cmd_alpha_explain)

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

    a_list.set_defaults(func=alphas.cmd_alpha_list)

    ledger_parser = sub.add_parser(
        "ledger", help="Record and explore the durable local research ledger.",
        description="List, label and relate recorded research offline. Only record "
                    "contacts BRAIN: it logs in and fetches an alpha's current metrics "
                    "before appending them; it never runs a simulation.")
    ledger_sub = ledger_parser.add_subparsers(dest="subcommand", metavar="<action>",
                                        parser_class=JsonArgumentParser)

    l_rec = ledger_sub.add_parser(
        "record", help="Fetch one alpha from BRAIN and append it to the ledger (online).",
        description="Requires a BRAIN session and an existing alpha id. Fetches "
                    "current metrics without a new simulation. To change "
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
    l_rec.set_defaults(func=ledger.cmd_ledger_record)

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
    l_list.set_defaults(func=ledger.cmd_ledger_list)

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
    l_lbl.set_defaults(func=ledger.cmd_ledger_label)

    l_imp = ledger_sub.add_parser(
        "impact",
        help="Which other alphas stand on the same legs (offline).",
        description="Leg names are positional, so exposure is matched on the leg "
                    "body: the same sub-expression, not the same letter.")
    l_imp.add_argument("alpha_id")
    l_imp.add_argument("--min-shared", type=int, default=1)
    l_imp.add_argument("--limit", type=int, default=25)
    l_imp.add_argument("--registry", default=None)
    l_imp.set_defaults(func=ledger.cmd_ledger_impact)

    l_lin = ledger_sub.add_parser(
        "lineage",
        help="Walk the parent edges of an alpha (offline).",
        description="Shows where a variant came from and by which operation.")
    l_lin.add_argument("alpha_id")
    l_lin.add_argument("--registry", default=None)
    l_lin.set_defaults(func=ledger.cmd_ledger_lineage)

    l_cov = ledger_sub.add_parser(
        "coverage",
        help="How much of the ledger still lacks labels (offline).",
        description="Independence is counted on the family column; this reports "
                    "how far that column is from being usable.")
    l_cov.add_argument("--registry", default=None)
    l_cov.add_argument("--list-missing", action="store_true",
                       help="Also list keys missing a family, most recent first.")
    l_cov.add_argument("--limit", type=int, default=40)
    l_cov.set_defaults(func=ledger.cmd_ledger_coverage)

    l_sync = ledger_sub.add_parser(
        "sync",
        help="Record platform alphas the ledger has never seen (online, no simulations).",
        description="The ledger only grows when something records an alpha, so it drifts "
                    "from the account after any sweep or manual submission. This walks "
                    "the account and appends rows for ids that are missing; it does not "
                    "re-record existing ones (append-only: that would duplicate a row).")
    l_sync.add_argument("--platform-status", default="ACTIVE",
                        help="Status filter for the listing (default ACTIVE).")
    l_sync.add_argument("--status", default=None,
                        help="Only record alphas whose own status is in this list.")
    l_sync.add_argument("--limit", type=int, default=100, help="Page size for the listing.")
    l_sync.add_argument("--pages", type=int, default=10)
    l_sync.add_argument("--order", default="-dateCreated")
    l_sync.add_argument("--limit-record", type=int, default=25, dest="limit_record")
    l_sync.add_argument("--dry-run", action="store_true", dest="dry_run")
    l_sync.add_argument("--registry", default=None)
    l_sync.set_defaults(func=ledger.cmd_ledger_sync)

    l_fmt = ledger_sub.add_parser(
        "fmt", help="Reformat stored expressions in the book's multi-line style.",
        description="The platform and the ledger both treat newlines as whitespace, so "
                    "this is presentational - but it refuses to write unless every row "
                    "keeps its canonical expression_hash, and it backs the file up first.")
    l_fmt.add_argument("--registry", default=None)
    l_fmt.add_argument("--dry-run", action="store_true", dest="dry_run")
    l_fmt.set_defaults(func=ledger.cmd_ledger_format)

    l_stab = ledger_sub.add_parser(
        "stability",
        help="Train first-half vs second-half Sharpe per alpha; flags sign flips (offline).",
        description="A single train Sharpe can be the average of two opposite regimes. "
                    "The halves stay inside train, so this is evidence selection may use "
                    "before choosing rather than test data after the fact.")
    l_stab.add_argument("--registry", default=None)
    l_stab.add_argument("--where", default=None, help="Ledger filter, e.g. family=quality.")
    l_stab.add_argument("--status", default=None, help="Bare status filter, e.g. ACTIVE.")
    l_stab.add_argument("--min-move", type=float, default=0.3,
                        help="Both halves must exceed this |Sharpe| to count as a flip.")
    l_stab.add_argument("--limit", type=int, default=10)
    l_stab.add_argument("--list", action="store_true", help="Also list every flip.")
    l_stab.set_defaults(func=ledger.cmd_ledger_stability)

    l_sty = ledger_sub.add_parser(
        "styles",
        help="Classify what each alpha compares against: history, peers, or nothing (offline).",
        description="One batch measured 0.87 for a value read cross-sectionally against "
                    "2.05 for the same value read against its own history. This reports "
                    "which book alphas have never been re-read that way, so the cheap "
                    "re-estimation can be aimed instead of guessed.")
    l_sty.add_argument("--registry", default=None)
    l_sty.add_argument("--where", default=None,
                       help="Ledger filter, e.g. status=ACTIVE or family=quality.")
    l_sty.add_argument("--status", default=None,
                       help="Bare status filter, e.g. ACTIVE (comma separated).")
    l_sty.add_argument("--list", action="store_true",
                       help="Also list the level-only alphas in full.")
    l_sty.set_defaults(func=ledger.cmd_ledger_styles)

    buf = sub.add_parser(
        "buffer", help="Hold alphas that passed every check but were not submitted.",
        description="A shelf for finished work. The platform's self-correlation is "
                    "measured against a snapshot, so a held alpha's value decays as the "
                    "book grows; each entry records the book size at shelving time and "
                    "`buffer submit` re-checks before spending the irreversible step.")
    buf_sub = buf.add_subparsers(dest="buffer_command", required=True,
                                 parser_class=JsonArgumentParser)

    b_add = buf_sub.add_parser("add", help="Run the checks and shelve the alpha.")
    b_add.add_argument("alpha_id")
    b_add.add_argument("--note", default=None)
    b_add.add_argument("--max-wait", type=int, default=None, dest="max_wait")
    b_add.add_argument("--buffer", default=None, help="Shelf file (default: in research dir).")
    b_add.add_argument("--registry", default=None)
    b_add.set_defaults(func=buffer_cmd.cmd_buffer_add)

    b_list = buf_sub.add_parser("list", help="Show the shelf and how stale it is.")
    b_list.add_argument("--buffer", default=None)
    b_list.add_argument("--registry", default=None)
    b_list.set_defaults(func=buffer_cmd.cmd_buffer_list)

    b_drop = buf_sub.add_parser("drop", help="Take an alpha off the shelf.")
    b_drop.add_argument("alpha_id")
    b_drop.add_argument("--note", default=None)
    b_drop.add_argument("--buffer", default=None)
    b_drop.add_argument("--registry", default=None)
    b_drop.set_defaults(func=buffer_cmd.cmd_buffer_drop)

    b_sub = buf_sub.add_parser("submit", help="Submit a shelved alpha (irreversible).")
    b_sub.add_argument("alpha_id")
    b_sub.add_argument("--yes", action="store_true")
    b_sub.add_argument("--force", action="store_true",
                       help="Submit even if the fresh self-correlation is above the gate.")
    b_sub.add_argument("--max-wait", type=int, default=None, dest="max_wait")
    b_sub.add_argument("--buffer", default=None)
    b_sub.add_argument("--registry", default=None)
    b_sub.set_defaults(func=buffer_cmd.cmd_buffer_submit)

    fmt = sub.add_parser(
        "fmt", help="Format an alpha expression (or every arm in a spec file).",
        description="The platform accepts newlines - all 55 stored book expressions are "
                    "multi-line with aligned continuations - but everything written here "
                    "ends up on one line. This reformats: one statement per line, wrapped "
                    "chains, indented legs. Idempotent and semantics-preserving.")
    fmt.add_argument("target", help="An expression string, or the path to a spec JSON.")
    fmt.add_argument("--write", action="store_true",
                     help="Rewrite the spec file in place (only for spec paths).")
    fmt.set_defaults(func=experiments.cmd_fmt)

    health = sub.add_parser(
        "health", help="Per-alpha marginal contribution to the book (offline, no sims).",
        description="A book of 55 is not 55 bets, and equal weighting pays every member "
                    "the same. This reports each alpha's marginal contribution "
                    "S_i - rho_i * S_book from cached PnL, so dead weight is visible.")
    health.add_argument("--window", choices=["train", "full"], default="train",
                        help="Train window by default: selecting on the full window "
                             "means selecting on the test year.")
    health.add_argument("--train-end", default="2023-01-01", dest="train_end",
                        help="Exclusive end of the train window (test starts here).")
    health.add_argument("--pool", action="store_true",
                        help="Also score every non-book alpha with cached PnL against "
                             "this book, ranked by marginal contribution.")
    health.add_argument("--min-marginal", type=float, default=0.3, dest="min_marginal",
                        help="Marginal contribution a pool alpha must reach to pass.")
    health.add_argument("--gate", type=float, default=0.70,
                        help="Max correlation to any single book member (platform gate).")
    health.add_argument("--min-sharpe", type=float, default=1.0, dest="min_sharpe",
                        help="A negative-Sharpe alpha can post a big marginal; require a "
                             "real Sharpe for a pool alpha to pass.")
    health.add_argument("--limit", type=int, default=8)
    health.add_argument("--min-overlap", type=int, default=500, dest="min_overlap")
    health.add_argument("--registry", default=None)
    health.set_defaults(func=campaign.cmd_health)

    br = sub.choices["brief"]
    br.add_argument("--drift", action="store_true",
                    help="Also compare the platform's ACTIVE set with the ledger "
                         "(online; the rest of brief is offline).")

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
    a_self.set_defaults(func=alphas.cmd_alpha_self_check)

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
    a_corr.set_defaults(func=alphas.cmd_alpha_corr)

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

    a_book.set_defaults(func=alphas.cmd_alpha_book)

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
    "ask": ["ask --by neutralization --stat n,median:sharpe,floor_share",
            "ask --where status=ACTIVE --list"],
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
    """Return structured usage and command options for a command path."""
    return _help_document(_find_command_parser(getattr(args, "path", [])))


def cmd_schema(args, _sink):
    """Emit the full machine-readable call contract or one command's schema."""
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
