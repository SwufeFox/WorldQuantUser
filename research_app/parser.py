"""Command parsing, terminal help, and the JSON command contract."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
from functools import partial
from importlib import import_module

import alpha_registry
import field_contract
import machine_lib as ml
import research_config as config_io
import research_paths
import simulation_multi

from .selection import DEFAULT_PROMOTION_RETENTION
from .runtime import VERSION, EXIT_CODES, ERROR_CODES, METRIC_KEYS, CliError
from .stats import MIN_PORTFOLIO_OVERLAP


def _dispatch_handler(target, args, sink):
    module_name, handler_name = target.split(".")
    module = import_module(f".{module_name}", __package__)
    return getattr(module, handler_name)(args, sink)


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

    def bind_handler(self, target):
        """Bind a command without importing its implementation while parsing."""
        self.set_defaults(func=partial(_dispatch_handler, target))

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


def _add_session_args(parser, *, reference=False):
    parser.add_argument("--no-cache", action="store_true",
                        help=("Bypass the response cache and re-authenticate." if reference else
                              "Ignore the cached session and re-authenticate."))
    parser.add_argument("--relogin", action="store_true",
                        help=("Refresh the response with a fresh login." if reference else
                              "Force a fresh login even if the cache looks valid."))
    cached = "reference content" if reference else "catalog metadata"
    cache_name = "reference cache" if reference else "catalog cache"
    parser.add_argument("--meta-max-age", type=_non_negative_seconds, default=None,
                        metavar="SECONDS",
                        help=f"Accept cached {cached} up to this age "
                             f"(default 86400; 0 disables the {cache_name}).")
    parser.add_argument("--refresh-meta", action="store_true",
                        help=f"Ignore cached {cached} and refetch it.")


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


def _cost_bps(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cost must be a number of basis points") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("cost must be finite and non-negative")
    return number


def _add_cost_args(parser):
    parser.add_argument("--cost-bps", nargs="+", type=_cost_bps, default=None, metavar="BPS",
                        help="One-way cost scenarios per dollar traded; reports net margin, "
                             "not an inferred daily cost path or net Sharpe.")


def _add_selection_args(parser, freeze=True):
    if freeze:
        parser.add_argument("--freeze", default=None, metavar="PATH",
                            help="Save an immutable training selection and its evidence.")
    parser.add_argument("--selection", default=None, metavar="PATH",
                        help="Frozen training choice; validation preserves its candidates and order.")


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


def _non_negative_integer(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def _positive_count(value):
    number = _non_negative_integer(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _setting_assignment(value):
    key, separator, raw = value.partition("=")
    if not separator or not key.strip():
        raise argparse.ArgumentTypeError("expected NAME=VALUE")

    def invalid_constant(token):
        raise ValueError("non-finite JSON constant %s" % token)

    try:
        parsed = json.loads(raw, parse_constant=invalid_constant)
        json.dumps(parsed, allow_nan=False)
    except json.JSONDecodeError as exc:
        if raw.lstrip().startswith(("{", "[", '"')):
            raise argparse.ArgumentTypeError("invalid JSON setting value") from exc
        parsed = raw
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return key.strip(), parsed


def _add_concurrency_args(parser):
    parser.add_argument("--batch-size", dest="concurrency", type=_positive_count, default=None,
                        help="Requested simulation concurrency. Published account limits take precedence; "
                             "unpublished capacity uses a reported local policy, not a claimed platform allowance.")


def _add_multi_size_arg(parser):
    parser.add_argument("--multi-size", type=int, choices=range(2, simulation_multi.MAX_CHILDREN + 1),
                        help="Opt in to native BRAIN Multi with up to this many Alphas per request. "
                             "Requires MULTI_SIMULATION; the size is kept for resume.")


def _add_settings_args(parser):
    parser.add_argument("--region", type=str.upper,
                        help="BRAIN region token; live account metadata determines availability.")
    parser.add_argument("--universe", type=str.upper, help="Universe, e.g. TOP3000.")
    parser.add_argument("--instrument-type", type=str.upper,
                        help="BRAIN instrument type.")
    parser.add_argument("--neutralization", type=str.upper,
                        help="Neutralization token; validated by machine_lib, with availability decided by BRAIN.")
    parser.add_argument("--decay", type=int,
                        help="Default decay for alphas that do not set their own.")
    parser.add_argument("--delay", type=_non_negative_integer,
                        help="Non-negative delay; allowed values come from the current account.")
    parser.add_argument("--truncation", type=float,
                        help="Truncation, e.g. 0.08.")
    parser.add_argument("--pasteurization", type=str.upper)
    parser.add_argument("--nan-handling", type=str.upper)
    parser.add_argument("--test-period", type=str.upper, help="e.g. P0Y, P1Y, P2Y.")
    parser.add_argument("--setting", action="append", type=_setting_assignment, default=None,
                        metavar="NAME=VALUE", help="Additional platform setting; repeatable. "
                        "Values accept JSON or unquoted text. Conflicting explicit flags are rejected.")


def _add_reference_commands(sub):
    def add(group, name, target, help_text):
        command = group.add_parser(name, help=help_text)
        _add_session_args(command, reference=True)
        command.add_argument("--offline", action="store_true",
                             help="Read only a fresh cache entry for the current account. "
                                  "A cache miss returns NOT_FOUND without a network request. "
                                  "Incompatible with --refresh-meta, --no-cache and --relogin.")
        command.bind_handler(target)
        return command

    def paginate(command, *, cursor=False):
        command.add_argument("--limit", type=int, default=10, help="Items per response.")
        if cursor:
            command.add_argument("--cursor", help="Continue from pagination.after_cursor.")
        else:
            command.add_argument("--page", type=int, default=1, help="Page number.")

    docs = sub.add_parser("docs", help="Read official BRAIN documentation and Help Center articles.")
    docs_sub = docs.add_subparsers(dest="subcommand", metavar="<action>", parser_class=JsonArgumentParser)
    docs_list = add(docs_sub, "list", "docs_reader.cmd_docs_list", "List BRAIN documentation by category.")
    docs_list.add_argument("--query", help="Filter category titles, page titles and IDs.")
    docs_list.add_argument("--summary", action="store_true", help="Return a flat list of page IDs, titles and links.")
    docs_get = add(docs_sub, "get", "docs_reader.cmd_docs_get", "Read a BRAIN documentation page by ID or keyword.")
    page = docs_get.add_mutually_exclusive_group(required=True)
    page.add_argument("slug", nargs="?", metavar="PAGE_ID", help="Page ID from docs list.")
    page.add_argument("--query", help="Read the unique page matching this title or keyword.")
    docs_get.add_argument("--text", action="store_true", help="Return readable text instead of original content blocks.")
    articles = add(docs_sub, "articles", "reference.cmd_docs_articles", "List English Help Center articles in a category.")
    articles.add_argument("category_id", help="Numeric ID from the Help Center category link.")
    paginate(articles)
    article = add(docs_sub, "article", "reference.cmd_docs_article", "Read a Help Center article's HTML body.")
    article.add_argument("article_id", help="Numeric ID from the Help Center article link.")

    forum = sub.add_parser("forum", help="Search and read the signed-in BRAIN community.")
    forum_sub = forum.add_subparsers(dest="subcommand", metavar="<action>", parser_class=JsonArgumentParser)
    topics = add(forum_sub, "topics", "reference.cmd_forum_topics", "List forum topics visible to this account.")
    paginate(topics, cursor=True)
    search = add(forum_sub, "search", "reference.cmd_forum_search", "Search forum posts by relevance.")
    search.add_argument("query", help="Search phrase.")
    paginate(search)
    post = add(forum_sub, "get", "reference.cmd_forum_get", "Read a forum post's HTML body.")
    post.add_argument("post_id", help="Numeric ID from the forum post link.")
    comments = add(forum_sub, "comments", "reference.cmd_forum_comments", "Read forum comments and their HTML bodies.")
    comments.add_argument("post_id", help="Numeric ID from the forum post link.")
    paginate(comments, cursor=True)


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
    schema.set_defaults(func=partial(cmd_schema, parser))

    help_command = sub.add_parser("help", help="Show a concise guide or help for a command path.")
    help_command.add_argument("path", nargs="*", metavar="COMMAND",
                              help="Command path, e.g. help catalog fields.")
    help_command.set_defaults(func=partial(cmd_help, parser))

    auth = sub.add_parser("auth", help="Credential and session checks.")
    auth_sub = auth.add_subparsers(dest="subcommand", metavar="<check>",
                                   parser_class=JsonArgumentParser)
    auth_check = auth_sub.add_parser(
        "check", help="Verify credentials and report account permissions.")
    _add_session_args(auth_check)
    auth_check.bind_handler("account.cmd_auth_check")

    auth_logout = auth_sub.add_parser(
        "logout", help="Forget the cached session cookie (local only).")
    auth_logout.bind_handler("account.cmd_auth_logout")

    pairs_parser = sub.add_parser(
        "pairs", help="Screen cached alphas for a combination worth simulating.",
        description="For every pair of non-book alphas with cached PnL, builds the "
                    "equal-weight composite, and keeps it only if it stays under the "
                    "self-correlation gate against every book alpha. Offline.")
    pairs_parser.add_argument("--min-leg-sharpe", type=float, default=0.7,
                              dest="min_leg_sharpe",
                              help="Ignore legs weaker than this in absolute Sharpe.")
    pairs_parser.add_argument("--max-leg-sharpe", type=float, default=2.0,
                              dest="max_leg_sharpe",
                              help="Ignore legs above this: they do not need a partner.")
    pairs_parser.add_argument("--max-book-rho", type=float, default=0.7, dest="max_book_rho",
                              help="Reject pairs whose composite correlates above this with "
                                   "any book alpha (the platform gate is 0.7).")
    pairs_parser.add_argument("--min-composite-sharpe", type=float, default=1.25,
                              dest="min_composite_sharpe")
    pairs_parser.add_argument("--min-marginal", type=float, default=None, dest="min_marginal",
                              help="Keep only pairs whose composite beats the book as a "
                                   "whole: S_composite - rho(book) * S_book >= this.")
    pairs_parser.add_argument("--min-support-ratio", type=float, default=None,
                              dest="min_support_ratio",
                              help="Keep pairs whose legs hold comparable numbers of names "
                                   "(min/max of long+short). A 400-name leg next to a "
                                   "3000-name leg cannot be added as alphas.")
    pairs_parser.add_argument("--plain-legs", action="store_true", dest="plain_legs",
                              help="Keep only pairs whose legs are plain signals, where "
                                   "the leg-sum prediction is exact. Centred legs "
                                   "(group_rank minus its mean) overstate the composite.")
    pairs_parser.add_argument("--rank", choices=("sharpe", "marginal"), default="sharpe",
                              help="Order the kept pairs by their own Sharpe or by what "
                                   "they would add to the book.")
    pairs_parser.add_argument("--limit", type=int, default=12)
    pairs_parser.add_argument("--registry", default=None)
    pairs_parser.bind_handler("pairs.cmd_pairs")

    marginal_parser = sub.add_parser(
        "marginal",
        help="[superseded by `triage --ids`] Slope vs the book's actual change.",
        description="Reports the theoretical marginal S_a - rho*S_B next to the Sharpe the "
                    "book actually moves by when the alpha is added equal-weight. The two "
                    "differ, and only the second one happened.")
    marginal_parser.add_argument("alpha_id")
    marginal_parser.add_argument("--registry", default=None)
    marginal_parser.bind_handler("independence.cmd_marginal")

    independence_parser = sub.add_parser(
        "independence", help="Are the book's alphas one bet or many, calm and stressed?",
        description="Pairwise correlations over the full sample, the worst decile of book "
                    "days and the best decile, plus an effective-bet count. Full-sample "
                    "independence is not independence when the book is losing.")
    independence_parser.add_argument("--threshold", type=float, default=0.7)
    independence_parser.add_argument("--registry", default=None)
    independence_parser.bind_handler("independence.cmd_independence")

    legs_parser = sub.add_parser(
        "legs", help="Did a composite behave like the sum of its legs? (cached PnL only)",
        description="Sums the cached daily PnL of the legs with the given weights and "
                    "compares that Sharpe to the composite's. Both are dollars on the same "
                    "book, so the sum is the leg model exactly; the ratio is what the "
                    "platform's per-expression re-neutralisation took away.")
    legs_parser.add_argument("--leg", action="append", metavar="ID[:WEIGHT]",
                             help="Leg alpha id with optional weight (default equal).")
    legs_parser.add_argument("--residual", action="store_true",
                             help="Fit r_composite ~ a*r_leg1 + b*r_leg2 and report the "
                                  "calibrated residual, coefficients and date coverage. "
                                  "Uncalibrated ratios confuse scale with structure.")
    legs_parser.add_argument("--composite", default=None, metavar="ID",
                             help="The simulated composite to compare against.")
    legs_parser.add_argument("--registry", default=None)
    legs_parser.bind_handler("legs.cmd_legs")

    mine_parser = sub.add_parser(
        "mine",
        help="Review cached fields, definition gaps and related ledger evidence.",
        description="Offline screening in the default simulation scope: separate numeric "
                    "fields from labels and VECTOR/EVENT reductions needing a definition. "
                    "Show cache provenance, same-description history and train-only dataset "
                    "turnover. Unused field ids are not new mechanisms. --emit drafts a "
                    "scan; it does not verify economic meaning, PIT or live availability.")
    mine_parser.add_argument("--dataset", default=None,
                             help="Restrict to one dataset id (default: every dataset).")
    mine_parser.add_argument("--min-coverage", type=float, default=0.3,
                             dest="min_coverage")
    mine_parser.add_argument("--turnover-band", default="0.08,0.25", dest="turnover_band",
                             help="MIN,MAX for sorting observed dataset train turnover; "
                                  "not a field cadence estimate.")
    mine_parser.add_argument("--limit", type=int, default=4,
                             help="Rows to show per dataset in each review category.")
    mine_parser.add_argument("--expr", default=None,
                             help="Semicolon-separated sub-expressions to apply --shape to. "
                                  "Choose vector reductions explicitly after reviewing their semantics.")
    mine_parser.add_argument("--shape", default="raw",
                             choices=("raw", "centre", "midtier", "change", "ratio_assets"),
                             help="Construction to apply to each --expr.")
    mine_parser.add_argument("--smooth", type=int, default=0,
                             help="Wrap each field in ts_mean(x, N) before ranking; daily "
                                  "attention fields reorder every day without it.")
    mine_parser.add_argument("--arms", type=int, default=8,
                             help="Arms to emit in the spec.")
    mine_parser.add_argument("--emit", default=None, metavar="PATH",
                             help="Draft a scan of numeric-review fields or explicit --expr; "
                                  "never silently aggregate vectors or overwrite an existing spec.")
    mine_parser.add_argument("--name", default=None, help="Spec batch name.")
    mine_parser.add_argument("--registry", default=None)
    mine_parser.bind_handler("mine.cmd_mine")

    prune_parser = sub.add_parser(
        "prune", help="Drop cached results that fail today's bar (dry run by default).",
        description="Reads <research>/prune-criteria.json so a changed requirement is a "
                    "file edit, not a code change. Reports what would go; --apply deletes "
                    "the cached PnL files. Book members, submitted alphas and anything "
                    "listed in `protect` are never touched.")
    prune_parser.add_argument("--apply", action="store_true",
                              help="Delete the files. Without this nothing is written.")
    prune_parser.add_argument("--criteria", default=None, metavar="PATH",
                              help="Criteria file (default <research>/prune-criteria.json).")
    prune_parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                              help="Override one criterion, e.g. --set min_fitness=1.2.")
    prune_parser.add_argument("--show-criteria", action="store_true", dest="show_criteria",
                              help="Print the criteria in force and stop.")
    prune_parser.add_argument("--limit", type=int, default=20)
    prune_parser.add_argument("--registry", default=None)
    prune_parser.bind_handler("prune.cmd_prune")

    bench_parser = sub.add_parser(
        "benchmark",
        help="[superseded by `triage --internal`] Internal fitness vs the platform's.",
        description="Discount platform Fitness using train-Sharpe priors, the book "
                    "delta and block retention. Operator count produces auditability "
                    "warnings without changing the score.")
    bench_parser.add_argument("--batch", default=None, metavar="BATCH_JSON")
    bench_parser.add_argument("--ids", nargs="+", default=None, metavar="ID")
    bench_parser.add_argument("--criteria", default=None,
                              help="Weights file (default: built-in priors).")
    bench_parser.add_argument("--registry", default=None)
    bench_parser.bind_handler("benchmark.cmd_benchmark")
    _add_selection_args(bench_parser, freeze=False)
    _add_cost_args(bench_parser)

    triage_parser = sub.add_parser(
        "triage",
        help="Train-only candidate screening; explicit test view after design freeze.",
        description="Compare candidates or book members using train metrics, costs and PnL. "
                    "Save a choice with --freeze; --show-test requires --selection and "
                    "cannot change the frozen candidates or their order.")
    triage_parser.add_argument("--batch", default=None, metavar="BATCH_JSON",
                               help="Triage every alpha id in this batch file.")
    triage_parser.add_argument("--ids", nargs="+", default=None, metavar="ID")
    triage_parser.add_argument("--book", action="store_true",
                               help="Score every book member instead of candidates "
                                    "(this is what `health` did).")
    triage_parser.add_argument("--internal", action="store_true",
                               help="Report test-calibrated internal fitness; requires --show-test.")
    triage_parser.add_argument("--show-test", action="store_true",
                               help="Validate --selection using full-window and test evidence; logs the view.")
    triage_parser.add_argument("--train-end", metavar="YYYY-MM-DD",
                               help="Exclusive train end; cannot extend past recorded test starts. "
                                    "Default: earliest recorded test start, else 2023-01-01. "
                                    "Shelf screening requires known test starts for the compared alphas.")
    triage_parser.add_argument("--min-delta", type=float, default=None, dest="min_delta",
                               help="Minimum train delta; validation retains the frozen criterion.")
    triage_parser.add_argument("--retention-floor", type=float, default=None,
                               dest="retention_floor",
                               help="Validation test/train floor; retained in the frozen choice.")
    triage_parser.add_argument("--registry", default=None)
    triage_parser.add_argument("--against-shelf", action="store_true",
                               help="Compare candidates against the shelf on the same training window.")
    triage_parser.add_argument("--max-shelf-correlation", type=float, default=None, metavar="RHO",
                               help="With --against-shelf, require positive train contribution and a "
                                    "maximum absolute shelf correlation below this ceiling "
                                    "before writing a requested freeze.")
    triage_parser.add_argument("--buffer", default=None, metavar="PATH",
                               help="Candidate shelf used by --against-shelf.")
    _add_selection_args(triage_parser)
    _add_cost_args(triage_parser)
    triage_parser.bind_handler("triage.cmd_triage")

    promote_parser = sub.add_parser(
        "promote", help="Freeze, validate, check and shelve one candidate.",
        description="Freeze the candidate or reuse --selection, then validate, check and add to the shelf. "
                    "Keep every completed step's evidence; stop on an operational failure. "
                    "Dry runs still open the test view and perform platform checks.")
    promote_parser.add_argument("alpha_id", metavar="ALPHA_ID")
    promote_parser.add_argument("--note", default=None)
    retention_option = promote_parser.add_argument(
        "--retention-floor", type=float, default=None,
        help=f"Minimum test/train Sharpe retention (new selections default to {DEFAULT_PROMOTION_RETENTION}).")
    retention_option.documented_default = DEFAULT_PROMOTION_RETENTION
    promote_parser.add_argument("--max-self-correlation", type=float, default=None, metavar="RHO",
                                help="Require a measured live self-correlation below this value.")
    promote_source = promote_parser.add_mutually_exclusive_group()
    promote_source.add_argument("--freeze", default=None, metavar="PATH",
                                help="Frozen selection path (default: promote-freeze-ID-timestamp.json in research dir).")
    promote_source.add_argument("--selection", default=None, metavar="PATH",
                                help="Validate an existing single-candidate training selection.")
    promote_parser.add_argument("--buffer", default=None, metavar="PATH",
                                help="Shelf file (default: in research dir).")
    promote_parser.add_argument("--dry-run", action="store_true",
                                help="Run validation and platform checks. Report would_buffer without "
                                     "writing the shelf if eligible.")
    _add_cost_args(promote_parser)
    promote_parser.bind_handler("promote.cmd_promote")

    feed_parser = sub.add_parser(
        "feed",
        help="A bounded research digest across papers, news, social sources, forums and market observations.",
        description="Collect selected sources, retain timestamps and provenance, and balance research leads. "
                    "Evidence, source claims and hypotheses remain distinct; collection does not run simulations.")
    feed_parser.add_argument("--topic", action="append", default=None,
                             help="Search this instead of deriving from open-threads.")
    feed_parser.add_argument("--source", action="append", default=None, metavar="ID",
                             help="Source ID: paper, news, social, forum, trends, all, or a configured ID; repeatable.")
    feed_parser.add_argument("--sources-file", default=None, metavar="PATH",
                             help="JSON source definitions for RSS/Atom, markets, local exports or structured harvests.")
    feed_parser.add_argument("--market", action="append", default=None, metavar="VENUE:SYMBOL",
                             help="Explicit market observation, e.g. coinbase:BTC-USD or alpaca:AAPL; repeatable.")
    feed_parser.add_argument("--book-depth", type=_positive_count, default=10,
                             help="Local order-book retention depth, 1-50; does not change provider coverage.")
    feed_parser.add_argument("--market-feed", type=str.lower, default="iex",
                             help="Alpaca stock feed; default iex is one exchange, with provider-specific access requirements.")
    feed_parser.add_argument("--summary", action="store_true",
                             help="Return a compact digest/table with source outcomes, coverage and evidence paths.")
    feed_parser.add_argument("--seed", action="append", default=None, metavar="ID",
                             help="Expand references/citations of a known OpenAlex ID or DOI; repeatable.")
    feed_parser.add_argument("--follow", choices=("references", "citations"), default="references",
                             help="Citation direction for --seed; an edge is a reading lead, not evidence of agreement.")
    feed_parser.add_argument("--from-threads", action="store_true", dest="from_threads",
                             help="Derive queries from open-threads.md.")
    feed_parser.add_argument("--topics", action="store_true",
                             help="Also read <research>/inspiration/topics.md "
                                  "(English queries, one per line).")
    feed_parser.add_argument("--topics-file", default=None, dest="topics_file")
    feed_parser.add_argument("--limit", type=int, default=5,
                             help="Items per retrieval job.")
    feed_parser.add_argument("--total", type=int, default=10,
                             help="Items in the digest.")
    feed_parser.add_argument("--max-topics", type=_non_negative_integer, default=4, dest="max_topics",
                             help="Maximum retrieval jobs across queries, seeds, fixed feeds and market observations.")
    feed_parser.add_argument("--workers", type=int, choices=range(1, 9), default=4,
                             help="Concurrent retrieval jobs, bounded independently of result count.")
    feed_parser.add_argument("--timeout", type=_positive_seconds, default=30.0,
                             metavar="SECONDS", help="Network timeout for each source request.")
    feed_parser.add_argument("--from-year", type=int, default=2005, dest="from_year")
    feed_parser.add_argument("--min-citations", type=_non_negative_integer, default=0,
                             dest="min_citations", help="Optional citation floor; default 0 keeps new papers eligible.")
    feed_parser.add_argument("--seen", default=None)
    feed_parser.add_argument("--draft-cards", default=None, dest="draft_cards",
                             metavar="PATH")
    feed_parser.add_argument("--action", action="store_true", dest="action",
                             help="Add declared book exposure and prior-trial context; unknown is not unexplored.")
    feed_parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                             help="Report only; do not update the seen store.")
    feed_parser.add_argument("--region", type=str.upper,
                             default=ml.DEFAULT_SETTINGS["region"])
    feed_parser.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    feed_parser.add_argument("--delay", type=_non_negative_integer, default=ml.DEFAULT_SETTINGS["delay"])
    feed_parser.add_argument("--instrument-type", type=str.upper,
                             default=ml.DEFAULT_SETTINGS["instrumentType"])
    feed_parser.add_argument("--meta-max-age", type=_non_negative_seconds, default=None,
                             metavar="SECONDS", help="Maximum age of local scoped capability evidence.")
    def _feed_dispatch(args, sink):
        target = "feed.cmd_feed_action" if args.action else "feed.cmd_feed"
        return _dispatch_handler(target, args, sink)

    feed_parser.set_defaults(func=_feed_dispatch)

    api_parser = sub.add_parser(
        "api", help="Probe which BRAIN endpoints this account can actually read.")
    api_sub = api_parser.add_subparsers(dest="action", metavar="<action>",
                                        parser_class=JsonArgumentParser)
    api_probe = api_sub.add_parser(
        "probe", help="GET a fixed list of endpoints and report status codes.")
    api_probe.add_argument("--alpha", default=None,
                           help="An alpha id for the per-alpha endpoints.")
    api_probe.bind_handler("account.cmd_api_probe")

    _add_reference_commands(sub)

    cache = sub.add_parser(
        "cache", help="Inspect or clear local caches (session, catalog metadata).")
    cache_sub = cache.add_subparsers(dest="subcommand", metavar="<action>",
                                     parser_class=JsonArgumentParser)
    cache_info = cache_sub.add_parser(
        "info", help="Where caches live, what they hold, and how stale they are.")
    cache_info.bind_handler("account.cmd_cache_info")
    cache_clear = cache_sub.add_parser(
        "clear", help="Drop cached data. Locks are kept for running processes.")
    cache_clear.add_argument("--kind", choices=["session", "metadata", "all"],
                             default="all", help="What to clear (default: all).")
    cache_clear.bind_handler("account.cmd_cache_clear")

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
    cat_op.bind_handler("catalog.cmd_catalog_operators")

    cat_ds = cat.add_parser("datasets", help="Datasets for a region/universe/delay.")
    cat_ds.add_argument("--instrument-type", type=str.upper, default=ml.DEFAULT_SETTINGS["instrumentType"])
    cat_ds.add_argument("--region", type=str.upper, default=ml.DEFAULT_SETTINGS["region"])
    cat_ds.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_ds.add_argument("--delay", type=_non_negative_integer, default=ml.DEFAULT_SETTINGS["delay"])
    cat_ds.add_argument("--limit", type=int, default=None)
    cat_ds.add_argument("--theme", choices=["true", "false"], default=None,
                        help="Filter datasets by the platform theme flag; omit to leave it unset.")
    cat_ds.add_argument("--summary", action="store_true",
                        help="Bound catalog rows and show reported Pyramid/Theme metadata.")

    _add_session_args(cat_ds)
    cat_ds.bind_handler("catalog.cmd_catalog_datasets")

    cat_find = cat.add_parser(
        "find", help="Search every dataset's fields at once (offline once cached).")
    cat_find.add_argument("query", nargs="?", help="Search terms; multi-word queries are ANDed. "
                          "May be omitted when filtering by usage, coverage or field type.")
    cat_find.add_argument("--dataset", default=None,
                          help="Comma-separated dataset ids to restrict to.")
    cat_find.add_argument("--min-coverage", type=float, default=None, dest="min_coverage",
                          help="Drop fields whose coverage is below this.")
    cat_find.add_argument("--max-alpha-count", type=int,
                          help="Keep fields with a reported alphaCount at or below this value.")
    cat_find.add_argument("--type", dest="field_type", type=str.upper, help="Keep a native BRAIN field type.")
    cat_find.add_argument("--sort", choices=["relevance", "pyramid-multiplier"], default="relevance",
                          help="Order by exact ID and coverage, or by descending reported Pyramid multiplier.")
    cat_find.add_argument("--stream", action="store_true", help="Emit catalog scan progress on stderr.")
    cat_find.add_argument("--fuzzy", action="store_true")
    cat_find.add_argument("--any-term", action="store_true", dest="any_term")
    cat_find.add_argument("--region", type=str.upper, default=ml.DEFAULT_SETTINGS["region"])
    cat_find.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_find.add_argument("--delay", type=_non_negative_integer, default=ml.DEFAULT_SETTINGS["delay"])
    cat_find.add_argument("--instrument-type", type=str.upper, default=ml.DEFAULT_SETTINGS["instrumentType"])
    cat_find.add_argument("--limit", type=int, default=25)
    _add_session_args(cat_find)
    cat_find.bind_handler("catalog.cmd_catalog_find")

    cat_unused = cat.add_parser(
        "unused",
        help="Which catalog fields the book has never touched, by dataset.")
    cat_unused.add_argument("--top", type=int, default=4,
                            help="Highest-coverage unused fields to show per dataset.")
    cat_unused.add_argument("--registry", default=None)
    cat_unused.bind_handler("catalog.cmd_catalog_unused")

    cat_f = cat.add_parser("fields", help="Datafields of one dataset.")
    cat_f.add_argument("--dataset", required=True, help="Dataset id, e.g. fundamental6.")
    cat_f.add_argument("--region", type=str.upper, default=ml.DEFAULT_SETTINGS["region"])
    cat_f.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_f.add_argument("--delay", type=_non_negative_integer, default=ml.DEFAULT_SETTINGS["delay"])
    cat_f.add_argument("--instrument-type", type=str.upper, default=ml.DEFAULT_SETTINGS["instrumentType"])
    cat_f.add_argument("--type", dest="field_type", type=str.upper,
                       help="Keep a native BRAIN field type, including types newly exposed by this account.")
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
    cat_f.add_argument("--summary", action="store_true",
                       help="Bound field rows and show reported Pyramid/Theme metadata.")

    _add_session_args(cat_f)
    cat_f.bind_handler("catalog.cmd_catalog_fields")

    cat_cc = cat.add_parser(
        "concepts",
        help="Group fields by concept and show variants side by side.")
    cat_cc.add_argument("--dataset", default=None,
                        help="Limit to one dataset id, e.g. analyst4.")
    cat_cc.add_argument("--region", type=str.upper, default=ml.DEFAULT_SETTINGS["region"])
    cat_cc.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_cc.add_argument("--delay", type=_non_negative_integer, default=ml.DEFAULT_SETTINGS["delay"])
    cat_cc.add_argument("--instrument-type", type=str.upper, default=ml.DEFAULT_SETTINGS["instrumentType"])
    cat_cc.add_argument("--type", dest="field_type", type=str.upper,
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
    cat_cc.bind_handler("catalog.cmd_catalog_concepts")

    cat_vf = cat.add_parser(
        "verify",
        help="Check field ids against the live catalog (catches typed-from-memory names).")
    cat_vf.add_argument("--fields", nargs="+", default=[],
                        help="Field ids, comma or space separated.")
    cat_vf.add_argument("--region", type=str.upper, default=ml.DEFAULT_SETTINGS["region"])
    cat_vf.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_vf.add_argument("--delay", type=_non_negative_integer, default=ml.DEFAULT_SETTINGS["delay"])
    cat_vf.add_argument("--instrument-type", type=str.upper, default=ml.DEFAULT_SETTINGS["instrumentType"])
    cat_vf.add_argument("--strict", action="store_true",
                        help="Exit with INVALID_INPUT when any name is unknown.")

    _add_session_args(cat_vf)
    cat_vf.bind_handler("catalog.cmd_catalog_verify")

    cat_c = cat.add_parser("categories", help="Data categories with crowding stats.")
    cat_c.add_argument("--region", type=str.upper, default=ml.DEFAULT_SETTINGS["region"])
    cat_c.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_c.add_argument("--delay", type=_non_negative_integer, default=ml.DEFAULT_SETTINGS["delay"])

    _add_session_args(cat_c)
    cat_c.bind_handler("catalog.cmd_catalog_categories")

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
    cat_audit.add_argument("--region", type=str.upper, default=ml.DEFAULT_SETTINGS["region"])
    cat_audit.add_argument("--universe", type=str.upper, default=ml.DEFAULT_SETTINGS["universe"])
    cat_audit.add_argument("--delay", type=_non_negative_integer, default=ml.DEFAULT_SETTINGS["delay"])
    cat_audit.add_argument("--instrument-type", type=str.upper, default=ml.DEFAULT_SETTINGS["instrumentType"])

    _add_session_args(cat_audit)
    cat_audit.bind_handler("catalog.cmd_catalog_field_audit")

    lint = sub.add_parser(
        "lint",
        help="Preflight FASTEXPR expressions or Python Alpha source (quota-free).")
    lint.add_argument("expression", nargs="?",
                      help="Expression to lint, e.g. 'rank(ts_delta(close, 5))'.")
    lint.add_argument("--file", dest="files", action="append", default=[],
                      help="Research file, Python source (.py), or - for stdin. "
                           "Later file settings override earlier files.")
    lint.add_argument("--expr", action="append", default=[], metavar="EXPRESSION",
                      help="Inline expression; repeat and combine with files or the positional expression.")
    lint.add_argument("--offline", action="store_true",
                      help="Check structure without platform catalogs or account access.")
    lint.add_argument("--dataset", default=None,
                      help="Treat this dataset's fields as known so unknown "
                           "identifiers are flagged.")
    lint.add_argument("--strict-fields", action="store_true",
                      help="Treat unknown identifiers as errors, not warnings.")
    _add_settings_args(lint)
    lint.add_argument("--denylist", default=None,
                      help="Override the learned denylist path.")

    _add_session_args(lint)
    lint.bind_handler("validation.cmd_lint")

    papers = sub.add_parser(
        "paper",
        help="Find, inspect and read papers or follow their citation graph via OpenAlex.")
    papers.add_argument("query", nargs="?", help="Keyword query; use --id to inspect a known paper.")
    papers.add_argument("--id", dest="paper_id", default=None, metavar="ID",
                        help="OpenAlex W ID or DOI, including their standard HTTPS URLs.")
    papers.add_argument("--related", choices=("references", "citations"), default=None,
                        help="Retrieve a bounded set of graph neighbours of --id.")
    papers.add_argument("--full-text", action="store_true",
                        help="Read advertised OpenAlex structured full text for --id; availability may be unknown.")
    papers.add_argument("--offset", type=_non_negative_integer, default=None,
                        help="Character offset for --full-text (default 0).")
    papers.add_argument("--max-chars", type=_positive_count, default=None,
                        help="Characters per full-text reading window (default 12000); complete text stays cached.")
    papers.add_argument("--timeout", type=_positive_seconds, default=30.0,
                        metavar="SECONDS", help="Network timeout per paper request.")
    papers.add_argument("--limit", type=int, default=10,
                        help="Number of papers, 1-50 (default 10).")
    papers.add_argument("--from-year", type=int, default=None,
                        help="Only papers published in or after this year.")
    papers.add_argument("--min-citations", type=int, default=None,
                        help="Drop papers with fewer citations than this.")

    papers.bind_handler("literature.cmd_paper")

    sim = sub.add_parser(
        "simulate", help="Simulate alphas; preview with --dry-run before using platform capacity.")
    sim.add_argument("file", nargs="?",
                     help="Research file: JSON, one FASTEXPR expression per line, or one Python Alpha (.py). "
                          "Omit when using --expr.")
    sim.add_argument("--expr", action="append", default=[], metavar="EXPRESSION",
                     help="Inline expression; repeat for several.")
    sim.add_argument("--file", dest="files", action="append", default=[],
                     help="Additional research file or - for stdin; repeat to merge expressions. "
                          "Later file settings override earlier files.")
    _add_settings_args(sim)
    _add_concurrency_args(sim)
    _add_multi_size_arg(sim)
    sim.add_argument("--max-wait-seconds", type=_positive_seconds, default=300.0,
                     help="Shared wait budget for login, preflight and simulation. Longer Retry-After "
                          "waits are saved for resume (default: 300).")
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
                     help="Permit a FASTEXPR launch with lint findings. Python preflight and "
                          "unsupported setting errors still block.")
    sim.add_argument("--sort", choices=["none", *METRIC_KEYS], default="none",
                     help="Sort results by this metric, best first.")
    sim.add_argument("--top", type=int, default=None,
                     help="Keep only the first N results after sorting.")
    sim.add_argument("--stream", action="store_true",
                     help="Emit JSON Lines progress events on stderr.")
    sim.add_argument("--verbose", action="store_true",
                     help="With --stream, also emit per-poll events.")
    _add_session_args(sim)
    sim.bind_handler("experiments.cmd_simulate")

    resume = sub.add_parser(
        "resume", help="Resume pending/queued simulations from a checkpoint.")
    resume.add_argument("checkpoint", help="Checkpoint JSON produced by simulate/resume.")
    _add_concurrency_args(resume)
    resume.add_argument("--max-wait-seconds", type=_positive_seconds, default=300.0,
                        help="Shared wait budget for login and resumed simulations (default: 300).")
    resume.add_argument("--sort", choices=["none", *METRIC_KEYS], default="none",
                        help="Sort results by this metric, best first.")
    resume.add_argument("--top", type=int, default=None,
                        help="Keep only the first N results after sorting.")
    resume.add_argument("--stream", action="store_true",
                        help="Emit JSON Lines progress events on stderr.")
    resume.add_argument("--verbose", action="store_true",
                        help="With --stream, also emit per-poll events.")
    _add_session_args(resume)
    resume.bind_handler("experiments.cmd_resume")

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

    abl.bind_handler("experiments.cmd_ablate")

    cmp_ = sub.add_parser(
        "compare",
        help="Find and compare saved research batches using training evidence.",
        description="Read explicit result files or discover matching batches in the research "
                    "directory. Select the latest files and inspect labelled train metrics "
                    "without globbing or parsing JSON in a separate script.")
    cmp_.add_argument("files", nargs="*", help="Explicit result JSON paths, or - for stdin.")
    cmp_.add_argument("--match", action="append", default=[], metavar="GLOB",
                      help="Find result files under the research directory; quote the glob. Repeatable.")
    cmp_.add_argument("--latest", nargs="?", type=int, const=1, default=None, metavar="N",
                      help="Read only the newest N matching files by modification time (1 when omitted).")
    cmp_.add_argument("--min-fitness", type=float, default=None)
    cmp_.add_argument("--sort", default=None,
                      choices=["none", "fitness", "sharpe", "turnover", "returns", "id"],
                      help="Train sort (default fitness); frozen validation retains saved order.")
    cmp_.add_argument("--frozen", action="store_true",
                      help="Validate --selection without filtering or reordering on test evidence.")
    cmp_.add_argument("--contributions", action="store_true",
                      help="Also compute per-leg contributions from ablate labels.")
    cmp_.add_argument("--per-file", action="store_true",
                      help="Report contributions per input file so identically named ablation arms stay separate.")
    _add_selection_args(cmp_)
    _add_cost_args(cmp_)
    cmp_.add_argument("--summary", action="store_true",
                      help="Show bounded evidence rows with run identity, scope and missing evidence.")
    cmp_.add_argument("--format", choices=("json", "markdown"), default="json",
                      help="Include a Markdown table inside the JSON result; missing evidence stays unknown.")

    cmp_.bind_handler("experiments.cmd_compare")

    brief = sub.add_parser(
        "brief",
        help="One bounded document describing where the research stands (no quota; train-only scores).",
        description="Book composition, train-only recent scores and setbacks, source "
                    "paths, new write-ups and open questions. Start here, then read "
                    "only the evidence needed for the current research question.")
    brief.add_argument("--days", type=float, default=7.0,
                       help="How far back 'recent' reaches (default 7).")
    brief.add_argument("--top", type=int, default=5,
                       help="Maximum items per list: scores, setbacks, write-ups, and each thread state (at least 1).")
    brief.add_argument("--registry", default=None)

    brief.bind_handler("campaign.cmd_brief")

    ask = sub.add_parser(
        "ask",
        help="Query the local ledger: filter, group, summarise, correlate.",
        description="Filter ledger entries, group statistics, measure within-panel slopes, "
                    "or correlate cached PnL without network requests.")
    ask.add_argument("--where", default="",
                     help="Comma list of field=value / field!=value; fields: " +
                          ", ".join(alpha_registry.QUERY_FIELDS) +
                          " plus check=NAME:RESULT and has=metrics|pnl|checks.")
    ask.add_argument("--by", default=None,
                     help="Grouping axis; one of " + ", ".join(alpha_registry.QUERY_FIELDS) + ".")
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
    ask.bind_handler("ledger.cmd_ask")

    probe = sub.add_parser(
        "probe",
        help="Run a batch of arms from a spec file (lint, simulate, table, record).",
        description="Prepare, check, run and recover labelled experiments from the same spec. "
                    "--dry-run is offline; --check verifies live metadata without simulations. "
                    "Results include durable path, run identity, per-arm outcomes and next actions. "
                    "--recover and --from-simulation only read platform jobs; "
                    "--resume explicitly starts unsubmitted work.")
    probe.add_argument("--allow-lint-errors", action="store_true", dest="allow_lint_errors",
                       help="Permit a FASTEXPR launch with lint findings. Python preflight and "
                            "unsupported setting errors still block.")
    probe.add_argument("--strict-gate", action="store_true", dest="strict_gate",
                       help="Refuse a finished prior result with the same expression and "
                            "effective settings. Dry-run previews the blocking evidence.")
    probe_source = probe.add_mutually_exclusive_group()
    probe_source.add_argument("--recover", default=None, metavar="BATCH_JSON",
                              help="Refresh accepted jobs and reconcile local evidence without new simulations.")
    probe_source.add_argument("--resume", default=None, metavar="BATCH_JSON",
                              help="Continue a saved batch, including definitely unsubmitted arms; "
                                   "use --dry-run to preview first.")
    probe_source.add_argument("--from-simulation", metavar="ID_OR_LINK",
                              help="Read an existing BRAIN multi-simulation and list its Alpha IDs.")
    probe.add_argument("spec", nargs="?", default=None,
                       help="JSON experiment spec. Arms take an inline expression or a relative Python "
                            "source_file. Use design=explore for scans or mixed-language batches. "
                            "Omitted with --recover, --resume or --from-simulation.")
    _add_settings_args(probe)
    _add_session_args(probe)
    _add_concurrency_args(probe)
    _add_multi_size_arg(probe)
    probe.add_argument("--name", default=None,
                       help="Batch name for the output file (default: the spec's name).")
    probe_mode = probe.add_mutually_exclusive_group()
    probe_mode.add_argument("--dry-run", action="store_true",
                            help="Offline request preview; unchecked catalogs remain explicit.")
    probe_mode.add_argument("--check", action="store_true",
                            help="Validate this spec against live scoped metadata without simulating.")
    probe.add_argument("--batch-path", default=None, metavar="PATH",
                       help="Durable batch/checkpoint path; distinct from the final --save response.")
    probe.add_argument("--summary", action="store_true",
                       help="Return a bounded train summary with a table, execution state and saved evidence paths.")
    probe.add_argument("--defer-quick-pnl", action="store_true",
                       help="Save QUICK results before fetching daily PnL. Run --recover without this flag to fetch it.")
    probe.add_argument("--max-wait-seconds", type=_positive_seconds, default=300.0,
                       help="Shared wait budget for login, preflight, simulation and recovery reads; "
                            "accepted jobs remain recoverable (default: 300).")
    probe.add_argument("--show-test", action="store_true",
                       help="Validate --selection while recovering saved results; never run new simulations.")
    probe.add_argument("--force", action="store_true",
                       help="Overwrite an existing batch file of the same name.")
    probe.add_argument("--no-record", action="store_true",
                       help="Do not append the arms to the ledger.")
    probe.add_argument("--registry", default=None)
    _add_selection_args(probe, freeze=False)
    _add_cost_args(probe)

    probe.bind_handler("experiments.cmd_probe")

    campaign = sub.add_parser("campaign", help="Preview, inspect and advance the durable research queue.")
    campaign_sub = campaign.add_subparsers(dest="subcommand", metavar="<action>",
                                           parser_class=JsonArgumentParser)
    campaign_preview = campaign_sub.add_parser(
        "preview", help="Inspect BRAIN arms or compare recorded SPC outputs without starting simulations.")
    campaign_preview.add_argument("spec", metavar="SPEC")
    _add_multi_size_arg(campaign_preview)
    campaign_preview.bind_handler("campaign.cmd_campaign_preview")

    campaign_prepare = campaign_sub.add_parser(
        "prepare-full", help="Draft bounded FULL simulations from a saved QUICK batch's training results.")
    campaign_prepare.add_argument("batch", metavar="BATCH")
    campaign_prepare.add_argument("--out", required=True, metavar="SPEC", help="Path for the raw Probe spec.")
    campaign_prepare.add_argument("--limit", required=True, type=_positive_count,
                                  help="Maximum FULL arms, including the control in a controlled experiment.")
    campaign_prepare.bind_handler("campaign.cmd_campaign_prepare_full")

    campaign_breed = campaign_sub.add_parser(
        "breed", help="Draft a paired offspring generation from saved Probe training results.")
    campaign_breed.add_argument("batch", metavar="BATCH")
    campaign_breed.add_argument("--out", required=True, metavar="SPEC",
                                help="Path for the raw offspring Probe spec.")
    campaign_breed.add_argument("--seed", required=True, type=int, help="Seed for the parent lottery and child sampling.")
    campaign_breed.add_argument("--limit", required=True, type=_positive_count,
                                help="Maximum offspring per policy, one per mechanism and settings stratum.")
    campaign_breed.bind_handler("campaign.cmd_campaign_breed")

    campaign_status = campaign_sub.add_parser("status", help="Read queued specs and their saved progress offline.")
    campaign_status.add_argument("--limit", type=_non_negative_integer, default=20,
                                 help="Maximum queued specs and archived PnL batches to display in each list.")
    campaign_status.bind_handler("campaign.cmd_campaign_status")

    campaign_pnl = campaign_sub.add_parser(
        "recover-pnl", help="Fill missing daily PnL for archived QUICK batches without starting simulations.")
    campaign_pnl.add_argument("--limit", type=_positive_count, default=10,
                              help="Maximum batches to attempt per pass.")
    campaign_pnl.add_argument("--watch", action="store_true", help="Revisit pending evidence in successive passes.")
    campaign_pnl.add_argument("--interval", type=_positive_seconds, default=30,
                              help="Seconds between passes in watch mode.")
    campaign_pnl.add_argument("--max-wait-seconds", type=_positive_seconds, default=60,
                              help="Maximum wait budget for each Probe recovery.")
    campaign_pnl.add_argument("--stream", action="store_true", help="Emit recovery events on stderr.")
    campaign_pnl.bind_handler("campaign.cmd_campaign_recover_pnl")

    campaign_run = campaign_sub.add_parser(
        "run", help="Advance one queued experiment or keep watching the same inbox.")
    campaign_run.add_argument("--watch", action="store_true", help="Keep consuming the research inbox.")
    campaign_run.add_argument("--interval", type=_positive_seconds, default=30,
                              help="Seconds between passes in watch mode.")
    campaign_run.add_argument("--max-wait-seconds", type=_positive_seconds, default=600,
                              help="Maximum wait budget for each Probe invocation.")
    campaign_run.add_argument("--recover-only", action="store_true",
                              help="Recover existing simulation tasks without posting unstarted arms.")
    campaign_run.add_argument("--defer-quick-pnl", action="store_true",
                              help="Let QUICK batches finish before fetching their daily PnL.")
    campaign_run.add_argument("--validate", action="store_true",
                              help="Validate eligible Regular FULL candidates after freezing training evidence.")
    campaign_run.add_argument("--stream", action="store_true", help="Emit campaign events on stderr.")
    _add_concurrency_args(campaign_run)
    _add_multi_size_arg(campaign_run)
    campaign_run.bind_handler("campaign.cmd_campaign_run")

    views = sub.add_parser(
        "views",
        help="Who looked at test-period numbers, when, and for which alphas.",
        description="Reads the append-only log of validation views and their frozen selection identity.")
    views.add_argument("--alpha", default=None, help="Only entries naming this alpha id.")
    views.add_argument("--limit", type=int, default=50)

    views.bind_handler("campaign.cmd_views")

    alpha = sub.add_parser("alpha", help="Inspect the account's alphas (quota-free).")
    alpha_sub = alpha.add_subparsers(dest="subcommand", metavar="<action>",
                                     parser_class=JsonArgumentParser)

    a_get = alpha_sub.add_parser("get", help="Fetch one alpha's metrics and checks.")
    a_get.add_argument("alpha_id")
    a_get.add_argument("--performance", nargs="?", const="summary", choices=["summary", "full"],
                       help="Also read platform merged before/after performance; full includes all native series.")
    a_get.add_argument("--recordsets", nargs="*", metavar="NAME",
                       help="Discover native diagnostic recordsets and optionally read names such as yearly-stats.")
    a_get_scope = a_get.add_mutually_exclusive_group()
    a_get_scope.add_argument("--team", dest="team_id", help="Team scope for --performance; default is the current user.")
    a_get_scope.add_argument("--competition", dest="competition_id", help="Competition scope for --performance.")
    a_get.add_argument("--max-wait", type=_non_negative_seconds, default=180.0,
                       help="Wait budget for each requested evidence view (performance or recordsets).")
    _add_session_args(a_get)

    a_get.bind_handler("alphas.cmd_alpha_get")

    a_near = alpha_sub.add_parser(
        "near", help="Book alphas closest in structure to an expression (offline).",
        description="Duplicate detection arrives too late by PnL (needs the simulation) "
                    "or by eye (unreliable at 55 alphas). This compares AST skeletons and "
                    "field sets instead: an identical skeleton means 'same construction, "
                    "different grid point'.")
    a_near.add_argument("--alpha-id", default=None, dest="alpha_id")
    a_near.add_argument("--expression", default=None)
    a_near.add_argument("--status", default=None, help="Filter candidates by status, e.g. ACTIVE.")
    a_near.add_argument("--limit", type=int, default=8)
    a_near.add_argument("--registry", default=None)
    a_near.bind_handler("alphas.cmd_alpha_near")

    a_chk = alpha_sub.add_parser(
        "check",
        help="Compute live checks for one or more alphas, preserving pending/error evidence.")
    a_chk.add_argument("alpha_ids", nargs="+", metavar="ALPHA_ID")
    a_chk.add_argument("--workers", type=int, choices=[1, 2, 3], default=1,
                       help="Concurrent alpha checks; each worker uses isolated authenticated session state.")
    a_chk.add_argument("--max-wait", type=_non_negative_seconds, default=180.0,
                       help="Seconds to wait for each alpha's async check job (default 180).")
    _add_session_args(a_chk)

    a_chk.bind_handler("alphas.cmd_alpha_check")

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
    a_sc.bind_handler("alphas.cmd_alpha_self_corr")

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
    a_desc.bind_handler("alphas.cmd_alpha_describe")

    a_sub = alpha_sub.add_parser(
        "submit",
        help="Submit one alpha to its competition (dry-run unless --yes).")
    a_sub.add_argument("alpha_id")
    a_sub.add_argument("--yes", action="store_true",
                       help="Actually submit. Without it nothing is written.")
    a_sub.add_argument("--max-wait", type=float, default=180.0)
    a_sub.add_argument("--registry", default=None)
    a_sub.add_argument("--buffer", default=None,
                       help="Shelf to reconcile after a confirmed submission.")
    _add_selection_args(a_sub, freeze=False)
    _add_cost_args(a_sub)
    _add_session_args(a_sub)

    a_sub.bind_handler("alphas.cmd_alpha_submit")

    a_expl = alpha_sub.add_parser(
        "explain",
        help="Explain one alpha: what each leg computes, and what the evidence says.")
    a_expl.add_argument("alpha_id")
    a_expl.add_argument("--no-corr", action="store_true",
                        help="skip the local correlation against the book (reads "
                             "<research>/pnl, which is a few hundred small files)")
    _add_session_args(a_expl)

    a_expl.bind_handler("alphas.cmd_alpha_explain")

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
                        help="Merge visible and hidden alphas, including SUPER; "
                             "offset/limit apply to the combined sorted page.")
    _add_session_args(a_list)

    a_list.bind_handler("alphas.cmd_alpha_list")

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
    l_rec.bind_handler("ledger.cmd_ledger_record")

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
    l_list.add_argument("--regex", action="store_true",
                        help="Treat --expression as a regular expression.")
    l_list.add_argument("--all-fields", default=None, dest="all_fields",
                        help="Comma-separated field ids that must ALL appear, e.g. "
                             "debt_st,assets.")
    l_list.add_argument("--limit", type=int, default=0,
                        help="Maximum entries to show, ranked by fitness; 0 shows all (default).")
    l_list.add_argument("--format", choices=["json", "markdown"], default="json",
                        help="Markdown adds a table inside the JSON envelope.")
    l_list.add_argument("--registry", default=None)
    l_list.bind_handler("ledger.cmd_ledger_list")

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
    l_lbl.bind_handler("ledger.cmd_ledger_label")

    l_imp = ledger_sub.add_parser(
        "impact",
        help="Which other alphas stand on the same legs (offline).",
        description="Leg names are positional, so exposure is matched on the leg "
                    "body: the same sub-expression, not the same letter.")
    l_imp.add_argument("alpha_id")
    l_imp.add_argument("--min-shared", type=int, default=1)
    l_imp.add_argument("--limit", type=int, default=25)
    l_imp.add_argument("--registry", default=None)
    l_imp.bind_handler("ledger.cmd_ledger_impact")

    for parser_name in ("l_lst",):
        pass
    l_lin = ledger_sub.add_parser(
        "lineage",
        help="Walk the parent edges of an alpha (offline).",
        description="Shows where a variant came from and by which operation.")
    l_lin.add_argument("alpha_id")
    l_lin.add_argument("--registry", default=None)
    l_lin.bind_handler("ledger.cmd_ledger_lineage")

    l_cov = ledger_sub.add_parser(
        "coverage",
        help="How much of the ledger still lacks labels (offline).",
        description="Independence is counted on the family column; this reports "
                    "how far that column is from being usable.")
    l_cov.add_argument("--registry", default=None)
    l_cov.add_argument("--list-missing", action="store_true",
                       help="Also list keys missing a family, most recent first.")
    l_cov.add_argument("--limit", type=int, default=40)
    l_cov.bind_handler("ledger.cmd_ledger_coverage")

    l_sync = ledger_sub.add_parser(
        "sync",
        help="Reconcile platform membership, local records and optional PnL evidence.",
        description="Read the complete requested platform scope before resolving reverse drift. "
                    "Report listing completeness, record limits, member identity and data coverage.")
    l_sync.add_argument("--platform-status", default="ACTIVE",
                        help="Status filter for the listing (default ACTIVE).")
    l_sync.add_argument("--status", default=None,
                        help="Only record alphas whose own status is in this list.")
    l_sync.add_argument("--limit", type=int, default=100, help="Page size for the listing.")
    l_sync.add_argument("--pages", type=int, default=10)
    l_sync.add_argument("--order", default="-dateCreated")
    l_sync.add_argument("--limit-record", type=int, default=25, dest="limit_record")
    l_sync.add_argument("--max-wait-seconds", type=float, default=300.0,
                        help="Network wait budget shared by login, listing, metrics and PnL (default 300s).")
    l_sync.add_argument("--dry-run", action="store_true", dest="dry_run")
    l_sync.add_argument("--registry", default=None)
    l_sync.add_argument("--hidden", action="store_true", help="Select the platform's hidden-alpha scope.")
    l_sync.add_argument("--alpha-type-not", default=None, metavar="TYPE",
                        help="Explicit type exclusion; omitted means no type exclusion.")
    l_sync.add_argument("--pnl", action="store_true",
                        help="Refresh daily PnL and record pagination/coverage evidence.")
    l_sync.bind_handler("ledger.cmd_ledger_sync")

    l_fmt = ledger_sub.add_parser(
        "fmt", help="Reformat stored expressions in the book's multi-line style.",
        description="The platform and the ledger both treat newlines as whitespace, so "
                    "this is presentational - but it refuses to write unless every row "
                    "keeps its canonical expression_hash, and it backs the file up first.")
    l_fmt.add_argument("--registry", default=None)
    l_fmt.add_argument("--dry-run", action="store_true", dest="dry_run")
    l_fmt.bind_handler("ledger.cmd_ledger_format")

    l_decay = ledger_sub.add_parser(
        "decay", help="How much train Sharpe survives to the full window, per block.")
    l_decay.add_argument("--min-train-sharpe", type=float, default=0.8,
                         dest="min_train_sharpe")
    l_decay.add_argument("--min-arms", type=int, default=15, dest="min_arms")
    l_decay.add_argument("--registry", default=None)
    l_decay.bind_handler("ledger.cmd_ledger_decay")

    l_form = ledger_sub.add_parser(
        "form", help="Level vs change construction, measured per information block.")
    l_form.add_argument("--min-arms", type=int, default=20, dest="min_arms")
    l_form.add_argument("--min-per-form", type=int, default=10, dest="min_per_form")
    l_form.add_argument("--include-book", action="store_true", dest="include_book")
    l_form.add_argument("--registry", default=None)
    l_form.bind_handler("ledger.cmd_ledger_form")

    l_cards = ledger_sub.add_parser(
        "cards", help="Mechanism cards: the queue of external knowledge (offline).",
        description="A card is the only shape external knowledge may enter in: source, "
                    "economic mechanism, falsification condition, closest existing family, "
                    "and fields split into conceptual vs verified. A card whose "
                    "verified_brain_fields names a field the catalog does not have is "
                    "reported and not stored.")
    l_cards.add_argument("--add", default=None, metavar="FILE",
                         help="Append cards from a JSON file (object or list).")
    l_cards.add_argument("--show", default=None, metavar="MECHANISM_ID")
    l_cards.add_argument("--status", default=None,
                         help="Filter by status, e.g. UNTESTED,TESTING.")
    l_cards.add_argument("--limit", type=int, default=20)
    l_cards.add_argument("--cards", default=None, help="Card store path.")
    l_cards.bind_handler("ledger.cmd_ledger_cards")

    l_mech = ledger_sub.add_parser(
        "mechanisms", help="Independent mechanisms and their outcomes (offline).",
        description="Groups by (economic block, AST skeleton): the campaign's rule that "
                    "parameter variants of one idea count as one bet, made countable "
                    "without a simulation.")
    l_mech.add_argument("--where", default=None)
    l_mech.add_argument("--status", default=None)
    l_mech.add_argument("--with-marginal", action="store_true", dest="with_marginal",
                        help="Also compute each book alpha's marginal contribution.")
    l_mech.add_argument("--list", action="store_true", help="Show every mechanism.")
    l_mech.add_argument("--limit", type=int, default=15)
    l_mech.add_argument("--registry", default=None)
    l_mech.bind_handler("ledger.cmd_ledger_mechanisms")

    l_conv = ledger_sub.add_parser(
        "convergence", help="Per-batch novelty and output, for divergence decisions.",
        description="Reports arms, distinct mechanisms, structural novelty and additive "
                    "output per batch. Structural novelty is a weak trigger: the last "
                    "hundred arms were 93% new skeletons and almost none were additive.")
    l_conv.add_argument("--batches", type=int, default=8)
    l_conv.add_argument("--no-marginal", action="store_true", dest="no_marginal")
    l_conv.add_argument("--registry", default=None)
    l_conv.bind_handler("ledger.cmd_ledger_convergence")

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
    l_stab.add_argument("--polars", action="store_true",
                        help="Use a rebuildable training PnL Parquet cache for bulk queries (optional Polars).")
    l_stab.bind_handler("ledger.cmd_ledger_stability")

    l_sty = ledger_sub.add_parser(
        "styles",
        help="Classify what each alpha compares against: history, peers, or nothing (offline).",
        description="Report each alpha's comparison set and identify variables that "
                    "have not been compared with their own history. This helps target "
                    "controlled re-estimation of the same input.")
    l_sty.add_argument("--registry", default=None)
    l_sty.add_argument("--where", default=None,
                       help="Ledger filter, e.g. status=ACTIVE or family=quality.")
    l_sty.add_argument("--status", default=None,
                       help="Bare status filter, e.g. ACTIVE (comma separated).")
    l_sty.add_argument("--list", action="store_true",
                       help="Also list the level-only alphas in full.")
    l_sty.bind_handler("ledger.cmd_ledger_styles")

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
    _add_selection_args(b_add, freeze=False)
    _add_cost_args(b_add)
    b_add.bind_handler("buffer_cmd.cmd_buffer_add")

    b_list = buf_sub.add_parser("list", help="Show the shelf and how stale it is.")
    b_list.add_argument("--buffer", default=None)
    b_list.add_argument("--registry", default=None)
    _add_cost_args(b_list)
    b_list.bind_handler("buffer_cmd.cmd_buffer_list")

    b_drop = buf_sub.add_parser("drop", help="Take an alpha off the shelf.")
    b_drop.add_argument("alpha_id")
    b_drop.add_argument("--note", default=None)
    b_drop.add_argument("--buffer", default=None)
    b_drop.add_argument("--registry", default=None)
    b_drop.bind_handler("buffer_cmd.cmd_buffer_drop")

    b_sub = buf_sub.add_parser("submit", help="Submit a shelved alpha (irreversible).")
    b_sub.add_argument("alpha_id")
    b_sub.add_argument("--yes", action="store_true")
    b_sub.add_argument("--force", action="store_true",
                       help="Submit even if the fresh self-correlation is above the gate.")
    b_sub.add_argument("--max-wait", type=int, default=None, dest="max_wait")
    b_sub.add_argument("--buffer", default=None)
    b_sub.add_argument("--registry", default=None)
    _add_selection_args(b_sub, freeze=False)
    _add_cost_args(b_sub)
    b_sub.bind_handler("buffer_cmd.cmd_buffer_submit")

    fmt = sub.add_parser(
        "fmt", help="Preview review-friendly formatting of native FASTEXPR.",
        description="Format an expression or the FASTEXPR arms in a draft spec. "
                    "Changes are checked against the parsed expression. Python arms are skipped.")
    fmt.add_argument("target", help="A native FASTEXPR expression or a draft spec JSON.")
    fmt.add_argument("--write", action="store_true",
                     help="Write verified formatting changes to a draft spec in place.")
    fmt.bind_handler("experiments.cmd_fmt")

    health = sub.add_parser(
        "health",
        help="[superseded by `triage --book`] Per-member contribution to the book.",
        description="A book of 55 is not 55 bets, and equal weighting pays every member "
                    "the same. This reports each alpha's marginal contribution "
                    "S_i - rho_i * S_book from cached PnL, so dead weight is visible.")
    health.add_argument("--window", choices=["train", "full"], default="train",
                        help="Train window by default: selecting on the full window "
                             "means selecting on the test year.")
    health.add_argument("--train-end", default="2023-01-01", dest="train_end",
                        help="Upper bound on the exclusive train cutoff, further capped by recorded test starts.")
    health.add_argument("--pool", action="store_true",
                        help="Also score every non-book alpha with cached PnL against "
                             "this book, ranked by marginal contribution.")
    health.add_argument("--min-marginal", type=float, default=0.3, dest="min_marginal",
                        help="Marginal contribution a pool alpha must reach to pass.")
    health.add_argument("--gate", type=float, default=0.70,
                        help="Maximum absolute daily PnL correlation to one member in this local screen.")
    health.add_argument("--min-sharpe", type=float, default=1.0, dest="min_sharpe",
                        help="A negative-Sharpe alpha can post a big marginal; require a "
                             "real Sharpe for a pool alpha to pass.")
    health.add_argument("--limit", type=int, default=8)
    health.add_argument("--min-overlap", type=int, default=MIN_PORTFOLIO_OVERLAP, dest="min_overlap")
    health.add_argument("--registry", default=None)
    health.bind_handler("campaign.cmd_health")

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
                        help="Screening adjustment added to absolute daily Pearson "
                             "(default 0.015); does not replace a platform check.")
    a_self.add_argument("--top", type=int, default=3,
                        help="How many counterparts to list (default 3).")
    a_self.add_argument("--min-overlap", type=int, default=200,
                        help="Minimum shared trading days (default 200).")
    a_self.add_argument("--peer-limit", type=int, default=300,
                        help="How many peers to scan per status (default 300).")
    a_self.add_argument("--max-age-hours", type=float, default=168.0,
                        help="Trust a cached PnL series younger than this (default 168).")
    _add_session_args(a_self)
    a_self.bind_handler("alphas.cmd_alpha_self_check")

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
    a_corr.bind_handler("alphas.cmd_alpha_corr")

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

    a_book.bind_handler("alphas.cmd_alpha_book")

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
    "docs": ['docs get --query "power pool" --text', "docs list --query python --summary"],
    "docs list": ["docs list --query python --summary"],
    "docs get": ['docs get --query "power pool" --text',
                 "docs get how-translate-fast-expression-alpha-python-alpha",
                 "docs get how-translate-fast-expression-alpha-python-alpha --offline"],
    "docs articles": ["docs articles 41681062887191 --limit 5"],
    "docs article": ["docs article 41681217386903"],
    "forum": ['forum search "Systematic Predictions Challenge" --limit 5'],
    "forum topics": ["forum topics --limit 5"],
    "forum search": ['forum search "Systematic Predictions Challenge" --limit 5'],
    "forum get": ["forum get 41113878259479", "forum get 41113878259479 --refresh-meta"],
    "forum comments": ["forum comments 41996433774359 --limit 5"],
    "cache": ["cache info"],
    "cache info": ["cache info"],
    "cache clear": ["cache clear --kind metadata"],
    "catalog": ["catalog datasets --limit 10", "catalog fields --dataset fundamental6 --limit 10"],
    "catalog operators": ["catalog operators --search rank --limit 10"],
    "catalog datasets": ["catalog datasets --region USA --limit 10",
                         "catalog datasets --theme true --summary"],
    "catalog fields": ["catalog fields --dataset fundamental6 --search liability --expressions --limit 10",
                       "catalog fields --dataset fundamental6 --summary"],
    "catalog find": ["catalog find debt --min-coverage 0.8",
                     "catalog find --region EUR --universe TOP2500 --dataset analyst69,analyst15 "
                     "--type MATRIX --max-alpha-count 0 --min-coverage 0.8 --sort pyramid-multiplier --stream"],
    "catalog concepts": ["catalog concepts --dataset fundamental6 --limit 10"],
    "catalog verify": ["catalog verify --fields close volume --strict"],
    "catalog categories": ["catalog categories"],
    "catalog field-audit": ["catalog field-audit --fields close,volume"],
    "lint": ['lint --expr "rank(close)" --offline', "lint --file research.json",
             "lint --file reversal.py --offline"],
    "paper": ['paper "accruals stock returns" --limit 3',
              'paper --id W2741809807', 'paper --id W2741809807 --related references --limit 5',
              'paper --id W2741809807 --full-text --max-chars 12000'],
    "feed": ["feed --from-threads --workers 4 --timeout 30 --dry-run",
             'feed --topic "analyst revisions" --topic "earnings quality" --action',
             'feed --seed W2741809807 --follow citations --dry-run',
             'feed --source all --topic "earnings guidance" --max-topics 12 --summary --dry-run',
             'feed --sources-file radar-sources.json --summary --dry-run',
             'feed --market coinbase:BTC-USD --book-depth 10 --summary --dry-run'],
    "simulate": ['simulate --expr "rank(close)" --dry-run',
                 "simulate reversal.py --setting lookback=0 --dry-run",
                 "simulate research.json --sort fitness --save results.json"],
    "resume": ["resume CHECKPOINT.json --save results.json"],
    "ablate": ['ablate --expr "a = rank(close); b = rank(volume); a + b" --out variants.json'],
    "compare": ["compare results.json --cost-bps 5 10 20 --freeze choice.json --summary",
                "compare --frozen --selection choice.json --summary",
                'compare --match "sample-batch*.json" --latest --sort none'],
    "triage": ["triage --batch BATCH.json --freeze choice.json",
               "triage --ids ALPHA_ID --against-shelf --max-shelf-correlation 0.7 --freeze choice.json",
               "triage --show-test --selection choice.json"],
    "promote": ['promote ALPHA_ID --note "frozen validation passed" --cost-bps 5 10 20',
                "promote ALPHA_ID --selection choice.json --max-self-correlation 0.7",
                "promote ALPHA_ID --dry-run"],
    "brief": ["brief", "brief --days 7 --top 5"],
    "ask": ["ask --by neutralization --stat n,median:sharpe,floor_share",
            "ask --where status=ACTIVE --list"],
    "probe": ["probe batch.spec.json --dry-run", "probe batch.spec.json --check",
              "probe python.spec.json --batch-path PYTHON_BATCH.json --dry-run",
              "probe batch.spec.json --batch-path BATCH.json --save response.json",
              "probe --recover BATCH.json --max-wait-seconds 60",
              "probe --resume BATCH.json --dry-run", "probe --from-simulation SIMULATION_ID"],
    "campaign": ["campaign status", "campaign preview batch.spec.json",
                 "campaign prepare-full quick-batch.json --out full.spec.json --limit 4",
                 "campaign breed batch.json --out offspring.spec.json --seed 17 --limit 4",
                 "campaign run --recover-only", "campaign run --watch --multi-size 4",
                 "campaign recover-pnl --watch --stream"],
    "views": ["views --limit 10"],
    "alpha": ["alpha list --limit 10", "alpha get ALPHA_ID"],
    "alpha get": ["alpha get ALPHA_ID", "alpha get ALPHA_ID --performance",
                  "alpha get ALPHA_ID --performance full --competition COMPETITION_ID",
                  "alpha get ALPHA_ID --recordsets yearly-stats"],
    "alpha check": ["alpha check ALPHA_ID", "alpha check ALPHA_A ALPHA_B --workers 2"],
    "alpha self-corr": ["alpha self-corr ALPHA_ID"],
    "alpha describe": ['alpha describe ALPHA_ID --name "quality signal" --category FUNDAMENTAL'],
    "alpha submit": ["alpha submit ALPHA_ID", "alpha submit ALPHA_ID --yes --selection choice.json"],
    "buffer submit": ["buffer submit ALPHA_ID --yes --selection choice.json"],
    "alpha explain": ["alpha explain ALPHA_ID --no-corr"],
    "alpha list": ["alpha list --limit 10 --min-fitness 1"],
    "alpha self-check": ["alpha self-check ALPHA_ID"],
    "alpha corr": ["alpha corr ALPHA_A ALPHA_B --vs-book"],
    "alpha book": ["alpha book --limit 10"],
    "ledger": ["ledger list --family quality", "ledger coverage --list-missing"],
    "ledger sync": ["ledger sync --dry-run", "ledger sync --pnl"],
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


def _find_command_parser(parser, path):
    parts = [word for part in path for word in part.split()]
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
        default = getattr(action, "documented_default", action.default)
        if default not in (None, False, [], argparse.SUPPRESS):
            entry["default"] = default
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


def cmd_help(parser, args, _sink):
    """Return structured usage and command options for a command path."""
    return _help_document(_find_command_parser(parser, args.path))


def cmd_schema(parser, args, _sink):
    """Emit the full machine-readable call contract or one command's schema."""
    from .research_route import SPEC_CONTRACT

    if args.name:
        selected = _find_command_parser(parser, args.name)
        return {"tool": "research_cli.py", "version": VERSION,
                "exit_codes": EXIT_CODES, "error_codes": ERROR_CODES,
                **({"campaign_spec": SPEC_CONTRACT} if args.name[0] in {"campaign", "probe"} else {}),
                "command": describe_parser(selected, " ".join(selected.prog.split()[1:]))}
    described = describe_parser(parser)
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
        "campaign_spec": SPEC_CONTRACT,
        "settings_defaults": ml.DEFAULT_SETTINGS,
        "settings_hints": {
            "region": ml.REGIONS,
            "neutralization": ml.NEUTRALIZATIONS,
            "universe": ml.UNIVERSES,
        },
        "settings_validation": {
            "offline": "structural validation only; descriptive hints are not permission limits",
            "online_source": "authenticated OPTIONS /simulations",
            "discovery": "auth check",
            "extensions": "--setting NAME=VALUE or settings in a research file",
            "unknown": "unpublished or stale capability evidence does not prove availability",
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
