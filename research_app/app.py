"""CLI execution boundary: workspace, output, error envelopes and exit codes."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import machine_lib as ml
import research_paths

from . import runtime
from .parser import JsonArgumentParser, _add_common_args, _help_document, build_parser
from .runtime import CliError, EXIT_CODES, _error_body, _prepare_output, emit, make_event_sink, note


def _dispatch(args):
    """Validate common argument constraints, configure logging and caches, and invoke command."""
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
    ml.configure_metadata_cache(
        max_age=getattr(args, "meta_max_age", None),
        refresh=getattr(args, "refresh_meta", None),
    )
    return args.func(args, sink)


def main(argv=None):
    """One boundary for parsing, workspace selection, saving and error envelopes."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    options = argparse.Namespace(save=None, compact=False, quiet=False, research_dir=None,
                                 command="help")
    prepared = None
    previous_quiet = runtime._QUIET
    previous_directory = os.environ.get(research_paths.ENV_VAR)
    previous_policy = ml.metadata_cache_policy()
    previous_handler = ml.set_diagnostic_handler(note)
    code = EXIT_CODES["OK"]
    try:
        try:
            # Read only common options first, so errors honour --save/--compact too.
            common = JsonArgumentParser(prog="research_cli.py", add_help=False, common=False)
            _add_common_args(common)
            common.parse_known_args(arguments, namespace=options)
            runtime._QUIET = options.quiet
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
        ml.set_diagnostic_handler(previous_handler)
        if prepared:
            prepared[1].unlink(missing_ok=True)
        runtime._QUIET = previous_quiet
        ml.configure_metadata_cache(**previous_policy)
        if previous_directory is None:
            os.environ.pop(research_paths.ENV_VAR, None)
        else:
            os.environ[research_paths.ENV_VAR] = previous_directory
