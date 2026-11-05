"""Shared CLI contract, JSON output, input settings and authenticated sessions."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import machine_lib as ml
import research_config as config_io


VERSION = "3.0.0"


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


METRIC_KEYS = ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin",
               "longCount", "shortCount", "pnl")


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


def collect_alphas(args, metadata=None):
    """Merge inputs in order, optionally collecting research-file side metadata."""
    file_settings, alphas = {}, []
    paths = ([args.file] if getattr(args, "file", None) else [])
    paths.extend(getattr(args, "files", None) or [])
    if sum(str(path) == "-" for path in paths) > 1:
        raise CliError("INVALID_INPUT", "stdin (-) may be read only once")
    for path in paths:
        try:
            text = config_io.read_input_text(path)
            settings, entries = config_io.parse_research_text(text, source=str(path))
        except config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc)) from exc
        if metadata is not None and text.lstrip().startswith("{"):
            payload = json.loads(text)
            for key in ("legs", "legend", "note"):
                if payload.get(key) is not None:
                    metadata[key] = payload[key]
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
