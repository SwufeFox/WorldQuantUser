#!/usr/bin/env python3
"""Research-file parsing and validation, shared by the CLI and research scripts.

A research file contains Alpha source and optional simulation settings.
The CLI and research scripts use this parser to interpret the same input.

* JSON: an array of expressions, or an object with ``settings`` and
  ``alphas``, or a single object with an ``expression``.
* A named ``.py`` file is one complete Python Alpha, including significant
  whitespace. This applies even if its first character looks like JSON.
* Other text is read as one FASTEXPR expression per line. Blank lines and
  lines starting with ``#`` are ignored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import machine_lib as ml

RESEARCH_FILE_SPEC = {
    "description": "Input to `lint`, `simulate`, and `ablate`: JSON or FASTEXPR text. "
                   "Use - as the file path to read UTF-8 stdin (an optional BOM is accepted).",
    "json": {
        "array": ["raw(liabilities/assets)", {"expression": "rank(cap)", "decay": 4}],
        "object": {
            "settings": {
                "region": "USA",
                "universe": "TOP3000",
                "neutralization": "SUBINDUSTRY",
            },
            "alphas": ["raw(liabilities/assets)", {"expression": "rank(cap)", "decay": 4}],
        },
        "notes": [
            "A bare object with 'expression' is treated as a single alpha.",
            "settings in the file are overridden by explicit command-line flags.",
            "Known snake_case setting names normalize to platform keys before "
            "merging layers; conflicting aliases in one settings object are rejected.",
            "lint and simulate merge repeated --file and --expr inputs; later file "
            "settings override earlier files. Each invocation accepts stdin only once.",
            "ablate requires exactly one composite; its file and --expr inputs are mutually exclusive.",
            "alphas entries may be plain strings, {expression, decay} objects, or "
            "[expression, decay] pairs.",
            "Optional role is candidate or diagnostic. Diagnostic evidence is kept for attribution, not candidate promotion.",
            "Additional setting keys remain available through settings or --setting NAME=VALUE; live account metadata determines support.",
        ],
    },
    "plain_text": {
        "description": "Other text that does not start with '{' or '[' is read line "
                       "by line: one expression per line, blank lines and '# comments skipped.",
        "example": "rank(liabilities/assets)-0.5\n# next one\nrank(cap)",
    },
    "python_source": {
        "description": "For lint and simulate, a .py file is one whole Python Alpha. "
                       "Use --setting lookback=VALUE to specify its history window.",
    },
}


class ResearchError(ValueError):
    """The research file or its settings are not usable as written."""


def parse_research_text(text, source="<text>"):
    """Return ``(settings, alphas)`` from a research file's text.

    Raises :class:`ResearchError` when neither shape applies. Setting names are
    canonicalized; call :func:`build_config` to validate their values.
    """
    if Path(source).suffix.lower() == ".py":
        if not text.strip():
            raise ResearchError("%s contains no Python source" % source)
        return {"language": "PYTHON"}, [ml.normalize_python_source(text)]
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ResearchError(
                "invalid JSON in %s: %s (see `research_cli.py schema` for the "
                "accepted shapes)" % (source, exc)
            ) from exc
        if isinstance(payload, list):
            return {}, payload
        if isinstance(payload, dict):
            settings = payload.get("settings") or {}
            if not isinstance(settings, dict):
                raise ResearchError("'settings' in %s must be an object" % source)
            try:
                settings = ml.normalize_setting_keys(settings)
            except ValueError as exc:
                raise ResearchError("settings in %s: %s" % (source, exc)) from exc
            if "alphas" in payload:
                alphas = payload["alphas"]
                if not isinstance(alphas, list) or not alphas:
                    raise ResearchError(
                        "'alphas' in %s must be a non-empty list" % source
                    )
                return settings, alphas
            if "expression" in payload:
                return settings, [payload]
            raise ResearchError(
                "JSON object in %s has neither 'alphas' nor 'expression'" % source
            )
        raise ResearchError("unsupported JSON top level in %s" % source)

    entries = []
    for line in text.splitlines():
        candidate = line.strip()
        if candidate and not candidate.startswith("#"):
            entries.append(candidate)
    if not entries:
        raise ResearchError("%s contains no expressions" % source)
    return {}, entries


def read_input_text(path):
    """Read UTF-8 text from a path or one complete stdin stream (``-``)."""
    try:
        if str(path) == "-":
            stream = getattr(sys.stdin, "buffer", sys.stdin)
            text = stream.read()
            return (text.decode("utf-8-sig") if isinstance(text, bytes)
                    else text.removeprefix("\ufeff"))
        return Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ResearchError("cannot read %s: %s" % (path, exc)) from exc


def load_research_file(path):
    """Read a research file or stdin and return ``(settings, alphas)``."""
    return parse_research_text(read_input_text(path), str(path))


def build_config(settings=None, alphas=None):
    """Validate and return ``(resolved_settings, [(expression, decay), ...])``.

    Raises :class:`ResearchError` for malformed settings, invalid decays or empty
    expressions. Account availability is checked separately against live metadata.
    """
    try:
        resolved = ml.resolve_settings(settings or {})
        pairs = ml.normalize_alphas(alphas or [], resolved)
    except ValueError as exc:
        raise ResearchError(str(exc)) from exc
    if not pairs:
        raise ResearchError("no alphas to simulate")
    return resolved, pairs
