#!/usr/bin/env python3
"""Research-file parsing and validation, shared by the CLI and the GUI.

A "research file" is the unit of work this toolchain trades in: a set of
expressions plus the settings to simulate them with. Keeping the format here
means the CLI, the GUI and any test agree on exactly one definition.

Two shapes are accepted:

* JSON -- an array of expressions, or an object with ``settings`` and
  ``alphas``, or a single object with an ``expression``.
* Plain text -- anything that does not start with ``{`` or ``[`` is read line
  by line: one expression per line, blank lines and ``#`` comments ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import machine_lib as ml

RESEARCH_FILE_SPEC = {
    "description": "Input to `simulate`. Either JSON or a plain line-based file.",
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
            "alphas entries may be plain strings, {expression, decay} objects, or "
            "[expression, decay] pairs.",
        ],
    },
    "plain_text": {
        "description": "Any file that does not start with '{' or '[' is read line "
                       "by line: one expression per line, blank lines and '# comments skipped.",
        "example": "rank(liabilities/assets)-0.5\n# next one\nrank(cap)",
    },
}


class ResearchError(ValueError):
    """The research file or its settings are not usable as written."""


def parse_research_text(text, source="<text>"):
    """Return ``(settings, alphas)`` from a research file's text.

    Raises :class:`ResearchError` when neither shape applies. Values are left
    unnormalized; call :func:`build_config` to validate them.
    """
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


def load_research_file(path):
    """Read a research file from disk and return ``(settings, alphas)``."""
    path = Path(path)
    if not path.is_file():
        raise ResearchError("research file not found: %s" % path)
    return parse_research_text(path.read_text(), str(path))


def build_config(settings=None, alphas=None):
    """Validate and return ``(resolved_settings, [(expression, decay), ...])``.

    Raises :class:`ResearchError` for unknown regions/neutralizations, bad
    decays or empty expressions -- i.e. everything worth catching before
    spending simulation quota.
    """
    try:
        resolved = ml.resolve_settings(settings or {})
        pairs = ml.normalize_alphas(alphas or [], resolved)
    except ValueError as exc:
        raise ResearchError(str(exc)) from exc
    if not pairs:
        raise ResearchError("no alphas to simulate")
    return resolved, pairs