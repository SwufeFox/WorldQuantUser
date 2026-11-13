"""Read the closed-direction list as data, so tools stop recommending dead ends.

Closed directions and their reasons belong in machine-readable research data.
A high historical fitness alone can rank a seed from a closed family first;
tools should surface the recorded reason alongside that recommendation.

The file lives in the research directory, not the repository: it is a judgement about
markets, revised when evidence changes, while the repository versions the tool. It is
loaded the same way as ``field-units.json`` and degrades to "nothing is closed" when
absent, because a missing research file must not break a tool.

Only entries carrying a ``match`` can be applied mechanically. Other entries may
describe a technique that cannot be detected from an expression; they remain
available so a reader gets the whole list from one place rather than a filtered one.
"""

from __future__ import annotations

import json
import os

import research_paths

__all__ = ["load", "closed_families", "reason", "entries", "is_closed_family",
           "path"]

_NAME = "closed-families.json"


def path(explicit=None):
    """Where the file is: an explicit path, else the research directory."""
    return explicit or research_paths.resolve(_NAME)


def load(explicit=None):
    """The parsed file, or an empty shell when it is missing or damaged.

    Returning a shell rather than raising is deliberate: a tool that refuses to run
    because a research note is missing is worse than a tool that runs without the
    note, as long as the absence is visible. ``note`` carries an ``error`` key in
    that case.
    """
    try:
        with open(path(explicit)) as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as error:
        return {"version": 0, "closed": [], "error": str(error), "path": path(explicit)}
    if not isinstance(payload, dict) or not isinstance(payload.get("closed"), list):
        return {"version": 0, "closed": [], "error": "no 'closed' list", "path": path(explicit)}
    payload.setdefault("path", path(explicit))
    return payload


def entries(explicit=None):
    """Every closed direction, in file order."""
    return load(explicit).get("closed") or []


def closed_families(explicit=None):
    """Mechanism families that are closed, e.g. ``{"option"}``.

    Only entries with a ``match.family`` count. A family closed for one construction
    is *not* closed in general, so this is a warning to show next to a recommendation,
    not a filter that silently deletes rows.
    """
    out = set()
    for entry in entries(explicit):
        match = entry.get("match") or {}
        if match.get("family"):
            out.add(match["family"])
    return out


def is_closed_family(family, explicit=None):
    return bool(family) and family in closed_families(explicit)


def reason(family, explicit=None):
    """The reason a family is closed, or ``None`` when it is not."""
    for entry in entries(explicit):
        match = entry.get("match") or {}
        if match.get("family") == family:
            return entry.get("reason")
    return None