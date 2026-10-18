#!/usr/bin/env python3
"""Where research *results* live, as opposed to the *tool*.

The repository holds code only. Everything a run produces -- simulation result
files, field-catalog snapshots, the alpha registry, the learned operator
denylist -- belongs outside the working tree, so that `git status` stays about
"did the tool change", not "did I run something today".

Persistence matters as much as location. In a container, ``$HOME`` frequently
sits on an ephemeral layer that is thrown away when the container is rebuilt,
while the mounted workspace survives. A registry that evaporates is no better
than one that was never written, so the persistent workspace wins when it is
available.

Resolution order:

1. ``BRAIN_RESEARCH_DIR`` environment variable (always wins).
2. ``<workspace>/worldquant-research`` for a writable, persistent workspace
   directory -- ``/data`` on this deployment.
3. ``$XDG_DATA_HOME/worldquant-research`` when ``XDG_DATA_HOME`` is set.
4. ``~/.local/share/worldquant-research``.

:func:`research_dir` only resolves the path; :func:`ensure_dir` creates it.
"""

from __future__ import annotations

import os

__all__ = ["research_dir", "ensure_dir", "resolve", "describe", "source"]

APP_DIR_NAME = "worldquant-research"
ENV_VAR = "BRAIN_RESEARCH_DIR"

# Mounted volumes that survive a container rebuild, in preference order.
PERSISTENT_ROOTS = ("/data",)


def _persistent_root():
    for root in PERSISTENT_ROOTS:
        if os.path.isdir(root) and os.access(root, os.W_OK):
            return root
    return None


def research_dir():
    """Absolute path of the research results directory."""
    explicit = os.environ.get(ENV_VAR)
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    root = _persistent_root()
    if root:
        return os.path.join(root, APP_DIR_NAME)
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        base = os.path.abspath(os.path.expanduser(xdg))
    else:
        base = os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, APP_DIR_NAME)


def source():
    """Why :func:`research_dir` returned what it did -- for diagnostics."""
    if os.environ.get(ENV_VAR):
        return "env:" + ENV_VAR
    if _persistent_root():
        return "persistent-workspace"
    return "xdg-or-home"


def ensure_dir():
    """Create the research directory if needed and return its path."""
    path = research_dir()
    os.makedirs(path, exist_ok=True)
    return path


def resolve(*parts):
    """Absolute path of ``parts`` joined under the research directory."""
    return os.path.join(research_dir(), *parts)


def describe():
    """Small dict for `schema`, so callers can discover the location."""
    return {
        "research_dir": research_dir(),
        "resolved_from": source(),
        "created_on_demand": True,
        "env_var": ENV_VAR,
        "note": (
            "Results live outside the git working tree and on the persistent "
            "volume; the repository holds code only."
        ),
    }