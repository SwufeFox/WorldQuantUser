#!/usr/bin/env python3
"""Compatibility alias and non-CLI guard for research_app.guarded_sim."""

import sys
from pathlib import Path

if __name__ == "__main__":
    print("guarded_sim is a library, not a command:\n"
          "  from guarded_sim import simulate, check\n"
          "  CLI entry point: research_cli.py probe SPEC.json",
          file=sys.stderr)
    raise SystemExit(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_app import guarded_sim as _impl  # noqa: E402

sys.modules[__name__] = _impl
