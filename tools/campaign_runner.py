#!/usr/bin/env python3
"""Compatibility CLI wrapper and module alias for research_app.campaign_runner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_app import campaign_runner as _impl  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_impl.main())

sys.modules[__name__] = _impl
