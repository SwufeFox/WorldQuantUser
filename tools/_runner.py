"""Compatibility alias for research_app.queue_runner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_app import queue_runner as _impl  # noqa: E402

sys.modules[__name__] = _impl
