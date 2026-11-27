"""Compatibility alias for the top-level spc_experiment module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spc_experiment as _impl  # noqa: E402

sys.modules[__name__] = _impl
