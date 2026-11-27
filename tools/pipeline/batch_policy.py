"""Compatibility alias for the top-level batch_policy module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import batch_policy as _impl  # noqa: E402

sys.modules[__name__] = _impl
