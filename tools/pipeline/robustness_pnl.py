#!/usr/bin/env python3
"""Compatibility entry point for the cumulative-PnL robustness screen."""
from __future__ import annotations

if __package__:
    from .robustness import main as run_robustness, yearly_sharpe
else:
    from robustness import main as run_robustness, yearly_sharpe


def main(argv=None):
    return run_robustness(argv, cumulative=True)


if __name__ == "__main__":
    raise SystemExit(main())
