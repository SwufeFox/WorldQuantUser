#!/usr/bin/env python3
"""Compatibility entry point for the shared, resumable Quick batch runner."""
from __future__ import annotations

import sys

if __package__:
    from .batch_quick import main as run_batch
else:
    from batch_quick import main as run_batch


def main():
    return run_batch(default_out="ledgers/async.jsonl", default_concurrency=6)


if __name__ == "__main__":
    sys.exit(main())
