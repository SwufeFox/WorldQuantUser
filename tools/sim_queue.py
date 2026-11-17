#!/usr/bin/env python3
"""Run probe specs from queue/ and keep unfinished batches available for recovery."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools._runner import research_root, run_queue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", "--root", dest="root")
    parser.add_argument("--max-wait", type=int, default=150)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--sleep", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.max_wait <= 0 or not math.isfinite(args.sleep) or args.sleep <= 0:
        parser.error("--max-wait and --sleep must be finite and positive")
    root = research_root(args.root)
    inbox = root / "queue"
    batches = inbox / "batches"
    batches.mkdir(parents=True, exist_ok=True)

    def paths(spec):
        return (batches / f"{spec.stem}-batch.json", root / f"{spec.stem}-result.json")

    def log(event):
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{stamp} {event['kind']} {event.get('spec', '')} {event.get('detail', '')}".rstrip()
        with (inbox / "queue.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    return run_queue(root, inbox, paths, log, max_wait=args.max_wait, interval=args.sleep,
                     once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
