#!/usr/bin/env python3
"""Screen saved ATOM results by the platform's reported sub-universe check."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools._runner import research_root


def _sharpe(row):
    value = row.get("sharpe")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return -math.inf


def load(path):
    """Keep the best measured Sharpe per alpha, with the latest row winning ties."""
    best = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON result") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: result must be a JSON object")
            alpha_id = row.get("alpha_id")
            if alpha_id and _sharpe(row) >= _sharpe(best.get(alpha_id, {})):
                best[alpha_id] = row
    return best


def _sub_status(row):
    check = (row.get("checks") or {}).get("LOW_SUB_UNIVERSE_SHARPE") or {}
    return check.get("result") or "MISSING", check.get("value"), check.get("limit")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", type=Path, help="Workspace containing atom-results.jsonl")
    parser.add_argument("--min-sharpe", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_sharpe) or args.top < 0:
        parser.error("--min-sharpe must be finite and --top must be non-negative")

    path = research_root(args.research_dir) / "atom-results.jsonl"
    try:
        rows = [row for row in load(path).values() if _sharpe(row) >= args.min_sharpe]
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rows.sort(key=_sharpe, reverse=True)

    passing, failing = [], []
    for row in rows:
        status, value, limit = _sub_status(row)
        if status == "PASS":
            passing.append((row, value, limit))
        elif status == "FAIL":
            failing.append((row, value, limit))

    missing = len(rows) - len(passing) - len(failing)
    print(f"Sharpe >= {args.min_sharpe}: {len(rows)} arms "
          f"({len(passing)} pass sub-universe, {len(failing)} fail, {missing} unmeasured)\n")
    print(f"{'alpha':10s} {'Sharpe':>6s} {'turnover':>8s} {'sub':>16s}  label")
    for row, value, limit in passing[: args.top]:
        sub = f"PASS {value}/{limit}" if value is not None else "PASS"
        print(f"{row['alpha_id']:10s} {row['sharpe']:6.2f} {row.get('turnover') or 0:8.3f} "
              f"{sub:>16s}  {row['label'][:36]}")

    if failing:
        print("\nlargest Sharpe that the liquid half rejects:")
        for row, value, limit in failing[:5]:
            print(f"  {row['alpha_id']:10s} S{row['sharpe']:.2f} sub {value}/{limit}  {row['label'][:34]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
