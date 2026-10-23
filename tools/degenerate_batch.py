#!/usr/bin/env python3
"""Flag a batch whose different expressions produced the same result.

Measured 2026-10-23: six alphas built on six different S&P Core Earnings adjustment fields all
returned Sharpe 2.09-2.11, turnover 0.81, returns 0.138 - identical to two decimals. The
fields are in the catalogue with coverage 0.5, but carry no values in this universe, so the
disclosure leg was null and each expression had quietly collapsed to the fast leg with a
flipped sign.

Two diagnostic runs confirmed it: the field alone returned Sharpe 0.0 and turnover 0.0, and
the fast leg alone returned -2.09 / 0.8111.

The tell is that identical numbers from different inputs are not a coincidence; they mean the
differing part was discarded. This checks for it after the fact, because the platform has no
way to ask a field whether it is populated.

Usage:
    python tools/degenerate_batch.py results.json
    python tools/degenerate_batch.py results.json --min-cluster 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)


def metric_key(record, places=2):
    """The rounded metrics that identify a result, ignoring which expression produced it."""
    metrics = record.get("metrics") or {}
    sharpe = metrics.get("sharpe")
    turnover = metrics.get("turnover")
    if sharpe is None or turnover is None:
        return None
    return (round(float(sharpe), places), round(float(turnover), places))


def clusters(records, places=2, minimum=2):
    """Group records whose metrics match, keeping only clusters of ``minimum`` or more."""
    buckets = defaultdict(list)
    for record in records:
        key = metric_key(record, places)
        if key is None:
            continue
        buckets[key].append(record)
    found = []
    for key, group in buckets.items():
        expressions = {record.get("expression") for record in group}
        # Same expression twice is a duplicate submission, not a degenerate batch.
        if len(group) >= minimum and len(expressions) >= minimum:
            found.append({"sharpe": key[0], "turnover": key[1], "records": group})
    found.sort(key=lambda item: -len(item["records"]))
    return found


def report(records, places=2, minimum=2):
    found = clusters(records, places, minimum)
    dead = [record for record in records
            if (record.get("metrics") or {}).get("turnover") in (0, 0.0)
            and (record.get("metrics") or {}).get("sharpe") in (0, 0.0)]
    return {
        "records": len(records),
        "empty_results": [record.get("candidate_id") or record.get("alpha_id") for record in dead],
        "identical_clusters": [
            {
                "sharpe": item["sharpe"],
                "turnover": item["turnover"],
                "candidates": [record.get("candidate_id") or record.get("alpha_id")
                               for record in item["records"]],
                "expressions": [record.get("expression") for record in item["records"]],
            }
            for item in found
        ],
        "verdict": ("degenerate: different expressions returned the same result, so the part "
                    "that differed was discarded - suspect an unpopulated field"
                    if found else "ok"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="A simulate results JSON.")
    parser.add_argument("--min-cluster", type=int, default=2)
    parser.add_argument("--places", type=int, default=2)
    args = parser.parse_args(argv)

    with open(args.results, encoding="utf-8") as handle:
        document = json.load(handle)
    body = document.get("data") or document
    records = body.get("results") or body.get("rows") or []
    print(json.dumps(report(records, args.places, args.min_cluster),
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())