#!/usr/bin/env python3
"""Find ledger alphas whose daily PnL is indistinguishable, and mark the extras.

Why: submitting the same expression twice returns a fresh alpha id, and an
algebraically zero term added to force a new simulation does the same, so the
ledger ends up holding several ids for one strategy. Clustering the 147 qualifying
alphas produced 494 pairs at |r| >= 0.90 and at least one group of four ids whose
PnL agreed to four decimals, which means every count derived from the ledger was
inflated — including "how many separate bets".

Detection is deliberately strict: a duplicate must match on essentially every day,
not merely correlate well. Two genuinely different alphas with the same theme can
correlate 0.98 and still be two decisions; identical PnL is a bookkeeping fact.

The canonical member is chosen by status first (an ACTIVE alpha is the one
submitted), then fitness, then the earlier record: whatever the account already
treats as the real one should stay the one that gets counted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import correlation  # noqa: E402
import machine_lib as ml  # noqa: E402
import research_paths  # noqa: E402


def status_rank(entry):
    """Submitted beats unsubmitted: the account already counts it."""
    status = (entry.get("status") or "").upper()
    return 0 if status == "ACTIVE" else (1 if status == "SUBMITTED" else 2)


def canonical_order(entry):
    metrics = entry.get("metrics") or {}
    fitness = metrics.get("fitness")
    return (
        status_rank(entry),
        -(fitness if isinstance(fitness, (int, float)) else -99),
        entry.get("recorded_at") or "",
    )


def identical(left, right, min_overlap=200, equal_fraction=0.99):
    """True when two series agree on almost every shared day.

    Returns ``(is_duplicate, equal_days, shared_days)``. The equal-day test is the
    point: correlation alone would call two same-theme alphas duplicates.
    """
    shared = sorted(set(left) & set(right))
    if len(shared) < min_overlap:
        return False, 0, len(shared)
    equal = sum(1 for day in shared
                if abs(left[day] - right[day]) <= 1e-6 * max(1.0, abs(left[day])))
    return equal / len(shared) >= equal_fraction, equal, len(shared)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-fitness", type=float, default=1.25,
                        help="Only inspect entries at or above this fitness.")
    parser.add_argument("--equal-fraction", type=float, default=0.99)
    parser.add_argument("--min-overlap", type=int, default=200)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--registry", default=None)
    args = parser.parse_args(argv)

    path = args.registry or alpha_registry.default_registry_path()
    entries = alpha_registry.load_entries(path)
    by_id = {}
    for entry in entries:
        alpha_id = entry.get("alpha_id")
        fitness = (entry.get("metrics") or {}).get("fitness")
        if not alpha_id or not isinstance(fitness, (int, float)):
            continue
        if fitness < args.min_fitness:
            continue
        current = by_id.get(alpha_id)
        if current is None or canonical_order(entry) < canonical_order(current):
            by_id[alpha_id] = entry

    ids = sorted(by_id)
    print("inspecting %d alphas at fitness >= %s" % (len(ids), args.min_fitness))

    cache = os.path.join(research_paths.ensure_dir(), "pnl")
    session = ml.load_session() or ml.authenticate()
    pnls = {}
    for alpha_id in ids:
        series = ml.fetch_daily_pnl(session, alpha_id, cache_dir=cache)
        if series:
            pnls[alpha_id] = series
    print("usable PnL: %d" % len(pnls))

    groups = []
    claimed = set()
    for i, left in enumerate(sorted(pnls)):
        if left in claimed:
            continue
        members = [left]
        for right in sorted(pnls):
            if right == left or right in claimed:
                continue
            same, equal, shared = identical(pnls[left], pnls[right],
                                            args.min_overlap, args.equal_fraction)
            if same:
                members.append(right)
        if len(members) > 1:
            ordered = sorted(members, key=lambda name: canonical_order(by_id[name]))
            claimed.update(ordered[1:])
            groups.append((ordered[0], ordered[1:], equal, shared))

    print()
    if not groups:
        print("no identical-PnL groups found")
    else:
        print("identical-PnL groups: %d (each line: canonical <- duplicates)"
              % len(groups))
        for canonical, duplicates, equal, shared in groups:
            head = by_id[canonical]
            print("  %-10s (%s, fitness %s) <- %s"
                  % (canonical, head.get("status"),
                     (head.get("metrics") or {}).get("fitness"),
                     ", ".join(duplicates)))
        print()
        print("distinct strategies among the qualifiers: %d (was %d)"
              % (len(pnls) - sum(len(d) for _, d, _, _ in groups), len(pnls)))

    if not groups or not args.write:
        if groups:
            print()
            print("dry run; pass --write to mark the duplicates")
        return 0

    written = 0
    for canonical, duplicates, _equal, _shared in groups:
        for alpha_id in duplicates:
            updated = alpha_registry.mark_duplicate(by_id[alpha_id], canonical)
            _, did_write = alpha_registry.record_entry(updated, path)
            written += 1 if did_write else 0
    print()
    print("marked %d duplicates" % written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())