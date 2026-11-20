#!/usr/bin/env python3
"""Screen native yearly-stats or cumulative PnL with the same reporting pipeline."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cache_store
import machine_lib as ml


def _error(alpha_id, reason):
    return {"alpha_id": alpha_id, "error": reason}


def _records(session, alpha_id, name, wait):
    """Read a named recordset without a second request for catalog discovery."""
    try:
        _, paths = ml._recordset_paths(alpha_id, [name])
        receipt = ml._read_recordset(session, paths[name], time.monotonic() + wait)
    except (ml.BrainError, ValueError) as exc:
        return [], str(exc)
    if receipt["state"] == "pending":
        return [], "recordset pending"
    if receipt["state"] != "available":
        status = receipt.get("http_status")
        return [], f"HTTP {status}" if status not in (None, 200) else receipt["reason"]
    payload = receipt["payload"]
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return [], f"invalid {name} records"
    count = payload.get("count")
    if payload.get("next") or (type(count) is int and count != len(records)):
        return [], f"incomplete {name} recordset"
    return records, None


def _summary(alpha_id, yearly, *, cumulative=False):
    years, values = zip(*yearly)
    first = statistics.mean(values[:2])
    last = statistics.mean(values[-3:])
    dead = longest = 0
    for value in values:
        dead = dead + 1 if value < 0.3 else 0
        longest = max(longest, dead)
    flags = []
    if last > 0 and first > 2 * last:
        flags.append("front_loaded")
    back_loaded = abs(first) < 0.3 and last > 0.8 if cumulative else first <= 0 < last
    if back_loaded:
        flags.append("back_loaded")
    if longest >= 3:
        flags.append("dead_span")
    if last < 0.6:
        flags.append("thin_tail")
    minimum = min(values)
    if cumulative and minimum < 0:
        flags.append("negative_year")
    precision = 2 if cumulative else 3
    report = {"alpha_id": alpha_id, "years": list(years), "sharpe_by_year": list(values),
              "first_two_mean": round(first, precision), "last_three_mean": round(last, precision),
              "min_year": round(minimum, precision), "longest_dead_span": longest, "flags": flags}
    if not cumulative:
        report["max_year"] = round(max(values), precision)
    return report


def fetch(session, alpha_id):
    """Summarize the platform's yearly Sharpe values in their reported order."""
    records, error = _records(session, alpha_id, "yearly-stats", 90)
    if error or not records:
        return _error(alpha_id, error or "empty yearly-stats")
    yearly = []
    for row in records:
        try:
            year, value = str(row[0]), float(row[6])
        except (IndexError, KeyError, TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            yearly.append((year, value))
    return _summary(alpha_id, yearly) if yearly else _error(alpha_id, "unparsable yearly-stats")


def yearly_sharpe(session, alpha_id):
    """Estimate annual Sharpe using daily differences of cumulative PnL."""
    records, error = _records(session, alpha_id, "pnl", 120)
    if error or len(records) < 60:
        return _error(alpha_id, error or f"only {len(records)} pnl rows")
    by_year = defaultdict(list)
    previous = None
    for row in records:
        try:
            year, cumulative = str(row[0])[:4], float(row[1])
        except (IndexError, KeyError, TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(cumulative):
            continue
        if previous is not None:
            delta = cumulative - previous
            if not math.isfinite(delta):
                return _error(alpha_id, "non-finite daily pnl")
            by_year[year].append(delta)
        previous = cumulative
    if not by_year:
        return _error(alpha_id, "no deltas")
    yearly = []
    for year, values in sorted(by_year.items()):
        if len(values) >= 20:
            deviation = statistics.pstdev(values)
            sharpe = statistics.mean(values) / deviation * math.sqrt(252) if deviation else 0.0
            yearly.append((year, round(sharpe, 2)))
    if len(yearly) < 3:
        return _error(alpha_id, "too few years")
    return _summary(alpha_id, yearly, cumulative=True)


def _read_batch(ids, reader, workers):
    with ml.load_session() as session:
        if workers == 1:
            return [reader(session, alpha_id) for alpha_id in ids]

        def read(alpha_id):
            with ml.isolated_session(session) as isolated:
                return reader(isolated, alpha_id)

        with ThreadPoolExecutor(max_workers=min(workers, len(ids))) as pool:
            return list(pool.map(read, ids))


def main(argv=None, *, cumulative=False):
    parser = argparse.ArgumentParser(description=yearly_sharpe.__doc__ if cumulative else fetch.__doc__)
    parser.add_argument("alpha_ids", nargs="*")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    ids = list(dict.fromkeys(args.alpha_ids + (args.ids_file.read_text().split() if args.ids_file else [])))
    if not ids:
        print("no ids" if cumulative else "no alpha ids", file=sys.stderr)
        return 2

    rows = _read_batch(ids, yearly_sharpe if cumulative else fetch, args.workers)
    for row in rows:
        if row.get("error"):
            print(f"{row['alpha_id']:10s} {row['error']}")
            continue
        flags = ",".join(row["flags"]) or "ok"
        print(f"{row['alpha_id']:10s} first2={row['first_two_mean']:+.2f} "
              f"last3={row['last_three_mean']:+.2f} min={row['min_year']:+.2f} [{flags}]")
        print("           " + " ".join(f"{value:+.1f}" for value in row["sharpe_by_year"]))

    if args.out:
        cache_store.write_text_atomic(args.out, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
