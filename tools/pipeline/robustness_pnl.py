#!/usr/bin/env python3
"""Year-by-year robustness screen computed from daily PnL.

`yearly-stats` only exists for already-submitted alphas, so this derives the same
shape from the cumulative `pnl` recordset instead. That makes the screen usable
before submission, which is the point: aggregate Sharpe hides front-loading.

Flags:
  front_loaded  first-two-year Sharpe > 2x last-three-year Sharpe
  back_loaded   signal effectively starts late (first two years near zero)
  dead_span     three or more consecutive years below 0.3
  thin_tail     last-three-year mean Sharpe below 0.6
  negative_year worst year is negative

Usage:
    python3 tools/robustness_pnl.py ALPHA_ID [ALPHA_ID ...]
    python3 tools/robustness_pnl.py --ids-file /tmp/ids.txt --out ledgers/robust.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import machine_lib as ml

API = "https://api.worldquantbrain.com"


def yearly_sharpe(session, alpha_id):
    r = session.get(f"{API}/alphas/{alpha_id}/recordsets/pnl", timeout=120)
    if r.status_code != 200:
        return {"alpha_id": alpha_id, "error": f"HTTP {r.status_code}"}
    try:
        payload = r.json()
    except Exception:
        return {"alpha_id": alpha_id, "error": "pnl not json"}
    records = payload.get("records") or []
    if len(records) < 60:
        return {"alpha_id": alpha_id, "error": f"only {len(records)} pnl rows"}

    daily = []
    prev = None
    for rec in records:
        try:
            day = str(rec[0])
            cum = float(rec[1])
        except (TypeError, ValueError, IndexError):
            continue
        if prev is not None:
            daily.append((day, cum - prev))
        prev = cum
    if not daily:
        return {"alpha_id": alpha_id, "error": "no deltas"}

    by_year = defaultdict(list)
    for day, d in daily:
        by_year[day[:4]].append(d)

    years, sharpes = [], []
    for year in sorted(by_year):
        values = by_year[year]
        if len(values) < 20:
            continue
        mean = statistics.mean(values)
        stdev = statistics.pstdev(values)
        sharpe = (mean / stdev * math.sqrt(252)) if stdev > 0 else 0.0
        years.append(year)
        sharpes.append(round(sharpe, 2))

    if len(sharpes) < 3:
        return {"alpha_id": alpha_id, "error": "too few years"}

    first_two = statistics.mean(sharpes[:2])
    last_three = statistics.mean(sharpes[-3:])
    dead = longest = 0
    for v in sharpes:
        if v < 0.3:
            dead += 1
            longest = max(longest, dead)
        else:
            dead = 0

    flags = []
    if last_three > 0 and first_two > 2 * last_three:
        flags.append("front_loaded")
    if abs(first_two) < 0.3 and last_three > 0.8:
        flags.append("back_loaded")
    if longest >= 3:
        flags.append("dead_span")
    if last_three < 0.6:
        flags.append("thin_tail")
    if min(sharpes) < 0:
        flags.append("negative_year")

    return {
        "alpha_id": alpha_id,
        "years": years,
        "sharpe_by_year": sharpes,
        "first_two_mean": round(first_two, 2),
        "last_three_mean": round(last_three, 2),
        "min_year": min(sharpes),
        "longest_dead_span": longest,
        "flags": flags,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("alpha_ids", nargs="*")
    ap.add_argument("--ids-file")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    ids = list(args.alpha_ids)
    if args.ids_file:
        ids += [x for x in Path(args.ids_file).read_text().split() if x]
    ids = list(dict.fromkeys(i for i in ids if i))
    if not ids:
        print("no ids", file=sys.stderr)
        return 2

    session = ml.load_session()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(lambda i: yearly_sharpe(session, i), ids):
            rows.append(row)

    for row in rows:
        if row.get("error"):
            print(f"{row['alpha_id']:10s} {row['error']}")
            continue
        vals = " ".join(f"{v:+.1f}" for v in row["sharpe_by_year"])
        flag = ",".join(row["flags"]) or "ok"
        print(f"{row['alpha_id']:10s} first2={row['first_two_mean']:+.2f} "
              f"last3={row['last_three_mean']:+.2f} min={row['min_year']:+.2f} [{flag}]")
        print(f"           {vals}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        print(f"wrote {len(rows)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())