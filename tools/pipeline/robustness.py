#!/usr/bin/env python3
"""Year-by-year robustness screen for BRAIN alphas.

Aggregate Sharpe hides front-loading: an alpha that made 3.5+ in its first two
years and near-zero since will still show Sharpe ~1.1 overall. This reads the
yearly-stats recordset and reports the shape, then flags:

  front_loaded  first-two-year Sharpe is more than 2x the last-three-year Sharpe
  dead_span     three or more consecutive years below 0.3
  thin_tail     last-three-year mean Sharpe below 0.6

Usage:
    python3 tools/robustness.py blabL9GM 0mPwXpo6 ...
    python3 tools/robustness.py --ids-file /tmp/ids.txt --out ledgers/robust.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import machine_lib as ml

API = "https://api.worldquantbrain.com"


def fetch(session, alpha_id):
    r = session.get(f"{API}/alphas/{alpha_id}/recordsets/yearly-stats", timeout=90)
    if r.status_code != 200:
        return {"alpha_id": alpha_id, "error": f"HTTP {r.status_code}"}
    try:
        payload = r.json()
    except Exception:
        return {"alpha_id": alpha_id, "error": "no yearly-stats payload"}
    records = payload.get("records") or []
    if not records:
        return {"alpha_id": alpha_id, "error": "empty yearly-stats"}
    # schema: year, pnl, bookSize, longCount, shortCount, turnover, sharpe, ...
    years = []
    for rec in records:
        try:
            years.append((str(rec[0]), float(rec[6])))
        except (TypeError, ValueError, IndexError):
            continue
    if not years:
        return {"alpha_id": alpha_id, "error": "unparsable yearly-stats"}

    values = [v for _, v in years]
    first_two = statistics.mean(values[:2]) if len(values) >= 2 else values[0]
    last_three = statistics.mean(values[-3:]) if len(values) >= 3 else statistics.mean(values)
    dead = 0
    longest = 0
    for v in values:
        if v < 0.3:
            dead += 1
            longest = max(longest, dead)
        else:
            dead = 0
    flags = []
    if last_three > 0 and first_two > 2 * last_three:
        flags.append("front_loaded")
    if first_two <= 0 < last_three:
        flags.append("back_loaded")
    if longest >= 3:
        flags.append("dead_span")
    if last_three < 0.6:
        flags.append("thin_tail")
    return {
        "alpha_id": alpha_id,
        "years": [y for y, _ in years],
        "sharpe_by_year": values,
        "first_two_mean": round(first_two, 3),
        "last_three_mean": round(last_three, 3),
        "min_year": round(min(values), 3),
        "max_year": round(max(values), 3),
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
        print("no alpha ids", file=sys.stderr)
        return 2

    session = ml.load_session()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(lambda i: fetch(session, i), ids):
            rows.append(row)

    for row in rows:
        if row.get("error"):
            print(f"{row['alpha_id']:10s} {row['error']}")
            continue
        vals = " ".join(f"{v:+.1f}" for v in row["sharpe_by_year"])
        flag = ",".join(row["flags"]) or "ok"
        print(f"{row['alpha_id']:10s} first2={row['first_two_mean']:+.2f} "
              f"last3={row['last_three_mean']:+.2f} min={row['min_year']:+.2f}  "
              f"[{flag}]")
        print(f"           {vals}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        print(f"wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())