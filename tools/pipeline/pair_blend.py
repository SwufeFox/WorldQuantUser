#!/usr/bin/env python3
"""Suggest cross-dataset blends from a set of validated alphas.

The rule measured on 2026-11-19: two channels blend into something stronger
than either leg only when the legs are close to independent. The same pair
traded on divergence instead of a blend flipped from +1.81 to -0.86, so the
leg correlation is the number that decides which logic to use.

This takes alpha ids that already exist on the platform, pulls their daily PnL,
computes pairwise correlations locally, and writes blend arms for the pairs
whose correlation is below a threshold. Nothing is simulated here.

Usage:
    python3 tools/pair_blend.py 78YN3Zw8 QP0731Wr GrmbLnzx --out arms-pairs.json
    python3 tools/pair_blend.py --ids-file /tmp/stable.txt --max-corr 0.35
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import machine_lib as ml

API = "https://api.worldquantbrain.com"
SETTINGS = {
    "instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000", "delay": 1,
    "decay": 0, "neutralization": "SUBINDUSTRY", "truncation": 0.08,
    "pasteurization": "ON", "nanHandling": "ON", "unitHandling": "VERIFY",
    "language": "FASTEXPR", "visualization": False, "testPeriod": "P1Y",
}


def daily_returns(session, alpha_id):
    r = session.get(f"{API}/alphas/{alpha_id}/recordsets/pnl", timeout=120)
    if r.status_code != 200:
        return None
    try:
        payload = r.json()
    except Exception:
        return None
    records = payload.get("records") or []
    out = {}
    prev = None
    for rec in records:
        try:
            day, cum = str(rec[0]), float(rec[1])
        except (TypeError, ValueError, IndexError):
            continue
        if prev is not None:
            out[day] = cum - prev
        prev = cum
    return out or None


def corr(a: dict, b: dict) -> float | None:
    days = sorted(set(a) & set(b))
    if len(days) < 120:
        return None
    x = [a[d] for d in days]
    y = [b[d] for d in days]
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((i - mx) * (j - my) for i, j in zip(x, y))
    dx = math.sqrt(sum((i - mx) ** 2 for i in x))
    dy = math.sqrt(sum((j - my) ** 2 for j in y))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("alpha_ids", nargs="*")
    ap.add_argument("--ids-file")
    ap.add_argument("--out", default="arms-pairs.json")
    ap.add_argument("--max-corr", type=float, default=0.35)
    args = ap.parse_args()

    ids = list(args.alpha_ids)
    if args.ids_file:
        ids += [x for x in Path(args.ids_file).read_text().split() if x]
    ids = list(dict.fromkeys(i for i in ids if i))
    if len(ids) < 2:
        print("need at least two alpha ids", file=sys.stderr)
        return 2

    session = ml.load_session()
    series, expressions = {}, {}
    for aid in ids:
        s = daily_returns(session, aid)
        if not s:
            print(f"  {aid}: no pnl", file=sys.stderr)
            continue
        series[aid] = s
        try:
            d = session.get(f"{API}/alphas/{aid}", timeout=60).json()
            expressions[aid] = (d.get("regular") or {}).get("code") or ""
        except Exception:
            expressions[aid] = ""

    rows = []
    for a, b in itertools.combinations(series, 2):
        c = corr(series[a], series[b])
        if c is not None:
            rows.append((c, a, b))
    rows.sort()

    print(f"{'corr':>6s}  pair")
    for c, a, b in rows:
        mark = " <- blend" if c <= args.max_corr else ""
        print(f"{c:6.3f}  {a} x {b}{mark}")

    arms = []
    for c, a, b in rows:
        if c > args.max_corr:
            continue
        ea, eb = expressions.get(a), expressions.get(b)
        if not ea or not eb:
            continue
        if "sector" not in ea or "sector" not in eb:
            print(f"  skip {a} x {b}: legs use different groupings", file=sys.stderr)
            continue
        arms.append({"label": f"pb_{a}_{b}"[:88],
                     "expr": f"0.5 * ({ea}) + 0.5 * ({eb})",
                     "settings": dict(SETTINGS), "legs": [a, b], "leg_corr": round(c, 3)})

    out = Path(args.out)
    out.write_text(json.dumps(arms, ensure_ascii=False, indent=2))
    print(f"\nwrote {len(arms)} blend arms -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())