#!/usr/bin/env python3
"""Scan existing alphas for Power Pool eligibility.

POWER_POOL_ELIGIBLE is assigned when the alpha is simulated, so the label is
the only reliable pre-submission gate: an alpha without it is judged against
Regular submission tests and rejected. This reads classifications for a list of
alpha ids and writes one row per alpha. No simulations are consumed.

Usage:
    python3 tools/pp_eligibility.py --ids-file /tmp/ids.txt --out ledgers/pp-elig.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import machine_lib as ml

API = "https://api.worldquantbrain.com"


def read_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("alpha_id"):
            done.add(row["alpha_id"])
    return done


def probe(session, alpha_id):
    try:
        d = session.get(f"{API}/alphas/{alpha_id}", timeout=60).json()
    except Exception as exc:
        return {"alpha_id": alpha_id, "error": str(exc)}
    is_ = d.get("is") or {}
    checks = {c.get("name"): c for c in (is_.get("checks") or [])}
    classes = ml.alpha_classification_ids(d)
    return {
        "alpha_id": alpha_id,
        "status": d.get("status"),
        "sharpe": is_.get("sharpe"),
        "fitness": is_.get("fitness"),
        "turnover": is_.get("turnover"),
        "universe": (d.get("settings") or {}).get("universe"),
        "region": (d.get("settings") or {}).get("region"),
        "pp_eligible": None if classes is None else any("POWER_POOL_ELIGIBLE" in c for c in classes),
        "single_dataset": None if classes is None else any("SINGLE_DATA_SET" in c for c in classes),
        "classifications": classes,
        "fails": sorted(n for n, c in checks.items() if c.get("result") == "FAIL"),
        "two_year": (checks.get("LOW_2Y_SHARPE") or {}).get("value"),
        "expression": (d.get("regular") or {}).get("code"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--out", default="ledgers/pp-elig.jsonl")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    ids = [x for x in Path(args.ids_file).read_text().split() if x]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(out)
    todo = [i for i in ids if i not in done]
    print(f"ids={len(ids)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return 0

    session = ml.load_session()
    eligible = []
    with out.open("a") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, row in enumerate(pool.map(lambda i: probe(session, i), todo), 1):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if row.get("pp_eligible"):
                eligible.append(row)
                print(f"[{n}] PP-ELIGIBLE {row['alpha_id']} S={row.get('sharpe')} "
                      f"T={row.get('turnover')} univ={row.get('universe')}", flush=True)
            elif n % 20 == 0:
                print(f"[{n}/{len(todo)}] ...", flush=True)
    print(f"eligible found: {len(eligible)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
