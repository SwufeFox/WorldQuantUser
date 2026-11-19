#!/usr/bin/env python3
"""Generate template arms for a region/dataset list.

Templates are idea-shaped, not parameter sweeps:
  L  level      group_rank(f, subindustry) - 0.5
  T  trend      group_rank(mean20/mean120, subindustry) - 0.5

Field selection: highest-coverage MATRIX/VECTOR fields, skipping obvious
identifier and price-like columns, capped per dataset.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import machine_lib

BASE = {
    "instrumentType": "EQUITY",
    "delay": 1,
    "decay": 0,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
    "pasteurization": "ON",
    "nanHandling": "ON",
    "unitHandling": "VERIFY",
    "language": "FASTEXPR",
    "visualization": False,
    "testPeriod": "P1Y",
    "simulationMode": "QUICK",
}

SKIP = re.compile(r"(^|_)(id|code|flag|name|date|ticker|isin|cusip|sic)($|_)", re.I)


def fetch_fields(session, region: str, dataset: str, universe: str, per_dataset: int):
    url = ("https://api.worldquantbrain.com/data-fields"
           f"?instrumentType=EQUITY&region={region}&delay=1&universe={universe}"
           f"&dataset.id={dataset}&limit=50")
    try:
        resp = session.get(url, timeout=60)
        if resp.status_code != 200:
            return region, dataset, []
        rows = resp.json().get("results") or []
    except Exception:
        return region, dataset, []
    rows = [r for r in rows if r.get("type") in ("MATRIX", "VECTOR")]
    rows.sort(key=lambda r: (-(r.get("coverage") or 0), r.get("id")))
    picked = []
    for r in rows:
        fid = str(r.get("id"))
        if SKIP.search(fid):
            continue
        picked.append(r)
        if len(picked) >= per_dataset:
            break
    return region, dataset, picked


def expr_for(row, template: str) -> str:
    fid = str(row["id"])
    field = f"vec_avg({fid})" if row.get("type") == "VECTOR" else fid
    if template == "L":
        return f"group_rank(ts_backfill({field},20), subindustry) - 0.5"
    return (f"group_rank(ts_mean(ts_backfill({field},20),20) / "
            f"ts_mean(ts_backfill({field},20),120), subindustry) - 0.5")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--out", default="arms-batch.json")
    ap.add_argument("--per-dataset", type=int, default=2)
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    session = machine_lib.load_session()

    jobs = [(s["region"], d, s.get("universe", "TOP3000")) for s in spec for d in s["datasets"]]
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for region, dataset, rows in pool.map(
            lambda j: fetch_fields(session, j[0], j[1], j[2], args.per_dataset), jobs
        ):
            results.append((region, dataset, rows))

    universe = {s["region"]: s.get("universe", "TOP3000") for s in spec}
    arms, seen = [], set()
    for region, dataset, rows in results:
        if not rows:
            print(f"  ! {region}/{dataset}: no usable fields", file=sys.stderr)
            continue
        for row in rows:
            for template in ("L", "T"):
                label = f"{region}_{dataset}_{str(row['id'])[:22]}_{template}"
                if label in seen:
                    continue
                seen.add(label)
                settings = dict(BASE, region=region, universe=universe[region])
                arms.append({"label": label, "expr": expr_for(row, template),
                             "settings": settings})
    Path(args.out).write_text(json.dumps(arms, ensure_ascii=False, indent=2))
    print(f"wrote {len(arms)} arms -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())