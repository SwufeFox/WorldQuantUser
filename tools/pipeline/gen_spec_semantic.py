#!/usr/bin/env python3
"""Generate a probe spec from semantic field selection.

Only fields whose id/description reads like a prediction, a behaviour trace or
an estimate are kept: those are the families that produced working alphas here.
Coverage-ranked blind sweeps were tested on 2026-11-19 and topped out at 0.59.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import machine_lib

SEMANTIC = re.compile(
    r"forecast|predict|prob|likelihood|confidence|score|quantile|label|"
    r"estimate|revision|surprise|trend|count|inflow|outflow|hires|employee|"
    r"engagement|session|traffic|signal|alpha|momentum|sentiment|rating",
    re.I,
)
SKIP = re.compile(r"(^|_)(id|code|flag|name|date|ticker|isin|cusip|sic|scale|unit)($|_)", re.I)

BASE = {
    "instrumentType": "EQUITY", "delay": 1, "decay": 0,
    "neutralization": "SUBINDUSTRY", "truncation": 0.08,
    "pasteurization": "ON", "nanHandling": "ON", "unitHandling": "VERIFY",
    "language": "FASTEXPR", "visualization": False, "testPeriod": "P1Y",
    "simulationMode": "QUICK",
}


def fields_for(session, region, universe, dataset, want):
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
    rows.sort(key=lambda r: -(r.get("coverage") or 0))
    out = []
    for r in rows:
        fid = str(r.get("id"))
        text = f"{fid} {r.get('description') or ''}"
        if SKIP.search(fid) or not SEMANTIC.search(text):
            continue
        out.append(r)
        if len(out) >= want:
            break
    return region, dataset, out


def expr_for(row, template):
    fid = str(row["id"])
    f = f"vec_avg({fid})" if row.get("type") == "VECTOR" else fid
    if template == "L":
        return f"group_rank(ts_backfill({f},20), subindustry) - 0.5"
    if template == "L-":
        return f"-(group_rank(ts_backfill({f},20), subindustry) - 0.5)"
    return (f"group_rank(ts_mean(ts_backfill({f},20),20) / "
            f"ts_mean(ts_backfill({f},20),120), subindustry) - 0.5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-dataset", type=int, default=2)
    ap.add_argument("--templates", default="L,T")
    args = ap.parse_args()

    cfg = json.loads(Path(args.spec).read_text())
    session = machine_lib.load_session()
    jobs = [(c["region"], c["universe"], d) for c in cfg for d in c["datasets"]]
    got = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for region, dataset, rows in pool.map(
            lambda j: fields_for(session, j[0], j[1], j[2], args.per_dataset), jobs
        ):
            got.append((region, dataset, rows))

    universes = {c["region"]: c for c in cfg}
    arms, seen = [], set()
    for region, dataset, rows in got:
        if not rows:
            print(f"  ! {region}/{dataset}: nothing semantic", file=sys.stderr)
            continue
        for row in rows:
            for tpl in args.templates.split(","):
                label = f"{region}_{dataset}_{str(row['id'])[:20]}_{tpl}"
                if label in seen:
                    continue
                seen.add(label)
                conf = universes[region]
                settings = dict(BASE, region=region, universe=conf["universe"])
                arms.append({"label": label, "expression": expr_for(row, tpl),
                             "settings": settings})

    spec = {
        "name": Path(args.out).stem,
        "idea": "semantic template sweep across high-multiplier regions",
        "mechanism": "prediction / behaviour-trace fields expressed as level or own-history trend",
        "note": f"{len(arms)} arms, semantic field selection only",
        "settings": dict(BASE, region=cfg[0]["region"], universe=cfg[0]["universe"]),
        "arms": arms,
    }
    Path(args.out).write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    print(f"wrote {len(arms)} arms -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())