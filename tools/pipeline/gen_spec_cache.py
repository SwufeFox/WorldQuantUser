#!/usr/bin/env python3
"""Build a probe spec from the local metadata cache (no platform calls).

Selects semantic fields per dataset: predictions, estimates, revisions,
behaviour counts. Coverage-ranked blind selection was measured on 2026-11-19
and topped out at 0.59 Sharpe, so the semantic filter is the point.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

SEMANTIC = re.compile(
    r"forecast|predict|probab|likelihood|confidence|score|quantile|estimate|"
    r"revision|surprise|trend|inflow|outflow|hires|employee|engagement|session|"
    r"traffic|signal|momentum|rating|sentiment|count|ratio",
    re.I,
)
SKIP = re.compile(r"(^|_)(id|code|flag|name|date|ticker|isin|cusip|sic|scale|unit|type)($|_)", re.I)

SETTINGS = {
    "instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000", "delay": 1,
    "decay": 0, "neutralization": "SUBINDUSTRY", "truncation": 0.08,
    "pasteurization": "ON", "nanHandling": "ON", "unitHandling": "VERIFY",
    "language": "FASTEXPR", "visualization": False, "testPeriod": "P1Y",
    "simulationMode": "QUICK",
}


def load_fields(research_dir: Path):
    rows = {}
    for path in (research_dir / "cache" / "metadata" / "datafields").glob("*.json"):
        try:
            payload = json.loads(path.read_text())["payload"]
        except Exception:
            continue
        for r in payload:
            if isinstance(r, dict) and r.get("id"):
                rows[r["id"]] = r
    return rows


def load_dataset_categories(research_dir: Path):
    cats = {}
    for path in (research_dir / "cache" / "metadata" / "datasets").glob("*.json"):
        try:
            payload = json.loads(path.read_text())["payload"]
        except Exception:
            continue
        for r in payload:
            if isinstance(r, dict) and r.get("id") and r.get("region") == "USA":
                cat = r.get("category")
                cats[r["id"]] = cat.get("id") if isinstance(cat, dict) else cat
    return cats


def expr(field_id, kind, template):
    field = f"vec_avg({field_id})" if kind == "VECTOR" else field_id
    if template == "L":
        return f"group_rank(ts_backfill({field},20), subindustry) - 0.5"
    if template == "L-":
        return f"-(group_rank(ts_backfill({field},20), subindustry) - 0.5)"
    return (f"group_rank(ts_mean(ts_backfill({field},20),20) / "
            f"ts_mean(ts_backfill({field},20),120), subindustry) - 0.5")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-dir", default="/data/worldquant-research")
    ap.add_argument("--datasets", required=True, help="comma separated")
    ap.add_argument("--per-dataset", type=int, default=3)
    ap.add_argument("--templates", default="L,T")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="cache-scan")
    args = ap.parse_args()

    research = Path(args.research_dir)
    fields = load_fields(research)
    cats = load_dataset_categories(research)
    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]

    by_dataset = defaultdict(list)
    for fid, r in fields.items():
        if r.get("region") != "USA" or r.get("type") not in ("MATRIX", "VECTOR"):
            continue
        ds = r.get("dataset")
        dsid = ds.get("id") if isinstance(ds, dict) else None
        if not dsid or dsid not in wanted:
            continue
        if SKIP.search(fid) or not SEMANTIC.search(f"{fid} {r.get('description') or ''}"):
            continue
        by_dataset[dsid].append((r.get("coverage") or 0, fid, r.get("type")))

    arms, seen = [], set()
    for dsid in wanted:
        rows = sorted(by_dataset.get(dsid, []), reverse=True)[: args.per_dataset]
        if not rows:
            print(f"  ! {dsid}: no semantic fields in cache")
            continue
        for coverage, fid, kind in rows:
            for tpl in args.templates.split(","):
                label = f"cache_{dsid}_{fid[:22]}_{tpl}"
                if label in seen:
                    continue
                seen.add(label)
                arms.append({"label": label, "expression": expr(fid, kind, tpl),
                             "settings": dict(SETTINGS)})

    spec = {
        "name": args.name,
        "idea": "semantic field scan over untouched datasets",
        "mechanism": "prediction / estimate / behaviour-trace fields as level or own-history trend",
        "note": f"{len(arms)} arms from the local metadata cache",
        "design": "explore",
        "settings": dict(SETTINGS),
        "arms": arms,
    }
    Path(args.out).write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    print(f"wrote {len(arms)} arms -> {args.out}")
    covered = {a['label'].split('_')[1] for a in arms}
    missing = [d for d in wanted if d not in covered]
    if missing:
        print("no coverage:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())