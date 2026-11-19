#!/usr/bin/env python3
"""Build a backlog of net-sentiment arms from every positive/negative field pair.

The template that worked on earnings calls is:
    net = positive_score - negative_score
    group_rank(ts_mean(net, 20), sector) - 0.5
and, as a second shape, the own-history acceleration of that net series.

This scans the local metadata cache for exact `*positive*` / `*negative*` twins,
pairs them by name, and emits a prioritised backlog. It is a backlog: nothing is
simulated here. Feed the output to tools/pipeline_pc.py.

Priority: coverage first, then datasets not yet used by the submitted book.

Usage:
    python3 tools/backlog_net_sentiment.py --out backlogs/net-sentiment.jsonl
    python3 tools/backlog_net_sentiment.py --datasets news_transformer_scores,ai_news_scores
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

RESEARCH = Path("/data/worldquant-research")

SETTINGS = {
    "instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000", "delay": 1,
    "decay": 0, "neutralization": "SUBINDUSTRY", "truncation": 0.08,
    "pasteurization": "ON", "nanHandling": "ON", "unitHandling": "VERIFY",
    "language": "FASTEXPR", "visualization": False, "testPeriod": "P1Y",
}

# Datasets that already produced a submitted Power Pool alpha, or whose family
# was measured dead. Kept explicit so the reason is visible.
DEPRIORITISED = {
    "other580": "ETF holdings change, measured dead 2026-11-18",
    "earningscall_sentiment": "already produced 78YN3Zw8 / E5QpNkEr, one mechanism",
}


def load_fields():
    rows = {}
    for path in glob.glob(str(RESEARCH / "cache/metadata/datafields/*.json")):
        try:
            payload = json.loads(Path(path).read_text())["payload"]
        except Exception:
            continue
        for r in payload:
            if isinstance(r, dict) and r.get("id"):
                rows[r["id"]] = r
    return rows


def _ref(field_id: str, kind: str) -> str:
    """VECTOR fields need an aggregation; MATRIX fields must not be wrapped.

    Applying vec_avg unconditionally made every MATRIX arm fail at simulation
    (mmp_nlp_sentiment, 2026-11-19).
    """
    return f"vec_avg({field_id})" if kind == "VECTOR" else field_id


def expr_net(pos, neg, kind="MATRIX"):
    p = f"ts_backfill({_ref(pos, kind)},60)"
    n = f"ts_backfill({_ref(neg, kind)},60)"
    return f"group_rank(ts_mean(({p} - {n}),20), sector) - 0.5"


def expr_trend(pos, neg, kind="MATRIX"):
    p = f"ts_backfill({_ref(pos, kind)},60)"
    n = f"ts_backfill({_ref(neg, kind)},60)"
    net = f"({p} - {n})"
    return f"group_rank(ts_mean({net},20) / ts_mean({net},120), sector) - 0.5"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="backlogs/net-sentiment.jsonl")
    ap.add_argument("--datasets", default="", help="comma separated subset")
    ap.add_argument("--per-dataset", type=int, default=4)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    args = ap.parse_args()

    fields = load_fields()
    wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}

    # Pair exact twins: id with "positive" swapped for "negative".
    pairs = []
    for fid, r in fields.items():
        if r.get("region") != "USA" or "positive" not in fid:
            continue
        twin = fid.replace("positive", "negative")
        if twin not in fields:
            continue
        ds_obj = r.get("dataset")
        ds = ds_obj.get("id") if isinstance(ds_obj, dict) else None
        if not ds or (wanted and ds not in wanted):
            continue
        cov = min(r.get("coverage") or 0, fields[twin].get("coverage") or 0)
        if cov < args.min_coverage:
            continue
        pairs.append({"dataset": ds, "pos": fid, "neg": twin, "coverage": round(cov, 3),
                      "type": r.get("type")})

    by_ds = {}
    for p in pairs:
        by_ds.setdefault(p["dataset"], []).append(p)
    for ds in by_ds:
        by_ds[ds].sort(key=lambda p: -p["coverage"])

    arms = []
    for ds, items in sorted(by_ds.items(), key=lambda kv: -len(kv[1])):
        note = DEPRIORITISED.get(ds)
        for p in items[: args.per_dataset]:
            stem = re.sub(r"[^a-z0-9]+", "_", p["pos"].replace("positive", ""))[:18].strip("_")
            for shape, fn in (("net", expr_net), ("trend", expr_trend)):
                arms.append({
                    "label": f"ns_{ds[:14]}_{stem}_{shape}"[:88],
                    "expr": fn(p["pos"], p["neg"], p["type"]),
                    "settings": dict(SETTINGS),
                    "dataset": ds,
                    "coverage": p["coverage"],
                    "priority": 2 if note else 1,
                    "note": note or "",
                })

    arms.sort(key=lambda a: (a["priority"], -a["coverage"]))
    out = RESEARCH / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in arms) + "\n")

    print(f"pairs={len(pairs)} datasets={len(by_ds)} arms={len(arms)} -> {out}")
    seen = set()
    for a in arms:
        if a["dataset"] in seen:
            continue
        seen.add(a["dataset"])
        print(f"  {a['dataset'][:28]:28s} cov={a['coverage']:.2f} prio={a['priority']} "
              f"{('(' + a['note'][:30] + ')') if a['note'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
