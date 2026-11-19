#!/usr/bin/env python3
"""Generate a large feed of valid arms from datasets not yet used.

Continuous running needs a long tail of arms that are actually runnable: right
field type (MATRIX must not be wrapped in vec_avg), right shape per field
semantics, and no duplicates against what already ran.

Shapes used, in order of measured clean rate:
  ratio_trend   group_rank(mean20/mean120, sector) - 0.5     23%
  level         group_rank(mean20, sector) - 0.5             22%
  delta         group_rank(ts_delta(f,20), sector) - 0.5     48%
Net-sentiment difference pairs are emitted by tools/backlog_net_sentiment.py.

Usage:
    python3 tools/gen_feed.py --limit 120 --out feeds/feed1.jsonl
    python3 tools/gen_feed.py --datasets foo,bar --shape ratio_trend
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
SKIP = re.compile(r"(^|_)(id|code|flag|name|date|ticker|isin|cusip|sic|scale|unit|type|count|num)($|_)", re.I)
SEMANTIC = re.compile(
    r"forecast|predict|probab|likelihood|confidence|score|quantile|estimate|revision|"
    r"surprise|trend|imbalance|utilization|efficien|quality|growth|margin|yield|beta|"
    r"volatil|correlat|momentum|sentiment|signal|rank", re.I)


def load():
    fields = {}
    for path in glob.glob(str(RESEARCH / "cache/metadata/datafields/*.json")):
        try:
            payload = json.loads(Path(path).read_text())["payload"]
        except Exception:
            continue
        for r in payload:
            if isinstance(r, dict) and r.get("id"):
                fields[r["id"]] = r
    return fields


def ref(fid, kind):
    return f"vec_avg({fid})" if kind == "VECTOR" else fid


def build(fid, kind, shape):
    f = f"ts_backfill({ref(fid, kind)},60)"
    if shape == "ratio_trend":
        return f"group_rank(ts_mean({f},20) / ts_mean({f},120), sector) - 0.5"
    if shape == "delta":
        return f"group_rank(ts_delta({f},20), sector) - 0.5"
    return f"group_rank(ts_mean({f},20), sector) - 0.5"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="feeds/feed.jsonl")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--shape", default="auto", choices=["auto", "ratio_trend", "delta", "level"])
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--min-coverage", type=float, default=0.6)
    ap.add_argument("--per-dataset", type=int, default=3)
    args = ap.parse_args()

    fields = load()
    wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}

    # datasets already exercised: referenced by any recorded arm
    used_ds = set()
    fds = {}
    for fid, r in fields.items():
        d = r.get("dataset")
        if isinstance(d, dict):
            fds[fid] = d.get("id")
    for path in glob.glob(str(RESEARCH / "*.jsonl")) + glob.glob(str(RESEARCH / "ledgers/*.jsonl")):
        try:
            text = Path(path).read_text()
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            expr = row.get("expr") or row.get("expression")
            if not expr:
                continue
            for tok in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", expr)):
                if tok in fds:
                    used_ds.add(fds[tok])

    by_ds = {}
    for fid, r in fields.items():
        if r.get("region") != "USA" or r.get("type") not in ("MATRIX", "VECTOR"):
            continue
        ds = fds.get(fid)
        if not ds or (wanted and ds not in wanted):
            continue
        if not wanted and ds in used_ds:
            continue
        cov = r.get("coverage") or 0
        if cov < args.min_coverage:
            continue
        if SKIP.search(fid) or not SEMANTIC.search(f"{fid} {r.get('description') or ''}"):
            continue
        by_ds.setdefault(ds, []).append((cov, fid, r.get("type")))

    arms = []
    for ds, items in sorted(by_ds.items(), key=lambda kv: -max(i[0] for i in kv[1])):
        items.sort(reverse=True)
        for cov, fid, kind in items[: args.per_dataset]:
            shapes = [args.shape] if args.shape != "auto" else ["ratio_trend", "delta"]
            for shape in shapes:
                arms.append({"label": f"fd_{ds[:14]}_{fid[:20]}_{shape[:5]}"[:88],
                             "expr": build(fid, kind, shape), "settings": dict(SETTINGS),
                             "dataset": ds, "coverage": round(cov, 2)})
            if len(arms) >= args.limit:
                break
        if len(arms) >= args.limit:
            break

    out = RESEARCH / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in arms) + "\n")
    print(f"datasets available={len(by_ds)} arms={len(arms)} -> {out}")
    seen = []
    for a in arms:
        if a["dataset"] not in seen:
            seen.append(a["dataset"])
    print("datasets used:", ", ".join(seen[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())