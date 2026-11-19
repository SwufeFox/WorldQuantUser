#!/usr/bin/env python3
"""Build a markdown ledger joining robustness, checks, PC and PP eligibility.

Inputs are the JSONL ledgers produced by tools/robustness_pnl.py,
tools/pp_eligibility.py and the pc-*.json files. Missing evidence stays blank;
nothing is inferred.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

RESEARCH = Path("/data/worldquant-research")


def load_jsonl(pattern: str) -> list[dict]:
    rows = []
    for path in glob.glob(str(RESEARCH / pattern)):
        try:
            text = Path(path).read_text()
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> int:
    # Later files must not overwrite a successful measurement with a failed
    # retry, so a row without "error" always wins.
    robust: dict[str, dict] = {}
    for row in load_jsonl("ledgers/robust*.jsonl"):
        aid = row.get("alpha_id")
        if not aid:
            continue
        existing = robust.get(aid)
        if existing is None or (existing.get("error") and not row.get("error")):
            robust[aid] = row
    elig = {r["alpha_id"]: r for r in load_jsonl("ledgers/pp-elig.jsonl") if r.get("alpha_id")}
    for extra in load_jsonl("ledgers/pp-elig-*.jsonl"):
        if extra.get("alpha_id"):
            elig.setdefault(extra["alpha_id"], extra)

    metrics = {}
    for row in load_jsonl("*.jsonl") + load_jsonl("ledgers/*.jsonl"):
        aid = row.get("alpha_id")
        sharpe = row.get("sharpe")
        if not aid or not isinstance(sharpe, (int, float)):
            continue
        prev = metrics.get(aid)
        if prev is None or sharpe > prev.get("sharpe", -9):
            metrics[aid] = {"sharpe": sharpe, "turnover": row.get("turnover"),
                            "label": row.get("label") or (prev or {}).get("label", ""),
                            "fails": row.get("fails") or []}

    pc = {}
    for path in glob.glob(str(RESEARCH / "ledgers/pc-*.json")):
        try:
            payload = json.loads(Path(path).read_text())
        except Exception:
            continue
        for res in (payload.get("data") or {}).get("results") or []:
            value = ((res.get("checks") or {}).get("PROD_CORRELATION") or {}).get("value")
            if isinstance(value, (int, float)):
                pc[res["alpha_id"]] = value

    submitted = {"P05OZO6q", "78YzRA3L", "0mPwXpo6", "QP0731Wr", "qM7jPP3K", "blabL9GM"}

    rows = []
    for aid, rob in robust.items():
        if rob.get("error"):
            continue
        m = metrics.get(aid, {})
        e = elig.get(aid, {})
        last3 = rob["last_three_mean"]
        miny = rob["min_year"]
        flags = rob["flags"]
        if miny > 0 and last3 >= 1.0 and "dead_span" not in flags:
            verdict = "STABLE"
        elif miny < 0 or last3 < 0.6:
            verdict = "FRAGILE"
        else:
            verdict = "WATCH"
        rows.append({
            "alpha_id": aid,
            "label": m.get("label", "")[:34],
            "sharpe": m.get("sharpe"),
            "turnover": m.get("turnover"),
            "first2": rob["first_two_mean"],
            "last3": last3,
            "min_year": miny,
            "flags": ",".join(flags) or "-",
            "pc": pc.get(aid),
            "sub_fail": "LOW_SUB_UNIVERSE_SHARPE" in (e.get("fails") or []),
            "pp": "yes" if e.get("pp_eligible") else ("no" if e else ""),
            "submitted": "yes" if aid in submitted else "",
            "verdict": verdict,
        })

    order = {"STABLE": 0, "WATCH": 1, "FRAGILE": 2}
    rows.sort(key=lambda r: (order[r["verdict"]], -(r["last3"] or -9)))

    lines = [
        "# 稳健性台账 2026-11-19",
        "",
        "判定口径：最差年 Sharpe > 0、后三年均值 ≥ 1.0、无连续三年 < 0.3 → STABLE；"
        "最差年为负或后三年 < 0.6 → FRAGILE；其余 WATCH。",
        "",
        "| alpha | 简称 | S | T | 前两年 | 后三年 | 最差年 | 标记 | PC | 子宇宙 | PP资格 | 已提交 | 判定 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        def fmt(value, precision=None):
            if value is None:
                return "—"
            if isinstance(value, float):
                return f"{value:+.{precision}f}" if precision is not None else f"{value:+.2f}"
            return str(value)
        lines.append(
            f"| `{r['alpha_id']}` | {r['label']} | {fmt(r['sharpe'])} | "
            f"{fmt(r['turnover'])} | {fmt(r['first2'], 2)} | {fmt(r['last3'], 2)} | "
            f"{fmt(r['min_year'], 2)} | {r['flags']} | {fmt(r['pc'], 3)} | "
            f"{'fail' if r['sub_fail'] else ('pass' if r['pp'] else '')} | "
            f"{r['pp']} | {r['submitted']} | **{r['verdict']}** |"
        )

    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    lines += ["", "汇总：" + "，".join(f"{k} {v}" for k, v in counts.items()), ""]

    out = RESEARCH / "reports" / "robust-ledger-20261119.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {len(rows)} rows -> {out}")
    print("verdicts:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
