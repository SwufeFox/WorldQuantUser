#!/usr/bin/env python3
"""Async batch runner built on wqb.

The article BRAIN Tips 004 recommends wqb for bulk work because it is async and
handles authentication and retries itself. The previous runner submitted one
synchronous POST per thread and polled with fixed sleeps, which wastes the
platform's concurrency window on retry backoff instead of keeping slots full.

This runner:
  - reads credentials from the repository .env (never hard-coded),
  - submits every arm through wqb's concurrent_simulate so the slot count stays
    saturated,
  - keeps a JSONL checkpoint so a rerun skips finished arms,
  - reports the same fields the rest of the pipeline expects.

Usage:
    python3 tools/async_batch.py arms.json --out ledgers/async.jsonl --concurrency 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import wqb

RESEARCH = Path("/data/worldquant-research")
ENV_CANDIDATES = [Path("/data/src/magix/WorldQuantUser/.env")]


def load_env() -> tuple[str, str]:
    for name in ("BRAIN_USERNAME", "BRAIN_PASSWORD"):
        if os.environ.get(name):
            continue
        for path in ENV_CANDIDATES:
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                if line.startswith(f"{name}="):
                    os.environ[name] = line.split("=", 1)[1].strip().strip('"').strip("'")
    user = os.environ.get("BRAIN_USERNAME")
    password = os.environ.get("BRAIN_PASSWORD")
    if not user or not password:
        raise SystemExit("BRAIN_USERNAME / BRAIN_PASSWORD not found")
    return user, password


def read_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("label") and row.get("status") in {"COMPLETE", "WARNING"}:
            done.add(row["label"])
    return done


def payload_for(arm: dict) -> dict:
    settings = dict(arm["settings"])
    settings.setdefault("simulationMode", "QUICK")
    settings.setdefault("visualization", False)
    return {"type": "REGULAR", "settings": settings, "regular": arm["expr"]}


async def run(arms: list[dict], out: Path, concurrency: int, session) -> int:
    done = read_done(out)
    todo = [a for a in arms if a["label"] not in done]
    print(f"arms={len(arms)} done={len(done)} todo={len(todo)} concurrency={concurrency}", flush=True)
    if not todo:
        return 0

    targets = [payload_for(a) for a in todo]
    started = time.time()
    results = await session.concurrent_simulate(
        targets, concurrency=concurrency, return_exceptions=True, log="", log_gap=10
    )
    elapsed = time.time() - started

    rows = []
    for arm, result in zip(todo, results):
        row = {"label": arm["label"], "expr": arm["expr"],
               "region": arm["settings"].get("region"),
               "universe": arm["settings"].get("universe")}
        if isinstance(result, BaseException):
            row.update(status="RUNNER_ERROR", detail=str(result)[:180])
            rows.append(row)
            continue
        try:
            body = result.json()
        except Exception:
            row.update(status="BAD_RESPONSE", detail=str(getattr(result, "text", ""))[:180])
            rows.append(row)
            continue
        status = body.get("status") or ("COMPLETE" if body.get("alpha") else "UNKNOWN")
        alpha_id = body.get("alpha")
        row.update(status=status, alpha_id=alpha_id)
        if alpha_id:
            try:
                d = await asyncio.to_thread(
                    lambda: session.get(f"{wqb.WQB_API_URL}/alphas/{alpha_id}").json()
                )
                is_ = d.get("is") or {}
                row.update(sharpe=is_.get("sharpe"), fitness=is_.get("fitness"),
                           turnover=is_.get("turnover"), returns=is_.get("returns"))
                checks = {c.get("name"): c for c in (is_.get("checks") or [])}
                row["fails"] = sorted(n for n, c in checks.items() if c.get("result") == "FAIL")
            except Exception as exc:
                row["detail"] = f"alpha read failed: {exc}"[:160]
        rows.append(row)

    with out.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()

    scored = [r for r in rows if isinstance(r.get("sharpe"), (int, float))]
    clean = [r for r in scored if not (r.get("fails") or []) and r["sharpe"] >= 1.0]
    rate = len(rows) / elapsed * 3600 if elapsed else 0
    print(f"finished {len(rows)} in {elapsed:.0f}s = {rate:.0f} arms/hour "
          f"| scored {len(scored)} clean {len(clean)}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms")
    ap.add_argument("--out", default="ledgers/async.jsonl")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    text = Path(args.arms).read_text()
    try:
        arms = json.loads(text)
    except json.JSONDecodeError:
        arms = [json.loads(line) for line in text.splitlines() if line.strip()]

    user, password = load_env()
    session = wqb.WQBSession(wqb_auth=(user, password))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(arms, out, args.concurrency, session))


if __name__ == "__main__":
    sys.exit(main())