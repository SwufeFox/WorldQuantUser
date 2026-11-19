#!/usr/bin/env python3
"""Full-mode pipeline: simulate arms, then check production correlation.

The submission gate here is PROD_CORRELATION, so it must be measured before a
family is expanded. This runner simulates in Full mode (Quick-mode alphas cannot
be checked), keeps only arms that clear the training bars, runs `alpha check` on
those, and appends one row per arm to a JSONL ledger.

It does not submit anything and does not decide anything: it produces a ranked
table for a human to read.

Usage:
    python3 tools/pipeline_pc.py arms.json --out ledgers/pipeline.jsonl \
        --concurrency 4 --min-sharpe 0.9 --max-turnover 0.7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import machine_lib as ml

API = "https://api.worldquantbrain.com"
DONE = {"COMPLETE", "WARNING"}


def read_ledger(path: Path) -> set[str]:
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
        if row.get("label") and row.get("stage") == "checked":
            done.add(row["label"])
    return done


def simulate(session, arm, poll_timeout=600):
    settings = dict(arm["settings"])
    settings["simulationMode"] = "FULL"
    settings["visualization"] = False
    payload = {"type": "REGULAR", "settings": settings, "regular": arm["expr"]}
    resp = None
    for attempt in range(5):
        resp = session.post(f"{API}/simulations", json=payload, timeout=90)
        if resp.status_code == 201:
            break
        # A long batch can outlive its cookies; re-signing in is cheaper than
        # losing every remaining arm (11 arms died this way, 2026-11-19).
        if resp.status_code in (401, 403):
            time.sleep(5)
            session = ml.load_session(refresh=True)
            continue
        if resp.status_code == 429:
            time.sleep(8 + attempt * 8)
            continue
        return {"label": arm["label"], "stage": "simulate", "status": "POST_FAIL",
                "http": resp.status_code, "detail": (resp.text or "")[:160], "expr": arm["expr"]}
    else:
        return {"label": arm["label"], "stage": "simulate", "status": "BUSY", "expr": arm["expr"]}

    loc = resp.headers.get("Location")
    started = time.time()
    while time.time() - started < poll_timeout:
        time.sleep(4)
        try:
            body = session.get(loc, timeout=60).json()
        except Exception:
            continue
        status = body.get("status")
        if status in DONE | {"ERROR", "FAIL"}:
            alpha_id = body.get("alpha")
            row = {"label": arm["label"], "stage": "simulate", "status": status,
                   "alpha_id": alpha_id, "expr": arm["expr"],
                   "region": settings.get("region"), "universe": settings.get("universe")}
            if alpha_id:
                a = session.get(f"{API}/alphas/{alpha_id}", timeout=60).json()
                is_ = a.get("is") or {}
                row.update(sharpe=is_.get("sharpe"), fitness=is_.get("fitness"),
                           turnover=is_.get("turnover"), returns=is_.get("returns"))
                checks = {c.get("name"): c for c in (is_.get("checks") or [])}
                row["fails"] = sorted(n for n, c in checks.items() if c.get("result") == "FAIL")
                for name in ("LOW_SUB_UNIVERSE_SHARPE", "CONCENTRATED_WEIGHT", "LOW_2Y_SHARPE"):
                    c = checks.get(name) or {}
                    row[name] = {"result": c.get("result"), "value": c.get("value")}
            return row
    return {"label": arm["label"], "stage": "simulate", "status": "TIMEOUT", "expr": arm["expr"]}


def check_pc(session, alpha_id, max_wait=300):
    """Poll the live check endpoint until PROD_CORRELATION resolves."""
    if max_wait <= 0:
        return {"pc_result": "SKIPPED"}
    started = time.time()
    last = {}
    while time.time() - started < max_wait:
        try:
            body = session.get(f"{API}/alphas/{alpha_id}/check", timeout=90)
            if body.status_code == 200:
                payload = body.json()
                checks = payload.get("checks") or {}
                pc = checks.get("PROD_CORRELATION") or {}
                last = {"prod_correlation": pc.get("value"), "pc_result": pc.get("result"),
                        "self_correlation": (checks.get("SELF_CORRELATION") or {}).get("value"),
                        "failed": sorted(k for k, v in checks.items()
                                         if isinstance(v, dict) and v.get("result") == "FAIL")}
                if pc.get("result") not in (None, "PENDING"):
                    return last
        except Exception:
            pass
        time.sleep(8)
    last["pc_result"] = last.get("pc_result") or "UNRESOLVED"
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms")
    ap.add_argument("--out", default="ledgers/pipeline.jsonl")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--min-sharpe", type=float, default=0.9)
    ap.add_argument("--max-turnover", type=float, default=0.7)
    ap.add_argument("--pc-max-wait", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    text = Path(args.arms).read_text()
    try:
        arms = json.loads(text)
    except json.JSONDecodeError:
        # Generators emit JSONL; accept both shapes rather than failing a batch
        # that is otherwise ready to run (feed2 crash, 2026-11-19).
        arms = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(arms, list):
        arms = [arms]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = read_ledger(out)
    todo = [a for a in arms if a["label"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"arms={len(arms)} already-checked={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return 0

    session = ml.load_session()
    probe = session.get(f"{API}/users/self", timeout=30)
    if probe.status_code in (401, 403):
        print("session rejected; signing in again", flush=True)
        session = ml.load_session(refresh=True)

    with out.open("a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(simulate, session, a): a for a in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as exc:
                row = {"label": futures[fut]["label"], "stage": "simulate",
                       "status": "RUNNER_ERROR", "detail": str(exc)}
            s = row.get("sharpe")
            t = row.get("turnover") or 0
            gates_ok = (isinstance(s, (int, float)) and s >= args.min_sharpe
                        and 0.01 <= t <= args.max_turnover and not row.get("fails"))
            if gates_ok and row.get("alpha_id"):
                row.update(check_pc(session, row["alpha_id"], args.pc_max_wait))
                row["stage"] = "checked"
            else:
                row.setdefault("stage", "simulate")
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            sv = f"{s:.2f}" if isinstance(s, (int, float)) else "-"
            print(f"[{n}/{len(todo)}] {row['label'][:34]:34s} {row.get('status'):8s} "
                  f"S={sv} PC={row.get('prod_correlation')} {row.get('pc_result') or ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
