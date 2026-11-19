#!/usr/bin/env python3
"""Batch runner for BRAIN simulations in QUICK mode.

Design notes:
- Ideas, not parameter sweeps: the arms file is expected to vary dataset,
  field or region. Callers should not feed the same expression with
  different numeric constants.
- Resume: labels already present in the output JSONL are skipped.
- Concurrency is capped by the platform (measured ~7 in 2026-11).

Usage:
    python3 tools/batch_quick.py arms.json --out results.jsonl [--concurrency 7]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import machine_lib

QUICK = "QUICK"


def load_done(path: Path) -> set[str]:
    """Labels already finished, plus a normalised-expression set.

    Label-only dedupe missed a real duplicate: ov_valrank_70 and bb_valrank_70
    were the same expression under different names and both consumed a
    simulation. Track the expression too so repeats cannot slip through.
    """
    done: set[str] = set()
    expressions: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") not in {"COMPLETE", "WARNING"}:
            continue
        if row.get("label"):
            done.add(row["label"])
        expr = row.get("expr")
        if expr:
            expressions.add("".join(expr.split()))
    done_with_expr = done
    done_with_expr.update({"expr:" + e for e in expressions})
    # Cross-batch dedupe: another ledger may already hold the same expression
    # under a different label (ov2_surp_np_80 and ov3_surp_np_80 cost two sims).
    for sibling in path.parent.glob("*.jsonl"):
        if sibling == path:
            continue
        try:
            text = sibling.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") not in {"COMPLETE", "WARNING"}:
                continue
            expr = row.get("expr")
            if expr:
                done_with_expr.add("expr:" + "".join(expr.split()))
    return done_with_expr


def simulate(session, arm: dict, poll_timeout: int = 420) -> dict:
    settings = dict(arm["settings"])
    settings.setdefault("simulationMode", QUICK)
    settings.setdefault("visualization", False)
    payload = {"type": "REGULAR", "settings": settings, "regular": arm["expr"]}

    resp = None
    for attempt in range(4):
        try:
            resp = session.post(
                "https://api.worldquantbrain.com/simulations", json=payload, timeout=90
            )
        except Exception as exc:
            # Dropped connections were being recorded as RUNNER_ERROR without a
            # retry, quietly losing arms from an otherwise healthy batch.
            time.sleep(5 + attempt * 10)
            if attempt == 3:
                return {"label": arm["label"], "status": "NETWORK_FAIL",
                        "detail": str(exc)[:180], "expr": arm["expr"]}
            continue
        if resp.status_code == 201:
            break
        if resp.status_code == 429:
            time.sleep(10 + attempt * 10)
            continue
        return {"label": arm["label"], "status": "POST_FAIL", "http": resp.status_code,
                "detail": (resp.text or "")[:180], "expr": arm["expr"]}
    else:
        return {"label": arm["label"], "status": "BUSY", "expr": arm["expr"]}

    loc = resp.headers.get("Location")
    if not loc:
        return {"label": arm["label"], "status": "NO_LOCATION", "expr": arm["expr"]}

    started = time.time()
    while time.time() - started < poll_timeout:
        time.sleep(4)
        try:
            body = session.get(loc, timeout=60).json()
        except Exception as exc:  # transient
            if time.time() - started > poll_timeout - 20:
                break
            continue
        status = body.get("status")
        if status in ("COMPLETE", "WARNING", "ERROR", "FAIL"):
            alpha_id = body.get("alpha") or (body.get("alphaId"))
            row = {"label": arm["label"], "status": status, "alpha_id": alpha_id,
                   "expr": arm["expr"], "region": settings.get("region"),
                   "delay": settings.get("delay"), "universe": settings.get("universe"),
                   "elapsed": round(time.time() - started)}
            if alpha_id:
                try:
                    alpha = session.get(
                        f"https://api.worldquantbrain.com/alphas/{alpha_id}", timeout=60
                    ).json()
                    is_ = alpha.get("is") or {}
                    row.update(sharpe=is_.get("sharpe"), fitness=is_.get("fitness"),
                               turnover=is_.get("turnover"), returns=is_.get("returns"),
                               longCount=is_.get("longCount"), shortCount=is_.get("shortCount"))
                    checks = {c.get("name"): c for c in (is_.get("checks") or [])}
                    row["fails"] = [n for n, c in checks.items() if c.get("result") == "FAIL"]
                    for name in ("LOW_SUB_UNIVERSE_SHARPE", "CONCENTRATED_WEIGHT",
                                 "LOW_2Y_SHARPE", "LOW_SHARPE", "LOW_FITNESS"):
                        c = checks.get(name) or {}
                        row[name] = {"result": c.get("result"), "value": c.get("value")}
                except Exception as exc:
                    row["detail"] = f"alpha read failed: {exc}"
            return row
    return {"label": arm["label"], "status": "TIMEOUT", "expr": arm["expr"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms")
    ap.add_argument("--out", default="batch-results.jsonl")
    ap.add_argument("--concurrency", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    arms = json.loads(Path(args.arms).read_text())
    out = Path(args.out)
    done = load_done(out)
    def already(arm: dict) -> bool:
        if arm["label"] in done:
            return True
        expr = arm.get("expr") or ""
        return ("expr:" + "".join(expr.split())) in done

    todo = [a for a in arms if not already(a)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"arms={len(arms)} done={len(done)} todo={len(todo)} concurrency={args.concurrency}",
          flush=True)
    if not todo:
        return 0

    session = machine_lib.load_session()
    # A killed run can invalidate the cached cookies for the same account:
    # every POST then answers 401 and the batch silently loses its arms.
    probe = session.get("https://api.worldquantbrain.com/users/self", timeout=30)
    if probe.status_code in (401, 403):
        print(f"session rejected ({probe.status_code}); signing in again", flush=True)
        session = machine_lib.load_session(refresh=True)
    started = time.time()
    # Write each result the moment it lands: pool.map yields in submission
    # order, so one slow arm hides every finished arm behind it (2026-11-19).
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, out.open("a") as fh:
        futures = {pool.submit(simulate, session, arm): arm for arm in todo}
        for done_count, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
            except Exception as exc:
                arm = futures[future]
                row = {"label": arm["label"], "status": "RUNNER_ERROR", "detail": str(exc)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            s = row.get("sharpe")
            s = f"{s:.2f}" if isinstance(s, (int, float)) else "-"
            print(f"[{done_count}/{len(todo)}] {row['label'][:34]:34s} {row.get('status')} "
                  f"S={s} T={row.get('turnover')} fails={row.get('fails')}", flush=True)
    print(f"elapsed {time.time()-started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
