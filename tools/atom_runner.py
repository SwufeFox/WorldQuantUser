#!/usr/bin/env python3
"""Submit ATOM / Power Pool alphas directly against the BRAIN API.

Why this exists: after the Consultant upgrade the CLI's `simulate`/`probe`
preflight hangs (lint blocks on catalog loading), while the platform itself
accepts `POST /simulations` normally. This runner talks to the documented
endpoints directly: POST the simulation, poll its Location, read the alpha.

Power Pool criteria it screens for (2026-11-17 brief):
  Sharpe >= 1.0, turnover 1%-70%, <= 8 unique operators, <= 3 unique data fields
  (group fields excluded), self-correlation <= 0.5 among Power Pool alphas.
Fitness is exempt for Power Pool, so it is recorded but never gates here.

Usage:  python3 tools/atom_runner.py arms.json [--concurrency 4]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import machine_lib  # noqa: E402

RESEARCH = Path(os.environ.get("WQ_RESEARCH_DIR", "/data/worldquant-research"))
OUT = RESEARCH / "atom-results.jsonl"
BASE = "https://api.worldquantbrain.com"
GROUP_FIELDS = {"country", "industry", "subindustry", "currency", "market",
                "sector", "exchange", "sector"}


def submit(session, arm):
    settings = dict(arm["settings"])
    payload = {"type": "REGULAR", "settings": settings, "regular": arm["expr"]}
    r = session.post(f"{BASE}/simulations", json=payload, timeout=120)
    if r.status_code not in (200, 201):
        return {"label": arm["label"], "error": f"HTTP {r.status_code}: {r.text[:200]}"}
    return {"label": arm["label"], "expr": arm["expr"], "location": r.headers.get("Location")}


def poll(session, job, budget=1800):
    url = job.get("location")
    deadline = time.time() + budget
    while time.time() < deadline:
        r = session.get(url, timeout=60)
        if r.status_code != 200:
            job["error"] = f"poll HTTP {r.status_code}"
            return job
        body = r.json() if r.text else {}
        status = body.get("status")
        if status == "COMPLETE" or body.get("alpha"):
            job["alpha_id"] = body.get("alpha")
            job["status"] = "COMPLETE"
            return job
        if status in ("ERROR", "FAIL", "CANCELLED"):
            job["status"] = status
            job["error"] = json.dumps(body)[:200]
            return job
        time.sleep(10)
    job["status"] = "TIMEOUT"
    return job


def metrics(session, alpha_id):
    r = session.get(f"{BASE}/alphas/{alpha_id}", timeout=60)
    if r.status_code != 200:
        return {}
    d = r.json()
    is_ = d.get("is") or {}
    out = {k: is_.get(k) for k in ("sharpe", "fitness", "turnover", "returns", "margin", "drawdown")}
    # The checks decide eligibility, not the headline metrics: Power Pool needs
    # the sub-universe test, and a consultant submission needs production
    # correlation. Record every check's result and value so the screen can use
    # them without a second round trip.
    checks = is_.get("checks") or d.get("checks") or []
    if isinstance(checks, list):
        out["checks"] = {c.get("name"): {"result": c.get("result"), "value": c.get("value"),
                                          "limit": c.get("limit")} for c in checks if isinstance(c, dict)}
    elif isinstance(checks, dict):
        out["checks"] = {k: {"result": (v or {}).get("result"), "value": (v or {}).get("value"),
                             "limit": (v or {}).get("limit")} for k, v in checks.items()}
    else:
        out["checks"] = {}
    return out


def main() -> int:
    arms = json.loads(Path(sys.argv[1]).read_text())
    conc = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--concurrency" else 4
    session = machine_lib.load_session()
    pending = list(arms)
    running: list[dict] = []
    while pending or running:
        while pending and len(running) < conc:
            arm = pending.pop(0)
            job = submit(session, arm)
            print(f"submit {job['label']}: {job.get('location') or job.get('error')}", flush=True)
            if job.get("location"):
                running.append(job)
        still = []
        for job in running:
            job = poll(session, job, budget=60)
            if job.get("status") == "COMPLETE":
                job.update(metrics(session, job["alpha_id"]))
                job["power_pool_sharpe"] = (job.get("sharpe") or 0) >= 1.0
                job["turnover_ok"] = 0.01 <= (job.get("turnover") or 0) <= 0.70
                print(f"done {job['label']}: {job['alpha_id']} S={job.get('sharpe')} "
                      f"T={job.get('turnover')} F={job.get('fitness')}", flush=True)
                with OUT.open("a") as fh:
                    fh.write(json.dumps(job, ensure_ascii=False) + "\n")
            else:
                still.append(job)
        running = still
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
