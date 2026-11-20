#!/usr/bin/env python3
"""Run Full simulations and check production correlation on eligible results.

Simulations and PC checks have independent workers. A finished simulation is
written to the JSONL ledger before its PC check starts, so a later invocation
can continue that check using the existing Alpha ID.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from requests.exceptions import RequestException

import cache_store
import machine_lib as ml
import simulation_capabilities
from tools.pipeline.batch_quick import (
    SUCCESS_STATUSES, Runner, _append, _batch_args, _batch_parser, _completed_rows,
    _finished, _load_arms, _open_ledger, _payload, _run_locked, _simulation_key,
)

RESOLVED_PC = {"PASS", "FAIL", "WARNING"}


def read_ledger(path):
    """Collect usable simulation evidence, including old simulate-only rows."""
    records = defaultdict(list)
    for row in _completed_rows(path):
        if row.get("label") and row.get("expr"):
            records[row["label"], row["expr"]].append(row)
    return records


def recorded_result(arm, records):
    prior = records.get((arm["label"], arm["expr"]), ())
    settings = arm["settings"]
    configured = [row for row in prior if isinstance(row.get("settings"), dict) or row.get("simulation_key")]
    if configured:
        key = _simulation_key(_payload(arm))
        return next((row for row in reversed(configured)
                     if row.get("simulation_key") == key or row.get("settings") == settings), None)
    # Legacy ledgers recorded only the output scope, so resume them using the
    # original arms file. Refuse known scope mismatches even in these ledgers.
    return next((row for row in reversed(prior)
                 if row.get("stage") in {"simulate", "checked", "check_pending"}
                 and all(row.get(key) in (None, settings.get(key))
                        for key in ("region", "universe", "delay"))), None)


def pc_eligible(row, args):
    sharpe = row.get("sharpe")
    turnover = row.get("turnover") or 0
    return (row.get("status") in SUCCESS_STATUSES and row.get("alpha_id")
            and isinstance(sharpe, (int, float)) and sharpe >= args.min_sharpe
            and 0.01 <= turnover <= args.max_turnover and not row.get("fails"))


def check_pc(runner, alpha_id, max_wait):
    """Fetch native checks within a bounded budget, leaving stalled PC retryable."""
    deadline = time.monotonic() + max_wait
    last = {"pc_result": "UNRESOLVED"}
    refreshed = False
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            result = ml.fetch_alpha_checks(runner._session(), alpha_id, max_wait_seconds=remaining)
        except ml.AuthError:
            if refreshed:
                break
            try:
                runner._session(refresh=True)
            except (ml.BrainError, RequestException, cache_store.CacheError) as exc:
                last["pc_error"] = type(exc).__name__
                break
            refreshed = True
            continue
        except (ml.BrainError, RequestException, cache_store.CacheError) as exc:
            last["pc_error"] = type(exc).__name__
            break
        checks = result.get("checks") or {}
        pc = checks.get("PROD_CORRELATION") or {}
        result = pc.get("result")
        last = {"prod_correlation": pc.get("value"),
                "pc_result": result if result in RESOLVED_PC else "UNRESOLVED",
                "self_correlation": (checks.get("SELF_CORRELATION") or {}).get("value"),
                "failed": sorted(name for name, check in checks.items()
                                 if check.get("result") == "FAIL")}
        if result in RESOLVED_PC:
            break
        time.sleep(min(8, max(0, deadline - time.monotonic())))
    return last


def _simulate(runner, arm):
    return {**runner.run(arm), "settings": arm["settings"], "stage": "simulate"}


def _check(runner, arm, row, max_wait):
    evidence = check_pc(runner, row["alpha_id"], max_wait)
    stage = "checked" if evidence["pc_result"] in RESOLVED_PC else "check_pending"
    return {**row, **evidence, "settings": arm["settings"], "stage": stage}


def _process(runner, items, args, out):
    with (
        _open_ledger(out) as stream,
        ThreadPoolExecutor(max_workers=args.concurrency) as simulations,
        ThreadPoolExecutor(max_workers=min(3, args.concurrency)) as checks,
    ):
        def check(arm, row):
            return checks.submit(_check, runner, arm, row, args.pc_max_wait)

        pending = {}
        for arm, row in items:
            future = (check(arm, row) if row else simulations.submit(_simulate, runner, arm))
            pending[future] = (arm, row)
        finished = 0
        for (arm, old), future in _finished(pending):
            try:
                row = future.result()
            except Exception as exc:
                if old:
                    row = {**old, "settings": arm["settings"], "stage": "check_pending",
                           "pc_result": "UNRESOLVED", "pc_error": type(exc).__name__}
                else:
                    row = {"label": arm["label"], "expr": arm["expr"],
                           "settings": arm["settings"], "stage": "simulate",
                           "status": "RUNNER_ERROR", "detail": type(exc).__name__}
            _append(stream, row)
            if old is None and args.pc_max_wait > 0 and pc_eligible(row, args):
                pending[check(arm, row)] = (arm, row)
                continue
            finished += 1
            sharpe = row.get("sharpe")
            display = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "-"
            print(f"[{finished}/{len(items)}] {row['label'][:34]:34s} {row.get('status'):8s} "
                  f"S={display} PC={row.get('prod_correlation')} {row.get('pc_result') or ''}", flush=True)


def _run_batch(args, out):
    arms = _load_arms(Path(args.arms))
    ledger = read_ledger(out)
    items = []
    seen = set()
    for raw in arms:
        arm = {**raw, "settings": {**raw["settings"], "simulationMode": "FULL", "visualization": False}}
        key = arm["label"], arm["expr"], json.dumps(arm["settings"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        prior = recorded_result(arm, ledger)
        if prior is None or (args.pc_max_wait > 0 and prior.get("pc_result") not in RESOLVED_PC
                             and pc_eligible(prior, args)):
            items.append((arm, prior))
    if args.limit:
        items = items[:args.limit]
    print(f"arms={len(arms)} todo={len(items)} reused={sum(row is not None for _, row in items)}", flush=True)
    if not items:
        return 0

    with ml.load_session() as session:
        account = simulation_capabilities.account_provenance(session)
    runner = Runner(out, account, args.account_slots, poll_timeout=600)
    try:
        _process(runner, items, args, out)
    finally:
        runner.close()
    return 0


def main():
    parser = _batch_parser(__doc__, "ledgers/pipeline.jsonl", 4)
    parser.add_argument("--min-sharpe", type=float, default=0.9)
    parser.add_argument("--max-turnover", type=float, default=0.7)
    parser.add_argument("--pc-max-wait", type=float, default=300)
    args = _batch_args(parser)
    if not math.isfinite(args.pc_max_wait) or args.pc_max_wait < 0:
        parser.error("PC max wait must be non-negative and finite")
    return _run_locked(parser, args, _run_batch)


if __name__ == "__main__":
    sys.exit(main())
