#!/usr/bin/env python3
"""Run Full simulations and check production correlation on eligible results.

Simulations and PC checks have independent workers. A finished simulation is
written to the JSONL ledger before its PC check starts, so a later invocation
can continue that check using the existing Alpha ID.
"""
from __future__ import annotations

import math
import re
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
import simulation_slots
from tools.pipeline import batch_quick
from tools.pipeline.batch_quick import (
    SUCCESS_STATUSES, Runner, _append, _batch_args, _batch_parser, _finished,
    _json_object, _jsonl_rows, _load_arms, _open_ledger, _payload, _retry_after,
    _run_locked, _simulation_key,
)

RESOLVED_PC = {"PASS", "FAIL", "WARNING"}


def read_ledger(path):
    """Collect completed simulations and known terminal failures."""
    records = defaultdict(list)
    for row in _jsonl_rows(path):
        status = row.get("status")
        complete = status in SUCCESS_STATUSES and row.get("alpha_id")
        terminal = status in simulation_slots.TERMINAL_FAILURE_STATUSES and row.get("simulation_job_key")
        if (complete or terminal) and row.get("label") and row.get("expr"):
            records[row["label"], row["expr"]].append(row)
    return records


def _verified_row(arm, row, account):
    return {**row, "settings": arm["settings"], "simulation_key": _simulation_key(_payload(arm)),
            "account_fingerprint": account}


def _alpha_snapshot(runner, alpha_id, deadline):
    if not isinstance(alpha_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", alpha_id):
        return None
    url = f"{ml.BRAIN_API_URL}/alphas/{alpha_id}"
    snapshot = runner._read(url, deadline)
    if isinstance(snapshot, dict):
        author = snapshot.get("author")
        if isinstance(author, dict):
            author = author.get("id")
        if type(author) in (str, int):
            session = runner._session(deadline=deadline)
            if str(author) != str(getattr(session, "_brain_account_id", None)):
                return None
    return snapshot


def _legacy_match(arm, row, snapshot):
    """Distinguish a contradiction from missing proof of the executed request."""
    if not isinstance(snapshot, dict) or snapshot.get("id") != row["alpha_id"]:
        return None
    expected = _payload(arm)
    observed = snapshot.get("settings")
    if snapshot.get("type") not in (None, expected["type"]):
        return False
    code, _, _ = ml.regular_expression(snapshot)
    if isinstance(code, str) and _simulation_key({**expected, "regular": code}) != _simulation_key(expected):
        return False
    if isinstance(observed, dict):
        requested = ml.normalize_setting_keys(expected["settings"])
        try:
            actual = ml.normalize_setting_keys(observed)
        except ValueError:
            return None
        if any(key in actual and actual[key] is not None and actual[key] != value
               for key, value in requested.items()):
            return False
    return True if batch_quick._matches_alpha_payload(snapshot, expected) else None


def recorded_result(arm, records, runner, snapshots, *, deadline):
    """Prefer proven completions, then check plausible unscoped receipts."""
    prior = records.get((arm["label"], arm["expr"]), ())
    account = runner.account["fingerprint"]
    payload = _payload(arm)
    key = _simulation_key(payload)
    for row in reversed(prior):
        if row.get("account_fingerprint") != account:
            continue
        saved_key = row.get("simulation_key")
        same_payload = saved_key == key if saved_key else row.get("settings") == arm["settings"]
        if same_payload:
            return row
    job_key = runner._job_key(arm["label"], payload)
    if job_key in runner.jobs or job_key in runner.shared_jobs:
        return None

    unresolved = None
    for row in reversed(prior):
        if (row.get("account_fingerprint") or not row.get("alpha_id")
                or row.get("stage") not in {"simulate", "checked", "check_pending"}
                or any(row.get(name) not in (None, arm["settings"].get(name))
                       for name in ("region", "universe", "delay"))):
            continue
        alpha_id = row["alpha_id"]
        if alpha_id not in snapshots:
            try:
                snapshots[alpha_id] = _alpha_snapshot(runner, alpha_id, deadline)
            except (ml.BrainError, RequestException, cache_store.CacheError):
                snapshots[alpha_id] = None
        match = _legacy_match(arm, row, snapshots[alpha_id])
        if match is True:
            return _verified_row(arm, row, account)
        if match is None and unresolved is None:
            unresolved = {"label": arm["label"], "expr": arm["expr"], "alpha_id": alpha_id,
                          "status": "REUSE_UNRESOLVED", "stage": "reuse_pending",
                          "requested_settings": arm["settings"],
                          "detail": "could not establish the legacy Alpha's executed settings"}
    return unresolved


def pc_eligible(row, args):
    sharpe = row.get("sharpe")
    turnover = row.get("turnover") or 0
    return (row.get("status") in SUCCESS_STATUSES and row.get("alpha_id")
            and isinstance(sharpe, (int, float)) and sharpe >= args.min_sharpe
            and 0.01 <= turnover <= args.max_turnover and not row.get("fails"))


def check_pc(runner, alpha_id, max_wait):
    """Fetch native checks within a bounded budget, leaving stalled PC retryable."""
    if not isinstance(alpha_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", alpha_id):
        return {"pc_result": "UNRESOLVED", "pc_error": "InvalidAlphaID"}
    deadline = time.monotonic() + max_wait
    last = {"pc_result": "UNRESOLVED"}
    refreshed = False
    url = f"{ml.BRAIN_API_URL}/alphas/{alpha_id}/check"
    while not runner.stop_requested.is_set() and time.monotonic() < deadline:
        retry_after = 0.0
        try:
            session = runner._session(deadline=deadline)
            if time.monotonic() >= deadline or runner.stop_requested.is_set():
                break
            response = session.get(url, timeout=ml._request_timeout(deadline), allow_redirects=False)
            retry_after = _retry_after(response)
            if response.status_code in (401, 403):
                raise ml.AuthError("authentication rejected for Alpha check")
        except RequestException as exc:
            last["pc_error"] = type(exc).__name__
        except ml.AuthError:
            if refreshed:
                last["pc_error"] = "AuthError"
                break
            try:
                runner._session(refresh=True, deadline=deadline)
            except (ml.BrainError, RequestException, cache_store.CacheError) as exc:
                last["pc_error"] = type(exc).__name__
                break
            refreshed = True
            continue
        except (ml.BrainError, cache_store.CacheError) as exc:
            last["pc_error"] = type(exc).__name__
            break
        else:
            if response.status_code == 429:
                last["pc_error"] = "RateLimited"
            elif response.status_code == 200 and not retry_after:
                body = _json_object(response)
                # Core alpha-check receipts use is.checks; keep checking while
                # the service is still returning an incomplete payload.
                checks = ml._extract_checks(body) if isinstance(body.get("is"), dict) else {}
                checks = checks or {}
                pc = checks.get("PROD_CORRELATION") or {}
                result = pc.get("result")
                last = {"prod_correlation": pc.get("value"),
                        "pc_result": result if result in RESOLVED_PC else "UNRESOLVED",
                        "self_correlation": (checks.get("SELF_CORRELATION") or {}).get("value"),
                        "failed": sorted(name for name, check in checks.items()
                                         if check.get("result") == "FAIL")}
                if result in RESOLVED_PC:
                    break
            elif not (response.status_code == 200 or response.status_code >= 500):
                last["pc_error"] = "NotFound" if response.status_code == 404 else "ServiceError"
                break
        delay = retry_after or 8
        if delay >= deadline - time.monotonic() or runner.stop_requested.wait(delay):
            break
    return last


def _simulate(runner, arm):
    row = runner.run(arm)
    if row.get("status") in simulation_slots.TERMINAL_STATUSES:
        row = _verified_row(arm, row, runner.account["fingerprint"])
    else:
        row["requested_settings"] = arm["settings"]
    return {**row, "simulation_job_key": runner._job_key(arm["label"], _payload(arm)),
            "stage": "simulate"}


def _check(runner, row, max_wait):
    evidence = check_pc(runner, row["alpha_id"], max_wait)
    stage = "checked" if evidence["pc_result"] in RESOLVED_PC else "check_pending"
    updated = {**row, **evidence, "stage": stage}
    if "pc_error" not in evidence:
        updated.pop("pc_error", None)
    return updated


def _process(runner, items, args, out):
    with (
        _open_ledger(out) as stream,
        ThreadPoolExecutor(max_workers=args.concurrency) as simulations,
        ThreadPoolExecutor(max_workers=min(3, args.concurrency)) as checks,
    ):
        pending = {}
        simulation_arms = iter(arm for arm, row in items if row is None)

        def submit_next_simulation():
            arm = next(simulation_arms, None)
            if arm is not None:
                pending[simulations.submit(_simulate, runner, arm)] = (arm, None)

        def submit_check(arm, row):
            pending[checks.submit(_check, runner, row, args.pc_max_wait)] = (arm, row)

        try:
            for arm, old in items:
                if old is not None:
                    submit_check(arm, old)
            for _ in range(args.concurrency):
                submit_next_simulation()
            finished = 0
            for (arm, old), future in _finished(pending):
                try:
                    row = future.result()
                except Exception as exc:
                    if old is not None:
                        row = {**old, "stage": "check_pending", "pc_result": "UNRESOLVED",
                               "pc_error": type(exc).__name__}
                    else:
                        row = {"label": arm["label"], "expr": arm["expr"],
                               "requested_settings": arm["settings"], "stage": "simulate",
                               "status": "RUNNER_ERROR", "detail": type(exc).__name__}
                _append(stream, row)
                if old is None:
                    submit_next_simulation()
                    if args.pc_max_wait > 0 and pc_eligible(row, args):
                        submit_check(arm, row)
                        continue
                finished += 1
                sharpe = row.get("sharpe")
                display = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "-"
                print(f"[{finished}/{len(items)}] {row['label'][:34]:34s} {row.get('status'):8s} "
                      f"S={display} PC={row.get('prod_correlation')} {row.get('pc_result') or ''}", flush=True)
        except KeyboardInterrupt:
            runner.stop_requested.set()
            for future in pending:
                future.cancel()
            raise


def _run_batch(args, out):
    arms = _load_arms(Path(args.arms))
    if not arms:
        return 0
    with ml.load_session() as session:
        account = simulation_capabilities.account_provenance(session)
    runner = Runner(out, account, args.account_slots, poll_timeout=600)
    try:
        ledger = read_ledger(out)
        snapshots = {}
        legacy_deadline = time.monotonic() + 30
        recovery, items, deferred = [], [], []
        seen = set()
        for raw in arms:
            arm = {**raw, "settings": {**raw["settings"], "simulationMode": "FULL", "visualization": False}}
            key = runner._job_key(arm["label"], _payload(arm))
            if key in seen:
                continue
            seen.add(key)
            prior = recorded_result(arm, ledger, runner, snapshots, deadline=legacy_deadline)
            if prior is None:
                queue = recovery if key in runner.jobs or key in runner.shared_jobs else items
                queue.append((arm, None))
            elif prior.get("stage") == "reuse_pending":
                deferred.append(prior)
            elif args.pc_max_wait > 0 and prior.get("pc_result") not in RESOLVED_PC and pc_eligible(prior, args):
                items.append((arm, prior))
        items = (recovery + items)[:args.limit or None]
        print(f"arms={len(arms)} todo={len(items)} reused={sum(row is not None for _, row in items)} "
              f"reuse_unresolved={len(deferred)}", flush=True)
        if deferred:
            with _open_ledger(out) as stream:
                for row in deferred:
                    _append(stream, row)
        if items:
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
