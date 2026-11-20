#!/usr/bin/env python3
"""Run Quick simulations with a shared account budget and durable progress.

Use `--concurrency` for this process's workers and `--account-slots` for
the budget shared with other runners in the research workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext, suppress
from pathlib import Path

from requests.exceptions import ConnectTimeout, RequestException

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cache_store
import fastexpr_ast
import machine_lib
import research_paths
import simulation_capabilities
import simulation_slots

QUICK = "QUICK"
DEFAULT_ACCOUNT_SLOTS = 5
SUCCESS_STATUSES = {"COMPLETE", "COMPLETED", "WARNING"}
CHECK_NAMES = ("LOW_SUB_UNIVERSE_SHARPE", "CONCENTRATED_WEIGHT",
               "LOW_2Y_SHARPE", "LOW_SHARPE", "LOW_FITNESS")
LEGACY_SCOPE = ("region", "universe")


def _simulation_key(payload):
    """Compare source and every setting sent to the simulation API."""
    source = payload["regular"]
    if machine_lib.is_python_language(machine_lib.alpha_language(payload)):
        source = machine_lib.normalize_python_source(source)
    else:
        source = fastexpr_ast.canonical(source)
    text = json.dumps({**payload, "regular": source}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _jsonl_rows(path):
    try:
        fh = path.open(encoding="utf-8")
    except FileNotFoundError:
        return
    with fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _completed_rows(path):
    return (row for row in _jsonl_rows(path) if row.get("status") in SUCCESS_STATUSES and row.get("alpha_id"))


@contextmanager
def _open_ledger(path):
    with path.open("a+", encoding="utf-8") as stream:
        # Preserve a partial record from an interrupted append on its own line.
        if stream.tell():
            with path.open("rb") as existing:
                existing.seek(-1, os.SEEK_END)
                if existing.read(1) != b"\n":
                    stream.write("\n")
        yield stream


def _append(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def load_done(path):
    """Index exact completions across sibling ledgers and local legacy rows."""
    exact, own = set(), {}
    siblings = (entry for entry in path.parent.glob("*.jsonl") if entry != path)
    for ledger in (path, *siblings):
        try:
            for row in _completed_rows(ledger):
                key = row.get("simulation_key")
                if isinstance(key, str) and key:
                    exact.add(key)
                if ledger == path and isinstance(row.get("label"), str):
                    own.setdefault(row["label"], []).append(row)
        except OSError:
            if ledger == path:
                raise
    return exact, own


def _legacy_done(row, arm, settings):
    # The old wqb output did not carry delay. Match only recorded settings
    # and keep this fallback local to the same output, label and source text.
    recorded = row.get("settings")
    if not isinstance(recorded, dict):
        recorded = {}
    known = {**row, **recorded}
    keys = settings.keys() | machine_lib.DEFAULT_SETTINGS.keys() | recorded.keys()
    return (not row.get("simulation_key") and row.get("expr") == arm["expr"]
            and all(known.get(name) is not None and known[name] == settings.get(name)
                    for name in LEGACY_SCOPE)
            and known.get("simulationMode", QUICK) == settings["simulationMode"]
            and all(known[name] == settings.get(name) for name in keys if known.get(name) is not None))


def _load_arms(path):
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        arms = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        arms = value if isinstance(value, list) else [value]
    if not all(isinstance(arm, dict) and isinstance(arm.get("label"), str)
               and isinstance(arm.get("expr"), str) and isinstance(arm.get("settings"), dict)
               for arm in arms):
        raise ValueError("arms must contain label, expr and settings per row")
    return arms


def _retry_after(response):
    value = machine_lib._retry_after_seconds(response, cap_error_wait=False)
    return value if math.isfinite(value) else 0


def _json_object(response):
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _terminal_receipt(response, body):
    status = str(body.get("status", "")).upper()
    statuses = {200: simulation_slots.TERMINAL_STATUSES, 400: simulation_slots.TERMINAL_FAILURE_STATUSES}
    alpha_id = body.get("alpha") or body.get("alphaId")
    if status in statuses.get(response.status_code, ()) and (status not in SUCCESS_STATUSES or alpha_id):
        return {"status": status, "alpha_id": alpha_id}


def _poll_interval(body, delay, previous_progress):
    progress = body.get("progress")
    if type(progress) in (float, int) and 0 <= progress <= 1:
        if previous_progress is None or progress > previous_progress:
            return max(delay, 0.5 + 3.5 * progress), progress
    return min(4.0, 1.5 * delay), previous_progress


def _payload(arm):
    settings = dict(arm["settings"])
    settings.setdefault("simulationMode", QUICK)
    settings.setdefault("visualization", False)
    return {"type": "REGULAR", "settings": settings, "regular": arm["expr"]}


class Runner:
    def __init__(self, out: Path, account: dict, slots: int, poll_timeout: float):
        if not account.get("fingerprint"):
            raise ValueError("quick runner needs an authenticated account for shared slots")
        self.out = out.resolve()
        self.account = account
        self.slots = slots
        self.poll_timeout = poll_timeout
        self.state_path = self.out.with_suffix(self.out.suffix + ".state")
        self.state_lock = threading.Lock()
        self.jobs = {}
        self.saved_labels = set()
        self.shared_jobs = simulation_slots.quick_known_jobs(self.account)
        for path in self.state_path.glob("*.json"):
            saved, error = cache_store.load_json(path)
            if error or not isinstance(saved, dict) or saved.get("account") != account["fingerprint"]:
                raise ValueError("quick runner checkpoint is unreadable or belongs to another account")
            self.jobs[path.stem] = saved
            self.saved_labels.add(saved["label"])
        sidecar = self.out.with_suffix(self.out.suffix + ".loc")
        self.legacy_locations = {row["label"]: row["loc"] for row in _jsonl_rows(sidecar)
                                 if row.get("label") and row.get("loc")}
        self.local = threading.local()
        self.sessions = []
        self.sessions_lock = threading.Lock()
        self.reap_lock = threading.Lock()
        self.next_reap = 0.0
        self.reap_offset = 0

    def _session_generation(self):
        return cache_store.read_entry(machine_lib.session_cache_path()).get("created_at")

    def _session(self, refresh=False):
        previous = getattr(self.local, "session", None)
        if previous is not None and not refresh:
            return previous

        # An old response can arrive after a different worker refreshed the cookie.
        guard = (cache_store.file_lock(research_paths.lock_path("quick-runner-refresh"), timeout=90)
                 if refresh else nullcontext())
        with guard:
            before = self._session_generation()
            force_login = refresh and (not before or before == getattr(self.local, "generation", None))
            session = machine_lib.load_session(refresh=force_login)
            if simulation_capabilities.account_provenance(session)["fingerprint"] != self.account["fingerprint"]:
                session.close()
                raise ValueError("authenticated account changed while running simulations")
            self.local.session = session
            self.local.generation = self._session_generation() if force_login else before
            with self.sessions_lock:
                self.sessions.append(session)
                if previous is not None:
                    previous.close()
                    self.sessions.remove(previous)
            return session

    def close(self):
        for session in self.sessions:
            session.close()

    def _key(self, label, payload):
        data = json.dumps([str(self.out), label, payload], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data.encode()).hexdigest()

    def _checkpoint(self, key, row):
        saved = {**row, "account": self.account["fingerprint"]}
        with self.state_lock:
            cache_store.write_json_atomic(self.state_path / f"{key}.json", saved)
            self.jobs[key] = row
            self.saved_labels.add(row["label"])

    def _existing(self, key, label):
        with self.state_lock:
            saved = self.jobs.get(key, {})
            reused_label = label in self.saved_labels
        if saved or reused_label or label not in self.legacy_locations:
            return saved
        return {"progress_url": machine_lib._validated_progress_url(self.legacy_locations[label])}

    def _reap(self, key):
        if time.monotonic() < self.next_reap or not self.reap_lock.acquire(blocking=False):
            return
        try:
            self.next_reap = time.monotonic() + 4
            orphans = simulation_slots.quick_orphans(self.account, exclude_job=key)
            if not orphans:
                return
            orphan = orphans[self.reap_offset % len(orphans)]
            self.reap_offset += 1
            url = machine_lib._validated_progress_url(orphan["progress_url"])
            try:
                response = self._session().get(url, timeout=5, allow_redirects=False)
            except (RequestException, machine_lib.AuthError, machine_lib.SessionBusy):
                return
            receipt = _terminal_receipt(response, _json_object(response))
            if receipt:
                simulation_slots.quick_update(self.account, orphan["id"], "terminal", retain=True,
                                              terminal_result=receipt)
        finally:
            self.reap_lock.release()

    def _read(self, url, deadline, *, progress=False):
        delay = 0.5
        previous_progress = None
        wait = delay if progress else 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or wait >= remaining:
                return None
            if wait:
                time.sleep(wait)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                response = self._session().get(url, timeout=min(60, remaining),
                                               allow_redirects=False)
            except RequestException:
                wait = delay = min(4.0, 1.5 * delay)
                continue
            body = _json_object(response)
            if progress and (receipt := _terminal_receipt(response, body)):
                return receipt
            retry_after = _retry_after(response)
            if response.status_code == 401:
                with suppress(machine_lib.AuthError, machine_lib.SessionBusy, cache_store.LockTimeout):
                    self._session(refresh=True)
            elif not progress and response.status_code == 200 and body.get("is") and not retry_after:
                return body
            if progress:
                delay, previous_progress = _poll_interval(body, delay, previous_progress)
            else:
                delay = min(4.0, 1.5 * delay)
            wait = max(delay, retry_after)

    def _post(self, payload, key, row, *, existing_url):
        delay = 0.5
        deadline = time.monotonic() + self.poll_timeout
        retry_at = 0.0
        while True:
            self._reap(key)
            slot = simulation_slots.quick_acquire(self.account, self.slots, key,
                                                  existing_url=existing_url, prepare=True)
            if slot is not None:
                if slot.get("progress_url"):
                    return slot, None
                if slot["state"] == "unknown":
                    return None, {**row, "status": "UNKNOWN_SUBMISSION",
                                  "detail": "previous POST outcome is unknown"}
                try:
                    quota = simulation_slots.quota_snapshot(self.account)
                    if quota["blocked"]:
                        simulation_slots.quick_update(self.account, slot["id"], "refused")
                        return None, {**row, "status": "QUEUED", "detail": "daily simulation quota exhausted",
                                      "next_retry_at": quota["reset_at"]}
                    session = self._session()
                except (RequestException, machine_lib.AuthError, machine_lib.SessionBusy):
                    simulation_slots.quick_update(self.account, slot["id"], "refused")
                    session = None
                except BaseException:
                    simulation_slots.quick_update(self.account, slot["id"], "refused")
                    raise
                if session is None:
                    time.sleep(min(delay, max(0, deadline - time.monotonic())))
                    if time.monotonic() >= deadline:
                        return None, {**row, "status": "QUEUED", "detail": "waiting for authentication"}
                    delay = min(5.0, 1.5 * delay)
                    continue
                if not simulation_slots.quick_update(self.account, slot["id"], "submitting"):
                    raise RuntimeError("quick reservation no longer authorizes a POST")
                try:
                    response = session.post(machine_lib.BRAIN_API_URL + "/simulations",
                                            json=payload, timeout=90, allow_redirects=False)
                except ConnectTimeout:
                    simulation_slots.quick_update(self.account, slot["id"], "refused")
                    retry_at = time.monotonic() + delay
                except RequestException:
                    simulation_slots.quick_update(self.account, slot["id"], "unknown")
                    return None, {**row, "status": "UNKNOWN_SUBMISSION",
                                  "detail": "connection dropped before POST outcome was known"}
                else:
                    status = response.status_code
                    observation = simulation_slots.observe_quota(self.account, response.headers)
                    if status == 201:
                        try:
                            url = machine_lib._validated_progress_url(response.headers.get("Location"))
                        except machine_lib.SimulationError:
                            simulation_slots.quick_update(self.account, slot["id"], "unknown")
                            return None, {**row, "status": "UNKNOWN_SUBMISSION",
                                          "detail": "accepted POST did not return a usable Location"}
                        simulation_slots.quick_update(self.account, slot["id"], "accepted", progress_url=url)
                        return {**slot, "progress_url": url}, None
                    if status in (401, 429):
                        simulation_slots.quick_update(self.account, slot["id"], "refused")
                        daily = observation is not None and observation["remaining"] == 0
                        daily = daily or _json_object(response).get("detail") == "DAILY_SIMULATION_LIMIT_EXCEEDED"
                        if status == 429 and daily:
                            return None, {**row, "status": "QUEUED", "http": status,
                                          "detail": "daily simulation quota exhausted",
                                          "next_retry_at": simulation_slots.quota_snapshot(self.account)["reset_at"]}
                        if status == 401:
                            with suppress(machine_lib.AuthError, machine_lib.SessionBusy, cache_store.LockTimeout):
                                self._session(refresh=True)
                        retry_at = time.monotonic() + max(delay, _retry_after(response))
                    elif 400 <= status < 500 and status != 408:
                        simulation_slots.quick_update(self.account, slot["id"], "refused")
                        return None, {**row, "status": "POST_FAIL", "http": status,
                                      "detail": response.text[:180]}
                    else:
                        simulation_slots.quick_update(self.account, slot["id"], "unknown")
                        return None, {**row, "status": "UNKNOWN_SUBMISSION", "http": status}
            wait = max(delay * random.uniform(0.8, 1.2), retry_at - time.monotonic())
            if time.monotonic() + wait >= deadline:
                return None, {**row, "status": "QUEUED", "detail": "waiting for simulation capacity or retry",
                              "next_retry_at": time.time() + wait}
            time.sleep(wait)
            delay = min(5.0, 1.5 * delay)

    def run(self, arm, *, fetch_metrics=True):
        payload = _payload(arm)
        settings = payload["settings"]
        row = {"label": arm["label"], "expr": arm["expr"],
               **{name: settings.get(name) for name in ("region", "delay", "universe")}}
        key = self._key(arm["label"], payload)
        saved = self._existing(key, arm["label"])
        if saved.get("simulation_status"):
            terminal = saved
        else:
            slot, deferred = self._post(payload, key, row, existing_url=saved.get("progress_url"))
            if deferred is not None:
                return deferred
            url = machine_lib._validated_progress_url(slot["progress_url"])
            started_at = saved.get("accepted_at", time.time())
            pending = {**row, "slot_id": slot["id"], "progress_url": url,
                       "accepted_at": started_at}
            try:
                self._checkpoint(key, pending)
                receipt = slot.get("terminal_result") or self._read(
                    url, time.monotonic() + self.poll_timeout, progress=True)
                if receipt is not None:
                    terminal = {**pending, "simulation_status": receipt["status"],
                                "alpha_id": receipt.get("alpha_id"),
                                "elapsed": round(time.time() - started_at)}
                    self._checkpoint(key, terminal)
            except BaseException:
                simulation_slots.quick_update(self.account, slot["id"], "parked")
                raise
            if receipt is None:
                simulation_slots.quick_update(self.account, slot["id"], "parked")
                return {**pending, "status": "PENDING"}
        simulation_slots.quick_update(self.account, terminal["slot_id"], "terminal")
        row.update({name: value for name, value in terminal.items() if name != "account"})
        row["status"] = row["simulation_status"]
        if row["alpha_id"]:
            row["metrics_pending"] = True
            row["status"] = "PENDING"
            if fetch_metrics:
                return self.finish_metrics(row)
        return row

    def finish_metrics(self, row):
        alpha_id = row["alpha_id"]
        if not isinstance(alpha_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", alpha_id):
            raise ValueError("simulation returned an invalid Alpha ID")
        payload = self._read(f"{machine_lib.BRAIN_API_URL}/alphas/{alpha_id}",
                             time.monotonic() + self.poll_timeout)
        if payload is None:
            return {**row, "detail": "waiting for Alpha metrics"}
        metrics = payload["is"]
        row = dict(row, status=row["simulation_status"])
        row.pop("metrics_pending", None)
        row.update({name: metrics.get(name) for name in ("sharpe", "fitness", "turnover",
                                                        "returns", "drawdown", "longCount", "shortCount")})
        checks = {item.get("name"): item for item in metrics.get("checks") or []}
        row["fails"] = [name for name, item in checks.items() if item.get("result") == "FAIL"]
        for name in CHECK_NAMES:
            row[name] = {key: checks.get(name, {}).get(key) for key in ("result", "value")}
        return row


def _finished(pending):
    """Yield finished futures while allowing the caller to enqueue a next stage."""
    while pending:
        ready, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in ready:
            yield pending.pop(future), future


def _completed_arms(runner, todo, simulation_pool, metrics_pool):
    """Stream outcomes without using simulation workers to wait for Alpha metrics."""
    pending = {simulation_pool.submit(runner.run, arm, fetch_metrics=False): (arm, False) for arm in todo}
    for (arm, reading_metrics), future in _finished(pending):
        try:
            row = future.result()
        except Exception as exc:
            row = {"label": arm["label"], "expr": arm["expr"], "status": "RUNNER_ERROR",
                   "detail": f"{type(exc).__name__}: {str(exc)[:150]}"}
        else:
            if row.get("metrics_pending") and not reading_metrics:
                pending[metrics_pool.submit(runner.finish_metrics, row)] = (arm, True)
                continue
        yield arm, row


def _select_arms(arms, runner, done):
    completed, own_rows = done
    prepared, variants = {}, {}
    for arm in arms:
        payload = _payload(arm)
        job_key = runner._key(arm["label"], payload)
        if job_key in prepared:
            continue
        simulation_key = _simulation_key(payload)
        prepared[job_key] = (arm, payload["settings"], simulation_key)
        variants.setdefault(arm["label"], set()).add(simulation_key)

    resume, fresh = [], []
    seen = set(completed)
    resumed_urls = set()
    for job_key, (arm, settings, simulation_key) in prepared.items():
        label = arm["label"]
        rows = own_rows.get(label, [])
        if any(row.get("simulation_job_key") == job_key for row in rows):
            continue
        if (len(variants[label]) > 1 and label in runner.legacy_locations
                and label not in runner.saved_labels):
            raise ValueError(f"legacy Location for {label!r} has multiple possible payloads")
        saved = runner._existing(job_key, label)
        url = saved.get("progress_url")
        # Older checkpoints contained the slot identity and URL but no job key.
        if url and any(url == row.get("progress_url") for row in rows):
            continue
        if saved or job_key in runner.shared_jobs:
            if not url or url not in resumed_urls:
                resume.append(arm)
            if url:
                resumed_urls.add(url)
            seen.add(simulation_key)
        elif (len(variants[label]) != 1
              or not any(_legacy_done(row, arm, settings) for row in rows)):
            fresh.append((arm, simulation_key))

    for arm, simulation_key in fresh:
        if simulation_key not in seen:
            resume.append(arm)
            seen.add(simulation_key)
    return resume


def _run_batch(args, out):
    arms = _load_arms(Path(args.arms))
    if not arms:
        return 0
    done = load_done(out)

    with machine_lib.load_session() as session:
        account = simulation_capabilities.account_provenance(session)
    runner = Runner(out, account, args.account_slots, args.poll_timeout)
    todo = _select_arms(arms, runner, done)
    if args.limit:
        todo = todo[:args.limit]
    print(f"arms={len(arms)} todo={len(todo)} workers={args.concurrency} account_slots={args.account_slots}",
          flush=True)
    if not todo:
        return 0

    started = time.monotonic()
    deferred = 0
    try:
        with (
            _open_ledger(out) as fh,
            ThreadPoolExecutor(max_workers=args.concurrency) as simulations,
            ThreadPoolExecutor(max_workers=min(3, args.concurrency)) as metrics,
        ):
            for count, (arm, row) in enumerate(_completed_arms(runner, todo, simulations, metrics), 1):
                payload = _payload(arm)
                row["simulation_key"] = _simulation_key(payload)
                row["simulation_job_key"] = runner._key(arm["label"], payload)
                _append(fh, row)
                deferred += row["status"] in {"QUEUED", "PENDING", "UNKNOWN_SUBMISSION", "RUNNER_ERROR"}
                print(f"[{count}/{len(todo)}] {row['label'][:34]:34s} {row['status']} "
                      f"S={row.get('sharpe')} T={row.get('turnover')} fails={row.get('fails')}", flush=True)
    finally:
        runner.close()
    print(f"elapsed {time.monotonic() - started:.0f}s, unresolved {deferred}", flush=True)
    return 0


def _batch_parser(description, default_out, default_concurrency):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("arms")
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--concurrency", type=int, default=default_concurrency)
    parser.add_argument("--account-slots", type=int, default=DEFAULT_ACCOUNT_SLOTS)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def _run_locked(parser, args, run_batch):
    out = Path(args.out).resolve()
    try:
        with cache_store.file_lock(str(out) + ".run.lock", timeout=0):
            return run_batch(args, out)
    except cache_store.LockTimeout:
        parser.error("another batch already owns this output ledger")


def _batch_args(parser):
    args = parser.parse_args()
    if min(args.concurrency, args.account_slots) <= 0 or args.limit < 0:
        parser.error("workers and account slots must be positive and limit cannot be negative")
    return args


def main(*, default_out="batch-results.jsonl", default_concurrency=DEFAULT_ACCOUNT_SLOTS):
    parser = _batch_parser(__doc__, default_out, default_concurrency)
    parser.add_argument("--poll-timeout", type=float, default=420)
    args = _batch_args(parser)
    if not math.isfinite(args.poll_timeout) or args.poll_timeout <= 0:
        parser.error("poll timeout must be positive and finite")
    return _run_locked(parser, args, _run_batch)


if __name__ == "__main__":
    sys.exit(main())
