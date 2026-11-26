#!/usr/bin/env python3
"""Run Quick simulations with a shared account budget and durable progress.

Use `--concurrency` for this process's workers and `--account-slots` for
the local budget shared with other runners in the research workspace. Only
verified capacity for the authenticated account can lower that slot budget;
submission rate limits are handled separately through Retry-After.
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
from contextlib import ExitStack, contextmanager
from pathlib import Path

from requests.exceptions import ConnectTimeout, InvalidJSONError, RequestException

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cache_store
import fastexpr_ast
import alpha_registry
import machine_lib
import simulation_capabilities
import simulation_slots
from check_evidence import normalize_checks, summarize_checks
from research_app import capabilities
from tools.pipeline import batch_policy

QUICK = "QUICK"
DEFAULT_ACCOUNT_SLOTS = 5
SUCCESS_STATUSES = simulation_slots.TERMINAL_SUCCESS_STATUSES
CHECK_NAMES = ("LOW_SUB_UNIVERSE_SHARPE", "CONCENTRATED_WEIGHT",
               "LOW_2Y_SHARPE", "LOW_SHARPE", "LOW_FITNESS")
METRIC_NAMES = ("sharpe", "fitness", "turnover", "returns", "drawdown", "longCount", "shortCount", "margin")
LEGACY_SCOPE = ("region", "universe")
RECONCILIATION_STATUSES = {"UNKNOWN_SUBMISSION", "UNVERIFIED_RESULT"}
TRANSIENT_AUTH_ERRORS = (RequestException, machine_lib.AuthenticationUnavailable, machine_lib.SessionBusy)
DEFERRED_AUTH_ERRORS = (machine_lib.AuthenticationDeadline, machine_lib.RateLimited)


def _simulation_key(payload):
    """Compare source and every setting sent to the simulation API."""
    source = payload["regular"]
    if machine_lib.is_python_language(machine_lib.alpha_language(payload)):
        source = machine_lib.normalize_python_source(source)
    else:
        source = fastexpr_ast.canonical(source)
    text = json.dumps({**payload, "regular": source}, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _matches_alpha_payload(alpha, expected):
    """Bind a historical Alpha only when its executed request can be verified."""
    code, _, _ = machine_lib.regular_expression(alpha)
    recorded = alpha.get("settings")
    if alpha.get("type") != expected["type"] or not isinstance(code, str) or not isinstance(recorded, dict):
        return False
    observed = machine_lib.normalize_setting_keys(recorded)
    requested = machine_lib.normalize_setting_keys(expected["settings"])
    language = machine_lib.alpha_language(expected) or "FASTEXPR"
    if machine_lib.alpha_language(alpha) != language or not all(
        name in observed and observed[name] == value for name, value in requested.items()
    ):
        return False
    return _simulation_key({**expected, "regular": code}) == _simulation_key(expected)


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


def _ledger_account(path, row):
    if fingerprint := row.get("account_fingerprint"):
        return fingerprint
    key = row.get("simulation_job_key")
    if isinstance(key, str) and re.fullmatch(r"[a-f0-9]{64}", key):
        state, error = cache_store.load_json(path.with_suffix(path.suffix + ".state") / f"{key}.json")
        if not error and isinstance(state, dict) and state.get("alpha_id") == row.get("alpha_id"):
            return state.get("account")
    return None


def _legacy_locations(path):
    sidecar = path.with_suffix(path.suffix + ".loc")
    return {row["label"]: row["loc"] for row in _jsonl_rows(sidecar)
            if isinstance(row.get("label"), str) and isinstance(row.get("loc"), str)}


def _from_legacy_location(row, locations):
    label = row.get("label")
    location = locations.get(label) if isinstance(label, str) else None
    if not location or row.get("identity_verified") is True:
        return False
    recorded = row.get("progress_url")
    if not recorded:
        return True
    try:
        return machine_lib._validated_progress_url(recorded) == machine_lib._validated_progress_url(location)
    except machine_lib.SimulationError:
        return True


def load_done(path, account):
    """Index exact completions across sibling ledgers and local legacy rows."""
    exact, own = set(), {}
    siblings = (entry for entry in path.parent.glob("*.jsonl") if entry != path)
    for ledger in (path, *siblings):
        try:
            legacy = _legacy_locations(ledger)
            for row in _jsonl_rows(ledger):
                status = row.get("status")
                if status in simulation_slots.TERMINAL_STATUSES:
                    if _from_legacy_location(row, legacy):
                        continue
                elif ledger != path or status not in RECONCILIATION_STATUSES:
                    continue
                provenance = _ledger_account(ledger, row)
                if provenance is not None and provenance != account["fingerprint"]:
                    continue
                key = row.get("simulation_key")
                if (provenance == account["fingerprint"] and isinstance(key, str) and key
                        and row.get("status") in SUCCESS_STATUSES and row.get("alpha_id")):
                    exact.add(key)
                if (ledger == path and isinstance(row.get("label"), str)
                        and (provenance is not None or not key and not row.get("simulation_job_key"))):
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
    return (row.get("status") in SUCCESS_STATUSES and not row.get("simulation_key")
            and row.get("expr") == arm["expr"]
            and all(known.get(name) is not None and known[name] == settings.get(name)
                    for name in LEGACY_SCOPE)
            and known.get("simulationMode", QUICK) == settings["simulationMode"]
            and all(known[name] == settings.get(name) for name in keys if known.get(name) is not None))


def _finite_number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite number in batch input: {value}")
    return number


def _load_arms(path):
    text = path.read_text(encoding="utf-8")
    options = {"parse_float": _finite_number, "parse_constant": _finite_number}
    try:
        value = json.loads(text, **options)
    except json.JSONDecodeError:
        arms = [json.loads(line, **options) for line in text.splitlines() if line.strip()]
    else:
        arms = value if isinstance(value, list) else [value]
    if not all(isinstance(arm, dict) and isinstance(arm.get("label"), str)
               and isinstance(arm.get("expr", arm.get("expression")), str) and isinstance(arm.get("settings"), dict)
               for arm in arms):
        raise ValueError("arms must contain label, expr (or expression) and settings per row")
    for arm in arms:
        arm.setdefault("expr", arm.get("expression"))
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


def _record_seed(arm, row, payload, source_spec, out):
    """Reconcile a Quick observation through the shared simulation ledger."""
    source_spec = str(Path(source_spec).resolve())
    record = {"alpha_id": row.get("alpha_id"), "expression": payload["regular"],
        "settings": payload["settings"], "status": row.get("status"),
        "metrics": {name: row[name] for name in METRIC_NAMES if row.get(name) is not None},
        # Old compact columns never established that the full checks payload was
        # observed. Keep that absence explicit instead of manufacturing a pass.
        "checks": row.get("checks"), "error": row.get("error") or row.get("detail"),
        "idea": arm.get("idea"), "mechanism": arm.get("mechanism") or arm["label"],
        "family": arm.get("family"), "role": arm.get("role"), "note": arm.get("note"),
        "provenance": {"run_id": row.get("simulation_job_key"),
            "request_fingerprint": row.get("simulation_key") or _simulation_key(payload),
            "source_spec": source_spec, "arm_label": arm["label"], "result_ledger": str(out),
            "account_fingerprint": row.get("account_fingerprint"),
            "progress_url": row.get("progress_url"), "checks_payload_present": "checks" in row},
        **{name: row[name] for name in ("submission_state", "metrics_pending", "progress_url",
                                       "multi_parent_url", "multi_parent_terminal") if name in row}}
    try:
        summary = alpha_registry.record_simulation([record], payload["settings"],
            source="quick-batch", batch=source_spec)
    except ValueError as exc:
        print(f"seed not recorded for {arm['label']}: {exc}; Quick result remains in {out}", file=sys.stderr)
        return None
    if summary["missing_idea_count"]:
        print(f"seed recorded without idea for {arm['label']}; add idea to its original arm "
              "and replay this batch to label it", file=sys.stderr)
    for skipped in summary["skipped"]:
        print(f"seed not recorded for {arm['label']}: {skipped['reason']}; "
              f"Quick result remains in {out}", file=sys.stderr)
    return summary


class Runner:
    def __init__(self, out: Path, account: dict, slots: int, poll_timeout: float, *, admission=None):
        if not account.get("fingerprint"):
            raise ValueError("quick runner needs an authenticated account for shared slots")
        self.out = out.resolve()
        self.account = account
        self.slots = slots
        self.poll_timeout = poll_timeout
        # Bulk-exploration admission belongs to the calling workflow. The same
        # transport also recovers historical work and serves FULL validation.
        self.admission = admission
        self.state_path = self.out.with_suffix(self.out.suffix + ".state")
        self.state_lock = threading.Lock()
        self.jobs = {}
        self.saved_labels = set()
        self.shared_job_evidence = simulation_slots.quick_job_snapshot(self.account)
        self.shared_jobs = set(self.shared_job_evidence)
        for path in self.state_path.glob("*.json"):
            saved, error = cache_store.load_json(path)
            if error or not isinstance(saved, dict) or saved.get("account") != account["fingerprint"]:
                raise ValueError("quick runner checkpoint is unreadable or belongs to another account")
            self.jobs[path.stem] = saved
            self.saved_labels.add(saved["label"])
        self.legacy_locations = _legacy_locations(self.out)
        self.local = threading.local()
        self.sessions = []
        self.sessions_lock = threading.Lock()
        self.reap_lock = threading.Lock()
        self.next_reap = 0.0
        self.reap_offset = 0
        self.stop_requested = threading.Event()
        self.quota_exhausted = threading.Event()
        self.post_retry_lock = threading.Lock()
        retry_at = max((job.get("next_retry_at") or 0 for job in self.jobs.values()), default=0)
        self.post_retry_at = time.monotonic() + max(0, retry_at - time.time())

    def _session(self, refresh=False, *, deadline=None):
        previous = getattr(self.local, "session", None)
        if previous is not None and not refresh:
            return previous

        meta = {}
        rejected = getattr(self.local, "generation", None) if refresh else None
        session = machine_lib.load_session(refresh=refresh, meta=meta, deadline=deadline,
                                           rejected_generation=rejected)
        if simulation_capabilities.account_provenance(session)["fingerprint"] != self.account["fingerprint"]:
            session.close()
            raise ValueError("authenticated account changed while running simulations")
        self.local.session = session
        self.local.generation = meta.get("generation")
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
        return self._raw_key(label, _simulation_key(payload))

    def _raw_key(self, label, payload):
        data = json.dumps([str(self.out), label, payload], sort_keys=True, ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(data.encode()).hexdigest()

    def _job_key(self, label, payload):
        key = self._key(label, payload)
        raw = self._raw_key(label, payload)
        with self.state_lock:
            if key in self.jobs or key in self.shared_jobs:
                return key
            if raw in self.jobs or raw in self.shared_jobs:
                return raw
            if label not in self.saved_labels:
                return key
            for prior_key, row in self.jobs.items():
                prior_source = row.get("expr")
                if row.get("label") != label or not isinstance(prior_source, str):
                    continue
                prior = {**payload, "regular": prior_source}
                if prior_key == self._raw_key(label, prior) and key == self._key(label, prior):
                    return prior_key
        return key

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
        if saved:
            if saved.get("progress_url") and _from_legacy_location(saved, self.legacy_locations):
                return {**saved, "legacy_unverified": True}
            return saved
        if reused_label or label not in self.legacy_locations:
            return saved
        return {"progress_url": machine_lib._validated_progress_url(self.legacy_locations[label]),
                "legacy_unverified": True}

    def needs_new_submission(self, arm):
        """Queued preparations need admission; accepted/ambiguous work does not."""
        key = self._job_key(arm["label"], _payload(arm))
        saved = self._existing(key, arm["label"])
        shared = self.shared_job_evidence.get(key, {})
        return not (saved.get("progress_url") or saved.get("simulation_status")
                    or saved.get("status") in RECONCILIATION_STATUSES
                    or shared.get("progress_url") or shared.get("state") in {"submitting", "unknown"})

    def _reap(self, key, deadline):
        if not self.reap_lock.acquire(blocking=False):
            return
        try:
            now = time.monotonic()
            if now < self.next_reap or now >= deadline:
                return
            self.next_reap = now + 4
            if not simulation_slots.quick_orphans(self.account, exclude_job=key):
                return
            deadline = min(deadline, now + machine_lib.QUICK_ORPHAN_SWEEP_SECONDS)
            offset = self.reap_offset
            self.reap_offset += 1
            for refresh in (False, True):
                session = self._session(refresh=refresh, deadline=deadline)
                try:
                    machine_lib.recover_quick_orphans(session, self.account, deadline=deadline,
                                                     exclude_job=key, limit=1, offset=offset)
                except machine_lib.AuthError:
                    continue
                break
        except (RequestException, machine_lib.BrainError, cache_store.LockTimeout):
            return
        finally:
            self.reap_lock.release()

    def _wait(self, seconds):
        until = time.monotonic() + seconds
        while not self.stop_requested.is_set() and (remaining := until - time.monotonic()) > 0:
            time.sleep(min(1, remaining))

    def _read(self, url, deadline, *, simulation_id=None, next_poll_at=0, defer=None):
        delay = 0.5
        previous_progress = None
        refresh = False
        initial_wait = delay if simulation_id else 0
        wait = max(0, next_poll_at - time.time()) if next_poll_at else initial_wait
        while not self.stop_requested.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0 or wait >= remaining:
                return None
            if wait:
                self._wait(wait)
            if time.monotonic() >= deadline or self.stop_requested.is_set():
                return None
            try:
                session = self._session(refresh=refresh, deadline=deadline)
                refresh = False
                if time.monotonic() >= deadline or self.stop_requested.is_set():
                    return None
                response = session.get(url, timeout=machine_lib._request_timeout(deadline), allow_redirects=False)
            except DEFERRED_AUTH_ERRORS:
                return None
            except TRANSIENT_AUTH_ERRORS:
                wait = delay = min(4.0, 1.5 * delay)
                continue
            body = _json_object(response)
            retry_after = _retry_after(response)
            if retry_after > 0 and defer is not None:
                defer(time.time() + retry_after)
            if response.status_code in (401, 403):
                refresh = True
            elif not retry_after:
                if simulation_id:
                    if receipt := machine_lib.simulation_progress_receipt(response, body, simulation_id):
                        return receipt
                elif response.status_code == 200 and body.get("is"):
                    return body
            if simulation_id:
                delay, previous_progress = _poll_interval(body, delay, previous_progress)
            else:
                delay = min(4.0, 1.5 * delay)
            wait = retry_after or delay

    def _post(self, payload, key, row, *, existing_url, next_retry_at=0, identity_unverified=False):
        # Old checkpoints may be keyed by the exact source from the arm file.
        wire_payload = machine_lib.format_simulation_payload(payload)
        delay = 0.5
        deadline = time.monotonic() + self.poll_timeout
        retry_at = time.monotonic() + max(0, next_retry_at - time.time())
        refresh = False
        while not self.stop_requested.is_set():
            admitted = self.admission is None or self.admission["ok"]
            if admitted:
                self._reap(key, deadline)
            if time.monotonic() >= deadline:
                return None, {**row, "status": "QUEUED", "detail": "simulation POST deferred"}
            # A zero reservation budget still claims accepted/unknown jobs, but
            # never creates a new POST or keeps an unposted preparation alive.
            slot = simulation_slots.quick_acquire(self.account, self.slots if admitted else 0, key,
                                                  existing_url=existing_url, prepare=True,
                                                  identity_unverified=identity_unverified)
            if not admitted and slot is None:
                return None, {**row, "status": "BLOCKED", "new_post_allowed": False,
                              "detail": "batch admission blocks new simulations; recovery remains available",
                              "admission_path": self.admission["path"],
                              "admission_codes": [item["code"] for item in self.admission["blocking"]]}
            if slot is None and self.slots == 0:
                return None, {**row, "status": "QUEUED", "detail": "account slot budget is zero"}
            if slot is not None:
                if slot.get("progress_url"):
                    return slot, None
                if slot["state"] == "unknown":
                    return None, {**row, "status": "UNKNOWN_SUBMISSION",
                                  "detail": "previous POST outcome is unknown"}
                if not admitted:
                    raise RuntimeError("blocked admission received a new POST reservation")
                try:
                    quota = simulation_slots.quota_snapshot(self.account)
                    if self.quota_exhausted.is_set() or quota["blocked"]:
                        simulation_slots.quick_update(self.account, slot["id"], "refused")
                        return None, {**row, "status": "QUEUED", "detail": "daily simulation quota exhausted",
                                      "next_retry_at": quota["reset_at"]}
                    with self.post_retry_lock:
                        retry_at = max(retry_at, self.post_retry_at)
                    shared_retry_at = simulation_slots.submission_retry_at(self.account)
                    if shared_retry_at is not None:
                        retry_at = max(retry_at, time.monotonic() + shared_retry_at - time.time())
                    if retry_at > time.monotonic():
                        simulation_slots.quick_update(self.account, slot["id"], "refused")
                        remaining = retry_at - time.monotonic()
                        if time.monotonic() + remaining >= deadline:
                            return None, {**row, "status": "QUEUED", "detail": "waiting for simulation retry",
                                          "next_retry_at": time.time() + max(0, remaining)}
                        self._wait(remaining)
                        continue
                    session = self._session(refresh=refresh, deadline=deadline)
                    refresh = False
                except DEFERRED_AUTH_ERRORS as exc:
                    simulation_slots.quick_update(self.account, slot["id"], "refused")
                    return None, {**row, "status": "QUEUED", "detail": str(exc)}
                except TRANSIENT_AUTH_ERRORS:
                    simulation_slots.quick_update(self.account, slot["id"], "refused")
                    session = None
                except BaseException:
                    simulation_slots.quick_update(self.account, slot["id"], "refused")
                    raise
                if session is None:
                    self._wait(min(delay, max(0, deadline - time.monotonic())))
                    if time.monotonic() >= deadline:
                        return None, {**row, "status": "QUEUED", "detail": "waiting for authentication"}
                    delay = min(5.0, 1.5 * delay)
                    continue
                if not simulation_slots.quick_update(self.account, slot["id"], "submitting"):
                    raise RuntimeError("quick reservation no longer authorizes a POST")
                try:
                    if self.stop_requested.is_set() or time.monotonic() >= deadline:
                        simulation_slots.quick_update(self.account, slot["id"], "refused")
                        return None, {**row, "status": "QUEUED", "detail": "simulation POST deferred"}
                    if self.quota_exhausted.is_set():
                        simulation_slots.quick_update(self.account, slot["id"], "refused")
                        return None, {**row, "status": "QUEUED", "detail": "daily simulation quota exhausted",
                                      "next_retry_at": simulation_slots.quota_snapshot(self.account)["reset_at"]}
                    if simulation_slots.submission_retry_at(self.account) is not None:
                        simulation_slots.quick_update(self.account, slot["id"], "refused")
                        continue
                    # Once dispatched, give the POST a full chance to return its Location.
                    response = session.post(machine_lib.BRAIN_API_URL + "/simulations",
                                            json=wire_payload, timeout=90, allow_redirects=False)
                except ConnectTimeout:
                    simulation_slots.quick_update(self.account, slot["id"], "refused")
                    retry_at = time.monotonic() + delay
                except InvalidJSONError:
                    simulation_slots.quick_update(self.account, slot["id"], "refused")
                    return None, {**row, "status": "POST_FAIL", "detail": "simulation payload is not valid JSON"}
                except RequestException:
                    simulation_slots.quick_update(self.account, slot["id"], "unknown")
                    return None, {**row, "status": "UNKNOWN_SUBMISSION",
                                  "detail": "connection dropped before POST outcome was known"}
                else:
                    status = response.status_code
                    if status == 201:
                        try:
                            url, _ = machine_lib._quick_progress_target(response.headers.get("Location"))
                        except machine_lib.SimulationError:
                            simulation_slots.quick_update(self.account, slot["id"], "unknown")
                            self._observe_quota(response)
                            return None, {**row, "status": "UNKNOWN_SUBMISSION",
                                          "detail": "accepted POST did not return a usable Location"}
                        simulation_slots.quick_update(self.account, slot["id"], "accepted", progress_url=url)
                        self._observe_quota(response)
                        return {**slot, "progress_url": url}, None
                    if status in (401, 403, 429):
                        simulation_slots.quick_update(self.account, slot["id"], "refused")
                        self._observe_quota(response)
                        retry_after = _retry_after(response)
                        if status == 429:
                            if self.quota_exhausted.is_set():
                                return None, {**row, "status": "QUEUED", "http": status,
                                              "detail": "daily simulation quota exhausted",
                                              "next_retry_at": simulation_slots.quota_snapshot(self.account)["reset_at"]}
                            retry_at = self._defer_post(key, row, max(delay, retry_after),
                                                       shared=True, deadline=deadline)
                        else:
                            if retry_after > 0:
                                retry_at = self._defer_post(key, row, retry_after)
                            refresh = True
                        retry_at = max(retry_at, time.monotonic() + delay)
                    elif 400 <= status < 500 and status != 408:
                        simulation_slots.quick_update(self.account, slot["id"], "refused")
                        self._observe_quota(response)
                        return None, {**row, "status": "POST_FAIL", "http": status,
                                      "detail": response.text[:180]}
                    else:
                        simulation_slots.quick_update(self.account, slot["id"], "unknown")
                        self._observe_quota(response)
                        return None, {**row, "status": "UNKNOWN_SUBMISSION", "http": status}
            wait = max(delay * random.uniform(0.8, 1.2), retry_at - time.monotonic())
            if time.monotonic() + wait >= deadline:
                return None, {**row, "status": "QUEUED", "detail": "waiting for simulation capacity or retry",
                              "next_retry_at": time.time() + wait}
            self._wait(wait)
            delay = min(5.0, 1.5 * delay)
        return None, {**row, "status": "QUEUED", "detail": "batch interrupted"}

    def _observe_quota(self, response):
        if response.status_code in (401, 403):
            return
        observation = simulation_slots.observe_quota(self.account, response.headers)
        if ((observation and observation["remaining"] == 0)
                or (response.status_code == 429
                    and _json_object(response).get("detail") == "DAILY_SIMULATION_LIMIT_EXCEEDED")):
            self.quota_exhausted.set()

    def _defer_post(self, key, row, delay, *, shared=False, deadline=None):
        retry_time = time.time() + delay
        if shared:
            retry_time = simulation_slots.defer_submission(self.account, delay, deadline=deadline) or retry_time
        retry_at = time.monotonic() + max(0, retry_time - time.time())
        with self.post_retry_lock:
            self.post_retry_at = max(self.post_retry_at, retry_at)
        self._checkpoint(key, {**row, "next_retry_at": retry_time})
        return retry_at

    def run(self, arm, *, fetch_metrics=True):
        payload = _payload(arm)
        settings = payload["settings"]
        key = self._job_key(arm["label"], payload)
        row = {"label": arm["label"], "expr": arm["expr"], "simulation_job_key": key,
               **{name: settings.get(name) for name in ("region", "delay", "universe")}}
        saved = self._existing(key, arm["label"])
        if (saved.get("status") in RECONCILIATION_STATUSES
                and not saved.get("progress_url") and key not in self.shared_jobs):
            # The append-only result may outlive a damaged/missing checkpoint.
            # Missing local files cannot prove that the earlier POST was unsent.
            return {**saved, **row, "status": saved["status"],
                    "recovery_hint": "find the original Location; no new POST is authorized"}
        if saved.get("status") in RECONCILIATION_STATUSES and saved.get("legacy_unverified"):
            # A compact unverified row is not a terminal identity receipt. Read
            # its original operation and validate the Alpha's executed payload.
            saved = {name: value for name, value in saved.items()
                     if name not in {"simulation_status", "alpha_id", "metrics_pending"}}
        legacy_payload = saved.get("legacy_payload")
        if legacy_payload is None and saved.get("legacy_unverified"):
            legacy_payload = payload
        awaiting_manual_confirmation = (saved.get("manual_location") or legacy_payload is not None) and not saved.get("alpha_id")
        if saved.get("simulation_status") and not awaiting_manual_confirmation:
            terminal = saved if legacy_payload is None else {**saved, "legacy_payload": legacy_payload}
        else:
            slot, deferred = self._post(payload, key, row, existing_url=saved.get("progress_url"),
                                        next_retry_at=saved.get("next_retry_at") or 0,
                                        identity_unverified=legacy_payload is not None)
            if deferred is not None:
                return deferred
            try:
                url, simulation_id = machine_lib._quick_progress_target(slot["progress_url"])
            except machine_lib.SimulationError as exc:
                simulation_slots.quick_update(self.account, slot["id"], "parked")
                return {**row, "status": "UNVERIFIED_RESULT", "detail": str(exc)}
            started_at = saved.get("accepted_at", time.time())
            pending = {**row, "slot_id": slot["id"], "progress_url": url,
                       "accepted_at": started_at}
            if slot.get("manual_failure_confirmed"):
                pending["identity_verified"] = True
            elif legacy_payload is not None or slot.get("identity_unverified"):
                pending["legacy_payload"] = legacy_payload if legacy_payload is not None else payload
            if slot.get("manual_location"):
                pending["manual_location"] = True
            next_poll_at = max(saved.get("next_poll_at") or 0, slot.get("next_poll_at") or 0)
            if next_poll_at:
                pending["next_poll_at"] = next_poll_at

            def defer(when):
                simulation_slots.quick_defer(self.account, slot["id"], when)
                pending["next_poll_at"] = when
                self._checkpoint(key, pending)

            try:
                receipt = slot.get("terminal_result")
                if receipt is None and awaiting_manual_confirmation and saved.get("simulation_status"):
                    receipt = {"status": saved["simulation_status"], "alpha_id": None}
                if receipt is None:
                    self._checkpoint(key, pending)
                    receipt = self._read(url, time.monotonic() + self.poll_timeout,
                                         simulation_id=simulation_id, next_poll_at=next_poll_at, defer=defer)
                if receipt is not None:
                    terminal = {**pending, "simulation_status": receipt["status"],
                                "alpha_id": receipt.get("alpha_id"),
                                "elapsed": round(time.time() - started_at)}
                    terminal.pop("next_poll_at", None)
                    self._checkpoint(key, terminal)
            except BaseException:
                simulation_slots.quick_update(self.account, slot["id"], "parked")
                raise
            if receipt is None:
                simulation_slots.quick_update(self.account, slot["id"], "parked")
                return {name: value for name, value in {**pending, "status": "PENDING"}.items()
                        if name != "legacy_payload"}
        outcome = "parked" if (terminal.get("manual_location") or terminal.get("legacy_payload") is not None) else "terminal"
        simulation_slots.quick_update(self.account, terminal["slot_id"], outcome)
        row.update({name: value for name, value in terminal.items() if name != "account"})
        row["status"] = row["simulation_status"]
        if row["alpha_id"]:
            row["metrics_pending"] = True
            row["status"] = "PENDING"
            if fetch_metrics:
                return self.finish_metrics(row)
        elif row.pop("legacy_payload", None) is not None:
            row.update(status="UNVERIFIED_RESULT", observed_alpha_id=None,
                       detail="verify the original Location and use --confirm-original-failure to release its slot")
        return row

    def finish_metrics(self, row):
        alpha_id = row["alpha_id"]
        if not isinstance(alpha_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", alpha_id):
            raise ValueError("simulation returned an invalid Alpha ID")
        row = dict(row)

        def defer(when):
            row["next_metrics_at"] = when
            self._checkpoint(row["simulation_job_key"], row)

        payload = self._read(f"{machine_lib.BRAIN_API_URL}/alphas/{alpha_id}",
                             time.monotonic() + self.poll_timeout,
                             next_poll_at=row.get("next_metrics_at") or 0, defer=defer)
        legacy_payload = row.pop("legacy_payload", None)
        if payload is None:
            if legacy_payload is not None:
                row.update(alpha_id=None, observed_alpha_id=alpha_id)
            return {**row, "detail": "waiting for Alpha metrics"}
        if legacy_payload is not None and (payload.get("id", alpha_id) != alpha_id
                                           or not _matches_alpha_payload(payload, legacy_payload)):
            row.pop("metrics_pending", None)
            return {**row, "status": "UNVERIFIED_RESULT", "alpha_id": None, "observed_alpha_id": alpha_id,
                    "detail": "BRAIN did not confirm the legacy simulation's expression and settings"}
        if row.pop("manual_location", False) or legacy_payload is not None:
            simulation_slots.quick_update(self.account, row["slot_id"], "terminal")
        metrics = payload["is"]
        row = dict(row, status=row["simulation_status"])
        if legacy_payload is not None:
            row["identity_verified"] = True
        row.pop("metrics_pending", None)
        row.pop("next_metrics_at", None)
        row.update({name: metrics.get(name) for name in METRIC_NAMES})
        checks = normalize_checks(metrics.get("checks"))
        row["checks"] = checks
        row["fails"] = summarize_checks(checks)["failed"]
        for name in CHECK_NAMES:
            row[name] = {key: checks.get(name, {}).get(key) for key in ("result", "value")}
        self._checkpoint(row["simulation_job_key"], row)
        return row


def _finished(pending):
    """Yield finished futures while allowing the caller to enqueue a next stage."""
    while pending:
        ready, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in ready:
            yield pending.pop(future), future


def _completed_arms(runner, todo, simulation_pool, metrics_pool, *, workers=1):
    """Stream outcomes without using simulation workers to wait for Alpha metrics."""
    arms = iter(todo)
    pending = {}

    def submit_next():
        arm = next(arms, None)
        if arm is not None:
            pending[simulation_pool.submit(runner.run, arm, fetch_metrics=False)] = (arm, False)

    try:
        for _ in range(workers):
            submit_next()
        for (arm, reading_metrics), future in _finished(pending):
            try:
                row = future.result()
            except Exception as exc:
                row = {"label": arm["label"], "expr": arm["expr"], "status": "RUNNER_ERROR",
                       "detail": f"{type(exc).__name__}: {str(exc)[:150]}"}
            if row.get("metrics_pending") and not reading_metrics:
                pending[metrics_pool.submit(runner.finish_metrics, row)] = (arm, True)
            else:
                yield arm, row
            if not reading_metrics:
                submit_next()
    finally:
        for future in pending:
            future.cancel()


def _select_arms(arms, runner, done):
    completed, own_rows = done
    prepared, variants = {}, {}
    for arm in arms:
        payload = _payload(arm)
        job_key = runner._job_key(arm["label"], payload)
        saved = runner._existing(job_key, arm["label"])
        authoritative = (saved.get("progress_url") or saved.get("simulation_status")
                         or saved.get("status") in RECONCILIATION_STATUSES)
        if not authoritative and job_key not in runner.shared_jobs:
            # Older runners hashed the complete raw payload. A matching hash
            # still binds all settings, source, label, account and output path.
            candidates = {job_key, runner._raw_key(arm["label"], payload)}

            def matches_prior(row):
                if row.get("simulation_job_key") in candidates:
                    return True
                source = row.get("expr")
                if not isinstance(source, str):
                    return False
                prior_payload = {**payload, "regular": source}
                try:
                    return (row.get("simulation_job_key") == runner._raw_key(arm["label"], prior_payload)
                            and runner._key(arm["label"], prior_payload) == runner._key(arm["label"], payload))
                except ValueError:
                    return False

            prior = next((row for row in reversed(own_rows.get(arm["label"], []))
                          if row.get("account_fingerprint") == runner.account["fingerprint"]
                          and row.get("status") in RECONCILIATION_STATUSES and matches_prior(row)), None)
            if prior is not None:
                with runner.state_lock:
                    runner.jobs[prior["simulation_job_key"]] = {**prior, "legacy_unverified": True}
                    runner.saved_labels.add(arm["label"])
                job_key = runner._job_key(arm["label"], payload)
        if job_key in prepared:
            continue
        simulation_key = _simulation_key(payload)
        prepared[job_key] = (arm, payload["settings"], simulation_key)
        variants.setdefault(arm["label"], set()).add(simulation_key)

    resume, fresh, reconciliation = [], [], []
    seen = set(completed)
    resumed_urls = set()
    for job_key, (arm, settings, simulation_key) in prepared.items():
        label = arm["label"]
        rows = own_rows.get(label, [])
        if any(row.get("simulation_job_key") == job_key and row.get("status") in simulation_slots.TERMINAL_STATUSES
               for row in rows):
            continue
        if (len(variants[label]) > 1 and label in runner.legacy_locations
                and label not in runner.saved_labels):
            raise ValueError(f"legacy Location for {label!r} has multiple possible payloads")
        saved = runner._existing(job_key, label)
        url = saved.get("progress_url")
        # Older checkpoints contained the slot identity and URL but no job key.
        if url and any(url == row.get("progress_url") and row.get("status") in SUCCESS_STATUSES
                       and row.get("alpha_id") for row in rows):
            continue
        if saved or job_key in runner.shared_jobs:
            if not url or url not in resumed_urls:
                previous = next((row for row in reversed(rows) if row.get("simulation_job_key") == job_key), {})
                pending_reconciliation = previous.get("status") in RECONCILIATION_STATUSES and not (
                    previous.get("status") == "UNKNOWN_SUBMISSION" and url)
                (reconciliation if pending_reconciliation else resume).append(arm)
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
    return resume + reconciliation


def _account_slot_plan(session, requested):
    """Share the native runner's evidence policy without discovering online limits."""
    plan = machine_lib.concurrency_plan(session, requested=requested)
    if plan["effective"] < requested:
        print(f"note: account slot budget reduced from {requested} to {plan['effective']} "
              "by verified capacity for the authenticated account", file=sys.stderr)
    return plan


def _build_admission(arms, session, account, *, recover_only=False):
    """Assess the complete exploration batch once, with one catalog budget.

    Only positive, account/scope-bound evidence contributes dataset IDs. Keep
    catalog uncertainty separate from a proven shortfall and from expression
    errors; none of these authorizes repeating an accepted or ambiguous POST.
    """
    analysis = batch_policy.analyze_arms(arms)
    evidence = []
    if analysis["ok"] and not recover_only:
        groups = {}
        for arm in analysis["arms"]:
            key = tuple(sorted(arm["scope"].items()))
            group = groups.setdefault(key, {"scope": arm["scope"], "fields": set(), "indices": []})
            group["fields"].update(arm["fields"])
            group["indices"].append(arm["index"])
        records_by_arm = [[] for _ in arms]
        deadline = time.monotonic() + capabilities._REQUIRED_FIELDS_WAIT_SECONDS
        proof_cache = {}
        for group in groups.values():
            try:
                scope = capabilities.cached_field_scope(group["scope"], account=account)
                if scope["evidence"]["state"] == "verified":
                    records = scope["records"]
                else:
                    scope = capabilities.required_field_scope(
                        session, group["scope"], group["fields"], deadline=deadline, proof_cache=proof_cache)
                    records = scope["verified_records"]
                observed = scope["evidence"]
            except (machine_lib.BrainError, RequestException) as exc:
                records = []
                observed = {"state": "unknown", "scope": group["scope"],
                            "error": getattr(exc, "code", type(exc).__name__), "message": str(exc)}
            evidence.append(observed)
            for index in group["indices"]:
                records_by_arm[index] = records
            # One rate limit also defers live reads for the remaining scopes;
            # those scopes may still reuse their already verified cache entries.
            if (observed.get("error") == "RATE_LIMITED" or any(
                    row.get("error") == "RATE_LIMITED" for row in observed.get("catalogs", []))):
                deadline = time.monotonic()
        analysis = batch_policy.assess_datasets(analysis, records_by_arm)
    elif analysis["ok"]:
        analysis = {**analysis, "ok": False, "state": "recovery_only",
                    "blocking": [{"code": "recovery_only",
                                  "message": "existing tasks need recovery, not admission for a new POST"}]}
    retry_times = [row["retry_at"] for scope in evidence for row in scope.get("catalogs", [])
                   if isinstance(row.get("retry_at"), (int, float))]
    return {**analysis, "new_posts_allowed": analysis["ok"], "recovery_allowed": True,
            "field_evidence": evidence, "next_retry_at": max(retry_times, default=None)}


def _write_admission(out, input_path, admission, account):
    path = out.with_suffix(out.suffix + ".admission.json")
    admission.update(path=str(path), input_path=str(Path(input_path).resolve()),
                     account_fingerprint=account["fingerprint"])
    cache_store.write_json_atomic(path, admission)
    print(json.dumps({"event": "batch_admission", "path": str(path),
                      "new_posts_allowed": admission["ok"], "recovery_allowed": True,
                      "blocking": admission["blocking"],
                      "next_retry_at": admission["next_retry_at"]}, ensure_ascii=False), file=sys.stderr)


def _run_batch(args, out):
    arms = _load_arms(Path(args.arms))
    if getattr(args, "admission_only", False):
        with machine_lib.load_session() as session:
            account = _account_slot_plan(session, args.account_slots)["account"]
            admission = _build_admission(arms, session, account)
            _write_admission(out, args.arms, admission, account)
        return 0 if admission["ok"] else 2
    if not arms:
        return 0
    recover_job = getattr(args, "recover_job", None)
    limits = {"deadline": time.monotonic() + args.poll_timeout} if recover_job else {}
    with machine_lib.load_session(**limits) as session:
        plan = _account_slot_plan(session, args.account_slots)
        account = plan["account"]
        runner = Runner(out, account, plan["effective"], args.poll_timeout)
        if recover_job:
            if not any(runner._job_key(arm["label"], _payload(arm)) == recover_job for arm in arms):
                raise ValueError("recovery job key does not match an arm in the original batch")
            confirm_failure = getattr(args, "confirm_original_failure", False)
            machine_lib.attach_quick_location(session, account, recover_job, args.recover_location,
                                              confirm_failure=confirm_failure, **limits)
            message = ("terminal failure confirmed and slot released" if confirm_failure
                       else "recovery Location recorded")
            print(f"{message} for {recover_job}", flush=True)
            return 0
        done = load_done(out, account)
        # Repair a durable result whose ledger append was interrupted, even when
        # its historical input lacks metadata required for new exploration.
        for arm in arms:
            payload = _payload(arm)
            job_key = runner._job_key(arm["label"], payload)
            previous = next((row for row in reversed(done[1].get(arm["label"], []))
                             if row.get("simulation_job_key") == job_key
                             and _ledger_account(out, row) == account["fingerprint"]), None)
            if previous is not None:
                _record_seed(arm, previous, payload, args.arms, out)
        todo = _select_arms(arms, runner, done)
        recover_only = getattr(args, "recover_only", False)
        if recover_only:
            todo = [arm for arm in todo if not runner.needs_new_submission(arm)]
        if args.limit:
            todo = todo[:args.limit]
        if todo:
            recover_only = recover_only or not any(runner.needs_new_submission(arm) for arm in todo)
            admission = _build_admission(arms, session, account, recover_only=recover_only)
            _write_admission(out, args.arms, admission, account)
            runner.admission = admission
    print(f"arms={len(arms)} todo={len(todo)} workers={args.concurrency} account_slots={plan['effective']}",
          flush=True)
    if not todo:
        return 0

    started = time.monotonic()
    deferred = blocked = 0
    try:
        with (
            _open_ledger(out) as fh,
            ThreadPoolExecutor(max_workers=args.concurrency) as simulations,
            ThreadPoolExecutor(max_workers=min(3, args.concurrency)) as metrics,
        ):
            rows = _completed_arms(runner, todo, simulations, metrics, workers=args.concurrency)
            try:
                for count, (arm, row) in enumerate(rows, 1):
                    payload = _payload(arm)
                    row["account_fingerprint"] = account["fingerprint"]
                    row["simulation_key"] = _simulation_key(payload)
                    row["simulation_job_key"] = runner._job_key(arm["label"], payload)
                    _append(fh, row)
                    if row["status"] != "BLOCKED":
                        _record_seed(arm, row, payload, args.arms, out)
                    blocked += row["status"] == "BLOCKED"
                    deferred += row["status"] in {"QUEUED", "PENDING", "UNKNOWN_SUBMISSION", "UNVERIFIED_RESULT", "RUNNER_ERROR"}
                    print(f"[{count}/{len(todo)}] {row['label'][:34]:34s} {row['status']} "
                          f"S={row.get('sharpe')} T={row.get('turnover')} fails={row.get('fails')}", flush=True)
            except BaseException:
                runner.stop_requested.set()
                rows.close()
                raise
    finally:
        runner.close()
    print(f"elapsed {time.monotonic() - started:.0f}s, unresolved {deferred}, admission_blocked {blocked}", flush=True)
    return 2 if blocked else 0


def _batch_parser(description, default_out, default_concurrency):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("arms")
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--concurrency", type=int, default=default_concurrency,
                        help="Number of workers in this process (default: %(default)s).")
    parser.add_argument("--account-slots", type=int, default=DEFAULT_ACCOUNT_SLOTS,
                        help="Local shared slot budget, bounded by verified account capacity "
                             "when known (default: %(default)s).")
    parser.add_argument("--limit", type=int, default=0, help="Maximum arms to process (0 means unlimited).")
    return parser


def _run_locked(parser, args, run_batch):
    out = Path(args.out).resolve()
    with ExitStack() as stack:
        try:
            stack.enter_context(cache_store.file_lock(str(out) + ".run.lock", timeout=0))
        except cache_store.LockTimeout:
            return parser.error("another batch already owns this output ledger")
        return run_batch(args, out)


def _batch_args(parser):
    args = parser.parse_args()
    if min(args.concurrency, args.account_slots) <= 0 or args.limit < 0:
        parser.error("workers and account slots must be positive and limit cannot be negative")
    args.concurrency = simulation_slots.quick_single_slot_limit(
        simulation_slots.quick_slot_limit(args.concurrency),
    )
    return args


def main(*, default_out="batch-results.jsonl", default_concurrency=DEFAULT_ACCOUNT_SLOTS):
    parser = _batch_parser(__doc__, default_out, default_concurrency)
    parser.add_argument("--poll-timeout", type=float, default=420,
                        help="Seconds per stage for queueing, progress, and metrics (default: %(default)s). "
                             "An in-flight POST uses its own network timeout.")
    parser.add_argument("--recover-job", help="Job key from a result with an uncertain POST outcome.")
    parser.add_argument("--admission-only", action="store_true",
                        help="Write the full Quick admission receipt without submitting or recovering tasks.")
    parser.add_argument("--recover-only", action="store_true",
                        help="Recover existing tasks and record their results without submitting queued arms.")
    parser.add_argument("--recover-location", help="Original BRAIN progress Location for that POST.")
    parser.add_argument("--confirm-original-failure", action="store_true",
                        help="Confirm the supplied Location is original and release its slot after a BRAIN terminal failure.")
    args = _batch_args(parser)
    if not math.isfinite(args.poll_timeout) or args.poll_timeout <= 0:
        parser.error("poll timeout must be positive and finite")
    if bool(args.recover_job) != bool(args.recover_location):
        parser.error("--recover-job and --recover-location must be used together")
    if args.confirm_original_failure and not args.recover_job:
        parser.error("--confirm-original-failure requires --recover-job and --recover-location")
    if args.admission_only and args.recover_job:
        parser.error("--admission-only and --recover-job are mutually exclusive")
    if args.recover_only and (args.admission_only or args.recover_job):
        parser.error("--recover-only cannot be combined with --admission-only or --recover-job")
    return _run_locked(parser, args, _run_batch)


if __name__ == "__main__":
    sys.exit(main())
