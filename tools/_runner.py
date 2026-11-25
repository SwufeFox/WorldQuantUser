"""Shared CLI and saved-batch handling for the background research tools."""

from __future__ import annotations

import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import alpha_registry
import cache_store
import research_paths
import simulation_slots

REPO = Path(__file__).resolve().parents[1]
_FINISHED = alpha_registry.FINISHED_STATUSES | {"ACTIVE", "SUBMITTED", "UNSUBMITTED"}
_FAILED = simulation_slots.TERMINAL_FAILURE_STATUSES | {"FAILED"}
_RETRYABLE = {"RATE_LIMITED", "SERVICE_ERROR", "AUTH_MISSING", "AUTH_FAILED", "AUTH_CAPTCHA", "INTERRUPTED", "TIMEOUT",
              "INVALID_RESPONSE"}
_OFFLINE_BLOCKS = {"empty", "unbalanced_string", "unbalanced_parens", "reserved_identifier",
                   "trailing_token", "unused_variable", "operator_budget", "python_syntax",
                   "python_alpha_entrypoint", "python_alpha_signature", "python_data_declaration",
                   "python_implicit_field"}


@dataclass(frozen=True)
class CliResult:
    ok: bool
    data: dict
    detail: str = ""
    code: str = ""


@dataclass(frozen=True)
class ProbeRun:
    state: str
    data: dict
    detail: str = ""


@dataclass(frozen=True)
class _ProbeSchedule:
    priority: int = 0
    poll_at: float | None = None
    recovery: bool = False
    occupied: frozenset[tuple[str, str]] = frozenset()
    queued: frozenset[tuple[str, str]] | None = None
    native_admission_scope: tuple[str, int, int] | None = None
    scheduling_events: int = 0


def research_root(explicit=None):
    """Resolve the CLI root while honoring the older runner setting as a fallback."""
    value = (explicit or os.environ.get(research_paths.ENV_VAR)
             or os.environ.get("WQ_RESEARCH_DIR") or research_paths.research_dir())
    return Path(value).expanduser().resolve()


def run_cli(root, *args, timeout=None, input_text=None):
    command = [sys.executable, str(REPO / "research_cli.py"), "--research-dir", str(root),
               "--quiet", *map(str, args)]
    try:
        completed = subprocess.run(command, cwd=REPO, text=True, input=input_text,
                                   capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return CliResult(False, {}, f"CLI timed out after {timeout:g}s", "TIMEOUT")
    try:
        payload = json.loads(completed.stdout)
    except ValueError:
        payload = {}
    if isinstance(payload, dict):
        data = payload.get("data")
        data = data if isinstance(data, dict) else {}
        if completed.returncode == 0 and payload.get("ok") is True:
            return CliResult(True, data)
        error = payload.get("error") or {}
        if isinstance(error, dict) and error.get("code"):
            return CliResult(False, data, error.get("message") or error["code"], error["code"])
    detail = (completed.stderr or completed.stdout).strip()[-600:]
    return CliResult(False, {}, detail or f"invalid CLI response (exit {completed.returncode})",
                     "INVALID_RESPONSE")


def probe_state(data):
    """Return the queue state and the next action justified by saved arm evidence."""
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        return "failed", None
    execution = data.get("execution") or {}
    actions = data.get("next_actions")
    if (isinstance(execution, dict) and isinstance(actions, list)
            and all(type(execution.get(key)) is int for key in (
                "total", "finished", "terminal", "unknown_submission"))):
        next_actions = {item.get("action") for item in actions if isinstance(item, dict)}
        if execution["unknown_submission"]:
            return "deferred", "recover"
        if execution["finished"] + execution["terminal"] >= execution["total"]:
            state = "failed" if execution["terminal"] else "done"
            return state, None
        if "recover" in next_actions:
            return "deferred", "recover"
        if "resume" in next_actions:
            return "deferred", "resume"
        return "deferred", "recover"

    # Older checkpoints did not carry canonical execution counts or actions.
    pending = []
    failures = 0
    for row in rows:
        status = str(row.get("status") or "").upper()
        if row.get("submission_state") == "terminal":
            failures += 1
        elif row.get("progress_url") or row.get("metrics_pending"):
            pending.append("resume")
        elif status in _FAILED:
            failures += 1
        elif status in _FINISHED and row.get("alpha_id"):
            continue
        elif status == "QUEUED" and row.get("submission_state") == "not_submitted":
            pending.append("resume")
        else:
            pending.append("recover")
    if pending:
        # Reconcile ambiguous POSTs before starting any known queued arms.
        return "deferred", "recover" if "recover" in pending else "resume"
    finished = execution.get("finished")
    if isinstance(finished, int) and finished + failures < len(rows):
        return "deferred", "recover"
    return ("failed", None) if failures else ("done", None)


_probe_state = probe_state


def _saved_probe(preview, batch):
    """Bind an existing batch to the queued spec before resuming its saved requests."""
    try:
        saved = json.loads(batch.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return CliResult(False, {}, f"cannot read saved batch {batch.name}: {error}")
    if not isinstance(saved, dict) or not saved.get("request_fingerprint"):
        return CliResult(False, {}, f"missing saved request fingerprint in {batch.name}")
    def roles(rows):
        return [(row.get("role"), row.get("design"), row.get("proxy_of")) for row in rows]

    if (preview.get("request_fingerprint") != saved["request_fingerprint"]
            or preview.get("design") != saved.get("design")
            or preview.get("quick_source") != saved.get("quick_source")
            or preview.get("breeding_source") != saved.get("breeding_source")
            or ("track" in saved and preview.get("track") != saved["track"])
            or roles(preview.get("results") or []) != roles(saved.get("arms") or [])):
        return CliResult(False, {}, f"queued spec does not match {batch.name}; choose a new spec name")
    return CliResult(True, saved)


def _receipt_batch(root, receipt):
    try:
        stored = json.loads(receipt.read_text(encoding="utf-8"))
        target = (stored.get("data") or {}).get("path")
    except (OSError, AttributeError, ValueError):
        return None
    if isinstance(target, str):
        candidate = Path(target).expanduser().resolve()
        if candidate.is_relative_to(root) and candidate.exists():
            return candidate
        raise ValueError(f"saved receipt references a missing or external batch: {candidate}")
    return None


def _legacy_batch(root, preview, receipt):
    """The old simulation queue used the probe's default batch path."""
    target = _receipt_batch(root, receipt) if receipt.exists() else None
    if target is not None:
        return target
    # An older worker may have saved a batch before it reached --save. Restrict
    # discovery to the exact default name and the date suffix used by probe.
    default = Path(preview["path"]).resolve()
    prefix = default.stem[:-10]
    if not default.is_relative_to(root) or not prefix:
        return None
    pattern = glob.escape(prefix) + "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"
    candidates = sorted(default.parent.glob(pattern))
    candidates = [path for path in candidates if _saved_probe(preview, path).ok]
    if len(candidates) > 1:
        raise ValueError("multiple matching legacy batches; use an explicit probe recovery")
    return candidates[0] if candidates else None


def run_probe(root, spec, batch, receipt, *, max_wait, concurrency=None,
              multi_size=None, recover_only=False, strict_gate=False, defer_quick_pnl=False,
              native_poll_once=False):
    """Run or continue a probe. A CLI exit of zero can still mean deferred work."""
    from research_app.experiments import probe_spec_identity
    from research_app.runtime import CliError

    if batch.exists():
        try:
            identity = probe_spec_identity(spec)
        except CliError as exc:
            return ProbeRun("failed", {}, exc.message)
        saved_path = batch
    else:
        preview_options = ("--multi-size", str(multi_size)) if multi_size is not None else ()
        if strict_gate:
            preview_options += ("--strict-gate",)
        preview = run_cli(root, "probe", spec, "--dry-run", *preview_options, timeout=120)
        if not preview.ok:
            return ProbeRun("deferred" if preview.code in _RETRYABLE else "failed", {}, preview.detail)
        identity = preview.data
        saved_path = _legacy_batch(root, identity, receipt)
    wait = ("--max-wait-seconds", str(max_wait))
    budget = () if concurrency is None else ("--batch-size", str(concurrency))
    if saved_path is not None:
        saved = _saved_probe(identity, saved_path)
        if not saved.ok:
            return ProbeRun("failed", {}, saved.detail)
        state, action = probe_state(saved.data)
        if state == "failed" and not (saved.data.get("execution") or {}).get("finished"):
            return ProbeRun(state, {}, f"saved batch has terminal failures ({saved_path.name})")
        if recover_only and action == "resume":
            rows = saved.data["rows"]
            if all(row.get("submission_state") == "not_submitted" for row in rows):
                return ProbeRun("deferred", {}, "awaiting authorization to launch queued arms")
            action = "recover"
        # A completed batch can still need idempotent ledger reconciliation after a crash.
        arguments = ("--" + (action or "recover"), saved_path)
    elif recover_only:
        return ProbeRun("deferred", {}, "awaiting authorization to launch a new batch")
    else:
        structural = sorted({item["code"] for item in preview.data.get("blocking", ())
                             if item.get("code") in _OFFLINE_BLOCKS})
        if structural:
            return ProbeRun("failed", preview.data, "offline syntax gate: " + ", ".join(structural))
        if strict_gate and preview.data.get("gate_blocking"):
            return ProbeRun("failed", preview.data, "exact finished requests repeat under --strict-gate")
        arguments = (spec, "--batch-path", batch, *preview_options)
    run_options = ("--defer-quick-pnl",) if defer_quick_pnl else ()
    if native_poll_once:
        run_options += ("--native-poll-once",)
    result = run_cli(root, "probe", *arguments, *wait, *budget, *run_options, "--save", receipt,
                     timeout=max_wait + 900)
    if not result.ok:
        state = "deferred" if result.code in _RETRYABLE else "failed"
        return ProbeRun(state, result.data, result.detail)
    state, action = probe_state(result.data)
    detail = {"done": "all arms finished", "deferred": f"awaiting probe --{action}",
              "failed": "one or more arms failed or the receipt has no arm evidence"}[state]
    field_deferred = result.data.get("preflight_deferred")
    if state == "deferred" and field_deferred:
        detail += f" (field preflight deferred for {len(field_deferred)} settings groups)"
    return ProbeRun(state, result.data, detail)


def probe_poll_at(document):
    """Earliest useful poll when remaining saved work is accepted and timed."""
    rows = document.get("rows") or []
    execution = document.get("execution") or {}
    if (not isinstance(rows, list) or not rows or not isinstance(execution, dict)
            or execution.get("total") != len(rows) or type(execution.get("finished")) is not int):
        return None
    # Native Retry-After is written once on the parent anchor, rather than on
    # every child row. A terminal parent may still expose new child evidence.
    records = document.get("records")
    parent_times = {
        (record["multi_request_id"], record["multi_parent_url"]): record.get("multi_next_parent_poll_at")
        for record in records if isinstance(record, dict)
        and type(record.get("multi_member")) is int and record["multi_member"] == 0
        and record.get("slot_reservation_state") == "accepted"
        and bool(record.get("slot_reservation"))
        and not record.get("multi_parent_terminal")
        and isinstance(record.get("multi_request_id"), str) and isinstance(record.get("multi_parent_url"), str)
    } if isinstance(records, list) else {}
    times, finished = [], 0
    for row in rows:
        if not isinstance(row, dict):
            return None
        state = row.get("submission_state")
        if state == "terminal":
            continue
        if state != "accepted":
            return None
        parent_url, request_id = row.get("multi_parent_url"), row.get("multi_request_id")
        if (not row.get("progress_url") and not row.get("metrics_pending")
                and (not parent_url or row.get("multi_parent_terminal"))
                and row.get("alpha_id") and str(row.get("status") or "").upper() in _FINISHED):
            finished += 1
            continue
        if isinstance(parent_url, str) and isinstance(request_id, str) and not row.get("multi_parent_terminal"):
            at = parent_times.get((request_id, parent_url))
        elif not parent_url and (row.get("progress_url") or (row.get("alpha_id") and row.get("metrics_pending"))):
            at = row.get("next_poll_at")
        else:
            return None
        if type(at) not in (int, float) or not math.isfinite(at) or at <= 0:
            return None
        times.append(at)
    return min(times, default=None) if finished == execution["finished"] else None


def _recovery_schedule(root, batch, receipt, include_requests):
    if not batch.is_file():
        if not receipt.is_file():
            return _ProbeSchedule(priority=2)
        try:
            batch = _receipt_batch(root, receipt)
        except (OSError, RuntimeError, ValueError):
            batch = None  # The executor handles broken receipts on their own turn.
        if batch is None:
            return _ProbeSchedule(priority=1)
    try:
        document = json.loads(batch.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _ProbeSchedule()
    if not isinstance(document, dict):
        return _ProbeSchedule()
    rows = document.get("rows") or []
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return _ProbeSchedule()
    queued = [row for row in rows if row.get("submission_state") == "not_submitted"]
    wholly_queued = bool(queued) and len(queued) == len(rows)
    state, action = probe_state(document)
    recovery = action == "recover" or state in {"done", "failed"} or bool(rows and not queued)
    # The pre-POST checkpoint also owns its request until reconciliation proves
    # what happened. A wholly unstarted batch does not reserve any request.
    active = (row for row in rows if row.get("submission_state") in {"accepted", "unknown_submission"}
              or row.get("alpha_id") or row.get("progress_url") or row.get("multi_parent_url"))
    scope = None
    events = document.get("scheduling")
    if wholly_queued and document.get("format") == "probe-batch" and isinstance(events, list):
        snapshot = document.get("capability_snapshot")
        account = snapshot.get("account") if isinstance(snapshot, dict) else None
        fingerprint = account.get("fingerprint") if isinstance(account, dict) else None
        concurrency, multi_size = document.get("concurrency"), document.get("multi_size")
        if (isinstance(fingerprint, str) and fingerprint and type(concurrency) is int and concurrency > 0
                and type(multi_size) is int and multi_size >= 2):
            scope = (fingerprint, concurrency, multi_size)
    return _ProbeSchedule(priority=2 if wholly_queued else 0,
                          poll_at=probe_poll_at(document), recovery=recovery,
                          occupied=_watch_requests(active) if include_requests else frozenset(),
                          queued=_watch_requests(queued) if include_requests else frozenset(),
                          native_admission_scope=scope, scheduling_events=len(events) if isinstance(events, list) else 0)


def _watch_requests(rows):
    from research_app.experiments import _probe_identity

    return frozenset(_probe_identity(row["expression"], row["settings"]) for row in rows
                     if isinstance(row, dict) and isinstance(row.get("expression"), str)
                     and isinstance(row.get("settings"), dict) and row["settings"])


def _watch_occupy(occupied, spec, requests):
    for key in requests:
        occupied.setdefault(key, set()).add(spec)


def _watch_conflict(spec, requests, occupied):
    if not occupied:
        return None
    if requests is None:
        from research_app.experiments import probe_spec_identity
        from research_app.runtime import CliError

        try:
            requests = _watch_requests(probe_spec_identity(spec).get("results") or ())
        except CliError:
            # The executor reports invalid Probe specs and handles local SPC specs.
            return None
    # An owner whose spec was archived (done/failed) can never release the
    # request key through its own terminal receipt, so it must not keep another
    # spec waiting: wait only for owners that are still in the inbox.
    conflicts = {owner for key in requests for owner in occupied.get(key, ())
                 if owner != spec and owner.exists()}
    return min(conflicts, default=None)


CONFLICT_DEFER_LIMIT = 3


def _conflict_rounds_exceeded(rounds):
    """A duplicate-wait must end: after this many rounds the spec proceeds.

    `waiting for matching request` exists to avoid posting the same request
    twice at once. If the other spec never reaches a terminal state the waiter
    would otherwise hang forever, so the budget forces a decision.
    """
    return rounds >= CONFLICT_DEFER_LIMIT


def run_queue(root, inbox, paths, log, *, max_wait, interval, once=False,
              concurrency=None, consume=None, execute=None, schedule_recovery=False, recover_only=False,
              launch_rank=None):
    """Lock one inbox and archive only terminal outcomes.

    A deferred job stays with its saved batch. In once mode, process at most
    one spec and return 0 for done or empty, 1 for failed, or 3 for deferred
    or busy. The optional consumer runs before terminal archival. Campaign
    watch gives recovery a shorter turn when additional work is ready.
    The optional launch rank orders only wholly unstarted specs within their tier.
    """
    if execute is None:
        execute = partial(run_probe, recover_only=True) if recover_only else run_probe
    inbox.mkdir(parents=True, exist_ok=True)
    for state in ("done", "failed"):
        (inbox / state).mkdir(exist_ok=True)
    try:
        with cache_store.file_lock(inbox / ".runner.lock", timeout=0):
            watch = schedule_recovery and not once
            guard = schedule_recovery and not recover_only
            full_recovery_turn = set()
            conflict_rounds = {}
            while True:
                scheduled = []
                occupied = {}
                specs = sorted(inbox.glob("*.json"))
                full_recovery_turn.intersection_update(specs)
                for spec in specs:
                    batch, receipt = paths(spec)
                    schedule = (_recovery_schedule(root, batch, receipt, guard)
                                if schedule_recovery else _ProbeSchedule())
                    scheduled.append((spec, batch, receipt, schedule))
                    _watch_occupy(occupied, spec, schedule.occupied)
                if schedule_recovery:
                    scheduled.sort(key=lambda item: (item[3].priority,
                        launch_rank(item[0]) if launch_rank and guard and item[3].priority == 2 else 0))
                now = time.time()
                ready = {spec for spec, _, _, task in scheduled
                         if watch and (not recover_only or task.priority < 2)
                         and (task.poll_at is None or task.poll_at <= now)}
                next_poll = None
                saturated = set()
                skipped_launches = 0
                for spec, batch, receipt, schedule in scheduled:
                    ready.discard(spec)
                    if schedule.native_admission_scope in saturated:
                        skipped_launches += 1
                        continue
                    poll_at = schedule.poll_at
                    if poll_at is not None and poll_at > time.time():
                        next_poll = poll_at if next_poll is None else min(next_poll, poll_at)
                        continue
                    log({"kind": "start", "spec": spec.name})
                    wait = max_wait
                    try:
                        source = spec.read_bytes()
                        recovery = schedule.recovery or recover_only
                        owner = _watch_conflict(spec, schedule.queued, occupied) if not recovery and guard else None
                        if owner is not None and _conflict_rounds_exceeded(conflict_rounds.get(spec.name, 0)):
                            owner = None  # bounded exit: stop waiting and decide
                        if owner is not None:
                            conflict_rounds[spec.name] = conflict_rounds.get(spec.name, 0) + 1
                            result = ProbeRun("deferred", {}, f"waiting for matching request in {owner.name}")
                        else:
                            conflict_rounds.pop(spec.name, None)
                            if ready and recovery and schedule.priority == 0 and spec not in full_recovery_turn:
                                wait = min(max_wait, interval)
                            full_recovery_turn.discard(spec)
                            result = execute(root, spec, batch, receipt, max_wait=wait, concurrency=concurrency)
                        if result.state != "deferred":
                            destination = inbox / result.state / spec.name
                            if destination.exists():
                                raise ValueError(f"archive already exists: {destination}")
                            if consume is not None and result.data.get("rows"):
                                consume(spec, result.data)
                            if spec.read_bytes() != source:
                                raise ValueError(f"spec changed during processing: {spec.name}")
                            shutil.move(spec, destination)
                    except Exception as error:
                        # Leave the spec and any saved POST intent for an exact retry.
                        result = ProbeRun("deferred", {}, f"{type(error).__name__}: {error}")
                    if watch:
                        updated = _recovery_schedule(root, batch, receipt, guard)
                        _watch_occupy(occupied, spec, updated.occupied)
                        events = result.data.get("scheduling")
                        # Reuse only fresh authenticated capacity evidence for this sweep.
                        # Saved evidence alone can outlive a released slot or account change.
                        if (guard and result.state == "deferred" and updated.native_admission_scope
                                and isinstance(events, list) and len(events) == updated.scheduling_events
                                and len(events) > schedule.scheduling_events and isinstance(events[-1], dict)
                                and events[-1].get("reason") == "shared_slots_occupied"):
                            saturated.add(updated.native_admission_scope)
                        updated_at = updated.poll_at
                        if result.state == "deferred" and updated_at is not None and updated_at > time.time():
                            next_poll = updated_at if next_poll is None else min(next_poll, updated_at)
                        elif result.state == "deferred" and wait < max_wait:
                            # Untimed recovery can need longer than a short turn.
                            full_recovery_turn.add(spec)
                    log({"kind": result.state, "spec": spec.name, "detail": result.detail,
                         "batch": str(batch), "receipt": str(receipt)})
                    if once:
                        return {"done": 0, "failed": 1, "deferred": 3}[result.state]
                if skipped_launches:
                    log({"kind": "waiting-for-slots", "deferred_specs": skipped_launches})
                if once:
                    if next_poll is not None:
                        log({"kind": "waiting", "next_poll_at": next_poll,
                             "detail": "waiting for the next saved simulation poll"})
                        return 3
                    return 0
                delay = interval if next_poll is None else min(interval, max(0, next_poll - time.time()))
                time.sleep(delay)
    except cache_store.LockTimeout:
        log({"kind": "busy", "detail": f"another runner is active for {inbox}"})
        return 3
