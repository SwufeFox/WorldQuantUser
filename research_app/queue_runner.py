"""Shared CLI and saved-batch handling for the background research tools."""

from __future__ import annotations

import glob
import hashlib
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
CLI_TIMEOUT_GRACE_SECONDS = 15
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
    """Structured result returned by invoking ``research_cli.py`` in a subprocess."""

    ok: bool
    data: dict
    detail: str = ""
    code: str = ""
    error_details: dict | None = None


@dataclass(frozen=True)
class ProbeRun:
    """State and payload produced by executing or resuming one probe batch."""

    state: str
    data: dict
    detail: str = ""


@dataclass(frozen=True)
class _ProbeSchedule:
    priority: int = 0
    poll_at: float | None = None
    recovery: bool = False
    terminal: bool = False
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
    """Run ``research_cli.py`` for *root* and parse its stdout JSON envelope."""
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
            details = error.get("details")
            return CliResult(False, data, error.get("message") or error["code"], error["code"],
                             details if isinstance(details, dict) else None)
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
        if execution["unknown_submission"] or any(row.get("submission_state") == "unknown_submission" for row in rows):
            return "deferred", "recover"
        if execution["finished"] + execution["terminal"] >= execution["total"]:
            state = "failed" if execution["terminal"] else "done"
            return state, None
        if "resume" in next_actions and any(row.get("submission_state") == "not_submitted" for row in rows):
            # Resume also recovers accepted rows, then refills available slots.
            # An ambiguous POST above still forces read-only reconciliation.
            return "deferred", "resume"
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
        return CliResult(False, {},
                         f"queued spec does not match {batch.name}; inspect the original spec and checkpoint, "
                         "then restore the queued spec or deliberately create a separate experiment",
                         error_details={"reason": "batch_identity_mismatch", "batch_path": str(batch),
                                        "next_action": "inspect_saved_identity"})
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
    from .experiments import probe_spec_identity
    from .runtime import CliError

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
            data = {}
            if saved.error_details:
                data["diagnosis"] = {"state": "blocked", "reason": saved.error_details["reason"],
                    "batch_path": str(saved_path), "spec_path": str(spec),
                    "next_action": {"action": saved.error_details["next_action"],
                        "argv": ["python", "research_cli.py", "--research-dir", str(root),
                                 "probe", str(spec), "--dry-run"]}}
            return ProbeRun("failed", data, saved.detail)
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
                     timeout=max_wait + CLI_TIMEOUT_GRACE_SECONDS)
    if not result.ok:
        state = "deferred" if result.code in _RETRYABLE else "failed"
        details = result.error_details or {}
        diagnosis = {"state": "blocked" if details.get("reason") == "batch_locked" else state,
                     "reason": details.get("reason") or result.code.lower(),
                     "error_code": result.code,
                     "batch_path": str(saved_path or batch),
                     "next_action": {"action": "status", "argv": ["python", "research_cli.py",
                         "--research-dir", str(root), "campaign", "status"]}}
        diagnosis.update({key: details[key] for key in ("batch_path", "lock_path", "owner") if key in details})
        return ProbeRun(state, {**result.data, "diagnosis": diagnosis}, result.detail)
    state, action = probe_state(result.data)
    detail = {"done": "all arms finished", "deferred": f"awaiting probe --{action}",
              "failed": "one or more arms failed or the receipt has no arm evidence"}[state]
    field_deferred = result.data.get("preflight_deferred")
    if state == "deferred" and field_deferred:
        detail += f" (field preflight deferred for {len(field_deferred)} settings groups)"
    return ProbeRun(state, result.data, detail)


def probe_poll_at(document):
    """Earliest useful accepted-job poll or queued group's preflight retry."""
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
    from .experiments import matching_preflight_deferrals

    snapshot = document.get("capability_snapshot")
    account = snapshot.get("account") if isinstance(snapshot, dict) else None
    deferred = (matching_preflight_deferrals(document, account.get("fingerprint"))
                if isinstance(account, dict) else {})
    account_retry_at = max((entry["retry_at"] for entry in deferred.values()
                           if entry.get("code") == "RATE_LIMITED"), default=0)
    queued_times = {label: entry["retry_at"] for entry in deferred.values() for label in entry["labels"]}
    times, finished = [], 0
    for row in rows:
        if not isinstance(row, dict):
            return None
        state = row.get("submission_state")
        if state == "terminal":
            continue
        if state == "not_submitted":
            retry_at = max(account_retry_at, queued_times.get(row.get("label"), 0))
            if not retry_at:
                return None
            times.append(retry_at)
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


def probe_diagnosis(document, batch, *, root=None):
    """Explain the next safe operation from saved evidence, without platform calls."""
    state, action = probe_state(document)
    execution = document.get("execution") or {}
    execution = execution if isinstance(execution, dict) else {}
    rows = document.get("rows") or []
    next_action = None
    if action:
        prefix = ["python", "research_cli.py", *(["--research-dir", str(root)] if root else [])]
        next_action = {"action": action, "argv": [*prefix, "probe",
                                                     "--" + action, str(batch)]}
    if execution.get("unknown_submission") or any(
            row.get("submission_state") == "unknown_submission" for row in rows
            if isinstance(row, dict)):
        return {"state": "blocked", "reason": "submission_unknown", "next_action": next_action}
    if state in {"done", "failed"}:
        return {"state": state, "reason": "complete" if state == "done" else "terminal_failure",
                "next_action": next_action}
    poll_at = probe_poll_at(document)
    deferred = document.get("preflight_deferred")
    field_waits = [entry.get("retry_at") for entry in deferred.values() if isinstance(entry, dict)] \
        if isinstance(deferred, dict) else []
    if poll_at is not None and poll_at > time.time():
        return {"state": "waiting", "reason": "preflight_backoff" if poll_at in field_waits else "platform_pending",
                "next_poll_at": poll_at, "next_action": next_action}
    events = document.get("scheduling")
    last = events[-1] if isinstance(events, list) and events and isinstance(events[-1], dict) else {}
    if action == "resume" and last.get("reason") in {"shared_slots_occupied", "submission_cooldown"}:
        return {"state": "waiting", "reason": last["reason"], "next_action": next_action}
    return {"state": "ready" if action == "resume" else "waiting",
            "reason": "queued_work" if action == "resume" else "platform_pending", "next_action": next_action}


def _progress_fingerprint(document):
    """Measure evidence changes, excluding saves, retry clocks and scheduling events."""
    if not isinstance(document, dict):
        return None
    keys = ("label", "status", "submission_state", "alpha_id", "progress_url", "simulation_status",
            "multi_parent_url", "multi_parent_terminal", "metrics_pending", "train_sharpe", "train_fitness",
            "train_sharpe_h1", "train_sharpe_h2", "failing_checks")
    deferred, proofs = document.get("preflight_deferred"), {}
    if isinstance(deferred, dict):
        for name, entry in deferred.items():
            fields = entry.get("required_fields") if isinstance(entry, dict) else None
            if isinstance(fields, dict):
                proofs[name] = fields.get("verified_ids")
    rows = document.get("rows")
    rows = rows if isinstance(rows, list) else []
    evidence = {"execution": document.get("execution"),
                "rows": [{key: row.get(key) for key in keys} for row in rows
                         if isinstance(row, dict)],
                "field_proofs": proofs}
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _saved_progress(batch):
    try:
        document = json.loads(batch.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return _progress_fingerprint(document)


def recent_campaign_attempts(root):
    """Read a bounded tail of the existing log; old observations are never live locks."""
    try:
        with (Path(root) / "campaign-log.jsonl").open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            start = max(0, handle.tell() - 256 * 1024)
            handle.seek(start)
            if start:
                handle.readline()
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    result = {}
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if (isinstance(event, dict) and isinstance(event.get("spec"), str)
                and isinstance(event.get("progress"), dict)):
            result[event["spec"]] = event
        elif isinstance(event, dict) and event.get("kind") == "busy":
            result[None] = event
    return result


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
                          terminal=bool(rows) and state in {"done", "failed"},
                          occupied=_watch_requests(active) if include_requests else frozenset(),
                          queued=_watch_requests(queued) if include_requests else frozenset(),
                          native_admission_scope=scope, scheduling_events=len(events) if isinstance(events, list) else 0)


def _watch_requests(rows):
    from .experiments import probe_identity

    return frozenset(probe_identity(row["expression"], row["settings"]) for row in rows
                     if isinstance(row, dict) and isinstance(row.get("expression"), str)
                     and isinstance(row.get("settings"), dict) and row["settings"])


def _watch_occupy(occupied, spec, requests):
    for key in requests:
        occupied.setdefault(key, set()).add(spec)


def _watch_refresh(occupied, spec, schedule):
    """Only saved terminal evidence releases an owner's request identities."""
    if schedule.terminal:
        for key, owners in list(occupied.items()):
            owners.discard(spec)
            if not owners:
                del occupied[key]
    else:
        _watch_occupy(occupied, spec, schedule.occupied)


def _watch_conflict(spec, requests, occupied):
    if not occupied:
        return None
    if requests is None:
        from .experiments import probe_spec_identity
        from .runtime import CliError

        try:
            requests = _watch_requests(probe_spec_identity(spec).get("results") or ())
        except CliError:
            # The executor reports invalid Probe specs and handles local SPC specs.
            return None
    # Moving or deleting a spec says nothing about whether its POST was accepted.
    # _watch_refresh releases ownership only from saved terminal evidence.
    conflicts = {owner for key in requests for owner in occupied.get(key, ())
                 if owner != spec}
    return min(conflicts, default=None)


STALLED_ROUNDS = 3


def _archive_target(directory, spec, source):
    """Return an archive path without ever overwriting different evidence.

    Retrying the same archived spec is a normal idempotent outcome, not an
    operational failure. Only content-different namesakes receive a suffix.
    """
    fingerprint = hashlib.sha256(source).hexdigest()
    destination = directory / spec.name
    suffix = f"-{fingerprint[:10]}"
    attempt = 1
    while True:
        try:
            if not destination.exists():
                return destination, False
            if hashlib.sha256(destination.read_bytes()).hexdigest() == fingerprint:
                return destination, True
        except FileNotFoundError:
            return destination, False
        tail = "" if attempt == 1 else f"-{attempt}"
        destination = directory / f"{spec.stem}{suffix}{tail}{spec.suffix}"
        attempt += 1


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
            attempts = recent_campaign_attempts(root)
            retry_deadlines = {}
            occupied = {}
            saved_owners = {}
            if guard:
                # Seed old archived work once. Later sweeps reread only owners
                # whose checkpoint still has accepted or ambiguous requests.
                for state in ("done", "failed"):
                    for archived in (inbox / state).glob("*.json"):
                        owner = inbox / archived.name
                        saved_owners.setdefault(owner, paths(owner))
            while True:
                scheduled = []
                specs = sorted(inbox.glob("*.json"))
                for stale in set(retry_deadlines) - set(specs):
                    retry_deadlines.pop(stale, None)
                full_recovery_turn.intersection_update(specs)
                for owner, (batch, receipt) in list(saved_owners.items()):
                    if owner in specs:
                        continue
                    owner_schedule = _recovery_schedule(root, batch, receipt, True)
                    _watch_refresh(occupied, owner, owner_schedule)
                    if not any(owner in owners for owners in occupied.values()):
                        saved_owners.pop(owner)
                for spec in specs:
                    batch, receipt = paths(spec)
                    schedule = (_recovery_schedule(root, batch, receipt, guard)
                                if schedule_recovery else _ProbeSchedule())
                    scheduled.append((spec, batch, receipt, schedule))
                    _watch_refresh(occupied, spec, schedule)
                    if schedule.occupied and not schedule.terminal:
                        saved_owners[spec] = (batch, receipt)
                if schedule_recovery:
                    scheduled.sort(key=lambda item: (item[3].priority,
                        launch_rank(item[0]) if launch_rank and guard and item[3].priority == 2 else 0))
                now = time.time()
                def due_at(spec, task):
                    return retry_deadlines.get(spec, task.poll_at)
                ready = {spec for spec, _, _, task in scheduled
                         if watch and (not recover_only or task.priority < 2)
                         and (due_at(spec, task) is None or due_at(spec, task) <= now)}
                next_poll = None
                saturated = set()
                skipped_launches = 0
                for spec, batch, receipt, schedule in scheduled:
                    ready.discard(spec)
                    if schedule.native_admission_scope in saturated:
                        skipped_launches += 1
                        continue
                    poll_at = due_at(spec, schedule)
                    if poll_at is not None and poll_at > time.time():
                        next_poll = poll_at if next_poll is None else min(next_poll, poll_at)
                        continue
                    log({"kind": "start", "spec": spec.name})
                    wait = max_wait
                    before = _saved_progress(batch)
                    source_fingerprint = None
                    try:
                        source = spec.read_bytes()
                        source_fingerprint = hashlib.sha256(source).hexdigest()
                        recovery = schedule.recovery or recover_only
                        owner = _watch_conflict(spec, schedule.queued, occupied) if not recovery and guard else None
                        if owner is not None:
                            owner_batch, _ = paths(owner)
                            diagnosis = {"state": "blocked", "reason": "matching_request", "owner_spec": str(owner),
                                         "owner_batch": str(owner_batch), "next_action": {"action": "recover",
                                             "argv": ["python", "research_cli.py", "--research-dir", str(root),
                                                      "probe", "--recover", str(owner_batch)]}}
                            result = ProbeRun("deferred", {"diagnosis": diagnosis},
                                              f"waiting for matching request in {owner.name}")
                        else:
                            # A local turn is not the remote job's lifetime. Give
                            # other ready work a turn; untimed recovery gets a
                            # full-budget turn next time if the short read deferred.
                            if ready and recovery and schedule.priority == 0 and spec not in full_recovery_turn:
                                wait = min(max_wait, interval)
                            full_recovery_turn.discard(spec)
                            result = execute(root, spec, batch, receipt, max_wait=wait, concurrency=concurrency)
                        if result.state != "deferred":
                            destination, already_archived = _archive_target(
                                inbox / result.state, spec, source)
                            if spec.read_bytes() != source:
                                raise ValueError(f"spec changed during processing: {spec.name}")
                            if already_archived:
                                spec.unlink(missing_ok=True)
                            else:
                                if consume is not None and result.data.get("rows"):
                                    consume(spec, result.data)
                                if result.state == "failed":
                                    reason = inbox / "failed" / ".reasons" / destination.name
                                    cache_store.write_json_atomic(reason, {
                                        "source_fingerprint": source_fingerprint,
                                        "detail": result.detail or "terminal probe failure",
                                        "diagnosis": result.data.get("diagnosis"),
                                        "receipt": str(receipt),
                                        "observed_at": time.time(),
                                    })
                                shutil.move(spec, destination)
                    except Exception as error:
                        # Leave the spec and any saved POST intent for an exact retry.
                        result = ProbeRun("deferred", {"diagnosis": {"state": "blocked", "reason": "runner_error",
                            "next_action": {"action": "status", "argv": ["python", "research_cli.py",
                                "--research-dir", str(root), "campaign", "status"]}}},
                            f"{type(error).__name__}: {error}")
                    updated = schedule
                    if watch:
                        updated = _recovery_schedule(root, batch, receipt, guard)
                        _watch_refresh(occupied, spec, updated)
                        if updated.occupied and not updated.terminal:
                            saved_owners[spec] = (batch, receipt)
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
                    after = _saved_progress(batch)
                    previous = attempts.get(spec.name) or {}
                    if previous.get("source_fingerprint") != source_fingerprint:
                        previous = {}
                    progress = previous.get("progress") or {}
                    prior_rounds = progress.get("unchanged_rounds", 0)
                    prior_rounds = prior_rounds if type(prior_rounds) is int and prior_rounds >= 0 else 0
                    changed = (before != after or result.state in {"done", "failed"}
                               or "progress_fingerprint" in previous and previous["progress_fingerprint"] != after)
                    rounds = 0 if changed else prior_rounds + 1
                    diagnosis = result.data.get("diagnosis")
                    if not isinstance(diagnosis, dict):
                        diagnosis = (probe_diagnosis(result.data, batch, root=root) if result.data.get("rows") else
                                     {"state": result.state, "reason": "no_progress_observed" if result.state == "deferred"
                                      else "complete" if result.state == "done" else "terminal_failure"})
                    observed = time.time()
                    timed_wait = type(diagnosis.get("next_poll_at")) in (int, float) and diagnosis["next_poll_at"] > observed
                    retry_at = observed + interval
                    for candidate in (updated.poll_at, diagnosis.get("next_poll_at")):
                        if type(candidate) in (int, float) and candidate > retry_at:
                            retry_at = candidate
                    if (result.state == "deferred" and diagnosis.get("reason") in {
                            "platform_pending", "queued_work", "shared_slots_occupied",
                            "submission_cooldown", "preflight_backoff"}):
                        retry_deadlines[spec] = retry_at
                    else:
                        retry_deadlines.pop(spec, None)
                    progress = {"changed": changed, "unchanged_rounds": rounds,
                                "last_progress_at": observed if changed else progress.get("last_progress_at"),
                                "stalled": result.state == "deferred" and rounds >= STALLED_ROUNDS and not timed_wait}
                    event = {"kind": result.state, "spec": spec.name, "detail": result.detail,
                             "batch": str(batch), "receipt": str(receipt), "observed_at": observed,
                             **({"retry_at": retry_at} if result.state == "deferred" else {}),
                             "diagnosis": diagnosis, "progress": progress, "source_fingerprint": source_fingerprint,
                             "progress_fingerprint": after}
                    attempts[spec.name] = event
                    log(event)
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
                # A poll can fall due while the sweep is still executing other
                # specs. Rebuild the clock from current saved evidence instead
                # of sleeping on a deadline observed at the start of the sweep.
                now = time.time()
                next_poll = None
                for spec, batch, receipt, _ in scheduled:
                    if not spec.exists():
                        continue
                    at = retry_deadlines.get(spec)
                    if at is None:
                        at = _recovery_schedule(root, batch, receipt, guard).poll_at
                    if at is not None and at > now:
                        next_poll = at if next_poll is None else min(next_poll, at)
                delay = interval if next_poll is None else min(interval, max(0, next_poll - now))
                time.sleep(delay)
    except cache_store.LockTimeout:
        log({"kind": "busy", "detail": f"another runner is active for {inbox}",
             "diagnosis": {"state": "blocked", "reason": "runner_locked", "lock_path": str(inbox / ".runner.lock"),
                 "next_action": {"action": "status", "argv": ["python", "research_cli.py",
                     "--research-dir", str(root), "campaign", "status"]}}})
        return 3
