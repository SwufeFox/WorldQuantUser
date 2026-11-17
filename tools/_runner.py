"""Shared CLI and saved-batch handling for the background research tools."""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import alpha_registry
import cache_store
import research_paths
import simulation_slots

REPO = Path(__file__).resolve().parents[1]
_FINISHED = alpha_registry.FINISHED_STATUSES | {"ACTIVE", "SUBMITTED", "UNSUBMITTED"}
_FAILED = simulation_slots.TERMINAL_FAILURE_STATUSES | {"FAILED"}
_RETRYABLE = {"RATE_LIMITED", "SERVICE_ERROR", "AUTH_CAPTCHA", "INTERRUPTED", "TIMEOUT",
              "INVALID_RESPONSE"}


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


def _probe_state(data):
    """Return the queue state and the next action justified by saved arm evidence."""
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        return "failed", None
    pending = []
    failures = 0
    for row in rows:
        status = str(row.get("status") or "").upper()
        if row.get("progress_url") or row.get("metrics_pending"):
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
    finished = (data.get("execution") or {}).get("finished")
    if isinstance(finished, int) and finished + failures < len(rows):
        return "deferred", "recover"
    return ("failed", None) if failures else ("done", None)


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
            or roles(preview.get("results") or []) != roles(saved.get("arms") or [])):
        return CliResult(False, {}, f"queued spec does not match {batch.name}; choose a new spec name")
    return CliResult(True, saved)


def _legacy_batch(root, preview, receipt):
    """The old simulation queue used the probe's default batch path."""
    if receipt.exists():
        try:
            stored = json.loads(receipt.read_text(encoding="utf-8"))
            target = (stored.get("data") or {}).get("path")
        except (OSError, AttributeError, ValueError):
            target = None
        if isinstance(target, str):
            candidate = Path(target).expanduser().resolve()
            if candidate.is_relative_to(root) and candidate.exists():
                return candidate
            raise ValueError(f"saved receipt references a missing or external batch: {candidate}")
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


def run_probe(root, spec, batch, receipt, *, max_wait, concurrency=None):
    """Run or continue a probe. A CLI exit of zero can still mean deferred work."""
    preview = run_cli(root, "probe", spec, "--dry-run", timeout=120)
    if not preview.ok:
        return ProbeRun("deferred" if preview.code in _RETRYABLE else "failed", {}, preview.detail)
    wait = ("--max-wait-seconds", str(max_wait))
    budget = () if concurrency is None else ("--batch-size", str(concurrency))
    saved_path = batch if batch.exists() else _legacy_batch(root, preview.data, receipt)
    if saved_path is not None:
        saved = _saved_probe(preview.data, saved_path)
        if not saved.ok:
            return ProbeRun("failed", {}, saved.detail)
        state, action = _probe_state(saved.data)
        if state == "failed":
            return ProbeRun(state, {}, f"saved batch has terminal failures ({saved_path.name})")
        # A completed batch can still need idempotent ledger reconciliation after a crash.
        arguments = ("--" + (action or "recover"), saved_path)
    else:
        arguments = (spec, "--batch-path", batch)
    result = run_cli(root, "probe", *arguments, *wait, *budget, "--save", receipt,
                     timeout=max_wait + 900)
    if not result.ok:
        state = "deferred" if result.code in _RETRYABLE else "failed"
        return ProbeRun(state, result.data, result.detail)
    state, action = _probe_state(result.data)
    detail = {"done": "all arms finished", "deferred": f"awaiting probe --{action}",
              "failed": "one or more arms failed or the receipt has no arm evidence"}[state]
    return ProbeRun(state, result.data, detail)


def run_queue(root, inbox, paths, log, *, max_wait, interval, once=False,
              concurrency=None, consume=None):
    """Visit one inbox at a time and archive only terminal outcomes."""
    inbox.mkdir(parents=True, exist_ok=True)
    for state in ("done", "failed"):
        (inbox / state).mkdir(exist_ok=True)
    try:
        with cache_store.file_lock(inbox / ".runner.lock", timeout=0):
            while True:
                specs = sorted(inbox.glob("*.json"))
                for spec in specs:
                    batch, receipt = paths(spec)
                    log({"kind": "start", "spec": spec.name})
                    try:
                        result = run_probe(root, spec, batch, receipt, max_wait=max_wait,
                                           concurrency=concurrency)
                        if result.state != "deferred":
                            destination = inbox / result.state / spec.name
                            if destination.exists():
                                raise ValueError(f"archive already exists: {destination}")
                            if consume is not None and result.data.get("rows"):
                                consume(spec, result.data)
                            shutil.move(spec, destination)
                    except Exception as error:
                        # Leave the spec and any saved POST intent for an exact retry.
                        result = ProbeRun("deferred", {}, f"{type(error).__name__}: {error}")
                    log({"kind": result.state, "spec": spec.name, "detail": result.detail,
                         "batch": str(batch), "receipt": str(receipt)})
                    if once:
                        return {"done": 0, "failed": 1, "deferred": 3}[result.state]
                if once:
                    return 0
                time.sleep(interval)
    except cache_store.LockTimeout:
        log({"kind": "busy", "detail": f"another runner is active for {inbox}"})
        return 3
