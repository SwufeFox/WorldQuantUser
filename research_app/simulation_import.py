"""Read parent and child simulation status and Alpha IDs for Probe."""

import math
import re
import time

import machine_lib as ml
import simulation_slots

from .runtime import CliError, session_reader


_SUCCESS = simulation_slots.TERMINAL_STATUSES - simulation_slots.TERMINAL_FAILURE_STATUSES
_FAILURE = simulation_slots.TERMINAL_FAILURE_STATUSES | {"FAILED"}
_PENDING = {"PENDING", "QUEUED", "WAITING", "SIMULATING", "RUNNING"}
_STATUS_TOKEN = re.compile(r"[A-Z][A-Z0-9_]{0,31}")
_MAX_CHILDREN = 10  # Documented BRAIN multi-simulation request size.


def _unknown(simulation_id, reason, status=None):
    return {"simulation_id": simulation_id, "status": status, "state": "unknown",
            "alpha_id": None, "reason": reason}


def _row(simulation_id, payload, retry_after, parent_id=None):
    """Extract the task state and check any identity echoed by the platform."""
    if payload.get("id") not in (None, simulation_id):
        return _unknown(simulation_id, "simulation_id_mismatch")
    if parent_id is not None and payload.get("parent") not in (None, parent_id):
        return _unknown(simulation_id, "parent_mismatch")

    status = payload.get("status")
    status = status.upper() if isinstance(status, str) else ""
    status = status if _STATUS_TOKEN.fullmatch(status) else None
    row = {"simulation_id": simulation_id, "status": status, "alpha_id": None}
    # The retry header takes precedence over status in the response body.
    if retry_after is not None:
        return {**row, "state": "pending", "retry_after_seconds": retry_after}
    if status in _PENDING:
        return {**row, "state": "pending"}
    if status in _FAILURE:
        return {**row, "state": "failed"}
    if status in _SUCCESS:
        alpha_id = payload.get("alpha")
        if ml.valid_simulation_id(alpha_id):
            return {**row, "state": "complete", "alpha_id": alpha_id}
        if parent_id is None:
            return {**row, "state": "complete"}
        return _unknown(simulation_id, "missing_or_invalid_alpha_id", status)
    return _unknown(simulation_id, "unrecognized_status", status)


def _read_children(read, children, parent_id, deadline):
    seen = {parent_id}
    rows = []
    pause_reason = None
    for child_id in children:
        if not ml.valid_simulation_id(child_id):
            rows.append(_unknown(None, "invalid_child_id"))
            continue
        if child_id in seen:
            rows.append(_unknown(child_id, "duplicate_child_id"))
            continue
        seen.add(child_id)
        if pause_reason or time.monotonic() >= deadline:
            rows.append(_unknown(child_id, pause_reason or "wait_budget_expired"))
            continue
        try:
            payload, retry_after = read(child_id)
        except ml.BrainError as exc:
            rows.append(_unknown(child_id, exc.code))
            if isinstance(exc, ml.RateLimited):
                pause_reason = exc.code
        else:
            rows.append(_row(child_id, payload, retry_after, parent_id))
    return rows


def read_existing_simulation(args, reference):
    """Read the parent and each declared child within one request budget."""
    try:
        simulation_id = ml.simulation_reference_id(reference)
    except ValueError as exc:
        raise CliError("INVALID_INPUT", str(exc),
                       "pass a simulation ID or https://api.worldquantbrain.com/simulations/<id> URL") from exc
    wait = getattr(args, "max_wait_seconds", 300.0)
    if type(wait) not in (int, float) or not math.isfinite(wait) or wait <= 0:
        raise CliError("INVALID_INPUT", "simulation inspection wait must be positive and finite",
                       "pass a positive number of seconds for --max-wait-seconds")
    deadline = time.monotonic() + wait
    run = session_reader(args, deadline=deadline)

    def read(identifier):
        return run(lambda session: ml.read_simulation_snapshot(session, identifier, deadline=deadline))

    payload, retry_after = read(simulation_id)
    parent = _row(simulation_id, payload, retry_after)
    kind, children_state, rows = "unknown", "unavailable", []
    children = payload.get("children")
    if parent.get("reason") != "simulation_id_mismatch" and retry_after is None:
        if isinstance(children, list) and children:
            if len(children) > _MAX_CHILDREN:
                raise CliError("SERVICE_ERROR", "BRAIN listed more than ten child simulations",
                               "inspect the parent simulation directly on BRAIN before proceeding")
            kind, children_state = "multi", "available"
            rows = _read_children(read, children, simulation_id, deadline)
        elif children not in (None, []):
            children_state = "malformed"
        elif parent["alpha_id"]:
            kind, children_state, rows = "single", "not_applicable", [parent]

    return {
        **parent,
        "kind": kind,
        "children_state": children_state,
        "rows": rows,
        "alpha_ids": list(dict.fromkeys(row["alpha_id"] for row in rows if row["alpha_id"])),
        "summary": {"total": len(rows), **{
            state: sum(row["state"] == state for row in rows)
            for state in ("complete", "failed", "pending", "unknown")
        }},
    }
