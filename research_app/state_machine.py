"""Persistent 4-gate state machine model, protocol firewall, and workspace migration."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import uuid

import alpha_registry
import cache_store
import fastexpr_ast
import machine_lib as ml
import pnl_store
import research_paths
import simulation_slots
import workspace_db

from . import brief, buffer_cmd, experiments, labs, research_route, selection, validation
from .runtime import CliError, parse_tags

AUTO_STATES = frozenset({
    "AUTO_PREFLIGHT",
    "AUTO_SPC",
    "AUTO_QUICK_SIM",
    "AUTO_HYDRATE_PNL",
    "AUTO_FULL_VALIDATE",
    "AUTO_BUFFER",
    "AUTO_SUBMITTING",
    "AUTO_CANCELLING",
})

GATE_STATES = frozenset({
    "GATE_PREFLIGHT_BLOCKED",
    "GATE_TRIAGE",
    "GATE_APPROVE_FULL",
    "GATE_DESCRIBE",
    "GATE_CONFIRM_SUBMIT",
    "GATE_LABS",
})

TERMINAL_STATES = frozenset({
    "COMPLETED",
    "DROPPED",
    "ARCHIVED",
    "REJECTED_AT_VALIDATION",
    "SHELVED",
    "SUBMITTED",
    "REJECTED_AT_SUBMIT",
    "CANCELLED",
    "FAILED",
})

PRE_FREEZE_STATES = frozenset({
    "AUTO_PREFLIGHT",
    "GATE_PREFLIGHT_BLOCKED",
    "AUTO_QUICK_SIM",
    "GATE_TRIAGE",
    "AUTO_HYDRATE_PNL",
    "GATE_APPROVE_FULL",
    "GATE_LABS",
})

ALL_STATES = AUTO_STATES | GATE_STATES | TERMINAL_STATES

DEFAULT_DAEMON_CONFIG = {
    "quota_floor": 0,
    "concurrency": 8,
    "multi_size": 10,
    "max_unchanged_defers": 3,
    "reconcile_interval_seconds": 300,
}

_DROP_VERDICTS = frozenset({"rejected", "falsified", "no_go"})
_REDACT_ROW_KEYS = frozenset({
    "test", "test_sharpe", "test_fitness", "test_turnover",
    "retention", "full_metrics", "validation", "checks", "cost_audit",
    "out_of_sample", "metrics", "is", "full", "test_metrics", "test_pnl", "test_pnl_path",
})
_REDACT_DOC_KEYS = frozenset({"validation", "cost_audit", "full_records", "full_checks"})


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _allowed_actions(doc):
    state = doc.get("state")
    run_id = doc.get("run_id")
    if state == "GATE_PREFLIGHT_BLOCKED":
        return [
            {"action": "retry-preflight", "command": f"research_cli.py act {run_id} retry-preflight"},
            {"action": "request-labs", "command": f"research_cli.py act {run_id} request-labs --trigger anomaly --fields ... --question \"...\" --expect \"...\" --branch \"...\""},
            {"action": "drop", "command": f"research_cli.py act {run_id} drop --reason \"...\""},
        ]
    if state == "GATE_TRIAGE":
        return [
            {"action": "hydrate-pnl",
             "command": f"research_cli.py act {run_id} hydrate-pnl --top 3"},
            {"action": "request-labs",
             "command": f"research_cli.py act {run_id} request-labs --trigger anomaly --fields ... --question \"...\" --expect \"...\" --branch \"...\""},
            {"action": "drop",
             "command": f"research_cli.py act {run_id} drop --verdict falsified --reason \"...\""},
        ]
    if state == "GATE_APPROVE_FULL":
        return [
            {"action": "approve-full",
             "command": f"research_cli.py act {run_id} approve-full"},
            {"action": "request-labs",
             "command": f"research_cli.py act {run_id} request-labs --trigger anomaly --fields ... --question \"...\" --expect \"...\" --branch \"...\""},
            {"action": "back-to-triage",
             "command": f"research_cli.py act {run_id} back-to-triage"},
            {"action": "drop",
             "command": f"research_cli.py act {run_id} drop --verdict rejected --reason \"...\""},
        ]
    if state == "GATE_LABS":
        ticket_id = doc.get("labs_ticket_id") or "TICKET_ID"
        return [
            {"action": "labs-resolve",
             "command": f"research_cli.py labs resolve {ticket_id} --summary \"...\""},
            {"action": "drop",
             "command": f"research_cli.py act {run_id} drop --verdict no_go --reason \"...\""},
        ]
    if state == "GATE_DESCRIBE":
        return [
            {"action": "describe",
             "command": f"research_cli.py act {run_id} describe --name \"...\" --category CAT --tags t1 t2 --description \"...\""},
            {"action": "drop",
             "command": f"research_cli.py act {run_id} drop --verdict rejected --reason \"...\""},
        ]
    if state == "GATE_CONFIRM_SUBMIT":
        return [
            {"action": "submit",
             "command": f"research_cli.py act {run_id} submit --yes"},
            {"action": "hold",
             "command": f"research_cli.py act {run_id} hold"},
        ]
    if state in AUTO_STATES:
        return [{"action": "cancel", "command": f"research_cli.py cancel {run_id}"}]
    return []


def _train_public_value(value):
    if isinstance(value, list):
        return [_train_public_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {key: _train_public_value(item) for key, item in value.items()
            if key not in _REDACT_ROW_KEYS and not key.startswith(("test_", "full_"))}


def public_run_view(doc, *, arm_label=None):
    """Return a JSON-safe run view obeying the protocol-level Train/Test firewall.

    Before a training selection is frozen (``PRE_FREEZE_STATES`` or
    ``selection_frozen`` is falsy), all test-period, post-validation, and live
    check metrics are redacted from the response.
    """
    state = doc.get("state")
    pre_freeze = state in PRE_FREEZE_STATES or not doc.get("selection_frozen")

    rows = []
    for row in doc.get("rows") or []:
        if arm_label is not None and row.get("label") != arm_label:
            continue
        if pre_freeze:
            item = _train_public_value(row)
        else:
            item = dict(row)
        rows.append(item)

    if arm_label is not None and not rows:
        raise CliError("NOT_FOUND", f"arm {arm_label!r} not found in run {doc.get('run_id')}",
                       f"run `research_cli.py inspect {doc.get('run_id')}` without --arm to see available labels")

    view = {
        "run_id": doc.get("run_id"),
        "name": doc.get("name"),
        "track": doc.get("track"),
        "design": doc.get("design"),
        "state": state,
        "metrics_scope": "train" if pre_freeze else "train+test",
        "test_view_open": not pre_freeze,
        "selection_frozen": bool(doc.get("selection_frozen")),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "next_wake_at": doc.get("next_wake_at"),
        "unchanged_defers": doc.get("unchanged_defers", 0),
        "stalled": bool(doc.get("stalled", False)),
        "request_fingerprint": doc.get("request_fingerprint"),
        "idea": doc.get("idea"),
        "mechanism": doc.get("mechanism"),
        "execution": doc.get("execution"),
        "selected_arms": doc.get("selected_arms"),
        "approved_arms": doc.get("approved_arms"),
        "comparisons": doc.get("comparisons"),
        "hydration": doc.get("hydration"),
        "labs_ticket_id": doc.get("labs_ticket_id"),
        "pre_labs_state": doc.get("pre_labs_state"),
        "labs_resolution": doc.get("labs_resolution"),
        "warnings": doc.get("warnings") or [],
        "preflight_block": doc.get("preflight_block"),
        "last_error": doc.get("last_error"),
        "rows": rows,
        "allowed_actions": _allowed_actions(doc),
    }
    if doc.get("track") == "SPC" and state in {"COMPLETED", "FAILED"}:
        view["experiment"] = doc.get("experiment")
    if not pre_freeze:
        view["selection_path"] = doc.get("selection_path")
        view["validation"] = doc.get("validation")
        view["cost_audit"] = doc.get("cost_audit")
        view["full_checks"] = doc.get("full_checks")
        view["candidate_metadata"] = doc.get("candidate_metadata")
        view["shelf_entry"] = doc.get("shelf_entry")
        view["submit_attempt"] = doc.get("submit_attempt")
    else:
        for key in ("comparisons", "hydration", "execution", "selected_arms", "approved_arms"):
            view[key] = _train_public_value(view[key])
        for key in _REDACT_DOC_KEYS:
            view.pop(key, None)
        if doc.get("records"):
            view["records_count"] = len(doc["records"])
    return view


def _run_summary(doc):
    execution = doc.get("execution") or {}
    return {
        "run_id": doc.get("run_id"),
        "name": doc.get("name"),
        "track": doc.get("track"),
        "state": doc.get("state"),
        "updated_at": doc.get("updated_at"),
        "next_wake_at": doc.get("next_wake_at"),
        "unchanged_defers": doc.get("unchanged_defers", 0),
        "stalled": bool(doc.get("stalled", False)),
        "execution": execution,
        "preflight_block": doc.get("preflight_block"),
        "last_error": doc.get("last_error"),
        "allowed_actions": _allowed_actions(doc),
    }


def _static_validate_arms(arms, *, source_dir=None):
    for arm in arms:
        settings = arm.get("settings") or {}
        expression = arm.get("expression") or ""
        if ml.is_python_language(settings.get("language")):
            report = validation.lint_python_source(expression, None, lookback=arm.get("lookback"))
            if not report["ok"]:
                raise CliError("INVALID_INPUT",
                               f"arm {arm.get('label')!r} failed static Python preflight: {report['errors']}",
                               "fix `@alpha(data=[...])`, lookback, or Python syntax before proposing",
                               details={"arm": arm.get("label"), "errors": report["errors"]})
        else:
            try:
                fastexpr_ast.parse_program(expression)
            except fastexpr_ast.ParseError as exc:
                raise CliError("INVALID_INPUT",
                               f"arm {arm.get('label')!r} has invalid FASTEXPR syntax: {exc}",
                               "run `research_cli.py fmt` or `lint` on the draft expression before proposing",
                               details={"arm": arm.get("label"), "error": str(exc)}) from exc


def shared_arm_settings(arms):
    """Resolve a homogeneous set of simulation settings for one run."""
    try:
        first, *rest = [ml.resolve_settings(arm.get("settings")) for arm in arms]
    except ValueError as exc:
        raise CliError("INVALID_INPUT", f"invalid simulation settings: {exc}",
                       "check each arm's settings in the spec") from exc
    if any(settings != first for settings in rest):
        raise CliError("INVALID_INPUT", "arms in one run must share simulation settings",
                       "split arms with different settings into separate specs")
    return first


def _allocate_run_id(runs_dir, name, *, research_dir=None):
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(name or "run")).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)[:28] or "run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    for _ in range(16):
        suffix = uuid.uuid4().hex[:6]
        run_id = f"{slug}-{stamp}-{suffix}"
        if workspace_db.get_run(run_id, research_dir=research_dir) is None and not (runs_dir / f"{run_id}.json").exists():
            return run_id
    return f"run-{uuid.uuid4().hex[:12]}"


class StateMachineEngine:
    """Durable, single-writer state machine managing experiment runs under a research directory."""

    def __init__(self, research_dir=None):
        """Initialize workspace directories under *research_dir* without spawning background tasks."""
        self.root = Path(research_dir or research_paths.ensure_dir()).expanduser().resolve()
        self.runs_dir = self.root / "runs"
        self.daemon_dir = self.root / "daemon"
        self.config_path = self.daemon_dir / "config.json"
        self.state_lock_path = self.daemon_dir / "state.lock"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.daemon_dir.mkdir(parents=True, exist_ok=True)

    def run_path(self, run_id):
        """Return canonical JSON path for *run_id*, validating safe filename characters."""
        if not isinstance(run_id, str) or not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
            raise CliError("INVALID_INPUT", f"invalid run_id {run_id!r}",
                           "pass a valid run ID returned by `propose`, `inbox`, or `runs`")
        path = (self.runs_dir / f"{run_id}.json").resolve()
        if path.parent != self.runs_dir:
            raise CliError("INVALID_INPUT", f"invalid run_id {run_id!r}",
                           "run ID must resolve directly inside runs/")
        return path

    def load_config(self):
        """Return daemon scheduling configuration merged over defaults from ``workspace.db``."""
        legacy = {}
        if self.config_path.is_file():
            raw, err = cache_store.load_json(self.config_path, default={})
            if not err and isinstance(raw, dict):
                legacy = {key: raw[key] for key in DEFAULT_DAEMON_CONFIG
                          if type(raw.get(key)) in (int, float) and 0 <= raw[key] < math.inf}
        return workspace_db.get_config(DEFAULT_DAEMON_CONFIG, legacy=legacy, research_dir=self.root)

    def update_config(self, updates):
        """Persist updated scheduling config fields atomically inside ``workspace.db``."""
        with cache_store.file_lock(self.state_lock_path):
            cfg = self.load_config()
            for key, value in (updates or {}).items():
                if key not in DEFAULT_DAEMON_CONFIG:
                    raise CliError("INVALID_INPUT", f"unknown config key {key!r}",
                                   f"allowed keys: {', '.join(sorted(DEFAULT_DAEMON_CONFIG))}")
                if value is None:
                    continue
                if type(value) not in (int, float) or not 0 <= value < math.inf:
                    raise CliError("INVALID_INPUT", f"{key} must be a non-negative number",
                                   f"pass a numeric value >= 0 for {key}")
                cfg[key] = int(value)
            workspace_db.set_config(cfg, research_dir=self.root)
            return cfg

    def _sync_legacy_runs(self):
        if self.runs_dir.is_dir():
            workspace_db.import_legacy_run_files(self.runs_dir, research_dir=self.root)

    def load_run_raw(self, run_id):
        """Load raw run document for *run_id* from ``workspace.db``, raising NOT_FOUND if absent."""
        self.run_path(run_id)
        doc = workspace_db.get_run(run_id, research_dir=self.root)
        if doc is None:
            self._sync_legacy_runs()
            doc = workspace_db.get_run(run_id, research_dir=self.root)
        if doc is None:
            path = self.run_path(run_id)
            if path.is_file():
                raw, err = cache_store.load_json(path)
                if err or not isinstance(raw, dict):
                    raise CliError("INVALID_INPUT", f"run {run_id!r} file is unreadable: {err}",
                                   f"inspect {path} for JSON corruption")
            raise CliError("NOT_FOUND", f"run {run_id!r} does not exist",
                           "run `research_cli.py runs` or `inbox` to list available runs")
        return doc

    def save_run(self, doc, *, from_state=None, trigger=None, detail=None, durable=False):
        """Atomically persist *doc* to ``workspace.db`` and append a transition event."""
        doc["updated_at"] = _utc_now()
        self.run_path(doc["run_id"])
        workspace_db.upsert_run(doc, from_state=from_state, trigger=trigger,
                                detail=detail, durable=durable, research_dir=self.root)
        return doc

    def iter_runs(self, *, states=None, limit=None):
        """Return the matching durable run documents, newest first."""
        self._sync_legacy_runs()
        return workspace_db.list_runs(states=states, limit=limit, research_dir=self.root)

    def propose(self, spec_payload, *, source_path=None):
        """Validate a spec offline and create a durable run."""
        if isinstance(spec_payload, (str, bytes)):
            try:
                spec_payload = json.loads(spec_payload)
            except ValueError as exc:
                raise CliError("INVALID_INPUT", f"spec must be valid JSON: {exc}",
                               "fix JSON syntax before calling `propose`") from exc
        if not isinstance(spec_payload, dict):
            raise CliError("INVALID_INPUT", "spec must be a JSON object",
                           "see `research_cli.py schema propose` for the required spec shape")

        try:
            track = research_route.normalize_route(spec_payload)["track"]
        except ValueError as exc:
            raise CliError("INVALID_INPUT", str(exc), "check the track and settings in the proposed spec") from exc
        if track == "SPC":
            return self._propose_spc(spec_payload, source_path)
        with cache_store.file_lock(self.state_lock_path):
            return self._propose_unlocked(spec_payload, source_path=source_path, track=track)

    def _propose_unlocked(self, spec_payload, *, source_path=None, track):
        spec_path_str = str(source_path or "-")
        normalized = experiments._normalize_probe_spec(dict(spec_payload), spec_path_str)
        _, _, _, arms, design = experiments._probe_inputs(
            type("Args", (), {"spec": spec_path_str, "name": normalized.get("name"),
                              "research_dir": str(self.root)})(),
            document=normalized,
        )
        spec_idea = (normalized.get("idea") or "").strip()
        missing_idea = [arm["label"] for arm in arms if not (arm.get("idea") or spec_idea).strip()]
        if missing_idea:
            raise CliError("INVALID_INPUT",
                           f"missing non-empty `idea` on arms: {', '.join(missing_idea[:10])}",
                           "set `idea` on the spec or every arm explaining the hypothesis",
                           details={"missing_idea_labels": missing_idea})
        shared_arm_settings(arms)
        _static_validate_arms(arms, source_dir=Path(source_path).parent if source_path else None)
        fingerprint = experiments._request_fingerprint(arms)

        self._sync_legacy_runs()
        existing = workspace_db.find_active_by_fingerprint(fingerprint, AUTO_STATES | GATE_STATES, research_dir=self.root)
        if existing is not None:
            eid = existing["run_id"]
            raise CliError(
                "INVALID_INPUT",
                f"identical request already active in run {eid}",
                f"run `research_cli.py inspect {eid}` instead of proposing a duplicate",
                details={"run_id": eid, "state": existing.get("state")},
            )

        run_id = _allocate_run_id(self.runs_dir, normalized.get("name") or (Path(source_path).stem if source_path else "run"), research_dir=self.root)
        now = _utc_now()
        records = [{
            "arm_index": arm["arm_index"],
            "label": arm["label"],
            "role": arm.get("role"),
            "expression": arm["expression"],
            "settings": arm["settings"],
            "ops": arm.get("ops"),
            "status": "QUEUED",
            "submission_attempted": False,
        } for arm in arms]
        doc = {
            "version": 4,
            "run_id": run_id,
            "name": normalized.get("name") or run_id,
            "track": track,
            "design": design,
            "state": "AUTO_PREFLIGHT",
            "selection_frozen": False,
            "request_fingerprint": fingerprint,
            "idea": normalized.get("idea"),
            "mechanism": normalized.get("mechanism"),
            "warnings": labs.open_ticket_warnings_for_arms(arms, research_dir=self.root),
            "source_spec": str(source_path) if source_path else None,
            "created_at": now,
            "updated_at": now,
            "next_wake_at": None,
            "unchanged_defers": 0,
            "stalled": False,
            "progress_hash": None,
            "arms": arms,
            "records": records,
            "rows": [{"label": arm["label"], "role": arm.get("role"), "status": "QUEUED"} for arm in arms],
            "execution": {"total": len(arms), "not_submitted": len(arms), "accepted": 0, "finished": 0, "terminal": 0},
            "config_snapshot": {k: v for k, v in self.load_config().items()},
        }
        self.save_run(doc, trigger="propose", detail={"arms": len(arms), "track": track})
        return public_run_view(doc)

    def _propose_spc(self, spec_payload, source_path):
        if source_path is None:
            raise CliError("INVALID_INPUT", "SPC `propose` requires a spec file path containing relative run files",
                           "pass `research_cli.py propose SPEC.json`")
        src = Path(source_path).expanduser().resolve()
        runs = spec_payload.get("runs")
        if not isinstance(runs, list) or not runs or any(not isinstance(item, str) or not item.strip() for item in runs):
            raise CliError("INVALID_INPUT", "SPC spec requires a non-empty list of run file paths",
                           "set `runs` in the SPC spec JSON before proposing")
        run_paths = [str((src.parent / Path(item).expanduser()).resolve()) for item in runs]
        ref_raw = spec_payload.get("reference")
        ref_path = str((src.parent / Path(ref_raw).expanduser()).resolve()) if isinstance(ref_raw, str) and ref_raw.strip() else None
        return self._queue_spc(spec_payload, src, run_paths, ref_path)

    def _queue_spc(self, spec_payload, src, run_paths, ref_path):
        fingerprint = hashlib.sha256(json.dumps(spec_payload, sort_keys=True).encode()).hexdigest()
        with cache_store.file_lock(self.state_lock_path):
            existing = workspace_db.find_active_by_fingerprint(fingerprint, AUTO_STATES, research_dir=self.root)
            if existing:
                raise CliError("INVALID_INPUT", f"identical SPC request already active in run {existing['run_id']}",
                               f"inspect {existing['run_id']} for its progress")
            run_id = _allocate_run_id(self.runs_dir, spec_payload.get("name") or "spc", research_dir=self.root)
            doc = {
                "version": 4,
                "run_id": run_id,
                "name": spec_payload.get("name") or run_id,
                "track": "SPC",
                "design": {"mode": "spc"},
                "state": "AUTO_SPC",
                "selection_frozen": False,
                "request_fingerprint": fingerprint,
                "source_spec": str(src),
                "spc_inputs": {"runs": run_paths, "reference": ref_path},
                "created_at": _utc_now(),
                "arms": [],
                "records": [],
                "rows": [],
                "execution": {"total": len(run_paths), "not_submitted": len(run_paths),
                              "accepted": 0, "finished": 0, "terminal": 0},
            }
            self.save_run(doc, trigger="propose-spc")
        return public_run_view(doc)

    def inbox(self, *, gate=None, limit=20):
        """Return runs paused at interaction gates, active AUTO counts, and OPEN Labs tickets."""
        self._sync_legacy_runs()
        counts = workspace_db.run_counts(research_dir=self.root)
        gates = GATE_STATES if gate is None else (GATE_STATES & {gate})
        actionable = workspace_db.list_runs(states=gates, limit=limit, research_dir=self.root) if gates else []
        active_auto = workspace_db.list_runs(states=AUTO_STATES, limit=limit, research_dir=self.root)
        open_labs = labs.open_ticket_summary(self.root, limit=limit)
        return {
            "research_dir": str(self.root),
            "actionable": [_run_summary(doc) for doc in actionable],
            "actionable_total": sum(counts.get(state, 0) for state in gates),
            "active_auto": [_run_summary(doc) for doc in active_auto],
            "active_auto_total": sum(counts.get(state, 0) for state in AUTO_STATES),
            "open_labs_tickets": open_labs["tickets"],
            "open_labs_total": open_labs["count"],
            "counts_by_state": counts,
            "quota": _read_latest_quota_snapshot(self.root),
        }

    def list_runs(self, *, state=None, limit=50):
        """List runs optionally filtered by state."""
        docs = self.iter_runs(states={state} if state else None, limit=limit)
        counts = workspace_db.run_counts(research_dir=self.root)
        return {
            "research_dir": str(self.root),
            "count": counts.get(state, 0) if state else sum(counts.values()),
            "runs": [_run_summary(doc) for doc in docs],
        }

    def inspect_run(self, run_id, *, arm_label=None):
        """Return firewall-filtered evidence for one run and record test access."""
        return self.read_run_view(self.load_run_raw(run_id), tool="inspect", arm_label=arm_label)

    def read_run_view(self, doc, *, tool, arm_label=None):
        """Build a run view for an explicit inspection or targeted wait."""
        view = public_run_view(doc, arm_label=arm_label)
        if view["test_view_open"] and doc.get("track") != "SPC":
            research_paths.log_view(tool, "test", _view_alpha_ids(doc, arm_label),
                                    note=doc["run_id"], research_dir=self.root)
        return view

    def act(self, run_id, action, params=None):
        """Apply a validated model decision at a GATE_* state and advance to the next state."""
        params = dict(params or {})
        with cache_store.file_lock(self.state_lock_path):
            doc = self.load_run_raw(run_id)
            state = doc.get("state")
            if action == "request-labs" and state in labs.REQUEST_STATES:
                _transition_to_labs(doc, state, params, self.root)
                return public_run_view(doc)

            if state == "GATE_PREFLIGHT_BLOCKED":
                if action == "retry-preflight":
                    doc["unchanged_defers"] = 0
                    doc["stalled"] = False
                    doc["preflight_block"] = None
                    doc["next_wake_at"] = None
                    doc["state"] = "AUTO_PREFLIGHT"
                elif action == "drop":
                    doc["drop_reason"] = params.get("reason")
                    doc["state"] = "DROPPED"
                else:
                    raise CliError("INVALID_INPUT", f"action {action!r} invalid in {state}",
                                   f"use `retry-preflight`, `request-labs`, or `drop` on run {run_id}")

            elif state == "GATE_TRIAGE":
                if action == "hydrate-pnl":
                    selected = _select_triage_arms(doc, params)
                    doc["selected_arms"] = selected
                    doc["next_wake_at"] = None
                    doc["state"] = "AUTO_HYDRATE_PNL"
                elif action == "drop":
                    _archive_with_verdict(doc, params, self.root)
                else:
                    raise CliError("INVALID_INPUT", f"action {action!r} invalid in {state}",
                                   f"use `hydrate-pnl`, `request-labs`, or `drop` on run {run_id}")

            elif state == "GATE_APPROVE_FULL":
                if action == "approve-full":
                    approved = _freeze_full_selection(doc, params, self.runs_dir, self.root)
                    doc["approved_arms"] = approved
                    doc["selection_frozen"] = True
                    doc["next_wake_at"] = None
                    doc["state"] = "AUTO_FULL_VALIDATE"
                elif action == "back-to-triage":
                    doc["state"] = "GATE_TRIAGE"
                elif action == "drop":
                    _archive_with_verdict(doc, params, self.root)
                else:
                    raise CliError("INVALID_INPUT", f"action {action!r} invalid in {state}",
                                   f"use `approve-full`, `request-labs`, `back-to-triage`, or `drop` on run {run_id}")

            elif state == "GATE_LABS":
                if action == "drop":
                    _archive_with_verdict(doc, params, self.root)
                else:
                    ticket_id = doc.get("labs_ticket_id")
                    raise CliError("INVALID_INPUT", f"action {action!r} invalid in {state}",
                                   f"resolve with `research_cli.py labs resolve {ticket_id} --summary \"...\"` or drop the run")

            elif state == "GATE_DESCRIBE":
                if action == "describe":
                    if not doc.get("full_alpha_id") or not doc.get("selection_frozen") or not doc.get("selection_id"):
                        raise CliError("INVALID_INPUT", "candidate lacks frozen FULL validation evidence",
                                       "drop the run or propose a new evaluation with a verified train selection")
                    name = (params.get("name") or "").strip()
                    category = params.get("category")
                    tags = parse_tags(params.get("tags"))
                    description = params.get("description")
                    problems = buffer_cmd.validate_seed(name, category, tags, description)
                    if problems:
                        raise CliError(
                            "INVALID_INPUT",
                            "invalid candidate metadata: " + "; ".join(problems),
                            "use official Power Pool headings `Idea:`, `Rationale for data used:`, "
                            "`Rationale for operators used:` with `- ` bullets and >=100 chars",
                            details={"problems": problems},
                        )
                    doc["candidate_metadata"] = {
                        "name": name,
                        "category": category,
                        "tags": tags,
                        "description": description.strip(),
                    }
                    doc["next_wake_at"] = None
                    doc["state"] = "AUTO_BUFFER"
                elif action == "drop":
                    _archive_with_verdict(doc, params, self.root)
                else:
                    raise CliError("INVALID_INPUT", f"action {action!r} invalid in {state}",
                                   f"use `describe --name ... --category ... --tags ... --description ...` or `drop` on run {run_id}")

            elif state == "GATE_CONFIRM_SUBMIT":
                if action == "submit":
                    if params.get("yes") is not True:
                        raise CliError(
                            "CONFIRM_REQUIRED",
                            f"submission for run {run_id} requires explicit --yes confirmation",
                            f"repeat `research_cli.py act {run_id} submit --yes` to authorize final platform submission",
                        )
                    if not ml.agent_submit_allowed():
                        raise CliError(
                            "SUBMIT_DISABLED",
                            "real alpha submission is disabled by submission policy",
                            f"set {ml.SUBMIT_UNLOCK_ENV}=true when authorized to submit, or run `hold` to keep on shelf",
                        )
                    doc["next_wake_at"] = None
                    doc["state"] = "AUTO_SUBMITTING"
                elif action == "hold":
                    doc["state"] = "SHELVED"
                else:
                    raise CliError("INVALID_INPUT", f"action {action!r} invalid in {state}",
                                   f"use `submit --yes` or `hold` on run {run_id}")

            else:
                raise CliError("INVALID_INPUT", f"run {run_id} is in state {state} and does not accept gate actions",
                               f"inspect allowed actions via `research_cli.py inspect {run_id}`")

            self.save_run(doc, from_state=state, trigger=f"act:{action}", detail=params.get("reason"))
            if action == "approve-full":
                research_paths.log_view("approve-full", "test", _view_alpha_ids(doc),
                                        note=run_id, research_dir=self.root)
            return public_run_view(doc)

    def cancel(self, run_id, *, reason=None):
        """Cancel a run, scheduling platform simulation cancellation if live tasks exist."""
        with cache_store.file_lock(self.state_lock_path):
            doc = self.load_run_raw(run_id)
            state = doc.get("state")
            if state in TERMINAL_STATES:
                return public_run_view(doc)
            if state == "AUTO_SUBMITTING" and (doc.get("submit_attempt") or {}).get("post_attempted"):
                raise CliError("INVALID_INPUT", "submission has already entered platform reconciliation",
                               "inspect this run for the platform submission outcome")
            doc["cancel_reason"] = reason
            doc["next_wake_at"] = None
            # A worker may be between sending a request and persisting its receipt.
            doc["state"] = "AUTO_CANCELLING" if state in {"AUTO_QUICK_SIM", "AUTO_FULL_VALIDATE", "AUTO_CANCELLING"} or _has_live_simulation(doc) else "CANCELLED"
            self.save_run(doc, from_state=state, trigger="cancel", detail=reason)
            return public_run_view(doc)

    def migrate_workspace(self, *, dry_run=False):
        """Import legacy v3 ``backlog/`` and ``backlog-results/`` state into v4 ``runs/``."""
        inbox = self.root / "backlog"
        results = self.root / "backlog-results"
        migrated = []
        skipped = []
        seen_batches = set()

        with cache_store.file_lock(self.state_lock_path):
            db_docs = workspace_db.list_runs(research_dir=self.root)
            legacy_docs = [doc for _, doc in workspace_db.legacy_run_documents(self.runs_dir)]
            known_fp = {doc["request_fingerprint"] for doc in db_docs + legacy_docs if doc.get("request_fingerprint")}
            batch_files = sorted(results.glob("*-batch.json")) if results.is_dir() else []
            for batch_path in batch_files:
                seen_batches.add(batch_path.name)
                raw, err = cache_store.load_json(batch_path)
                if err or not isinstance(raw, dict):
                    skipped.append({"path": str(batch_path), "reason": err or "invalid"})
                    continue
                fp = raw.get("request_fingerprint")
                if fp and fp in known_fp:
                    skipped.append({"path": str(batch_path), "reason": "already_migrated"})
                    continue
                arms = raw.get("arms") or []
                records = raw.get("records") or []
                rows = raw.get("rows") or []
                complete = bool(raw.get("complete"))
                has_full = any(((r.get("settings") or {}).get("simulationMode") or "").upper() == "FULL" for r in records)
                exec_summary = raw.get("execution") or {}
                if has_full:
                    target_state = "ARCHIVED"
                elif not complete:
                    target_state = "AUTO_QUICK_SIM" if (exec_summary.get("accepted") or exec_summary.get("not_submitted")) else "FAILED"
                elif any(type(r.get("train_sharpe")) in (int, float) for r in rows):
                    target_state = "GATE_TRIAGE"
                else:
                    target_state = "FAILED"

                stem = batch_path.name.removesuffix("-batch.json")
                run_id = _allocate_run_id(self.runs_dir, f"migrated-{stem}", research_dir=self.root)
                now = _utc_now()
                doc = {
                    "version": 4,
                    "run_id": run_id,
                    "name": raw.get("name") or stem,
                    "track": raw.get("track") or "REGULAR",
                    "design": raw.get("design") or {"mode": "explore"},
                    "state": target_state,
                    "selection_frozen": False,
                    "request_fingerprint": fp or hashlib.sha256(str(batch_path).encode()).hexdigest(),
                    "idea": raw.get("idea"),
                    "mechanism": raw.get("mechanism"),
                    "migrated_from": str(batch_path),
                    "created_at": now,
                    "updated_at": now,
                    "next_wake_at": None,
                    "unchanged_defers": 0,
                    "stalled": False,
                    "arms": arms,
                    "records": records,
                    "rows": rows,
                    "execution": exec_summary,
                }
                if has_full:
                    doc["migration_note"] = "Archived legacy FULL results without a verified frozen train selection."
                if not dry_run:
                    self.save_run(doc, trigger="migrate-workspace", detail={"source": str(batch_path)})
                known_fp.add(doc["request_fingerprint"])
                migrated.append({"run_id": run_id, "state": target_state, "source": str(batch_path)})

            if inbox.is_dir():
                for spec_path in sorted(inbox.glob("*.json")):
                    stem = spec_path.stem
                    if f"{stem}-batch.json" in seen_batches:
                        continue
                    spec_doc, err = cache_store.load_json(spec_path)
                    if err or not isinstance(spec_doc, dict):
                        skipped.append({"path": str(spec_path), "reason": err or "invalid"})
                        continue
                    try:
                        track = research_route.normalize_route(spec_doc)["track"]
                        if track == "SPC":
                            skipped.append({"path": str(spec_path), "reason": "propose this SPC spec separately"})
                            continue
                        view = self._propose_unlocked(spec_doc, source_path=spec_path, track=track) if not dry_run else {
                            "run_id": f"dry-run-{stem}", "state": "AUTO_PREFLIGHT"
                        }
                        migrated.append({"run_id": view["run_id"], "state": view["state"], "source": str(spec_path)})
                    except (CliError, ValueError) as exc:
                        skipped.append({"path": str(spec_path), "reason": str(exc)})

            archive_root = self.root / "legacy-v3-archive"
            runs_json_res = workspace_db.import_legacy_run_files(
                self.runs_dir,
                research_dir=self.root,
                dry_run=dry_run,
            )
            archived_to = None
            if not dry_run:
                moved_any = False
                for legacy_name in ("backlog", "backlog-results", "backlog-claimed", "backlog-consumed"):
                    src_dir = self.root / legacy_name
                    if src_dir.exists():
                        archive_root.mkdir(parents=True, exist_ok=True)
                        dest = archive_root / legacy_name
                        if dest.exists():
                            dest = archive_root / f"{legacy_name}-{uuid.uuid4().hex[:8]}"
                        shutil.move(str(src_dir), str(dest))
                        moved_any = True
                if moved_any:
                    archived_to = str(archive_root)

        pnl_res = pnl_store.migrate_legacy_pnl_dir(
            self.root / "pnl",
            archive_dir=archive_root / "pnl-json",
            dry_run=dry_run,
        )
        catalog_res = ml.migrate_legacy_datafield_cache(
            self.root / "cache" / "metadata", archive_dir=archive_root / "datafields-json", dry_run=dry_run,
        )
        slot_dir = self.root / "simulation-slots"
        slots_migrated = 0
        if slot_dir.is_dir():
            for spath in sorted(slot_dir.glob("*.json")):
                if spath.name.startswith("."):
                    continue
                sraw, serr = cache_store.load_json(spath)
                if not serr and isinstance(sraw, dict) and sraw.get("account_fingerprint"):
                    archive_path = archive_root / "simulation-slots-json" / spath.name
                    if dry_run or simulation_slots.import_legacy_journal(
                        spath, research_dir=self.root, archive_path=archive_path,
                    ):
                        slots_migrated += 1

        if not dry_run and (pnl_res.get("migrated") or catalog_res.get("migrated") or slots_migrated):
            archived_to = str(archive_root)

        return {
            "dry_run": bool(dry_run),
            "migrated": migrated,
            "skipped": skipped,
            "runs_json_migrated": runs_json_res.get("imported", 0),
            "pnl_json_migrated": pnl_res.get("migrated", 0),
            "slots_json_migrated": slots_migrated,
            "datafields_json_migrated": catalog_res.get("migrated", 0),
            "datafields_json_skipped": catalog_res.get("skipped", 0),
            "archived_to": archived_to,
        }


def _view_alpha_ids(doc, arm_label=None):
    alpha_ids = [row.get("alpha_id") for row in doc.get("approved_arms") or []
                 if arm_label is None or row.get("label") == arm_label]
    if alpha_ids and doc.get("full_alpha_id"):
        alpha_ids.append(doc["full_alpha_id"])
    return list(dict.fromkeys(alpha_ids))


def _has_live_simulation(doc):
    terminal = alpha_registry.FINISHED_STATUSES | simulation_slots.TERMINAL_FAILURE_STATUSES | {"FAILED"}
    return any(
        (rec.get("progress_url") or rec.get("multi_parent_url")) and
        str(rec.get("status") or "").upper() not in terminal
        for rec in (doc.get("records") or []) + (doc.get("full_records") or [])
    )


def _select_triage_arms(doc, params):
    rows = doc.get("rows") or []
    arms_by_label = {arm["label"]: arm for arm in (doc.get("arms") or []) if "label" in arm}
    explicit = params.get("arms")
    if explicit:
        wanted = [label.strip() for label in (explicit if isinstance(explicit, list) else str(explicit).split(",")) if label.strip()]
        row_map = {row.get("label"): row for row in rows}
        selected = []
        for label in wanted:
            row = row_map.get(label)
            if row is None or not row.get("alpha_id"):
                raise CliError("INVALID_INPUT", f"arm {label!r} is missing or lacks a completed alpha_id",
                               f"inspect completed arms with `research_cli.py inspect {doc['run_id']}`")
            if row.get("role") == "diagnostic" or (arms_by_label.get(label) or {}).get("design") == "control":
                raise CliError("INVALID_INPUT", f"arm {label!r} is diagnostic/control and cannot enter FULL confirmation",
                               "select non-diagnostic candidate arms for `hydrate-pnl`")
            selected.append({"label": label, "alpha_id": row["alpha_id"],
                             "train_sharpe": row.get("train_sharpe"), "train_fitness": row.get("train_fitness")})
        return selected

    top_n = int(params.get("top") or 3)
    eligible = []
    for row in rows:
        label = row.get("label")
        arm = arms_by_label.get(label) or {}
        if row.get("role") == "diagnostic" or arm.get("design") == "control":
            continue
        sharpe, fitness = row.get("train_sharpe"), row.get("train_fitness")
        if row.get("alpha_id") and type(sharpe) in (int, float) and type(fitness) in (int, float) and sharpe > 0 and fitness > 0:
            eligible.append({"label": label, "alpha_id": row["alpha_id"],
                             "train_sharpe": sharpe, "train_fitness": fitness})
    eligible.sort(key=lambda item: (-item["train_sharpe"], str(item["label"])))
    if not eligible:
        raise CliError("INVALID_INPUT", "no positive-train candidate arms available to hydrate",
                       f"drop this run with `research_cli.py act {doc['run_id']} drop --verdict falsified`")
    return eligible[:max(1, top_n)]


def _freeze_full_selection(doc, params, runs_dir, root):
    selected = doc.get("selected_arms") or []
    if not selected:
        raise CliError("INVALID_INPUT", "no hydrated arms available to freeze for FULL confirmation",
                       "run `hydrate-pnl` at GATE_TRIAGE before `approve-full`")
    hydration = (doc.get("hydration") or {}).get("by_label") or {}
    passing = [item for item in selected if (hydration.get(item["label"]) or {}).get("gate_pass") is True
               and hydration[item["label"]].get("alpha_id") == item.get("alpha_id")]
    explicit = params.get("arms")
    if explicit:
        labels = explicit if isinstance(explicit, list) else str(explicit).split(",")
        wanted = [label.strip() for label in labels if isinstance(label, str) and label.strip()]
        if len(wanted) != 1:
            raise CliError("INVALID_INPUT", "FULL approval requires exactly one arm",
                           "pass `approve-full --arms LABEL` with one arm that passed hydration")
        passing = [item for item in passing if item["label"] == wanted[0]]
    if not passing:
        raise CliError("INVALID_INPUT", "no requested arm passed the train-only PnL gate",
                       "inspect hydration evidence, hydrate another arm, or drop this run")
    chosen = passing[:1]

    arms_by_label = {arm["label"]: arm for arm in (doc.get("arms") or []) if "label" in arm}
    freeze_rows = []
    for item in chosen:
        arm = arms_by_label.get(item["label"]) or {}
        freeze_rows.append({
            "alpha_id": item["alpha_id"],
            "label": item["label"],
            "role": arm.get("role"),
            "expression": arm.get("expression"),
            "settings": arm.get("settings"),
            "train_sharpe": item.get("train_sharpe"),
            "train_fitness": item.get("train_fitness"),
        })

    sel_path = runs_dir / f"{doc['run_id']}-selection.json"
    if sel_path.exists():
        sel_path = runs_dir / f"{doc['run_id']}-selection-{uuid.uuid4().hex[:6]}.json"
    manifest = selection.freeze(
        sel_path,
        source="triage",
        rows=freeze_rows,
        criteria={"min_train_sharpe": 1.0, "mode": "state_machine_gate"},
        inputs={"run_id": doc["run_id"], "research_dir": str(root),
                "registry_path": str(Path(root) / "alpha-registry.jsonl")},
        window={"train_only": True},
    )
    doc["selection_id"] = manifest["selection_id"]
    doc["selection_path"] = str(sel_path)
    return chosen


def _archive_with_verdict(doc, params, root):
    verdict = (params.get("verdict") or "rejected").strip().lower()
    if verdict not in _DROP_VERDICTS:
        raise CliError("INVALID_INPUT", f"verdict must be one of {sorted(_DROP_VERDICTS)}, got {verdict!r}",
                       "pass --verdict rejected, falsified, or no_go")
    reason = params.get("reason")
    doc["drop_reason"] = reason
    doc["drop_verdict"] = verdict
    registry_path = str(Path(root) / "alpha-registry.jsonl")
    latest = {e.get("alpha_id"): e for e in alpha_registry.load_entries(registry_path) if e.get("alpha_id")} if Path(registry_path).exists() else {}
    for row in doc.get("rows") or []:
        alpha_id = row.get("alpha_id")
        if alpha_id and alpha_id in latest:
            alpha_registry.update_entry(alpha_id, {"verdict": verdict, "note": reason}, path=registry_path)
    doc["state"] = "ARCHIVED"


def _transition_to_labs(doc, from_state, params, root):
    (region, universe, delay), fields = labs.select_request_scope(doc.get("arms") or [], params.get("fields"))
    ticket = labs.build_ticket({
        "trigger_rule": params.get("trigger") or "anomaly",
        "region": region,
        "universe": universe,
        "delay": delay,
        "dataset": params.get("dataset"),
        "fields": fields,
        "questions": params.get("question"),
        "expected_metrics": params.get("expect"),
        "branches": params.get("branch"),
        "source_run_id": doc["run_id"],
    })
    doc.update(labs_ticket_id=ticket["ticket_id"], pre_labs_state=from_state, labs_resolution=None,
               next_wake_at=None, state="GATE_LABS", updated_at=_utc_now())
    workspace_db.insert_labs_ticket(ticket, run=doc, from_state=from_state, research_dir=root)


def _read_latest_quota_snapshot(root):
    return workspace_db.latest_quota_snapshot(research_dir=root, legacy_dir=Path(root) / "simulation-slots")
