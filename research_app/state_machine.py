"""Persistent 4-gate state machine model, protocol firewall, and workspace migration."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import alpha_registry
import cache_store
import fastexpr_ast
import machine_lib as ml
import research_paths
import simulation_slots
import spc_experiment

from . import brief, buffer_cmd, experiments, research_route, selection, validation
from .runtime import CliError, parse_tags

AUTO_STATES = frozenset({
    "AUTO_PREFLIGHT",
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
    "out_of_sample",
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
            {"action": "drop", "command": f"research_cli.py act {run_id} drop --reason \"...\""},
        ]
    if state == "GATE_TRIAGE":
        return [
            {"action": "hydrate-pnl",
             "command": f"research_cli.py act {run_id} hydrate-pnl --top 3"},
            {"action": "drop",
             "command": f"research_cli.py act {run_id} drop --verdict falsified --reason \"...\""},
        ]
    if state == "GATE_APPROVE_FULL":
        return [
            {"action": "approve-full",
             "command": f"research_cli.py act {run_id} approve-full"},
            {"action": "back-to-triage",
             "command": f"research_cli.py act {run_id} back-to-triage"},
            {"action": "drop",
             "command": f"research_cli.py act {run_id} drop --verdict rejected --reason \"...\""},
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


def _redact_record_if_pre_freeze(record):
    if not isinstance(record, dict):
        return record
    cleaned = {k: v for k, v in record.items() if k not in _REDACT_ROW_KEYS}
    split = record.get("out_of_sample")
    if isinstance(split, dict) and "train" in split:
        cleaned["out_of_sample"] = {"train": split["train"]}
    return cleaned


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
            item = {k: v for k, v in row.items() if k not in _REDACT_ROW_KEYS}
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
        "preflight_block": doc.get("preflight_block"),
        "last_error": doc.get("last_error"),
        "rows": rows,
        "allowed_actions": _allowed_actions(doc),
    }
    if not pre_freeze:
        view["selection_path"] = doc.get("selection_path")
        view["validation"] = doc.get("validation")
        view["cost_audit"] = doc.get("cost_audit")
        view["full_checks"] = doc.get("full_checks")
        view["candidate_metadata"] = doc.get("candidate_metadata")
        view["shelf_entry"] = doc.get("shelf_entry")
        view["submit_attempt"] = doc.get("submit_attempt")
    else:
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


def _allocate_run_id(runs_dir, name):
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(name or "run")).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)[:28] or "run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    for _ in range(16):
        suffix = uuid.uuid4().hex[:6]
        run_id = f"{slug}-{stamp}-{suffix}"
        if not (runs_dir / f"{run_id}.json").exists():
            return run_id
    return f"run-{uuid.uuid4().hex[:12]}"


class StateMachineEngine:
    """Durable, single-writer state machine managing experiment runs under a research directory."""

    def __init__(self, research_dir=None):
        """Initialize workspace directories under *research_dir* without spawning background tasks."""
        self.root = Path(research_dir or research_paths.ensure_dir()).expanduser().resolve()
        self.runs_dir = self.root / "runs"
        self.daemon_dir = self.root / "daemon"
        self.events_path = self.runs_dir / "events.jsonl"
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
        """Return daemon scheduling configuration merged over defaults."""
        raw, err = cache_store.load_json(self.config_path, default={})
        cfg = dict(DEFAULT_DAEMON_CONFIG)
        if not err and isinstance(raw, dict):
            for key in DEFAULT_DAEMON_CONFIG:
                if isinstance(raw.get(key), (int, float)) and raw[key] >= 0:
                    cfg[key] = raw[key]
        return cfg

    def update_config(self, updates):
        """Persist updated scheduling config fields atomically under state lock."""
        with cache_store.file_lock(self.state_lock_path):
            cfg = self.load_config()
            for key, value in (updates or {}).items():
                if key not in DEFAULT_DAEMON_CONFIG:
                    raise CliError("INVALID_INPUT", f"unknown config key {key!r}",
                                   f"allowed keys: {', '.join(sorted(DEFAULT_DAEMON_CONFIG))}")
                if value is None:
                    continue
                if type(value) not in (int, float) or value < 0:
                    raise CliError("INVALID_INPUT", f"{key} must be a non-negative number",
                                   f"pass a numeric value >= 0 for {key}")
                cfg[key] = int(value)
            cache_store.write_json_atomic(self.config_path, cfg)
            return cfg

    def load_run_raw(self, run_id):
        """Load raw run document for *run_id*, raising NOT_FOUND if absent."""
        path = self.run_path(run_id)
        doc, err = cache_store.load_json(path)
        if err == "missing":
            raise CliError("NOT_FOUND", f"run {run_id!r} does not exist",
                           "run `research_cli.py runs` or `inbox` to list available runs")
        if err or not isinstance(doc, dict):
            raise CliError("INVALID_INPUT", f"run {run_id!r} file is unreadable: {err}",
                           f"inspect {path} for JSON corruption")
        return doc

    def save_run(self, doc, *, from_state=None, trigger=None, detail=None):
        """Atomically persist *doc* and append a transition event when state changed."""
        doc["updated_at"] = _utc_now()
        path = self.run_path(doc["run_id"])
        cache_store.write_json_atomic(path, doc, indent=2)
        to_state = doc.get("state")
        if from_state is not None and from_state != to_state:
            self._append_event(doc["run_id"], from_state, to_state, trigger or "transition", detail)
        return doc

    def _append_event(self, run_id, from_state, to_state, trigger, detail=None):
        event = {
            "at": _utc_now(),
            "run_id": run_id,
            "from_state": from_state,
            "to_state": to_state,
            "trigger": trigger,
        }
        if detail is not None:
            event["detail"] = detail
        line = json.dumps(event, ensure_ascii=False) + "\n"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def iter_runs(self):
        """Yield raw run documents stored under ``runs/``."""
        if not self.runs_dir.is_dir():
            return []
        docs = []
        for path in sorted(self.runs_dir.glob("*.json")):
            if path.name.endswith("-selection.json") or path.name.startswith("."):
                continue
            doc, err = cache_store.load_json(path)
            if not err and isinstance(doc, dict) and doc.get("run_id"):
                docs.append(doc)
        return docs

    def propose(self, spec_payload, *, source_path=None):
        """Validate a spec synchronously without network calls and create a run in AUTO_PREFLIGHT."""
        with cache_store.file_lock(self.state_lock_path):
            return self._propose_unlocked(spec_payload, source_path=source_path)

    def _propose_unlocked(self, spec_payload, *, source_path=None):
        if isinstance(spec_payload, (str, bytes)):
            try:
                spec_payload = json.loads(spec_payload)
            except ValueError as exc:
                raise CliError("INVALID_INPUT", f"spec must be valid JSON: {exc}",
                               "fix JSON syntax before calling `propose`") from exc
        if not isinstance(spec_payload, dict):
            raise CliError("INVALID_INPUT", "spec must be a JSON object",
                           "see `research_cli.py schema propose` for the required spec shape")

        spec_path_str = str(source_path or "-")
        route = research_route.normalize_route(spec_payload)
        track = route["track"]

        if track == "SPC":
            return self._propose_spc(spec_payload, source_path)

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
        _static_validate_arms(arms, source_dir=Path(source_path).parent if source_path else None)
        fingerprint = experiments._request_fingerprint(arms)

        for existing in self.iter_runs():
            if existing.get("state") in (AUTO_STATES | GATE_STATES) and existing.get("request_fingerprint") == fingerprint:
                eid = existing["run_id"]
                raise CliError(
                    "INVALID_INPUT",
                    f"identical request already active in run {eid}",
                    f"run `research_cli.py inspect {eid}` instead of proposing a duplicate",
                    details={"run_id": eid, "state": existing.get("state")},
                )

        run_id = _allocate_run_id(self.runs_dir, normalized.get("name") or (Path(source_path).stem if source_path else "run"))
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
        cache_store.write_json_atomic(self.run_path(run_id), doc, indent=2)
        self._append_event(run_id, None, "AUTO_PREFLIGHT", "propose", {"arms": len(arms), "track": track})
        return public_run_view(doc)

    def _propose_spc(self, spec_payload, source_path):
        run_id = _allocate_run_id(self.runs_dir, spec_payload.get("name") or "spc")
        now = _utc_now()
        if source_path is None:
            raise CliError("INVALID_INPUT", "SPC `propose` requires a spec file path containing relative run files",
                           "pass `research_cli.py propose SPEC.json`")
        src = Path(source_path).expanduser().resolve()
        runs = spec_payload.get("runs")
        if not isinstance(runs, list) or not runs:
            raise CliError("INVALID_INPUT", "SPC spec requires a non-empty `runs` list",
                           "set `runs` in the SPC spec JSON before proposing")
        run_paths = [(src.parent / Path(item).expanduser()).resolve() for item in runs]
        ref_raw = spec_payload.get("reference")
        ref_path = (src.parent / Path(ref_raw).expanduser()).resolve() if isinstance(ref_raw, str) and ref_raw.strip() else None
        report = spc_experiment.experiment(src, run_paths, reference_path=ref_path)
        state = "COMPLETED" if report.get("ok") else "FAILED"
        doc = {
            "version": 4,
            "run_id": run_id,
            "name": spec_payload.get("name") or run_id,
            "track": "SPC",
            "design": {"mode": "spc"},
            "state": state,
            "selection_frozen": True,
            "request_fingerprint": hashlib.sha256(json.dumps(spec_payload, sort_keys=True).encode()).hexdigest(),
            "source_spec": str(src),
            "created_at": now,
            "updated_at": now,
            "experiment": report,
            "arms": [],
            "records": [],
            "rows": [],
            "execution": {"total": len(run_paths), "not_submitted": 0, "accepted": len(run_paths),
                          "finished": len(run_paths), "terminal": 0 if report.get("ok") else 1},
        }
        cache_store.write_json_atomic(self.run_path(run_id), doc, indent=2)
        self._append_event(run_id, None, state, "propose-spc", {"ok": report.get("ok")})
        return public_run_view(doc)

    def inbox(self, *, gate=None, limit=20):
        """Return runs paused at interaction gates plus active background run counts."""
        docs = self.iter_runs()
        docs.sort(key=lambda d: d.get("updated_at") or "", reverse=True)
        counts = {}
        actionable = []
        active_auto = []
        for doc in docs:
            st = doc.get("state") or "UNKNOWN"
            counts[st] = counts.get(st, 0) + 1
            if st in GATE_STATES:
                if gate is None or st == gate:
                    actionable.append(_run_summary(doc))
            elif st in AUTO_STATES:
                active_auto.append(_run_summary(doc))
        return {
            "research_dir": str(self.root),
            "actionable": actionable[:limit],
            "actionable_total": len(actionable),
            "active_auto": active_auto[:limit],
            "active_auto_total": len(active_auto),
            "counts_by_state": counts,
            "quota": _read_latest_quota_snapshot(self.root),
        }

    def list_runs(self, *, state=None, limit=50):
        """List runs optionally filtered by state."""
        docs = self.iter_runs()
        if state:
            docs = [d for d in docs if d.get("state") == state]
        docs.sort(key=lambda d: d.get("updated_at") or "", reverse=True)
        return {
            "research_dir": str(self.root),
            "count": len(docs),
            "runs": [_run_summary(d) for d in docs[:limit]],
        }

    def inspect_run(self, run_id, *, arm_label=None):
        """Return firewall-filtered inspection payload for one run."""
        return public_run_view(self.load_run_raw(run_id), arm_label=arm_label)

    def act(self, run_id, action, params=None):
        """Apply a validated model decision at a GATE_* state and advance to the next state."""
        params = dict(params or {})
        with cache_store.file_lock(self.state_lock_path):
            doc = self.load_run_raw(run_id)
            state = doc.get("state")

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
                                   f"use `retry-preflight` or `drop` on run {run_id}")

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
                                   f"use `hydrate-pnl` or `drop` on run {run_id}")

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
                                   f"use `approve-full`, `back-to-triage`, or `drop` on run {run_id}")

            elif state == "GATE_DESCRIBE":
                if action == "describe":
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
            return public_run_view(doc)

    def cancel(self, run_id, *, reason=None):
        """Cancel a run, scheduling platform simulation cancellation if live tasks exist."""
        with cache_store.file_lock(self.state_lock_path):
            doc = self.load_run_raw(run_id)
            state = doc.get("state")
            if state in TERMINAL_STATES:
                return public_run_view(doc)
            live = any(
                (rec.get("progress_url") or rec.get("multi_parent_url"))
                and str(rec.get("status") or "").upper() not in (alpha_registry.FINISHED_STATUSES | simulation_slots.TERMINAL_FAILURE_STATUSES | {"FAILED"})
                for rec in (doc.get("records") or []) + (doc.get("full_records") or [])
            )
            doc["cancel_reason"] = reason
            doc["next_wake_at"] = None
            doc["state"] = "AUTO_CANCELLING" if live else "CANCELLED"
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
            def _existing_fingerprints():
                return {d.get("request_fingerprint") for d in self.iter_runs() if d.get("request_fingerprint")}

            known_fp = _existing_fingerprints()
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
                if not complete:
                    target_state = "AUTO_QUICK_SIM" if (exec_summary.get("accepted") or exec_summary.get("not_submitted")) else "FAILED"
                elif has_full and any(r.get("alpha_id") for r in rows):
                    target_state = "GATE_DESCRIBE"
                elif any(type(r.get("train_sharpe")) in (int, float) for r in rows):
                    target_state = "GATE_TRIAGE"
                else:
                    target_state = "FAILED"

                stem = batch_path.name.removesuffix("-batch.json")
                run_id = _allocate_run_id(self.runs_dir, f"migrated-{stem}")
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
                if not dry_run:
                    cache_store.write_json_atomic(self.run_path(run_id), doc, indent=2)
                    self._append_event(run_id, None, target_state, "migrate-workspace", {"source": str(batch_path)})
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
                        view = self._propose_unlocked(spec_doc, source_path=spec_path) if not dry_run else {
                            "run_id": f"dry-run-{stem}", "state": "AUTO_PREFLIGHT"
                        }
                        migrated.append({"run_id": view["run_id"], "state": view["state"], "source": str(spec_path)})
                    except CliError as exc:
                        skipped.append({"path": str(spec_path), "reason": exc.message})

            archived_to = None
            if not dry_run:
                archive_root = self.root / "legacy-v3-archive"
                moved_any = False
                for legacy_name in ("backlog", "backlog-results", "backlog-claimed", "backlog-consumed"):
                    src_dir = self.root / legacy_name
                    if src_dir.exists():
                        archive_root.mkdir(parents=True, exist_ok=True)
                        dest = archive_root / legacy_name
                        if dest.exists():
                            dest = archive_root / f"{legacy_name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
                        shutil.move(str(src_dir), str(dest))
                        moved_any = True
                if moved_any:
                    archived_to = str(archive_root)

        return {"dry_run": bool(dry_run), "migrated": migrated, "skipped": skipped, "archived_to": archived_to}


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
    explicit = params.get("arms")
    if explicit:
        wanted = {label.strip() for label in (explicit if isinstance(explicit, list) else str(explicit).split(",")) if label.strip()}
        chosen = [item for item in selected if item["label"] in wanted]
        if len(chosen) != len(wanted):
            raise CliError("INVALID_INPUT", "one or more requested arms were not hydrated in GATE_APPROVE_FULL",
                           "choose arms listed in `selected_arms`")
    else:
        hydration = (doc.get("hydration") or {}).get("by_label") or {}
        passing = [item for item in selected if (hydration.get(item["label"]) or {}).get("gate_pass", True)]
        chosen = passing or selected

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
        inputs={"run_id": doc["run_id"], "research_dir": str(root)},
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


def _read_latest_quota_snapshot(root):
    slot_dir = Path(root) / "simulation-slots"
    if not slot_dir.is_dir():
        return None
    latest = None
    for path in sorted(slot_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        raw, err = cache_store.load_json(path)
        if err or not isinstance(raw, dict):
            continue
        quota = raw.get("quota")
        if isinstance(quota, dict) and quota.get("observed_at") is not None:
            if latest is None or quota["observed_at"] > latest.get("observed_at", 0):
                latest = quota
    return latest
