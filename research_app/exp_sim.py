"""Standalone simulation execution, checkpointing, resumption, and ledger recording."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import threading
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

import simulation_multi
from . import experiments as _exp
from .runtime import CliError

def _write_simulation_checkpoint(path, checkpoint):
    _exp.cache_store.write_json_atomic(path, checkpoint)

_CHECKPOINT_EVENTS = {"submitting", "submitted", "pending", "completed", "failed", "deferred"}

_CHECKPOINT_POLL_FIELDS = ("progress_url", "next_poll_at", "poll_after_seconds", "metrics_pending")

_MULTI_CHECKPOINT_EVENTS = {"multi_submitting", "multi_submitted", "multi_pending"}

_MULTI_CHECKPOINT_FIELDS = ("multi_request_id", "multi_member", "multi_size", "multi_parent_url",
                            "multi_child_id", "multi_parent_terminal", "multi_parent_status",
                            "multi_terminal_evidence", "multi_failure_attribution",
                            *simulation_multi.PARENT_TIMING_FIELDS)

def _merge_checkpoint_record(previous, payload):
    updated = {**previous, **{key: value for key, value in payload.items() if key != "index"}}
    if "multi_request_id" in payload and payload["multi_request_id"] is None:
        for key in (*_exp.simulation_multi.STATE_KEYS, *_exp.simulation_multi.PARENT_TIMING_FIELDS):
            updated.pop(key, None)
    if payload.get("alpha_id") or payload.get("status") in _exp.simulation_slots.TERMINAL_STATUSES:
        # Once a poll yields an Alpha ID, subsequent metrics reads use that ID.
        # Missing transport markers in the new event supersede the old ones.
        for key in (*_exp._CHECKPOINT_POLL_FIELDS, "simulation_status", "simulation_error"):
            if key not in payload:
                updated.pop(key, None)
    return updated

def _checkpoint_event_updates(kind, payload, size):
    if kind in _exp._MULTI_CHECKPOINT_EVENTS:
        rows = payload["records"]
        indices = [row.get("index") for row in rows]
        if (not indices or any(type(index) is not int or not 0 <= index < size for index in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("invalid native Multi checkpoint event indices")
        return zip(indices, rows)
    index = payload.get("index")
    if kind in _exp._CHECKPOINT_EVENTS and type(index) is int and 0 <= index < size:
        return ((index, payload),)
    return ()

def _commit_checkpoint_records(document, updates, save):
    """Write all replacements before publishing any of them to the live batch."""
    if not updates:
        return
    records = document["records"]
    pending = records.copy()
    for index, record in updates.items():
        pending[index] = record
    checkpoint = {**document, "records": pending}
    save(checkpoint)
    # Active callers keep a reference to the records list while scheduling work.
    records[:] = pending
    document.update({**checkpoint, "records": records})

def _simulation_checkpoint_callback(path, checkpoint, sink):
    lock = threading.Lock()

    def on_event(kind, payload):
        with lock:
            records = checkpoint["records"]
            updates = {}
            for index, row in _exp._checkpoint_event_updates(kind, payload, len(records)):
                previous = records[index]
                updated = _exp._merge_checkpoint_record(previous, row)
                updated["role"] = _exp.alpha_registry.evidence_role(previous, row)
                if updated != previous:
                    updates[index] = updated
            _exp._commit_checkpoint_records(checkpoint, updates, lambda pending: _exp._write_simulation_checkpoint(path, pending))
            if sink:
                sink(kind, payload)

    return on_event

def _sort_simulation_results(records, args):
    for record in records:
        _exp._compact_metrics(record)
    if args.sort != "none":
        records.sort(key=lambda r: _exp._metric_value(r, args.sort), reverse=True)
    if args.top is not None:
        records = records[:args.top]
    return records

def _simulation_result(records, settings, summary, checkpoint_path):
    if summary["pending"] or summary["queued"]:
        status = "PENDING"
    elif summary["failed"]:
        status = "PARTIAL" if summary["completed"] else "FAILED"
    else:
        status = "COMPLETE"
    result = {
        "status": status,
        "dry_run": False,
        "settings": settings,
        "summary": summary,
        "results": records,
    }
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        result["checkpoint"] = str(Path(checkpoint_path))
    return result

def _simulation_checkpoint_document(settings, pairs, batch_size, labels, families, candidates, roles):
    # The label columns must be written here, not only in the dry-run branch.
    # These records are handed to simulate_alphas as initial_records, so anything
    # omitted from this dict is lost for the whole real run: mechanism tags
    # arrived empty in the ledger for exactly this reason.
    records = [
        {
            "expression": expression,
            "decay": decay,
            "candidate_id": candidates[index],
            "role": roles[index],
            "mechanism": labels[index],
            "family": families[index],
            "alpha_id": None,
            "status": "QUEUED",
            "metrics": None,
            "checks": None,
            "error": None,
        }
        for index, (expression, decay) in enumerate(pairs)
    ]
    return {
        "format": "worldquantuser.simulation-checkpoint",
        "version": 1,
        "settings": settings,
        "concurrency": batch_size,
        "records": records,
    }

def _checkpoint_lock_details(path):
    """Describe contention without treating an old lock file as a live lease."""
    path = Path(path)
    lock_path = path.with_name(path.name + ".lock")
    details = {"reason": "batch_locked", "batch_path": str(path),
               "checkpoint": str(path), "lock_path": str(lock_path),
               "next_action": "inspect_existing_writer"}
    try:
        with lock_path.open(encoding="utf-8") as handle:
            pid, stamp = handle.readline(200).strip().split(" ", 1)
        if pid.isdigit():
            datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            details["owner"] = {"pid": int(pid), "acquired_at": stamp, "observed": True}
    except (OSError, ValueError):
        pass
    return details

@contextmanager
def _locked_simulation_checkpoint(path, args):
    lock_path = path.with_name(path.name + ".lock")
    _exp.protect_output(args, path, lock_path)
    try:
        with _exp.cache_store.file_lock(lock_path, timeout=0):
            try:
                yield
            except KeyboardInterrupt as exc:
                if not path.is_file():
                    raise
                raise CliError("INTERRUPTED", "cancelled", "run `resume` with the saved checkpoint",
                               details={"checkpoint": str(path)}) from exc
    except _exp.cache_store.LockTimeout as exc:
        raise CliError("SERVICE_ERROR", "this simulation checkpoint is already being updated",
                       "inspect the existing writer; do not delete its lock or start another writer",
                       details=_exp._checkpoint_lock_details(path)) from exc

def _finalize_simulation(records, settings, args, checkpoint_path, checkpoint, explicit):
    for record, previous in zip(records, checkpoint["records"]):
        record["role"] = _exp.alpha_registry.evidence_role(previous, record)
    checkpoint["records"] = [dict(record) for record in records]
    _exp._write_simulation_checkpoint(checkpoint_path, checkpoint)
    has_unfinished = any(
        record.get("progress_url")
        or record.get("metrics_pending")
        or _exp._native_parent_pending(record)
        or record.get("status") in {"PENDING", "QUEUED", "UNKNOWN_SUBMISSION"}
        for record in records
    )
    discard_checkpoint = not explicit and not has_unfinished

    # The display view honours --sort/--top; the batch, the summary and the ledger
    # must not. Truncating before recording lost every arm outside the top N - and
    # with the checkpoint deleted at the end, those alphas had been paid for and
    # left no trace, which is both wasted quota and a selection bias in the archive.
    display = _exp._sort_simulation_results([dict(record) for record in records], args)
    summary = _exp.summarize(records)
    unknown = [index for index, record in enumerate(records) if record.get("status") == "UNKNOWN_SUBMISSION"]
    recording = None
    if getattr(args, "record", False):
        try:
            recording = _exp._record_simulation(records, settings, args)
        except (OSError, _exp.cache_store.CacheError) as exc:
            raise CliError("SERVICE_ERROR", "cannot record simulation results: %s" % exc,
                           "run `resume` with the saved checkpoint to retry recording",
                           details={"checkpoint": str(checkpoint_path)}) from exc
    if summary["completed"] == 0 and summary["total"] and not (
        summary["pending"] or summary["queued"] or unknown
    ):
        if discard_checkpoint:
            Path(checkpoint_path).unlink(missing_ok=True)
        first_error = next(
            (r.get("error") for r in records if r.get("error")), "unknown error"
        )
        raise CliError(
            "SIMULATION_FAILED",
            "all %d simulations failed" % summary["total"],
            str(first_error),
            details={"checkpoint": None if discard_checkpoint else str(checkpoint_path),
                     "summary": summary, "recording": recording},
        )
    if discard_checkpoint:
        Path(checkpoint_path).unlink(missing_ok=True)
        checkpoint_path = None
    result = _exp._simulation_result(display, settings, summary, checkpoint_path)
    result.update({key: checkpoint[key] for key in ("concurrency", "scheduling", "capability_snapshot", "multi_size")
                   if key in checkpoint})
    result["unknown_submission"] = unknown
    result["results_total"] = len(records)
    for key in ("legs", "legend", "note"):
        value = (checkpoint.get("metadata") or {}).get(key)
        if value is not None:
            result[key] = value
    pending_ids = [record.get("alpha_id") or record.get("candidate_id")
                   for record in records
                   if (record.get("status") or "").upper() in {"PENDING", "QUEUED"}]
    if pending_ids:
        result["pending"] = pending_ids
    if unknown:
        result["status"] = "PENDING"
        result["next"] = "submission outcome is unknown; reconcile it using the saved checkpoint"
    elif pending_ids:
        result["next"] = ("some arms are still running; the checkpoint was kept - "
                          "run `resume %s` before judging them"
                          % (checkpoint_path or "<checkpoint>"))
    if recording is not None:
        result["registry"] = recording
    return result

def cmd_simulate(args, sink):
    """Simulate a batch of expressions, or dry-run exact payloads offline."""
    concurrency = _exp._requested_concurrency(args)
    multi = _exp._multi_options(args)
    metadata = {}
    file_settings, alphas = _exp.collect_alphas(args, metadata=metadata)
    settings = _exp.merge_settings(file_settings, args)
    paths = [getattr(args, "file", None), *(getattr(args, "files", None) or ())]
    sources = [path for path in paths if path and Path(path).suffix.lower() == ".py"]
    _exp.protect_output(args, *sources)
    try:
        settings, pairs = _exp.config_io.build_config(settings, alphas)
    except _exp.config_io.ResearchError as exc:
        raise CliError("INVALID_INPUT", str(exc),
                       "check expression syntax, decay, and settings via `python research_cli.py schema simulate`")
    labels = _exp.ml.alpha_labels(alphas, len(pairs))
    families = _exp.ml.alpha_families(alphas, len(pairs))
    candidates = _exp.ml.alpha_candidate_ids(alphas, len(pairs))
    roles = [_exp._selection_call(_exp.alpha_registry.evidence_role, alpha) if isinstance(alpha, dict) else None
             for alpha in alphas]

    if args.dry_run:
        scheduling = {}
        records = _exp.ml.simulate_alphas(
            None, pairs, settings=settings, dry_run=True, labels=labels,
            families=families, candidates=candidates,
            concurrency=concurrency, capability_snapshot=getattr(args, "capability_snapshot", None),
            scheduling_report=scheduling,
            **multi,
        )
        for record, role in zip(records, roles):
            record["role"] = role
        # Offline lint: catches structural problems and learned rejections. The
        # authoritative operator check needs the catalog, so it runs on `lint`
        # and on a real submission below, not here.
        offline_lint = _exp._offline_simulation_lint(pairs, settings)
        return {
            "status": "DRY_RUN",
            "dry_run": True,
            "settings": settings,
            "concurrency": concurrency, "scheduling": scheduling,
            **multi,
            "summary": {"total": len(records), "payloads": len(records)},
            "lint": {
                "operator_catalog_checked": False,
                "hint": ("run `lint` to check Python source and platform fields"
                         if _exp.ml.is_python_language(settings.get("language"))
                         else "run `lint` for the authoritative operator check"),
                "ok": offline_lint["ok"],
                "errors": offline_lint["errors"],
                "warnings": offline_lint["warnings"],
            },
            "results": records,
            **metadata,
        }

    checkpoint_path = Path(args.checkpoint or ".brain_checkpoints/simulate-%s.json" % uuid.uuid4().hex)
    checkpoint_path = checkpoint_path.expanduser().resolve()
    with _exp._locked_simulation_checkpoint(checkpoint_path, args):
        if checkpoint_path.exists():
            raise CliError("INVALID_INPUT", "checkpoint already exists; use `resume` or choose another path",
                           str(checkpoint_path))

        deadline = _exp._simulation_deadline(args)
        run = _exp.session_reader(args, deadline=deadline)

        def preflight(session):
            snapshot = _exp._invocation_capabilities(session, args, deadline=deadline)
            _exp._preflight_lint(session, pairs, settings, args, deadline=deadline)
            return session, snapshot

        validated_session, snapshot = run(preflight)
        checkpoint = _exp._simulation_checkpoint_document(settings, pairs, concurrency, labels, families, candidates, roles)
        checkpoint.update(multi)
        if metadata:
            checkpoint["metadata"] = metadata
        checkpoint["capability_snapshot"] = snapshot
        checkpoint["scheduling"] = {}
        if getattr(args, "record", False):
            registry = getattr(args, "registry", None) or _exp.alpha_registry.default_registry_path()
            args.registry = str(Path(registry).expanduser().resolve())
        checkpoint["recording_policy"] = {key: getattr(args, key, None)
            for key in ("record", "registry", "idea", "mechanism", "family", "note")}
        _exp._write_simulation_checkpoint(checkpoint_path, checkpoint)
        _exp.note("simulation checkpoint %s" % checkpoint_path)
        on_event = _exp._simulation_checkpoint_callback(checkpoint_path, checkpoint, sink)

        def simulate(session):
            if session is not validated_session:
                _, checkpoint["capability_snapshot"] = preflight(session)
                _exp._write_simulation_checkpoint(checkpoint_path, checkpoint)
            return _exp.ml.simulate_alphas(
                session,
                pairs,
                settings=settings,
                concurrency=concurrency, capability_snapshot=checkpoint["capability_snapshot"],
                scheduling_report=checkpoint["scheduling"],
                on_event=on_event,
                deadline=deadline,
                initial_records=checkpoint["records"],
                labels=labels,
                families=families,
                candidates=candidates,
                **multi,
            )

        records = run(simulate)
        return _exp._finalize_simulation(records, settings, args, checkpoint_path, checkpoint,
                                    explicit=bool(args.checkpoint))

def cmd_resume(args, sink):
    """Resume pending, queued or unpolled simulations from an atomic checkpoint file."""
    _exp._reject_saved_overrides(args)
    path = Path(args.checkpoint).expanduser().resolve()
    with _exp._locked_simulation_checkpoint(path, args):
        return _exp._resume_simulation(args, sink, path)

def _resume_simulation(args, sink, path):
    deadline = _exp._simulation_deadline(args)
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError("INVALID_INPUT", "cannot read simulation checkpoint", str(exc))
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != "worldquantuser.simulation-checkpoint"
        or checkpoint.get("version") != 1
        or not isinstance(checkpoint.get("records"), list)
    ):
        raise CliError("INVALID_INPUT", "unsupported or malformed simulation checkpoint",
                       "pass a checkpoint created by `simulate` (or use `probe --resume` for Probe batches)")

    settings = checkpoint.get("settings")
    if not isinstance(settings, dict) or not settings:
        raise CliError("INVALID_INPUT", "simulation checkpoint is missing saved settings",
                       "restore the original checkpoint file or start a fresh simulation")

    records = checkpoint["records"]
    try:
        pairs = [(record["expression"], record.get("decay")) for record in records]
    except (KeyError, TypeError) as exc:
        raise CliError("INVALID_INPUT", "checkpoint is missing alpha inputs", str(exc))
    # Keep the per-expression labels recorded in the checkpoint; a resumed run
    # must not silently drop the mechanism tags gathered before the pause.
    labels = [
        record.get("mechanism") if isinstance(record, dict) else None
        for record in records
    ]
    # Same for the candidate id: an ablation resumed halfway must keep the
    # variant label, or the contribution table comes out unlabelled.
    candidates = [
        record.get("candidate_id") if isinstance(record, dict) else None
        for record in records
    ]
    multi = _exp._multi_options(args, checkpoint)
    if not multi and any(record.get("multi_request_id") for record in records):
        raise CliError("INVALID_INPUT", "native Multi checkpoint lacks its saved request size",
                       "restore the original checkpoint file with its saved multi_size field")
    concurrency = _exp._requested_concurrency(args, checkpoint)
    checkpoint["concurrency"] = concurrency
    policy = checkpoint.get("recording_policy")
    if isinstance(policy, dict):
        for key, value in policy.items():
            requested = getattr(args, key, None)
            if key == "registry" and requested is not None and value is not None:
                requested = str(Path(requested).expanduser().resolve())
                value = str(Path(value).expanduser().resolve())
            if requested is not None and requested is not False and requested != value:
                raise CliError("USAGE", "resume uses the saved recording policy; %s cannot change" % key,
                               "omit --%s when resuming this checkpoint" % key.replace("_", "-"))
            setattr(args, key, value)
    for record in records:
        if not record.get("multi_request_id"):
            _exp._selection_call(_exp.simulation_slots.restore_record, record)
    if not any(
        record.get("progress_url")
        or record.get("metrics_pending")
        or _exp._native_parent_pending(record)
        or (record.get("multi_request_id") and record.get("status") in {"PENDING", "UNKNOWN_SUBMISSION"})
        or record.get("status") == "QUEUED"
        for record in records
        if isinstance(record, dict)
    ):
        return _exp._finalize_simulation(
            records, settings, args, path, checkpoint, explicit=True
        )
    on_event = _exp._simulation_checkpoint_callback(path, checkpoint, sink)
    checkpoint["scheduling"] = {}
    def resume(session):
        _exp._validate_saved_native_owner(session, checkpoint)
        checkpoint["capability_snapshot"] = _exp._invocation_capabilities(session, args, deadline=deadline)
        return _exp.ml.simulate_alphas(session, pairs, settings=settings, concurrency=concurrency,
            capability_snapshot=checkpoint["capability_snapshot"], scheduling_report=checkpoint["scheduling"],
            on_event=on_event, deadline=deadline,
            initial_records=checkpoint["records"], labels=labels, candidates=candidates, **multi)
    records = _exp.with_session(args, resume, deadline=deadline)
    return _exp._finalize_simulation(
        records,
        settings,
        args,
        path,
        checkpoint,
        explicit=True,
    )

def _record_simulation(records, settings, args):
    """Append finished simulation records to the registry.

    The mapping lives in :func:`alpha_registry.record_simulation`, because the
    research template records the same records and had been copying this body by
    hand - or, more often, skipping it.
    """
    return _exp.alpha_registry.record_simulation(
        records, settings,
        idea=getattr(args, "idea", None),
        mechanism=getattr(args, "mechanism", None),
        family=getattr(args, "family", None),
        note=getattr(args, "note", None),
        registry_path=getattr(args, "registry", None),
        source="simulate")
