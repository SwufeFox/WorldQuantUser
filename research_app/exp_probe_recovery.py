"""Saved probe batch loading and platform task recovery."""

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

def _recovery_expression_key(expression, language):
    if not expression:
        return None
    if _exp.ml.is_python_language(language):
        return _exp.alpha_registry.expression_hash(expression, language=language)
    from fastexpr_ast import ParseError, parse

    # A formatted submission and its captured source have the same tree.
    # Keep opaque source exact instead of dropping whitespace inside strings.
    try:
        return "fastexpr", parse(expression).render()
    except (ParseError, RecursionError):
        return "raw", expression.strip()

def _probe_recovery_index(session, limit, pages, deadline):
    index = {}
    for page in range(pages):
        if _exp.time.monotonic() >= deadline:
            raise _exp.ml.RateLimited("probe recovery wait budget expired during alpha lookup")
        batch = _exp.ml.list_alphas(session, limit=limit, offset=page * limit,
                               order="-dateCreated", status="UNSUBMITTED", deadline=deadline)
        for alpha in batch:
            code, _description, _ops = _exp.ml.regular_expression(alpha)
            key = _exp._recovery_expression_key(code, _exp.ml.alpha_language(alpha))
            if key:
                index.setdefault(key, []).append({"alpha_id": alpha.get("id"),
                    "signature": _exp._settings_signature(alpha.get("settings"))})
        if len(batch) < limit:
            break
    return index

def _terminal_probe_without_alpha(record):
    attempted = record.get("submission_attempted")
    if record.get("alpha_id"):
        return False
    if record.get("submission_refused") is True and _exp.alpha_registry.terminal_execution_failure(record):
        return True
    if attempted is False:
        return False
    status = str(record.get("status") or "").upper()
    # Legacy rows can describe unlocated successful results. An explicit POST
    # attempt distinguishes today's terminal no-ID result from that old layout.
    return (status in _exp.simulation_slots.TERMINAL_FAILURE_STATUSES
            or (attempted is True and status in _exp.simulation_slots.TERMINAL_STATUSES))

def _probe_needs_recovery(row):
    if _exp._native_parent_pending(row):
        return True
    return (row.get("submission_attempted") is not False
            and (row.get("train_fitness") is None or row.get("metrics_pending"))
            and not _exp._terminal_probe_without_alpha(row))

def _native_parent_pending(record):
    return bool(record.get("multi_parent_url") and not record.get("multi_parent_terminal"))

def _recover_probe_rows(session, rows, limit=400, pages=3, max_wait_seconds=300.0,
                        on_progress=None, *, deadline=None):
    """Read known jobs first; use expression/settings lookup only for unidentified arms."""
    from requests.exceptions import RequestException

    limit = min(100, max(1, limit))
    missing = [row for row in rows if row.get("train_fitness") is None or row.get("metrics_pending")]
    matched, still_missing, ambiguous = {}, set(), []
    not_submitted = [row["label"] for row in missing if row.get("submission_attempted") is False]
    if deadline is None:
        deadline = _exp.time.monotonic() + max_wait_seconds
    index = lookup_error = None
    for row in sorted(missing, key=lambda item: not (item.get("alpha_id") or item.get("progress_url"))):
        label = row["label"]
        if (not _exp._probe_needs_recovery(row) or _exp.time.monotonic() >= deadline
                or (row.get("next_poll_at") or 0) > _exp.time.time()):
            still_missing.add(label)
            continue
        try:
            alpha_id = row.get("alpha_id")
            if not alpha_id and row.get("progress_url"):
                try:
                    progress = _exp.ml.poll_simulation(session, row["progress_url"], deadline=deadline)
                except _exp.ml.NotFound:
                    # An expired or temporarily invisible receipt does not prove
                    # that POST never happened. An older matching expression is
                    # not proof of this job's outcome either; retain the receipt.
                    row["error"] = "simulation receipt not found; submission remains unresolved"
                    row["next_poll_at"] = _exp.time.time() + 15
                    still_missing.add(label)
                    if on_progress:
                        on_progress(row)
                    continue
                if progress is not None:
                    status = str(progress.get("status") or "").upper()
                    row["status"] = status or None
                    row["submission_attempted"] = True
                    alpha_id = progress.get("alpha")
                    if not alpha_id:
                        row["error"] = progress.get("message")
                        if _exp._terminal_probe_without_alpha(row):
                            for key in _exp._CHECKPOINT_POLL_FIELDS:
                                row.pop(key, None)
                        else:
                            wait = progress.get("next_poll_after_seconds")
                            if wait is not None:
                                row["next_poll_at"] = _exp.time.time() + max(0, float(wait))
                        still_missing.add(label)
                        if on_progress:
                            on_progress(row)
                        continue
                    row["alpha_id"] = alpha_id
                    row["simulation_status"] = row["status"]
                    row["metrics_pending"] = True
                    row.pop("progress_url", None)
                    row.pop("next_poll_at", None)
                    row.pop("poll_after_seconds", None)
                    if on_progress:
                        on_progress(row)
            if not alpha_id:
                if not row.get("settings") and row.get("settings_signature") in (None, "", "{}"):
                    still_missing.add(label)
                    continue
                if index is None and lookup_error is None:
                    try:
                        index = _exp._probe_recovery_index(session, limit, pages, deadline)
                    except _exp.ml.AuthError:
                        raise
                    except (_exp.ml.BrainError, RequestException) as exc:
                        lookup_error = exc
                if lookup_error is not None:
                    row["error"] = str(lookup_error)
                    still_missing.add(label)
                    if on_progress:
                        on_progress(row)
                    continue
                wanted = (_exp._settings_signature(row["settings"]) if "settings" in row
                          else row.get("settings_signature") or _exp._settings_signature(None))
                key = _exp._recovery_expression_key(row.get("expression"), _exp._recovery_language(row, wanted))
                exact = [candidate for candidate in index.get(key, [])
                    if candidate["signature"] == wanted]
                if len(exact) != 1:
                    if len(exact) > 1:
                        ambiguous.append({"label": label,
                                          "candidates": [item["alpha_id"] for item in exact]})
                    still_missing.add(label)
                    continue
                alpha_id = exact[0]["alpha_id"]
                row["alpha_id"] = alpha_id
                row["metrics_pending"] = True
                if on_progress:
                    on_progress(row)
            payload = _exp.ml.fetch_alpha_metrics(session, alpha_id, deadline=deadline)
            if _exp.ml.defer_alpha_metrics(row, payload):
                still_missing.add(label)
                if on_progress:
                    on_progress(row)
                continue
            split = _exp.ml.extract_split(payload) or {}
            train = split.get("train") or {}
            if train.get("fitness") is None:
                still_missing.add(label)
            row.update({"status": payload.get("status") or row.get("simulation_status") or row.get("status"),
                        "train_sharpe": train.get("sharpe"),
                        "train_fitness": train.get("fitness"), "train_turnover": train.get("turnover"),
                        "train_returns": train.get("returns"), "error": None,
                        "settings_signature": _exp._settings_signature(row.get("settings"))})
            row.pop("metrics_pending", None)
            row.pop("next_poll_at", None)
            row.pop("poll_after_seconds", None)
            row.pop("simulation_status", None)
            raw = {"alpha_id": alpha_id, "status": row["status"],
                   "role": row.get("role"),
                   "expression": row.get("expression"), "settings": row.get("settings") or {},
                   "metrics": payload.get("is"), "out_of_sample": split,
                   "checks": _exp.ml._extract_checks(payload), "error": None,
                   "submission_attempted": True, "mechanism": label,
                   "decay": (row.get("settings") or {}).get("decay")}
            classifications = _exp.ml.alpha_classification_ids(payload)
            if classifications is not None:
                raw["classification_ids"] = classifications
            if train.get("fitness") is not None:
                matched[label] = alpha_id
        except _exp.ml.AuthError:
            raise
        except _exp.ml.NotFound:
            row["error"] = "alpha evidence not found; keep the accepted request for read-only recovery"
            row["next_poll_at"] = _exp.time.time() + 15
            still_missing.add(label)
            if on_progress:
                on_progress(row)
            continue
        except (_exp.ml.BrainError, RequestException) as exc:
            row["error"] = str(exc)
            still_missing.add(label)
            if on_progress:
                on_progress(row)
            continue
        if on_progress:
            on_progress(row, raw)
    return {"recovered": len(matched), "still_missing": [row["label"] for row in missing
                                                           if row["label"] in still_missing],
            "ambiguous": ambiguous, "matched": matched, "not_submitted": not_submitted}

def _recovery_language(row, signature):
    settings = row.get("settings")
    if not isinstance(settings, dict):
        try:
            settings = json.loads(signature)
        except (TypeError, ValueError):
            return None
    return settings.get("language") if isinstance(settings, dict) else None

def _load_probe_batch(value, *, source_paths=None, within=None, observe_source=None):
    alias = Path(value).expanduser()
    visited = set()
    while True:
        try:
            path = alias.resolve()
        except (OSError, RuntimeError) as exc:
            raise CliError("INVALID_INPUT", "cannot resolve probe response path %s" % alias, str(exc)) from exc
        if within is not None and not path.is_relative_to(within):
            raise CliError("INVALID_INPUT", "probe response path is outside the research directory",
                           "keep probe batches and responses inside the active research directory")
        if path in visited:
            raise CliError("INVALID_INPUT", "cyclic probe response path",
                           "pass the canonical batch file path directly")
        visited.add(path)
        if source_paths is not None:
            source_paths.add(path)
        if observe_source is not None:
            observe_source(alias, path)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CliError("INVALID_INPUT", "cannot read batch file %s" % path, str(exc)) from exc
        if isinstance(document, dict) and isinstance(document.get("data"), dict):
            document = document["data"]
        if not isinstance(document, dict):
            raise CliError("INVALID_INPUT", "probe batch must be an object",
                           "pass a JSON batch or response saved by `probe`")
        if document.get("path") and not document.get("arms"):
            if not isinstance(document["path"], str):
                raise CliError("INVALID_INPUT", "probe response path must be a string",
                               "inspect the saved probe response JSON")
            destination = Path(document["path"]).expanduser()
            alias = destination if destination.is_absolute() else path.parent / destination
            continue
        break
    rows = document.get("rows") or []
    if not rows:
        raise CliError("INVALID_INPUT", "%s has no probe rows" % path.name,
                       "verify this is a saved `probe` batch or response file")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise CliError("INVALID_INPUT", "probe rows must be objects",
                       "verify this is a valid `probe` batch file")
    saved = document.get("records") or [{} for _ in rows]
    if not isinstance(saved, list) or len(saved) != len(rows) or not all(isinstance(record, dict) for record in saved):
        raise CliError("INVALID_INPUT", "probe batch records do not align with rows",
                       "recover from the intact batch file or check for partial edits")
    if not document.get("arms"):
        arms = []
        for index, (row, record) in enumerate(zip(rows, saved)):
            settings = row.get("settings") or record.get("settings")
            if not settings and row.get("settings_signature"):
                try:
                    settings = json.loads(row["settings_signature"])
                except (ValueError, TypeError) as exc:
                    raise CliError("INVALID_INPUT", "legacy settings signature is not recoverable",
                                   "supply a spec or batch with complete settings") from exc
            if settings is not None and not isinstance(settings, dict):
                raise CliError("INVALID_INPUT", "probe arm settings must be an object",
                               "inspect the saved arm settings in the batch file")
            arms.append({"label": row.get("label", "arm%d" % (index + 1)),
                "role": _exp.alpha_registry.evidence_role(row, record),
                "expression": row.get("expression") or record.get("expression") or "",
                "settings": settings or {}, "arm_index": index, "ops": row.get("ops"),
                "design": row.get("design"), "proxy_of": row.get("proxy_of")})
        document["arms"] = arms
    arms = document["arms"]
    if (not isinstance(arms, list) or len(arms) != len(rows)
            or not all(isinstance(arm, dict) and isinstance(arm.get("settings"), dict)
                       and isinstance(arm.get("expression"), str) and arm.get("label") for arm in arms)
            or len({arm["label"] for arm in arms}) != len(arms)):
        raise CliError("INVALID_INPUT", "probe batch arm identities are inconsistent",
                       "restore the original saved batch checkpoint without editing arm labels or rows")
    records = []
    for index, (arm, row, record) in enumerate(zip(arms, rows, saved)):
        arm["arm_index"] = index
        arm.setdefault("ops", row.get("ops"))
        role = _exp._selection_call(_exp.alpha_registry.evidence_role, arm, row, record)
        arm["role"] = role
        record = {**record, "expression": arm["expression"], "settings": arm["settings"],
                  "decay": arm["settings"].get("decay"), "mechanism": arm["label"]}
        record["role"] = role
        for key in ("alpha_id", "status", "submission_attempted", "submission_refused",
                    "error", "simulation_status", "classification_ids",
                    *_exp._CHECKPOINT_POLL_FIELDS, *_exp._MULTI_CHECKPOINT_FIELDS):
            if key not in record and key in row:
                record[key] = row[key]
        if not record.get("out_of_sample"):
            record["out_of_sample"] = {"train": {key: row.get("train_" + key)
                for key in ("sharpe", "fitness", "turnover", "returns")}}
        if record.get("alpha_id") and not record.get("metrics"):
            record["metrics_pending"] = True
        if _exp._terminal_probe_without_alpha(record):
            for key in _exp._CHECKPOINT_POLL_FIELDS:
                record.pop(key, None)
        records.append(record)
    document["records"] = records
    if any(record.get("multi_request_id") for record in records) and not _exp._multi_options(None, document):
        raise CliError("INVALID_INPUT", "native Multi batch lacks its saved request size",
                       "restore the original batch file with its saved multi_size field")
    fingerprint = _exp._request_fingerprint(arms)
    if document.get("request_fingerprint") and document["request_fingerprint"] != fingerprint:
        raise CliError("INVALID_INPUT", "saved probe request fingerprint does not match its arms",
                       "keep the original arms and settings fixed; create a new spec for a different experiment")
    document.setdefault("request_fingerprint", fingerprint)
    document.setdefault("arm_order", [arm["label"] for arm in arms])
    document.setdefault("run_id", None)
    document.setdefault("started_at", None)
    track = document.get("track")
    if track is None and document.get("mechanism") == _exp.research_route.ATOM_RUNNER_MECHANISM:
        track = "ATOM"
    document["routes"] = _exp._probe_routes(track, arms)
    document["track"] = document["routes"][0]["track"]
    _exp._probe_document_update(document, path)
    return path, document

def load_probe_batch(value, *, source_paths=None, within=None, observe_source=None):
    """Load and validate a saved probe batch checkpoint from path or payload."""
    return _exp._load_probe_batch(value, source_paths=source_paths, within=within,
                             observe_source=observe_source)

def _cmd_probe_recover(args):
    path, _document = _exp._load_probe_batch(args.recover)
    _exp.protect_output(args, path)
    try:
        with _exp.cache_store.file_lock(str(path) + ".lock", timeout=0):
            return _exp._recover_probe_batch(args, path)
    except _exp.cache_store.LockTimeout as exc:
        raise CliError("SERVICE_ERROR", "this probe batch is already being updated",
                       "inspect the existing writer; keep this batch and its lock",
                       details=_exp._checkpoint_lock_details(path)) from exc

def _native_probe_groups(document):
    """Keep every member of any unresolved native parent together on recovery."""
    requests = {record["multi_request_id"] for row, record in zip(document["rows"], document["records"])
                if record.get("multi_request_id") and (_exp._native_parent_pending(record)
                    or (not _exp._has_finished_result(record) and _exp._probe_needs_recovery(row)))}
    groups = {}
    for index, record in enumerate(document["records"]):
        if record.get("multi_request_id") in requests:
            settings = document["arms"][index]["settings"]
            groups.setdefault(_exp._settings_signature(settings), []).append(index)
    return groups.values()

def _recover_probe_batch(args, path):
    deadline = _exp._simulation_deadline(args)
    path, document = _exp._load_probe_batch(path)
    show_test = bool(getattr(args, "show_test", False))
    if show_test and not getattr(args, "selection", None):
        raise CliError("USAGE", "--show-test requires --selection",
                       "pass --selection PATH from a training freeze before opening the test view")
    manifest = _exp._selection_call(_exp.selection.load, args.selection) if show_test else None
    rows = document["rows"]
    multi = _exp._multi_options(args, document)
    if getattr(args, "native_poll_once", False):
        multi["native_poll_once"] = True
    defer_quick_pnl = getattr(args, "defer_quick_pnl", False)
    pnl_alpha_ids = set(getattr(args, "pnl_alpha_id", None) or ())
    if pnl_alpha_ids:
        known_quick_ids = {record.get("alpha_id") for record in document["records"]
                           if _exp.ml.is_quick_simulation(record)}
        unknown_ids = pnl_alpha_ids - known_quick_ids
        if unknown_ids:
            raise CliError("INVALID_INPUT", "--pnl-alpha-id requires QUICK Alpha IDs in this batch",
                           "choose QUICK Alpha IDs from `rows[].alpha_id` in this batch",
                           details={"unknown_quick_alpha_ids": sorted(unknown_ids)})
    native_indices = list(_exp._native_probe_groups(document))
    if document.get("recording"):
        document["recording"]["appended"] = 0

    def pnl_records():
        return (record for record in document["records"]
                if not pnl_alpha_ids or (_exp.ml.is_quick_simulation(record)
                                         and record.get("alpha_id") in pnl_alpha_ids))

    def recover(session):
        records = document["records"]
        account = None
        if session is not None:
            _exp._validate_saved_native_owner(session, document)
        unresolved = [record for record in records if record.get("slot_reservation")
                      and not record.get("multi_request_id") and not _exp._terminal_probe_without_alpha(record)]
        if session is not None and unresolved:
            account = _exp.simulation_capabilities.account_provenance(session)
            restored = [_exp._selection_call(_exp.simulation_slots.restore_record, record, account)
                        for record in unresolved]
            if any(restored):
                _exp._save_probe_checkpoint(document, path, indent=1)
        native_before = {row["label"] for row in document["rows"]
                         if row.get("multi_request_id") and row.get("train_fitness") is None}
        if session is not None:
            progress_lock = threading.Lock()
            for indices in native_indices:
                if _exp.time.monotonic() >= deadline:
                    break
                arms = [document["arms"][index] for index in indices]
                settings = arms[0]["settings"]

                def progress(kind, payload):
                    with progress_lock:
                        _exp._save_probe_progress(document, path, indices, kind, payload)

                produced = _exp.ml.simulate_alphas(session, [(arm["expression"], arm["settings"]["decay"])
                    for arm in arms], settings=settings, labels=[arm["label"] for arm in arms],
                    initial_records=[records[index] for index in indices], deadline=deadline,
                    concurrency=document.get("concurrency"), on_event=progress, recover_only=True, **multi)
                _exp._save_probe_records(document, path, zip(indices, produced))
        active_rows = document["rows"]
        for row, record in zip(active_rows, records):
            if "simulation_status" in record:
                row["simulation_status"] = record["simulation_status"]
        positions = {id(row): index for index, row in enumerate(active_rows)}

        def save_recovered(row, raw=None):
            index = positions[id(row)]
            record = records[index]
            if raw is not None:
                record = records[index] = _exp._merge_checkpoint_record(record, raw)
            for key in ("alpha_id", "status", "submission_attempted", "error"):
                if key in row:
                    record[key] = row[key]
            for key in (*_exp._CHECKPOINT_POLL_FIELDS, "simulation_status"):
                if key in row:
                    record[key] = row[key]
                else:
                    record.pop(key, None)
            _exp.simulation_slots.reconcile_record(record, account)
            _exp._save_probe_checkpoint(document, path, indent=1)

        outcome = _exp._recover_probe_rows(session, [row for row in active_rows
            if not row.get("multi_request_id") and row["label"] not in native_before],
            on_progress=save_recovered, deadline=deadline)
        for row in active_rows:
            label = row["label"]
            if label in native_before:
                if row.get("train_fitness") is None:
                    outcome["still_missing"].append(label)
                    if row.get("submission_attempted") is False:
                        outcome["not_submitted"].append(label)
                elif row.get("alpha_id"):
                    outcome["matched"][label] = row["alpha_id"]
                    outcome["recovered"] += 1
        _exp._probe_halves(session, pnl_records(), deadline=deadline, defer_quick_pnl=defer_quick_pnl)
        return outcome

    now = _exp.time.time()
    needs_platform = any(_exp._probe_needs_recovery(row) and (row.get("next_poll_at") or 0) <= now
                         for row in rows)
    needs_halves = any(_exp._probe_needs_train_pnl(record) for record in pnl_records()
                       if not (defer_quick_pnl and _exp.ml.is_quick_simulation(record)))
    outcome = _exp.with_session(args, recover, deadline=deadline) if needs_platform or needs_halves else recover(None)
    _exp._probe_reconcile(document)
    _exp._save_probe_checkpoint(document, path, indent=1)
    result = _exp._probe_response(document, path, show_test, manifest, summary=getattr(args, "summary", False))
    if show_test:
        _exp.research_paths.log_view("probe-recover", "test", [row.get("alpha_id") for row in result["rows"]],
                                note={"selection_id": manifest["selection_id"], "batch_path": str(path)})
    return {**result, **outcome}
