"""Simulation, resumption, labelled probes, ablation and comparison workflows."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

import ablation
import alpha_registry
import alpha_unit
import cache_store
import expression_lint
import machine_lib as ml
import research_config as config_io
import research_paths
import simulation_capabilities
import simulation_multi
import simulation_slots
from . import guarded_sim

from .runtime import CliError, METRIC_KEYS, collect_alphas, merge_settings, note, protect_output, session_reader, with_session
from . import costs, incentives, research_route, selection, stats
from .validation import (FieldEvidenceRefreshError, _preflight_lint, attach_field_contracts,
                         cached_field_scope, lint_python_source)



#: Operators per arm above which the report flags complexity. Not a platform ceiling
#: (the platform allows up to 64); a research discipline threshold to keep arms
#: interpretable and avoid overfitting multi-term composites.
SIMPLE_OPERATOR_LIMIT = 8

def _train_table(rows):
    columns = (("label", "label"), ("role", "role"), ("state", "status"),
               ("train S", "train_sharpe"), ("train F", "train_fitness"),
               ("turnover", "train_turnover"), ("margin_bps", "margin_bps"),
               ("half 1 S", "train_sharpe_h1"), ("half 2 S", "train_sharpe_h2"),
               ("alpha_id", "alpha_id"))

    def cell(row, key):
        value = row.get(key)
        if key == "role" and value is None:
            value = "unspecified"
        if key == "margin_bps":
            value = (row.get("cost_evidence") or {}).get("margin_bps")
        return "-" if value is None else str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(title for title, _ in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(row, key) for _, key in columns) + " |" for row in rows)
    return "\n".join(lines)

def _probe_summary(response):
    rows = response.get("rows", response.get("results", []))
    fields = ("train_sharpe", "train_fitness", "train_turnover", "train_margin",
              "train_sharpe_h1", "train_sharpe_h2")
    quota = next((report["quota"] for report in reversed(response.get("scheduling") or [])
                  if report.get("quota") is not None), None)
    incentive_rows = [row for row in rows if (row.get("incentives") or {}).get("state") == "reported"]
    checkpoint_path = response.get("path")
    saved = (not response.get("dry_run") and not response.get("check")
             and isinstance(checkpoint_path, str) and Path(checkpoint_path).is_file())
    return {"path": response.get("path"), "run_id": response.get("run_id"),
            "request_fingerprint": response.get("request_fingerprint"), "view": "train",
            "full_evidence": {"available": saved, "source": "checkpoint" if saved else None,
                              "path": checkpoint_path if saved else None},
            "total_count": (response.get("execution") or {}).get("total", response.get("arms", len(rows))),
            "displayed_count": min(20, len(rows)), "rows_omitted": max(0, len(rows) - 20),
            "execution": response.get("execution"), "recording": response.get("recording"),
            "quota": quota,
            "incentives": [{"label": row.get("label"), "alpha_id": row.get("alpha_id"), **row["incentives"]}
                           for row in incentive_rows[:incentives.SUMMARY_LIMIT]],
            "incentives_reported_count": len(incentive_rows),
            "incentives_unknown_count": len(rows) - len(incentive_rows),
            "incentives_omitted": max(0, len(incentive_rows) - incentives.SUMMARY_LIMIT),
            "roles": {role: sum((row.get("role") or "unspecified") == role for row in rows)
                      for role in ("candidate", "diagnostic", "unspecified")},
            "missing_evidence_count": sum(any(row.get(key) is None for key in fields) for row in rows),
            "missing_evidence": [{"label": row.get("label"),
                                  "fields": [key for key in fields if row.get(key) is None]}
                                 for row in rows[:20] if any(row.get(key) is None for key in fields)],
            "next_actions": response.get("next_actions", []), "table": _train_table(rows[:20])}

def _compact_preflight(report):
    if not isinstance(report, dict):
        return {"state": "unavailable"}
    result = {key: report[key] for key in ("ok", "preflight_complete", "operator_catalog_checked",
              "field_catalog_checked", "capability_id") if key in report}
    for key in ("errors", "warnings", "results", "checked_requests"):
        items = report.get(key)
        if isinstance(items, list):
            result[key + "_count"] = len(items)
            visible = items[:20] if key in ("errors", "warnings") else []
            result[key + "_omitted"] = len(items) - len(visible)
            if visible:
                result[key] = visible
    result["truncated"] = (any(result.get(key + "_omitted", 0)
                               for key in ("errors", "warnings", "results", "checked_requests"))
                           or bool(set(report) - set(result)
                                   - {"errors", "warnings", "results", "checked_requests"}))
    return result

def _summarize_probe_response(response):
    response["summary"] = _probe_summary(response)
    truncated = bool(response["summary"]["rows_omitted"] or response["summary"]["incentives_omitted"])
    deferred_preflights = response.get("preflight_deferred")
    if deferred_preflights:
        response["preflight_deferred_total"] = len(deferred_preflights)
        response["preflight_deferred_omitted"] = max(0, len(deferred_preflights) - 20)
        response["preflight_deferred_truncated"] = len(deferred_preflights) > 20
        visible_keys = ("code", "message", "hint", "scope", "retry_at")
        response["preflight_deferred"] = {
            label: {key: group[key] for key in visible_keys}
            for label, group in list(deferred_preflights.items())[:20]
        }
        truncated |= (response["preflight_deferred_truncated"]
                      or any(set(group) - set(visible_keys) for group in deferred_preflights.values()))
    quick_pnl = response.get("quick_pnl")
    if quick_pnl:
        labels = quick_pnl["pending_labels"]
        response["quick_pnl"] = {**quick_pnl, "pending_labels": labels[:20],
                                 "pending_labels_total": len(labels),
                                 "pending_labels_omitted": max(0, len(labels) - 20),
                                 "pending_labels_truncated": len(labels) > 20}
        truncated |= len(labels) > 20
    for key in ("rows", "results", "incomplete", "comparisons", "arm_order", "preflight", "multi_requests"):
        items = response.get(key)
        if isinstance(items, list):
            response[key + "_total"] = len(items)
            response[key + "_omitted"] = max(0, len(items) - 20)
            response[key + "_truncated"] = len(items) > 20
            response[key] = [_compact_preflight(item) for item in items[:20]] if key == "preflight" else items[:20]
            truncated |= len(items) > 20
            if key == "preflight":
                truncated |= any(item.get("truncated", False) for item in response[key])
    if isinstance(response.get("lint"), dict):
        lint = response["lint"]
        response["lint"] = _compact_preflight(lint)
        response["lint"]["groups_count"] = len(lint.get("groups") or [])
        truncated |= response["lint"]["truncated"]
    snapshot = response.get("capability_snapshot")
    if isinstance(snapshot, dict):
        response["capability_snapshot"] = {key: snapshot[key] for key in (
            "state", "account", "source", "as_of", "age_seconds", "capability_id", "concurrency") if key in snapshot}
        truncated |= bool(set(snapshot) - set(response["capability_snapshot"]))
    response["truncated"] = response["summary"]["truncated"] = bool(truncated)
    return response

def _requested_concurrency(args, saved=None):
    requested = getattr(args, "concurrency", None)
    if requested is None:
        requested = getattr(args, "batch_size", None)
    if requested is None and saved:
        requested = saved.get("concurrency", saved.get("batch_size"))
    if requested is not None and (isinstance(requested, bool) or not isinstance(requested, int) or requested < 1):
        raise CliError("INVALID_INPUT", "--batch-size must be a positive integer",
                       "pass an integer >= 1, e.g. --batch-size 4")
    return requested

def _multi_options(args, saved=None):
    size = getattr(args, "multi_size", None) if saved is None else saved.get("multi_size")
    if size is None:
        return {}
    if type(size) is not int or not 2 <= size <= simulation_multi.MAX_CHILDREN:
        raise CliError("INVALID_INPUT", f"--multi-size must be an integer from 2 to {simulation_multi.MAX_CHILDREN}",
                       f"pass --multi-size between 2 and {simulation_multi.MAX_CHILDREN}")
    return {"multi_size": size}

def _validate_saved_native_owner(session, document):
    if not document.get("multi_size"):
        return
    # Finished parents and singleton tails still bind a batch to its account
    # when the scheduler only receives the remaining queued rows.
    attempted = [record for record in document["records"] if (
        record.get("submission_attempted") is True or record.get("alpha_id")
        or record.get("progress_url") or record.get("multi_request_id"))]
    if not attempted:
        return
    account = simulation_capabilities.account_provenance(session)
    saved = ((document.get("capability_snapshot") or {}).get("account") or {}).get("fingerprint")
    if saved and saved != account.get("fingerprint"):
        raise CliError("INVALID_INPUT", "native Multi checkpoint belongs to another or unidentified account",
                       "switch back to the account that created the checkpoint before resuming or recovering")
    for record in attempted:
        _selection_call(simulation_slots.validate_reservation_owner, record, account)

def _invocation_capabilities(session, args, *, deadline=None):
    snapshot = getattr(args, "capability_snapshot", None)
    current = simulation_capabilities.account_provenance(session)
    prior_account = (snapshot or {}).get("account") or {}
    changed = (current.get("fingerprint") and prior_account.get("fingerprint")
               and current["fingerprint"] != prior_account["fingerprint"])
    if snapshot is None or changed:
        snapshot = simulation_capabilities.fetch(session, refresh=True, deadline=deadline)
        args.capability_snapshot = snapshot
    return snapshot

def _simulation_deadline(args):
    return time.monotonic() + getattr(args, "max_wait_seconds", 300.0)

def _probe_routes(track, arms):
    """Describe resolved routes without changing their platform request identity."""
    try:
        routes = [research_route.normalize_route({"track": track}, settings=arm["settings"])
                  for arm in arms]
    except ValueError as exc:
        raise CliError("INVALID_INPUT", str(exc),
                       "check spec track and settings against `python research_cli.py schema probe`") from exc
    return list({(row["language"], row["simulation_mode"]): row for row in routes}.values())

def cmd_probe(args, sink):
    """Read existing tasks or run a spec through the saved-batch workflow."""
    reference = getattr(args, "from_simulation", None)
    if getattr(args, "native_poll_once", False) and (reference is not None
            or getattr(args, "dry_run", False) or getattr(args, "check", False)):
        raise CliError("USAGE", "--native-poll-once requires a live probe run, --resume or --recover",
                       "omit --native-poll-once during --dry-run, --check, or --from-simulation")
    if reference is not None:
        from .simulation_import import read_existing_simulation

        _reject_saved_overrides(args, mode="--from-simulation")
        incompatible = ("spec", "recover", "resume", "dry_run", "check", "show_test", "selection",
                        "name", "batch_path", "registry", "cost_bps", "concurrency", "force",
                        "allow_lint_errors", "strict_gate", "defer_quick_pnl", "pnl_alpha_id")
        if any(getattr(args, key, None) is not None and getattr(args, key) is not False
               for key in incompatible):
            raise CliError("USAGE", "--from-simulation is a read-only operation on existing simulations",
                           "pass only `probe --from-simulation ID [--max-wait-seconds N]`")
        return read_existing_simulation(args, reference)
    _requested_concurrency(args)
    result = _cmd_probe(args, sink)
    if getattr(args, "summary", False):
        _summarize_probe_response(result)
    return result

def _offline_simulation_lint(pairs, settings, scoped=None):
    if ml.is_python_language(settings.get("language")):
        known_fields = (scoped or {}).get("known_fields")
        reports = [lint_python_source(source, known_fields=known_fields, lookback=settings.get("lookback"))
                   for source, _ in pairs]
        return {"ok": all(report["ok"] for report in reports), "language": "PYTHON",
                "errors": [error for report in reports for error in report["errors"]],
                "warnings": [warning for report in reports for warning in report["warnings"]],
                "results": reports}
    if scoped is None:
        return expression_lint.lint_all(pairs, denylist=expression_lint.load_denylist())
    report = expression_lint.lint_all(
        pairs, denylist=expression_lint.load_denylist(),
        known_fields=scoped.get("known_fields"), field_types=scoped.get("field_types"),
        strict_fields=scoped.get("known_fields") is not None)
    attach_field_contracts(report, scoped)
    return report

def _preview_probe_group(args, group, concurrency, multi):
    scheduling = {}
    arms = group["arms"]
    records = ml.simulate_alphas(None, group["pairs"], settings=group["settings"],
        labels=[arm["label"] for arm in arms], dry_run=True, concurrency=concurrency,
        capability_snapshot=getattr(args, "capability_snapshot", None), scheduling_report=scheduling,
        **multi)
    requests = []
    for request in scheduling.pop("multi_requests", []):
        included = [arms[index] for index in request["indices"]]
        requests.append({**request, "indices": [arm["arm_index"] for arm in included],
                         "labels": [arm["label"] for arm in included]})
    return records, scheduling, requests

def _cmd_probe(args, _sink):
    """Prepare exact requests before checking, executing or continuing a batch."""
    recover = getattr(args, "recover", None)
    resume = getattr(args, "resume", None)
    check = bool(getattr(args, "check", False))
    dry_run = bool(getattr(args, "dry_run", False))
    if sum(bool(value) for value in (recover, resume, check)) > 1 or check and dry_run:
        raise CliError("USAGE", "--check, --recover and --resume are exclusive; --check cannot use --dry-run",
                       "choose one mode: --dry-run (offline), --check (live preflight), --recover, or --resume")
    if getattr(args, "pnl_alpha_id", None) and (not recover or dry_run
            or getattr(args, "show_test", False) or getattr(args, "defer_quick_pnl", False)):
        raise CliError("USAGE", "--pnl-alpha-id requires --recover and cannot use "
                       "--dry-run, --show-test or --defer-quick-pnl",
                       "use `probe --recover BATCH.json --pnl-alpha-id ALPHA_ID`")
    if getattr(args, "show_test", False) and (not recover or not getattr(args, "selection", None)):
        raise CliError("USAGE", "--show-test requires --recover and --selection; it cannot start simulations",
                       "use `probe --recover BATCH.json --show-test --selection CHOICE.json`")
    if recover:
        _reject_saved_overrides(args)
        return _cmd_probe_recover(args)
    if resume:
        return _cmd_probe_resume(args)
    multi = _multi_options(args)
    spec, name, groups, arms, design, gate = _prepare_probe(args)
    target = Path(getattr(args, "batch_path", None) or research_paths.batch_path(name)).expanduser().resolve()
    source_paths = [Path(arm["source_file"]) for arm in arms if "source_file" in arm]
    if target in source_paths:
        raise CliError("INVALID_INPUT", "batch path would overwrite an input source_file",
                       "choose a --batch-path distinct from all Python source_file paths")
    protect_output(args, target, *source_paths)
    context = {key: spec[key] for key in _PROBE_CONTEXT_FIELDS if key in spec}
    base = {"name": name, **context, "path": str(target), "arms": len(arms),
            "track": spec["track"], "routes": _probe_routes(spec["track"], arms),
            "groups": len(groups), "design": design, "arm_order": [arm["label"] for arm in arms],
            "request_fingerprint": _request_fingerprint(arms),
            "concurrency": _requested_concurrency(args),
            "settings": [group["settings"] for group in groups.values()], **multi}
    gate_blocking = [{"code": "exact_repeat", "label": arm["label"],
                      "evidence_keys": verdict["exact_repeat_keys"],
                      "completed_refs": verdict.get("completed_refs", []),
                      "inspect_argv": ["python", "research_cli.py", "ledger", "list",
                                       "--expression", arm["expression"],
                                       "--registry", str(getattr(args, "registry", None)
                                                         or alpha_registry.default_registry_path())]}
                     for arm, verdict in zip(arms, gate)
                     if getattr(args, "strict_gate", False) and verdict.get("already_simulated")]
    if dry_run:
        recovery_actions = _probe_checkpoint_refs(target, arms, gate)
        results = [None] * len(arms)
        reports = []
        requests = []
        catalog = turnover_stats = None
        for group in groups.values():
            scoped = cached_field_scope(group["settings"])
            report = _offline_simulation_lint(group["pairs"], group["settings"], scoped)
            reports.append({**report, "scope": scoped.get("evidence")})
            preview, scheduling, group_requests = _preview_probe_group(
                args, group, base["concurrency"], multi)
            requests.extend(group_requests)
            reports[-1]["scheduling"] = scheduling
            python = ml.is_python_language(group["settings"].get("language"))
            if not python and catalog is None:
                from . import mine

                catalog = mine.load_catalog()
                datasets = {dataset for arm in arms
                            if not ml.is_python_language(arm["settings"].get("language"))
                            for dataset in mine.expression_datasets(arm["expression"], catalog)}
                turnover_stats = mine.dataset_turnover(
                    catalog=catalog, datasets=datasets, registry_path=getattr(args, "registry", None))
            for record, arm in zip(preview, group["arms"]):
                index = arm["arm_index"]
                results[index] = {**record, "label": arm["label"], "arm_index": index,
                    "role": arm.get("role"),
                    "design": arm.get("design"), "proxy_of": arm.get("proxy_of"),
                    "settings": arm["settings"], "gate": gate[index],
                    "turnover_band": None if python else _turnover_band(
                        arm["expression"], catalog=catalog, stats=turnover_stats)}
                if "source_file" in arm:
                    results[index]["source_file"] = arm["source_file"]
        blocking = [item for report in reports for item in report.get("errors", [])]
        return {**base, "dry_run": True, "results": results, "blocking": blocking,
                **({"multi_requests": requests} if multi else {}),
                "gate_blocking": gate_blocking, "would_block": bool(blocking or gate_blocking),
                "preflight_complete": False, "run_id": None, "started_at": None,
                "comparisons": _probe_comparisons(results, design),
                "lint": {"ok": not blocking, "errors": blocking,
                         "warnings": [item for report in reports for item in report.get("warnings", [])],
                         "operator_catalog_checked": False, "field_catalog_checked": False,
                         "hint": "Live catalogs were not checked; run probe --check for exact authenticated preflight.",
                         "groups": reports},
                "next_actions": recovery_actions or [{"action": "check", "spec": str(args.spec), **multi}]}
    if gate_blocking:
        raise CliError("INVALID_INPUT", "exact finished requests repeat under --strict-gate",
                       "inspect the recorded evidence; resume the original batch for unfinished work",
                       details={"reason": "exact_repeat", "next_action": "inspect_existing_evidence",
                                "gate_blocking": gate_blocking})
    if check:
        deadline = _simulation_deadline(args)
        proof_cache = {}

        def check_groups(session):
            base["capability_snapshot"] = _invocation_capabilities(session, args, deadline=deadline)
            reports = []
            for group in groups.values():
                if time.monotonic() >= deadline:
                    break
                reports.append(_preflight_lint(session, group["pairs"], group["settings"], args,
                                               deadline=deadline, proof_cache=proof_cache))
            return reports
        reports = with_session(args, check_groups, deadline=deadline)
        return {**base, "check": True, "preflight": reports,
                "preflight_complete": len(reports) == len(groups)
                and all(bool(report) and report.get("ok", False) for report in reports),
                "run_id": None, "started_at": None,
                "next_actions": [{"action": "run", "spec": str(args.spec), "batch_path": str(target), **multi}]}
    try:
        with cache_store.file_lock(str(target) + ".lock", timeout=0):
            if target.exists() and not getattr(args, "force", False):
                raise CliError("INVALID_INPUT", "%s already exists" % target.name,
                               "recover or resume the saved batch, or choose another --batch-path")
            return _run_probe(args, spec, name, groups, design, gate, target)
    except cache_store.LockTimeout as exc:
        raise CliError("SERVICE_ERROR", "this probe batch is already being updated",
                       "inspect the existing writer; keep this batch and its lock",
                       details=_checkpoint_lock_details(target)) from exc

def _probe_row(record, arm, gate, show_test):
    train = _train_metrics(record)
    row = {"label": arm["label"], "alpha_id": record.get("alpha_id"),
           "role": alpha_registry.evidence_role(arm, record),
           "design": arm.get("design"), "proxy_of": arm.get("proxy_of"),
           "expression": arm["expression"], "decay": record.get("decay"),
           "settings": dict(record["settings"]),
           "settings_signature": _settings_signature(record["settings"]),
           "status": record.get("status"), "error": record.get("error"),
           "train_sharpe": train.get("sharpe"), "train_fitness": train.get("fitness"),
           "train_turnover": train.get("turnover"), "train_returns": train.get("returns"),
           "train_margin": train.get("margin"),
           "train_sharpe_h1": (record.get("half_sharpes") or (None, None))[0],
           "train_sharpe_h2": (record.get("half_sharpes") or (None, None))[1],
           "ops": arm["ops"], "gate": gate}
    row["incentives"] = incentives.check_summary(record.get("checks"))
    if "classification_ids" in record:
        row["classification_ids"] = record["classification_ids"]
    for key in (*_CHECKPOINT_POLL_FIELDS, *_MULTI_CHECKPOINT_FIELDS,
                "submission_attempted", "submission_refused"):
        if key in record:
            row[key] = record[key]
    if show_test:
        metrics = record.get("metrics") or {}
        row.update({"sharpe": metrics.get("sharpe"), "fitness": metrics.get("fitness"),
                    "turnover": metrics.get("turnover"),
                    "test_sharpe": ((record.get("out_of_sample") or {}).get("test") or {}).get("sharpe"),
                    "failed_checks": sorted(key for key, body in (record.get("checks") or {}).items()
                                            if isinstance(body, dict) and body.get("result") == "FAIL")})
    return row

def _submission_state(record):
    if _terminal_probe_without_alpha(record):
        return "terminal"
    if record.get("alpha_id") or record.get("progress_url") or record.get("multi_parent_url"):
        return "accepted"
    return "not_submitted" if record.get("submission_attempted") is False else "unknown_submission"

def _probe_document_update(document, target):
    arms, records = document["arms"], document["records"]
    rows = [_probe_row(record, arm, (document.get("gate") or [{}] * len(arms))[index], True)
            for index, (record, arm) in enumerate(zip(records, arms))]
    for index, row in enumerate(rows):
        row["arm_index"] = index
        row["submission_state"] = _submission_state(records[index])
        row["cost_evidence"] = costs.evidence(_train_metrics(records[index]), "train", document.get("cost_bps"))
    incomplete = _batch_incomplete_labels(rows, records)
    document.update({"rows": rows, "complete": not incomplete, "incomplete": incomplete,
                     "all_empty": bool(rows) and len(incomplete) == len(rows),
                     "comparisons": _probe_comparisons(rows, document.get("design"))})
    document["execution"] = {"total": len(rows), "finished": sum(_has_finished_result(r) for r in records),
        "train_evidence": len(rows) - len(incomplete),
        **{state: sum(row["submission_state"] == state for row in rows)
           for state in ("accepted", "not_submitted", "unknown_submission", "terminal")}}
    pending_quick_pnl = [arm["label"] for arm, record in zip(arms, records)
                         if ml.is_quick_simulation(record) and _probe_needs_train_pnl(record)
                         and (record.get("pnl_evidence") or {}).get("complete") is not True]
    if pending_quick_pnl:
        document["quick_pnl"] = {
            "pending_labels": pending_quick_pnl, "platform_read_only": True,
            "recover_argv": ["python", "research_cli.py", "--research-dir", research_paths.research_dir(),
                             "probe", "--recover", str(target)],
        }
    else:
        document.pop("quick_pnl", None)
    actions = []
    if any(_probe_needs_recovery(row) for row in rows):
        actions.append({"action": "recover", "batch_path": str(target), "platform_read_only": True})
    if any(row["submission_state"] == "not_submitted"
           or (row["submission_state"] == "accepted"
               and (row.get("progress_url") or _native_parent_pending(row)))
           for row in rows):
        actions.append({"action": "resume", "batch_path": str(target), "requires_explicit_request": True})
    if document.get("recording_policy", {}).get("enabled") and not document.get("recording"):
        actions.append({"action": "recover", "batch_path": str(target), "reason": "reconcile_recording"})
    document["next_actions"] = actions
    for action in actions:
        action["argv"] = ["python", "research_cli.py", "--research-dir", research_paths.research_dir(),
                          "probe", "--" + action["action"], str(target)]

def _probe_response(document, path, show_test=False, manifest=None, summary=False):
    keys = ("name", *_PROBE_CONTEXT_FIELDS, "track", "routes", "run_id", "started_at", "request_fingerprint",
            "arm_order", "design", "complete", "incomplete", "all_empty", "comparisons",
            "execution", "recording", "recording_policy", "next_actions", "quick_pnl", "preflight",
            "preflight_deferred",
            "concurrency", "scheduling", "capability_snapshot", "multi_size")
    rows = [_probe_visible_row(row, show_test) for row in document["rows"]]
    if manifest is not None:
        sources = {row.get("source_file") for row in manifest["rows"] if row.get("source_file")}
        matching_sources = []
        for source in sources:
            if Path(source).resolve() == Path(path).resolve():
                matching_sources.append(source)
                continue
            try:
                canonical, _saved = _load_probe_batch(source)
            except CliError:
                continue
            if canonical == Path(path).resolve():
                matching_sources.append(source)
        current = [{**row, "source_file": source} for source in matching_sources for row in rows
                   if row.get("alpha_id")]
        rows = _selection_call(selection.ordered_rows, manifest, current)
    response = {key: document[key] for key in keys if key in document}
    response.update({"path": str(path), "rows": rows, "show_test": show_test,
                     "simulated": document["execution"]["accepted"]})
    if document.get("recording"):
        response["registry"] = document["recording"]
    if manifest:
        response["selection_id"] = manifest["selection_id"]
    if summary:
        response["summary"] = _probe_summary(response)
    return response

def _probe_reconcile(document):
    policy = document.get("recording_policy")
    if not policy or not policy.get("enabled"):
        document["recording"] = {"state": "skipped" if policy else "deferred",
            "reason": "recording_disabled" if policy else "legacy_recording_policy_unknown",
            "recorded": 0, "appended": 0}
        return
    records = []
    for arm, record in zip(document["arms"], document["records"]):
        records.append({**record, "idea": arm.get("idea") or document.get("idea"),
            "note": arm.get("note") or document.get("note"),
            "provenance": {"run_id": document.get("run_id"),
                "request_fingerprint": document.get("request_fingerprint"),
                "source_spec": document.get("spec"), "arm_label": arm["label"]}})
    summary = alpha_registry.record_simulation(
        records, {}, idea=document.get("idea"), mechanism=document.get("mechanism"),
        note=document.get("note"), registry_path=policy["registry"], source="simulate", batch=document.get("name"))
    # Checkpoint callbacks may have already written rows during this invocation.
    summary["appended"] += (document.get("recording") or {}).get("appended", 0)
    document["recording"] = {"state": "reconciled", **summary}

def _probe_needs_train_pnl(record):
    return (bool(record.get("alpha_id")) and _has_finished_result(record)
            and bool(((record.get("out_of_sample") or {}).get("test") or {}).get("startDate"))
            and (not record.get("half_sharpes") or any(value is None for value in record["half_sharpes"])))

def _probe_halves(session, records, *, deadline=None, defer_quick_pnl=False):
    for record in records:
        if deadline is not None and time.monotonic() >= deadline:
            break
        if defer_quick_pnl and ml.is_quick_simulation(record):
            continue
        if not _has_finished_result(record) or not record.get("alpha_id"):
            continue
        boundary = ((record.get("out_of_sample") or {}).get("test") or {}).get("startDate")
        if not boundary:
            record["half_sharpes"] = [None, None]
            continue
        if record.get("half_sharpes") and all(value is not None for value in record["half_sharpes"]):
            continue
        from requests.exceptions import RequestException

        evidence = {}
        try:
            pnl = ml.fetch_daily_pnl(session, record["alpha_id"],
                cache_dir=Path(research_paths.ensure_dir()) / "pnl", max_age_hours=24, report=evidence,
                deadline=deadline)
        except (ml.BrainError, RequestException, OSError) as exc:
            record["half_sharpes"] = [None, None]
            record["half_sharpes_error"] = str(exc)
        else:
            record["pnl_evidence"] = evidence
            record["half_sharpes"] = half_split_sharpes(pnl, boundary) if evidence.get("complete") is True else (None, None)
            record.pop("half_sharpes_error", None)

def _probe_checkpoint_record(previous, arm, payload, *, submitting):
    merged = _merge_checkpoint_record(previous, payload)
    merged.update({"settings": dict(arm["settings"]), "expression": arm["expression"],
                   "mechanism": arm["label"], "decay": arm["settings"]["decay"]})
    merged["role"] = alpha_registry.evidence_role(arm, previous, payload)
    if submitting:
        merged.update({"submission_attempted": True, "status": "UNKNOWN_SUBMISSION"})
    if merged.get("alpha_id") or merged.get("progress_url") or merged.get("multi_parent_url"):
        merged["submission_attempted"] = True
    if _has_finished_result(merged):
        for key in ("progress_url", "next_poll_at", "metrics_pending"):
            if key not in payload:
                merged.pop(key, None)
    return merged

def _save_probe_checkpoint(document, target, *, indent=None):
    _probe_document_update(document, target)
    cache_store.write_json_atomic(target, document, indent=indent)

def _save_probe_records(document, target, updates, *, submitting=False):
    records, arms = document["records"], document["arms"]
    replacements = {
        index: _probe_checkpoint_record(records[index], arms[index], payload, submitting=submitting)
        for index, payload in updates
    }
    def persist(pending):
        if not submitting and (pending.get("recording_policy") or {}).get("enabled") and any(
            _has_finished_result(row) or _terminal_probe_without_alpha(row)
            for row in replacements.values()
        ):
            try:
                _probe_reconcile(pending)
            except (OSError, ValueError, cache_store.CacheError) as exc:
                note("probe seed recording deferred: %s" % exc)
        _save_probe_checkpoint(pending, target)

    _commit_checkpoint_records(document, replacements, persist)

def _save_probe_progress(document, target, indices, kind, payload):
    updates = ((indices[index], row) for index, row in _checkpoint_event_updates(kind, payload, len(indices)))
    _save_probe_records(document, target, updates, submitting=kind in {"submitting", "multi_submitting"})

def _matching_preflight_deferrals(document, account_fingerprint):
    """Reuse retry state only for the same account and exact saved requests."""
    entries, saved_arms = document.get("preflight_deferred"), document.get("arms")
    if not isinstance(entries, dict) or not isinstance(saved_arms, list):
        return {}
    arms = {arm["label"]: arm for arm in saved_arms
            if isinstance(arm, dict) and isinstance(arm.get("label"), str)}
    matched = {}
    for label, entry in entries.items():
        if not isinstance(entry, dict) or entry.get("account_fingerprint") != account_fingerprint:
            continue
        retry_at, labels = entry.get("retry_at"), entry.get("labels")
        if (type(retry_at) not in (int, float) or not math.isfinite(retry_at) or retry_at <= 0
                or not isinstance(labels, list) or not labels
                or any(not isinstance(name, str) or name not in arms for name in labels)):
            continue
        try:
            fingerprint = _request_fingerprint([arms[name] for name in labels])
        except (KeyError, TypeError, ValueError):
            continue
        if (entry.get("request_fingerprint") == fingerprint
                or (entry.get("request_fingerprint") is None and entry.get("code") == "RATE_LIMITED")):
            # Older checkpoints lack group digests, but an observed account
            # rate limit still applies to that endpoint's new requests.
            matched[label] = entry
    return matched

def matching_preflight_deferrals(document, account_fingerprint):
    """Return prior preflight deferrals in *document* matching *account_fingerprint*."""
    return _matching_preflight_deferrals(document, account_fingerprint)

def _execute_probe(args, document, target, groups):
    deadline = _simulation_deadline(args)
    proof_cache = {}
    deferred_preflights = document.get("preflight_deferred") or {}
    records = document["records"]
    state_lock = threading.Lock()
    if document.get("recording"):
        document["recording"]["appended"] = 0
    multi = _multi_options(args, document)
    if multi and getattr(args, "native_poll_once", False):
        multi["native_poll_once"] = True

    def save():
        _save_probe_checkpoint(document, target)

    def progress(indices, kind, payload):
        with state_lock:
            _save_probe_progress(document, target, indices, kind, payload)
        note("  %s" % kind)

    def run_group(session, group, *, launching=False, preflight=None, fetch_metrics=True):
        indices = [arm["arm_index"] for arm in group["arms"]]
        scheduling = {}
        document.setdefault("scheduling", []).append(scheduling)
        options = dict(labels=[arm["label"] for arm in group["arms"]],
            concurrency=document["concurrency"], capability_snapshot=document.get("capability_snapshot"),
            scheduling_report=scheduling, initial_records=[records[index] for index in indices],
            deadline=deadline, fetch_metrics=fetch_metrics,
            on_event=lambda kind, payload: progress(indices, kind, payload), **multi)
        if launching:
            produced = guarded_sim.simulate([expression for expression, _ in group["pairs"]],
                group["settings"], session=session, preflight=preflight,
                force=getattr(args, "allow_lint_errors", False), **options)
        else:
            options["recover_only"] = True
            produced = ml.simulate_alphas(session, group["pairs"], settings=group["settings"], **options)
        with state_lock:
            _save_probe_records(document, target, zip(indices, produced))
        return produced

    def recover_group(session, group):
        if time.monotonic() < deadline and any(
                _probe_needs_recovery(document["rows"][arm["arm_index"]]) for arm in group["arms"]):
            run_group(session, group)

    def work(session):
        _validate_saved_native_owner(session, document)
        account = simulation_capabilities.account_provenance(session)
        fingerprint = account.get("fingerprint")
        matched = _matching_preflight_deferrals(document, fingerprint)
        deferred_preflights.clear()
        deferred_preflights.update(matched)
        document.setdefault("preflight", [])
        save()
        def launches(group):
            return any(_submission_state(records[arm["arm_index"]]) == "not_submitted"
                       for arm in group["arms"])

        ordered = sorted(((launches(group), group) for group in groups.values()),
                         key=lambda item: (item[0], deferred_preflights.get(
                             item[1]["arms"][0]["label"], {}).get("observed_at", 0)))
        field_retry_at = max((item.get("retry_at", 0) for item in deferred_preflights.values()
                              if item.get("code") == "RATE_LIMITED"), default=0)
        def group_retry_at(group):
            prior = deferred_preflights.get(group["arms"][0]["label"], {})
            return max(field_retry_at, prior.get("retry_at", 0))

        last_launch = max((index for index, (launching, group) in enumerate(ordered)
                           if launching and group_retry_at(group) <= time.time()), default=-1)
        launch_limit = None
        capabilities_ready = False
        slots_reconciled = False
        deferred_metrics = []
        for position, (launching, group) in enumerate(ordered):
            if time.monotonic() >= deadline:
                break
            preflight = None
            if launching:
                if group_retry_at(group) > time.time():
                    recover_group(session, group)
                    continue
                retry_at = simulation_slots.submission_retry_at(account)
                deferred = None
                if retry_at is not None:
                    deferred = {
                        "state": "deferred", "reason": "submission_cooldown",
                        "submission_retry_at": retry_at,
                    }
                else:
                    if not capabilities_ready:
                        # Recovery and cooled launches need no fresh OPTIONS or catalogs.
                        document["capability_snapshot"] = _invocation_capabilities(session, args, deadline=deadline)
                        plan = ml.concurrency_plan(session, document["concurrency"], document["capability_snapshot"])
                        launch_limit = plan["effective"]
                        if multi:
                            launch_limit = min(launch_limit, simulation_multi.CONSULTANT_PARENT_LIMIT)
                        capabilities_ready = True
                if deferred is None and launch_limit:
                    accepted = simulation_slots.confirmed_accepted_count(account)
                    if accepted is not None and accepted >= launch_limit and not slots_reconciled:
                        slots_reconciled = True
                        reconciliation = ml.recover_accepted_slots(session, account, deadline=deadline)
                        document.setdefault("scheduling", []).append({
                            "reason": "accepted_slot_reconciliation", **reconciliation})
                        accepted = simulation_slots.confirmed_accepted_count(account)
                    if accepted is not None and accepted >= launch_limit:
                        deferred = {
                            "state": "deferred", "reason": "shared_slots_occupied",
                            "confirmed_accepted_slots": accepted, "effective_limit": launch_limit,
                        }
                if deferred is not None:
                    document.setdefault("scheduling", []).append(deferred)
                    recover_group(session, group)
                    continue
                label = group["arms"][0]["label"]
                try:
                    preflight = _preflight_lint(session, group["pairs"], group["settings"], args,
                                                deadline=deadline, proof_cache=proof_cache)
                except FieldEvidenceRefreshError as exc:
                    evidence = exc.details["field_catalog"]
                    catalogs = evidence.get("catalogs") or []
                    observed_at = time.time()
                    attempts = deferred_preflights.get(label, {}).get("attempts", 0)
                    attempts = (attempts if type(attempts) is int and attempts >= 0 else 0) + 1
                    server_retry_at = max((row["retry_at"] for row in catalogs
                        if type(row.get("retry_at")) in (int, float) and math.isfinite(row["retry_at"])), default=0)
                    retry_at = max(server_retry_at, observed_at + min(300, 15 * 2 ** min(attempts - 1, 5)))
                    deferred_preflights[label] = {
                        "labels": [arm["label"] for arm in group["arms"]],
                        "code": exc.code, "message": exc.message, "hint": exc.hint,
                        "observed_at": observed_at, "retry_at": retry_at, "attempts": attempts,
                        "account_fingerprint": fingerprint,
                        "request_fingerprint": _request_fingerprint(group["arms"]),
                        "scope": evidence.get("scope"), "required_fields": evidence.get("required_fields"),
                        "catalogs": [row for row in catalogs if row.get("state") != "verified"],
                    }
                    document["preflight_deferred"] = deferred_preflights
                    save()
                    recover_group(session, group)
                    if exc.code == "RATE_LIMITED":
                        field_retry_at = retry_at
                        break
                    continue
                deferred_preflights.pop(label, None)
                document["preflight"].append(preflight)
                save()
            if time.monotonic() >= deadline:
                break
            fetch_metrics = position >= last_launch
            produced = run_group(session, group, launching=launching, preflight=preflight,
                                 fetch_metrics=fetch_metrics)
            if not fetch_metrics:
                waiting = [(arm, pair) for arm, pair, record in zip(group["arms"], group["pairs"], produced)
                           if record.get("alpha_id") and record.get("metrics_pending")
                           and not record.get("progress_url")]
                if waiting:
                    arms, pairs = zip(*waiting)
                    deferred_metrics.append(group if multi else
                        {"settings": group["settings"], "arms": arms, "pairs": pairs})
            if launching and any(
                record.get("progress_url") or record.get("status") in {"QUEUED", "UNKNOWN_SUBMISSION"}
                or (record.get("status") == "PENDING"
                    and not (record.get("alpha_id") and record.get("metrics_pending")))
                for record in produced
            ):
                break
        for group in deferred_metrics:
            if time.monotonic() >= deadline:
                break
            run_group(session, group)
        _probe_halves(session, records, deadline=deadline,
                      defer_quick_pnl=getattr(args, "defer_quick_pnl", False))

    document["concurrency"] = _requested_concurrency(args, document)
    save()
    if groups:
        with_session(args, work, deadline=deadline)
    _probe_reconcile(document)
    save()
    return _probe_response(document, target, summary=getattr(args, "summary", False))

def _run_probe(args, spec, name, groups, design, gate, target):
    arms = sorted([arm for group in groups.values() for arm in group["arms"]], key=lambda arm: arm["arm_index"])
    document = {"format": "probe-batch", "version": 2, "spec": str(args.spec), "name": name,
        "track": spec["track"], "routes": _probe_routes(spec["track"], arms),
        **{key: spec[key] for key in _PROBE_CONTEXT_FIELDS if key in spec},
        **_multi_options(args),
        "run_id": str(uuid.uuid4()), "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "request_fingerprint": _request_fingerprint(arms), "arm_order": [arm["label"] for arm in arms],
        "arms": arms, "design": design, "gate": gate, "cost_bps": getattr(args, "cost_bps", None),
        "recording_policy": {"enabled": not getattr(args, "no_record", False),
            "registry": str(Path(getattr(args, "registry", None) or alpha_registry.default_registry_path()).resolve())},
        "records": [{"expression": arm["expression"], "decay": arm["settings"]["decay"],
                     "settings": dict(arm["settings"]), "mechanism": arm["label"], "status": "QUEUED",
                     "role": arm.get("role"),
                     "alpha_id": None, "metrics": None, "submission_attempted": False} for arm in arms]}
    return _execute_probe(args, document, target, groups)

def _reject_saved_overrides(args, mode="resume/recover"):
    changing = ("setting", "region", "universe", "instrument_type", "neutralization",
                "decay", "delay", "truncation", "pasteurization", "nan_handling", "test_period", "multi_size")
    if (any(getattr(args, key, None) is not None for key in changing)
            or getattr(args, "quick_mode", False) or getattr(args, "full_mode", False)):
        raise CliError("USAGE", f"{mode} uses saved settings; explicit overrides are not allowed",
                       "omit simulation settings flags; start a new spec if you need different settings")

def _cmd_probe_resume(args):
    _reject_saved_overrides(args)
    changing = ("spec", "name", "batch_path", "registry", "cost_bps", "region", "universe",
                "instrument_type", "neutralization", "decay", "delay", "truncation",
                "pasteurization", "nan_handling", "test_period", "force", "no_record")
    if any(getattr(args, key, None) is not None and getattr(args, key) is not False for key in changing):
        raise CliError("USAGE", "--resume uses saved requests and recording policy; parameters cannot change",
                       "pass only `probe --resume BATCH.json` (plus optional --batch-size / --max-wait-seconds / --dry-run)")
    path, document = _load_probe_batch(args.resume)
    protect_output(args, path)

    def groups_for(saved):
        groups = {}
        native_members = {index for indices in _native_probe_groups(saved) for index in indices}
        for arm, record in zip(saved["arms"], saved["records"]):
            if record.get("multi_request_id"):
                if arm["arm_index"] not in native_members:
                    continue
                state = "native_parent"
            else:
                if _terminal_probe_without_alpha(record) or _has_finished_result(record):
                    continue
                _selection_call(simulation_slots.restore_record, record)
                state = _submission_state(record)
                if state == "unknown_submission":
                    continue
            if not arm.get("settings") or not arm.get("expression"):
                raise CliError("INVALID_INPUT", "legacy batch lacks exact requests; cannot resume",
                               "use `probe --recover` to read already-accepted jobs, or run a new spec")
            if state == "not_submitted" and ml.resolve_settings(arm["settings"]) != arm["settings"]:
                raise CliError("INVALID_INPUT", "legacy batch lacks complete resolved settings; cannot start a new POST",
                               "use `probe --recover` for existing tasks or draft a new spec")
            group = groups.setdefault((_settings_signature(arm["settings"]), state),
                                      {"settings": arm["settings"], "arms": [], "pairs": []})
            group["arms"].append(arm)
            group["pairs"].append((arm["expression"], arm["settings"].get("decay")))
        return groups

    if getattr(args, "dry_run", False):
        document["concurrency"] = _requested_concurrency(args, document)
        groups = groups_for(document)
        multi = _multi_options(args, document)
        requests = []
        if multi:
            for (_signature, state), group in groups.items():
                if state == "not_submitted":
                    _records, _scheduling, group_requests = _preview_probe_group(
                        args, group, document["concurrency"], multi)
                    requests.extend(group_requests)
        _probe_document_update(document, path)
        return {**_probe_response(document, path, summary=getattr(args, "summary", False)), "dry_run": True, "resume": True,
                **({"multi_requests": requests} if multi else {}),
                "continuing": [arm["label"] for group in groups.values() for arm in group["arms"]],
                "blocked_unknown": [arm["label"] for arm, record in zip(document["arms"], document["records"])
                                    if _submission_state(record) == "unknown_submission"]}
    try:
        with cache_store.file_lock(str(path) + ".lock", timeout=0):
            path, document = _load_probe_batch(path)
            return _execute_probe(args, document, path, groups_for(document))
    except cache_store.LockTimeout as exc:
        raise CliError("SERVICE_ERROR", "this probe batch is already being updated",
                       "inspect the existing writer; keep this batch and its lock",
                       details=_checkpoint_lock_details(path)) from exc

from .exp_compare import (  # noqa: E402
    half_split_sharpes, train_half_sharpes, _train_metrics, _finished, summarize, _metric_value, _compact_metrics, cmd_ablate, _compare_inputs, _compare_document, _compare_row, _selection_call, cmd_compare, _require_spec, _format_fastexpr, _write_formatted_spec, cmd_fmt,
)
from .exp_sim import (  # noqa: E402
    _write_simulation_checkpoint, _CHECKPOINT_EVENTS, _CHECKPOINT_POLL_FIELDS, _MULTI_CHECKPOINT_EVENTS, _MULTI_CHECKPOINT_FIELDS, _merge_checkpoint_record, _checkpoint_event_updates, _commit_checkpoint_records, _simulation_checkpoint_callback, _sort_simulation_results, _simulation_result, _simulation_checkpoint_document, _checkpoint_lock_details, _locked_simulation_checkpoint, _finalize_simulation, cmd_simulate, cmd_resume, _resume_simulation, _record_simulation,
)
from .exp_probe_spec import (  # noqa: E402
    _PROBE_CONTEXT_FIELDS, _load_probe_spec, _normalize_probe_spec, _DESIGNS, _probe_design, validate_probe_design, _finite_metric, finite_metric, _has_finished_result, _batch_incomplete_labels, _probe_comparisons, _probe_visible_row, _turnover_band, _settings_signature, _PROBE_REFERENCE_LIMIT, _probe_history_ref, _probe_identity, probe_identity, _scan_novelty_history, _novelty_gate, _probe_checkpoint_refs, _recoverable_probe_reference, _probe_inputs, _prepare_probe, probe_spec_identity, preview_probe_spec, _request_fingerprint,
)
from .exp_probe_recovery import (  # noqa: E402
    _recovery_expression_key, _probe_recovery_index, _terminal_probe_without_alpha, _probe_needs_recovery, _native_parent_pending, _recover_probe_rows, _recovery_language, _load_probe_batch, load_probe_batch, _cmd_probe_recover, _native_probe_groups, _recover_probe_batch,
)
