"""Autonomous state-machine step workers and orphan slot reconciliation."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from datetime import date
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean, stdev
from threading import Lock
import time

from requests.exceptions import RequestException

import alpha_registry
import cache_store
from check_evidence import summarize_checks
import machine_lib as ml
import simulation_capabilities
import simulation_multi
import simulation_slots
import spc_experiment

from . import batch_admission, brief, buffer_cmd, experiments, pp_budget, selection, state_machine, validation


def progress_hash(doc):
    """Return a stable fingerprint of durable scientific and platform execution state."""
    records = doc.get("records") or []
    full_records = doc.get("full_records") or []
    payload = {
        "state": doc.get("state"),
        "records": [
            {k: rec.get(k) for k in ("label", "status", "alpha_id", "progress_url", "multi_parent_url", "metrics_pending", "error")}
            for rec in records if isinstance(rec, dict)
        ],
        "full_records": [
            {k: rec.get(k) for k in ("label", "status", "alpha_id", "progress_url", "multi_parent_url", "metrics_pending", "error")}
            for rec in full_records if isinstance(rec, dict)
        ],
        "submit_attempt": doc.get("submit_attempt"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _record_deferral(doc, cfg, *, retry_at=None, reason=None):
    current_hash = progress_hash(doc)
    if current_hash != doc.get("progress_hash"):
        doc["progress_hash"] = current_hash
        doc["unchanged_defers"] = 0
        doc["stalled"] = False
    elif retry_at is None or retry_at <= time.time():
        doc["unchanged_defers"] = int(doc.get("unchanged_defers") or 0) + 1
    else:
        # An explicit future Retry-After window explains waiting without declaring stall.
        doc["unchanged_defers"] = 0

    max_defers = int(cfg.get("max_unchanged_defers", 3))
    if retry_at is not None and retry_at > time.time():
        doc["next_wake_at"] = retry_at
    elif doc["unchanged_defers"] >= max_defers:
        if doc.get("state") == "AUTO_PREFLIGHT":
            doc["state"] = "GATE_PREFLIGHT_BLOCKED"
            doc["preflight_block"] = {"reason": reason or "stalled_deferral", "defers": doc["unchanged_defers"]}
            doc["next_wake_at"] = None
        else:
            doc["stalled"] = True
            doc["next_wake_at"] = time.time() + 600
    else:
        doc["next_wake_at"] = time.time() + 10


def _step_preflight(engine, doc, session, cfg):
    arms = doc.get("arms") or []
    try:
        settings = state_machine.shared_arm_settings(arms)
    except state_machine.CliError as exc:
        doc["state"] = "GATE_PREFLIGHT_BLOCKED"
        doc["preflight_block"] = {"reason": "incompatible_arm_settings", "message": str(exc)}
        doc["next_wake_at"] = None
        return
    registry_path = str(engine.root / "alpha-registry.jsonl")

    # 1. Exact duplicate gate against previously finished ledger items.
    novelty = experiments._novelty_gate(arms, registry_path=registry_path)
    exact_completed = [
        item for item in (novelty or [])
        if isinstance(item, dict) and item.get("completed_refs")
    ]
    if exact_completed and len(exact_completed) == len(arms):
        doc["state"] = "GATE_PREFLIGHT_BLOCKED"
        doc["preflight_block"] = {"reason": "exact_repeat", "matches": exact_completed}
        doc["next_wake_at"] = None
        return

    # 2. Scope field proof warming + exploration diversity checks.
    account = simulation_capabilities.account_provenance(session)

    if doc.get("design", {}).get("mode") == "explore" and len(arms) >= 16:
        admission = batch_admission.build_admission(arms, session, account)
        if not admission["ok"]:
            retry_at = admission.get("next_retry_at")
            if retry_at:
                _record_deferral(doc, cfg, retry_at=retry_at, reason="admission_backoff")
                return
            doc["state"] = "GATE_PREFLIGHT_BLOCKED"
            doc["preflight_block"] = {"reason": "exploration_admission", "blocking": admission.get("blocking")}
            doc["next_wake_at"] = None
            return

    # 3. Scoped field warming & online preflight lint.
    pairs = ml.normalize_alphas([arm["expression"] for arm in arms], settings)
    try:
        from types import SimpleNamespace
        validation._preflight_lint(session, pairs, settings, SimpleNamespace(no_cache=False), proof_cache={})
    except (ml.CaptchaRequired, ml.RateLimited, ml.ServiceError, RequestException):
        raise
    except (ml.BrainError, state_machine.CliError, ValueError) as exc:
        doc["state"] = "GATE_PREFLIGHT_BLOCKED"
        doc["preflight_block"] = {
            "reason": "preflight_rejected",
            "code": getattr(exc, "code", type(exc).__name__),
            "message": str(exc),
            "details": getattr(exc, "details", None),
        }
        doc["next_wake_at"] = None
        return

    doc["preflight_complete"] = True
    doc["preflight_block"] = None
    doc["unchanged_defers"] = 0
    doc["stalled"] = False
    doc["next_wake_at"] = None
    doc["state"] = "AUTO_QUICK_SIM"


def _simulation_checkpoint(engine, doc, records_key):
    state = doc["state"]
    lock = Lock()

    def on_event(kind, payload):
        if kind not in {"submitting", "submitted", "completed", "failed", "deferred", "pending"} and "records" not in payload:
            return
        snapshots = payload.get("records", [payload])
        with lock:
            records = doc[records_key]
            for snapshot in snapshots:
                index = snapshot.get("index")
                if type(index) is int and 0 <= index < len(records):
                    # The simulator emits the whole receipt, including removals after polling.
                    records[index] = {key: value for key, value in snapshot.items() if key != "index"}
            _persist_worker_progress(
                engine, doc, state, (records_key,),
                merge_on_cancel=kind not in {"submitting", "multi_submitting"},
            )

    return on_event


def _step_quick_sim(engine, doc, session, cfg):
    arms = doc.get("arms") or []
    records = doc.get("records") or []
    try:
        settings = state_machine.shared_arm_settings(arms)
    except state_machine.CliError as exc:
        doc["state"] = "GATE_PREFLIGHT_BLOCKED"
        doc["preflight_block"] = {"reason": "incompatible_arm_settings", "message": str(exc)}
        doc["next_wake_at"] = None
        return
    account = simulation_capabilities.account_provenance(session)
    quota = simulation_slots.quota_snapshot(account)
    quota_floor = int(cfg.get("quota_floor", 0))
    pending_new = any(not rec.get("submission_attempted") for rec in records)
    if pending_new and quota.get("blocked"):
        doc["next_wake_at"] = quota.get("reset_at") or (time.time() + 120)
        doc["last_error"] = {"code": "RATE_LIMITED", "message": "daily quota blocked"}
        return
    if pending_new and isinstance(quota.get("remaining"), (int, float)) and quota["remaining"] <= quota_floor and quota_floor > 0:
        doc["next_wake_at"] = quota.get("reset_at") or (time.time() + 120)
        doc["last_error"] = {"code": "RATE_LIMITED", "message": f"quota floor {quota_floor} reached (remaining={quota['remaining']})"}
        return

    expressions = [arm["expression"] for arm in arms]
    labels = [arm["label"] for arm in arms]
    multi_size = min(int(cfg.get("multi_size", 10) or 1), len(arms))
    if multi_size < 2 or ml.is_python_language(settings.get("language")):
        multi_size = None
    doc["records"] = records

    try:
        result_records = ml.simulate_alphas(
            session,
            expressions,
            settings=settings,
            labels=labels,
            concurrency=int(cfg.get("concurrency", 8)),
            multi_size=multi_size,
            deadline=time.monotonic() + 25.0,
            fetch_metrics=True,
            initial_records=records,
            on_event=_simulation_checkpoint(engine, doc, "records"),
        )
    except ml.RateLimited as exc:
        wait = getattr(exc, "retry_after", None) or 30
        _record_deferral(doc, cfg, retry_at=time.time() + wait, reason="RATE_LIMITED")
        doc["last_error"] = {"code": "RATE_LIMITED", "message": str(exc)}
        return

    for idx, rec in enumerate(result_records):
        if idx < len(records):
            records[idx] = {**records[idx], **rec}
    doc["records"] = records

    # Refresh execution summary & rows using existing probe row summarization.
    rows = []
    registry_path = str(engine.root / "alpha-registry.jsonl")
    finished_terminal = 0
    failed_terminal = 0
    accepted = 0
    for arm, rec in zip(arms, records):
        status = str(rec.get("status") or "QUEUED").upper()
        if rec.get("submission_attempted") or rec.get("progress_url") or rec.get("alpha_id"):
            accepted += 1
        split = rec.get("out_of_sample") or {}
        train = split.get("train") if isinstance(split, dict) else {}
        if status in alpha_registry.FINISHED_STATUSES and rec.get("alpha_id") and not rec.get("metrics_pending"):
            finished_terminal += 1
            entry = alpha_registry.build_entry(
                {"alpha_id": rec["alpha_id"], "expression": arm["expression"],
                 "settings": arm["settings"], "metrics": rec.get("metrics") or {},
                 "checks": rec.get("checks") or {}, "out_of_sample": split,
                 "status": status},
                idea=arm.get("idea") or doc.get("idea"),
                mechanism=arm.get("mechanism") or doc.get("mechanism") or arm["label"],
                source="simulate",
                family=arm.get("family"),
                batch=doc["run_id"],
            )
            entry["role"] = arm.get("role")
            alpha_registry.record_entry(entry, path=registry_path)
        elif status in (simulation_slots.TERMINAL_FAILURE_STATUSES | {"FAILED"}):
            failed_terminal += 1
            alpha_registry.record_entry({
                "key": f"failure:{hashlib.sha256((arm['expression'] + str(doc['run_id']) + arm['label']).encode()).hexdigest()[:16]}",
                "alpha_id": None,
                "execution_outcome": "failed",
                "expression": arm["expression"],
                "settings": arm["settings"],
                "status": status,
                "error": rec.get("error"),
                "idea": arm.get("idea") or doc.get("idea"),
                "mechanism": arm.get("mechanism") or doc.get("mechanism"),
                "source": "simulate",
                "batch": doc["run_id"],
            }, path=registry_path)

        rows.append({
            "label": arm["label"],
            "role": arm.get("role"),
            "status": status,
            "alpha_id": rec.get("alpha_id"),
            "train_sharpe": train.get("sharpe") if isinstance(train, dict) else None,
            "train_fitness": train.get("fitness") if isinstance(train, dict) else None,
            "train_turnover": train.get("turnover") if isinstance(train, dict) else None,
            "train_margin": train.get("margin") if isinstance(train, dict) else None,
            "error": rec.get("error"),
        })

    doc["rows"] = rows
    total = len(arms)
    not_sub = sum(1 for rec in records if not rec.get("submission_attempted") and str(rec.get("status") or "").upper() == "QUEUED")
    doc["execution"] = {
        "total": total,
        "not_submitted": not_sub,
        "accepted": accepted,
        "finished": finished_terminal,
        "terminal": failed_terminal,
    }

    if finished_terminal + failed_terminal < total:
        poll_times = [rec.get("next_poll_at") for rec in records if isinstance(rec.get("next_poll_at"), (int, float))]
        next_poll = min(poll_times) if poll_times else None
        _record_deferral(doc, cfg, retry_at=next_poll, reason="platform_pending")
        return

    doc["comparisons"] = experiments._probe_comparisons(rows, doc.get("design") or {"mode": "explore"})
    has_candidate = any(
        row.get("alpha_id") and experiments.finite_metric(row.get("train_sharpe"))
        for row in rows
    )
    doc["unchanged_defers"] = 0
    doc["stalled"] = False
    doc["next_wake_at"] = None
    doc["state"] = "GATE_TRIAGE" if has_candidate else "FAILED"


def _step_hydrate_pnl(engine, doc, session, cfg):
    from . import pnl_snapshot

    selected = doc.get("selected_arms") or []
    pnl_dir = engine.root / "pnl"
    pnl_dir.mkdir(parents=True, exist_ok=True)
    registry_path = str(engine.root / "alpha-registry.jsonl")
    entries = {e.get("alpha_id"): e for e in alpha_registry.load_entries(registry_path) if e.get("alpha_id")}

    by_label = {}
    fetched_items = []
    for item in selected:
        alpha_id = item["alpha_id"]
        split = (entries.get(alpha_id) or {}).get("out_of_sample") or {}
        train_start = (split.get("train") or {}).get("startDate")
        test_start = (split.get("test") or {}).get("startDate")
        try:
            has_train_window = date.fromisoformat(str(train_start)[:10]) < date.fromisoformat(str(test_start)[:10])
        except ValueError:
            has_train_window = False
        if not has_train_window:
            by_label[item["label"]] = {
                "alpha_id": alpha_id,
                "yearly_train": {"state": "unavailable", "reason": "train_window_not_verified"},
                "gate_pass": False,
            }
            continue
        report = {}
        try:
            ml.fetch_daily_pnl(session, alpha_id, cache_dir=pnl_dir, report=report, max_wait_seconds=25)
        except (ml.CaptchaRequired, ml.RateLimited, ml.ServiceError, RequestException):
            raise
        except ml.BrainError as exc:
            by_label[item["label"]] = {"alpha_id": alpha_id, "yearly_train": {"state": "unavailable", "reason": str(exc)}, "gate_pass": False}
            continue
        fetched_items.append((item, report, {"alpha_id": alpha_id, "train_start": train_start, "test_start": test_start}))

    if fetched_items:
        yearly_results = pnl_snapshot.yearly_train_batch([row for _, _, row in fetched_items], pnl_dir)
        for index, (item, report, _) in enumerate(fetched_items):
            yearly = yearly_results[index] if index < len(yearly_results) else None
            yearly = yearly or {"state": "unavailable", "reason": "missing_yearly_train_result"}
            by_label[item["label"]] = {
                "alpha_id": item["alpha_id"],
                "yearly_train": yearly,
                "pnl_fetch": report,
                "gate_pass": yearly.get("state") == "pass",
            }

    doc["hydration"] = {"by_label": by_label}
    doc["unchanged_defers"] = 0
    doc["stalled"] = False
    doc["next_wake_at"] = None
    doc["state"] = "GATE_APPROVE_FULL"


def _series_map(series_entry, field_name):
    if not isinstance(series_entry, dict) or series_entry.get("complete") is False:
        return {}
    properties = (series_entry.get("schema") or {}).get("properties") or []
    columns = {prop.get("name"): i for i, prop in enumerate(properties) if isinstance(prop, dict)}
    indices = (columns.get("date"), columns.get(field_name))
    result = {}
    for row in series_entry.get("records") or []:
        if isinstance(row, dict):
            day, value = row.get("date"), row.get(field_name)
        elif (isinstance(row, (list, tuple)) and None not in indices
              and len(row) > max(indices)):
            day, value = row[indices[0]], row[indices[1]]
        else:
            continue
        if not isinstance(day, str) or not experiments.finite_metric(value):
            continue
        try:
            day = date.fromisoformat(day).isoformat()
        except ValueError:
            continue
        if day in result and result[day] != value:
            return {}
        result[day] = value
    return result


def _compute_cost_audit(payload, recordsets):
    metrics = payload.get("is") or payload.get("metrics") or {}
    book_size = metrics.get("bookSize")
    if not experiments.finite_metric(book_size) or book_size <= 0:
        return None
    series = (recordsets or {}).get("series") or {}
    # The same daily-pnl series feeds yearly Sharpe and summed weekly PnL.
    pnl_map = _series_map(series.get("daily-pnl"), "pnl")
    turnover_map = _series_map(series.get("turnover"), "turnover")
    if any(value < 0 for value in turnover_map.values()):
        return None
    days = sorted(pnl_map.keys() & turnover_map.keys())
    if len(days) < 20:
        return None
    daily_traded = [turnover_map[day] * book_size for day in days]
    traded = sum(daily_traded)
    if not experiments.finite_metric(traded) or traded <= 0:
        return None
    scenarios = []
    for bps in (5, 10, 20):
        net = [pnl_map[day] - bps / 10000.0 * amount for day, amount in zip(days, daily_traded)]
        if not all(experiments.finite_metric(value) for value in net):
            return None
        volatility = stdev(net)
        scenarios.append({
            "cost_bps": bps,
            "net_sharpe": round(mean(net) / volatility * sqrt(252), 3) if volatility > 0 else None,
            "net_margin_bps": round(sum(net) / traded * 10000.0, 2),
        })
    return {"joint_sessions": len(days), "scenarios": scenarios}


def _step_full_validate(engine, doc, session, cfg):
    approved = doc.get("approved_arms") or []
    try:
        if doc.get("selection_frozen") is not True or len(approved) != 1:
            raise ValueError("FULL confirmation needs exactly one frozen selection")
        manifest = selection.load(doc.get("selection_path"), source="triage")
        if (manifest["selection_id"] != doc.get("selection_id")
                or manifest["inputs"].get("run_id") != doc["run_id"]
                or manifest["window"].get("train_only") is not True
                or len(manifest["rows"]) != 1):
            raise ValueError("FULL confirmation selection manifest does not match the run")
        frozen = manifest["rows"][0]
        if (frozen.get("label"), frozen.get("alpha_id")) != (approved[0].get("label"), approved[0].get("alpha_id")):
            raise ValueError("approved arm does not match the frozen selection")
        full_arm = brief.full_arm(frozen)
        settings = ml.resolve_settings(full_arm["settings"])
    except (KeyError, TypeError, ValueError) as exc:
        doc["state"] = "FAILED"
        doc["next_wake_at"] = None
        doc["last_error"] = {"code": "INVALID_INPUT", "message": str(exc)}
        return

    receipt_identity = {"label": full_arm["label"], "expression": full_arm["expression"], "settings": settings}
    full_records = doc.get("full_records") or [{
        **receipt_identity, "arm_index": 0, "status": "QUEUED", "submission_attempted": False,
    }]
    if len(full_records) != 1 or any(full_records[0].get(key) != value for key, value in receipt_identity.items()):
        doc["state"] = "FAILED"
        doc["next_wake_at"] = None
        doc["last_error"] = {"code": "INVALID_INPUT", "message": "FULL receipt does not match the frozen selection"}
        return

    doc["full_records"] = full_records
    terminal = alpha_registry.FINISHED_STATUSES | simulation_slots.TERMINAL_FAILURE_STATUSES | {"FAILED"}
    candidate_rec = full_records[0]
    if str(candidate_rec.get("status") or "").upper() not in terminal or candidate_rec.get("metrics_pending"):
        if not candidate_rec.get("submission_attempted"):
            # FULL ignores the soft quota floor. Accepted work remains readable when quota is exhausted.
            quota = simulation_slots.quota_snapshot(simulation_capabilities.account_provenance(session))
            if quota.get("blocked"):
                doc["next_wake_at"] = quota.get("reset_at") or (time.time() + 120)
                return
        result_records = ml.simulate_alphas(
            session,
            [full_arm["expression"]],
            settings=settings,
            labels=[full_arm["label"]],
            concurrency=int(cfg.get("concurrency", 8)),
            deadline=time.monotonic() + 25.0,
            fetch_metrics=True,
            initial_records=full_records,
            on_event=_simulation_checkpoint(engine, doc, "full_records"),
        )
        if (len(result_records) != 1 or result_records[0].get("index", 0) != 0
                or any(result_records[0].get(key, value) != value for key, value in receipt_identity.items())):
            doc["state"] = "FAILED"
            doc["next_wake_at"] = None
            doc["last_error"] = {"code": "SIMULATION_FAILED", "message": "FULL result does not match the frozen selection"}
            return
        candidate_rec = {**receipt_identity, "arm_index": 0, **result_records[0]}
        doc["full_records"] = [candidate_rec]
        _persist_worker_progress(engine, doc, "AUTO_FULL_VALIDATE", ("full_records",))
        if str(candidate_rec.get("status") or "").upper() not in terminal or candidate_rec.get("metrics_pending"):
            _record_deferral(doc, cfg, retry_at=candidate_rec.get("next_poll_at"), reason="full_pending")
            return

    full_alpha_id = candidate_rec.get("alpha_id")
    if (str(candidate_rec.get("status") or "").upper() not in alpha_registry.FINISHED_STATUSES
            or not full_alpha_id or full_alpha_id == frozen["alpha_id"]):
        doc["state"] = "FAILED"
        doc["next_wake_at"] = None
        doc["last_error"] = {"code": "SIMULATION_FAILED", "message": "FULL confirmation simulation failed"}
        return

    payload = ml.fetch_alpha_metrics(session, full_alpha_id)
    if payload.get("id") not in (None, full_alpha_id):
        doc["state"] = "FAILED"
        doc["next_wake_at"] = None
        doc["last_error"] = {"code": "SIMULATION_FAILED", "message": "FULL metrics refer to a different Alpha ID"}
        return
    if payload.get("_deferred"):
        _record_deferral(doc, cfg, retry_at=time.time() + 15, reason="full_metrics_pending")
        return
    checks_report = ml.fetch_alpha_checks(session, full_alpha_id, max_wait_seconds=10)
    self_corr = ml.fetch_self_correlation(session, full_alpha_id)
    reported_rows = self_corr.get("records") or []
    correlation_rows = [row for row in reported_rows if isinstance(row, (list, tuple))
                        and len(row) > 5 and experiments.finite_metric(row[5]) and abs(row[5]) <= 1]
    sc_verdict = ml.self_correlation_verdict({**self_corr, "records": correlation_rows})
    if len(correlation_rows) != len(reported_rows) and not sc_verdict["blocked"]:
        sc_verdict.update(known=False, reason="malformed self-correlation records")
    sc_check = checks_report.get("self_correlation") or {}
    sc_result = str(sc_check.get("result") or "").strip().upper()
    check_value = sc_check.get("value")
    sc_verdict["value"] = (check_value if experiments.finite_metric(check_value) and abs(check_value) <= 1
                           else max((abs(row[5]) for row in correlation_rows), default=None))
    if sc_result == "PASS" and not sc_verdict["known"]:
        sc_verdict.update(known=True, reason="live self-correlation check passed")
    recordsets = {}
    try:
        recordsets = ml.fetch_alpha_recordsets(session, full_alpha_id, ("daily-pnl", "turnover"), max_wait_seconds=10)
    except (ml.CaptchaRequired, ml.RateLimited, ml.ServiceError, RequestException):
        raise
    except ml.BrainError:
        pass

    if recordsets and recordsets.get("alpha_id") not in (None, full_alpha_id):
        recordsets = {}

    split = ml.extract_split(payload) or {}
    test_sharpe = (split.get("test") or {}).get("sharpe")
    if not experiments.finite_metric(test_sharpe) and split.get("test"):
        split["test"] = {**split["test"], "sharpe": None}
    checks = checks_report.get("checks") or {}
    check_summary = summarize_checks(checks)
    failed_checks = sorted(set(check_summary["failed"]) - set(check_summary["errored"]))
    validation_result = {
        "full_alpha_id": full_alpha_id,
        "test_sharpe": test_sharpe if experiments.finite_metric(test_sharpe) else None,
        "self_correlation": sc_verdict,
        "failed_checks": failed_checks,
        "checks_complete": checks_report.get("complete") is True and check_summary["resolved"],
    }
    doc.update(full_alpha_id=full_alpha_id, full_checks=checks_report,
               cost_audit=_compute_cost_audit(payload, recordsets), validation=validation_result)

    failed_reason = None
    if not experiments.finite_metric(test_sharpe):
        failed_reason = "test_sharpe_unavailable"
    elif test_sharpe <= 0:
        failed_reason = "nonpositive_test_sharpe"
    elif failed_checks or sc_verdict.get("blocked") or sc_result == "FAIL":
        failed_reason = "failed_platform_checks"

    if doc.get("track") == "POWER_POOL":
        budget = pp_budget.alpha_budget(full_arm["expression"], settings, checks=checks)
        validation_result["pp_budget"] = budget
        if isinstance(budget, dict) and ((budget.get("gates") or {}).get("blocking")
                                         or (budget.get("budget") or {}).get("ok") is False):
            failed_reason = failed_reason or "power_pool_budget"

    if not failed_reason and not (validation_result["checks_complete"] and sc_verdict["known"]):
        _record_deferral(doc, cfg, retry_at=time.time() + 15, reason="checks_pending")
        return

    # Record one final observation, after checks have resolved or a rejection is proven.
    arm_meta = next((arm for arm in doc.get("arms", []) if arm.get("label") == frozen["label"]), {})
    entry = alpha_registry.build_entry(
        {"alpha_id": full_alpha_id, "expression": full_arm["expression"],
         "settings": settings, "metrics": payload.get("is") or {}, "checks": checks,
         "out_of_sample": split, "status": payload.get("status") or "COMPLETE"},
        idea=arm_meta.get("idea") or doc.get("idea"),
        mechanism=arm_meta.get("mechanism") or doc.get("mechanism") or frozen["label"],
        source="simulate", family=arm_meta.get("family"), batch=doc["run_id"],
    )
    if failed_reason:
        validation_result["failed_reason"] = failed_reason
        entry.update(verdict="rejected", validation_failure=failed_reason)
    alpha_registry.record_entry(entry, path=str(engine.root / "alpha-registry.jsonl"))

    doc["unchanged_defers"] = 0
    doc["stalled"] = False
    doc["next_wake_at"] = None
    doc["state"] = "REJECTED_AT_VALIDATION" if failed_reason else "GATE_DESCRIBE"


def _step_buffer(engine, doc, session, cfg):
    alpha_id = doc.get("full_alpha_id")
    meta = doc.get("candidate_metadata") or {}
    if not alpha_id or not meta:
        doc["state"] = "FAILED"
        doc["last_error"] = {"code": "INVALID_INPUT", "message": "missing full_alpha_id or candidate_metadata"}
        return

    parity_key = {"alpha_id": alpha_id, "metadata": meta}
    _persist_worker_progress(engine, doc, "AUTO_BUFFER", ())
    if doc.get("buffer_metadata_verified") != parity_key:
        ml.set_alpha_metadata(
            session,
            alpha_id,
            name=meta["name"],
            category=meta["category"],
            tags=meta["tags"],
            description=meta["description"],
        )
    payload = ml.fetch_alpha_metrics(session, alpha_id)
    code, live_desc, _ = ml.regular_expression(payload)
    if payload.get("_deferred"):
        _record_deferral(doc, cfg, reason="buffer_metadata_pending")
        return
    if payload.get("id") not in (None, alpha_id) or (live_desc or "").strip() != meta["description"].strip():
        doc.pop("buffer_metadata_verified", None)
        doc["state"] = "GATE_DESCRIBE"
        doc["next_wake_at"] = None
        doc["last_error"] = {"code": "INVALID_INPUT", "message": "remote description did not match submitted description"}
        return

    doc["buffer_metadata_verified"] = parity_key
    _persist_worker_progress(engine, doc, "AUTO_BUFFER", ("buffer_metadata_verified",))
    seed_path = str(engine.root / buffer_cmd.SEEDS_FILENAME)
    shelf_path = str(engine.root / buffer_cmd.BUFFER_FILENAME)
    split = ml.extract_split(payload) or {}
    seed_row = {
        "alpha_id": alpha_id,
        "name": meta["name"],
        "category": meta["category"],
        "tags": meta["tags"],
        "description": meta["description"],
        "expression": code,
        "settings": dict(payload.get("settings") or {}),
        "metrics": {"train": split.get("train"), "test": split.get("test"), "full": payload.get("is")},
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    buffer_cmd._write_seed(seed_path, seed_row)
    checks_map = (doc.get("full_checks") or {}).get("checks") or {}
    shelf_entry = {
        "event": "shelved",
        "alpha_id": alpha_id,
        "selection_id": doc.get("selection_id"),
        "name": meta["name"],
        "category": meta["category"],
        "self_correlation": ((doc.get("validation") or {}).get("self_correlation") or {}).get("value"),
        "train": split.get("train"),
        "test": split.get("test"),
        "checks": buffer_cmd._checks_summary(checks_map),
        "eligible": True,
        "at": seed_row["recorded_at"],
    }
    buffer_cmd._append(shelf_path, shelf_entry)
    doc["shelf_entry"] = shelf_entry
    doc["next_wake_at"] = None
    doc["state"] = "GATE_CONFIRM_SUBMIT"


def _step_submitting(engine, doc, session, cfg):
    alpha_id = doc.get("full_alpha_id")
    prior = doc.get("submit_attempt") or {}
    if prior.get("state") == "UNCERTAIN" or prior.get("post_attempted"):
        current = ml.fetch_alpha_metrics(session, alpha_id)
        if current.get("dateSubmitted") or str(current.get("status") or "").upper() in {"ACTIVE", "SUBMITTED"}:
            doc["submit_attempt"] = {
                **prior,
                "state": "CONFIRMED",
                "submitted": True,
                "date_submitted": current.get("dateSubmitted"),
            }
            doc["state"] = "SUBMITTED"
            doc["next_wake_at"] = None
            return
        recovery = ml.read_alpha_submission_status(session, alpha_id, max_wait_seconds=10)
        doc["submit_attempt"] = {**prior, "recovery": recovery}
        if recovery.get("state") == "rejected":
            doc["state"] = "REJECTED_AT_SUBMIT"
            doc["next_wake_at"] = None
            return
        retry_at = recovery.get("retry_at") or (time.time() + 15)
        _record_deferral(doc, cfg, retry_at=retry_at, reason="submission_uncertain")
        return

    def on_event(event):
        if event.get("event") == "submit_posting":
            doc["submit_attempt"] = {"state": "UNCERTAIN", "post_attempted": True, "at": time.time()}
            _persist_worker_progress(engine, doc, "AUTO_SUBMITTING", ("submit_attempt",), merge_on_cancel=False)

    try:
        res = ml.submit_alpha(session, alpha_id, confirm=True, max_wait_seconds=15, on_event=on_event)
    except (ml.CaptchaRequired, ml.RateLimited, ml.ServiceError, RequestException):
        raise
    except ml.BrainError as exc:
        if (doc.get("submit_attempt") or {}).get("post_attempted"):
            raise
        doc["state"] = "REJECTED_AT_SUBMIT"
        doc["last_error"] = {"code": getattr(exc, "code", type(exc).__name__), "message": str(exc)}
        doc["next_wake_at"] = None
        return

    doc["submit_attempt"] = {"state": res.get("submission") or "CONFIRMED", **res}
    if res.get("submitted") or res.get("submission") == "CONFIRMED":
        doc["state"] = "SUBMITTED"
        doc["next_wake_at"] = None
    elif res.get("submission") == "REJECTED":
        doc["state"] = "REJECTED_AT_SUBMIT"
        doc["next_wake_at"] = None
    else:
        doc["submit_attempt"]["state"] = "UNCERTAIN"
        _record_deferral(doc, cfg, retry_at=time.time() + 15, reason="submission_pending")


def _cancellation_targets(doc):
    terminal = alpha_registry.FINISHED_STATUSES | simulation_slots.TERMINAL_FAILURE_STATUSES | {"FAILED"}
    urls = set()
    unknown = False
    for rec in (doc.get("records") or []) + (doc.get("full_records") or []):
        if not rec.get("multi_parent_terminal") and rec.get("multi_parent_url"):
            urls.add(rec["multi_parent_url"])
        status = str(rec.get("simulation_status") or rec.get("status") or "").upper()
        if rec.get("slot_reservation_state") in {"cancelled", "already_gone"}:
            continue
        if status not in terminal:
            if rec.get("progress_url"):
                urls.add(rec["progress_url"])
            elif status == "UNKNOWN_SUBMISSION" and not rec.get("multi_parent_url"):
                unknown = True
    return sorted(urls), unknown


def _step_cancelling(engine, doc, session, cfg):
    urls, unknown = _cancellation_targets(doc)
    receipts = doc.setdefault("cancellation_receipts", {})
    reservations = [record for key in ("records", "full_records") for record in (doc.get(key) or [])
                    if record.get("slot_reservation") and (record.get("progress_url") in urls
                                                          or record.get("multi_parent_url") in urls)]
    if reservations:
        account = simulation_capabilities.account_provenance(session)
        for record in reservations:
            try:
                simulation_slots.validate_reservation_owner(record, account, research_dir=engine.root)
            except ValueError as exc:
                raise ml.AuthError(str(exc)) from exc
    for url in urls:
        previous = (receipts.get(url) or {}).get("state")
        if previous in {"cancelled", "already_gone", "terminal"}:
            continue
        if previous == "cancellation_pending":
            try:
                progress = ml.poll_simulation(session, url, deadline=time.monotonic() + 15)
            except ml.NotFound:
                result = {"state": "already_gone", "http_status": 404}
            else:
                if str(progress.get("status") or "").upper() not in simulation_slots.ORPHAN_RELEASE_STATUSES:
                    continue
                result = {"state": "terminal", "platform": progress}
        else:
            result = ml.cancel_simulation(session, url, deadline=time.monotonic() + 15)
        receipts[url] = result
        _persist_worker_progress(engine, doc, "AUTO_CANCELLING", ("cancellation_receipts",))
    if unknown or any((receipts.get(url) or {}).get("state") not in {"cancelled", "already_gone", "terminal"} for url in urls):
        reason = "simulation outcome unresolved" if unknown else "platform cancellation pending"
        doc["last_error"] = {"code": "CANCELLATION_PENDING", "message": reason}
        _record_deferral(doc, cfg, reason=reason)
        return
    doc["next_wake_at"] = None
    doc["last_error"] = None
    doc["state"] = "CANCELLED"


def _step_spc(engine, doc, session, cfg):
    inputs = doc["spc_inputs"]
    try:
        report = spc_experiment.experiment(doc["source_spec"], inputs["runs"], reference_path=inputs["reference"])
    except (OSError, ValueError) as exc:
        doc["last_error"] = {"code": "INVALID_INPUT", "message": str(exc)}
        report = {"ok": False}
    total = len(inputs["runs"])
    doc["experiment"] = report
    doc["state"] = "COMPLETED" if report.get("ok") else "FAILED"
    doc["execution"] = {"total": total, "not_submitted": 0, "accepted": total,
                        "finished": total, "terminal": 0 if report.get("ok") else 1}
    doc["next_wake_at"] = None


_HANDLERS = {
    "AUTO_SPC": _step_spc,
    "AUTO_PREFLIGHT": _step_preflight,
    "AUTO_QUICK_SIM": _step_quick_sim,
    "AUTO_HYDRATE_PNL": _step_hydrate_pnl,
    "AUTO_FULL_VALIDATE": _step_full_validate,
    "AUTO_BUFFER": _step_buffer,
    "AUTO_SUBMITTING": _step_submitting,
    "AUTO_CANCELLING": _step_cancelling,
}


@contextmanager
def _open_session(session_factory):
    session = session_factory()
    try:
        yield session
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()


class _StepSuperseded(Exception):
    """Stop a worker whose run was changed while it was outside the state lock."""


def _merge_cancel_receipts(engine, current, doc):
    if current.get("state") != "AUTO_CANCELLING":
        return
    for key in ("records", "full_records", "cancellation_receipts"):
        if key in doc:
            current[key] = doc[key]
    engine.save_run(current)


def _persist_worker_progress(engine, doc, expected_state, fields, *, merge_on_cancel=True):
    with cache_store.file_lock(engine.state_lock_path):
        current = engine.load_run_raw(doc["run_id"])
        if current.get("state") != expected_state:
            if merge_on_cancel:
                _merge_cancel_receipts(engine, current, doc)
            raise _StepSuperseded
        for key in fields:
            current[key] = doc[key]
        if fields:
            engine.save_run(current, durable=not merge_on_cancel)


def _defer_platform_error(doc, cfg, exc):
    retry_after = getattr(exc, "retry_after", None)
    if isinstance(exc, RequestException) and exc.response is not None:
        retry_after = ml._retry_after_seconds(exc.response, cap_error_wait=False)
    retry_at = None
    if isinstance(retry_after, (int, float)) and isfinite(retry_after) and retry_after > 0:
        retry_at = time.time() + retry_after
    _record_deferral(doc, cfg, retry_at=retry_at, reason=str(exc))
    doc["last_error"] = {"code": getattr(exc, "code", type(exc).__name__), "message": str(exc)}


def advance_run_once(engine, run_id, session_factory):
    """Run one autonomous step with the workspace state lock released during I/O."""
    claim = engine.daemon_dir / f"worker-{run_id}.lock"
    engine.run_path(run_id)
    try:
        with cache_store.file_lock(claim, timeout=0):
            return _advance_claimed_run(engine, run_id, session_factory)
    except cache_store.LockTimeout as exc:
        if str(claim) not in str(exc):
            raise
        return engine.load_run_raw(run_id)


def _advance_claimed_run(engine, run_id, session_factory):
    with cache_store.file_lock(engine.state_lock_path):
        doc = engine.load_run_raw(run_id)
        from_state = doc.get("state")
        handler = _HANDLERS.get(from_state)
        if handler is None:
            return doc
        cfg = engine.load_config()
    if (doc.get("last_error") or {}).get("code") == "AUTH_CAPTCHA":
        doc["last_error"] = None
    try:
        if from_state == "AUTO_CANCELLING":
            for key in ("records", "full_records"):
                for record in doc.get(key) or []:
                    if record.get("slot_reservation"):
                        try:
                            simulation_slots.restore_record(record, research_dir=engine.root)
                        except ValueError as exc:
                            raise ml.AuthError(str(exc)) from exc
        if from_state == "AUTO_SPC" or (from_state == "AUTO_CANCELLING" and not _cancellation_targets(doc)[0]):
            handler(engine, doc, None, cfg)
        else:
            with _open_session(session_factory) as session:
                handler(engine, doc, session, cfg)
    except _StepSuperseded:
        return engine.load_run_raw(run_id)
    except ml.CaptchaRequired as exc:
        doc["state"] = from_state
        doc["stalled"] = True
        doc["next_wake_at"] = None
        doc["last_error"] = {"code": exc.code, "message": str(exc)}
    except (ml.BrainError, RequestException) as exc:
        doc["state"] = from_state
        _defer_platform_error(doc, cfg, exc)
    with cache_store.file_lock(engine.state_lock_path):
        current = engine.load_run_raw(run_id)
        if current.get("state") != from_state:
            _merge_cancel_receipts(engine, current, doc)
            return current
        engine.save_run(doc, from_state=from_state, trigger="worker")
        return doc


def reconcile_orphan_slots(engine, session_factory, *, max_age_min=45.0):
    """Reconcile the authenticated account's reservations and harvest completed alphas."""
    if not simulation_slots.has_slot_reservations(engine.root):
        return {"released": [], "harvested": []}

    harvested = []
    registry_path = str(engine.root / "alpha-registry.jsonl")

    with _open_session(session_factory) as session:
        account = simulation_capabilities.account_provenance(session)
        if not account.get("fingerprint"):
            return {"released": [], "harvested": []}

        deadline = time.monotonic() + 15

        def poll(url):
            try:
                target, simulation_id = ml._quick_progress_target(url)
                progress, delay = ml._read_simulation_snapshot(session, target, simulation_id, deadline=deadline)
            except ml.NotFound:
                return {"status": "NOT_FOUND", "http_status": 404}
            except (ml.BrainError, RequestException):
                return None
            if delay or progress.get("id") not in (None, simulation_id):
                return None
            alpha_id = progress.get("alpha") or progress.get("alphaId")
            return {"status": progress.get("status"), "alpha": alpha_id if ml.valid_simulation_id(alpha_id) else None}

        res = simulation_slots.reconcile_orphans(account, poll, max_age_min=max_age_min,
                                                  deadline=deadline, research_dir=engine.root)
        released_all = res.get("released") or []
        if released_all:
            known_ids = {e.get("alpha_id") for e in alpha_registry.load_entries(registry_path)}
        for item in released_all:
            alpha_id = item.get("alpha")
            if not alpha_id or alpha_id in known_ids:
                continue
            try:
                payload = ml.fetch_alpha_metrics(session, alpha_id)
                code, _, _ = ml.regular_expression(payload)
                entry = alpha_registry.build_entry(
                    {"alpha_id": alpha_id, "expression": code or "",
                     "settings": payload.get("settings") or {},
                     "metrics": payload.get("is") or {},
                     "out_of_sample": ml.extract_split(payload) or {},
                     "status": payload.get("status") or "COMPLETE"},
                    source="reconcile_slots",
                )
                alpha_registry.record_entry(entry, path=registry_path)
                harvested.append(alpha_id)
                known_ids.add(alpha_id)
            except (ml.BrainError, RequestException):
                continue

    return {"released": released_all, "harvested": harvested}
