"""Autonomous state-machine step workers and orphan slot reconciliation."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

from requests.exceptions import RequestException

import alpha_registry
import cache_store
import machine_lib as ml
import simulation_capabilities
import simulation_multi
import simulation_slots

from . import batch_admission, brief, buffer_cmd, experiments, ledger_analysis, pp_budget, selection, state_machine, validation


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
    try:
        account = simulation_capabilities.account_provenance(session)
    except (ml.BrainError, RequestException) as exc:
        _record_deferral(doc, cfg, retry_at=getattr(exc, "retry_after", None) and time.time() + exc.retry_after, reason=str(exc))
        doc["last_error"] = {"code": getattr(exc, "code", type(exc).__name__), "message": str(exc)}
        return

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
    settings = arms[0].get("settings") or {}
    pairs = ml.normalize_alphas([arm["expression"] for arm in arms], settings)
    try:
        from types import SimpleNamespace
        validation._preflight_lint(session, pairs, settings, SimpleNamespace(no_cache=False), proof_cache={})
    except ml.RateLimited as exc:
        wait = getattr(exc, "retry_after", None) or 20
        _record_deferral(doc, cfg, retry_at=time.time() + wait, reason="RATE_LIMITED")
        doc["last_error"] = {"code": "RATE_LIMITED", "message": str(exc)}
        return
    except (ml.BrainError, Exception) as exc:
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


def _step_quick_sim(engine, doc, session, cfg):
    arms = doc.get("arms") or []
    records = doc.get("records") or []
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

    settings = ml.resolve_settings(arms[0].get("settings") or {})
    expressions = [arm["expression"] for arm in arms]
    labels = [arm["label"] for arm in arms]
    checkpoint_path = engine.run_path(doc["run_id"])
    multi_size = min(int(cfg.get("multi_size", 10) or 1), len(arms)) if not ml.is_python_language(settings.get("language")) else 1

    def on_event(kind, payload):
        idx = payload.get("index") if isinstance(payload, dict) else None
        if isinstance(idx, int) and 0 <= idx < len(records):
            if kind in {"submitted", "completed", "failed", "deferred", "pending"}:
                if kind == "submitted" or payload.get("progress_url") or payload.get("status"):
                    records[idx]["submission_attempted"] = True
                for key in ("status", "alpha_id", "progress_url", "next_poll_at", "metrics",
                            "out_of_sample", "checks", "error", "metrics_pending", "submission_state"):
                    if key in payload:
                        records[idx][key] = payload[key]
                cache_store.write_json_atomic(checkpoint_path, doc, indent=2)

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
            fetch_checks=False,
            initial_records=records,
            checkpoint=str(checkpoint_path),
            on_event=on_event,
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
        if not train and isinstance(rec.get("metrics"), dict) and status in alpha_registry.FINISHED_STATUSES:
            train = rec["metrics"]
        if status in alpha_registry.FINISHED_STATUSES and rec.get("alpha_id"):
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
        entry = entries.get(alpha_id) or {}
        split = entry.get("out_of_sample") or {}
        train_start = (split.get("train") or {}).get("startDate")
        test_start = (split.get("test") or {}).get("startDate")
        report = {}
        try:
            ml.fetch_daily_pnl(session, alpha_id, report=report)
        except ml.RateLimited as exc:
            _record_deferral(doc, cfg, retry_at=time.time() + (getattr(exc, "retry_after", None) or 20), reason="RATE_LIMITED")
            return
        except (ml.BrainError, RequestException) as exc:
            by_label[item["label"]] = {"alpha_id": alpha_id, "yearly_train": {"state": "unavailable", "reason": str(exc)}, "gate_pass": True}
            continue
        if train_start and test_start:
            fetched_items.append((item, report, {"alpha_id": alpha_id, "train_start": train_start, "test_start": test_start}))
        else:
            by_label[item["label"]] = {
                "alpha_id": alpha_id,
                "yearly_train": {"state": "unavailable", "reason": "train_window_not_verified"},
                "pnl_fetch": report,
                "gate_pass": True,
            }

    if fetched_items:
        yearly_results = pnl_snapshot.yearly_train_batch([row for _, _, row in fetched_items], pnl_dir)
        for (item, report, _), yearly in zip(fetched_items, yearly_results):
            by_label[item["label"]] = {
                "alpha_id": item["alpha_id"],
                "yearly_train": yearly,
                "pnl_fetch": report,
                "gate_pass": yearly.get("state") != "fail",
            }

    doc["hydration"] = {"by_label": by_label}
    doc["unchanged_defers"] = 0
    doc["next_wake_at"] = None
    doc["state"] = "GATE_APPROVE_FULL"


def _series_map(series_entry, field_name):
    records = (series_entry or {}).get("records") or []
    props = ((series_entry or {}).get("schema") or {}).get("properties") or []
    date_idx, val_idx = 0, 1
    if isinstance(props, list) and props:
        cols = {prop.get("name"): i for i, prop in enumerate(props) if isinstance(prop, dict) and prop.get("name")}
        if "date" in cols and field_name in cols:
            date_idx, val_idx = cols["date"], cols[field_name]
    out = {}
    for row in records:
        if isinstance(row, dict) and "date" in row and field_name in row:
            out[row["date"]] = row[field_name]
        elif isinstance(row, (list, tuple)) and len(row) > max(date_idx, val_idx):
            out[row[date_idx]] = row[val_idx]
    return out


def _compute_cost_audit(payload, recordsets):
    metrics = payload.get("is") or payload.get("metrics") or {}
    book_size = metrics.get("bookSize")
    series = (recordsets or {}).get("series") or {}
    pnl_map = _series_map(series.get("daily-pnl"), "pnl")
    to_map = _series_map(series.get("turnover"), "turnover")
    if type(book_size) not in (int, float) or book_size <= 0 or not pnl_map or not to_map:
        return None
    days = sorted(pnl_map.keys() & to_map.keys())
    if len(days) < 20:
        return None
    scenarios = []
    from math import sqrt
    from statistics import mean, stdev
    daily_pnl = [pnl_map[d] for d in days]
    daily_traded = [to_map[d] * book_size for d in days]
    traded = sum(daily_traded)
    if traded <= 0 or stdev(daily_pnl) == 0:
        return None
    for bps in (5, 10, 20):
        net = [p - bps / 10000.0 * amt for p, amt in zip(daily_pnl, daily_traded)]
        scenarios.append({
            "cost_bps": bps,
            "net_sharpe": round(mean(net) / stdev(net) * sqrt(252), 3),
            "net_margin_bps": round(sum(net) / traded * 10000.0, 2),
        })
    return {"joint_sessions": len(days), "scenarios": scenarios}


def _step_full_validate(engine, doc, session, cfg):
    # FULL confirmation lane deliberately ignores soft quota_floor, pausing only if platform quota is blocked.
    account = simulation_capabilities.account_provenance(session)
    quota = simulation_slots.quota_snapshot(account)
    if quota.get("blocked"):
        doc["next_wake_at"] = quota.get("reset_at") or (time.time() + 120)
        return

    approved = doc.get("approved_arms") or []
    arms_by_label = {arm["label"]: arm for arm in (doc.get("arms") or []) if "label" in arm}
    full_arms = [brief.full_arm(arms_by_label[item["label"]]) for item in approved if item["label"] in arms_by_label]
    if not full_arms:
        doc["state"] = "FAILED"
        doc["last_error"] = {"code": "INVALID_INPUT", "message": "no approved arms matched spec"}
        return

    full_records = doc.get("full_records") or [{
        "arm_index": idx,
        "label": arm["label"],
        "expression": arm["expression"],
        "settings": arm["settings"],
        "status": "QUEUED",
        "submission_attempted": False,
    } for idx, arm in enumerate(full_arms)]

    need_sim = any(
        str(rec.get("status") or "").upper() not in (alpha_registry.FINISHED_STATUSES | simulation_slots.TERMINAL_FAILURE_STATUSES | {"FAILED"})
        for rec in full_records
    )
    if need_sim:
        settings = ml.resolve_settings(full_arms[0]["settings"])
        result_records = ml.simulate_alphas(
            session,
            [arm["expression"] for arm in full_arms],
            settings=settings,
            labels=[arm["label"] for arm in full_arms],
            concurrency=int(cfg.get("concurrency", 8)),
            multi_size=1,
            deadline=time.monotonic() + 25.0,
            fetch_metrics=True,
            fetch_checks=False,
            initial_records=full_records,
        )
        full_records = [{**full_records[i], **rec} for i, rec in enumerate(result_records)]
        doc["full_records"] = full_records
        if any(str(rec.get("status") or "").upper() not in (alpha_registry.FINISHED_STATUSES | simulation_slots.TERMINAL_FAILURE_STATUSES | {"FAILED"})
               for rec in full_records):
            _record_deferral(doc, cfg, retry_at=time.time() + 15, reason="full_pending")
            return

    doc["full_records"] = full_records
    candidate_rec = next((rec for rec in full_records if rec.get("alpha_id")), None)
    if candidate_rec is None:
        doc["state"] = "FAILED"
        doc["last_error"] = {"code": "SIMULATION_FAILED", "message": "FULL confirmation simulation failed"}
        return

    full_alpha_id = candidate_rec["alpha_id"]
    payload = ml.fetch_alpha_metrics(session, full_alpha_id)
    checks_report = ml.fetch_alpha_checks(session, full_alpha_id, max_wait_seconds=10)
    self_corr = ml.fetch_self_correlation(session, full_alpha_id)
    sc_verdict = ml.self_correlation_verdict(self_corr)
    recordsets = {}
    try:
        recordsets = ml.fetch_alpha_recordsets(session, full_alpha_id, ("daily-pnl", "turnover"), max_wait_seconds=10)
    except (ml.BrainError, RequestException):
        recordsets = {}

    registry_path = str(engine.root / "alpha-registry.jsonl")
    entry = alpha_registry.build_entry(
        {"alpha_id": full_alpha_id, "expression": full_arms[0]["expression"],
         "settings": full_arms[0]["settings"], "metrics": payload.get("is") or {},
         "checks": checks_report.get("checks") or {}, "out_of_sample": ml.extract_split(payload) or {},
         "status": payload.get("status") or "COMPLETE"},
        idea=full_arms[0].get("idea") or doc.get("idea"),
        mechanism=full_arms[0].get("mechanism") or doc.get("mechanism") or full_arms[0]["label"],
        source="simulate",
        family=full_arms[0].get("family"),
        batch=doc["run_id"],
    )
    alpha_registry.record_entry(entry, path=registry_path)

    split = ml.extract_split(payload) or {}
    test_metrics = split.get("test") or {}
    test_sharpe = test_metrics.get("sharpe")
    checks = checks_report.get("checks") or {}
    failed_checks = sorted(name for name, item in checks.items() if isinstance(item, dict) and item.get("result") == "FAIL")

    doc["full_alpha_id"] = full_alpha_id
    doc["full_checks"] = checks_report
    doc["cost_audit"] = _compute_cost_audit(payload, recordsets)
    doc["validation"] = {
        "full_alpha_id": full_alpha_id,
        "test_sharpe": test_sharpe,
        "self_correlation": sc_verdict,
        "failed_checks": failed_checks,
        "checks_complete": bool(checks_report.get("complete")),
    }

    if doc.get("track") == "POWER_POOL":
        budget = pp_budget.alpha_budget(full_arms[0]["expression"], full_arms[0]["settings"], checks=checks)
        doc["validation"]["pp_budget"] = budget
        if isinstance(budget, dict):
            gates = budget.get("gates") or {}
            inner = budget.get("budget") or {}
            if gates.get("blocking") or inner.get("ok") is False:
                doc["state"] = "REJECTED_AT_VALIDATION"
                return

    if failed_checks or bool(sc_verdict.get("blocked")):
        doc["state"] = "REJECTED_AT_VALIDATION"
        return
    if type(test_sharpe) in (int, float) and test_sharpe <= 0:
        doc["validation"]["failed_reason"] = "nonpositive_test_sharpe"
        doc["state"] = "REJECTED_AT_VALIDATION"
        return
    if not checks_report.get("complete"):
        _record_deferral(doc, cfg, retry_at=time.time() + 15, reason="checks_pending")
        return

    doc["unchanged_defers"] = 0
    doc["next_wake_at"] = None
    doc["state"] = "GATE_DESCRIBE"


def _step_buffer(engine, doc, session, cfg):
    alpha_id = doc.get("full_alpha_id")
    meta = doc.get("candidate_metadata") or {}
    if not alpha_id or not meta:
        doc["state"] = "FAILED"
        doc["last_error"] = {"code": "INVALID_INPUT", "message": "missing full_alpha_id or candidate_metadata"}
        return

    ml.set_alpha_metadata(
        session,
        alpha_id,
        name=meta["name"],
        category=meta["category"],
        tags=meta["tags"],
        description=meta["description"],
    )
    payload = ml.fetch_alpha_metrics(session, alpha_id)
    _, live_desc, _ = ml.regular_expression(payload)
    if (live_desc or "").strip() != meta["description"].strip():
        doc["state"] = "GATE_DESCRIBE"
        doc["last_error"] = {"code": "INVALID_INPUT", "message": "remote description did not match submitted description"}
        return

    seed_path = str(engine.root / buffer_cmd.SEEDS_FILENAME)
    shelf_path = str(engine.root / buffer_cmd.BUFFER_FILENAME)
    split = ml.extract_split(payload) or {}
    seed_row = {
        "alpha_id": alpha_id,
        "name": meta["name"],
        "category": meta["category"],
        "tags": meta["tags"],
        "description": meta["description"],
        "expression": (payload.get("regular") or {}).get("code"),
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
        "self_correlation": ((doc.get("full_checks") or {}).get("self_correlation") or {}).get("value"),
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
    if prior.get("state") == "UNCERTAIN":
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
            engine.save_run(doc)

    try:
        res = ml.submit_alpha(session, alpha_id, confirm=True, max_wait_seconds=15, on_event=on_event)
    except (ml.BrainError, RequestException) as exc:
        if (doc.get("submit_attempt") or {}).get("post_attempted"):
            _record_deferral(doc, cfg, retry_at=time.time() + 15, reason=str(exc))
        else:
            doc["state"] = "REJECTED_AT_SUBMIT"
            doc["last_error"] = {"code": getattr(exc, "code", type(exc).__name__), "message": str(exc)}
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


def _step_cancelling(engine, doc, session, cfg):
    for rec in (doc.get("records") or []) + (doc.get("full_records") or []):
        url = rec.get("progress_url") or rec.get("multi_parent_url")
        if url:
            try:
                ml.cancel_simulation(session, url)
            except (ml.BrainError, RequestException):
                pass
    doc["next_wake_at"] = None
    doc["state"] = "CANCELLED"


_HANDLERS = {
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


def advance_run_once(engine, run_id, session_factory):
    """Execute one non-blocking state-machine step for *run_id* and persist any transition."""
    with cache_store.file_lock(engine.state_lock_path):
        doc = engine.load_run_raw(run_id)
        from_state = doc.get("state")
        handler = _HANDLERS.get(from_state)
        if handler is None:
            return doc
        cfg = engine.load_config()
        with _open_session(session_factory) as session:
            handler(engine, doc, session, cfg)
        engine.save_run(doc, from_state=from_state, trigger="worker")
        return doc


def reconcile_orphan_slots(engine, session_factory, *, max_age_min=45.0):
    """Sweep orphan slot reservations and harvest any completed alphas into the local ledger."""
    slot_dir = engine.root / "simulation-slots"
    if not slot_dir.is_dir():
        return {"released": [], "harvested": []}
    journals = sorted(slot_dir.glob("*.json"))
    if not journals:
        return {"released": [], "harvested": []}
    raw, err = cache_store.load_json(journals[0])
    if err or not isinstance(raw, dict) or not raw.get("account_fingerprint"):
        return {"released": [], "harvested": []}
    account = {"fingerprint": raw["account_fingerprint"]}

    released_all = []
    harvested = []
    registry_path = str(engine.root / "alpha-registry.jsonl")
    known_ids = {e.get("alpha_id") for e in alpha_registry.load_entries(registry_path) if e.get("alpha_id")} if Path(registry_path).exists() else set()

    with _open_session(session_factory) as session:
        def poll(url):
            try:
                progress = ml.poll_simulation(session, url, deadline=time.monotonic() + 15)
            except ml.NotFound:
                return "NOT_FOUND"
            except (ml.BrainError, RequestException):
                return None
            return {"status": str(progress.get("status") or "").upper(), "alpha": progress.get("alpha")}

        res = simulation_slots.reconcile_orphans(account, poll, max_age_min=max_age_min)
        released_all = res.get("released") or []
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
