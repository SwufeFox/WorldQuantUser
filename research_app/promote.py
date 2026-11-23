"""Freeze, validate, check and shelve one candidate, retaining each step's evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import machine_lib as ml
import requests
import research_paths

from . import alphas, buffer_cmd, selection, triage
from .experiments import _selection_call
from .runtime import CliError, note, protect_output, with_session


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validation_reasons(row, retention_floor):
    reasons = []
    test, retention = row.get("test_sharpe"), row.get("retention")
    if not _finite(test):
        reasons.append("test_sharpe_missing_or_nonfinite")
    elif test <= 0:
        reasons.append("test_sharpe <= 0")
    if not _finite(retention):
        reasons.append("retention_missing_or_nonfinite")
    elif retention < retention_floor:
        reasons.append("retention < retention_floor (%s < %s)" % (retention, retention_floor))
    return reasons


def _read_production_distribution(path, alpha_id, limit):
    path = Path(path).expanduser().resolve()
    try:
        archive = json.loads(path.read_text(encoding="utf-8"))
        source = archive["source"]
        requested_ids = archive["requested_ids"]
        if (source != "GET /alphas/{id}/correlations/prod"
                or not isinstance(requested_ids, list) or alpha_id not in requested_ids):
            raise ValueError("archive does not identify this alpha and the production endpoint")
        captures = archive["results"][alpha_id]
        if not isinstance(captures, list) or not captures:
            raise ValueError("archive has no captures for this alpha")
        dated = [(datetime.fromisoformat(capture["captured_at"].replace("Z", "+00:00")), capture)
                 for capture in captures]
        if any(stamp.utcoffset() is None for stamp, _ in dated):
            raise ValueError("captures must have time-zone-aware timestamps")
        stamp, capture = max(dated, key=lambda item: item[0])
        status, retry_after = capture["http_status"], capture.get("retry_after")
        diagnostic = {
            "source": source.replace("{id}", alpha_id), "path": str(path),
            "captured_at": stamp.astimezone(timezone.utc).isoformat(),
            "http_status": status, "retry_after": retry_after,
            "recorded_response_sha256": capture.get("response_sha256"),
            "state": "unavailable", "max": None, "bins": [],
            "count_in_bins_at_or_above_formal_limit": None,
        }
        try:
            retry_delay = float(str(retry_after).strip() or 0) if retry_after is not None else 0
        except (TypeError, ValueError, OverflowError):
            retry_delay = None
        response = capture.get("raw_json_response")
        if status != 200 or not _finite(retry_delay) or retry_delay > 0 or not response:
            return diagnostic
        schema = response["schema"]
        columns = [prop["name"] for prop in schema["properties"]]
        if (schema["name"] != "prodCorrelation" or columns != ["min", "max", "alphas"]
                or not isinstance(response["records"], list)):
            raise ValueError("capture has an unsupported production histogram")
        observed_max = response.get("max")
        if observed_max is not None and not alphas._correlation_number(observed_max):
            raise ValueError("capture has an invalid maximum correlation")
        bins = []
        for lower, upper, count in response["records"]:
            if (not alphas._correlation_number(lower) or not alphas._correlation_number(upper)
                    or lower >= upper or type(count) is not int or count < 0
                    or (bins and lower < bins[-1]["upper"])):
                raise ValueError("capture has an invalid or overlapping correlation bin")
            bins.append({"lower": lower, "upper": upper, "count": count})
        if observed_max is not None or bins:
            diagnostic.update(state="available", max=observed_max, bins=bins)
        if bins and alphas._correlation_number(limit):
            diagnostic["count_in_bins_at_or_above_formal_limit"] = sum(
                row["count"] for row in bins if row["lower"] >= limit)
        return diagnostic
    except (OSError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise CliError("INVALID_INPUT", f"cannot review production distribution: {exc}") from exc


def _review_receipt(args):
    """Review historical Regular promotion evidence without contacting BRAIN."""
    if (not args.selection or args.dry_run or any(getattr(args, key) is not None
            for key in ("freeze", "buffer", "note", "retention_floor", "cost_bps",
                        "max_self_correlation"))):
        raise CliError("INVALID_INPUT", "--review-receipt requires --selection and cannot use live promotion options")
    protect_output(args)
    manifest = _selection_call(selection.load, args.selection, source="triage")
    retention_floor = manifest["criteria"].get("retention_floor", selection.DEFAULT_PROMOTION_RETENTION)
    if not _finite(retention_floor):
        raise CliError("INVALID_INPUT", "frozen retention floor must be finite")
    path = Path(args.review_receipt).expanduser().resolve()
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        data = receipt["data"]
        steps = data["steps"]
        frozen = steps["freeze"].get("evidence") or {}
        validated = steps["validate"].get("evidence") or {}
        checks = data["checks"]
        rows = validated.get("rows", [])
        results = checks.get("results", [])
        if (receipt.get("command") != "promote" or data.get("alpha_id") != args.alpha_id
                or [row["alpha_id"] for row in manifest["rows"]] != [args.alpha_id]
                or (frozen and frozen.get("selection_id") != manifest["selection_id"])
                or (validated and (not frozen
                    or validated.get("selection_id") != manifest["selection_id"]
                    or validated.get("frozen_window") != manifest["window"]
                    or [selection.row_identity(row) for row in rows] != manifest["identities"]))
                or (results and (not validated or [row["alpha_id"] for row in results] != [args.alpha_id]))):
            raise ValueError("receipt, candidate and frozen selection do not match")
        row = rows[0] if rows else {}
        checked = results[0] if results else {}
        observed = checked["checks"] if checked.get("payload_received") is True and not checked.get("error") else {}
        verdicts = {name: check.get("result") for name, check in observed.items()}
        saved_pc = (checked.get("production_correlation_evidence") or {}).get("formal") or {}
        captured_at = saved_pc.get("captured_at")
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
        raise CliError("INVALID_INPUT", f"cannot review promotion receipt: {exc}") from exc

    reasons = _validation_reasons(row, retention_floor)
    source_fields = ("source_expression", "source_settings")
    source_changed = [key for key in source_fields if row and not row.get("missing")
                      and row.get(key) != manifest["rows"][0].get(key)]
    missing = [reason for reason in reasons if reason.endswith("_missing_or_nonfinite")]
    failures = [reason for reason in reasons if reason not in missing]
    failed_checks = sorted(name for name, result in verdicts.items() if result == "FAIL")
    unresolved = sorted(name for name, result in verdicts.items()
                        if result not in ("PASS", "FAIL", "WARNING"))
    checks_complete = bool(verdicts) and not unresolved and checked.get("complete") is True
    production = observed.get("PROD_CORRELATION") or {}
    if not checks_complete:
        missing.append("platform_checks_not_resolved")
    if production.get("result") not in ("PASS", "FAIL"):
        missing.append("production_correlation_verdict_missing")
    error = receipt.get("error") or checked.get("error")
    if error:
        missing.append("operational_error")
    if source_changed:
        missing.append("frozen_source_changed")
    if not row or row.get("missing"):
        missing.append("frozen_validation_missing")
    settings = row.get("source_settings") or {}
    if ml.is_quick_simulation({"settings": settings}):
        missing.append("quick_simulation")
    outcome = ("deferred" if source_changed or error or "quick_simulation" in missing
               else "blocked" if failures or failed_checks
               else "deferred" if missing else "passes_review")
    pc_evidence = alphas.production_correlation_evidence(
        args.alpha_id, observed, basis="saved_receipt", captured_at=captured_at)
    distribution_path = getattr(args, "pc_distribution", None)
    if distribution_path:
        pc_evidence["distribution"] = _read_production_distribution(
            distribution_path, args.alpha_id, production.get("limit"))
    return {
        "alpha_id": args.alpha_id, "selection_id": manifest["selection_id"],
        "receipt_path": str(path), "basis": "saved_receipt", "live_verified": False,
        "recorded_outcome": data.get("outcome"), "review_outcome": outcome,
        "simulation_mode": settings.get("simulationMode"),
        "validation": {key: row.get(key) for key in ("train_sharpe", "test_sharpe", "retention", "costs")},
        "production_correlation": production or None, "failed_checks": failed_checks,
        "production_correlation_evidence": pc_evidence,
        "unresolved_checks": unresolved, "validation_failures": failures,
        "missing_evidence": missing, "source_changes": source_changed, "recorded_error": error,
    }


def cmd_promote(args, sink):
    """Compose existing commands; operational failures carry the completed evidence."""
    if getattr(args, "review_receipt", None):
        return _review_receipt(args)
    if getattr(args, "pc_distribution", None):
        raise CliError("INVALID_INPUT", "--pc-distribution requires --review-receipt")
    alpha_id = args.alpha_id
    alphas._candidate_role(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", alpha_id)
    saved_choice = getattr(args, "selection", None)
    freeze = Path(saved_choice or args.freeze or research_paths.resolve(
        "promote-freeze-%s-%s.json" % (safe_id, stamp))).expanduser().resolve()
    shelf = buffer_cmd.buffer_path(args).resolve()
    protect_output(args, freeze, shelf)
    if freeze == shelf:
        raise CliError("INVALID_INPUT", "--freeze and --buffer must use distinct paths")

    data = {"alpha_id": alpha_id, "freeze_path": str(freeze),
            "validate": dict.fromkeys(("train_sharpe", "test_sharpe", "retention",
                                       "rho_book", "max_member_rho")),
            "checks": {}, "buffered": False, "outcome": "rejected", "reasons": [],
            "steps": {name: {"status": "pending"}
                      for name in ("freeze", "validate", "checks", "buffer")}}
    step = "freeze"
    try:
        # Use the command parser's defaults, so promotion follows the standalone
        # commands when their optional settings change.
        from .parser import build_parser
        parser = build_parser()
        retention_floor, cost_bps = args.retention_floor, args.cost_bps
        max_corr = getattr(args, "max_self_correlation", None)
        if max_corr is not None and (not _finite(max_corr) or not 0 <= max_corr <= 1):
            raise CliError("INVALID_INPUT", "--max-self-correlation must be between 0 and 1")
        if saved_choice:
            # Campaign already selected the candidate using training evidence.
            # Reuse its manifest to keep validation bound to that decision.
            manifest = _selection_call(selection.load, freeze, source="triage")
            if [row.get("alpha_id") for row in manifest["rows"]] != [alpha_id]:
                raise CliError("INVALID_INPUT", "promotion selection must contain exactly this candidate")
            frozen_shelf = manifest["inputs"].get("shelf")
            if frozen_shelf and Path(frozen_shelf["path"]).resolve() != shelf:
                raise CliError("INVALID_INPUT", "--buffer conflicts with the frozen shelf")
            saved_criteria = manifest["criteria"]
            for key, provided in (("retention_floor", retention_floor), ("cost_bps", cost_bps)):
                if provided is not None and provided != saved_criteria.get(key):
                    raise CliError("INVALID_INPUT", f"{key} conflicts with the frozen selection")
            retention_floor = saved_criteria.get("retention_floor", selection.DEFAULT_PROMOTION_RETENTION)
            cost_bps = saved_criteria.get("cost_bps")
            frozen = {"selection_id": manifest["selection_id"], "rows": manifest["rows"],
                      "criteria": saved_criteria, "window": manifest["window"]}
        else:
            if retention_floor is None:
                retention_floor = selection.DEFAULT_PROMOTION_RETENTION
        if not _finite(retention_floor):
            raise CliError("INVALID_INPUT", "retention floor must be finite")
        payload = with_session(args, lambda session: ml.fetch_alpha_metrics(session, alpha_id))
        alphas._reject_quick_simulation(args, payload)
        if not saved_choice:
            freeze_args = parser.parse_args(["triage", "--ids", alpha_id, "--freeze", str(freeze)])
            freeze_args.retention_floor = retention_floor
            freeze_args.cost_bps = cost_bps
            frozen = _selection_call(triage.cmd_triage, freeze_args, sink)
        data["steps"][step] = {"status": "completed", "evidence": frozen}

        step = "validate"
        validate_args = parser.parse_args(["triage", "--selection", str(freeze), "--show-test"])
        if not saved_choice:
            validate_args.retention_floor = retention_floor
            validate_args.cost_bps = cost_bps
        validated = _selection_call(triage.cmd_triage, validate_args, sink)
        data["steps"][step] = {"status": "completed", "evidence": validated}
        row = next((row for row in validated.get("rows", [])
                    if row.get("alpha_id") == alpha_id), {})
        data["validate"] = {key: row.get(key) for key in data["validate"]}

        reasons = data["reasons"]
        reasons.extend(_validation_reasons(data["validate"], retention_floor))
        if reasons:
            data["steps"]["checks"] = {"status": "skipped"}
            data["steps"]["buffer"] = {"status": "skipped"}
            return data

        step = "checks"
        check_args = parser.parse_args(["alpha", "check", alpha_id])
        checked = alphas.cmd_alpha_check(check_args, sink)
        data["checks"] = checked
        data["steps"][step] = {"status": "completed"}
        result = next((row for row in checked.get("results", [])
                       if row.get("alpha_id") == alpha_id), {})
        if result.get("error"):
            error = result["error"]
            raise CliError(error["code"], error["message"], error.get("hint"))

        summary = buffer_cmd._checks_summary(result.get("checks"))
        if (not checked.get("complete") or not result.get("complete")
                or not result.get("checks_resolved") or not result.get("checks")
                or summary["pending"]):
            reasons.append("platform_checks_not_resolved")
        if summary["failed"]:
            reasons.append("platform_checks_failed: " + ", ".join(summary["failed"]))
        if max_corr is not None:
            value = (result.get("self_correlation") or {}).get("value")
            if not _finite(value):
                reasons.append("self_correlation_missing_or_nonfinite")
            elif value >= max_corr:
                reasons.append("self_correlation_at_or_above_limit")
        if reasons:
            data["steps"]["buffer"] = {"status": "skipped"}
            return data
        if args.dry_run:
            data["outcome"] = "would_buffer"
            data["steps"]["buffer"] = {"status": "would_buffer"}
            return data

        step = "buffer"
        buffer_args = parser.parse_args(["buffer", "add", alpha_id, "--selection", str(freeze)])
        buffer_args.buffer, buffer_args.note = args.buffer, args.note
        buffer_args.cost_bps = cost_bps
        buffered = buffer_cmd.cmd_buffer_add(buffer_args, sink, checked=result)
        data["steps"][step] = {"status": "completed", "evidence": buffered}
        data.update(buffered=True, outcome="buffered")
        return data
    except (Exception, KeyboardInterrupt) as exc:
        data["steps"][step]["status"] = "failed"
        if step != "buffer":
            data["steps"]["buffer"]["status"] = "skipped"
        if isinstance(exc, CliError):
            error = exc
        elif isinstance(exc, (ml.BrainError, requests.RequestException)):
            body = alphas._check_error(exc)
            error = CliError(body["code"], body["message"], body.get("hint"))
        elif isinstance(exc, KeyboardInterrupt):
            error = CliError("INTERRUPTED", "cancelled")
        else:
            error = CliError("SERVICE_ERROR", "%s: %s" % (type(exc).__name__, exc))
        if error.code.startswith("AUTH") and ml.invalidate_session():
            note("session cache cleared after %s" % error.code)
        data["reasons"].append("%s: %s: %s" % (step, error.code, error.message))
        data["failed_step"] = step
        error.data = data
        if error is exc:
            raise
        raise error from exc
