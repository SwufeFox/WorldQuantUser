"""Freeze, validate, check and shelve one candidate, retaining each step's evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import re

import machine_lib as ml
import requests
import research_paths

from . import alphas, buffer_cmd, triage
from .experiments import _selection_call
from .runtime import CliError, note, protect_output


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def cmd_promote(args, sink):
    """Compose existing commands; operational failures carry the completed evidence."""
    alpha_id = args.alpha_id
    alphas._candidate_role(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", alpha_id)
    freeze = Path(args.freeze or research_paths.resolve(
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
        freeze_args = parser.parse_args(["triage", "--ids", alpha_id, "--freeze", str(freeze)])
        freeze_args.retention_floor = args.retention_floor
        freeze_args.cost_bps = args.cost_bps
        frozen = _selection_call(triage.cmd_triage, freeze_args, sink)
        data["steps"][step] = {"status": "completed", "evidence": frozen}

        step = "validate"
        validate_args = parser.parse_args(["triage", "--selection", str(freeze), "--show-test"])
        validate_args.retention_floor = args.retention_floor
        validate_args.cost_bps = args.cost_bps
        validated = _selection_call(triage.cmd_triage, validate_args, sink)
        data["steps"][step] = {"status": "completed", "evidence": validated}
        row = next((row for row in validated.get("rows", [])
                    if row.get("alpha_id") == alpha_id), {})
        data["validate"] = {key: row.get(key) for key in data["validate"]}

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

        reasons = data["reasons"]
        summary = buffer_cmd._checks_summary(result.get("checks"))
        if (not checked.get("complete") or not result.get("complete")
                or not result.get("checks_resolved") or not result.get("checks")
                or summary["pending"]):
            reasons.append("platform_checks_not_resolved")
        if summary["failed"]:
            reasons.append("platform_checks_failed: " + ", ".join(summary["failed"]))
        test = data["validate"]["test_sharpe"]
        retention = data["validate"]["retention"]
        if not _finite(test):
            reasons.append("test_sharpe_missing_or_nonfinite")
        elif test <= 0:
            reasons.append("test_sharpe <= 0")
        if not _finite(retention):
            reasons.append("retention_missing_or_nonfinite")
        elif retention < args.retention_floor:
            reasons.append("retention < retention_floor (%s < %s)" % (retention, args.retention_floor))
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
        buffer_args.cost_bps = args.cost_bps
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
