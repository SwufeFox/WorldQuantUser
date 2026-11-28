"""BRAIN Labs diagnostic tickets and scoped blind-simulation guardrails."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import textwrap
import uuid

import alpha_registry
from check_evidence import normalize_checks
import fastexpr_ast
import machine_lib as ml
import research_paths
import workspace_db

from . import validation
from .runtime import CliError

TRIGGER_RULES = {
    "anomaly": "同一数据集或字段连续多次逻辑合理但模拟结果异常（高换手、收益极端集中、子样本骤降）。",
    "distribution": "不确定字段的真实更新频率、覆盖率、长尾极值或行业缺失特征。",
    "interaction": "需要验证多字段共线性、去冗余或探索非线性/条件分段交互结构。",
}
_TRIGGER_ALIASES = {
    "repeated_anomaly": "anomaly",
    "field_distribution": "distribution",
    "field_distribution_uncertainty": "distribution",
    "multicollinearity": "interaction",
    "multicollinearity_or_interaction": "interaction",
}
REQUEST_STATES = frozenset({"GATE_PREFLIGHT_BLOCKED", "GATE_TRIAGE", "GATE_APPROVE_FULL"})
_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_MAX_FIELDS = 32
_MAX_ITEMS = 32
_MAX_LIST = 100
_MAX_METRICS_BYTES = 65536


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text(value, name, *, required=True, max_length=2000):
    if value is None and not required:
        return None
    if isinstance(value, str) and value.strip() and len(value.strip()) <= max_length:
        return value.strip()
    raise CliError("INVALID_INPUT", f"{name} must be non-empty text of at most {max_length} characters",
                   f"provide concise text for {name}")


def _token(value, name, *, required=True):
    if value is None and not required:
        return None
    if isinstance(value, str) and _TOKEN.fullmatch(value.strip()):
        return value.strip()
    raise CliError("INVALID_INPUT", f"invalid {name}", f"provide a single identifier for {name}")


def _items(value, name, *, limit=_MAX_ITEMS):
    if not isinstance(value, list) or not 1 <= len(value) <= limit:
        raise CliError("INVALID_INPUT", f"{name} must contain 1–{limit} entries",
                       f"supply a list of entries for {name}")
    return value


def _fields(value):
    if isinstance(value, str):
        value = re.split(r"[,\n]", value)
    return list(dict.fromkeys(_token(item, "field") for item in _items(value, "fields", limit=_MAX_FIELDS)))


def _branch(value):
    if isinstance(value, str):
        condition, separator, action = value.partition("->")
        value = {"condition": condition, "action": action if separator else condition}
    if not isinstance(value, dict):
        raise CliError("INVALID_INPUT", "invalid Labs branch", "use `condition -> action` or a branch object")
    return {"condition": _text(value.get("condition", value.get("if")), "branch condition"),
            "action": _text(value.get("action", value.get("then")), "branch action")}


def build_labs_notebook_snippet(ticket):
    """Generate a summary-only Python snippet for In-Sample BRAIN Labs matrices."""
    return f"FIELDS = {ticket['fields']!r}\n" + textwrap.dedent('''\
        import json
        from itertools import combinations

        import numpy as np
        import pandas as pd

        # Supply load_is_matrix(field) in BRAIN Labs. It must return an In-Sample
        # date-by-stock DataFrame, with absent or out-of-universe values missing.
        # Share aggregate findings only. Keep the input matrices inside Labs.
        def rounded(value):
            return round(float(value), 6) if np.isfinite(value) else None

        summary = {"fields": {}, "pairwise_rank_corr": {}}
        ranks = {}
        for name in FIELDS:
            df = load_is_matrix(name).sort_index().apply(pd.to_numeric, errors="coerce")
            df = df.replace([np.inf, -np.inf], np.nan)
            values = df.to_numpy(dtype=float)
            valid = np.isfinite(values)
            finite_values = values[valid]
            coverage = valid.mean(axis=1) if values.size else np.array([])
            pairs = valid[1:] & valid[:-1]
            changed = np.abs(values[1:][pairs] - values[:-1][pairs]) > 1e-12
            ranks[name] = df.rank(axis=1)
            lag = ranks[name].corrwith(ranks[name].shift(1), axis=1).dropna()
            summary["fields"][name] = {
                "mean_coverage": rounded(coverage.mean()) if coverage.size else None,
                "min_coverage": rounded(coverage.min()) if coverage.size else None,
                "daily_update_rate": rounded(changed.mean()) if changed.size else None,
                "quantiles_0.1_1_5_50_95_99_99.9": [rounded(x) for x in np.percentile(
                    finite_values, [0.1, 1, 5, 50, 95, 99, 99.9])] if finite_values.size else None,
                "median_lag1_rank_autocorr": rounded(lag.median()) if not lag.empty else None,
            }
        for left, right in combinations(FIELDS, 2):
            a, b = ranks[left].align(ranks[right], join="inner")
            correlations = a.iloc[::20].corrwith(b.iloc[::20], axis=1).dropna()
            if not correlations.empty:
                summary["pairwise_rank_corr"].setdefault(left, {})[right] = rounded(correlations.median())
        print(json.dumps(summary, allow_nan=False))
        # Use the ticket's questions and branches to explain the summary.
        ''')


def format_ticket_markdown(ticket):
    """Render a BRAIN Labs diagnostic ticket in the collaborator template."""
    lines = [
        f"### BRAIN Labs 诊断工单 `{ticket['ticket_id']}` ({ticket['status']})",
        f"- **Region/Universe/Delay** `{_scope_label(ticket['region'], ticket['universe'], ticket['delay'])}`"
        + (f"（数据集 `{ticket['dataset']}`）" if ticket.get("dataset") else ""),
        f"- **目标字段** {', '.join(f'`{f}`' for f in ticket['fields'])}",
        f"- **触发规则** `{ticket['trigger_rule']}` — {TRIGGER_RULES.get(ticket['trigger_rule'], '')}",
    ]
    if ticket.get("source_run_id"):
        lines.append(f"- **关联 Run** `{ticket['source_run_id']}`")
    lines.append("- **待验证的具体定量问题**")
    lines.extend(f"  {idx}. {question}" for idx, question in enumerate(ticket["questions"], 1))
    lines.append("- **期望回传的汇总统计指标（禁止导出原始数据）**")
    lines.extend(f"  - {metric}" for metric in ticket["expected_metrics"])
    lines.append("- **基于诊断结果的下一步分支方案**")
    for branch in ticket["branches"]:
        condition, action = branch["condition"], branch["action"]
        lines.append(f"  - {condition}" if condition == action else f"  - 若 {condition} → {action}")
    return "\n".join(lines)


def build_ticket(payload):
    """Validate a Labs ticket and construct its summary-only collaborator payload."""
    if not isinstance(payload, dict):
        raise CliError("INVALID_INPUT", "Labs spec must be a JSON object", "provide a valid `labs open --spec` object")
    raw_trigger = payload.get("trigger_rule", payload.get("trigger"))
    trigger_rule = raw_trigger.strip().lower() if isinstance(raw_trigger, str) else ""
    trigger_rule = _TRIGGER_ALIASES.get(trigger_rule, trigger_rule)
    if trigger_rule not in TRIGGER_RULES:
        raise CliError("INVALID_INPUT", "invalid Labs trigger_rule", "choose anomaly, distribution, or interaction")
    delay = payload.get("delay")
    if type(delay) is not int or not 0 <= delay <= 2**63 - 1:
        raise CliError("INVALID_INPUT", "Labs ticket requires a non-negative integer delay",
                       "pass --delay 1 or the delay for the affected simulation scope")
    now = _utc_now()
    ticket = {
        "ticket_id": f"labs-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}",
        "status": "OPEN",
        "trigger_rule": trigger_rule,
        "region": _token(payload.get("region"), "region").upper(),
        "universe": _token(payload.get("universe"), "universe").upper(),
        "delay": delay,
        "dataset": _token(payload.get("dataset"), "dataset", required=False),
        "fields": _fields(payload.get("fields")),
        "questions": [_text(item, "question") for item in _items(payload.get("questions"), "questions")],
        "expected_metrics": [_text(item, "expected metric") for item in
                             _items(payload.get("expected_metrics"), "expected_metrics")],
        "branches": [_branch(item) for item in _items(payload.get("branches"), "branches")],
        "source_run_id": _token(payload.get("source_run_id"), "source run ID", required=False),
        "resolution": None,
        "selected_branch": None,
        "created_at": now,
        "updated_at": now,
        "resolved_at": None,
    }
    ticket["notebook_snippet"] = build_labs_notebook_snippet(ticket)
    return ticket


def _workspace_engine(research_dir):
    # State machine imports Labs. Defer this import to standalone CLI mutations.
    from .state_machine import StateMachineEngine
    return StateMachineEngine(research_dir)


def open_ticket(payload, *, research_dir=None):
    """Open a standalone ticket serialized with proposal and gate decisions."""
    import cache_store

    ticket = build_ticket(payload)
    engine = _workspace_engine(research_dir)
    with cache_store.file_lock(engine.state_lock_path):
        if ticket["source_run_id"]:
            engine.load_run_raw(ticket["source_run_id"])
        workspace_db.insert_labs_ticket(ticket, research_dir=engine.root)
    return {**ticket, "markdown_card": format_ticket_markdown(ticket)}


def _aggregate_value(value, depth=0):
    if depth > 5:
        return False
    if value is None or type(value) in (int, bool):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, str):
        return len(value) <= 256 and "\n" not in value and "\r" not in value
    if isinstance(value, list):
        return len(value) <= _MAX_ITEMS and all(
            item is None or type(item) in (int, float, bool) and _aggregate_value(item) for item in value)
    if isinstance(value, dict):
        return len(value) <= _MAX_ITEMS and all(
            isinstance(key, str) and 0 < len(key.strip()) <= 128 and "\n" not in key and "\r" not in key
            and _aggregate_value(item, depth + 1) for key, item in value.items())
    return False


def _metrics_object(metrics):
    if metrics is None:
        return {}
    try:
        if isinstance(metrics, str):
            if len(metrics.encode("utf-8")) > _MAX_METRICS_BYTES:
                raise ValueError
            metrics = json.loads(metrics)
        if (not isinstance(metrics, dict) or not _aggregate_value(metrics)
                or len(json.dumps(metrics, allow_nan=False).encode("utf-8")) > _MAX_METRICS_BYTES):
            raise ValueError
    except (ValueError, RecursionError) as exc:
        raise CliError("INVALID_INPUT", "invalid Labs summary metrics",
                       "pass a compact JSON object of finite aggregates, such as coverage, quantiles, or correlations") from exc
    return metrics


def resolve_ticket(ticket_id, *, summary, metrics=None, chosen_branch=None, status="RESOLVED", research_dir=None):
    """Close an open ticket and restore the corresponding linked gate, if any."""
    import cache_store

    if status not in {"RESOLVED", "CANCELLED"}:
        raise CliError("INVALID_INPUT", "invalid Labs ticket closing status", "use `labs resolve` or `labs cancel`")
    summary = _text(summary, "summary" if status == "RESOLVED" else "cancellation reason", max_length=4000)
    metrics = _metrics_object(metrics)
    chosen_branch = _text(chosen_branch, "chosen branch", required=False)
    engine = _workspace_engine(research_dir)
    with cache_store.file_lock(engine.state_lock_path):
        now = _utc_now()
        try:
            updated, transition, applied = workspace_db.close_labs_ticket(ticket_id, {
                "status": status,
                "resolution": {"summary": summary, "metrics": metrics, "resolved_at": now},
                "selected_branch": chosen_branch,
                "resolved_at": now,
                "updated_at": now,
            }, resumable_states=REQUEST_STATES, research_dir=engine.root)
        except ValueError as exc:
            raise CliError("INVALID_INPUT", "linked run has an invalid Labs resume state",
                           "inspect the linked run and repair its recorded state before closing the ticket") from exc
        if updated is None:
            raise CliError("NOT_FOUND", f"Labs ticket {ticket_id!r} not found", "run `research_cli.py labs list`")
        if not applied:
            raise CliError("INVALID_INPUT", f"Labs ticket {ticket_id} is already {updated['status']}",
                           f"inspect it via `research_cli.py labs show {ticket_id}`")
    result = {**updated, "markdown_card": format_ticket_markdown(updated)}
    if transition:
        result["run_transition"] = transition
    return result


def scope_fields(arms):
    """Collect effective scope and referenced datafields for each FASTEXPR or Python arm."""
    by_scope = defaultdict(set)
    for arm in arms:
        settings = ml.normalize_setting_keys(arm.get("settings") or {})
        region = settings.get("region") or ml.DEFAULT_SETTINGS["region"]
        universe = settings.get("universe") or ml.DEFAULT_SETTINGS["universe"]
        delay = settings.get("delay")
        if delay is None:
            delay = ml.DEFAULT_SETTINGS["delay"]
        expression = arm.get("expression") or ""
        if ml.is_python_language(settings.get("language")):
            report = validation.lint_python_source(expression, lookback=arm.get("lookback"))
            fields = report["declared_fields"] or []
        else:
            try:
                fields = fastexpr_ast.field_names(fastexpr_ast.parse_program(expression))
            except fastexpr_ast.ParseError:
                fields = []
        if isinstance(region, str) and isinstance(universe, str) and type(delay) is int and 0 <= delay <= 2**63 - 1:
            by_scope[(region.strip().upper(), universe.strip().upper(), delay)].update(fields)
    return by_scope


def select_request_scope(arms, fields):
    """Pick the unique affected scope for a linked diagnostic and validate its fields."""
    scopes = scope_fields(arms)
    if fields is None and len(scopes) == 1:
        scope, referenced = next(iter(scopes.items()))
        return scope, sorted(referenced)
    if fields is not None:
        targets = _fields(fields)
        candidates = [scope for scope, referenced in scopes.items()
                      if {field.casefold() for field in targets} <= {field.casefold() for field in referenced}]
        if len(candidates) == 1:
            return candidates[0], targets
    raise CliError("INVALID_INPUT", "the Labs request must identify one affected scope in the linked run",
                   "use --fields to select fields from one scope, or open a standalone ticket with an explicit scope")


def _scope_label(region, universe, delay):
    return f"{region}/{universe}/D{delay}"


def open_ticket_warnings_for_arms(arms, *, research_dir=None):
    """Return one warning per open ticket and scope, combining fields across arms."""
    limit = 20
    warnings = []
    for (region, universe, delay), fields in sorted(scope_fields(arms).items()):
        hits = workspace_db.find_open_blocking_tickets(region, universe, delay, fields,
                                                       limit=limit - len(warnings), research_dir=research_dir)
        scope = _scope_label(region, universe, delay)
        warnings.extend({"code": "LABS_TICKET_OPEN", "ticket_id": hit["ticket_id"],
                         "overlapping_fields": hit["overlapping_fields"], "scope": scope}
                        for hit in hits)
        if len(warnings) == limit:
            break
    return warnings


def open_ticket_summary(research_dir=None, *, limit=50):
    """Return a bounded list of open ticket metadata and the total open count."""
    count, rows = workspace_db.summarize_open_labs_tickets(limit=min(limit, _MAX_LIST), research_dir=research_dir)
    tickets = [{"scope": _scope_label(row["region"], row["universe"], row["delay"]),
                **{key: row[key] for key in ("ticket_id", "trigger_rule", "fields", "source_run_id")}} for row in rows]
    return {"count": count, "tickets": tickets}


def _entry_anomalies(entry):
    split = entry.get("out_of_sample") or {}
    turnover = (split.get("train") or {}).get("turnover")
    if ((type(turnover) is int or type(turnover) is float and math.isfinite(turnover)) and turnover > 0.70):
        yield "high_turnover"
    if entry.get("checks_state") in {"stale", "missing", "partial"}:
        return
    checks = normalize_checks(entry.get("checks"))
    for name, anomaly in (("LOW_SUB_UNIVERSE_SHARPE", "low_sub_universe"),
                          ("CONCENTRATED_WEIGHT", "concentrated_weight")):
        if str(checks.get(name, {}).get("result") or "").strip().upper() == "FAIL":
            yield anomaly


def suggest_delegations(research_dir=None, *, region=None, field=None, min_attempts=3, limit=50, entries=None):
    """Suggest diagnostics for recurring market anomalies on distinct completed requests."""
    root = Path(research_dir or research_paths.ensure_dir()).expanduser().resolve()
    if entries is None:
        entries = alpha_registry.load_entries(root / alpha_registry.DEFAULT_FILENAME)
    buckets = defaultdict(dict)
    for entry in entries:
        if (entry.get("role") in {"control", "diagnostic"} or entry.get("execution_outcome") == "failed"
                or entry.get("metrics_pending") or not entry.get("expression")):
            continue
        if str(entry.get("status") or "").upper() not in alpha_registry.FINISHED_STATUSES | {"UNSUBMITTED"}:
            continue
        settings = ml.normalize_setting_keys(entry.get("settings") or {})
        entry_region, universe, delay = (settings.get(key) for key in ("region", "universe", "delay"))
        if (not isinstance(entry_region, str) or not entry_region.strip() or not isinstance(universe, str)
                or not universe.strip() or type(delay) is not int or not 0 <= delay <= 2**63 - 1):
            continue
        entry_region = entry_region.strip().upper()
        universe = universe.strip().upper()
        if region and entry_region != region.strip().upper():
            continue
        settings["language"] = alpha_registry.entry_language(entry)
        fields = entry.get("fields") or scope_fields([{**entry, "settings": settings}]).get(
            (entry_region, universe, delay), ())
        request = alpha_registry.placeholder_key(entry["expression"], settings)
        anomalies = set(_entry_anomalies(entry))
        for name in fields:
            if isinstance(name, str) and (field is None or field.casefold() == name.casefold()):
                by_request = buckets[(entry_region, universe, delay, name)]
                by_request.setdefault(request, set()).update(anomalies)

    suggestions = []
    for (reg, univ, delay, name), requests in buckets.items():
        if len(requests) < max(2, min_attempts):
            continue
        counts = Counter(anomaly for findings in requests.values() for anomaly in findings)
        anomalies = [f"{anomaly}({count})" for anomaly, count in sorted(counts.items()) if count >= 2]
        if anomalies:
            suggestions.append({"field": name, "scope": _scope_label(reg, univ, delay), "attempts": len(requests),
                                "anomalies": anomalies, "recommended_trigger": "anomaly", "_scope": (reg, univ, delay)})
    suggestions.sort(key=lambda item: (-len(item["anomalies"]), -item["attempts"], item["field"], item["scope"]))
    bounded = suggestions[:min(limit, _MAX_LIST)]
    for item in bounded:
        hits = workspace_db.find_open_blocking_tickets(*item.pop("_scope"), [item["field"]], limit=1, research_dir=root)
        item.update(has_open_ticket=bool(hits), open_ticket_id=hits[0]["ticket_id"] if hits else None)
    return bounded


def cmd_labs_open(args, _sink):
    """Open a structured diagnostic ticket for a specified scope and fields."""
    if args.spec:
        spec_path = Path(args.spec).expanduser().resolve()
        if not spec_path.is_file():
            raise CliError("NOT_FOUND", f"Labs spec file not found: {spec_path}", "pass a valid JSON ticket spec")
        try:
            payload = json.loads(spec_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, RecursionError) as exc:
            raise CliError("INVALID_INPUT", "invalid Labs spec JSON", "fix JSON syntax in --spec") from exc
    else:
        payload = {"trigger_rule": args.trigger, "region": args.region, "universe": args.universe,
                   "delay": args.delay, "dataset": args.dataset, "fields": args.fields, "questions": args.question,
                   "expected_metrics": args.expect, "branches": args.branch, "source_run_id": args.run_id}
    return open_ticket(payload, research_dir=args.research_dir)


def cmd_labs_list(args, _sink):
    """List a bounded set of diagnostic tickets in the workspace."""
    tickets = workspace_db.list_labs_tickets(status=args.status, field=args.field,
                                             limit=min(args.limit, _MAX_LIST), research_dir=args.research_dir)
    return {"count": len(tickets), "tickets": [{key: ticket[key] for key in (
        "ticket_id", "status", "trigger_rule", "region", "universe", "delay", "dataset", "fields", "source_run_id",
        "created_at", "updated_at", "resolved_at") } for ticket in tickets]}


def cmd_labs_show(args, _sink):
    """Show one complete ticket, including its collaborator card and notebook snippet."""
    ticket = workspace_db.get_labs_ticket(args.ticket_id, research_dir=args.research_dir)
    if ticket is None:
        raise CliError("NOT_FOUND", f"Labs ticket {args.ticket_id!r} not found", "run `research_cli.py labs list`")
    return {**ticket, "markdown_card": format_ticket_markdown(ticket)}


def cmd_labs_resolve(args, _sink):
    """Attach collaborator summary metrics, close the ticket, and resume its linked gate."""
    return resolve_ticket(args.ticket_id, summary=args.summary, metrics=args.metrics, chosen_branch=args.chosen_branch,
                          research_dir=args.research_dir)


def cmd_labs_cancel(args, _sink):
    """Cancel an open diagnostic ticket and resume its linked gate."""
    return resolve_ticket(args.ticket_id, summary=args.reason or "cancelled", status="CANCELLED",
                          research_dir=args.research_dir)


def cmd_labs_suggest(args, _sink):
    """Report fields with recurring simulation anomalies in local evidence."""
    suggestions = suggest_delegations(args.research_dir, region=args.region, field=args.field,
                                      min_attempts=args.min_attempts, limit=args.limit)
    return {"count": len(suggestions), "suggestions": suggestions}
