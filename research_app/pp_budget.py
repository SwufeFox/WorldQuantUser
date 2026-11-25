"""Power Pool budget: operator, field and dataset gates read from the parsed signal.

The platform publishes the structural gates (Sharpe >= 1.0; at most 8 operators
counting repeats, with ``ts_backfill``/``group_backfill`` exempt; at most 3 data
fields excluding grouping keys; Power Pool correlation < 0.5; turnover,
sub-universe and robust-universe tests PASS; an active theme; a description of
at least 100 characters).  This module computes only the structural part, from
the same AST the linter uses, and reports the evidence behind every number.

It never scores a candidate and never decides eligibility alone: correlation,
tests and themes come from the platform check, and a missing input stays
missing instead of passing.
"""

from __future__ import annotations

import fastexpr_ast

PP_MAX_OPERATORS = 8
PP_MAX_FIELDS = 3

#: Calls the platform does not charge against the Power Pool operator budget.
FREE_OPERATORS = frozenset({"ts_backfill", "group_backfill"})

#: Grouping keys: they select the bucket, they are not data fields.
GROUPING_FIELDS = frozenset({
    "country", "industry", "subindustry", "currency", "market", "sector", "exchange",
})


def operator_usage(program):
    """Count operators the way the platform's ``regular.operatorCount`` does.

    Every function call and every infix operator is charged, including a unary
    minus (it is a reversal operator, not a sign).  ``?:`` is charged twice
    because both the comparison and the branch are operators.  Calls in
    :data:`FREE_OPERATORS` are reported separately and not charged.
    """
    total = free = 0
    by_name = {}
    for node in program.walk():
        if isinstance(node, fastexpr_ast.Call):
            total += 1
            by_name[node.name] = by_name.get(node.name, 0) + 1
            if node.name in FREE_OPERATORS:
                free += 1
        elif isinstance(node, fastexpr_ast.BinOp):
            total += 1
        elif isinstance(node, fastexpr_ast.UnaryOp):
            total += 1
        elif isinstance(node, fastexpr_ast.Ternary):
            total += 2
    return {"total": total, "free": free, "charged": total - free, "by_name": by_name}


def data_fields(program):
    """Data fields, excluding grouping keys, locals, named literals and keywords."""
    return [name for name in fastexpr_ast.field_names(program) if name not in GROUPING_FIELDS]


def pp_budget(expression, *, field_datasets=None, platform_operator_count=None):
    """Structural Power Pool budget of one expression.

    ``field_datasets`` maps a field id to its dataset id (from the catalog); a
    field missing from that mapping is reported as unresolved, never assumed to
    share another field's dataset.  ``platform_operator_count`` is the value the
    platform returned for a simulated alpha; when present it is authoritative
    for the total, and the AST is used only to subtract the exempt calls.
    """
    text = expression or ""
    try:
        program = fastexpr_ast.parse(text)
    except fastexpr_ast.ParseError as exc:
        return {"ok": False, "expression": text, "issues": ["parse error: %s" % exc],
                "operators": None, "operators_total": None, "free_operators": None,
                "fields": [], "n_fields": None, "datasets": [], "unresolved_fields": [],
                "operator_count_source": "unparsed"}

    usage = operator_usage(program)
    platform = platform_operator_count if isinstance(platform_operator_count, int) else None
    total = platform if platform is not None else usage["total"]
    charged = max(0, total - usage["free"]) if platform is not None else usage["charged"]
    fields = data_fields(program)
    mapping = field_datasets or {}
    unresolved = [f for f in fields if mapping.get(f) in (None, "")]
    datasets = sorted({mapping[f] for f in fields if mapping.get(f) not in (None, "")})

    issues = []
    if charged > PP_MAX_OPERATORS:
        issues.append("%d charged operators > %d" % (charged, PP_MAX_OPERATORS))
    if len(fields) > PP_MAX_FIELDS:
        issues.append("%d data fields > %d" % (len(fields), PP_MAX_FIELDS))
    if unresolved:
        issues.append("no dataset evidence for %s" % ", ".join(sorted(unresolved)))

    return {
        "ok": not issues,
        "expression": text,
        "operators": charged,
        "operators_total": total,
        "free_operators": usage["free"],
        "operator_count_source": "platform" if platform is not None else "ast",
        "operator_calls": usage["by_name"],
        "fields": fields,
        "n_fields": len(fields),
        "datasets": datasets,
        "single_dataset": len(datasets) == 1 and not unresolved,
        "unresolved_fields": sorted(unresolved),
        "issues": issues,
        "limits": {"operators": PP_MAX_OPERATORS, "fields": PP_MAX_FIELDS},
        "exempt_operators": sorted(FREE_OPERATORS),
    }


def gate_summary(budget, check=None):
    """Combine the structural budget with a platform check into named gates.

    Only inputs that are present are judged.  A gate whose evidence is missing
    is reported as ``unknown``; it is never ``pass``.  ``check`` is the payload
    from ``alpha check`` (name -> record or a list of records).
    """
    records = {}
    if isinstance(check, list):
        for record in check:
            if isinstance(record, dict) and record.get("name"):
                records.setdefault(record["name"], record)
    elif isinstance(check, dict):
        for name, record in check.items():
            records[name] = record if isinstance(record, dict) else {"result": record}

    def result(name):
        record = records.get(name) or {}
        return record.get("result")

    def value(name):
        return (records.get(name) or {}).get("value")

    gates = {
        "operators": {"value": budget.get("operators"), "limit": PP_MAX_OPERATORS,
                      "state": "pass" if budget.get("operators") is not None
                               and budget["operators"] <= PP_MAX_OPERATORS else
                               ("fail" if budget.get("operators") is not None else "unknown")},
        "fields": {"value": budget.get("n_fields"), "limit": PP_MAX_FIELDS,
                   "state": "pass" if budget.get("n_fields") is not None
                            and budget["n_fields"] <= PP_MAX_FIELDS else
                            ("fail" if budget.get("n_fields") is not None else "unknown")},
        "datasets": {"value": budget.get("datasets"), "state":
                     "unknown" if budget.get("unresolved_fields") else
                     ("pass" if budget.get("single_dataset") else "note")},
        "sharpe": {"value": value("LOW_SHARPE"), "limit": 1.0,
                   "state": _state(result("LOW_SHARPE"), value("LOW_SHARPE"), 1.0)},
        "power_pool_correlation": {"value": value("POWER_POOL_CORRELATION"), "limit": 0.5,
                                   "state": _state(result("POWER_POOL_CORRELATION"),
                                                   value("POWER_POOL_CORRELATION"), 0.5,
                                                   upper=True)},
        "turnover": {"value": value("HIGH_TURNOVER"),
                     "state": _pass_state(result("HIGH_TURNOVER"))},
        "sub_universe": {"value": value("LOW_SUB_UNIVERSE_SHARPE"),
                         "state": _pass_state(result("LOW_SUB_UNIVERSE_SHARPE"))},
        "themes": {"value": [t.get("name") for t in (records.get("MATCHES_THEMES") or {}).get("themes") or []],
                   "state": _pass_state(result("MATCHES_THEMES"))},
        "description": {"value": value("POWER_POOL_DESCRIPTION_LENGTH"), "limit": 100,
                        "state": _pass_state(result("POWER_POOL_DESCRIPTION_LENGTH"))},
    }
    blocking = sorted(name for name, gate in gates.items() if gate["state"] == "fail")
    unknown = sorted(name for name, gate in gates.items() if gate["state"] == "unknown")
    return {"gates": gates, "blocking": blocking, "unknown": unknown,
            "ok": not blocking and not unknown}


def _pass_state(result):
    if result in ("PASS", "WARNING"):
        return "pass" if result == "PASS" else "note"
    if result in ("FAIL", "ERROR"):
        return "fail"
    return "unknown"


def _state(result, number, limit, *, upper=False):
    if isinstance(number, (int, float)):
        bad = number > limit if upper else number < limit
        return "fail" if bad else "pass"
    return _pass_state(result)