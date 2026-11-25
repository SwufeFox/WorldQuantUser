"""Shared interpretation of observed platform checks.

This module describes the records supplied by a caller. It does not establish
that an endpoint finished, that every applicable check was returned, or that a
numeric value overrides the platform's verdict.
"""

from __future__ import annotations

from copy import deepcopy


# This entry is a client diagnostic, never a platform check or a coverage rule.
CLIENT_PARSE_ERROR = "__CLIENT_CHECK_PARSE_ERROR__"


def _parse_error(code, records):
    return {"result": None, "source": "client_parser", "client_parse_error": code,
            "raw_records": deepcopy(records)}


def normalize_checks(checks):
    """Copy a platform list or name-to-record mapping without losing metadata.

    Legacy mappings may contain a result string instead of a record. Duplicate
    names remain unresolved with every original occurrence attached. Unnamed or
    malformed rows are retained in an explicitly client-owned diagnostic entry.
    Neither case fabricates a platform verdict or disappears on JSON replay.
    """
    if isinstance(checks, dict):
        records = checks.items()
        mapping = True
    elif isinstance(checks, list):
        records = ((record.get("name") if isinstance(record, dict) else None, record)
                   for record in checks)
        mapping = False
    else:
        return {} if checks is None else {
            CLIENT_PARSE_ERROR: _parse_error("invalid_checks_container", [checks])}

    normalized = {}
    occurrences = {}
    malformed = []
    for name, record in records:
        if name == CLIENT_PARSE_ERROR and mapping and isinstance(record, dict) and (
                record.get("source") == "client_parser" and record.get("client_parse_error")):
            # Preserve our own persisted diagnostic without nesting it again.
            normalized[name] = deepcopy(record)
            continue
        if not isinstance(name, str) or not name.strip() or name == CLIENT_PARSE_ERROR:
            malformed.append({"key": name, "record": record} if mapping else record)
            continue
        body = dict(record) if isinstance(record, dict) else {"result": record}
        body.pop("name", None)
        prior = occurrences.setdefault(name, [])
        prior.append(record)
        normalized[name] = body if len(prior) == 1 else _parse_error("duplicate_check_name", prior)
    if malformed:
        previous = normalized.get(CLIENT_PARSE_ERROR) or {}
        normalized[CLIENT_PARSE_ERROR] = _parse_error(
            "invalid_check_records", list(previous.get("raw_records") or []) + malformed)
    return normalized


def platform_check_names(checks):
    """Observed platform names, excluding the client's malformed-payload record."""
    return set(normalize_checks(checks)) - {CLIENT_PARSE_ERROR}


def summarize_checks(checks, required=()):
    """Report observed outcomes and explicitly required but absent checks.

    ``passed`` includes nonblocking warnings, matching the existing shelf
    summary; ``warned`` identifies that subset. ``failed`` includes FAIL and
    operational ERROR so neither can permit progression. ERROR is also exposed
    in ``errored`` and is not resolved scientific evidence. Every other unknown
    result and every absent required name remains ``pending``.

    ``resolved`` means a nonempty observed set with no pending/error outcomes.
    It is separate from any caller's endpoint-completion or coverage evidence.
    """
    normalized = normalize_checks(checks)
    buckets = {key: set() for key in ("passed", "failed", "pending", "warned", "errored")}
    for name, record in normalized.items():
        if record.get("source") == "client_parser" and record.get("client_parse_error"):
            buckets["pending"].add(name)
            continue
        value = record.get("result")
        result = value.strip().upper() if isinstance(value, str) else None
        if result in {"PASS", "WARNING"}:
            buckets["passed"].add(name)
            if result == "WARNING":
                buckets["warned"].add(name)
        elif result in {"FAIL", "ERROR"}:
            buckets["failed"].add(name)
            if result == "ERROR":
                buckets["errored"].add(name)
        else:
            buckets["pending"].add(name)
    buckets["pending"].update(name for name in required
                              if name != CLIENT_PARSE_ERROR and name not in normalized)
    return {
        **{key: sorted(names) for key, names in buckets.items()},
        "available": bool(platform_check_names(normalized)),
        "resolved": bool(normalized) and not buckets["pending"] and not buckets["errored"],
    }
