"""Pre-flight linting for FASTEXPR expressions.

The expensive failure mode this module removes: an expression passes
``simulate --dry-run`` (which only checks payload shape), gets submitted, and
only then does BRAIN reject it -- after the submit quota is spent and after the
polling wait. That is exactly what happened with ``regression_neut``, which has
a documentation page but is *not* one of the 66 operators in ``GET /operators``.

So the lint compares every operator name in the expression against the
authoritative operator list, and additionally against a small learned denylist
of operators the simulator has rejected in this config. The denylist is written
by :func:`record_rejection`, which the simulation loop calls when BRAIN returns
an operator-level error, so the same trap cannot be stepped in twice.

Checks, in the order they run:

``empty``              expression is blank
``unbalanced_parens``  parentheses do not balance
``unbalanced_string``  a quoted string literal is not closed
``unknown_operator``   identifier called as a function is not a real operator
``operator_rejected``  operator is on the learned denylist
``arity_suspicious``   argument count is outside the documented signature
``unknown_identifier`` bare identifier is neither a known field nor a literal
"""

from __future__ import annotations

import json
import os
import re

__all__ = [
    "build_operator_index",
    "lint_all",
    "extract_calls",
    "lint_expression",
    "load_denylist",
    "record_rejection",
    "default_denylist_path",
]

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LITERALS = frozenset({"true", "false", "nan", "inf", "null", "none"})


def _mask_strings(expression):
    """Blank out string literals so commas/parens inside them are ignored.

    Character positions are preserved so reported offsets stay meaningful.
    """
    characters = list(expression)
    inside = False
    quote = ""
    for index, char in enumerate(expression):
        if inside:
            if char == quote:
                inside = False
            characters[index] = " "
        elif char in {'"', "'"}:
            inside = True
            quote = char
            characters[index] = " "
    return "".join(characters), inside


def _parse_signature(definition):
    """Extract ``(min_arity, max_arity, signature)`` from an operator definition.

    Definitions look like ``add(x, y, filter = false), x + y`` -- the parameter
    list is the first balanced parenthesised group. Parameters carrying ``=`` are
    optional; a trailing ``...`` means variadic.
    """
    if not isinstance(definition, str) or "(" not in definition:
        return None, None, None
    start = definition.index("(")
    depth = 0
    end = None
    for position in range(start, len(definition)):
        char = definition[position]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                end = position
                break
    if end is None:
        return None, None, None
    inner = definition[start + 1:end].strip()
    signature = definition[:end + 1]
    if not inner:
        return 0, 0, signature
    params = [part.strip() for part in inner.split(",")]
    required = 0
    variadic = False
    for param in params:
        if not param:
            continue
        if param in {"...", "*args", "*"}:
            variadic = True
            continue
        if "=" in param:
            continue
        required += 1
    maximum = None if variadic else len([p for p in params if p])
    return required, maximum, signature


def build_operator_index(operator_records):
    """Index ``GET /operators`` records by name with parsed arity.

    Returns ``{name: {category, signature, min_arity, max_arity}}``.
    """
    index = {}
    for record in operator_records or []:
        name = record.get("name")
        if not name:
            continue
        min_arity, max_arity, signature = _parse_signature(record.get("definition"))
        index[name] = {
            "category": record.get("category"),
            "signature": signature,
            "definition": record.get("definition"),
            "min_arity": min_arity,
            "max_arity": max_arity,
        }
    return index


def _count_args(text, open_index):
    """Count top-level arguments of the call whose ``(`` is at ``open_index``.

    Returns ``(arg_count, close_index)``; ``close_index`` is ``None`` when the
    group never closes.
    """
    depth = 0
    commas = 0
    has_content = False
    for position in range(open_index, len(text)):
        char = text[position]
        if char == "(":
            depth += 1
            if depth > 1:
                has_content = True
        elif char == ")":
            depth -= 1
            if depth == 0:
                return (commas + 1 if has_content else 0), position
        elif char == "," and depth == 1:
            commas += 1
            has_content = True
        elif not char.isspace() and depth == 1:
            has_content = True
    return (commas + 1 if has_content else 0), None


def extract_calls(expression):
    """Return every function call in the expression as ``{name, arg_count, offset}``.

    Named arguments such as ``filter=true`` or ``range="0.1,1,0.1"`` are not
    calls and are skipped, as are bare literal identifiers.
    """
    masked, _ = _mask_strings(expression)
    calls = []
    for match in _IDENT.finditer(masked):
        name = match.group(0)
        position = match.end()
        while position < len(masked) and masked[position].isspace():
            position += 1
        if position >= len(masked) or masked[position] != "(":
            continue
        arg_count, _ = _count_args(masked, position)
        calls.append({"name": name, "arg_count": arg_count, "offset": match.start()})
    return calls


def extract_identifiers(expression):
    """Return bare (non-called, non-assigned) identifiers with their offsets."""
    masked, _ = _mask_strings(expression)
    identifiers = []
    for match in _IDENT.finditer(masked):
        name = match.group(0)
        position = match.end()
        while position < len(masked) and masked[position].isspace():
            position += 1
        if position < len(masked) and masked[position] in "(=":
            continue
        identifiers.append({"name": name, "offset": match.start()})
    return identifiers


def default_denylist_path():
    """Denylist location: ``<repo>/research/operator-denylist.json``."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "research", "operator-denylist.json")


def load_denylist(path=None):
    """Load the learned denylist as ``{operator: {reason, ...}}``."""
    path = path or default_denylist_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        # A corrupt denylist must not block research; degrade to empty.
        return {}
    entries = payload.get("operators") if isinstance(payload, dict) else None
    return entries or {}


def record_rejection(operator, message, context=None, path=None, severity="block"):
    """Persist an operator rejection so future lints flag it immediately.

    Called when BRAIN rejects a submitted expression with an operator-level
    error. Appends a timestamp and the rejection message; repeat rejections bump
    a counter instead of duplicating the entry.

    ``severity`` is ``block`` when the operator itself is unknown or unavailable
    (lint fails), and ``warn`` when the operator is real but rejected this
    particular input kind (lint warns). Defaulting to ``block`` is deliberate:
    silently downgrading an unknown operator is the failure this module exists
    to prevent.
    """
    import datetime

    if severity not in {"block", "warn"}:
        raise ValueError("severity must be 'block' or 'warn', got %r" % (severity,))
    path = path or default_denylist_path()
    entries = load_denylist(path)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    existing = entries.get(operator)
    if existing:
        existing["count"] = int(existing.get("count", 1)) + 1
        existing["last_seen"] = stamp
        existing["message"] = message
        # A hard rejection outranks a soft one; never soften an existing block.
        if severity == "block":
            existing["severity"] = "block"
    else:
        entries[operator] = {
            "reason": "rejected by the BRAIN simulator",
            "message": message,
            "severity": severity,
            "context": context or {},
            "first_seen": stamp,
            "last_seen": stamp,
            "count": 1,
        }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    document = {
        "_comment": "Learned operator denylist. Written by expression_lint."
                    "record_rejection when the simulator rejects an operator.",
        "operators": entries,
    }
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)
    return entries[operator]


def lint_expression(expression, operator_index=None, denylist=None,
                    known_fields=None, strict_fields=False):
    """Lint one expression. Returns ``{ok, errors, warnings, ...}``.

    ``operator_index`` comes from :func:`build_operator_index`; when omitted the
    operator-name check is skipped (offline use). ``known_fields`` enables the
    unknown-identifier check.
    """
    errors = []
    warnings = []
    expression = expression or ""

    if not expression.strip():
        errors.append({"code": "empty", "message": "expression is blank"})
        return {
            "ok": False,
            "operator_count": 0,
            "operators": [],
            "errors": errors,
            "warnings": warnings,
        }

    masked, unterminated = _mask_strings(expression)
    if unterminated:
        errors.append({"code": "unbalanced_string",
                       "message": "a quoted string literal is never closed"})

    depth = 0
    for char in masked:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                break
    if depth != 0:
        errors.append({"code": "unbalanced_parens",
                       "message": "parentheses do not balance (net %+d)" % depth})

    calls = extract_calls(expression)
    used = []
    for call in calls:
        name = call["name"]
        used.append(name)
        entry = {"name": name, "arg_count": call["arg_count"],
                 "offset": call["offset"]}
        if denylist and name in denylist:
            record = denylist[name] or {}
            reason = record.get("message") or record.get("reason") or \
                "previously rejected"
            severity = record.get("severity", "block")
            item = {
                "code": "operator_rejected",
                "operator": name,
                "offset": call["offset"],
                "message": "operator %r was rejected by the simulator before: %s"
                           % (name, reason),
            }
            if severity == "block":
                errors.append(item)
                entry["status"] = "denylisted"
            else:
                warnings.append(item)
                entry["status"] = "denylisted_soft"
        elif operator_index is not None:
            meta = operator_index.get(name)
            if meta is None:
                errors.append({
                    "code": "unknown_operator",
                    "operator": name,
                    "offset": call["offset"],
                    "message": "operator %r is not in the BRAIN operator list"
                               % name,
                })
                entry["status"] = "unknown"
            else:
                entry["status"] = "known"
                entry["category"] = meta.get("category")
                minimum = meta.get("min_arity")
                maximum = meta.get("max_arity")
                count = call["arg_count"]
                if minimum is not None and count < minimum:
                    warnings.append({
                        "code": "arity_suspicious",
                        "operator": name,
                        "message": "%s expects at least %d argument(s) per "
                                   "`%s`, got %d"
                                   % (name, minimum, meta.get("signature"), count),
                    })
                if maximum is not None and count > maximum:
                    warnings.append({
                        "code": "arity_suspicious",
                        "operator": name,
                        "message": "%s expects at most %d argument(s) per `%s`, "
                                   "got %d"
                                   % (name, maximum, meta.get("signature"), count),
                    })
        else:
            entry["status"] = "unchecked"
        # Nested calls are visited by the same scan, so no recursion needed.

    if known_fields is not None:
        known = set(known_fields)
        seen = set()
        for identifier in extract_identifiers(expression):
            name = identifier["name"]
            if name in seen or name.lower() in _LITERALS:
                continue
            seen.add(name)
            if name in known:
                continue
            # Only flag things that actually look like datafield ids; a bare
            # short word is more likely a keyword we do not model.
            if "_" not in name and len(name) < 6:
                continue
            item = {
                "code": "unknown_identifier",
                "identifier": name,
                "offset": identifier["offset"],
                "message": "%r is not a known datafield in this config" % name,
            }
            if strict_fields:
                errors.append(item)
            else:
                warnings.append(item)

    return {
        "ok": not errors,
        "operator_count": len(used),
        "operators": used,
        "calls": [call for call in calls],
        "details": None,
        "errors": errors,
        "warnings": warnings,
    }

def lint_all(pairs, operator_index=None, denylist=None, known_fields=None,
             strict_fields=False):
    """Lint a batch of ``(expression, decay)`` pairs.

    Returns ``{ok, results, errors, warnings}`` where ``errors`` and
    ``warnings`` are flattened with the offending expression attached, which is
    the shape a CLI wants to report. Per-expression reports stay in ``results``.
    """
    results = []
    errors = []
    warnings = []
    for pair in pairs or []:
        expression = pair[0] if isinstance(pair, (tuple, list)) else pair
        report = lint_expression(
            expression,
            operator_index=operator_index,
            denylist=denylist,
            known_fields=known_fields,
            strict_fields=strict_fields,
        )
        report["expression"] = expression
        results.append(report)
        for item in report["errors"]:
            errors.append(dict(item, expression=expression))
        for item in report["warnings"]:
            warnings.append(dict(item, expression=expression))
    return {
        "ok": not errors,
        "expressions": len(results),
        "results": results,
        "errors": errors,
        "warnings": warnings,
    }
