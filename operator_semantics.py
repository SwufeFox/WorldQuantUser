#!/usr/bin/env python3
"""Semantic rules the BRAIN operator catalog does not publish.

The catalog tells you an operator's *name* and its *signature string*. It does
not tell you that ``kth_element`` needs ``k <= d``, that ``bucket`` takes exactly
one of ``range``/``buckets``, or that ``buckets`` must be strictly increasing.
Those invariants live only in the documentation prose and in the simulator's
rejections, so they cannot be derived and cannot be fetched.

This module keeps them as **data**, not code: a JSON file whose location follows
the same resolution as every other metadata snapshot, so the rule set can be
edited, diffed and re-pointed without touching the linter. Parameter names are
read from the live catalog's own ``definition`` string rather than hardcoded, so
a renamed parameter degrades to "skipped" instead of "wrongly rejected".

Design stance: catalogs (operator names, field names, arity) are fetched and
cached. Invariants (cross-argument constraints, return types) are curated, because
no endpoint exposes them. Keeping the two in different places is the point --
the curated file is small, stable, and says where it came from.
"""

from __future__ import annotations

import json
import os
import re

__all__ = [
    "default_rules_path",
    "load_rules",
    "rule_findings",
    "return_type_findings",
]

_DEFAULT_NAME = "operator-rules.json"


def default_rules_path():
    """Return the curated rule file inside the metadata cache root."""
    import research_paths

    return os.path.join(str(research_paths.metadata_root()), _DEFAULT_NAME)


def packaged_default_path():
    """The copy shipped with the tool, used when the cache has no rule file."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "operator_rules.default.json")


def resolve_rules_path(path=None):
    """Cache copy first, packaged default second.

    Two locations on purpose: the cache copy is where tuning happens (it survives
    a code pull and needs no commit), and the packaged copy keeps a fresh clone
    working instead of silently linting with zero rules.
    """
    if path:
        return path
    cached = default_rules_path()
    if os.path.exists(cached):
        return cached
    return packaged_default_path()


def load_rules(path=None):
    """Load the rule file. Missing or malformed means "no rules", never a crash."""
    path = resolve_rules_path(path)
    if not os.path.exists(path):
        return {"returns": {}, "constraints": [], "path": path, "present": False}
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {"returns": {}, "constraints": [], "path": path, "present": False}
    payload.setdefault("returns", {})
    payload.setdefault("constraints", [])
    payload["path"] = path
    payload["present"] = True
    return payload


def _split_args(text, start):
    """Split a call's argument text at top-level commas, respecting quotes.

    ``start`` indexes the opening parenthesis. Returns ``(args, end)`` where each
    arg is ``(name_or_None, raw_text)`` and ``end`` is the index of the closing
    parenthesis (or ``len(text)`` when unbalanced).
    """
    args, depth, index = [], 0, start
    quote = None
    current = []
    while index < len(text):
        char = text[index]
        if quote:
            current.append(char)
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            current.append(char)
        elif char == "(":
            depth += 1
            if depth > 1:
                current.append(char)
        elif char == ")":
            depth -= 1
            if depth == 0:
                break
            current.append(char)
        elif char == "," and depth == 1:
            args.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    tail = "".join(current)
    if tail.strip() or args:
        args.append(tail)
    parsed = []
    for raw in args:
        text_arg = raw.strip()
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)\s*(.+)$", text_arg, re.S)
        if match:
            parsed.append((match.group(1), match.group(2).strip()))
        else:
            parsed.append((None, text_arg))
    return parsed, index


def _param_names(definition):
    """Parameter names in order, read from an operator's signature string.

    ``ts_backfill(x,lookback = d, k=1)`` -> ``["x", "lookback", "k"]``.
    Returns ``[]`` when the definition cannot be parsed, which makes every rule
    on that operator skip rather than misfire.
    """
    if not definition:
        return []
    text = definition.replace("“", '"').replace("”", '"')
    open_index = text.find("(")
    if open_index < 0:
        return []
    args, _ = _split_args(text, open_index)
    names = []
    for name, value in args:
        if name:
            names.append(name)
            continue
        # A bare identifier is how the catalogue writes a positional parameter
        # (``kth_element(x, d, k, ...)``). Anything else means the signature is
        # not something this parser can index, so skip the rule entirely.
        token = value.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            names.append(token)
        else:
            return []
    return names


def _number(value):
    if value is None:
        return None
    try:
        return float(value.strip().strip("\"'"))
    except (TypeError, ValueError):
        return None


def _is_valid_literal(value):
    if value is None:
        return False
    return value.strip().strip("\"'").lower() not in {"nan", "none", "null", ""}


def _number_list(value):
    if value is None:
        return None
    cleaned = value.strip().strip("\"'")
    parts = [part.strip() for part in cleaned.split(",")]
    out = []
    for part in parts:
        try:
            out.append(float(part))
        except ValueError:
            return None
    return out


def rule_findings(expression, operator_index=None, rules=None, operator_name=None):
    """Check curated cross-argument invariants for whichever operators appear.

    ``operator_name`` limits the work to one call (used by tests); otherwise every
    known operator with a rule is checked.
    """
    rules = rules if rules is not None else load_rules()
    constraints = rules.get("constraints") or []
    if not constraints or not expression:
        return []

    by_operator = {}
    for rule in constraints:
        if rule.get("operator"):
            by_operator.setdefault(rule["operator"], []).append(rule)

    findings = []
    for name, operator_rules in by_operator.items():
        if operator_name and name != operator_name:
            continue
        for match in re.finditer(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\s*\(", expression):
            args, _end = _split_args(expression, match.end() - 1)
            definition = None
            if operator_index and name in operator_index:
                definition = operator_index[name].get("definition")
            params = _param_names(definition)

            values = {}
            for position, (arg_name, raw) in enumerate(args):
                if arg_name:
                    values[arg_name] = raw
                elif position < len(params):
                    values[params[position]] = raw

            for rule in operator_rules:
                kind = rule.get("kind")
                where = {"operator": name, "offset": match.start(), "rule": kind}
                if kind in {"le", "lt"}:
                    left = _number(values.get(rule.get("left")))
                    right = _number(values.get(rule.get("right")))
                    if left is None or right is None:
                        continue
                    bad = left > right if kind == "le" else left >= right
                    if bad:
                        findings.append(dict(
                            where,
                            code="operator_arg_constraint",
                            severity="error",
                            message=rule.get("message") or (
                                "%s requires %s %s %s"
                                % (name, rule.get("left"),
                                   "<=" if kind == "le" else "<", rule.get("right"))
                            ),
                        ))
                elif kind == "exactly_one":
                    keys = rule.get("keys") or []
                    valid = [key for key in keys if _is_valid_literal(values.get(key))]
                    if len(valid) != 1:
                        findings.append(dict(
                            where,
                            code="operator_arg_constraint",
                            severity="error",
                            message=rule.get("message") or (
                                "%s requires exactly one of %s to be set (got %s)"
                                % (name, "/".join(keys), ", ".join(valid) or "none")
                            ),
                        ))
                elif kind == "increasing":
                    numbers = _number_list(values.get(rule.get("key")))
                    if numbers and any(b <= a for a, b in zip(numbers, numbers[1:])):
                        findings.append(dict(
                            where,
                            code="operator_arg_constraint",
                            severity="error",
                            message=rule.get("message") or (
                                "%s requires %s to be strictly increasing"
                                % (name, rule.get("key"))
                            ),
                        ))
    return findings


def return_type_findings(expression, operator_index=None, rules=None):
    """Reject a signal whose outermost call returns something that is not Matrix.

    Only the curated return types are consulted (``bucket`` and ``densify`` on
    this account), so an unknown operator is silently allowed through rather than
    guessed at.
    """
    rules = rules if rules is not None else load_rules()
    returns = rules.get("returns") or {}
    if not returns or not expression:
        return []
    tail = expression.rstrip().rstrip(";").strip()
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", tail)
    if not match or match.group(1) not in returns or match.start() != 0:
        return []
    name = match.group(1)
    return [{
        "code": "signal_type",
        "severity": "error",
        "operator": name,
        "offset": 0,
        "message": "signal is the output of %s(), which is %s-typed; BRAIN needs a "
                   "Matrix-valued signal" % (name, returns[name]),
    }]