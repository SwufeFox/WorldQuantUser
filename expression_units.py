#!/usr/bin/env python3
"""Dimensional analysis for FASTEXPR, from units the simulator itself reported.

The failure that prompted this: `cap + abs(liabilities)`. Market cap carries
`Unit[CSPrice:1, CSShare:1]` - price times shares - while total liabilities is
dimensionless in this catalogue, so the addition is refused. It cost a submission to learn,
and the same shape appeared in three expressions.

The authoritative source of a field's unit is the simulator's own rejection message, which
names it: `expected "Unit[CSPrice:1,CSShare:1]", found "Unit[]"`. `tools/probe_units.py`
already harvests that into `<research_dir>/field-units.json`, and errors observed at
simulation time are recorded there too.

This module only reasons from units that were measured. An unknown unit yields no finding,
because a guess that misfires is worse than silence - the same rule the type checker
follows.

Scope is deliberately narrow: `add`, `subtract` and the comparisons must see two operands of
the same unit, which is exactly the class of error that costs a round trip. Multiplication
and division produce combined units, and are computed only so that a nested expression can
be judged; a fractional exponent drops the unit rather than inventing a square root.
"""

from __future__ import annotations

import re

UNKNOWN = None  # a unit we do not know; distinct from dimensionless, which is {}

_DIMENSIONLESS = frozenset({"", "dimensionless", "unit[]"})
_UNIT = re.compile(r"Unit\[(.*?)\]")
_TERM = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(-?\d+)")

#: Operators whose result is a rank-like score, so the unit is dropped.
_DIMENSIONLESS_RESULT = frozenset({
    "rank", "zscore", "scale", "winsorize", "normalize", "quantile", "group_rank",
    "group_zscore", "group_scale", "bucket", "densify", "sign", "is_nan", "not",
    "and", "or", "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "ts_rank", "ts_quantile", "ts_zscore", "ts_arg_max", "ts_arg_min", "ts_count_nans",
    "ts_step", "days_from_last_change", "ts_corr",
})
#: Operators that pass the first argument's unit through.
_UNIT_PRESERVING = frozenset({
    "abs", "ts_delta", "ts_delay", "ts_mean", "ts_sum", "ts_std_dev", "ts_av_diff",
    "ts_backfill", "ts_decay_linear", "ts_product", "hump", "kth_element",
    "last_diff_value", "reverse", "inverse", "signed_power", "max", "min",
    "trade_when", "if_else", "vec_avg", "vec_sum",
})


def parse_unit(text):
    """Parse `Unit[CSPrice:1,CSShare:1]` or `dimensionless` into a dict, or None."""
    if text is None:
        return UNKNOWN
    raw = str(text).strip()
    if raw.lower() in _DIMENSIONLESS:
        return {}
    match = _UNIT.search(raw)
    if not match:
        return UNKNOWN
    body = match.group(1).strip()
    if not body:
        return {}
    unit = {}
    for name, exponent in _TERM.findall(body):
        exponent = int(exponent)
        if exponent:
            unit[name] = exponent
    return unit


def to_text(unit):
    """Render a unit the way the simulator writes it."""
    if unit is UNKNOWN:
        return "unknown"
    if not unit:
        return "dimensionless"
    return "Unit[%s]" % ",".join("%s:%d" % (k, v) for k, v in sorted(unit.items()))


def _combine(left, right, sign):
    if left is UNKNOWN or right is UNKNOWN:
        return UNKNOWN
    unit = dict(left)
    for name, exponent in right.items():
        unit[name] = unit.get(name, 0) + sign * exponent
        if not unit[name]:
            del unit[name]
    return unit


def _split_arguments(text):
    parts, depth, buffer = [], 0, ""
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(buffer.strip())
            buffer = ""
        else:
            buffer += char
    if buffer.strip():
        parts.append(buffer.strip())
    return parts


def resolve(text, field_units, _depth=0):
    """Unit of one expression fragment, or UNKNOWN."""
    if _depth > 30:
        return UNKNOWN
    fragment = (text or "").strip()
    if not fragment:
        return UNKNOWN
    if re.fullmatch(r"-?\d+(\.\d+)?", fragment):
        return {}
    call = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$", fragment, re.S)
    if call:
        operator = call.group(1).lower()
        arguments = _split_arguments(call.group(2))
        if operator in _DIMENSIONLESS_RESULT:
            return {}
        if operator in _UNIT_PRESERVING:
            return resolve(arguments[0], field_units, _depth + 1) if arguments else UNKNOWN
        if operator == "subtract" and len(arguments) >= 2:
            left = resolve(arguments[0], field_units, _depth + 1)
            return left
        if operator == "add" and len(arguments) >= 2:
            return resolve(arguments[0], field_units, _depth + 1)
        if operator in ("multiply", "divide") and len(arguments) >= 2:
            left = resolve(arguments[0], field_units, _depth + 1)
            right = resolve(arguments[1], field_units, _depth + 1)
            return _combine(left, right, 1 if operator == "multiply" else -1)
        if operator == "sqrt" and arguments:
            inner = resolve(arguments[0], field_units, _depth + 1)
            return UNKNOWN if inner is UNKNOWN else UNKNOWN
        return UNKNOWN
    field = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", fragment)
    if field:
        measured = (field_units or {}).get(fragment)
        if measured is None:
            return UNKNOWN
        return measured if isinstance(measured, dict) else parse_unit(measured)
    return UNKNOWN


_ADDITIVE = frozenset({"add", "subtract", "equal", "not_equal", "less", "less_equal",
                       "greater", "greater_equal", "max", "min"})


def unit_findings(expression, field_units):
    """Report additive operators whose operands carry different measured units."""
    if not field_units:
        return []
    findings = []
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression or ""):
        operator = match.group(1).lower()
        if operator not in _ADDITIVE:
            continue
        start = match.end() - 1
        depth = 0
        end = None
        for index in range(start, len(expression)):
            if expression[index] == "(":
                depth += 1
            elif expression[index] == ")":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end is None:
            continue
        arguments = _split_arguments(expression[start + 1:end])
        if len(arguments) < 2:
            continue
        left = resolve(arguments[0], field_units)
        right = resolve(arguments[1], field_units)
        if left is UNKNOWN or right is UNKNOWN or left == right:
            continue
        findings.append({
            "code": "unit_mismatch",
            "severity": "error",
            "operator": operator,
            "left": to_text(left),
            "right": to_text(right),
            "message": "%s adds %s to %s; the simulator refuses operands of different "
                       "units. Divide by a common scale instead of adding."
                       % (operator, to_text(left), to_text(right)),
        })
    return findings

#: Binary infix operators that require both sides to share a unit. `=` and `!=` are written
#: as words in this language, so only the arithmetic and ordering forms appear here.
_INFIX = ("+", "-", ">", "<")


def _top_level_pieces(text):
    """Split on binary infix operators that sit at bracket depth zero."""
    pieces, buffer, depth = [], "", 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if depth == 0 and char in _INFIX:
            previous = buffer.rstrip()
            unary = (not previous) or previous[-1] in "+-*/<>=,("
            # `>=`, `<=` and `!=` are comparisons, handled by their word forms.
            if not unary and not (char in "<>" and text[index:index + 2] in (">=", "<=")):
                pieces.append(buffer.strip())
                buffer = ""
                index += 1
                continue
        buffer += char
        index += 1
    if buffer.strip():
        pieces.append(buffer.strip())
    return [piece for piece in pieces if piece]


def infix_findings(expression, field_units):
    """Report infix expressions whose pieces carry different measured units."""
    if not field_units:
        return []
    findings = []
    for statement in re.split(r";", expression or ""):
        body = statement.split("=", 1)[-1] if "=" in statement else statement
        pieces = _top_level_pieces(body)
        if len(pieces) < 2:
            continue
        resolved = [(piece, resolve(piece, field_units)) for piece in pieces]
        known = [(piece, unit) for piece, unit in resolved if unit is not UNKNOWN]
        if len(known) < 2:
            continue
        first_piece, first_unit = known[0]
        for piece, unit in known[1:]:
            if unit == first_unit:
                continue
            findings.append({
                "code": "unit_mismatch",
                "severity": "error",
                "left": to_text(first_unit),
                "right": to_text(unit),
                "message": "infix arithmetic mixes %s (%s) with %s (%s); the simulator "
                           "refuses operands of different units. Divide by a common "
                           "scale instead of adding."
                           % (first_piece[:32], to_text(first_unit),
                              piece[:32], to_text(unit)),
            })
            break
    return findings


def load_field_units(path=None):
    """Measured units from the probe cache, or an empty mapping.

    The cache is written by ``tools/probe_units.py`` and by errors observed at simulation
    time. A missing cache means the check is silent, which is the correct answer when
    nothing has been measured.
    """
    import json
    import os
    if path is None:
        try:
            import research_paths
            path = os.path.join(research_paths.ensure_dir(), "field-units.json")
        except Exception:  # noqa: BLE001 - no research directory means no cache
            return {}
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        return {}
