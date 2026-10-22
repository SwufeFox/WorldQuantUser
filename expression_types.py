#!/usr/bin/env python3
"""A type system for FASTEXPR, so operator/field mismatches are caught before lint.

The platform's own field filter offers five types - Matrix, Vector, Group, Universe,
Symbol - and its operators accept specific ones. Until now the linter carried a hand-built
denylist of "operators that reject VECTOR fields", which was assembled from error messages
and was necessarily incomplete. A type table derives those rules instead of listing them.

Signatures come from the operator documentation; field types come from the field catalogue.
Where an operator's signature is not declared in the documentation (the window arguments,
the `filter=` flags), the convention used here is: integer and string parameters are
scalars and are not type-checked, only the value arguments are.

Inference is deliberately pragmatic. A full parser for FASTEXPR arithmetic would be a
project in itself, so an argument's type is resolved as:

* a bare field name -> that field's declared type;
* a function call -> that call's return type, recursively;
* a numeric or string literal -> scalar;
* anything else (arithmetic, comparisons) -> the type of the first typed subexpression
  inside it, or UNKNOWN.

That resolves every case that has actually caused a rejection here - a VECTOR field handed
to `rank`, `ts_delta`, `ts_backfill` or `multiply`, and a MATRIX handed to a group slot -
while staying silent instead of inventing a verdict where it cannot tell.
"""

from __future__ import annotations

import re

MATRIX = "MATRIX"
VECTOR = "VECTOR"
GROUP = "GROUP"
UNIVERSE = "UNIVERSE"
SYMBOL = "SYMBOL"
SCALAR = "SCALAR"
UNKNOWN = "UNKNOWN"

#: The five field types the platform's own filter offers.
FIELD_TYPES = (MATRIX, VECTOR, GROUP, UNIVERSE, SYMBOL)

#: operator -> (positional parameter types, return type). Value parameters only; window
#: lengths and flag parameters are scalars and are accepted as anything.
SIGNATURES = {
    # Arithmetic. These operate on matrices; a vector has to be reduced first, which is
    # exactly why `multiply` rejects event inputs.
    "abs": ([MATRIX], MATRIX),
    "add": ([MATRIX, MATRIX], MATRIX),
    "subtract": ([MATRIX, MATRIX], MATRIX),
    "multiply": ([MATRIX, MATRIX], MATRIX),
    "divide": ([MATRIX, MATRIX], MATRIX),
    "inverse": ([MATRIX], MATRIX),
    "log": ([MATRIX], MATRIX),
    "max": ([MATRIX, MATRIX], MATRIX),
    "min": ([MATRIX, MATRIX], MATRIX),
    "power": ([MATRIX, SCALAR], MATRIX),
    "signed_power": ([MATRIX, SCALAR], MATRIX),
    "sqrt": ([MATRIX], MATRIX),
    "reverse": ([MATRIX], MATRIX),
    "sign": ([MATRIX], MATRIX),
    # A group field is a different kind of object, not a number.
    "densify": ([GROUP], GROUP),
    # Logical and comparison operators all consume and return matrices.
    "and": ([MATRIX, MATRIX], MATRIX),
    "or": ([MATRIX, MATRIX], MATRIX),
    "not": ([MATRIX], MATRIX),
    "if_else": ([MATRIX, MATRIX, MATRIX], MATRIX),
    "is_nan": ([MATRIX], MATRIX),
    "equal": ([MATRIX, MATRIX], MATRIX),
    "not_equal": ([MATRIX, MATRIX], MATRIX),
    "less": ([MATRIX, MATRIX], MATRIX),
    "less_equal": ([MATRIX, MATRIX], MATRIX),
    "greater": ([MATRIX, MATRIX], MATRIX),
    "greater_equal": ([MATRIX, MATRIX], MATRIX),
    # Cross sectional: matrix in, matrix out.
    "rank": ([MATRIX], MATRIX),
    "zscore": ([MATRIX], MATRIX),
    "scale": ([MATRIX], MATRIX),
    "winsorize": ([MATRIX], MATRIX),
    "normalize": ([MATRIX], MATRIX),
    "quantile": ([MATRIX], MATRIX),
    # Vector reducers: the documented route from VECTOR to MATRIX.
    "vec_avg": ([VECTOR], MATRIX),
    "vec_sum": ([VECTOR], MATRIX),
    # Transformational. `bucket` produces a grouping, which is why it can be passed to the
    # group operators and to nothing else.
    "bucket": ([MATRIX], GROUP),
    "trade_when": ([MATRIX, MATRIX, MATRIX], MATRIX),
    # Group operators: a value per instrument, plus the group it is measured against.
    "group_rank": ([MATRIX, GROUP], MATRIX),
    "group_neutralize": ([MATRIX, GROUP], MATRIX),
    "group_zscore": ([MATRIX, GROUP], MATRIX),
    "group_scale": ([MATRIX, GROUP], MATRIX),
    "group_backfill": ([MATRIX, GROUP], MATRIX),
    "group_mean": ([MATRIX, MATRIX, GROUP], MATRIX),
    # Time series: matrix in, matrix out; windows are scalars.
    "ts_delay": ([MATRIX], MATRIX),
    "ts_delta": ([MATRIX], MATRIX),
    "ts_mean": ([MATRIX], MATRIX),
    "ts_sum": ([MATRIX], MATRIX),
    "ts_product": ([MATRIX], MATRIX),
    "ts_std_dev": ([MATRIX], MATRIX),
    "ts_zscore": ([MATRIX], MATRIX),
    "ts_rank": ([MATRIX], MATRIX),
    "ts_quantile": ([MATRIX], MATRIX),
    "ts_scale": ([MATRIX], MATRIX),
    "ts_arg_max": ([MATRIX], MATRIX),
    "ts_arg_min": ([MATRIX], MATRIX),
    "ts_av_diff": ([MATRIX], MATRIX),
    "ts_backfill": ([MATRIX], MATRIX),
    "ts_count_nans": ([MATRIX], MATRIX),
    "ts_decay_linear": ([MATRIX], MATRIX),
    "kth_element": ([MATRIX], MATRIX),
    "last_diff_value": ([MATRIX], MATRIX),
    "days_from_last_change": ([MATRIX], MATRIX),
    "hump": ([MATRIX], MATRIX),
    "ts_corr": ([MATRIX, MATRIX], MATRIX),
    "ts_covariance": ([MATRIX, MATRIX], MATRIX),
    "ts_regression": ([MATRIX, MATRIX], MATRIX),
    "ts_step": ([SCALAR], MATRIX),
}

#: Operators that appear in the course material and in some documentation, but are not in
#: this account's operator catalogue. Recording them here means a batch that uses one gets
#: a typed reason ("documented, unavailable") rather than a generic unknown-operator error.
DOCUMENTED_UNAVAILABLE = {
    "vec_choose": ([VECTOR, SCALAR], MATRIX,
                   "documented as a vector-to-matrix selector, absent from /operators"),
    "ts_max": ([MATRIX], MATRIX, "documented in the formulae paper, absent from /operators"),
    "ts_min": ([MATRIX], MATRIX, "documented in the formulae paper, absent from /operators"),
    "floor": ([MATRIX], MATRIX, "documented, absent from /operators"),
}

_number = re.compile(r"^-?\d+(\.\d+)?$")
_field = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_call = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$", re.S)


def split_arguments(text):
    """Split a call's arguments on commas that are not nested inside brackets or quotes."""
    parts, depth, buffer, quoted = [], 0, "", False
    for char in text:
        if char == '"':
            quoted = not quoted
        if not quoted:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append(buffer.strip())
                buffer = ""
                continue
        buffer += char
    if buffer.strip():
        parts.append(buffer.strip())
    return parts


def infer(expression, field_types, _depth=0):
    """Infer the type of an expression, or UNKNOWN when it cannot be told."""
    if _depth > 40:
        return UNKNOWN
    text = (expression or "").strip()
    if not text:
        return UNKNOWN
    if _number.match(text):
        return SCALAR
    if text.startswith('"') and text.endswith('"'):
        return SCALAR
    match = _call.match(text)
    if match:
        name = match.group(1).lower()
        if name in SIGNATURES:
            return SIGNATURES[name][1]
        if name in DOCUMENTED_UNAVAILABLE:
            return DOCUMENTED_UNAVAILABLE[name][1]
        return UNKNOWN
    if _field.match(text):
        declared = (field_types or {}).get(text)
        if declared:
            return str(declared).upper()
        # A bare identifier that is not a field is almost always an assigned variable.
        return UNKNOWN
    # Arithmetic or a comparison: fall back to the first typed piece inside.
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        declared = (field_types or {}).get(token)
        if declared:
            return str(declared).upper()
    return UNKNOWN


def _calls(expression):
    """Yield ``(operator, argument_text, offset)`` for each call, outermost first."""
    text = expression or ""
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        start = match.end() - 1
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    yield match.group(1).lower(), text[start + 1:index], match.start()
                    break


def type_findings(expression, field_types, strict=False):
    """Report arguments whose type the operator cannot accept.

    Each finding names the operator, the argument position, what it wanted and what it
    got, because "operator does not support event inputs" is the message the simulator
    gives and it does not say which argument or which type.
    """
    findings = []
    for operator, arguments, offset in _calls(expression):
        if operator in DOCUMENTED_UNAVAILABLE:
            _params, _ret, note = DOCUMENTED_UNAVAILABLE[operator]
            findings.append({
                "code": "operator_documented_unavailable",
                "severity": "error" if strict else "warning",
                "operator": operator,
                "offset": offset,
                "message": "%s is documented but not in this account's operator "
                           "catalogue (%s)" % (operator, note),
            })
            continue
        signature = SIGNATURES.get(operator)
        if not signature:
            continue
        expected = signature[0]
        for position, raw in enumerate(split_arguments(arguments)):
            if position >= len(expected):
                break
            want = expected[position]
            if want == SCALAR:
                continue
            got = infer(raw, field_types)
            if got in (UNKNOWN, SCALAR):
                continue
            if got != want:
                findings.append({
                    "code": "type_mismatch",
                    "severity": "error",
                    "operator": operator,
                    "position": position,
                    "expected": want,
                    "actual": got,
                    "offset": offset,
                    "message": "%s argument %d expects %s, got %s (%s)"
                               % (operator, position + 1, want, got, raw.strip()[:48]),
                })
    return findings