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

import fastexpr_ast as ast

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
    # Arithmetic signatures require matrix values; event error text is not type evidence.
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



def _typed_program(expression, field_types):
    """Infer from the syntax tree, preserving assignment and reduction order."""
    try:
        program = ast.parse(expression)
    except Exception:
        return UNKNOWN, []
    bindings, findings = {}, []

    def mismatch(operator, position, want, got, node):
        if got in (UNKNOWN, SCALAR) or got == want:
            return
        finding = {"code": "type_mismatch", "severity": "error", "operator": operator,
                   "position": position, "expected": want, "actual": got,
                   "expression": node.render(),
                   "message": "%s argument %d expects %s, got %s" % (operator, position + 1, want, got)}
        if got == VECTOR and want == MATRIX:
            finding["suggestion"] = ("A confirmed VECTOR requires an appropriate vector reduction before this operator; "
                                     "choose the reduction from field semantics and the current operator catalog.")
        findings.append(finding)

    def infer_node(node):
        if isinstance(node, (ast.Number, ast.String)):
            return SCALAR
        if isinstance(node, ast.Field):
            if node.name in bindings:
                return bindings[node.name]
            return str((field_types or {}).get(node.name) or UNKNOWN).upper()
        if isinstance(node, ast.Call):
            values = [infer_node(arg) for arg in node.args]
            for _key, value in node.kwargs:
                infer_node(value)
            signature = SIGNATURES.get(node.name.lower())
            if signature is None:
                return UNKNOWN
            for position, (value, want) in enumerate(zip(values, signature[0])):
                if want != SCALAR:
                    mismatch(node.name, position, want, value, node.args[position])
            return signature[1]
        if isinstance(node, ast.BinOp):
            left, right = infer_node(node.left), infer_node(node.right)
            mismatch(node.op, 0, MATRIX, left, node.left)
            mismatch(node.op, 1, MATRIX, right, node.right)
            return SCALAR if left == right == SCALAR else MATRIX if UNKNOWN not in (left, right) else UNKNOWN
        if isinstance(node, ast.UnaryOp):
            value = infer_node(node.operand)
            mismatch(node.op, 0, MATRIX, value, node.operand)
            return value
        if isinstance(node, ast.Ternary):
            for index, child in enumerate((node.condition, node.when_true, node.when_false)):
                mismatch("conditional", index, MATRIX, infer_node(child), child)
            return MATRIX
        return UNKNOWN

    for name, value in program.assignments:
        bindings[name] = infer_node(value)
    result = infer_node(program.signal) if program.signal is not None else UNKNOWN
    return result, findings


def infer(expression, field_types, _depth=0):
    """Return the AST-derived type, or UNKNOWN without declared evidence."""
    return _typed_program(expression, field_types)[0]


def type_findings(expression, field_types, strict=False):
    """Declared type mismatches are hard errors; unknown types remain unknown."""
    result, findings = _typed_program(expression, field_types)
    if result in (VECTOR, UNIVERSE, GROUP, SYMBOL):
        findings.append({"code": "invalid_signal_type", "severity": "error",
                         "actual": result, "expected": MATRIX,
                         "message": "%s is not a matrix signal" % result})
    return findings