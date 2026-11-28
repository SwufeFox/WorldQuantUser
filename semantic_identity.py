"""Semantic identity for FASTEXPR expressions.

Two expressions share an identity when the platform would read them as the same
signal: commutative arguments compare unordered, a positive scalar factor on the
root is scale rather than signal, a negative factor is direction, and a constant
additive offset disappears when the neutralisation demeans the book anyway.

This is deliberately conservative. It normalises the root only - nested constants
stay where they are - because a nested offset can still change what an operator
sees (a percentile, a threshold, a partition).
"""

from __future__ import annotations

import hashlib

import fastexpr_ast as ast

#: Operators whose arguments compare unordered.
COMMUTATIVE = frozenset({"add", "multiply", "max", "min"})

#: Neutralisation settings that leave a constant offset in the final weights.
UNTILTED = frozenset({"", "NONE"})


def _number(node):
    """Numeric value of a literal, including one written with a unary sign."""
    sign = 1
    while isinstance(node, ast.UnaryOp) and node.op in {"-", "+"}:
        if node.op == "-":
            sign = -sign
        node = node.operand
    if isinstance(node, ast.Number):
        try:
            return sign * float(node.text)
        except ValueError:
            return None
    return None


def _canonical(node):
    """Render a node with commutative arguments ordered and keywords sorted."""
    if isinstance(node, ast.Call):
        args = [_canonical(arg) for arg in node.args]
        if node.name in COMMUTATIVE:
            args.sort()
        keywords = sorted((key, _canonical(value)) for key, value in node.kwargs)
        parts = args + ["%s=%s" % (key, value) for key, value in keywords]
        return "%s(%s)" % (node.name, ", ".join(parts))
    if isinstance(node, ast.BinOp):
        left, right = _canonical(node.left), _canonical(node.right)
        if node.op in {"+", "*"} and left > right:
            left, right = right, left
        return "(%s%s%s)" % (left, node.op, right)
    if isinstance(node, ast.UnaryOp):
        return "(%s%s)" % (node.op, _canonical(node.operand))
    return node.render()


def _demeaning(neutralization):
    return str(neutralization or "").upper() not in UNTILTED


def _root_sign_and_core(node, demeaned):
    """Strip root scale and (when demeaning) a constant offset, keeping direction."""
    sign = 1
    while True:
        if isinstance(node, ast.UnaryOp) and node.op == "-":
            sign = -sign
            node = node.operand
            continue
        if not isinstance(node, ast.BinOp):
            break
        left_value, right_value = _number(node.left), _number(node.right)
        if node.op in {"+", "-"} and demeaned:
            if right_value is not None and left_value is None:
                node = node.left
                continue
            if left_value is not None and right_value is None:
                node = node.right
                continue
        if node.op == "*":
            if right_value is not None and left_value is None:
                factor, node = right_value, node.left
            elif left_value is not None and right_value is None:
                factor, node = left_value, node.right
            else:
                factor = None
            if factor is not None:
                if factor < 0:
                    sign = -sign
                if factor == 0:
                    return 1, "0"
                continue
        if node.op == "/" and right_value is not None and right_value > 0 and left_value is None:
            node = node.left
            continue
        break
    return sign, _canonical(node)


def identity(expression, *, neutralization="INDUSTRY"):
    """Return the canonical identity string for one expression."""
    program = ast.parse(expression)
    signal = program.signal
    if signal is None:
        raise ValueError("expression has no signal")
    sign, core = _root_sign_and_core(signal, _demeaning(neutralization))
    return core if sign > 0 else "-" + core


def fingerprint(expression, *, neutralization="INDUSTRY"):
    """Return the SHA-256 of the canonical identity."""
    return hashlib.sha256(identity(expression, neutralization=neutralization).encode("utf-8")).hexdigest()


def same_identity(left, right, *, neutralization="INDUSTRY"):
    """True when both expressions share one identity under the given neutralisation."""
    return identity(left, neutralization=neutralization) == identity(right, neutralization=neutralization)
