#!/usr/bin/env python3
"""Sugar that compiles down to FASTEXPR.

The point is not brevity. Each sugar form below encodes a *decision* that has
already been paid for:

* ``rrank`` exists because a centred rank and a raw rank differ by a constant,
  which neutralisation erases -- writing ``- 0.5`` by hand invites forgetting it
  exactly where it does matter (before a multiplication).
* ``band`` exists because the offset constant in ``-(abs(r) - c)`` is invisible
  under neutralisation, so the only meaningful parameter is the *shape*. Naming it
  ``band`` stops the width parameter from looking tunable when it is not.
* ``combo`` exists because equal-weight assembly of legs kept getting written by
  hand with per-leg coefficients that drifted.

Everything here expands at parse time into plain FASTEXPR, so what gets simulated
and what gets logged is the expanded string. That is the "post-compile" property:
sugar is a source language, FASTEXPR is the object language, and the registry is
the linker map.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/data/src/magix/WorldQuantUser")

from fastexpr_ast import (  # noqa: E402
    BinOp,
    Call,
    Field,
    Number,
    UnaryOp,
    macro,
    parse,
)

__all__ = ["load", "fs"]


def _num(node, default=None):
    from fastexpr_ast import Number

    if isinstance(node, Number):
        return node.text
    if isinstance(node, UnaryOp) and node.op == "-" and isinstance(node.operand, Number):
        return "-" + node.operand.text
    if default is not None:
        return default
    raise ValueError("expected a numeric literal, got %r" % (node,))


@macro("rrank")
def _rrank(args, kwargs):
    """rrank(x, g) -> group_rank(x, g) - 0.5"""
    if len(args) != 2:
        raise ValueError("rrank(x, group)")
    return BinOp("-", Call("group_rank", (args[0], args[1])), Number("0.5"))


@macro("rrank_cs")
def _rrank_cs(args, kwargs):
    """rrank_cs(x) -> rank(x) - 0.5"""
    if len(args) != 1:
        raise ValueError("rrank_cs(x)")
    return BinOp("-", Call("rank", (args[0],)), Number("0.5"))


@macro("band")
def _band(args, kwargs):
    """band(x, w=0.25) -> -(abs(x) - w); mid is good, constant is cosmetic."""
    if len(args) != 1:
        raise ValueError("band(x, w=0.25)")
    width = _num(kwargs.get("w"), "0.25") if kwargs else "0.25"
    return UnaryOp("-", BinOp("-", Call("abs", (args[0],)), Number(width)))


@macro("trough")
def _trough(args, kwargs):
    """trough(x, w=0.25) -> abs(x) - w; both ends good, the mirror of band."""
    if len(args) != 1:
        raise ValueError("trough(x, w=0.25)")
    width = _num(kwargs.get("w"), "0.25") if kwargs else "0.25"
    return BinOp("-", Call("abs", (args[0],)), Number(width))


def _abs_denominator(node):
    """Abs the denominator unless the caller already did.

    ``abs(abs(x))`` is numerically a no-op but costs an operator instance, and the
    simulator caps the whole expression at 64. Sugar that quietly inflates the
    operator count is worse than no sugar.
    """
    if isinstance(node, Call) and node.name == "abs":
        return node
    return Call("abs", (node,))


@macro("ratio")
def _ratio(args, kwargs):
    """ratio(a, b, eps=0) -> a / (abs(b) + eps); keeps the sign of b out of it."""
    if len(args) != 2:
        raise ValueError("ratio(a, b, eps=0)")
    eps = _num(kwargs.get("eps"), "0") if kwargs else "0"
    denominator = _abs_denominator(args[1])
    if float(eps) != 0:
        denominator = BinOp("+", denominator, Number(eps))
    return BinOp("/", args[0], denominator)


@macro("spread")
def _spread(args, kwargs):
    """spread(hi, lo, eps=0.05) -> (hi - lo) / (abs(mid) + eps), mid from hi/lo."""
    if len(args) != 2:
        raise ValueError("spread(high, low, eps=0.05)")
    eps = _num(kwargs.get("eps"), "0.05") if kwargs else "0.05"
    mid = BinOp("/", BinOp("+", args[0], args[1]), Number("2"))
    denominator = _abs_denominator(mid)
    if float(eps) != 0:
        denominator = BinOp("+", denominator, Number(eps))
    return BinOp("/", BinOp("-", args[0], args[1]), denominator)


@macro("mix")
def _mix(args, kwargs):
    """mix(a, b, w=0.5) -> w*a + (1-w)*b, evaluated at compile time."""
    if len(args) != 2:
        raise ValueError("mix(a, b, w=0.5)")
    weight = _num(kwargs.get("w"), "0.5") if kwargs else "0.5"
    other = str(1 - float(weight))
    return BinOp("+",
                 BinOp("*", Number(weight), args[0]),
                 BinOp("*", Number(other), args[1]))


@macro("combo")
def _combo(args, kwargs):
    """combo(a, b, c, ...) -> equal-weight sum; no coefficients to drift."""
    if not args:
        raise ValueError("combo(leg1, leg2, ...)")
    total = args[0]
    for leg in args[1:]:
        total = BinOp("+", total, leg)
    return BinOp("*", Number(str(round(1 / len(args), 6))), total)


@macro("lag")
def _lag(args, kwargs):
    if len(args) != 2:
        raise ValueError("lag(x, n)")
    return Call("ts_delay", (args[0], args[1]))


@macro("chg")
def _chg(args, kwargs):
    if len(args) != 2:
        raise ValueError("chg(x, n)")
    return Call("ts_delta", (args[0], args[1]))


@macro("avg")
def _avg(args, kwargs):
    if len(args) != 2:
        raise ValueError("avg(x, n)")
    return Call("ts_mean", (args[0], args[1]))


@macro("sd")
def _sd(args, kwargs):
    if len(args) != 2:
        raise ValueError("sd(x, n)")
    return Call("ts_std_dev", (args[0], args[1]))


@macro("gate")
def _gate(args, kwargs):
    """gate(cond, x, else=0) -> if_else(cond, x, else)."""
    if len(args) != 2:
        raise ValueError("gate(cond, x, else=0)")
    fallback = kwargs.get("else", Number("0")) if kwargs else Number("0")
    return Call("if_else", (args[0], args[1], fallback))


@macro("deadband")
def _deadband(args, kwargs):
    """deadband(x, cut=0.02) -> x if |x| > cut else 0, as an if_else.

    Useful for turnover: it stops a leg from trading on values that are inside
    the noise. Expressed as sugar because the naive form (multiplying by a mask)
    silently fails under neutralisation -- a masked zero becomes the group mean's
    mirror image, not a flat position.
    """
    if len(args) != 1:
        raise ValueError("deadband(x, cut=0.02)")
    cut = _num(kwargs.get("cut"), "0.02") if kwargs else "0.02"
    return Call("if_else",
                (BinOp(">", Call("abs", (args[0],)), Number(cut)), args[0], Number("0")))


SUGAR = {
    "rrank", "rrank_cs", "band", "trough", "ratio", "spread", "mix", "combo",
    "lag", "chg", "avg", "sd", "gate", "deadband",
}


def load():
    """Importing this module registers the sugar; returns the names."""
    return sorted(SUGAR)


def fs(expression: str) -> str:
    """Sugar-flavoured source -> plain FASTEXPR (what actually gets simulated)."""
    return parse(expression).render()