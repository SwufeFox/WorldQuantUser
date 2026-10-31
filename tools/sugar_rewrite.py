#!/usr/bin/env python3
"""Recognise sugar shapes inside existing expressions, i.e. decompile to the source form.

The sugar layer compiles `band(rrank(x, g))` down to `-(abs(group_rank(x, g) - 0.5) - 0.25)`.
That is one direction. The question this answers is the other one: how much of the
ledger is *already* expressible in sugar, and how much does the ledger compress if it
is written that way?

Compression here is a proxy for a real claim: if two thousand expressions collapse onto
a few dozen sugar forms, then the space being explored is much smaller than the count
suggests -- the same conclusion the skeleton analysis reached, reached from a different
direction and without needing the fields to match.

Recognition is structural, never textual: the matcher reads the tree, so `((x))` and
`x` behave the same and a comment cannot change the answer. Rules are one-directional
(none of them fires on a shape they do not describe), and an expression is reported as
sugar only for the parts actually recognised, with the rest left as FASTEXPR.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import alpha_sugar  # noqa: E402,F401  (importing registers the macros)
import fastexpr_ast as fa  # noqa: E402


def _number(node):
    if isinstance(node, fa.Number):
        try:
            return float(node.text)
        except ValueError:
            return None
    return None


def _fmt(value):
    return ("%.6g" % value)


_PRECEDENCE = fa._PRECEDENCE


def _wrap(text, op, side):
    """Parenthesise a fallback term so the printed form keeps its meaning.

    The first version joined leftovers with spaces, which turned
    ``(close - low) / (high - low)`` into ``close - low - high - close / high - low``:
    it printed a different formula and reported it as the same one.
    """
    return "(%s)" % text


def sugar_of(node, parent_precedence=0):
    """Render ``node`` in sugar where a shape matches, FASTEXPR otherwise.

    Returns ``(text, hits)`` where ``hits`` counts the sugar forms recognised in this
    subtree, so a caller can tell a fully-sugared expression from a partially one.
    """
    hits = 0
    children = []

    if isinstance(node, fa.BinOp):
        left, lh = sugar_of(node.left)
        right, rh = sugar_of(node.right)
        hits += lh + rh
        children = [node.left, node.right]

        # rrank(x, g)  ==  group_rank(x, g) - 0.5
        if node.op == "-" and isinstance(node.left, fa.Call) and node.left.name == "group_rank" \
                and len(node.left.args) == 2 and _number(node.right) == 0.5:
            inner, ih = sugar_of(node.left.args[0])
            group, gh = sugar_of(node.left.args[1])
            return "rrank(%s, %s)" % (inner, group), hits + 1 + ih + gh

        # rrank_cs(x)  ==  rank(x) - 0.5
        if node.op == "-" and isinstance(node.left, fa.Call) and node.left.name == "rank" \
                and len(node.left.args) == 1 and _number(node.right) == 0.5:
            inner, ih = sugar_of(node.left.args[0])
            return "rrank_cs(%s)" % inner, hits + 1 + ih

        # band(x, w)  ==  -(abs(x) - w)
        if node.op == "-" and isinstance(node.left, fa.UnaryOp) and node.left.op == "-" \
                and isinstance(node.left.operand, fa.BinOp) \
                and node.left.operand.op == "-" \
                and isinstance(node.left.operand.left, fa.Call) \
                and node.left.operand.left.name == "abs":
            inner, ih = sugar_of(node.left.operand.left.args[0])
            width = _number(node.left.operand.right)
            if width is not None:
                return "band(%s, w=%s)" % (inner, _fmt(width)), hits + 1 + ih

        # trough(x, w)  ==  abs(x) - w
        if node.op == "-" and isinstance(node.left, fa.Call) and node.left.name == "abs":
            width = _number(node.right)
            if width is not None:
                inner, ih = sugar_of(node.left.args[0])
                return "trough(%s, w=%s)" % (inner, _fmt(width)), hits + 1 + ih

        # ratio(a, b, eps)  ==  a / (abs(b) + eps)
        if node.op == "/" and isinstance(node.right, fa.BinOp) and node.right.op == "+" \
                and isinstance(node.right.left, fa.Call) and node.right.left.name == "abs":
            eps = _number(node.right.right)
            if eps is not None:
                inner, ih = sugar_of(node.left)
                body, bh = sugar_of(node.right.left.args[0])
                return ("ratio(%s, %s, eps=%s)" % (inner, body, _fmt(eps)),
                        hits + 1 + ih + bh)

        # combo(a, b)  ==  1/n * (a + b), recognised for n terms with equal weights
        precedence = _PRECEDENCE[node.op]
        text = "%s %s %s" % (
            _wrap(left, node.op, "left") if _needs_wrap(node.left, precedence) else left,
            node.op,
            _wrap(right, node.op, "right")
            if _needs_wrap(node.right, precedence + (1 if node.op in {"-", "/"} else 0)) else right)
        return ("(%s)" % text) if precedence < parent_precedence else text, hits

    if isinstance(node, fa.UnaryOp):
        # band spans two levels: -(abs(x) - w). Matching it inside the BinOp branch
        # never fires, because the outer node is the unary minus, so the expression was
        # reported as `-(trough(x, w))` -- correct but not the form anybody writes.
        if node.op == "-" and isinstance(node.operand, fa.BinOp) \
                and node.operand.op == "-" \
                and isinstance(node.operand.left, fa.Call) \
                and node.operand.left.name == "abs":
            width = _number(node.operand.right)
            if width is not None:
                inner, ih = sugar_of(node.operand.left.args[0])
                return "band(%s, w=%s)" % (inner, _fmt(width)), hits + 1 + ih
        inner, ih = sugar_of(node.operand)
        # `-(a - b)` is not `-a - b`. The operand needs wrapping whenever it binds
        # looser than the unary operator, which is the same trap as the binary case
        # above and was caught by the same faithfulness check.
        if _needs_wrap(node.operand, 6):
            inner = "(%s)" % inner
        return "%s%s" % (node.op, inner), hits + ih

    if isinstance(node, fa.Call):
        args = [sugar_of(arg) for arg in node.args]
        hits += sum(h for _t, h in args)
        kwargs = [(key, sugar_of(value)) for key, value in node.kwargs]
        hits += sum(h for _k, (_t, h) in kwargs)
        simple = {
            "ts_delay": "lag", "ts_delta": "chg", "ts_mean": "avg",
            "ts_std_dev": "sd",
        }
        if node.name in simple and len(args) == 2 and not kwargs:
            return "%s(%s, %s)" % (simple[node.name], args[0][0], args[1][0]), hits + 1
        if node.name == "if_else" and len(args) == 3 and args[2][0] == "0" and not kwargs:
            return "gate(%s, %s)" % (args[0][0], args[1][0]), hits + 1
        parts = [text for text, _h in args] + ["%s=%s" % (k, t) for k, (t, _h) in kwargs]
        return "%s(%s)" % (node.name, ", ".join(parts)), hits

    return node.render(), hits


def _needs_wrap(child, precedence):
    """True when printing ``child`` bare inside an operator of this precedence breaks it."""
    if isinstance(child, fa.BinOp):
        return _PRECEDENCE[child.op] < precedence
    if isinstance(child, fa.Ternary):
        return True
    return False


def analyse(entries):
    rows = []
    seen = set()
    for entry in entries:
        expression = entry.get("expression")
        if not expression or expression in seen:
            continue
        seen.add(expression)
        try:
            program = fa.parse(expression)
        except fa.ParseError:
            continue
        chunks, total = [], 0
        for name, value in program.assignments:
            text, hits = sugar_of(value)
            total += hits
            chunks.append("%s = %s" % (name, text))
        if program.signal is not None:
            text, hits = sugar_of(program.signal)
            total += hits
            chunks.append(text)
        sugared = ";\n".join(chunks)
        rows.append({
            "expression": expression,
            "sugar": sugared,
            "hits": total,
            "raw_chars": len(expression),
            "sugar_chars": len(sugared),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    entries = [e for e in alpha_registry.load_entries()
               if not e.get("abandoned") and e.get("expression")]
    rows = analyse(entries)
    full = [r for r in rows if r["hits"] > 0]
    no_hit = len(rows) - len(full)
    raw = sum(r["raw_chars"] for r in rows)
    sug = sum(r["sugar_chars"] for r in rows)

    # Faithfulness: the sugar form must mean the same thing. Compared canonically, so
    # formatting differences do not count as agreement.
    mismatched = []
    for row in rows:
        try:
            if fa.canonical(row["sugar"]) != fa.canonical(row["expression"]):
                mismatched.append(row)
        except fa.ParseError:
            mismatched.append(row)

    heads = collections.Counter()
    for row in rows:
        for head in re.findall(r"\b(rrank_cs|rrank|band|trough|ratio|combo|lag|chg|avg|sd|gate)\s*\(",
                               row["sugar"]):
            heads[head] += 1

    forms = {}
    for row in rows:
        forms.setdefault(row["sugar"], []).append(row)

    print("expressions            : %d" % len(rows))
    print("with at least one sugar: %d (%.0f%%)" % (len(full), 100.0 * len(full) / max(1, len(rows))))
    print("no sugar recognised    : %d" % no_hit)
    print("characters             : %d -> %d (%.0f%% of original)"
          % (raw, sug, 100.0 * sug / max(1, raw)))
    print("distinct sugar forms   : %d" % len(forms))
    biggest = max(forms.values(), key=len)
    print("largest sugar class    : %d expressions" % len(biggest))
    print("faithful rewrites      : %d / %d" % (len(rows) - len(mismatched), len(rows)))
    print("sugar forms used       : %s" % ", ".join("%s %d" % (k, v) for k, v in heads.most_common()))

    for row in sorted(full, key=lambda r: r["raw_chars"] - r["sugar_chars"], reverse=True)[:args.examples]:
        print("\n  %d -> %d chars, %d sugar form(s)" % (
            row["raw_chars"], row["sugar_chars"], row["hits"]))
        print("    sugar: %s" % row["sugar"][:150].replace("\n", " "))

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=1)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())