#!/usr/bin/env python3
"""Detect and rewrite the "ranking a level instead of a ratio" mistake, on the tree.

The pattern is easy to describe and easy to miss: a raw accounting amount is
dominated by company size across the cross-section, so ``group_rank(field, g)`` on
that field mostly ranks size. Dividing by a size measure first leaves the structure
of the business, which is what the idea was about.

Doing this by string search is what produced the original problem: the same shape
appears as ``group_rank(x, subindustry)``, ``group_rank((x), subindustry)`` and
inside a sum, and a rename turns the search into a no-op that still reports success.
On the tree the check is structural -- "is the whole argument one bare field, and is
that field a level" -- so it cannot half-match.

Two guards keep it from guessing:

* the field must be in the curated ``level_fields`` list (below), because a field
  whose unit was never measured cannot be classified by name without inventing
  knowledge; and
* the rewrite is only *offered*, never applied silently. ``plan`` returns the
  variant plus a one-line reason, and the caller decides.

The size measure is a parameter, not a policy: ``atq`` (total assets) is the default
because it is available for every filer and is already used as the denominator in
this account's working composites; ``cap`` gives a market-based version and is
accepted just as readily.
"""

from __future__ import annotations

import json
import os

import fastexpr_ast as fa

__all__ = ["load_rules", "level_ranks", "apply_size", "plan"]

_RULES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intent-rules.json")


def load_rules(path=None):
    """Read the curated level/size lists; a missing file means "rewrite nothing"."""

    try:
        with open(path or _RULES) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {"level_fields": [], "rank_operators": {}, "size_fields": {}}


def _is_rank_call(node, rank_operators):
    return isinstance(node, fa.Call) and node.name in rank_operators


def level_ranks(program, rules=None):
    """Find ``rank(bare_level_field)`` calls, i.e. the pattern worth rewriting.

    Returns ``[(call, field_name), ...]`` so the caller can see exactly which call
    fired, not just that something did.
    """
    rules = rules or load_rules()
    levels = set(rules.get("level_fields") or [])
    rank_operators = rules.get("rank_operators") or {}
    found = []
    for node in program.walk():
        if not _is_rank_call(node, rank_operators):
            continue
        wanted = rank_operators.get(node.name) or []
        for index, argument in enumerate(node.args):
            if wanted and index not in wanted:
                continue
            if isinstance(argument, fa.Field) and argument.name in levels:
                found.append((node, argument.name))
    return found


def _replace_rank_argument(node, field_name, replacement):
    """Rebuild a rank call with one argument swapped, keeping the others."""
    if isinstance(node, fa.Call) and node.name:
        args = []
        swapped = False
        for argument in node.args:
            if not swapped and isinstance(argument, fa.Field) and argument.name == field_name:
                args.append(replacement)
                swapped = True
            else:
                args.append(argument)
        result = fa.Call(node.name, tuple(args), node.kwargs)
        return result
    return node


def apply_size(program, size_field, rules=None):
    """Rewrite every level-rank in the program to divide by ``size_field``."""
    rules = rules or load_rules()
    found = level_ranks(program, rules)
    if not found:
        return program
    names = {field for _call, field in found}
    tree = program.signal
    for _call, field in found:
        tree = _walk_and_rewrite(tree, field, size_field)
    assignments = []
    for key, value in program.assignments:
        for field in names:
            value = _walk_and_rewrite(value, field, size_field)
        assignments.append((key, value))
    return fa.Program(tuple(assignments), tree, program.annotations)


def _walk_and_rewrite(node, field, size_field):
    if isinstance(node, fa.Call):
        rebuilt = _replace_rank_argument(node, field,
                                         fa.BinOp("/", fa.Field(field),
                                                  fa.Call("ts_backfill",
                                                          (fa.Field(size_field),
                                                           fa.Number("252")))))
        return fa.Call(rebuilt.name,
                       tuple(_walk_and_rewrite(a, field, size_field) for a in rebuilt.args),
                       tuple((k, _walk_and_rewrite(v, field, size_field))
                             for k, v in rebuilt.kwargs))
    if isinstance(node, fa.BinOp):
        return fa.BinOp(node.op, _walk_and_rewrite(node.left, field, size_field),
                        _walk_and_rewrite(node.right, field, size_field))
    if isinstance(node, fa.UnaryOp):
        return fa.UnaryOp(node.op, _walk_and_rewrite(node.operand, field, size_field))
    if isinstance(node, fa.Ternary):
        return fa.Ternary(_walk_and_rewrite(node.condition, field, size_field),
                          _walk_and_rewrite(node.when_true, field, size_field),
                          _walk_and_rewrite(node.when_false, field, size_field))
    return node


def plan(program, size_field="fnd6_cptnewqv1300_atq", rules=None):
    """Offer the rewrite with a reason, or explain why there is nothing to do."""
    rules = rules or load_rules()
    found = level_ranks(program, rules)
    if not found:
        return []
    fields = sorted({field for _call, field in found})
    rewritten = apply_size(program, size_field, rules)
    return [{
        "kind": "size_normalisation",
        "fields": fields,
        "size_field": size_field,
        "reason": ("%s ranked as a level; across the cross-section that mostly "
                   "ranks company size. Dividing by %s leaves the composition."
                   % (", ".join(fields), size_field)),
        "before": program.render(),
        "after": rewritten.render(),
        "program": rewritten,
    }]