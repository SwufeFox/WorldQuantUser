#!/usr/bin/env python3
"""Family trees over alpha expressions, built from structure instead of PnL.

Correlation tells you two alphas move together but costs a request per pair and
says nothing about *why*. The skeleton tells you they are the same expression with
different inputs, which is the answer to "is this a new bet or the same bet again"
and costs nothing.

Three levels, increasingly strict:

``skeleton``   structure only: fields -> ``F``, literals -> ``L``
``fields``     which fields it reads (locals excluded)
``program``    the exact expression

A family is a skeleton. Two members of one family differ only in which field they
point at, so adding a member is a data substitution, not a new hypothesis. That is
the difference between growing a book and growing a filing cabinet.
"""

from __future__ import annotations

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fastexpr_ast as fa  # noqa: E402

__all__ = ["analyze", "report", "lineage", "leg_skeletons", "graft", "swap"]


def analyze(entries):
    """``entries`` is ``[(label, expression), ...]`` -> one row per parseable entry."""
    rows = []
    for label, expression in entries:
        try:
            program = fa.parse(expression)
        except fa.ParseError as exc:
            rows.append({"label": label, "expression": expression,
                         "error": str(exc).splitlines()[0]})
            continue
        rows.append({
            "label": label,
            "expression": expression,
            "program": program,
            "skeleton": fa.skeleton(program),
            "fields": tuple(fa.field_names(program)),
            "operators": tuple(fa.operator_names(program)),
            "nodes": sum(1 for _ in program.walk()),
        })
    return rows


def report(entries, top=10, stream=None):
    """Print family counts. Returns the summary dict so callers can assert on it."""
    rows = analyze(entries)
    ok = [row for row in rows if "error" not in row]
    by_skeleton = collections.defaultdict(list)
    for row in ok:
        by_skeleton[row["skeleton"]].append(row)
    sizes = sorted((len(members) for members in by_skeleton.values()), reverse=True)
    summary = {
        "entries": len(rows),
        "parseable": len(ok),
        "unparseable": len(rows) - len(ok),
        "distinct_skeletons": len(by_skeleton),
        "singletons": sum(1 for size in sizes if size == 1),
        "largest": sizes[0] if sizes else 0,
        "distinct_field_sets": len({row["fields"] for row in ok}),
    }

    stream = stream or sys.stdout
    print("entries %(entries)d  parseable %(parseable)d  unparseable %(unparseable)d"
          % summary, file=stream)
    print("distinct skeletons %(distinct_skeletons)d  singletons %(singletons)d  "
          "largest family %(largest)d  distinct field sets %(distinct_field_sets)d"
          % summary, file=stream)
    for skeleton, members in sorted(by_skeleton.items(),
                                    key=lambda item: -len(item[1]))[:top]:
        fields = {field for member in members for field in member["fields"]}
        print("\n  [%d] %s" % (len(members), skeleton[:110]), file=stream)
        print("      fields: %s" % ", ".join(sorted(fields)[:6]), file=stream)
        print("      e.g. %s" % ", ".join(m["label"] for m in members[:4]), file=stream)
    return summary


def leg_skeletons(program):
    """Skeleton of each assignment, so legs can be compared as sets.

    Whole-program skeletons cannot be nested by substring: the signal segment (a
    bare field reference or a sum of locals) differs between a base and its
    extension, which breaks the containment test. Comparing the *legs* fixes it:
    B extends A when B's leg set is a superset of A's.
    """
    return frozenset(fa.skeleton(fa.Program((), value)) for _name, value in program.assignments)


def lineage(entries):
    """Attach each program to the widest other program whose legs it contains.

    A program with more legs is a descendant of one whose legs it fully reuses,
    which is the "this alpha is that alpha plus a leg" relation. Deliberately
    strict: a descendant must contain *every* leg of the ancestor, so a
    re-parameterised copy is a sibling, not a child.
    """
    rows = [row for row in analyze(entries) if "error" not in row]
    legs = {row["label"]: leg_skeletons(row["program"]) for row in rows}
    parents = {}
    for row in rows:
        candidates = [
            other for other in rows
            if other is not row
            and legs[other["label"]] and legs[other["label"]] < legs[row["label"]]
        ]
        if candidates:
            parents[row["label"]] = max(candidates,
                                        key=lambda other: len(legs[other["label"]]))
    return parents


def graft(base_expression, leg_expression, weight=1.0):
    """Put a leg inside a base program, as a compiled expression.

    Tree surgery, not string concatenation: both sides are parsed, so precedence
    and locals cannot leak across the join.
    """
    base = fa.parse(base_expression)
    leg = fa.parse(leg_expression)
    base_signal = base.signal or fa.Number("0")
    leg_signal = leg.signal or fa.Number("0")
    if leg.assignments:
        leg_signal = fa.Program(leg.assignments, leg_signal).render()
        leg_signal = fa.parse(leg_signal).signal
    combined = fa.BinOp("+", base_signal,
                        fa.BinOp("*", fa.Number(str(weight)), leg_signal))
    return fa.Program(base.assignments, combined).render()


def swap(expression, mapping):
    """Return the same program reading different fields."""
    return fa.map_fields(fa.parse(expression), mapping).render()