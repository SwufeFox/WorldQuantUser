#!/usr/bin/env python3
"""Leave-one-out and solo variants for a composite alpha expression.

Why this exists: every time a leg was added, the question "did that help?" was
answered by hand-writing the variant expressions, running a batch, and eyeballing
numbers from different files. That is slow and it produced at least one wrong
conclusion, because a five-leg version scoring below the four-leg version was
only visible after two separate batches.

The rule this enforces is narrow on purpose. Variants are generated only when the
combination is a plain sum of named legs:

    a = ...; n = ...; c = ...; a + n + c

Anything else — a product, a gated leg like ``r*(1-rank(cap))``, a nested
expression — is refused with an explanation. Guessing what "remove a leg" means
from syntax would produce a variant that is not the experiment anyone intended,
and a silently wrong variant is worse than no variant: it consumes quota and
returns a number that looks like evidence.

Every variant carries a ``candidate_id`` so the contribution table can map results
back to legs without relying on string matching against expressions.
"""

from __future__ import annotations

import re

__all__ = ["AblationError", "parse", "variants", "report", "MODES"]

MODES = ("full", "loo", "solo")

_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", re.S)


class AblationError(ValueError):
    """The expression cannot be ablated, with the reason and the way out."""


def parse(expression):
    """Split a composite into ``{"legs", "order", "combination"}``.

    Raises :class:`AblationError` when the final term is not a plain sum of the
    names declared before it.
    """
    text = (expression or "").strip()
    if not text:
        raise AblationError("expression is empty")

    parts = [part.strip() for part in text.split(";") if part.strip()]
    if len(parts) < 2:
        raise AblationError(
            "no legs to ablate: the expression has no 'name = ...; ...' structure")

    legs = {}
    order = []
    for part in parts[:-1]:
        match = _ASSIGNMENT.match(part)
        if not match:
            raise AblationError(
                "every segment before the last must be 'name = expression'; "
                "got %r" % part[:80])
        name, body = match.group(1), match.group(2).strip()
        if name in legs:
            raise AblationError("duplicate leg name %r" % name)
        legs[name] = body
        order.append(name)

    combination = parts[-1]
    tokens = [token.strip() for token in combination.split("+")]
    if len(tokens) < 2:
        raise AblationError(
            "the last segment must be a sum of legs such as 'a + b'; got %r. "
            "If the combination is a product or is gated, declare the legs "
            "explicitly with --legs and ablate by hand" % combination[:80])
    unknown = [token for token in tokens if token not in legs]
    if unknown:
        raise AblationError(
            "the last segment is not a plain sum of declared legs: %s. "
            "A product, a function call or a constant in the combination cannot "
            "be removed syntactically" % ", ".join(sorted(set(unknown))))

    if len(tokens) != len(set(tokens)):
        raise AblationError("the combination repeats a leg; remove the duplicate first")

    unused = [name for name in order if name not in tokens]
    return {
        "legs": legs,
        "order": order,
        "combination": combination,
        "used": tokens,
        "unused": unused,
    }


def _rebuild(parsed, final_term):
    head = ["%s = %s" % (name, parsed["legs"][name]) for name in parsed["order"]]
    return "; ".join(head + [final_term])


def variants(expression, modes=MODES):
    """Return the variant list, each with a stable ``candidate_id``.

    ``full`` is always included and always first, because a contribution is only
    meaningful next to the thing it is a contribution to.
    """
    parsed = parse(expression)
    wanted = tuple(modes or MODES)
    for mode in wanted:
        if mode not in MODES:
            raise AblationError("unknown mode %r; choose from %s"
                                % (mode, ", ".join(MODES)))

    used = parsed["used"]
    out = []
    if "full" in wanted:
        out.append({
            "candidate_id": "full",
            "mode": "full",
            "expression": _rebuild(parsed, " + ".join(used)),
            "legs": list(used),
        })
    if "loo" in wanted and len(used) > 1:
        for name in used:
            rest = [other for other in used if other != name]
            out.append({
                "candidate_id": "loo:" + name,
                "mode": "loo",
                "expression": _rebuild(parsed, " + ".join(rest)),
                "legs": rest,
                "removed": name,
            })
    if "solo" in wanted:
        for name in used:
            out.append({
                "candidate_id": "solo:" + name,
                "mode": "solo",
                "expression": _rebuild(parsed, name),
                "legs": [name],
                "solo": name,
            })
    return out


def _fitness(record):
    metrics = record.get("metrics") or {}
    value = metrics.get("fitness")
    return value if isinstance(value, (int, float)) else None


def report(records):
    """Turn simulation records into per-leg contributions.

    ``contribution`` is ``fitness(full) - fitness(full without this leg)``, so a
    positive number means the leg earns its place and a negative one means the
    composite would score higher without it. ``solo`` is the leg alone, which
    separates a driver from a carrier: a carrier can add to a composite while
    scoring weakly on its own.
    """
    by_id = {}
    for record in records or []:
        if not isinstance(record, dict):
            continue
        name = record.get("candidate_id")
        if name and name not in by_id:
            by_id[name] = record

    full = by_id.get("full")
    full_fitness = _fitness(full) if full else None
    legs = []
    for name in sorted(by_id):
        if name.startswith("solo:"):
            leg = name.split(":", 1)[1]
            if any(row["leg"] == leg for row in legs):
                continue
            legs.append({"leg": leg, "solo": _fitness(by_id[name])})
    for row in legs:
        loo = by_id.get("loo:" + row["leg"])
        row["without"] = _fitness(loo) if loo else None
        if full_fitness is not None and row["without"] is not None:
            row["contribution"] = round(full_fitness - row["without"], 4)
        else:
            row["contribution"] = None
        if row["contribution"] is None:
            row["verdict"] = "unmeasured"
        elif row["contribution"] > 0.02:
            row["verdict"] = "adds"
        elif row["contribution"] < -0.02:
            row["verdict"] = "hurts"
        else:
            row["verdict"] = "neutral"

    hurts = [row["leg"] for row in legs if row["verdict"] == "hurts"]
    return {
        "full_fitness": full_fitness,
        "full_alpha_id": (full or {}).get("alpha_id"),
        "legs": legs,
        "hurts": hurts,
        "measured": sum(1 for row in legs if row["contribution"] is not None),
    }