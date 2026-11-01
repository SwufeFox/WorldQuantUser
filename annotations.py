#!/usr/bin/env python3
"""Read the annotations on a program, check them, and derive ablations from them.

Annotations record intent that the simulator never sees: which leg is the carrier,
what mechanism the composite is supposed to be, and which leg is worth removing to
test whether it earns its place. Because they are metadata, they can go stale
silently -- the leg gets renamed, the annotation keeps pointing at a name that no
longer exists, and the ablation it requests quietly becomes a no-op. Checking that
is the point of :func:`check`.

Two operations, deliberately different:

``zero_leg``   keep the assignment, set its value to 0. The book keeps its shape
               and the leg stops contributing information. This isolates "does
               this leg's *content* matter" from "does this leg's *weight* matter".
``drop_term``  remove the term from the signal and renormalise the remaining
               coefficients. This answers the cruder question "what if we never
               had this leg", which is what a real ablation asks.

The vocabulary lives in ``annotation-schema.json`` next to this module, so adding a
kind of annotation is a data edit and an unknown name is reported instead of being
ignored.
"""

from __future__ import annotations

import json
import os

import fastexpr_ast as fa

__all__ = ["load_schema", "check", "derive_ablations", "zero_leg", "drop_term",
           "annotations_of"]

_SCHEMA_NAME = "annotation-schema.json"


def load_schema(path=None):
    """Read the annotation vocabulary; an unreadable file means "no vocabulary"."""

    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), _SCHEMA_NAME)
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {"annotations": {}}


def annotations_of(program):
    """``[(name, args, kwargs), ...]`` with rendered strings, for easy comparison."""
    out = []
    for note in program.annotations:
        out.append((note.name,
                    [value.render() for value in note.args],
                    {key: value.render() for key, value in note.kwargs}))
    return out


def _string_value(raw):
    """Annotation arguments are literal strings or numbers; unwrap the quotes."""
    if raw is None:
        return None
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def check(program, schema=None, known_fields=None):
    """Return findings: unknown annotation, missing required argument, stale target.

    Staleness is decided structurally: an annotation that names a leg must name one
    that exists in this program. A name that has drifted is the common failure, and
    it is invisible without this check because the annotation still *renders*.
    """
    schema = schema or load_schema()
    vocabulary = schema.get("annotations", {})
    assigned = {name for name, _value in program.assignments}
    findings = []

    for name, args, kwargs in annotations_of(program):
        spec = vocabulary.get(name)
        if spec is None:
            findings.append({"code": "unknown_annotation", "annotation": name,
                             "message": "@%s is not in the annotation schema" % name})
            continue
        for required in spec.get("requires", []):
            if required not in kwargs and not args:
                findings.append({"code": "missing_argument", "annotation": name,
                                 "message": "@%s requires %r" % (name, required)})
        leg_key = spec.get("names_a_leg")
        if leg_key and leg_key in kwargs:
            target = _string_value(kwargs[leg_key])
            if target not in assigned:
                findings.append({
                    "code": "stale_annotation",
                    "annotation": name,
                    "target": target,
                    "message": "@%s points at leg %r, which this program does not "
                               "define (it defines %s)"
                               % (name, target, ", ".join(sorted(assigned)) or "none"),
                })
        field_key = spec.get("names_a_field")
        if field_key and field_key in kwargs and known_fields is not None:
            target = _string_value(kwargs[field_key])
            if target not in known_fields:
                findings.append({"code": "stale_field", "annotation": name,
                                 "target": target,
                                 "message": "@%s names field %r, which is not in the "
                                            "catalogue" % (name, target)})
    return findings


def zero_leg(program, name):
    """Neutralise one leg's information while keeping every weight in place."""
    assignments = tuple((key, fa.Number("0") if key == name else value)
                        for key, value in program.assignments)
    return fa.Program(assignments, program.signal, program.annotations)


def _terms(node, sign=1.0):
    """Signed additive decomposition: ``[(coefficient, body), ...]``.

    Splits ``+``, ``-`` and unary minus, and pulls a numeric factor out of a
    product, so the coefficient is what renormalisation needs and the body is what
    "does this term reference the leg" needs.

    This used to split only ``+``, which quietly made ``drop_term`` a no-op for
    every composite written with subtraction -- and most composites are. A leg that
    is never dropped is invisible in the numbers: the ablation table simply has one
    fewer row and looks complete.
    """
    if isinstance(node, fa.BinOp) and node.op == "+":
        return _terms(node.left, sign) + _terms(node.right, sign)
    if isinstance(node, fa.BinOp) and node.op == "-":
        return _terms(node.left, sign) + _terms(node.right, -sign)
    if isinstance(node, fa.UnaryOp) and node.op == "-":
        return _terms(node.operand, -sign)
    if isinstance(node, fa.BinOp) and node.op == "*":
        for side in (0, 1):
            value = _numeric(node.left if side == 0 else node.right)
            if value is not None:
                return [(sign * value, node.right if side == 0 else node.left)]
    return [(sign, node)]


def _numeric(node):
    """The value of a bare numeric factor, including ``-1`` written as unary minus.

    ``-1 * a`` parses as ``UnaryOp('-', 1) * a``, not as ``Number('-1') * a``, so a
    check for ``Number`` alone misses the sign and treats the whole product as one
    unsplittable term.
    """
    if isinstance(node, fa.Number):
        try:
            return float(node.text)
        except ValueError:
            return None
    if isinstance(node, fa.UnaryOp) and node.op == "-":
        inner = _numeric(node.operand)
        if inner is not None:
            return -inner
    return None


def _references(node, name):
    return any(isinstance(child, fa.Field) and child.name == name for child in node.walk())


def _prune_assignments(assignments, signal):
    """Drop assignments nothing reads, transitively.

    A dropped leg keeps its assignment under the old behaviour, so the variable stays
    declared and unread -- and the simulator rejects the whole expression for that
    alone. The ablation is right and the program is invalid, which is the worst of
    both: it looks like a result until you try to run it.
    """
    alive = [(key, value) for key, value in assignments]
    changed = True
    while changed:
        changed = False
        used = set()
        for node in (signal.walk() if signal is not None else []):
            if isinstance(node, fa.Field):
                used.add(node.name)
        kept = []
        for key, value in alive:
            if key in used or any(isinstance(node, fa.Field) and node.name == key
                                  for other_key, other_value in alive
                                  if other_key != key for node in other_value.walk()):
                kept.append((key, value))
            else:
                changed = True
        alive = kept
    return tuple(alive)


def drop_term(program, name):
    """Remove the terms that reference ``name`` and renormalise the rest.

    Returns ``None`` when the signal is not a flat sum or when dropping would leave
    nothing -- inventing a fallback there would produce an ablation that measures a
    different construction.

    The kept coefficients are divided by the **absolute** value of their sum. Using
    the signed sum made every net-short composite undroppable: a signal that is
    ``-l0 - l1 - 0.5*l2`` sums to a negative number and the guard rejected it,
    although the ablation is perfectly well defined.
    """
    if program.signal is None:
        return None
    terms = _terms(program.signal)
    if len(terms) < 2:
        return None
    kept = [(coefficient, body) for coefficient, body in terms
            if not _references(body, name)]
    if not kept or len(kept) == len(terms):
        return None
    total = sum(coefficient for coefficient, _body in kept)
    if total == 0:
        return None
    scale = abs(total)
    rebuilt = None
    for coefficient, body in kept:
        scaled = fa.BinOp("*", fa.Number(str(round(coefficient / scale, 6))), body)
        rebuilt = scaled if rebuilt is None else fa.BinOp("+", rebuilt, scaled)
    return fa.Program(_prune_assignments(program.assignments, rebuilt), rebuilt,
                      program.annotations)


def derive_ablations(program, schema=None):
    """Every ablation this program's structure and annotations ask for.

    Explicit ``@ablate(leg=...)`` requests win when present; otherwise each leg is
    ablated, because a composite leg that nobody annotated is exactly the case where
    "does it earn its place" has never been asked.
    """
    schema = schema or load_schema()
    vocabulary = schema.get("annotations", {})
    requested = []
    for name, _args, kwargs in annotations_of(program):
        spec = vocabulary.get(name) or {}
        if spec.get("kind") == "ablation" and "leg" in kwargs:
            requested.append(_string_value(kwargs["leg"]))
    targets = requested or [name for name, _value in program.assignments]

    variants = []
    for target in targets:
        zeroed = zero_leg(program, target)
        variants.append({"label": "zero:%s" % target, "kind": "zero_leg",
                         "leg": target, "program": zeroed,
                         "expression": zeroed.render()})
        dropped = drop_term(program, target)
        if dropped is not None:
            variants.append({"label": "drop:%s" % target, "kind": "drop_term",
                             "leg": target, "program": dropped,
                             "expression": dropped.render()})
    return variants