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


def _terms(node):
    if isinstance(node, fa.BinOp) and node.op == "+":
        return _terms(node.left) + _terms(node.right)
    return [node]


def _references(node, name):
    return any(isinstance(child, fa.Field) and child.name == name for child in node.walk())


def _coefficient(term):
    if isinstance(term, fa.BinOp) and term.op == "*" and isinstance(term.left, fa.Number):
        try:
            return float(term.left.text), term.right
        except ValueError:
            return None, term
    return 1.0, term


def drop_term(program, name):
    """Remove the terms that reference ``name`` and renormalise the rest.

    Returns ``None`` when the signal is not a flat sum or when dropping would leave
    nothing -- inventing a fallback there would produce an ablation that measures a
    different construction.
    """
    if program.signal is None:
        return None
    terms = _terms(program.signal)
    if len(terms) < 2:
        return None
    kept = [term for term in terms if not _references(term, name)]
    if not kept or len(kept) == len(terms):
        return None
    total = 0.0
    weights = []
    for term in kept:
        coefficient, body = _coefficient(term)
        if coefficient is None:
            return None
        weights.append((coefficient, body))
        total += coefficient
    if total <= 0:
        return None
    rebuilt = None
    for coefficient, body in weights:
        scaled = fa.BinOp("*", fa.Number(str(round(coefficient / total, 6))), body)
        rebuilt = scaled if rebuilt is None else fa.BinOp("+", rebuilt, scaled)
    return fa.Program(program.assignments, rebuilt, program.annotations)


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