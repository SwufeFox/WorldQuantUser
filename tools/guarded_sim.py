#!/usr/bin/env python3
"""Simulate only after checking names against the *cached* catalog.

Why this exists: on 2026-10-30 three separate batches died on names that were
never real (`historical_volatility_252` does not exist; the catalog stops at 180)
and on variables that were declared but unused. Every one of those was catchable
locally. The lint already knew how -- the research scripts simply called
``ml.simulate_alphas`` directly and skipped it.

So the fix is not more rules, it is a single entry point that cannot be skipped:

    from guarded_sim import simulate
    records = simulate(["rank(close)"], {"universe": "TOP500"})

The catalog comes from the on-disk metadata cache and is fetched only when a
scope has never been seen. Nothing here hardcodes an operator or field name.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import expression_lint  # noqa: E402
import field_index  # noqa: E402
import machine_lib as ml  # noqa: E402


class UnknownNameError(RuntimeError):
    """Raised when the expression references a name the catalog does not have."""


def _catalog(session, settings):
    """Return (operator_index, known_fields, field_types) from cache, fetching on miss."""
    operators = ml.fetch_operators(session)
    operator_index = expression_lint.build_operator_index(operators)

    datasets = ml.get_datasets(
        session,
        instrument_type=settings.get("instrumentType", "EQUITY"),
        region=settings.get("region", "USA"),
        delay=settings.get("delay", 1),
        universe=settings.get("universe", "TOP3000"),
    ) or []
    records = []
    for item in datasets:
        dataset_id = item.get("id")
        if not dataset_id:
            continue
        records.extend(ml.get_datafields(
            session,
            instrument_type=settings.get("instrumentType", "EQUITY"),
            region=settings.get("region", "USA"),
            delay=settings.get("delay", 1),
            universe=settings.get("universe", "TOP3000"),
            dataset_id=dataset_id,
        ) or [])
    known = field_index.field_ids(records)
    types = {r.get("id"): r.get("type") for r in records if r.get("id")}
    return operator_index, known, types


def check(expressions, settings=None, session=None):
    """Lint against the live catalog and return the report (no simulation)."""
    settings = dict(settings or {})
    session = session or ml.login()
    operator_index, known, types = _catalog(session, settings)
    pairs = [(expr, settings.get("decay", 4)) for expr in expressions]
    return expression_lint.lint_all(
        pairs,
        operator_index=operator_index,
        denylist=expression_lint.load_denylist(),
        known_fields=known,
        strict_fields=True,
        field_types=types,
        field_units=None,
    )


def simulate(expressions, settings=None, labels=None, force=False, session=None, **kwargs):
    """Lint first, then simulate. ``force=True`` downgrades unknown names to a warning."""
    settings = dict(settings or {})
    session = session or ml.login()
    report = check(expressions, settings, session=session)
    blocking = [e for e in report["errors"]
                if e["code"] in {"unknown_operator", "unknown_identifier",
                                 "unused_variable", "operator_arg_constraint",
                                 "signal_type", "undefined_variable"}]
    if blocking and not force:
        lines = []
        for item in blocking:
            subject = item.get("identifier") or item.get("operator") or item["code"]
            lines.append("  %-14s %s" % (item["code"], subject))
            if item.get("message"):
                lines.append("      " + item["message"])
        raise UnknownNameError(
            "refusing to spend a simulation: %d blocking finding(s)\n%s"
            % (len(blocking), "\n".join(lines))
        )
    return ml.simulate_alphas(
        session,
        expressions,
        settings=settings,
        labels=labels,
        **kwargs,
    )