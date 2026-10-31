#!/usr/bin/env python3
"""Say *why* a simulation was rejected, not just that it was.

`failing_checks` reports which gate failed, and that is not the same thing. On
2026-10-31 the ledger held 1513 rejected rows and 1119 of them carried the single
label `LOW_FITNESS`. Answering "how many of those were low amplitude rather than
excess turnover" was impossible, because the label cannot distinguish them and the
difference decides what to change.

The decomposition comes from the platform's own formula:

    Fitness = Sharpe * sqrt(|returns| / max(turnover, 0.125))

so a low Fitness has exactly three independent causes on this account:

``turnover_excess``   turnover above the 0.125 floor and the signal would clear 1.0
                      if it traded less. The signal is fine; the trading is not.
``low_amplitude``     turnover at or below the floor, so the denominator is pinned and
                      the only lever left is returns.
``weak_signal``       Sharpe itself below 1.0, so no amount of return or turnover work
                      saves it.

Everything else is a named gate and is reported as itself. ``unclassified`` is a real
answer too: rows without metrics, and rows whose failure this module does not model.
"""

from __future__ import annotations

__all__ = ["classify", "FITNESS_FLOOR", "TURNOVER_FLOOR"]

FITNESS_FLOOR = 1.0
TURNOVER_FLOOR = 0.125


def _fitness(sharpe, returns, turnover):
    if sharpe is None or returns is None or turnover is None:
        return None
    return sharpe * (abs(returns) / max(turnover, TURNOVER_FLOOR)) ** 0.5


def classify(entry):
    """Return ``(code, detail)`` for one ledger row.

    ``code`` is a stable slug suitable for counting; ``detail`` is a short sentence
    for a human. The function never guesses: missing metrics give ``unclassified``.
    """
    checks = entry.get("failing_checks") or []
    if isinstance(checks, str):
        checks = [checks]
    checks = set(checks)
    metrics = entry.get("metrics") or {}
    sharpe = metrics.get("sharpe")
    fitness = metrics.get("fitness")
    turnover = metrics.get("turnover")
    returns = metrics.get("returns")

    if "CONCENTRATED_WEIGHT" in checks:
        return "weight_concentration", (
            "the return comes from a few oversized positions; bounding the weights "
            "collapses the score rather than fixing the check")
    if "LOW_TURNOVER" in checks:
        return "turnover_floor", (
            "a stock variable: it barely moves, so it cannot be traded at all")
    if "HIGH_TURNOVER" in checks:
        return "turnover_high", "trading too fast to survive the fitness denominator"
    if "SELF_CORRELATION" in checks:
        return "crowded", "too close to an alpha already in the book"
    # Weak signal is a statement about the metric, not about which gate tripped: a
    # Sharpe below 1.0 cannot be rescued by trading slower or by more return, so it
    # outranks the decomposition below regardless of how the gates were labelled.
    if sharpe is not None and sharpe < 1.0:
        return "weak_signal", (
            "Sharpe %.2f is below 1.0: neither returns nor turnover work saves it" % sharpe)

    if "LOW_SUB_UNIVERSE_SHARPE" in checks and not checks & {"LOW_SHARPE", "LOW_FITNESS"}:
        return "fragile_subuniverse", (
            "works in the full cross-section but not in the sub-universe, so it "
            "depends on names the book cannot rely on")
    if "LOW_FITNESS" in checks:
        if None in (sharpe, returns, turnover):
            return "unclassified", "fitness failed but the metrics needed to decompose it are missing"
        capped = _fitness(sharpe, returns, TURNOVER_FLOOR)
        if turnover > TURNOVER_FLOOR and capped is not None and capped >= FITNESS_FLOOR:
            return "turnover_excess", (
                "would clear %.2f at the turnover floor (%.2f); the trading is the problem, "
                "not the signal" % (capped, TURNOVER_FLOOR))
        needed = TURNOVER_FLOOR / (sharpe ** 2) if sharpe else None
        detail = "the denominator is pinned at %.3f" % TURNOVER_FLOOR
        if needed is not None:
            detail += "; clearing 1.0 needs returns of %.2f%% against %.2f%% actual" % (
                100 * needed, 100 * abs(returns))
        return "low_amplitude", detail

    if not checks:
        # A row with a passing Fitness and no recorded gate was rejected for something
        # the ledger never wrote down -- usually a submission-time check. Calling that
        # "unclassified" would blame the classifier for a recording gap.
        if fitness is not None and fitness >= FITNESS_FLOOR:
            return "unrecorded_check", (
                "Fitness %.2f clears the gate but no failing check was stored; the "
                "cause is a recording gap, not a missing rule" % fitness)
        return "unclassified", "no failing check and no usable metrics"
    return "other_gate", "failed %s" % ", ".join(sorted(checks))