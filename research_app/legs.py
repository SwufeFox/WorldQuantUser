"""Did the composite behave like the sum of its legs?

The question keeps coming up the same way: two legs each look fine, their daily PnL
correlation is near zero, the arithmetic says the equal-weight combination should have
Sharpe 1.7 - and the composite comes back at 0.07. The composite is re-neutralised and
re-decayed as one expression, so the legs do not enter the portfolio the way the two
separate alphas did.

Once both legs *and* the composite are simulated, that gap is measurable rather than
arguable: sum the cached daily PnL of the legs with the chosen weights and compare its
Sharpe to the composite's. Both series are dollars on the same book, so the sum is the
leg model exactly, and the ratio is how much the platform's per-expression treatment took
away.

The ratio is worth recording per leg-pair: a pair that collapses once will collapse again,
and the shape of the legs (level transforms, autocorrelations, ranks) is visible in the
expression.
"""

import json
import math
import os

import alpha_registry
import research_paths


def load_pnl(alpha_id, pnl_dir=None):
    """``{date: pnl}`` for one alpha, or ``{}`` when it was never cached."""
    directory = pnl_dir or os.path.join(research_paths.ensure_dir(), "pnl")
    path = os.path.join(directory, "%s.json" % alpha_id)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sharpe_of(series):
    """Annualised Sharpe of a daily PnL series."""
    values = [v for v in series if v is not None]
    if len(values) < 30:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    if variance <= 0:
        return None
    return mean / math.sqrt(variance) * math.sqrt(252)


def correlation(a, b):
    if len(a) < 30 or len(b) != len(a):
        return None
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 0 or var_b <= 0:
        return None
    return cov / math.sqrt(var_a * var_b)


def combine(legs, pnl_dir=None):
    """Align the legs on shared dates and return ``(dates, {id: series}, summed series)``."""
    series = {}
    for alpha_id in legs:
        series[alpha_id] = load_pnl(alpha_id, pnl_dir)
    shared = None
    for values in series.values():
        dates = set(values)
        shared = dates if shared is None else (shared & dates)
    shared = sorted(shared or [])
    if len(shared) < 30:
        return shared, series, []
    aligned = {alpha_id: [values[d] for d in shared] for alpha_id, values in series.items()}
    summed = [0.0] * len(shared)
    for alpha_id, weight in legs.items():
        for index, value in enumerate(aligned[alpha_id]):
            summed[index] += weight * value
    return shared, aligned, summed


def cmd_legs(args, _sink):
    """Compare a weighted sum of leg PnLs with the composite that was actually simulated.

    Weights default to equal. ``--composite`` is optional: without it the command reports
    the legs' own Sharpe and the pairwise correlations, which is enough to see whether the
    arithmetic promises anything before spending a simulation on the composite.
    """
    legs = {}
    for spec in args.leg:
        if ":" in spec:
            alpha_id, weight = spec.split(":", 1)
            legs[alpha_id] = float(weight)
        else:
            legs[spec] = 1.0
    if not legs:
        raise ValueError("--leg is required, e.g. --leg ABC123:0.5")
    total = sum(legs.values())
    if total <= 0:
        raise ValueError("the weights sum to %s" % total)
    legs = {k: v / total for k, v in legs.items()}

    pnl_dir = os.path.join(research_paths.ensure_dir(), "pnl")
    dates, aligned, summed = combine(legs, pnl_dir)
    if not dates:
        raise ValueError("no cached PnL overlaps; simulate the legs first")

    per_leg = {}
    for alpha_id, series in aligned.items():
        per_leg[alpha_id] = {"weight": legs[alpha_id],
                             "sharpe": sharpe_of(series)}
    pairs = []
    ids = list(aligned)
    for i, first in enumerate(ids):
        for second in ids[i + 1:]:
            pairs.append({"a": first, "b": second,
                          "r": correlation(aligned[first], aligned[second])})

    predicted = sharpe_of(summed)
    latest = {}
    for entry in alpha_registry.load_entries(getattr(args, "registry", None)):
        key = entry.get("alpha_id") or entry.get("key")
        if key:
            latest[key] = entry

    payload = {
        "window": {"start": dates[0], "end": dates[-1], "days": len(dates)},
        "legs": per_leg,
        "pairs": pairs,
        "predicted_sharpe": predicted,
    }

    composite = getattr(args, "composite", None)
    if composite and getattr(args, "residual", False):
        return residual_report(composite, legs, pnl_dir)
    if composite:
        actual_series = load_pnl(composite, pnl_dir)
        actual = None
        if actual_series:
            common = [d for d in dates if d in actual_series]
            if len(common) >= 30:
                actual = sharpe_of([actual_series[d] for d in common])
        entry = latest.get(composite) or {}
        metrics = entry.get("metrics") or {}
        ratio = None
        if predicted not in (None, 0) and actual is not None:
            ratio = actual / predicted
        verdict = None
        if ratio is not None:
            if ratio < 0.85:
                verdict = "collapsed: the composite gave back %.0f%% of the leg model" % (
                    (1 - ratio) * 100)
            elif ratio > 1.15:
                verdict = "amplified by %.0f%%" % ((ratio - 1) * 100)
            else:
                verdict = "additive within 15%"
        payload["composite"] = {
            "alpha_id": composite,
            "predicted_sharpe": predicted,
            "actual_sharpe": actual,
            "ledger_sharpe": metrics.get("sharpe"),
            "ledger_fitness": metrics.get("fitness"),
            "ratio": ratio,
            "verdict": verdict,
        }
    return payload

def residual_report(composite, leg_weights, pnl_dir=None):
    """Where the leg sum misses, after calibrating the scale.

    The plain ratio "actual / predicted" mixes two failures: the composite may be a
    different *magnitude* of the same bet (normalisation), or a different bet entirely.
    Fitting ``r_C ~ a*r_A + b*r_B`` separates them: coefficients near the nominal weights
    mean the scale is right and the residual is the real deviation; coefficients far from
    it mean the legs entered the composite with different sizes, and comparing without
    calibration would blame the wrong layer.

    The date coverage is reported alongside because a residual computed on the wrong
    overlap looks exactly like a structural break.
    """
    import numpy as np

    directory = pnl_dir or os.path.join(research_paths.ensure_dir(), "pnl")
    ids = list(leg_weights)
    series = {}
    coverage = {}
    for alpha_id in ids + [composite]:
        values = load_pnl(alpha_id, directory)
        series[alpha_id] = values
        if values:
            dates = sorted(values)
            coverage[alpha_id] = {"days": len(dates), "start": dates[0], "end": dates[-1]}
        else:
            coverage[alpha_id] = {"days": 0, "start": None, "end": None}
    if not series.get(composite):
        raise ValueError("%s has no cached PnL" % composite)

    shared = set(series[composite])
    for alpha_id in ids:
        if not series[alpha_id]:
            raise ValueError("%s has no cached PnL" % alpha_id)
        shared &= set(series[alpha_id])
    dates = sorted(shared)
    if len(dates) < 120:
        raise ValueError("only %d shared days; the arms may not cover the same window" % len(dates))

    design = np.array([[series[alpha_id][d] for alpha_id in ids] for d in dates])
    target = np.array([series[composite][d] for d in dates])
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    residual = target - fitted
    ss_res = float((residual ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    r_squared = 1 - ss_res / ss_tot if ss_tot else None

    # The uncalibrated comparison, kept so the two numbers are not confused.
    nominal = design @ np.array([leg_weights[alpha_id] for alpha_id in ids])
    nominal_residual = target - nominal

    return {
        "composite": composite,
        "legs": ids,
        "window": {"start": dates[0], "end": dates[-1], "days": len(dates)},
        "coverage": coverage,
        "coefficients": {alpha_id: round(float(value), 3)
                         for alpha_id, value in zip(ids, coefficients)},
        "nominal_weights": {alpha_id: leg_weights[alpha_id] for alpha_id in ids},
        "r_squared": None if r_squared is None else round(float(r_squared), 4),
        "residual_sharpe": None if sharpe_of(list(residual)) is None
        else round(sharpe_of(list(residual)), 3),
        "residual_vol_share": round(float(residual.std() / target.std()), 4) if target.std() else None,
        "nominal_residual_sharpe": None if sharpe_of(list(nominal_residual)) is None
        else round(sharpe_of(list(nominal_residual)), 3),
        "reading": ("coefficients near the nominal weights mean the scale is right and the "
                    "residual is structure; far from them means the legs entered at "
                    "different sizes and the raw ratio was measuring scale, not structure."),
    }
