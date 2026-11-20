"""One Sharpe, one correlation, one place.

There were four Sharpe implementations in this repo: ``ledger`` and the half-split in
``experiments`` divided the variance by ``n``, ``legs`` divided by ``n-1`` and required at
least 30 points, and ``pairs`` / ``independence`` leaned on numpy's ``ddof=1``. On a
1236-day series the difference is 0.04% and nobody notices; on a half-split of 600 days or
a short window it is a silent fork in the numbers two commands report for the same alpha.

The convention here is the population form (divide by ``n``), because that is what the
ledger and the half-split already used and those are the numbers every decision was made
on. Changing the convention would move historical reports; unifying the code does not.
"""

import math

from correlation import pearson

#: Below this many points a Sharpe is not reported, it is guessed.
MIN_POINTS = 30
MIN_PORTFOLIO_OVERLAP = 120
FITNESS_TURNOVER_FLOOR = 0.125
# Recheck decisive comparisons with the scalar kernel within this band.
CORRELATION_ROUNDOFF = 1e-10


def platform_fitness(sharpe, returns, turnover):
    """BRAIN Fitness for a complete set of reported metrics."""
    return sharpe * math.sqrt(abs(returns) / max(turnover, FITNESS_TURNOVER_FLOOR))


def sharpe(values, min_points=MIN_POINTS):
    """Annualised Sharpe of a daily PnL series, ``None`` when it cannot be measured."""
    cleaned = [float(v) for v in values if v is not None]
    if len(cleaned) < min_points:
        return None
    mean = sum(cleaned) / len(cleaned)
    variance = sum((value - mean) ** 2 for value in cleaned) / len(cleaned)
    if variance <= 0:
        return None
    return mean / math.sqrt(variance) * math.sqrt(252)


def correlation(a, b, min_points=MIN_POINTS):
    """Pearson correlation of two equal-length series, aligned by position."""
    return pearson(a, b, min_points=min_points)


def unit_correlation_rows(rows):
    """Center and normalize aligned rows so dot products give Pearson r.

    Return None when the scalar kernel is needed, including when NumPy is
    absent or a row is constant or numerically sensitive.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        values = np.asarray(rows)
    except (TypeError, ValueError, OverflowError):
        return None
    if values.ndim != 2 or values.shape[1] < MIN_POINTS or values.dtype.kind not in "biuf":
        return None
    if values.dtype.kind in "iu" and (values.max() > 2**53 or values.min() < -(2**53)):
        return None
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        return None
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        low, high = values.min(axis=1), values.max(axis=1)
        if not ((high - low) > 1e-5 * np.maximum(abs(low), abs(high))).all():
            return None
        centered = values - values.mean(axis=1, keepdims=True)
        norms = np.sqrt(np.einsum("ij,ij->i", centered, centered))
        # Keep extreme scales on the scalar path.
        if not ((norms > 1e-70) & (norms < 1e70)).all():
            return None
        return centered / norms[:, None]


def member_correlations(values, series, dates, unit_book=None, min_points=MIN_POINTS):
    """Return the measured count and strongest signed correlation to a member."""
    if len(dates) < min_points:
        return 0, None
    members = list(series.items())
    unit_candidate = unit_correlation_rows([values]) if unit_book is not None else None
    if unit_candidate is not None:
        approximate = abs((unit_book * unit_candidate[0]).sum(axis=1))
        maximum = approximate.max()
        members_to_check = [member for member, rho in zip(members, approximate)
                            if maximum - rho <= CORRELATION_ROUNDOFF]
    else:
        members_to_check = members
    # Recompute close contenders with the scalar kernel, including ties.
    measured = []
    for member, member_values in members_to_check:
        rho = correlation(values, [member_values[day] for day in dates], min_points=min_points)
        if rho is not None:
            measured.append((rho, member))
    count = len(members) if unit_candidate is not None else len(measured)
    return count, max(measured, key=lambda item: abs(item[0]), default=None)


def effective_bets(pair_correlations, members):
    """``N / (1 + (N-1) * mean_rho)``: the correlation-equivalent count of bets."""
    if not pair_correlations or not members:
        return None
    mean = sum(pair_correlations) / len(pair_correlations)
    denominator = 1 + (members - 1) * max(mean, 0)
    if denominator <= 0:
        return None
    return round(members / denominator, 2)
