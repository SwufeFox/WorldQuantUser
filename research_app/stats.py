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

#: Below this many points a Sharpe is not reported, it is guessed.
MIN_POINTS = 30


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
    if len(a) != len(b) or len(a) < min_points:
        return None
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 0 or var_b <= 0:
        return None
    return cov / math.sqrt(var_a * var_b)


def effective_bets(pair_correlations, members):
    """``N / (1 + (N-1) * mean_rho)``: the correlation-equivalent count of bets."""
    if not pair_correlations or not members:
        return None
    mean = sum(pair_correlations) / len(pair_correlations)
    denominator = 1 + (members - 1) * max(mean, 0)
    if denominator <= 0:
        return None
    return round(members / denominator, 2)