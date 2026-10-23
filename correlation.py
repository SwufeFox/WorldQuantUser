#!/usr/bin/env python3
"""Correlation maths for daily PnL series.

Why this is a module and not a script: the same handful of calculations came up
every time an alpha was judged, and each time they were rewritten by hand. That
is where mistakes come from. Two in particular were made repeatedly and are now
encoded once, with the trap named:

1. **Two alphas that both look independent of the existing pool can still be
   ~0.6 correlated with each other.** Judging a pair only against the book hides
   the fact that the pair is one bet. :func:`matrix` therefore reports every pair,
   not just each candidate's maximum.

2. **A shared factor makes unrelated signals look alike.** If two alphas load on
   the same leg, their raw correlation is mostly that leg.
   :func:`residualize` strips a chosen control series first, and the honest
   question is whether correlation survives that.

Everything here is pure: no network, no files, no globals. Dates are compared as
strings (``YYYY-MM-DD``), which is how the API returns them and sorts correctly.
"""

from __future__ import annotations

__all__ = [
    "align",
    "pearson",
    "matrix",
    "max_against",
    "residualize",
    "beta",
]


def align(left, right, min_overlap=200):
    """Return ``(x, y, dates)`` on the dates both series share.

    ``min_overlap`` guards against the classic mistake of correlating two series
    that overlap for a few weeks during an unusual stretch and treating the
    result as a property of the strategies.
    """
    shared = sorted(set(left) & set(right))
    if len(shared) < min_overlap:
        return [], [], []
    return ([left[d] for d in shared], [right[d] for d in shared], shared)


def pearson(x, y):
    """Pearson correlation, or ``None`` when it is not defined.

    A constant series has no correlation with anything; returning ``None`` rather
    than 0.0 keeps that distinct from "measured as uncorrelated".
    """
    n = len(x)
    if n < 3 or n != len(y):
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxx = sum((v - mean_x) ** 2 for v in x)
    syy = sum((v - mean_y) ** 2 for v in y)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def beta(target, control, min_overlap=200):
    """OLS slope of ``target`` on ``control``, or ``None`` if undefined."""
    x, y, _ = align(control, target, min_overlap)
    if not x:
        return None
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxx = sum((v - mean_x) ** 2 for v in x)
    if sxx <= 0:
        return None
    sxy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    return sxy / sxx


def residualize(target, controls, min_overlap=200):
    """Remove the projection of ``target`` onto ``controls``, returning a series.

    ``controls`` may be a single series or a list. Each is removed in turn from
    the running residual, which is the usual sequential projection and is exact
    for orthogonal controls and approximate otherwise. The point is not perfect
    econometrics; it is answering "does this correlation survive removing the
    factor we both load on".
    """
    if isinstance(controls, dict):
        controls = [controls]
    series = dict(target)
    dates = sorted(series)
    for control in controls:
        shared = [d for d in dates if d in control]
        if len(shared) < min_overlap:
            continue
        # beta() works on date-keyed dicts, not bare lists: passing the extracted
        # values would make it try to index a list with a date.
        slope = beta({d: series[d] for d in shared},
                     {d: control[d] for d in shared}, min_overlap=1)
        if slope is None:
            continue
        for d in shared:
            series[d] = series[d] - slope * control[d]
    return series


def matrix(pnls, min_overlap=200):
    """Pairwise correlation for ``{name: series}``.

    Returns ``{"ids", "pairs", "measured", "skipped"}``. Pairs whose overlap is
    too short are listed in ``skipped`` rather than silently dropped, because a
    missing pair and a zero pair mean different things.
    """
    ids = sorted(pnls)
    pairs = []
    skipped = []
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            x, y, dates = align(pnls[left], pnls[right], min_overlap)
            r = pearson(x, y) if x else None
            if r is None:
                skipped.append({"a": left, "b": right, "n": len(dates)})
            else:
                pairs.append({"a": left, "b": right, "r": r, "n": len(dates)})
    return {"ids": ids, "pairs": pairs, "measured": len(pairs),
            "skipped": skipped}


def max_against(target, refs, min_overlap=200):
    """Highest absolute correlation of ``target`` against each reference.

    Returns ``{"max": {...} or None, "all": [...]}`` sorted by absolute value, so
    the caller can see the runner-up and tell a single near-duplicate apart from
    a broad resemblance.
    """
    rows = []
    for name, series in refs.items():
        x, y, dates = align(series, target, min_overlap)
        if not x:
            continue
        r = pearson(x, y)
        if r is None:
            continue
        rows.append({"with": name, "r": r, "n": len(dates)})
    rows.sort(key=lambda row: abs(row["r"]), reverse=True)
    return {"max": rows[0] if rows else None, "all": rows}

def aggregate(series, frequency="d"):
    """Aggregate a daily PnL series to weekly or monthly buckets.

    The platform's SELF_CORRELATION was calibrated against daily Pearson: over ten alphas
    with a known platform value, daily came within 0.006 to 0.016 and always slightly
    below, while weekly and monthly overstated it by 0.03 to 0.08. Daily is therefore the
    default; the slower frequencies are diagnostic, useful for telling a shared signal
    that drifts apart from a mechanically identical one.

    Buckets are keyed by calendar period, so two series only line up when they share the
    bucket - which is what ``align`` needs.
    """
    if frequency == "d":
        return dict(series or {})
    out = {}
    for key, value in (series or {}).items():
        if not isinstance(key, str) or len(key) < 10:
            continue
        if frequency == "m":
            bucket = key[:7]
        else:
            try:
                day = int(key[8:10])
            except ValueError:
                continue
            bucket = "%s-w%02d" % (key[:7], (day - 1) // 7 + 1)
        out[bucket] = out.get(bucket, 0.0) + float(value)
    return out


#: Typical gap between daily Pearson and the platform's SELF_CORRELATION, same book, same
#: moment: measured +0.013, +0.023 and +0.126 on three alphas.
#:
#: That spread is the point. The gap is not a constant, so this number orders candidates and
#: does not gate them: an alpha at 0.62 locally was refused at 0.745, which no correction of
#: this size would have predicted. Local correlation is a screen; the platform is the only
#: test that counts, and a refused submission costs nothing but a call.
#:
#: The +0.126 case also rules out the first explanation tried here - that the platform's
#: reference set is the ACTIVE book. It is not, or not only: that alpha's book-only local
#: maximum was 0.6195 against an ACTIVE counterpart, and the platform still measured 0.745.
PLATFORM_BIAS = 0.015

#: How close to the limit counts as undecided rather than clear. A candidate sitting just
#: under the line is worth submitting, but not worth trusting as a new bet.
BORDERLINE_BAND = 0.03


def classify(daily_correlation, threshold=0.7, platform_bias=PLATFORM_BIAS):
    """Turn a daily correlation into the verdict the platform will probably reach.

    Returns ``(verdict, estimated_platform)``. The estimate *adds* the measured bias, which
    is why an alpha at 0.66 locally is worth trying and one at 0.70 is not. Getting the sign
    wrong here is expensive in one direction only: it spends submissions on alphas that were
    always going to be refused.
    """
    estimate = round(min(1.0, daily_correlation + platform_bias), 4)
    if estimate >= threshold:
        return "blocked", estimate
    if estimate >= threshold - BORDERLINE_BAND:
        return "borderline", estimate
    return "clear", estimate
