#!/usr/bin/env python3
"""Correlation maths for daily PnL series.

Why this is a module and not a script: the same handful of calculations came up
every time an alpha was judged, and each time they were rewritten by hand. That
is where mistakes come from. Two in particular were made repeatedly and are now
encoded once, with the trap named:

1. **Two alphas that each look independent of the existing pool can still be
   correlated with each other.** Judging a pair only against the book hides
   shared exposures. :func:`matrix` therefore reports every pair, not just
   each candidate's maximum.

2. **A shared factor makes unrelated signals look alike.** If two alphas load on
   the same leg, their raw correlation is mostly that leg.
   :func:`residualize` strips a chosen control series first, and the honest
   question is whether correlation survives that.

Everything here is pure: no network, no files, no globals. Dates are compared as
strings (``YYYY-MM-DD``), which is how the API returns them and sorts correctly.
"""

from __future__ import annotations

import math
from operator import mul

__all__ = [
    "align",
    "spectrum",
    "pearson",
    "center_for_pearson",
    "pearson_centered",
    "matrix",
    "max_against",
    "matrix_against_refs",
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


def pearson(x, y, *, min_points=3):
    """Pearson correlation, or ``None`` when it is not defined.

    A constant series has no correlation with anything; returning ``None`` rather
    than 0.0 keeps that distinct from "measured as uncorrelated".
    """
    n = len(x)
    if n < min_points or n != len(y):
        return None
    return pearson_centered(center_for_pearson(x, min_points=min_points),
                            center_for_pearson(y, min_points=min_points))


def center_for_pearson(values, *, min_points=3):
    """Center one aligned series and compute its norm once for reuse."""
    if len(values) < max(1, min_points):
        return None
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    norm_squared = sum(value ** 2 for value in centered)
    return None if norm_squared <= 0 else (centered, norm_squared ** 0.5)


def pearson_centered(left, right):
    """Correlate two prepared series with the same observations in the same order."""
    if left is None or right is None:
        return None
    x, x_norm = left
    y, y_norm = right
    if len(x) != len(y):
        return None
    return sum(map(mul, x, y)) / (x_norm * y_norm)


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


def _encode_panel(pnls, ids):
    """Build masked, scale-normalized, pre-centered dense ``(N, D)`` float64 panels."""
    try:
        import numpy as np
    except ImportError:
        return None
    n_rows = len(ids)
    row_idxs, day_keys, vals = [], [], []
    for r_idx, aid in enumerate(ids):
        series = pnls.get(aid) or {}
        for day, val in series.items():
            if type(val) in (int, float) and math.isfinite(val):
                row_idxs.append(r_idx)
                day_keys.append(day)
                vals.append(float(val))
    if not row_idxs:
        return (
            np.zeros((n_rows, 0), dtype=np.float64),
            np.zeros((n_rows, 0), dtype=np.float64),
            np.zeros(n_rows, dtype=np.float64),
        )
    try:
        import polars as pl
        frame = pl.DataFrame({"r": row_idxs, "d": day_keys, "v": vals})
        encoded = frame.select(
            pl.col("r").cast(pl.Intp),
            (pl.col("d").rank("dense") - 1).cast(pl.Intp).alias("c"),
            pl.col("v").cast(pl.Float64),
        )
        r_arr = encoded.get_column("r").to_numpy()
        c_arr = encoded.get_column("c").to_numpy()
        v_arr = encoded.get_column("v").to_numpy()
        n_cols = int(c_arr.max()) + 1 if len(c_arr) else 0
    except Exception:  # noqa: BLE001 - fall back to pure-Python day index when Polars unavailable
        unique_days = sorted(set(day_keys))
        day_map = {day: idx for idx, day in enumerate(unique_days)}
        r_arr = np.asarray(row_idxs, dtype=np.intp)
        c_arr = np.asarray([day_map[d] for d in day_keys], dtype=np.intp)
        v_arr = np.asarray(vals, dtype=np.float64)
        n_cols = len(unique_days)

    V = np.zeros((n_rows, n_cols), dtype=np.float64)
    M = np.zeros((n_rows, n_cols), dtype=np.float64)
    V[r_arr, c_arr] = v_arr
    M[r_arr, c_arr] = 1.0

    # Pre-scale rows above 1e150 so V.sum() never overflows float64, then center observed
    # entries before scaling by max(|V - mu|) to preserve full precision on large offsets.
    raw_max = np.max(np.abs(V), axis=1)
    V = np.where(M > 0.0, V / np.where(raw_max > 1e150, raw_max, 1.0)[:, None], 0.0)
    counts = M.sum(axis=1)
    means = np.divide(V.sum(axis=1), counts, out=np.zeros_like(counts), where=counts > 0.0)
    V_centered_raw = np.where(M > 0.0, V - means[:, None], 0.0)
    row_max = np.max(np.abs(V_centered_raw), axis=1)
    safe_scale = np.where(row_max > 0.0, row_max, 1.0)[:, None]
    Vc = np.where(M > 0.0, V_centered_raw / safe_scale, 0.0)
    return Vc, M, row_max


def _finalize_pair_corr(cov, vx, vy, max_x, max_y, series_x, series_y, min_overlap):
    """Resolve Pearson r from centered moments, falling back to exact scalar on degenerate pairs."""
    if max_x == 0.0 or max_y == 0.0:
        return None
    if vx <= 1e-22 or vy <= 1e-22 or not (math.isfinite(cov) and math.isfinite(vx) and math.isfinite(vy)):
        x, y, _ = align(series_x, series_y, min_overlap)
        return pearson(x, y) if x else None
    denom = math.sqrt(vx * vy)
    if denom <= 0.0:
        return None
    return max(-1.0, min(1.0, cov / denom))


def matrix(pnls, min_overlap=200):
    """Pairwise correlation for ``{name: series}``.

    Returns ``{"ids", "pairs", "measured", "skipped"}``. Pairs whose overlap is
    too short are listed in ``skipped`` rather than silently dropped, because a
    missing pair and a zero pair mean different things. Each skip carries a
    ``reason``: ``short_overlap`` is a data problem that waiting can fix, while
    ``undefined`` means one series is flat, which is a property of the alpha and
    will not improve.
    """
    ids = sorted(pnls)
    pairs = []
    skipped = []
    if len(ids) >= 4:
        panel = _encode_panel(pnls, ids)
        if panel is not None:
            import numpy as np
            Vc, M, row_max = panel
            N_mat = np.rint(M @ M.T).astype(np.int64)
            S_x = Vc @ M.T
            S_xx = (Vc * Vc) @ M.T
            S_xy = Vc @ Vc.T
            n_ids = len(ids)
            for i in range(n_ids):
                left = ids[i]
                left_series = pnls[left]
                max_i = float(row_max[i])
                for j in range(i + 1, n_ids):
                    right = ids[j]
                    n = int(N_mat[i, j])
                    if n < min_overlap:
                        skipped.append({"a": left, "b": right, "n": 0, "reason": "short_overlap"})
                        continue
                    sx = float(S_x[i, j])
                    sy = float(S_x[j, i])
                    cov = float(S_xy[i, j]) - (sx * sy) / n
                    vx = float(S_xx[i, j]) - (sx * sx) / n
                    vy = float(S_xx[j, i]) - (sy * sy) / n
                    r = _finalize_pair_corr(cov, vx, vy, max_i, float(row_max[j]),
                                            left_series, pnls[right], min_overlap)
                    if r is None:
                        skipped.append({"a": left, "b": right, "n": n, "reason": "undefined"})
                    else:
                        pairs.append({"a": left, "b": right, "r": r, "n": n})
            return {"ids": ids, "pairs": pairs, "measured": len(pairs), "skipped": skipped}

    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            x, y, dates = align(pnls[left], pnls[right], min_overlap)
            r = pearson(x, y) if x else None
            if r is None:
                skipped.append({"a": left, "b": right, "n": len(dates),
                                "reason": "short_overlap" if not x else "undefined"})
            else:
                pairs.append({"a": left, "b": right, "r": r, "n": len(dates)})
    return {"ids": ids, "pairs": pairs, "measured": len(pairs),
            "skipped": skipped}


def matrix_against_refs(targets, refs, min_overlap=200):
    """Compute ``max_against`` results for many targets against shared references in one BLAS pass."""
    target_ids = list(targets)
    ref_ids = list(refs)
    if not target_ids:
        return {}
    if not ref_ids:
        return {tid: {"max": None, "all": []} for tid in target_ids}

    total_targets = len(target_ids)
    total_refs = len(ref_ids)
    if total_targets * total_refs >= 8:
        ordered_keys = [f"__t_{i}" for i in range(total_targets)] + [f"__r_{j}" for j in range(total_refs)]
        combined = {f"__t_{i}": targets[tid] for i, tid in enumerate(target_ids)}
        combined.update({f"__r_{j}": refs[rid] for j, rid in enumerate(ref_ids)})
        panel = _encode_panel(combined, ordered_keys)
        if panel is not None:
            import numpy as np
            Vc, M, row_max = panel
            Vc_t, Vc_r = Vc[:total_targets], Vc[total_targets:]
            M_t, M_r = M[:total_targets], M[total_targets:]
            rmax_t, rmax_r = row_max[:total_targets], row_max[total_targets:]

            N_mat = np.rint(M_t @ M_r.T).astype(np.int64)
            S_t = Vc_t @ M_r.T
            S_r = M_t @ Vc_r.T
            S_tt = (Vc_t * Vc_t) @ M_r.T
            S_rr = M_t @ (Vc_r * Vc_r).T
            S_tr = Vc_t @ Vc_r.T

            results = {}
            for i, tid in enumerate(target_ids):
                t_series = targets[tid]
                max_t = float(rmax_t[i])
                rows = []
                for j, rid in enumerate(ref_ids):
                    n = int(N_mat[i, j])
                    if n < min_overlap:
                        continue
                    st = float(S_t[i, j])
                    sr = float(S_r[i, j])
                    cov = float(S_tr[i, j]) - (st * sr) / n
                    vt = float(S_tt[i, j]) - (st * st) / n
                    vr = float(S_rr[i, j]) - (sr * sr) / n
                    r = _finalize_pair_corr(cov, vt, vr, max_t, float(rmax_r[j]),
                                            t_series, refs[rid], min_overlap)
                    if r is not None:
                        rows.append({"with": rid, "r": r, "n": n})
                rows.sort(key=lambda row: abs(row["r"]), reverse=True)
                results[tid] = {"max": rows[0] if rows else None, "all": rows}
            return results

    results = {}
    for tid in target_ids:
        t_series = targets[tid]
        rows = []
        for rid in ref_ids:
            x, y, dates = align(refs[rid], t_series, min_overlap)
            if not x:
                continue
            r = pearson(x, y)
            if r is not None:
                rows.append({"with": rid, "r": r, "n": len(dates)})
        rows.sort(key=lambda row: abs(row["r"]), reverse=True)
        results[tid] = {"max": rows[0] if rows else None, "all": rows}
    return results


def max_against(target, refs, min_overlap=200):
    """Highest absolute correlation of ``target`` against each reference.

    Returns ``{"max": {...} or None, "all": [...]}`` sorted by absolute value, so
    the caller can see the runner-up and tell a single near-duplicate apart from
    a broad resemblance.
    """
    if len(refs) >= 8:
        return matrix_against_refs({"__target__": target}, refs, min_overlap=min_overlap)["__target__"]
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

    Daily values are used for correlation decisions. Weekly and monthly aggregates
    are diagnostics for distinguishing slower shared exposure from identical daily
    behavior; aggregation changes the statistic being compared.

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


#: Screening adjustment for local daily Pearson. Differences in peer sets and
#: timestamps can exceed this adjustment, so it cannot replace the platform check.
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

    The limit is on ``|r|``. The platform blocks a candidate that mirrors an existing alpha
    whatever the direction, so a peer at -0.95 blocks exactly as hard as one at +0.95; an
    earlier version compared the signed value and called a perfect mirror ``clear``.
    ``None`` means nothing was measured, which is ``unknown`` rather than ``clear``.
    """
    if daily_correlation is None:
        return "unknown", None
    estimate = round(min(1.0, abs(daily_correlation) + platform_bias), 4)
    if estimate >= threshold:
        return "blocked", estimate
    if estimate >= threshold - BORDERLINE_BAND:
        return "borderline", estimate
    return "clear", estimate


def spectrum(ids, pairs):
    """How many *independent* bets a correlated set really is.

    ``effective_bets`` is ``(sum lambda)^2 / sum lambda^2`` - the squared Frobenius
    norm gives the denominator without ever forming the eigenvalues - and
    ``pc1_share`` is the largest eigenvalue over the trace, found by power iteration
    so this module stays free of numpy.

    The cluster count answers a different question. Diffuse correlation can leave
    most pairs below a clustering threshold while substantially reducing the
    number of effective bets.
    """
    n = len(ids)
    if n < 2:
        return {"effective_bets": None, "pc1_share": None, "median_abs_r": None,
                "pairs_measured": 0}
    index = {name: position for position, name in enumerate(ids)}
    matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    off = []
    for pair in pairs:
        i, j = index[pair["a"]], index[pair["b"]]
        value = pair["r"]
        matrix[i][j] = matrix[j][i] = value
        off.append(abs(value))
    frobenius = sum(cell * cell for row in matrix for cell in row)
    effective = (n * n / frobenius) if frobenius else None
    vector = [1.0] * n
    for _ in range(80):
        product = [sum(matrix[i][j] * vector[j] for j in range(n)) for i in range(n)]
        scale = max(abs(value) for value in product) or 1.0
        vector = [value / scale for value in product]
    numerator = sum(sum(matrix[i][j] * vector[j] for j in range(n)) * vector[i]
                    for i in range(n))
    denominator = sum(value * value for value in vector)
    lam1 = numerator / denominator if denominator else 0.0
    ordered = sorted(off)
    median = ordered[len(ordered) // 2] if ordered else None
    return {"effective_bets": round(effective, 2) if effective else None,
            "pc1_share": round(lam1 / n, 3),
            "median_abs_r": round(median, 3) if median is not None else None,
            "pairs_measured": len(off)}
