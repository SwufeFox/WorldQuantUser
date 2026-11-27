"""Is the book one bet or sixty? And what does adding an alpha actually do?

Two questions the campaign kept answering with approximations.

**Marginal contribution.** ``M = S_a - rho * S_B`` is a slope: the first-order change in
portfolio Sharpe per unit of the new signal's weight, under standardisation and a linear
combination. It is not the Sharpe the book gains by adding the alpha, because the book is
an equal-weight sum of already-neutralised alphas and the new member arrives with its own
scaling. Both numbers are reported here, side by side, because quoting only the first as
"the marginal" overstates what the reader thinks happened.

**Independence.** Counting mechanisms counts *categories*. Two alphas can use different
economic inputs, look different, and still be one bet in a crisis: full-sample correlation
is measured over a period where both the calm and the stressed days are averaged together.
The honest test splits the sample by the book's own returns and asks whether the pairs
that were independent in calm markets stay independent when the book is losing.
"""

import json
import math
import os

import alpha_registry
import research_paths

from . import book_state, legs, stats


def _series(alpha_ids, directory):
    out = {}
    for alpha_id in alpha_ids:
        series = legs.load_pnl(alpha_id, directory)
        if series:
            out[alpha_id] = series
    return out


def _aligned(series):
    dates = None
    for values in series.values():
        keys = set(values)
        dates = keys if dates is None else (dates & keys)
    return sorted(dates or [])


def _sharpe(values):
    return legs.sharpe_of(values)


def _near_rounding_boundary(value, places, tolerance=stats.CORRELATION_ROUNDOFF):
    if not math.isfinite(value):
        return False
    scale = 10 ** places
    scaled = value * scale
    return (abs(value) <= tolerance
            or abs(abs(scaled - round(scaled)) - 0.5) <= tolerance * scale)


def marginal_report(alpha_id, registry_path=None, pnl_dir=None):
    """Theoretical marginal, actual book change, and the gap between them."""
    latest = {}
    for entry in alpha_registry.load_entries(registry_path):
        key = entry.get("alpha_id") or entry.get("key")
        if key:
            latest[key] = entry
    directory = pnl_dir or os.path.join(research_paths.ensure_dir(), "pnl")
    book = [aid for aid, entry in latest.items()
            if str(entry.get("status") or "").upper() in book_state.BOOK_STATUSES
            and aid != alpha_id]
    series = _series(book + [alpha_id], directory)
    if alpha_id not in series:
        raise ValueError("%s has no cached PnL" % alpha_id)
    book = [aid for aid in book if aid in series]
    dates = _aligned(series)
    if len(dates) < 120:
        raise ValueError("only %d shared days" % len(dates))

    book_daily = [sum(series[aid][d] for aid in book) for d in dates]
    candidate = [series[alpha_id][d] for d in dates]
    book_sharpe = _sharpe(book_daily)
    candidate_sharpe = _sharpe(candidate)
    rho = legs.correlation(candidate, book_daily)
    theoretical = candidate_sharpe - rho * book_sharpe

    n = len(book)
    with_member = [(b + c) / (n + 1) for b, c in zip(book_daily, candidate)]
    actual = _sharpe(with_member) - book_sharpe

    return {
        "alpha_id": alpha_id,
        "book_members": n,
        "book": book_state.describe_book(latest, series=series, registry_path=registry_path,
                                           pnl_dir=directory),
        "window": {"start": dates[0], "end": dates[-1], "days": len(dates)},
        "book_sharpe": round(book_sharpe, 3),
        "candidate_sharpe": round(candidate_sharpe, 3),
        "correlation": round(rho, 3),
        "theoretical_marginal": round(theoretical, 3),
        "actual_book_change": round(actual, 4),
        "note": ("theoretical = S_a - rho*S_B (first-order slope); actual = Sharpe after "
                 "adding the alpha equal-weight. The second is what the book does."),
    }


def independence_report(registry_path=None, pnl_dir=None, threshold=0.7):
    """Pairwise correlation in calm vs stressed days, plus an effective-bet count."""
    latest = {}
    for entry in alpha_registry.load_entries(registry_path):
        key = entry.get("alpha_id") or entry.get("key")
        if key:
            latest[key] = entry
    book = [aid for aid, entry in latest.items()
            if str(entry.get("status") or "").upper() in book_state.BOOK_STATUSES]
    directory = pnl_dir or os.path.join(research_paths.ensure_dir(), "pnl")
    series = _series(book, directory)
    book = sorted(series)
    dates = _aligned(series)
    if len(dates) < 250:
        raise ValueError("only %d shared days" % len(dates))

    book_daily = [sum(series[aid][d] for aid in book) for d in dates]
    ordered = sorted(range(len(dates)), key=lambda i: book_daily[i])
    cut = max(10, len(ordered) // 10)
    stressed = sorted(ordered[:cut])
    calm = sorted(ordered[-cut:])
    all_rows = [[series[aid][day] for day in dates] for aid in book]
    stress_rows = [[row[index] for index in stressed] for row in all_rows]
    calm_rows = [[row[index] for index in calm] for row in all_rows]

    def corr_on(values, *, scalar=False):
        if len(values[0]) < stats.MIN_POINTS:
            return [], False
        units = stats.unit_correlation_rows(values) if not scalar else None
        matrix = units @ units.T if units is not None else None
        pairs = []
        for i, first in enumerate(book):
            for j in range(i + 1, len(book)):
                if matrix is None:
                    rho = legs.correlation(values[i], values[j])
                else:
                    rho = float(matrix[i, j])
                    # The report rounds the quantiles and compares the raw
                    # values to a threshold. Keep scalar arithmetic at either boundary.
                    if (abs(rho - threshold) <= stats.CORRELATION_ROUNDOFF
                            or _near_rounding_boundary(rho, 3)):
                        rho = legs.correlation(values[i], values[j])
                if rho is not None:
                    pairs.append({"a": first, "b": book[j], "r": rho})
        return pairs, matrix is not None

    full, full_vectorized = corr_on(all_rows)
    stress, stress_vectorized = corr_on(stress_rows)
    calm_pairs, calm_vectorized = corr_on(calm_rows)

    def summary(pairs, rows, vectorized):
        if not pairs:
            return {"pairs": 0}
        values = sorted(p["r"] for p in pairs)
        mean = sum(values) / len(values)
        effective = len(book) / (1 + (len(book) - 1) * max(mean, 0))
        effective_tolerance = stats.CORRELATION_ROUNDOFF * len(book) ** 2
        if vectorized and (_near_rounding_boundary(mean, 3)
                           or _near_rounding_boundary(effective, 2, effective_tolerance)):
            exact, _ = corr_on(rows, scalar=True)
            return summary(exact, rows, False)
        high = [p for p in pairs if p["r"] >= threshold]
        return {
            "pairs": len(pairs),
            "mean": round(mean, 3),
            "median": round(values[len(values) // 2], 3),
            "p90": round(values[int(0.9 * (len(values) - 1))], 3),
            "above_threshold": len(high),
            "effective_bets": round(effective, 2),
        }

    full_map = {key: value for key, value in
                (((p["a"], p["b"]), p["r"]) for p in full)}
    stress_map = {(p["a"], p["b"]): p["r"] for p in stress}
    shifted = [(a, b, value, stress_map[(a, b)])
               for (a, b), value in full_map.items() if (a, b) in stress_map]
    # The interesting number: how much higher does correlation get when the book bleeds?
    positions = {aid: index for index, aid in enumerate(book)}
    jumps = []
    for a, b, fr, sr in shifted:
        if (full_vectorized or stress_vectorized) and _near_rounding_boundary(
                sr - fr, 3, 2 * stats.CORRELATION_ROUNDOFF):
            left, right = positions[a], positions[b]
            fr = legs.correlation(all_rows[left], all_rows[right])
            sr = legs.correlation(stress_rows[left], stress_rows[right])
        jumps.append({"a": a, "b": b, "full": round(fr, 3), "stressed": round(sr, 3),
                      "jump": round(sr - fr, 3)})
    jumps.sort(key=lambda row: -row["jump"])

    return {
        "book_members": len(book),
        "book": book_state.describe_book(latest, series=series, registry_path=registry_path,
                                           pnl_dir=directory),
        "window": {"start": dates[0], "end": dates[-1], "days": len(dates)},
        "stress_days": len(stressed),
        # Historical reports export the number of calm pairs in this field.
        "calm_days": len(calm_pairs),
        "full_sample": summary(full, all_rows, full_vectorized),
        "stressed": summary(stress, stress_rows, stress_vectorized),
        "calm": summary(calm_pairs, calm_rows, calm_vectorized),
        "largest_correlation_jumps": jumps[:10],
        "note": ("effective_bets uses the average pairwise correlation over the full "
                 "sample; the stress block is the worst decile of book days, where the "
                 "diversification that the full sample advertises is what matters."),
    }


def cmd_marginal(args, _sink):
    """CLI handler for ``marginal``: compare theoretical vs actual book Sharpe change."""
    from .runtime import CliError

    try:
        return marginal_report(args.alpha_id, getattr(args, "registry", None))
    except ValueError as exc:
        raise CliError(
            "INVALID_INPUT", str(exc),
            "ensure the alpha and book have cached daily PnL (run `ledger sync --pnl` or `alpha get ID`)",
        ) from exc


def cmd_independence(args, _sink):
    """CLI handler for ``independence``: calm vs stressed book correlation summary."""
    from .runtime import CliError

    try:
        return independence_report(getattr(args, "registry", None),
                                   threshold=getattr(args, "threshold", 0.7))
    except ValueError as exc:
        raise CliError(
            "INVALID_INPUT", str(exc),
            "warm cached daily PnL for ACTIVE book alphas with `ledger sync --pnl` first",
        ) from exc
