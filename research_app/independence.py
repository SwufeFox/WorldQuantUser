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

from . import legs


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


def marginal_report(alpha_id, registry_path=None, pnl_dir=None):
    """Theoretical marginal, actual book change, and the gap between them."""
    latest = {}
    for entry in alpha_registry.load_entries(registry_path):
        key = entry.get("alpha_id") or entry.get("key")
        if key:
            latest[key] = entry
    directory = pnl_dir or os.path.join(research_paths.ensure_dir(), "pnl")
    book = [aid for aid, entry in latest.items()
            if str(entry.get("status") or "").upper() in ("ACTIVE", "SUBMITTED")
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
            if str(entry.get("status") or "").upper() in ("ACTIVE", "SUBMITTED")]
    directory = pnl_dir or os.path.join(research_paths.ensure_dir(), "pnl")
    series = _series(book, directory)
    book = sorted(series)
    dates = _aligned(series)
    if len(dates) < 250:
        raise ValueError("only %d shared days" % len(dates))

    book_daily = [sum(series[aid][d] for aid in book) for d in dates]
    ordered = sorted(range(len(dates)), key=lambda i: book_daily[i])
    cut = max(10, len(ordered) // 10)
    stressed = {dates[i] for i in ordered[:cut]}
    calm = {dates[i] for i in ordered[-cut:]}

    def corr_on(days):
        picked = [i for i, d in enumerate(dates) if d in days]
        values = {aid: [series[aid][dates[i]] for i in picked] for aid in book}
        pairs = []
        for i, first in enumerate(book):
            for second in book[i + 1:]:
                rho = legs.correlation(values[first], values[second])
                if rho is not None:
                    pairs.append({"a": first, "b": second, "r": rho})
        return pairs

    full = corr_on(set(dates))
    stress = corr_on(stressed)
    calm = corr_on(calm)

    def summary(pairs):
        if not pairs:
            return {"pairs": 0}
        values = sorted(p["r"] for p in pairs)
        mean = sum(values) / len(values)
        high = [p for p in pairs if p["r"] >= threshold]
        return {
            "pairs": len(pairs),
            "mean": round(mean, 3),
            "median": round(values[len(values) // 2], 3),
            "p90": round(values[int(0.9 * (len(values) - 1))], 3),
            "above_threshold": len(high),
            "effective_bets": round(len(book) / (1 + (len(book) - 1) * max(mean, 0)), 2),
        }

    full_map = {key: value for key, value in
                (((p["a"], p["b"]), p["r"]) for p in full)}
    stress_map = {(p["a"], p["b"]): p["r"] for p in stress}
    shifted = [(a, b, value, stress_map[(a, b)])
               for (a, b), value in full_map.items() if (a, b) in stress_map]
    # The interesting number: how much higher does correlation get when the book bleeds?
    jumps = sorted(({"a": a, "b": b, "full": round(fr, 3), "stressed": round(sr, 3),
                     "jump": round(sr - fr, 3)} for a, b, fr, sr in shifted),
                   key=lambda row: -row["jump"])

    return {
        "book_members": len(book),
        "window": {"start": dates[0], "end": dates[-1], "days": len(dates)},
        "stress_days": len(stressed),
        "calm_days": len(calm),
        "full_sample": summary(full),
        "stressed": summary(stress),
        "calm": summary(calm),
        "largest_correlation_jumps": jumps[:10],
        "note": ("effective_bets uses the average pairwise correlation over the full "
                 "sample; the stress block is the worst decile of book days, where the "
                 "diversification that the full sample advertises is what matters."),
    }


def cmd_marginal(args, _sink):
    return marginal_report(args.alpha_id, getattr(args, "registry", None))


def cmd_independence(args, _sink):
    return independence_report(getattr(args, "registry", None),
                               threshold=getattr(args, "threshold", 0.7))
