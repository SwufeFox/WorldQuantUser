"""Which two rejected arms should be combined - screened from cached PnL, before simulating.

The addition rule is old news and keeps being ignored: an equal-weight pair beats its
better leg only when ``rho < S_weak / S_strong``. So the question "is there a pair worth
running" is answerable offline, from the PnL cache, for every pair that was ever
simulated - as long as the composite is also checked against the book, because a pair
that is an improvement but sits 0.8 away from a book member cannot be submitted.

That is the whole point of this screen: rank pairs by the composite's Sharpe *subject to*
staying under the self-correlation gate against every book alpha. A pair that fails the
gate is not a candidate, however good its own numbers look.

Returns and turnover are averaged from the two legs' ledger metrics, which is what the
platform's own F arithmetic does for a rank-blend of similar signals, so the fitness
column is an estimate. The Sharpe and the correlations are computed from actual daily
PnL and are not estimates.
"""

import json
import math
import os

import numpy as np

import alpha_registry
import research_paths

from . import legs


def _ledger(registry_path=None):
    latest = {}
    for entry in alpha_registry.load_entries(registry_path):
        key = entry.get("alpha_id") or entry.get("key")
        if key:
            latest[key] = entry
    return latest


def _load(directory, alpha_ids):
    series = {}
    for alpha_id in alpha_ids:
        values = legs.load_pnl(alpha_id, directory)
        if values:
            series[alpha_id] = values
    return series


def _sharpe(matrix):
    """Sharpe per row of a (n, days) array."""
    mean = matrix.mean(axis=1)
    std = matrix.std(axis=1, ddof=1)
    std[std == 0] = np.nan
    return mean / std * math.sqrt(252)


def cmd_pairs(args, _sink):
    directory = os.path.join(research_paths.ensure_dir(), "pnl")
    latest = _ledger(getattr(args, "registry", None))

    book = [aid for aid, entry in latest.items()
            if str(entry.get("status") or "").upper() in ("ACTIVE", "SUBMITTED")]
    pool = []
    for alpha_id, entry in latest.items():
        if alpha_id in book:
            continue
        sharpe = (entry.get("metrics") or {}).get("sharpe")
        if sharpe is None:
            continue
        if args.min_leg_sharpe is not None and abs(sharpe) < args.min_leg_sharpe:
            continue
        if args.max_leg_sharpe is not None and abs(sharpe) > args.max_leg_sharpe:
            continue
        pool.append(alpha_id)

    pool_series = _load(directory, pool)
    book_series = _load(directory, book)
    pool = [aid for aid in pool if aid in pool_series]
    book = [aid for aid in book if aid in book_series]
    if len(pool) < 2:
        raise ValueError("fewer than two pool alphas have cached PnL")

    dates = set(pool_series[pool[0]])
    for values in pool_series.values():
        dates &= set(values)
    for values in book_series.values():
        dates &= set(values)
    dates = sorted(dates)
    if len(dates) < 120:
        raise ValueError("only %d shared days across the pool" % len(dates))

    pool_matrix = np.array([[pool_series[a][d] for d in dates] for a in pool], dtype=float)
    book_matrix = np.array([[book_series[a][d] for d in dates] for a in book], dtype=float)
    book_norm = (book_matrix - book_matrix.mean(axis=1, keepdims=True))
    book_scale = np.linalg.norm(book_norm, axis=1)
    book_scale[book_scale == 0] = np.nan
    book_norm = book_norm / book_scale[:, None]

    book_total = book_matrix.sum(axis=0)
    book_sharpe = _sharpe(book_total[None, :])[0]
    book_centred = book_total - book_total.mean()
    book_norm_total = np.linalg.norm(book_centred)
    if book_norm_total:
        book_centred = book_centred / book_norm_total

    sharpes = _sharpe(pool_matrix)
    rows = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            composite = 0.5 * (pool_matrix[i] + pool_matrix[j])
            centred = composite - composite.mean()
            norm = np.linalg.norm(centred)
            if norm == 0:
                continue
            rho = float(np.max(np.abs(book_norm @ (centred / norm)))) if len(book) else 0.0
            rho_book = float(book_centred @ (centred / norm)) if book_norm_total else 0.0
            sharpe = _sharpe(composite[None, :])[0]
            if math.isnan(sharpe):
                continue
            if args.max_book_rho is not None and rho > args.max_book_rho:
                continue
            # The gate against the worst *member* is not the addition test: what a new
            # alpha has to beat is the book as a whole, because that is what it is being
            # added to. A pair can pass 0.7 against every member and still be dilutive.
            marginal = None
            if not math.isnan(book_sharpe):
                marginal = float(sharpe) - rho_book * float(book_sharpe)
            if args.min_marginal is not None and (marginal is None or marginal < args.min_marginal):
                continue
            if args.min_composite_sharpe is not None and sharpe < args.min_composite_sharpe:
                continue
            metrics_a = (latest[pool[i]].get("metrics") or {})
            metrics_b = (latest[pool[j]].get("metrics") or {})
            returns = (float(metrics_a.get("returns") or 0) + float(metrics_b.get("returns") or 0)) / 2
            turnover = (float(metrics_a.get("turnover") or 0) + float(metrics_b.get("turnover") or 0)) / 2
            estimated_fitness = None
            if returns and turnover:
                estimated_fitness = sharpe * math.sqrt(abs(returns) / max(turnover, 0.125))
            rows.append({
                "a": pool[i], "b": pool[j],
                "sharpe_a": None if np.isnan(sharpes[i]) else round(float(sharpes[i]), 3),
                "sharpe_b": None if np.isnan(sharpes[j]) else round(float(sharpes[j]), 3),
                "composite_sharpe": round(float(sharpe), 3),
                "composite_fitness_estimate": None if estimated_fitness is None
                else round(estimated_fitness, 3),
                "max_book_rho": round(rho, 3),
                "book_rho": round(rho_book, 3),
                "marginal_vs_book": None if marginal is None else round(marginal, 3),
            })

    rows.sort(key=lambda row: -row["composite_sharpe"]
              if args.rank == "sharpe" else -row["marginal_vs_book"])
    return {
        "window": {"start": dates[0], "end": dates[-1], "days": len(dates)},
        "pool": len(pool),
        "book": len(book),
        "pairs_tested": len(rows),
        "criteria": {"min_leg_sharpe": args.min_leg_sharpe,
                     "max_leg_sharpe": args.max_leg_sharpe,
                     "max_book_rho": args.max_book_rho,
                     "min_composite_sharpe": args.min_composite_sharpe,
                     "min_marginal": args.min_marginal,
                     "rank": args.rank},
        "book_sharpe": None if math.isnan(book_sharpe) else round(float(book_sharpe), 3),
        "note": ("Sharpe and correlations come from cached daily PnL; fitness is estimated "
                 "from the legs' ledger returns and turnover, so treat it as a hint."),
        "pairs": rows[: args.limit],
    }