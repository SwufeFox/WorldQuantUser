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

from . import book_state, legs, pnl_snapshot, stats


def _ledger(registry_path=None):
    latest = {}
    for entry in alpha_registry.load_entries(registry_path):
        key = entry.get("alpha_id") or entry.get("key")
        if key:
            latest[key] = entry
    return latest


def _load(directory, alpha_ids):
    return pnl_snapshot.load_series_map(directory, alpha_ids)


def _sharpe(matrix):
    """Sharpe per row of a (n, days) array, using the campaign-wide convention."""
    values = []
    for row in matrix:
        measured = stats.sharpe(list(row), min_points=2)
        values.append(float("nan") if measured is None else measured)
    return np.array(values)


def cmd_pairs(args, _sink):
    """Screen non-book cached PnL pairs for equal-weight composites under the book gate."""
    from .runtime import CliError

    directory = os.path.join(research_paths.ensure_dir(), "pnl")
    latest = _ledger(getattr(args, "registry", None))

    book = [aid for aid, entry in latest.items()
            if str(entry.get("status") or "").upper() in book_state.BOOK_STATUSES]
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
        raise CliError(
            "INVALID_INPUT", "fewer than two pool alphas have cached PnL",
            "adjust --min-leg-sharpe/--max-leg-sharpe or recover PnL via `probe --recover` / `ledger sync --pnl`",
        )

    dates = set(pool_series[pool[0]])
    for values in pool_series.values():
        dates &= set(values)
    for values in book_series.values():
        dates &= set(values)
    dates = sorted(dates)
    if len(dates) < 120:
        raise CliError(
            "INVALID_INPUT", "only %d shared days across the pool" % len(dates),
            "refresh missing PnL caches with `ledger sync --pnl` or inspect individual series",
        )

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
    n_days = len(dates)
    row_means = pool_matrix.mean(axis=1)
    pool_centred = pool_matrix - row_means[:, None]
    cov = (pool_centred @ pool_centred.T) / n_days
    var = np.diag(cov)
    comp_mean = 0.5 * (row_means[:, None] + row_means[None, :])
    comp_var = 0.25 * (var[:, None] + var[None, :] + 2.0 * cov)
    comp_norm = np.sqrt(np.maximum(comp_var * n_days, 0.0))
    comp_std = np.sqrt(np.maximum(comp_var, 0.0))
    scale = math.sqrt(252.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        comp_sharpe_mat = np.where(comp_std > 0.0, comp_mean / comp_std * scale, np.nan)

    has_book_members = len(book) > 0
    has_book_total = bool(book_norm_total)
    book_member_proj = (book_norm @ pool_centred.T) if has_book_members else None
    book_total_proj = (book_centred @ pool_centred.T) if has_book_total else None

    centered_token = ("group_rank(", "group_zscore(", "group_neutralize(")
    centered_flags = [
        any(token in (latest[aid].get("expression") or "") for token in centered_token)
        for aid in pool
    ]
    breadths = [
        ((latest[aid].get("metrics") or {}).get("longCount") or 0)
        + ((latest[aid].get("metrics") or {}).get("shortCount") or 0)
        for aid in pool
    ]
    returns_arr = [float((latest[aid].get("metrics") or {}).get("returns") or 0) for aid in pool]
    turnover_arr = [float((latest[aid].get("metrics") or {}).get("turnover") or 0) for aid in pool]

    ii, jj = np.triu_indices(len(pool), k=1)
    norms_pair = comp_norm[ii, jj]
    sharpes_pair = comp_sharpe_mat[ii, jj]
    keep = (norms_pair > 0.0) & np.isfinite(sharpes_pair)
    if args.min_composite_sharpe is not None:
        keep &= sharpes_pair >= args.min_composite_sharpe

    if has_book_total and not math.isnan(book_sharpe):
        rho_book_all = np.where(
            norms_pair > 0.0,
            0.5 * (book_total_proj[ii] + book_total_proj[jj]) / np.where(norms_pair > 0.0, norms_pair, 1.0),
            0.0,
        )
        marginal_all = sharpes_pair - rho_book_all * float(book_sharpe)
        if args.min_marginal is not None:
            keep &= np.isfinite(marginal_all) & (marginal_all >= args.min_marginal)
    else:
        rho_book_all = np.zeros(len(ii), dtype=np.float64)
        marginal_all = None
        if args.min_marginal is not None:
            keep[:] = False

    cand_idx = np.flatnonzero(keep)
    cand_i = ii[cand_idx]
    cand_j = jj[cand_idx]
    if has_book_members and len(cand_idx):
        # Compute max(|book_norm @ comp_norm|) in chunks to avoid (B, M) memory blowup.
        max_rho_cand = np.empty(len(cand_idx), dtype=np.float64)
        chunk = 16384
        for start in range(0, len(cand_idx), chunk):
            end = min(start + chunk, len(cand_idx))
            ci = cand_i[start:end]
            cj = cand_j[start:end]
            cn = norms_pair[cand_idx[start:end]]
            proj = 0.5 * (book_member_proj[:, ci] + book_member_proj[:, cj]) / cn[None, :]
            max_rho_cand[start:end] = np.max(np.abs(proj), axis=0)
    else:
        max_rho_cand = np.zeros(len(cand_idx), dtype=np.float64)

    plain_only = bool(getattr(args, "plain_legs", False))
    rows = []
    for k, flat_idx in enumerate(cand_idx):
        i = int(cand_i[k])
        j = int(cand_j[k])
        rho = float(max_rho_cand[k])
        if args.max_book_rho is not None and rho > args.max_book_rho:
            continue
        centred_legs = centered_flags[i] or centered_flags[j]
        if plain_only and centred_legs:
            continue
        breadth_a, breadth_b = breadths[i], breadths[j]
        support_ratio = None
        if breadth_a and breadth_b:
            support_ratio = min(breadth_a, breadth_b) / max(breadth_a, breadth_b)
        if args.min_support_ratio is not None:
            if support_ratio is None or support_ratio < args.min_support_ratio:
                continue
        sharpe = float(sharpes_pair[flat_idx])
        rho_book = float(rho_book_all[flat_idx])
        marginal = None if marginal_all is None else float(marginal_all[flat_idx])
        returns = 0.5 * (returns_arr[i] + returns_arr[j])
        turnover = 0.5 * (turnover_arr[i] + turnover_arr[j])
        estimated_fitness = stats.platform_fitness(sharpe, returns, turnover) if (returns and turnover) else None
        rows.append({
            "a": pool[i], "b": pool[j],
            "sharpe_a": None if np.isnan(sharpes[i]) else round(float(sharpes[i]), 3),
            "sharpe_b": None if np.isnan(sharpes[j]) else round(float(sharpes[j]), 3),
            "composite_sharpe": round(sharpe, 3),
            "composite_fitness_estimate": None if estimated_fitness is None
            else round(estimated_fitness, 3),
            "max_book_rho": round(rho, 3),
            "book_rho": round(rho_book, 3),
            "marginal_vs_book": None if marginal is None else round(marginal, 3),
            "leg_sum_risk": "centred" if centred_legs else "plain",
            "support_ratio": None if support_ratio is None else round(support_ratio, 3),
        })

    rows.sort(key=lambda row: -row["composite_sharpe"]
              if args.rank == "sharpe" else -row["marginal_vs_book"])
    return {
        "window": {"start": dates[0], "end": dates[-1], "days": len(dates)},
        "pool": len(pool),
        "book": len(book),
        "book_evidence": book_state.describe_book(
            latest, series=book_series, registry_path=getattr(args, "registry", None), pnl_dir=directory),
        "pairs_tested": len(rows),
        "criteria": {"min_leg_sharpe": args.min_leg_sharpe,
                     "max_leg_sharpe": args.max_leg_sharpe,
                     "max_book_rho": args.max_book_rho,
                     "min_composite_sharpe": args.min_composite_sharpe,
                     "min_marginal": args.min_marginal,
                     "min_support_ratio": args.min_support_ratio,
                     "rank": args.rank},
        "book_sharpe": None if math.isnan(book_sharpe) else round(float(book_sharpe), 3),
        "gate_basis": ("worst single member (--max-book-rho); marginal_vs_book is the "
                       "number that says whether the pair helps the book"),
        "note": ("Sharpe and correlations come from cached daily PnL; fitness is estimated "
                 "from the legs' ledger returns and turnover, so treat it as a hint."),
        "pairs": rows[: args.limit],
    }
