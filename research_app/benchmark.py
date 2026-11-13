"""Discount platform Fitness using scoring priors and portfolio contribution.

The train-Sharpe bands and optional economic-block factors are scoring coefficients.
Raw calibration observations, sample sizes and individual alpha outcomes belong in
the research directory. Operator count warns about auditability without multiplying
the score. Treat this benchmark as a ranking aid, not evidence of future performance.
"""

import json
import os

import research_paths

from . import legs, mechanisms

#: Piecewise scoring coefficients: strongest confidence in the middle Sharpe band.
SHRINK_BANDS = (
    (1.0, 0.48),
    (1.5, 0.91),
    (2.0, 1.00),
    (2.5, 0.75),
    (99.0, 0.58),
)

#: Normalised scoring priors; raw performance measurements are not shipped.
BLOCK_PRIORS = {
    'price_path': 1.1199999999999999,
    'analyst_yield': 1.2133333333333334,
    'value_quality': 0.27999999999999997,
    'turnover_structure': 0.9199999999999999,
    'other': 0.25,
    'volatility_structure': 0.8933333333333334,
    'derivative_pricing': 0.6133333333333334,
    'analyst_dispersion': 2.0,
    'news_sentiment': 0.25,
    'debt_structure': 0.3333333333333333,
}

DEFAULTS = {
    "weight_shrink": 1.0,
    "weight_contribution": 1.0,
    "weight_retention": 1.0,
    "weight_block": 1.0,
    # Off by default: the factor only exists when a caller opts in, so the score keeps
    # its old meaning unless the criteria file says otherwise.
    "block_ratio": 0,
    "delta_full_credit": 0.05,
    "ops_warn": [8, 16, 32],
    "note": ("Weights default to 1.0 because there is no evidence to prefer another "
             "value; the factors themselves carry the information. Treat this as a "
             "ranking aid with a stated basis, not a calibrated score."),
}


def refine_criteria(path):
    """File over defaults, exactly like ``prune-criteria.json``."""
    criteria = dict(DEFAULTS)
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            raise ValueError("benchmark criteria %s is unreadable: %s" % (path, exc))
        if isinstance(payload, dict):
            criteria.update({k: v for k, v in payload.items() if k in DEFAULTS})
    return criteria


def shrink_factor(train_sharpe):
    """How much of the train-period signal to believe, by the hump above."""
    if train_sharpe is None:
        return 0.6
    for limit, factor in SHRINK_BANDS:
        if train_sharpe < limit:
            return factor
    return 0.55


def contribution_factor(delta, full_credit):
    if delta is None:
        return 0.5
    if delta <= 0:
        return 0.0
    return min(1.0, delta / full_credit)


def retention_factor(block_retention, baseline=0.85):
    if block_retention is None:
        return 1.0
    return max(0.5, min(1.2, block_retention / baseline))


def block_factor(block, criteria):
    """Apply an optional block prior; an unknown block receives no penalty."""
    if not criteria.get("block_ratio"):
        return 1.0
    return BLOCK_PRIORS.get(block or "", 1.0)


def ops_warning(ops, thresholds):
    """A warning string when the expression is long, or ``None``.

    Long expressions cost more to audit; complexity alone does not justify a penalty.
    """
    if ops is None:
        return None
    for limit in thresholds:
        if ops <= limit:
            if limit == thresholds[0]:
                return None
            return "ops<=%d (no score penalty; audit expression complexity)" % limit
    return "ops>%d: long expression, consider splitting before it becomes unauditable" % thresholds[-1]


def benchmark_rows(ids, registry_path=None, pnl_dir=None, criteria=None,
                   block_retention=None):
    import alpha_registry

    criteria = criteria or dict(DEFAULTS)
    latest = {}
    for entry in alpha_registry.load_entries(registry_path):
        key = entry.get("alpha_id") or entry.get("key")
        if key:
            latest[key] = entry

    directory = pnl_dir or os.path.join(research_paths.ensure_dir(), "pnl")
    series = {}
    for key, entry in latest.items():
        if str(entry.get("status") or "").upper() in ("ACTIVE", "SUBMITTED"):
            values = legs.load_pnl(key, directory)
            if values:
                series[key] = values
    dates = None
    for values in series.values():
        dates = set(values) if dates is None else dates & set(values)
    dates = sorted(dates or [])
    book_daily = [sum(values[day] for values in series.values()) for day in dates]
    n = len(series)

    rows = []
    for alpha_id in ids:
        entry = latest.get(alpha_id) or {}
        metrics = entry.get("metrics") or {}
        split = entry.get("out_of_sample") or {}
        train = (split.get("train") or {}).get("sharpe")
        test = (split.get("test") or {}).get("sharpe")
        expression = entry.get("expression") or ""
        ops = expression.count("(")
        candidate = legs.load_pnl(alpha_id, directory)
        delta = rho = None
        if candidate and dates:
            common = [day for day in dates if day in candidate]
            if len(common) >= 120:
                book_slice = [sum(values[day] for values in series.values()) for day in common]
                values = [candidate[day] for day in common]
                rho = legs.correlation(values, book_slice)
                now, after = legs.sharpe_of(book_slice), legs.sharpe_of(
                    [(b + v) / (n + 1) for b, v in zip(book_slice, values)])
                if now is not None and after is not None:
                    delta = after - now

        fitness = metrics.get("fitness")
        block = entry.get("block") or mechanisms.block_of(expression)
        factors = {
            "shrink": shrink_factor(train),
            "contribution": contribution_factor(delta, criteria["delta_full_credit"]),
            "retention": retention_factor(
                (block_retention or {}).get(block or "", None)),
            "block": block_factor(block, criteria),
        }
        weights = {
            "shrink": criteria["weight_shrink"],
            "contribution": criteria["weight_contribution"],
            "retention": criteria["weight_retention"],
            "block": criteria["weight_block"],
        }
        combined = 1.0
        for name, value in factors.items():
            combined *= value ** weights[name]
        internal = None if fitness is None else round(fitness * combined, 3)
        rows.append({
            "alpha_id": alpha_id,
            "platform_fitness": fitness,
            "internal_fitness": internal,
            "train_sharpe": train,
            "test_sharpe": test,
            "retention": None if not train or test is None else round(test / train, 2),
            "rho_book": None if rho is None else round(rho, 3),
            "delta": None if delta is None else round(delta, 4),
            "ops": ops,
            "ops_warning": ops_warning(ops, criteria["ops_warn"]),
            "factors": {k: round(v, 3) for k, v in factors.items()},
        })
    rows.sort(key=lambda row: -(row["internal_fitness"] or -99))
    return {
        "criteria": criteria,
        "book_members": n,
        "rows": rows,
        "note": ("internal_fitness = platform_fitness * shrink * contribution * retention"
                 + (" * block" if criteria.get("block_ratio") else "") + ". "
                 "Shrink and optional block factors are scoring priors; "
                 "operator count produces auditability warnings without a score penalty."),
    }


def cmd_benchmark(args, _sink):
    criteria = refine_criteria(getattr(args, "criteria", None))
    ids = list(getattr(args, "ids", None) or [])
    if getattr(args, "batch", None):
        path = os.path.expanduser(args.batch)
        try:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, ValueError) as exc:
            raise ValueError("cannot read batch file %s: %s" % (path, exc))
        ids.extend(row.get("alpha_id") for row in document.get("rows") or []
                   if row.get("alpha_id"))
    ids = [alpha_id for alpha_id in dict.fromkeys(ids) if alpha_id and alpha_id != "None"]
    if not ids:
        raise ValueError("nothing to benchmark: pass --batch or --ids")
    return benchmark_rows(ids, getattr(args, "registry", None), criteria=criteria)
