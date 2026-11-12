"""An internal bar: the platform's Fitness, discounted by what our own data says decays.

The platform's Fitness is a *train-heavy* number: the full window is four-fifths training,
so an arm can clear the floor while its out-of-sample behaviour is poor. The campaign
needs a bar of its own - one that is strictly harder to clear - and the question is what
belongs in it.

Two candidates were proposed and both failed their own test on this ledger:

* **operator count**. The intuition is that a 40-operator expression is overfitted. It is
  not, here: median retention is 0.45 for 1-4 operators, 0.75 for 5-8, 0.73 for 9-16,
  0.81 for 17-32 and 1.83 above 32. Complexity does *not* predict decay, and a penalty
  would push work toward the arms that decay most.
* **field crowding** (the "cold field" bonus). Also backwards: the least-crowded third of
  the ledger retains 0.52, the middle 0.74, the most-crowded 0.94. Crowded fields are
  crowded because they keep working.

What the ledger does support is a **hump in train Sharpe**: retention is 0.41 below 1.0,
0.77 in 1.0-1.5, **0.85 in 1.5-2.0**, 0.64 in 2.0-2.5 and 0.49 at 2.5 and above. Weak arms
are noise; very strong arms are the overfitted ones. That is the shape a benchmark should
encode, and it is the opposite of rewarding raw Fitness.

So this module keeps three factors, each with a stated basis:

* ``shrink`` - train-Sharpe hump (the table above);
* ``contribution`` - the book delta, because an alpha that does not move the book is not a
  finding however good its own Sharpe;
* ``retention`` - the block's median test/train ratio, as a family-level prior.

Operator count is *reported* and warned about (a 32-operator arm cannot be reasoned about
by a reader) but it does not multiply the score: a warning is honest, a penalty would be
a superstition with a number attached.
"""

import json
import os

import research_paths

from . import legs, mechanisms

#: Train-Sharpe hump, read off the ledger (retention by bucket, 1427 arms with a split).
#: Below 1.0 the arms are noise; above 2.5 they are the overfitted tail.
#: Measured retention / best retention, so the factor is a *ratio* rather than a guess:
#: 0.41/0.85 = 0.48, 0.77/0.85 = 0.91, 1.00, 0.64/0.85 = 0.75, 0.49/0.85 = 0.58.
SHRINK_BANDS = (
    (1.0, 0.48),
    (1.5, 0.91),
    (2.0, 1.00),
    (2.5, 0.75),
    (99.0, 0.58),
)

#: Median test/train Sharpe ratio per economic block, measured on 2026-11-12 over every
#: ledger arm that carries a train/test split and |train S| >= 0.3 (n in the comment).
#: The spread is large and it is *not* inversion: the train-test correlation stays
#: positive in every block (0.10 to 0.62), so these are shrinkage differences, not
#: sign flips. A block at 0.21 needs roughly five times the train Sharpe that a block at
#: 1.74 needs in order to deliver the same out-of-sample signal.
BLOCK_RATIO = {
    "price_path": (0.84, 318),
    "analyst_yield": (0.91, 268),
    "value_quality": (0.21, 187),
    "turnover_structure": (0.69, 185),
    "other": (0.10, 178),
    "volatility_structure": (0.67, 133),
    "derivative_pricing": (0.46, 117),
    "analyst_dispersion": (1.74, 97),
    "news_sentiment": (0.03, 32),
    "debt_structure": (0.25, 26),
}

#: The ledger-wide median ratio these block numbers are normalised against.
BLOCK_BASELINE = 0.75

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
    """How much a block's train Sharpe survives into the test period, versus the norm.

    Measured, not assumed: see ``BLOCK_RATIO``. Unknown blocks get 1.0 rather than a
    penalty, because "not in the table" is not evidence of decay.
    """
    if not criteria.get("block_ratio"):
        return 1.0
    entry = BLOCK_RATIO.get(block or "")
    if entry is None:
        return 1.0
    return max(0.25, min(2.0, entry[0] / BLOCK_BASELINE))


def ops_warning(ops, thresholds):
    """A warning string when the expression is long, or ``None``.

    Long expressions are hard to audit, which is a real cost to the reader - but the
    ledger says they are not more likely to decay, so this never changes the score.
    """
    if ops is None:
        return None
    for limit in thresholds:
        if ops <= limit:
            if limit == thresholds[0]:
                return None
            return "ops<=%d (no penalty; complexity does not predict decay here)" % limit
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
                 "The shrink factor follows the train-Sharpe hump measured on this ledger; "
                 "operator count is warned about but never scored, because it does not "
                 "predict decay here."),
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
