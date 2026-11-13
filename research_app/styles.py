"""What each alpha compares against: other stocks, its own history, or nothing.

A cross-sectional rank, a rank against a value's own history, and a historical
change ask different questions of the same input. The comparison set can change
the resulting signal even when the underlying variable stays the same.

That makes "which comparison does this expression use" a property of the whole book,
worth measuring once instead of rediscovering per batch. This module classifies an
expression by the outermost comparison it performs:

``history`` it compares a value to its own past (``ts_rank``, ``ts_delta``,
            ``ts_zscore``, ``ts_argmax``, ``ts_av_diff``, ``ts_returns``).
``window``  it uses history as a window (``ts_mean``, ``ts_sum``, ``ts_std_dev``)
            without comparing against it.
``static``  only cross-sectional comparisons and cleaning, such as a fixed ratio.
``level``   no comparison at all: pure arithmetic.

Known blind spots, stated rather than hidden: a windowed aggregate of a *flow*
(``ts_sum(returns, 5)``) is a change in substance but not in this vocabulary, so it
is not counted as ``time``; ``group_rank`` counts as ``cross`` even though the
grouping is by industry rather than the whole market. The label is a screen for
"where has nobody re-read this variable against its own history", not a taxonomy.
"""

from __future__ import annotations

import re

#: Operators that compare a value to its *own past*: a percentile, a change, a gap.
HISTORY_OPERATORS = (
    "ts_rank", "ts_delta", "ts_zscore", "ts_argmax", "ts_argmin",
    "ts_av_diff", "ts_returns", "ts_ir", "ts_skewness", "ts_kurtosis",
)
#: Operators that use history but do not compare against it: smoothing, windows,
#: spreads. `ts_sum(returns, 5)` is a five-day move, not a percentile of one.
WINDOW_OPERATORS = (
    "ts_mean", "ts_sum", "ts_std_dev", "ts_min", "ts_max", "ts_median",
    "ts_decay_linear", "ts_decay_exp_window", "ts_delay", "ts_quantile",
    "ts_scale", "ts_corr", "ts_covariance", "ts_regression", "ts_product",
)
#: Cleaning operators: transparent for this question, like a cast.
CLEAN_OPERATORS = ("ts_backfill", "ts_step", "ts_count_nans", "ts_rank_by_side")
CROSS_OPERATORS = (
    "group_rank", "group_zscore", "group_neutralize", "zscore", "quantile",
    "rank", "rank_by_side",
)

_OPERATOR_PATTERN = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(")


def operators(expression):
    """Every operator name called in an expression, in order of appearance."""
    return [name for name in _OPERATOR_PATTERN.findall(expression or "")
            if name not in {"subindustry", "sector", "industry", "country"}]


def classify(expression):
    """Return ``{"comparison", "time", "cross", "operators"}`` for one expression.

    ``comparison`` is ``time`` when any self-history operator appears, ``cross`` when
    only cross-sectional ones do, and ``level`` when neither does. An expression that
    uses both is reported as ``time`` *and* keeps ``cross`` true: the interesting case
    for re-estimation is the one that has never been read against its own past.
    """
    names = operators(expression)
    history = [name for name in names if name in HISTORY_OPERATORS]
    window = [name for name in names if name in WINDOW_OPERATORS]
    cross = [name for name in names if name in CROSS_OPERATORS]
    if history:
        label = "history"
    elif window:
        label = "window"
    elif cross:
        label = "static"
    else:
        label = "level"
    return {"comparison": label, "history": history, "window": window,
            "cross": cross, "operators": sorted(set(names))}


def summarize(entries):
    """Group ledger entries by comparison and report the counts plus the level list.

    Entries without an expression are skipped instead of counted as ``level``: a
    missing expression is missing evidence, not evidence of a level construct.
    """
    by_comparison = {"history": 0, "window": 0, "static": 0, "level": 0}
    rows = []
    skipped = []
    for entry in entries:
        expression = entry.get("expression") or ""
        if not expression.strip():
            skipped.append(entry.get("alpha_id"))
            continue
        verdict = classify(expression)
        by_comparison[verdict["comparison"]] += 1
        metrics = entry.get("metrics") or {}
        train = (entry.get("out_of_sample") or {}).get("train") or {}
        rows.append({
            "alpha_id": entry.get("alpha_id"),
            "status": entry.get("status"),
            "family": entry.get("family"),
            "comparison": verdict["comparison"],
            "train_fitness": train.get("fitness"),
            "fitness": metrics.get("fitness"),
            "sharpe": metrics.get("sharpe"),
            "expression": expression,
        })
    order = {"static": 0, "level": 1, "window": 2, "history": 3}
    rows.sort(key=lambda row: (order.get(row["comparison"], 9), -(row["fitness"] or -9)))
    return {"counts": by_comparison, "rows": rows, "skipped": skipped,
            "measured": len(rows)}