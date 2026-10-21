#!/usr/bin/env python3
"""Report the qualifying pool as a distribution, the way a paper would.

Method borrowed from Kakushadze (2015), "101 Formulaic Alphas": that paper's Table 1
reports quartiles for Sharpe, turnover, holding period, cents-per-share, volatility,
return and pairwise correlation across its whole pool, and the correlation
*distribution* is a headline result rather than an afterthought.

The point is that per-alpha reporting hides what a pool is. Two pools can share a
best alpha and differ completely in spread: one narrow and correlated, one broad and
independent. The paper's 101 alphas are "mostly price-volume" - it also uses some
fundamental input, market cap, and an industry classification - and still show a
median pairwise correlation of 14.3 percent, which is only visible in a table like
this.

Measures follow the paper where the platform allows: holding period is 1/turnover,
volatility comes from the daily PnL series, and pairwise correlations are computed on
overlapping dates (theirs are on 1006 common observations).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import correlation  # noqa: E402
import research_paths  # noqa: E402


def quartiles(values):
    """Min, first quartile, median, mean, third quartile, max."""
    clean = sorted(value for value in values if isinstance(value, (int, float)))
    if not clean:
        return None
    count = len(clean)

    def pick(fraction):
        if count == 1:
            return clean[0]
        position = fraction * (count - 1)
        low = int(math.floor(position))
        high = min(low + 1, count - 1)
        return clean[low] + (clean[high] - clean[low]) * (position - low)

    return {
        "min": round(clean[0], 4),
        "q1": round(pick(0.25), 4),
        "median": round(pick(0.5), 4),
        "mean": round(sum(clean) / count, 4),
        "q3": round(pick(0.75), 4),
        "max": round(clean[-1], 4),
        "n": count,
    }


def load_pnls(alpha_ids, cache_dir, min_overlap=200):
    """Cached daily PnL only: this is a report, not a fetch job."""
    out = {}
    for alpha_id in alpha_ids:
        path = os.path.join(cache_dir, alpha_id + ".json")
        if not os.path.exists(path):
            continue
        try:
            series = json.loads(open(path, encoding="utf-8").read())
        except ValueError:
            continue
        if series:
            out[alpha_id] = {day: float(value) for day, value in series.items()}
    return out


def pnl_volatility(series):
    """Standard deviation of daily PnL changes, in the recordset's own units.

    Deliberately not converted to a percentage. The recordset is cumulative dollar
    PnL and the platform does not expose the capital behind it, so a "volatility"
    computed as a ratio of PnL levels is meaningless — an early version divided each
    day's level by the previous one and produced numbers from 9.7 to 1216. The figure
    here is comparable between alphas of this account and not to any published table.
    """
    values = [series[day] for day in sorted(series)]
    if len(values) < 3:
        return None
    changes = [current - previous for previous, current in zip(values, values[1:])]
    mean = sum(changes) / len(changes)
    variance = sum((value - mean) ** 2 for value in changes) / (len(changes) - 1)
    return math.sqrt(variance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-fitness", type=float, default=1.25)
    parser.add_argument("--min-overlap", type=int, default=200)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    path = args.registry or alpha_registry.default_registry_path()
    entries = [entry for entry in alpha_registry.unique_entries(
        alpha_registry.load_entries(path))
        if isinstance((entry.get("metrics") or {}).get("fitness"), (int, float))
        and entry["metrics"]["fitness"] >= args.min_fitness]

    metrics = [entry.get("metrics") or {} for entry in entries]
    sharpe = [m.get("sharpe") for m in metrics]
    turnover = [m.get("turnover") for m in metrics]
    returns = [m.get("returns") for m in metrics]
    holding = [1.0 / t if t else None for t in turnover]

    pair_returns = [m.get("returns") for m in metrics]

    cache = os.path.join(research_paths.ensure_dir(), "pnl")
    ids = [entry.get("alpha_id") for entry in entries if entry.get("alpha_id")]
    pnls = load_pnls(ids, cache)

    correlations = []
    for left, right in itertools.combinations(sorted(pnls), 2):
        x, y, _days = correlation.align(pnls[left], pnls[right], args.min_overlap)
        value = correlation.pearson(x, y) if x else None
        if value is not None:
            correlations.append(value)

    volatilities = [pnl_volatility(series) for series in pnls.values()]
    print("daily PnL std is in the recordset's own units: capital is not exposed, so")
    print("it cannot be turned into the paper's percent-of-capital volatility.")
    print()

    table = {
        "sharpe": quartiles(sharpe),
        "turnover": quartiles(turnover),
        "holding_days": quartiles(holding),
        "returns": quartiles(returns),
        "daily_pnl_std": quartiles(volatilities),
        "pairwise_correlation": quartiles(correlations),
    }

    labels = {
        "sharpe": "Sharpe",
        "turnover": "turnover",
        "holding_days": "holding days (1/turnover)",
        "returns": "annual return",
        "daily_pnl_std": "daily PnL std (own units)",
        "pairwise_correlation": "pairwise correlation",
    }
    print("pool: %d alphas at fitness >= %s, %d with cached PnL, %d pairs"
          % (len(entries), args.min_fitness, len(pnls), len(correlations)))
    print()
    print("%-28s %8s %8s %8s %8s %8s %8s" %
          ("measure", "min", "q1", "median", "mean", "q3", "max"))
    for key in ("sharpe", "turnover", "holding_days", "returns",
                "daily_pnl_std", "pairwise_correlation"):
        stats = table[key]
        if not stats:
            print("%-28s %8s" % (labels[key], "no data"))
            continue
        print("%-28s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f"
              % (labels[key], stats["min"], stats["q1"], stats["median"],
                 stats["mean"], stats["q3"], stats["max"]))
    print()
    print("for comparison, Kakushadze (2015) reports over its 101 alphas")
    print("  sharpe median 2.224, turnover median 0.4752, holding 2.10 days,")
    print("  pairwise correlation median 0.1431, mean 0.1586, max 0.8733")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(table, handle, ensure_ascii=False, indent=1)
        print()
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())