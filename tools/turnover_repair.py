#!/usr/bin/env python3
"""Find alphas whose Fitness is being held down by turnover.

Fitness is ``S * sqrt(R / max(T, 0.125))``, so turnover above 0.125 sits directly in
the denominator: a strategy that trades twice as much is scored as though it earned
half as much per unit of trading, even when its raw return is higher. Smoothing the
signal (a larger decay, or averaging a noisy term) lowers turnover and usually
lowers Sharpe a little — the question is whether the denominator gain beats the
numerator loss, and that can be estimated before spending a simulation.

The output deliberately does not predict the gain. An earlier version did, by
scaling from one repair observed in this ledger (turnover x0.51, Sharpe x0.91), and
it was wrong on three of four alphas: smoothing cost 17 percent of Sharpe rather
than 9, and the return fell with it, so the numerator loss ate the denominator gain.

What is computed instead is the **break-even Sharpe**: holding volatility roughly
constant, return scales with Sharpe, so with turnover at the floor

    F' = S' * sqrt(R * S' / (S * 0.125))

and setting F' = F solves for the Sharpe below which the repair loses:

    S'_break_even = (F * sqrt(S * 0.125 / R)) ^ (2/3)

That number is a property of the current alpha, not a guess about the future. If a
decay increase is likely to cost more Sharpe than the gap between the current and
break-even values, the repair is not worth a simulation — which is the decision the
tool exists to support.

Measured outcome, so the number is not over-read: eight alphas in this ledger have
been through this repair. Net Fitness changes were +0.22, +0.10, +0.06, +0.03,
+0.01, -0.01, -0.02 and -0.12 — median +0.03. The headroom column ranged 0.34 to
0.61 across them and never paid out in full, because Sharpe fell 0.25 to 0.46 each
time rather than the small loss the model assumes. The room is therefore an upper
bound on the payoff, not a forecast, and smoothing is worth one simulation only when
the alpha is isolated enough that another variant of it is itself valuable.

An alpha below the floor gets no suggestion: turnover is already free there, and the
only way to raise its Fitness is to earn more.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402

FLOOR = 0.125


def fitness(sharpe, returns, turnover):
    """Fitness under the platform's own formula."""
    if None in (sharpe, returns, turnover):
        return None
    denominator = max(turnover, FLOOR)
    if denominator <= 0 or returns <= 0:
        return None
    return sharpe * (returns / denominator) ** 0.5


def score_row(entry):
    metrics = entry.get("metrics") or {}
    sharpe = metrics.get("sharpe")
    returns = metrics.get("returns")
    turnover = metrics.get("turnover")
    current = fitness(sharpe, returns, turnover)
    if current is None or turnover <= FLOOR:
        return None
    ceiling = fitness(sharpe, returns, FLOOR)
    # Sharpe below which moving turnover to the floor stops paying, with return
    # scaling with Sharpe at constant volatility.
    inner = current * (sharpe * FLOOR / returns) ** 0.5 if returns else None
    break_even = round(inner ** (2.0 / 3.0), 4) if inner else None
    return {
        "alpha_id": entry.get("alpha_id"),
        "family": entry.get("family"),
        "sharpe": sharpe,
        "returns": returns,
        "turnover": turnover,
        "fitness": round(current, 4),
        "ceiling": round(ceiling, 4) if ceiling else None,
        "break_even_sharpe": break_even,
        "sharpe_room": round(sharpe - break_even, 4) if break_even else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-fitness", type=float, default=1.0)
    parser.add_argument("--min-turnover", type=float, default=FLOOR)
    parser.add_argument("--min-room", type=float, default=0.05,
                        help="Minimum Sharpe headroom above break-even.")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    path = args.registry or alpha_registry.default_registry_path()
    entries = alpha_registry.load_entries(path)
    rows = []
    for entry in alpha_registry.unique_entries(entries):
        row = score_row(entry)
        if row is None:
            continue
        if row["fitness"] < args.min_fitness:
            continue
        if row["turnover"] < args.min_turnover:
            continue
        if row["sharpe_room"] is None or row["sharpe_room"] < args.min_room:
            continue
        rows.append(row)
    rows.sort(key=lambda row: -row["sharpe_room"])

    print("%d alphas sit above the %.3f floor with Sharpe room >= %.2f"
          % (len(rows), FLOOR, args.min_room))
    print("(room = how much Sharpe the repair may cost before it stops paying)")
    print()
    print("%-10s %-9s %6s %6s %7s %6s %9s %6s"
          % ("alpha", "family", "S", "R", "T", "now", "break-even", "room"))
    for row in rows[: args.limit]:
        print("%-10s %-9s %6.2f %6.3f %7.4f %6.2f %9.2f %+6.2f"
              % (row["alpha_id"], str(row["family"])[:9], row["sharpe"],
                 row["returns"], row["turnover"], row["fitness"],
                 row["break_even_sharpe"], row["sharpe_room"]))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=1)
        print()
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())