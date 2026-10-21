#!/usr/bin/env python3
"""Propose a ``family`` label for ledger entries that lack one.

Why a script and not a bulk edit: 319 of 481 keys carry no family, and the family
column is what independence gets counted on, so leaving it empty makes "how many
separate mechanisms" unanswerable. Filling it by hand is slow and unauditable.

How it decides: the family is read off the datafield namespace of the **first
leg**, because that leg is the carrier in every composite inspected so far — the
overnight/intraday composite is filed under oi, the leverage-times-revision
composite under leverage. It records the field that triggered the decision, and
writes with ``source: relabel:derived`` so a derived label stays distinguishable
from a human one.

What it refuses to do: guess. An expression with no recognisable namespace, or one
whose first leg cannot be parsed, stays unlabelled and is listed for review. A
wrong family silently corrupts the count; an absent one merely shows up as debt.
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

#: Datafield namespace to family. Order matters only within one leg: the first
#: match wins, and the table is read top to bottom.
NAMESPACE = (
    # Order is a priority, not a list of equals: a composite belongs to the
    # return source that carries it, and the price-structure families are
    # distinctive enough to claim an alpha before a fundamentals leg does.
    ("oi", ("overnight", "intraday", "ts_delay(close,1)")),
    ("option", ("implied_volatility", "pcr_", "forward_price",
                "parkinson_volatility")),
    ("event", ("mean_news", "rp_ess", "rp_nip", "rp_css", "nws18", "news_",
               "mean_event", "mean_entity", "story_event",
               "pct_change_thirty_minute")),
    ("attention", ("scl12", "snt_", "social")),
    # Leverage sits above analyst on purpose: the composites that combine a
    # distress leg with an estimate revision are carried by distress, and filing
    # them under analyst would split one bet across two families.
    ("leverage", ("liabilities", "cogs", "sga_expense", "debt_",
                  "equity/assets")),
    ("analyst", ("anl4_",)),
    ("risk", ("historical_volatility", "beta_last", "systematic_risk",
              "unsystematic_risk", "correlation_last")),
    # Relationship and network data has no standard factor bucket; filing it under
    # "other" keeps it countable without pretending it is a quality signal.
    ("other", ("rel_num_cust", "rel_num_comp", "rel_ret", "ownership_percent",
               "equity_method", "customergraph")),
    ("quality", ("fnd6_", "fnd2_", "lease", "pension", "goodwill", "intangible",
                 "deferred_tax", "valuation_allowance", "receivable",
                 "inventory", "oiadp", "comp_not_rec", "sbp_expense",
                 "recurring_assets_fair_value", "rank_derivative", "fscore",
                 "cashflow_efficiency", "earnings_certainty", "growth_potential",
                 "relative_valuation", "contingent_consideration")),
)


def classify(expression):
    """Return ``(family, evidence)`` or ``(None, reason)``.

    The whole expression is scanned rather than the first leg: a composite's
    carrier is not reliably first in the sum, and taking the first leg filed the
    overnight/intraday alphas under leverage because their leverage term happened
    to be written first. Priority resolves the overlap, and every family that
    matched is recorded so an override has something to look at.
    """
    body = " ".join((expression or "").split()).lower()
    hits = []
    for family, needles in NAMESPACE:
        for needle in needles:
            if needle in body:
                hits.append((family, needle))
                break
    # No momentum/reversal fallback on "ts_mean(returns": that fragment appears
    # inside volatility-shape composites, so it labels unrelated alphas. Those
    # entries stay blank, which is honest debt rather than a wrong label.
    if not hits:
        return None, {"reason": "no recognised namespace", "body": body[:120]}
    family, needle = hits[0]
    return family, {"field": needle,
                    "matched": [name for name, _ in hits],
                    "ambiguous": len(hits) > 1}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="Append derived labels to the ledger.")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--limit", type=int, default=8,
                        help="Examples to print per family (default 8).")
    args = parser.parse_args(argv)

    path = args.registry or alpha_registry.default_registry_path()
    entries = alpha_registry.load_entries(path)
    missing = [entry for entry in entries if not entry.get("family")]
    print("ledger %d entries, %d missing a family" % (len(entries), len(missing)))

    plan, unlabelled = {}, []
    for entry in missing:
        family, evidence = classify(entry.get("expression"))
        if family is None:
            unlabelled.append((entry, evidence))
        else:
            plan.setdefault(family, []).append((entry, evidence))

    print()
    for family in sorted(plan, key=lambda name: -len(plan[name])):
        rows = plan[family]
        print("%-10s %3d entries" % (family, len(rows)))
        for entry, evidence in rows[: args.limit]:
            print("    %-10s %-16s %s"
                  % (entry.get("alpha_id") or entry.get("key"),
                     evidence["field"], (entry.get("expression") or "")[:58]))
        if len(rows) > args.limit:
            print("    ... %d more" % (len(rows) - args.limit))

    if unlabelled:
        print()
        print("left unlabelled: %d" % len(unlabelled))
        for entry, evidence in unlabelled[: args.limit]:
            print("    %-10s %s" % (entry.get("alpha_id") or entry.get("key"),
                                    evidence.get("body", "")[:70]))

    if not args.write:
        print()
        print("dry run; pass --write to append these labels")
        return 0

    written = 0
    for family, rows in plan.items():
        for entry, evidence in rows:
            updated = alpha_registry.relabel(entry, family=family)
            updated["source"] = "relabel:derived"
            updated["note"] = (updated.get("note") or "") + (
                " [family derived from first leg: %s]" % evidence["field"]
                if evidence.get("field") else " [family derived]")
            _, did_write = alpha_registry.record_entry(updated, path)
            written += 1 if did_write else 0
    ambiguous = sum(1 for rows in plan.values() for _, ev in rows if ev.get("ambiguous"))
    print()
    print("appended %d derived labels (%d had more than one namespace match)"
          % (written, ambiguous))
    summary = alpha_registry.unlabeled_summary(alpha_registry.load_entries(path))
    print("family coverage now:", summary["family_coverage"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())