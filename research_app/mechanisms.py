"""Mechanism identity: what the book and the ledger actually bet on.

The campaign counts candidates by *independent economic mechanism*, not by expression
count. Two things were missing to make that countable offline:

* a mechanism key. Family labels are too coarse (11 buckets, one of which holds half the
  book) and PnL correlation is too late (it needs a simulation). The key used here is
  ``(economic block, AST skeleton)``: the block says what information is being read, the
  skeleton says how. Fields and literals are blanked, so a window change or a field swap
  stays inside the same mechanism - which is exactly the rule "100 parameter variants
  count as one".
* a convergence signal. On 2026-11-08 the book held 56 alphas but only 38 distinct
  skeletons, and the last 100 simulated arms were 93% structurally novel while only one
  of the last seven batches produced anything additive. Structural novelty is therefore
  a weak trigger; the monitor keys on outcomes, and this module reports both so the gap
  stays visible.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath

import fastexpr_ast

#: Which information the expression reads, in the order that decides ties: the first
#: rule that matches wins, so the list runs from most specific to most generic.
BLOCK_RULES = (
    ("derivative_pricing", ("implied_volatility", "option", "straddle", "pcr_")),
    ("analyst_dispersion", ("eps_high", "eps_low", "sales_high", "sales_low",
                            "dts_spe", "_std", "_high", "_low")),
    ("analyst_yield", ("eps_mean", "median_eps", "afv4_eps", "bvps", "cfps",
                       "estimate")),
    ("analyst_coverage", ("_number", "count")),
    ("turnover_structure", ("sharesout",)),
    ("volatility_structure", ("ts_corr(abs(returns)", "ts_std_dev(ts_sum(returns",
                              "historical_volatility")),
    ("price_path", ("ts_arg_max", "ts_arg_min", "open / ts_delay(close",
                    "close / open", "high / low")),
    ("debt_structure", ("debt_st", "debt_lt", "liabilities")),
    ("value_quality", ("equity / cap", "atq", "assets", "revenue", "cogs", "saleq",
                       "ebitda")),
    ("news_sentiment", ("news_", "scl12_", "sent")),
)


def block_of(expression):
    """The information block an expression reads (``value_quality`` when nothing matches)."""
    text = " ".join((expression or "").split())
    for name, needles in BLOCK_RULES:
        if any(needle in text for needle in needles):
            return name
    return "other"


def skeleton_of(expression):
    """AST skeleton with fields and literals blanked, or ``None`` when unparseable."""
    text = " ".join((expression or "").split())
    if not text:
        return None
    try:
        return fastexpr_ast.skeleton(fastexpr_ast.parse(text))
    except Exception:  # noqa: BLE001 - an unparseable row must not break the census
        return None


def mechanism_key(expression):
    """``(block, skeleton)``: the unit the campaign counts as one bet."""
    skeleton = skeleton_of(expression)
    if skeleton is None:
        return None
    return (block_of(expression), skeleton[:200])


def summarise(entries, marginals=None):
    """Group entries by mechanism and report size, families and outcomes.

    ``marginals`` maps alpha id to its marginal contribution to the book; when given, the
    median per mechanism is reported. Without it the table still answers "how many
    mechanisms do we have", which is the number the registry exists to make visible.
    """
    marginals = marginals or {}
    groups = defaultdict(lambda: {"members": [], "families": set(), "statuses": [],
                                  "fitness": [], "marginal": []})
    skipped = []
    for entry in entries:
        key = mechanism_key(entry.get("expression"))
        alpha_id = entry.get("alpha_id") or entry.get("key")
        if key is None:
            skipped.append(alpha_id)
            continue
        bucket = groups[key]
        bucket["members"].append(alpha_id)
        bucket["families"].add(entry.get("family") or "-")
        bucket["statuses"].append(str(entry.get("status") or "").upper())
        fitness = (entry.get("metrics") or {}).get("fitness")
        if isinstance(fitness, (int, float)):
            bucket["fitness"].append(fitness)
        if alpha_id in marginals:
            bucket["marginal"].append(marginals[alpha_id])

    def median(values):
        if not values:
            return None
        ordered = sorted(values)
        return round(ordered[len(ordered) // 2], 2)

    rows = []
    for (block, skeleton), bucket in groups.items():
        rows.append({
            "block": block,
            "members": len(bucket["members"]),
            "alpha_ids": bucket["members"][:6],
            "families": sorted(name for name in bucket["families"] if name != "-"),
            "median_fitness": median(bucket["fitness"]),
            "median_marginal": median(bucket["marginal"]),
            "skeleton": skeleton[:120],
        })
    rows.sort(key=lambda row: (-row["members"], row["block"]))
    by_block = defaultdict(lambda: {"mechanisms": 0, "members": 0, "marginal": []})
    for row in rows:
        bucket = by_block[row["block"]]
        bucket["mechanisms"] += 1
        bucket["members"] += row["members"]
        if row["median_marginal"] is not None:
            bucket["marginal"].append(row["median_marginal"])
    blocks = [{"block": name, "mechanisms": value["mechanisms"],
               "members": value["members"],
               "median_marginal": median(value["marginal"])}
              for name, value in by_block.items()]
    blocks.sort(key=lambda row: -row["members"])
    return {"mechanisms": len(rows), "members": len(entries) - len(skipped),
            "skipped": skipped[:10], "blocks": blocks, "rows": rows}


def convergence(entries, marginals=None, recent_batches=8):
    """Per-batch outcome signals, for deciding whether to widen the search.

    A batch is the ledger's ``idea`` label, ordered by first record. For each batch:
    how many arms, how many distinct mechanisms, how many mechanisms the book did not
    already hold, and how many members turned out additive (marginal >= threshold).
    The third number is the one worth triggering on - structural novelty sat at 93% even
    across the run of batches that produced nothing.
    """
    marginals = marginals or {}
    batches = defaultdict(list)
    for entry in entries:
        idea = (entry.get("idea") or "").strip() or "untagged"
        batches[idea].append(entry)
    book_keys = {mechanism_key(e.get("expression")) for e in entries
                 if str(e.get("status") or "").upper() == "ACTIVE"}
    book_keys.discard(None)

    rows = []
    for idea, members in batches.items():
        keys = [mechanism_key(e.get("expression")) for e in members]
        keys = [key for key in keys if key]
        if not keys:
            continue
        # A positive marginal alone is not enough: an alpha with train Sharpe 0.5 and no
        # correlation scores +0.5 and is useless. Same floor as the pool screen.
        additive = 0
        for entry in members:
            key = entry.get("alpha_id") or entry.get("key")
            marginal = marginals.get(key)
            train_sharpe = ((entry.get("out_of_sample") or {}).get("train") or {}).get("sharpe")
            if (marginal is not None and marginal >= 0.3
                    and isinstance(train_sharpe, (int, float)) and train_sharpe >= 1.0):
                additive += 1
        rows.append({
            "batch": idea[:30],
            "arms": len(members),
            "mechanisms": len(set(keys)),
            "new_to_book": len(set(keys) - book_keys),
            "additive": additive,
            "first_record": min((e.get("recorded_at") or "")[:10] for e in members),
        })
    rows.sort(key=lambda row: row["first_record"])
    tail = rows[-recent_batches:]
    productive = [row for row in tail if row["additive"] > 0]
    return {"batches": len(rows), "recent": tail,
            "recent_productive": len(productive),
            "recent_additive": sum(row["additive"] for row in tail),
            "book_mechanisms": len(book_keys)}