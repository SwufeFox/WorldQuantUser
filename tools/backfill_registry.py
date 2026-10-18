"""One-off backfill of the alpha registry from verifiable sources.

Two sources, both checked rather than remembered:

1. Alphas that exist on the BRAIN account. The expression, settings, metrics and
   platform checks are re-read from the API right now, so the ledger cannot drift
   from what the platform actually holds.
2. Alpha ids already written into this repo's ``research/*-results.json`` files,
   paired index-wise with the matching ``research/*.json`` config so each record
   gets the expression it was simulated from.

Annotations (idea / mechanism / note) are written by hand here on purpose: the
ledger should record *why* something mattered, and that judgement cannot be
derived from the data.

Run: ``python3 tools/backfill_registry.py``
"""

from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import machine_lib as ml  # noqa: E402
import research_paths  # noqa: E402

RESEARCH = research_paths.research_dir()

# Alphas worth remembering, with the annotation that makes them meaningful.
# Expressions were re-read from the platform, so these labels sit next to the
# real expression and can be checked against it.
ACCOUNT_ALPHAS = {
    "d5KVgrbJ": {
        "idea": "main line",
        "mechanism": "overnight/intraday reversal composite plus liabilities/assets",
        "verdict": "candidate",
        "note": "Main research line: 10d/42d overnight-minus-intraday reversal, "
                "subindustry neutralized, plus a raw liabilities/assets leg. "
                "Strongest known member of the existing family.",
    },
    "3qVNpxkX": {
        "idea": "main line / decomposition",
        "mechanism": "overnight-minus-intraday reversal only",
        "verdict": "candidate",
        "note": "Ablation of d5KVgrbJ: the reversal leg alone, no leverage term. "
                "Kept to show how much of the composite comes from price structure.",
    },
    "2rjnEe1J": {
        "idea": "main line / decomposition",
        "mechanism": "liabilities/assets level",
        "verdict": "candidate",
        "note": "Ablation of d5KVgrbJ: the leverage leg alone. Very low turnover "
                "(0.017), which is why it survived fit checks.",
    },
    "d51oN1O2": {
        "idea": "IV skew x leverage",
        "mechanism": "implied-vol call-put spread modulated by a leverage proxy",
        "verdict": "rejected",
        "note": "High headline numbers but fails CONCENTRATED_WEIGHT and "
                "LOW_SUB_UNIVERSE_SHARPE: the signal is carried by a small group, "
                "so it is not a usable standalone despite the score.",
    },
    "6XAY8n0E": {
        "idea": "Idea 30 (leverage x FCF revision)",
        "mechanism": "leverage level gated by FCFPS estimate revision",
        "verdict": "candidate",
        "note": "Adding the high-leverage x positive-revision interaction to the "
                "additive baseline. The interaction alone scored far lower; the "
                "value is in the interaction, not the standalone part.",
    },
    "XgRYr31m": {
        "idea": "Idea 30 (leverage x FCF revision, lagged)",
        "mechanism": "same interaction with leverage frozen 63 days earlier",
        "verdict": "candidate",
        "note": "Lagging the leverage term to 63 days back changed nothing "
                "materially: evidence the interaction is not a same-day artifact.",
    },
    "P0Mq0wEM": {
        "idea": "leverage level, size-bucketed",
        "mechanism": "liabilities/assets, neutralized inside market-cap deciles",
        "verdict": "candidate",
        "note": "Minimal reference alpha used to verify the toolchain end to end. "
                "Group-neutralizing by size bucket rather than subindustry keeps "
                "the leverage ordering but removes the size bet.",
    },
    "xA19XpVn": {
        "idea": "Idea 44 round 1",
        "mechanism": "63-day FCF-per-share estimate revision",
        "verdict": "watch",
        "note": "Best field idea of the round-1 sweep but still below the bar "
                "(sharpe 0.80). Recorded because the field pair itself is usable; "
                "the expression is what failed, not the data.",
    },
    "88v7gGzm": {
        "idea": "round 1 (customer/supplier graph)",
        "mechanism": "customer count relative to competitor count",
        "verdict": "watch",
        "note": "Second-best round-1 idea at 0.73 with very low turnover. The "
                "relationship fields are legitimate; the level form is too weak.",
    },
}

# research file prefix -> idea label
FILE_IDEAS = {
    "cfo-surprise-screen": "Idea 44 (CFO surprise)",
    "supplier-finance-proxy-screen": "Idea 45 (supplier finance proxy)",
    "accrual-state-reversal-screen": "Idea 46 (accrual reversal)",
    "capex-investment-discipline-screen": "Idea 47 fallback (CapEx discipline)",
    "idea48-asset-growth-revenue-conversion-screen": "Idea 48 (asset growth vs revenue)",
    "orderflow-proxy-screen": "order-flow proxy (no-go)",
    "orderflow-proxy-followup": "order-flow proxy (no-go)",
    "orderflow-proxy-residual-check": "order-flow proxy (no-go)",
}


def find_alpha_records(node):
    """Find the first list of dicts that look like alpha results."""
    if isinstance(node, list):
        if node and isinstance(node[0], dict) and (
                "alpha_id" in node[0] or "sharpe" in node[0]):
            return node
        for item in node:
            found = find_alpha_records(item)
            if found:
                return found
    elif isinstance(node, dict):
        for key in ("results", "alphas"):
            if key in node:
                found = find_alpha_records(node[key])
                if found:
                    return found
        for value in node.values():
            found = find_alpha_records(value)
            if found:
                return found
    return None


def expressions_from_config(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return []
    alphas = payload.get("alphas")
    if isinstance(alphas, list):
        result = []
        for item in alphas:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                result.append(item.get("expression") or "")
            else:
                result.append("")
        return result
    return []


def record_research_files(session, registry_path):
    """Seed from research/*-results.json paired with their configs."""
    added = 0
    seen = set()
    for results_path in sorted(glob.glob(os.path.join(RESEARCH, "*-results.json"))):
        stem = os.path.basename(results_path)[:-len("-results.json")]
        config_path = os.path.join(RESEARCH, stem + ".json")
        try:
            with open(results_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        records = find_alpha_records(payload)
        if not records:
            continue
        expressions = expressions_from_config(config_path) if os.path.exists(config_path) else []
        idea = FILE_IDEAS.get(stem)
        for index, record in enumerate(records):
            alpha_id = record.get("alpha_id")
            if not alpha_id or alpha_id in seen:
                continue
            seen.add(alpha_id)
            expression = expressions[index] if index < len(expressions) else ""
            sharpe = record.get("sharpe")
            fitness = record.get("fitness")
            verdict = "candidate" if (
                isinstance(sharpe, (int, float)) and isinstance(fitness, (int, float))
                and sharpe >= 1.25 and fitness >= 1.0
            ) else "rejected"
            name = record.get("name")
            note = name
            if isinstance(sharpe, (int, float)):
                note = "%s -- screened out (sharpe %.2f, fitness %.2f)" % (
                    name or "variant", sharpe,
                    fitness if isinstance(fitness, (int, float)) else float("nan"))
            entry = alpha_registry.build_entry(
                {
                    "alpha_id": alpha_id,
                    "expression": expression,
                    "settings": record.get("settings") or {},
                    "metrics": record,
                    "checks": {},
                    "status": "COMPLETE" if record.get("sharpe") is not None else None,
                },
                idea=idea,
                mechanism=name,
                verdict=verdict,
                note=note,
                source="backfill:research",
                extra={"source_file": os.path.basename(results_path),
                       "record": record},
            )
            _, wrote = alpha_registry.record_entry(entry, registry_path)
            added += 1 if wrote else 0
            print("  research %-9s %-46s %s" % (alpha_id, (name or "")[:46], verdict))
    return added


def record_account_alphas(session, registry_path):
    """Seed from the platform, re-reading every value instead of trusting notes."""
    added = 0
    for alpha_id, annotation in ACCOUNT_ALPHAS.items():
        try:
            payload = ml.fetch_alpha_metrics(session, alpha_id)
        except Exception as exc:  # noqa: BLE001 - a missing alpha must not abort
            print("  SKIP %-9s %s" % (alpha_id, exc))
            continue
        in_sample = payload.get("is") or {}
        expression = (payload.get("regular") or {}).get("code") or ""
        entry = alpha_registry.build_entry(
            {
                "alpha_id": payload.get("id") or alpha_id,
                "expression": expression,
                "settings": payload.get("settings") or {},
                "metrics": in_sample,
                "checks": in_sample.get("checks") or [],
                "status": payload.get("status"),
            },
            idea=annotation["idea"],
            mechanism=annotation["mechanism"],
            verdict=annotation["verdict"],
            note=annotation["note"],
            source="backfill:account",
        )
        _, wrote = alpha_registry.record_entry(entry, registry_path)
        added += 1 if wrote else 0
        print("  account  %-9s sharpe %-5s %s" % (
            alpha_id, in_sample.get("sharpe"), annotation["verdict"]))
    return added




def record_account_scan(session, registry_path, min_sharpe=1.25, min_fitness=1.0,
                        limit=40):
    """Record clean, strong alphas the curated list does not cover.

    The platform already stores all 547 alphas; copying them into the ledger
    would turn a memory aid into a dump. So this keeps only alphas that pass
    every platform check and clear the bar, deduped by expression, capped at
    ``limit`` and ordered by fitness. Anything already in the ledger is skipped.
    """
    existing = alpha_registry.load_entries(registry_path)
    known_ids = {entry.get("alpha_id") for entry in existing}
    known_expr = {entry.get("expression_hash") for entry in existing}

    candidates = []
    offset = 0
    while len(candidates) < 400:
        batch = ml.list_alphas(session, limit=100, offset=offset,
                               status="UNSUBMITTED", min_sharpe=min_sharpe,
                               min_fitness=min_fitness)
        if not batch:
            break
        for alpha in batch:
            checks = (alpha.get("is") or {}).get("checks") or []
            if any(str(check.get("result")).upper() in {"FAIL", "ERROR"}
                   for check in checks):
                continue
            if alpha.get("id") in known_ids:
                continue
            candidates.append(alpha)
        if len(batch) < 100:
            break
        offset += 100

    candidates.sort(key=lambda alpha: (alpha.get("is") or {}).get("fitness") or 0,
                    reverse=True)
    added = 0
    skipped_dup = 0
    for alpha in candidates:
        if added >= limit:
            break
        alpha_id = alpha.get("id")
        try:
            payload = ml.fetch_alpha_metrics(session, alpha_id)
        except Exception as exc:  # noqa: BLE001
            print("  SKIP %-9s %s" % (alpha_id, exc))
            continue
        in_sample = payload.get("is") or {}
        expression = (payload.get("regular") or {}).get("code") or ""
        digest = alpha_registry.expression_hash(expression)
        if digest in known_expr:
            skipped_dup += 1
            continue
        known_expr.add(digest)
        entry = alpha_registry.build_entry(
            {
                "alpha_id": alpha_id,
                "expression": expression,
                "settings": payload.get("settings") or {},
                "metrics": in_sample,
                "checks": in_sample.get("checks") or [],
                "status": payload.get("status"),
            },
            idea=None,
            mechanism="account scan: passes all checks, clears the bar",
            verdict="candidate",
            note="account scan: all platform checks pass and sharpe/fitness "
                 "clear the bar; no research label yet",
            source="backfill:account-scan",
        )
        _, wrote = alpha_registry.record_entry(entry, registry_path)
        added += 1 if wrote else 0
        print("  scan     %-9s sharpe %-5s fitness %-5s" % (
            alpha_id, in_sample.get("sharpe"), in_sample.get("fitness")))
    if skipped_dup:
        print("  (%d candidate(s) skipped as duplicate expressions)" % skipped_dup)
    return added



def main():
    registry_path = alpha_registry.default_registry_path()
    print("registry:", registry_path)
    session = ml.load_session()
    if session is None:
        session = ml.authenticate()

    print("account alphas:")
    account_added = record_account_alphas(session, registry_path)
    print("research files:")
    research_added = record_research_files(session, registry_path)
    print("account scan (clean alphas above the bar):")
    scan_added = record_account_scan(session, registry_path)

    entries = alpha_registry.load_entries(registry_path)
    summary = alpha_registry.summarize(entries)
    print()
    print("appended: account %d, research %d, scan %d"
          % (account_added, research_added, scan_added))
    print("registry now holds %d alphas" % summary["total"])
    print("by verdict:", summary["by_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
