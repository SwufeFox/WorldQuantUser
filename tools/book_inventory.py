#!/usr/bin/env python3
"""Inventory the live book: what we actually hold, grouped by construction.

Why this exists: "48 alphas" is not a description of a book. The number that matters
is how many *constructions* those 48 reduce to, and the family graph answers that
without a single correlation request. This script writes the inventory to a dated
markdown file so the shape of the book can be read at a glance and compared later.

What it reports, and why each part earns its place:

* headline counts -- alphas, unique expressions, families, and the split by universe
  and neutralisation, because a book concentrated in one universe has one liquidity
  profile however many alphas it holds;
* one line per family -- size, universes, best Fitness, representative alpha, and the
  ledger's own mechanism label, so a 13-member family cannot hide among 31 singletons;
* the largest family called out separately -- a template instantiated over N fields is
  one idea wearing N names, and that is the single most useful thing to see;
* the backlog -- ledger entries that clear every recorded check but were never
  submitted, listed with their Fitness and settings. Self-correlation is *not*
  verified here (that costs a request per pair), so the section says so rather than
  implying these are submittable.

Usage: ``python3 tools/book_inventory.py [--out PATH]``
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import alpha_registry  # noqa: E402
import family_graph as fg  # noqa: E402
import fastexpr_ast as fa  # noqa: E402
import machine_lib as ml  # noqa: E402
import research_paths  # noqa: E402

BASE = "https://api.worldquantbrain.com"


def live_book(session):
    """Ids the platform currently reports as submitted, in stable order."""
    response = session.get(BASE + "/users/self/alphas",
                           params={"status": "ACTIVE", "limit": 100})
    return sorted({item["id"] for item in response.json().get("results", [])})


def build(session):
    """Read the live book and group it by construction, not by alpha.

    The live list comes from the platform and the structure comes from the ledger, so
    an alpha that is live but missing from the ledger is dropped rather than guessed
    at -- it has no expression to decompose. That gap is what :func:`backlog` exists
    to report.
    """
    by_id = {e.get("alpha_id"): e for e in alpha_registry.load_entries() if e.get("alpha_id")}
    ids = [i for i in live_book(session) if i in by_id]
    entries = [by_id[i] for i in ids]
    nodes = fg.build_nodes(entries)
    groups = fg.components(nodes, fg.build_edges(nodes))

    families = []
    for group in groups:
        stats = [((by_id[n].get("metrics") or {}), n) for n in group]
        best_fitness, best_id = max(stats, key=lambda pair: pair[0].get("fitness") or 0)
        universes = sorted({(by_id[n].get("settings") or {}).get("universe") for n in group})
        labels = collections.Counter((by_id[n].get("family") or "other") for n in group)
        try:
            fields = fa.field_names(fa.parse(by_id[best_id]["expression"]))
        except fa.ParseError:
            fields = []
        families.append({
            "members": sorted(group),
            "size": len(group),
            "universes": universes,
            "best_fitness": best_fitness.get("fitness"),
            "best_id": best_id,
            "best_sharpe": best_fitness.get("sharpe"),
            "label": labels.most_common(1)[0][0],
            "fields": fields[:4],
            "skeleton": fg.fa.skeleton(fa.parse(by_id[best_id]["expression"]))
            if by_id[best_id].get("expression") else "",
        })
    families.sort(key=lambda f: (-(f["best_fitness"] or 0), -f["size"]))
    return entries, families


def backlog(session, book_expressions):
    """Unsubmitted entries that pass every recorded check, newest first."""
    rows = []
    for entry in alpha_registry.load_entries():
        if (entry.get("status") or "").upper() == "ACTIVE":
            continue
        if entry.get("abandoned") or not entry.get("alpha_id"):
            continue
        if entry.get("expression") in book_expressions:
            continue
        if entry.get("failing_checks"):
            continue
        fitness = (entry.get("metrics") or {}).get("fitness") or 0
        if fitness < 1.2:
            continue
        rows.append(entry)
    rows.sort(key=lambda e: -((e.get("metrics") or {}).get("fitness") or 0))
    return rows


def distant(session, book_entries, min_fitness=1.2):
    """Unsubmitted candidates whose construction shares nothing with the book.

    These are the entries worth verifying first: the backlog sorted by Fitness is
    dominated by near-duplicates of families already in the book, which is exactly the
    population self-correlation will reject. A candidate with no sibling, no ancestor
    and no shared leg cannot be a copy of something already submitted.
    """
    book_nodes = fg.build_nodes(book_entries)
    book_edges = fg.build_edges(book_nodes)
    book_neighbours = collections.defaultdict(set)
    for left, right, _kind in book_edges:
        book_neighbours[left].add(right)
        book_neighbours[right].add(left)
    book_skeletons = {node["skeleton"] for node in book_nodes.values()}
    book_legs = set()
    for node in book_nodes.values():
        book_legs |= node["legs"]

    seen = {e.get("expression") for e in book_entries}
    out = []
    for entry in alpha_registry.load_entries():
        expression = entry.get("expression")
        if not expression or expression in seen:
            continue
        if (entry.get("status") or "").upper() == "ACTIVE" or entry.get("abandoned"):
            continue
        fitness = (entry.get("metrics") or {}).get("fitness") or 0
        if fitness < min_fitness:
            continue
        try:
            program = fa.parse(expression)
        except fa.ParseError:
            continue
        if fa.skeleton(program) in book_skeletons:
            continue
        if fg.legs(program) & book_legs:
            continue
        seen.add(expression)
        out.append(entry)
    out.sort(key=lambda e: -((e.get("metrics") or {}).get("fitness") or 0))
    return out


def render(entries, families, pending=(), far=()):
    """Render the inventory as markdown: what we hold, grouped by what it is.

    `pending` and `far` are listed rather than merged: an alpha that is waiting and an
    alpha that is structurally distant from the book are different kinds of not-yet,
    and collapsing them into "candidates" loses the reason someone would look.
    """
    today = datetime.date.today().isoformat()
    lines = ["# Book inventory — %s" % today, ""]
    universes = collections.Counter((e.get("settings") or {}).get("universe") for e in entries)
    neuts = collections.Counter((e.get("settings") or {}).get("neutralization") for e in entries)
    labels = collections.Counter((e.get("family") or "other") for e in entries)
    fitness = sorted(((e.get("metrics") or {}).get("fitness") or 0) for e in entries)

    lines += [
        "| | |",
        "| --- | --- |",
        "| ACTIVE alphas | %d |" % len(entries),
        "| unique expressions | %d |" % len({e.get("expression") for e in entries}),
        "| **constructions (families)** | **%d** |" % len(families),
        "| universe | %s |" % ", ".join("%s %d" % (k, v) for k, v in universes.most_common()),
        "| neutralisation | %s |" % ", ".join("%s %d" % (k, v) for k, v in neuts.most_common()),
        "| ledger mechanism labels | %s |" % ", ".join("%s %d" % (k, v) for k, v in labels.most_common()),
        "| Fitness | min %.2f / median %.2f / max %.2f |"
        % (fitness[0], fitness[len(fitness) // 2], fitness[-1]),
        "",
        "## Families",
        "",
        "| # | size | universes | best F | best S | label | representative |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, family in enumerate(families, 1):
        lines.append("| %d | %d | %s | %.2f | %.2f | %s | `%s` |" % (
            index, family["size"], "/".join(u or "?" for u in family["universes"]),
            family["best_fitness"] or 0, family["best_sharpe"] or 0,
            family["label"], family["best_id"]))

    biggest = max(families, key=lambda f: f["size"])
    lines += ["", "## The largest family", "",
              "%d of %d alphas (**%.0f%% of the book**) come from one construction:"
              % (biggest["size"], len(entries), 100.0 * biggest["size"] / len(entries)), "",
              "```", biggest["skeleton"][:220], "```", "",
              "Representative fields: %s" % ", ".join(biggest["fields"]), "",
              "Members: %s" % ", ".join("`%s`" % m for m in biggest["members"]), ""]
    if pending:
        lines += ["", "## Backlog (passes every recorded check, never submitted)", "",
                  "Self-correlation is **not** checked here; each row is a candidate to "
                  "verify, not a candidate to submit.", "",
                  "| alpha | S | F | universe | note |", "| --- | --- | --- | --- | --- |"]
        for entry in pending[:15]:
            metrics = entry.get("metrics") or {}
            settings = entry.get("settings") or {}
            lines.append("| `%s` | %.2f | %.2f | %s | %s |" % (
                entry.get("alpha_id"), metrics.get("sharpe") or 0,
                metrics.get("fitness") or 0, settings.get("universe"),
                (entry.get("note") or "")[:48]))
        lines.append("")
        lines.append("### Backlog that shares nothing with the book")
        lines.append("")
        lines.append("No shared skeleton and no shared leg: these cannot be copies of "
                     "anything already submitted.")
        lines.append("")
        lines.append("| alpha | S | F | universe | note |")
        lines.append("| --- | --- | --- | --- | --- |")
        for entry in far[:10]:
            metrics = entry.get("metrics") or {}
            settings = entry.get("settings") or {}
            lines.append("| `%s` | %.2f | %.2f | %s | %s |" % (
                entry.get("alpha_id"), metrics.get("sharpe") or 0,
                metrics.get("fitness") or 0, settings.get("universe"),
                (entry.get("note") or "")[:48]))
        lines.append("")
    return "\n".join(lines) + "\n"


def mechanism_table(entries):
    """Per-mechanism counts and Fitness spread, as a markdown table.

    Kept here rather than in the atlas document so the numbers can be regenerated:
    a hand-maintained statistics table goes stale silently, and a stale atlas is worse
    than none because it reads as current.
    """
    import statistics

    groups = collections.defaultdict(list)
    for entry in entries:
        groups[entry.get("family") or "other"].append(entry)
    rows = []
    for label, members in groups.items():
        fitness = [(m.get("metrics") or {}).get("fitness") or 0 for m in members]
        best = max(members, key=lambda m: (m.get("metrics") or {}).get("fitness") or 0)
        rows.append((label, len(members), statistics.median(fitness), max(fitness),
                     best.get("alpha_id")))
    rows.sort(key=lambda row: -row[1])
    lines = ["| mechanism | count | median F | best F | best alpha |",
             "| --- | --- | --- | --- | --- |"]
    for label, count, median, best_f, best_id in rows:
        lines.append("| %s | %d | %.2f | %.2f | `%s` |" % (label, count, median, best_f, best_id))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanisms", action="store_true",
                        help="print the per-mechanism table instead of writing the report")
    parser.add_argument("--live", action="store_true",
                        help="read the book from the platform instead of the ledger "
                             "(the report always does; --mechanisms does not need it)")
    parser.add_argument("--out", default=None,
                        help="write the inventory as markdown here (not JSON: "
                             "this one is a document meant to be read)")
    args = parser.parse_args()

    if args.mechanisms and not args.live:
        # The per-mechanism table is a ledger aggregation. It used to log in and read
        # the live book first, which made the cheapest question in the toolkit the
        # slowest one, and impossible offline.
        entries = [e for e in alpha_registry.load_entries()
                   if (e.get("status") or "").upper() == "ACTIVE"]
        print("source: ledger, %d ACTIVE (pass --live to read the platform; run "
              "tools/reconcile_status.py if the column may have drifted)" % len(entries))
        print(mechanism_table(entries))
        return 0

    session = ml.login()
    entries, families = build(session)
    if args.mechanisms:
        print(mechanism_table(entries))
        return 0
    book_expressions = {e.get("expression") for e in entries}
    document = render(entries, families, backlog(session, book_expressions),
                      distant(session, entries))
    out = args.out or os.path.join(
        research_paths.ensure_dir(),
        "book-inventory-%s.md" % datetime.date.today().strftime("%Y%m%d"))
    with open(out, "w") as handle:
        handle.write(document)
    print(document)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())