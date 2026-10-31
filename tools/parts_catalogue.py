#!/usr/bin/env python3
"""Catalogue the reusable parts of every expression in the ledger.

An alpha is assembled from legs, and a leg is a shape applied to data. Both halves
matter and they are different indexes:

* the **shape** (fields replaced by F, literals by L) says what the leg does, and two
  alphas reading different fields through the same shape share that leg's behaviour;
* the **fields** say what data it reads, which is what decides whether a leg is
  redundant with something already in the book.

Indexing only by shape loses "this shape works on balance-sheet data but not on price
data"; indexing only by expression loses the fact that the shape is reusable. So this
records both, and reports the pair.

What the counts are used for: a part that appears in book alphas is proven; a part with
a high best Fitness that appears in one or two alphas is an untapped assembly; a part
that appears many times and never scores is a dead end that should stop being reused.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import fastexpr_ast as fa  # noqa: E402


def _nodes(node):
    return sum(1 for _ in node.walk())


def parts_of(expression, min_nodes=6):
    """Return ``[(shape, fields, kind), ...]`` for one expression.

    ``min_nodes`` drops fragments that are not parts. Without it the catalogue fills
    with single-operator legs like ``ts_mean(F, L)``, which appear in hundreds of
    expressions and say nothing: they are the vocabulary, not a construction.

    ``kind`` distinguishes a named leg (an assignment) from the assembling expression
    that combines them, because those are reused in different ways: legs get swapped,
    the combiner gets re-weighted.
    """
    program = fa.parse(expression)
    out = []
    for _name, value in program.assignments:
        if _nodes(value) < min_nodes:
            continue
        out.append((fa.skeleton(fa.Program((), value)),
                    tuple(fa.field_names(fa.Program((), value))),
                    "leg"))
    if program.signal is not None and _nodes(program.signal) >= min_nodes:
        out.append((fa.skeleton(fa.Program((), program.signal)),
                    tuple(fa.field_names(fa.Program((), program.signal))),
                    "combiner"))
    return out


def build(entries):
    """Aggregate the ledger into ``{shape: stats}``."""
    catalogue = {}
    for entry in entries:
        expression = entry.get("expression")
        if not expression:
            continue
        try:
            parts = parts_of(expression)
        except fa.ParseError:
            continue
        metrics = entry.get("metrics") or {}
        fitness = metrics.get("fitness")
        in_book = (entry.get("status") or "").upper() == "ACTIVE"
        for shape, fields, kind in parts:
            slot = catalogue.setdefault(shape, {
                "shape": shape, "kind": kind,
                "expressions": set(), "book": set(), "fields": set(),
                "best_fitness": None, "best_alpha": None, "fitness_sum": 0.0,
                "fitness_n": 0, "example": expression,
            })
            slot["expressions"].add(expression)
            slot["fields"].update(fields)
            if in_book:
                slot["book"].add(entry.get("alpha_id") or expression)
            if fitness is not None:
                slot["fitness_sum"] += fitness
                slot["fitness_n"] += 1
                if slot["best_fitness"] is None or fitness > slot["best_fitness"]:
                    slot["best_fitness"] = fitness
                    slot["best_alpha"] = entry.get("alpha_id")
    for slot in catalogue.values():
        slot["mean_fitness"] = (slot["fitness_sum"] / slot["fitness_n"]
                                if slot["fitness_n"] else None)
        slot["n_expressions"] = len(slot["expressions"])
        slot["n_book"] = len(slot["book"])
    return catalogue


def screen(catalogue, min_fitness=1.5, max_expressions=3):
    """Split parts into proven / untapped / dead.

    ``proven``     used by at least one submitted alpha -- safe to reuse, already
                   represented in the book's exposures.
    ``untapped``   a high best Fitness in very few expressions: the shape works but the
                   assembly has not been pushed, which is where new candidates come from.
    ``dead``       many expressions, no score: reusing it again is a waste.
    """
    proven, untapped, dead = [], [], []
    for slot in catalogue.values():
        if slot["n_book"] > 0:
            proven.append(slot)
        elif (slot["best_fitness"] or 0) >= min_fitness and slot["n_expressions"] <= max_expressions:
            untapped.append(slot)
        elif slot["n_expressions"] >= 5 and (slot["best_fitness"] or 0) < 1.0:
            dead.append(slot)
    proven.sort(key=lambda s: -(s["best_fitness"] or 0))
    untapped.sort(key=lambda s: -(s["best_fitness"] or 0))
    dead.sort(key=lambda s: -s["n_expressions"])
    return proven, untapped, dead


def seeds(entries, catalogue, min_fitness=1.5):
    """Proven parts that have met the fewest partners, plus what they never met.

    A part in the book is proven; a part in the book that has only ever appeared
    alongside two other parts has barely been tested as an ingredient. The suggestion
    attached to each seed is the set of *other* proven parts it has never co-occurred
    with -- assemblies nobody has run, built only from parts that already work.

    This is deliberately not a ranking by Fitness: the point is coverage of
    combinations, not another leaderboard.
    """
    # Co-occurrence uses *every* part, including the simple legs the display filter
    # drops. With the filter applied here, `x = ts_backfill(f, 252)` disappeared and an
    # expression reduced to one part, so every seed reported zero partners -- a fact
    # about the filter, not about the book.
    partners = collections.defaultdict(set)
    for entry in entries:
        expression = entry.get("expression")
        if not expression:
            continue
        try:
            shapes = {shape for shape, _f, _k in parts_of(expression, min_nodes=1)}
        except fa.ParseError:
            continue
        for shape in shapes:
            partners[shape] |= (shapes - {shape})
    proven = [slot for slot in catalogue.values() if slot["n_book"] > 0]
    proven_shapes = {slot["shape"] for slot in proven}
    ranked = sorted(proven, key=lambda s: (len(partners[s["shape"]]), -(s["best_fitness"] or 0)))
    out = []
    for slot in ranked:
        met = partners[slot["shape"]]
        never = sorted(proven_shapes - met - {slot["shape"]})
        out.append({"slot": slot, "partners": sorted(met), "never": never,
                    "best_fitness": slot["best_fitness"]})
    return out


def portable_seeds(catalogue, min_fitness=1.2, max_book=1):
    """Proven parts that appear in few book alphas: ingredients never ported.

    The co-occurrence version of this question was degenerate -- most book alphas are a
    single expression with no named leg, so "partners" was empty, and with 61 proven
    parts almost every pair is untried, so "never met" was always true. Counting how
    many *families* a proven part appears in is the sharper question: an ingredient
    that carries one family at F 1.8 and appears nowhere else has been tested once.
    """
    rows = [slot for slot in catalogue.values()
            if slot["n_book"] > 0 and slot["n_book"] <= max_book
            and (slot["best_fitness"] or 0) >= min_fitness]
    rows.sort(key=lambda s: -(s["best_fitness"] or 0))
    return rows


def report_seeds(entries, catalogue, top=8, stream=None):
    stream = stream or sys.stdout
    rows = [r for r in seeds(entries, catalogue) if (r["best_fitness"] or 0) >= 1.2]
    print("seed parts (in the book, fewest partners):", file=stream)
    for row in rows[:top]:
        slot = row["slot"]
        print("\n  %s" % slot["shape"][:96], file=stream)
        print("    F=%.2f  used in %d expressions, %d book alphas, %d partners"
              % (row["best_fitness"] or 0, slot["n_expressions"], slot["n_book"],
                 len(row["partners"])), file=stream)
        if row["never"]:
            print("    never met: %s" % " | ".join(x[:44] for x in row["never"][:3]), file=stream)
    return rows


def report(catalogue, stream=None):
    stream = stream or sys.stdout
    proven, untapped, dead = screen(catalogue)
    print("parts: %d  (proven %d, untapped %d, dead %d)"
          % (len(catalogue), len(proven), len(untapped), len(dead)), file=stream)

    def line(slot):
        return ("  F=%-5s n=%-3d book=%-2d  %-58s  %s"
                % ("%.2f" % slot["best_fitness"] if slot["best_fitness"] is not None else "-",
                   slot["n_expressions"], slot["n_book"], slot["shape"][:58],
                   ", ".join(sorted(slot["fields"])[:2])))

    for title, rows in (("PROVEN (in the book)", proven[:8]),
                        ("UNTAPPED (strong, barely used)", untapped[:12]),
                        ("DEAD (reused, never scored)", dead[:6])):
        print("\n%s" % title, file=stream)
        for slot in rows:
            print(line(slot), file=stream)
            if title.startswith("UNTAPPED") and slot.get("example"):
                print("        e.g. %s" % slot["example"][:110].replace("\n", " "), file=stream)
    return {"proven": proven, "untapped": untapped, "dead": dead}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None, help="write the full catalogue here")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--seeds", action="store_true",
                        help="show proven parts confined to one book family")
    args = parser.parse_args()

    entries = [e for e in alpha_registry.load_entries()
               if not e.get("abandoned") and e.get("expression")]
    catalogue = build(entries)
    if args.seeds:
        rows = portable_seeds(catalogue)
        print("proven parts confined to a single book family (%d):" % len(rows))
        for slot in rows[:args.top]:
            print("\n  F=%.2f  n=%-3d book=%-2d  %s"
                  % (slot["best_fitness"] or 0, slot["n_expressions"], slot["n_book"],
                     slot["shape"][:92]))
            print("    fields: %s" % ", ".join(sorted(slot["fields"])[:4]))
            print("    e.g. %s" % slot["example"][:110].replace("\n", " "))
        return 0
    summary = report(catalogue)

    if args.json:
        payload = {}
        for slot in catalogue.values():
            payload[slot["shape"]] = {
                "kind": slot["kind"],
                "n_expressions": slot["n_expressions"],
                "n_book": slot["n_book"],
                "best_fitness": slot["best_fitness"],
                "best_alpha": slot["best_alpha"],
                "mean_fitness": slot["mean_fitness"],
                "fields": sorted(slot["fields"]),
            }
        with open(args.json, "w") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())