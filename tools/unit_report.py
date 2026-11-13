#!/usr/bin/env python3
"""Report the unit layer: how few constructions the ledger really is.

Run this instead of counting by hand. Every number below is recomputed from the
ledger on each run, because the counts in the prose go stale the moment an alpha is
filed -- the mechanism table already had to be re-derived once for exactly that
reason.

What the report is for is *reading a corpus*, not scoring it. Three questions, in the
order they get asked:

1. **How many constructions are there really?** Report the hierarchy of assemblies,
   kinds and templates. See ``docs/unit-compression.md`` for their definitions.
2. **What has a construction already tried, and what has it not?** The grid per unit;
   ``--gaps`` lists the derivations the book never ran.
3. **Which constructions are worth attention?** Size and best Fitness together, since
   field variants and settings variants answer different questions.

Usage:
    python3 tools/unit_report.py                 # the report
    python3 tools/unit_report.py --gaps          # book ablations never run
    python3 tools/unit_report.py --unit 3f2a    # one unit, in full
    python3 tools/unit_report.py --derive 3f2a  # what that unit has not tried
    python3 tools/unit_report.py --json units.json
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
sys.path.insert(0, HERE)

from _output import add_json_flag, guard, reserve_stdout, write  # noqa: E402

import alpha_registry  # noqa: E402
import alpha_unit as au  # noqa: E402
import closed_families  # noqa: E402
import fastexpr_ast as fa  # noqa: E402

#: How an assembly token reads in prose. Assemblies combine legs and offsets.
_TOKENS = {"G": "leg", "-G": "−leg", "L": "offset", "-L": "−offset"}


def describe_assembly(assembly):
    """``G -L -G`` -> ``leg −offset −leg``: the pattern in words, not in symbols.

    The tokens are the whole vocabulary -- there are only two, a leg and a constant --
    so the translation is three lines and removes the "what is G" step from every
    reading of the report.
    """
    return " ".join(_TOKENS.get(token, token) for token in assembly.split())


def fitness_of(point):
    value = (point.metrics or {}).get("fitness")
    return value if isinstance(value, (int, float)) else None


def load(entries=None):
    """Build the corpus, plus the set of every canonical form ever filed.

    The second set deliberately includes abandoned rows: an expression that was run
    and thrown away is still an expression that was run, and calling its ablation
    "novel" would send the next pass to redo it.
    """
    entries = list(entries if entries is not None else alpha_registry.load_entries())
    datasets = au.load_field_datasets()
    corpus = au.build(entries, datasets=datasets)
    known = set()
    for entry in entries:
        expression = entry.get("expression")
        if not expression:
            continue
        point = au.split(expression, datasets=datasets)
        if point is not None:
            known.add(point.canonical)
    return corpus, known, datasets


def report(corpus, counts, stream=None):
    """The hierarchy plus the top of each level, in the order the questions are asked.

    Every line names its counting unit: rows carrying ACTIVE and distinct ACTIVE
    keys describe different things, so the report must distinguish them.
    """
    stream = sys.stdout if stream is None else stream
    print("corpus", file=stream)
    print("  expressions      %d  (after inlining; variables are not content)"
          % counts["expressions"], file=stream)
    print("  assemblies       %d  (how legs are mixed)"
          % counts["assemblies"], file=stream)
    print("  kinds            %d  (assembly x data family), largest %d"
          % (counts["kinds"], counts["largest_kind"]), file=stream)
    print("  units            %d  (exact template), largest %d, %d singletons"
          % (counts["units"], counts["largest_unit"], counts["unit_singletons"]),
          file=stream)
    inside = counts["expressions"] - counts["unit_singletons"]
    print("  in a shared unit %d / %d = %.0f%%"
          % (inside, counts["expressions"], 100.0 * inside / max(1, counts["expressions"])),
          file=stream)
    print("  unit sizes       %s" % counts["unit_sizes"], file=stream)
    print("  one-step moves   %d  (variants reachable without inventing a value)"
          % counts["one_step"], file=stream)

    book = [point for point in corpus.points
            if any((status or "").upper() == "ACTIVE" for status in point.statuses)]
    filers = sum(sum(1 for status in point.statuses
                     if (status or "").upper() == "ACTIVE") for point in book)
    print("  book             %d alphas over %d constructions, %d units, %d kinds"
          % (filers, len(book), len({point.unit for point in book}),
             len({point.kind for point in book})), file=stream)

    print("\nassemblies by coverage", file=stream)
    by_assembly = collections.Counter(point.assembly for point in corpus.points)
    for assembly, count in by_assembly.most_common(10):
        print("  %4d  %-22s %s"
              % (count, assembly, describe_assembly(assembly)), file=stream)

    print("\nkinds by size", file=stream)
    kinds = sorted(corpus.kinds.values(), key=lambda k: (-k.size, k.assembly))
    closed = closed_families.closed_families()
    for kind in kinds[:12]:
        best = max((fitness_of(point) or 0) for unit in kind.units
                   for point in unit.points) or None
        flag = ""
        hit = sorted(closed.intersection(kind.families))
        if hit:
            # High fitness does not override a recorded closure reason. Show the
            # warning so a closed family does not read as an unexplored opportunity.
            flag = "  [CLOSED: %s]" % ",".join(hit)
        print("  %4d  %-14s %-22s  units=%d  best F=%s%s"
              % (kind.size, ",".join(kind.families) or "-", kind.assembly,
                 len(kind.units), "%.2f" % best if best else "-", flag), file=stream)
        print("        e.g. %s"
              % kind.units[0].points[0].expression.replace("\n", " ")[:120], file=stream)

    # A closed family that scores well is the most dangerous row in this report,
    # and it is always small, so the size-ranked table above never shows it. Listed
    # separately, by score, with the reason.
    dead = []
    for kind in corpus.kinds.values():
        hit = sorted(closed.intersection(kind.families))
        if not hit:
            continue
        best = max((fitness_of(point) or 0) for unit in kind.units
                   for point in unit.points) or 0
        dead.append((best, kind, hit))
    if dead:
        print("\nclosed families that still score well (do not read the F as an "
              "opportunity)", file=stream)
        for best, kind, hit in sorted(dead, key=lambda row: -row[0])[:5]:
            print("  %4d  %-14s %-20s  best F=%.2f   [CLOSED: %s]"
                  % (kind.size, ",".join(kind.families), kind.assembly, best,
                     ",".join(hit)), file=stream)

    print("\nunits by size  (the 8-char id works with --unit / --derive)", file=stream)
    units = sorted(corpus.units.values(), key=lambda u: -u.size)
    for unit in units[:10]:
        best = max((fitness_of(point) or 0) for point in unit.points) or None
        print("  %s  %3d points  %-22s  slots=%2d  best F=%s  %s"
              % (unit.short, unit.size, unit.assembly, len(unit.grid),
                 "%.2f" % best if best else "-", unit.roles[0][:40]), file=stream)
    return units


def gaps(corpus, known, stream=None, top=12):
    """Derivations of book constructions that the ledger has never run.

    Book constructions can contain legs that have never been individually removed.
    Reporting those ablations exposes untested questions about existing holdings.
    """
    stream = sys.stdout if stream is None else stream
    book = [point for point in corpus.points
            if any((status or "").upper() == "ACTIVE" for status in point.statuses)]
    totals = collections.Counter()
    rows = []
    for point in book:
        variants = au.derive(point, grid=corpus.units[point.unit].grid, observed=known)
        novel = [variant for variant in variants if variant.status == "novel"]
        seen = set()
        unique = []
        for variant in novel:
            if variant.canonical in seen:
                continue
            seen.add(variant.canonical)
            unique.append(variant)
        totals.update(variant.rule for variant in unique)
        rows.append((point, unique))

    print("book: %d constructions, %d derivations never run"
          % (len(book), sum(totals.values())), file=stream)
    for rule, count in totals.most_common():
        print("  %-10s %d" % (rule, count), file=stream)
    rows.sort(key=lambda row: -len(row[1]))
    print("\nbiggest gaps", file=stream)
    for point, variants in rows[:top]:
        label = point.alpha_ids[:1] or ("expr",)
        print("  %3d open  %-8s  %s"
              % (len(variants), str(label[0]), point.expression.replace("\n", " ")[:96]),
              file=stream)
    return rows


def dataset_view(corpus, dataset, stream=None, top=8):
    """What a data family has already been asked, in shapes rather than rows.

    Before spending simulations on a dataset the useful question is not "has anyone
    used field X" but "which shapes has this family already been run in, and how did
    they score". Each unit is one exact template and each kind is assembly x family,
    so the answer is a short list, not a grep.

    A dataset absent from the book may still have many measured expressions in the
    ledger. Inspecting the unit layer before writing probes avoids repeating them.
    """
    stream = sys.stdout if stream is None else stream
    # The unit layer groups by *family* (price / fundamental / analyst / news /
    # option / model), not by dataset id; `unit-rules.json` holds the mapping. Take
    # either and say which one answered, so "model51" and "model" are not a silent
    # difference.
    mapping = (au.load_rules() or {}).get("datasets") or {}
    family = mapping.get(dataset, dataset)
    if family != dataset:
        print("\n%s → family %s (unit-rules.json)" % (dataset, family), file=stream)
    dataset = family
    points = [point for point in corpus.points if dataset in point.datasets]
    units = [unit for unit in corpus.units.values() if dataset in unit.datasets]
    in_book = [point for point in points
               if any((status or "").upper() == "ACTIVE" for status in point.statuses)]
    best_fitness = max((fitness_of(point) or 0) for point in points) if points else None
    best_sharpe = max(((point.metrics or {}).get("sharpe")
                       or -9) for point in points) if points else None
    families = sorted({family for unit in units for family in unit.datasets})

    print("\ndataset %s" % dataset, file=stream)
    print("  expressions %d · units %d · kinds %d · in the book %d"
          % (len(points), len(units), len({point.kind for point in points}), len(in_book)),
          file=stream)
    print("  families    %s" % ", ".join(families or ["?"]), file=stream)
    print("  best        F=%s%s"
          % ("%.2f" % best_fitness if best_fitness else "-",
             "  S=%.2f" % best_sharpe if isinstance(best_sharpe, float) else ""),
          file=stream)

    print("\nshapes already run (by number of points)", file=stream)
    for unit in sorted(units, key=lambda item: -len(item.points))[:top]:
        best = max((fitness_of(point) or 0) for point in unit.points) or None
        sample = unit.points[0].expression.replace("\n", " ")[:96]
        print("  %-22s points=%-3d slots=%2d best F=%s"
              % (unit.assembly, len(unit.points), len(unit.grid),
                 "%.2f" % best if best else "-"), file=stream)
        print("        %s" % sample, file=stream)
    return {"dataset": dataset, "expressions": len(points), "units": len(units),
            "in_book": len(in_book), "best_fitness": best_fitness,
            "best_sharpe": best_sharpe,
            "assemblies": sorted({unit.assembly for unit in units})}


def crossings(corpus, known, stream=None, top=10):
    """Templates nobody ran, assembled from legs that each already exists.

    Only a handful survive the guards (same role, same data families, one leg
    already shared), which is the point: a port has to be able to say *why* the two
    legs are interchangeable, and "both are ranked ratios over fundamentals" is a
    reason while "both are subexpressions" is not.
    """
    stream = sys.stdout if stream is None else stream
    total = novel = 0
    rows = []
    for unit in corpus.units.values():
        variants = au.crossover(unit, corpus, observed=known)
        for variant in variants:
            total += 1
            if variant.status == "novel":
                novel += 1
                rows.append((unit, variant))
    print("crossovers: %d candidates, %d never run" % (total, novel), file=stream)
    for unit, variant in rows[:top]:
        print("  [%s] %s" % (variant.label, variant.expression.replace("\n", " ")[:120]),
              file=stream)
    return rows


def _payload(corpus):
    out = []
    for unit in corpus.units.values():
        points = []
        for point in unit.points:
            points.append({
                "expression": point.expression,
                "canonical": point.canonical,
                "alpha_ids": list(point.alpha_ids),
                "statuses": list(point.statuses),
                "fitness": (point.metrics or {}).get("fitness"),
                "sharp": (point.metrics or {}).get("sharpe"),
                "filling": point.filling,
            })
        out.append({
            "key": unit.key,
            "assembly": unit.assembly,
            "families": list(unit.datasets),
            "roles": list(unit.roles),
            "grid": {name: values for name, values in unit.grid.items()},
            "points": points,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    add_json_flag(parser, help_="write every unit (default: stdout)")
    parser.add_argument("--unit", default=None, help="show one unit by key prefix")
    parser.add_argument("--derive", default=None, help="derivations for one unit prefix")
    parser.add_argument("--gaps", action="store_true", help="book derivations never run")
    parser.add_argument("--cross", action="store_true",
                        help="templates assembled from two existing constructions")
    parser.add_argument("--dataset", default=None,
                        help="what one data family has already been asked: shapes, "
                             "counts and the best score seen, before spending a "
                             "simulation on it")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    reserve_stdout(args)

    entries = alpha_registry.load_entries()
    corpus, known, _datasets = load(entries)
    counts = au.summary(corpus)

    if args.unit or args.derive:
        prefix = args.unit or args.derive
        matches = [unit for unit in corpus.units.values()
                   if unit.short == prefix or unit.short.startswith(prefix)
                   or unit.key.startswith(prefix)]
        if not matches:
            print("no unit matches %r (pass an 8-char id from the report, or a key "
                  "prefix; `--json` lists every key)" % prefix, file=sys.stderr)
            return 2
        for unit in matches[:3]:
            print(au.render_unit(unit, preview=args.top))
            if args.derive:
                print("\n  derivations:")
                for point in unit.points[:2]:
                    variants = au.derive(point, grid=unit.grid, observed=known)
                    novel = [variant for variant in variants if variant.status == "novel"]
                    seen, unique = set(), []
                    for variant in novel:
                        if variant.canonical not in seen:
                            seen.add(variant.canonical)
                            unique.append(variant)
                    print("    from %s: %d novel" % (point.canonical[:52], len(unique)))
                    for variant in unique[:args.top]:
                        print("      [%s] %s" % (variant.rule,
                                                 variant.expression.replace("\n", " ")[:110]))
            print()
        return 0

    if args.gaps:
        gaps(corpus, known, top=args.top)
        return 0

    if args.dataset:
        dataset_view(corpus, args.dataset, top=args.top)
        return 0

    if args.cross:
        crossings(corpus, known, top=args.top)
        return 0

    checked, failures = au.verify(corpus)
    report(corpus, counts)
    print("\nround-trip: %d/%d templates refill to their own expression, %d failures"
          % (checked - len(failures), checked, len(failures)))
    if failures:
        for expression, expected, got in failures[:5]:
            print("  FAIL %s\n    expected %s\n    got      %s"
                  % (expression[:80], expected[:80], got[:80]))
        return 1

    if args.json:
        target = write(_payload(corpus), args.json)
        if target and target != "-":
            print("\nwrote %s (%d units)" % (target, len(corpus.units)))
    return 0


if __name__ == "__main__":
    raise SystemExit(guard(main))
