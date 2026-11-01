#!/usr/bin/env python3
"""Report the unit layer: how few constructions the ledger really is.

Run this instead of counting by hand. Every number below is recomputed from the
ledger on each run, because the counts in the prose go stale the moment an alpha is
filed -- the mechanism table already had to be re-derived once for exactly that
reason.

What the report is for is *reading a corpus*, not scoring it. Three questions, in the
order they get asked:

1. **How many constructions are there really?** The answer is a hierarchy, not a
   number, and the shape of the hierarchy is the finding: assemblies collapse by 20x,
   templates barely collapse at all. See ``docs/unit-compression.md``.
2. **What has a construction already tried, and what has it not?** The grid per unit;
   ``--gaps`` lists the derivations the book never ran.
3. **Which constructions are worth attention?** Size and best Fitness together, since
   a template with 42 field variants and one with 42 settings are different things.

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
import fastexpr_ast as fa  # noqa: E402

#: How an assembly token reads in prose. The vocabulary is small on purpose: there
#: are 40 assemblies in the corpus and they are all combinations of these two.
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

    Every line is printed with the unit it counts: "117 rows say ACTIVE" and "48
    ACTIVE keys" are different numbers about different things, and the report has to
    say which is which or the reader supplies their own.
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
    for kind in kinds[:12]:
        best = max((fitness_of(point) or 0) for unit in kind.units
                   for point in unit.points) or None
        print("  %4d  %-14s %-22s  units=%d  best F=%s"
              % (kind.size, ",".join(kind.families) or "-", kind.assembly,
                 len(kind.units), "%.2f" % best if best else "-"), file=stream)
        print("        e.g. %s"
              % kind.units[0].points[0].expression.replace("\n", " ")[:120], file=stream)

    print("\nunits by size", file=stream)
    units = sorted(corpus.units.values(), key=lambda u: -u.size)
    for unit in units[:10]:
        best = max((fitness_of(point) or 0) for point in unit.points) or None
        print("  %3d points  %-22s  slots=%2d  best F=%s  %s"
              % (unit.size, unit.assembly, len(unit.grid),
                 "%.2f" % best if best else "-", unit.roles[0][:44]), file=stream)
    return units


def gaps(corpus, known, stream=None, top=12):
    """Derivations of book constructions that the ledger has never run.

    The book is where the money is, so "which questions about it were never asked" is
    the cheapest question in the corpus. Ablations dominate: 43 of the 48 book alphas
    are composites whose legs were never individually removed.
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
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    reserve_stdout(args)

    entries = alpha_registry.load_entries()
    corpus, known, _datasets = load(entries)
    counts = au.summary(corpus)

    if args.unit or args.derive:
        prefix = args.unit or args.derive
        matches = [unit for key, unit in corpus.units.items() if key.startswith(prefix)]
        if not matches:
            print("no unit key starts with %r" % prefix, file=sys.stderr)
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