#!/usr/bin/env python3
"""Skeleton for a new experiment. Copy this file, do not start from scratch.

Why a template instead of a convention: the guard only works if new scripts go
through it, and the path of least resistance has to be the correct one. This file is
runnable as-is -- it simulates nothing until EXPRESSIONS is filled in -- and it
already does the four things every experiment here needs:

1. sugar and annotations compile to plain FASTEXPR before anything is submitted;
2. the catalogue check runs before the simulation, not after a rejection;
3. results land in the research tree with a date-stamped name, never in the repo;
4. the summary prints the checks that fail, not just the score.

To run it::

    python3 tools/research_template.py --dry-run     # parse, lint, print, submit nothing
    python3 tools/research_template.py               # simulate

The dry run is the default in spirit: it costs nothing and catches the name errors
that used to cost a round trip.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_sugar  # noqa: E402
import annotations  # noqa: E402
import fastexpr_ast  # noqa: E402
import research_paths  # noqa: E402
import guarded_sim  # noqa: E402

SETTINGS = {
    "universe": "TOP3000",
    "region": "USA",
    "delay": 1,
    "decay": 4,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
    "testPeriod": "P1Y",
}

#: Sugar and annotations are allowed here; both compile away before submission.
EXPRESSIONS = [
    # '@mechanism("...")  a = rrank(...); b = ...; 0.7 * a + 0.3 * b',
]

NAME = "template"


def prepare(source):
    """Return ``(compiled_expression, annotation_findings)`` for one source string."""
    program = fastexpr_ast.parse(alpha_sugar.fs(source))
    return program.render(), annotations.check(program)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="parse and lint only; never submits")
    parser.add_argument("--force", action="store_true",
                        help="simulate even if the catalogue check complains")
    args = parser.parse_args()

    if not EXPRESSIONS:
        print("EXPRESSIONS is empty: add one, or read the docstring first.")
        return 0

    compiled = []
    for source in EXPRESSIONS:
        expression, findings = prepare(source)
        if findings:
            for finding in findings:
                print("annotation: %s -- %s" % (finding["code"], finding["message"]))
        compiled.append(expression)
        print("\n%s\n  -> %s" % (source.replace("\n", "\n     "), expression))

    if args.dry_run:
        report = guarded_sim.check(compiled, SETTINGS)
        for item in report["errors"]:
            print("error  %-24s %s" % (item["code"], item.get("message", "")[:100]))
        print("\ndry run: %d expression(s), %d error(s)"
              % (len(compiled), len(report["errors"])))
        return 0

    records = guarded_sim.simulate(compiled, SETTINGS, force=args.force)

    out = os.path.join(research_paths.ensure_dir(), "%s-%s.json" % (
        NAME, __import__("datetime").date.today().isoformat()))
    with open(out, "w") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=1)

    for record in records:
        metrics = record.get("metrics") or {}
        failed = [name for name, body in (record.get("checks") or {}).items()
                  if body.get("result") == "FAIL"]
        print("%-12s S=%-6s F=%-6s T=%-7s %s" % (
            record.get("alpha_id"), metrics.get("sharpe"), metrics.get("fitness"),
            metrics.get("turnover"), "fail=" + ",".join(failed) if failed else "all pass"))
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())