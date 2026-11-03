#!/usr/bin/env python3
"""Skeleton for a new experiment. Copy this file, do not start from scratch.

Why a template instead of a convention: the guard only works if new scripts go
through it, and the path of least resistance has to be the correct one. This file is
runnable as-is -- it simulates nothing until EXPRESSIONS or PARENT is filled in --
and it already does the five things every experiment here needs:

1. sugar and annotations compile to plain FASTEXPR before anything is submitted;
2. the catalogue check runs before the simulation, not after a rejection;
3. results land in the research tree with a date-stamped name, never in the repo;
4. results also land in the ledger, which is the only durable index of what was tried;
5. the summary prints the **train** period next to the score and hides the test
   period unless you ask for it. `testPeriod` is the only out-of-sample evidence the
   API exposes - BRAIN's full window already contains it, so selecting on the
   full-window number trains on the test year. Print it once, after the design is
   frozen, with ``--show-test``.

Two shapes of experiment, because that is what the last week actually was:

* ``EXPRESSIONS`` -- a handful of hand-written formulas;
* ``PARENT`` + ``SUBSTITUTIONS`` -- one proven alpha with one token swapped out N
  times. Eleven scripts in four days were this shape, each re-writing the
  substitution, the table and the ledger call. Write the token and the values here
  instead; the parent's settings and expression come from the ledger, so "only this
  token changed" is true by construction.

To run it::

    python3 tools/research_template.py --dry-run     # parse, lint, print, submit nothing
    python3 tools/research_template.py               # simulate and record

The dry run is the default in spirit: it costs nothing and catches the name errors
that used to cost a round trip.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import alpha_sugar  # noqa: E402
import annotations  # noqa: E402
import fastexpr_ast  # noqa: E402
import guarded_sim  # noqa: E402
import research_paths  # noqa: E402

SETTINGS = {
    "universe": "TOP3000",
    "region": "USA",
    "delay": 1,
    "decay": 4,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
    "testPeriod": "P1Y",
}

#: The settings a parent's frozen expression should inherit. Everything else in a
#: stored settings block -- startDate, endDate, simulationMode -- is assigned by BRAIN
#: and would be a lie if resent.
SETTINGS_KEYS = ("region", "universe", "delay", "decay", "neutralization", "truncation",
                 "testPeriod", "nanHandling", "pasteurization", "unitHandling",
                 "instrumentType", "language")

#: Sugar and annotations are allowed here; both compile away before submission.
#: An entry may be a plain string or ``(label, string)``. Use a label when the batch
#: is meant to be read later: the label travels into the ledger as the entry's
#: mechanism, and a truncated first line of an expression is not a mechanism.
EXPRESSIONS = [
    # '@mechanism("...")  a = rrank(...); b = ...; 0.7 * a + 0.3 * b',
]

#: Set both to run the swapped-token shape instead of ``EXPRESSIONS``. The parent is
#: read from the ledger, so its expression and settings are the ones that actually
#: ran -- not a retyped copy.
PARENT = ""
SUBSTITUTIONS = {
    # "accrued_liabilities_total_2": ["taxes_due_liabilities", "senior_notes_balance"],
}

NAME = "template"
IDEA = ""
MECHANISM = ""
NOTE = ""


def prepare(source):
    """Return ``(compiled_expression, annotation_findings)`` for one source string."""
    program = fastexpr_ast.parse(alpha_sugar.fs(source))
    return program.render(), annotations.check(program)


def handwritten(expressions):
    """``(label, expression)`` pairs for the ``EXPRESSIONS`` shape."""
    pairs = []
    for index, item in enumerate(expressions, 1):
        label, source = item if isinstance(item, tuple) else ("expr%d" % index, item)
        expression, findings = prepare(source)
        for finding in findings:
            print("annotation: %s -- %s" % (finding["code"], finding["message"]))
        print("\n%s\n  -> %s" % (source.replace("\n", "\n     "), expression))
        pairs.append((label, expression))
    return pairs


def expand(expression, substitutions):
    """One ``(label, expression)`` pair per combination of replacement values.

    Substitution is textual and exact, which is the point: the experiment claims that
    *only* this token changed. A target that does not appear in the parent is an error
    rather than a silent no-op, because a batch of identical expressions looks exactly
    like a batch of variants that all failed.
    """
    targets = list(substitutions)
    patterns = {}
    for target in targets:
        # Whole identifiers only. `str.replace("sales", "revenue")` also rewrites
        # `sales_growth` and `sales_number`, so a "single-factor" swap silently
        # changed several inputs - and both fields exist, so the lint passed.
        pattern = re.compile(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(target))
        if not pattern.search(expression):
            raise SystemExit("%r is not in the parent expression; nothing to swap"
                             % target)
        patterns[target] = pattern
    pairs = []
    for combination in itertools.product(*(substitutions[t] for t in targets)):
        text = expression
        for target, value in zip(targets, combination):
            text = patterns[target].sub(value, text)
        pairs.append((",".join("%s=%s" % (t, v) for t, v in zip(targets, combination)),
                      text))
    return pairs


def parent_experiment(alpha_id, registry_path=None):
    """``(settings, pairs)`` from a ledger entry, for the swapped-token shape."""
    entries = alpha_registry.load_entries(registry_path)
    entry = next((item for item in entries if item.get("alpha_id") == alpha_id), None)
    if entry is None:
        raise SystemExit("%s is not in the ledger; record it first, or use EXPRESSIONS"
                         % alpha_id)
    expression = entry.get("expression") or ""
    if not expression:
        raise SystemExit("%s has no expression in the ledger" % alpha_id)
    settings = {key: value for key, value in (entry.get("settings") or {}).items()
                if key in SETTINGS_KEYS}
    if not SUBSTITUTIONS:
        raise SystemExit("PARENT is set but SUBSTITUTIONS is empty")
    # The parent expression is already compiled; running it through the sugar pass
    # again would re-render it and weaken the claim that only the token changed.
    return settings, expand(expression, SUBSTITUTIONS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="parse and lint only; never submits")
    parser.add_argument("--force", action="store_true",
                        help="simulate even if the catalogue check complains")
    parser.add_argument("--no-record", action="store_true",
                        help="do not append the results to the ledger")
    parser.add_argument("--show-test", action="store_true",
                        help="print the platform's test-period numbers. Do not pass this "
                             "while choosing between arms: the test year is the only "
                             "out-of-sample evidence there is, and reading it during "
                             "selection turns it into training data.")
    parser.add_argument("--registry", default=None)
    args = parser.parse_args()

    if PARENT:
        settings, pairs = parent_experiment(PARENT, args.registry)
        print("parent %s: %d variant(s)" % (PARENT, len(pairs)))
    else:
        if not EXPRESSIONS:
            print("EXPRESSIONS and PARENT are both empty: add one, or read the docstring.")
            return 0
        settings = SETTINGS
        pairs = handwritten(EXPRESSIONS)

    compiled = [expression for _label, expression in pairs]
    labels = [label for label, _expression in pairs]

    if args.dry_run:
        report = guarded_sim.check(compiled, settings)
        for item in report["errors"]:
            print("error  %-24s %s" % (item["code"], item.get("message", "")[:100]))
        print("\ndry run: %d expression(s), %d error(s)"
              % (len(compiled), len(report["errors"])))
        return 0

    records = guarded_sim.simulate(compiled, settings, labels=labels, force=args.force)

    out = os.path.join(research_paths.ensure_dir(), "%s-%s.json" % (
        NAME, __import__("datetime").date.today().isoformat()))
    with open(out, "w") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=1)

    print()
    header = "%-28s %6s %6s %7s %7s" % ("label", "S", "F", "T", "train")
    if args.show_test:
        header += " %7s" % "test"
    print(header)
    for record in records:
        metrics = record.get("metrics") or {}
        split = record.get("out_of_sample") or {}
        failed = [name for name, body in (record.get("checks") or {}).items()
                  if isinstance(body, dict) and body.get("result") == "FAIL"]
        line = ("%-28s %6s %6s %7s %7s"
                % (str(record.get("mechanism") or record.get("alpha_id"))[:28],
                   metrics.get("sharpe"), metrics.get("fitness"), metrics.get("turnover"),
                   (split.get("train") or {}).get("sharpe")))
        if args.show_test:
            line += " %7s" % (split.get("test") or {}).get("sharpe")
        print(line + ("  fail=" + ",".join(failed) if failed else ""))

    if not args.no_record:
        summary = alpha_registry.record_simulation(
            records, settings, idea=IDEA or NAME, mechanism=MECHANISM or None,
            note=NOTE or None, registry_path=args.registry)
        print("\nledger: %d recorded, %d new" % (summary["recorded"], summary["appended"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
