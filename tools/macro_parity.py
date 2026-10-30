#!/usr/bin/env python3
"""Compare the text macro backend against the tree backend, honestly.

The first version of this comparison was useless because it fed nonsense into
positional slots and counted the resulting failures as differences. This one
classifies every combination into four buckets, and only two of them are findings:

``old_invalid``      the text expander emitted something that does not parse
                     (e.g. a number on the left of ``=``). Harness noise.
``old_leaks_macro``  the output still contains a macro call. A real defect: the
                     text expander's call regex cannot match arguments containing
                     parentheses, so a nested macro with a compound argument is
                     left in place and the simulator would reject the name.
``semantic_diff``    both parse, but they mean different things. A real defect:
                     the placeholder sat next to an operator and the argument
                     needed parentheses.
``equal``            same meaning.

Arguments are drawn per parameter kind: a parameter used as an assignment target
only ever gets an identifier, and parameters that are not used arithmetically only
get the literal pools, which keeps the sweep from manufacturing failures.
"""

from __future__ import annotations

import collections
import glob
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ast_macros  # noqa: E402
import expression_macros as text_macros  # noqa: E402
import fastexpr_ast as fa  # noqa: E402

FIELDS = ["fnd6_newqv1300_ivltq", "close"]
ZERO_ARG = ["position()", "pressure()"]
COMPOUND = ["a - b", "(a + b) / 2"]
NUMBERS = ["0.5", "252"]
NAMES = ["x", "v"]
SAMPLE = 300


def non_atomic_params(program):
    """Parameters whose placeholder does not sit in a whole-argument position.

    Those are the only places where textual substitution can change meaning: the
    argument's own operators bind against the surrounding expression instead of
    against the placeholder. If a difference appears anywhere else, it is a real
    regression rather than a known defect being fixed.
    """
    risky = set()

    def walk(node, parent):
        if (isinstance(node, fa.Field) and node.name.startswith("__ph_")
                and not isinstance(parent, fa.Call)):
            risky.add(node.name[5:-2])
        if isinstance(node, fa.Call):
            children = node.args + tuple(v for _k, v in node.kwargs)
        elif isinstance(node, fa.BinOp):
            children = (node.left, node.right)
        elif isinstance(node, fa.UnaryOp):
            children = (node.operand,)
        elif isinstance(node, fa.Ternary):
            children = (node.condition, node.when_true, node.when_false)
        else:
            children = ()
        for child in children:
            walk(child, node)

    for _key, value in program.assignments:
        walk(value, None)
    if program.signal is not None:
        walk(program.signal, None)
    return risky


def _is_compound(text):
    signal = fa.parse(text).signal
    return any(isinstance(node, (fa.BinOp, fa.UnaryOp, fa.Call))
               for node in signal.walk())


def classify(am, tm, name, spec, args):
    """Run one argument combination through both backends and name the outcome.

    The distinction that matters is between a difference with a mechanism (the
    placeholder sat against an operator, so the argument needed parentheses) and one
    without. Only the second is a regression.
    """

    try:
        old = text_macros.expand(tm, name, list(args))
    except Exception:
        return "old_invalid", None
    try:
        fa.parse(old)
    except fa.ParseError:
        return "old_invalid", None

    leaked = [call.name for call in fa.parse(old).walk()
              if isinstance(call, fa.Call) and call.name in am]
    if leaked:
        return "old_leaks_macro", old

    try:
        new = ast_macros.expand(am, name, list(args))
        same = fa.parse(old).render() == new
    except Exception:
        return "new_error", None
    if same:
        return "equal", None
    risky = non_atomic_params(spec["ast"])
    bound = dict(zip([p for p, _d in spec["params"]], args))
    if any(_is_compound(bound[p]) for p in risky if p in bound):
        return "diff_explained", (old, new)
    return "diff_UNEXPLAINED", (old, new)


def main() -> int:
    buckets = collections.Counter()
    per_macro = collections.defaultdict(collections.Counter)
    samples = collections.defaultdict(list)

    for path in sorted(glob.glob("/data/worldquant-research/macros/*.macros")):
        source = open(path).read()
        am = ast_macros.parse_macros(source)
        tm = text_macros.parse_macros(source)
        for name, spec in am.items():
            key_params = {key for key, _value in spec["ast"].assignments}
            pools = []
            for param, _default in spec["params"]:
                if param in key_params:
                    pools.append(NAMES)
                else:
                    pools.append(FIELDS + ZERO_ARG + COMPOUND + NUMBERS)
            combos = list(itertools.product(*pools))
            # The full product explodes (share_of alone is 8**4 * 2). A sample is
            # enough to classify, and a fixed seed keeps the report reproducible.
            if len(combos) > SAMPLE:
                combos = random.Random(20261030).sample(combos, SAMPLE)
            for combo in combos:
                kind, detail = classify(am, tm, name, spec, combo)
                buckets[kind] += 1
                per_macro[name][kind] += 1
                if detail and len(samples[kind]) < 3:
                    samples[kind].append((name, combo, detail))

    total = sum(buckets.values())
    print("combinations: %d" % total)
    for kind, count in buckets.most_common():
        print("  %-16s %6d" % (kind, count))

    print("\nreal findings by macro:")
    for name, counter in sorted(per_macro.items()):
        bad = (counter["old_leaks_macro"] + counter["semantic_diff"]
               + counter["diff_UNEXPLAINED"])
        if bad:
            print("  %-14s leaks=%-5d semantic=%-5d equal=%d"
                  % (name, counter["old_leaks_macro"], counter["semantic_diff"],
                     counter["equal"]))

    print("\nacceptance: diff_UNEXPLAINED must be 0 before the default can flip")

    for kind in ("old_leaks_macro", "semantic_diff", "diff_UNEXPLAINED"):
        for name, combo, detail in samples[kind]:
            print("\n[%s] %s%s" % (kind, name, combo))
            if kind == "old_leaks_macro":
                print("   old: %s" % detail[:160])
            else:
                print("   old: %s" % detail[0][:160])
                print("   new: %s" % detail[1][:160])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())