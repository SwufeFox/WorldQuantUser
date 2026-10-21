#!/usr/bin/env python3
"""Transpile the 101 formulaic alphas into this platform's FASTEXPR dialect.

The paper's operators are renamed rather than reimplemented: ``correlation`` is
``ts_corr`` here, ``decay_linear`` is ``ts_decay_linear``, and
``indneutralize(x, IndClass.sector)`` becomes ``group_neutralize(x, sector)``. Three
things genuinely do not exist on this platform — ``ts_min``, ``ts_max`` and
``floor`` — so any formula that needs them is reported instead of being silently
rewritten into something else.

Refusals are the point. A transpiler that guesses produces formulas that look like
the paper and measure something else, which is worse than an empty slot: it would
be recorded as a test of the paper's claim.

"Salting" lives here too, because the user asked for volume rather than rarity. Salt
is applied only along axes the paper itself leaves free — decay, neutralisation, and
the constants it scribbles with arbitrary precision (``19.8975``, ``6.28259``) — not
by changing the economic content of a formula, which would make the result
unattributable to anything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

#: Paper operator -> platform operator. Identity names are listed so the diff is
#: explicit rather than implied.
RENAME = {
    "correlation": "ts_corr",
    "covariance": "ts_covariance",
    "delta": "ts_delta",
    "delay": "ts_delay",
    "sum": "ts_sum",
    "ts_sum": "ts_sum",
    "stddev": "ts_std_dev",
    "ts_stddev": "ts_std_dev",
    "decay_linear": "ts_decay_linear",
    "ts_rank": "ts_rank",
    "ts_argmax": "ts_arg_max",
    "ts_argmin": "ts_arg_min",
    "signedpower": "signed_power",
    "ts_product": "ts_product",
    "product": "ts_product",
    "ln": "log",
    "log": "log",
    "abs": "abs",
    "sign": "sign",
    "rank": "rank",
    "scale": "scale",
    "min": "min",
    "max": "max",
    "power": "power",
    "indneutralize": "group_neutralize",
    "sqrt": "sqrt",
}

#: Operators this platform does not provide at all.
ABSENT = {"ts_min", "ts_max", "floor"}

_INDUSTRY = re.compile(r"IndClass\.(sector|industry|subindustry)", re.I)
_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_NUMBER = re.compile(r"(?<![\w.])(\d+\.\d+|\d+)(?![\w.])")


def strip_appendix(text):
    """Cut the trailing prose the PDF extractor glued onto the last formula."""
    for marker in (" A.1.", " Appendix", " Functions and Operators"):
        if marker in text:
            text = text.split(marker)[0]
    return " ".join(text.split())


def normalize_syntax(expr):
    """Turn the paper's infix/ternary syntax into FASTEXPR.

    Only three rewrites happen, each mechanical:

    * ``&&`` / ``||`` / ``!`` become ``and`` / ``or`` / ``not``;
    * a *flat* ``cond ? a : b`` becomes ``if_else(cond, a, b)``;
    * ``IndClass.x`` becomes the group field name.

    A nested ternary is refused: rewriting it without a parser risks changing which
    branch belongs to which condition, and a wrong branch is indistinguishable from
    a real result.
    """
    text = expr
    text = text.replace("&&", " and ").replace("||", " or ")
    text = _INDUSTRY.sub(lambda m: m.group(1).lower(), text)

    if text.count("?") != text.count(":"):
        raise ValueError("unbalanced ternary")
    if text.count("?") > 1:
        raise ValueError("nested ternary; needs a real parser")

    if "?" in text:
        # The ternary may sit inside other calls, as in
        # SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2).
        # Depth must therefore be measured from the start of the string and the
        # colon accepted only at the same depth as the question mark; scanning from
        # the "?" alone put the split in the wrong place and produced
        # if_else(...) with the wrong arguments.
        depth = 0
        q_index = q_depth = None
        colon_index = None
        for index, char in enumerate(text):
            if char in "([":
                depth += 1
            elif char in ")]":
                depth -= 1
            elif char == "?" and q_index is None:
                q_index, q_depth = index, depth
            elif char == ":" and q_index is not None and depth == q_depth:
                colon_index = index
                break
        if q_index is None or colon_index is None:
            raise ValueError("ternary without a matching colon at the same depth")

        # Extend back to the delimiter that opens the conditional expression.
        start = 0
        depth = q_depth
        for index in range(q_index - 1, -1, -1):
            char = text[index]
            if char in ")]":
                depth += 1
            elif char in "([":
                depth -= 1
                if depth < q_depth:
                    start = index + 1
                    break
            elif char == "," and depth == q_depth:
                start = index + 1
                break

        # And forward to the delimiter that closes it.
        end = len(text)
        depth = q_depth
        for index in range(colon_index + 1, len(text)):
            char = text[index]
            if char in "([":
                depth += 1
            elif char in ")]":
                depth -= 1
                if depth < q_depth:
                    end = index
                    break
            elif char == "," and depth == q_depth:
                end = index
                break

        condition = text[start:q_index].strip()
        when_true = text[q_index + 1:colon_index].strip()
        when_false = text[colon_index + 1:end].strip()
        text = "%sif_else(%s, %s, %s)%s" % (
            text[:start], condition, when_true, when_false, text[end:])
    return " ".join(text.split())


#: The paper writes ``adv50``/``adv81``/``adv180``; this platform only publishes
#: ``adv20``. Substituting a different averaging window changes what the formula
#: measures, so by default such a formula is REFUSED. ``--adapt-adv`` opts into the
#: substitution and tags the variant, because a silent rewrite would be recorded as a
#: test of the paper's claim when it is a test of a different one.
ADV_FALLBACK = {"adv50", "adv60", "adv81", "adv120", "adv150", "adv180"}


def canonical_case(text):
    """Rewrite every function name to the platform's own spelling.

    The paper is CamelCase (``Ts_Rank``, ``Sign``, ``IndNeutralize``) and this
    platform is snake_case. A case-insensitive replacement of renamed operators left
    the identity ones untouched, so 46 of 80 translations were rejected for
    ``Ts_Rank`` alone.
    """
    def swap(match):
        name = match.group(1)
        ours = RENAME.get(name.lower())
        return (ours + "(") if ours else (name + "(")

    return _CALL.sub(swap, text)


def transpile(expr, adapt_adv=False):
    """Return ``(fastexpr, notes)``; raise ``ValueError`` when not translatable.

    ``adapt_adv`` is opt-in: without it, a formula using a wider adv window is
    refused rather than quietly measured with a different window.
    """
    text = strip_appendix(normalize_syntax(expr))
    used = {name.lower() for name in _CALL.findall(text)}
    absent = sorted(used & ABSENT)
    if absent:
        raise ValueError("platform has no %s" % ", ".join(absent))
    unknown = sorted(name for name in used
                     if name not in RENAME and name not in {"if_else"})
    if unknown:
        raise ValueError("unmapped function(s): %s" % ", ".join(unknown))

    text = canonical_case(text)
    adapted = []
    for name in sorted(ADV_FALLBACK):
        pattern = r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(name)
        if not re.search(pattern, text):
            continue
        if not adapt_adv:
            raise ValueError("uses %s, which this platform does not publish; "
                             "pass --adapt-adv to substitute adv20 and label the "
                             "result as adapted" % name)
        text = re.sub(pattern, "adv20", text)
        adapted.append(name + "->adv20")
    # Single quotes around numbers are not valid here.
    text = text.replace("'", "")
    return text, adapted


def salt(expr, decay_variants, jitter=0.0, seed=0):
    """Yield salted variants: a different decay, and optionally jittered constants.

    The paper's own constants are arbitrary to five decimals; nudging them is a
    legitimate axis to vary. Changing an operator or a comparison is not: that would
    alter what the formula measures and the result could no longer be attributed to
    the paper's claim.
    """
    import random
    for index, decay in enumerate(decay_variants):
        if not jitter:
            yield "%s@decay%s" % (expr, decay), expr, decay
            continue
        rng = random.Random(seed + index * 7919)

        def bump(match):
            value = float(match.group(1))
            # Integers are window lengths and counts. Jittering 20 into 19.9 makes
            # an invalid lookback and the expression is rejected outright, so only
            # the paper's arbitrary-precision constants are nudged.
            if value == int(value):
                return match.group(1)
            return ("%.5f" % (value * (1 + rng.uniform(-jitter, jitter)))).rstrip("0").rstrip(".")

        jittered = _NUMBER.sub(bump, expr)
        yield "%s@decay%s+j" % (expr, decay), jittered, decay


def load_formulas(path):
    formulas = {}
    for line in open(path, encoding="utf-8"):
        if "\t" not in line:
            continue
        number, expr = line.strip().split("\t", 1)
        formulas[int(number)] = expr
    return formulas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="/tmp/alphas101.txt")
    parser.add_argument("--out", default=None,
                        help="Write a research JSON of the translatable formulas.")
    parser.add_argument("--salts", default="4,8,12",
                        help="Decays to generate per formula (comma separated).")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="Relative constant jitter, e.g. 0.25.")
    parser.add_argument("--adapt-adv", action="store_true",
                        help="Substitute adv20 for the wider windows the platform "
                             "does not publish, tagging the result as adapted.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    formulas = load_formulas(args.source)
    decays = [int(part) for part in args.salts.split(",") if part.strip()]
    ok, refused, adaptations = {}, {}, {}
    for number in sorted(formulas):
        try:
            expr, adapted = transpile(formulas[number], adapt_adv=args.adapt_adv)
        except ValueError as exc:
            refused[number] = str(exc)
            continue
        ok[number] = expr
        if adapted:
            adaptations[number] = adapted

    print("translations: %d ok, %d refused" % (len(ok), len(refused)))
    if adaptations:
        print("adapted (declared substitutions): %d formulas use adv20 in place of"
              " a wider window" % len(adaptations))
    reasons = {}
    for number, reason in refused.items():
        key = reason.split(":")[0] if ":" in reason else reason
        reasons.setdefault(key, []).append(number)
    for reason, numbers in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        print("  %-42s %3d  %s" % (reason[:42], len(numbers),
                                   ", ".join("#%d" % n for n in numbers[:8])
                                   + (" ..." if len(numbers) > 8 else "")))

    alphas, seen = [], set()
    for number in sorted(ok):
        for label, expr, decay in salt(ok[number], decays, args.jitter, seed=number):
            # Key on the settings too: three decay variants of one formula are three
            # alphas, and deduping on the expression alone silently kept only one.
            key = (expr, decay)
            if key in seen:
                continue
            seen.add(key)
            alphas.append({
                "expression": expr,
                "decay": decay,
                "candidate_id": "WQ%03d-%s" % (number, label.split("@")[-1]),
                "mechanism": "paper alpha #%d%s%s" % (
                    number,
                    " jittered" if "+j" in label else "",
                    " (adv window adapted)" if number in adaptations else ""),
                "family": "other",
            })
    if args.limit:
        alphas = alphas[: args.limit]
    print("variants generated: %d (%d formulas x %d decays)"
          % (len(alphas), len(ok), len(decays)))

    if args.out:
        document = {
            "settings": {"instrumentType": "EQUITY", "region": "USA",
                         "universe": "TOP3000", "delay": 1, "decay": 4,
                         "neutralization": "SUBINDUSTRY", "truncation": 0.08,
                         "testPeriod": "P1Y"},
            "alphas": alphas,
        }
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=1)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())