#!/usr/bin/env python3
"""Named expression macros, so templates are written once and instantiated in bulk.

The campaign's repeated cost was hand-writing the same shape with a new field:
``ts_delta(ts_backfill(X, 252), 63) / (abs(ts_delay(ts_backfill(X, 252), 63)) + 1)``
appeared in a dozen batches with only ``X`` changing, and each transcription risked a
typo that lint would only sometimes catch.

A macro file is plain text::

    def growth(f, w=63):
        ts_delta(ts_backfill({f}, 252), {w}) / (abs(ts_delay(ts_backfill({f}, 252), {w})) + 1)

    def position():
        ((close - low) / (high - low) - 0.5)

    def carrier(f, fast, fw=1.0, w=63):
        -rank(growth({f}, {w})) - {fw}*rank({fast})

Macros may call other macros, so ``carrier`` expands ``growth`` itself. Expansion is
recursive with cycle detection, and every argument is substituted as text - which is all
FASTEXPR needs, because it has no types to check.

Two properties matter more than the substitution:

* an expansion is deterministic, so the same macro and arguments always produce the same
  expression, which is what makes the ledger's expression-level dedupe meaningful;
* every expansion is checked against the ledger before it is written into a batch, because
  a template applied twice to the same field is a duplicate that costs a simulation and
  proves nothing.
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

_DEF = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*:\s*$")
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_MACRO_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)")


class MacroError(ValueError):
    """A macro file that cannot be parsed, or a call that cannot be expanded."""


def parse_macros(text):
    """Return ``{name: {"params": [(name, default)], "body": str}}``."""
    macros = {}
    current = None
    body_lines = []

    def flush():
        if current is None:
            return
        body = " ".join(line.strip() for line in body_lines if line.strip())
        if not body:
            raise MacroError("macro %r has no body" % current["name"])
        current["body"] = body
        macros[current["name"]] = current

    for raw in text.splitlines():
        match = _DEF.match(raw)
        if match:
            flush()
            name, param_text = match.group(1), match.group(2)
            params = []
            for part in [p.strip() for p in param_text.split(",") if p.strip()]:
                if "=" in part:
                    key, default = part.split("=", 1)
                    params.append((key.strip(), default.strip()))
                else:
                    params.append((part, None))
            current, body_lines = {"name": name, "params": params}, []
            continue
        if current is not None:
            # A comment line between two definitions belongs to neither body. Without
            # this the trailing prose of a macro file is glued onto the previous body
            # and shows up inside the expanded expression.
            if raw.strip().startswith("#"):
                continue
            body_lines.append(raw)
    flush()
    return macros


def _split_args(text):
    """Split a call's arguments on commas that are not inside brackets."""
    parts, depth, buffer = [], 0, ""
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(buffer.strip())
            buffer = ""
        else:
            buffer += char
    if buffer.strip():
        parts.append(buffer.strip())
    return parts


def expand(macros, name, args=None, _stack=()):
    """Expand one macro call into an expression string."""
    if name in _stack:
        raise MacroError("macro cycle: %s" % " -> ".join(_stack + (name,)))
    if name not in macros:
        raise MacroError("unknown macro %r" % name)
    spec = macros[name]
    args = list(args or [])
    if len(args) > len(spec["params"]):
        raise MacroError("macro %r takes %d arguments, %d given"
                         % (name, len(spec["params"]), len(args)))
    bound = {}
    for index, (param, default) in enumerate(spec["params"]):
        if index < len(args):
            bound[param] = args[index]
        elif default is not None:
            bound[param] = default
        else:
            raise MacroError("macro %r is missing argument %r" % (name, param))

    text = spec["body"]
    # Substitute this macro's own parameters FIRST. Expanding nested calls before
    # substituting passes the outer macro's raw placeholders - "{f1}", "{w}" - into the
    # inner macro as literal arguments, and the inner one then fails with unresolved
    # placeholders of its own.
    for key, value in bound.items():
        text = text.replace("{%s}" % key, value)

    # Now expand any remaining macro calls, innermost first. Calls that are not macros
    # (rank, ts_delta, ...) are left alone: they are the platform's own operators.
    for _ in range(50):
        replaced = False
        for match in _MACRO_CALL.finditer(text):
            callee = match.group(1)
            if callee not in macros:
                continue
            inner = _split_args(match.group(2))
            resolved = expand(macros, callee, inner, _stack + (name,))
            text = text[:match.start()] + resolved + text[match.end():]
            replaced = True
            break
        if not replaced:
            break

    leftover = _PLACEHOLDER.findall(text)
    if leftover:
        raise MacroError("macro %r left placeholders unresolved: %s"
                         % (name, ", ".join(sorted(set(leftover)))))
    return " ".join(text.split())


def build(macros, plan, built=None):
    """Turn a plan into alphas, dropping duplicates of the ledger and of itself.

    ``plan`` entries are ``{macro, args, decay, candidate_id, mechanism, family}``. The
    return value is ``(alphas, skipped)`` where ``skipped`` says why an entry was dropped
    - a duplicate expression is the common case and must not be silently re-submitted.
    """
    if built is None:
        import alpha_registry as registry
        built = {entry.get("expression") for entry in registry.load_entries()
                 if entry.get("expression")}
    alphas, skipped, seen = [], [], set(built)
    for item in plan:
        expression = expand(macros, item["macro"], item.get("args") or [])
        key = (expression, item.get("decay"))
        if expression in seen:
            skipped.append({"candidate_id": item.get("candidate_id"),
                            "reason": "duplicate expression", "expression": expression})
            continue
        seen.add(expression)
        alphas.append({
            "expression": expression,
            "decay": item.get("decay", 4),
            "candidate_id": item.get("candidate_id"),
            "mechanism": item.get("mechanism"),
            "family": item.get("family", "other"),
        })
    return alphas, skipped


def lint_alphas(alphas, session=None, strict=True):
    """Run the shared linter over generated expressions.

    Macros must not become a way around the checks that were built today: unknown
    operators, VECTOR fields read by an operator that cannot take them, trailing
    tokens, and the operator budget. Generating a batch is exactly where those are
    cheapest to catch, because nothing has been submitted yet.

    Returns ``(clean, findings)``; ``findings`` carries one entry per problem with the
    candidate id attached so a report can point at the macro that produced it.
    """
    import expression_lint
    import machine_lib

    own_session = session is None
    if own_session:
        session = machine_lib.load_session() or machine_lib.authenticate()
    operator_index = expression_lint.build_operator_index(
        machine_lib.fetch_operators(session))
    # Field names come from the same index the CLI uses. There is no fetch_fields on
    # the client, so reading the cached catalogue is the way to get the field-level
    # checks rather than silently running operator checks only.
    known, types = None, None
    try:
        import research_paths
        index_path = os.path.join(research_paths.ensure_dir(), "index", "fields_live.json")
        with open(index_path, encoding="utf-8") as handle:
            fields = [f for f in json.load(handle) if isinstance(f, dict)]
        known = {f.get("id") for f in fields if f.get("id")}
        types = {f.get("id"): f.get("type") for f in fields if f.get("id")}
    except Exception:  # noqa: BLE001 - a missing index degrades to operator checks only
        pass

    findings, clean = [], 0
    for alpha in alphas:
        report = expression_lint.lint_expression(
            alpha["expression"], operator_index=operator_index,
            denylist=expression_lint.load_denylist(), known_fields=known,
            strict_fields=bool(known), field_types=types)
        problems = report["errors"] + ([w for w in report["warnings"]] if strict else [])
        if not problems:
            clean += 1
        for item in problems:
            findings.append({"candidate_id": alpha.get("candidate_id"),
                             "code": item.get("code"),
                             "message": item.get("message")})
    return clean, findings


def document(alphas, settings=None):
    """Wrap alphas in the research-file shape the CLI's simulate expects."""
    base = {"instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000",
            "delay": 1, "decay": 4, "neutralization": "SUBINDUSTRY",
            "truncation": 0.08, "testPeriod": "P1Y"}
    base.update(settings or {})
    return {"settings": base, "alphas": alphas}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--macros", required=True, help="Path to the macro file.")
    parser.add_argument("--expand", help="Print one expansion: NAME(arg1,arg2,...)")
    parser.add_argument("--plan", help="JSON plan to build into a research file.")
    parser.add_argument("--out", help="Where to write the research file.")
    parser.add_argument("--list", action="store_true", help="List macro names.")
    parser.add_argument("--lint", action="store_true",
                        help="Run the shared linter over the generated batch and refuse "
                             "to write a file that has errors.")
    args = parser.parse_args(argv)

    with open(args.macros, encoding="utf-8") as handle:
        macros = parse_macros(handle.read())
    if args.list:
        for name, spec in sorted(macros.items()):
            signature = ", ".join(p if d is None else "%s=%s" % (p, d)
                                  for p, d in spec["params"])
            print("%s(%s)" % (name, signature))
        return 0
    if args.expand:
        call = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$", args.expand.strip())
        if not call:
            raise MacroError("--expand expects NAME(arg,arg)")
        print(expand(macros, call.group(1), _split_args(call.group(2))))
        return 0
    if args.plan:
        with open(args.plan, encoding="utf-8") as handle:
            plan = json.load(handle)
        alphas, skipped = build(macros, plan.get("plan") or [])
        for item in skipped:
            print("skipped %s: %s" % (item["candidate_id"], item["reason"]))
        if args.lint:
            clean, findings = lint_alphas(alphas)
            print("lint: %d clean of %d" % (clean, len(alphas)))
            for item in findings:
                print("  %s %s: %s" % (item["candidate_id"], item["code"], item["message"]))
            if findings:
                raise MacroError("generated batch has %d lint problem(s); nothing written"
                                 % len(findings))
        target = args.out or (args.plan.replace(".json", "-alphas.json"))
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(document(alphas, plan.get("settings")), handle,
                      ensure_ascii=False, indent=1)
        print("wrote %s with %d alphas (%d skipped)" % (target, len(alphas), len(skipped)))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())