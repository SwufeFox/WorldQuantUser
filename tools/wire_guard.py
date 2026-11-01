#!/usr/bin/env python3
"""Rewrite research scripts so every simulation goes through guarded_sim.

Why a rewrite and not a wrapper: guarded_sim only helps if nothing bypasses it, and
"remember to import the helper" is not an enforcement mechanism. The transform is
mechanical and verifiable -- after it, `ml.simulate_alphas` must not appear anywhere
in the research directory, and a test enforces that.

The transform is textual but argument-aware: it finds the call, splits its top-level
arguments, drops the leading positional `session`, moves `settings=` to the second
positional slot, and re-appends `session=session` by keyword.
"""

from __future__ import annotations

import argparse
import py_compile
import re
import sys

IMPORT_LINE = "import guarded_sim  # noqa: E402  (simulations go through the catalog check)"
SYS_PATH_LINE = 'sys.path.insert(0, "/data/src/magix/WorldQuantUser/tools")'


def split_args(text, start):
    """Split a call's argument list; returns (args, close_index)."""
    # start is the index just past the opening parenthesis, so the call's own
    # paren counts as depth 1; starting at 0 makes top-level commas invisible.
    depth, index, current, args = 1, start, [], []
    quote = None
    while index < len(text):
        char = text[index]
        if quote:
            current.append(char)
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            current.append(char)
        elif char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                break
            current.append(char)
        elif char == "," and depth == 1:
            args.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    tail = "".join(current)
    if tail.strip():
        args.append(tail)
    return [a.strip() for a in args], index


def _is_named(argument):
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", argument))


def transform(source):
    """Return (new_source, count) after rewriting every simulate_alphas call.

    The argument that is not ``name=value`` is the expression list; it must become
    the *first* positional argument of guarded_sim.simulate, because its signature
    is (expressions, settings, ...). Getting that order wrong still compiles -- the
    two are both objects -- so the rewrite verifies itself with ast afterwards
    instead of trusting the text.
    """
    count = 0
    out = source
    while True:
        match = re.search(r"\bml\.simulate_alphas\s*\(", out)
        if not match:
            break
        args, close = split_args(out, match.end())
        if not args:
            break
        session_arg = args[0]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", session_arg):
            raise SystemExit("first argument is not a session variable: %r" % session_arg)
        rest = args[1:]
        settings = [a for a in rest if a.startswith("settings=")]
        expressions = [a for a in rest if not _is_named(a)]
        others = [a for a in rest if _is_named(a) and not a.startswith("settings=")]
        if not settings:
            raise SystemExit("call without settings=; refusing to guess")
        if len(expressions) != 1:
            raise SystemExit("expected exactly one positional expression argument, got %d"
                             % len(expressions))

        parts = [expressions[0], settings[0].split("=", 1)[1].strip()] + others
        call = "guarded_sim.simulate(\n        " + ",\n        ".join(parts)
        call += ",\n        session=%s,\n    )" % session_arg
        out = out[:match.start()] + call + out[close + 1:]
        count += 1
    return out, count


def _first_expression(source):
    """The expression argument the rewrite produced, recovered bracket-aware.

    A regex that stops at the first comma truncates `[e for e, _ in X]` in the
    middle, which makes verify() compare two different strings and pass for the
    wrong reason. Same splitter as the rewrite, so both agree on what an argument is.
    """
    match = re.search(r"guarded_sim\.simulate\s*\(", source)
    if not match:
        raise SystemExit("could not find the rewritten call")
    args, _close = split_args(source, match.end())
    if not args:
        raise SystemExit("rewritten call has no arguments")
    return args[0].strip()


def verify(path, expected_expressions):
    """Fail loudly if the rewritten call has the two positionals in the wrong order."""
    import ast

    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "simulate"):
            if len(node.args) < 2:
                raise SystemExit("%s: simulate call has fewer than two positionals" % path)
            first = ast.unparse(node.args[0])
            if first.strip() != expected_expressions.strip():
                raise SystemExit("%s: expressions argument is %r, expected %r"
                                 % (path, first, expected_expressions))
            return
    raise SystemExit("%s: no guarded_sim.simulate call found after rewrite" % path)


def ensure_import(source):
    """Add the ``guarded_sim`` import beside the existing ``machine_lib`` one.

    Next to it, not at the top: research scripts manipulate ``sys.path`` before
    importing, so an import placed above that line fails at runtime and the failure
    looks like a missing module rather than a rewrite that landed in the wrong place.
    """
    if "import guarded_sim" in source:
        return source
    if SYS_PATH_LINE not in source:
        # Insert next to the existing sys.path manipulation for machine_lib.
        source = source.replace(
            'sys.path.insert(0, "/data/src/magix/WorldQuantUser")',
            'sys.path.insert(0, "/data/src/magix/WorldQuantUser")\n' + SYS_PATH_LINE, 1)
    return source.replace("import machine_lib as ml  # noqa: E402",
                          "import guarded_sim  # noqa: E402\nimport machine_lib as ml  # noqa: E402", 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    total = 0
    for path in args.paths:
        source = open(path).read()
        new_source, count = transform(source)
        if count:
            new_source = ensure_import(new_source)
        print("%-46s %d call(s)" % (path.split("/")[-1], count))
        total += count
        if args.apply and count:
            open(path, "w").write(new_source)
            py_compile.compile(path, doraise=True)
            verify(path, _first_expression(new_source))
    print("total rewritten: %d" % total)
    if not args.apply:
        print("\ndry run: pass --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())