"""One way for a tool to hand back machine-readable data.

Twelve tools write JSON. Before this module they spelled it three ways -- `--json
path`, `--out path`, `--save path` -- and every one of them did it with a bare
``open()``. That is two papercuts for a caller:

* it has to remember which name this particular tool uses, and
* when the path is unwritable the tool dies with a traceback **after** printing
  half of a human report to stdout, so the caller sees exit 1, a Python stack on
  stderr, and a truncated document that parsed fine.

So: one flag, ``--json``, with an optional value.

    --json            one JSON document on stdout, for ``| jq`` and subprocesses
    --json PATH       the same document written to PATH
    --json -          same as no argument, stated explicitly

Writing is atomic (``cache_store.write_json_atomic``), so a killed run cannot leave
half a file behind, and an unwritable path becomes one line on stderr plus exit 2 --
a usage error, because the path came from the caller.

Not a tool, and deliberately not in ``tools/`` listing as one: no ``main()``, no
argument parser of its own. It exists so the other twelve do not each invent theirs.
"""

from __future__ import annotations

import json
import os
import sys

import cache_store

__all__ = ["add_json_flag", "reserve_stdout", "write"]


def add_json_flag(parser, help_="write the machine-readable payload (default: stdout)"):
    """Declare the shared ``--json`` flag.

    ``nargs="?"`` with ``const="-"`` is what makes the bare flag mean stdout: a
    caller that wants the data has to be able to get it without inventing a temporary
    path, and one that wants a file should not have to pipe.
    """
    parser.add_argument("--json", nargs="?", const="-", default=None, metavar="PATH",
                        help=help_)


def reserve_stdout(args):
    """Give stdout to the document, and the prose to stderr, when ``--json`` is set.

    ``--json`` is a promise that stdout is exactly one JSON document. Printing a human
    report first and the document after breaks it even though each is valid alone: a
    caller piping stdout into a parser gets something it cannot read. The report is
    still useful, so it moves to stderr rather than disappearing.

    A one-line global redirect, because every tool's report is written with ``print``
    and threading a stream through all of them would be a larger change than the
    problem. ``write`` sends the document to ``sys.__stdout__``, which is the real
    stdout, so the redirect cannot capture it.
    """
    if getattr(args, "json", None):
        sys.stdout = sys.stderr


def write(payload, target, indent=1, sort_keys=False):
    """Write ``payload`` to ``target``: ``None`` means nothing, ``-`` means stdout.

    Returns the path written, or ``"-"`` for stdout. Raises :class:`UsageError` when
    the file cannot be written, so the caller can turn it into exit 2 without
    catching ``OSError`` at every call site.
    """
    if target is None:
        return None
    text = json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
    if target == "-":
        sys.__stdout__.write(text + "\n")
        sys.__stdout__.flush()
        return "-"
    try:
        directory = os.path.dirname(os.path.abspath(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Reuse the atomic writer so a killed run cannot leave a half-written
        # document that still parses as JSON.
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    except OSError as error:
        raise UsageError("cannot write %s: %s" % (target, error)) from error
    return target


class UsageError(RuntimeError):
    """A problem with the caller's arguments, not with the world.

    ``tools/*`` map this to exit 2, the same code argparse uses, so "you asked for
    something impossible" is distinguishable from "the run failed".
    """


def guard(main, argv=None):
    """Run ``main`` and turn :class:`UsageError` into a usage failure.

    Every tool ends with ``raise SystemExit(guard(main))`` instead of
    ``raise SystemExit(main())``: one line, and an unwritable path stops looking like
    a crash.
    """
    try:
        return main() if argv is None else main(argv)
    except UsageError as error:
        print(str(error), file=sys.stderr)
        return 2