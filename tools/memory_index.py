#!/usr/bin/env python3
"""Regenerate MEMORY.md from the memory bundle, so the index stops being a version.

Why this exists: `MEMORY.md` was rebuilt by hand three times in two days, each time
with a slightly different ad-hoc script, and the backend re-materialises it in its own
long form whenever a memory is written. That produced exactly what it sounds like -
several competing versions of "the index", each claiming to be current.

The fix is to make the index a *projection*: one generator, one rule, and a `--check`
mode that says whether the checked-in file matches the bundle. Nothing about the
knowledge lives here; the 24 files under `memory/<layer>/` are the truth, and this
prints a one-line pointer to each.

    python3 tools/memory_index.py              # write MEMORY.md
    python3 tools/memory_index.py --check      # exit 1 if it is out of date
    python3 tools/memory_index.py --print      # show what it would write
"""

from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from _output import add_json_flag, guard, reserve_stdout, write  # noqa: E402

TITLE_LIMIT = 62
HEADER = """# MEMORY

索引，不是正文。每条一行，正文在 `memory/<layer>/<slug>.md`。
研究协议见 `/data/worldquant-research/protocol.md`：**训练不看 test**；**模拟不花钱**（省的是时间不是钱）。
研究状态别从这份索引读：用 `research_cli.py brief`、`open-threads.md`、`research-index-20261104.md`。
"""


def workspace_root(explicit=None):
    """The bot workspace: the directory that owns AGENTS.md and memory/."""
    if explicit:
        return os.path.abspath(explicit)
    for candidate in (os.environ.get("MEMOH_WORKSPACE"), "/data", ROOT):
        if candidate and os.path.isdir(os.path.join(candidate, "memory")):
            return os.path.abspath(candidate)
    raise SystemExit("cannot find the workspace; pass --workspace")


def _front_matter(text):
    match = re.search(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    return meta, match.group(2)


def _short_title(meta, path):
    """One line, no dates, no merge provenance: those are versions, not labels."""
    title = (meta.get("title") or os.path.splitext(os.path.basename(path))[0])
    title = title.split("｜")[0]
    # A date *range* ("2026-11-03/04") leaves a stray "/04" if only the first
    # date is stripped, which is how one index line ended up starting with "/04".
    title = re.sub(r"^20\d\d-\d\d-\d\d(/\d{1,2})?\s*", "", title)
    for cut in ("。", "（合并自", "（原文在", "；"):
        title = title.split(cut)[0]
    title = re.sub(r"\s+", " ", title).strip(" .")
    if len(title) > TITLE_LIMIT:
        title = title[:TITLE_LIMIT].rstrip() + "…"
    return title


def collect(root):
    rows = []
    for base, _dirs, files in os.walk(os.path.join(root, "memory")):
        for name in files:
            if not name.endswith(".md"):
                continue
            path = os.path.join(base, name)
            meta, _body = _front_matter(open(path, encoding="utf-8").read())
            rows.append((meta.get("updated_at") or meta.get("timestamp") or "1970",
                         os.path.relpath(path, root), _short_title(meta, path)))
    rows.sort(key=lambda row: row[0], reverse=True)
    return rows


def render(rows):
    lines = HEADER.splitlines() + [""]
    for index, (_ts, path, title) in enumerate(rows, 1):
        lines.append("%d. [%s](%s)" % (index, title, path))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default=None,
                        help="Directory holding memory/ (default: /data).")
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 when MEMORY.md differs from the generated text.")
    parser.add_argument("--print", dest="show", action="store_true",
                        help="Print the generated index instead of writing it.")
    add_json_flag(parser, help_="write the entry list as JSON")
    args = parser.parse_args()
    reserve_stdout(args)

    root = workspace_root(args.workspace)
    rows = collect(root)
    text = render(rows)
    target = os.path.join(root, "MEMORY.md")
    current = ""
    if os.path.exists(target):
        current = open(target, encoding="utf-8").read()

    if args.show:
        print(text, end="")
        return 0
    if args.check:
        stale = current != text
        if stale:
            print("MEMORY.md is out of date (%d entries in the bundle)"
                  % len(rows), file=sys.stderr)
        return 1 if stale else 0
    if current != text:
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(text)
    print("wrote %s (%d entries, %d bytes)" % (target, len(rows), len(text.encode())))
    if args.json:
        write([{"title": title, "path": path} for _ts, path, title in rows], args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(guard(main))