#!/usr/bin/env python3
"""Tests for the generated memory index.

The index used to be rebuilt by hand, once per merge, with a slightly different
throwaway script each time - which is how the workspace ended up with several
competing "versions" of it. These tests pin the two properties that make it a
projection instead: it is derived from the bundle, and `--check` notices when the
file has drifted.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "memory_index.py")


def build_workspace(entries):
    root = tempfile.mkdtemp()
    for layer, slug, title, ts in entries:
        folder = os.path.join(root, "memory", layer)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, slug + ".md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("---\ntype: memory\ntitle: %s\nlayer: %s\nupdated_at: %s\n---\n\nbody\n"
                         % (title, layer, ts))
    return root


def run(workspace, *args):
    return subprocess.run([sys.executable, TOOL, "--workspace", workspace, *args],
                          capture_output=True, text=True)


class MemoryIndexTests(unittest.TestCase):
    def test_the_index_is_one_line_per_concept_newest_first(self):
        root = build_workspace([
            ("experience", "old", "2026-11-01 旧的一条", "2026-11-01T00:00:00Z"),
            ("context", "new", "2026-11-04 新的一条", "2026-11-04T00:00:00Z"),
        ])
        self.assertEqual(run(root).returncode, 0)
        text = open(os.path.join(root, "MEMORY.md"), encoding="utf-8").read()
        lines = [line for line in text.splitlines() if line.startswith(("1.", "2."))]
        self.assertEqual(len(lines), 2)
        self.assertIn("新的一条", lines[0])
        self.assertIn("memory/context/new.md", lines[0])

    def test_a_date_range_does_not_leave_a_fragment_in_the_title(self):
        # A title starting "2026-11-03/04 ..." used to come out as "/04 ...".
        root = build_workspace([("experience", "range", "2026-11-03/04 四轮实验",
                                 "2026-11-04T00:00:00Z")])
        run(root)
        text = open(os.path.join(root, "MEMORY.md"), encoding="utf-8").read()
        line = [row for row in text.splitlines() if row.startswith("1.")][0]
        self.assertIn("四轮实验", line)
        self.assertNotIn("/04", line)

    def test_check_reports_a_stale_index(self):
        root = build_workspace([("context", "a", "2026-11-04 甲", "2026-11-04T00:00:00Z")])
        self.assertEqual(run(root, "--check").returncode, 1)     # not generated yet
        run(root)
        self.assertEqual(run(root, "--check").returncode, 0)
        with open(os.path.join(root, "MEMORY.md"), "a", encoding="utf-8") as handle:
            handle.write("stale line\n")
        self.assertEqual(run(root, "--check").returncode, 1)


if __name__ == "__main__":
    unittest.main()
