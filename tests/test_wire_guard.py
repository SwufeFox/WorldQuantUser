#!/usr/bin/env python3
"""Tests for the guard wiring: correct argument order, and no bypass left behind.

The argument-order bug this pins is worth stating: `simulate(settings, exprs)` and
`simulate(exprs, settings)` both compile, and py_compile cannot tell them apart. The
first run of the transformer produced the wrong one in all eight scripts, so the
rewrite now verifies itself with ast instead of trusting its own text.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import wire_guard  # noqa: E402

SAMPLE = '''
def main():
    session = ml.login()
    records = ml.simulate_alphas(
        session,
        [expr for expr, _ in ALPHAS],
        settings=SETTINGS,
        batch_size=3,
        labels=labels,
    )
    return records
'''


class Transform(unittest.TestCase):
    def test_expressions_come_first(self):
        out, count = wire_guard.transform(SAMPLE)
        self.assertEqual(count, 1)
        self.assertIn("guarded_sim.simulate(", out)
        first = wire_guard._first_expression(out)
        self.assertEqual(first, "[expr for expr, _ in ALPHAS]")

    def test_settings_is_second_and_session_is_a_keyword(self):
        out, _ = wire_guard.transform(SAMPLE)
        self.assertIn("SETTINGS,", out)
        self.assertIn("session=session", out)
        self.assertNotIn("ml.simulate_alphas", out)

    def test_verify_rejects_a_reversed_call(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "x.py")
            open(path, "w").write(
                "guarded_sim.simulate(\n    SETTINGS,\n    [a],\n    session=s,\n)\n")
            with self.assertRaises(SystemExit):
                wire_guard.verify(path, "[a]")

    def test_verify_accepts_a_correct_call(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "x.py")
            open(path, "w").write(
                "guarded_sim.simulate(\n    [a],\n    SETTINGS,\n    session=s,\n)\n")
            wire_guard.verify(path, "[a]")

    def test_call_without_settings_is_refused(self):
        with self.assertRaises(SystemExit):
            wire_guard.transform("x = ml.simulate_alphas(session, exprs)\n")


class NoBypass(unittest.TestCase):
    """Every simulation in the research directory must go through the guard."""

    def test_research_scripts_do_not_call_the_api_directly(self):
        research = os.environ.get("BRAIN_RESEARCH_DIR", "/data/worldquant-research")
        if not os.path.isdir(research):
            self.skipTest("no research directory on this machine")
        offenders = []
        for name in sorted(os.listdir(research)):
            if not name.endswith(".py"):
                continue
            text = open(os.path.join(research, name)).read()
            if "ml.simulate_alphas" in text:
                offenders.append(name)
        self.assertEqual(offenders, [],
                         "these scripts bypass guarded_sim: %s" % ", ".join(offenders))


if __name__ == "__main__":
    unittest.main()