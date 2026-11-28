#!/usr/bin/env python3
"""Tests for the reinstated `alpha describe` command and its ledger journal."""

import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research_app import alphas


def _args(**overrides):
    base = dict(alpha_id="A1", name="name", category=None, tags=None,
                description=None, color=None, hide=False, unhide=False)
    base.update(overrides)
    return Namespace(**base)


class DescribeJournalTests(unittest.TestCase):
    def test_metadata_change_is_appended_to_the_workspace_log(self):
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": root}), \
                mock.patch.object(alphas, "with_session", side_effect=lambda _a, work: work(object())), \
                mock.patch.object(alphas.ml, "set_alpha_metadata",
                                  return_value={"id": "A1", "name": "name", "tags": ["t"]}):
            data = alphas.cmd_alpha_describe(_args(tags=["t"]), None)
            log = os.path.join(root, "metadata-log.jsonl")
            self.assertTrue(os.path.isfile(log))
            rows = [json.loads(line) for line in open(log, encoding="utf-8")]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["alpha_id"], "A1")
        self.assertEqual(rows[0]["changed"]["name"], "name")
        self.assertEqual(data["metadata_log"], log)

    def test_nothing_to_change_is_a_usage_error(self):
        from research_app.runtime import CliError

        with self.assertRaises(CliError):
            alphas.cmd_alpha_describe(_args(name=None), None)


if __name__ == "__main__":
    unittest.main()
