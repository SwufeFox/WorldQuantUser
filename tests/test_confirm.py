"""Irreversible submissions take two consecutive calls, not one."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_app import confirm  # noqa: E402
from research_app.runtime import CliError  # noqa: E402


class DoubleCallTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = self.dir.name

    def call(self, key="A1", now=1000.0, ttl=600, context=None):
        return confirm.require_double_call("alpha_submit", key, now=now, ttl=ttl,
                                           research_dir=self.root, context=context)

    def test_first_call_stages_and_refuses(self):
        with self.assertRaises(CliError) as ctx:
            self.call()
        self.assertEqual(ctx.exception.code, "CONFIRM_REQUIRED")

    def test_second_call_confirms(self):
        with self.assertRaises(CliError):
            self.call(now=1000.0)
        self.assertTrue(self.call(now=1010.0))

    def test_stage_is_consumed_by_the_confirmation(self):
        with self.assertRaises(CliError):
            self.call(now=1000.0)
        self.assertTrue(self.call(now=1010.0))
        with self.assertRaises(CliError):
            self.call(now=1020.0)

    def test_stale_stage_does_not_authorise(self):
        with self.assertRaises(CliError):
            self.call(now=1000.0)
        with self.assertRaises(CliError):
            self.call(now=1000.0 + 601)

    def test_keys_are_independent(self):
        with self.assertRaises(CliError):
            self.call(key="A1", now=1000.0)
        with self.assertRaises(CliError):
            self.call(key="A2", now=1000.0)
        self.assertTrue(self.call(key="A2", now=1001.0))
        self.assertTrue(self.call(key="A1", now=1001.0))

    def test_changed_request_must_be_confirmed_separately(self):
        with self.assertRaises(CliError):
            self.call(context={"selection_id": "first"})
        with self.assertRaises(CliError) as caught:
            self.call(now=1001.0, context={"selection_id": "second"})
        self.assertEqual(caught.exception.code, "CONFIRM_REQUIRED")
        self.assertTrue(self.call(now=1002.0, context={"selection_id": "second"}))

    def test_stage_file_is_written_in_the_research_dir(self):
        with self.assertRaises(CliError):
            self.call()
        data = json.loads((Path(self.root) / confirm.FILENAME).read_text())
        self.assertIn("A1", data["alpha_submit"])


if __name__ == "__main__":
    unittest.main()
