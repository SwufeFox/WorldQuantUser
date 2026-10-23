#!/usr/bin/env python3
"""The submit lock: an agent may not take an irreversible step by default.

Submission cannot be undone from the account holder's side, and an agent that can take it
eventually will - at 3am, without being asked. So absence of the variable means no, and only
an exact `ALLOW_AGENT_SUBMIT=true` opens it.

These tests check the two things that matter: the predicate refuses every ambiguous value,
and a real submission is refused *before* any HTTP call is made, not after.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import machine_lib as ml  # noqa: E402


class SubmitLockPredicateTests(unittest.TestCase):
    def test_missing_variable_is_closed(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(ml, "_load_dotenv", lambda: None):
            self.assertFalse(ml.agent_submit_allowed())

    def test_only_exact_true_opens_it(self):
        # Whitespace is stripped first: a stray space from a copy-paste carries no ambiguity
        # about intent, and silently keeping the lock shut would just confuse the person who
        # set it. Every other spelling stays closed, because there the intent *is* ambiguous.
        for value in ("1", "yes", "TRUE", "True", "on", "false", "tru", "yes please"):
            with mock.patch.dict(os.environ, {ml.SUBMIT_UNLOCK_ENV: value}, clear=True), \
                    mock.patch.object(ml, "_load_dotenv", lambda: None):
                self.assertFalse(ml.agent_submit_allowed(),
                                 "%r must not open the lock" % value)

    def test_surrounding_whitespace_is_forgiven(self):
        for value in (" true", "true ", "\ttrue\n"):
            with mock.patch.dict(os.environ, {ml.SUBMIT_UNLOCK_ENV: value}, clear=True), \
                    mock.patch.object(ml, "_load_dotenv", lambda: None):
                self.assertTrue(ml.agent_submit_allowed(), "%r is unambiguous" % value)

    def test_exact_true_opens_it(self):
        with mock.patch.dict(os.environ, {ml.SUBMIT_UNLOCK_ENV: "true"}, clear=True), \
                mock.patch.object(ml, "_load_dotenv", lambda: None):
            self.assertTrue(ml.agent_submit_allowed())


class SubmitLockEnforcementTests(unittest.TestCase):
    class Session:
        """Records anything that would touch the network."""

        def __init__(self):
            self.calls = []

        def get(self, *args, **kwargs):
            self.calls.append(("get", args))
            raise AssertionError("the lock must refuse before any request")

        def post(self, *args, **kwargs):
            self.calls.append(("post", args))
            raise AssertionError("the lock must refuse before any request")

        def patch(self, *args, **kwargs):
            self.calls.append(("patch", args))
            raise AssertionError("the lock must refuse before any request")

    def test_real_submission_is_refused_with_the_policy_message(self):
        session = self.Session()
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(ml, "_load_dotenv", lambda: None):
            with self.assertRaises(ml.SubmissionDisabled) as ctx:
                ml.submit_alpha(session, "ABC123", confirm=True)
        message = str(ctx.exception)
        self.assertIn(ml.SUBMIT_UNLOCK_ENV, message)
        self.assertIn("disabled", message)
        self.assertIn("self-correlation", message)
        self.assertEqual(session.calls, [], "nothing may reach the network")
        self.assertEqual(ml.SubmissionDisabled.code, "SUBMIT_DISABLED")

    def test_dry_run_is_not_gated(self):
        # A dry run is how a caller inspects checks; it must keep working with the lock shut.
        session = self.Session()
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(ml, "_load_dotenv", lambda: None):
            with self.assertRaises(AssertionError):
                # It proceeds far enough to touch the session, which is the point: the lock
                # did not stop it, the stub did.
                ml.submit_alpha(session, "ABC123", confirm=False)


if __name__ == "__main__":
    unittest.main()