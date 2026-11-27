"""BRAIN client exceptions and diagnostic output handlers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timezone
from functools import cache
from os import environ
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import cache_store
from check_evidence import normalize_checks, platform_check_names
import machine_lib as _ml
import research_paths


def set_diagnostic_handler(handler):
    """Install a diagnostic handler and return the previous one for restoration."""
    pass
    previous = _ml._diagnostic_handler
    _ml._diagnostic_handler = handler
    return previous


def note(message):
    """Emit one diagnostic line. Never writes to stdout."""
    handler = _ml._diagnostic_handler
    if handler is not None:
        handler(message)
    else:
        print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Errors
#
# Everything this module raises on purpose derives from BrainError and carries
# a stable ``code``. Callers map ``code`` -> process exit code, so failures stay
# machine-readable instead of leaking a traceback shape.
# ---------------------------------------------------------------------------
class BrainError(RuntimeError):
    """Base exception raised by ``machine_lib`` with a stable CLI error code."""

    code = "BRAIN_ERROR"
    #: Optional actionable next step, surfaced verbatim by the CLI.
    hint = None


class MissingCredentials(BrainError):
    """Raised when ``BRAIN_USERNAME`` or ``BRAIN_PASSWORD`` is missing."""

    code = "AUTH_MISSING"
    hint = "set BRAIN_USERNAME and BRAIN_PASSWORD in the environment, or copy .env.example to .env next to machine_lib.py"


class AuthError(BrainError):
    """Raised when BRAIN rejects the configured credentials or session."""

    code = "AUTH_FAILED"
    hint = "check BRAIN_USERNAME/BRAIN_PASSWORD and retry with --relogin (or `auth check`)"


class CaptchaRequired(AuthError):
    """BRAIN wants a human captcha before it will hand out a session."""

    code = "AUTH_CAPTCHA"
    hint = ("too many logins in a short window; wait a few minutes, then retry. "
            "Reusing the cached session avoids this.")


class NotFound(BrainError):
    """Raised when a BRAIN endpoint or alpha/simulation ID is not found (HTTP 404)."""

    code = "NOT_FOUND"


class RateLimited(BrainError):
    """Raised when BRAIN returns HTTP 429 or an async read exceeds its wait budget."""

    code = "RATE_LIMITED"
    hint = "respect Retry-After / retry_at before resuming or polling the existing task"

    def __init__(self, *args, retry_after=None):
        super().__init__(*args)
        self.retry_after = retry_after


class SimulationError(BrainError):
    """Raised when a BRAIN simulation fails or rejects its payload."""

    code = "SIMULATION_FAILED"
    hint = "inspect the reported error/checks and run `lint` or `probe --dry-run` before retrying"


class SubmissionDisabled(BrainError):
    """Raised when a real submission is attempted while the lock is closed."""

    code = "SUBMIT_DISABLED"
    hint = ("this is a policy lock, not a failure: set ALLOW_AGENT_SUBMIT=true in the "
            "environment or .env to allow real submissions on this account")


def agent_submit_allowed():
    """Whether an agent may take the irreversible step of submitting an alpha.

    Only an explicit ``ALLOW_AGENT_SUBMIT=true`` opens the lock. Surrounding whitespace is
    forgiven - a stray space from a copy-paste says nothing about intent - but every other
    spelling (``1``, ``yes``, ``TRUE``, ``on``) leaves it closed: there the intent is
    ambiguous, and ambiguity resolves to "no".
    """
    _ml._load_dotenv()
    return (environ.get(_ml.SUBMIT_UNLOCK_ENV, "") or "").strip() == "true"


class ServiceError(BrainError):
    """A transport or unexpected HTTP failure while talking to BRAIN."""

    code = "SERVICE_ERROR"


class ReadInterrupted(ServiceError):
    """A read could not finish after transient transport or gateway errors."""

    hint = "retry or recover the existing checkpoint/alpha ID; do not resubmit accepted tasks"


class AuthenticationUnavailable(ServiceError):
    """A transient transport or server failure interrupted authentication."""

    hint = "BRAIN authentication was temporarily unavailable; wait briefly and retry"


class AuthenticationDeadline(ServiceError):
    """The caller's budget for starting or waiting for authentication expired."""

    hint = "increase --max-wait-seconds or retry once authentication completes"


class SessionBusy(ServiceError):
    """Another process is acquiring the shared account session."""

    code = "SESSION_BUSY"
    hint = "retry shortly to reuse the session created by the active login"


class PermissionDenied(ServiceError):
    """BRAIN denied access to an endpoint with HTTP 403."""

    code = "ACCESS_DENIED"
    hint = "this account tier lacks access to the endpoint; verify permissions with `auth check` or `api probe`"
