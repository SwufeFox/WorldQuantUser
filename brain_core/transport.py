"""HTTP polling, timeout calculation, URL validation and request_json transport."""

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


_SIMULATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def valid_simulation_id(value):
    """Check the shape of a bare id before using it as a path segment."""
    return isinstance(value, str) and bool(_ml._SIMULATION_ID.fullmatch(value))


def simulation_reference_id(reference):
    """Extract a bare id or the id from one exact BRAIN API simulation URL."""
    if _ml.valid_simulation_id(reference):
        return reference
    message = "expected a simulation id or https://api.worldquantbrain.com/simulations/ID"
    if not isinstance(reference, str) or any(
            character.isspace() or ord(character) < 32 or character in "\\?#"
            for character in reference):
        raise ValueError(message)
    try:
        url = urlsplit(reference)
    except ValueError as exc:
        raise ValueError(message) from exc
    base, separator, simulation_id = url.path.removesuffix("/").rpartition("/")
    if (url.scheme != "https" or url.netloc.lower() not in {
            "api.worldquantbrain.com", "api.worldquantbrain.com:443"}
            or not separator or base != "/simulations" or not _ml.valid_simulation_id(simulation_id)):
        raise ValueError(message)
    return simulation_id


def _validated_progress_url(location):
    """Resolve a BRAIN Location header and reject unsafe resume targets.

    Progress URLs are persisted in local checkpoints. Treat those files as
    untrusted input: a forged URL must not make the authenticated session send
    cookies to an arbitrary host when ``resume`` polls it.
    """
    if not isinstance(location, str) or not location or any(
        ord(character) < 32 or character == "\\" for character in location
    ):
        raise _ml.SimulationError("invalid simulation progress URL")
    resolved = urljoin(_ml.BRAIN_API_URL + "/", location)
    try:
        parsed = urlsplit(resolved)
        port = parsed.port
    except ValueError as exc:
        raise _ml.SimulationError("invalid simulation progress URL") from exc
    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() != "api.worldquantbrain.com"
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or not parsed.path.startswith("/simulations/")
        or parsed.fragment
    ):
        raise _ml.SimulationError(
            "simulation progress URL must target the BRAIN HTTPS simulations endpoint"
        )
    return resolved


def _quick_progress_target(location):
    """Return the original request URL and exact simulation ID for a Quick receipt."""
    url = _ml._validated_progress_url(location)
    try:
        simulation_id = _ml.simulation_reference_id(urlsplit(url)._replace(query="").geturl())
    except ValueError as exc:
        raise _ml.SimulationError("Quick Location must identify one BRAIN simulation") from exc
    return url, simulation_id


def _request_timeout(deadline):
    """Cap a Requests connect/read timeout at the remaining phase budget."""
    if deadline is None:
        return 60
    return max(0.001, min(60, deadline - _ml.time.monotonic()))


def _retry_after_seconds(response, cap_error_wait=True):
    """Seconds BRAIN wants us to wait, or 0 when the request is ready.

    ``Retry-After`` drives both the progress protocol and submission rate
    limiting. It is parsed leniently; callers may cap non-2xx backoff for
    ordinary requests, while bounded simulation polling honors the full delay
    and checkpoints instead of sleeping past its deadline.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return 0.0
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not response.ok and cap_error_wait:
        return min(seconds, 120.0)
    return max(seconds, 0.0)


_TRANSIENT_READ_STATUSES = {408, 502, 503, 504}


def _transient_read_error(exc):
    from requests.exceptions import ChunkedEncodingError, ConnectionError, ContentDecodingError, SSLError, Timeout

    return (isinstance(exc, (ChunkedEncodingError, ConnectionError, ContentDecodingError, Timeout))
            and not isinstance(exc, SSLError))


def _wait_for_read_retry(attempt, attempts, deadline, retry_after=0):
    wait = retry_after if math.isfinite(retry_after) and retry_after > 0 else min(2.0, 0.5 * attempt)
    if attempt >= attempts or (deadline is not None and wait >= deadline - _ml.time.monotonic()):
        return False
    _ml.sleep(wait)
    return True


def _get_poll_response(s, url, deadline):
    """Retry interrupted reads while leaving the endpoint's readiness protocol intact."""
    from requests.exceptions import RequestException

    attempts = 3
    last = None
    for attempt in range(1, attempts + 1):
        if last is not None and deadline is not None and _ml.time.monotonic() >= deadline:
            raise last
        try:
            response = s.get(url, timeout=_ml._request_timeout(deadline))
        except RequestException as exc:
            last = _ml.ReadInterrupted(f"BRAIN read interrupted ({type(exc).__name__})")
            if _ml._transient_read_error(exc) and _ml._wait_for_read_retry(attempt, attempts, deadline):
                continue
            raise last from exc
        status = response.status_code
        if status != 429 and status not in _ml._TRANSIENT_READ_STATUSES:
            return response
        delay = _ml._retry_after_seconds(response, cap_error_wait=False)
        if status in (429, 503) and math.isfinite(delay) and delay > 0:
            return response
        last = _ml.ReadInterrupted(f"BRAIN read interrupted (HTTP {status})")
        if not _ml._wait_for_read_retry(attempt, attempts, deadline, delay):
            raise last


def _poll_json_when_ready(s, url, on_event, deadline, *, endpoint, status_errors):
    """Return the JSON payload or the delay for the caller to checkpoint."""
    while deadline is None or _ml.time.monotonic() < deadline:
        response = _ml._get_poll_response(s, url, deadline)
        error = status_errors.get(response.status_code)
        if error is not None:
            raise error

        delay = _ml._retry_after_seconds(response, cap_error_wait=False)
        if not math.isfinite(delay):
            raise _ml.ReadInterrupted("%s returned an invalid Retry-After" % endpoint)
        if response.status_code not in (429, 503) or not delay:
            response.raise_for_status()
        if not delay:
            try:
                return response.json(), None
            except ValueError as exc:
                raise _ml.SimulationError("%s endpoint returned non-JSON: %s"
                                      % (endpoint, response.text[:300])) from exc
        if deadline is not None and delay >= deadline - _ml.time.monotonic():
            return None, delay
        if on_event:
            on_event("polling", {"retry_after": delay})
        _ml.sleep(delay)
    return None, 0


def request_json(s, method, path, params=None, json_body=None, max_retries=4, *, deadline=None):
    """Decode a BRAIN API response or raise a typed HTTP failure.

    Responses with status 429 use backoff within the attempt and time budgets.
    Safe reads also retry interrupted transfers and transient gateway responses.
    Other failures raise AuthError, PermissionDenied, NotFound or ServiceError.
    Successful responses without decodable JSON yield an empty dict.
    """
    from requests.exceptions import RequestException

    url = path if path.startswith("http") else _ml.BRAIN_API_URL + path
    read_only = method.upper() in {"GET", "HEAD"}
    if deadline is None:
        deadline = _ml.time.monotonic() + _ml._request_timeout(None)
    last = None
    for attempt in range(1, max_retries + 1):
        if _ml.time.monotonic() >= deadline:
            raise last or _ml.ServiceError("request wait budget expired for %s %s" % (method, url))
        try:
            response = s.request(method, url, params=params, json=json_body,
                                 timeout=_ml._request_timeout(deadline))
        except RequestException as exc:
            last = _ml.ServiceError("BRAIN request failed (%s) for %s %s"
                                % (type(exc).__name__, method, url))
            if (read_only and _ml._transient_read_error(exc)
                    and _ml._wait_for_read_retry(attempt, max_retries, deadline)):
                continue
            raise last from exc
        if response.status_code == 401:
            raise _ml.AuthError("BRAIN rejected the session (HTTP 401) for %s %s" % (method, url))
        if response.status_code == 403:
            raise _ml.PermissionDenied("BRAIN denied access (HTTP 403) for %s %s" % (method, url))
        if response.status_code == 404:
            raise _ml.NotFound("%s %s -> 404 (unknown id or endpoint)" % (method, url))
        if response.status_code == 429:
            wait = _ml._retry_after_seconds(response)
            if not math.isfinite(wait) or wait <= 0:
                wait = min(60.0, 5.0 * attempt)
            retry_after = _ml._retry_after_seconds(response, cap_error_wait=False)
            last = _ml.RateLimited("%s %s is rate limited" % (method, url),
                               retry_after=retry_after if math.isfinite(retry_after) and retry_after > 0 else wait)
            if attempt < max_retries and wait < deadline - _ml.time.monotonic():
                _ml.sleep(wait)
                continue
            raise last
        if not response.ok:
            last = _ml.ServiceError(
                "HTTP %s from %s %s: %s"
                % (response.status_code, method, url, response.text[:300])
            )
            if (read_only and response.status_code in _ml._TRANSIENT_READ_STATUSES
                    and _ml._wait_for_read_retry(attempt, max_retries, deadline,
                                             _ml._retry_after_seconds(response, cap_error_wait=False))):
                continue
            raise last
        try:
            return response.json()
        except ValueError:
            return {}
    raise last or _ml.RateLimited("%s %s" % (method, url))
