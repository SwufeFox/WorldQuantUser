"""Single/multi simulation submission, settings normalization, slot recovery and execution."""

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


def read_simulation_snapshot(s, simulation_id, *, deadline):
    """Read a snapshot, preserving terminal failure evidence in HTTP 400 replies."""
    if not _ml.valid_simulation_id(simulation_id):
        raise ValueError("invalid simulation id")
    url = f"{_ml.BRAIN_API_URL}/simulations/{simulation_id}"
    return _ml._read_simulation_snapshot(s, url, simulation_id, deadline=deadline)


def _read_simulation_snapshot(s, url, simulation_id, *, deadline):
    from requests.exceptions import RequestException

    if deadline is not None and _ml.time.monotonic() >= deadline:
        raise _ml.ServiceError("simulation inspection wait budget expired")
    try:
        response = s.get(url, timeout=_ml._request_timeout(deadline), allow_redirects=False)
    except RequestException as exc:
        raise _ml.ServiceError("simulation inspection request failed") from exc
    status = response.status_code
    if status in (401, 403):
        raise _ml.AuthError("BRAIN denied access to this simulation")
    if status == 404:
        raise _ml.NotFound("simulation not found")
    if status not in (200, 400, 429):
        raise _ml.ServiceError(f"simulation inspection returned HTTP {status}")
    seconds = _ml._retry_after_seconds(response, cap_error_wait=False)
    if not math.isfinite(seconds):
        raise _ml.ServiceError("simulation inspection returned an invalid Retry-After")
    retry_after = seconds if seconds > 0 else None
    if status == 429:
        wait = f", retry after {retry_after:g} seconds" if retry_after is not None else ""
        raise _ml.RateLimited(f"simulation inspection rate limited{wait}", retry_after=retry_after)
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if status == 400:
        from simulation_slots import TERMINAL_FAILURE_STATUSES

        task = payload if isinstance(payload, dict) else {}
        echoed_id, children = task.get("id"), task.get("children")
        failed = str(task.get("status")).upper() in TERMINAL_FAILURE_STATUSES
        identified = echoed_id == simulation_id or (
            echoed_id is None and isinstance(children, list) and bool(children))
        if not (failed and identified):
            raise _ml.ServiceError("simulation inspection returned HTTP 400")
    if not isinstance(payload, dict):
        if retry_after is None:
            raise _ml.ServiceError("simulation inspection returned an invalid response")
        payload = {}
    return payload, retry_after


def cancel_simulation(s, reference, *, deadline=None):
    """Cancel a running BRAIN simulation via ``DELETE /simulations/{id}`` and release its local slot."""
    from requests.exceptions import RequestException
    import simulation_capabilities
    import simulation_slots

    simulation_id = _ml.simulation_reference_id(reference)
    url = f"{_ml.BRAIN_API_URL}/simulations/{simulation_id}"
    if deadline is not None and _ml.time.monotonic() >= deadline:
        raise _ml.ServiceError("simulation cancellation wait budget expired")
    try:
        response = s.delete(url, timeout=_ml._request_timeout(deadline), allow_redirects=False)
    except RequestException as exc:
        raise _ml.ServiceError("simulation cancellation request failed") from exc
    status = response.status_code
    if status in (401, 403):
        raise _ml.AuthError("BRAIN denied simulation cancellation (HTTP %s)" % status)
    if status in (200, 204):
        state = "cancelled"
    elif status == 202:
        state = "cancellation_pending"
    elif status == 404:
        state = "already_gone"
    elif status == 429:
        retry_after = _ml._retry_after_seconds(response, cap_error_wait=False)
        raise _ml.RateLimited(
            "simulation cancellation rate limited",
            retry_after=retry_after if math.isfinite(retry_after) and retry_after > 0 else None,
        )
    else:
        raise _ml.ServiceError("simulation cancellation returned HTTP %s" % status)

    released_slots = []
    if state != "cancellation_pending":
        released_slots = simulation_slots.cancel_reservation_by_url(
            simulation_capabilities.account_provenance(s), url, state=state, status_code=status
        )
    return {
        "simulation_id": simulation_id,
        "url": url,
        "state": state,
        "cancelled": state == "cancelled",
        "http_status": status,
        "released_reservations": released_slots,
    }


def simulation_progress_receipt(response, body, simulation_id):
    """Return a terminal snapshot that belongs to the requested simulation."""
    import simulation_slots

    if not isinstance(body, dict) or response.status_code not in (200, 400):
        return None
    echoed_id = body.get("id")
    status = str(body.get("status", "")).upper()
    failed = status in simulation_slots.TERMINAL_FAILURE_STATUSES
    if echoed_id not in (None, simulation_id) or status not in simulation_slots.TERMINAL_STATUSES:
        return None
    if response.status_code == 400 and (echoed_id != simulation_id or not failed):
        return None
    alpha_id = body.get("alpha") or body.get("alphaId")
    if (alpha_id is not None and not _ml.valid_simulation_id(alpha_id)) or (not failed and not alpha_id):
        return None
    return {"status": status, "alpha_id": alpha_id}


def recover_accepted_slots(s, account, *, deadline=None):
    """Read ordinary/native acceptances before declaring the shared slots full.

    The sweep has one short budget, never posts, and never turns age or a
    missing response into proof of completion. Native reservations point to
    their parent, so child metrics cannot release or multiply parent slots.
    """
    import simulation_capabilities
    import simulation_slots

    fingerprint = account.get("fingerprint")
    if not fingerprint or simulation_capabilities.account_provenance(s).get("fingerprint") != fingerprint:
        return {"state": "account_unverified", "released": [], "kept": [], "checked": 0}
    end = _ml.time.monotonic() + _ml.QUICK_ORPHAN_SWEEP_SECONDS
    if deadline is not None:
        end = min(end, deadline)
    remaining = max(1, simulation_slots.confirmed_accepted_count(account) or 0)
    observed = {}

    def poll(url):
        nonlocal remaining
        try:
            simulation_id = _ml.simulation_reference_id(url)
        except ValueError:
            return None
        if simulation_id in observed:
            return observed[simulation_id]
        now = _ml.time.monotonic()
        request_deadline = now + max(0, end - now) / remaining
        remaining = max(1, remaining - 1)
        try:
            payload, delay = _ml.read_simulation_snapshot(s, simulation_id, deadline=request_deadline)
        except _ml.AuthError:
            raise
        except _ml.NotFound:
            answer = {"status": "NOT_FOUND", "http_status": 404}
        except Exception:
            answer = None
        else:
            status = str(payload.get("status") or "").upper()
            alpha = payload.get("alpha") or payload.get("alphaId")
            if (payload.get("id") not in (None, simulation_id)
                    or (alpha is not None and not _ml.valid_simulation_id(alpha))):
                answer = None
            elif delay:
                answer = "PENDING"
            else:
                answer = {"status": status, "alpha": alpha, "progress": payload.get("progress"),
                          "children": payload.get("children")}
        observed[simulation_id] = answer
        return answer

    def cancel(url):
        from requests.exceptions import RequestException
        try:
            response = s.delete(url, timeout=_ml._request_timeout(end), allow_redirects=False)
        except RequestException:
            return None
        if response.status_code in (401, 403):
            raise _ml.AuthError("BRAIN denied simulation cancellation")
        return response.status_code

    try:
        report = simulation_slots.reconcile_orphans(
            account, poll, deadline=end, ordinary_only=True, stuck_after_min=45.0,
            cancel=cancel)
    except _ml.cache_store.LockTimeout:
        report = {"state": "journal_busy", "released": [], "kept": []}
    report.setdefault("state", "observed")
    report["checked"] = len(observed)
    return report


def recover_quick_orphans(s, account, *, deadline=None, exclude_job=None, limit=None, offset=0):
    """Reclaim orphaned Quick slots within a short, shared recovery budget."""
    import simulation_slots
    from requests.exceptions import RequestException

    if not account.get("fingerprint"):
        return 0
    short_deadline = _ml.time.monotonic() + _ml.QUICK_ORPHAN_SWEEP_SECONDS
    deadline = min(deadline, short_deadline) if deadline is not None else short_deadline
    if _ml.time.monotonic() >= deadline:
        return 0
    jobs = simulation_slots.quick_orphans(account, exclude_job=exclude_job)
    if not jobs:
        return 0
    start = offset % len(jobs)
    jobs = (jobs[start:] + jobs[:start])[:limit]
    released = 0
    for position, job in enumerate(jobs):
        now = _ml.time.monotonic()
        if now >= deadline:
            break
        try:
            url, simulation_id = _ml._quick_progress_target(job["progress_url"])
            # Leave time to inspect later reservations if this endpoint times out.
            request_deadline = now + (deadline - now) / (len(jobs) - position)
            response = s.get(url, timeout=_ml._request_timeout(request_deadline), allow_redirects=False)
        except (_ml.SimulationError, RequestException):
            continue
        if response.status_code in (401, 403):
            raise _ml.AuthError("session rejected while recovering a Quick simulation")
        delay = _ml._retry_after_seconds(response, cap_error_wait=False)
        if math.isfinite(delay) and delay > 0:
            try:
                simulation_slots.quick_defer(account, job["id"], _ml.time.time() + delay, deadline=deadline)
            except _ml.cache_store.LockTimeout:
                break
            continue
        if response.status_code not in (200, 400):
            continue
        try:
            payload = response.json()
        except ValueError:
            continue
        receipt = _ml.simulation_progress_receipt(response, payload, simulation_id)
        if not receipt:
            continue
        try:
            released += simulation_slots.quick_update(account, job["id"], "terminal", retain=True,
                                                      terminal_result=receipt, deadline=deadline)
        except _ml.cache_store.LockTimeout:
            break
    return released


def attach_quick_location(s, account, job_key, location, *, deadline=None, confirm_failure=False):
    """Attach the original, independently recovered Location of an ambiguous Quick POST."""
    import simulation_capabilities
    import simulation_slots

    fingerprint = account.get("fingerprint")
    if not fingerprint or simulation_capabilities.account_provenance(s).get("fingerprint") != fingerprint:
        raise ValueError("Quick Location recovery requires its authenticated account")
    if not isinstance(job_key, str) or not job_key:
        raise ValueError("Quick Location recovery requires a job key")
    url, simulation_id = _ml._quick_progress_target(location)
    payload, delay = _ml._read_simulation_snapshot(s, url, simulation_id, deadline=deadline)
    status = str(payload.get("status") or "").upper()
    progress = payload.get("progress")
    if (payload.get("id") not in (None, simulation_id)
            or not (payload.get("id") == simulation_id or status
                    or type(progress) in (int, float) and 0 <= progress <= 1)):
        raise _ml.ServiceError("BRAIN did not confirm the recovered Quick simulation identity")
    if confirm_failure:
        if (delay or payload.get("id") != simulation_id
                or status not in simulation_slots.TERMINAL_FAILURE_STATUSES
                or payload.get("alpha") or payload.get("alphaId")):
            raise _ml.ServiceError("BRAIN did not confirm a terminal failure without an Alpha for this Location")
        return simulation_slots.quick_attach_location(account, job_key, url, terminal_failure=status)
    next_poll_at = _ml.time.time() + delay if delay else None
    return simulation_slots.quick_attach_location(account, job_key, url, next_poll_at=next_poll_at)


def poll_simulation(s, progress_url, on_event=None, deadline=None):
    """Poll a simulation until completion or an optional monotonic deadline.

    If BRAIN's ``Retry-After`` would run past ``deadline``, return a resumable
    ``PENDING`` payload instead of sleeping for hours. The progress URL is kept
    so callers can checkpoint it and continue in a later invocation.
    """
    progress_url = _ml._validated_progress_url(progress_url)
    payload, delay = _ml._poll_json_when_ready(
        s, progress_url, on_event, deadline, endpoint="progress",
        status_errors={status: _ml.AuthError("session rejected while polling (HTTP %s)" % status)
                       for status in (401, 403)},
    )
    if delay is None:
        return payload
    return {"status": "PENDING", "progress_url": progress_url, "next_poll_after_seconds": delay}


def submit_single_simulation(
    s, simulation_data, max_retries=5, on_event=None, deadline=None
):
    """POST one single-alpha simulation and return its progress URL.

    Daily-quota headers are retained separately from transient HTTP throttling.
    Confirmed exhaustion defers new work; accepted jobs remain recoverable.
    """
    return _ml._submit_simulation_request(s, simulation_data, max_retries, on_event, deadline)


def submit_multi_simulation(s, simulations, max_retries=5, on_event=None, deadline=None):
    """POST an array of REGULAR simulation requests and return the parent URL."""
    if not isinstance(simulations, list) or not 2 <= len(simulations) <= 10:
        raise ValueError("native multi-simulation needs 2–10 requests")
    if any(not isinstance(request, dict) or request.get("type") != "REGULAR"
           or not isinstance(request.get("settings"), dict) for request in simulations):
        raise ValueError("native multi-simulation accepts REGULAR request objects")
    first = simulations[0]["settings"]
    common = ("instrumentType", "region", "delay", "language")
    if any(any(request["settings"].get(key) != first.get(key) for key in common)
           for request in simulations):
        raise ValueError("native multi-simulation needs common instrumentType, region, delay and language")
    if _ml._has_multi_simulation_permission(s) is False:
        raise _ml.PermissionDenied("native multi-simulation requires MULTI_SIMULATION permission")
    parent_url = _ml._submit_simulation_request(s, simulations, max_retries, on_event, deadline)
    try:
        _ml.simulation_reference_id(parent_url)
    except ValueError as exc:
        raise _ml.SimulationError("native multi-simulation returned an invalid parent Location") from exc
    return parent_url


def _has_multi_simulation_permission(session):
    permissions = getattr(session, "_brain_permissions", None)
    if isinstance(permissions, (list, tuple, set, frozenset)):
        return "MULTI_SIMULATION" in permissions
    return None


def _submit_simulation_request(s, simulation_data, max_retries, on_event, deadline):
    import simulation_capabilities
    import simulation_slots

    native = isinstance(simulation_data, list)
    body = ([_ml.format_simulation_payload(item) for item in simulation_data]
            if native else _ml.format_simulation_payload(simulation_data))
    redirect_limits = {"allow_redirects": False} if native else {}
    account = simulation_capabilities.account_provenance(s)
    for attempt in range(1, max_retries + 1):
        if deadline is not None and deadline - _ml.time.monotonic() <= 0:
            raise _ml.RateLimited("submission deferred by the run time limit")
        quota = simulation_slots.quota_snapshot(account)
        if quota["blocked"]:
            if on_event:
                on_event("quota", quota)
            raise _ml.RateLimited("daily simulation quota exhausted; resume after the reported reset")
        if simulation_slots.submission_retry_at(account) is not None:
            raise _ml.RateLimited("simulation submission deferred by shared Retry-After cooldown")
        response = s.post(
            _ml.BRAIN_API_URL + "/simulations", json=body,
            timeout=_ml._request_timeout(deadline), **redirect_limits,
        )
        observation = (simulation_slots.observe_quota(account, response.headers)
                       if response.status_code not in (401, 403) else None)
        if observation is not None and on_event:
            on_event("quota", observation)
        if (response.status_code == 429 and observation is not None
                and simulation_slots.quota_snapshot(account, observation)["blocked"]):
            raise _ml.RateLimited("daily simulation quota exhausted; resume after the reported reset")
        if response.status_code == 429:
            wait = _ml._retry_after_seconds(response, cap_error_wait=False)
            if not math.isfinite(wait) or wait <= 0:
                wait = min(60.0, 5.0 * attempt)
            retry_at = simulation_slots.defer_submission(account, wait, deadline=deadline)
            if retry_at is not None:
                wait = max(wait, retry_at - _ml.time.time())
            if on_event:
                request = ({"child_count": len(simulation_data)} if isinstance(simulation_data, list)
                           else {"expression": simulation_data.get("regular")})
                on_event("rate_limited", {"attempt": attempt, "wait_seconds": wait, **request})
            if attempt < max_retries:
                if deadline is not None:
                    remaining = max(0.0, deadline - _ml.time.monotonic())
                    if wait >= remaining:
                        raise _ml.RateLimited(
                            "submission deferred until after the run time limit"
                        )
                _ml.sleep(wait)
                continue
            raise _ml.RateLimited(
                "submission still rate limited after %d attempts" % max_retries
            )
        if response.status_code in (401, 403):
            raise _ml.AuthError(
                "submission rejected (HTTP %s)" % response.status_code
            )
        if native and 300 <= response.status_code < 400:
            raise _ml.SimulationError("native multi-simulation returned an unexpected redirect")
        response.raise_for_status()
        location = response.headers.get("Location")
        if not location:
            raise _ml.SimulationError(
                "submit response has no Location header (HTTP %s): %s"
                % (response.status_code, response.text[:500])
            )
        return _ml._validated_progress_url(location)


def fetch_alpha_metrics(s, alpha_id, on_event=None, deadline=None):
    """Read one alpha's settings, in-sample metrics and platform checks.

    Returns the raw ``GET /alphas/{id}`` payload (``regular``, ``settings``,
    ``is``, ``checks``, ...). Quota-free: it never consumes a simulation slot.
    Raises ``NotFound`` for an unknown id. Returns a private ``_deferred``
    sentinel when a Retry-After would exceed ``deadline``.
    """
    payload, delay = _ml._poll_json_when_ready(
        s, _ml.BRAIN_API_URL + "/alphas/" + alpha_id, on_event, deadline, endpoint="alpha",
        status_errors={404: _ml.NotFound("alpha %s not found" % alpha_id)},
    )
    if delay is None:
        return payload
    return {"_deferred": True, "next_poll_after_seconds": delay}


def is_quick_simulation(alpha):
    """Identify a Quick simulation from its recorded settings."""
    settings = alpha.get("settings")
    mode = settings.get("simulationMode") if isinstance(settings, dict) else None
    return isinstance(mode, str) and mode.strip().upper() == "QUICK"


def alpha_classification_ids(alpha):
    """Return native classification IDs, preserving missing versus empty evidence."""
    classifications = alpha.get("classifications")
    if not isinstance(classifications, list):
        return None
    identifiers = [item.get("id") if isinstance(item, dict) else item for item in classifications]
    if not all(isinstance(identifier, str) and identifier for identifier in identifiers):
        return None
    return list(dict.fromkeys(identifiers))


def defer_alpha_metrics(record, payload):
    """Keep a simulation resumable when the metrics response asks us to wait."""
    if not payload.get("_deferred"):
        return False
    wait = max(0.0, float(payload.get("next_poll_after_seconds") or 0))
    record.update(status="PENDING", metrics_pending=True, poll_after_seconds=wait)
    if wait:
        record["next_poll_at"] = _ml.time.time() + wait
    else:
        record.pop("next_poll_at", None)
    return True


def normalize_setting_keys(overrides=None):
    """Canonicalize known key spellings within one input layer, without defaults.

    This must run before merging files, arms or CLI flags: spelling must not
    decide which layer wins. Unknown platform extensions remain untouched.
    """
    canonical_keys = {key.replace("_", "").casefold(): key for key in (*_ml.DEFAULT_SETTINGS, "simulationMode")}
    settings, origins = {}, {}
    for key, value in (overrides or {}).items():
        canonical = canonical_keys.get(str(key).replace("_", "").casefold(), key)
        if canonical == "simulationMode" and isinstance(value, str):
            mode = value.strip().upper()
            if mode in {"QUICK", "FULL"}:
                value = mode
        if canonical in settings:
            previous = settings[canonical]
            if value is None and previous is not None:
                continue
            if previous is not None and value is not None and previous != value:
                raise ValueError("conflicting settings %r and %r for %r" %
                                 (origins[canonical], key, canonical))
        settings[canonical], origins[canonical] = value, key
    return settings


def resolve_settings(overrides=None):
    """Merge defaults and validate structure, leaving platform support to evidence.

    Observed enum lists are hints, not permission gates. Explicit null extension
    values are retained; null known settings retain their historical default.
    """
    settings = dict(_ml.DEFAULT_SETTINGS)
    for key, value in _ml.normalize_setting_keys(overrides).items():
        if value is not None or key not in _ml.DEFAULT_SETTINGS:
            settings[key] = value
    if _ml.is_python_language(settings.get("language")):
        settings["language"] = "PYTHON"
    for key in ("region", "instrumentType", "universe", "neutralization",
                "pasteurization", "nanHandling"):
        value = settings.get(key)
        if not isinstance(value, str) or not _ml._SETTING_TOKEN.fullmatch(value):
            raise ValueError("%s must be a well-formed settings token, got %r" % (key, value))
    delay = settings["delay"]
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
        raise ValueError("delay must be a non-negative integer, got %r" % (delay,))
    return settings


#: Settings the OPTIONS schema advertises as required but the POST contract
#: rejects for ``type=REGULAR`` ("Selection handling ... is not allowed for
#: simulation type REGULAR"). They belong to a selection simulation type this
#: REGULAR path never sends, even if legacy specs include them. Preflight and
#: dispatch check the actual REGULAR settings against the account schema.
_REGULAR_UNSENT_SETTINGS = ("selectionHandling", "selectionLimit")


#: The native schema advertises these FASTEXPR controls, but the Python
#: REGULAR endpoint rejects them. Keep the original settings in the spec and
#: checkpoint; use the wire subset for dispatch and Python identity.
_PYTHON_UNSENT_SETTINGS = ("unitHandling", "nanHandling")


def _regular_unsent_settings(settings):
    if _ml.is_python_language(settings.get("language")):
        return _ml._REGULAR_UNSENT_SETTINGS + _ml._PYTHON_UNSENT_SETTINGS
    return _ml._REGULAR_UNSENT_SETTINGS


def _omit_regular_settings(mapping, excluded):
    spellings = {name.casefold() for name in excluded}
    return {key: value for key, value in mapping.items()
            if str(key).replace("_", "").casefold() not in spellings}


def regular_wire_settings(settings):
    """Return the writable settings present in a REGULAR simulation request."""
    return _ml._omit_regular_settings(settings, _ml._regular_unsent_settings(settings))


def format_simulation_payload(payload):
    """Apply safe default FASTEXPR formatting to a native REGULAR request."""
    if (not isinstance(payload, dict) or payload.get("type") != "REGULAR"
            or not isinstance(payload.get("regular"), str)
            or (_ml.alpha_language(payload) or "FASTEXPR") != "FASTEXPR"):
        return payload
    from expression_lint import format_fastexpr

    source, _reason = format_fastexpr(payload["regular"])
    return {**payload, "regular": source} if source != payload["regular"] else payload


def _single_simulation_data(expression, decay, settings, *, format_source=True):
    """Build one ``POST /simulations`` payload from a resolved settings dict."""
    body = _ml.regular_wire_settings(settings)
    body["decay"] = decay
    payload = {"type": "REGULAR", "settings": body, "regular": expression}
    return _ml.format_simulation_payload(payload) if format_source else payload


def regular_settings_evidence(settings, snapshot):
    """Check the settings that the REGULAR request will actually send."""
    from simulation_capabilities import settings_evidence

    wire = _ml.regular_wire_settings(settings)
    schema = snapshot.get("settings_schema")
    if isinstance(schema, dict):
        snapshot = {**snapshot, "settings_schema":
                    _ml._omit_regular_settings(schema, _ml._regular_unsent_settings(settings))}
    report = settings_evidence(wire, snapshot)
    report["omitted_from_payload"] = [key for key in settings if key not in wire]
    return report


#: Keys a per-alpha dict in a research file may carry. Anything else is an
#: error rather than a silent drop: a config that says
#: ``{"expression": ..., "neutralization": "SUBINDUSTRY"}`` looks like it
#: overrides the setting, but neutralisation is a settings-level field, so the
#: value would vanish and the round would quietly test nothing. Raising on the
#: unknown key turns a wasted experiment into a message.
_ALPHA_ENTRY_KEYS = frozenset({
    "expression", "decay", "mechanism", "label", "name",
    "family", "candidate_id", "idea", "hypothesis", "note", "role",
})


def normalize_alphas(alphas, settings):
    """Normalize a research-file ``alphas`` entry list into ``(expr, decay)`` pairs.

    Accepts plain strings, ``{"expression": ..., "decay": ...}`` dicts, or
    ``(expression, decay)`` tuples. The default decay comes from ``settings``.
    Unknown keys in a dict entry raise, because silently ignoring them lets a
    config appear to control something it does not.
    """
    default_decay = settings.get("decay", _ml.DEFAULT_SETTINGS["decay"])
    normalized = []
    for index, item in enumerate(alphas):
        if isinstance(item, str):
            expression, decay = item, default_decay
        elif isinstance(item, dict):
            unknown = set(item) - _ml._ALPHA_ENTRY_KEYS
            if unknown:
                raise ValueError(
                    "alphas[%d] has unsupported key(s) %s; only %s are read"
                    % (index, ", ".join(sorted(unknown)),
                       ", ".join(sorted(_ml._ALPHA_ENTRY_KEYS))))
            expression = item.get("expression")
            decay = item.get("decay", default_decay)
            if "role" in item and (not isinstance(item["role"], str)
                                   or item["role"] not in {"candidate", "diagnostic"}):
                raise ValueError("alphas[%d] role must be candidate or diagnostic" % index)
        elif isinstance(item, (list, tuple)) and item:
            expression, decay = item[0], (item[1] if len(item) > 1 else default_decay)
        else:
            raise ValueError("alphas[%d] is not a usable expression entry" % index)
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("alphas[%d] has an empty or non-string expression" % index)
        if not isinstance(decay, int) or isinstance(decay, bool) or decay < 0:
            raise ValueError("alphas[%d] has an invalid decay %r" % (index, decay))
        if _ml.is_python_language(settings.get("language")):
            expression = _ml.normalize_python_source(expression)
        else:
            expression = expression.strip()
        normalized.append((expression, decay))
    return normalized


def alpha_candidate_ids(alphas, count):
    """Per-expression ``candidate_id`` declarations, aligned with pairs.

    Same contract as :func:`alpha_families`: when the list does not line up with
    the normalized pairs it is dropped wholesale. A misaligned label is worse
    than a missing one, because it silently attaches an ablation variant's
    numbers to the wrong leg.
    """
    if not alphas or not isinstance(alphas[0], dict):
        return [None] * count
    ids = [item.get("candidate_id") for item in alphas if isinstance(item, dict)]
    if len(ids) != count:
        return [None] * count
    return ids


def alpha_families(alphas, count):
    """Per-expression ``family`` declarations from a config, aligned with pairs.

    Same contract as :func:`alpha_labels`: a config entry may carry ``family``,
    and the list is dropped wholesale when it does not line up with the
    normalized pairs. A wrong family silently corrupts the diversity count that
    ``alpha book`` reports, so dropping is safer than guessing.
    """
    if not alphas or not isinstance(alphas[0], dict):
        return [None] * count
    families = [item.get("family") for item in alphas if isinstance(item, dict)]
    if len(families) != count:
        return [None] * count
    return families


def alpha_labels(alphas, count):
    """Per-expression labels from a config, aligned with the normalized pairs.

    A config entry may carry ``mechanism`` / ``label`` / ``name``. When the
    shapes do not line up with the normalized pairs (an entry expanded into
    several expressions, plain strings, tuples) the labels are dropped rather
    than mis-assigned: a wrong label in the ledger is worse than no label.
    """
    if not alphas or not isinstance(alphas[0], dict):
        return [None] * count
    labels = [
        item.get("mechanism") or item.get("label") or item.get("name")
        for item in alphas
        if isinstance(item, dict)
    ]
    if len(labels) != count:
        return [None] * count
    return labels


def concurrency_plan(session, requested=None, capability_snapshot=None):
    """Resolve an invocation budget from supplied/cached account evidence.

    Never discovers capabilities online. Three is a conservative local default,
    not a platform allowance; any explicit positive integer budget is valid.
    Only a verified capacity bound to the current account can clamp the budget.
    """
    import simulation_capabilities

    if requested is not None and (
        isinstance(requested, bool) or not isinstance(requested, int) or requested < 1
    ):
        raise ValueError("concurrency must be a positive integer")
    evidence = capability_snapshot if capability_snapshot is not None else simulation_capabilities.cached()
    account = simulation_capabilities.account_provenance(session) if session is not None else evidence.get("account", {})
    capacity = evidence.get("concurrency") or {}
    evidence_account = evidence.get("account") or {}
    matched = bool(account.get("fingerprint")) and account.get("fingerprint") == evidence_account.get("fingerprint")
    limit = capacity.get("limit")
    verified = (evidence.get("state") == "verified" and capacity.get("state") == "verified"
                and matched and isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0)
    budget = requested if requested is not None else (limit if verified else _ml.DEFAULT_LOCAL_CONCURRENCY)
    return {
        "requested": requested,
        "effective": min(budget, limit) if verified else budget,
        "platform_limit": limit if verified else None,
        "source": ("explicit_local_budget" if requested is not None else
                   "verified_platform_limit" if verified else "conservative_local_default"),
        "platform_source": capacity.get("source") if verified else None,
        "account": account,
        "capability_id": evidence.get("capability_id"),
    }


def simulate_alphas(
    s,
    alphas,
    settings=None,
    *,
    dry_run=False,
    on_event=None,
    max_wait_seconds=None,
    deadline=None,
    initial_records=None,
    labels=None,
    families=None,
    candidates=None,
    concurrency=None,
    capability_snapshot=None,
    scheduling_report=None,
    fetch_metrics=True,
    multi_size=None,
    recover_only=False,
    native_poll_once=False,
):
    """Simulate a list of alphas and return one record per expression.

    Each record is ``{"expression", "decay", "alpha_id", "status", "metrics",
    "checks", "error"}``. ``metrics`` is the in-sample block from
    ``GET /alphas/{id}`` for completed runs; ``checks`` maps platform check
    names to ``{"result", "limit", "value"}``. A failed expression is recorded
    with a populated ``error`` instead of aborting the run. When
    ``max_wait_seconds`` or an absolute monotonic ``deadline`` is set, unfinished work is returned as ``PENDING`` (or
    ``QUEUED`` if it has not yet been submitted) instead of keeping the caller
    blocked. Pass those records back as ``initial_records`` to resume without
    resubmitting accepted jobs.

    ``dry_run`` validates and builds the payloads but never touches the network
    (and requires no credentials). ``on_event(kind, payload)`` receives
    ``submitted`` / ``polling`` / ``quota`` / ``rate_limited`` / ``completed`` events.
    Feeding it the result of ``resolve_settings`` is the caller's job.
    ``concurrency`` is a positive local budget, clamped only by verified account
    capacity. A terminal progress response releases its slot independently
    of the Alpha metrics read. With ``fetch_metrics=False``, keep Alpha IDs in
    the checkpoint for a later metrics-only read of those records.
    ``multi_size`` sends native batches of compatible arms with a maximum size
    between two and ten. ``recover_only`` reads existing work without creating it.
    ``native_poll_once`` checks each due native parent once and saves its next poll time.
    Reservations survive interruptions and are
    reconciled by ``simulation_slots.reconcile_record``; other workspaces and
    unidentified accounts cannot share these reservations.
    """
    if native_poll_once and dry_run:
        raise ValueError("native_poll_once requires live simulation or recovery")
    settings = _ml.resolve_settings(settings)
    pairs = _ml.normalize_alphas(alphas, settings)
    roles = [item.get("role") if isinstance(item, dict) else None for item in alphas]
    if labels is None:
        labels = _ml.alpha_labels(alphas, len(pairs))
    elif len(labels) != len(pairs):
        labels = [None] * len(pairs)
    if families is None:
        families = _ml.alpha_families(alphas, len(pairs))
    elif len(families) != len(pairs):
        families = [None] * len(pairs)
    if candidates is None:
        candidates = _ml.alpha_candidate_ids(alphas, len(pairs))
    elif len(candidates) != len(pairs):
        candidates = [None] * len(pairs)

    import simulation_capabilities
    import simulation_slots

    native_enabled = multi_size is not None
    if native_enabled or (initial_records and any(row.get("multi_request_id") for row in initial_records)):
        import simulation_multi
    if native_enabled and (type(multi_size) is not int or not 2 <= multi_size <= simulation_multi.MAX_CHILDREN):
        raise ValueError("native multi-simulation size must be an integer between 2 and 10")

    def original_payload_for(index):
        expression, decay = pairs[index]
        return _ml._single_simulation_data(expression, decay, settings, format_source=False)

    @cache
    def payload_for(index):
        return _ml.format_simulation_payload(original_payload_for(index))

    if capability_snapshot is None and not dry_run:
        capability_snapshot = simulation_capabilities.cached()
    plan = _ml.concurrency_plan(None if dry_run else s, concurrency,
                            (capability_snapshot or {}) if dry_run else capability_snapshot)
    if settings.get("simulationMode") == "QUICK":
        plan["effective"] = simulation_slots.quick_slot_limit(plan["effective"])
    if native_enabled:
        plan["multi_parent_limit"] = simulation_multi.CONSULTANT_PARENT_LIMIT
        plan["effective"] = min(plan["effective"], plan["multi_parent_limit"])
    batch_size = max(1, plan["effective"])
    if scheduling_report is not None:
        scheduling_report.update(plan)
        scheduling_report["shared_slots"] = {
            "state": "dry_run", "scope": "not_reserved", "cross_workspace_coordinated": False,
        }
        scheduling_report["quota"] = simulation_slots.quota_snapshot({} if dry_run else plan["account"])
    if max_wait_seconds is not None and (
        not math.isfinite(max_wait_seconds) or max_wait_seconds < 0
    ):
        raise ValueError("max_wait_seconds must be a finite, non-negative number")

    if dry_run:
        if native_enabled and scheduling_report is not None:
            requests = simulation_multi.request_groups(range(len(pairs)), payload_for, multi_size)
            scheduling_report["multi_requests"] = [
                {"indices": list(group), "payload": [payload_for(index) for index in group]
                 if len(group) > 1 else payload_for(group[0])}
                for group in requests
            ]
        return [
            {
                "expression": expression,
                "decay": decay,
                "candidate_id": candidates[index],
                "mechanism": labels[index],
                "family": families[index],
                **({"role": roles[index]} if roles[index] is not None else {}),
                "alpha_id": None,
                "status": "DRY_RUN",
                "metrics": None,
                "checks": None,
                "error": None,
                "payload": payload_for(index),
            }
            for index, (expression, decay) in enumerate(pairs)
        ]

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from requests.exceptions import HTTPError, RequestException

    if initial_records is None:
        records = [
            {
                "expression": expression,
                "decay": decay,
                "candidate_id": candidates[index],
                "mechanism": labels[index],
                "family": families[index],
                **({"role": roles[index]} if roles[index] is not None else {}),
                "alpha_id": None,
                "status": "QUEUED",
                "submission_attempted": False,
                "metrics": None,
                "checks": None,
                "error": None,
            }
            for index, (expression, decay) in enumerate(pairs)
        ]
    else:
        records = [dict(record) for record in initial_records]
        if len(records) != len(pairs):
            raise ValueError("initial_records must match the number of alphas")
        for index, record in enumerate(records):
            if "role" not in record and roles[index] is not None:
                record["role"] = roles[index]


    def shared_state():
        state = simulation_slots.snapshot(plan["account"], plan["effective"])
        if scheduling_report is not None:
            scheduling_report["shared_slots"] = state
            scheduling_report["submission_retry_at"] = simulation_slots.submission_retry_at(plan["account"])
        return state

    if max_wait_seconds is not None:
        relative_deadline = _ml.time.monotonic() + max_wait_seconds
        deadline = relative_deadline if deadline is None else min(deadline, relative_deadline)

    def emit(kind, index, payload=None):
        if on_event:
            data = dict(payload or {})
            data["index"] = index
            on_event(kind, data)

    def checkpoint_submission(index, record, identity):
        if identity is not None:
            record.update(slot_reservation=identity, slot_reservation_state="submitting")
        record.update(submission_attempted=True, status="UNKNOWN_SUBMISSION")
        records[index] = record
        emit("submitting", index, record)

    quota_exhausted = False

    def submission_event(kind, index, payload):
        nonlocal quota_exhausted
        if kind == "quota":
            payload = simulation_slots.quota_snapshot(plan["account"], observation=payload)
            quota_exhausted = quota_exhausted or payload["blocked"]
            if scheduling_report is not None:
                scheduling_report["quota"] = payload
        emit(kind, index, payload)

    has_saved_native = any(record.get("multi_request_id") for record in records)
    native = (simulation_multi.MultiRuns(s, records, payload_for, plan["account"],
              plan["effective"], deadline if deadline is not None else float("inf"), emit,
              submission_event, original_payload_for=original_payload_for, poll_once=native_poll_once)
              if native_enabled or has_saved_native else None)
    saved_groups = list(native.restore()) if has_saved_native else []
    for record in records:
        if not record.get("multi_request_id"):
            simulation_slots.reconcile_record(record, plan["account"])
    shared_state()

    def fetch_metrics_record(index):
        record = records[index]
        next_poll_at = record.pop("next_poll_at", None)
        if next_poll_at:
            wait = max(0.0, float(next_poll_at) - _ml.time.time())
            remaining = deadline - _ml.time.monotonic() if deadline is not None else None
            if remaining is not None and wait >= remaining:
                record["status"] = "PENDING"
                record["metrics_pending"] = True
                record["poll_after_seconds"] = wait
                record["next_poll_at"] = next_poll_at
                emit("pending", index, record)
                return
            if wait:
                _ml.sleep(wait)

        def metrics_event(kind, payload):
            emit(kind, index, payload)

        try:
            payload = _ml.fetch_alpha_metrics(
                s,
                record["alpha_id"],
                on_event=metrics_event,
                deadline=deadline,
            )
            if record.get("multi_request_id"):
                if not isinstance(payload, dict) or payload.get("id") not in (None, record["alpha_id"]):
                    raise _ml.ServiceError("multi-simulation Alpha metrics identity mismatch")
                metrics = payload.get("is")
                if not payload.get("_deferred") and (not isinstance(metrics, dict) or not metrics):
                    raise _ml.ReadInterrupted("multi-simulation Alpha metrics are not available yet")
        except (_ml.AuthError, _ml.ReadInterrupted) as exc:
            record["status"] = "PENDING"
            record["metrics_pending"] = True
            record["error"] = str(exc)
            emit("pending", index, record)
            return
        except Exception as exc:
            record["error"] = "metrics fetch failed: %s" % exc
            if record.get("multi_request_id"):
                record.update(status="PENDING", metrics_pending=True)
                emit("pending", index, record)
                return
        else:
            if _ml.defer_alpha_metrics(record, payload):
                emit("pending", index, record)
                return
            record["error"] = record.pop("simulation_error", None)
            record["metrics"] = payload.get("is")
            record["checks"] = _ml._extract_checks(payload)
            classifications = _ml.alpha_classification_ids(payload)
            if classifications is not None:
                record["classification_ids"] = classifications
            split = _ml.extract_split(payload)
            if split:
                record["out_of_sample"] = split

        record["status"] = record.pop("simulation_status", record.get("status"))
        record.pop("metrics_pending", None)
        record.pop("next_poll_at", None)
        record.pop("poll_after_seconds", None)
        emit("completed", index, record)

    def poll_record(index):
        record = records[index]
        progress_url = record.get("progress_url")
        if not progress_url:
            return

        next_poll_at = record.pop("next_poll_at", None)
        if next_poll_at:
            wait = max(0.0, float(next_poll_at) - _ml.time.time())
            remaining = deadline - _ml.time.monotonic() if deadline is not None else None
            if remaining is not None and wait >= remaining:
                record["status"] = "PENDING"
                record["poll_after_seconds"] = wait
                record["next_poll_at"] = next_poll_at
                emit("pending", index, record)
                return
            if wait:
                _ml.sleep(wait)

        def poll_event(kind, payload):
            emit(kind, index, payload)

        try:
            progress = _ml.poll_simulation(
                s, progress_url, on_event=poll_event, deadline=deadline
            )
            if progress.get("status") == "PENDING":
                wait = max(
                    0.0, float(progress.get("next_poll_after_seconds") or 0)
                )
                record["status"] = "PENDING"
                record["progress_url"] = progress_url
                record["poll_after_seconds"] = wait
                if wait:
                    record["next_poll_at"] = _ml.time.time() + wait
                emit("pending", index, record)
                return

            record["status"] = progress.get("status") or "UNKNOWN"
            record["alpha_id"] = progress.get("alpha")
            if not record["alpha_id"] and record["status"] not in simulation_slots.TERMINAL_STATUSES:
                record["status"] = "PENDING"
                record["progress_url"] = progress_url
                emit("pending", index, record)
                return
            simulation_slots.reconcile_record(record, plan["account"], outcome="terminal")
            record["error"] = None
            if progress.get("status") in simulation_slots.TERMINAL_FAILURE_STATUSES or progress.get("message"):
                record["error"] = progress.get("message") or "simulation ended with %s" % record["status"]
                if not _ml.is_python_language(settings.get("language")):
                    _ml.note_operator_rejection(record.get("expression"), record["error"])
            if record["alpha_id"]:
                record["simulation_status"] = record["status"]
                if record["error"]:
                    record["simulation_error"] = record["error"]
                record["status"] = "PENDING"
                record["metrics_pending"] = True
            record.pop("progress_url", None)
            if record.get("metrics_pending"):
                emit("pending", index, record)
            else:
                record.pop("next_poll_at", None)
                record.pop("poll_after_seconds", None)
                emit("completed", index, record)
        except (_ml.AuthError, _ml.ReadInterrupted) as exc:
            record["status"] = "PENDING"
            record["error"] = str(exc)
            record["progress_url"] = progress_url
            emit("pending", index, record)
        except Exception as exc:  # preserve the accepted job for a later resume
            record["status"] = "FAILED"
            record["error"] = "poll failed: %s" % exc
            record["progress_url"] = progress_url
            emit("failed", index, record)

    snapshot_account = (capability_snapshot or {}).get("account") or {}
    settings_snapshot = capability_snapshot if (
        plan["account"].get("fingerprint")
        and plan["account"]["fingerprint"] == snapshot_account.get("fingerprint")
    ) else {}
    settings_checks = {}

    def settings_allowed(index):
        decay = pairs[index][1]
        if decay not in settings_checks:
            settings_checks[decay] = _ml.regular_settings_evidence(
                {**settings, "decay": decay}, settings_snapshot)
        capability_check = settings_checks[decay]
        if not capability_check["unsupported"]:
            return True
        record = records[index]
        record.update(status="FAILED", submission_attempted=False, submission_refused=True,
                      error="settings are unsupported by current account metadata",
                      settings_capabilities=capability_check)
        emit("failed", index, record)
        return False

    def submit_record(index):
        record = dict(records[index])
        payload = payload_for(index)
        identity, slots = simulation_slots.prepare_submission(
            plan["account"], plan["effective"],
            lambda reservation: checkpoint_submission(index, record, reservation),
        )
        if scheduling_report is not None:
            scheduling_report["shared_slots"] = slots
        if identity is None and slots["available"] == 0:
            record["error"] = "shared simulation slots are occupied; reconcile or resume existing attempts"
            records[index] = record
            emit("deferred", index, record)
            return "contended"
        try:
            location = _ml.submit_single_simulation(
                s, payload,
                on_event=lambda kind, payload, i=index: submission_event(kind, i, payload),
                deadline=deadline,
            )
        except _ml.RateLimited as exc:
            simulation_slots.reconcile_record(record, plan["account"], outcome="refused")
            record["status"] = "QUEUED"
            record["submission_attempted"] = False
            record["error"] = "%s: %s" % (exc.code, exc)
            records[index] = record
            emit("deferred", index, record)
            return "blocked"
        except _ml.AuthError as exc:
            simulation_slots.reconcile_record(record, plan["account"], outcome="refused")
            shared_state()
            record.update(status="QUEUED", submission_attempted=False, error=str(exc))
            records[index] = record
            emit("deferred", index, record)
            raise
        except _ml.SimulationError as exc:
            record.update(status="UNKNOWN_SUBMISSION", error=str(exc))
            records[index] = record
            emit("pending", index, record)
            return "blocked"
        except _ml.BrainError as exc:
            record["status"] = "UNKNOWN_SUBMISSION"
            record["error"] = "%s: %s" % (exc.code, exc)
            records[index] = record
            emit("pending", index, record)
            return "blocked"
        except HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            refused = status_code is not None and 400 <= status_code < 500 and status_code != 408
            record["status"] = "FAILED" if refused else "UNKNOWN_SUBMISSION"
            record["error"] = "submit failed: %s" % exc
            if refused:
                detail = getattr(exc.response, "text", "")
                if _ml.is_python_language(settings.get("language")) and isinstance(detail, str) and detail.strip():
                    record["error"] += " (" + detail.strip()[:500] + ")"
                simulation_slots.reconcile_record(record, plan["account"], outcome="refused")
                record.update(status="FAILED", submission_refused=True)
                if not _ml.is_python_language(settings.get("language")):
                    _ml.note_operator_rejection(record.get("expression"), record["error"])
            records[index] = record
            emit("failed" if refused else "pending", index, record)
            return "skipped" if refused else "blocked"
        except RequestException as exc:
            record.update(status="UNKNOWN_SUBMISSION", error="submit transport failed: %s" % type(exc).__name__)
            records[index] = record
            emit("pending", index, record)
            return "blocked"
        record["status"] = "PENDING"
        record["progress_url"] = location
        record["error"] = None
        simulation_slots.reconcile_record(record, plan["account"], outcome="accepted")
        records[index] = record
        emit("submitted", index, record)
        return "accepted"

    def read_result(future):
        try:
            future.result()
        except Exception as exc:  # One read failure should not discard the other jobs.
            _ml.note("simulation read worker raised %s: %s" % (type(exc).__name__, exc))

    def dispatch(group):
        for index in group:
            record = records[index]
            if record.get("submission_attempted"):
                record["status"] = "UNKNOWN_SUBMISSION"
                record["error"] = "submission outcome needs reconciliation before another POST"
                emit("pending", index, record)
                return "blocked", ()
        if quota_exhausted:
            for index in group:
                record = records[index]
                record["error"] = "daily simulation quota exhausted; resume after the reported reset"
                emit("deferred", index, record)
            return "blocked", ()
        if simulation_slots.submission_retry_at(plan["account"]) is not None:
            for index in group:
                record = records[index]
                record["error"] = "simulation submission deferred by shared Retry-After cooldown"
                emit("deferred", index, record)
            return "blocked", ()
        allowed = tuple(index for index in group if records[index].get("status") != "FAILED"
                        and settings_allowed(index))
        if not allowed:
            return "skipped", ()
        if len(allowed) == 1:
            return submit_record(allowed[0]), allowed
        return native.submit(allowed), allowed

    queued = [index for index, record in enumerate(records)
              if record.get("status") in {"QUEUED", None}
              and not (record.get("progress_url") or record.get("metrics_pending") or record.get("multi_request_id"))]
    groups = (simulation_multi.request_groups(queued, payload_for, multi_size) if native_enabled
              else ((index,) for index in queued))
    next_group = next(groups, None)
    stop_submitting = recover_only or any(record.get("status") == "UNKNOWN_SUBMISSION" for record in records)
    unresolved_polls = 0
    read_workers = max(2, batch_size)
    attempted_orphan_recovery = launched_here = False

    with (ThreadPoolExecutor(max_workers=read_workers) as progress_pool,
          ThreadPoolExecutor(max_workers=read_workers) as metrics_pool):
        progress_reads = {}
        metrics_reads = []

        def read_progress(group):
            future = (progress_pool.submit(poll_record, group[0]) if len(group) == 1
                      else progress_pool.submit(native.poll_parent, group))
            progress_reads[future] = group

        def read_metrics(index):
            if fetch_metrics and records[index].get("metrics_pending"):
                metrics_reads.append(metrics_pool.submit(fetch_metrics_record, index))

        def read_children_and_metrics(group):
            native.collect_children(group)
            for index in group:
                if fetch_metrics and records[index].get("metrics_pending"):
                    fetch_metrics_record(index)

        def read_native_results(group):
            if native.needs_children(group):
                metrics_reads.append(metrics_pool.submit(read_children_and_metrics, group))
            else:
                for index in group:
                    read_metrics(index)

        for group in saved_groups:
            if native.needs_parent_read(group):
                read_progress(group)
            else:
                read_native_results(group)
        for index, record in enumerate(records):
            if record.get("multi_request_id"):
                continue
            if record.get("progress_url"):
                read_progress((index,))
            else:
                read_metrics(index)

        # Authenticated recovery restores shared reservations before dispatch. With
        # unidentified legacy jobs, wait for the original reads before using slots.
        untracked_reads = set() if plan["account"].get("fingerprint") else set(progress_reads)
        while True:
            shared_slot_wait = None
            while (next_group is not None and not stop_submitting and not untracked_reads
                   and len(progress_reads) + unresolved_polls < plan["effective"]
                   and (deadline is None or _ml.time.monotonic() < deadline)):
                outcome, accepted = dispatch(next_group)
                if outcome == "accepted":
                    launched_here = True
                    read_progress(accepted)
                elif outcome in {"contended", "blocked"}:
                    if outcome == "contended" and not attempted_orphan_recovery:
                        attempted_orphan_recovery = True
                        try:
                            recovery_deadline = _ml.time.monotonic() + _ml.QUICK_ORPHAN_SWEEP_SECONDS
                            if deadline is not None:
                                recovery_deadline = min(recovery_deadline, deadline)
                            recovered = _ml.recover_accepted_slots(s, plan["account"], deadline=recovery_deadline)
                            if scheduling_report is not None:
                                scheduling_report["accepted_slot_recovery"] = recovered
                            if recovered["released"]:
                                continue
                            if _ml.recover_quick_orphans(s, plan["account"], deadline=recovery_deadline):
                                continue
                        except _ml.AuthError as exc:
                            if not launched_here:
                                raise
                            # Keep this invocation's accepted jobs available to its caller.
                            _ml.note(str(exc))
                    if (outcome == "contended" and len(next_group) > 1
                            and not native_poll_once and deadline is not None):
                        if progress_reads:
                            # A different runner can release capacity while our
                            # own accepted parents still need to be polled.
                            shared_slot_wait = min(0.25, max(0.0, deadline - _ml.time.monotonic()))
                        else:
                            while _ml.time.monotonic() < deadline and not shared_state()["available"]:
                                _ml.sleep(min(0.25, max(0.0, deadline - _ml.time.monotonic())))
                            if _ml.time.monotonic() < deadline:
                                continue
                    stop_submitting = outcome == "blocked"
                    break
                next_group = next(groups, None)

            if not progress_reads:
                break
            ready, _ = wait(progress_reads, timeout=shared_slot_wait, return_when=FIRST_COMPLETED)
            for future in ready:
                group = progress_reads.pop(future)
                read_result(future)
                pending = (records[group[0]].get("progress_url") if len(group) == 1
                           else native.pending_on_platform(group))
                if future in untracked_reads:
                    untracked_reads.remove(future)
                    stop_submitting = stop_submitting or bool(pending)
                if pending:
                    # The attempt still occupies a slot even though this read has yielded.
                    unresolved_polls += 1
                elif len(group) > 1:
                    read_native_results(group)
                else:
                    read_metrics(group[0])

        for future in metrics_reads:
            read_result(future)

    shared_state()
    return records
