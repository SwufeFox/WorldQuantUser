"""Durable simulation reservations scoped to one account and research workspace.

Locks protect local preparation and journal updates. Reservations that reached
POST have no expiry: an interrupted POST may have been accepted. Abandoned
preparation can be reclaimed before any POST. Other workspaces and
unidentifiable accounts are not covered.
"""

import hashlib
import math
import os
import time
import uuid
from collections.abc import Mapping

import cache_store
import research_paths


TERMINAL_FAILURE_STATUSES = frozenset({"ERROR", "FAIL", "TIMEOUT", "CANCELLED", "CANCELED"})
TERMINAL_STATUSES = frozenset({"COMPLETE", "COMPLETED", "WARNING"}) | TERMINAL_FAILURE_STATUSES


def _workspace():
    return os.path.normcase(os.path.abspath(research_paths.research_dir()))


def _fingerprint(account):
    return account.get("fingerprint") if isinstance(account, dict) else None


def _paths(fingerprint):
    key = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    root = research_paths.resolve("simulation-slots")
    # Keep the lock beside the durable journal, outside the disposable cache.
    return os.path.join(root, key + ".json"), os.path.join(root, key + ".lock")


def _load(path, fingerprint):
    journal, error = cache_store.load_json(path)
    if error == "missing":
        return {"version": 1, "account_fingerprint": fingerprint, "reservations": {}}
    if error or not isinstance(journal, dict) or journal.get("version") != 1 or (
        journal.get("account_fingerprint") != fingerprint
        or not isinstance(journal.get("reservations"), dict)
        or any(not isinstance(value, dict) for value in journal.get("reservations", {}).values())
        or not isinstance(journal.get("not_submitted", {}), dict)
    ):
        raise cache_store.CacheError("simulation slot journal cannot be read safely: %s" % (error or path))
    return journal


def _report(account, limit, active=None):
    shared = bool(_fingerprint(account))
    return {
        "state": "shared_workspace" if shared else "unidentifiable_account",
        "scope": "account_and_research_workspace" if shared else "invocation_only",
        "workspace": _workspace(),
        "active": active,
        "available": max(0, limit - active) if active is not None else limit,
        "cross_workspace_coordinated": False,
        "age_expiry": False,
    }


def snapshot(account, limit):
    """Return current coordination evidence; unknown accounts create no files."""
    fingerprint = _fingerprint(account)
    if not fingerprint:
        return _report(account, limit)
    path, lock = _paths(fingerprint)
    with cache_store.file_lock(lock):
        journal = _load(path, fingerprint)
        return _report(account, limit, len(journal["reservations"]))


def reserve(account, limit):
    """Atomically reserve before POST; return ``(identity, shared_state)``.

    A null identity with zero available slots means defer. For an unidentified
    account the caller uses its invocation budget without shared coordination.
    """
    fingerprint = _fingerprint(account)
    if not fingerprint:
        return None, _report(account, limit)
    path, lock = _paths(fingerprint)
    with cache_store.file_lock(lock):
        journal = _load(path, fingerprint)
        identity, state = _reserve(journal, account, limit, "submitting")
        if identity:
            cache_store.write_json_atomic(path, journal)
        return identity, state


def _reserve(journal, account, limit, phase):
    active = len(journal["reservations"])
    if active >= limit:
        return None, _report(account, limit, active)
    identity = {"id": uuid.uuid4().hex, "account_fingerprint": _fingerprint(account),
                "workspace": _workspace()}
    journal["reservations"][identity["id"]] = {"state": phase, "created_at": time.time()}
    return identity, _report(account, limit, active + 1)


def _record_unposted(journal, identity):
    entry = journal["reservations"].pop(identity)
    # Keep proof for a checkpoint that was saved before preparation stopped.
    journal.setdefault("not_submitted", {})[identity] = entry.get("created_at")


def prepare_submission(account, limit, checkpoint):
    """Reserve, checkpoint, then authorize the caller to POST.

    The preparation lock is never held during network I/O. If its owner dies,
    a reservation still marked ``preparing`` could not yet have been sent.
    """
    fingerprint = _fingerprint(account)
    if not fingerprint:
        state = _report(account, limit)
        checkpoint(None)
        return None, state
    path, lock = _paths(fingerprint)
    with cache_store.file_lock(lock + ".prepare"):
        with cache_store.file_lock(lock):
            journal = _load(path, fingerprint)
            abandoned = [identity for identity, entry in journal["reservations"].items()
                         if entry.get("state") == "preparing"]
            for identity in abandoned:
                _record_unposted(journal, identity)
            identity, state = _reserve(journal, account, limit, "preparing")
            if identity or abandoned:
                cache_store.write_json_atomic(path, journal)
        if identity is None:
            return identity, state

        ready = False
        try:
            checkpoint(identity)
            ready = True
        finally:
            with cache_store.file_lock(lock):
                journal = _load(path, fingerprint)
                entry = journal["reservations"][identity["id"]]
                if ready:
                    entry["state"] = "submitting"
                else:
                    _record_unposted(journal, identity["id"])
                cache_store.write_json_atomic(path, journal)
        return identity, state


def _reservation_identity(record, account):
    identity = record.get("slot_reservation")
    if identity:
        if identity.get("workspace") != _workspace():
            raise ValueError("simulation reservation belongs to another research workspace")
        fingerprint = identity.get("account_fingerprint")
        if not fingerprint or not identity.get("id"):
            raise ValueError("simulation reservation identity is incomplete")
        if account is not None and _fingerprint(account) != fingerprint:
            raise ValueError("simulation reservation belongs to another or unidentified account")
    return identity


def validate_reservation_owner(record, account):
    """Verify a saved reservation belongs to the current account and workspace."""
    _reservation_identity(record, account)


def _has_terminal_result(record):
    status = record.get("simulation_status", record.get("status"))
    return bool(record.get("alpha_id") or status in TERMINAL_STATUSES)


def _restore_progress(record, entry):
    url = entry.get("progress_url")
    if entry.get("state") != "accepted" or not isinstance(url, str) or not url:
        return None
    if record.get("progress_url") != url or record.get("status") == "UNKNOWN_SUBMISSION":
        record.update(status="PENDING", submission_attempted=True, progress_url=url,
                      slot_reservation_state="accepted", error=None)
    return url


def _restore_unposted(record):
    if record.get("submission_attempted") is False:
        return False
    record.update(status="QUEUED", submission_attempted=False,
                  slot_reservation_state="released", error=None)
    return True


def restore_record(record, account=None):
    """Restore a missing progress URL or a definitely unposted attempt.

    Return whether the in-memory record changed. Local inspection can read
    completed copies across workspaces. Pass the authenticated account before
    querying BRAIN for an unresolved attempt.
    """
    terminal = _has_terminal_result(record)
    if terminal and account is None:
        return False
    identity = _reservation_identity(record, account)
    if not identity or record.get("progress_url") or terminal:
        return False
    fingerprint = identity["account_fingerprint"]
    path, lock = _paths(fingerprint)
    # Writers replace the complete journal atomically.
    journal = _load(path, fingerprint)
    attempt = identity["id"]
    entry = journal["reservations"].get(attempt, {})
    if attempt in journal.get("not_submitted", {}):
        return _restore_unposted(record)
    if entry.get("state") == "preparing":
        try:
            with cache_store.file_lock(lock + ".prepare", timeout=0, poll=0):
                journal = _load(path, fingerprint)
                entry = journal["reservations"].get(attempt, {})
                if attempt in journal.get("not_submitted", {}) or entry.get("state") == "preparing":
                    return _restore_unposted(record)
        except cache_store.LockTimeout:
            return False
    return bool(_restore_progress(record, entry))


def reconcile_record(record, account=None, *, outcome=None, progress_url=None, alpha_id=None):
    """Reconcile one saved attempt using authoritative recovery evidence.

    Accepted jobs keep their slot and recover a missing URL from the journal.
    Observed terminal outcomes release the slot. Definite refusals also make
    the job retryable. Indeterminate failures keep their reservation.
    Persist the returned record with the batch checkpoint.
    """
    if outcome not in {None, "accepted", "terminal", "refused"}:
        raise ValueError("unknown simulation reservation outcome %r" % outcome)
    identity = _reservation_identity(record, account)
    if identity and outcome is None and not progress_url and not alpha_id:
        restore_record(record, account)
    fingerprint = identity["account_fingerprint"] if identity else _fingerprint(account)
    effective_url = progress_url or record.get("progress_url")
    if outcome is None:
        if alpha_id or _has_terminal_result(record):
            outcome = "terminal"
        elif effective_url:
            outcome = "accepted"
    unresolved = outcome == "accepted" or (outcome is None and (
        record.get("submission_attempted") or record.get("status") == "UNKNOWN_SUBMISSION"
    ))
    if not identity and fingerprint and unresolved:
        identity = {"id": uuid.uuid4().hex, "account_fingerprint": fingerprint,
                    "workspace": _workspace()}
        record["slot_reservation"] = identity
    if identity:
        path, lock = _paths(fingerprint)
        with cache_store.file_lock(lock):
            journal = _load(path, fingerprint)
            if outcome in {"terminal", "refused"}:
                journal["reservations"].pop(identity["id"], None)
            else:
                entry = journal["reservations"].get(identity["id"], {})
                if outcome is None and identity["id"] in journal.get("not_submitted", {}):
                    _restore_unposted(record)
                    return record
                if entry.get("state") == "preparing":
                    return record
                effective_url = _restore_progress(record, entry) or effective_url
                if entry.get("state") == "accepted":
                    outcome = "accepted"
                if outcome or unresolved:
                    entry.update(state="accepted" if outcome == "accepted" else "unknown",
                                 progress_url=effective_url)
                    journal["reservations"][identity["id"]] = entry
            if outcome or unresolved:
                cache_store.write_json_atomic(path, journal)
    if outcome == "accepted" and effective_url:
        record["progress_url"] = effective_url
    elif progress_url:
        record["progress_url"] = progress_url
    if alpha_id:
        record["alpha_id"] = alpha_id
    if outcome == "refused":
        record.update(status="QUEUED", submission_attempted=False)
        record.pop("progress_url", None)
    if identity and (outcome or unresolved):
        record["slot_reservation_state"] = (
            "released" if outcome in {"terminal", "refused"}
            else "accepted" if outcome == "accepted" else "unknown"
        )
    return record


def _quota_number(value, integer=False):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    if integer and isinstance(value, float):
        return None
    try:
        number = int(value) if integer else float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if number >= 0 and (integer or math.isfinite(number)) else None


def observe_quota(account, headers):
    """Retain only documented simulation-quota headers, never unrelated headers."""
    if not isinstance(headers, Mapping):
        return None
    headers = {str(key).lower(): value for key, value in headers.items()}
    observation = {
        "limit": _quota_number(headers.get("x-ratelimit-limit"), integer=True),
        "remaining": _quota_number(headers.get("x-ratelimit-remaining"), integer=True),
        "reset_seconds": _quota_number(headers.get("x-ratelimit-reset")),
    }
    if all(value is None for value in observation.values()):
        return None
    if (observation["limit"] is not None and observation["remaining"] is not None
            and observation["remaining"] > observation["limit"]):
        observation["remaining"] = None
    observation.update(observed_at=time.time(), source="POST /simulations response headers")
    fingerprint = _fingerprint(account)
    if fingerprint:
        path, lock = _paths(fingerprint)
        try:
            with cache_store.file_lock(lock):
                journal = _load(path, fingerprint)
                journal["quota"] = observation
                cache_store.write_json_atomic(path, journal)
        except (OSError, cache_store.CacheError) as error:
            # Losing telemetry must not lose an already accepted progress URL.
            observation["storage_error"] = type(error).__name__
    return observation


def quota_snapshot(account, observation=None):
    """Describe the latest observation; a reset makes it stale, never refills it.

    Without a reset time, zero stops its current run only. A later invocation
    may obtain a fresh response instead of being blocked indefinitely by history.
    """
    current_run = observation is not None
    fingerprint = _fingerprint(account)
    if observation is None and fingerprint:
        path, lock = _paths(fingerprint)
        with cache_store.file_lock(lock):
            observation = _load(path, fingerprint).get("quota")
    observation = observation if isinstance(observation, dict) else {}
    limit = _quota_number(observation.get("limit"), integer=True)
    remaining = _quota_number(observation.get("remaining"), integer=True)
    observed_at = _quota_number(observation.get("observed_at"))
    reset_seconds = _quota_number(observation.get("reset_seconds"))
    if observed_at is None or (limit is not None and remaining is not None and remaining > limit):
        remaining = None
    reset_at = observed_at + reset_seconds if observed_at is not None and reset_seconds is not None else None
    now = time.time()
    state = ("unknown" if observed_at is None or remaining is None else
             "reset_unknown" if reset_at is None else
             "reset_due" if reset_at <= now else "observed")
    return {
        "state": state, "limit": limit, "remaining": remaining,
        "observed_at": observed_at, "reset_seconds": reset_seconds,
        "reset_at": reset_at,
        "reset_after_seconds": max(0.0, reset_at - now) if reset_at is not None else None,
        "blocked": remaining == 0 and (state == "observed" or (current_run and state == "reset_unknown")),
        "source": observation.get("source"),
        "scope": "account_and_research_workspace" if fingerprint and not observation.get("storage_error") else "invocation_only",
        **({"storage_error": observation["storage_error"]} if observation.get("storage_error") else {}),
    }
