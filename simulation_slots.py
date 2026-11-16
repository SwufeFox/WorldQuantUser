"""Durable simulation reservations scoped to one account and research workspace.

Locks protect journal updates only. Reservations have no expiry: an interrupted
POST may have been accepted, so only definite refusal or observed completion
releases its slot. Other workspaces and unidentifiable accounts are not covered.
"""

import hashlib
import os
import time
import uuid

import cache_store
import research_paths


TERMINAL_STATUSES = frozenset({"COMPLETE", "COMPLETED", "ERROR", "FAIL", "CANCELLED", "CANCELED"})


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
        active = len(journal["reservations"])
        if active >= limit:
            return None, _report(account, limit, active)
        identity = {"id": uuid.uuid4().hex, "account_fingerprint": fingerprint,
                    "workspace": _workspace()}
        journal["reservations"][identity["id"]] = {
            "state": "submitting", "created_at": time.time(),
        }
        cache_store.write_json_atomic(path, journal)
        return identity, _report(account, limit, active + 1)


def reconcile_record(record, account=None, *, outcome=None, progress_url=None, alpha_id=None):
    """Reconcile one saved attempt using authoritative recovery evidence.

    ``accepted`` preserves the reservation and may attach a progress URL.
    ``terminal`` releases it; ``refused`` releases and makes the record retryable.
    With no explicit outcome, infer only a terminal platform status/alpha ID or
    an accepted progress URL. A generic FAILED/unknown record is never released.
    A record copied from another workspace cannot reconcile this workspace.
    Mutates and returns the record; the caller must persist it afterwards.
    """
    if outcome not in {None, "accepted", "terminal", "refused"}:
        raise ValueError("unknown simulation reservation outcome %r" % outcome)
    identity = record.get("slot_reservation")
    if identity:
        if identity.get("workspace") != _workspace():
            raise ValueError("simulation reservation belongs to another research workspace")
        fingerprint = identity.get("account_fingerprint")
        if not fingerprint or not identity.get("id"):
            raise ValueError("simulation reservation identity is incomplete")
        if account is not None and _fingerprint(account) != fingerprint:
            raise ValueError("simulation reservation belongs to another or unidentified account")
    else:
        fingerprint = _fingerprint(account)
    effective_alpha = alpha_id or record.get("alpha_id")
    effective_url = progress_url or record.get("progress_url")
    if outcome is None:
        if effective_alpha or record.get("simulation_status", record.get("status")) in TERMINAL_STATUSES:
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
    if identity and (outcome or unresolved):
        path, lock = _paths(fingerprint)
        with cache_store.file_lock(lock):
            journal = _load(path, fingerprint)
            if outcome in {"terminal", "refused"}:
                journal["reservations"].pop(identity["id"], None)
            else:
                entry = journal["reservations"].setdefault(identity["id"], {})
                entry.update(state="accepted" if outcome == "accepted" else "unknown",
                             progress_url=effective_url)
            cache_store.write_json_atomic(path, journal)
    if progress_url:
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
