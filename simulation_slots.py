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
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import cache_store
import research_paths


TERMINAL_FAILURE_STATUSES = frozenset({"ERROR", "FAIL", "TIMEOUT", "CANCELLED", "CANCELED"})
TERMINAL_SUCCESS_STATUSES = frozenset({"COMPLETE", "COMPLETED", "WARNING"})
TERMINAL_STATUSES = TERMINAL_SUCCESS_STATUSES | TERMINAL_FAILURE_STATUSES


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
        or not isinstance(journal.get("quick_resolved", {}), dict)
        or any(not isinstance(value, dict) for value in journal.get("quick_resolved", {}).values())
        or not isinstance(journal.get("resolved", {}), dict)
        or any(not isinstance(value, dict) for value in journal.get("resolved", {}).values())
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
        return _report(account, _active_limit(journal, limit), len(journal["reservations"]))


def confirmed_accepted_count(account):
    """Read a lower bound for admission without reserving or repairing slots."""
    fingerprint = _fingerprint(account)
    if not fingerprint:
        return None
    path, _ = _paths(fingerprint)
    try:
        # Each writer replaces the journal atomically. This hint never holds a lock.
        reservations = _load(path, fingerprint)["reservations"]
    except (OSError, cache_store.CacheError):
        return None

    from machine_lib import simulation_reference_id

    accepted = set()
    for entry in reservations.values():
        # Legacy Quick reservations and unfinished preparations can be reclaimed by
        # normal dispatch. They cannot establish that preflight should be deferred.
        url = entry.get("progress_url")
        if (entry.get("state") != "accepted" or "quick_job_key" in entry
                or not isinstance(url, str) or not url.startswith("https://")):
            continue
        try:
            accepted.add(simulation_reference_id(url))
        except ValueError:
            continue
    return len(accepted)


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
        pruned = _prune_quick_preparations(journal)
        identity, state = _reserve(journal, account, limit, "submitting")
        if identity or pruned:
            cache_store.write_json_atomic(path, journal)
        return identity, state


def release_url(url):
    """Drop reservations that point at a simulation URL the platform no longer knows.

    A 404 means the job was cancelled outside this workspace; keeping its
    reservation would hold a slot forever and block new submissions.
    """
    if not isinstance(url, str) or not url.startswith("https://"):
        return []
    root = Path(research_paths.resolve("simulation-slots"))
    released = []
    for path in sorted(root.glob("*.json")):
        journal, error = cache_store.load_json(str(path))
        if error or not isinstance(journal, dict) or not isinstance(journal.get("reservations"), dict):
            continue
        reservations = journal["reservations"]
        dropped = [key for key, entry in reservations.items()
                   if isinstance(entry, dict) and entry.get("progress_url") == url]
        if not dropped:
            continue
        for key in dropped:
            reservations.pop(key, None)
        journal.setdefault("quick_resolved", {})[url.rstrip("/").split("/")[-1]] = {
            "state": "released", "progress_url": url}
        cache_store.write_json_atomic(str(path), journal)
        released.extend(dropped)
    return released


ORPHAN_RELEASE_STATUSES = TERMINAL_STATUSES


def reconcile_orphans(account, poll, *, max_age_min=45.0, hard_max_age_min=360.0,
                      now=None, deadline=None, ordinary_only=False):
    """Release only observed terminal acceptances, retaining their receipts.

    ``poll(url)`` supplies an authenticated observation for that exact URL.
    Age is diagnostic: time passing, a failed read or HTTP 404 cannot prove a
    live job finished or an ambiguous POST never happened. Network reads run
    outside the journal lock, and changed reservations are not overwritten.
    """
    fingerprint = _fingerprint(account)
    if not fingerprint:
        return {"released": [], "kept": [], "state": "account_unknown"}
    path, lock = _paths(fingerprint)
    stamp = time.time() if now is None else now

    def lock_options():
        return {} if deadline is None else {"timeout": max(0, deadline - time.monotonic())}

    # Snapshot under the lock, then poll outside it. Holding the lock across
    # network reads made every other writer time out at 60s.
    with cache_store.file_lock(lock, **lock_options()):
        snapshot = dict((_load(path, fingerprint).get("reservations") or {}))

    decisions, kept = [], []
    for key, entry in snapshot.items():
        if ordinary_only and entry.get("quick_job_key"):
            continue
        url = entry.get("progress_url")
        created = _quota_number(entry.get("created_at"))
        age_min = max(0, stamp - created) / 60.0 if created is not None else None
        item = {"id": key, "age_min": round(age_min, 1) if age_min is not None else None,
                "overdue": age_min is not None and age_min > min(max_age_min, hard_max_age_min)}
        unavailable = (entry.get("state") != "accepted" or not url
                       or entry.get("manual_location") or entry.get("identity_unverified")
                       or (entry.get("quick_job_key") and _quick_owner_is_live(entry.get("quick_owner"))))
        if unavailable:
            kept.append({**item, "status": None, "reason": "acceptance_unverified_or_owned"})
            continue
        if deadline is not None and time.monotonic() >= deadline:
            kept.append({**item, "status": None, "reason": "recovery_budget_exhausted"})
            continue
        answer = poll(url)
        if isinstance(answer, dict):
            status, alpha = answer.get("status"), answer.get("alpha") or answer.get("alpha_id")
        else:
            status, alpha = answer, None
        status = status.upper() if isinstance(status, str) else None
        if status in ORPHAN_RELEASE_STATUSES:
            decisions.append({**item, "status": status, "alpha": alpha})
        else:
            kept.append({**item, "status": status,
                         "reason": "platform_pending" if status not in {None, "NOT_FOUND"} else "outcome_unknown"})

    released = []
    if decisions:
        with cache_store.file_lock(lock, **lock_options()):
            journal = _load(path, fingerprint)
            reservations = journal.get("reservations") or {}
            for item in decisions:
                key = item["id"]
                if reservations.get(key) != snapshot[key]:
                    kept.append({**item, "reason": "reservation_changed"})
                    continue
                entry = reservations.pop(key)
                collection = "quick_resolved" if entry.get("quick_job_key") else "resolved"
                journal.setdefault(collection, {})[key] = {
                    **entry, "quick_owner": None, "resolved_at": stamp,
                    "terminal_result": {"status": item["status"], "alpha_id": item["alpha"]},
                }
                released.append(item)
            if released:
                cache_store.write_json_atomic(path, journal)
    return {"released": released, "kept": kept}


def _active_limit(journal, limit):
    for entry in journal["reservations"].values():
        shared_limit = entry.get("quick_limit")
        if type(shared_limit) is int and shared_limit > 0:
            owner = entry.get("quick_owner")
            # A reservation whose worker is gone must not keep throttling the
            # workspace at the budget it happened to request. Probe reservations
            # carry no owner and no quick_limit, so they are unaffected.
            if owner is not None and not _quick_owner_is_live(owner):
                continue
            limit = min(limit, shared_limit)
    return limit


def _reserve(journal, account, limit, phase):
    limit = _active_limit(journal, limit)
    active = len(journal["reservations"])
    if active >= limit:
        return None, _report(account, limit, active)
    identity = {"id": uuid.uuid4().hex, "account_fingerprint": _fingerprint(account),
                "workspace": _workspace()}
    journal["reservations"][identity["id"]] = {"state": phase, "created_at": time.time()}
    return identity, _report(account, limit, active + 1)


def _process_stat(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]
    except (OSError, IndexError):
        return None, None


def _quick_owner():
    pid = os.getpid()
    return {"pid": pid, "start": _process_stat(pid)[1]}


def _quick_owner_is_live(owner):
    pid = owner.get("pid") if isinstance(owner, dict) else None
    if type(pid) is not int or pid < 1:
        return False
    state, start = _process_stat(pid)
    if state is not None:
        return state not in {"Z", "X"} and (owner.get("start") is None or owner["start"] == start)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _prune_quick_preparations(journal):
    reservations = journal["reservations"]
    abandoned = [key for key, entry in reservations.items()
                 if entry.get("state") == "quick_preparing"
                 and not _quick_owner_is_live(entry.get("quick_owner"))]
    for key in abandoned:
        del reservations[key]
    return bool(abandoned)


@contextmanager
def _locked_quick_journal(account, *, deadline=None):
    fingerprint = _fingerprint(account)
    if not fingerprint:
        raise ValueError("shared quick simulation slots require an identified account")
    path, lock = _paths(fingerprint)
    lock_options = {} if deadline is None else {"timeout": max(0, deadline - time.monotonic())}
    with cache_store.file_lock(lock, **lock_options):
        yield _load(path, fingerprint), path


def quick_acquire(account, limit, job_key, *, existing_url=None, prepare=False,
                  identity_unverified=False):
    """Claim a recorded QUICK job or reserve one shared slot before its POST.

    A new ``submitting`` result authorizes exactly one POST. An old submitting
    reservation without a URL is indeterminate, even when its owner has died.
    With ``prepare=True``, a new reservation is reclaimable until submitting.
    An imported URL belongs to an already accepted job and counts until terminal.
    A zero budget blocks new reservations while allowing accepted jobs to recover.
    """
    if type(limit) is not int or limit < 0 or not isinstance(job_key, str) or not job_key:
        raise ValueError("quick simulation slots require a non-negative limit and job key")
    owner = _quick_owner()
    with _locked_quick_journal(account) as (journal, path):
        if _prune_quick_preparations(journal):
            cache_store.write_json_atomic(path, journal)
        resolved = next(((key, entry) for key, entry in journal.get("quick_resolved", {}).items()
                         if entry.get("quick_job_key") == job_key), None)
        if resolved:
            key, entry = resolved
            return {"id": key, **entry, "resolved": True}
        reservations = journal["reservations"]
        prior = next(((key, entry) for key, entry in reservations.items()
                      if entry.get("quick_job_key") == job_key), None)
        if prior:
            key, entry = prior
            current_owner = entry.get("quick_owner")
            if current_owner != owner and _quick_owner_is_live(current_owner):
                return None
            stored_url = entry.get("progress_url")
            if existing_url and stored_url and existing_url != stored_url:
                raise ValueError("saved progress URL differs from the shared quick reservation")
            if stored_url:
                entry.update(state="accepted", quick_owner=owner)
            elif entry.get("state") == "quick_preparing" and not existing_url and limit == 0:
                # No POST has begun. A later zero budget revokes this owner's
                # preparation instead of allowing its old limit to authorize work.
                del reservations[key]
                cache_store.write_json_atomic(path, journal)
                return None
            elif entry.get("state") != "quick_preparing" or existing_url:
                # An existing unknown POST cannot inherit an unverified legacy address.
                entry.update(state="unknown", quick_owner=None)
        else:
            if existing_url:
                if any(entry.get("progress_url") == existing_url for entry in reservations.values()):
                    raise ValueError("progress URL already belongs to another simulation reservation")
                key = uuid.uuid4().hex
                entry = {"state": "accepted", "progress_url": existing_url, "created_at": time.time()}
                if identity_unverified:
                    entry["identity_unverified"] = True
                reservations[key] = entry
            else:
                phase = "quick_preparing" if prepare else "submitting"
                identity, _ = _reserve(journal, account, limit, phase)
                if identity is None:
                    return None
                key = identity["id"]
                entry = reservations[key]
            entry.update(quick_job_key=job_key, quick_limit=limit, quick_owner=owner)
        cache_store.write_json_atomic(path, journal)
        return {"id": key, **entry}


def quick_update(account, identity, outcome, *, progress_url=None, retain=False,
                 terminal_result=None, deadline=None):
    """Reconcile a QUICK reservation using the result of the corresponding HTTP request.

    Parking yields ownership of the poll without releasing server occupancy.
    An orphan's observed terminal URL can be retained for the original ledger.
    """
    if outcome not in {"submitting", "accepted", "refused", "terminal", "unknown", "parked"}:
        raise ValueError(f"unknown quick simulation reservation outcome {outcome!r}")
    receipt = None
    if terminal_result is not None:
        from machine_lib import valid_simulation_id

        if not retain or outcome != "terminal" or not isinstance(terminal_result, dict):
            raise ValueError("only retained terminal quick simulations can store a result")
        status, alpha_id = terminal_result.get("status"), terminal_result.get("alpha_id")
        if not isinstance(status, str) or status not in TERMINAL_STATUSES:
            raise ValueError("retained quick simulation result requires a terminal status")
        if alpha_id is not None and not valid_simulation_id(alpha_id):
            raise ValueError("invalid Alpha ID in retained quick simulation result")
        receipt = {"status": status, "alpha_id": alpha_id}
    with _locked_quick_journal(account, deadline=deadline) as (journal, path):
        reservations = journal["reservations"]
        entry = reservations.get(identity)
        if not entry or not entry.get("quick_job_key"):
            resolved = journal.get("quick_resolved", {})
            if outcome == "terminal" and not retain and identity in resolved:
                resolved.pop(identity)
                cache_store.write_json_atomic(path, journal)
                return True
            return False
        if retain and (entry.get("state") != "accepted" or entry.get("manual_location")
                       or _quick_owner_is_live(entry.get("quick_owner"))):
            return False
        if outcome == "submitting":
            if entry.get("state") != "quick_preparing" or entry.get("quick_owner") != _quick_owner():
                return False
            entry["state"] = "submitting"
        elif outcome == "accepted":
            if not progress_url:
                raise ValueError("accepting a quick simulation requires its progress URL")
            if entry.get("progress_url") not in (None, progress_url):
                raise ValueError("cannot replace a quick simulation's progress URL")
            entry.update(state="accepted", progress_url=progress_url)
        elif outcome == "parked":
            if entry.get("state") != "accepted" or not entry.get("progress_url"):
                raise ValueError("only accepted quick simulations can be parked")
            entry["quick_owner"] = None
        elif outcome == "unknown":
            if entry.get("state") == "accepted":
                raise ValueError("an accepted quick simulation keeps its progress URL")
            entry.update(state="unknown", quick_owner=None)
        else:
            if outcome == "refused" and entry.get("state") == "accepted":
                raise ValueError("an accepted quick simulation is not a refused POST")
            if outcome == "terminal" and retain and entry.get("progress_url"):
                journal.setdefault("quick_resolved", {})[identity] = {
                    **entry, "quick_owner": None, "terminal_result": receipt,
                }
            reservations.pop(identity)
        cache_store.write_json_atomic(path, journal)
        return True


def quick_orphans(account, *, exclude_job=None):
    """List accepted Quick jobs that are ready for automatic recovery."""
    fingerprint = _fingerprint(account)
    if not fingerprint:
        raise ValueError("shared quick simulation slots require an identified account")
    path, _ = _paths(fingerprint)
    # The journal is atomically replaced and updates recheck ownership under the lock.
    reservations = _load(path, fingerprint)["reservations"]
    now = time.time()
    return [
        {"id": key, **entry, "job_key": entry["quick_job_key"]}
        for key, entry in reservations.items()
        if entry.get("quick_job_key") and entry["quick_job_key"] != exclude_job
        and entry.get("state") == "accepted" and entry.get("progress_url")
        and not entry.get("manual_location")
        and (_quota_number(entry.get("next_poll_at")) or 0) <= now
        and not _quick_owner_is_live(entry.get("quick_owner"))
    ]


def quick_defer(account, identity, next_poll_at, *, deadline=None):
    """Persist a server polling delay without changing accepted slot ownership."""
    next_poll_at = _quota_number(next_poll_at)
    if next_poll_at is None:
        raise ValueError("quick polling deadline must be finite and non-negative")
    with _locked_quick_journal(account, deadline=deadline) as (journal, path):
        entry = journal["reservations"].get(identity)
        if not entry or not entry.get("quick_job_key") or entry.get("state") != "accepted":
            return False
        if (_quota_number(entry.get("next_poll_at")) or 0) < next_poll_at:
            entry["next_poll_at"] = next_poll_at
            cache_store.write_json_atomic(path, journal)
        return True


def quick_attach_location(account, job_key, progress_url, *, terminal_failure=None, next_poll_at=None):
    """Bind the caller's verified original Location to an unresolved Quick lease."""
    if terminal_failure is not None and terminal_failure not in TERMINAL_FAILURE_STATUSES:
        raise ValueError("manual reconciliation requires a terminal failure")
    target_path = urlsplit(progress_url).path.rstrip("/")
    with _locked_quick_journal(account) as (journal, path):
        reservations = journal["reservations"]
        resolved = journal.get("quick_resolved", {})
        if terminal_failure:
            for key, entry in resolved.items():
                if (entry.get("quick_job_key") == job_key and entry.get("progress_url") == progress_url
                        and entry.get("manual_failure_confirmed")):
                    return {"id": key, **entry, "resolved": True}
        match = next(((key, entry) for key, entry in reservations.items()
                      if entry.get("quick_job_key") == job_key), None)
        if not match:
            raise ValueError("no unresolved Quick reservation matches this job key")
        key, entry = match
        new_location = entry.get("state") in {"unknown", "submitting"} and not entry.get("progress_url")
        parked_manual = (terminal_failure and entry.get("state") == "accepted"
                         and entry.get("manual_location") and entry.get("progress_url") == progress_url)
        if not (new_location or parked_manual) or _quick_owner_is_live(entry.get("quick_owner")):
            raise ValueError("Quick reservation already has a Location or may still be posting")
        for collection in (reservations, resolved):
            if any(other != key and isinstance(row.get("progress_url"), str)
                   and urlsplit(row["progress_url"]).path.rstrip("/") == target_path
                   for other, row in collection.items()):
                raise ValueError("Location already belongs to another simulation reservation")
        entry.update(state="accepted", progress_url=progress_url, quick_owner=None)
        if terminal_failure:
            entry.pop("next_poll_at", None)
            entry.pop("manual_location", None)
            entry.pop("identity_unverified", None)
            entry.update(manual_failure_confirmed=True,
                         terminal_result={"status": terminal_failure, "alpha_id": None})
            journal.setdefault("quick_resolved", {})[key] = reservations.pop(key)
        else:
            entry.update(identity_unverified=True, manual_location=True)
            if next_poll_at is not None:
                entry["next_poll_at"] = next_poll_at
        cache_store.write_json_atomic(path, journal)
        return {"id": key, **entry}


def quick_known_jobs(account):
    """Return durable Quick job keys without changing the shared journal."""
    with _locked_quick_journal(account) as (journal, _):
        return {
            entry["quick_job_key"]
            for collection in (journal["reservations"], journal.get("quick_resolved", {}))
            for entry in collection.values()
            if entry.get("quick_job_key")
        }


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
            pruned = _prune_quick_preparations(journal)
            abandoned = [identity for identity, entry in journal["reservations"].items()
                         if entry.get("state") == "preparing"]
            for identity in abandoned:
                _record_unposted(journal, identity)
            identity, state = _reserve(journal, account, limit, "preparing")
            if identity or abandoned or pruned:
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


def _restore_terminal(record, entry):
    receipt = entry.get("terminal_result") or {}
    if not isinstance(receipt, dict):
        return False
    status, alpha_id = receipt.get("status"), receipt.get("alpha_id")
    if status not in TERMINAL_STATUSES:
        return False
    record.update(status=status, simulation_status=status, submission_attempted=True,
                  slot_reservation_state="released")
    record.pop("progress_url", None)
    record.pop("next_poll_at", None)
    record.pop("poll_after_seconds", None)
    if alpha_id:
        record.update(alpha_id=alpha_id, status="PENDING", metrics_pending=True)
    if status in TERMINAL_FAILURE_STATUSES and not record.get("error"):
        record["error"] = "simulation ended with %s" % status
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
    if not identity or terminal:
        return False
    fingerprint = identity["account_fingerprint"]
    path, lock = _paths(fingerprint)
    # Writers replace the complete journal atomically.
    journal = _load(path, fingerprint)
    attempt = identity["id"]
    resolved = journal.get("resolved", {}).get(attempt)
    if resolved and _restore_terminal(record, resolved):
        return True
    if record.get("progress_url"):
        return False
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


def submission_retry_at(account):
    """Return an active shared simulation POST cooldown, as a wall-clock timestamp."""
    fingerprint = _fingerprint(account)
    if not fingerprint:
        return None
    path, _ = _paths(fingerprint)
    # The journal is replaced atomically. Do not block this read on preparation.
    retry_at = _quota_number(_load(path, fingerprint).get("submit_not_before"))
    return retry_at if retry_at is not None and retry_at > time.time() else None


def defer_submission(account, wait_seconds, *, deadline=None):
    """Share a transient POST refusal's retry time without affecting accepted jobs."""
    delay = _quota_number(wait_seconds)
    if not delay:
        return None
    retry_at = time.time() + delay
    if not math.isfinite(retry_at):
        return None
    fingerprint = _fingerprint(account)
    if fingerprint:
        path, lock = _paths(fingerprint)
        lock_options = ({} if deadline is None or not math.isfinite(deadline)
                        else {"timeout": max(0, deadline - time.monotonic())})
        try:
            with cache_store.file_lock(lock, **lock_options):
                journal = _load(path, fingerprint)
                previous = _quota_number(journal.get("submit_not_before")) or 0
                if previous >= retry_at:
                    return previous
                journal["submit_not_before"] = retry_at
                cache_store.write_json_atomic(path, journal)
        except (OSError, cache_store.CacheError):
            # The caller still honors the observed delay if it cannot be shared.
            pass
    return retry_at


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
