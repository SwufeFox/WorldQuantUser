"""Durable simulation reservations scoped to one account and research workspace.

Locks protect local preparation and journal updates. Reservations that reached
POST have no expiry: an interrupted POST may have been accepted. Abandoned
preparation can be reclaimed before any POST. Other workspaces and
unidentifiable accounts are not covered.
"""

import hashlib
import math
import os
import sqlite3
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import cache_store
import research_paths
import workspace_db


TERMINAL_FAILURE_STATUSES = frozenset({"ERROR", "FAIL", "TIMEOUT", "CANCELLED", "CANCELED"})
TERMINAL_SUCCESS_STATUSES = frozenset({"COMPLETE", "COMPLETED", "WARNING"})
TERMINAL_STATUSES = TERMINAL_SUCCESS_STATUSES | TERMINAL_FAILURE_STATUSES


def _workspace():
    return os.path.normcase(os.path.abspath(research_paths.research_dir()))


def _fingerprint(account):
    return account.get("fingerprint") if isinstance(account, dict) else None


def _configured_slot_limit(requested, name):
    path = research_paths.resolve("simulation-slots", name)
    try:
        with open(path, encoding="ascii") as handle:
            ceiling = int(handle.read().strip())
    except FileNotFoundError:
        return requested
    if ceiling < 1:
        raise ValueError(f"{name} must be a positive integer")
    return min(requested, ceiling)


def quick_slot_limit(requested):
    """Apply the optional workspace QUICK ceiling to new reservations."""
    return _configured_slot_limit(requested, "quick-slot-limit")


def quick_single_slot_limit(requested):
    """Apply the optional ceiling on compatibility-runner SINGLE requests."""
    return _configured_slot_limit(requested, "quick-single-slot-limit")


def _paths(fingerprint, research_dir=None):
    key = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    root = Path(research_dir or research_paths.research_dir()) / "simulation-slots"
    # Keep the lock beside the durable journal, outside the disposable cache.
    return os.path.join(root, key + ".json"), os.path.join(root, key + ".lock")


def _validate_journal(journal, fingerprint, path):
    if not isinstance(journal, dict) or journal.get("version") != 1 or (
        journal.get("account_fingerprint") != fingerprint
        or not isinstance(journal.get("reservations"), dict)
        or any(not isinstance(value, dict) for value in journal.get("reservations", {}).values())
        or not isinstance(journal.get("not_submitted", {}), dict)
        or not isinstance(journal.get("quick_resolved", {}), dict)
        or any(not isinstance(value, dict) for value in journal.get("quick_resolved", {}).values())
        or not isinstance(journal.get("resolved", {}), dict)
        or any(not isinstance(value, dict) for value in journal.get("resolved", {}).values())
        or not isinstance(journal.get("stuck", {}), dict)
        or any(not isinstance(value, dict) for value in journal.get("stuck", {}).values())
        or not isinstance(journal.get("cancelled", {}), dict)
        or any(not isinstance(value, dict) for value in journal.get("cancelled", {}).values())
    ):
        raise cache_store.CacheError("simulation slot journal cannot be read safely: %s" % path)
    return journal


def _load(path, fingerprint, *, research_dir=None):
    try:
        journal = workspace_db.load_slot_journal(fingerprint, research_dir=research_dir, legacy_path=path)
    except sqlite3.Error as exc:
        raise cache_store.CacheError("could not read the SQLite simulation slot journal") from exc
    return _validate_journal(journal, fingerprint, path)


def _save(path, journal, *, research_dir=None):
    try:
        workspace_db.save_slot_journal(journal["account_fingerprint"], journal,
                                       research_dir=research_dir)
    except sqlite3.Error as exc:
        raise cache_store.CacheError("could not save the SQLite simulation slot journal") from exc


def has_slot_reservations(research_dir):
    """Check for SQLite reservations or legacy journals that may need reconciliation."""
    if workspace_db.has_slot_reservations(research_dir=research_dir):
        return True
    return any((Path(research_dir) / "simulation-slots").glob("*.json"))


def import_legacy_journal(path, *, research_dir, archive_path):
    """Archive a valid legacy journal after preserving any newer SQLite state."""
    path = Path(path)
    raw, error = cache_store.load_json(path)
    fingerprint = raw.get("account_fingerprint") if isinstance(raw, dict) else None
    if error or not isinstance(fingerprint, str) or not fingerprint:
        return False
    expected_path, lock = _paths(fingerprint, research_dir)
    if path.resolve() != Path(expected_path).resolve():
        return False
    with cache_store.file_lock(lock):
        raw, error = cache_store.load_json(path)
        if error:
            return False
        _validate_journal(raw, fingerprint, path)
        workspace_db.save_slot_journal(fingerprint, raw, research_dir=research_dir, import_only=True)
        archived = Path(archive_path)
        snapshot = path.read_bytes()
        try:
            cache_store.link_bytes_exclusive(archived.parent, archived, snapshot)
        except FileExistsError:
            if archived.read_bytes() != snapshot:
                return False
        path.unlink()
    return True


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
            _save(path, journal)
        return identity, state


ORPHAN_RELEASE_STATUSES = TERMINAL_STATUSES


def reconcile_orphans(account, poll, *, max_age_min=45.0, hard_max_age_min=360.0,
                      now=None, deadline=None, ordinary_only=False, stuck_after_min=None,
                      cancel=None, research_dir=None):
    """Release only observed terminal acceptances, retaining their receipts.

    ``poll(url)`` supplies an authenticated observation for that exact URL.
    Age and missing receipts are diagnostic. A confirmed HTTP 404 on a verified
    ordinary acceptance records that the task is gone. Network reads run
    outside the journal lock, and changed reservations are not overwritten.

    ``max_age_min`` and ``hard_max_age_min`` retain their compatibility names
    and defaults (45 and 360 minutes). They describe the reported age only:
    ``overdue`` uses the earlier threshold and ``past_hard_age`` uses the
    second one. Neither threshold permits releasing a reservation.
    """
    age_diagnostics = {"diagnostic_only": True, "max_age_min": max_age_min,
                       "hard_max_age_min": hard_max_age_min,
                       "overdue_after_min": min(max_age_min, hard_max_age_min)}
    fingerprint = _fingerprint(account)
    if not fingerprint:
        return {"released": [], "cancelled": [], "kept": [], "state": "account_unknown",
                "age_diagnostics": age_diagnostics}
    path, lock = _paths(fingerprint, research_dir)
    stamp = time.time() if now is None else now

    def lock_options():
        return {} if deadline is None else {"timeout": max(0, deadline - time.monotonic())}

    # Snapshot under the lock, then poll outside it. Holding the lock across
    # network reads made every other writer time out at 60s.
    with cache_store.file_lock(lock, **lock_options()):
        snapshot = dict(_load(path, fingerprint, research_dir=research_dir)["reservations"])

    decisions, kept, observations = [], [], {}
    for key, entry in snapshot.items():
        if ordinary_only and entry.get("quick_job_key"):
            continue
        url = entry.get("progress_url")
        created = _quota_number(entry.get("created_at"))
        age_min = max(0, stamp - created) / 60.0 if created is not None else None
        item = {"id": key, "age_min": round(age_min, 1) if age_min is not None else None,
                "overdue": age_min is not None and age_min > age_diagnostics["overdue_after_min"],
                "past_hard_age": age_min is not None and age_min > hard_max_age_min}
        unavailable = (entry.get("state") != "accepted" or not url
                       or entry.get("manual_location") or entry.get("identity_unverified")
                       or (entry.get("quick_job_key") and _quick_owner_is_live(entry.get("quick_owner"))))
        if unavailable:
            kept.append({**item, "status": None, "reason": "acceptance_unverified_or_owned"})
            continue
        if deadline is not None and time.monotonic() >= deadline:
            kept.append({**item, "status": None, "reason": "recovery_budget_exhausted"})
            continue
        if (_quota_number(entry.get("next_poll_at")) or 0) > stamp:
            kept.append({**item, "status": None, "reason": "poll_deferred"})
            continue
        answer = poll(url)
        if isinstance(answer, dict):
            status, alpha = answer.get("status"), answer.get("alpha") or answer.get("alpha_id")
            progress = answer.get("progress")
            children = answer.get("children")
        else:
            status, alpha = answer, None
            progress = children = None
        status = status.upper() if isinstance(status, str) else None
        if (status == "NOT_FOUND" and isinstance(answer, dict)
                and answer.get("http_status") == 404 and not entry.get("quick_job_key")):
            decisions.append({**item, "kind": "already_gone", "reason": "already_gone",
                              "state": "already_gone",
                              "status": status, "status_code": 404,
                              "progress_url": url, "progress": entry.get("progress"),
                              "last_progress_at": entry.get("last_progress_at"),
                              "cancelled_at": stamp})
            continue
        if status in ORPHAN_RELEASE_STATUSES:
            decisions.append({**item, "kind": "terminal", "status": status, "alpha": alpha})
            continue
        progress_value = _quota_number(progress)
        if (stuck_after_min is not None and not entry.get("quick_job_key")
                and status not in {None, "NOT_FOUND"} and progress_value is not None):
            child_count = len(children) if isinstance(children, list) else 0
            progress_fingerprint = f"{status}:{progress_value:.12g}:{child_count}"
            previous = _quota_number(entry.get("last_progress_at"))
            prior_fingerprint = entry.get("progress_fingerprint")
            if prior_fingerprint == progress_fingerprint:
                last_progress_at = previous if previous is not None else stamp
            elif prior_fingerprint is None:
                last_progress_at = created if created is not None else stamp
                last_progress_at = min(last_progress_at, stamp)
            else:
                last_progress_at = stamp
            unchanged_min = max(0, stamp - last_progress_at) / 60.0
            observed = {
                "progress": progress_value,
                "progress_fingerprint": progress_fingerprint,
                "last_progress_at": last_progress_at,
                "progress_observed_at": stamp,
                "platform_status": status,
            }
            observations[key] = observed
            evidence = {
                "reason": "progress_unchanged",
                "minutes_unchanged": round(unchanged_min, 1),
                "threshold_min": stuck_after_min,
                "progress_url": url,
                "created_at": created,
                "stuck_at": stamp,
                **observed,
            }
            if unchanged_min >= stuck_after_min and cancel is not None:
                status_code = cancel(url)
                evidence.update(status_code=status_code, cancelled_at=stamp)
                if status_code in {200, 204}:
                    decisions.append({**item, "kind": "cancelled", "reason": "cancelled",
                                      "state": "cancelled",
                                      "status": status, "status_code": status_code,
                                      "progress": progress_value,
                                      "last_progress_at": last_progress_at,
                                      "minutes_unchanged": round(unchanged_min, 1),
                                      "cancelled_at": stamp, "evidence": evidence})
                    continue
                if status_code == 404:
                    decisions.append({**item, "kind": "already_gone", "reason": "already_gone",
                                      "state": "already_gone",
                                      "status": status, "status_code": 404,
                                      "progress": progress_value,
                                      "last_progress_at": last_progress_at,
                                      "minutes_unchanged": round(unchanged_min, 1),
                                      "cancelled_at": stamp, "evidence": evidence})
                    continue
                kept.append({**item, "status": status, "reason": "cancel_failed",
                             "status_code": status_code, "progress": progress_value,
                             "minutes_unchanged": round(unchanged_min, 1)})
                continue
            kept.append({**item, "status": status, "reason": "progress_unchanged",
                         "progress": progress_value, "minutes_unchanged": round(unchanged_min, 1)})
            continue
        kept.append({**item, "status": status,
                     "reason": "platform_pending" if status not in {None, "NOT_FOUND"} else "outcome_unknown"})

    released = []
    cancelled = []
    if decisions or observations:
        with cache_store.file_lock(lock, **lock_options()):
            journal = _load(path, fingerprint, research_dir=research_dir)
            reservations = journal.get("reservations") or {}
            for item in decisions:
                key = item["id"]
                if reservations.get(key) != snapshot[key]:
                    kept.append({**item, "reason": "reservation_changed"})
                    continue
                if item["kind"] in {"cancelled", "already_gone"}:
                    entry = reservations.pop(key)
                    evidence = item.get("evidence") or {
                        "reason": item["reason"], "progress_url": item.get("progress_url"),
                        "progress": item.get("progress"), "last_progress_at": item.get("last_progress_at"),
                        "status_code": item.get("status_code"), "cancelled_at": item.get("cancelled_at"),
                    }
                    entry.update(evidence, state=item["kind"], cancel_evidence=evidence)
                    journal.setdefault("cancelled", {})[key] = entry
                    cancelled.append({k: v for k, v in item.items() if k != "evidence"})
                    released.append({k: v for k, v in item.items() if k != "evidence"})
                    continue
                entry = reservations.pop(key)
                collection = "quick_resolved" if entry.get("quick_job_key") else "resolved"
                journal.setdefault(collection, {})[key] = {
                    **entry, "quick_owner": None, "resolved_at": stamp,
                    "terminal_result": {"status": item["status"], "alpha_id": item["alpha"]},
                }
                released.append(item)
            for key, observed in observations.items():
                if key in reservations and reservations.get(key) == snapshot.get(key):
                    reservations[key].update(observed)
            if decisions or observations:
                _save(path, journal, research_dir=research_dir)
    return {"released": released, "cancelled": cancelled, "kept": kept,
            "age_diagnostics": age_diagnostics}


def cancel_reservation_by_url(account, progress_url, *, state="cancelled", status_code=204):
    """Release accepted local slot reservations matching a cancelled simulation URL."""
    fingerprint = _fingerprint(account)
    if not fingerprint or not isinstance(progress_url, str) or not progress_url:
        return []
    path, lock = _paths(fingerprint)
    stamp = time.time()
    released_ids = []
    with cache_store.file_lock(lock):
        journal = _load(path, fingerprint)
        reservations = journal.get("reservations") or {}
        matching = [
            key for key, entry in reservations.items()
            if entry.get("progress_url") == progress_url and entry.get("state") == "accepted"
        ]
        for key in matching:
            entry = reservations.pop(key)
            evidence = {
                "reason": "explicit_cancel" if state == "cancelled" else state,
                "progress_url": progress_url,
                "progress": entry.get("progress"),
                "last_progress_at": entry.get("last_progress_at"),
                "status_code": status_code,
                "cancelled_at": stamp,
            }
            entry.update(evidence, state=state, quick_owner=None, cancel_evidence=evidence)
            journal.setdefault("cancelled", {})[key] = entry
            released_ids.append(key)
        if released_ids:
            _save(path, journal)
    return released_ids


def _active_limit(journal, limit):
    for entry in journal["reservations"].values():
        shared_limit = entry.get("quick_limit")
        if type(shared_limit) is int and shared_limit > 0:
            owner = entry.get("quick_owner")
            # A parked or abandoned reservation still occupies a slot. Only a
            # live worker's requested limit can constrain the workspace budget.
            if not _quick_owner_is_live(owner):
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


@lru_cache(maxsize=1)
def _win32_kernel():
    import ctypes
    from ctypes import wintypes

    class _FileTime(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_FileTime), ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime), ctypes.POINTER(_FileTime),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return ctypes, wintypes, _FileTime, kernel32


def _windows_process_stat(pid):
    """Query process liveness and creation time via Win32 without console signals."""
    if pid > 0xFFFFFFFF:
        return "X", None
    ctypes, wintypes, file_time_cls, kernel32 = _win32_kernel()
    # PROCESS_QUERY_LIMITED_INFORMATION (0x1000) inspects exit status and times
    # without sending CTRL_C_EVENT (which CPython's os.kill(pid, 0) triggers on Windows).
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return ("R", None) if ctypes.get_last_error() == 5 else ("X", None)
    try:
        exit_code = wintypes.DWORD()
        code_ok = bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)))
        creation = file_time_cls()
        exited = file_time_cls()
        kernel = file_time_cls()
        user = file_time_cls()
        times_ok = bool(kernel32.GetProcessTimes(
            handle, ctypes.byref(creation), ctypes.byref(exited),
            ctypes.byref(kernel), ctypes.byref(user),
        ))
        start = (str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
                 if times_ok else None)
        # STILL_ACTIVE is 259; also check exited FILETIME because an open parent
        # handle keeps the kernel process object addressable after termination.
        if (code_ok and exit_code.value != 259) or (
            times_ok and (exited.dwHighDateTime or exited.dwLowDateTime)
        ):
            return "Z", start
        return "R", start
    finally:
        kernel32.CloseHandle(handle)


def _process_stat(pid):
    if os.name == "nt":
        return _windows_process_stat(pid)
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
        if state in {"Z", "X"}:
            return False
        recorded_start = owner.get("start")
        return recorded_start is None or (start is not None and recorded_start == start)
    if os.name == "nt":
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (OverflowError, OSError):
        return False
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
            _save(path, journal)
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
                if identity_unverified:
                    entry["identity_unverified"] = True
            elif entry.get("state") == "quick_preparing" and not existing_url and limit == 0:
                # No POST has begun. A later zero budget revokes this owner's
                # preparation instead of allowing its old limit to authorize work.
                del reservations[key]
                _save(path, journal)
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
                single_limit = quick_single_slot_limit(limit)
                if single_limit < limit and sum(
                    bool(slot.get("quick_job_key")) for slot in reservations.values()
                ) >= single_limit:
                    return None
                phase = "quick_preparing" if prepare else "submitting"
                identity, _ = _reserve(journal, account, quick_slot_limit(limit), phase)
                if identity is None:
                    return None
                key = identity["id"]
                entry = reservations[key]
            entry.update(quick_job_key=job_key, quick_limit=limit, quick_owner=owner)
        _save(path, journal)
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
                _save(path, journal)
                return True
            return False
        if retain and (entry.get("state") != "accepted" or entry.get("manual_location")
                       or entry.get("identity_unverified")
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
        _save(path, journal)
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
        and not entry.get("manual_location") and not entry.get("identity_unverified")
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
            _save(path, journal)
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
        parked_unverified = (terminal_failure and entry.get("state") == "accepted"
                             and (entry.get("manual_location") or entry.get("identity_unverified"))
                             and entry.get("progress_url") == progress_url)
        if not (new_location or parked_unverified) or _quick_owner_is_live(entry.get("quick_owner")):
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
        _save(path, journal)
        return {"id": key, **entry}


def quick_job_snapshot(account):
    """Read durable Quick identities and POST evidence without changing the journal.

    This snapshot is a scheduling hint. ``quick_acquire`` rechecks current
    ownership and state under its lock before authorizing any work.
    """
    with _locked_quick_journal(account) as (journal, _):
        return {
            entry["quick_job_key"]: {"id": identity, **entry}
            for collection in (journal["reservations"], journal.get("quick_resolved", {}))
            for identity, entry in collection.items()
            if entry.get("quick_job_key")
        }


def quick_known_jobs(account):
    """Return durable Quick job keys without changing the shared journal."""
    return set(quick_job_snapshot(account))


def _record_unposted(journal, identity):
    entry = journal["reservations"].pop(identity)
    # Keep proof for a checkpoint that was saved before preparation stopped.
    journal.setdefault("not_submitted", {})[identity] = entry.get("created_at")


def prepare_submission(account, limit, checkpoint, *, quick_native=False):
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
            native_available = True
            if quick_native:
                native_limit = _configured_slot_limit(limit, "quick-native-slot-limit")
                native_active = sum(
                    bool(entry.get("quick_native")) for entry in journal["reservations"].values()
                )
                native_available = native_active < native_limit
            if native_available:
                identity, state = _reserve(journal, account, limit, "preparing")
                if identity and quick_native:
                    journal["reservations"][identity["id"]]["quick_native"] = True
            else:
                identity, state = None, _report(account, limit, len(journal["reservations"]))
                state.update(available=0, lane="quick_native")
            if identity or abandoned or pruned:
                _save(path, journal)
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
                _save(path, journal)
        return identity, state


def _reservation_identity(record, account, research_dir=None):
    identity = record.get("slot_reservation")
    if identity:
        workspace = os.path.normcase(os.path.abspath(research_dir)) if research_dir is not None else _workspace()
        if identity.get("workspace") != workspace:
            raise ValueError("simulation reservation belongs to another research workspace")
        fingerprint = identity.get("account_fingerprint")
        if not fingerprint or not identity.get("id"):
            raise ValueError("simulation reservation identity is incomplete")
        if account is not None and _fingerprint(account) != fingerprint:
            raise ValueError("simulation reservation belongs to another or unidentified account")
    return identity


def validate_reservation_owner(record, account, *, research_dir=None):
    """Verify a saved reservation belongs to the current account and workspace."""
    _reservation_identity(record, account, research_dir)


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


def restore_record(record, account=None, *, research_dir=None):
    """Restore a missing progress URL or a definitely unposted attempt.

    Return whether the in-memory record changed. Local inspection can read
    completed copies across workspaces. Pass the authenticated account before
    querying BRAIN for an unresolved attempt.
    """
    terminal = _has_terminal_result(record)
    if terminal and account is None:
        return False
    identity = _reservation_identity(record, account, research_dir)
    if not identity or terminal:
        return False
    fingerprint = identity["account_fingerprint"]
    path, lock = _paths(fingerprint, research_dir)
    # Writers replace the complete journal atomically.
    journal = _load(path, fingerprint, research_dir=research_dir)
    attempt = identity["id"]
    resolved = journal.get("resolved", {}).get(attempt)
    if resolved and _restore_terminal(record, resolved):
        return True
    stuck = journal.get("stuck", {}).get(attempt)
    if stuck:
        record["slot_reservation_state"] = "stuck"
        record["slot_stuck_evidence"] = stuck.get("stuck_evidence") or stuck
        return True
    cancelled = journal.get("cancelled", {}).get(attempt)
    if cancelled:
        record["slot_reservation_state"] = cancelled.get("state", "cancelled")
        record["slot_cancel_evidence"] = cancelled.get("cancel_evidence") or cancelled
        return True
    if record.get("progress_url"):
        return False
    entry = journal["reservations"].get(attempt, {})
    if attempt in journal.get("not_submitted", {}):
        return _restore_unposted(record)
    if entry.get("state") == "preparing":
        try:
            with cache_store.file_lock(lock + ".prepare", timeout=0, poll=0):
                journal = _load(path, fingerprint, research_dir=research_dir)
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
                stuck = journal.get("stuck", {}).get(identity["id"])
                if stuck:
                    record["slot_reservation_state"] = "stuck"
                    record["slot_stuck_evidence"] = stuck.get("stuck_evidence") or stuck
                    return record
                cancelled = journal.get("cancelled", {}).get(identity["id"])
                if cancelled:
                    record["slot_reservation_state"] = cancelled.get("state", "cancelled")
                    record["slot_cancel_evidence"] = cancelled.get("cancel_evidence") or cancelled
                    return record
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
                _save(path, journal)
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
                _save(path, journal)
        except (OSError, cache_store.CacheError):
            # The caller still honors the observed delay if it cannot be shared.
            pass
    return retry_at


def _quota_reset_at(observation):
    observed = _quota_number(observation.get("observed_at"))
    delay = _quota_number(observation.get("reset_seconds"))
    return observed + delay if observed is not None and delay is not None else None


def _merge_quota(previous, incoming):
    """Select one authoritative balance without resurrecting stale quota."""
    if not isinstance(previous, dict):
        return incoming
    past = _quota_number(previous.get("observed_at"))
    now = incoming["observed_at"]
    if past is None:
        return incoming
    previous_header = _quota_number(previous.get("header_at"))
    incoming_header = incoming.get("header_at")
    old_reset, new_reset = _quota_reset_at(previous), _quota_reset_at(incoming)
    old_remaining = _quota_number(previous.get("remaining"), integer=True)
    remaining = incoming["remaining"]
    if now < past or (previous_header is not None and incoming_header is not None
                      and incoming_header < previous_header):
        # A slower writer can still bring a lower balance for the same window.
        # Retain the journal's newer timestamp and its established boundary.
        if (old_reset is not None and new_reset is not None and abs(old_reset - new_reset) <= 2
                and remaining is not None and (old_remaining is None or remaining < old_remaining)):
            return {**previous, "remaining": remaining}
        return previous
    if old_reset is None and new_reset is not None and new_reset > now and remaining is not None:
        # A header without a reset cannot establish a durable window. Use the
        # first complete observation when one arrives.
        return incoming
    # Reaching the old deadline is insufficient. A subsequent window must
    # provide both a fresh balance and a new future reset boundary.
    if old_reset is not None and now >= old_reset:
        issued = incoming.get("header_at", now)
        if remaining is not None and issued >= old_reset and new_reset is not None and new_reset > now:
            return incoming
        return previous
    if remaining is None or (old_remaining is not None and remaining > old_remaining):
        return previous
    if incoming["limit"] is None:
        incoming = {**incoming, "limit": previous.get("limit")}
    if old_reset is not None:
        # Keep the established boundary despite header/transport jitter.
        incoming = {**incoming, "reset_seconds": old_reset - now}
    return incoming


def observe_quota(account, headers):
    """Publish response quota through the account journal's exclusive writer lock."""
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
    now = time.time()
    observation.update(observed_at=now, source="POST /simulations response headers")
    if headers.get("date"):
        try:
            issued = parsedate_to_datetime(headers["date"]).timestamp()
            if issued <= now + 60:
                observation["header_at"] = issued
                if observation["reset_seconds"] is not None:
                    observation["reset_seconds"] = max(0.0, issued + observation["reset_seconds"] - now)
        except (AttributeError, TypeError, ValueError, OverflowError, OSError):
            pass
    fingerprint = _fingerprint(account)
    if fingerprint:
        path, lock = _paths(fingerprint)
        try:
            with cache_store.file_lock(lock):
                journal = _load(path, fingerprint)
                previous = journal.get("quota")
                observation = _merge_quota(previous, observation)
                if observation != previous:
                    journal["quota"] = observation
                    _save(path, journal)
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
    reset_at = _quota_reset_at(observation)
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
