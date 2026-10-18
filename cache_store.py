#!/usr/bin/env python3
"""Shared on-disk cache plumbing: atomic writes, TTL envelopes, file locks.

Standard library only, so the CLI, the research scripts and the offline tests
can all import it without pulling in ``requests``.

Three properties matter here, each one learned from a real failure:

* **Atomic replacement.** Research runs do get killed mid-write (deadlines,
  killed background tasks, a dropped executor). ``Path.write_text`` can leave a
  truncated JSON document on disk; writing a temporary file in the same
  directory and calling ``os.replace`` cannot -- a reader sees either the old
  file or the new one, never half of each.
* **An honest staleness signal.** Every entry records when it was created and
  when it expires, so a caller can tell *fresh*, *stale but usable* and *absent*
  apart instead of inferring meaning from a bare mtime.
* **Single-flight writes.** Concurrent CLI invocations used to race each other:
  both missed the session cache, both logged in, and BRAIN answered the second
  login with a captcha challenge. :func:`file_lock` makes the loser wait for the
  winner and then reuse its result.

Nothing here decides *what* to cache or for how long; callers pass a TTL.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone

try:  # POSIX advisory locks; absent on Windows.
    import fcntl
except ImportError:  # pragma: no cover - exercised only off Linux/macOS
    fcntl = None

__all__ = [
    "CACHE_VERSION",
    "LockTimeout",
    "CacheError",
    "build_entry",
    "cache_stats",
    "clear_dir",
    "file_lock",
    "load_json",
    "prune_corrupt",
    "quarantine",
    "read_entry",
    "remove",
    "utc_now",
    "write_json_atomic",
]

#: Bumped when the on-disk envelope changes shape. A mismatched version is
#: treated as a cache miss (and rewritten), never as an error.
CACHE_VERSION = 1


class CacheError(Exception):
    """A cache file exists but cannot be used."""


class LockTimeout(CacheError):
    """Another process held the lock for longer than the caller allowed."""


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def utc_now():
    """Timezone-aware ``datetime`` for cache timestamps."""
    return datetime.now(timezone.utc)


def _to_iso(moment):
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _from_iso(text):
    if not text:
        return None
    cleaned = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        moment = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


# ---------------------------------------------------------------------------
# Atomic JSON I/O
# ---------------------------------------------------------------------------
def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return parent


def write_json_atomic(path, payload, mode=0o600, indent=None):
    """Serialise ``payload`` to ``path`` atomically.

    The temporary file is created in the *same directory* so ``os.replace``
    stays a rename on one filesystem. ``mode`` defaults to ``0o600`` because
    cache files can carry session cookies; widening it is an explicit choice.

    Returns the path written.
    """
    path = os.path.abspath(path)
    parent = _ensure_parent(path)
    data = json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=False)
    fd, tmp_path = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".",
                                    suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(tmp_path, mode)
        except OSError:  # pragma: no cover - exotic filesystems
            pass
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    return path


def load_json(path, default=None):
    """Read JSON, returning ``(data, error)`` instead of raising.

    A caller that only cares about the happy path can use
    ``data = load_json(path)[0] or default``. ``error`` is a short string when
    the file is missing, unreadable or malformed; the file itself is left alone
    (see :func:`quarantine` for the explicit "move it aside" step).
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), None
    except FileNotFoundError:
        return default, "missing"
    except PermissionError as exc:
        return default, "permission denied: %s" % exc
    except json.JSONDecodeError as exc:
        return default, "malformed json: %s" % exc
    except OSError as exc:
        return default, "unreadable: %s" % exc


def remove(path):
    """Delete ``path`` if present. Returns True when something was removed."""
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# TTL envelope
# ---------------------------------------------------------------------------
def build_entry(kind, payload, ttl=None, source=None, now=None):
    """Wrap ``payload`` in the cache envelope.

    ``ttl`` is seconds from now, or ``None`` for an entry that never expires on
    its own (session validity still comes from BRAIN, not from us).
    """
    created = now or utc_now()
    expires = None if ttl is None else created + timedelta(seconds=float(ttl))
    return {
        "_cache": {
            "version": CACHE_VERSION,
            "kind": kind,
            "created_at": _to_iso(created),
            "expires_at": None if expires is None else _to_iso(expires),
            "source": source,
        },
        "payload": payload,
    }


def read_entry(path, now=None):
    """Read a cache entry and describe it.

    Returns a dict with:

    ``payload``       the stored payload, or ``None``
    ``present``       a file exists and parsed
    ``fresh``         present, right version, and not past ``expires_at``
    ``age_seconds``   how long ago the entry was created (``None`` if unknown)
    ``expires_at``    ISO timestamp or ``None``
    ``expires_in``    seconds until expiry (negative when already stale)
    ``kind``/``version``/``source``  envelope metadata
    ``error``         why the entry is unusable, when it is

    A wrong envelope version is reported as ``error="version mismatch"`` rather
    than raising, so an upgrade degrades into a cache miss.
    """
    data, error = load_json(path)
    result = {
        "path": os.path.abspath(path),
        "payload": None,
        "present": False,
        "fresh": False,
        "age_seconds": None,
        "created_at": None,
        "expires_at": None,
        "expires_in": None,
        "kind": None,
        "version": None,
        "source": None,
        "error": error,
    }
    if data is None:
        return result

    if not isinstance(data, dict) or "_cache" not in data or "payload" not in data:
        result["error"] = "not a cache envelope"
        return result

    meta = data.get("_cache") or {}
    version = meta.get("version")
    result.update({
        "payload": data.get("payload"),
        "version": version,
        "kind": meta.get("kind"),
        "source": meta.get("source"),
        "created_at": meta.get("created_at"),
        "expires_at": meta.get("expires_at"),
    })
    if version != CACHE_VERSION:
        result["error"] = "version mismatch: found %r, want %r" % (
            version, CACHE_VERSION)
        return result

    result["present"] = True
    moment = now or utc_now()
    created = _from_iso(meta.get("created_at"))
    if created is not None:
        result["age_seconds"] = max(0.0, (moment - created).total_seconds())
    expires = _from_iso(meta.get("expires_at"))
    if expires is None:
        result["fresh"] = True
    else:
        remaining = (expires - moment).total_seconds()
        result["expires_in"] = remaining
        result["fresh"] = remaining > 0
    return result


# ---------------------------------------------------------------------------
# Corrupt-file handling
# ---------------------------------------------------------------------------
def quarantine(path, keep=3):
    """Move a corrupt file aside instead of silently deleting it.

    Silent deletion hides a bug that will simply happen again; keeping one or
    two samples around is what made the earlier truncated-session bug
    diagnosable. Old samples beyond ``keep`` are pruned.
    """
    if not os.path.exists(path):
        return None
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    target = "%s.corrupt-%s" % (path, stamp)
    try:
        os.replace(path, target)
    except OSError:
        return None
    prune_corrupt(path, keep=keep)
    return target


def prune_corrupt(path, keep=3):
    """Keep only the newest ``keep`` corrupt samples for ``path``."""
    parent = os.path.dirname(os.path.abspath(path)) or "."
    prefix = os.path.basename(path) + ".corrupt-"
    try:
        names = [name for name in os.listdir(parent) if name.startswith(prefix)]
    except OSError:
        return []
    names.sort(reverse=True)
    removed = []
    for name in names[keep:]:
        if remove(os.path.join(parent, name)):
            removed.append(name)
    return removed


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def file_lock(path, timeout=60.0, poll=0.1):
    """Hold an exclusive advisory lock on ``path`` for the duration of the block.

    Used to make login single-flight: the first process authenticates and writes
    the session cache, everyone else blocks and then finds a valid cache instead
    of logging in again (repeat logins are what trigger BRAIN's captcha).

    On POSIX this is ``flock``, which the kernel releases when the process dies,
    so a killed run cannot leave a permanent lock. Raises :class:`LockTimeout`
    when the deadline passes; callers decide whether that is fatal.
    """
    path = os.path.abspath(path)
    _ensure_parent(path)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + float(timeout)
    acquired = False
    try:
        if fcntl is None:  # pragma: no cover - fallback path
            while True:
                try:
                    lock_fd = os.open(path + ".pid", os.O_CREAT | os.O_EXCL, 0o600)
                    os.close(lock_fd)
                    acquired = True
                    break
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise
                    if time.monotonic() > deadline:
                        raise LockTimeout(
                            "timed out after %.1fs waiting for %s" % (timeout, path))
                    time.sleep(poll)
        else:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    if time.monotonic() > deadline:
                        raise LockTimeout(
                            "timed out after %.1fs waiting for %s" % (timeout, path))
                    time.sleep(poll)
        try:
            os.ftruncate(fd, 0)
            os.write(fd, ("%d %s\n" % (os.getpid(), _to_iso(utc_now()))).encode())
        except OSError:
            pass
        yield
    finally:
        if acquired and fcntl is None:  # pragma: no cover
            with contextlib.suppress(OSError):
                os.unlink(path + ".pid")
        if fcntl is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)


# ---------------------------------------------------------------------------
# Introspection / maintenance
# ---------------------------------------------------------------------------
def walk_files(root):
    """Every regular file under ``root``, recursively and deterministically."""
    for current, _dirs, names in os.walk(root):
        for name in sorted(names):
            yield os.path.join(current, name)


def cache_stats(root):
    """Summarise a cache tree: file count, bytes, corrupt samples, oldest/newest."""
    root = os.path.abspath(root)
    stats = {
        "root": root,
        "exists": os.path.isdir(root),
        "files": 0,
        "bytes": 0,
        "corrupt_files": 0,
        "newest_mtime": None,
        "oldest_mtime": None,
        "by_kind": {},
    }
    if not stats["exists"]:
        return stats
    for path in walk_files(root):
        try:
            info = os.stat(path)
        except OSError:
            continue
        stats["files"] += 1
        stats["bytes"] += info.st_size
        name = os.path.basename(path)
        if ".corrupt-" in name:
            stats["corrupt_files"] += 1
        rel = os.path.relpath(os.path.dirname(path), root)
        bucket = "." if rel == "." else rel.split(os.sep)[0]
        entry = stats["by_kind"].setdefault(bucket, {"files": 0, "bytes": 0})
        entry["files"] += 1
        entry["bytes"] += info.st_size
        mtime = info.st_mtime
        if stats["newest_mtime"] is None or mtime > stats["newest_mtime"]:
            stats["newest_mtime"] = mtime
        if stats["oldest_mtime"] is None or mtime < stats["oldest_mtime"]:
            stats["oldest_mtime"] = mtime
    for key in ("newest_mtime", "oldest_mtime"):
        if stats[key] is not None:
            stats[key] = _to_iso(datetime.fromtimestamp(stats[key], timezone.utc))
    return stats


def clear_dir(root, keep_locks=False):
    """Remove every file under ``root``. Returns the number of files removed.

    Lock files are kept unless ``keep_locks`` is false, because deleting a lock
    another process is holding would let two writers in at once.
    """
    removed = 0
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return 0
    for path in list(walk_files(root)):
        if keep_locks and os.path.basename(path).endswith(".lock"):
            continue
        if remove(path):
            removed += 1
    for current, dirs, _names in os.walk(root, topdown=False):
        for name in dirs:
            with contextlib.suppress(OSError):
                os.rmdir(os.path.join(current, name))
    return removed