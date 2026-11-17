"""Two-step confirmation for irreversible actions.

Submitting an alpha cannot be undone, so one CLI invocation must never be enough.
`require_double_call` stages the intent on the first call and lets a second call
through only inside a short window; the stage is consumed as it is used.

The stage lives in the research directory so separate CLI invocations can share
it. Consumption and restaging use one lock. Expired records and records with
different caller context cannot confirm a submission.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cache_store
import research_paths

from .runtime import CliError

TTL_SECONDS = 600
FILENAME = "submit-confirm.json"


def stage_path(research_dir=None) -> Path:
    root = Path(research_dir or research_paths.research_dir())
    return root / FILENAME


def require_double_call(kind: str, key: str, *, now: float | None = None,
                        ttl: int = TTL_SECONDS, research_dir=None, context=None) -> bool:
    """Return True only on the confirming (second) call.

    Raises CONFIRM_REQUIRED on the staging call. Callers may already have run
    read-only checks, but must acquire confirmation before sending a submission.
    """
    path = stage_path(research_dir)
    with cache_store.file_lock(path.with_suffix(".lock")):
        moment = time.time() if now is None else now
        data, error = cache_store.load_json(path, default={})
        if error not in (None, "missing") or not isinstance(data, dict):
            raise CliError("INVALID_INPUT", "submission confirmation file is unreadable",
                           "inspect the local confirmation state before retrying")
        bucket = data.setdefault(kind, {})
        if not isinstance(bucket, dict):
            raise CliError("INVALID_INPUT", "submission confirmation group must be an object")
        entry = bucket.get(key)
        staged_at = entry.get("at") if isinstance(entry, dict) else None
        valid = (type(staged_at) in (int, float) and math.isfinite(staged_at)
                 and 0 <= moment - staged_at <= ttl and entry.get("context") == context)
        if valid:
            del bucket[key]
            if not bucket:
                del data[kind]
        else:
            bucket[key] = {"at": moment, "context": context}
        cache_store.write_json_atomic(path, data)
    if valid:
        return True
    raise CliError(
        "CONFIRM_REQUIRED",
        "%s is staged but not performed: run the same command once more within %d minutes" % (key, ttl // 60),
        "repeat the matching request within the confirmation window; no submission was sent",
    )
