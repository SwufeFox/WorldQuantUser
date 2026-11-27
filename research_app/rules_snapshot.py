"""Rule snapshots: what the platform enforced, when, and for which scope.

Thresholds, quotas and theme multipliers move.  A rule remembered from a forum
post is not evidence about today's account, so every value here carries where it
came from, when it was observed and which scope it applies to.  A rule that was
never captured stays absent: callers must treat a missing snapshot as unknown,
never as a pass.

Sources are deliberately few and explicit:

* ``platform_check`` -- a value read from ``GET /alphas/{id}/check``;
* ``official_doc`` -- a statement read from a BRAIN documentation page;
* ``measured`` -- an inference from repeated platform observations, with the
  sample size that produced it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import cache_store
from check_evidence import CLIENT_PARSE_ERROR, normalize_checks
import research_paths

#: Names that describe a quota rather than a test.
QUOTA_SUFFIX = "_SUBMISSION"
SNAPSHOT_FILENAME = "rules-snapshot.jsonl"


def _now_iso():
    return cache_store._to_iso(cache_store.utc_now())


def _timestamp(value):
    moment = cache_store._from_iso(value)
    return None if moment is None else moment.timestamp()


def default_snapshot_path():
    """Default JSONL path under the active research directory."""
    return Path(research_paths.research_dir()) / SNAPSHOT_FILENAME


def extract_check_rules(check, *, scope, observed_at=None):
    """Turn a platform check payload into rule records with provenance.

    ``check`` is either the list the endpoint returns or a name -> record map.
    Only values the payload actually carries are recorded; a missing limit is
    omitted rather than filled with a remembered default.
    """
    records = []
    observed = observed_at or _now_iso()
    if isinstance(check, list):
        items = check
    else:
        normalized = normalize_checks(check)
        items = [
            dict(record, name=name)
            for name, record in normalized.items()
            if name != CLIENT_PARSE_ERROR
            and not (record.get("source") == "client_parser" and record.get("client_parse_error"))
        ]
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        base = {
            "name": name,
            "result": item.get("result"),
            "source": "platform_check",
            "observed_at": observed,
            "scope": dict(scope or {}),
        }
        kind = "quota" if name.endswith(QUOTA_SUFFIX) else "check"
        record = dict(base, kind=kind)
        for field in ("value", "limit"):
            if item.get(field) is not None:
                record[field] = item[field]
        if name == "MATCHES_THEMES":
            record["kind"] = "themes"
            record["themes"] = [
                {"id": t.get("id"), "name": t.get("name"), "multiplier": t.get("multiplier")}
                for t in item.get("themes") or [] if isinstance(t, dict)]
        records.append(record)
    return records


def official_rule(name, value, *, source_id, source_url, scope=None, statement=None,
                  observed_at=None):
    """One rule taken verbatim from an official page."""
    return {
        "name": name,
        "value": value,
        "kind": "official_rule",
        "statement": statement,
        "source": "official_doc",
        "source_id": source_id,
        "source_url": source_url,
        "observed_at": observed_at or _now_iso(),
        "scope": dict(scope or {}),
    }


def measured_rule(name, value, *, sample_size, scope=None, note=None, observed_at=None):
    """A rule inferred from repeated observations, with the sample behind it."""
    return {
        "name": name,
        "value": value,
        "kind": "measured_rule",
        "source": "measured",
        "sample_size": sample_size,
        "note": note,
        "observed_at": observed_at or _now_iso(),
        "scope": dict(scope or {}),
    }


def append(path, records):
    """Append rule dictionaries atomically under a file lock."""
    if isinstance(records, dict):
        records = [records]
    valid = [record for record in (records or ()) if isinstance(record, dict)]
    target = os.path.abspath(os.fspath(path))
    if not valid:
        return {"path": target, "appended": 0}
    with cache_store.file_lock(target + ".lock"):
        existing = ""
        if os.path.exists(target):
            with open(target, encoding="utf-8") as handle:
                existing = handle.read()
        prefix = existing if not existing or existing.endswith("\n") else existing + "\n"
        delta = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                        for record in valid)
        cache_store.write_text_atomic(target, prefix + delta)
    return {"path": target, "appended": len(valid)}


def record_observed_checks(check, *, scope, path=None, observed_at=None):
    """Persist changed check/quota/theme observations to the workspace snapshot."""
    candidates = extract_check_rules(check, scope=scope, observed_at=observed_at)
    target = os.path.abspath(os.fspath(path or default_snapshot_path()))
    if not candidates:
        return {"path": target, "appended": 0}
    comparable = lambda row: {k: v for k, v in row.items() if k != "observed_at"}
    with cache_store.file_lock(target + ".lock"):
        prior = load(target)
        latest = {}
        for row in prior:
            key = (row.get("name"), json.dumps(row.get("scope") or {}, sort_keys=True))
            latest[key] = comparable(row)
        changed = []
        for record in candidates:
            key = (record.get("name"), json.dumps(record.get("scope") or {}, sort_keys=True))
            body = comparable(record)
            if latest.get(key) != body:
                latest[key] = body
                changed.append(record)
        if not changed:
            return {"path": target, "appended": 0}
        existing = ""
        if os.path.exists(target):
            with open(target, encoding="utf-8") as handle:
                existing = handle.read()
        prefix = existing if not existing or existing.endswith("\n") else existing + "\n"
        delta = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                        for record in changed)
        cache_store.write_text_atomic(target, prefix + delta)
    return {"path": target, "appended": len(changed)}


def load(path):
    """Load all valid JSON rule snapshot records from ``path``, skipping blank or corrupt lines."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def _matches(record_scope, wanted):
    if not isinstance(record_scope, dict):
        record_scope = {}
    for key, value in (wanted or {}).items():
        if value is None:
            continue
        if record_scope.get(key) != value:
            return False
    return True


def lookup(records, name, scope=None, *, max_age_seconds=None, now=None):
    """Newest record for a rule and scope, with its UTC age and staleness.

    Returns ``None`` when nothing matches: absence is information the caller has
    to handle, not a default value.
    """
    matches = [r for r in (records or ())
               if isinstance(r, dict) and r.get("name") == name
               and _matches(r.get("scope"), scope)]
    if not matches:
        return None

    def rank(record):
        ts = _timestamp(record.get("observed_at"))
        return (ts is not None, ts if ts is not None else float("-inf"))

    newest = max(matches, key=rank)
    newest_ts = _timestamp(newest.get("observed_at"))
    current = time.time() if now is None else float(now)
    age = None if newest_ts is None else max(0.0, current - newest_ts)
    stale = None if (max_age_seconds is None or age is None) else age > max_age_seconds
    return {"rule": newest, "age_seconds": age, "stale": stale,
            "alternatives": len(matches) - 1}
