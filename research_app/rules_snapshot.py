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
import time

#: Names that describe a quota rather than a test.
QUOTA_SUFFIX = "_SUBMISSION"


def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def extract_check_rules(check, *, scope, observed_at=None):
    """Turn a platform check payload into rule records with provenance.

    ``check`` is either the list the endpoint returns or a name -> record map.
    Only values the payload actually carries are recorded; a missing limit is
    omitted rather than filled with a remembered default.
    """
    records = []
    observed = observed_at or _now_iso()
    items = check if isinstance(check, list) else [
        dict(record or {}, name=name) for name, record in (check or {}).items()]
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
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
    """Append records as JSON lines; the file is created when absent."""
    if isinstance(records, dict):
        records = [records]
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {"path": path, "appended": len(records)}


def load(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _matches(record_scope, wanted):
    for key, value in (wanted or {}).items():
        if value is None:
            continue
        if (record_scope or {}).get(key) != value:
            return False
    return True


def lookup(records, name, scope=None, *, max_age_seconds=None, now=None):
    """Newest record for a rule and scope, with its age and staleness.

    Returns ``None`` when nothing matches: absence is information the caller has
    to handle, not a default value.
    """
    matches = [r for r in records
               if r.get("name") == name and _matches(r.get("scope"), scope)]
    if not matches:
        return None
    def stamp(record):
        try:
            return time.mktime(time.strptime(record.get("observed_at"), "%Y-%m-%dT%H:%M:%SZ"))
        except (TypeError, ValueError):
            return 0.0
    newest = max(matches, key=stamp)
    age = None
    if newest.get("observed_at"):
        age = max(0.0, (now or time.time()) - stamp(newest))
    stale = None if (max_age_seconds is None or age is None) else age > max_age_seconds
    return {"rule": newest, "age_seconds": age, "stale": stale,
            "alternatives": len(matches) - 1}