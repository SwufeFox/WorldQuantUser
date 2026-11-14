"""Immutable training choices and identity-preserving validation joins."""

import hashlib
import json
import os
from datetime import datetime, timezone

import cache_store

FORMAT = "research.train-selection"
VERSION = 1
IDENTITY_FIELDS = ("alpha_id", "source_file", "label", "candidate_id")


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("selection must contain finite JSON values") from exc


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def row_identity(row):
    """Return a canonical identity; source metadata distinguishes repeated IDs."""
    if not isinstance(row, dict) or not isinstance(row.get("alpha_id"), str) or not row["alpha_id"].strip() or row["alpha_id"] == "None":
        raise ValueError("selection rows require a non-empty alpha_id")
    identity = {key: row[key] for key in IDENTITY_FIELDS if row.get(key) is not None}
    for key, value in identity.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError("selection identity %s must be non-empty text" % key)
    return _canonical(identity)


def _identities(rows):
    if not isinstance(rows, list):
        raise ValueError("selection rows must be a list")
    identities = [row_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("ambiguous duplicate selection row identity")
    return identities


def _validate(manifest, source=None):
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise ValueError("unsupported selection manifest")
    if not isinstance(manifest.get("source"), str) or not manifest["source"]:
        raise ValueError("selection source is missing")
    if source is not None and manifest["source"] not in ((source,) if isinstance(source, str) else source):
        raise ValueError("selection source does not match %s" % source)
    for key in ("criteria", "inputs", "window"):
        if not isinstance(manifest.get(key), dict):
            raise ValueError("selection %s must be an object" % key)
    identities = _identities(manifest.get("rows"))
    if identities != manifest.get("identities"):
        raise ValueError("selection identities or order were modified")
    try:
        created = datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))
        if created.tzinfo is None:
            raise ValueError
    except (KeyError, TypeError, AttributeError, ValueError):
        raise ValueError("selection creation time is invalid") from None
    content = {key: value for key, value in manifest.items()
               if key not in ("selection_id", "created_at", "integrity_hash")}
    if manifest.get("selection_id") != _digest(content):
        raise ValueError("selection content hash mismatch")
    signed = {key: value for key, value in manifest.items() if key != "integrity_hash"}
    if manifest.get("integrity_hash") != _digest(signed):
        raise ValueError("selection integrity hash mismatch")
    return manifest


def freeze(path, *, source, rows, criteria=None, inputs=None, window=None):
    """Atomically persist a choice, allowing only identical idempotent writes."""
    content = {"format": FORMAT, "version": VERSION, "source": source,
               "rows": rows, "identities": _identities(rows),
               "criteria": criteria or {}, "inputs": inputs or {}, "window": window or {}}
    # Detach caller-owned values before hashing and writing.
    manifest = json.loads(_canonical(content))
    manifest["selection_id"] = _digest(content)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["integrity_hash"] = _digest(manifest)
    _validate(manifest)
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    with cache_store.file_lock(path + ".lock"):
        if os.path.lexists(path):
            previous = load(path)
            if previous["selection_id"] != manifest["selection_id"]:
                raise ValueError("selection is immutable; choose a new path")
            return previous
        cache_store.write_json_atomic(path, manifest, indent=2)
    return manifest


def load(path, source=None):
    """Read and verify every identity, ordered row and manifest content hash."""
    try:
        with open(os.path.expanduser(os.fspath(path)), encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read selection: %s" % exc) from exc
    return _validate(manifest, source)


def ordered_rows(manifest, rows):
    """Join current evidence to the frozen order; never fill absent metrics."""
    _validate(manifest)
    wanted = set(manifest["identities"])
    wanted_ids = {row["alpha_id"] for row in manifest["rows"]}
    current = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("alpha_id") not in wanted_ids:
            continue
        identity = row_identity(row)
        if identity not in wanted:
            continue
        if identity in current:
            raise ValueError("ambiguous duplicate validation row identity")
        current[identity] = row
    ordered = []
    for identity in manifest["identities"]:
        row = current.get(identity)
        if row is None:
            row = {**json.loads(identity), "missing": True,
                   "evidence_state": "missing", "missing_reason": "frozen candidate absent from current evidence"}
        ordered.append(dict(row))
    return ordered
