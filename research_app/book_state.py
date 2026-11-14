"""Explicit local membership, platform snapshots and PnL coverage evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import alpha_registry
import cache_store
import machine_lib as ml
import research_paths

BOOK_STATUSES = frozenset({"ACTIVE", "SUBMITTED"})


def member_digest(ids):
    """Digest identities, so replacing a member changes evidence at equal count."""
    return hashlib.sha256(json.dumps(sorted(set(ids)), separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def snapshot_path(registry_path=None):
    registry = registry_path or alpha_registry.default_registry_path()
    return Path(str(registry) + ".platform.json")


def persist_snapshot(snapshot, registry_path=None):
    path = snapshot_path(registry_path)
    with cache_store.file_lock(str(path) + ".lock"):
        cache_store.write_json_atomic(path, snapshot)


def invalidate_snapshot(alpha_id, registry_path=None):
    """A confirmed individual submission cannot certify the rest of the book."""
    path = snapshot_path(registry_path)
    with cache_store.file_lock(str(path) + ".lock"):
        snapshot, error = cache_store.load_json(path)
        if error and error != "missing":
            raise ValueError("cannot invalidate platform snapshot: " + error)
        snapshot = snapshot or {"complete": False, "as_of": None, "scope": None}
        if not isinstance(snapshot, dict):
            raise ValueError("invalid platform snapshot")
        snapshot["invalidated"] = True
        snapshot["invalidation_reason"] = "confirmed submission requires a new account capture"
        snapshot["confirmed_alpha_ids"] = sorted(
            set(snapshot.get("confirmed_alpha_ids") or []) | {alpha_id})
        cache_store.write_json_atomic(path, snapshot)


def describe_book(entries, series=None, registry_path=None, pnl_dir=None):
    """Describe evidence without promoting a local ledger into platform truth."""
    latest = {}
    for entry in entries.values() if isinstance(entries, dict) else entries:
        if entry.get("alpha_id"):
            latest[entry["alpha_id"]] = entry
    ids = sorted(alpha_id for alpha_id, entry in latest.items()
                 if str(entry.get("status") or "").upper() in BOOK_STATUSES)
    digest = member_digest(ids)
    snapshot, error = cache_store.load_json(snapshot_path(registry_path))
    platform = dict(snapshot) if isinstance(snapshot, dict) else {}
    platform["available"] = isinstance(snapshot, dict) and not error
    platform["complete"] = bool(platform.get("complete"))
    platform["matches_local_members"] = (
        platform.get("member_digest") == digest if platform["available"] else None)
    platform["current_capture"] = bool(platform["complete"]
                                       and not platform.get("invalidated"))
    platform["freshness"] = "as_of_only" if platform["available"] else "unknown"
    platform["live_verified"] = False
    if error and error != "missing":
        platform["error"] = error
    metadata_origin_known = series is None or pnl_dir is not None
    pnl_dir = Path(pnl_dir or research_paths.resolve("pnl"))
    available, missing, complete, unknown, incomplete = [], [], [], [], []
    coverage = {}
    for alpha_id in ids:
        if series is None:
            values, _ = cache_store.load_json(pnl_dir / (alpha_id + ".json"))
        else:
            values = series.get(alpha_id)
        if not values:
            missing.append(alpha_id)
            continue
        available.append(alpha_id)
        evidence = (ml.pnl_cache_status(pnl_dir, alpha_id, values)
                    if metadata_origin_known else
                    {"complete": None, "error": "series origin is not supplied"})
        coverage[alpha_id] = evidence
        if evidence.get("complete") is True:
            complete.append(alpha_id)
        elif evidence.get("complete") is False:
            incomplete.append(alpha_id)
        else:
            unknown.append(alpha_id)
    return {"local": {"source": "ledger", "statuses": sorted(BOOK_STATUSES),
                      "member_ids": ids, "member_count": len(ids),
                      "member_digest": digest, "platform_confirmed": False},
            "platform": platform,
            "pnl": {"available_ids": available, "missing_ids": missing,
                    "complete_ids": complete, "unknown_ids": unknown,
                    "incomplete_ids": incomplete, "coverage": coverage,
                    "complete": not missing and not unknown and not incomplete,
                    "scope": "cached daily PnL; completeness is per-alpha recordset coverage"}}
