"""Explicit local membership, platform snapshots and PnL coverage evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import alpha_registry
import cache_store
import machine_lib as ml
import research_paths

BOOK_STATUSES = frozenset({"ACTIVE", "SUBMITTED"})
QUARTERLY_MIN_SUBMISSION_DAYS = 20
OSMOSIS_MIN_SCOPES = 3
OSMOSIS_MIN_ALPHAS_PER_SCOPE = 10


def income_readiness(session, *, as_of=None):
    """Count observable income prerequisites from a live ACTIVE listing."""
    alphas = {}
    for hidden in (False, True):
        seen = set()
        offset = 0
        while True:
            page = ml.list_alphas(session, status="ACTIVE", order="-dateSubmitted",
                                  hidden=hidden, alpha_type_not=None, offset=offset, limit=100)
            for alpha in page:
                alpha_id = alpha.get("id")
                if not isinstance(alpha_id, str) or not alpha_id or alpha.get("status") != "ACTIVE":
                    raise ml.ServiceError("ACTIVE Alpha listing returned an invalid record")
                if alpha_id in seen:
                    raise ml.ServiceError("ACTIVE Alpha listing repeated an ID within one visibility scope")
                seen.add(alpha_id)
                alphas.setdefault(alpha_id, alpha)
            if len(page) < 100:
                break
            offset += len(page)

    eastern = ZoneInfo("America/New_York")
    timestamp = datetime.now(eastern) if as_of is None else as_of.astimezone(eastern)
    today = timestamp.date()
    quarter_start = today.replace(month=((today.month - 1) // 3) * 3 + 1, day=1)
    submission_days = set()
    missing_dates = 0
    scope_counts = Counter()
    for alpha in alphas.values():
        settings = alpha.get("settings") or {}
        region, delay = settings.get("region"), settings.get("delay")
        if isinstance(region, str) and region and type(delay) is int and delay >= 0:
            scope_counts[region, delay] += 1

        stamp = alpha.get("dateSubmitted")
        try:
            submitted = datetime.fromisoformat(stamp.replace("Z", "+00:00")) if isinstance(stamp, str) else None
        except ValueError:
            submitted = None
        if submitted is None or submitted.utcoffset() is None or submitted > timestamp:
            missing_dates += 1
            continue
        day = submitted.astimezone(eastern).date()
        if quarter_start <= day <= today:
            submission_days.add(day)

    scopes = [
        {"region": region, "delay": delay, "active_alphas": count,
         "active_count_gap": max(0, OSMOSIS_MIN_ALPHAS_PER_SCOPE - count)}
        for (region, delay), count in sorted(scope_counts.items())
    ]
    scopes_at_minimum = sum(row["active_count_gap"] == 0 for row in scopes)
    return {
        "source": "live ACTIVE Alpha listing (visible and hidden, all Alpha types)",
        "as_of": timestamp.isoformat(),
        "active_alphas": len(alphas),
        "quarterly": {
            "quarter_start": quarter_start.isoformat(),
            "counting_timezone": eastern.key,
            "observed_active_submission_days": len(submission_days),
            "unverified_active_submission_dates": missing_dates,
            "minimum_submission_days": QUARTERLY_MIN_SUBMISSION_DAYS,
            "observed_submission_day_gap": max(0, QUARTERLY_MIN_SUBMISSION_DAYS - len(submission_days)),
            "source": "https://support.worldquantbrain.com/hc/en-us/articles/4702810979351",
        },
        "osmosis": {
            "scopes": scopes,
            "unscoped_active_alphas": len(alphas) - sum(scope_counts.values()),
            "minimum_scopes": OSMOSIS_MIN_SCOPES,
            "minimum_eligible_alphas_per_scope": OSMOSIS_MIN_ALPHAS_PER_SCOPE,
            "scopes_at_minimum_active_count": scopes_at_minimum,
            "observed_active_scope_gap": max(0, OSMOSIS_MIN_SCOPES - scopes_at_minimum),
            "source": "https://platform.worldquantbrain.com/learn/documentation/consultant-information/osmosis-allocation-guide-consultants",
        },
    }


def test_boundaries(latest, ids):
    """Read recorded, exclusive training cutoffs and identify missing boundaries."""
    starts, missing = {}, []
    for alpha_id in sorted(set(ids)):
        split = (latest.get(alpha_id) or {}).get("out_of_sample") or {}
        start = (split.get("test") or {}).get("startDate")
        if not start:
            missing.append(alpha_id)
            continue
        try:
            starts[alpha_id] = date.fromisoformat(str(start)[:10]).isoformat()
        except ValueError:
            raise ValueError("invalid recorded test boundary for %s" % alpha_id) from None
    return starts, missing


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
