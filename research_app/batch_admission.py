"""Shared QUICK exploration admission checks using verified account-and-scope evidence."""

from __future__ import annotations

import time
from requests.exceptions import RequestException

import batch_policy
import machine_lib
from . import capabilities


def build_admission(arms, session, account, *, recover_only=False):
    """Assess the complete exploration batch once, with one catalog budget.

    Only positive, account/scope-bound evidence contributes dataset IDs. Keep
    catalog uncertainty separate from a proven shortfall and from expression
    errors; none of these authorizes repeating an accepted or ambiguous POST.
    """
    analysis = batch_policy.analyze_arms(arms)
    evidence = []
    if analysis["ok"] and not recover_only:
        groups = {}
        for arm in analysis["arms"]:
            key = tuple(sorted(arm["scope"].items()))
            group = groups.setdefault(key, {"scope": arm["scope"], "fields": set(), "indices": []})
            group["fields"].update(arm["fields"])
            group["indices"].append(arm["index"])
        records_by_arm = [[] for _ in arms]
        deadline = time.monotonic() + capabilities._REQUIRED_FIELDS_WAIT_SECONDS
        proof_cache = {}
        for group in groups.values():
            try:
                scope = capabilities.cached_field_scope(group["scope"], account=account)
                if scope["evidence"]["state"] == "verified":
                    records = scope["records"]
                else:
                    scope = capabilities.required_field_scope(
                        session, group["scope"], group["fields"], deadline=deadline, proof_cache=proof_cache)
                    records = scope["verified_records"]
                observed = scope["evidence"]
            except (machine_lib.BrainError, RequestException) as exc:
                records = []
                observed = {"state": "unknown", "scope": group["scope"],
                            "error": getattr(exc, "code", type(exc).__name__), "message": str(exc)}
            evidence.append(observed)
            for index in group["indices"]:
                records_by_arm[index] = records
            # One rate limit also defers live reads for the remaining scopes;
            # those scopes may still reuse their already verified cache entries.
            if (observed.get("error") == "RATE_LIMITED" or any(
                    row.get("error") == "RATE_LIMITED" for row in observed.get("catalogs", []))):
                deadline = time.monotonic()
        analysis = batch_policy.assess_datasets(analysis, records_by_arm)
    elif analysis["ok"]:
        analysis = {**analysis, "ok": False, "state": "recovery_only",
                    "blocking": [{"code": "recovery_only",
                                  "message": "existing tasks need recovery, not admission for a new POST"}]}
    retry_times = [row["retry_at"] for scope in evidence for row in scope.get("catalogs", [])
                   if isinstance(row.get("retry_at"), (int, float))]
    return {**analysis, "new_posts_allowed": analysis["ok"], "recovery_allowed": True,
            "field_evidence": evidence, "next_retry_at": max(retry_times, default=None)}


_build_admission = build_admission
