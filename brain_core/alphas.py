"""Alpha checks, performance/recordsets, daily PnL, submission and correlations."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
import threading
import time
from datetime import datetime, timezone
from functools import cache
from os import environ
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import cache_store
from check_evidence import normalize_checks, platform_check_names
import machine_lib as _ml
import research_paths


_OPERATOR_MENTION = re.compile(r"[Oo]perator\s+[`'\"]?([A-Za-z_][A-Za-z0-9_]*)")


_OPERATOR_FATAL = re.compile(
    r"unknown operator|not a valid operator|operator .{0,40}(not found|does not "
    r"exist|is not supported|not available|not permitted|is not allowed)",
    re.IGNORECASE,
)


def note_operator_rejection(expression, message, context=None):
    """Learn an operator-level rejection so the linter flags it next time.

    BRAIN answers some rejections with ``Operator <name> ...``. Two flavours
    matter and they are not equally severe:

    * the operator is unknown or unavailable in this config -- ``block``;
    * the operator exists but does not accept this kind of input, e.g.
      ``Operator subtract does not support event inputs`` -- ``warn``.

    Blanket-denying the second kind would disable perfectly good operators, so
    the two are stored with different severities.
    """
    if not message:
        return None
    match = _ml._OPERATOR_MENTION.search(message)
    if not match:
        return None
    operator = match.group(1)
    severity = "block" if _ml._OPERATOR_FATAL.search(message) else "warn"
    try:
        import expression_lint
        return expression_lint.record_rejection(
            operator, message, severity=severity,
            context=context or {"expression": expression},
        )
    except Exception:
        # Learning from a rejection must never break a simulation run.
        return None


# Checks whose limit is a ceiling: PASS when value <= limit. Every other check
# in the catalog uses a floor (value >= limit). The displayed value and formal
# verdict can differ, including when platform exceptions apply. Numeric
# comparison is diagnostic evidence and never replaces the formal result.
_CEILING_CHECKS = frozenset({
    "SELF_CORRELATION",
    "PROD_CORRELATION",
    "CONCENTRATED_WEIGHT",
    "HIGH_TURNOVER",
})


def reconcile_check(name, check):
    """Compare a check's value and limit without changing its formal verdict.

    Returns ``{"expected", "consistent"}``, or None when the check carries no
    value/limit pair to compare (CONCENTRATED_WEIGHT often has neither) or is
    still pending.
    """
    check = check or {}
    value = check.get("value")
    limit = check.get("limit")
    result = check.get("result")
    if value is None or limit is None or result in (None, "PENDING"):
        return None
    lower_is_better = (name or "").upper() in _ml._CEILING_CHECKS
    passed = value <= limit if lower_is_better else value >= limit
    expected = "PASS" if passed else "FAIL"
    return {"expected": expected, "consistent": expected == result}


def _extract_checks(alpha_payload):
    """Normalize platform checks, preserving ambiguity and numeric diagnostics."""
    checks = normalize_checks((alpha_payload.get("is") or {}).get("checks"))
    for name, info in checks.items():
        info.pop("expected_result", None)
        info.pop("value_consistent", None)
        for key in ("result", "limit", "value"):
            info.setdefault(key, None)
        verdict = _ml.reconcile_check(name, info)
        if verdict is not None:
            info["expected_result"] = verdict["expected"]
            info["value_consistent"] = verdict["consistent"]
    return checks or None


def fetch_alpha_checks(s, alpha_id, max_wait_seconds=180.0, on_event=None):
    """Return live platform checks, including computed SELF_CORRELATION.

    ``GET /alphas/{id}`` embeds a *cached* check snapshot, which reports
    SELF_CORRELATION as PENDING or omits it entirely. The dedicated
    ``GET /alphas/{id}/check`` endpoint triggers the computation and answers
    with ``Retry-After`` until the job finishes, so it is the only way to read
    the real correlation value. This polls it within a bounded budget.

    ``complete`` indicates a nonempty check payload was received, not that every
    check resolved. ``payload_received`` also records successful empty payloads.
    The CLI reports check resolution separately from transport completion.
    A zero wait budget allows one request with the default transport timeout.
    """
    if max_wait_seconds is not None and (not math.isfinite(max_wait_seconds) or max_wait_seconds < 0):
        raise ValueError("max_wait_seconds must be finite and non-negative")
    deadline = _ml.time.monotonic() + max_wait_seconds if max_wait_seconds else None
    url = _ml.BRAIN_API_URL + "/alphas/" + alpha_id + "/check"
    payload = None
    checks = None
    while True:
        response = s.get(url, timeout=_ml._request_timeout(deadline))
        if response.status_code in (401, 403):
            raise _ml.AuthError("BRAIN rejected the session (HTTP %s) for alpha %s checks"
                            % (response.status_code, alpha_id))
        if response.status_code == 404:
            raise _ml.NotFound("alpha %s not found" % alpha_id)
        if response.status_code == 429:
            raise _ml.RateLimited("checks for alpha %s are rate limited" % alpha_id)
        if not 200 <= response.status_code < 300:
            raise _ml.ServiceError("HTTP %s fetching checks for alpha %s"
                               % (response.status_code, alpha_id))
        retry_after = _ml._retry_after_seconds(response, cap_error_wait=False)
        if retry_after > 0:
            wait = retry_after
            if on_event and deadline is not None and _ml.time.monotonic() < deadline:
                on_event({"event": "check_pending", "alpha_id": alpha_id,
                          "retry_after_seconds": wait})
        else:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if payload is not None:
                try:
                    checks = _ml._extract_checks(payload)
                except (AttributeError, TypeError) as exc:
                    raise _ml.ServiceError("invalid check payload for alpha %s" % alpha_id) from exc
                if checks:
                    if not platform_check_names(checks):
                        raise _ml.ServiceError("invalid check payload for alpha %s" % alpha_id)
                    if any(check["result"] is not None and not isinstance(check["result"], str)
                           for check in checks.values()):
                        raise _ml.ServiceError("invalid check result for alpha %s" % alpha_id)
                    break
            wait = 5.0
        if deadline is None or wait >= deadline - _ml.time.monotonic():
            break
        _ml.time.sleep(wait)
        if _ml.time.monotonic() >= deadline:
            break
    return {
        "alpha_id": alpha_id,
        "checks": checks,
        "self_correlation": (checks or {}).get("SELF_CORRELATION"),
        "complete": bool(checks),
        "payload_received": payload is not None,
    }


def _performance_request(alpha_id, team_id=None, competition_id=None):
    if team_id is not None and competition_id is not None:
        raise ValueError("choose either a team or a competition, not both")
    for name, value in (("alpha_id", alpha_id), ("team_id", team_id), ("competition_id", competition_id)):
        if (name == "alpha_id" or value is not None) and (not isinstance(value, str) or not value.strip()):
            raise ValueError("%s must be a nonempty identifier" % name)
    kind, identifier = ("competition", competition_id) if competition_id is not None else (
        ("team", team_id) if team_id is not None else ("user", "self"))
    collection = {"competition": "competitions", "team": "teams", "user": "users"}[kind]
    path = "/%s/%s/alphas/%s/before-and-after-performance" % (
        collection, quote(identifier, safe=""), quote(alpha_id, safe=""))
    return {"scope": {"kind": kind, "id": identifier}, "source": _ml.BRAIN_API_URL + path}


def fetch_merged_performance(s, alpha_id, team_id=None, competition_id=None,
                             max_wait_seconds=180.0, on_event=None):
    """Read the platform's merged before/after comparison without submitting an alpha.

    The provider owns the partition and metric period. A ready response is not
    proof of train-only evidence or of a causal marginal contribution.
    """
    request = _ml._performance_request(alpha_id, team_id, competition_id)
    if (isinstance(max_wait_seconds, bool) or not isinstance(max_wait_seconds, (int, float))
            or not math.isfinite(max_wait_seconds) or max_wait_seconds < 0):
        raise ValueError("max_wait_seconds must be a finite, non-negative number")
    deadline = _ml.time.monotonic() + max_wait_seconds
    while True:
        response = s.get(request["source"], timeout=_ml._request_timeout(deadline))
        status = response.status_code
        if status in (401, 403):
            raise _ml.AuthError("merged performance access rejected (HTTP %s)" % status)
        if status == 404:
            raise _ml.NotFound("merged performance is unavailable for the requested alpha and scope")
        retry_after = _ml._retry_after_seconds(response, cap_error_wait=False)
        if not math.isfinite(retry_after):
            raise _ml.ServiceError("merged performance returned an invalid Retry-After")
        observed = {**request, "retrieved_at": datetime.now(timezone.utc).isoformat(), "http_status": status}
        if retry_after > 0 and (200 <= status < 300 or status in (429, 503)):
            if retry_after >= deadline - _ml.time.monotonic():
                return {**observed, "complete": False, "payload": None,
                        "next_poll_after_seconds": retry_after}
            if on_event:
                on_event("performance_pending", {"alpha_id": alpha_id, "retry_after_seconds": retry_after})
            _ml.sleep(retry_after)
            if _ml.time.monotonic() >= deadline:
                return {**observed, "complete": False, "payload": None,
                        "next_poll_after_seconds": 0.0}
            continue
        if status == 429:
            raise _ml.RateLimited("merged performance read was rate limited")
        if not 200 <= status < 300:
            raise _ml.ServiceError("HTTP %s fetching merged performance" % status)
        try:
            payload = response.json()
            json.dumps(payload, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise _ml.ServiceError("merged performance response was not finite JSON") from exc
        stats = payload.get("stats") if isinstance(payload, dict) else None
        if not isinstance(stats, dict) or any(
            side not in stats or (stats[side] is not None and not isinstance(stats[side], dict))
            for side in ("before", "after")
        ):
            raise _ml.ServiceError("merged performance response has no usable before/after statistics object")
        return {**observed, "complete": True, "payload": payload, "next_poll_after_seconds": None}


def _recordset_paths(alpha_id, names):
    if not isinstance(alpha_id, str) or not re.fullmatch(r"[A-Za-z0-9]+", alpha_id):
        raise ValueError("recordset reads require a BRAIN alpha ID")
    if any(not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) for name in names):
        raise ValueError("invalid recordset name; use e.g. yearly-stats")
    base = f"{_ml.BRAIN_API_URL}/alphas/{alpha_id}/recordsets"
    return base, {name: f"{base}/{name}" for name in names}


def _read_recordset(s, url, deadline, *, one_shot=False):
    """Read a native response within the shared wait budget or in one attempt."""
    from requests.exceptions import RequestException

    result = {"source": url, "state": "unavailable", "payload": None}
    while True:
        remaining = deadline - _ml.time.monotonic()
        if remaining <= 0 and not one_shot:
            return {**result, "reason": "wait_budget_exhausted"}
        timeout = 12.0 if one_shot else min(12.0, remaining)
        try:
            response = s.get(url, timeout=timeout, allow_redirects=False)
        except RequestException:
            return {**result, "reason": "transport_error"}
        status = response.status_code
        result.update(http_status=status, retrieved_at=_ml.cache_store.utc_now().isoformat())
        if status == 401:
            raise _ml.AuthError("BRAIN rejected the session while reading alpha recordsets")
        if status in (403, 404):
            return {**result, "reason": "access_denied" if status == 403 else "not_found"}
        delay = _ml._retry_after_seconds(response, cap_error_wait=False)
        if not math.isfinite(delay):
            return {**result, "reason": "invalid_retry_after"}
        if delay > 0 and (200 <= status < 300 or status in (429, 503)):
            if delay >= deadline - _ml.time.monotonic():
                return {**result, "state": "pending", "reason": "retry_after",
                        "next_poll_after_seconds": delay}
            _ml.sleep(delay)
            if _ml.time.monotonic() >= deadline:
                return {**result, "state": "pending", "reason": "retry_after",
                        "next_poll_after_seconds": 0.0}
            continue
        if not 200 <= status < 300:
            return {**result, "reason": "rate_limited" if status == 429 else "http_error"}
        if not response.content.strip():
            return {**result, "reason": "empty_response"}
        try:
            payload = response.json()
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError):
            return {**result, "reason": "invalid_json"}
        return {**result, "state": "available", "payload": payload}


def fetch_alpha_recordsets(s, alpha_id, names=(), max_wait_seconds=180.0):
    """Discover native series and read explicitly named recordsets.

    The discovery response and each named read have separate availability
    evidence. Schemas and rows retain platform dates and units. Completeness
    stays unknown when the response has no pagination evidence.
    """
    base, paths = _ml._recordset_paths(alpha_id, names)
    if (type(max_wait_seconds) not in (int, float) or not math.isfinite(max_wait_seconds)
            or max_wait_seconds < 0):
        raise ValueError("max_wait_seconds must be a finite, non-negative number")
    deadline = _ml.time.monotonic() + max_wait_seconds
    one_shot = max_wait_seconds == 0
    discovery = _ml._read_recordset(s, base, deadline, one_shot=one_shot)
    payload = discovery.pop("payload")
    rows = payload if isinstance(payload, list) else (payload.get("results") if isinstance(payload, dict) else None)
    discovery["complete"] = None
    if isinstance(rows, list) and all(isinstance(row, dict) and isinstance(row.get("name"), str)
                                     and row["name"] for row in rows):
        discovery["available"] = rows
        if isinstance(payload, list):
            discovery["complete"] = True
        elif payload.get("next"):
            discovery["complete"] = False
        elif type(payload.get("count")) is int:
            discovery["complete"] = payload["count"] == len(rows)
    else:
        if discovery["state"] == "available":
            discovery.update(state="unavailable", reason="missing_catalog" if not payload else "invalid_catalog")
        discovery["available"] = None
    advertised = discovery["available"]
    advertised_names = {row["name"] for row in advertised or []}
    datasets = {}
    for name, url in paths.items():
        # A catalog can omit an available series, including yearly statistics.
        entry = _ml._read_recordset(s, url, deadline, one_shot=one_shot)
        payload = entry.pop("payload")
        entry.update(advertised=True if name in advertised_names else False if discovery["complete"] else None,
                     schema=None, records=None, records_returned=None, reported_count=None, complete=None)
        if entry["state"] == "available":
            if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
                entry.update(state="unavailable", reason="missing_records" if not payload else "invalid_records")
            else:
                records = payload["records"]
                count = payload.get("count")
                count = count if type(count) is int and count >= 0 else None
                complete = False if payload.get("next") else (
                    count == len(records) if count is not None else None)
                entry.update(schema=payload.get("schema"), records=records, records_returned=len(records),
                             reported_count=count, complete=complete)
        datasets[name] = entry
    return {"alpha_id": alpha_id, "train_only_verified": False,
            "discovery": discovery, "series": datasets}


_PNL_EVIDENCE_FORMAT = "worldquantuser.pnl-evidence"


def _pnl_digest(series):
    return hashlib.sha256(json.dumps(series, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _matching_pnl_evidence(metadata, alpha_id, digest):
    for candidate in (metadata, metadata.get("previous")):
        if (isinstance(candidate, dict) and candidate.get("format") == _ml._PNL_EVIDENCE_FORMAT
                and candidate.get("alpha_id") == alpha_id and candidate.get("data_sha256") == digest):
            return candidate
    return None


def pnl_cache_status(cache_dir, alpha_id, series=None):
    """Describe cache evidence without assuming a legacy file was fully fetched."""
    import pnl_store

    store_evidence = pnl_store.pnl_evidence_status(cache_dir, alpha_id, series=series)
    if store_evidence is not None:
        return store_evidence

    path = Path(cache_dir) / (alpha_id + ".json")
    metadata_path = Path(cache_dir) / ".meta" / (alpha_id + ".json")
    evidence = {"source": "cache", "path": str(path), "metadata_path": str(metadata_path),
                "complete": None, "as_of": None, "error": None, "points": 0}
    if series is None:
        series, data_error = _ml.cache_store.load_json(path)
        if data_error:
            evidence["error"] = data_error
            return evidence
    if not isinstance(series, dict):
        evidence.update(complete=False, error="invalid_pnl_cache")
        return evidence
    evidence["points"] = len(series)
    metadata, evidence["error"] = _ml.cache_store.load_json(metadata_path)
    if not isinstance(metadata, dict) or metadata.get("format") != _ml._PNL_EVIDENCE_FORMAT:
        return evidence
    try:
        digest = _ml._pnl_digest(series)
    except (TypeError, ValueError):
        evidence.update(complete=False, error="invalid_pnl_values")
        return evidence
    matched = _ml._matching_pnl_evidence(metadata, alpha_id, digest)
    if matched is None and path.exists():
        # Columnar Parquet caches normalize JSON integers (e.g. 100) to float64 (100.0).
        disk_series, disk_err = _ml.cache_store.load_json(path)
        if not disk_err and isinstance(disk_series, dict) and disk_series.keys() == series.keys():
            try:
                disk_digest = _ml._pnl_digest(disk_series)
                disk_matched = _ml._matching_pnl_evidence(metadata, alpha_id, disk_digest)
                if disk_matched is not None and all(float(disk_series[k]) == series[k] for k in series):
                    matched = disk_matched
            except (TypeError, ValueError, OverflowError):
                matched = None
    metadata = matched
    if metadata is None:
        evidence.update(complete=False, error="pnl_metadata_mismatch")
        return evidence
    evidence.update({key: metadata.get(key) for key in
                     ("complete", "as_of", "pages", "start", "end", "missing_values")})
    evidence["error"] = None
    return evidence


def _write_pnl_cache(path, metadata_path, series, metadata, deadline):
    """Persist daily PnL atomically to Delta-Parquet + SQLite WAL (with lock protection)."""
    import pnl_store

    cache_dir = Path(path).parent
    alpha_id = metadata.get("alpha_id") or Path(path).stem
    lock_timeout = max(0, min(1, deadline - _ml.time.monotonic()))
    return pnl_store.write_pnl(cache_dir, alpha_id, series, metadata, timeout=lock_timeout)


def fetch_daily_pnl(s, alpha_id, cache_dir=None, max_age_hours=24.0,
                    page_limit=10000, attempts=3, report=None, max_wait_seconds=180.0, *, deadline=None):
    """Read daily PnL; never return or cache a silently truncated recordset.

    Data lives primarily in Delta-Parquet + SQLite ``meta.sqlite`` (with transparent
    legacy JSON fallback). A hash-bound metadata record records successful pagination.
    Retry-After takes precedence over a nonempty body.
    """
    if not isinstance(page_limit, int) or isinstance(page_limit, bool) or page_limit < 1:
        raise ValueError("page_limit must be a positive integer")
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    if (isinstance(max_wait_seconds, bool) or not isinstance(max_wait_seconds, (int, float))
            or not math.isfinite(max_wait_seconds) or max_wait_seconds < 0):
        raise ValueError("max_wait_seconds must be a finite, non-negative number")
    evidence = {"source": "platform", "complete": False, "pages": 0,
                "points": 0, "as_of": None, "error": None, "missing_values": 0}

    def publish():
        if isinstance(report, dict):
            report.clear()
            report.update(evidence)

    if cache_dir and max_age_hours > 0:
        import pnl_store

        try:
            cached = pnl_store.read_pnl(cache_dir, alpha_id, max_age_hours=max_age_hours)
            if isinstance(cached, dict) and cached:
                evidence = _ml.pnl_cache_status(cache_dir, alpha_id, cached)
                if evidence["complete"] is True:
                    publish()
                    return {key: float(value) for key, value in cached.items()}
        except (OSError, ValueError, TypeError, _ml.cache_store.CacheError):
            pass

    evidence = {"source": "platform", "complete": False, "pages": 0,
                "points": 0, "as_of": None, "error": None, "missing_values": 0}
    from requests.exceptions import RequestException

    relative_deadline = _ml.time.monotonic() + max_wait_seconds
    deadline = relative_deadline if deadline is None else min(deadline, relative_deadline)
    out = {}
    offset = 0
    total = None
    seen_pages = set()
    try:
        while True:
            url = _ml.BRAIN_API_URL + "/alphas/" + alpha_id + "/recordsets/daily-pnl"
            for attempt in range(attempts):
                if _ml.time.monotonic() >= deadline:
                    raise _ml.RateLimited("daily PnL read exceeded its wait budget")
                try:
                    response = s.get(url, params={"limit": page_limit, "offset": offset},
                                     timeout=_ml._request_timeout(deadline))
                except RequestException as exc:
                    if (_ml._transient_read_error(exc)
                            and _ml._wait_for_read_retry(attempt + 1, attempts, deadline)):
                        continue
                    raise
                if response.status_code in (401, 403):
                    raise _ml.AuthError("session rejected while reading daily PnL")
                if response.status_code == 404:
                    raise _ml.NotFound("daily PnL is unavailable for %s" % alpha_id)
                retry_after = _ml._retry_after_seconds(response, cap_error_wait=False)
                if not math.isfinite(retry_after):
                    raise _ml.ServiceError("daily PnL returned an invalid Retry-After")
                if retry_after > 0:
                    evidence["retry_after_seconds"] = retry_after
                    if attempt + 1 == attempts or retry_after >= deadline - _ml.time.monotonic():
                        raise _ml.RateLimited("daily PnL is not ready; retry after the reported delay")
                    _ml.time.sleep(retry_after)
                    continue
                if response.status_code == 200 and response.text.strip():
                    evidence.pop("retry_after_seconds", None)
                    break
                if attempt + 1 < attempts:
                    if 3 >= deadline - _ml.time.monotonic():
                        raise _ml.RateLimited("daily PnL read exceeded its wait budget")
                    _ml.time.sleep(3)
            if response.status_code == 429:
                raise _ml.RateLimited("daily PnL read was rate limited")
            if response.status_code != 200 or not response.text.strip():
                raise _ml.ServiceError("daily PnL page unavailable (HTTP %s)" % response.status_code)
            try:
                payload = response.json()
            except ValueError as exc:
                raise _ml.ServiceError("daily PnL page was not JSON") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
                raise _ml.ServiceError("daily PnL page has no record array")
            records = payload["records"]
            evidence["pages"] += 1
            count = payload.get("count")
            if count is not None:
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise _ml.ServiceError("daily PnL page has an invalid count")
                if total is not None and total != count:
                    raise _ml.ServiceError("daily PnL changed during pagination")
                total = count
            more = bool(payload.get("next"))
            encoded_page = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
            signature = hashlib.sha256(encoded_page).digest()
            if records and signature in seen_pages:
                raise _ml.ServiceError("daily PnL repeated a page")
            seen_pages.add(signature)
            for row in records:
                if not isinstance(row, (list, tuple)) or len(row) < 2 or not isinstance(row[0], str):
                    raise _ml.ServiceError("daily PnL contains a malformed record")
                value = row[1]
                if value is None:
                    evidence["missing_values"] += 1
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise _ml.ServiceError("daily PnL contains a non-finite or nonnumeric value")
                if row[0] in out and out[row[0]] != float(value):
                    raise _ml.ServiceError("daily PnL changed within an overlapping page")
                out[row[0]] = float(value)
            offset += len(records)
            evidence["points"] = len(out)
            if total is not None:
                if offset > total:
                    raise _ml.ServiceError("daily PnL exceeded its declared count")
                if offset == total:
                    break
                if not records:
                    raise _ml.ServiceError("daily PnL ended before its declared count")
            elif not more and len(records) < page_limit:
                break
            elif not records:
                raise _ml.ServiceError("daily PnL pagination did not advance")
    except RequestException as exc:
        evidence["error"] = "SERVICE_ERROR"
        publish()
        raise _ml.ServiceError("daily PnL transport failed: %s" % type(exc).__name__) from exc
    except _ml.BrainError as exc:
        evidence["error"] = exc.code
        publish()
        raise

    evidence.update(complete=True, as_of=datetime.now(timezone.utc).isoformat(),
                    start=min(out) if out else None, end=max(out) if out else None)
    if cache_dir:
        path = Path(cache_dir) / (alpha_id + ".json")
        metadata_path = Path(cache_dir) / ".meta" / (alpha_id + ".json")
        metadata = {**evidence, "format": _ml._PNL_EVIDENCE_FORMAT, "version": 1,
                    "alpha_id": alpha_id, "data_sha256": _ml._pnl_digest(out)}
        try:
            receipt = _ml._write_pnl_cache(path, metadata_path, out, metadata, deadline)
            evidence.update(path=receipt["path"], metadata_path=receipt["metadata_path"], cached=True)
        except (OSError, _ml.cache_store.CacheError) as exc:
            evidence.update(cached=False, cache_error=type(exc).__name__)
    publish()
    return out


def read_alpha_submission_status(s, alpha_id, *, max_wait_seconds=30.0):
    """Read an earlier submission's outcome without issuing another POST.

    An UNSUBMITTED alpha snapshot is not a rejection receipt: evaluation may
    still be running. Only a ready submission endpoint with resolved failing
    checks proves that this evaluation ended in rejection.
    """
    from check_evidence import summarize_checks
    from requests.exceptions import RequestException

    if not _ml.valid_simulation_id(alpha_id):
        raise ValueError("invalid Alpha ID")
    report = {"state": "unknown", "source": "submission_status", "alpha_id": alpha_id}
    if max_wait_seconds is not None and max_wait_seconds <= 0:
        return {**report, "reason": "read_budget_exhausted"}
    deadline = _ml.time.monotonic() + min(30, max_wait_seconds or 30)
    try:
        response = s.get(_ml.BRAIN_API_URL + "/alphas/" + alpha_id + "/submit",
                         timeout=_ml._request_timeout(deadline), allow_redirects=False)
    except RequestException as exc:
        raise _ml.ReadInterrupted("submission status read failed (%s)" % type(exc).__name__) from exc
    status = response.status_code
    report["status_code"] = status
    if status in (401, 403):
        raise _ml.AuthError("BRAIN denied the submission status read")
    delay = _ml._retry_after_seconds(response, cap_error_wait=False)
    if not math.isfinite(delay):
        return {**report, "reason": "invalid_retry_after"}
    if delay > 0:
        return {**report, "state": "pending", "reason": "platform_retry_after",
                "retry_at": _ml.time.time() + delay}
    if status not in (200, 400):
        return {**report, "reason": "submission_status_unavailable"}
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not isinstance(payload, dict) or payload.get("id") not in (None, alpha_id):
        return {**report, "reason": "submission_response_unverified"}
    platform_status = str(payload.get("status") or "").upper()
    if platform_status in {"PENDING", "RUNNING", "SIMULATING", "CHECKING", "ACCEPTED"}:
        return {**report, "state": "pending", "reason": "platform_evaluating"}
    if payload.get("is") is not None and not isinstance(payload["is"], dict):
        return {**report, "reason": "submission_response_unverified"}
    checks = _ml._extract_checks(payload)
    summary = summarize_checks(checks)
    report["checks"] = checks
    if summary["resolved"] and summary["failed"]:
        return {**report, "state": "rejected", "failed_checks": summary["failed"],
                "reason": "completed_submission_checks_failed"}
    return {**report, "reason": "no_terminal_submission_evidence"}


def submit_alpha(s, alpha_id, confirm=False, max_wait_seconds=180.0, on_event=None):
    """Submit one alpha to its competition, or report what would happen.

    BRAIN exposes no separate "test submission": the pre-submission test *is*
    ``GET /alphas/{id}/check`` (see :func:`fetch_alpha_checks`), and the only
    write action is ``POST /alphas/{id}/submit``. Submission is irreversible
    from a user's point of view, so ``confirm`` must be True to send it;
    otherwise a dry-run document describing the current checks is returned.
    """
    if confirm and not _ml.agent_submit_allowed():
        raise _ml.SubmissionDisabled(
            "submission is disabled: %s is not 'true' in the environment or .env, so an "
            "agent may not submit alphas on this account. Backtesting, checks and "
            "self-correlation analysis remain available; set %s=true to allow real "
            "submissions." % (_ml.SUBMIT_UNLOCK_ENV, _ml.SUBMIT_UNLOCK_ENV)
        )
    current = _ml.fetch_alpha_metrics(s, alpha_id)
    status = current.get("status")
    document = {
        "alpha_id": alpha_id,
        "status": status,
        "submitted": False,
        "post_attempted": False,
    }
    if status in ("ACTIVE", "SUBMITTED"):
        document.update({
            "checks": _ml._extract_checks(current) or {},
            "submission": "CONFIRMED",
            "submitted": True,
            "date_submitted": current.get("dateSubmitted"),
            "reason": "already %s" % status,
        })
        return document
    if _ml.is_quick_simulation(current):
        document.update(
            submission="NOT_SENT",
            post_attempted=False,
            reason="Quick simulation results are for screening; rerun with simulationMode=FULL",
        )
        return document
    if not confirm:
        # the platform has no separate test-submission call: the test *is* the
        # check flow, so run it live instead of echoing the cached snapshot.
        live = _ml.fetch_alpha_checks(s, alpha_id,
                                  max_wait_seconds=max_wait_seconds,
                                  on_event=on_event)
        document["checks"] = live.get("checks")
        document["self_correlation"] = live.get("self_correlation")
        document["checks_complete"] = live.get("complete")
        document["submission"] = "NOT_SENT"
        document["reason"] = "confirmation required: rerun with --yes"
        return document
    deadline = _ml.time.monotonic() + (max_wait_seconds or 0)
    url = _ml.BRAIN_API_URL + "/alphas/" + alpha_id + "/submit"
    if on_event:
        # The caller checkpoints this boundary before transport can accept the
        # irreversible request. A failed callback prevents the POST entirely.
        on_event({"event": "submit_posting", "alpha_id": alpha_id})
    document["post_attempted"] = True
    response = s.post(url, timeout=_ml._request_timeout(deadline))
    post_status = response.status_code
    retry_after = _ml._retry_after_seconds(response, cap_error_wait=False)
    outcome_error = None
    if on_event:
        on_event({"event": "submit_sent", "alpha_id": alpha_id,
                  "status_code": post_status,
                  "retry_after_seconds": retry_after if math.isfinite(retry_after) else None})
    while 200 <= post_status < 300 and 0 < retry_after < deadline - _ml.time.monotonic():
        _ml.time.sleep(retry_after)
        if _ml.time.monotonic() >= deadline:
            break
        try:
            response = _ml._get_poll_response(s, url, deadline)
        except _ml.ReadInterrupted as exc:
            outcome_error = str(exc)
            break
        if not 200 <= response.status_code < 300:
            outcome_error = "submission status read returned HTTP %s" % response.status_code
            break
        retry_after = _ml._retry_after_seconds(response, cap_error_wait=False)
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    if post_status >= 500 or post_status == 408 or 300 <= post_status < 400:
        outcome_error = "submission request returned HTTP %s" % post_status
    if outcome_error:
        document.update(submission="UNCERTAIN", status_code=response.status_code,
                        reason=outcome_error + "; verify platform status before another attempt",
                        response=payload)
        return document

    # Four outcomes, not two. A 2xx from this endpoint means "accepted for
    # evaluation", not "submitted": the platform answers 201 with an empty body while
    # the checks are still being computed, and an alpha that then fails them stays
    # UNSUBMITTED with no error anywhere. Treating the status code as success is the
    # mistake this function used to make, and it is the same mistake that produced a
    # false "twenty submitted" report during the campaign.
    checks = (payload.get("is") or {}).get("checks") or []
    failed = [check for check in checks
              if check.get("result") not in (None, "PASS", "PENDING")]

    if post_status >= 400:
        document.update({
            "submission": "REJECTED",
            "submitted": False,
            "status_code": response.status_code,
            "checks": _ml._extract_checks({"is": {"checks": checks}}) or {},
            "failed_checks": [check.get("name") for check in failed],
            "reason": ("refused: " + ", ".join(
                "%s=%s" % (check.get("name"), check.get("value")) for check in failed)
                if failed else "refused with HTTP %s" % response.status_code),
            "response": payload,
        })
        return document

    # A 2xx only buys an evaluation. The read-back decides.
    try:
        refreshed = _ml.fetch_alpha_metrics(s, alpha_id)
    except _ml.BrainError as exc:
        document.update({
            "submission": "UNCERTAIN",
            "submitted": False,
            "status_code": response.status_code,
            "reason": "accepted but the read-back failed (%s); re-check dateSubmitted "
                      "before treating this as submitted" % type(exc).__name__,
            "response": payload,
        })
        return document

    date_submitted = refreshed.get("dateSubmitted")
    document.update({
        "submitted": bool(date_submitted),
        "submission": "CONFIRMED" if date_submitted else "ACCEPTED",
        "status": refreshed.get("status"),
        "date_submitted": date_submitted,
        "status_code": response.status_code,
        "checks": _ml._extract_checks(refreshed) or {},
        "response": payload,
    })
    if not date_submitted:
        document["reason"] = (
            "accepted for asynchronous evaluation but dateSubmitted is still empty; "
            "reconcile this attempt with read-only status checks before another submission")
    return document


def list_alphas(s, limit=100, offset=0, order="-is.sharpe", status="UNSUBMITTED",
                region=None, universe=None, min_sharpe=None, min_fitness=None,
                alpha_type_not="SUPER", hidden=False, search=None, *,
                delay=None, instrument_type=None, category=None, color=None,
                tag=None, max_turnover=None, date_created_from=None,
                date_created_to=None, date_submitted_from=None, deadline=None):
    """List the account's own alphas (``GET /users/self/alphas``) as ``list[dict]``.

    Metadata only: this never re-simulates anything and consumes no quota.
    Filters are pushed server-side, so paging a large account stays cheap.
    ``order`` uses the API syntax, e.g. ``-is.sharpe`` or ``is.fitness``.

    Careful: ``status`` defaults to ``"UNSUBMITTED"``, which **hides the
    submitted book**. Paging the default filter and then reporting "every alpha
    is unsubmitted" is a wrong conclusion that has already cost one analysis
    pass. Pass ``status="ACTIVE"`` to read what is actually submitted, or
    ``status=None`` for no filter at all.
    """
    params = {
        "limit": min(int(limit), 100),
        "offset": int(offset),
        "order": order,
        "hidden": "true" if hidden else "false",
    }
    if alpha_type_not:
        params["type!="] = alpha_type_not
    if status:
        params["status"] = status
    if region:
        params["settings.region"] = region
    if universe:
        params["settings.universe"] = universe
    if delay is not None:
        params["settings.delay"] = int(delay)
    if instrument_type:
        params["settings.instrumentType"] = instrument_type
    if category:
        params["category"] = category
    if color:
        params["color"] = color
    if tag:
        params["tag"] = tag
    if min_sharpe is not None:
        params["is.sharpe>"] = min_sharpe
    if min_fitness is not None:
        params["is.fitness>"] = min_fitness
    if max_turnover is not None:
        params["is.turnover<"] = max_turnover
    if date_created_from:
        params["dateCreated>="] = date_created_from
    if date_created_to:
        params["dateCreated<"] = date_created_to
    if date_submitted_from:
        params["dateSubmitted>="] = date_submitted_from
    if search:
        params["search"] = search
    limits = {} if deadline is None else {"deadline": deadline}
    payload = _ml.request_json(s, "GET", "/users/self/alphas", params=params, **limits)
    return _ml._as_results(payload, "alphas")


def extract_split(payload):
    """Pull BRAIN's own train/test split out of an alpha payload.

    A regular simulation always runs the platform's fixed window (this account
    reports 2019-01-01..2023-12-31; ``startDate``/``endDate`` in the settings
    are assigned by BRAIN and are silently ignored when a caller sends their
    own). The only out-of-sample evidence the API exposes is ``testPeriod``:
    the tail of that window becomes ``test`` and everything before it ``train``.

    Returns ``{"test_period", "train", "test"}`` with ``None`` blocks when the
    alpha was simulated with ``testPeriod: P0Y`` (no split exists).
    """
    if not isinstance(payload, dict):
        return None
    settings = payload.get("settings") or {}
    train = payload.get("train")
    test = payload.get("test")
    if not isinstance(train, dict) and not isinstance(test, dict):
        return None

    def block(source):
        if not isinstance(source, dict):
            return None
        picked = {k: source.get(k) for k in _ml.SPLIT_METRIC_KEYS if k in source}
        for key in ("startDate", "endDate"):
            value = source.get(key)
            if value is not None:
                picked[key] = value
        return picked or None

    return {
        "test_period": settings.get("testPeriod"),
        "train": block(train),
        "test": block(test),
    }


def set_alpha_metadata(s, alpha_id, name=None, category=None, tags=None,
                       description=None, *, color=_ml._UNSET, hidden=None):
    """Set an alpha's writable metadata and return the updated payload.

    ``name``, ``category``, ``tags``, ``color`` and ``hidden`` are top-level properties;
    ``tags`` must be a list, not a string. ``description`` is nested under ``regular``
    (``{"regular": {"description": ...}}``).

    The update is verified by re-reading rather than trusting the PATCH status, for
    the same reason a submission is verified by ``dateSubmitted``: this API answers
    with success codes for requests it has only accepted.
    """
    payload = {}
    if name is not None:
        payload["name"] = name
    if category is not None:
        if category not in _ml.ALPHA_CATEGORIES:
            raise ValueError("category must be one of %s, got %r"
                             % (", ".join(_ml.ALPHA_CATEGORIES), category))
        payload["category"] = category
    if tags is not None:
        payload["tags"] = list(tags)
    if description is not None:
        payload["regular"] = {"description": str(description)}
    if color is not _ml._UNSET:
        if color is None or (isinstance(color, str) and color.strip().upper() == "NONE"):
            payload["color"] = None
        else:
            normalized_color = str(color).strip().upper()
            if normalized_color not in _ml.ALPHA_COLORS:
                raise ValueError("color must be one of %s or None, got %r"
                                 % (", ".join(_ml.ALPHA_COLORS), color))
            payload["color"] = normalized_color
    if hidden is not None:
        if not isinstance(hidden, bool):
            raise ValueError("hidden must be a boolean")
        payload["hidden"] = hidden
    if not payload:
        raise ValueError("nothing to update")

    response = s.patch(_ml.BRAIN_API_URL + "/alphas/" + alpha_id, json=payload,
                       timeout=_ml._request_timeout(None))
    if response.status_code != 200:
        raise _ml.ServiceError("metadata update failed: HTTP %d %s"
                           % (response.status_code, (response.text or "")[:200]))

    current = _ml.fetch_alpha_metrics(s, alpha_id)
    mismatched = []
    for key, value in payload.items():
        if key == "regular":
            # `regular` is a whole object on the way back: code, language, limits and the
            # description among them. Comparing objects would always fail; the only part
            # this function writes is the description.
            if (current.get("regular") or {}).get("description") != value.get("description"):
                mismatched.append("regular.description")
        elif current.get(key) != value:
            mismatched.append(key)
    if mismatched:
        raise _ml.ServiceError("metadata did not stick: %s" % ", ".join(mismatched))
    return current


_CORRELATION_KINDS = ("self", "prod", "power-pool")


def _fetch_alpha_correlation_endpoint(
    s, alpha_id, kind, *, max_wait_seconds=60.0, max_step_wait=None,
    max_attempts=None, on_event=None
):
    """Poll ``GET /alphas/{id}/correlations/{kind}`` until ready or budget expires."""
    from requests.exceptions import RequestException

    if not _ml.valid_simulation_id(alpha_id):
        raise ValueError("invalid Alpha ID")
    if kind not in _ml._CORRELATION_KINDS:
        raise ValueError("correlation kind must be one of %s" % (", ".join(_ml._CORRELATION_KINDS),))
    if (isinstance(max_wait_seconds, bool) or not isinstance(max_wait_seconds, (int, float))
            or not math.isfinite(max_wait_seconds) or max_wait_seconds < 0):
        raise ValueError("max_wait_seconds must be a finite, non-negative number")

    url = f"{_ml.BRAIN_API_URL}/alphas/{alpha_id}/correlations/{kind}"
    deadline = _ml.time.monotonic() + max_wait_seconds
    attempt = 0
    last_status = None
    last_retry_after = None

    while True:
        attempt += 1
        try:
            response = s.get(
                url,
                timeout=_ml._request_timeout(deadline if max_wait_seconds > 0 else None),
            )
        except RequestException as exc:
            raise _ml.ReadInterrupted(
                f"{kind} correlation read failed ({type(exc).__name__})"
            ) from exc

        status = getattr(response, "status_code", 200)
        last_status = status
        if status == 401:
            raise _ml.AuthError(f"BRAIN rejected the session (HTTP 401) for {kind} correlation")
        if status == 403:
            raise _ml.PermissionDenied(
                f"BRAIN denied access (HTTP 403) to {kind} correlation for {alpha_id}"
            )
        if status == 404:
            raise _ml.NotFound(f"alpha {alpha_id} or {kind} correlation not found")

        headers = getattr(response, "headers", None) or {}
        raw_retry = headers.get("Retry-After")
        if raw_retry is None:
            retry_after = 0.0
        else:
            retry_after = _ml._retry_after_seconds(response, cap_error_wait=False)
            if not math.isfinite(retry_after):
                raise _ml.ServiceError(f"{kind} correlation returned an invalid Retry-After")
        last_retry_after = retry_after

        if status == 429 and retry_after <= 0:
            raise _ml.RateLimited(f"{kind} correlation for {alpha_id} is rate limited")
        if not 200 <= status < 300 and status not in (429, 503):
            raise _ml.ServiceError(f"HTTP {status} fetching {kind} correlation for {alpha_id}")

        body_text = getattr(response, "text", "") or ""
        if 200 <= status < 300 and retry_after <= 0 and body_text.strip():
            try:
                payload = response.json()
            except ValueError as exc:
                raise _ml.ServiceError(f"{kind} correlation returned non-JSON") from exc
            if not isinstance(payload, dict):
                raise _ml.ServiceError(f"{kind} correlation returned a non-object payload")
            raw_bytes = getattr(response, "content", None)
            if not isinstance(raw_bytes, (bytes, bytearray)):
                raw_bytes = body_text.encode("utf-8")
            return {
                "alpha_id": alpha_id,
                "kind": kind,
                "source": f"GET /alphas/{alpha_id}/correlations/{kind}",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "http_status": status,
                "response_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "schema": payload.get("schema"),
                "max": payload.get("max"),
                "min": payload.get("min"),
                "records": payload.get("records") or [],
                "raw_json_response": payload,
            }

        if max_attempts is not None and attempt >= max_attempts:
            break
        if max_wait_seconds <= 0 or _ml.time.monotonic() >= deadline:
            break
        wait = retry_after if retry_after > 0 else 2.0
        if max_step_wait is not None:
            wait = min(wait, max_step_wait)
        remaining = deadline - _ml.time.monotonic()
        if remaining <= 0:
            break
        wait = min(wait, remaining) if max_step_wait is not None else wait
        if wait > remaining:
            break
        if on_event:
            on_event({
                "event": "correlation_pending",
                "alpha_id": alpha_id,
                "kind": kind,
                "retry_after_seconds": wait,
            })
        _ml.time.sleep(wait)

    if last_status == 429:
        raise _ml.RateLimited(
            f"{kind} correlation for {alpha_id} exceeded its wait budget",
            retry_after=last_retry_after if last_retry_after and last_retry_after > 0 else None,
        )
    if max_attempts is not None:
        raise _ml.ServiceError(f"{kind}-correlation stayed empty after {max_attempts} attempts")
    raise _ml.ServiceError(f"{kind}-correlation stayed pending or empty for {alpha_id}")


def fetch_self_correlation(s, alpha_id, tries=6, max_wait=5.0, *,
                           max_wait_seconds=None, on_event=None):
    """Return the platform's own self-correlation view for one alpha.

    The counterpart's Sharpe determines whether the substitute route clears its
    required margin; a bare maximum correlation cannot answer that question.

    The endpoint frequently answers HTTP 200 with an *empty body* plus a Retry-After
    header. That means "not now", not "no data": reading it as absence silently
    reports an alpha as uncorrelated. The response is retried while the body is blank.
    """
    if max_wait_seconds is None:
        capture = _ml._fetch_alpha_correlation_endpoint(
            s, alpha_id, "self",
            max_wait_seconds=max(0.0, float(max_wait) * int(tries)),
            max_step_wait=float(max_wait),
            max_attempts=int(tries),
            on_event=on_event,
        )
    else:
        capture = _ml._fetch_alpha_correlation_endpoint(
            s, alpha_id, "self",
            max_wait_seconds=max_wait_seconds,
            on_event=on_event,
        )
    return {
        "alpha_id": alpha_id,
        "max": capture["max"],
        "min": capture["min"],
        "records": capture["records"],
        "schema": capture["schema"],
        "captured_at": capture["captured_at"],
    }


def fetch_prod_correlation(s, alpha_id, *, max_wait_seconds=60.0, on_event=None):
    """Fetch production correlation histogram/records via ``GET /alphas/{id}/correlations/prod``."""
    return _ml._fetch_alpha_correlation_endpoint(
        s, alpha_id, "prod", max_wait_seconds=max_wait_seconds, on_event=on_event
    )


def build_prod_correlation_archive(alpha_id, capture):
    """Package a ``fetch_prod_correlation`` capture into the ``--pc-distribution`` archive contract."""
    if not _ml.valid_simulation_id(alpha_id):
        raise ValueError("invalid Alpha ID")
    if not isinstance(capture, dict) or not isinstance(capture.get("raw_json_response"), dict):
        raise ValueError("capture must contain raw_json_response from fetch_prod_correlation")
    return {
        "source": "GET /alphas/{id}/correlations/prod",
        "requested_ids": [alpha_id],
        "results": {
            alpha_id: [{
                "captured_at": capture["captured_at"],
                "http_status": capture.get("http_status", 200),
                "retry_after": capture.get("retry_after"),
                "response_sha256": capture.get("response_sha256"),
                "raw_json_response": capture["raw_json_response"],
            }],
        },
    }


def fetch_power_pool_correlation(s, alpha_id, *, max_wait_seconds=60.0, on_event=None):
    """Fetch Power Pool correlation detail via ``GET /alphas/{id}/correlations/power-pool``."""
    return _ml._fetch_alpha_correlation_endpoint(
        s, alpha_id, "power-pool", max_wait_seconds=max_wait_seconds, on_event=on_event
    )


def power_pool_correlation_verdict(payload, limit=0.5):
    """Evaluate Power Pool correlation against its strict ``< limit`` ceiling (default 0.5)."""
    records = payload.get("records") or []
    max_corr = payload.get("max")
    max_valid = type(max_corr) in (int, float) and math.isfinite(max_corr)
    counterparts = []
    bins_at_or_above = 0
    is_histogram = bool(
        records and all(
            isinstance(row, (list, tuple)) and len(row) == 3
            and type(row[0]) in (int, float) and type(row[1]) in (int, float)
            and type(row[2]) is int
            for row in records
        )
    )
    if is_histogram:
        bins_at_or_above = sum(
            row[2] for row in records if row[0] >= limit or (row[1] > limit and row[2] > 0 and max_valid and max_corr >= limit)
        )
    else:
        for row in records:
            if (isinstance(row, (list, tuple)) and len(row) > 5
                    and type(row[5]) in (int, float) and math.isfinite(row[5])
                    and abs(row[5]) >= limit):
                counterparts.append({
                    "alpha_id": row[0],
                    "name": row[1] if len(row) > 1 else None,
                    "correlation": row[5],
                    "sharpe": row[6] if len(row) > 6 else None,
                })
    known = bool(records) or max_valid
    blocked = bool(counterparts) or bins_at_or_above > 0 or (max_valid and abs(max_corr) >= limit)
    if not known:
        reason = "no Power Pool correlation records: not scored yet"
    elif blocked:
        reason = "Power Pool correlation at or above limit (< %g required)" % limit
    else:
        reason = "Power Pool correlation strictly below limit"
    return {
        "blocked": blocked,
        "known": known,
        "limit": limit,
        "max_correlation": max_corr if max_valid else None,
        "records": len(records),
        "counterparts": counterparts,
        "bins_at_or_above_limit": bins_at_or_above if is_histogram else None,
        "reason": reason,
    }


def self_correlation_verdict(payload, limit=0.7, sharpe_margin=1.10):
    """Decide whether a submission is blocked, and whether the substitute route holds.

    The rule allows a correlated alpha through when its Sharpe exceeds, by the margin,
    the Sharpe of *every* alpha it correlates with above the limit - not merely the
    worst one. Checking only the maximum correlation's counterpart is the tempting
    shortcut and it is wrong in the direction that matters: a second, higher-Sharpe
    counterpart raises the bar.
    """
    records = payload.get("records") or []
    correlated = [row for row in records
                  if len(row) > 5 and isinstance(row[5], (int, float))
                  and abs(row[5]) > limit]
    if not correlated:
        # `known` separates two things that look identical in `blocked`: a real
        # measurement where everything sits under the limit, and a payload with no
        # records at all because the platform has not scored this alpha yet. Reading
        # the second as "clear" is the same mistake as reading an empty body as
        # zero correlation, one layer up -- and it is how an unscored alpha gets
        # reported as submittable.
        return {"blocked": False, "known": bool(records),
                "reason": "nothing above the correlation limit" if records
                          else "no correlation records: not scored yet",
                "counterparts": [], "sharpe_needed": None, "records": len(records)}

    sharpes = [row[6] for row in correlated if len(row) > 6 and isinstance(row[6], (int, float))]
    needed = max(sharpes) * sharpe_margin if sharpes else None
    return {
        "blocked": True,
        "known": True,
        "records": len(records),
        "reason": "self-correlation above the limit",
        "counterparts": [
            {"alpha_id": row[0], "name": row[1], "correlation": row[5],
             "sharpe": row[6] if len(row) > 6 else None,
             "fitness": row[9] if len(row) > 9 else None}
            for row in correlated
        ],
        "sharpe_needed": round(needed, 4) if needed else None,
    }
