"""BRAIN catalog discovery, field metadata caching and operator/category lookups."""

from __future__ import annotations

import hashlib
import json
import math
import os
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
from .errors import BrainError


def configure_metadata_cache(max_age=None, refresh=None, enabled=None):
    """Set the process-wide catalog cache policy (the CLI maps its flags here)."""
    if max_age is not None:
        _ml._metadata_policy["max_age"] = float(max_age)
    if refresh is not None:
        _ml._metadata_policy["refresh"] = bool(refresh)
    if enabled is not None:
        _ml._metadata_policy["enabled"] = bool(enabled)
    return dict(_ml._metadata_policy)


def metadata_cache_policy():
    """Current catalog cache policy, for `cache info` and `schema`."""
    return dict(_ml._metadata_policy)


def metadata_cache_path(kind, params):
    """Deterministic path for one cached catalog response.

    The key is a hash of the query parameters, so different regions/universes/
    datasets never collide, while a repeat of the same query lands on the same
    file and hits.
    """
    blob = json.dumps({"kind": kind, "params": params}, sort_keys=True, default=str)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", kind)[:40] or "meta"
    return Path(_ml.research_paths.metadata_root()) / label / (digest + ".json")


def cached_metadata(kind, params, fetch, max_age=None, refresh=None, report=None, *,
                    use_cache=True, account=None, derive_from=None):
    """Read catalog data and return it with cache evidence.

    A hit must satisfy both the stored expiry and the requested maximum age.
    Read the source on a miss, a forced refresh, or when caching is disabled.
    On BrainError, reuse an available snapshot from the same cache key and mark
    it stale and degraded. A derived query may use a separate, fresh full
    snapshot before attempting the network. Pass a report dict for cache evidence.
    """
    policy = _ml._metadata_policy
    ttl = policy["max_age"] if max_age is None else float(max_age)
    do_refresh = policy["refresh"] if refresh is None else bool(refresh)

    def cache_path(query):
        path = _ml.metadata_cache_path(kind, query)
        return path.with_name(f"{account}-{path.name}") if account is not None else path

    path = cache_path(params) if use_cache else None
    status = {
        "kind": kind,
        "path": None if path is None else str(path),
        "hit": False,
        "stale": False,
        "degraded": False,
        "disabled": False,
        "age_seconds": None,
        "expires_in_seconds": None,
        "refreshed_at": None,
        "error": None,
        "complete": not bool(params.get("limit")),
        "account_bound": account is not None,
    }

    def done(data):
        if isinstance(report, dict):
            report.clear()
            report.update(status)
        return {"data": data, "cache": status}

    if path is None or not policy["enabled"] or ttl <= 0:
        status["disabled"] = True
        data = fetch()
        status["refreshed_at"] = datetime.now(timezone.utc).isoformat()
        return done(data)

    entry = _ml.cache_store.read_entry(path)
    usable = entry["present"] and entry["kind"] == kind

    def cached_result(source, data, error=None):
        remaining = source["expires_in"]
        status.update(hit=True, stale=error is not None, degraded=error is not None,
                      age_seconds=round(source["age_seconds"], 1), refreshed_at=source["created_at"],
                      expires_in_seconds=None if remaining is None else round(remaining, 1),
                      error=error)
        return done(data)

    if usable and not do_refresh and entry["fresh"] and entry["age_seconds"] <= ttl:
        return cached_result(entry, entry["payload"])
    if not do_refresh and derive_from is not None:
        source_params, derive = derive_from
        source_path = cache_path(source_params)
        source = _ml.cache_store.read_entry(source_path)
        if (source["present"] and source["kind"] == kind and source["fresh"]
                and source["age_seconds"] <= ttl):
            derived = derive(source["payload"])
            if derived is not None:
                status.update(path=str(source_path), derived_from_complete=True)
                return cached_result(source, derived)
    if entry.get("error") not in (None, "missing"):
        _ml.cache_store.quarantine(path)

    try:
        data = fetch()
    except _ml.BrainError as exc:
        if usable:
            _ml.note("metadata cache: serving stale %s snapshot (%s)" % (kind, exc.code))
            return cached_result(entry, entry["payload"], exc.code)
        raise
    entry = _ml.cache_store.build_entry(kind, data, ttl=ttl, source="brain-api")
    try:
        _ml.cache_store.write_json_atomic(path, entry)
    except OSError as exc:
        _ml.note("metadata cache: could not write %s (%s)" % (path, exc))
    status.update(refreshed_at=entry["_cache"]["created_at"], age_seconds=0.0,
                  expires_in_seconds=ttl)
    return done(data)


class CatalogAccess:
    """Pair the catalog account identity with a runner that opens the session on demand."""

    def __init__(self, username, run):
        self.username = username
        self.run = run


def _catalog_account(username):
    name = username.strip().casefold() if isinstance(username, str) else ""
    return hashlib.sha256(name.encode()).hexdigest() if name else None


def catalog_field_cache_paths():
    """Yield (path, account_bound) for current-account and legacy field snapshots."""
    account = _ml._catalog_account(_ml.configured_username())
    directory = Path(_ml.research_paths.metadata_root()) / "datafields"
    for path in sorted(directory.glob("*.json")):
        namespace, separator, _ = path.stem.partition("-")
        if not separator or namespace == account:
            yield path, bool(separator)


def _cached_account_catalog(session, kind, params, fetch, refresh, report, *, derive_from=None):
    if isinstance(session, _ml.CatalogAccess):
        username, run = session.username, session.run
    else:
        auth = getattr(session, "auth", None)
        username = auth[0] if isinstance(auth, (tuple, list)) and auth else None

        def run(work):
            return work(session)

    account = _ml._catalog_account(username)
    return _ml.cached_metadata(kind, params, lambda: run(fetch), refresh=refresh, report=report,
                           use_cache=account is not None, account=account,
                           derive_from=derive_from)["data"]


def get_datasets(s, instrument_type="EQUITY", region="USA", delay=1,
                 universe="TOP3000", limit=None, refresh=None, report=None, theme=None, *, deadline=None):
    """List datasets available for a region/universe/delay as ``list[dict]``.

    Quota-free and cached (see :func:`cached_metadata`). An empty result concerns
    this exact combination; it does not establish access for other universes or
    delays, or distinguish an unsupported combination from an empty catalog.
    """
    if theme is not None and not isinstance(theme, bool):
        raise ValueError("theme must be true, false or unspecified")
    params = {"instrumentType": instrument_type, "region": region,
              "delay": delay, "universe": universe, "limit": limit}
    if theme is not None:
        params["theme"] = theme
    request_limits = {} if deadline is None else {"deadline": deadline}
    return _ml._cached_account_catalog(
        s, "datasets", params,
        lambda session: _ml._get_datasets(session, instrument_type, region, delay, universe, limit, theme,
                                      **request_limits),
        refresh, report,
    )


def _get_datasets(s, instrument_type="EQUITY", region="USA", delay=1,
                  universe="TOP3000", limit=None, theme=None, *, deadline=None):
    """List datasets available for a region/universe/delay as ``list[dict]``.

    Quota-free. An empty result concerns only this exact combination.
    """
    results = []
    params = {"instrumentType": instrument_type, "region": region,
              "delay": delay, "universe": universe}
    if theme is not None:
        params["theme"] = "true" if theme else "false"
    request_limits = {} if deadline is None else {"deadline": deadline}
    for batch in _ml._catalog_pages(s, "/data-sets", params, keys=("data-sets",), **request_limits):
        if any(not isinstance(row.get("id"), str) or not row["id"] for row in batch):
            raise _ml.ServiceError("catalog returned an invalid dataset ID")
        results.extend(batch)
        if limit and len(results) >= limit:
            return results[:limit]
    return results


def _as_results(payload, *keys):
    """Normalize a BRAIN collection response into a list.

    Some endpoints wrap rows in ``{"results": [...]}`` and some return a bare
    JSON array, so callers should not have to care which.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        for key in ("results", "operators", "categories", "data-sets", "datafields", "competitions"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _catalog_pages(session, path, params=None, keys=(), page_size=None, *, deadline=None):
    """Yield complete collection pages; incomplete or repeated pages are errors."""
    offset = 0
    seen = set()
    total = None
    request_limits = {} if deadline is None else {"deadline": deadline, "max_retries": 1}
    while True:
        query = dict(params or {})
        if page_size is not None or offset:
            query.update(limit=page_size or 50, offset=offset)
        payload = _ml.request_json(session, "GET", path, params=query or None, **request_limits)
        collections = (*keys, "results", "operators", "categories", "data-sets", "datafields")
        if not isinstance(payload, list) and not (
            isinstance(payload, dict) and any(isinstance(payload.get(key), list) for key in collections)
        ):
            raise _ml.ServiceError("catalog %s returned no collection" % path)
        batch = _ml._as_results(payload, *keys)
        if any(not isinstance(row, dict) for row in batch):
            raise _ml.ServiceError("catalog %s returned an invalid row" % path)
        reported = payload.get("count") if isinstance(payload, dict) else None
        if reported is not None:
            if isinstance(reported, bool) or not isinstance(reported, int) or reported < 0:
                raise _ml.ServiceError("catalog %s returned an invalid count" % path)
            if total is not None and total != reported:
                raise _ml.ServiceError("catalog %s changed during pagination" % path)
            total = reported
        more = bool(payload.get("next")) if isinstance(payload, dict) else False
        if not batch:
            if more or (total is not None and offset < total):
                raise _ml.ServiceError("catalog %s ended before all rows were received" % path)
            return
        signature = tuple(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in batch)
        if signature in seen:
            raise _ml.ServiceError("catalog %s repeated a page" % path)
        seen.add(signature)
        offset += len(batch)
        if total is not None and offset > total:
            raise _ml.ServiceError("catalog %s returned more rows than its count" % path)
        yield batch
        if total is not None and offset == total:
            return
        if total is None and not more and (page_size is None or len(batch) < page_size):
            return
        if page_size is None:
            page_size = len(batch)


def fetch_operators(s, search="", group="", refresh=None, report=None, *, deadline=None):
    """List FASTEXPR operators as ``list[dict]`` (``name``, ``category``, ...).

    Quota-free and cached. ``search`` matches name or description, ``group``
    selects an exact category such as ``Time Series``.
    """
    operators = _ml.cached_metadata(
        "operators", {}, lambda: _ml._fetch_operators(s, deadline=deadline), refresh=refresh, report=report)["data"]
    if group:
        operators = [o for o in operators if (o.get("category") or "") == group]
    if search:
        needle = search.lower()
        operators = [
            o for o in operators
            if needle in ((o.get("name") or "") + " " + (o.get("description") or "")).lower()
        ]
    return operators


def _fetch_operators(s, search="", group="", *, deadline=None):
    """List FASTEXPR operators as ``list[dict]`` (``name``, ``category``, ...).

    Quota-free. ``search`` matches the name or description, ``group`` an exact
    category such as ``Time Series``.
    """
    operators = [row for batch in _ml._catalog_pages(s, "/operators", deadline=deadline) for row in batch]
    if group:
        operators = [o for o in operators if (o.get("category") or "") == group]
    if search:
        needle = search.lower()
        operators = [
            o for o in operators
            if needle in ((o.get("name") or "") + " " + (o.get("description") or "")).lower()
        ]
    return operators


def fetch_operator_doc(s, name, refresh=None, report=None):
    """Return one operator's documentation page (markdown body + metadata).

    Cached: the wording changes rarely and this is the most repeated lookup in
    expression work.
    """
    return _ml.cached_metadata(
        "operator-doc", {"name": name},
        lambda: _ml._fetch_operator_doc(s, name), refresh=refresh, report=report)["data"]


def _fetch_operator_doc(s, name):
    """Return one operator's documentation page (markdown body + metadata)."""
    return _ml.request_json(s, "GET", "/operators/" + name)


def fetch_data_categories(s, region="USA", universe="TOP3000", delay=1,
                          refresh=None, report=None):
    """Return data categories with their field/alpha counts and value score.

    ``alphaCount`` is a rough crowding signal: a high count means many people
    have already mined that category. Cached.
    """
    params = {"region": region, "universe": universe, "delay": delay}
    return _ml._cached_account_catalog(
        s, "data-categories", params,
        lambda session: _ml._fetch_data_categories(session, region, universe, delay),
        refresh, report,
    )


def _fetch_data_categories(s, region="USA", universe="TOP3000", delay=1):
    """Return the data categories with their field/alpha counts and value score.

    ``alphaCount`` is a rough crowding signal: high counts mean "many people
    already mined this".
    """
    payload = _ml.request_json(
        s, "GET", "/data-categories",
        params={"region": region, "universe": universe, "delay": delay},
    )
    return _ml._as_results(payload, "categories")


def get_datafields(s, instrument_type="EQUITY", region="USA", delay=1,
                   universe="TOP3000", dataset_id="", search="", field_type=None,
                   limit=None, refresh=None, report=None):
    """List datafields as ``list[dict]`` (``id``, ``description``, ``type``, ...).

    Paginates over ``GET /data-fields`` and stops once ``limit`` rows are
    collected. ``field_type`` filters on the BRAIN type (MATRIX, VECTOR, GROUP).
    Quota-free and cached per query, so re-auditing 4367 fields costs nothing.
    """
    params = {"instrumentType": instrument_type, "region": region, "delay": delay,
              "universe": universe, "dataset": dataset_id, "search": search,
              "type": field_type, "limit": limit}
    derive_from = None
    if search or field_type or limit:
        def filter_complete(rows):
            if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
                try:
                    return _ml._filter_datafields(rows, field_type, search, limit)
                except _ml.ServiceError:
                    return None

        derive_from = ({**params, "search": "", "type": None, "limit": None}, filter_complete)
    return _ml._cached_account_catalog(
        s, "datafields", params,
        lambda session: _ml._get_datafields(session, instrument_type, region, delay, universe,
                                        dataset_id, search, field_type, limit),
        refresh, report, derive_from=derive_from,
    )


def _get_datafields(s, instrument_type="EQUITY", region="USA", delay=1,
                   universe="TOP3000", dataset_id="", search="", field_type=None,
                   limit=None):
    """List datafields as ``list[dict]`` (``id``, ``description``, ``type``, ...).

    Paginates over ``GET /datafields`` and stops early once ``limit`` rows are
    collected. ``field_type`` filters on the BRAIN type (MATRIX, VECTOR, GROUP).
    Quota-free.
    """
    params = {"instrumentType": instrument_type, "region": region, "delay": delay,
              "universe": universe, "dataset.id": dataset_id}
    # BRAIN's MATRIX filter has been verified against complete scoped catalogs.
    if field_type == "MATRIX":
        params["type"] = field_type
    pages = _ml._catalog_pages(s, "/data-fields", params, keys=("datafields",), page_size=50)
    return _ml._filter_datafields((item for page in pages for item in page), field_type, search, limit)


def _filter_datafields(rows, field_type, search, limit):
    fields = []
    search = search.lower() if search else ""
    for item in rows:
        field_id = item.get("id")
        if not isinstance(field_id, str) or not field_id:
            raise _ml.ServiceError("catalog returned an invalid field ID")
        if field_type and item.get("type") != field_type:
            continue
        if search and search not in (
            field_id + " " + (item.get("description") or "")
        ).lower():
            continue
        fields.append(item)
        if limit and len(fields) >= limit:
            break
    return fields


def search_datafield(s, field, scope, *, deadline):
    """Read one scoped catalog search page for an exact field ID, without caching."""
    params = {key: scope[key] for key in ("instrumentType", "region", "delay", "universe")}
    params.update(search=field, limit=50, offset=0)
    payload = _ml.request_json(s, "GET", "/data-fields", params=params, max_retries=1, deadline=deadline)
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise _ml.ServiceError("field search returned no valid results page")
    count = payload.get("count")
    if type(count) is not int or count < len(rows):
        count = None
    return {"matches": [row for row in rows if row.get("id") == field],
            "returned_count": len(rows), "reported_count": count,
            "complete": count == len(rows) and not bool(payload.get("next"))}


def get_dataset(s, dataset_id, *, refresh=None, report=None, deadline=None):
    """Fetch one dataset's metadata directly via ``GET /data-sets/{id}`` (cached per account)."""
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a nonempty string")
    clean_id = dataset_id.strip()
    limits = {} if deadline is None else {"deadline": deadline}
    return _ml._cached_account_catalog(
        s, "dataset", {"id": clean_id},
        lambda session: _ml.request_json(
            session, "GET", "/data-sets/" + quote(clean_id, safe=""), **limits
        ),
        refresh, report,
    )


def get_datafield(s, field_id, *, refresh=None, report=None, deadline=None):
    """Fetch one datafield's metadata directly via ``GET /data-fields/{id}`` (cached per account)."""
    if not isinstance(field_id, str) or not field_id.strip():
        raise ValueError("field_id must be a nonempty string")
    clean_id = field_id.strip()
    limits = {} if deadline is None else {"deadline": deadline}
    return _ml._cached_account_catalog(
        s, "datafield", {"id": clean_id},
        lambda session: _ml.request_json(
            session, "GET", "/data-fields/" + quote(clean_id, safe=""), **limits
        ),
        refresh, report,
    )


def field_ids(fields, field_type=None, search=""):
    """Return the ``id`` of each field, optionally filtered by type/substring."""
    return [
        f["id"]
        for f in fields
        if f.get("id")
        and (not field_type or f.get("type") == field_type)
        and (
            not search
            or search.lower() in ((f.get("id") or "") + " " + (f.get("description") or "")).lower()
        )
    ]


def fields_to_df(fields):
    """Convert a list of field metadata dicts into a Polars DataFrame."""
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("polars is required for fields_to_df(); pip install polars") from exc
    rows = list(fields or [])
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None, strict=False)


def process_datafields(fields):
    """Turn field metadata into directly usable FASTEXPR snippets.

    Accepts the ``list[dict]`` from :func:`get_datafields` (or a Polars/pandas
    DataFrame). MATRIX fields are winsorized + backfilled, VECTOR fields are
    converted to a scalar via ``vec_avg``.
    """
    if hasattr(fields, "to_dicts"):
        fields = fields.to_dicts()
    elif hasattr(fields, "to_dict"):
        fields = fields.to_dict("records")
    matrix = [
        'winsorize(ts_backfill(%s, 120), std=4)' % f["id"]
        for f in fields if f.get("type") == "MATRIX" and f.get("id")
    ]
    vector = [
        'vec_avg(%s)' % f["id"]
        for f in fields if f.get("type") == "VECTOR" and f.get("id")
    ]
    return matrix + vector
