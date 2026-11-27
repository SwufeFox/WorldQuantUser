"""Store daily PnL in immutable Parquet files indexed by a SQLite WAL manifest."""

from __future__ import annotations

from contextlib import closing, contextmanager, suppress
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from uuid import uuid4

import cache_store

_EVIDENCE_FORMAT = "worldquantuser.pnl-evidence"
_META_DB = "meta.sqlite"
_DEFAULT_MAX_DELTAS = 24

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pnl_evidence (
    alpha_id TEXT PRIMARY KEY,
    format TEXT NOT NULL,
    complete INTEGER,
    as_of TEXT,
    data_sha256 TEXT NOT NULL,
    pages INTEGER,
    start_day TEXT,
    end_day TEXT,
    missing_values INTEGER,
    points INTEGER NOT NULL,
    delta_file TEXT NOT NULL DEFAULT '',
    previous_json TEXT,
    updated_at REAL NOT NULL
)
"""

_UPSERT = """
INSERT INTO pnl_evidence (
    alpha_id, format, complete, as_of, data_sha256, pages, start_day,
    end_day, missing_values, points, delta_file, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(alpha_id) DO UPDATE SET
    format=excluded.format, complete=excluded.complete, as_of=excluded.as_of,
    data_sha256=excluded.data_sha256, pages=excluded.pages, start_day=excluded.start_day,
    end_day=excluded.end_day, missing_values=excluded.missing_values, points=excluded.points,
    delta_file=excluded.delta_file, updated_at=excluded.updated_at
"""


def _digest(series):
    blob = json.dumps(series, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def canonical_digest(series):
    """Compute the PnL digest after normalizing values to finite floats."""
    return _digest(_normalize_series(series))


def matching_legacy_evidence(metadata, alpha_id, series):
    """Return the current or previous legacy proof bound to this exact JSON payload."""
    if not isinstance(metadata, dict) or metadata.get("format") != _EVIDENCE_FORMAT:
        return None
    _normalize_series(series)
    digest = _digest(series)
    for candidate in (metadata, metadata.get("previous")):
        if (isinstance(candidate, dict) and candidate.get("format") == _EVIDENCE_FORMAT
                and candidate.get("alpha_id") == alpha_id and candidate.get("data_sha256") == digest):
            return candidate
    return None


def _normalize_series(series):
    if not isinstance(series, dict):
        raise ValueError("series must be a dict mapping dates to numbers")
    result = {}
    for day, value in series.items():
        if not isinstance(day, str) or not day:
            raise ValueError("PnL dates must be nonempty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"PnL value for {day!r} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"PnL value for {day!r} must be finite")
        result[day] = number
    return result


def _import_polars():
    try:
        import polars
    except ImportError:
        return None
    return polars


@contextmanager
def _manifest(cache_path, *, write=False, lock_missing=False, timeout=30):
    db_path = cache_path / _META_DB
    if not write and not db_path.is_file() and not (lock_missing and cache_path.is_dir()):
        yield None
        return
    # Prevent retirement of files while a reader is using references from SQLite.
    with cache_store.file_lock(cache_path / "compact.lock", timeout=timeout):
        if not write and not db_path.is_file():
            yield None
            return
        try:
            with closing(sqlite3.connect(db_path, timeout=timeout)) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA synchronous=FULL")
                if write:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute(_SCHEMA)
                yield conn
        except sqlite3.Error as exc:
            raise cache_store.CacheError(f"could not access PnL manifest: {exc}") from exc


def _records(conn, alpha_ids=None):
    if conn is None:
        return []
    if alpha_ids is None:
        return conn.execute("SELECT * FROM pnl_evidence ORDER BY alpha_id").fetchall()
    ids = list(dict.fromkeys(alpha_ids))
    rows = []
    for offset in range(0, len(ids), 500):
        batch = ids[offset:offset + 500]
        params = ",".join("?" for _ in batch)
        rows.extend(conn.execute(f"SELECT * FROM pnl_evidence WHERE alpha_id IN ({params})", batch))
    return rows


def _record_path(cache_path, reference):
    if not reference:
        # Rows compacted by earlier drafts pointed to the fixed base filename.
        return cache_path / "base.parquet"
    if Path(reference).name != reference or not reference.endswith(".parquet"):
        raise cache_store.CacheError("invalid Parquet reference in PnL manifest")
    if reference.startswith("base-"):
        return cache_path / reference
    return cache_path / "deltas" / reference


def _delta_reference(reference):
    return reference and not reference.startswith("base-")


def _frame(pl, series_by_alpha):
    ids, days, values = [], [], []
    for alpha_id, series in sorted(series_by_alpha.items()):
        for day, value in sorted(series.items()):
            ids.append(alpha_id)
            days.append(day)
            values.append(value)
    return pl.DataFrame({"alpha_id": ids, "day": days, "pnl": values},
                        schema={"alpha_id": pl.String, "day": pl.String, "pnl": pl.Float64})


def _publish_frame(frame, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.stem}-", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        try:
            frame.write_parquet(temporary)
        except _import_polars().exceptions.PolarsError as exc:
            raise cache_store.CacheError(f"could not write PnL Parquet: {exc}") from exc
        with open(temporary, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        cache_store.fsync_directory(target.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def _publish_delta_file(cache_dir, alpha_id, series):
    pl = _import_polars()
    path = Path(cache_dir) / "deltas" / f"{uuid4().hex}.parquet"
    return _publish_frame(_frame(pl, {alpha_id: series}), path)


def _values(alpha_id, series, metadata, reference, updated_at):
    complete = metadata.get("complete")
    return (alpha_id, _EVIDENCE_FORMAT, int(complete) if type(complete) is bool else None,
            metadata.get("as_of"), canonical_digest(series), metadata.get("pages"),
            metadata.get("start"), metadata.get("end"), metadata.get("missing_values"),
            len(series), reference, updated_at)


def _retire(cache_path, conn, references):
    for reference in set(references):
        active = conn.execute("SELECT 1 FROM pnl_evidence WHERE delta_file = ? LIMIT 1", (reference,))
        if active.fetchone() is None:
            with suppress(OSError):
                _record_path(cache_path, reference).unlink(missing_ok=True)


def _write_legacy(cache_path, alpha_id, series, metadata):
    path = cache_path / f"{alpha_id}.json"
    sidecar = cache_path / ".meta" / path.name
    previous, _ = cache_store.load_json(path)
    prior_metadata, _ = cache_store.load_json(sidecar)
    with suppress(TypeError, ValueError, OverflowError):
        match = matching_legacy_evidence(prior_metadata, alpha_id, previous)
        if match:
            metadata["previous"] = {key: value for key, value in match.items() if key != "previous"}
    cache_store.write_json_atomic(sidecar, metadata)
    cache_store.write_json_atomic(path, series)
    return {"storage": "json_fallback", "path": str(path), "metadata_path": str(sidecar)}


def write_pnl(cache_dir, alpha_id, series, metadata, *, max_deltas=_DEFAULT_MAX_DELTAS, timeout=30):
    """Publish one whole recordset, then commit its file reference and evidence together."""
    cache_path = Path(cache_dir)
    normalized = _normalize_series(series)
    digest = canonical_digest(normalized)
    metadata = {**(metadata or {}), "format": _EVIDENCE_FORMAT, "alpha_id": alpha_id,
                "data_sha256": digest, "points": len(normalized)}
    if _import_polars() is None:
        with cache_store.file_lock(cache_path / "compact.lock", timeout=timeout):
            if (cache_path / _META_DB).is_file():
                raise cache_store.CacheError("Polars is required to update an existing Parquet PnL store")
            receipt = _write_legacy(cache_path, alpha_id, normalized, metadata)
    else:
        with _manifest(cache_path, write=True, timeout=timeout) as conn:
            prior = _records(conn, [alpha_id])
            delta = _publish_delta_file(cache_path, alpha_id, normalized)
            with conn:
                conn.execute(_UPSERT, _values(alpha_id, normalized, metadata, delta.name, time.time()))
            _retire(cache_path, conn, (row["delta_file"] for row in prior))
            # A maintenance failure does not invalidate the already committed write.
            with suppress(OSError, cache_store.CacheError, sqlite3.Error):
                _compact(cache_path, conn, max_deltas=max_deltas)
            reference = _records(conn, [alpha_id])[0]["delta_file"]
            receipt = {"storage": "parquet", "path": str(_record_path(cache_path, reference)),
                       "metadata_path": str(cache_path / _META_DB)}
            if _delta_reference(reference):
                receipt["delta_file"] = reference
    return {**receipt, "data_sha256": digest, "points": len(normalized)}


def _read_records(cache_path, rows):
    pl = _import_polars()
    if pl is None or not rows:
        return {}
    by_reference = {}
    for row in rows:
        by_reference.setdefault(row["delta_file"], {})[row["alpha_id"]] = row
    output = {}
    for reference, records in by_reference.items():
        try:
            frame = (pl.scan_parquet(_record_path(cache_path, reference))
                     .filter(pl.col("alpha_id").is_in(list(records))).collect())
            frame = frame.select("alpha_id", "day", "pnl")
        except (OSError, pl.exceptions.PolarsError):
            continue
        values = {alpha_id: {} for alpha_id in records}
        for alpha_id, day, pnl in frame.iter_rows():
            values[alpha_id][day] = pnl
        for alpha_id, series in values.items():
            row = records[alpha_id]
            try:
                if len(series) == row["points"] and canonical_digest(series) == row["data_sha256"]:
                    output[alpha_id] = series
            except (TypeError, ValueError, OverflowError):
                continue
    return output


def _compact(cache_path, conn, *, max_deltas, force=False):
    references = {row[0] for row in conn.execute("SELECT DISTINCT delta_file FROM pnl_evidence")}
    delta_count = sum(bool(_delta_reference(reference)) for reference in references)
    if not delta_count or (not force and delta_count < max(1, int(max_deltas))):
        return {"compacted": False, "deltas_merged": 0}
    rows = _records(conn)
    series = _read_records(cache_path, rows)
    if len(series) != len(rows):
        raise cache_store.CacheError("compaction could not verify every committed PnL recordset")
    target = cache_path / f"base-{uuid4().hex}.parquet"
    _publish_frame(_frame(_import_polars(), series), target)
    with conn:
        conn.execute("UPDATE pnl_evidence SET delta_file = ?", (target.name,))
    _retire(cache_path, conn, references)
    return {"compacted": True, "deltas_merged": delta_count, "base_path": str(target)}


def maybe_compact(cache_dir, *, max_deltas=_DEFAULT_MAX_DELTAS, force=False):
    """Merge committed data into a new base generation and retire its predecessors."""
    cache_path = Path(cache_dir)
    with _manifest(cache_path) as conn:
        if conn is None or _import_polars() is None:
            return {"compacted": False, "deltas_merged": 0}
        return _compact(cache_path, conn, max_deltas=max_deltas, force=force)


@contextmanager
def backup_members(cache_dir):
    """Pin the manifest while yielding committed Parquet paths relative to the cache."""
    cache_path = Path(cache_dir)
    with _manifest(cache_path, lock_missing=True) as conn:
        if conn is None:
            yield ()
        else:
            references = conn.execute("SELECT DISTINCT delta_file FROM pnl_evidence ORDER BY delta_file")
            yield tuple(_record_path(cache_path, row[0]).relative_to(cache_path) for row in references)


def committed_stamps(cache_dir, alpha_ids=None):
    """Return the manifest stamps of active recordsets for incremental consumers."""
    cache_path = Path(cache_dir)
    with _manifest(cache_path) as conn:
        return {row["alpha_id"]: ["parquet", row["data_sha256"], row["updated_at"]]
                for row in _records(conn, alpha_ids)}


def read_committed(cache_dir, alpha_ids=None):
    """Read verified committed series and their declared IDs in one snapshot."""
    cache_path = Path(cache_dir)
    with _manifest(cache_path) as conn:
        rows = _records(conn, alpha_ids)
        return _read_records(cache_path, rows), {row["alpha_id"] for row in rows}


def read_pnl(cache_dir, alpha_id, *, max_age_hours=None):
    """Load one verified series, falling back to JSON for an unregistered alpha."""
    if not cache_dir or not alpha_id or (max_age_hours is not None and max_age_hours <= 0):
        return None
    cache_path = Path(cache_dir)
    oldest = None if max_age_hours is None else time.time() - max_age_hours * 3600
    with _manifest(cache_path) as conn:
        rows = _records(conn, [alpha_id])
        if rows:
            if oldest is not None and rows[0]["updated_at"] <= oldest:
                return None
            return _read_records(cache_path, rows).get(alpha_id)
    path = cache_path / f"{alpha_id}.json"
    try:
        if oldest is not None and path.stat().st_mtime <= oldest:
            return None
        payload, error = cache_store.load_json(path)
        if error is None:
            return _normalize_series(payload)
    except (OSError, TypeError, ValueError, OverflowError):
        pass
    return None


def pnl_evidence_status(cache_dir, alpha_id, series=None):
    """Return completeness only for values bound to the committed PnL evidence."""
    if not cache_dir or not alpha_id:
        return None
    cache_path = Path(cache_dir)
    with _manifest(cache_path) as conn:
        rows = _records(conn, [alpha_id])
        if not rows:
            return None
        row = rows[0]
        loaded_series = series if series is not None else _read_records(cache_path, rows).get(alpha_id)
    evidence = {"source": "cache", "path": str(_record_path(cache_path, row["delta_file"])),
                "metadata_path": str(cache_path / _META_DB), "complete": None, "as_of": None,
                "error": "missing", "points": 0}
    if loaded_series is None:
        return evidence
    try:
        digest = canonical_digest(loaded_series)
    except (TypeError, ValueError, OverflowError):
        return dict(evidence, complete=False, error="invalid_pnl_values")
    evidence["points"] = len(loaded_series)
    if (digest != row["data_sha256"] or len(loaded_series) != row["points"]
            or row["format"] != _EVIDENCE_FORMAT):
        return dict(evidence, complete=False, error="pnl_metadata_mismatch")
    return dict(evidence, complete=None if row["complete"] is None else bool(row["complete"]),
                as_of=row["as_of"], pages=row["pages"], start=row["start_day"], end=row["end_day"],
                missing_values=row["missing_values"], error=None)


def read_parquet_series_map(cache_dir, alpha_ids=None, *, start_day=None, end_day=None, cutoffs=None):
    """Return nonempty committed series sliced to the requested date bounds."""
    series, _ = read_committed(cache_dir, alpha_ids)
    output = {}
    for alpha_id, values in series.items():
        bounds = [bound for bound in (end_day, (cutoffs or {}).get(alpha_id)) if bound]
        end = min(bounds) if bounds else None
        sliced = {day: pnl for day, pnl in values.items()
                  if (not start_day or day >= start_day) and (not end or day < end)}
        if sliced:
            output[alpha_id] = sliced
    return output


def _archive_files(archive, paths):
    # Publish both archive copies before retiring either source. Exclusive links
    # also protect an archive shared with another migration from overwrite.
    copies = [(source, archive / relative, source.read_bytes())
              for source, relative in paths if source.is_file()]
    if any(target.exists() and target.read_bytes() != payload for _, target, payload in copies):
        return False
    for _, target, payload in copies:
        try:
            cache_store.link_bytes_exclusive(target.parent, target, payload)
        except FileExistsError:
            if target.read_bytes() != payload:
                return False
    for source, _, _ in copies:
        source.unlink()
    return True


def migrate_legacy_pnl_dir(pnl_dir, *, archive_dir=None, dry_run=False):
    """Import legacy JSON without replacing a newer recordset or inventing completeness."""
    cache_path = Path(pnl_dir).resolve()
    files = sorted(path for path in cache_path.glob("*.json") if not path.name.startswith(".") and path.is_file())
    result = {"migrated": 0, "skipped": 0, "archived_to": None}
    if dry_run:
        result["dry_run"] = True
    if not files:
        return result
    pl = _import_polars()
    if pl is None:
        return dict(result, skipped=len(files), reason="polars_unavailable")
    archive = Path(archive_dir).resolve() if archive_dir is not None and not dry_run else None
    with _manifest(cache_path, write=not dry_run) as conn:
        existing = {row["alpha_id"]: row for row in _records(conn, [path.stem for path in files])}
        to_import, to_archive = {}, []
        for path in files:
            alpha_id = path.stem
            metadata_path = cache_path / ".meta" / path.name
            try:
                raw, error = cache_store.load_json(path)
                if error:
                    result["skipped"] += 1
                    continue
                series = _normalize_series(raw)
                metadata, _ = cache_store.load_json(metadata_path)
                proof = matching_legacy_evidence(metadata, alpha_id, raw) or {}
                modified = path.stat().st_mtime
            except (OSError, TypeError, ValueError, OverflowError):
                result["skipped"] += 1
                continue
            if alpha_id in existing:
                result["skipped"] += 1
                if existing[alpha_id]["data_sha256"] != canonical_digest(series):
                    continue
                if _read_records(cache_path, [existing[alpha_id]]).get(alpha_id) is None:
                    continue
            else:
                to_import[alpha_id] = (series, proof, modified)
            to_archive.append((path, metadata_path))
        if to_import and not dry_run:
            target = cache_path / f"base-{uuid4().hex}.parquet"
            _publish_frame(_frame(pl, {aid: record[0] for aid, record in to_import.items()}), target)
            with conn:
                conn.executemany(_UPSERT, (_values(aid, series, proof, target.name, modified)
                                           for aid, (series, proof, modified) in to_import.items()))
        result["migrated"] = len(to_import)
        if archive is not None and archive != cache_path:
            for path, metadata_path in to_archive:
                if _archive_files(archive, [(metadata_path, Path(".meta") / path.name), (path, Path(path.name))]):
                    result["archived_to"] = str(archive)
                else:
                    result["archive_conflicts"] = result.get("archive_conflicts", 0) + 1
    return result
