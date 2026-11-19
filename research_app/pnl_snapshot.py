"""Rebuildable training PnL snapshot for bulk local analysis."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import cache_store

from .runtime import CliError


_VERSION = 1


def read_series(path):
    """Match the ledger's float conversion and distinguish absent from empty."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            values = {day: float(value) for day, value in payload.items()}
            if all(math.isfinite(value) for value in values.values()):
                return values
    except (OSError, TypeError, ValueError, OverflowError):
        pass
    return None


def _stamp(path):
    try:
        metadata = path.stat()
    except OSError:
        return None
    return [metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns]


def _sources(pnl_dir, cutoffs):
    rows = []
    for alpha_id, cutoff in sorted(cutoffs.items()):
        stamp = _stamp(pnl_dir / f"{alpha_id}.json")
        if stamp is not None:
            rows.append((alpha_id, cutoff, stamp))
    signature = hashlib.sha256(json.dumps([_VERSION, str(pnl_dir), rows],
                                          separators=(",", ":")).encode()).hexdigest()
    return rows, signature


def _query(pl, path, present, selected):
    values = {alpha_id: [] for alpha_id in selected if alpha_id in present}
    if values:
        rows = (pl.scan_parquet(path).filter(pl.col("alpha_id").is_in(list(values)))
                .group_by("alpha_id").agg(pl.col("pnl").sort_by("day"))
                .collect(engine="streaming").iter_rows())
        values.update(rows)
    return values


def _rebuild(pl, path, manifest, pnl_dir, cutoffs, sources, signature):
    present = []
    alpha_ids, dates, pnls = [], [], []
    for alpha_id, cutoff, _stamp_value in sources:
        series = read_series(pnl_dir / f"{alpha_id}.json")
        if series is None:
            continue
        present.append(alpha_id)
        if cutoff:
            for day, value in series.items():
                if day < cutoff:
                    alpha_ids.append(alpha_id)
                    dates.append(day)
                    pnls.append(value)
    frame = pl.DataFrame({"alpha_id": alpha_ids, "day": dates, "pnl": pnls},
                         schema={"alpha_id": pl.String, "day": pl.String, "pnl": pl.Float64})
    fd, temporary = tempfile.mkstemp(prefix=".pnl-train-", suffix=".parquet", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            frame.write_parquet(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if _sources(pnl_dir, cutoffs)[1] != signature:
            return None
        os.replace(temporary, path)
        cache_store.write_json_atomic(manifest, {"sources": signature, "parquet": _stamp(path), "present": present})
        return set(present)
    finally:
        Path(temporary).unlink(missing_ok=True)


def training_values(pnl_dir, path, cutoffs, selected):
    """Return date-ordered rows for the selected alphas and cache hit evidence."""
    try:
        import polars as pl
    except ImportError as exc:
        raise CliError("INVALID_INPUT", "--polars requires Polars",
                       'install the optional package with pip install "polars>=2.1,<3"') from exc

    pnl_dir = pnl_dir.resolve()
    manifest = path.with_suffix(".json")
    errors = (OSError, pl.exceptions.PolarsError)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with cache_store.file_lock(path.with_suffix(".lock"), timeout=30):
            sources, signature = _sources(pnl_dir, cutoffs)
            saved, _ = cache_store.load_json(manifest)
            stamp = _stamp(path)
            if (isinstance(saved, dict) and saved.get("sources") == signature and stamp is not None
                    and saved.get("parquet") == stamp and isinstance(saved.get("present"), list)
                    and all(isinstance(item, str) for item in saved["present"])):
                try:
                    return _query(pl, path, set(saved["present"]), selected), "hit"
                except errors:
                    pass
            present = _rebuild(pl, path, manifest, pnl_dir, cutoffs, sources, signature)
            if present is not None:
                return _query(pl, path, present, selected), "rebuilt"
            return None, "source_changed"
    except (OSError, pl.exceptions.PolarsError, cache_store.CacheError) as exc:
        raise CliError("SERVICE_ERROR", "could not access the training PnL snapshot", str(exc)) from exc
