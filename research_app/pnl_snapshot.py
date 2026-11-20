"""Rebuildable training PnL snapshot for bulk local analysis."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import cache_store

from .runtime import CliError


_VERSION = 3


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
    rows = {}
    for alpha_id, cutoff in sorted(cutoffs.items()):
        stamp = _stamp(pnl_dir / f"{alpha_id}.json")
        if stamp is not None:
            rows[alpha_id] = [cutoff, stamp]
    return rows, _source_signature(pnl_dir, rows)


def _source_signature(pnl_dir, rows):
    encoded = json.dumps([_VERSION, str(pnl_dir), rows], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _parquet_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _cached_snapshot(manifest, path, pnl_dir):
    saved, _ = cache_store.load_json(manifest)
    if not isinstance(saved, dict):
        return None
    sources = saved.get("source_files")
    stamp = _stamp(path)
    sharpes = saved.get("half_sharpes")
    present = saved.get("present")
    if (stamp is None or saved.get("parquet") != stamp or not isinstance(sources, dict)
            or saved.get("sources") != _source_signature(pnl_dir, sources)
            or not isinstance(sharpes, dict) or not sharpes.keys() <= sources.keys()
            or not isinstance(present, list) or any(not isinstance(alpha_id, str) for alpha_id in present)
            or set(present) != sharpes.keys()):
        return None
    if any(pair is not None and (not isinstance(pair, list) or len(pair) != 2
           or any(value is not None and (type(value) is not float or not math.isfinite(value))
                  for value in pair))
           for pair in sharpes.values()):
        return None
    try:
        if _parquet_digest(path) == saved.get("parquet_sha256"):
            return saved
    except OSError:
        pass
    return None


def _rebuild(pl, path, manifest, pnl_dir, cutoffs, sources, signature, previous):
    from .experiments import train_half_sharpes

    old_sources = previous["source_files"] if previous is not None else {}
    changed = {alpha_id for alpha_id in sources.keys() | old_sources.keys()
               if sources.get(alpha_id) != old_sources.get(alpha_id)}
    sharpes = {} if previous is None else {
        alpha_id: pair for alpha_id, pair in previous["half_sharpes"].items() if alpha_id not in changed
    }
    alpha_ids, dates, pnls = [], [], []
    for alpha_id, (cutoff, _) in sources.items():
        if alpha_id not in changed:
            continue
        series = read_series(pnl_dir / f"{alpha_id}.json")
        if series is None:
            continue
        training_days = sorted(day for day in series if day < cutoff) if cutoff else []
        training_values = [series[day] for day in training_days]
        alpha_ids.extend([alpha_id] * len(training_days))
        dates.extend(training_days)
        pnls.extend(training_values)
        try:
            pair = train_half_sharpes(training_values)
        except ArithmeticError:
            pair = None
        # Defer extreme values to the legacy reader if this row is selected.
        # A rebuild must not fail on a row that the current report excludes.
        sharpes[alpha_id] = (pair if pair is not None and
                             all(value is None or math.isfinite(value) for value in pair) else None)
    delta = pl.DataFrame({"alpha_id": alpha_ids, "day": dates, "pnl": pnls},
                         schema={"alpha_id": pl.String, "day": pl.String, "pnl": pl.Float64}).lazy()
    if old_sources.keys() - changed:
        retained = pl.scan_parquet(path).filter(~pl.col("alpha_id").is_in(list(changed)))
        # Both inputs are sorted and contain disjoint Alpha IDs.
        delta = retained.merge_sorted(delta, key=["alpha_id", "day"])
    fd, temporary = tempfile.mkstemp(prefix=".pnl-train-", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        delta.sink_parquet(temporary, engine="streaming", maintain_order=True, sync_on_close="all")
        digest = _parquet_digest(temporary)
        if _sources(pnl_dir, cutoffs)[1] != signature:
            return None
        os.replace(temporary, path)
        cache_store.write_json_atomic(manifest, {
            "sources": signature,
            "source_files": sources,
            "parquet": _stamp(path),
            "parquet_sha256": digest,
            "present": list(sharpes),
            "half_sharpes": sharpes,
        })
        return sharpes
    finally:
        Path(temporary).unlink(missing_ok=True)


def training_half_sharpes(pnl_dir, path, cutoffs, selected):
    """Return exact train-half values for the selected alphas and cache hit evidence."""
    pnl_dir = pnl_dir.resolve()
    manifest = path.with_suffix(".json")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with cache_store.file_lock(path.with_suffix(".lock"), timeout=30):
            sources, signature = _sources(pnl_dir, cutoffs)
            previous = _cached_snapshot(manifest, path, pnl_dir)
            sharpes = previous["half_sharpes"] if previous is not None else None
            state = "hit"
            if previous is None or previous["sources"] != signature:
                try:
                    import polars as pl
                except ImportError as exc:
                    raise CliError("INVALID_INPUT", "--polars requires Polars",
                                   'install the optional package with pip install "polars>=2.1,<3"') from exc
                try:
                    sharpes = _rebuild(pl, path, manifest, pnl_dir, cutoffs, sources, signature, previous)
                except pl.exceptions.PolarsError as exc:
                    raise cache_store.CacheError(str(exc)) from exc
                state = "rebuilt"
            if sharpes is None:
                return None, "source_changed"
            return {alpha_id: sharpes[alpha_id] for alpha_id in selected if alpha_id in sharpes}, state
    except (OSError, cache_store.CacheError) as exc:
        raise CliError("SERVICE_ERROR", "could not access the training PnL snapshot", str(exc)) from exc
