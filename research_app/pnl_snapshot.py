"""Rebuildable training PnL snapshot for bulk local analysis."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import cache_store

from .runtime import CliError


_VERSION = 4


def read_series(path, *, raise_io_errors=False):
    """Match the ledger's float conversion and distinguish absent from empty."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            values = {day: float(value) for day, value in payload.items()}
            if all(math.isfinite(value) for value in values.values()):
                return values
    except OSError:
        if raise_io_errors:
            raise
    except (TypeError, ValueError, OverflowError):
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
        try:
            series = read_series(pnl_dir / f"{alpha_id}.json", raise_io_errors=True)
        except OSError:
            # A failed read has not established that this source is missing or malformed.
            return None, "source_unavailable"
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
            return None, "source_changed"
        os.replace(temporary, path)
        cache_store.write_json_atomic(manifest, {
            "sources": signature,
            "source_files": sources,
            "parquet": _stamp(path),
            "parquet_sha256": digest,
            "present": list(sharpes),
            "half_sharpes": sharpes,
        })
        return sharpes, "rebuilt"
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
                    sharpes, state = _rebuild(pl, path, manifest, pnl_dir, cutoffs, sources, signature, previous)
                except pl.exceptions.PolarsError as exc:
                    raise cache_store.CacheError(str(exc)) from exc
            if sharpes is None:
                return None, state
            return {alpha_id: sharpes[alpha_id] for alpha_id in selected if alpha_id in sharpes}, state
    except (OSError, cache_store.CacheError) as exc:
        raise CliError("SERVICE_ERROR", "could not access the training PnL snapshot", str(exc)) from exc


_UNIFIED_CACHE_VERSION = 1
_UNIFIED_SUBDIR = ".parquet_cache"
_UNIFIED_PARQUET_NAME = "pnl.parquet"
_UNIFIED_MANIFEST_NAME = "pnl.json"
_UNIFIED_LOCK_NAME = "pnl.lock"
_MIN_COMPLETE_YEAR_POINTS = 145


def _import_polars():
    try:
        import polars as pl
        return pl
    except ImportError:
        return None


def _scan_pnl_stamps(pnl_dir, alpha_ids=None):
    stamps = {}
    if alpha_ids is None:
        if not pnl_dir.is_dir():
            return stamps
        for path in sorted(pnl_dir.glob("*.json")):
            if path.name.startswith(".") or not path.is_file():
                continue
            stamp = _stamp(path)
            if stamp is not None:
                stamps[path.stem] = stamp
        return stamps
    for alpha_id in sorted({aid for aid in alpha_ids if isinstance(aid, str) and aid}):
        stamp = _stamp(pnl_dir / f"{alpha_id}.json")
        if stamp is not None:
            stamps[alpha_id] = stamp
    return stamps


def _load_unified_manifest(manifest_path, parquet_path):
    saved, err = cache_store.load_json(manifest_path)
    if err or not isinstance(saved, dict):
        return None
    if saved.get("version") != _UNIFIED_CACHE_VERSION:
        return None
    files = saved.get("files")
    pq_stamp = _stamp(parquet_path)
    if pq_stamp is None or saved.get("parquet") != pq_stamp or not isinstance(files, dict):
        return None
    return saved


def sync_pnl_parquet(pnl_dir, alpha_ids=None):
    """Synchronize ``<pnl_dir>/.parquet_cache/pnl.parquet`` incrementally with ``*.json`` PnL files."""
    pnl_dir = Path(pnl_dir)
    if not pnl_dir.is_dir():
        return {"state": "empty", "parquet_path": None, "files": 0, "reloaded": 0, "removed": 0}
    pl = _import_polars()
    if pl is None:
        return {"state": "polars_unavailable", "parquet_path": None, "files": 0, "reloaded": 0, "removed": 0}

    resolved_dir = pnl_dir.resolve()
    cache_dir = resolved_dir / _UNIFIED_SUBDIR
    parquet_path = cache_dir / _UNIFIED_PARQUET_NAME
    manifest_path = cache_dir / _UNIFIED_MANIFEST_NAME
    lock_path = cache_dir / _UNIFIED_LOCK_NAME

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with cache_store.file_lock(lock_path, timeout=30):
            target_set = None if alpha_ids is None else sorted(
                {aid for aid in alpha_ids if isinstance(aid, str) and aid}
            )
            current_stamps = _scan_pnl_stamps(resolved_dir, target_set)
            manifest = _load_unified_manifest(manifest_path, parquet_path)
            old_files = manifest["files"] if manifest is not None else {}

            if target_set is None:
                changed_ids = sorted(aid for aid, st in current_stamps.items() if old_files.get(aid) != st)
                removed_ids = sorted(aid for aid in old_files if aid not in current_stamps)
                merged_files = current_stamps
            else:
                changed_ids = sorted(aid for aid, st in current_stamps.items() if old_files.get(aid) != st)
                removed_ids = sorted(aid for aid in target_set if aid in old_files and aid not in current_stamps)
                merged_files = {k: v for k, v in old_files.items() if k not in set(removed_ids)}
                merged_files.update(current_stamps)

            if manifest is not None and not changed_ids and not removed_ids:
                return {
                    "state": "hit",
                    "parquet_path": parquet_path,
                    "files": len(old_files),
                    "reloaded": 0,
                    "removed": 0,
                }

            # When prior manifest is missing/stale on a full sync, rebuild from all disk files.
            can_increment = manifest is not None
            if not can_increment and target_set is None:
                changed_ids = sorted(current_stamps)
                removed_ids = []
            elif not can_increment and target_set is not None:
                # Cold build on subset: start fresh with target_set only.
                changed_ids = sorted(current_stamps)
                removed_ids = []
                merged_files = dict(current_stamps)

            row_aids, row_days, row_pnls = [], [], []
            for aid in changed_ids:
                try:
                    series = read_series(resolved_dir / f"{aid}.json", raise_io_errors=True)
                except OSError:
                    return {
                        "state": "source_unavailable",
                        "parquet_path": None,
                        "files": len(merged_files),
                        "reloaded": 0,
                        "removed": 0,
                    }
                if not series:
                    continue
                for day, value in sorted(series.items()):
                    if isinstance(day, str) and len(day) >= 4:
                        row_aids.append(aid)
                        row_days.append(day)
                        row_pnls.append(float(value))

            delta_df = pl.DataFrame(
                {"alpha_id": row_aids, "day": row_days, "pnl": row_pnls},
                schema={"alpha_id": pl.String, "day": pl.String, "pnl": pl.Float64},
            )
            delta_lf = delta_df.lazy()

            drop_ids = sorted(set(changed_ids) | set(removed_ids))
            if can_increment and parquet_path.exists():
                kept_lf = pl.scan_parquet(parquet_path)
                if drop_ids:
                    kept_lf = kept_lf.filter(~pl.col("alpha_id").is_in(drop_ids))
                combined_lf = pl.concat([kept_lf, delta_lf]).sort(["alpha_id", "day"])
            else:
                combined_lf = delta_lf

            fd, temporary = tempfile.mkstemp(prefix=".pnl-", suffix=".parquet", dir=cache_dir)
            os.close(fd)
            try:
                combined_lf.sink_parquet(temporary, engine="streaming", maintain_order=True, sync_on_close="all")
                # Re-verify source stamps did not race mid-write.
                if _scan_pnl_stamps(resolved_dir, target_set) != current_stamps:
                    return {
                        "state": "source_changed",
                        "parquet_path": None,
                        "files": len(merged_files),
                        "reloaded": 0,
                        "removed": 0,
                    }
                os.replace(temporary, parquet_path)
                cache_store.write_json_atomic(manifest_path, {
                    "version": _UNIFIED_CACHE_VERSION,
                    "parquet": _stamp(parquet_path),
                    "files": merged_files,
                })
                return {
                    "state": "rebuilt" if not can_increment else "updated",
                    "parquet_path": parquet_path,
                    "files": len(merged_files),
                    "reloaded": len(changed_ids),
                    "removed": len(removed_ids),
                }
            finally:
                Path(temporary).unlink(missing_ok=True)
    except (OSError, cache_store.CacheError) as exc:
        return {
            "state": "source_unavailable",
            "parquet_path": None,
            "error": str(exc),
            "files": 0,
            "reloaded": 0,
            "removed": 0,
        }
    except Exception as exc:  # noqa: BLE001 - PolarsError or unexpected cache read fault falls back to JSON
        pl_exc = getattr(getattr(pl, "exceptions", None), "PolarsError", None)
        if pl_exc is not None and isinstance(exc, pl_exc):
            return {
                "state": "source_unavailable",
                "parquet_path": None,
                "error": str(exc),
                "files": 0,
                "reloaded": 0,
                "removed": 0,
            }
        raise


def _filter_series(series, start_day=None, end_day=None, cutoff=None):
    upper = end_day
    if cutoff and (upper is None or cutoff < upper):
        upper = cutoff
    if not start_day and not upper:
        return dict(series)
    return {
        day: value for day, value in series.items()
        if (not start_day or day >= start_day) and (not upper or day < upper)
    }


def load_series_map(pnl_dir, alpha_ids=None, *, start_day=None, end_day=None, cutoffs=None):
    """Load ``{alpha_id: {day: float}}`` from incremental Parquet cache with optional date slicing."""
    pnl_path = Path(pnl_dir)
    if alpha_ids is not None:
        target_ids = list(dict.fromkeys(aid for aid in alpha_ids if isinstance(aid, str) and aid))
        if not target_ids:
            return {}
    else:
        target_ids = None

    if not pnl_path.is_dir():
        return {}

    pl = _import_polars()
    sync = sync_pnl_parquet(pnl_path, alpha_ids=target_ids) if pl is not None else {"parquet_path": None}
    parquet_path = sync.get("parquet_path")
    cutoff_map = {str(k): str(v) for k, v in (cutoffs or {}).items() if isinstance(k, str) and v}

    if pl is not None and parquet_path is not None and Path(parquet_path).exists():
        try:
            lf = pl.scan_parquet(parquet_path)
            if target_ids is not None:
                lf = lf.filter(pl.col("alpha_id").is_in(target_ids))
            if start_day:
                lf = lf.filter(pl.col("day") >= str(start_day))
            if end_day:
                lf = lf.filter(pl.col("day") < str(end_day))
            if cutoff_map:
                cut_df = pl.DataFrame(
                    {"alpha_id": list(cutoff_map.keys()), "_cutoff": list(cutoff_map.values())},
                    schema={"alpha_id": pl.String, "_cutoff": pl.String},
                ).lazy()
                lf = (
                    lf.join(cut_df, on="alpha_id", how="left")
                    .filter(pl.col("_cutoff").is_null() | (pl.col("day") < pl.col("_cutoff")))
                    .drop("_cutoff")
                )
            df = lf.collect()
            if df.is_empty():
                return {}
            out = {}
            partitions = df.partition_by("alpha_id", as_dict=True, maintain_order=True)
            for key, sub in partitions.items():
                aid = key[0] if isinstance(key, tuple) else key
                days = sub.get_column("day").to_list()
                pnls = sub.get_column("pnl").to_list()
                if days:
                    out[aid] = dict(zip(days, pnls))
            if target_ids is not None:
                return {aid: out[aid] for aid in target_ids if aid in out}
            return out
        except Exception:  # noqa: BLE001 - fall back to direct JSON reads on transient Parquet error
            pass

    fallback_ids = (
        target_ids
        if target_ids is not None
        else sorted(p.stem for p in pnl_path.glob("*.json") if not p.name.startswith(".") and p.is_file())
    )
    out = {}
    for aid in fallback_ids:
        series = read_series(pnl_path / f"{aid}.json")
        if not series:
            continue
        filtered = _filter_series(series, start_day=start_day, end_day=end_day, cutoff=cutoff_map.get(aid))
        if filtered:
            out[aid] = filtered
    return out


def yearly_train_batch(rows, pnl_dir):
    """Evaluate yearly train Sharpe breakdown for multiple ledger rows via unified Parquet + Polars."""
    import machine_lib as ml
    from datetime import date as _date
    from .stats import sharpe as _sharpe

    row_list = list(rows)
    if not row_list:
        return []

    pnl_path = Path(pnl_dir)
    results = [None] * len(row_list)
    valid_indices = []
    proofs = {}

    candidate_aids = sorted({
        row["alpha_id"]
        for row in row_list
        if row.get("alpha_id") and ml.valid_simulation_id(row["alpha_id"])
        and row.get("train_start") and row.get("test_start")
    })
    series_map = load_series_map(pnl_path, candidate_aids) if candidate_aids else {}

    for idx, row in enumerate(row_list):
        alpha_id = row.get("alpha_id")
        if not alpha_id or not ml.valid_simulation_id(alpha_id):
            results[idx] = {"state": "unavailable", "reason": "invalid_alpha_id"}
            continue
        if not row.get("train_start") or not row.get("test_start"):
            results[idx] = {"state": "unavailable", "reason": "train_window_not_verified"}
            continue
        proof = proofs.get(alpha_id)
        if proof is None:
            if alpha_id in series_map:
                proof = ml.pnl_cache_status(pnl_path, alpha_id, series_map[alpha_id])
            else:
                raw_series, err = cache_store.load_json(pnl_path / f"{alpha_id}.json")
                if err:
                    proofs[alpha_id] = {"error_state": "cache_missing" if err == "missing" else "cache_unreadable"}
                    results[idx] = {"state": "unavailable", "reason": proofs[alpha_id]["error_state"]}
                    continue
                proof = ml.pnl_cache_status(pnl_path, alpha_id, raw_series)
            proofs[alpha_id] = proof
        if "error_state" in proof:
            results[idx] = {"state": "unavailable", "reason": proof["error_state"]}
            continue
        if proof["complete"] is not True:
            results[idx] = {"state": "unavailable", "reason": "cache_completeness_unverified"}
            continue
        valid_indices.append(idx)

    if not valid_indices:
        return results

    pl = _import_polars()
    valid_aids = sorted({row_list[idx]["alpha_id"] for idx in valid_indices})
    sync = sync_pnl_parquet(pnl_path, alpha_ids=valid_aids) if pl is not None else {"parquet_path": None}
    parquet_path = sync.get("parquet_path")

    if pl is not None and parquet_path is not None and Path(parquet_path).exists():
        try:
            meta_rows = []
            for idx in valid_indices:
                row = row_list[idx]
                meta_rows.append({
                    "_row_idx": idx,
                    "alpha_id": row["alpha_id"],
                    "_start": str(row["train_start"])[:10],
                    "_cutoff": str(row["test_start"])[:10],
                })
            meta_df = pl.DataFrame(meta_rows).lazy()
            lf = (
                pl.scan_parquet(parquet_path)
                .filter(pl.col("alpha_id").is_in(valid_aids))
                .join(meta_df, on="alpha_id", how="inner")
                .filter(
                    (pl.col("day") >= pl.col("_start"))
                    & (pl.col("day") < pl.col("_cutoff"))
                    & pl.col("day").str.contains(r"^\d{4}-\d{2}-\d{2}$")
                )
                .with_columns(
                    _year=pl.col("day").str.slice(0, 4).cast(pl.Int32),
                    _month=pl.col("day").str.slice(5, 2).cast(pl.Int32),
                )
                .sort(["_row_idx", "day"])
                .group_by(["_row_idx", "_year"])
                .agg(
                    points=pl.len(),
                    first_month=pl.col("_month").first(),
                    last_month=pl.col("_month").last(),
                    mean=pl.col("pnl").mean(),
                    std=pl.col("pnl").std(ddof=0),
                )
                .sort(["_row_idx", "_year"])
            )
            agg_df = lf.collect()
            grouped = {}
            for rec in agg_df.to_dicts():
                grouped.setdefault(rec["_row_idx"], {})[rec["_year"]] = rec

            scale = math.sqrt(252.0)
            for idx in valid_indices:
                row = row_list[idx]
                start, cutoff = str(row["train_start"])[:10], str(row["test_start"])[:10]
                first, last = _date.fromisoformat(start), _date.fromisoformat(cutoff)
                by_year = grouped.get(idx, {})
                years = []
                for year in range(first.year + (first.month != 1 or first.day != 1), last.year):
                    info = by_year.get(year)
                    points = int(info["points"]) if info else 0
                    score = None
                    if (
                        info
                        and points >= _MIN_COMPLETE_YEAR_POINTS
                        and info["first_month"] == 1
                        and info["last_month"] == 12
                        and info["std"] is not None
                        and info["std"] > 0
                    ):
                        score = float(info["mean"] / info["std"] * scale)
                    years.append({"year": year, "points": points, "sharpe": score})
                output = {"years": years, "as_of": proofs[row["alpha_id"]]["as_of"], "state": "insufficient_full_train_years"}
                if len(years) >= 3 and all(item["sharpe"] is not None for item in years):
                    scores = [item["sharpe"] for item in years]
                    worst = min(scores)
                    recent = sum(scores[-3:]) / 3
                    weak_run = any(max(scores[i:i + 3]) < 0.3 for i in range(len(scores) - 2))
                    output.update(
                        state="pass" if worst > 0 and recent >= 1.0 and not weak_run else "fail",
                        worst_sharpe=worst,
                        recent_three_mean=recent,
                        three_weak_years=weak_run,
                    )
                results[idx] = output
            return results
        except Exception:  # noqa: BLE001 - fall back to exact Python computation on error
            pass

    for idx in valid_indices:
        row = row_list[idx]
        alpha_id = row["alpha_id"]
        series, error = cache_store.load_json(pnl_path / f"{alpha_id}.json")
        if error:
            results[idx] = {"state": "unavailable", "reason": "cache_missing" if error == "missing" else "cache_unreadable"}
            continue
        start, cutoff = str(row["train_start"])[:10], str(row["test_start"])[:10]
        by_year = {}
        for day, value in sorted(series.items()):
            if day < start or day >= cutoff:
                continue
            try:
                as_date = _date.fromisoformat(day)
            except ValueError:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                by_year.setdefault(as_date.year, []).append((as_date, value))
        first, last = _date.fromisoformat(start), _date.fromisoformat(cutoff)
        years = []
        for year in range(first.year + (first.month != 1 or first.day != 1), last.year):
            points = by_year.get(year, [])
            score = None
            if len(points) >= _MIN_COMPLETE_YEAR_POINTS and points[0][0].month == 1 and points[-1][0].month == 12:
                score = _sharpe([value for _, value in points], min_points=_MIN_COMPLETE_YEAR_POINTS)
            years.append({"year": year, "points": len(points), "sharpe": score})
        output = {"years": years, "as_of": proofs[alpha_id]["as_of"], "state": "insufficient_full_train_years"}
        if len(years) >= 3 and all(item["sharpe"] is not None for item in years):
            scores = [item["sharpe"] for item in years]
            worst = min(scores)
            recent = sum(scores[-3:]) / 3
            weak_run = any(max(scores[i:i + 3]) < 0.3 for i in range(len(scores) - 2))
            output.update(
                state="pass" if worst > 0 and recent >= 1.0 and not weak_run else "fail",
                worst_sharpe=worst,
                recent_three_mean=recent,
                three_weak_years=weak_run,
            )
        results[idx] = output
    return results
