"""Bounded workspace brief, book health, out-of-sample views log, and breeding helper."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import json
from pathlib import Path

import alpha_registry
import cache_store
import correlation
import machine_lib
import research_paths

from . import book_state, experiments, pnl_snapshot, stats
from .runtime import CliError, protect_output


def full_arm(arm):
    """Build a FULL simulation arm payload from a completed QUICK arm definition."""
    context = ("label", "expression", "role", "design", "proxy_of", "mechanism", "name",
               "family", "candidate_id", "idea", "hypothesis", "note")
    settings = machine_lib.normalize_setting_keys(arm["settings"])
    settings["simulationMode"] = "FULL"
    if settings.get("region") == "ASI":
        settings["visualization"] = True
    return {**{key: arm[key] for key in context
               if arm.get(key) is not None},
            "settings": settings}


def _first_heading(path):
    """The document's title, or its first non-empty line."""
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text.startswith("#"):
                    return text.lstrip("#").strip()
                if text:
                    return text[:120]
    except OSError:
        return None
    return None


def _convergence_snapshot(max_age_hours=48.0):
    """Read the snapshot ``ledger convergence`` writes without recomputing correlations."""
    path = Path(research_paths.ensure_dir()) / "convergence.json"
    if not path.exists():
        return {"stale": True, "hint": "run `ledger convergence` to record a baseline"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"stale": True, "hint": "snapshot unreadable; rerun `ledger convergence`"}
    as_of = payload.get("as_of") or ""
    age_hours = None
    try:
        stamp = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        age_hours = round((datetime.now(timezone.utc) - stamp).total_seconds() / 3600, 1)
    except ValueError:
        pass
    return {"as_of": as_of, "age_hours": age_hours,
            "stale": age_hours is not None and age_hours > max_age_hours,
            "batches": payload.get("batches"),
            "recent_productive": payload.get("recent_productive"),
            "recent_additive": payload.get("recent_additive"),
            "book_mechanisms": payload.get("book_mechanisms"),
            "recent": (payload.get("recent") or [])[-3:]}


def _drift_check():
    """Compare the platform's ACTIVE set with the ledger's, online."""
    def work(session):
        platform = {}
        for page in range(3):
            batch = machine_lib.list_alphas(session, limit=100, offset=page * 100,
                                            order="-dateCreated", status="ACTIVE")
            for alpha in batch:
                alpha_id = alpha.get("id")
                if alpha_id:
                    platform[alpha_id] = (alpha.get("status") or "").upper()
            if len(batch) < 100:
                break
        return platform

    args = type("DriftArgs", (), {"no_cache": False, "relogin": False, "meta_max_age": None})()
    try:
        from .runtime import with_session
        platform = with_session(args, work)
    except Exception as error:  # noqa: BLE001 - a brief must survive a network failure
        return {"error": "%s: %s" % (type(error).__name__, str(error)[:120])}
    return {"platform": platform}


def _book_drift(book, platform):
    """Ids where the platform says ACTIVE and the ledger disagrees, both ways."""
    ledger_active = {alpha_id for alpha_id, entry in (book or {}).items()
                     if str(entry.get("status") or "").upper() == "ACTIVE"}
    platform_ids = set(platform or {})
    return {"platform_active": len(platform_ids), "ledger_active": len(ledger_active),
            "on_platform_not_in_ledger": sorted(platform_ids - ledger_active),
            "in_ledger_not_on_platform": sorted(ledger_active - platform_ids),
            "in_sync": platform_ids == ledger_active}


def _cached_book_series(book, pnl_dir):
    return pnl_snapshot.load_series_map(pnl_dir, list(book))


def _book_stability(book, min_move=0.3, limit=3, *, series=None):
    """Report how many book alphas earn their train Sharpe in only one half of train."""
    if series is None:
        series = _cached_book_series(book, Path(research_paths.ensure_dir()) / "pnl")
    gaps, flips, late = [], [], []
    for alpha_id, values in series.items():
        if not isinstance(values, dict):
            continue
        try:
            pnl = {day: float(value) for day, value in values.items()}
        except (TypeError, ValueError):
            continue
        entry = (book.get(alpha_id) if isinstance(book, dict) else None) or {}
        test_start = ((entry.get("out_of_sample") or {}).get("test") or {}).get("startDate") or "2023-01-01"
        first, second = experiments.half_split_sharpes(pnl, test_start)
        if first is None or second is None:
            continue
        gaps.append(abs(first - second))
        if first * second < 0 and min(abs(first), abs(second)) >= min_move:
            flips.append({"alpha_id": alpha_id, "h1": round(first, 2), "h2": round(second, 2)})
        elif first < 0.5 and second > 1.5:
            late.append({"alpha_id": alpha_id, "h1": round(first, 2), "h2": round(second, 2)})
    gaps.sort()
    return {"measured": len(gaps), "flips": len(flips),
            "late_bloomers": len(late),
            "median_gap": round(gaps[len(gaps) // 2], 2) if gaps else None,
            "worst": flips[:limit] + late[:limit]}


def _book_correlator(series):
    ordered = {name: tuple(sorted(values)) for name, values in series.items()}
    centered = {}

    def prepare(name):
        if name not in centered:
            centered[name] = correlation.center_for_pearson([series[name][day] for day in ordered[name]])
        return centered[name]

    def correlate(left, right, min_overlap):
        days = ordered[left]
        if days == ordered[right]:
            if len(days) < min_overlap:
                return None, len(days)
            return correlation.pearson_centered(prepare(left), prepare(right)), len(days)
        x, y, shared = correlation.align(series[left], series[right], min_overlap=min_overlap)
        return correlation.pearson(x, y), len(shared)

    return correlate


def _book_duplicates(book, gate=0.70, prefilter=0.75, limit=8, *, series=None):
    """Book pairs above the correlation gate, measured inside their train windows."""
    if series is None:
        series = _cached_book_series(book, Path(research_paths.ensure_dir()) / "pnl")
    train = {}
    for alpha_id, pnl in series.items():
        if not isinstance(pnl, dict):
            continue
        test = ((book[alpha_id].get("out_of_sample") or {}).get("test") or {})
        train_end = test.get("startDate") or "2023-01-01"
        train[alpha_id] = {day: value for day, value in pnl.items() if day < train_end}
    result = correlation.matrix(train, min_overlap=200)
    sampled = {}
    hits = []
    for pair in result["pairs"]:
        if abs(pair["r"]) < gate:
            continue
        left, right = pair["a"], pair["b"]
        for side in (left, right):
            if side not in sampled:
                vals = train[side]
                sampled[side] = {day: vals[day] for day in sorted(vals)[::5]}
        sx, sy, _ = correlation.align(sampled[left], sampled[right], min_overlap=40)
        strided_r = correlation.pearson(sx, sy) if sx else None
        hits.append({
            "a": left,
            "b": right,
            "r": round(pair["r"], 4),
            "strided_r": round(strided_r, 3) if strided_r is not None else round(pair["r"], 3),
            "days": pair["n"],
        })
    hits.sort(key=lambda row: abs(row["r"]), reverse=True)
    return {"gate": gate, "prefilter": prefilter, "frequency": "daily", "window": "train",
            "count": len(hits), "pairs": hits[:limit]}


def _health_pool(args, latest, pnl_dir, series, book_daily, train_end, *, pool_series=None):
    @lru_cache(maxsize=4)
    def reference(days):
        values = [book_daily[day] for day in days]
        units = stats.unit_correlation_rows([[member[day] for day in days] for member in series.values()])
        return values, stats.sharpe(values, min_points=1), units

    candidate_ids = [
        alpha_id for alpha_id, entry in latest.items()
        if str(entry.get("status") or "").upper() not in book_state.BOOK_STATUSES
    ]
    if pool_series is None:
        pool_series = pnl_snapshot.load_series_map(pnl_dir, candidate_ids)

    rows = []
    for alpha_id in candidate_ids:
        entry = latest[alpha_id]
        candidate = pool_series.get(alpha_id)
        if not candidate:
            continue
        cutoff = train_end
        if cutoff is not None:
            try:
                bounds, _ = book_state.test_boundaries(latest, [alpha_id])
            except ValueError:
                continue
            cutoff = min(cutoff, bounds.get(alpha_id, cutoff))
        days = tuple(day for day in book_daily if day in candidate and (cutoff is None or day < cutoff))
        if len(days) < args.min_overlap:
            continue
        values = [candidate[day] for day in days]
        book_values, book_sharpe, units = reference(days)
        sharpe = stats.sharpe(values, min_points=1)
        rho = correlation.pearson(values, book_values)
        if sharpe is None or rho is None or book_sharpe is None:
            continue
        marginal = sharpe - rho * book_sharpe
        measured, worst = stats.member_correlations(values, series, days, units, min_points=3)
        rows.append({
            "alpha_id": alpha_id, "family": entry.get("family"), "status": entry.get("status"),
            "sharpe": sharpe, "rho_book": round(rho, 3), "marginal": marginal,
            "overlap_days": len(days), "train_end": cutoff,
            "max_single": worst[0] if worst is not None else None,
            "max_single_vs": worst[1] if worst is not None else None,
            "passes": (worst is not None and measured == len(series) and abs(worst[0]) < args.gate
                       and marginal >= args.min_marginal and sharpe >= args.min_sharpe),
        })
    return sorted(rows, key=lambda row: -row["marginal"])


def cmd_health(args, _sink):
    """Analyze the book's marginal contribution, train-window duplicates, and candidate pool."""
    entries = alpha_registry.load_entries(args.registry or alpha_registry.default_registry_path())
    latest = {entry["alpha_id"]: entry for entry in entries if entry.get("alpha_id")}
    book = {alpha_id: entry for alpha_id, entry in latest.items()
            if str(entry.get("status") or "").upper() in book_state.BOOK_STATUSES}
    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"
    include_pool = bool(getattr(args, "pool", False))
    load_ids = sorted(latest) if include_pool else sorted(book)
    all_series = pnl_snapshot.load_series_map(pnl_dir, load_ids)
    series = {alpha_id: all_series[alpha_id] for alpha_id in sorted(book) if alpha_id in all_series}
    if not series:
        raise CliError("INVALID_INPUT", "no cached PnL for the book",
                       "run `ledger sync` or `alpha get` to warm the cache")
    dates = sorted(set.intersection(*(set(values) for values in series.values())))
    window = getattr(args, "window", "train")
    train_end = None
    if window == "train":
        requested_end = getattr(args, "train_end", None) or "2023-01-01"
        try:
            if date.fromisoformat(requested_end).isoformat() != requested_end:
                raise ValueError("--train-end must be YYYY-MM-DD")
            bounds, _ = book_state.test_boundaries(latest, series)
        except ValueError as exc:
            raise CliError("INVALID_INPUT", str(exc),
                           "pass --train-end YYYY-MM-DD and ensure recorded test boundaries are valid ISO dates") from exc
        train_end = min(requested_end, min(bounds.values(), default=requested_end))
        dates = [day for day in dates if day < train_end]
    if len(dates) < args.min_overlap:
        raise CliError("INVALID_INPUT", "only %d common days" % len(dates),
                       "lower --min-overlap or refresh the PnL cache")

    aligned = {alpha_id: [values[day] for day in dates] for alpha_id, values in series.items()}
    count = len(aligned)
    total = [sum(values) for values in zip(*aligned.values())]
    book_daily = {day: value / count for day, value in zip(dates, total)}
    book_sharpe = stats.sharpe(book_daily.values(), min_points=1)
    rows = []
    for alpha_id, values in aligned.items():
        peers = [row for member, row in aligned.items() if member != alpha_id]
        rest = [sum(day) / len(peers) for day in zip(*peers)]
        sharpe = stats.sharpe(values, min_points=1)
        rho = correlation.pearson(values, rest)
        rest_sharpe = stats.sharpe(rest, min_points=1)
        marginal = (round(sharpe - rho * book_sharpe, 2)
                    if sharpe is not None and rho is not None and book_sharpe is not None else None)
        delta = round(book_sharpe - rest_sharpe, 4) if rest_sharpe is not None and book_sharpe is not None else None
        rows.append({"alpha_id": alpha_id,
                     "family": (book.get(alpha_id) or {}).get("family"),
                     "sharpe": None if sharpe is None else round(sharpe, 2),
                     "rho_rest": None if rho is None else round(rho, 3),
                     "slope": marginal, "delta": delta, "marginal": marginal})

    pool = (_health_pool(args, latest, pnl_dir, series, book_daily, train_end, pool_series=all_series)
            if include_pool else None)
    rows.sort(key=lambda row: (row["marginal"] if row["marginal"] is not None else 0))
    measured = [row for row in rows if row["marginal"] is not None]
    return {"window": window, "train_end": train_end,
            "days": len(dates), "count": len(measured),
            "book": book_state.describe_book(latest, series=series, registry_path=args.registry,
                                               pnl_dir=pnl_dir),
            "book_sharpe": None if book_sharpe is None else round(book_sharpe, 2),
            "negative": [row for row in measured if row["marginal"] < 0][: args.limit],
            "weakest": measured[: args.limit],
            "strongest": sorted(measured, key=lambda row: -(row["marginal"] or 0))[: args.limit],
            "negative_count": sum(1 for row in measured if row["marginal"] < 0),
            "pool": pool[: args.limit] if pool is not None else None,
            "pool_size": len(pool) if pool is not None else None,
            "pool_passing": sum(row["passes"] for row in pool) if pool is not None else None,
            "gate": {"min_marginal": getattr(args, "min_marginal", 0.3),
                     "max_single": getattr(args, "gate", 0.70),
                     "min_sharpe": getattr(args, "min_sharpe", 1.0)}}


def cmd_views(args, _sink):
    """Read the log of every time out-of-sample numbers were looked at."""
    path = research_paths.views_log_path()
    entries = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if args.alpha and args.alpha not in (entry.get("alpha_ids") or []):
                continue
            entries.append(entry)
    entries = entries[-args.limit:]
    return {"path": path, "count": len(entries),
            "by_tool": {tool: sum(1 for entry in entries if entry.get("tool") == tool)
                        for tool in sorted({entry.get("tool") for entry in entries if entry.get("tool")})},
            "entries": entries}


def cmd_breed(args, _sink):
    """Freeze one train-Fitness parent-selection comparison as a draft spec for ``propose``."""
    from . import breeding

    if getattr(args, "out", None) == "-":
        raise CliError("USAGE", "--out requires a file path for the breeding spec",
                       "pass a destination file path, e.g. --out offspring.json")
    output = Path(args.out).expanduser().resolve()
    protect_output(args, output)

    import workspace_db

    root = Path(research_paths.ensure_dir()).resolve()
    runs_dir = root / "runs"
    raw_ref = str(args.source)
    storage = workspace_db.db_path(root)
    batch = workspace_db.get_run(raw_ref, research_dir=root)
    if batch is None:
        candidate_run = runs_dir / f"{raw_ref}.json"
        source_path = candidate_run if candidate_run.is_file() else Path(raw_ref).expanduser().resolve()
        protect_output(args, source_path)
        protected = set()
        try:
            path, batch = experiments.load_probe_batch(source_path, source_paths=protected)
        finally:
            protect_output(args, *protected)
        source = {"batch_path": str(path)}
    else:
        source = {"workspace_db": str(storage)}
        if batch.get("track") == "SPC" or not all(isinstance(batch.get(key), list) for key in ("arms", "rows", "records")):
            raise CliError("INVALID_INPUT", "the run lacks comparable simulation arms",
                           "choose a REGULAR run with finished train evidence or a saved probe batch")

    lock = Path(f"{output}.lock")
    protect_output(args, lock)
    if lock.is_symlink():
        raise CliError("INVALID_INPUT", "--out lock path must not be a symlink",
                       "choose a clean destination filename for --out")

    arms, report = breeding.generation(batch, alpha_registry.load_entries(),
                                      seed=args.seed, limit=args.limit)
    source_meta = {**source, "run_id": batch.get("run_id"),
                   "request_fingerprint": batch.get("request_fingerprint"), **report}
    spec = {"name": output.stem, "track": batch.get("track", "REGULAR"), "design": "explore",
            "idea": "Compare train-Fitness parent selection with a seeded lottery at equal budget.",
            "breeding_source": source_meta, "arms": arms}
    experiments.validate_probe_design(spec, {})
    try:
        with cache_store.file_lock(lock, timeout=0):
            if output.exists():
                if json.loads(output.read_text(encoding="utf-8")) != spec:
                    raise CliError("INVALID_INPUT", "refusing to replace a different breeding spec",
                                   "choose a new --out filename or remove the obsolete draft")
            else:
                cache_store.write_json_atomic(output, spec, indent=2)
    except cache_store.LockTimeout as exc:
        raise CliError("SERVICE_ERROR", "the breeding spec is being prepared by another process",
                       "wait for the active writer to finish before retrying") from exc
    except (OSError, ValueError) as exc:
        raise CliError("INVALID_INPUT", f"cannot save breeding spec: {exc}",
                       "check directory permissions and disk space for --out") from exc
    return {"path": str(output), "track": spec["track"], "arms": len(arms),
            "breeding_source": source_meta}


def _state_machine_summary(research_dir, limit):
    import workspace_db
    from .state_machine import AUTO_STATES, GATE_STATES

    runs_dir = research_dir / "runs"
    if runs_dir.is_dir():
        workspace_db.import_legacy_run_files(runs_dir, research_dir=research_dir)
    counts = workspace_db.run_counts(research_dir=research_dir)
    gates = workspace_db.list_runs(states=GATE_STATES, limit=limit, research_dir=research_dir)
    return {
        "db_path": str(workspace_db.db_path(research_dir)),
        "runs_dir": str(runs_dir),
        "actionable_gates": [
            {key: doc.get(key) for key in ("run_id", "name", "state", "track", "updated_at")}
            for doc in gates
        ],
        "actionable_gates_total": sum(counts.get(state, 0) for state in GATE_STATES),
        "active_auto_count": sum(counts.get(state, 0) for state in AUTO_STATES),
        "by_state": counts,
    }


def cmd_brief(args, _sink):
    """Emit the bounded state of research, book health, recent results, and pending state-machine gates."""
    if args.top < 1:
        raise CliError("INVALID_INPUT", "--top must be at least 1",
                       "pass --top 1 or greater")

    def excerpt(text):
        if isinstance(text, str) and len(text) > 240:
            return text[:239] + "…"
        return text

    def moment(entry):
        raw = entry.get("recorded_at") or ""
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return None

    days = max(0.0, args.days)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    registry_path = args.registry or alpha_registry.default_registry_path()
    entries = alpha_registry.load_entries(registry_path)

    def metric(entry, name):
        return experiments._train_metrics(entry).get(name)

    latest = {entry["alpha_id"]: entry for entry in entries if entry.get("alpha_id")}
    book = {alpha_id: entry for alpha_id, entry in latest.items()
            if str(entry.get("status") or "").upper() in book_state.BOOK_STATUSES}

    sharpes = [metric(entry, "sharpe") for entry in book.values()
               if isinstance(metric(entry, "sharpe"), (int, float))]
    families, universes = {}, {}
    for entry in book.values():
        key = entry.get("family") or "unlabeled"
        families[key] = families.get(key, 0) + 1
        universe = (entry.get("settings") or {}).get("universe") or "?"
        universes[universe] = universes.get(universe, 0) + 1

    recent = []
    for entry in entries:
        when = moment(entry)
        if when is not None and when >= cutoff:
            recent.append(entry)
    runs = [entry for entry in recent if (entry.get("source") or "") in {"simulate", "record", "submit"}]
    by_verdict = {}
    for entry in runs:
        verdict = entry.get("verdict") or "unlabeled"
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
    scored = [entry for entry in runs if isinstance(metric(entry, "sharpe"), (int, float))]
    scored.sort(key=lambda entry: -metric(entry, "sharpe"))
    best = [{"alpha_id": entry.get("alpha_id"), "sharpe": metric(entry, "sharpe"),
             "fitness": metric(entry, "fitness"),
             "verdict": entry.get("verdict"), "idea": excerpt(entry.get("idea")),
             "family": entry.get("family"), "mechanism": excerpt(entry.get("mechanism")),
             "when": entry.get("recorded_at")}
            for entry in scored[:args.top]]

    setbacks = [entry for entry in recent if entry.get("verdict") in {"rejected", "falsified", "no_go"}]
    setbacks.sort(key=moment, reverse=True)
    seen, lessons = set(), []
    for entry in setbacks:
        group = (entry.get("family"), entry.get("mechanism") or entry.get("key"), entry.get("verdict"))
        if group in seen:
            continue
        seen.add(group)
        lessons.append({"key": entry.get("key"), "alpha_id": entry.get("alpha_id"),
                        "family": entry.get("family"),
                        "mechanism": excerpt(entry.get("mechanism")),
                        "verdict": entry.get("verdict"), "note": excerpt(entry.get("note")),
                        "when": entry.get("recorded_at")})
        if len(lessons) >= args.top:
            break

    research_dir = Path(research_paths.ensure_dir())
    documents = []
    for path in sorted(research_dir.glob("*-results.md"),
                       key=lambda item: item.stat().st_mtime, reverse=True):
        written = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if written < cutoff:
            break
        documents.append({"file": path.name, "when": written.isoformat(timespec="minutes"),
                          "title": _first_heading(path)})
        if len(documents) >= args.top:
            break

    thread_file = research_dir / "open-threads.md"
    threads = {"path": str(thread_file), "exists": thread_file.exists(),
               "counts": {"open": 0, "closed": 0}, "open": [], "closed": []}
    if threads["exists"]:
        for line in thread_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            for tag, key in (("[open]", "open"), ("[closed]", "closed")):
                if text.startswith("- " + tag):
                    body = text[len(tag) + 2:].strip()
                    if body:
                        threads["counts"][key] += 1
                        if len(threads[key]) < args.top:
                            threads[key].append(excerpt(body))

    unlabeled = sum(1 for entry in book.values() if not entry.get("family") or not entry.get("mechanism"))
    no_checks = sum(1 for entry in entries if (entry.get("metrics") or {}) and not (entry.get("checks") or {}))
    pnl_dir = research_dir / "pnl"
    series = _cached_book_series(book, pnl_dir)
    evidence = book_state.describe_book(latest, series=series, registry_path=registry_path, pnl_dir=pnl_dir)
    capture = evidence["platform"]
    book_evidence = {
        "source": "local_ledger", "member_digest": evidence["local"]["member_digest"],
        "statuses": evidence["local"]["statuses"],
        "platform": {key: capture.get(key) for key in
                     ("available", "as_of", "scope", "complete", "matches_local_members",
                      "current_capture", "freshness", "live_verified")},
        "pnl": {key: len(evidence["pnl"][key + "_ids"])
                for key in ("available", "missing", "complete", "unknown", "incomplete")}}
    return {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_days": days,
        "research_dir": str(research_dir),
        "state_machine": _state_machine_summary(research_dir, args.top),
        "ledger": {"rows": len(entries), "path": registry_path,
                   "exists": Path(registry_path).exists(),
                   "book_rows_missing_labels": unlabeled,
                   "rows_with_metrics_but_no_checks": no_checks},
        "book": {
            "count": len(book),
            "evidence": book_evidence,
            "metrics_scope": "train",
            "sharpe_median": (round(sorted(sharpes)[len(sharpes) // 2], 3) if sharpes else None),
            "by_family": dict(sorted(families.items(), key=lambda item: -item[1])),
            "by_universe": dict(sorted(universes.items(), key=lambda item: -item[1])),
            "self_duplicates": _book_duplicates(book, series=series),
            "stability": _book_stability(book, series=series),
        },
        "recent": {"runs": len(runs), "by_verdict": by_verdict, "best": best,
                   "metrics_scope": "train", "missing_train_metrics": len(runs) - len(scored),
                   "setbacks": {"count": len(setbacks), "items": lessons}},
        "documents": documents,
        "convergence": _convergence_snapshot(),
        "drift": (_book_drift(book, _drift_check().get("platform")) if getattr(args, "drift", False) else None),
        "labs": _labs_summary(research_dir, args.top, entries),
        "threads": threads,
    }


def _labs_summary(research_dir, limit, entries):
    from . import labs

    summary = labs.open_ticket_summary(research_dir, limit=limit)
    return {
        "open_count": summary["count"],
        "open_tickets": summary["tickets"],
        "suggested_delegations": labs.suggest_delegations(research_dir, limit=limit, entries=entries),
    }
