"""Bounded research briefs and train/test view history."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
import alpha_registry
import cache_store
import correlation
import json
import research_paths

from . import book_state, experiments, pnl_snapshot, stats
from .runtime import CliError



def cmd_views(args, _sink):
    """Read the log of every time out-of-sample numbers were looked at.

    The point is auditability, not prevention: a frozen validation and an iterative
    peek look identical in a results table, and only the log tells them apart.
    """
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
    """Read the snapshot `ledger convergence` writes; never compute it here.

    The marginal pass needs hundreds of PnL correlations, which would push `brief` from
    one second to tens of seconds. Reporting a stale snapshot with its age is the honest
    alternative to both: the trigger stays visible, and nobody mistakes a 3-day-old
    reading for the current state.
    """
    from datetime import datetime, timezone

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
    """Compare the platform's ACTIVE set with the ledger's, online.

    `brief` is offline by design, so this runs only when asked. It exists because the
    ledger only grows when something records an alpha: on 2026-11-07 an alpha was
    submitted on the platform while the local row still said COMPLETE, so the reported
    book was one short and the alpha looked like an unused candidate.
    """
    import machine_lib as ml
    from .runtime import CliError

    def work(session):
        platform = {}
        for page in range(3):
            batch = ml.list_alphas(session, limit=100, offset=page * 100,
                                   order="-dateCreated", status="ACTIVE")
            for alpha in batch:
                alpha_id = alpha.get("id")
                if alpha_id:
                    platform[alpha_id] = (alpha.get("status") or "").upper()
            if len(batch) < 100:
                break
        return platform

    args = type("DriftArgs", (), {"no_cache": False, "relogin": False,
                                  "meta_max_age": None})()
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


def _health_pool(args, latest, pnl_dir, series, book_daily, train_end):
    @lru_cache(maxsize=4)
    def reference(days):
        values = [book_daily[day] for day in days]
        units = stats.unit_correlation_rows([[member[day] for day in days] for member in series.values()])
        return values, stats.sharpe(values, min_points=1), units

    rows = []
    for alpha_id, entry in latest.items():
        if str(entry.get("status") or "").upper() in book_state.BOOK_STATUSES:
            continue
        candidate = pnl_snapshot.read_series(pnl_dir / f"{alpha_id}.json")
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
    """The book's own accounting: who contributes, who is dead weight, what is duplicated.

    A book's alpha count is not its independent bet count, and equal weighting pays
    every member the same regardless of what it adds. The number that decides whether a member
    earns its slot is its marginal contribution, `S_i - rho_i * S_book`, and it needs
    two things this workspace already has: each alpha's daily PnL and its correlation
    against the rest of the book.

    Read-only and offline: it never simulates, and it never submits.
    """
    entries = alpha_registry.load_entries(args.registry or alpha_registry.default_registry_path())
    latest = {entry["alpha_id"]: entry for entry in entries if entry.get("alpha_id")}
    book = {alpha_id: entry for alpha_id, entry in latest.items()
            if str(entry.get("status") or "").upper() in book_state.BOOK_STATUSES}
    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"
    series = {alpha_id: values for alpha_id in sorted(book)
              if (values := pnl_snapshot.read_series(pnl_dir / f"{alpha_id}.json"))}
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
            raise CliError("INVALID_INPUT", str(exc)) from exc
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

    pool = _health_pool(args, latest, pnl_dir, series, book_daily, train_end) if getattr(args, "pool", False) else None

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


def _cached_book_series(book, pnl_dir):
    series = {}
    for alpha_id in book:
        values, _ = cache_store.load_json(pnl_dir / (alpha_id + ".json"))
        if values is not None:
            series[alpha_id] = values
    return series


def _book_stability(book, min_move=0.3, limit=3, *, series=None):
    """How many book alphas earn their train Sharpe in only one half of train.

    A single train Sharpe can be the average of two opposite regimes, and the test
    year may not be used to select. Splitting train in half keeps the evidence inside
    the window selection is allowed to read, so a flip is visible at session start
    instead of after a submission disappoints.

    This catches *train-internal* regime dependence, not future decay. Stability
    within the training window does not guarantee positive returns in a later period.
    """
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
        # The halves must stay inside train: the test year is not selection evidence,
        # and splitting the whole series would quietly make it so.
        entry = (book.get(alpha_id) if isinstance(book, dict) else None) or {}
        test_start = ((entry.get("out_of_sample") or {}).get("test") or {}).get("startDate")
        # Same conservative default as the duplicate screen: an entry that predates the
        # recorded split must not silently split the whole series and read test data.
        test_start = test_start or "2023-01-01"
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
    """Book pairs above the correlation gate, measured inside their train windows.

    A stride prefilter keeps startup work bounded; surviving pairs are checked on
    daily train PnL. Test-period returns must not change the session-start signal.
    """
    if series is None:
        series = _cached_book_series(book, Path(research_paths.ensure_dir()) / "pnl")
    train = {}
    for alpha_id, pnl in series.items():
        if not isinstance(pnl, dict):
            continue
        test = ((book[alpha_id].get("out_of_sample") or {}).get("test") or {})
        train_end = test.get("startDate") or "2023-01-01"
        train[alpha_id] = {day: value for day, value in pnl.items() if day < train_end}
    # Monthly averaging can distort daily pair correlations. Sample every fifth
    # day for the prefilter, then confirm each surviving pair on daily train PnL.
    # The prefilter is an approximation and does not guarantee an exhaustive scan.
    sampled = {name: {day: values[day] for day in sorted(values)[::5]}
               for name, values in train.items()}
    sampled_correlation = _book_correlator(sampled)
    daily_correlation = _book_correlator(train)
    names = sorted(sampled)
    suspects = []
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            value, _ = sampled_correlation(left, right, min_overlap=40)
            if value is None or abs(value) < prefilter:
                continue
            suspects.append((left, right, value))
    hits = []
    for left, right, strided_r in suspects:
        value, days = daily_correlation(left, right, min_overlap=200)
        if value is None or abs(value) < gate:
            continue
        hits.append({"a": left, "b": right, "r": round(value, 4),
                     "strided_r": round(strided_r, 3), "days": days})
    hits.sort(key=lambda row: abs(row["r"]), reverse=True)
    return {"gate": gate, "prefilter": prefilter, "frequency": "daily", "window": "train",
            "count": len(hits), "pairs": hits[:limit]}


def cmd_brief(args, _sink):
    """The state of the campaign in one bounded document.

    Every session starts by reconstructing where the research stands, and the
    material for that is scattered: a multi-megabyte ledger, a directory of
    date-stamped write-ups, a hand-written weekly note, and whatever the last
    session remembered. Reading all of it is the expensive part of the work, and it
    grows every week while the answer stays the same size.

    Emit scores alongside recent setbacks, source paths and bounded open questions.
    Labels and notes are recorded judgements, not inferred research conclusions.

    The open-question list is read from ``open-threads.md`` in the research
    directory, one ``- [open] ...`` or ``- [closed] ...`` line each. That file is
    maintained by hand on purpose: a question worth carrying across sessions is a
    judgement, and anything derived automatically would be a summary of the past
    rather than a list of what is still undecided.
    """
    from datetime import datetime, timedelta, timezone

    if args.top < 1:
        raise CliError("INVALID_INPUT", "--top must be at least 1")

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
        # A row with no timestamp cannot be said to be recent; including it would
        # make the window meaningless on a ledger that predates `recorded_at`.
        if when is not None and when >= cutoff:
            recent.append(entry)
    runs = [entry for entry in recent
            if (entry.get("source") or "") in {"simulate", "record", "submit"}]
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

    # Include relabelled decisions: a conclusion often arrives after the simulation.
    setbacks = [entry for entry in recent
                if entry.get("verdict") in {"rejected", "falsified", "no_go"}]
    setbacks.sort(key=moment, reverse=True)
    seen, lessons = set(), []
    for entry in setbacks:
        group = (entry.get("family"), entry.get("mechanism") or entry.get("key"),
                 entry.get("verdict"))
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
        documents.append({"file": path.name,
                          "when": written.isoformat(timespec="minutes"),
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

    # Label debt and recording gaps: both were invisible until someone counted by
    # hand, and both make the ledger look complete while it is not.
    unlabeled = sum(1 for entry in book.values()
                    if not entry.get("family") or not entry.get("mechanism"))
    no_checks = sum(1 for entry in entries
                    if (entry.get("metrics") or {}) and not (entry.get("checks") or {}))
    pnl_dir = research_dir / "pnl"
    series = _cached_book_series(book, pnl_dir)
    evidence = book_state.describe_book(latest, series=series, registry_path=registry_path,
                                       pnl_dir=pnl_dir)
    capture = evidence["platform"]
    book_evidence = {
        "source": "local_ledger", "member_digest": evidence["local"]["member_digest"],
        "statuses": evidence["local"]["statuses"],
        "platform": {key: capture.get(key) for key in
                     ("available", "as_of", "scope", "complete", "matches_local_members",
                      "current_capture", "freshness", "live_verified")},
        "pnl": {key: len(evidence["pnl"][key + "_ids"])
                for key in ("available", "missing", "complete", "unknown", "incomplete")}}
    data = {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_days": days,
        "research_dir": str(research_dir),
        "ledger": {"rows": len(entries), "path": registry_path,
                   "exists": Path(registry_path).exists(),
                   "book_rows_missing_labels": unlabeled,
                   "rows_with_metrics_but_no_checks": no_checks},
        "book": {
            "count": len(book),
            "evidence": book_evidence,
            "metrics_scope": "train",
            "sharpe_median": (round(sorted(sharpes)[len(sharpes) // 2], 3)
                              if sharpes else None),
            "by_family": dict(sorted(families.items(), key=lambda item: -item[1])),
            "by_universe": dict(sorted(universes.items(), key=lambda item: -item[1])),
            # Same-bet pairs the submission gate could have missed (snapshot lag).
            "self_duplicates": _book_duplicates(book, series=series),
            # A train Sharpe can hide a single-regime edge; the halves stay in train.
            "stability": _book_stability(book, series=series),
        },
        "recent": {"runs": len(runs), "by_verdict": by_verdict, "best": best,
                   "metrics_scope": "train", "missing_train_metrics": len(runs) - len(scored),
                   "setbacks": {"count": len(setbacks), "items": lessons}},
        "documents": documents,
        # The divergence trigger: how many of the last batches produced anything additive.
        "convergence": _convergence_snapshot(),
        "drift": (_book_drift(book, _drift_check().get("platform"))
                  if getattr(args, "drift", False) else None),
        "threads": threads,
    }

    return data
