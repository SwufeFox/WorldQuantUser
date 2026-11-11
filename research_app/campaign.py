"""Bounded research briefs and train/test view history."""

from __future__ import annotations

from pathlib import Path
import alpha_registry
import correlation
import json
import research_paths

from . import experiments



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


def _annual_sharpe(values):
    if not values:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    if var <= 0:
        return None
    return mean / (var ** 0.5) * (252 ** 0.5)


def cmd_health(args, _sink):
    """The book's own accounting: who contributes, who is dead weight, what is duplicated.

    A book of 55 alphas is not 55 bets, and equal weighting means every member is paid
    the same regardless of what it adds. The one number that decides whether a member
    earns its slot is its marginal contribution, `S_i - rho_i * S_book`, and it needs
    two things this workspace already has: each alpha's daily PnL and its correlation
    against the rest of the book.

    Read-only and offline: it never simulates, and it never submits.
    """
    entries = alpha_registry.load_entries(args.registry or alpha_registry.default_registry_path())
    latest = {}
    for entry in entries:
        alpha_id = entry.get("alpha_id")
        if alpha_id:
            latest[alpha_id] = entry
    book = {alpha_id: entry for alpha_id, entry in latest.items()
            if str(entry.get("status") or "").upper() == "ACTIVE"}
    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"
    series = {}
    for alpha_id in book:
        path = pnl_dir / ("%s.json" % alpha_id)
        if not path.exists():
            continue
        try:
            series[alpha_id] = {k: float(v) for k, v in json.loads(path.read_text()).items()}
        except (OSError, ValueError):
            continue
    ids = sorted(series)
    if not ids:
        raise CliError("INVALID_INPUT", "no cached PnL for the book",
                       "run `ledger sync` or `alpha get` to warm the cache")
    dates = sorted(set.intersection(*(set(series[a]) for a in ids)))
    # Default to the train window. The cached series runs to 2023; ranking candidates on
    # the full window would be ranking on the test year, which is the one thing this
    # workspace refuses to select on.
    if getattr(args, "window", "train") == "train" and getattr(args, "train_end", None):
        dates = [day for day in dates if day < args.train_end]
        if not dates:
            raise CliError("INVALID_INPUT", "no days before --train-end %s" % args.train_end)
    if len(dates) < args.min_overlap:
        raise CliError("INVALID_INPUT", "only %d common days" % len(dates),
                       "lower --min-overlap or refresh the PnL cache")
    aligned = {a: [series[a][d] for d in dates] for a in ids}
    sharpes = {a: _annual_sharpe(aligned[a]) for a in ids}
    book_series = [sum(aligned[a][i] for a in ids) / len(ids) for i in range(len(dates))]
    book_sharpe = _annual_sharpe(book_series)
    rows = []
    for alpha_id in ids:
        # Marginal is a statement about the *other* members, so a single-alpha book has
        # no marginal to report (and dividing by len-1 would crash).
        rest = ([sum(aligned[b][i] for b in ids if b != alpha_id) / (len(ids) - 1)
                 for i in range(len(dates))] if len(ids) > 1 else [])
        rho = correlation.pearson(aligned[alpha_id], rest) if rest else None
        sharpe = sharpes[alpha_id]
        marginal = None
        if sharpe is not None and rho is not None and book_sharpe is not None:
            marginal = sharpe - rho * book_sharpe
        # Two different numbers, deliberately side by side. `slope` is the first-order
        # term S_i - rho*S_book; `delta` is what the book's own Sharpe does when this
        # member is removed. They differ by roughly the member's weight - on a 60-alpha
        # book the slope can read +1.4 while the book moves 0.03 - and calling both
        # "marginal" is how the slope gets quoted as an outcome.
        delta = None
        if len(ids) > 1 and rest and book_sharpe is not None:
            rest_sharpe = _annual_sharpe(rest)
            if rest_sharpe is not None:
                delta = book_sharpe - rest_sharpe
        rows.append({"alpha_id": alpha_id,
                     "family": (book.get(alpha_id) or {}).get("family"),
                     "sharpe": None if sharpe is None else round(sharpe, 2),
                     "rho_rest": None if rho is None else round(rho, 3),
                     "slope": None if marginal is None else round(marginal, 2),
                     "delta": None if delta is None else round(delta, 4),
                     "marginal": None if marginal is None else round(marginal, 2)})
    # Optional second pass: score the *pool* (alphas the ledger knows, PnL cached, not in
    # the book) against this same book. Ranking a finished candidate by its Fitness hid
    # the alphas that are weak alone but genuinely add; ranking by marginal is the
    # criterion the book is actually optimised on.
    pool = []
    if getattr(args, "pool", False):
        book_ids = set(ids)
        for alpha_id, entry in latest.items():
            if alpha_id in book_ids:
                continue
            if str(entry.get("status") or "").upper() == "ACTIVE":
                continue
            path = pnl_dir / ("%s.json" % alpha_id)
            if not path.exists():
                continue
            try:
                series_map = {k: float(v) for k, v in json.loads(path.read_text()).items()}
            except (OSError, ValueError):
                continue
            common = sorted(set(series_map) & set(dates))
            if len(common) < args.min_overlap:
                continue
            values = [series_map[d] for d in common]
            sharpe = _annual_sharpe(values)
            index = {d: i for i, d in enumerate(dates)}
            book_aligned = [book_series[index[d]] for d in common]
            rho = correlation.pearson(values, book_aligned)
            if sharpe is None or rho is None or book_sharpe is None:
                continue
            pool.append({"alpha_id": alpha_id,
                         "family": entry.get("family"),
                         "status": entry.get("status"),
                         "sharpe": round(sharpe, 2),
                         "rho_book": round(rho, 3),
                         "marginal": round(sharpe - rho * book_sharpe, 2)})
        pool.sort(key=lambda row: -(row["marginal"] or 0))
        # Marginal is measured against the *portfolio*, but the platform's gate is the
        # max against any single member - and they disagree in practice: `LL58p8ee`
        # scored rho_book 0.18 yet 0.972 against `rKd0g0bj`, i.e. a duplicate of one
        # member that looks diversified against the average. Report both for the rows we
        # show, otherwise the marginal alone would wave duplicates through.
        # Screen every pool row, not just the visible ones: the point of the pool pass is
        # to answer "what is left after the two quantitative legs of the gate", and that
        # answer is a count. The strided series keeps 433 x 55 correlations cheap.
        stride = max(1, len(dates) // 250)
        strided_dates = dates[::stride]
        book_strided = {a: [series[a][d] for d in strided_dates] for a in ids}
        for row in pool:
            path = pnl_dir / ("%s.json" % row["alpha_id"])
            try:
                series_map = {k: float(v) for k, v in json.loads(path.read_text()).items()}
            except (OSError, ValueError):
                continue
            values = [series_map.get(d) for d in strided_dates]
            if any(v is None for v in values):
                row["max_single"] = None
            else:
                worst, partner = 0.0, None
                for other in ids:
                    r = correlation.pearson(values, book_strided[other])
                    if r is not None and abs(r) > abs(worst):
                        worst, partner = r, other
                row["max_single"] = round(worst, 3)
                row["max_single_vs"] = partner
            # A negative-Sharpe alpha with negative correlation scores a big "marginal"
            # and would be nonsense to submit, so the screen also requires a real Sharpe.
            row["passes"] = bool(row.get("max_single") is not None
                                 and abs(row["max_single"]) < args.gate
                                 and row["marginal"] >= args.min_marginal
                                 and (row.get("sharpe") or 0) >= args.min_sharpe)

        for row in pool[: max(args.limit, 1) * 2]:
            path = pnl_dir / ("%s.json" % row["alpha_id"])
            try:
                series_map = {k: float(v) for k, v in json.loads(path.read_text()).items()}
            except (OSError, ValueError):
                continue
            values = [series_map.get(d) for d in dates]
            if any(v is None for v in values):
                row["max_single"] = None
                continue
            worst, partner = 0.0, None
            for other in ids:
                r = correlation.pearson(values, aligned[other])
                if r is not None and abs(r) > abs(worst):
                    worst, partner = r, other
            row["max_single"] = round(worst, 3)
            row["max_single_vs"] = partner

    rows.sort(key=lambda row: (row["marginal"] if row["marginal"] is not None else 0))
    measured = [row for row in rows if row["marginal"] is not None]
    return {"window": getattr(args, "window", "train"), "train_end": getattr(args, "train_end", None),
            "days": len(dates), "count": len(measured),
            "book_sharpe": None if book_sharpe is None else round(book_sharpe, 2),
            "negative": [row for row in measured if row["marginal"] < 0][: args.limit],
            "weakest": measured[: args.limit],
            "strongest": sorted(measured, key=lambda row: -(row["marginal"] or 0))[: args.limit],
            "negative_count": sum(1 for row in measured if row["marginal"] < 0),
            "pool": pool[: args.limit] if getattr(args, "pool", False) else None,
            "pool_size": len(pool) if getattr(args, "pool", False) else None,
            "pool_passing": (sum(1 for row in pool if row.get("passes"))
                             if getattr(args, "pool", False) else None),
            "gate": {"min_marginal": getattr(args, "min_marginal", 0.3),
                     "max_single": getattr(args, "gate", 0.70),
                     "min_sharpe": getattr(args, "min_sharpe", 1.0)}}


def _book_stability(book, min_move=0.3, limit=3):
    """How many book alphas earn their train Sharpe in only one half of train.

    A single train Sharpe can be the average of two opposite regimes, and the test
    year may not be used to select. Splitting train in half keeps the evidence inside
    the window selection is allowed to read, so a flip is visible at session start
    instead of after a submission disappoints.

    Honest limit, worth keeping in the code: this catches *train-internal* regime
    dependence, not future decay. `omEaaMl2` is stable by this measure (2.50 -> 1.81)
    and still turned negative the following year.
    """
    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"
    gaps, flips, late = [], [], []
    for alpha_id in book:
        path = pnl_dir / ("%s.json" % alpha_id)
        if not path.exists():
            continue
        try:
            pnl = {k: float(v) for k, v in json.loads(path.read_text()).items()}
        except (OSError, ValueError):
            continue
        # The halves must stay inside train: the test year is not selection evidence,
        # and splitting the whole series would quietly make it so.
        entry = (book.get(alpha_id) if isinstance(book, dict) else None) or {}
        test_start = ((entry.get("out_of_sample") or {}).get("test") or {}).get("startDate")
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


def _book_duplicates(book, gate=0.70, prefilter=0.75, limit=8):
    """Book pairs that break the platform's own correlation gate.

    Why this belongs in `brief`: the platform compares a submission against a
    *periodically refreshed snapshot*, so two alphas 0.88 correlated can both sit in
    the book when the second was submitted before the snapshot refreshed. Measured on
    2026-11-05, the book holds **eight** pairs above the 0.70 gate - `O0J6Aeag` and
    `gJWre33J` at 0.975 among them - and nothing was scanning the book after the fact.

    Two stages, because 1275 pairs on 1236 daily points is too slow for a session
    start: a stride prefilter on every fifth day at 0.75, then the survivors are
    confirmed on full daily PnL at the real gate. The summary therefore reports the
    gate the platform would have applied, not the prefilter.
    """
    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"
    series = {}
    for alpha_id in book:
        path = pnl_dir / ("%s.json" % alpha_id)
        if not path.exists():
            continue
        try:
            series[alpha_id] = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    # Prefilter on every fifth day rather than on monthly buckets: monthly averaging
    # lost two of the eight violations (a 0.77 daily pair can read 0.80 monthly), and a
    # screen that under-reports is worse than no screen. A stride keeps ~250 points per
    # pair, which is cheap and tracks daily closely enough to be conservative at 0.80.
    sampled = {name: {day: values[day] for day in sorted(values)[::5]}
               for name, values in series.items()}
    names = sorted(sampled)
    suspects = []
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            x, y, dates = correlation.align(sampled[left], sampled[right], min_overlap=40)
            if not x:
                continue
            value = correlation.pearson(x, y)
            if value is None or abs(value) < prefilter:
                continue
            suspects.append((left, right, value))
    hits = []
    for left, right, strided_r in suspects:
        x, y, dates = correlation.align(series[left], series[right], min_overlap=200)
        if not x:
            continue
        value = correlation.pearson(x, y)
        if value is None or abs(value) < gate:
            continue
        hits.append({"a": left, "b": right, "r": round(value, 4),
                     "strided_r": round(strided_r, 3), "days": len(dates)})
    hits.sort(key=lambda row: abs(row["r"]), reverse=True)
    return {"gate": gate, "prefilter": prefilter, "frequency": "daily",
            "count": len(hits), "pairs": hits[:limit]}


def cmd_brief(args, _sink):
    """The state of the campaign in one bounded document.

    Every session starts by reconstructing where the research stands, and the
    material for that is scattered: a multi-megabyte ledger, a directory of
    date-stamped write-ups, a hand-written weekly note, and whatever the last
    session remembered. Reading all of it is the expensive part of the work, and it
    grows every week while the answer stays the same size.

    So this emits the small part that moves - what the book is, what ran lately and
    how it scored, which write-ups are new, and the standing open questions - and
    nothing else. It is deliberately not a report; it is the input to one.

    The open-question list is read from ``open-threads.md`` in the research
    directory, one ``- [open] ...`` or ``- [closed] ...`` line each. That file is
    maintained by hand on purpose: a question worth carrying across sessions is a
    judgement, and anything derived automatically would be a summary of the past
    rather than a list of what is still undecided.
    """
    from datetime import datetime, timedelta, timezone

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
        return (entry.get("metrics") or {}).get(name)

    book = {}
    for entry in entries:
        if (entry.get("status") or "").upper() != "ACTIVE":
            continue
        alpha_id = entry.get("alpha_id")
        if alpha_id:
            # The ledger is append-only, so the last row for an alpha is its current
            # truth. Keeping the first one hid fields added later - the out-of-sample
            # split among them, which is what the stability split needs.
            book[alpha_id] = entry

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
             "verdict": entry.get("verdict"), "idea": entry.get("idea"),
             "when": entry.get("recorded_at")}
            for entry in scored[:args.top]]

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
        if len(documents) >= args.top * 2:
            break

    threads = {"open": [], "closed": []}
    thread_file = research_dir / "open-threads.md"
    if thread_file.exists():
        for line in thread_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            for tag, key in (("[open]", "open"), ("[closed]", "closed")):
                if text.startswith("- " + tag):
                    body = text[len(tag) + 2:].strip()
                    if body:
                        threads[key].append(body)

    # Label debt and recording gaps: both were invisible until someone counted by
    # hand, and both make the ledger look complete while it is not.
    unlabeled = sum(1 for entry in book.values()
                    if not entry.get("family") or not entry.get("mechanism"))
    no_checks = sum(1 for entry in entries
                    if (entry.get("metrics") or {}) and not (entry.get("checks") or {}))
    data = {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_days": days,
        "ledger": {"rows": len(entries), "path": registry_path,
                   "book_rows_missing_labels": unlabeled,
                   "rows_with_metrics_but_no_checks": no_checks},
        "book": {
            "count": len(book),
            "sharpe_median": (round(sorted(sharpes)[len(sharpes) // 2], 3)
                              if sharpes else None),
            "by_family": dict(sorted(families.items(), key=lambda item: -item[1])),
            "by_universe": dict(sorted(universes.items(), key=lambda item: -item[1])),
            # Same-bet pairs the submission gate could have missed (snapshot lag).
            "self_duplicates": _book_duplicates(sorted(book)),
            # A train Sharpe can hide a single-regime edge; the halves stay in train.
            "stability": _book_stability(book),
        },
        "recent": {"runs": len(runs), "by_verdict": by_verdict, "best": best},
        "documents": documents,
        # The divergence trigger: how many of the last batches produced anything additive.
        "convergence": _convergence_snapshot(),
        "drift": (_book_drift(book, _drift_check().get("platform"))
                  if getattr(args, "drift", False) else None),
        "threads": threads,
    }

    return data
