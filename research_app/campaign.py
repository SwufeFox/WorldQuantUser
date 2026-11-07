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
        "threads": threads,
    }

    return data
