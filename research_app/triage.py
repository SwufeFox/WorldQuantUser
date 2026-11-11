"""One command that answers "what do I do with this".

After a probe the same questions came up every time, and each was answered by a throwaway
script: does the platform accept it, how correlated is it with the book, how much would the
book actually move, and how much of it survives the test year. `triage`, `health`,
`marginal` and `benchmark` grew out of that one need, one command per question - which is
how a toolset starts costing more than it saves.

This module is the consolidation: **one scorer**, applied to a candidate, a batch file, or
the whole book, with the internal benchmark available as a flag rather than a religion.

The two numbers that matter and are easy to confuse:

* ``delta`` - the book's Sharpe after adding this alpha equal-weight, minus the book now.
  Not the slope ``S_a - rho*S_B``, which on a sixty-member book is a different magnitude.
* ``retention`` - test Sharpe over train Sharpe. It can veto, never rank: preferring a
  better test number would be selecting on the test year.
"""

import json
import os

import alpha_registry
import research_paths

from . import benchmark, legs


def _book_series(latest, directory):
    book = [key for key, entry in latest.items()
            if str(entry.get("status") or "").upper() in ("ACTIVE", "SUBMITTED")]
    series = {}
    for alpha_id in book:
        values = legs.load_pnl(alpha_id, directory)
        if values:
            series[alpha_id] = values
    dates = None
    for values in series.values():
        dates = set(values) if dates is None else dates & set(values)
    return series, sorted(dates or [])


def _failing(entry):
    return sorted(entry.get("failing_checks") or [])


def _round(value, places=3):
    return None if value is None else round(value, places)


def _score(alpha_id, entry, series, dates, criteria):
    """The four questions plus the internal bar, for one alpha."""
    directory = criteria["pnl_dir"]
    n = len(series)
    metrics = entry.get("metrics") or {}
    split = entry.get("out_of_sample") or {}
    train = (split.get("train") or {}).get("sharpe")
    test = (split.get("test") or {}).get("sharpe")
    retention = None if not train or test is None else round(test / train, 2)

    row = {
        "alpha_id": alpha_id,
        "sharpe": metrics.get("sharpe"),
        "fitness": metrics.get("fitness"),
        "train_sharpe": train,
        "test_sharpe": test,
        "retention": retention,
        "failing": _failing(entry),
        "rho_book": None,
        "slope": None,
        "delta": None,
        "ops": (entry.get("expression") or "").count("("),
    }

    in_book = alpha_id in series
    candidate = legs.load_pnl(alpha_id, directory)
    if candidate and dates:
        common = [day for day in dates if day in candidate]
        if len(common) >= 120:
            book_slice = [sum(values[day] for values in series.values()) for day in common]
            values = [candidate[day] for day in common]
            rho = legs.correlation(values, book_slice)
            own = legs.sharpe_of(values)
            now = legs.sharpe_of(book_slice)
            if rho is not None:
                row["rho_book"] = round(rho, 3)
                if own is not None and now is not None:
                    row["slope"] = round(own - rho * now, 3)
            if in_book:
                # A member is already in the book, so the question is what removing it
                # costs: leave-one-out, the same number `health` reported.
                without = [(b - v) / (n - 1) for b, v in zip(book_slice, values)] if n > 1 else []
                if without:
                    left = legs.sharpe_of(without)
                    if now is not None and left is not None:
                        row["delta"] = round(now - left, 4)
                        row["member"] = True
            else:
                after = legs.sharpe_of([(b + v) / (n + 1) for b, v in zip(book_slice, values)])
                if now is not None and after is not None:
                    row["delta"] = round(after - now, 4)

    if criteria.get("internal"):
        factors = {
            "shrink": benchmark.shrink_factor(train),
            "contribution": benchmark.contribution_factor(
                row["delta"], criteria["delta_full_credit"]),
        }
        combined = 1.0
        for value in factors.values():
            combined *= value
        fitness = row["fitness"]
        row["internal_fitness"] = None if fitness is None else round(fitness * combined, 3)
        row["factors"] = {name: round(value, 3) for name, value in factors.items()}
        row["ops_warning"] = benchmark.ops_warning(row["ops"], criteria["ops_warn"])

    row["verdict"], row["why"] = _verdict(row, criteria)
    return row


def _verdict(row, criteria):
    if row.get("member"):
        if not row["delta"]:
            return "member", "no measurable contribution"
        direction = "raises" if row["delta"] < 0 else "lowers"
        return ("drag" if row["delta"] < 0 else "carries"), (
            "removing it %s the book by %.4f" % (direction, abs(row["delta"])))
    if row["failing"]:
        return "blocked", "fails %s" % ", ".join(row["failing"])
    if row["delta"] is None:
        return "unknown", "no cached PnL to measure the book against"
    if row["delta"] <= 0:
        return "dilutive", "adding it lowers the book"
    floor = criteria["retention_floor"]
    if row["retention"] is not None and row["retention"] < floor:
        return "overfit-risk", "test keeps %.0f%% of train" % (row["retention"] * 100)
    if row["delta"] < criteria["min_delta"]:
        return "weak", "adds %.3f, under the %.2f bar" % (row["delta"], criteria["min_delta"])
    return "shelve", "passes checks and moves the book +%.3f" % row["delta"]


def triage(ids=None, registry_path=None, pnl_dir=None, min_delta=0.02,
           retention_floor=0.2, internal=False, whole_book=False, criteria=None):
    """Score candidates, a batch, or the book itself with one scorer."""
    latest = {}
    for entry in alpha_registry.load_entries(registry_path):
        key = entry.get("alpha_id") or entry.get("key")
        if key:
            latest[key] = entry

    directory = pnl_dir or os.path.join(research_paths.ensure_dir(), "pnl")
    series, dates = _book_series(latest, directory)
    if not dates:
        raise ValueError("no cached book PnL; the book cannot be scored")
    book_sharpe = legs.sharpe_of([sum(values[day] for values in series.values())
                                  for day in dates])
    if book_sharpe is None:
        raise ValueError("the book's own Sharpe is not measurable on the cached days")

    settings = {"pnl_dir": directory, "min_delta": min_delta,
                "retention_floor": retention_floor, "internal": internal}
    settings.update(benchmark.DEFAULTS)
    if criteria:
        settings.update(criteria)

    targets = sorted(series) if whole_book else list(ids or [])
    rows = []
    for alpha_id in targets:
        entry = latest.get(alpha_id)
        if entry is None:
            rows.append({"alpha_id": alpha_id, "verdict": "unknown",
                         "why": "not in the ledger"})
            continue
        rows.append(_score(alpha_id, entry, series, dates, settings))

    rows.sort(key=lambda item: -(item.get("delta") or -9))
    return {
        "book_sharpe": round(book_sharpe, 3),
        "book_members": len(series),
        "mode": "book" if whole_book else "candidates",
        "internal": bool(internal),
        "criteria": {"min_delta": min_delta, "retention_floor": retention_floor},
        "rows": rows,
        "note": ("delta is the book's Sharpe change from adding this alpha; retention is "
                 "test/train and can only veto, never rank. `shelve` means the checks pass "
                 "and the book moves up - a decision, not a permission."),
    }


def cmd_triage(args, _sink):
    """Triage a batch file, explicit ids, or the whole book."""
    ids = list(getattr(args, "ids", None) or [])
    if getattr(args, "batch", None):
        path = os.path.expanduser(args.batch)
        try:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, ValueError) as exc:
            raise ValueError("cannot read batch file %s: %s" % (path, exc))
        ids.extend(row.get("alpha_id") for row in document.get("rows") or []
                   if row.get("alpha_id"))
    ids = [alpha_id for alpha_id in dict.fromkeys(ids) if alpha_id and alpha_id != "None"]
    if not ids and not getattr(args, "book", False):
        raise ValueError("nothing to triage: pass a batch file, ids, or --book")
    return triage(ids, getattr(args, "registry", None),
                  min_delta=args.min_delta, retention_floor=args.retention_floor,
                  internal=getattr(args, "internal", False),
                  whole_book=getattr(args, "book", False))