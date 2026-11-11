"""One command that answers "what do I do with this batch".

After a probe the same four questions came up every time, and each one was answered by a
throwaway script: how does this row look out of sample, how correlated is it with the book,
how much would the book actually move if it were added, and does the platform accept it.
That is the drift the tooling had become: the loop worked, but the last mile was manual.

``triage`` reads a batch file (or explicit ids) and returns one table with the verdict. It
does not simulate and it does not submit.

The two numbers worth explaining:

* ``delta`` - the book's Sharpe after adding this alpha equal-weight, minus the book's
  Sharpe now. Not the slope ``S_a - rho * S_B``, which on a sixty-member book is a
  different magnitude entirely.
* ``retention`` - test-period Sharpe divided by train-period Sharpe. It is reported and
  can *veto* a candidate, never rank one: preferring the better test number would be
  selecting on the test year, which is the one thing this workspace refuses to do.
"""

import json
import os

import alpha_registry
import research_paths

from . import legs


def _load_pnl(alpha_id, directory):
    return legs.load_pnl(alpha_id, directory)


def _book_series(latest, directory):
    book = [key for key, entry in latest.items()
            if str(entry.get("status") or "").upper() in ("ACTIVE", "SUBMITTED")]
    series = {}
    for alpha_id in book:
        values = _load_pnl(alpha_id, directory)
        if values:
            series[alpha_id] = values
    dates = None
    for values in series.values():
        dates = set(values) if dates is None else dates & set(values)
    return series, sorted(dates or [])


def _split_sharpes(entry):
    split = entry.get("out_of_sample") or {}
    train = (split.get("train") or {}).get("sharpe")
    test = (split.get("test") or {}).get("sharpe")
    return train, test


def _retention(train, test):
    if not train or test is None:
        return None
    return round(test / train, 2)


def _failing(entry):
    return sorted(entry.get("failing_checks") or [])


def triage(ids, registry_path=None, pnl_dir=None, min_delta=0.02, retention_floor=0.2):
    """Score each id on the four questions and return one row per id."""
    latest = {}
    for entry in alpha_registry.load_entries(registry_path):
        key = entry.get("alpha_id") or entry.get("key")
        if key:
            latest[key] = entry

    directory = pnl_dir or os.path.join(research_paths.ensure_dir(), "pnl")
    series, dates = _book_series(latest, directory)
    if not dates:
        raise ValueError("no cached book PnL; the book cannot be scored")
    book_daily = [sum(values[day] for values in series.values()) for day in dates]
    book_sharpe = legs.sharpe_of(book_daily)
    if book_sharpe is None:
        raise ValueError("the book's own Sharpe is not measurable on the cached days")
    n = len(series)

    rows = []
    for alpha_id in ids:
        entry = latest.get(alpha_id)
        if entry is None:
            rows.append({"alpha_id": alpha_id, "verdict": "unknown",
                         "note": "not in the ledger"})
            continue
        candidate = _load_pnl(alpha_id, directory)
        train, test = _split_sharpes(entry)
        retention = _retention(train, test)
        metrics = entry.get("metrics") or {}
        row = {
            "alpha_id": alpha_id,
            "sharpe": metrics.get("sharpe"),
            "fitness": metrics.get("fitness"),
            "retention": retention,
            "failing": _failing(entry),
            "rho_book": None, "delta": None,
        }
        if candidate:
            common = [day for day in dates if day in candidate]
            if len(common) >= 120:
                book_slice = [sum(values[day] for values in series.values()) for day in common]
                values = [candidate[day] for day in common]
                row["rho_book"] = _round(legs.correlation(values, book_slice))
                combined = [(b + v) / (n + 1) for b, v in zip(book_slice, values)]
                now = legs.sharpe_of(book_slice)
                after = legs.sharpe_of(combined)
                if now is not None and after is not None:
                    row["delta"] = round(after - now, 4)
        row["verdict"], row["why"] = _verdict(row, min_delta, retention_floor)
        rows.append(row)

    rows.sort(key=lambda item: -(item.get("delta") or -9))
    return {
        "book_sharpe": round(book_sharpe, 3),
        "book_members": n,
        "criteria": {"min_delta": min_delta, "retention_floor": retention_floor},
        "rows": rows,
        "note": ("delta is the book's Sharpe change from adding this alpha; retention is "
                 "test/train and can only veto, never rank. `shelve` means all checks "
                 "pass, the book moves up, and the out-of-sample behaviour is not "
                 "obviously broken - it is still a decision, not a permission."),
    }


def _round(value):
    return None if value is None else round(value, 3)


def _verdict(row, min_delta, retention_floor):
    if row["failing"]:
        return "blocked", "fails %s" % ", ".join(row["failing"])
    if row["delta"] is None:
        return "unknown", "no cached PnL to measure the book against"
    if row["delta"] <= 0:
        return "dilutive", "adding it lowers the book"
    if row["retention"] is not None and row["retention"] < retention_floor:
        return "overfit-risk", "test keeps %.0f%% of train" % (row["retention"] * 100)
    if row["delta"] < min_delta:
        return "weak", "adds %.3f, under the %.2f bar" % (row["delta"], min_delta)
    return "shelve", "passes checks and moves the book +%.3f" % row["delta"]


def cmd_triage(args, _sink):
    """Triage a batch file, a list of ids, or the whole pool."""
    ids = list(getattr(args, "ids", None) or [])
    if getattr(args, "batch", None):
        path = os.path.expanduser(args.batch)
        try:
            document = json.loads(open(path, encoding="utf-8").read())
        except (OSError, ValueError) as exc:
            raise ValueError("cannot read batch file %s: %s" % (path, exc))
        ids.extend(row.get("alpha_id") for row in document.get("rows") or []
                   if row.get("alpha_id"))
    if not ids:
        raise ValueError("nothing to triage: pass a batch file or ids")
    return triage([alpha_id for alpha_id in dict.fromkeys(ids)
                   if alpha_id and alpha_id != "None"],
                  getattr(args, "registry", None),
                  min_delta=args.min_delta, retention_floor=args.retention_floor)