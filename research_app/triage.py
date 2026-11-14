"""One command that answers "what do I do with this".

After a probe the same questions came up every time, and each was answered by a throwaway
script: does the platform accept it, how correlated is it with the book, and how much would
the book actually move. Selection uses train metrics and train PnL. ``--freeze`` saves
the ordered training choice; ``--show-test --selection`` adds validation evidence
in exactly that order, with the saved criteria and explicit source/book drift.

This module is the consolidation: **one scorer**, applied to a candidate, a batch file, or
the whole book, with the internal benchmark available as a flag rather than a religion.

The two numbers that matter and are easy to confuse:

* ``delta`` - the book's Sharpe after adding this alpha equal-weight, minus the book now.
  Not the slope ``S_a - rho*S_B``, which on a sixty-member book is a different magnitude.
* ``retention`` - test Sharpe over train Sharpe, available only with ``show_test``
  and a persisted training selection. Cost scenarios remain period-specific evidence.
"""

import json
import os

import alpha_registry
import research_paths

from . import benchmark, costs, legs, mechanisms, selection
from .runtime import CliError


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
    failures = set(entry.get("failing_checks") or [])
    for name, body in (entry.get("checks") or {}).items():
        result = body.get("result") if isinstance(body, dict) else body
        if result in ("FAIL", "ERROR"):
            failures.add(name)
    return sorted(failures)


def _sub_universe(entry):
    """How much room the sub-universe check has: value / limit.

    The check compares Sharpe in the smaller half of the universe against a floor. An
    alpha that clears it by two percent is not the same as one that clears it by fifty:
    the first may be earning its edge in names whose execution is worse, which is the one
    thing a daily P&L cannot show. Reported as a margin so the reader can judge.
    """
    body = (entry.get("checks") or {}).get("LOW_SUB_UNIVERSE_SHARPE")
    if not isinstance(body, dict):
        return None
    value, limit = body.get("value"), body.get("limit")
    if value is None or not limit:
        return None
    return {"value": value, "limit": limit, "margin": round(value / limit, 3)}


def _unresolved(entry):
    """Checks the platform never resolved for this record.

    A simulation response carries SELF_CORRELATION as PENDING: it is computed by the
    check endpoint afterwards. Treating a PENDING as "not failing" made every unresolved
    candidate look clean, which is how a 0.99 self-correlation sat on the shelf looking
    like 0.44.
    """
    checks = entry.get("checks") or {}
    names = []
    for name in sorted(set(checks) | {"SELF_CORRELATION", "MATCHES_COMPETITION"}):
        body = checks.get(name)
        result = (body or {}).get("result") if isinstance(body, dict) else body
        if result not in ("PASS", "FAIL"):
            names.append(name)
    return names


def _round(value, places=3):
    return None if value is None else round(value, places)


def _cohort_retentions(latest, days=45):
    """Retention distribution of recently recorded arms, for a percentile.

    A fixed floor can hide cohort drift. Compare a candidate with its recent peers
    to distinguish a weak cohort from an unusually weak member of that cohort.
    """
    import datetime

    cutoff = (datetime.datetime.now(datetime.timezone.utc)
              - datetime.timedelta(days=days)).isoformat()[:10]
    values = []
    for entry in latest.values():
        recorded = (entry.get("recorded_at") or "")[:10]
        if not recorded or recorded < cutoff:
            continue
        split = entry.get("out_of_sample") or {}
        train = (split.get("train") or {}).get("sharpe")
        test = (split.get("test") or {}).get("sharpe")
        if not train or test is None or train <= 1.0:
            continue
        values.append(test / train)
    return sorted(values)


def _percentile(values, value):
    if not values or value is None:
        return None
    below = sum(1 for item in values if item < value)
    return round(100.0 * below / len(values), 0)


def _score(alpha_id, entry, series, dates, criteria):
    """The four questions plus the internal bar, for one alpha."""
    directory = criteria["pnl_dir"]
    n = len(series)
    split = entry.get("out_of_sample") or {}
    train_metrics = split.get("train") or {}
    train = train_metrics.get("sharpe")
    show_test = criteria["show_test"]
    metrics = (entry.get("metrics") or {}) if show_test else train_metrics

    row = {
        "alpha_id": alpha_id,
        "sharpe": metrics.get("sharpe"),
        "fitness": metrics.get("fitness"),
        "train_sharpe": train,
        "metrics_scope": "full" if show_test else "train",
        "overlap_days": 0,
        "rho_book": None,
        "slope": None,
        "delta": None,
        "ops": (entry.get("expression") or "").count("("),
    }
    row["costs"] = costs.evidence(metrics, row["metrics_scope"], criteria.get("cost_bps"))
    row["source_settings"] = entry.get("settings") or {}
    row["source_expression"] = entry.get("expression")
    row["correlation_policy"] = {"local_duplicate_threshold": 0.9,
                                 "basis": "local_cached_pnl_heuristic",
                                 "platform_pass_inferred": False}
    if show_test:
        test = (split.get("test") or {}).get("sharpe")
        retention = None if not train or test is None else round(test / train, 2)
        row.update({"test_sharpe": test, "retention": retention,
                    "retention_percentile": _percentile(criteria.get("cohort") or [], retention),
                    "failing": _failing(entry), "unresolved": _unresolved(entry),
                    "sub_universe": _sub_universe(entry)})
        row["costs_by_period"] = {
            "train": costs.evidence(train_metrics, "train", criteria.get("cost_bps")),
            "test": costs.evidence(split.get("test") or {}, "test", criteria.get("cost_bps")),
        }
        row["platform_checks"] = entry.get("checks") or {}

    in_book = alpha_id in series
    candidate = legs.load_pnl(alpha_id, directory)
    if candidate and dates:
        common = [day for day in dates if day in candidate]
        row["overlap_days"] = len(common)
        if len(common) >= 120:
            book_slice = [sum(values[day] for values in series.values()) for day in common]
            values = [candidate[day] for day in common]
            rho = legs.correlation(values, book_slice)
            # Local cached-PnL correlations are heuristic evidence. Platform
            # check values and thresholds are displayed separately when available.
            member_rhos = []
            for member, member_values in series.items():
                other = legs.correlation(values, [member_values[day] for day in common])
                if other is not None:
                    member_rhos.append((other, member))
            own = legs.sharpe_of(values)
            now = legs.sharpe_of(book_slice)
            if member_rhos:
                worst = max(member_rhos, key=lambda item: abs(item[0]))
                # Magnitude reports strong relationships in either direction;
                # it does not establish the platform's correlation verdict.
                row["max_member_rho"] = round(abs(worst[0]), 3)
                row["max_member_rho_signed"] = round(worst[0], 3)
                row["max_member"] = worst[1]
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
        # The row carries neither field: the block has to come off the ledger entry, or
        # be derived from its expression. Reading the row silently gave "other" for
        # everything, which is the one block with the worst measured ratio.
        block = (row.get("block") or entry.get("block")
                 or mechanisms.block_of(entry.get("expression") or ""))
        factors = {
            "shrink": benchmark.shrink_factor(train),
            "contribution": benchmark.contribution_factor(
                row["delta"], criteria["delta_full_credit"]),
            "block": benchmark.block_factor(block, criteria),
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
    if row.get("failing"):
        return "blocked", "fails %s" % ", ".join(row["failing"])
    if (row.get("max_member_rho") or 0) > 0.9:
        return ("duplicate", "%.2f correlated with %s; exceeds local 0.90 duplicate "
                "heuristic, not a platform check" % (row["max_member_rho"], row.get("max_member")))
    if row["delta"] is None:
        return "unknown", "insufficient overlapping PnL in this view (%d days; need 120)" % row["overlap_days"]
    # Disqualifying facts come before "not measured yet": an alpha that already lowers
    # the book does not become interesting once its checks resolve.
    if row["delta"] <= 0:
        return "dilutive", "adding it lowers the book"
    floor = criteria["retention_floor"]
    if row.get("retention") is not None and row["retention"] < floor:
        return "overfit-risk", "test keeps %.0f%% of train" % (row["retention"] * 100)
    if row.get("unresolved"):
        return "unverified", ("never resolved: %s - run `alpha check` before treating "
                              "this as passing" % ", ".join(row["unresolved"]))
    if row["delta"] < criteria["min_delta"]:
        return "weak", "adds %.3f, under the %.2f bar" % (row["delta"], criteria["min_delta"])
    if not criteria["show_test"]:
        return "validate", "train contribution clears the bar; platform and test checks not inspected"
    return "shelve", "recorded checks resolved and moves the book +%.3f; recheck platform eligibility before submission" % row["delta"]


def triage(ids=None, registry_path=None, pnl_dir=None, min_delta=None,
           retention_floor=None, internal=False, whole_book=False, criteria=None,
           show_test=False, train_end=None, freeze_path=None, selection_path=None,
           cost_bps=None):
    """Freeze training decisions and join validation evidence without reranking."""
    from datetime import date
    from .book_state import describe_book

    if show_test and not selection_path:
        raise ValueError("--show-test requires --selection from a training freeze")
    if freeze_path and (show_test or selection_path):
        raise ValueError("--freeze is only available for a new train selection")
    if selection_path and not show_test:
        raise ValueError("--selection requires --show-test")
    if internal and not show_test:
        raise ValueError("--internal uses test-calibrated factors; freeze the design and pass --show-test")
    manifest = selection.load(selection_path, source="triage") if selection_path else None
    if train_end is not None:
        try:
            if date.fromisoformat(train_end).isoformat() != train_end:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError("--train-end must be an ISO date (YYYY-MM-DD)") from None
    if manifest:
        frozen_ids = [row["alpha_id"] for row in manifest["rows"]]
        if ids and set(ids) != set(frozen_ids):
            raise ValueError("candidate IDs conflict with the frozen selection")
        inputs = manifest["inputs"]
        for key, provided in (("registry_path", registry_path), ("pnl_dir", pnl_dir)):
            if provided is not None and os.path.abspath(os.path.expanduser(os.fspath(provided))) != inputs.get(key):
                raise ValueError("%s conflicts with the frozen selection" % key)
        if whole_book and not inputs.get("whole_book"):
            raise ValueError("--book conflicts with the frozen selection")
        if train_end is not None and train_end != manifest["window"].get("end"):
            raise ValueError("train end conflicts with the frozen selection")
        registry_path, pnl_dir = inputs.get("registry_path"), inputs.get("pnl_dir")
        whole_book = inputs.get("whole_book", False)
        saved = manifest["criteria"]
        for key, value in (("min_delta", min_delta), ("retention_floor", retention_floor)):
            if value is not None and value != saved.get(key):
                raise ValueError("%s conflicts with frozen criteria" % key)
        if criteria:
            requested = benchmark.refine_criteria(criteria) if isinstance(criteria, str) else criteria
            if any(saved.get(key) != value for key, value in requested.items()):
                raise ValueError("criteria conflict with the frozen selection")
        if cost_bps is not None and list(cost_bps) != saved.get("cost_bps"):
            raise ValueError("cost scenarios conflict with the frozen selection")
    latest = {}
    for entry in alpha_registry.load_entries(registry_path):
        key = entry.get("alpha_id") or entry.get("key")
        if key:
            latest[key] = entry
    directory = os.path.abspath(os.path.expanduser(os.fspath(
        pnl_dir or os.path.join(research_paths.ensure_dir(), "pnl"))))
    series, dates = _book_series(latest, directory)
    if not dates and not show_test:
        raise ValueError("no cached book PnL; the book cannot be scored")
    book_state = describe_book(latest, series=series, registry_path=registry_path, pnl_dir=directory)
    targets = frozen_ids if manifest else (sorted(series) if whole_book else list(dict.fromkeys(ids or [])))
    window = {"mode": "full" if show_test else "train", "end": None, "source": None}
    if not show_test:
        boundaries = []
        for alpha_id in set(series) | set(targets):
            split = (latest.get(alpha_id) or {}).get("out_of_sample") or {}
            start = (split.get("test") or {}).get("startDate")
            if start:
                try:
                    boundaries.append(date.fromisoformat(str(start)[:10]).isoformat())
                except ValueError:
                    raise ValueError("invalid recorded test boundary for %s" % alpha_id) from None
        cutoff = min(boundaries) if boundaries else "2023-01-01"
        source = "recorded_test_start" if boundaries else "default_train_end"
        if train_end is not None:
            cutoff = min(cutoff, train_end) if boundaries else train_end
            source = "explicit_and_recorded" if boundaries else "explicit"
        window.update({"end": cutoff, "source": source, "end_exclusive": True})
        dates = [day for day in dates if day < cutoff]
        if not dates:
            raise ValueError("no cached book PnL before train end %s" % cutoff)
    window.update({"observed_start": dates[0] if dates else None,
                   "observed_end": dates[-1] if dates else None, "days": len(dates)})
    book_sharpe = legs.sharpe_of([sum(values[day] for values in series.values()) for day in dates])
    if book_sharpe is None and not show_test:
        raise ValueError("the book's own Sharpe is not measurable on the cached days")
    if manifest:
        decision_criteria = dict(manifest["criteria"])
    else:
        default_criteria = os.path.join(research_paths.ensure_dir(), "benchmark-criteria.json")
        decision_criteria = benchmark.refine_criteria(default_criteria)
        if criteria:
            decision_criteria.update(benchmark.refine_criteria(criteria)
                                     if isinstance(criteria, str) else criteria)
        for key in ("show_test", "internal", "cohort", "pnl_dir"):
            decision_criteria.pop(key, None)
        decision_criteria.update({"min_delta": 0.02 if min_delta is None else min_delta,
                                  "retention_floor": 0.2 if retention_floor is None else retention_floor,
                                  "cost_bps": list(costs.DEFAULT_COST_BPS if cost_bps is None else cost_bps)})
    costs.evidence({}, "train", decision_criteria["cost_bps"])
    settings = {**decision_criteria, "pnl_dir": directory, "internal": bool(internal),
                "show_test": bool(show_test), "cohort": _cohort_retentions(latest) if show_test else []}
    rows = []
    for alpha_id in targets:
        entry = latest.get(alpha_id)
        if entry is None:
            if not manifest:
                rows.append({"alpha_id": alpha_id, "verdict": "unknown", "missing": True,
                             "why": "not in the ledger", "costs": costs.evidence({}, "train", settings["cost_bps"])})
            continue
        rows.append(_score(alpha_id, entry, series, dates, settings))
    if manifest:
        rows = selection.ordered_rows(manifest, rows)
        for row, frozen in zip(rows, manifest["rows"]):
            row["source_changes"] = {
                key: {"frozen": frozen.get(key), "current": row.get(key)}
                for key in ("source_settings", "source_expression")
                if not row.get("missing") and frozen.get(key) != row.get(key)}
    else:
        rows.sort(key=lambda item: (item.get("delta") is None,
                                   -(item["delta"] if item.get("delta") is not None else 0)))
    result = {
        "book_sharpe": _round(book_sharpe), "book_members": len(series),
        "book_state": book_state, "mode": "book" if whole_book else "candidates",
        "internal": bool(internal), "window": window, "criteria": decision_criteria,
        "rows": rows,
        "note": ("Validation evidence in immutable training order; missing candidates remain missing."
                 if show_test else "Selection view: train metrics and PnL only. Missing train metrics stay unknown."),
    }
    if freeze_path:
        manifest = selection.freeze(freeze_path, source="triage", rows=rows,
                                    criteria=decision_criteria, window=window,
                                    inputs={"registry_path": os.path.abspath(os.path.expanduser(os.fspath(
                                                registry_path or alpha_registry.default_registry_path()))),
                                            "pnl_dir": directory, "whole_book": bool(whole_book),
                                            "book_state": book_state})
    if manifest:
        result["selection_id"] = manifest["selection_id"]
        result["selection_path"] = os.path.abspath(os.fspath(selection_path or freeze_path))
        result["frozen_window"] = manifest["window"]
        result["frozen_book_state"] = manifest["inputs"].get("book_state")
        result["book_changed"] = book_state != result["frozen_book_state"]
    if show_test:
        research_paths.log_view("triage", "test", [row["alpha_id"] for row in rows],
                                note={"selection_id": manifest["selection_id"],
                                      "selection_path": result["selection_path"]})
    return result


def cmd_triage(args, _sink):
    """Triage a batch file, explicit ids, or the whole book."""
    try:
        return _cmd_triage(args)
    except ValueError as exc:
        raise CliError("INVALID_INPUT", str(exc)) from exc


def _cmd_triage(args):
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
    if not ids and not getattr(args, "book", False) and not getattr(args, "selection", None):
        raise ValueError("nothing to triage: pass a batch file, ids, --book, or --selection")
    show_test = bool(getattr(args, "show_test", False))
    result = triage(ids, getattr(args, "registry", None),
                    min_delta=getattr(args, "min_delta", None), retention_floor=getattr(args, "retention_floor", None),
                    internal=getattr(args, "internal", False),
                    whole_book=getattr(args, "book", False), show_test=show_test,
                    train_end=getattr(args, "train_end", None),
                    freeze_path=getattr(args, "freeze", None),
                    selection_path=getattr(args, "selection", None),
                    cost_bps=getattr(args, "cost_bps", None))
    return result
