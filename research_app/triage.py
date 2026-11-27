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
import math
import os
from datetime import date

import alpha_registry
from check_evidence import normalize_checks, platform_check_names, summarize_checks
import machine_lib as ml
import research_paths

from . import benchmark, costs, legs, mechanisms, selection, stats, submission_planner
from .book_state import describe_book, member_digest, test_boundaries
from .runtime import CliError


MIN_OVERLAP_DAYS = stats.MIN_PORTFOLIO_OVERLAP


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
    failures.update(summarize_checks(entry.get("checks"))["failed"])
    return sorted(failures)


def _sub_universe(entry):
    """How much room the sub-universe check has: value / limit.

    The check compares Sharpe in the smaller half of the universe against a floor. An
    alpha that clears it by two percent is not the same as one that clears it by fifty:
    the first may be earning its edge in names whose execution is worse, which is the one
    thing a daily P&L cannot show. Reported as a margin so the reader can judge.
    """
    body = normalize_checks(entry.get("checks")).get("LOW_SUB_UNIVERSE_SHARPE")
    if not isinstance(body, dict):
        return None
    value, limit = body.get("value"), body.get("limit")
    if value is None or not limit:
        return None
    return {"value": value, "limit": limit, "margin": round(value / limit, 3)}


def _unresolved(entry):
    """Checks the platform has not yet resolved for this record.

    A simulation snapshot often carries SELF_CORRELATION as PENDING until the check
    endpoint evaluates it. Treating PENDING or ERROR as passing would let unverified
    candidates advance without a resolved correlation verdict.
    """
    required = {"SELF_CORRELATION", "MATCHES_COMPETITION"}
    if entry.get("checks_state") == "partial":
        prior = entry.get("last_check_snapshot") or {}
        required.update(platform_check_names(prior.get("checks")))
    summary = summarize_checks(entry.get("checks"), required=required)
    return sorted(set(summary["pending"]) | set(summary["errored"]))


def _round(value, places=3):
    return None if value is None else round(value, places)


def _observed_window(dates):
    return {"observed_start": dates[0] if dates else None,
            "observed_end": dates[-1] if dates else None, "days": len(dates)}


def _shelf_scope(path, targets):
    from . import buffer_cmd

    state = buffer_cmd.shelf_state(buffer_cmd.read_events(path))
    # A single candidate cannot duplicate itself. Its own shelf writes do not
    # invalidate a retry, but adding another peer changes the screening set.
    excluded = set(targets) if len(targets) == 1 else set()
    ids = sorted(alpha_id for alpha_id, entry in state.items()
                 if alpha_id not in excluded and entry.get("state") not in {"consumed", "dropped"})
    return {"path": os.path.abspath(os.path.expanduser(os.fspath(path))),
            "member_ids": ids, "member_digest": member_digest(ids)}, {
                alpha_id for alpha_id, entry in state.items()
                if entry.get("eligible") and entry.get("state") not in {"consumed", "dropped"}
            }


def _shelf_correlations(alpha_id, candidate, dates, peers, coverage, missing_boundaries, candidate_complete):
    measured, missing = [], []
    for peer, values in peers.items():
        if peer == alpha_id:
            continue
        shared = [day for day in dates if day in candidate and day in values]
        reason = None
        if alpha_id in missing_boundaries or peer in missing_boundaries:
            reason = "missing_train_boundary"
        elif not candidate or not values:
            reason = "missing_pnl"
        elif candidate_complete is not True or coverage[peer]["complete"] is not True:
            reason = "unverified_pnl_coverage"
        elif len(shared) < MIN_OVERLAP_DAYS:
            reason = "insufficient_overlap"
        else:
            rho = legs.correlation([candidate[day] for day in shared], [values[day] for day in shared])
            if rho is None:
                reason = "undefined_correlation"
            else:
                measured.append((rho, peer, len(shared)))
        if reason:
            missing.append({"alpha_id": peer, "reason": reason, "overlap_days": len(shared)})
    rho, peer, days = max(measured, key=lambda item: abs(item[0]), default=(None, None, None))
    return {"state": "unknown" if missing else "measured", "eligible": len(measured) + len(missing),
            "measured": len(measured), "missing": missing, "max_member": peer,
            "max_rho": abs(rho) if rho is not None else None, "max_rho_signed": rho,
            "max_member_overlap_days": days}


def _candidate_evidence(row, candidate, dates, window, book_state, missing_boundaries, directory):
    book, coverage = book_state["local"], book_state["pnl"]
    estimate = coverage["coverage"].get(row["alpha_id"])
    if estimate is None:
        estimate = ml.pnl_cache_status(directory, row["alpha_id"], candidate)
    overlap = [day for day in dates if day in candidate]
    reasons = []
    if missing_boundaries:
        reasons.append("missing_train_boundary")
    if estimate["complete"] is not True:
        reasons.append("unverified_candidate_pnl")
    if not coverage["complete"]:
        reasons.append("unverified_book_pnl")
    if row.get("delta") is None:
        reasons.append("unmeasurable_marginal")
    if row.get("book_correlations_measured") != book["member_count"] or row.get("rho_book") is None:
        reasons.append("incomplete_book_correlations")
    return {"state": "unknown" if reasons else "measured", "reasons": reasons,
            "window": window, "book_member_digest": book["member_digest"],
            "missing_train_boundary_ids": missing_boundaries,
            "candidate_pnl": {key: estimate.get(key) for key in ("points", "complete", "as_of", "error")},
            "book_pnl": {"complete": coverage["complete"], "members": book["member_count"],
                         "available": len(coverage["available_ids"]), "verified": len(coverage["complete_ids"])},
            "overlap": _observed_window(overlap)}


def _training_gate(row, min_delta, max_shelf_correlation):
    """Decide whether a training row can be frozen for automatic validation."""
    if row.get("shelf_correlation", {}).get("already_shelved"):
        return {"state": "rejected", "reason": "already_shelved"}
    if row.get("evidence", {}).get("state") != "measured":
        return {"state": "unknown", "reason": "incomplete_book_or_candidate_evidence"}
    shelf = row["shelf_correlation"]
    if shelf["state"] != "measured":
        return {"state": "unknown", "reason": "incomplete_shelf_evidence"}
    if row.get("member") or row.get("role") == "diagnostic":
        return {"state": "rejected", "reason": "not_a_candidate"}
    if row["delta"] <= 0 or row["delta"] < min_delta:
        return {"state": "rejected", "reason": "insufficient_train_marginal"}
    if row["verdict"] == "duplicate":
        return {"state": "rejected", "reason": "book_duplicate"}
    if shelf["max_rho"] is not None and shelf["max_rho"] >= max_shelf_correlation:
        return {"state": "rejected", "reason": "shelf_duplicate"}
    return {"state": "pass", "reason": None}


def _delta_segments(book, comparison, dates, member):
    midpoint = len(dates) // 2
    result = []
    for segment in (slice(None, midpoint), slice(midpoint, None)):
        before, after = legs.sharpe_of(book[segment]), legs.sharpe_of(comparison[segment])
        delta = None if before is None or after is None else (before - after if member else after - before)
        result.append({**_observed_window(dates[segment]), "delta": _round(delta, 4)})
    return result


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


def _score(alpha_id, entry, series, dates, criteria, candidate=None, *, book_daily=None, unit_book=None):
    """The four questions plus the internal bar, for one alpha."""
    directory = criteria["pnl_dir"]
    n = len(series)
    split = entry.get("out_of_sample") or {}
    train_metrics = split.get("train") or {}
    train = train_metrics.get("sharpe")
    show_test = criteria["show_test"]
    metrics = (entry.get("metrics") or {}) if show_test else train_metrics
    language = alpha_registry.entry_language(entry)
    expression = entry.get("expression") or ""

    row = {
        "alpha_id": alpha_id,
        "role": alpha_registry.evidence_role(entry),
        "sharpe": metrics.get("sharpe"),
        "fitness": metrics.get("fitness"),
        "train_sharpe": train,
        "metrics_scope": "full" if show_test else "train",
        "overlap_days": 0,
        "rho_book": None,
        "slope": None,
        "delta": None,
        "ops": None if ml.is_python_language(language) else expression.count("("),
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
    if candidate is None:
        candidate = legs.load_pnl(alpha_id, directory)
    if candidate and dates:
        common = [day for day in dates if day in candidate]
        row["overlap_days"] = len(common)
        if len(common) >= MIN_OVERLAP_DAYS:
            book_slice = ([book_daily[day] for day in common] if book_daily is not None else
                          [sum(values[day] for values in series.values()) for day in common])
            values = [candidate[day] for day in common]
            rho = legs.correlation(values, book_slice)
            # Local cached-PnL correlations are heuristic evidence. Platform
            # check values and thresholds are displayed separately when available.
            complete_book = unit_book if len(common) == len(dates) else None
            count, worst = stats.member_correlations(values, series, common, complete_book)
            row["book_correlations_measured"] = count
            own = legs.sharpe_of(values)
            now = legs.sharpe_of(book_slice)
            if worst is not None:
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
                comparison = [(b - v) / (n - 1) for b, v in zip(book_slice, values)] if n > 1 else []
                if comparison:
                    left = legs.sharpe_of(comparison)
                    if now is not None and left is not None:
                        row["delta"] = round(now - left, 4)
                        row["member"] = True
            else:
                comparison = [(b + v) / (n + 1) for b, v in zip(book_slice, values)]
                after = legs.sharpe_of(comparison)
                if now is not None and after is not None:
                    row["delta"] = round(after - now, 4)
            if comparison and not show_test:
                row["train_delta_by_half"] = _delta_segments(book_slice, comparison, common, in_book)

    if criteria.get("internal"):
        # The row carries neither field: the block has to come off the ledger entry, or
        # be derived from its expression. Reading the row silently gave "other" for
        # everything, which is the one block with the worst measured ratio.
        block = (row.get("block") or entry.get("block")
                 or mechanisms.block_of(expression, language))
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
        ops = row["ops"]
        row["ops_warning"] = None if ops is None else benchmark.ops_warning(ops, criteria["ops_warn"])

    row["verdict"], row["why"] = _verdict(row, criteria)
    return row


def _verdict(row, criteria):
    if row.get("role") == "diagnostic":
        return "diagnostic", "attribution evidence; not eligible for candidate selection"
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


def _candidate_submission_plan(rows, candidate_pnls, scoring_days, max_correlation=None):
    """Suggest an independent candidate subset using train-only PnL and marginal weights."""
    import itertools

    eligible = [
        row for row in rows
        if not row.get("missing") and not row.get("member") and row.get("role") != "diagnostic"
        and type(row.get("delta")) in (int, float) and math.isfinite(row["delta"]) and row["delta"] > 0
    ]
    if len(eligible) < 2:
        return None
    ids = [row["alpha_id"] for row in eligible]
    correlations = {}
    for a, b in itertools.combinations(ids, 2):
        series_a = candidate_pnls.get(a) or {}
        series_b = candidate_pnls.get(b) or {}
        if not series_a or not series_b:
            continue
        shared = [day for day in scoring_days if day in series_a and day in series_b]
        if len(shared) < MIN_OVERLAP_DAYS:
            continue
        rho = legs.correlation([series_a[day] for day in shared], [series_b[day] for day in shared])
        if rho is not None:
            correlations[(a, b)] = round(rho, 4)
    threshold = (max_correlation if max_correlation is not None
                 else submission_planner.DEFAULT_MAX_CORRELATION)
    planned = submission_planner.plan(
        [{"id": row["alpha_id"], "weight": row["delta"]} for row in eligible],
        correlations,
        max_correlation=threshold,
    )
    planned["weight_basis"] = "train_marginal_sharpe"
    return planned


def triage(ids=None, registry_path=None, pnl_dir=None, min_delta=None,
           retention_floor=None, internal=False, whole_book=False, criteria=None,
           show_test=False, train_end=None, freeze_path=None, selection_path=None,
           cost_bps=None, roles=None, source_files=None, against_shelf=False, shelf_path=None,
           max_shelf_correlation=None):
    """Freeze training decisions and join validation evidence without reranking."""
    if show_test and not selection_path:
        raise ValueError("--show-test requires --selection from a training freeze")
    if freeze_path and (show_test or selection_path):
        raise ValueError("--freeze is only available for a new train selection")
    if selection_path and not show_test:
        raise ValueError("--selection requires --show-test")
    if internal and not show_test:
        raise ValueError("--internal uses test-calibrated factors; freeze the design and pass --show-test")
    if against_shelf and (show_test or whole_book):
        raise ValueError("--against-shelf only applies to candidate screening on train PnL")
    if shelf_path is not None and not against_shelf:
        raise ValueError("--buffer requires --against-shelf")
    if max_shelf_correlation is not None:
        if not against_shelf:
            raise ValueError("--max-shelf-correlation requires --against-shelf")
        if (type(max_shelf_correlation) not in (int, float) or not math.isfinite(max_shelf_correlation)
                or not 0 < max_shelf_correlation <= 1):
            raise ValueError("--max-shelf-correlation must be in (0, 1]")
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
    if not dates and not show_test and not against_shelf:
        raise ValueError("no cached book PnL; the book cannot be scored")
    book_state = describe_book(latest, series=series, registry_path=registry_path, pnl_dir=directory)
    targets = frozen_ids if manifest else (sorted(series) if whole_book else list(dict.fromkeys(ids or [])))
    book = book_state["local"]
    shelf, shelved = None, set()
    frozen_shelf = manifest["inputs"].get("shelf") if manifest else None
    if frozen_shelf is not None:
        # Check the scope before scoring or recording a test view. A changed
        # membership invalidates the training evidence used by Campaign.
        shelf, _ = _shelf_scope(frozen_shelf["path"], targets)
        if (shelf["member_digest"] != frozen_shelf["member_digest"]
                or manifest["inputs"]["book_state"]["local"]["member_digest"] != book["member_digest"]):
            raise ValueError("book or shelf membership changed since training freeze")
        for row in manifest["rows"]:
            current = latest.get(row["alpha_id"]) or {}
            if (row.get("source_expression") != current.get("expression")
                    or row.get("source_settings") != (current.get("settings") or {})):
                raise ValueError("candidate expression or settings changed since training freeze")
    elif against_shelf:
        if shelf_path is None:
            from . import buffer_cmd
            shelf_path = buffer_cmd.buffer_path()
        shelf, shelved = _shelf_scope(shelf_path, targets)
        if freeze_path and os.path.realpath(os.path.expanduser(freeze_path)) == os.path.realpath(shelf["path"]):
            raise ValueError("--freeze and --buffer must use distinct paths")
    window = {"mode": "full" if show_test else "train", "end": None, "source": None}
    missing_boundaries = []
    if not show_test:
        relevant = set(book["member_ids"]) | set(targets) if against_shelf else set(series) | set(targets)
        if against_shelf:
            relevant.update(shelf["member_ids"])
        bounds, missing_boundaries = test_boundaries(latest, relevant)
        boundaries = bounds.values()
        cutoff = min(boundaries) if boundaries else "2023-01-01"
        source = "recorded_test_start" if boundaries else "default_train_end"
        if train_end is not None:
            cutoff = min(cutoff, train_end) if boundaries else train_end
            source = "explicit_and_recorded" if boundaries else "explicit"
        window.update({"end": cutoff, "source": source, "end_exclusive": True})
        dates = [day for day in dates if day < cutoff]
        if not dates and not against_shelf:
            raise ValueError("no cached book PnL before train end %s" % cutoff)
    window.update(_observed_window(dates))
    # The ordinary train view can report on its legacy/default cutoff. The
    # scoped gate requires a known boundary for every compared Alpha.
    scoring_days = [] if against_shelf and missing_boundaries else dates
    book_daily = {day: sum(values[day] for values in series.values()) for day in scoring_days}
    book_sharpe = legs.sharpe_of(book_daily.values())
    if scoring_days and book_sharpe is None and not show_test and not against_shelf:
        raise ValueError("the book's own Sharpe is not measurable on the cached days")
    unit_book = (stats.unit_correlation_rows([[values[day] for day in scoring_days]
                                             for values in series.values()])
                 if len(targets) > 1 and len(scoring_days) >= MIN_OVERLAP_DAYS else None)
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
        if max_shelf_correlation is not None:
            decision_criteria["max_shelf_correlation"] = max_shelf_correlation
    costs.evidence({}, "train", decision_criteria["cost_bps"])
    settings = {**decision_criteria, "pnl_dir": directory, "internal": bool(internal),
                "show_test": bool(show_test), "cohort": _cohort_retentions(latest) if show_test else []}
    peer_series = {peer: legs.load_pnl(peer, directory) for peer in shelf["member_ids"]} if against_shelf else {}
    peer_coverage = {peer: ml.pnl_cache_status(directory, peer, values) for peer, values in peer_series.items()}
    rows = []
    candidate_pnls = {}
    for alpha_id in targets:
        entry = latest.get(alpha_id)
        if entry is not None and roles and alpha_id in roles:
            entry = {**entry, "role": alpha_registry.evidence_role(entry, {"role": roles[alpha_id]})}
        if entry is None:
            if not manifest:
                row = {"alpha_id": alpha_id, "verdict": "unknown", "missing": True,
                       "why": "not in the ledger", "costs": costs.evidence({}, "train", settings["cost_bps"])}
                if max_shelf_correlation is not None:
                    row["training_gate"] = _training_gate(row, settings["min_delta"], max_shelf_correlation)
                rows.append(row)
            continue
        candidate = series[alpha_id] if alpha_id in series else legs.load_pnl(alpha_id, directory)
        candidate_pnls[alpha_id] = candidate
        row = _score(alpha_id, entry, series, scoring_days, settings, candidate=candidate,
                     book_daily=book_daily, unit_book=unit_book)
        row["evidence"] = _candidate_evidence(
            row, candidate, scoring_days, window, book_state, missing_boundaries, directory)
        if against_shelf:
            row["shelf_correlation"] = _shelf_correlations(
                alpha_id, candidate, scoring_days, peer_series, peer_coverage,
                missing_boundaries, row["evidence"]["candidate_pnl"]["complete"])
            row["shelf_correlation"]["already_shelved"] = alpha_id in shelved
            if max_shelf_correlation is not None:
                row["training_gate"] = _training_gate(row, settings["min_delta"], max_shelf_correlation)
        rows.append(row)
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
    if not whole_book and not show_test and len(rows) >= 2:
        planned = _candidate_submission_plan(rows, candidate_pnls, scoring_days, max_shelf_correlation)
        if planned is not None:
            result["submission_plan"] = planned
    if freeze_path and (max_shelf_correlation is None or (
            rows and all(row["training_gate"]["state"] == "pass" for row in rows))):
        inputs = {"registry_path": os.path.abspath(os.path.expanduser(os.fspath(
                      registry_path or alpha_registry.default_registry_path()))),
                  "pnl_dir": directory, "whole_book": bool(whole_book),
                  "files": source_files or [], "book_state": book_state}
        if against_shelf:
            inputs["shelf"] = shelf
        manifest = selection.freeze(freeze_path, source="triage", rows=rows,
                                    criteria=decision_criteria, window=window,
                                    inputs=inputs)
    if manifest:
        result["selection_id"] = manifest["selection_id"]
        result["selection_path"] = os.path.abspath(os.fspath(selection_path or freeze_path))
        result["frozen_window"] = manifest["window"]
        result["frozen_book_state"] = manifest["inputs"].get("book_state")
        result["book_changed"] = book_state != result["frozen_book_state"]
        frozen_digest = ((result["frozen_book_state"] or {}).get("local") or {}).get("member_digest")
        if show_test and frozen_digest and frozen_digest != book["member_digest"]:
            for row in rows:
                if "evidence" in row:
                    row["evidence"]["state"] = "stale"
                    row["evidence"]["reasons"].append("book_membership_changed_since_freeze")
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
        raise CliError("INVALID_INPUT", str(exc),
                       "run `python research_cli.py help triage` for valid targets, --freeze, and --selection usage") from exc


def _cmd_triage(args):
    ids = []
    for chunk in getattr(args, "ids", None) or []:
        for part in str(chunk).replace(",", " ").split():
            if part:
                ids.append(part)
    batch_path = getattr(args, "batch", None)
    for item in getattr(args, "targets", None) or []:
        if not isinstance(item, str) or not item.strip():
            continue
        token = item.strip()
        if token.lower().endswith(".json") or os.path.isfile(os.path.expanduser(token)):
            if batch_path and os.path.abspath(os.path.expanduser(batch_path)) != os.path.abspath(os.path.expanduser(token)):
                raise ValueError("at most one batch file may be triaged at a time")
            batch_path = token
        else:
            for part in token.replace(",", " ").split():
                if part:
                    ids.append(part)
    roles = {}
    if batch_path:
        path = os.path.expanduser(batch_path)
        try:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, ValueError) as exc:
            raise ValueError("cannot read batch file %s: %s" % (path, exc))
        while isinstance(document, dict) and isinstance(document.get("data"), dict):
            document = document["data"]
        if not isinstance(document, dict):
            raise ValueError("triage batch must be an object")
        evidence = []
        batch_ids = []
        arms = document.get("arms")
        for key in ("rows", "records", "results"):
            source_rows = document.get(key)
            if source_rows is None:
                continue
            if not isinstance(source_rows, list) or not all(isinstance(row, dict) for row in source_rows):
                raise ValueError("triage batch %s must be an array of objects" % key)
            if not batch_ids:
                batch_ids = [row["alpha_id"] for row in source_rows if row.get("alpha_id")]
            for index, row in enumerate(source_rows):
                arm = arms[index] if isinstance(arms, list) and len(arms) == len(source_rows) else {}
                role = alpha_registry.evidence_role(row, arm)
                evidence.append({**row, "role": role})
                if row.get("alpha_id"):
                    roles[row["alpha_id"]] = alpha_registry.evidence_role(
                        {"role": roles.get(row["alpha_id"])}, {"role": role})
        if getattr(args, "freeze", None) or getattr(args, "selection", None):
            alpha_registry.require_candidate_eligible(evidence,
                                                     registry_path=getattr(args, "registry", None))
        ids.extend(batch_ids)
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
                    cost_bps=getattr(args, "cost_bps", None), roles=roles,
                    against_shelf=getattr(args, "against_shelf", False),
                    shelf_path=getattr(args, "buffer", None),
                    max_shelf_correlation=getattr(args, "max_shelf_correlation", None),
                    source_files=[os.path.abspath(path)] if batch_path else None)
    return result
