"""Alpha inspection, submission, explanations and portfolio correlation commands."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import hashlib
import math
from pathlib import Path

import requests

import alpha_registry
import cache_store
import correlation
import machine_lib as ml
import research_paths

from .runtime import CliError, METRIC_KEYS, note, with_session


def cmd_alpha_get(args, _sink):
    """Fetch one alpha's metrics, checks, metadata and out-of-sample split from BRAIN."""
    payload = with_session(args, lambda s: ml.fetch_alpha_metrics(s, args.alpha_id))
    in_sample = payload.get("is") or {}
    code, description, operator_count = ml.regular_expression(payload)
    document = {
        "alpha_id": payload.get("id", args.alpha_id),
        # Name, category and tags are what a submission carries and what makes an
        # alpha attributable later. They live at the top of the payload, not under
        # `is`, so they were invisible here - the submit path even refuses an alpha
        # without them, and the only way to see why was to read the raw payload.
        "name": payload.get("name"),
        "category": payload.get("category"),
        "tags": payload.get("tags") or [],
        "expression": code,
        "expression_description": description,
        "operator_count": operator_count,
        "settings": payload.get("settings"),
        "status": payload.get("status"),
        "date_created": payload.get("dateCreated"),
        "grade": payload.get("grade"),
        "metrics": {k: in_sample.get(k) for k in METRIC_KEYS if k in in_sample},
        "checks": ml._extract_checks(payload),
    }
    split = ml.extract_split(payload)
    if split:
        document["out_of_sample"] = split

    return document


def cmd_alpha_near(args, _sink):
    """Find the book alphas closest in *structure* to an expression, before simulating.

    Duplicate detection came in two flavours and both arrive too late: PnL correlation
    needs the simulation, and eyeballing expressions is not reliable at 55 alphas. On
    2026-11-08 several composites were built only to discover they were structural
    variants of what the book already held.

    Structure is compared with the AST skeleton (fields blanked) plus the field set:
    an identical skeleton means "same construction, different grid point", which is the
    definition of a duplicate the campaign uses. Overlapping fields with a different
    skeleton is a much weaker signal and is reported separately.
    """
    import fastexpr_ast

    if args.alpha_id and args.expression:
        raise CliError("USAGE", "pass either --alpha-id or --expression, not both")
    registry = args.registry or alpha_registry.default_registry_path()
    entries = alpha_registry.load_entries(registry)
    latest = {}
    for entry in entries:
        alpha_id = entry.get("alpha_id")
        if alpha_id:
            latest[alpha_id] = entry

    if args.alpha_id:
        entry = latest.get(args.alpha_id)
        if entry is None:
            raise CliError("INVALID_INPUT", "%s is not in the ledger" % args.alpha_id)
        expression = entry.get("expression") or ""
    else:
        expression = args.expression or ""
    expression = " ".join(expression.split())
    if not expression:
        raise CliError("INVALID_INPUT", "empty expression")
    try:
        program = fastexpr_ast.parse(expression)
    except Exception as exc:  # noqa: BLE001 - unparseable input is a usage error
        raise CliError("INVALID_INPUT", "cannot parse expression: %s" % exc,
                       "check the syntax, or pass --alpha-id for a stored alpha")

    skeleton = fastexpr_ast.skeleton(program)
    fields = set(fastexpr_ast.field_names(program))
    operators = set(fastexpr_ast.operator_names(program))

    rows = []
    for alpha_id, entry in latest.items():
        if args.alpha_id and alpha_id == args.alpha_id:
            continue
        if args.status:
            wanted = {part.strip().upper() for part in args.status.split(",")}
            if str(entry.get("status") or "").upper() not in wanted:
                continue
        other = " ".join((entry.get("expression") or "").split())
        if not other:
            continue
        try:
            other_program = fastexpr_ast.parse(other)
        except Exception:  # noqa: BLE001 - a damaged row must not break the search
            continue
        other_skeleton = fastexpr_ast.skeleton(other_program)
        other_fields = set(fastexpr_ast.field_names(other_program))
        other_ops = set(fastexpr_ast.operator_names(other_program))
        shared = fields & other_fields
        union = fields | other_fields
        rows.append({
            "alpha_id": alpha_id,
            "family": entry.get("family"),
            "status": entry.get("status"),
            "fitness": (entry.get("metrics") or {}).get("fitness"),
            "same_skeleton": other_skeleton == skeleton,
            "field_jaccard": round(len(shared) / len(union), 3) if union else 0.0,
            "operator_jaccard": (round(len(operators & other_ops) / len(operators | other_ops), 3)
                                 if (operators | other_ops) else 0.0),
            "shared_fields": sorted(shared)[:8],
            "skeleton": other_skeleton[:160],
        })
    rows.sort(key=lambda row: (not row["same_skeleton"], -row["field_jaccard"]))
    limited = rows[: args.limit]
    same = [row["alpha_id"] for row in rows if row["same_skeleton"]]
    return {"expression": expression[:200], "skeleton": skeleton[:300],
            "fields": sorted(fields), "operators": sorted(operators),
            "count": len(rows), "same_skeleton_count": len(same),
            "same_skeleton": same[:10], "near": limited}


def _check_error(exc):
    if isinstance(exc, requests.RequestException):
        return {"code": "SERVICE_ERROR", "message": "check request failed (%s)" % type(exc).__name__,
                "hint": "Retry this alpha when the platform is reachable."}
    return {"code": exc.code, "message": str(exc), "hint": getattr(exc, "hint", None)}


def _check_result(alpha_id, payload=None, error=None):
    payload = payload or {}
    checks = payload.get("checks") or {}
    verdicts = [check.get("result") for check in checks.values()]
    resolved = bool(verdicts) and all(value in {"PASS", "FAIL", "WARNING"}
                                      for value in verdicts)
    complete = error is None and bool(payload.get("complete")) and resolved
    status = "error" if error else "complete" if complete else "pending"
    outcome = ("error" if error else "failed" if "FAIL" in verdicts else
               "pending" if not complete else "warning" if "WARNING" in verdicts else "passed")
    return {"alpha_id": alpha_id, "status": status, "outcome": outcome,
            "complete": complete, "payload_received": bool(payload.get("payload_received", checks)),
            "checks_resolved": resolved, "checks": checks,
            "self_correlation": payload.get("self_correlation"), "error": error}


def _isolated_check_session(source):
    """Copy authenticated state without sharing mutable cookies or connection pools."""
    session = requests.Session()
    for name in ("auth", "headers", "cookies", "params", "proxies", "hooks",
                 "verify", "cert", "trust_env", "max_redirects", "stream"):
        setattr(session, name, deepcopy(getattr(source, name)))
    return session


def cmd_alpha_check(args, _sink):
    """Compute live checks for ordered unique IDs, preserving every partial outcome."""
    if args.max_wait is not None and (not math.isfinite(args.max_wait) or args.max_wait < 0):
        raise CliError("USAGE", "--max-wait must be finite and non-negative",
                       "pass a number of seconds, or omit the flag for the default")
    if args.workers not in (1, 2, 3):
        raise CliError("USAGE", "--workers must be 1, 2 or 3")
    alpha_ids = list(dict.fromkeys(args.alpha_ids))
    results, auth_failed = {}, set()
    attempts = 0

    def work(session):
        nonlocal attempts
        attempts += 1
        remaining = [alpha_id for alpha_id in alpha_ids
                     if alpha_id not in results or alpha_id in auth_failed]
        auth_failed.clear()

        def fetch(alpha_id):
            try:
                if args.workers == 1:
                    payload = ml.fetch_alpha_checks(session, alpha_id, max_wait_seconds=args.max_wait)
                else:
                    with _isolated_check_session(session) as isolated:
                        payload = ml.fetch_alpha_checks(isolated, alpha_id, max_wait_seconds=args.max_wait)
                return _check_result(alpha_id, payload), None
            except (ml.BrainError, requests.RequestException) as exc:
                return _check_result(alpha_id, error=_check_error(exc)), exc

        def collect(outcomes):
            first_auth_error = None
            for alpha_id, (result, exc) in zip(remaining, outcomes):
                results[alpha_id] = result
                if isinstance(exc, ml.AuthError):
                    auth_failed.add(alpha_id)
                    first_auth_error = first_auth_error or exc
            return first_auth_error

        if args.workers == 1 or not remaining:
            auth_error = collect(map(fetch, remaining))
        else:
            with ThreadPoolExecutor(max_workers=min(args.workers, len(remaining))) as executor:
                auth_error = collect(executor.map(fetch, remaining))
        if auth_error is not None and attempts == 1:
            # with_session owns the single refresh. Its second call only retries
            # rejected authentication work; successful and pending peers stay put.
            raise auth_error

    try:
        with_session(args, work)
    except (ml.BrainError, requests.RequestException) as exc:
        if not results:
            raise
        # A failed refresh must not erase the first pass's successful evidence.
        for alpha_id in auth_failed:
            results[alpha_id] = _check_result(alpha_id, error=_check_error(exc))
    ordered = [results[alpha_id] for alpha_id in alpha_ids]
    summary = {status: sum(row["status"] == status for row in ordered)
               for status in ("complete", "pending", "error")}
    summary.update({outcome: sum(row["outcome"] == outcome for row in ordered)
                    for outcome in ("passed", "failed", "warning")})
    research_paths.log_view("alpha check", "validation", alpha_ids)
    return {"count": len(ordered), "complete": all(row["complete"] for row in ordered),
            "results": ordered, "summary": summary}


def cmd_alpha_submit(args, _sink):
    """Precheck, or submit and independently reconcile each local side effect."""
    if getattr(args, "max_wait", None) is not None and args.max_wait < 0:
        raise CliError("USAGE", "--max-wait must be non-negative")
    return with_session(args, lambda session: submit_workflow(session, args))


def _selection_id(args, required=False):
    path = getattr(args, "selection", None)
    if not path:
        if required:
            raise CliError("INVALID_INPUT", "a new submission requires --selection",
                           "freeze a training selection containing this alpha first")
        return None
    from . import selection
    try:
        manifest = selection.load(path)
    except ValueError as exc:
        raise CliError("INVALID_INPUT", str(exc), "use a valid frozen training selection") from exc
    if not any(row.get("alpha_id") == args.alpha_id for row in manifest.get("rows", [])):
        raise CliError("INVALID_INPUT", "alpha is absent from the frozen selection",
                       "submit a candidate included in the training selection")
    return manifest["selection_id"]


def _submission_path(alpha_id):
    identity = hashlib.sha256(alpha_id.encode("utf-8")).hexdigest()
    return Path(research_paths.resolve("submissions", identity + ".json"))


def _confirmed(current):
    return (str(current.get("status") or "").upper() in {"ACTIVE", "SUBMITTED"}
            or bool(current.get("dateSubmitted")))


def _platform_entry(current, alpha_id, existing=None, selection_id=None, source="submit"):
    """Refresh platform facts while preserving research labels and provenance."""
    existing = existing or {}
    code, _description, _count = ml.regular_expression(current)
    checks = ml._extract_checks(current)
    extra = dict(existing.get("extra") or {})
    extra.update({"dateSubmitted": current.get("dateSubmitted") or extra.get("dateSubmitted"),
                  "platform_status": current.get("status"),
                  "platform_hidden": current.get("hidden"),
                  "platform_type": current.get("type")})
    if selection_id:
        extra["selection_id"] = selection_id
    facts = {"alpha_id": alpha_id, "expression": code,
             "settings": current.get("settings") or {},
             "metrics": current.get("is") or {}, "checks": checks,
             "status": current.get("status"),
             "out_of_sample": ml.extract_split(current) or {}}
    entry = alpha_registry.build_entry(
        facts, verdict=alpha_registry.auto_verdict(facts), source=source,
        idea=existing.get("idea"), mechanism=existing.get("mechanism"),
        family=existing.get("family"), batch=existing.get("batch"),
        note=existing.get("note"), extra=extra)
    return {**existing, **entry}


def _reconcile_submission(session, args, current, selection_id):
    from . import book_state, buffer_cmd
    registry = str(getattr(args, "registry", None) or alpha_registry.default_registry_path())
    report = {}
    try:
        entries = alpha_registry.load_entries(registry)
        existing = next((row for row in entries if row.get("alpha_id") == args.alpha_id), None)
        entry = _platform_entry(current, args.alpha_id, existing, selection_id)
        stored, written = alpha_registry.record_entry(entry, registry)
        report["ledger"] = {"complete": True, "written": written,
                            "key": stored.get("key"), "status": stored.get("status"),
                            "date_submitted": current.get("dateSubmitted")}
    except Exception as exc:
        report["ledger"] = {"complete": False, "error": str(exc)}
    try:
        report["shelf"] = buffer_cmd.consume_submission(args, selection_id)
    except Exception as exc:
        report["shelf"] = {"complete": False, "error": str(exc)}
    try:
        book_state.invalidate_snapshot(args.alpha_id, registry)
        report["book_snapshot"] = {"complete": True, "freshness": "invalidated"}
    except Exception as exc:
        report["book_snapshot"] = {"complete": False, "error": str(exc)}
    pnl_report = {}
    try:
        series = ml.fetch_daily_pnl(session, args.alpha_id,
                                    cache_dir=Path(research_paths.resolve("pnl")),
                                    max_age_hours=0, report=pnl_report)
        report["pnl"] = {**pnl_report, "points": len(series or {}),
                         "complete": pnl_report.get("complete") is True}
    except Exception as exc:
        report["pnl"] = {**pnl_report, "complete": False, "error": str(exc)}
    return report


def submit_workflow(session, args, shelf_entry=None):
    """One state machine for both commands; uncertain attempts are read-only on retry."""
    from datetime import datetime, timezone
    from . import costs
    max_wait = getattr(args, "max_wait", None)
    if max_wait is not None and max_wait < 0:
        raise CliError("USAGE", "--max-wait must be non-negative")
    max_wait = 180 if max_wait is None else max_wait
    yes = bool(getattr(args, "yes", False))
    path = _submission_path(args.alpha_id)

    def run():
        prior, error = cache_store.load_json(path)
        if error and error != "missing":
            raise CliError("INVALID_INPUT", "submission journal is unreadable: " + error,
                           "recover the journal before attempting a new submission")
        prior = prior or {}
        if not isinstance(prior, dict):
            raise CliError("INVALID_INPUT", "submission journal must be an object")
        readback_error = None
        try:
            current = ml.fetch_alpha_metrics(session, args.alpha_id)
        except Exception as exc:
            readback_error = str(exc)
            if not prior.get("state"):
                raise
            current = prior.get("platform") or {}
            if prior.get("state") != "CONFIRMED":
                return {"alpha_id": args.alpha_id, "submission": prior["state"],
                        "submitted": False, "post_attempted": False,
                        "reason": "read-back failed; no repeated POST", "error": str(exc)}
        result = {"alpha_id": args.alpha_id, "status": current.get("status"),
                  "post_attempted": False,
                  "costs": costs.evidence((ml.extract_split(current) or {}).get("train") or {},
                                          "train", getattr(args, "cost_bps", None))}
        from .buffer_cmd import _book_evidence
        try:
            result["book"] = _book_evidence(args)
        except Exception as exc:
            result["book"] = {"available": False, "error": str(exc)}
        result["confirmation_evidence"] = "durable_journal" if readback_error else "platform_readback"
        if readback_error:
            result["readback_error"] = readback_error
        confirmed = _confirmed(current) or prior.get("state") == "CONFIRMED"
        if prior.get("state") == "CONFIRMED" and not _confirmed(current):
            result["confirmation_evidence"] = "durable_journal"
            current = {**current, "dateSubmitted": (prior.get("platform") or {}).get("dateSubmitted")}
        selection_id = prior.get("selection_id")
        if not confirmed:
            selection_id = _selection_id(args, required=False) or selection_id
        elif getattr(args, "selection", None):
            selection_id = _selection_id(args)
        if not yes:
            if confirmed:
                result.update(submitted=True, submission="CONFIRMED",
                              date_submitted=current.get("dateSubmitted"))
            elif prior.get("state") in {"INTENT", "ACCEPTED", "UNCERTAIN", "UNKNOWN"}:
                result.update(submitted=False, submission=prior["state"],
                              reason="previous attempt requires platform read-back; no repeated POST")
            else:
                result.update(ml.submit_alpha(session, args.alpha_id, confirm=False,
                                               max_wait_seconds=max_wait))
            result.update(dry_run=True, reconciled=False)
            return result
        if not confirmed and prior.get("state") in {"INTENT", "ACCEPTED", "UNCERTAIN", "UNKNOWN"}:
            result.update(submitted=False, submission=prior["state"],
                          reason="previous attempt is unresolved; no repeated POST",
                          repair_actions=["read platform status again with this command"])
            return result
        if not confirmed:
            selection_id = _selection_id(args, required=True)
            missing = [field for field in ("name", "category") if not current.get(field)]
            if missing:
                raise CliError("INVALID_INPUT", "%s has no %s" % (args.alpha_id, ", ".join(missing)),
                               "set the missing metadata with alpha describe")
            if shelf_entry is not None:
                if shelf_entry.get("state") in {"dropped", "consumed"}:
                    raise CliError("INVALID_INPUT", "candidate is no longer on the shelf",
                                   "shelve it again before a new buffer submission")
                from .buffer_cmd import GATE
                checks = ml.fetch_alpha_checks(session, args.alpha_id, max_wait_seconds=max_wait)
                fresh = (checks.get("self_correlation") or {}).get("value")
                result["self_correlation_now"] = fresh
                if not checks.get("complete") or not checks.get("checks") or fresh is None:
                    raise CliError("INVALID_INPUT", "fresh submission checks are incomplete",
                                   "rerun the read-only precheck after the platform finishes")
                if fresh is not None and fresh >= GATE and not getattr(args, "force", False):
                    raise CliError("INVALID_INPUT", "fresh self-correlation exceeds the shelf gate",
                                   "pass --force to attempt the platform gate anyway")
            journal = {"alpha_id": args.alpha_id, "state": "INTENT",
                       "selection_id": selection_id,
                       "started_at": datetime.now(timezone.utc).isoformat()}
            cache_store.write_json_atomic(path, journal)
            result["post_attempted"] = True
            try:
                response = ml.submit_alpha(session, args.alpha_id, confirm=True,
                                            max_wait_seconds=max_wait)
                result.update(response)
            except ml.SubmissionDisabled:
                journal["state"] = "NOT_SENT"
                cache_store.write_json_atomic(path, journal)
                raise
            except Exception as exc:
                result.update(submission="UNKNOWN", submitted=False, error=str(exc))
            try:
                refreshed = ml.fetch_alpha_metrics(session, args.alpha_id)
                if _confirmed(refreshed):
                    current, confirmed = refreshed, True
            except Exception as exc:
                result["readback_error"] = str(exc)
            if not confirmed and result.get("submission") == "CONFIRMED" and result.get("submitted"):
                confirmed = True
                result["confirmation_evidence"] = "submit_result"
                current = {**current, "status": result.get("status"),
                           "dateSubmitted": result.get("date_submitted")}
            if not confirmed:
                state = result.get("submission") or "UNKNOWN"
                if state not in {"REJECTED", "NOT_SENT", "ACCEPTED", "UNCERTAIN", "UNKNOWN"}:
                    state = "UNKNOWN"
                result.update(submission=state, submitted=False)
                journal.update(state=state, result=result)
                cache_store.write_json_atomic(path, journal)
                return result
        result.update(submitted=True, submission="CONFIRMED", status=current.get("status"),
                      date_submitted=current.get("dateSubmitted"), selection_id=selection_id)
        journal = {"alpha_id": args.alpha_id, "state": "CONFIRMED", "platform": current,
                   "selection_id": selection_id, "result": dict(result)}
        # Confirmation is durable before any local repair. Failure here must still
        # report remote success, and the old intent prevents a repeated POST.
        try:
            cache_store.write_json_atomic(path, journal)
        except Exception as exc:
            result["journal_error"] = str(exc)
        result["reconciliation"] = _reconcile_submission(session, args, current, selection_id)
        try:
            result["book"] = _book_evidence(args)
        except Exception as exc:
            result["book"] = {"available": False, "error": str(exc)}
        result["repair_actions"] = ["repair " + name for name, value in result["reconciliation"].items()
                                    if not value.get("complete")]
        if result.get("journal_error"):
            result["repair_actions"].append("repair submission journal")
        result["reconciled"] = not result["repair_actions"]
        journal["result"] = result
        try:
            cache_store.write_json_atomic(path, journal)
        except Exception as exc:
            result["journal_error"] = str(exc)
            if "repair submission journal" not in result["repair_actions"]:
                result["repair_actions"].append("repair submission journal")
            result["reconciled"] = False
        return result

    if not yes:
        return run()
    with cache_store.file_lock(str(path) + ".lock", timeout=max(30, max_wait + 30)):
        return run()


def cmd_alpha_list(args, _sink):
    """List platform alphas matching filters such as status, region, universe or minimum Sharpe."""
    alphas = with_session(args, lambda s: ml.list_alphas(
        s, limit=args.limit, offset=args.offset, order=args.order,
        status=args.status, region=args.region, universe=args.universe,
        min_sharpe=args.min_sharpe, min_fitness=args.min_fitness, search=args.search,
        hidden=args.all, alpha_type_not="" if args.all else "SUPER",
    ))
    compact = []
    for alpha in alphas:
        in_sample = alpha.get("is") or {}
        settings = alpha.get("settings") or {}
        code, description, operator_count = ml.regular_expression(alpha)
        compact.append({
            "alpha_id": alpha.get("id"),
            "expression": code,
            "expression_description": description,
            "operator_count": operator_count,
            "sharpe": in_sample.get("sharpe"),
            "fitness": in_sample.get("fitness"),
            "turnover": in_sample.get("turnover"),
            "region": settings.get("region"),
            "universe": settings.get("universe"),
            "neutralization": settings.get("neutralization"),
            "status": alpha.get("status"),
            "grade": alpha.get("grade"),
            "date_created": alpha.get("dateCreated"),
        })
    document = {"count": len(compact), "offset": args.offset, "alphas": compact}

    return document


def cmd_alpha_explain(args, _sink):
    """Explain one alpha: what each leg computes, and what the evidence says.

    Three layers, kept apart on purpose, because mixing them is how a description
    starts to sound like a justification:

    * **construction** -- what the expression does, leg by leg, in words. Read off the
      tree, so it cannot drift from the formula.
    * **evidence** -- the metrics, where the Fitness comes from arithmetically, the
      measured ablations if any were run, and the local correlation with the book.
    * **risk** -- whether the family is on the closed list, and what the leg weights do
      *not* tell you (a weight is a multiplier, not a contribution).
    """
    import alpha_unit as au

    entries = alpha_registry.load_entries(getattr(args, "registry", None))
    target = alpha_registry.resolve(entries, args.alpha_id)
    if target is None:
        raise CliError("NOT_FOUND", "no ledger entry for %s" % args.alpha_id,
                       "run `alpha list` or `alpha get` to find the id")

    point = au.split(target.get("expression") or "",
                     datasets=au.load_field_datasets())
    if point is None:
        raise CliError("INVALID_INPUT", "%s does not parse" % args.alpha_id,
                       "the ledger row may be damaged; `alpha get` shows the raw text")

    explanation = au.explain(point)
    metrics = target.get("metrics") or {}
    settings = target.get("settings") or {}

    # The arithmetic that decides whether the score is earned or borrowed.
    sharpe, turnover = metrics.get("sharpe"), metrics.get("turnover")
    returns, fitness = metrics.get("returns"), metrics.get("fitness")
    denominator = None
    if isinstance(turnover, (int, float)):
        denominator = max(turnover, 0.125)
    implied = None
    if isinstance(sharpe, (int, float)) and isinstance(returns, (int, float)) and denominator:
        implied = round(sharpe * (returns / denominator) ** 0.5, 3)

    ablations = []
    for entry in entries:
        idea = entry.get("idea") or ""
        if idea.endswith(" of %s" % args.alpha_id) or idea.endswith("of %s" % args.alpha_id):
            m = entry.get("metrics") or {}
            ablations.append({"label": idea.split(":")[-1].split(" of ")[0].strip(),
                              "alpha_id": entry.get("alpha_id"),
                              "sharpe": m.get("sharpe"), "fitness": m.get("fitness")})

    gate = None if args.no_corr else _local_book_gate(args.alpha_id, entries)

    document = {
        "alpha_id": args.alpha_id,
        "status": target.get("status"),
        "family": target.get("family"),
        "settings": {k: settings.get(k) for k in
                     ("universe", "neutralization", "decay", "truncation", "delay")},
        "construction": explanation,
        "metrics": {k: metrics.get(k) for k in
                    ("sharpe", "fitness", "turnover", "returns", "drawdown")},
        "fitness_arithmetic": {
            "formula": "sharpe * sqrt(returns / max(turnover, 0.125))",
            "denominator_used": denominator,
            "below_turnover_floor": bool(isinstance(turnover, (int, float))
                                         and turnover < 0.125),
            "implied_fitness": implied,
            # The gap is the integrity check this block exists for: a row whose
            # stored Fitness cannot be rebuilt from its own Sharpe/returns/turnover
            # is stale or damaged, and every ranking built on it inherits that.
            "reported_fitness": fitness,
            "gap": (round(fitness - implied, 3)
                    if isinstance(fitness, (int, float)) and implied is not None
                    else None),
        },
        "ablations_measured": ablations,
        "book_correlation": gate,
        "closed_family": _closed_reason(target.get("family")),
        "caveat": "a leg weight is a multiplier, not a contribution; which leg carries "
                  "the alpha is only known if an ablation was run",
    }

    return document


def _local_book_gate(alpha_id, entries):
    """Daily-correlation of one alpha against the book, computed from cached PnL.

    Local correlations screen candidates; the platform makes the submission decision.
    Values near the gate require the platform's own check.
    """
    from pathlib import Path

    import correlation

    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"
    own = pnl_dir / ("%s.json" % alpha_id)
    if not own.exists():
        return {"available": False, "reason": "no cached PnL for this alpha"}
    book = {e.get("alpha_id") for e in entries
            if (e.get("status") or "").upper() == "ACTIVE" and e.get("alpha_id")}
    try:
        with open(own) as handle:
            target = json.load(handle)
    except (OSError, ValueError):
        return {"available": False, "reason": "cached PnL is unreadable"}
    scores, missing = [], 0
    for peer in sorted(book):
        if peer == alpha_id:
            continue
        path = pnl_dir / ("%s.json" % peer)
        if not path.exists():
            missing += 1
            continue
        try:
            with open(path) as handle:
                series = json.load(handle)
        except (OSError, ValueError):
            missing += 1
            continue
        x, y, _dates = correlation.align(target, series)
        value = correlation.pearson(x, y)
        if value is not None:
            scores.append((value, peer))
    if not scores:
        return {"available": False, "reason": "no overlapping book PnL"}
    best = max(scores, key=lambda pair: abs(pair[0]))
    return {"available": True, "max": round(best[0], 4), "peer": best[1],
            "peers": len(scores), "missing_pnl": missing,
            "verdict": ("clear" if abs(best[0]) < 0.67 else
                        "verify on the platform (>= 0.67)"),
            "method": "daily Pearson on cached PnL"}


def _closed_reason(family):
    try:
        import closed_families
        reason = closed_families.reason(family) if family else None
    except Exception:  # noqa: BLE001 - an explain must work without the file
        return None
    return reason


def cmd_alpha_self_corr(args, _sink):
    """Report the platform's self-correlation and the substitute route's bar."""
    payload = with_session(
        args, lambda s: ml.fetch_self_correlation(s, args.alpha_id))
    verdict = ml.self_correlation_verdict(payload, limit=args.limit,
                                          sharpe_margin=args.sharpe_margin)
    current = with_session(args, lambda s: ml.fetch_alpha_metrics(s, args.alpha_id))
    sharpe = (current.get("is") or {}).get("sharpe")
    needed = verdict.get("sharpe_needed")
    qualifies = None
    if verdict["blocked"] and needed is not None and isinstance(sharpe, (int, float)):
        qualifies = sharpe >= needed
    data = {
        "alpha_id": args.alpha_id,
        "sharpe": sharpe,
        "max_correlation": payload.get("max"),
        "limit": args.limit,
        "sharpe_margin": args.sharpe_margin,
        "sharpe_needed": needed,
        "qualifies_via_sharpe": qualifies,
        "verdict": ("clear" if not verdict["blocked"]
                    else ("sharpe route" if qualifies else "blocked")),
        "counterparts": verdict["counterparts"],
    }

    return data


def cmd_alpha_describe(args, _sink):
    """Write an alpha's metadata and verify it stuck."""
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    if not any((args.name, args.category, tags, args.description)):
        raise CliError("USAGE", "nothing to set",
                       "pass --name, --category, --tags and/or --description")

    def work(session):
        return ml.set_alpha_metadata(session, args.alpha_id, name=args.name,
                                     category=args.category,
                                     tags=tags or None,
                                     description=args.description)

    current = with_session(args, work)
    data = {
        "alpha_id": current.get("id"),
        "name": current.get("name"),
        "category": current.get("category"),
        "tags": current.get("tags"),
        "verified": True,
        "description": (current.get("regular") or {}).get("description"),
        "note": "name, category and tags are top-level; description is nested under "
                "`regular`",
    }

    return data


def _rank_peers(scored):
    """Order peers by absolute correlation, largest first.

    The platform's limit is on ``|r|``: a candidate that mirrors an existing alpha
    has a large negative correlation and is blocked exactly as hard as a positive
    one. Sorting on the signed value put such a peer last and reported the
    runner-up as the counterpart.
    """
    return sorted(scored, key=lambda item: -abs(item.get("correlation") or 0.0))


def _independence_state(flagged, skipped, measured):
    """Three states, because "no measured pair cleared the line" is not independence.

    A pair whose overlap is too short lands in ``skipped``. Treating that as a pass
    is absence of evidence read as evidence of absence, and the unmeasurable pair is
    the one most likely to be a duplicate.
    """
    if flagged:
        return "dependent"
    if skipped or not measured:
        return "unknown"
    return "independent"


def _verdict_with_gaps(verdict, gaps):
    """A gap can only hide a block, never a clear.

    ``clear`` means every peer was measured and none came close. When a peer's
    PnL is missing, too short, or flat, that statement is not supported: the
    pair nobody could measure is the likeliest duplicate, so ``clear`` degrades
    to ``unknown`` instead of surviving on the pairs that did work.

    ``blocked`` and ``borderline`` already mean do not submit, and a missing
    measurement can only add a reason, never remove one, so they stand.

    Returns ``(verdict, note)``; the note names the gaps whenever there are any,
    including when the verdict was already ``unknown`` because nothing at all
    could be measured - that case is a gap too, and the least informative one.
    """
    if not gaps:
        return verdict, None
    counts = {}
    for gap in gaps:
        counts[gap["reason"]] = counts.get(gap["reason"], 0) + 1
    note = ("%d peer(s) could not be measured (%s)"
            % (len(gaps), ", ".join("%s %d" % (reason, count)
                                    for reason, count in sorted(counts.items()))))
    if verdict == "clear":
        return "unknown", (note + "; an unmeasured pair is the one most likely to be a "
                                   "duplicate, so `clear` is not claimed")
    return verdict, note


def cmd_alpha_self_check(args, sink):
    """Reproduce the platform's self-correlation check locally, and better.

    Three measured facts shape this command.

    Daily Pearson is the decision measure; weekly and monthly aggregates remain
    diagnostics because aggregation changes the comparison.

    The platform compares against a periodically refreshed snapshot of the book.
    A local check using only ACTIVE alphas can miss in-flight submissions, so recent
    submissions, including ones not submitted yet, are included by default.

    A published platform value is historical. ``--as-of`` restricts peers to the book
    at that timestamp; peers submitted later can otherwise change the comparison.

    Every peer that yields no number is named with a reason - no PnL, too little overlap,
    or an undefined Pearson because one series is flat. A flat series is the sharp case:
    ``round(None, 4)`` used to be a TypeError, and the tempting fix, dropping the pair,
    would read as a measured zero. ``_verdict_with_gaps`` says why a gap downgrades
    ``clear`` to ``unknown`` and leaves ``blocked`` alone.
    """
    from datetime import datetime, timedelta, timezone

    def parse(moment):
        if not moment:
            return None
        try:
            return datetime.fromisoformat(str(moment).replace("Z", "+00:00"))
        except ValueError:
            return None

    cutoff = parse(args.as_of) if args.as_of else None
    if args.as_of and cutoff is None:
        raise CliError("INVALID_INPUT", "cannot read --as-of %r" % args.as_of,
                       "use an ISO timestamp such as 2026-10-23T00:15:00+08:00")
    horizon = (datetime.now(timezone.utc)
               - timedelta(hours=max(0.0, args.include_unsent_hours)))

    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"

    def collect(session):
        candidate = ml.fetch_daily_pnl(session, args.alpha_id, cache_dir=pnl_dir,
                                       max_age_hours=args.max_age_hours)
        if not candidate:
            raise CliError("NOT_FOUND", "no daily PnL for %s" % args.alpha_id,
                           "the alpha must have finished simulating")
        record = ml.fetch_alpha_metrics(session, args.alpha_id)
        peers, excluded = [], []
        seen = set()
        statuses = ["ACTIVE"]
        if args.include_unsent_hours > 0:
            statuses.append("UNSUBMITTED")
        for status in statuses:
            for entry in ml.list_alphas(session, limit=args.peer_limit, status=status,
                                        alpha_type_not=""):
                peer_id = entry.get("id")
                if not peer_id or peer_id == args.alpha_id:
                    continue
                when = parse(entry.get("dateSubmitted")) or parse(entry.get("dateCreated"))
                if cutoff is not None and when is not None and when > cutoff:
                    excluded.append({"alpha_id": peer_id, "when": entry.get("dateSubmitted")
                                     or entry.get("dateCreated")})
                    continue
                if status == "UNSUBMITTED" and when is not None and when < horizon:
                    continue
                if peer_id in seen:
                    # An alpha submitted minutes ago is ACTIVE and was also created
                    # inside the unsubmitted window, so it arrives twice and would be
                    # counted, correlated and gap-reported twice.
                    continue
                seen.add(peer_id)
                peers.append({"alpha_id": peer_id, "when": when,
                              "status": status})
        series = {}
        for peer in peers:
            data = ml.fetch_daily_pnl(session, peer["alpha_id"], cache_dir=pnl_dir,
                                      max_age_hours=args.max_age_hours)
            if data:
                series[peer["alpha_id"]] = data
        return candidate, record, peers, series, excluded

    candidate, record, peers, series, excluded = with_session(args, collect)

    # Every peer that produced no number is named with a reason. The crash this
    # replaces was `round(None, 4)` on a flat PnL series; the quieter defect was
    # that a peer with no PnL at all was skipped without a word, so the maximum
    # was a maximum over whatever happened to be measurable.
    gaps = []
    usable = []
    for peer in peers:
        if series.get(peer["alpha_id"]):
            usable.append(peer)
        else:
            gaps.append({"alpha_id": peer["alpha_id"], "reason": "no_pnl", "n": 0})

    broadcast = {}
    for frequency in ("d", "w", "m"):
        target = correlation.aggregate(candidate, frequency)
        minimum = args.min_overlap if frequency == "d" else 12
        scored = []
        for peer in usable:
            left, right, _ = correlation.align(
                target, correlation.aggregate(series[peer["alpha_id"]], frequency),
                minimum)
            if len(left) < minimum:
                if frequency == "d":
                    gaps.append({"alpha_id": peer["alpha_id"],
                                 "reason": "short_overlap", "n": len(left)})
                continue
            value = correlation.pearson(left, right)
            if value is None:
                if frequency == "d":
                    gaps.append({"alpha_id": peer["alpha_id"],
                                 "reason": "undefined", "n": len(left)})
                continue
            scored.append({"alpha_id": peer["alpha_id"],
                           "correlation": round(value, 4),
                           "when": peer["when"].isoformat() if peer["when"] else None,
                           "status": peer["status"]})
        scored[:] = _rank_peers(scored)
        broadcast[frequency] = scored

    daily = broadcast["d"]
    best = daily[0] if daily else {"alpha_id": None, "correlation": None}
    # The platform reads a little lower than daily Pearson; the offset is measured, not
    # assumed, and it is what makes `borderline` a useful verdict instead of a shrug.
    verdict, estimate = correlation.classify(best["correlation"], args.threshold,
                                              args.platform_bias)
    verdict, verdict_note = _verdict_with_gaps(verdict, gaps)

    submitted = parse(record.get("dateSubmitted")) or parse(record.get("dateCreated"))
    counterpart_when = parse(best.get("when"))
    newer = bool(submitted and counterpart_when and counterpart_when > submitted)

    return {
        "alpha_id": args.alpha_id,
        "candidate_submitted": record.get("dateSubmitted"),
        "as_of": args.as_of,
        "peers_considered": len(peers),
        "peers_measured": len(daily),
        "peers_unmeasured": gaps,
        "peers_excluded_as_newer": [item["alpha_id"] for item in excluded],
        "threshold": args.threshold,
        "platform_bias": args.platform_bias,
        "max_daily": best["correlation"],
        "estimated_platform": estimate,
        "counterpart": best["alpha_id"],
        "counterpart_submitted": best.get("when"),
        "counterpart_entered_after_candidate": newer,
        "verdict": verdict,
        "verdict_note": verdict_note,
        "frequencies": {name: (items[0]["correlation"] if items else None)
                        for name, items in broadcast.items()},
        "top": daily[:args.top],
        "note": ("the counterpart entered after this alpha, so the platform could not have "
                 "seen it - set --as-of to reproduce the original check"
                 if newer else None),
    }


def cmd_alpha_corr(args, sink):
    """Correlation on daily PnL: the evidence independence claims need.

    Three questions this answers that nothing else in the toolkit could:
    whether a pair is really two bets, whether a candidate matches anything in
    the submitted book, and whether a resemblance survives removing the leg both
    sides share.
    """
    ids = []
    for value in args.alpha_id:
        for part in str(value).replace(",", " ").split():
            if part and part not in ids:
                ids.append(part)
    if len(ids) < 2 and not args.vs_book:
        raise CliError("USAGE", "at least two alpha ids are needed",
                       "pass two or more ids, or one id with --vs-book")

    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"

    def load(session):
        series = {}
        for alpha_id in ids:
            series[alpha_id] = ml.fetch_daily_pnl(
                session, alpha_id, cache_dir=pnl_dir,
                max_age_hours=args.max_age_hours)
        book = {}
        if args.vs_book:
            for entry in ml.list_alphas(session, limit=200, status="ACTIVE",
                                        alpha_type_not=""):
                book_id = entry.get("id")
                if book_id and book_id not in series:
                    book[book_id] = ml.fetch_daily_pnl(
                        session, book_id, cache_dir=pnl_dir,
                        max_age_hours=args.max_age_hours)
        control = None
        if args.control:
            control = ml.fetch_daily_pnl(
                session, args.control, cache_dir=pnl_dir,
                max_age_hours=args.max_age_hours)
        return series, book, control

    series, book, control = with_session(args, load)

    empty = [name for name, data in series.items() if not data]
    if empty:
        raise CliError(
            "NOT_FOUND",
            "no daily PnL for %s" % ", ".join(sorted(empty)),
            "the recordset is only populated for alphas that completed a "
            "simulation; check the id with `alpha get`")

    dates = sorted(set.intersection(*(set(v) for v in series.values())))
    matrix = correlation.matrix(series, min_overlap=args.min_overlap)
    high = [p for p in matrix["pairs"] if abs(p["r"]) >= args.threshold]
    # Absence of evidence is not evidence of absence. A pair whose overlap is too
    # short lands in `skipped`, and an earlier version then reported `independent`
    # because no measured pair cleared the threshold -- exactly backwards: the
    # unmeasurable pair is the one most likely to be a duplicate.
    skipped = matrix.get("skipped") or []
    measured = len(matrix["pairs"])
    total_pairs = measured + len(skipped)
    independence = _independence_state(high, skipped, measured)
    data = {
        "ids": ids,
        "window": {
            "start": dates[0] if dates else None,
            "end": dates[-1] if dates else None,
            "shared_days": len(dates),
        },
        "matrix": matrix,
        "flagged": high,
        "threshold": args.threshold,
        "independence": independence,
        "independent": independence == "independent",
        "coverage": {
            "measured_pairs": measured,
            "skipped_pairs": len(skipped),
            "ratio": round(measured / total_pairs, 4) if total_pairs else None,
            "note": ("unmeasurable pairs make independence unknowable"
                     if skipped else None),
        },
    }

    if book:
        against = {}
        for alpha_id, values in series.items():
            if not values:
                continue
            against[alpha_id] = correlation.max_against(
                values, book, min_overlap=args.min_overlap)
        data["vs_book"] = {
            "book_size": len(book),
            "per_alpha": against,
        }

    if control:
        residuals = {}
        for alpha_id, values in series.items():
            if values:
                residuals[alpha_id] = correlation.residualize(
                    values, control, min_overlap=args.min_overlap)
        residual_matrix = correlation.matrix(residuals,
                                             min_overlap=args.min_overlap)
        data["residual"] = {
            "control": args.control,
            "matrix": residual_matrix,
        }


    return data


def cmd_alpha_book(args, sink):
    """Inspect currently submitted/ACTIVE alphas joined with local ledger mechanisms and families."""
    registry_path = (getattr(args, "registry", None)
                     or alpha_registry.default_registry_path())
    entries = alpha_registry.load_entries(registry_path)

    by_id = {}
    by_hash = {}
    for entry in entries:
        alpha_id = entry.get("alpha_id")
        if alpha_id:
            by_id.setdefault(alpha_id, entry)
        digest = entry.get("expression_hash")
        if digest:
            by_hash.setdefault(digest, entry)

    def load(session):
        return ml.list_alphas(session, limit=getattr(args, "limit", 50),
                              status="ACTIVE", alpha_type_not="")

    alphas = with_session(args, load)

    items = []
    unknown = []
    mechanism_counts = {}
    family_counts = {}
    unclassified = 0
    for alpha in alphas:
        alpha_id = alpha.get("id")
        metrics = alpha.get("is") or {}
        code, _, _ = ml.regular_expression(alpha)
        entry = by_id.get(alpha_id) or by_hash.get(
            alpha_registry.expression_hash(code)) if code else by_id.get(alpha_id)
        mechanism = (entry or {}).get("mechanism")
        family = (entry or {}).get("family")
        if not mechanism:
            unknown.append(alpha_id)
            mechanism = None
        else:
            mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1
        else:
            unclassified += 1
        items.append({
            "alpha_id": alpha_id,
            "status": alpha.get("status"),
            "sharpe": metrics.get("sharpe"),
            "fitness": metrics.get("fitness"),
            "turnover": metrics.get("turnover"),
            "mechanism": mechanism,
            "family": family,
            "idea": (entry or {}).get("idea"),
            "expression": code,
        })

    document = {
        "registry": registry_path,
        "book_size": len(items),
        "mechanism_count": len(mechanism_counts),
        "mechanism_counts": mechanism_counts,
        "family_count": len(family_counts),
        "family_counts": family_counts,
        "unclassified_family": unclassified,
        "unlabelled": unknown,
        "alphas": items,
    }

    candidate_id = getattr(args, "check", None)
    if candidate_id:
        checks = with_session(args, lambda s: ml.fetch_alpha_checks(
            s, candidate_id,
            max_wait_seconds=getattr(args, "max_wait", 180) or 180))
        self_correlation = checks.get("self_correlation") or {}
        document["candidate"] = {
            "alpha_id": candidate_id,
            "self_correlation": self_correlation,
            "clear": (self_correlation.get("value") is not None
                      and self_correlation.get("limit") is not None
                      and self_correlation["value"] <= self_correlation["limit"]),
        }


    return document
