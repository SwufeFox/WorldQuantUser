"""Commands for recording, labelling and inspecting the local research ledger."""

from __future__ import annotations

import json
import math
from pathlib import Path

import alpha_registry
import correlation
import machine_lib as ml
import research_paths

from . import experiments, mechanisms, styles
from .runtime import CliError, with_session


def cmd_ledger_record(args, sink):
    """Fetch metrics for an alpha from the platform and record a new entry in the local ledger."""
    alpha_id = getattr(args, "alpha_id", None)
    if not alpha_id:
        raise CliError(
            "USAGE", "ledger record needs an alpha id",
            "use `alpha list` to find one; a record without an alpha id has no "
            "metrics to persist")

    def work(session):
        return ml.fetch_alpha_metrics(session, alpha_id)

    payload = with_session(args, work)
    in_sample = payload.get("is") or {}
    expression, _, _ = ml.regular_expression(payload)
    entry = alpha_registry.build_entry(
        {
            "alpha_id": payload.get("id") or alpha_id,
            "expression": expression,
            "settings": payload.get("settings") or {},
            "metrics": in_sample,
            "checks": in_sample.get("checks") or {},
            "status": payload.get("status") or "UNSUBMITTED",
            "out_of_sample": ml.extract_split(payload),
        },
        idea=getattr(args, "idea", None),
        mechanism=getattr(args, "mechanism", None),
        verdict=getattr(args, "verdict", "watch"),
        note=getattr(args, "note", None),
        family=getattr(args, "family", None),
        source="record",
    )
    registry_path = (getattr(args, "registry", None)
                     or alpha_registry.default_registry_path())
    stored, written = alpha_registry.record_entry(entry, registry_path)
    return {
        "registry": registry_path,
        "appended": written,
        "entry": stored,
    }


def cmd_ledger_list(args, _sink):
    """List and filter ledger entries by field, mechanism, verdict, family or metrics."""
    registry_path, entries = _ledger(args)
    limit = getattr(args, "limit", 0)
    if limit is not None and limit < 0:
        raise CliError("INVALID_INPUT", "--limit must be zero or positive",
                       "use --limit 0 to show all matching entries")
    matched = alpha_registry.find_entries(
        entries,
        field=getattr(args, "field", None),
        mechanism=getattr(args, "mechanism", None),
        idea=getattr(args, "idea", None),
        expression=getattr(args, "expression", None),
        regex=getattr(args, "regex", False),
        all_fields=getattr(args, "all_fields", None),
        verdicts=getattr(args, "verdict", None),
        min_fitness=getattr(args, "min_fitness", None),
        family=getattr(args, "family", None),
        min_sharpe=getattr(args, "min_sharpe", None),
    )

    def fitness_key(entry):
        value = (entry.get("metrics") or {}).get("fitness")
        return value if isinstance(value, (int, float)) else float("-inf")

    matched.sort(key=fitness_key, reverse=True)
    shown = matched[:limit] if limit else matched
    verdict_counts, mechanisms, families = {}, {}, {}
    for entry in matched:
        verdict = entry.get("verdict") or "unknown"
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        mechanism = entry.get("mechanism") or "unlabelled"
        mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1
        family = entry.get("family") or "unclassified"
        families[family] = families.get(family, 0) + 1
    document = {
        "registry": registry_path,
        "registry_entries": len(entries),
        "matched": len(matched),
        "shown": len(shown),
        "query": {
            "field": getattr(args, "field", None),
            "mechanism": getattr(args, "mechanism", None),
            "idea": getattr(args, "idea", None),
            "expression": getattr(args, "expression", None),
            "verdict": list(getattr(args, "verdict", None) or []),
            "family": getattr(args, "family", None),
            "min_fitness": getattr(args, "min_fitness", None),
            "min_sharpe": getattr(args, "min_sharpe", None),
        },
        "verdict_counts": verdict_counts,
        "mechanism_counts": mechanisms,
        "family_counts": families,
        "summary": alpha_registry.summarize(matched),
        "already_tested": bool(matched),
        "entries": shown,
    }
    if getattr(args, "format", "json") == "markdown":
        document["markdown"] = alpha_registry.render_markdown(shown, title="Research ledger")
    return document


def _ledger(args):
    path = (getattr(args, "registry", None)
            or alpha_registry.default_registry_path())
    return path, alpha_registry.load_entries(path)


def cmd_ledger_label(args, _sink):
    """Rewrite the judgement columns of one entry.

    Appends rather than edits: the ledger is append-only and the newest record for
    a key wins, so a relabel keeps the earlier judgement visible.

    The response contains the stored ``entry`` and whether a new history row was
    ``written``, alongside the changed judgement columns and their earlier values.
    """
    path, entries = _ledger(args)
    target = alpha_registry.resolve(entries, args.alpha_id)
    if target is None:
        raise CliError("NOT_FOUND", "no ledger entry for %s" % args.alpha_id,
                       "run `ledger list` to see what is recorded")
    if all(getattr(args, name, None) is None
           for name in ("family", "verdict", "mechanism", "idea", "note")):
        raise CliError("USAGE", "nothing to change",
                       "pass at least one of --family/--verdict/--mechanism/"
                       "--idea/--note")
    try:
        updated = alpha_registry.relabel(
            target, family=args.family, verdict=args.verdict,
            mechanism=args.mechanism, idea=args.idea, note=args.note)
    except ValueError as exc:
        raise CliError("INVALID_INPUT", str(exc)) from exc
    stored, written = alpha_registry.record_entry(updated, path)
    data = {
        "registry": path,
        "written": written,
        "entry": stored,
        "key": stored.get("key"),
        "alpha_id": stored.get("alpha_id"),
        "family": stored.get("family"),
        "verdict": stored.get("verdict"),
        "mechanism": stored.get("mechanism"),
        "before": {"family": target.get("family"),
                   "verdict": target.get("verdict"),
                   "mechanism": target.get("mechanism")},
    }

    return data


def cmd_ledger_impact(args, _sink):
    """Alphas that share a leg body with the given one."""
    _path, entries = _ledger(args)
    target = alpha_registry.resolve(entries, args.alpha_id)
    if target is None:
        raise CliError("NOT_FOUND", "no ledger entry for %s" % args.alpha_id)
    bodies = alpha_registry.leg_bodies(target)
    rows = alpha_registry.shared_legs(entries, args.alpha_id,
                                      min_shared=args.min_shared)
    data = {
        "alpha_id": target.get("alpha_id"),
        "key": target.get("key"),
        "legs": [{"name": name, "body": body}
                 for name, body in bodies.items()],
        "shared_count": len(rows),
        "rows": rows[: args.limit],
        "note": ("legs are matched on the sub-expression body; a shared letter "
                 "means nothing on its own"),
    }

    return data


def cmd_ledger_lineage(args, _sink):
    """Parent chain for one alpha."""
    _path, entries = _ledger(args)
    chain = alpha_registry.lineage(entries, args.alpha_id)
    if not chain:
        raise CliError("NOT_FOUND", "no ledger entry for %s" % args.alpha_id)
    children = alpha_registry.descendants(entries, args.alpha_id)
    data = {
        "alpha_id": args.alpha_id,
        "depth": len(chain),
        "chain": [
            {"key": item.get("key"), "alpha_id": item.get("alpha_id"),
             "op": item.get("op"), "verdict": item.get("verdict"),
             "family": item.get("family"),
             "fitness": (item.get("metrics") or {}).get("fitness")}
            for item in chain
        ],
        "children": [{"key": item.get("key"), "alpha_id": item.get("alpha_id")}
                     for item in children],
        "note": ("lineage is recorded, never inferred; an empty op means the "
                 "record predates lineage capture"),
    }

    return data


def cmd_ledger_coverage(args, _sink):
    """How far the ledger is from having a countable family column."""
    _path, entries = _ledger(args)
    summary = alpha_registry.unlabeled_summary(entries)
    data = {"summary": summary}
    if args.list_missing:
        missing = [entry for entry in entries if not entry.get("family")]
        missing.sort(key=lambda entry: entry.get("recorded_at") or "", reverse=True)
        data["missing"] = [
            {"key": entry.get("key"), "alpha_id": entry.get("alpha_id"),
             "verdict": entry.get("verdict"),
             "fitness": (entry.get("metrics") or {}).get("fitness"),
             "mechanism": entry.get("mechanism")}
            for entry in missing[: args.limit]
        ]

    return data


def cmd_ledger_sync(args, _sink):
    """Reconcile the ledger with the account: record alphas it has never seen.

    The ledger is append-only and only ever grows when something explicitly records an
    alpha, so the platform and the ledger drift: alphas are created in a sweep, or
    submitted by hand, and the local book is quietly wrong afterwards. This walks the
    account, finds ids the ledger has no row for, and records them.

    It does not re-record alphas that already exist: the ledger is append-only and a
    re-record would add a duplicate row without new information. Use `ledger record`
    for a deliberate refresh of one alpha.
    """
    wanted = [part.strip().upper() for part in str(args.status).split(",") if part.strip()]

    def work(session):
        platform = {}
        for page in range(max(1, args.pages)):
            batch = ml.list_alphas(session, limit=args.limit, offset=page * args.limit,
                                   order=args.order, status=args.platform_status)
            for alpha in batch:
                alpha_id = alpha.get("id")
                if alpha_id:
                    platform[alpha_id] = alpha
            if len(batch) < args.limit:
                break
        return platform

    platform = with_session(args, work)
    _path, entries = _ledger(args)
    latest = {}
    for entry in entries:
        alpha_id = entry.get("alpha_id")
        if alpha_id:
            latest[alpha_id] = entry
    known = set(latest)
    missing = [alpha_id for alpha_id in platform if alpha_id not in known]
    # Status drift is the more dangerous half of a sync: `P0o5QGZK` was submitted on the
    # platform while the ledger still said COMPLETE, so every local book count was one
    # short and the alpha looked like a candidate that had never been spent.
    drifted = [alpha_id for alpha_id in platform
               if alpha_id in latest
               and str((latest[alpha_id] or {}).get("status") or "").upper()
               != str((platform[alpha_id] or {}).get("status") or "").upper()]
    if wanted:
        missing = [alpha_id for alpha_id in missing
                   if str((platform[alpha_id] or {}).get("status") or "").upper() in wanted]
    missing.sort()
    drifted.sort()
    if args.dry_run:
        return {"platform": len(platform), "ledger_ids": len(known),
                "missing": missing, "drifted": drifted,
                "recorded": 0, "dry_run": True}

    def record(session):
        written = []
        targets = missing + drifted
        for alpha_id in targets[: args.limit_record]:
            current = ml.fetch_alpha_metrics(session, alpha_id)
            code, _description, _count = ml.regular_expression(current)
            checks = ml._extract_checks(current)
            entry = alpha_registry.build_entry(
                {"alpha_id": alpha_id, "expression": code,
                 "settings": current.get("settings") or {},
                 "metrics": current.get("is") or {},
                 "checks": checks, "status": current.get("status"),
                 "out_of_sample": {"train": current.get("train") or {},
                                   "test": current.get("test") or {},
                                   "test_period": "P1Y"}},
                verdict=alpha_registry.auto_verdict(
                    {"status": current.get("status"),
                     "metrics": current.get("is") or {}, "checks": checks}),
                note="recorded by `ledger sync`" + (" (status drift)" if alpha_id in drifted else ""),
                source="sync")
            stored, was_written = alpha_registry.record_entry(
                entry, getattr(args, "registry", None))
            written.append({"alpha_id": alpha_id, "written": was_written,
                            "status": stored.get("status")})
        return written

    written = with_session(args, record)
    return {"platform": len(platform), "ledger_ids": len(known),
            "missing": missing, "drifted": drifted,
            "recorded": len(written), "rows": written,
            "skipped_beyond_limit": max(0, len(missing) + len(drifted) - len(written))}


def cmd_ledger_format(args, _sink):
    """Rewrite the ledger's stored expressions in the book's own multi-line style.

    The ledger is append-only and it is the record of what was actually simulated, so this
    refuses to write unless *every* row keeps its canonical identity: the expression text
    changes, `expression_hash` does not. A formatting pass that altered a hash would
    silently re-key the row and change what future dedupe matches against.

    A timestamped backup is written first; there is no undo otherwise.
    """
    import shutil
    from datetime import datetime, timezone
    import expression_lint

    path, entries = _ledger(args)
    registry = Path(path)
    if not registry.exists():
        raise CliError("INVALID_INPUT", "no ledger at %s" % registry)
    changed = 0
    rewritten = []
    for entry in entries:
        original = entry.get("expression") or ""
        if not original.strip():
            rewritten.append(entry)
            continue
        formatted = expression_lint.pretty_expression(original)
        if formatted != original:
            changed += 1
        entry = dict(entry)
        entry["expression"] = formatted
        rewritten.append(entry)

    # Identity guard: hash both sides *now* rather than trusting the stored field, which
    # rows written before 2026-10-31 carry in the older `expr:` form. Comparing a fresh
    # hash against a legacy one flagged 1208 untouched rows on the first attempt.
    mismatches = []
    for before, after in zip(entries, rewritten):
        old_text = before.get("expression") or ""
        new_text = after.get("expression") or ""
        if not old_text.strip():
            continue
        if alpha_registry.expression_hash(old_text) != alpha_registry.expression_hash(new_text):
            mismatches.append(after.get("alpha_id") or after.get("key"))
    if mismatches:
        raise CliError("INVALID_INPUT",
                       "%d row(s) would change identity: %s" % (len(mismatches),
                                                                ", ".join(mismatches[:5])),
                       "formatting must be semantic-preserving; nothing was written")

    if args.dry_run:
        return {"path": str(registry), "rows": len(rewritten), "changed": changed,
                "written": False, "dry_run": True}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = registry.with_suffix(registry.suffix + ".bak-%s" % stamp)
    shutil.copy2(registry, backup)
    temporary = registry.with_suffix(registry.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for entry in rewritten:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    temporary.replace(registry)
    return {"path": str(registry), "rows": len(rewritten), "changed": changed,
            "written": True, "backup": str(backup)}


def _book_marginals(entries, window_end="2023-01-01"):
    """Marginal contribution of every alpha with cached PnL, measured against the book.

    Two definitions were wrong before this one, and both looked plausible:

    * leave-one-out over *everything cached* made every dead-end arm look additive, because
      the "portfolio" was hundreds of unrelated alphas;
    * leave-one-out over the book only cannot score a candidate that is not in the book.

    So the book is the fixed reference: ``S_own - rho(own, book) * S_book``. For a book
    member this is the usual marginal (its own weight is 1/56 of the reference, which is
    negligible), and for an outside candidate it is exactly the entry test.
    """
    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"

    def load(alpha_id):
        path = pnl_dir / ("%s.json" % alpha_id)
        if not path.exists():
            return None
        try:
            return {k: float(v) for k, v in json.loads(path.read_text()).items()}
        except (OSError, ValueError):
            return None

    def sharpe(values):
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return None if var <= 0 else mean / (var ** 0.5) * (252 ** 0.5)

    series, book_ids = {}, []
    for entry in entries:
        alpha_id = entry.get("alpha_id")
        if not alpha_id:
            continue
        values = load(alpha_id)
        if values is None:
            continue
        series[alpha_id] = values
        if str(entry.get("status") or "").upper() == "ACTIVE":
            book_ids.append(alpha_id)
    if len(book_ids) < 2:
        return {}
    dates = sorted(set.intersection(*(set(series[a]) for a in book_ids)))
    dates = [day for day in dates if day < window_end]
    if len(dates) < 200:
        return {}
    book = [sum(series[a][d] for a in book_ids) / len(book_ids) for d in dates]
    book_sharpe = sharpe(book)
    if book_sharpe is None:
        return {}
    marginals = {}
    for alpha_id, values in series.items():
        if any(day not in values for day in dates):
            continue
        own = [values[d] for d in dates]
        rho = correlation.pearson(own, book)
        own_sharpe = sharpe(own)
        if rho is None or own_sharpe is None:
            continue
        marginals[alpha_id] = round(own_sharpe - rho * book_sharpe, 3)
    return marginals


def cmd_ledger_mechanisms(args, _sink):
    """How many independent mechanisms the book (or the ledger) actually holds.

    The campaign counts candidates by mechanism, but the only offline keys were the family
    label (too coarse) and PnL correlation (needs a simulation). This groups by
    ``(economic block, AST skeleton)`` - the definition "100 parameter variants count as
    one" - and reports size and median marginal per mechanism.
    """
    _path, entries = _ledger(args)
    selected = [entry for entry in entries
                if _ask_passes(entry, _ask_filters(args.where))]
    if args.status:
        wanted = {part.strip().upper() for part in args.status.split(",")}
        selected = [entry for entry in selected
                    if str(entry.get("status") or "").upper() in wanted]
    marginals = _book_marginals(entries) if args.with_marginal else {}
    data = mechanisms.summarise(selected, marginals)
    data["filter"] = {"where": args.where, "status": args.status}
    if not args.list:
        data["rows"] = [row for row in data["rows"][: args.limit]]
    return data


def cmd_ledger_convergence(args, _sink):
    """Per-batch signals for "is this line of work exhausted".

    Structural novelty is not a usable trigger: the last hundred arms were 93% new
    skeletons and almost none of them were additive. This reports the outcome number next
    to the structural one so the difference stays visible.
    """
    _path, entries = _ledger(args)
    marginals = _book_marginals(entries) if not args.no_marginal else {}
    data = mechanisms.convergence(entries, marginals, recent_batches=args.batches)
    data["thresholds"] = {"additive_marginal": 0.3,
                          "note": "trigger when recent_productive covers < 2 batches"}
    return data


def cmd_ledger_stability(args, _sink):
    """Which book alphas earn their train Sharpe in only one half of train.

    `omEaaMl2` made the case: a 2.11 train Sharpe whose following year went negative.
    A single aggregate number cannot show that, and the test year may not be used to
    select. Splitting train in half keeps the evidence inside the period selection is
    allowed to see, and a sign flip between halves is the cheapest warning that an
    edge is regime-bound rather than structural.
    """
    _path, entries = _ledger(args)
    selected = [entry for entry in entries if _ask_passes(entry, _ask_filters(args.where))]
    if getattr(args, "status", None):
        wanted = {part.strip().upper() for part in str(args.status).split(",") if part.strip()}
        selected = [entry for entry in selected
                    if str(entry.get("status") or "").upper() in wanted]
    latest = {}
    for entry in selected:
        key = entry.get("alpha_id")
        if key:
            latest[key] = entry          # ledger is append-only; last row wins
    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"
    rows = []
    for alpha_id, entry in latest.items():
        path = pnl_dir / ("%s.json" % alpha_id)
        if not path.exists():
            rows.append({"alpha_id": alpha_id, "status": entry.get("status"),
                         "family": entry.get("family"), "missing_pnl": True})
            continue
        try:
            pnl = {k: float(v) for k, v in json.loads(path.read_text()).items()}
        except (OSError, ValueError):
            rows.append({"alpha_id": alpha_id, "status": entry.get("status"),
                         "family": entry.get("family"), "missing_pnl": True})
            continue
        test_start = (entry.get("out_of_sample", {}).get("test") or {}).get("startDate")
        first, second = experiments.half_split_sharpes(pnl, test_start)
        flip = (first is not None and second is not None
                and first * second < 0 and min(abs(first), abs(second)) >= args.min_move)
        rows.append({"alpha_id": alpha_id, "status": entry.get("status"),
                     "family": entry.get("family"),
                     "h1": None if first is None else round(first, 2),
                     "h2": None if second is None else round(second, 2),
                     "gap": None if first is None or second is None else round(abs(first - second), 2),
                     "flip": flip})
    measured = [row for row in rows if not row.get("missing_pnl")]
    flips = [row for row in measured if row.get("flip")]
    measured.sort(key=lambda row: -(row.get("gap") or 0))
    data = {"count": len(measured), "flips": len(flips),
            "min_move": args.min_move,
            "median_gap": (sorted(row["gap"] for row in measured)[len(measured) // 2]
                           if measured else None),
            "worst": measured[: args.limit]}
    if args.list:
        data["flip_list"] = flips
    return data


def cmd_ledger_styles(args, _sink):
    """What each alpha compares against: other stocks, its own history, or nothing.

    The screen exists because one batch measured a 2.3x Fitness gap between the same
    variable read cross-sectionally (0.87) and against its own history (2.05). Where a
    book alpha never reads a field against its own past, that re-read is cheap and has
    a prior; where it already does, the question is answered.
    """
    _path, entries = _ledger(args)
    selected = [entry for entry in entries
                if _ask_passes(entry, _ask_filters(args.where))]
    if getattr(args, "status", None):
        wanted = {part.strip().upper() for part in str(args.status).split(",") if part.strip()}
        selected = [entry for entry in selected
                    if str(entry.get("status") or "").upper() in wanted]
    data = styles.summarize(selected)
    data["filter"] = {"where": getattr(args, "where", None),
                      "status": getattr(args, "status", None)}
    if args.list:
        data["static"] = [row for row in data["rows"] if row["comparison"] == "static"]
    if not args.list:
        for row in data["rows"]:
            row.pop("expression", None)
    return data


#: Columns the ledger can be filtered and grouped by. Everything here is read from
#: an entry, so a new axis is one line rather than a new command.
_ASK_FIELDS = ("status", "verdict", "family", "region", "universe", "neutralization",
               "delay", "decay", "month")

#: Aggregations --stat understands. ``floor_share`` is separate from the metric
#: aggregations because it answers a question about the sample rather than about a
#: column: how much of it is paying the ``max(turnover, 0.125)`` tax.
_ASK_STATS = ("n", "floor_share", "median", "mean", "p25", "p75", "min", "max")

_ASK_TURNOVER_FLOOR = 0.125


def _ask_field(entry, field):
    """Read one grouping/filter field off a ledger entry."""
    if field in ("status", "verdict", "family"):
        return entry.get(field)
    if field == "month":
        return (entry.get("recorded_at") or "")[:7] or None
    return (entry.get("settings") or {}).get(field)


def _ask_has(entry, what):
    if what == "metrics":
        return bool(entry.get("metrics"))
    if what == "checks":
        return bool(entry.get("checks"))
    if what == "pnl":
        alpha = entry.get("alpha_id")
        return bool(alpha) and (Path(research_paths.ensure_dir()) / "pnl" /
                                ("%s.json" % alpha)).exists()
    raise CliError("USAGE", "has=%r is not a thing" % what,
                   "use has=metrics, has=checks or has=pnl")


def _ask_filters(text):
    """Parse ``--where`` into ``[(field, op, value)]``.

    Two names carry a colon rather than a value: ``check=SELF_CORRELATION:PENDING``
    and ``has=pnl``. Both exist because the questions that started this command were
    "which rows never had their checks computed" and "which rows have a PnL".
    """
    filters = []
    for item in [piece.strip() for piece in (text or "").split(",") if piece.strip()]:
        op = "!=" if "!=" in item else "="
        field, _, value = item.partition(op)
        field, value = field.strip(), value.strip()
        if field not in _ASK_FIELDS + ("check", "has"):
            raise CliError("USAGE", "unknown --where field %r" % field,
                           "supported: " + ", ".join(_ASK_FIELDS + ("check", "has")))
        filters.append((field, op, value))
    return filters


def _ask_passes(entry, filters):
    for field, op, value in filters:
        if field == "check":
            name, _, expected = value.partition(":")
            actual = ((entry.get("checks") or {}).get(name) or {}).get("result")
            hit = str(actual or "") == expected
        elif field == "has":
            hit = _ask_has(entry, value)
        else:
            actual = _ask_field(entry, field)
            hit = str(actual) == value if actual is not None else False
        if hit != (op == "="):
            return False
    return True


def _ask_quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _ask_stat(entries, spec):
    """Compute one ``--stat`` term over a group."""
    if spec == "n":
        return len(entries)
    if spec == "floor_share":
        turnovers = [(entry.get("metrics") or {}).get("turnover") for entry in entries]
        turnovers = [value for value in turnovers if isinstance(value, (int, float))]
        if not turnovers:
            return None
        return round(sum(1 for value in turnovers if value < _ASK_TURNOVER_FLOOR)
                     / float(len(turnovers)), 4)
    name, _, metric = spec.partition(":")
    if name not in _ASK_STATS or not metric:
        raise CliError("USAGE", "unknown --stat %r" % spec,
                       "use n, floor_share or <agg>:<metric>; " + ", ".join(_ASK_STATS))
    values = [(entry.get("metrics") or {}).get(metric) for entry in entries]
    values = [value for value in values if isinstance(value, (int, float))]
    if not values:
        return None
    if name == "median":
        return round(_ask_quantile(values, 0.5), 4)
    if name == "mean":
        return round(sum(values) / float(len(values)), 4)
    if name == "p25":
        return round(_ask_quantile(values, 0.25), 4)
    if name == "p75":
        return round(_ask_quantile(values, 0.75), 4)
    if name == "min":
        return min(values)
    return max(values)


def _ask_panel(entries, axis, metrics):
    """Within-panel slope of ``axis`` against each metric.

    A panel is one construction under one set of settings, so the only thing varying
    inside it is the knob itself. Pooling the corpus instead measures how big alphas
    differ from small ones, which is a different question and a misleading answer.
    """
    panels = {}
    for entry in entries:
        settings = entry.get("settings") or {}
        key = (entry.get("expression_hash"), settings.get("region"),
               settings.get("universe"), settings.get("neutralization"),
               settings.get("truncation"), settings.get("delay"))
        panels.setdefault(key, []).append(entry)
    report = {}
    for metric in [piece.strip() for piece in metrics.split(",") if piece.strip()]:
        slopes = []
        detail = []
        for key, rows in panels.items():
            if len(rows) < 3:
                continue
            xs = [row.get("settings", {}).get(axis) for row in rows]
            ys = [(row.get("metrics") or {}).get(metric) for row in rows]
            if any(not isinstance(x, (int, float)) for x in xs):
                continue
            if any(not isinstance(y, (int, float)) for y in ys):
                continue
            r = correlation.pearson(xs, ys)
            if r is None:
                continue
            slopes.append(r)
            detail.append({"panel": key[0], "n": len(rows), "r": round(r, 4)})
        report[metric] = {
            "panels": len(slopes),
            "median_r": round(_ask_quantile(slopes, 0.5), 4) if slopes else None,
            "detail": sorted(detail, key=lambda item: abs(item["r"]), reverse=True),
        }
    return report


def _ask_matrix(entries, within, min_overlap):
    """Pairwise correlation of the matched rows' cached PnL."""
    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"
    groups = {}
    missing = []
    for entry in entries:
        alpha = entry.get("alpha_id")
        path = pnl_dir / ("%s.json" % alpha) if alpha else None
        if not path or not path.exists():
            if alpha:
                missing.append(alpha)
            continue
        try:
            series = json.loads(path.read_text())
        except (OSError, ValueError):
            missing.append(alpha)
            continue
        if within == "none":
            key = None
        elif within == "expr":
            key = entry.get("expression_hash")
        elif within == "family":
            key = entry.get("family")
        else:
            key = (entry.get("settings") or {}).get("neutralization")
        groups.setdefault(key, {}).setdefault(alpha, series)

    pairs, skipped = [], []
    for key, pnls in groups.items():
        if len(pnls) < 2:
            continue
        block = correlation.matrix(pnls, min_overlap=min_overlap)
        for pair in block["pairs"]:
            pairs.append({**pair, "within": key})
        for pair in block["skipped"]:
            skipped.append({**pair, "within": key})
    ordered = sorted(pairs, key=lambda item: abs(item["r"]), reverse=True)
    absolute = [abs(pair["r"]) for pair in pairs]
    # Every measured id belongs to one group; the spectrum is computed per group and
    # once over the whole matched set. `median_abs_r` says how tight the typical pair
    # is, the effective bet count says how many bets the set actually is - 24 clusters
    # can still be 7 bets.
    members = {}
    for key, pnls in groups.items():
        if len(pnls) >= 2:
            members[key] = sorted(pnls)
    per_group = []
    for key, names in members.items():
        block = [pair for pair in pairs if pair["within"] == key]
        spectrum = correlation.spectrum(names, block)
        per_group.append({"within": key, "n": len(names), **spectrum})
    everything = sorted({name for names in members.values() for name in names})
    overall = correlation.spectrum(everything, pairs) if len(members) == 1 else None
    return {
        "pairs_measured": len(pairs),
        "pairs_skipped": len(skipped),
        "median_abs_r": round(_ask_quantile(absolute, 0.5), 4) if absolute else None,
        "largest": ordered[0] if ordered else None,
        "effective_bets": (overall or {}).get("effective_bets"),
        "pc1_share": (overall or {}).get("pc1_share"),
        "groups": per_group,
        "pairs": ordered,
        "skipped": skipped[:20],
        "missing_pnl": missing[:20],
    }



def _ask_factor(entries, factor_ids, min_overlap, threshold=0.8):
    """Correlate every matched alpha against the average PnL of ``factor_ids``.

    This is the question a cluster count cannot answer: if a dozen alphas are all a
    re-parameterisation of one price leg, they correlate ~1 with the same factor and
    the book holds one bet, not a dozen. Building the factor from *already measured*
    members is what makes it a decomposition rather than a guess - the members come
    from an ablation that said "this leg carries that alpha".
    """
    pnl_dir = Path(research_paths.ensure_dir()) / "pnl"
    series, missing = {}, []
    for alpha in factor_ids:
        path = pnl_dir / ("%s.json" % alpha)
        if not path.exists():
            missing.append(alpha)
            continue
        try:
            series[alpha] = json.loads(path.read_text())
        except (OSError, ValueError):
            missing.append(alpha)
    if not series:
        return {"ids": list(factor_ids), "n": 0, "missing_pnl": missing[:20],
                "note": "no cached PnL for any factor member"}
    dates = sorted(set.intersection(*(set(value) for value in series.values())))
    if len(dates) < min_overlap:
        return {"ids": list(factor_ids), "n": len(series), "missing_pnl": missing[:20],
                "note": "factor members overlap only %d days" % len(dates)}
    rows = []
    for value in series.values():
        window = [value[day] for day in dates]
        mean = sum(window) / len(window)
        spread = (sum((item - mean) ** 2 for item in window) / (len(window) - 1)) ** 0.5
        rows.append([(item - mean) / spread if spread else 0.0 for item in window])
    factor = {day: sum(row[index] for row in rows) / len(rows)
              for index, day in enumerate(dates)}
    correlations = []
    for entry in entries:
        alpha = entry.get("alpha_id")
        if not alpha or alpha in series:
            continue
        path = pnl_dir / ("%s.json" % alpha)
        if not path.exists():
            continue
        try:
            other = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        left, right, shared = correlation.align(factor, other, min_overlap)
        if not left:
            continue
        value = correlation.pearson(left, right)
        if value is None:
            continue
        correlations.append({"alpha_id": alpha, "r": round(value, 4), "n": len(shared)})
    correlations.sort(key=lambda row: abs(row["r"]), reverse=True)
    above = [row for row in correlations if abs(row["r"]) >= threshold]
    others = [abs(row["r"]) for row in correlations if abs(row["r"]) < threshold]
    return {"ids": list(factor_ids), "n": len(series), "missing_pnl": missing[:20],
            "days": len(dates), "threshold": threshold,
            "correlations": correlations,
            "at_or_above_threshold": [row["alpha_id"] for row in above],
            "count_at_or_above_threshold": len(above),
            "median_abs_r_below_threshold": (round(sorted(others)[len(others) // 2], 4)
                                             if others else None)}


def cmd_ask(args, _sink):
    """Answer one read-only question about the ledger, in the shape it is asked.

    Every analysis in the 2026-11 campaign was the same four moves -- filter, group,
    aggregate, compare -- retyped as an inline script, which meant each one was
    written outside the test suite and re-derived the next time it was needed. This
    verb is that shape once: the filter is ``--where``, the group is ``--by``, the
    aggregate is ``--stat``, and the comparison is ``--panel``/``--matrix``.
    """
    _path, entries = _ledger(args)
    matched = [entry for entry in entries if _ask_passes(entry, _ask_filters(args.where))]
    stats = [piece.strip() for piece in (args.stat or "n").split(",") if piece.strip()]
    document = {
        "query": {"where": args.where, "by": args.by, "stat": args.stat,
                  "matrix_within": args.matrix_within if args.matrix else None},
        "rows_matched": len(matched),
    }
    if args.by:
        if args.by not in _ASK_FIELDS:
            raise CliError("USAGE", "unknown --by axis %r" % args.by,
                           "supported: " + ", ".join(_ASK_FIELDS))
        buckets = {}
        for entry in matched:
            buckets.setdefault(_ask_field(entry, args.by), []).append(entry)
        document["groups"] = [
            {"value": value, "n": len(rows),
             **{name: _ask_stat(rows, name) for name in stats if name != "n"}}
            for value, rows in sorted(buckets.items(),
                                      key=lambda item: -len(item[1]))
        ]
    else:
        document["summary"] = {name: _ask_stat(matched, name) for name in stats}
    if args.panel:
        if not args.by:
            raise CliError("USAGE", "--panel needs --by",
                           "a panel is the grouping axis whose slope you want")
        document["panel"] = _ask_panel(matched, args.by, args.panel)
    if args.matrix:
        document["matrix"] = _ask_matrix(matched, args.matrix_within, args.min_overlap)
    factor_ids = None
    if getattr(args, "factor_where", None):
        # Build the factor from a query instead of a pasted id list: "correlate every
        # candidate against the book" is `--factor-where status=ACTIVE`, and typing 51
        # ids by hand is how a probe becomes stale the moment the book changes.
        factor_ids = [entry.get("alpha_id") for entry in entries
                      if _ask_passes(entry, _ask_filters(args.factor_where))
                      and entry.get("alpha_id")]
    elif getattr(args, "factor_ids", None):
        factor_ids = [part.strip() for part in str(args.factor_ids).replace(",", " ").split()
                      if part.strip()]
    if factor_ids:
        document.setdefault("query", {})["factor_where"] = getattr(args, "factor_where", None)
        document["factor"] = _ask_factor(matched, factor_ids, args.min_overlap,
                                         getattr(args, "factor_threshold", 0.8))
    if args.list:
        document["rows"] = [
            {"alpha_id": entry.get("alpha_id"), "status": entry.get("status"),
             "verdict": entry.get("verdict"), "family": entry.get("family"),
             "neutralization": (entry.get("settings") or {}).get("neutralization"),
             "decay": (entry.get("settings") or {}).get("decay"),
             "sharpe": (entry.get("metrics") or {}).get("sharpe"),
             "fitness": (entry.get("metrics") or {}).get("fitness")}
            for entry in matched[: args.limit]
        ]
    return document
