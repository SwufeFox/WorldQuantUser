"""Commands for recording, labelling and inspecting the local research ledger."""

from __future__ import annotations

import json
import math
from pathlib import Path

import alpha_registry
import correlation
import machine_lib as ml
import research_paths

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
    if getattr(args, "factor_ids", None):
        factor_ids = [part.strip() for part in str(args.factor_ids).replace(",", " ").split()
                      if part.strip()]
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
