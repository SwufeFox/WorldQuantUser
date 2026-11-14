"""Simulation, resumption, labelled probes, ablation and comparison workflows."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path

import ablation
import alpha_registry
import alpha_unit
import cache_store
import expression_lint
import machine_lib as ml
import research_config as config_io
import research_paths
from tools import guarded_sim

from .runtime import CliError, METRIC_KEYS, collect_alphas, merge_settings, note, protect_output, with_session
from . import costs, selection, stats
from .validation import _preflight_lint, attach_field_contracts, cached_field_scope


#: Operators per arm above which the report says so. Not a hard limit (the platform's
#: is 64); a research limit. The user's rule from 2026-11-07: many operators means more
#: room to fit, and mixed signals blur what a result actually tests.
SIMPLE_OPERATOR_LIMIT = 8


def half_split_sharpes(pnl, test_start=None):
    """Sharpe of the first and second half of the train window.

    Selection uses the train period, and a train Sharpe can be the average of two very
    different halves. Future test performance cannot drive selection. The halves stay
    inside train, so they can reveal whether an edge depends on one train regime
    before choosing a candidate.

    PnL rather than returns: only the PnL series is exposed, and Sharpe is
    scale-invariant, so a within-alpha comparison is unaffected.
    """
    if not pnl or not test_start:
        return None, None
    days = sorted(day for day in pnl if day < test_start)
    if len(days) < 120:
        return None, None
    middle = len(days) // 2

    def sharpe(subset):
        return stats.sharpe([pnl[day] for day in subset], min_points=2)

    return sharpe(days[:middle]), sharpe(days[middle:])


def _train_metrics(record):
    """The metrics a selection decision is allowed to see: the train period only.

    The platform's headline numbers use the full window, which contains the test
    year, so every path that ranks, filters or ablates must come through here unless
    the caller has explicitly frozen the design (``--frozen`` / ``--show-test``).
    """
    return (record.get("out_of_sample") or {}).get("train") or {}


def _finished(record):
    """A run that produced final numbers.

    ``WARNING`` is a finished run: BRAIN marks a simulation WARNING when a check
    warns without failing (UNITS is the usual one) and every metric is final.
    Counting only COMPLETE made a batch of WARNING records look like zero
    completions, so `simulate` reported SIMULATION_FAILED and skipped recording.
    """
    return (record.get("status") or "").upper() in alpha_registry.FINISHED_STATUSES


def summarize(records):
    """Count outcomes and rank the scored alphas so a caller can pick winners."""
    completed = [r for r in records if _finished(r)]
    failed = [r for r in records if (r.get("status") or "").upper() == "FAILED"]
    scored = [r for r in completed if isinstance(r.get("metrics"), dict)]
    ranking = sorted(
        (
            {
                "alpha_id": r.get("alpha_id"),
                "expression": r.get("expression"),
                **{k: r["metrics"].get(k) for k in METRIC_KEYS if k in r["metrics"]},
            }
            for r in scored
        ),
        key=lambda item: item.get("sharpe") if item.get("sharpe") is not None else float("-inf"),
        reverse=True,
    )
    return {
        "total": len(records),
        "completed": len(completed),
        "failed": len(failed),
        "pending": sum(
            (r.get("status") or "").upper() == "PENDING" or bool(r.get("progress_url"))
            for r in records
        ),
        "queued": sum((r.get("status") or "").upper() == "QUEUED" for r in records),
        "other": len(records) - len(completed) - len(failed),
        "ranking": ranking,
        "best": ranking[0] if ranking else None,
    }


def _metric_value(record, key):
    metrics = record.get("metrics") or {}
    value = metrics.get(key)
    return value if value is not None else float("-inf")


def _compact_metrics(record):
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        record.pop("metrics", None)
        return
    record["metrics"] = {k: metrics.get(k) for k in METRIC_KEYS if k in metrics}


def _write_simulation_checkpoint(path, checkpoint):
    cache_store.write_json_atomic(path, checkpoint, indent=2)


def _simulation_checkpoint_callback(path, checkpoint, sink):
    lock = threading.Lock()

    def on_event(kind, payload):
        with lock:
            if kind in {"submitting", "submitted", "pending", "completed", "failed", "deferred"}:
                index = payload.get("index")
                if isinstance(index, int) and 0 <= index < len(checkpoint["records"]):
                    checkpoint["records"][index] = {
                        key: value for key, value in payload.items() if key != "index"
                    }
                    _write_simulation_checkpoint(path, checkpoint)
            if sink:
                sink(kind, payload)

    return on_event


def _sort_simulation_results(records, args):
    for record in records:
        _compact_metrics(record)
    if args.sort != "none":
        records.sort(key=lambda r: _metric_value(r, args.sort), reverse=True)
    if args.top is not None:
        records = records[:args.top]
    return records


def _simulation_result(records, settings, checkpoint_path=None, dry_run=False):
    summary = summarize(records)
    if summary["pending"] or summary["queued"]:
        status = "PENDING"
    elif summary["failed"]:
        status = "PARTIAL" if summary["completed"] else "FAILED"
    else:
        status = "COMPLETE"
    result = {
        "status": status,
        "dry_run": dry_run,
        "settings": settings,
        "summary": summary,
        "results": records,
    }
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        result["checkpoint"] = str(Path(checkpoint_path))
    return result


def _prepare_simulation_run(settings, pairs, batch_size, checkpoint_arg,
                            labels=None, families=None, candidates=None, metadata=None):
    if checkpoint_arg:
        path = Path(checkpoint_arg).expanduser().resolve()
        if path.exists():
            raise CliError(
                "INVALID_INPUT",
                "checkpoint already exists; use `resume` or choose another path",
                str(path),
            )
    else:
        path = Path(".brain_checkpoints") / ("simulate-%s.json" % uuid.uuid4().hex)
        path = path.resolve()
    # The label columns must be written here, not only in the dry-run branch.
    # These records are handed to simulate_alphas as initial_records, so anything
    # omitted from this dict is lost for the whole real run: mechanism tags
    # arrived empty in the ledger for exactly this reason.
    labels = labels or [None] * len(pairs)
    families = families or [None] * len(pairs)
    candidates = candidates or [None] * len(pairs)
    records = [
        {
            "expression": expression,
            "decay": decay,
            "candidate_id": candidates[index],
            "mechanism": labels[index],
            "family": families[index],
            "alpha_id": None,
            "status": "QUEUED",
            "metrics": None,
            "checks": None,
            "error": None,
        }
        for index, (expression, decay) in enumerate(pairs)
    ]
    checkpoint = {
        "format": "worldquantuser.simulation-checkpoint",
        "version": 1,
        "settings": settings,
        "batch_size": batch_size,
        "records": records,
    }
    if metadata:
        checkpoint["metadata"] = dict(metadata)
    _write_simulation_checkpoint(path, checkpoint)
    return path, checkpoint


def _finalize_simulation(records, settings, args, checkpoint_path, checkpoint, explicit):
    checkpoint["records"] = [dict(record) for record in records]
    _write_simulation_checkpoint(checkpoint_path, checkpoint)
    has_unfinished = any(
        record.get("progress_url")
        or record.get("status") in {"PENDING", "QUEUED", "UNKNOWN_SUBMISSION"}
        for record in records
    )
    if not explicit and not has_unfinished:
        Path(checkpoint_path).unlink(missing_ok=True)
        checkpoint_path = None

    # The display view honours --sort/--top; the batch, the summary and the ledger
    # must not. Truncating before recording lost every arm outside the top N - and
    # with the checkpoint deleted at the end, those alphas had been paid for and
    # left no trace, which is both wasted quota and a selection bias in the archive.
    every = list(records)
    display = _sort_simulation_results(records, args)
    summary = summarize(every)
    if summary["completed"] == 0 and summary["total"] and not (
        summary["pending"] or summary["queued"] or any(
            record.get("status") == "UNKNOWN_SUBMISSION" for record in every)
    ):
        first_error = next(
            (r.get("error") for r in every if r.get("error")), "unknown error"
        )
        raise CliError(
            "SIMULATION_FAILED",
            "all %d simulations failed" % summary["total"],
            str(first_error),
        )
    result = _simulation_result(every, settings, checkpoint_path)
    result["results"] = display
    result["unknown_submission"] = [index for index, record in enumerate(every)
                                    if record.get("status") == "UNKNOWN_SUBMISSION"]
    result["results_total"] = len(every)
    for key in ("legs", "legend", "note"):
        value = (checkpoint.get("metadata") or {}).get(key)
        if value is not None:
            result[key] = value
    pending_ids = [record.get("alpha_id") or record.get("candidate_id")
                   for record in every
                   if (record.get("status") or "").upper() in {"PENDING", "QUEUED"}]
    if pending_ids:
        result["pending"] = pending_ids
        result["next"] = ("some arms are still running; the checkpoint was kept - "
                          "run `resume %s` before judging them"
                          % (checkpoint_path or "<checkpoint>"))
    if getattr(args, "record", False):
        result["registry"] = _record_simulation(every, settings, args)
    return result


def cmd_ablate(args, _sink):
    """Write a research file holding full / leave-one-out / solo variants.

    Deliberately does not simulate: generating and spending stay separate, so a
    mis-specified ablation costs nothing and can be inspected first.
    """
    expression = args.expr
    decay = args.decay
    if args.file and expression is not None:
        raise CliError("INVALID_INPUT", "ablate accepts either a file or --expr, not both")
    if args.ids and (args.file or expression is not None):
        raise CliError("INVALID_INPUT", "--ids cannot be combined with a file or --expr")
    settings = {}
    settings_source = "defaults"
    alphas = None
    if args.file:
        try:
            settings, alphas = config_io.load_research_file(args.file)
        except config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc)) from exc
        settings_source = args.file
    resolved = merge_settings(settings, args)
    if any(getattr(args, flag, None) is not None for flag in (
            "region", "universe", "instrument_type", "neutralization", "decay",
            "delay", "truncation", "pasteurization", "nan_handling", "test_period")):
        settings_source = "flags"
    if alphas is not None:
        try:
            _settings, pairs = config_io.build_config(resolved, alphas)
        except config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc)) from exc
        if len(pairs) != 1:
            raise CliError("INVALID_INPUT",
                           "ablate takes exactly one composite expression, found %d" % len(pairs),
                           "split the file, or pass --expr with one composite")
        expression = pairs[0][0]
        if decay is None:
            decay = pairs[0][1]
    elif expression is not None:
        try:
            _settings, pairs = config_io.build_config(resolved, [expression])
        except config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc)) from exc
        expression, decay = pairs[0]

    if args.ids:
        # Batch form. One file per alpha, because `compare --contributions` groups by
        # candidate_id and reads `full` / `loo:<leg>` / `solo:<leg>`; merging six
        # alphas into one file would collide on those names and silently attribute
        # one alpha's leg contributions to another.
        entries = alpha_registry.load_entries(getattr(args, "registry", None))
        out_dir = Path(args.out_dir or research_paths.ensure_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        modes = [part.strip() for part in str(args.modes).split(",") if part.strip()]
        done, skipped = [], []
        for raw in str(args.ids).replace(",", " ").split():
            entry = alpha_registry.resolve(entries, raw)
            if entry is None:
                skipped.append({"alpha_id": raw, "reason": "not in the ledger"})
                continue
            source = entry.get("expression") or ""
            legs_legend = None
            try:
                plan = ablation.variants(source, modes)
            except ablation.AblationError as exc:
                point = alpha_unit.split(source)
                if point is None:
                    skipped.append({"alpha_id": raw,
                                    "reason": "%s; and it does not parse" % exc})
                    continue
                legs_legend = {
                    "l%d" % index: {"fields": list(leg.fields),
                                    "role": leg.role, "weight": leg.weight}
                    for index, leg in enumerate(getattr(point, "legs", ()) or ())
                }
                plan = []
                if "full" in modes:
                    plan.append({"candidate_id": "full", "expression": source})
                for variant in alpha_unit.derive(point):
                    if variant.rule != "drop_term":
                        continue
                    name = variant.detail or "t%d" % len(plan)
                    plan.append({"candidate_id": "loo:%s" % name,
                                 "expression": variant.expression})
                if not plan:
                    skipped.append({"alpha_id": raw,
                                    "reason": "unit layer derived no ablations"})
                    continue
            alpha_settings = merge_settings(entry.get("settings") or {}, args)
            alphas = []
            for item in plan:
                item_entry = {"expression": item["expression"],
                              "candidate_id": item["candidate_id"],
                              "mechanism": entry.get("mechanism") or raw}
                item_entry["decay"] = alpha_settings["decay"]
                alphas.append(item_entry)
            target = out_dir / ("%s-ablation.json" % raw)
            document = {"settings": alpha_settings, "alphas": alphas}
            if legs_legend:
                document["legs"] = legs_legend
            target.write_text(json.dumps(document, ensure_ascii=False, indent=2))
            done.append({"alpha_id": raw, "file": str(target), "variants": len(plan),
                         "source": "unit" if legs_legend else "named-legs"})
            note("wrote %s (%d variants)" % (target, len(plan)))
        return {"mode": "ids", "written": done, "skipped": skipped,
                "count": sum(item["variants"] for item in done),
                "next": "simulate each file, then `compare <results> --contributions`"}

    if not expression:
        raise CliError("USAGE", "nothing to ablate",
                       "pass a research file, --expr, or --ids")

    modes = [part.strip() for part in str(args.modes).split(",") if part.strip()]
    try:
        plan = ablation.variants(expression, modes)
    except ablation.AblationError as exc:
        raise CliError("INVALID_INPUT", str(exc),
                       "ablation needs 'name = ...; name = ...; name + name'; "
                       "declare the legs explicitly and test by hand otherwise")

    alphas = []
    for item in plan:
        entry = {
            "expression": item["expression"],
            "candidate_id": item["candidate_id"],
            "mechanism": item["candidate_id"],
        }
        if decay is not None:
            entry["decay"] = decay
        alphas.append(entry)
    document = {"settings": resolved, "alphas": alphas}

    if args.out:
        Path(args.out).write_text(json.dumps(document, ensure_ascii=False, indent=2))
        note("wrote %s" % args.out)

    data = {
        "source_expression": expression,
        "modes": modes,
        "variants": plan,
        "count": len(plan),
        "settings": resolved,
        "settings_source": settings_source,
        "written_to": args.out,
        "next": ("run with `simulate %s`, then `compare <results> --contributions`"
                 % (args.out or "<file>")),
    }

    return data


def _compare_inputs(args):
    """Select files before loading JSON, retaining where each selection came from."""
    import glob

    explicit = [str(path) for path in args.files]
    patterns = list(getattr(args, "match", None) or [])
    latest = getattr(args, "latest", None)
    if not explicit and not patterns:
        raise CliError("USAGE", "compare needs files or --match")
    if explicit.count("-") > 1:
        raise CliError("INVALID_INPUT", "stdin (-) may be read only once")
    if latest is not None and (not isinstance(latest, int) or latest < 1):
        raise CliError("USAGE", "--latest must be a positive integer")
    if latest is not None and "-" in explicit:
        raise CliError("USAGE", "--latest cannot select from stdin")
    root = Path(research_paths.research_dir()).resolve()
    candidates = {}

    def add(path, source):
        resolved = "-" if str(path) == "-" else str(Path(path).resolve())
        if resolved not in candidates:
            if resolved == "-":
                modified = None
            else:
                try:
                    if not Path(resolved).is_file():
                        raise OSError("not a file")
                    modified = Path(resolved).stat().st_mtime_ns
                except OSError as exc:
                    raise CliError("INVALID_INPUT", "cannot read %s: %s" % (path, exc)) from exc
            candidates[resolved] = {"path": resolved, "sources": [], "mtime_ns": modified}
        if source not in candidates[resolved]["sources"]:
            candidates[resolved]["sources"].append(source)

    for path in explicit:
        add(path, "explicit")
    for pattern in patterns:
        matches = [path for path in glob.iglob(str(root / pattern), recursive=True)
                   if Path(path).is_file()]
        if not matches:
            raise CliError("NOT_FOUND", "no files match %s" % pattern,
                           "patterns are resolved under %s" % root)
        for path in matches:
            add(path, pattern)
    selected = list(candidates.values())
    if patterns or latest is not None:
        selected.sort(key=lambda item: (item["mtime_ns"] is None,
                                       -(item["mtime_ns"] or 0), item["path"]))
    if latest is not None:
        selected = selected[:latest]
    paths = [item["path"] for item in selected]
    return paths, {"research_root": str(root), "patterns": patterns,
                   "explicit_files": explicit, "latest": latest,
                   "matched_count": len(candidates), "selected_paths": paths,
                   "provenance": selected}


def _compare_document(path, payload):
    """Keep raw records for ablation; normalize only explicit probe display fields."""
    while isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if isinstance(payload, dict) and payload.get("path") and not payload.get("records"):
        destination = Path(payload["path"]).expanduser()
        if not destination.is_absolute() and str(path) != "-":
            destination = Path(path).parent / destination
        _canonical, payload = _load_probe_batch(destination)
    display = []
    probe = False
    ordered_probe = False
    if isinstance(payload, list):
        records = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
        records = [item for item in payload["results"] if isinstance(item, dict)]
    elif isinstance(payload, dict):
        display = payload.get("rows")
        # A labelled row with explicit train fields is safe to normalize even in
        # an older snapshot without probe metadata. Headline-only tables are not.
        ordered_probe = "spec" in payload or ("name" in payload and any(
            key in payload for key in ("simulated", "incomplete", "design")))
        probe = isinstance(display, list) and (ordered_probe or bool(display) and all(
            isinstance(row, dict) and row.get("label")
            and any(key.startswith("train_") for key in row) for row in display))
        if isinstance(payload.get("records"), list):
            records = [item for item in payload["records"] if isinstance(item, dict)]
        elif probe:
            records = []
        else:
            raise CliError("INVALID_INPUT", "%s is not a simulate or probe result document" % path,
                           "expected results, raw records, or labelled probe rows with train fields")
    else:
        raise CliError("INVALID_INPUT", "%s is not a simulate or probe result document" % path)

    if not probe:
        if any(any(key in record for key in ("sharpe", "fitness", "train_sharpe", "train_fitness"))
               and not any(key in record for key in ("metrics", "out_of_sample")) for record in records):
            raise CliError("INVALID_INPUT", "%s contains flattened rows without probe metadata" % path,
                           "use raw simulation records or labelled probe rows with explicit train fields")
        return records, [(record, None) for record in records]

    display = [row for row in display if isinstance(row, dict)]
    used = set()
    aligned = []

    def label(record):
        return record.get("label") or record.get("mechanism")

    def compatible(row, record):
        return not (row.get("alpha_id") and record.get("alpha_id")
                    and row["alpha_id"] != record["alpha_id"]) and not (
                        row.get("label") and label(record) and row["label"] != label(record))

    for position, row in enumerate(display):
        matches = [index for index, record in enumerate(records) if index not in used
                   and compatible(row, record) and
                   ((row.get("alpha_id") and row["alpha_id"] == record.get("alpha_id"))
                    or (row.get("label") and row["label"] == label(record)))]
        index = matches[0] if len(matches) == 1 else None
        # Probe checkpoints save parallel arrays; order is evidence only when no
        # identity conflicts and no ambiguous identity match exist.
        if index is None and not matches and ordered_probe and len(display) == len(records) and position not in used:
            candidate = records[position]
            claimed_elsewhere = any(other is not row and (
                candidate.get("alpha_id") and candidate["alpha_id"] == other.get("alpha_id")
                or label(candidate) and label(candidate) == other.get("label")) for other in display)
            if compatible(row, candidate) and not claimed_elsewhere:
                index = position
        if index is not None:
            used.add(index)
            aligned.append((records[index], row))
        else:
            aligned.append(({}, row))
    aligned.extend((record, None) for index, record in enumerate(records) if index not in used)
    if not records:
        # Row-only snapshots still provide explicit train evidence for ablation.
        records = [{"candidate_id": row.get("candidate_id"), "alpha_id": row.get("alpha_id"),
                    "metrics": {key: row.get(key) for key in ("sharpe", "fitness", "turnover", "returns")},
                    "out_of_sample": {"train": {key: row.get("train_" + key)
                                                for key in ("sharpe", "fitness", "turnover", "returns")}}}
                   for row in display]
    return records, aligned


def _compare_row(record, display, path, frozen):
    display = display or {}
    train = _train_metrics(record)
    halves = record.get("half_sharpes") or (None, None)
    row = {key: record[key] if record.get(key) is not None else display.get(key)
           for key in ("candidate_id", "alpha_id", "decay", "status")}
    row.update({"label": display.get("label") or record.get("label") or record.get("mechanism"),
                "source_file": path,
                "source_expression": record.get("expression") or display.get("expression"),
                "source_settings": record.get("settings") or display.get("settings"),
                "train_period": {key: train.get(key) for key in ("startDate", "endDate", "turnoverPeriod")},
                "train_sharpe_h1": halves[0] if len(halves) > 0 and halves[0] is not None
                else display.get("train_sharpe_h1"),
                "train_sharpe_h2": halves[1] if len(halves) > 1 and halves[1] is not None
                else display.get("train_sharpe_h2"),
                "failed_checks": None, "checks_state": "hidden"})
    for key in ("sharpe", "fitness", "turnover", "returns", "margin"):
        row["train_" + key] = train.get(key, display.get("train_" + key))
    if frozen:
        metrics = record.get("metrics") or {}
        row.update({key: metrics.get(key, display.get(key))
                    for key in ("sharpe", "fitness", "turnover", "returns")})
        test = (record.get("out_of_sample") or {}).get("test") or {}
        row["test_sharpe"] = test.get("sharpe", display.get("test_sharpe"))
        checks = record.get("checks")
        if isinstance(checks, dict):
            row["failed_checks"] = sorted(name for name, value in checks.items()
                                          if isinstance(value, dict)
                                          and value.get("result") in ("FAIL", "ERROR"))
        elif isinstance(display.get("failed_checks"), list):
            row["failed_checks"] = display["failed_checks"]
        row["checks_state"] = "available" if row["failed_checks"] is not None else "unavailable"
    return row

def _selection_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (ValueError, OSError) as exc:
        raise CliError("INVALID_INPUT", str(exc)) from exc



def cmd_compare(args, _sink):
    """Merge result documents into one table.

    Variants used to live in separate batch files and were aligned by eye, which
    is how a five-leg version scoring below its four-leg parent went unnoticed.

    Two views, because the default one must not leak the test year. ``train`` ranks
    on the train period and does not print the full-window numbers; ``frozen`` shows
    them for a design that is already fixed.
    """
    frozen = bool(getattr(args, "frozen", False))
    sort_key = getattr(args, "sort", None) or "fitness"
    cost_bps = getattr(args, "cost_bps", None)
    cost_bps = list(costs.DEFAULT_COST_BPS) if cost_bps is None else list(cost_bps)
    manifest = None
    if frozen:
        if not getattr(args, "selection", None):
            raise CliError("USAGE", "--frozen requires --selection from a train --freeze")
        if getattr(args, "freeze", None):
            raise CliError("USAGE", "cannot freeze a validation view")
        if getattr(args, "sort", None) is not None or getattr(args, "min_fitness", None) is not None:
            raise CliError("USAGE", "frozen validation cannot change --sort or --min-fitness")
        manifest = _selection_call(selection.load, args.selection, source="compare")
        saved_costs = manifest["criteria"].get("cost_bps", list(costs.DEFAULT_COST_BPS))
        if getattr(args, "cost_bps", None) is not None and cost_bps != saved_costs:
            raise CliError("USAGE", "cost scenarios conflict with the frozen selection")
        cost_bps = saved_costs
        paths = list((manifest.get("inputs") or {}).get("files") or [])
        if not paths:
            raise CliError("INVALID_INPUT", "selection has no saved compare input files")
        provenance = (manifest.get("inputs") or {}).get("provenance") or {}
        supplied = [str(Path(path).resolve()) for path in getattr(args, "files", [])]
        if supplied and supplied != paths or getattr(args, "match", None) or getattr(args, "latest", None):
            raise CliError("USAGE", "frozen comparison reuses the saved input files")
    else:
        if getattr(args, "selection", None):
            raise CliError("USAGE", "--selection requires --frozen")
        paths, provenance = _compare_inputs(args)
    records = []
    inputs = []
    rows = []
    for path in paths:
        try:
            payload = json.loads(config_io.read_input_text(path))
        except (config_io.ResearchError, ValueError) as exc:
            if frozen:
                continue
            raise CliError("INVALID_INPUT", "cannot read %s: %s" % (path, exc),
                           "point at a file written by `simulate` or `probe`") from exc
        file_records, aligned = _compare_document(path, payload)
        records.extend(file_records)
        inputs.append((path, payload, file_records))
        rows.extend(_compare_row(record, display, path, frozen) for record, display in aligned)

    if not rows and not frozen:
        raise CliError("NOT_FOUND", "no result records in the given files",
                       "the file must come from `simulate` or `probe`")

    fitness_key = "train_fitness"
    if not frozen and args.min_fitness is not None:
        rows = [row for row in rows
                if isinstance(row.get(fitness_key), (int, float))
                and row[fitness_key] >= args.min_fitness]

    if not frozen and sort_key != "none":
        key = "alpha_id" if sort_key == "id" else sort_key
        if key in {"sharpe", "fitness", "turnover", "returns"}:
            key = "train_" + key
        rows.sort(key=lambda row: (row[key] is None, -(row[key] or 0)
                                   if isinstance(row[key], (int, float))
                                   else str(row[key])))

    for row in rows:
        row["cost_evidence"] = costs.evidence(
            {**row["train_period"], **{key: row.get("train_" + key) for key in ("margin", "turnover")}},
            "train", cost_bps)
    if manifest:
        rows = _selection_call(selection.ordered_rows, manifest, [row for row in rows if row.get("alpha_id")])
        for row, saved in zip(rows, manifest["rows"]):
            row["source_changes"] = {key: {"frozen": saved.get(key), "current": row.get(key)}
                                     for key in ("source_expression", "source_settings", "train_period")
                                     if not row.get("missing") and saved.get(key) != row.get(key)}
    data = {"files": paths, "selection": provenance, "count": len(rows),
            "view": "frozen" if frozen else "train", "rows": rows,
            "missing_evidence": [{"alpha_id": row.get("alpha_id"), "source_file": row.get("source_file"),
                "train_missing": [key for key in ("sharpe", "fitness", "turnover", "margin")
                                  if row.get("train_" + key) is None]} for row in rows
                if any(row.get("train_" + key) is None for key in ("sharpe", "fitness", "turnover", "margin"))]}
    if getattr(args, "freeze", None):
        if "-" in paths:
            raise CliError("USAGE", "freeze requires durable input files, not stdin")
        manifest = _selection_call(selection.freeze, args.freeze, source="compare", rows=rows,
            criteria={"min_fitness": args.min_fitness, "sort": sort_key, "scope": "train", "cost_bps": cost_bps},
            inputs={"files": paths, "provenance": provenance}, window={"scope": "train"})
        data["selection_path"] = str(Path(args.freeze).resolve())
    if manifest:
        data["selection_id"] = manifest["selection_id"]
    if frozen:
        research_paths.log_view("compare", "frozen", [row.get("alpha_id") for row in rows],
                                note={"selection_id": manifest["selection_id"], "files": paths})
    if getattr(args, "per_file", False):
        # Variant ids repeat across files, so keep their contributions separate.
        rules = alpha_unit.load_rules() or {}
        dataset_family = rules.get("datasets") or {}
        field_family = {field: dataset_family.get(dataset, dataset)
                        for field, dataset in (alpha_unit.load_field_datasets() or {}).items()}
        per_file = []
        for path, payload, file_records in inputs:
            while isinstance(payload, dict) and "data" in payload:
                payload = payload["data"]
            legend = payload.get("legs") if isinstance(payload, dict) else None
            if not legend and str(path) != "-":
                sibling = Path(path).with_name(
                    Path(path).name.replace("-results.json", "-ablation.json"))
                if sibling != Path(path) and sibling.exists():
                    try:
                        sibling_payload = json.loads(config_io.read_input_text(sibling))
                        if isinstance(sibling_payload, dict):
                            legend = sibling_payload.get("legs")
                    except (config_io.ResearchError, ValueError):
                        pass
            report = ablation.report(
                file_records, metrics_of=_train_metrics)
            for row in report["legs"]:
                if legend and row["leg"] in legend:
                    row["fields"] = legend[row["leg"]].get("fields")
                    row["role"] = legend[row["leg"]].get("role")
                    row["families"] = sorted({field_family.get(field, "?")
                                              for field in (row["fields"] or [])})
            scored = [row for row in report["legs"]
                      if isinstance(row.get("contribution"), (int, float))]
            per_file.append({
                "file": path,
                "alpha_id": Path(path).stem.split("-")[0],
                "full_fitness": report["full_fitness"],
                "top_leg": max(scored, key=lambda row: row["contribution"]) if scored else None,
                "contributions": report,
            })
        data["per_file"] = per_file
        data["per_file_view"] = "train"
    elif args.contributions:
        # The contribution is a selection decision ("does this leg earn its place"),
        # so in the exploration view it has to be computed from train-period fitness.
        # It used to read the full-window block regardless, which put the test year
        # back into the answer the table had just hidden.
        data["contributions"] = ablation.report(
            records, metrics_of=_train_metrics)
        data["contributions"]["view"] = "train"
    if getattr(args, "summary", False):
        data["rows"] = rows[:20]
        data["rows_omitted"] = max(0, len(rows) - 20)

    return data


def _load_probe_spec(path):
    """Read and validate a probe spec: settings plus a list of labelled arms."""
    try:
        spec = json.loads(config_io.read_input_text(path))
    except (config_io.ResearchError, ValueError) as exc:
        raise CliError("INVALID_INPUT", "cannot read spec %s: %s" % (path, exc),
                       "pass a JSON file shaped like "
                       "{\"name\": ..., \"settings\": {...}, \"arms\": [{\"label\", "
                       "\"expression\"}]}") from exc
    if not isinstance(spec, dict):
        raise CliError("INVALID_INPUT", "spec must be a JSON object",
                       "see `research_cli.py schema probe`")
    arms = spec.get("arms")
    if not isinstance(arms, list) or not arms:
        raise CliError("INVALID_INPUT", "spec has no arms",
                       "each arm needs at least {\"expression\": ...}")
    known = {"name", "idea", "mechanism", "note", "settings", "arms", "design"}
    unknown = sorted(set(spec) - known)
    if unknown:
        # A typo here is silent otherwise: `"setting"` instead of `"settings"` would
        # run the whole batch on the defaults and look like a failed hypothesis.
        note("spec: ignoring unknown key(s) %s (known: %s)"
             % (", ".join(unknown), ", ".join(sorted(known))))
    if "settings" in spec and not isinstance(spec["settings"], dict):
        raise CliError("INVALID_INPUT", "spec settings must be an object")
    for index, arm in enumerate(arms, 1):
        if not isinstance(arm, dict) or not isinstance(arm.get("expression"), str) \
                or not arm["expression"].strip():
            raise CliError("INVALID_INPUT", "arm %d has no expression" % index,
                           "an arm is {\"label\": ..., \"expression\": ..., "
                           "\"settings\": {...} (optional)}")
        if "settings" in arm and not isinstance(arm["settings"], dict):
            raise CliError("INVALID_INPUT", "arm %d settings must be an object" % index)
        arm.setdefault("label", "arm%d" % index)
    return spec


# Explore is an explicit, recorded opt-out from the controlled-design rules.
_DESIGNS = ("control", "add", "drop", "parallel", "rescue")


def _probe_design(spec, settings):
    """Validate a control and its interventions before authenticating or spending."""
    mode = spec.get("design")
    if mode is not None and mode not in _DESIGNS + ("explore",):
        raise CliError("INVALID_INPUT", "spec design %r is not a design" % (mode,),
                       "use one of " + ", ".join(_DESIGNS + ("explore",)))
    if mode == "explore":
        return {"mode": "explore", "control": None,
                "note": "scan, not an experiment: no arm pair is being compared"}

    arms = spec["arms"]
    unlabelled = [arm["label"] for arm in arms if not arm.get("design")]
    if unlabelled:
        raise CliError(
            "INVALID_INPUT", "arm(s) %s carry no design" % ", ".join(unlabelled),
            "label every arm control/add/drop/parallel/rescue, or set "
            "\"design\": \"explore\" on the spec to record that this is a scan")
    controls = [arm for arm in arms if arm.get("design") == "control"]
    if not controls:
        raise CliError("INVALID_INPUT", "no control arm",
                       "run the parent expression unchanged as a control arm")
    if len(controls) > 1:
        raise CliError("INVALID_INPUT",
                       "%d control arms; a batch has exactly one" % len(controls),
                       "the other arms compare against it")
    control = controls[0]
    base = {**settings, **(control.get("settings") or {})}
    for arm in arms:
        design = arm.get("design")
        if design == "control":
            continue
        if design not in _DESIGNS:
            raise CliError("INVALID_INPUT",
                           "arm %r has design %r" % (arm["label"], design),
                           "use " + ", ".join(_DESIGNS))
        if design == "rescue" and not arm.get("proxy_of"):
            raise CliError(
                "INVALID_INPUT", "rescue arm %r has no proxy_of" % (arm["label"],),
                "name the different implementation of the same economic quantity, "
                "e.g. \"proxy_of\": \"gross_profit/assets\"")
        effective = {**settings, **(arm.get("settings") or {})}
        moved = sorted(key for key in set(base) | set(effective)
                       if base.get(key) != effective.get(key))
        if design == "parallel":
            if len(moved) != 1:
                raise CliError(
                    "INVALID_INPUT",
                    "parallel arm %r moves %d settings (%s)"
                    % (arm["label"], len(moved), ", ".join(moved) or "none"),
                    "a dose series moves exactly one knob; split the batch instead")
        elif moved:
            raise CliError(
                "INVALID_INPUT",
                "%s arm %r also moves %s" % (design, arm["label"], ", ".join(moved)),
                "only a parallel arm may change a setting; an add/drop/rescue arm "
                "changes the expression and nothing else")
    return {"mode": "controlled", "control": control["label"],
            "designs": {arm["label"]: arm["design"] for arm in arms}}


def _finite_metric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _has_finished_result(record):
    status = str(record.get("status") or "").upper()
    if status not in alpha_registry.FINISHED_STATUSES | {"ACTIVE", "SUBMITTED", "UNSUBMITTED"}:
        return False
    metrics = record.get("metrics") or _train_metrics(record)
    return any(_finite_metric(metrics.get(key)) for key in ("sharpe", "fitness", "returns", "turnover"))


def _probe_comparisons(rows, design):
    """Describe intervention minus control using only explicit train evidence."""
    result = {"scope": "train", "rows": []}
    if not design or design.get("mode") != "controlled":
        result["unavailable_reason"] = ("explore_no_attribution" if design and
                                        design.get("mode") == "explore" else "design_unavailable")
        return result
    controls = [row for row in rows if row.get("label") == design.get("control")]
    control = controls[0] if len(controls) == 1 else None
    for row in rows:
        if row is control:
            continue
        comparison = {"control": design.get("control"), "label": row.get("label"),
                      "design": row.get("design"), "proxy_of": row.get("proxy_of"),
                      "deltas": {}, "unavailable": {}}
        for metric in ("sharpe", "fitness", "turnover", "returns"):
            arm_value = row.get("train_" + metric)
            control_value = control.get("train_" + metric) if control else None
            reasons = []
            if control is None:
                reasons.append("control_unavailable")
            elif not _finite_metric(control_value):
                reasons.append("control_train_metric_unavailable")
            if not _finite_metric(arm_value):
                reasons.append("arm_train_metric_unavailable")
            if reasons:
                comparison["unavailable"][metric] = reasons
            else:
                comparison["deltas"][metric] = arm_value - control_value
        result["rows"].append(comparison)
    return result




def _probe_visible_row(row, show_test):
    validation = {"sharpe", "fitness", "turnover", "test_sharpe", "failed_checks"}
    return {key: value for key, value in row.items() if show_test or key not in validation}

def _turnover_band(expression):
    """``{low, high, verdict, caution}`` from the fields an expression reads."""
    from . import mine

    band = mine.expected_turnover(expression)
    if band is None:
        return None
    return {**band, "verdict": mine.turnover_verdict(band),
            "caution": mine.long_window_caution(expression)}


def _settings_signature(settings):
    """Stable identity of the complete settings, without inventing missing defaults."""
    return json.dumps(settings or {}, sort_keys=True, separators=(",", ":"))


def _normalized_expression(text):
    """Whitespace-free form so a platform expression matches its spec arm."""
    return "".join((text or "").split())


def _recover_probe_rows(session, rows, limit=400, pages=3, max_wait_seconds=300.0,
                        on_progress=None):
    """Read known jobs first; use expression/settings lookup only for unidentified arms."""
    limit = min(100, max(1, limit))
    missing = [row for row in rows if row.get("train_fitness") is None or row.get("metrics_pending")]
    matched, still_missing, ambiguous = {}, [], []
    not_submitted = [row["label"] for row in missing if row.get("submission_attempted") is False]
    deadline = time.monotonic() + max_wait_seconds
    lookup = [row for row in missing if row.get("submission_attempted") is not False
              and not row.get("alpha_id") and not row.get("progress_url")
              and (row.get("settings") or row.get("settings_signature") not in (None, "", "{}"))]
    index = {}
    if lookup:
        for page in range(pages):
            if time.monotonic() >= deadline:
                break
            batch = ml.list_alphas(session, limit=limit, offset=page * limit,
                                   order="-dateCreated", status="UNSUBMITTED")
            for alpha in batch:
                code, _description, _ops = ml.regular_expression(alpha)
                key = _normalized_expression(code)
                if key:
                    index.setdefault(key, []).append({"alpha_id": alpha.get("id"),
                        "signature": _settings_signature(alpha.get("settings"))})
            if len(batch) < limit:
                break
    for row in missing:
        label = row["label"]
        if row.get("submission_attempted") is False or time.monotonic() >= deadline:
            still_missing.append(label)
            continue
        try:
            alpha_id = row.get("alpha_id")
            if not alpha_id and row.get("progress_url"):
                if (row.get("next_poll_at") or 0) > time.time():
                    still_missing.append(label)
                    continue
                progress = ml.poll_simulation(session, row["progress_url"], deadline=deadline)
                row["status"] = progress.get("status")
                alpha_id = progress.get("alpha")
                if not alpha_id:
                    wait = progress.get("next_poll_after_seconds")
                    if wait is not None:
                        row["next_poll_at"] = time.time() + max(0, float(wait))
                    row["error"] = progress.get("message")
                    still_missing.append(label)
                    if on_progress:
                        on_progress(row)
                    continue
                row["alpha_id"] = alpha_id
                row.pop("progress_url", None)
                row.pop("next_poll_at", None)
                if on_progress:
                    on_progress(row)
            if not alpha_id:
                if not row.get("settings") and row.get("settings_signature") in (None, "", "{}"):
                    still_missing.append(label)
                    continue
                wanted = (_settings_signature(row["settings"]) if "settings" in row
                          else row.get("settings_signature") or _settings_signature(None))
                exact = [candidate for candidate in index.get(
                    _normalized_expression(row.get("expression")), [])
                    if candidate["signature"] == wanted]
                if len(exact) != 1:
                    if len(exact) > 1:
                        ambiguous.append({"label": label,
                                          "candidates": [item["alpha_id"] for item in exact]})
                    still_missing.append(label)
                    continue
                alpha_id = exact[0]["alpha_id"]
                row["alpha_id"] = alpha_id
                if on_progress:
                    on_progress(row)
            payload = ml.fetch_alpha_metrics(session, alpha_id, deadline=deadline)
            split = ml.extract_split(payload) or {}
            train = split.get("train") or {}
            if train.get("fitness") is None:
                still_missing.append(label)
            row.update({"status": payload.get("status"), "train_sharpe": train.get("sharpe"),
                        "train_fitness": train.get("fitness"), "train_turnover": train.get("turnover"),
                        "train_returns": train.get("returns"), "error": None,
                        "settings_signature": _settings_signature(row.get("settings"))})
            row.pop("metrics_pending", None)
            raw = {"alpha_id": alpha_id, "status": row["status"],
                   "expression": row.get("expression"), "settings": row.get("settings") or {},
                   "metrics": payload.get("is"), "out_of_sample": split,
                   "checks": ml._extract_checks(payload), "error": None,
                   "submission_attempted": True, "mechanism": label,
                   "decay": (row.get("settings") or {}).get("decay")}
            if train.get("fitness") is not None:
                matched[label] = alpha_id
        except ml.AuthError:
            raise
        except (ml.BrainError, ml.requests.RequestException) as exc:
            row["error"] = str(exc)
            still_missing.append(label)
            if on_progress:
                on_progress(row)
            continue
        if on_progress:
            on_progress(row, raw)
    return {"recovered": len(matched), "still_missing": still_missing,
            "ambiguous": ambiguous, "matched": matched, "not_submitted": not_submitted}


def _load_probe_batch(value):
    path = Path(value).expanduser().resolve()
    visited = set()
    while True:
        if path in visited:
            raise CliError("INVALID_INPUT", "cyclic probe response path")
        visited.add(path)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CliError("INVALID_INPUT", "cannot read batch file %s" % path, str(exc)) from exc
        if isinstance(document, dict) and isinstance(document.get("data"), dict):
            document = document["data"]
        if not isinstance(document, dict):
            raise CliError("INVALID_INPUT", "probe batch must be an object")
        if document.get("path") and not document.get("arms"):
            destination = Path(document["path"]).expanduser()
            path = (destination if destination.is_absolute() else path.parent / destination).resolve()
            continue
        break
    rows = document.get("rows") or []
    if not rows:
        raise CliError("INVALID_INPUT", "%s has no probe rows" % path.name)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise CliError("INVALID_INPUT", "probe rows must be objects")
    saved = document.get("records") or [{} for _ in rows]
    if not isinstance(saved, list) or len(saved) != len(rows) or not all(isinstance(record, dict) for record in saved):
        raise CliError("INVALID_INPUT", "probe batch records do not align with rows")
    if not document.get("arms"):
        arms = []
        for index, (row, record) in enumerate(zip(rows, saved)):
            settings = row.get("settings") or record.get("settings")
            if not settings and row.get("settings_signature"):
                try:
                    settings = json.loads(row["settings_signature"])
                except (ValueError, TypeError) as exc:
                    raise CliError("INVALID_INPUT", "legacy settings signature is not recoverable") from exc
            if settings is not None and not isinstance(settings, dict):
                raise CliError("INVALID_INPUT", "probe arm settings must be an object")
            arms.append({"label": row.get("label", "arm%d" % (index + 1)),
                "expression": row.get("expression") or record.get("expression") or "",
                "settings": settings or {}, "arm_index": index, "ops": row.get("ops"),
                "design": row.get("design"), "proxy_of": row.get("proxy_of")})
        document["arms"] = arms
    arms = document["arms"]
    if (not isinstance(arms, list) or len(arms) != len(rows)
            or not all(isinstance(arm, dict) and isinstance(arm.get("settings"), dict)
                       and isinstance(arm.get("expression"), str) and arm.get("label") for arm in arms)
            or len({arm["label"] for arm in arms}) != len(arms)):
        raise CliError("INVALID_INPUT", "probe batch arm identities are inconsistent")
    records = []
    for index, (arm, row, record) in enumerate(zip(arms, rows, saved)):
        arm["arm_index"] = index
        arm.setdefault("ops", row.get("ops"))
        record = {**record, "expression": arm["expression"], "settings": arm["settings"],
                  "decay": arm["settings"].get("decay"), "mechanism": arm["label"]}
        for key in ("alpha_id", "status", "progress_url", "next_poll_at", "submission_attempted", "error"):
            if key not in record and key in row:
                record[key] = row[key]
        if not record.get("out_of_sample"):
            record["out_of_sample"] = {"train": {key: row.get("train_" + key)
                for key in ("sharpe", "fitness", "turnover", "returns")}}
        if record.get("alpha_id") and not record.get("metrics"):
            record["metrics_pending"] = True
        records.append(record)
    document["records"] = records
    fingerprint = _request_fingerprint(arms)
    if document.get("request_fingerprint") and document["request_fingerprint"] != fingerprint:
        raise CliError("INVALID_INPUT", "saved probe request fingerprint does not match its arms")
    document.setdefault("request_fingerprint", fingerprint)
    document.setdefault("arm_order", [arm["label"] for arm in arms])
    document.setdefault("run_id", None)
    document.setdefault("started_at", None)
    _probe_document_update(document, path)
    return path, document


def _cmd_probe_recover(args):
    path, _document = _load_probe_batch(args.recover)
    protect_output(args, path)
    try:
        with cache_store.file_lock(str(path) + ".lock", timeout=0):
            return _recover_probe_batch(args, path)
    except cache_store.LockTimeout as exc:
        raise CliError("SERVICE_ERROR", "this probe batch is already being updated", str(path)) from exc


def _recover_probe_batch(args, path):
    path, document = _load_probe_batch(path)
    show_test = bool(getattr(args, "show_test", False))
    if show_test and not getattr(args, "selection", None):
        raise CliError("USAGE", "--show-test requires --selection")
    manifest = _selection_call(selection.load, args.selection) if show_test else None
    rows = document["rows"]
    positions = {id(row): index for index, row in enumerate(rows)}

    def save_recovered(row, raw=None):
        index = positions[id(row)]
        record = document["records"][index]
        if raw is not None:
            record.update(raw)
        for key in ("alpha_id", "status", "progress_url", "next_poll_at", "error"):
            if key in row:
                record[key] = row[key]
            elif key in {"progress_url", "next_poll_at"}:
                record.pop(key, None)
        if raw is not None:
            record.pop("metrics_pending", None)
        _probe_document_update(document, path)
        cache_store.write_json_atomic(path, document, indent=1)

    def recover(session):
        outcome = _recover_probe_rows(session, rows, on_progress=save_recovered,
                                     max_wait_seconds=getattr(args, "max_wait_seconds", 300.0))
        _probe_halves(session, document["records"])
        return outcome

    needs_platform = any((row.get("train_fitness") is None or row.get("metrics_pending"))
                         and row.get("submission_attempted") is not False for row in rows)
    needs_halves = any(record.get("alpha_id") and _has_finished_result(record)
        and ((record.get("out_of_sample") or {}).get("test") or {}).get("startDate")
        and (not record.get("half_sharpes") or any(value is None for value in record["half_sharpes"]))
        for record in document["records"])
    outcome = with_session(args, recover) if needs_platform or needs_halves else recover(None)
    _probe_reconcile(document)
    _probe_document_update(document, path)
    cache_store.write_json_atomic(path, document, indent=1)
    result = _probe_response(document, path, show_test, manifest)
    if show_test:
        research_paths.log_view("probe-recover", "test", [row.get("alpha_id") for row in result["rows"]],
                                note={"selection_id": manifest["selection_id"], "batch_path": str(path)})
    return {**result, **outcome}


def _require_spec(args):
    """`probe --recover` needs no spec; every other mode does."""
    spec = getattr(args, "spec", None)
    if spec:
        return spec
    raise CliError("USAGE", "probe needs a spec file",
                   "pass spec.json, or --recover BATCH_JSON to repair an existing batch")


def cmd_fmt(args, _sink):
    """Format alpha expressions: one statement per line, wrapped chains, indented legs.

    Every expression written here lives as a single long line - in a spec file, in a
    commit message, in a probe invocation - while the book's own 55 expressions are
    stored multi-line with aligned continuations. A six-leg composite on one line cannot
    be reviewed, and review is the only defence against silent mistakes.
    """
    target = args.target
    path = Path(target).expanduser()
    if path.suffix == ".json" and path.exists():
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CliError("INVALID_INPUT", "cannot read %s: %s" % (path, exc))
        arms = document.get("arms") or []
        if not arms:
            raise CliError("INVALID_INPUT", "%s has no arms" % path.name)
        changed = 0
        for arm in arms:
            original = arm.get("expression") or ""
            formatted = expression_lint.pretty_expression(original)
            if formatted != original:
                arm["expression"] = formatted
                changed += 1
        if args.write:
            path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        return {"path": str(path), "arms": len(arms), "changed": changed,
                "written": bool(args.write), "formatted": document}

    # Not a spec file: treat the argument as one expression.
    formatted = expression_lint.pretty_expression(target)
    if args.write:
        raise CliError("USAGE", "--write needs a spec file",
                       "pass the path to a JSON spec instead of a bare expression")
    return {"formatted": formatted}


def _novelty_gate(arms, registry_path=None):
    """Annotate resolved requests with exact-result and structural evidence.

    Missing historical settings never imply today's defaults. Structural similarity
    is independent of an exact repeat and cannot block a simulation.
    """
    from . import mechanisms

    registry = registry_path or alpha_registry.default_registry_path()
    try:
        entries = alpha_registry.load_entries(registry)
    except OSError as exc:
        return [{"available": False, "reason": str(exc), "already_simulated": False,
                 "exact_repeat_keys": [], "structural_similarity_keys": []}
                for _ in arms]
    latest = {}
    for entry in entries:
        key = entry.get("alpha_id") or entry.get("key")
        if key:
            latest[key] = entry

    book = [entry for entry in latest.values()
            if str(entry.get("status") or "").upper() == "ACTIVE"]
    book_keys = {}
    for entry in book:
        key = mechanisms.mechanism_key(entry.get("expression"))
        if key:
            book_keys.setdefault(key, []).append(entry.get("key") or entry.get("alpha_id"))

    block_marginals = {}
    snapshot = Path(research_paths.ensure_dir()) / "convergence.json"
    if snapshot.exists():
        try:
            payload = json.loads(snapshot.read_text(encoding="utf-8"))
            block_marginals = {row["block"]: row for row in payload.get("blocks") or []}
        except (OSError, ValueError):
            block_marginals = {}

    exact_results = {}
    for entry in latest.values():
        prior_settings = entry.get("settings")
        if not entry.get("expression") or not _has_finished_result(entry):
            continue
        if not isinstance(prior_settings, dict) or any(
                prior_settings.get(key) is None for key in ml.DEFAULT_SETTINGS):
            continue
        try:
            resolved = ml.resolve_settings(prior_settings)
        except (ValueError, TypeError):
            continue
        identity = (alpha_registry.expression_hash(entry["expression"]),
                    _settings_signature(resolved))
        exact_results.setdefault(identity, []).append(entry.get("key") or entry.get("alpha_id"))
    verdicts = []
    for arm in arms:
        text = arm["expression"]
        key = mechanisms.mechanism_key(text)
        block = key[0] if key else None
        block_row = block_marginals.get(block) or {}
        exact = exact_results.get((alpha_registry.expression_hash(text),
                                   _settings_signature(arm["settings"])), [])
        structural = book_keys.get(key, []) if key else []
        verdicts.append({
            "available": True,
            "block": block,
            "block_median_marginal": block_row.get("median_marginal"),
            "duplicate_skeleton_of": structural[0] if structural else None,
            "structural_similarity_keys": structural,
            "exact_repeat_keys": exact,
            "already_simulated": bool(exact),
        })
    return verdicts


def _prepare_probe(args):
    """Resolve one ordered request used by preview, preflight and execution."""
    spec = _load_probe_spec(_require_spec(args))
    name = getattr(args, "name", None) or spec.get("name") or Path(args.spec).stem
    groups = {}
    arms = []
    for index, original in enumerate(spec["arms"]):
        overrides = merge_settings({**(spec.get("settings") or {}),
                                    **(original.get("settings") or {})}, args)
        entry = {key: value for key, value in original.items()
                 if key not in {"settings", "design", "proxy_of"}}
        try:
            resolved, pairs = config_io.build_config(overrides, [entry])
        except config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc)) from exc
        expression, decay = pairs[0]
        settings = {**resolved, "decay": decay}
        arm = {**original, "expression": expression, "settings": settings,
               "arm_index": index, "ops": expression_lint.estimate_operator_count(expression)}
        arms.append(arm)
        group = groups.setdefault(_settings_signature(settings),
                                  {"settings": settings, "arms": [], "pairs": []})
        group["arms"].append(arm)
        group["pairs"].append((expression, decay))
    design = _probe_design({**spec, "arms": arms}, {})
    gate = _novelty_gate(arms, registry_path=getattr(args, "registry", None))
    return spec, name, groups, arms, design, gate


def _request_fingerprint(arms):
    request = [{"label": arm["label"], "expression": arm["expression"],
                "settings": arm["settings"]} for arm in arms]
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def cmd_probe(args, _sink):
    """Prepare exact requests before checking, executing or continuing a batch."""
    recover = getattr(args, "recover", None)
    resume = getattr(args, "resume", None)
    check = bool(getattr(args, "check", False))
    dry_run = bool(getattr(args, "dry_run", False))
    if sum(bool(value) for value in (recover, resume, check)) > 1 or check and dry_run:
        raise CliError("USAGE", "--check, --recover and --resume are exclusive; --check cannot use --dry-run")
    if getattr(args, "show_test", False) and (not recover or not getattr(args, "selection", None)):
        raise CliError("USAGE", "--show-test requires --recover and --selection; it cannot start simulations")
    if recover:
        return _cmd_probe_recover(args)
    if resume:
        return _cmd_probe_resume(args)
    spec, name, groups, arms, design, gate = _prepare_probe(args)
    target = Path(getattr(args, "batch_path", None) or research_paths.batch_path(name)).expanduser().resolve()
    protect_output(args, target)
    context = {key: spec[key] for key in ("idea", "mechanism", "note") if key in spec}
    base = {"name": name, **context, "path": str(target), "arms": len(arms),
            "groups": len(groups), "design": design, "arm_order": [arm["label"] for arm in arms],
            "request_fingerprint": _request_fingerprint(arms),
            "settings": [group["settings"] for group in groups.values()]}
    gate_blocking = [{"code": "exact_repeat", "label": arm["label"],
                      "evidence_keys": verdict["exact_repeat_keys"]}
                     for arm, verdict in zip(arms, gate)
                     if getattr(args, "strict_gate", False) and verdict.get("already_simulated")]
    if dry_run:
        results = [None] * len(arms)
        reports = []
        for group in groups.values():
            scoped = cached_field_scope(group["settings"])
            report = expression_lint.lint_all(group["pairs"],
                denylist=expression_lint.load_denylist(),
                known_fields=scoped.get("known_fields"), field_types=scoped.get("field_types"),
                strict_fields=scoped.get("known_fields") is not None)
            attach_field_contracts(report, scoped)
            reports.append({**report, "scope": scoped.get("evidence")})
            preview = ml.simulate_alphas(None, group["pairs"], settings=group["settings"],
                                        labels=[arm["label"] for arm in group["arms"]], dry_run=True)
            for record, arm in zip(preview, group["arms"]):
                index = arm["arm_index"]
                results[index] = {**record, "label": arm["label"], "arm_index": index,
                    "design": arm.get("design"), "proxy_of": arm.get("proxy_of"),
                    "settings": arm["settings"], "gate": gate[index],
                    "turnover_band": _turnover_band(arm["expression"])}
        blocking = [item for report in reports for item in report.get("errors", [])]
        return {**base, "dry_run": True, "results": results, "blocking": blocking,
                "gate_blocking": gate_blocking, "would_block": bool(blocking or gate_blocking),
                "preflight_complete": False, "run_id": None, "started_at": None,
                "comparisons": _probe_comparisons(results, design),
                "lint": {"ok": not blocking, "errors": blocking,
                         "warnings": [item for report in reports for item in report.get("warnings", [])],
                         "operator_catalog_checked": False, "field_catalog_checked": False,
                         "hint": "Live catalogs were not checked; run probe --check for exact authenticated preflight.",
                         "groups": reports},
                "next_actions": [{"action": "check", "spec": str(args.spec)}]}
    if gate_blocking:
        raise CliError("INVALID_INPUT", "exact finished requests repeat under --strict-gate",
                       details={"gate_blocking": gate_blocking})
    if check:
        reports = with_session(args, lambda session: [
            _preflight_lint(session, group["pairs"], group["settings"], args)
            for group in groups.values()])
        return {**base, "check": True, "preflight": reports,
                "preflight_complete": all(bool(report) and report.get("ok", False) for report in reports),
                "run_id": None, "started_at": None,
                "next_actions": [{"action": "run", "spec": str(args.spec), "batch_path": str(target)}]}
    try:
        with cache_store.file_lock(str(target) + ".lock", timeout=0):
            if target.exists() and not getattr(args, "force", False):
                raise CliError("INVALID_INPUT", "%s already exists" % target.name,
                               "recover or resume the saved batch, or choose another --batch-path")
            return _run_probe(args, spec, name, groups, design, gate, target)
    except cache_store.LockTimeout as exc:
        raise CliError("SERVICE_ERROR", "this probe batch is already being updated", str(target)) from exc


def _probe_row(record, arm, gate, show_test):
    train = _train_metrics(record)
    row = {"label": arm["label"], "alpha_id": record.get("alpha_id"),
           "design": arm.get("design"), "proxy_of": arm.get("proxy_of"),
           "expression": arm["expression"], "decay": record.get("decay"),
           "settings": dict(record["settings"]),
           "settings_signature": _settings_signature(record["settings"]),
           "status": record.get("status"), "error": record.get("error"),
           "train_sharpe": train.get("sharpe"), "train_fitness": train.get("fitness"),
           "train_turnover": train.get("turnover"), "train_returns": train.get("returns"),
           "train_sharpe_h1": (record.get("half_sharpes") or (None, None))[0],
           "train_sharpe_h2": (record.get("half_sharpes") or (None, None))[1],
           "ops": arm["ops"], "gate": gate}
    for key in ("progress_url", "metrics_pending", "next_poll_at", "submission_attempted"):
        if key in record:
            row[key] = record[key]
    if show_test:
        metrics = record.get("metrics") or {}
        row.update({"sharpe": metrics.get("sharpe"), "fitness": metrics.get("fitness"),
                    "turnover": metrics.get("turnover"),
                    "test_sharpe": ((record.get("out_of_sample") or {}).get("test") or {}).get("sharpe"),
                    "failed_checks": sorted(key for key, body in (record.get("checks") or {}).items()
                                            if isinstance(body, dict) and body.get("result") == "FAIL")})
    return row


def _submission_state(record):
    if record.get("alpha_id") or record.get("progress_url"):
        return "accepted"
    return "not_submitted" if record.get("submission_attempted") is False else "unknown_submission"


def _probe_document_update(document, target):
    arms, records = document["arms"], document["records"]
    rows = [_probe_row(record, arm, (document.get("gate") or [{}] * len(arms))[index], True)
            for index, (record, arm) in enumerate(zip(records, arms))]
    for index, row in enumerate(rows):
        row["arm_index"] = index
        row["submission_state"] = _submission_state(records[index])
        row["cost_evidence"] = costs.evidence(_train_metrics(records[index]), "train", document.get("cost_bps"))
    incomplete = [row["label"] for row in rows if row.get("train_fitness") is None]
    document.update({"rows": rows, "complete": not incomplete, "incomplete": incomplete,
                     "all_empty": bool(rows) and len(incomplete) == len(rows),
                     "comparisons": _probe_comparisons(rows, document.get("design"))})
    document["execution"] = {"total": len(rows), "finished": sum(_has_finished_result(r) for r in records),
        "train_evidence": len(rows) - len(incomplete),
        **{state: sum(row["submission_state"] == state for row in rows)
           for state in ("accepted", "not_submitted", "unknown_submission")}}
    actions = []
    if any(row["submission_state"] != "not_submitted" and row["label"] in incomplete for row in rows):
        actions.append({"action": "recover", "batch_path": str(target), "platform_read_only": True})
    if any(row["submission_state"] == "not_submitted" or row.get("progress_url") for row in rows):
        actions.append({"action": "resume", "batch_path": str(target), "requires_explicit_request": True})
    if document.get("recording_policy", {}).get("enabled") and not document.get("recording"):
        actions.append({"action": "recover", "batch_path": str(target), "reason": "reconcile_recording"})
    document["next_actions"] = actions


def _probe_response(document, path, show_test=False, manifest=None):
    keys = ("name", "idea", "mechanism", "note", "run_id", "started_at", "request_fingerprint",
            "arm_order", "design", "complete", "incomplete", "all_empty", "comparisons",
            "execution", "recording", "recording_policy", "next_actions", "preflight")
    rows = [_probe_visible_row(row, show_test) for row in document["rows"]]
    if manifest is not None:
        sources = {row.get("source_file") for row in manifest["rows"] if row.get("source_file")}
        matching_sources = []
        for source in sources:
            if Path(source).resolve() == Path(path).resolve():
                matching_sources.append(source)
                continue
            try:
                canonical, _saved = _load_probe_batch(source)
            except CliError:
                continue
            if canonical == Path(path).resolve():
                matching_sources.append(source)
        current = [{**row, "source_file": source} for source in matching_sources for row in rows
                   if row.get("alpha_id")]
        rows = _selection_call(selection.ordered_rows, manifest, current)
    response = {key: document[key] for key in keys if key in document}
    response.update({"path": str(path), "rows": rows, "show_test": show_test,
                     "simulated": document["execution"]["accepted"]})
    if document.get("recording"):
        response["registry"] = document["recording"]
    if manifest:
        response["selection_id"] = manifest["selection_id"]
    return response


def _probe_reconcile(document):
    policy = document.get("recording_policy")
    if not policy or not policy.get("enabled"):
        document["recording"] = {"state": "skipped" if policy else "deferred",
            "reason": "recording_disabled" if policy else "legacy_recording_policy_unknown",
            "recorded": 0, "appended": 0}
        return
    document["recording"] = {"state": "reconciled", **alpha_registry.record_simulation(
        document["records"], {}, idea=document.get("idea"), mechanism=document.get("mechanism"),
        note=document.get("note"), registry_path=policy["registry"], source="simulate", batch=document.get("name"))}


def _probe_halves(session, records):
    for record in records:
        if not _has_finished_result(record) or not record.get("alpha_id"):
            continue
        boundary = ((record.get("out_of_sample") or {}).get("test") or {}).get("startDate")
        if not boundary:
            record["half_sharpes"] = [None, None]
            continue
        if record.get("half_sharpes") and all(value is not None for value in record["half_sharpes"]):
            continue
        evidence = {}
        try:
            pnl = ml.fetch_daily_pnl(session, record["alpha_id"],
                cache_dir=Path(research_paths.ensure_dir()) / "pnl", max_age_hours=24, report=evidence)
        except (ml.BrainError, ml.requests.RequestException, OSError) as exc:
            record["half_sharpes"] = [None, None]
            record["half_sharpes_error"] = str(exc)
        else:
            record["pnl_evidence"] = evidence
            record["half_sharpes"] = half_split_sharpes(pnl, boundary) if evidence.get("complete") is True else (None, None)
            record.pop("half_sharpes_error", None)


def _execute_probe(args, document, target, groups):
    records = document["records"]
    state_lock = threading.Lock()

    def save():
        _probe_document_update(document, target)
        cache_store.write_json_atomic(target, document, indent=1)

    def update(index, payload, kind=None):
        merged = {**records[index], **{key: value for key, value in payload.items() if key != "index"}}
        arm = document["arms"][index]
        merged.update({"settings": dict(arm["settings"]), "expression": arm["expression"],
                       "mechanism": arm["label"], "decay": arm["settings"]["decay"]})
        if kind == "submitting":
            merged.update({"submission_attempted": True, "status": "UNKNOWN_SUBMISSION"})
        if merged.get("alpha_id") or merged.get("progress_url"):
            merged["submission_attempted"] = True
        if _has_finished_result(merged):
            for key in ("progress_url", "next_poll_at", "metrics_pending"):
                if key not in payload:
                    merged.pop(key, None)
        records[index] = merged

    def progress(indices, kind, payload):
        index = payload.get("index")
        if kind in {"submitting", "submitted", "pending", "completed", "failed", "deferred"} \
                and isinstance(index, int) and 0 <= index < len(indices):
            with state_lock:
                update(indices[index], payload, kind)
                save()
        note("  %s" % kind)

    def work(session):
        document["preflight"] = [_preflight_lint(session, group["pairs"], group["settings"], args)
                                 for group in groups.values()]
        save()
        deadline = time.monotonic() + getattr(args, "max_wait_seconds", 300.0)
        for group in groups.values():
            indices = [arm["arm_index"] for arm in group["arms"]]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            produced = guarded_sim.simulate([expression for expression, _ in group["pairs"]],
                group["settings"], labels=[arm["label"] for arm in group["arms"]], session=session,
                force=getattr(args, "allow_lint_errors", False),
                initial_records=[records[index] for index in indices], max_wait_seconds=remaining,
                on_event=lambda kind, payload, positions=indices: progress(positions, kind, payload))
            with state_lock:
                for index, record in zip(indices, produced):
                    update(index, record)
                save()
            if any(record.get("progress_url") or record.get("metrics_pending")
                   or record.get("status") in {"PENDING", "QUEUED", "UNKNOWN_SUBMISSION"} for record in produced):
                break
        _probe_halves(session, records)

    save()
    if groups:
        with_session(args, work)
    _probe_reconcile(document)
    save()
    return _probe_response(document, target)


def _run_probe(args, spec, name, groups, design, gate, target):
    arms = sorted([arm for group in groups.values() for arm in group["arms"]], key=lambda arm: arm["arm_index"])
    document = {"format": "probe-batch", "version": 2, "spec": str(args.spec), "name": name,
        **{key: spec[key] for key in ("idea", "mechanism", "note") if key in spec},
        "run_id": str(uuid.uuid4()), "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "request_fingerprint": _request_fingerprint(arms), "arm_order": [arm["label"] for arm in arms],
        "arms": arms, "design": design, "gate": gate, "cost_bps": getattr(args, "cost_bps", None),
        "recording_policy": {"enabled": not getattr(args, "no_record", False),
            "registry": str(Path(getattr(args, "registry", None) or alpha_registry.default_registry_path()).resolve())},
        "records": [{"expression": arm["expression"], "decay": arm["settings"]["decay"],
                     "settings": dict(arm["settings"]), "mechanism": arm["label"], "status": "QUEUED",
                     "alpha_id": None, "metrics": None, "submission_attempted": False} for arm in arms]}
    return _execute_probe(args, document, target, groups)


def _cmd_probe_resume(args):
    changing = ("spec", "name", "batch_path", "registry", "cost_bps", "region", "universe",
                "instrument_type", "neutralization", "decay", "delay", "truncation",
                "pasteurization", "nan_handling", "test_period", "force", "no_record")
    if any(getattr(args, key, None) is not None and getattr(args, key) is not False for key in changing):
        raise CliError("USAGE", "--resume uses saved requests and recording policy; parameters cannot change")
    path, document = _load_probe_batch(args.resume)
    protect_output(args, path)

    def groups_for(saved):
        groups = {}
        for arm, record in zip(saved["arms"], saved["records"]):
            if _submission_state(record) == "unknown_submission" or _has_finished_result(record):
                continue
            if not arm.get("settings") or not arm.get("expression"):
                raise CliError("INVALID_INPUT", "legacy batch lacks exact requests; cannot resume")
            if _submission_state(record) == "not_submitted" and ml.resolve_settings(arm["settings"]) != arm["settings"]:
                raise CliError("INVALID_INPUT", "legacy batch lacks complete resolved settings; cannot start a new POST")
            group = groups.setdefault(_settings_signature(arm["settings"]),
                                      {"settings": arm["settings"], "arms": [], "pairs": []})
            group["arms"].append(arm)
            group["pairs"].append((arm["expression"], arm["settings"].get("decay")))
        return groups

    if getattr(args, "dry_run", False):
        groups = groups_for(document)
        _probe_document_update(document, path)
        return {**_probe_response(document, path), "dry_run": True, "resume": True,
                "continuing": [arm["label"] for group in groups.values() for arm in group["arms"]],
                "blocked_unknown": [arm["label"] for arm, record in zip(document["arms"], document["records"])
                                    if _submission_state(record) == "unknown_submission"]}
    try:
        with cache_store.file_lock(str(path) + ".lock", timeout=0):
            path, document = _load_probe_batch(path)
            return _execute_probe(args, document, path, groups_for(document))
    except cache_store.LockTimeout as exc:
        raise CliError("SERVICE_ERROR", "this probe batch is already being updated", str(path)) from exc


def cmd_simulate(args, sink):
    """Simulate a batch of expressions, or dry-run exact payloads offline."""
    if not 1 <= args.batch_size <= 3:
        raise CliError("INVALID_INPUT", "--batch-size must be between 1 and 3")
    metadata = {}
    file_settings, alphas = collect_alphas(args, metadata=metadata)
    settings = merge_settings(file_settings, args)
    try:
        settings, pairs = config_io.build_config(settings, alphas)
    except config_io.ResearchError as exc:
        raise CliError("INVALID_INPUT", str(exc))
    labels = ml.alpha_labels(alphas, len(pairs))
    families = ml.alpha_families(alphas, len(pairs))
    candidates = ml.alpha_candidate_ids(alphas, len(pairs))

    if args.dry_run:
        records = ml.simulate_alphas(
            None, pairs, settings=settings, dry_run=True, labels=labels,
            families=families, candidates=candidates,
        )
        # Offline lint: catches structural problems and learned rejections. The
        # authoritative operator check needs the catalog, so it runs on `lint`
        # and on a real submission below, not here.
        offline_lint = expression_lint.lint_all(
            pairs,
            denylist=expression_lint.load_denylist(),
        )
        return {
            "status": "DRY_RUN",
            "dry_run": True,
            "settings": settings,
            "summary": {"total": len(records), "payloads": len(records)},
            "lint": {
                "operator_catalog_checked": False,
                "hint": "run `lint` for the authoritative operator check",
                "ok": offline_lint["ok"],
                "errors": offline_lint["errors"],
                "warnings": offline_lint["warnings"],
            },
            "results": records,
            **metadata,
        }

    # Real submission: lint against the operator catalog first. An unknown
    # operator would otherwise burn a submission slot and a polling wait before
    # BRAIN rejected it -- exactly how `regression_neut` cost a round trip.
    with_session(args, lambda s: _preflight_lint(s, pairs, settings, args))

    checkpoint_path, checkpoint = _prepare_simulation_run(
        settings, pairs, args.batch_size, args.checkpoint,
        labels=labels, families=families, candidates=candidates, metadata=metadata,
    )
    on_event = _simulation_checkpoint_callback(checkpoint_path, checkpoint, sink)
    records = with_session(
        args,
        lambda s: ml.simulate_alphas(
            s,
            pairs,
            settings=settings,
            batch_size=args.batch_size,
            on_event=on_event,
            max_wait_seconds=args.max_wait_seconds,
            initial_records=checkpoint["records"],
            labels=labels,
            families=families,
            candidates=candidates,
        ),
    )
    return _finalize_simulation(
        records,
        settings,
        args,
        checkpoint_path,
        checkpoint,
        explicit=bool(args.checkpoint),
    )


def cmd_resume(args, sink):
    """Resume pending, queued or unpolled simulations from an atomic checkpoint file."""
    path = Path(args.checkpoint).expanduser().resolve()
    try:
        with cache_store.file_lock(str(path) + ".lock", timeout=0):
            return _resume_simulation(args, sink, path)
    except cache_store.LockTimeout as exc:
        raise CliError("SERVICE_ERROR", "this simulation checkpoint is already being resumed",
                       str(path)) from exc


def _resume_simulation(args, sink, path):
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError("INVALID_INPUT", "cannot read simulation checkpoint", str(exc))
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != "worldquantuser.simulation-checkpoint"
        or checkpoint.get("version") != 1
        or not isinstance(checkpoint.get("records"), list)
    ):
        raise CliError("INVALID_INPUT", "unsupported or malformed simulation checkpoint")

    records = checkpoint["records"]
    try:
        pairs = [(record["expression"], record.get("decay")) for record in records]
    except (KeyError, TypeError) as exc:
        raise CliError("INVALID_INPUT", "checkpoint is missing alpha inputs", str(exc))
    # Keep the per-expression labels recorded in the checkpoint; a resumed run
    # must not silently drop the mechanism tags gathered before the pause.
    labels = [
        record.get("mechanism") if isinstance(record, dict) else None
        for record in records
    ]
    # Same for the candidate id: an ablation resumed halfway must keep the
    # variant label, or the contribution table comes out unlabelled.
    candidates = [
        record.get("candidate_id") if isinstance(record, dict) else None
        for record in records
    ]
    settings = checkpoint.get("settings") or ml.default_settings()
    batch_size = checkpoint.get("batch_size", 3)
    if not any(
        record.get("progress_url")
        or record.get("metrics_pending")
        or record.get("status") == "QUEUED"
        for record in records
        if isinstance(record, dict)
    ):
        return _finalize_simulation(
            records, settings, args, path, checkpoint, explicit=True
        )
    on_event = _simulation_checkpoint_callback(path, checkpoint, sink)
    records = with_session(
        args,
        lambda s: ml.simulate_alphas(
            s,
            pairs,
            settings=settings,
            batch_size=batch_size,
            on_event=on_event,
            max_wait_seconds=args.max_wait_seconds,
            initial_records=checkpoint["records"],
            labels=labels,
            candidates=candidates,
        ),
    )
    return _finalize_simulation(
        records,
        settings,
        args,
        path,
        checkpoint,
        explicit=True,
    )


def _record_simulation(records, settings, args):
    """Append finished simulation records to the registry.

    The mapping lives in :func:`alpha_registry.record_simulation`, because the
    research template records the same records and had been copying this body by
    hand - or, more often, skipping it.
    """
    return alpha_registry.record_simulation(
        records, settings,
        idea=getattr(args, "idea", None),
        mechanism=getattr(args, "mechanism", None),
        family=getattr(args, "family", None),
        note=getattr(args, "note", None),
        registry_path=getattr(args, "registry", None),
        source="simulate")
