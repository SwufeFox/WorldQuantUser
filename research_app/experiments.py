"""Simulation, resumption, labelled probes, ablation and comparison workflows."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import ablation
import alpha_registry
import alpha_unit
import cache_store
import expression_lint
import machine_lib as ml
import research_config as config_io
import research_paths
import simulation_capabilities
import simulation_multi
import simulation_slots
from tools import guarded_sim

from .runtime import CliError, METRIC_KEYS, collect_alphas, merge_settings, note, protect_output, session_reader, with_session
from . import costs, incentives, selection, stats
from .validation import _preflight_lint, attach_field_contracts, cached_field_scope, lint_python_source


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
    return train_half_sharpes([pnl[day] for day in days])


def train_half_sharpes(values):
    """Measure two halves of an already sorted, train-only series."""
    if len(values) < 120:
        return None, None
    middle = len(values) // 2
    return (stats.sharpe(values[:middle], min_points=2),
            stats.sharpe(values[middle:], min_points=2))


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
    failed_statuses = simulation_slots.TERMINAL_FAILURE_STATUSES | {"FAILED"}
    failed = [r for r in records if (r.get("status") or "").upper() in failed_statuses]
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


_CHECKPOINT_EVENTS = {"submitting", "submitted", "pending", "completed", "failed", "deferred"}
_CHECKPOINT_POLL_FIELDS = ("progress_url", "next_poll_at", "poll_after_seconds", "metrics_pending")
_MULTI_CHECKPOINT_EVENTS = {"multi_submitting", "multi_submitted", "multi_pending"}
_MULTI_CHECKPOINT_FIELDS = ("multi_request_id", "multi_member", "multi_size", "multi_parent_url",
                            "multi_child_id", "multi_parent_terminal", "multi_parent_status")


def _merge_checkpoint_record(previous, payload):
    updated = {**previous, **{key: value for key, value in payload.items() if key != "index"}}
    if "multi_request_id" in payload and payload["multi_request_id"] is None:
        for key in simulation_multi.STATE_KEYS:
            updated.pop(key, None)
    if payload.get("alpha_id") or payload.get("status") in simulation_slots.TERMINAL_STATUSES:
        # Once a poll yields an Alpha ID, subsequent metrics reads use that ID.
        # Missing transport markers in the new event supersede the old ones.
        for key in (*_CHECKPOINT_POLL_FIELDS, "simulation_status", "simulation_error"):
            if key not in payload:
                updated.pop(key, None)
    return updated


def _checkpoint_event_updates(kind, payload, size):
    if kind in _MULTI_CHECKPOINT_EVENTS:
        rows = payload["records"]
        indices = [row.get("index") for row in rows]
        if (not indices or any(type(index) is not int or not 0 <= index < size for index in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("invalid native Multi checkpoint event indices")
        return zip(indices, rows)
    index = payload.get("index")
    if kind in _CHECKPOINT_EVENTS and type(index) is int and 0 <= index < size:
        return ((index, payload),)
    return ()


def _simulation_checkpoint_callback(path, checkpoint, sink):
    lock = threading.Lock()

    def on_event(kind, payload):
        with lock:
            records = checkpoint["records"]
            updates = {}
            for index, row in _checkpoint_event_updates(kind, payload, len(records)):
                previous = records[index]
                updated = _merge_checkpoint_record(previous, row)
                updated["role"] = alpha_registry.evidence_role(previous, row)
                if updated != previous:
                    updates[index] = updated
            if updates:
                previous = {index: records[index] for index in updates}
                for index, updated in updates.items():
                    records[index] = updated
                try:
                    _write_simulation_checkpoint(path, checkpoint)
                except BaseException:
                    for index, record in previous.items():
                        records[index] = record
                    raise
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


def _simulation_result(records, settings, summary, checkpoint_path):
    if summary["pending"] or summary["queued"]:
        status = "PENDING"
    elif summary["failed"]:
        status = "PARTIAL" if summary["completed"] else "FAILED"
    else:
        status = "COMPLETE"
    result = {
        "status": status,
        "dry_run": False,
        "settings": settings,
        "summary": summary,
        "results": records,
    }
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        result["checkpoint"] = str(Path(checkpoint_path))
    return result


def _simulation_checkpoint_document(settings, pairs, batch_size, labels, families, candidates, roles):
    # The label columns must be written here, not only in the dry-run branch.
    # These records are handed to simulate_alphas as initial_records, so anything
    # omitted from this dict is lost for the whole real run: mechanism tags
    # arrived empty in the ledger for exactly this reason.
    records = [
        {
            "expression": expression,
            "decay": decay,
            "candidate_id": candidates[index],
            "role": roles[index],
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
    return {
        "format": "worldquantuser.simulation-checkpoint",
        "version": 1,
        "settings": settings,
        "concurrency": batch_size,
        "records": records,
    }


@contextmanager
def _locked_simulation_checkpoint(path, args):
    lock_path = path.with_name(path.name + ".lock")
    protect_output(args, path, lock_path)
    try:
        with cache_store.file_lock(lock_path, timeout=0):
            try:
                yield
            except KeyboardInterrupt as exc:
                if not path.is_file():
                    raise
                raise CliError("INTERRUPTED", "cancelled", "run `resume` with the saved checkpoint",
                               details={"checkpoint": str(path)}) from exc
    except cache_store.LockTimeout as exc:
        raise CliError("SERVICE_ERROR", "this simulation checkpoint is already being updated",
                       details={"checkpoint": str(path)}) from exc


def _finalize_simulation(records, settings, args, checkpoint_path, checkpoint, explicit):
    for record, previous in zip(records, checkpoint["records"]):
        record["role"] = alpha_registry.evidence_role(previous, record)
    checkpoint["records"] = [dict(record) for record in records]
    _write_simulation_checkpoint(checkpoint_path, checkpoint)
    has_unfinished = any(
        record.get("progress_url")
        or record.get("metrics_pending")
        or _native_parent_pending(record)
        or record.get("status") in {"PENDING", "QUEUED", "UNKNOWN_SUBMISSION"}
        for record in records
    )
    discard_checkpoint = not explicit and not has_unfinished

    # The display view honours --sort/--top; the batch, the summary and the ledger
    # must not. Truncating before recording lost every arm outside the top N - and
    # with the checkpoint deleted at the end, those alphas had been paid for and
    # left no trace, which is both wasted quota and a selection bias in the archive.
    display = _sort_simulation_results([dict(record) for record in records], args)
    summary = summarize(records)
    unknown = [index for index, record in enumerate(records) if record.get("status") == "UNKNOWN_SUBMISSION"]
    if summary["completed"] == 0 and summary["total"] and not (
        summary["pending"] or summary["queued"] or unknown
    ):
        if discard_checkpoint:
            Path(checkpoint_path).unlink(missing_ok=True)
        first_error = next(
            (r.get("error") for r in records if r.get("error")), "unknown error"
        )
        raise CliError(
            "SIMULATION_FAILED",
            "all %d simulations failed" % summary["total"],
            str(first_error),
        )
    recording = None
    if getattr(args, "record", False):
        try:
            recording = _record_simulation(records, settings, args)
        except (OSError, cache_store.CacheError) as exc:
            raise CliError("SERVICE_ERROR", "cannot record simulation results: %s" % exc,
                           "run `resume` with the saved checkpoint to retry recording",
                           details={"checkpoint": str(checkpoint_path)}) from exc
    if discard_checkpoint:
        Path(checkpoint_path).unlink(missing_ok=True)
        checkpoint_path = None
    result = _simulation_result(display, settings, summary, checkpoint_path)
    result.update({key: checkpoint[key] for key in ("concurrency", "scheduling", "capability_snapshot", "multi_size")
                   if key in checkpoint})
    result["unknown_submission"] = unknown
    result["results_total"] = len(records)
    for key in ("legs", "legend", "note"):
        value = (checkpoint.get("metadata") or {}).get(key)
        if value is not None:
            result[key] = value
    pending_ids = [record.get("alpha_id") or record.get("candidate_id")
                   for record in records
                   if (record.get("status") or "").upper() in {"PENDING", "QUEUED"}]
    if pending_ids:
        result["pending"] = pending_ids
    if unknown:
        result["status"] = "PENDING"
        result["next"] = "submission outcome is unknown; reconcile it using the saved checkpoint"
    elif pending_ids:
        result["next"] = ("some arms are still running; the checkpoint was kept - "
                          "run `resume %s` before judging them"
                          % (checkpoint_path or "<checkpoint>"))
    if recording is not None:
        result["registry"] = recording
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
    if ml.is_python_language(resolved.get("language")):
        raise CliError("INVALID_INPUT", "ablate only supports FASTEXPR")
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
            if ml.is_python_language(ml.alpha_language(entry)):
                skipped.append({"alpha_id": raw, "reason": "ablate only supports FASTEXPR"})
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

    if isinstance(payload, dict) and isinstance(payload.get("arms"), list) and len(payload["arms"]) == len(records):
        records = [{**record, "role": alpha_registry.evidence_role(record, arm)}
                   for record, arm in zip(records, payload["arms"])]

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
                    "role": row.get("role"),
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
    row["role"] = alpha_registry.evidence_role(record, display)
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



def _train_table(rows):
    columns = (("label", "label"), ("role", "role"), ("state", "status"),
               ("train S", "train_sharpe"), ("train F", "train_fitness"),
               ("turnover", "train_turnover"), ("margin_bps", "margin_bps"),
               ("half 1 S", "train_sharpe_h1"), ("half 2 S", "train_sharpe_h2"),
               ("alpha_id", "alpha_id"))

    def cell(row, key):
        value = row.get(key)
        if key == "role" and value is None:
            value = "unspecified"
        if key == "margin_bps":
            value = (row.get("cost_evidence") or {}).get("margin_bps")
        return "-" if value is None else str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(title for title, _ in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(row, key) for _, key in columns) + " |" for row in rows)
    return "\n".join(lines)


def _probe_summary(response):
    rows = response.get("rows", response.get("results", []))
    fields = ("train_sharpe", "train_fitness", "train_turnover", "train_margin",
              "train_sharpe_h1", "train_sharpe_h2")
    quota = next((report["quota"] for report in reversed(response.get("scheduling") or [])
                  if report.get("quota") is not None), None)
    incentive_rows = [row for row in rows if (row.get("incentives") or {}).get("state") == "reported"]
    return {"path": response.get("path"), "run_id": response.get("run_id"),
            "request_fingerprint": response.get("request_fingerprint"), "view": "train",
            "total_count": (response.get("execution") or {}).get("total", response.get("arms", len(rows))),
            "displayed_count": min(20, len(rows)), "rows_omitted": max(0, len(rows) - 20),
            "execution": response.get("execution"), "recording": response.get("recording"),
            "quota": quota,
            "incentives": [{"label": row.get("label"), "alpha_id": row.get("alpha_id"), **row["incentives"]}
                           for row in incentive_rows[:incentives.SUMMARY_LIMIT]],
            "incentives_reported_count": len(incentive_rows),
            "incentives_unknown_count": len(rows) - len(incentive_rows),
            "incentives_omitted": max(0, len(incentive_rows) - incentives.SUMMARY_LIMIT),
            "roles": {role: sum((row.get("role") or "unspecified") == role for row in rows)
                      for role in ("candidate", "diagnostic", "unspecified")},
            "missing_evidence_count": sum(any(row.get(key) is None for key in fields) for row in rows),
            "missing_evidence": [{"label": row.get("label"),
                                  "fields": [key for key in fields if row.get(key) is None]}
                                 for row in rows[:20] if any(row.get(key) is None for key in fields)],
            "next_actions": response.get("next_actions", []), "table": _train_table(rows[:20])}


def _compact_preflight(report):
    if not isinstance(report, dict):
        return {"state": "unavailable"}
    result = {key: report[key] for key in ("ok", "preflight_complete", "operator_catalog_checked",
              "field_catalog_checked", "capability_id") if key in report}
    for key in ("errors", "warnings", "results", "checked_requests"):
        items = report.get(key)
        if isinstance(items, list):
            result[key + "_count"] = len(items)
            visible = items[:20] if key in ("errors", "warnings") else []
            result[key + "_omitted"] = len(items) - len(visible)
            if visible:
                result[key] = visible
    return result


def _summarize_probe_response(response):
    response["summary"] = _probe_summary(response)
    for key in ("rows", "results", "incomplete", "comparisons", "arm_order", "preflight", "multi_requests"):
        items = response.get(key)
        if isinstance(items, list):
            response[key + "_total"] = len(items)
            response[key + "_omitted"] = max(0, len(items) - 20)
            response[key] = [_compact_preflight(item) for item in items[:20]] if key == "preflight" else items[:20]
    if isinstance(response.get("lint"), dict):
        lint = response["lint"]
        response["lint"] = _compact_preflight(lint)
        response["lint"]["groups_count"] = len(lint.get("groups") or [])
    snapshot = response.get("capability_snapshot")
    if isinstance(snapshot, dict):
        response["capability_snapshot"] = {key: snapshot[key] for key in (
            "state", "account", "source", "as_of", "age_seconds", "capability_id", "concurrency") if key in snapshot}
    return response


def _requested_concurrency(args, saved=None):
    requested = getattr(args, "concurrency", None)
    if requested is None:
        requested = getattr(args, "batch_size", None)
    if requested is None and saved:
        requested = saved.get("concurrency", saved.get("batch_size"))
    if requested is not None and (isinstance(requested, bool) or not isinstance(requested, int) or requested < 1):
        raise CliError("INVALID_INPUT", "--batch-size must be a positive integer")
    return requested


def _multi_options(args, saved=None):
    size = getattr(args, "multi_size", None) if saved is None else saved.get("multi_size")
    if size is None:
        return {}
    if type(size) is not int or not 2 <= size <= simulation_multi.MAX_CHILDREN:
        raise CliError("INVALID_INPUT", f"--multi-size must be an integer from 2 to {simulation_multi.MAX_CHILDREN}")
    return {"multi_size": size}


def _validate_saved_native_owner(session, document):
    if not document.get("multi_size"):
        return
    # Finished parents and singleton tails still bind a batch to its account
    # when the scheduler only receives the remaining queued rows.
    attempted = [record for record in document["records"] if (
        record.get("submission_attempted") is True or record.get("alpha_id")
        or record.get("progress_url") or record.get("multi_request_id"))]
    if not attempted:
        return
    account = simulation_capabilities.account_provenance(session)
    saved = ((document.get("capability_snapshot") or {}).get("account") or {}).get("fingerprint")
    if saved and saved != account.get("fingerprint"):
        raise CliError("INVALID_INPUT", "native Multi checkpoint belongs to another or unidentified account")
    for record in attempted:
        _selection_call(simulation_slots.validate_reservation_owner, record, account)


def _invocation_capabilities(session, args, *, deadline=None):
    snapshot = getattr(args, "capability_snapshot", None)
    current = simulation_capabilities.account_provenance(session)
    prior_account = (snapshot or {}).get("account") or {}
    changed = (current.get("fingerprint") and prior_account.get("fingerprint")
               and current["fingerprint"] != prior_account["fingerprint"])
    if snapshot is None or changed:
        snapshot = simulation_capabilities.fetch(session, refresh=True, deadline=deadline)
        args.capability_snapshot = snapshot
    return snapshot


def _simulation_deadline(args):
    return time.monotonic() + getattr(args, "max_wait_seconds", 300.0)


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

    total_rows = len(rows)
    diagnostic_ids = {row.get("alpha_id") for row in rows
                      if row.get("role") == "diagnostic" and row.get("alpha_id")}
    for row in rows:
        if row.get("alpha_id") in diagnostic_ids:
            row["role"] = "diagnostic"
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
            "total_count": total_rows, "filtered_count": total_rows - len(rows),
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
    if getattr(args, "format", None) == "markdown":
        data["table"] = _train_table(data["rows"])

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
    if "settings" in spec:
        try:
            spec["settings"] = ml.normalize_setting_keys(spec["settings"])
        except ValueError as exc:
            raise CliError("INVALID_INPUT", "spec settings: %s" % exc) from exc
    for index, arm in enumerate(arms, 1):
        if not isinstance(arm, dict):
            raise CliError("INVALID_INPUT", "arm %d has no expression" % index,
                           "an arm is {\"label\": ..., \"expression\": ..., "
                           "\"settings\": {...} (optional)}")
        _selection_call(alpha_registry.evidence_role, arm)
        if "source_file" in arm:
            if "expression" in arm:
                raise CliError("INVALID_INPUT", "arm %d must choose expression or source_file" % index)
            if not isinstance(arm["source_file"], str) or not arm["source_file"].strip() \
                    or arm["source_file"] == "-":
                raise CliError("INVALID_INPUT", "arm %d source_file must be a file path" % index)
            if str(path) == "-":
                raise CliError("INVALID_INPUT", "source_file requires a probe spec file",
                               "use inline expression when reading a spec from stdin")
        elif not isinstance(arm.get("expression"), str) or not arm["expression"].strip():
            raise CliError("INVALID_INPUT", "arm %d has no expression" % index,
                           "provide expression or a Python source_file")
        if "settings" in arm and not isinstance(arm["settings"], dict):
            raise CliError("INVALID_INPUT", "arm %d settings must be an object" % index)
        if "settings" in arm:
            try:
                arm["settings"] = ml.normalize_setting_keys(arm["settings"])
            except ValueError as exc:
                raise CliError("INVALID_INPUT", "arm %d settings: %s" % (index, exc)) from exc
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
        if ml.is_python_language(base.get("language")) != ml.is_python_language(effective.get("language")):
            raise CliError("INVALID_INPUT", "arm %r changes alpha language" % arm["label"],
                           "use design explore or run a separate batch for each language")
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
    """Compare known settings without inventing missing defaults.

    Python uses the effective wire settings. FASTEXPR retains the signature
    used by earlier batches and ledger lookups.
    """
    settings = settings or {}
    if ml.is_python_language(settings.get("language")):
        settings = ml.regular_wire_settings(settings)
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def _normalized_expression(text):
    """Legacy FASTEXPR text key for matching platform expressions to saved arms."""
    return "".join((text or "").split())


def _recovery_expression_key(expression, language):
    if not expression:
        return None
    if ml.is_python_language(language):
        return alpha_registry.expression_hash(expression, language=language)
    return _normalized_expression(expression)


def _probe_recovery_index(session, limit, pages, deadline):
    index = {}
    for page in range(pages):
        if time.monotonic() >= deadline:
            raise ml.RateLimited("probe recovery wait budget expired during alpha lookup")
        batch = ml.list_alphas(session, limit=limit, offset=page * limit,
                               order="-dateCreated", status="UNSUBMITTED", deadline=deadline)
        for alpha in batch:
            code, _description, _ops = ml.regular_expression(alpha)
            key = _recovery_expression_key(code, ml.alpha_language(alpha))
            if key:
                index.setdefault(key, []).append({"alpha_id": alpha.get("id"),
                    "signature": _settings_signature(alpha.get("settings"))})
        if len(batch) < limit:
            break
    return index


def _terminal_probe_without_alpha(record):
    attempted = record.get("submission_attempted")
    if record.get("alpha_id") or attempted is False:
        return False
    status = str(record.get("status") or "").upper()
    # Legacy rows can describe unlocated successful results. An explicit POST
    # attempt distinguishes today's terminal no-ID result from that old layout.
    return (status in simulation_slots.TERMINAL_FAILURE_STATUSES
            or (attempted is True and status in simulation_slots.TERMINAL_STATUSES))


def _probe_needs_recovery(row):
    if _native_parent_pending(row):
        return True
    return (row.get("submission_attempted") is not False
            and (row.get("train_fitness") is None or row.get("metrics_pending"))
            and not _terminal_probe_without_alpha(row))


def _native_parent_pending(record):
    return bool(record.get("multi_parent_url") and not record.get("multi_parent_terminal"))


def _recover_probe_rows(session, rows, limit=400, pages=3, max_wait_seconds=300.0,
                        on_progress=None, *, deadline=None):
    """Read known jobs first; use expression/settings lookup only for unidentified arms."""
    from requests.exceptions import RequestException

    limit = min(100, max(1, limit))
    missing = [row for row in rows if row.get("train_fitness") is None or row.get("metrics_pending")]
    matched, still_missing, ambiguous = {}, set(), []
    not_submitted = [row["label"] for row in missing if row.get("submission_attempted") is False]
    if deadline is None:
        deadline = time.monotonic() + max_wait_seconds
    index = lookup_error = None
    for row in sorted(missing, key=lambda item: not (item.get("alpha_id") or item.get("progress_url"))):
        label = row["label"]
        if (not _probe_needs_recovery(row) or time.monotonic() >= deadline
                or (row.get("next_poll_at") or 0) > time.time()):
            still_missing.add(label)
            continue
        try:
            alpha_id = row.get("alpha_id")
            if not alpha_id and row.get("progress_url"):
                progress = ml.poll_simulation(session, row["progress_url"], deadline=deadline)
                status = str(progress.get("status") or "").upper()
                row["status"] = status or None
                row["submission_attempted"] = True
                alpha_id = progress.get("alpha")
                if not alpha_id:
                    row["error"] = progress.get("message")
                    if _terminal_probe_without_alpha(row):
                        for key in _CHECKPOINT_POLL_FIELDS:
                            row.pop(key, None)
                    else:
                        wait = progress.get("next_poll_after_seconds")
                        if wait is not None:
                            row["next_poll_at"] = time.time() + max(0, float(wait))
                    still_missing.add(label)
                    if on_progress:
                        on_progress(row)
                    continue
                row["alpha_id"] = alpha_id
                row["simulation_status"] = row["status"]
                row["metrics_pending"] = True
                row.pop("progress_url", None)
                row.pop("next_poll_at", None)
                row.pop("poll_after_seconds", None)
                if on_progress:
                    on_progress(row)
            if not alpha_id:
                if not row.get("settings") and row.get("settings_signature") in (None, "", "{}"):
                    still_missing.add(label)
                    continue
                if index is None and lookup_error is None:
                    try:
                        index = _probe_recovery_index(session, limit, pages, deadline)
                    except ml.AuthError:
                        raise
                    except (ml.BrainError, RequestException) as exc:
                        lookup_error = exc
                if lookup_error is not None:
                    row["error"] = str(lookup_error)
                    still_missing.add(label)
                    if on_progress:
                        on_progress(row)
                    continue
                wanted = (_settings_signature(row["settings"]) if "settings" in row
                          else row.get("settings_signature") or _settings_signature(None))
                key = _recovery_expression_key(row.get("expression"), _recovery_language(row, wanted))
                exact = [candidate for candidate in index.get(key, [])
                    if candidate["signature"] == wanted]
                if len(exact) != 1:
                    if len(exact) > 1:
                        ambiguous.append({"label": label,
                                          "candidates": [item["alpha_id"] for item in exact]})
                    still_missing.add(label)
                    continue
                alpha_id = exact[0]["alpha_id"]
                row["alpha_id"] = alpha_id
                row["metrics_pending"] = True
                if on_progress:
                    on_progress(row)
            payload = ml.fetch_alpha_metrics(session, alpha_id, deadline=deadline)
            if ml.defer_alpha_metrics(row, payload):
                still_missing.add(label)
                if on_progress:
                    on_progress(row)
                continue
            split = ml.extract_split(payload) or {}
            train = split.get("train") or {}
            if train.get("fitness") is None:
                still_missing.add(label)
            row.update({"status": payload.get("status") or row.get("simulation_status") or row.get("status"),
                        "train_sharpe": train.get("sharpe"),
                        "train_fitness": train.get("fitness"), "train_turnover": train.get("turnover"),
                        "train_returns": train.get("returns"), "error": None,
                        "settings_signature": _settings_signature(row.get("settings"))})
            row.pop("metrics_pending", None)
            row.pop("next_poll_at", None)
            row.pop("poll_after_seconds", None)
            row.pop("simulation_status", None)
            raw = {"alpha_id": alpha_id, "status": row["status"],
                   "role": row.get("role"),
                   "expression": row.get("expression"), "settings": row.get("settings") or {},
                   "metrics": payload.get("is"), "out_of_sample": split,
                   "checks": ml._extract_checks(payload), "error": None,
                   "submission_attempted": True, "mechanism": label,
                   "decay": (row.get("settings") or {}).get("decay")}
            if train.get("fitness") is not None:
                matched[label] = alpha_id
        except ml.AuthError:
            raise
        except (ml.BrainError, RequestException) as exc:
            row["error"] = str(exc)
            still_missing.add(label)
            if on_progress:
                on_progress(row)
            continue
        if on_progress:
            on_progress(row, raw)
    return {"recovered": len(matched), "still_missing": [row["label"] for row in missing
                                                           if row["label"] in still_missing],
            "ambiguous": ambiguous, "matched": matched, "not_submitted": not_submitted}


def _recovery_language(row, signature):
    settings = row.get("settings")
    if not isinstance(settings, dict):
        try:
            settings = json.loads(signature)
        except (TypeError, ValueError):
            return None
    return settings.get("language") if isinstance(settings, dict) else None


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
                "role": alpha_registry.evidence_role(row, record),
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
        role = _selection_call(alpha_registry.evidence_role, arm, row, record)
        arm["role"] = role
        record = {**record, "expression": arm["expression"], "settings": arm["settings"],
                  "decay": arm["settings"].get("decay"), "mechanism": arm["label"]}
        record["role"] = role
        for key in ("alpha_id", "status", "submission_attempted", "error", "simulation_status",
                    *_CHECKPOINT_POLL_FIELDS, *_MULTI_CHECKPOINT_FIELDS):
            if key not in record and key in row:
                record[key] = row[key]
        if not record.get("out_of_sample"):
            record["out_of_sample"] = {"train": {key: row.get("train_" + key)
                for key in ("sharpe", "fitness", "turnover", "returns")}}
        if record.get("alpha_id") and not record.get("metrics"):
            record["metrics_pending"] = True
        if _terminal_probe_without_alpha(record):
            for key in _CHECKPOINT_POLL_FIELDS:
                record.pop(key, None)
        records.append(record)
    document["records"] = records
    if any(record.get("multi_request_id") for record in records) and not _multi_options(None, document):
        raise CliError("INVALID_INPUT", "native Multi batch lacks its saved request size")
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


def _native_probe_groups(document):
    """Keep every member of any unresolved native parent together on recovery."""
    requests = {record["multi_request_id"] for row, record in zip(document["rows"], document["records"])
                if record.get("multi_request_id") and (_native_parent_pending(record)
                    or (not _has_finished_result(record) and _probe_needs_recovery(row)))}
    groups = {}
    for index, record in enumerate(document["records"]):
        if record.get("multi_request_id") in requests:
            settings = document["arms"][index]["settings"]
            groups.setdefault(_settings_signature(settings), []).append(index)
    return groups.values()


def _recover_probe_batch(args, path):
    deadline = _simulation_deadline(args)
    path, document = _load_probe_batch(path)
    show_test = bool(getattr(args, "show_test", False))
    if show_test and not getattr(args, "selection", None):
        raise CliError("USAGE", "--show-test requires --selection")
    manifest = _selection_call(selection.load, args.selection) if show_test else None
    rows = document["rows"]
    multi = _multi_options(args, document)
    native_indices = list(_native_probe_groups(document))

    def recover(session):
        records = document["records"]
        account = None
        if session is not None:
            _validate_saved_native_owner(session, document)
        unresolved = [record for record in records if record.get("slot_reservation")
                      and not record.get("multi_request_id") and not _terminal_probe_without_alpha(record)]
        if session is not None and unresolved:
            account = simulation_capabilities.account_provenance(session)
            restored = [_selection_call(simulation_slots.restore_record, record, account)
                        for record in unresolved]
            if any(restored):
                _probe_document_update(document, path)
                cache_store.write_json_atomic(path, document, indent=1)
        native_before = {row["label"] for row in document["rows"]
                         if row.get("multi_request_id") and row.get("train_fitness") is None}
        if session is not None:
            progress_lock = threading.Lock()
            for indices in native_indices:
                if time.monotonic() >= deadline:
                    break
                arms = [document["arms"][index] for index in indices]
                settings = arms[0]["settings"]

                def progress(kind, payload):
                    with progress_lock:
                        _save_probe_progress(document, path, indices, kind, payload)

                produced = ml.simulate_alphas(session, [(arm["expression"], arm["settings"]["decay"])
                    for arm in arms], settings=settings, labels=[arm["label"] for arm in arms],
                    initial_records=[records[index] for index in indices], deadline=deadline,
                    concurrency=document.get("concurrency"), on_event=progress, recover_only=True, **multi)
                for index, record in zip(indices, produced):
                    _update_probe_record(document, index, record)
                _save_probe_checkpoint(document, path)
        active_rows = document["rows"]
        for row, record in zip(active_rows, records):
            if "simulation_status" in record:
                row["simulation_status"] = record["simulation_status"]
        positions = {id(row): index for index, row in enumerate(active_rows)}

        def save_recovered(row, raw=None):
            index = positions[id(row)]
            record = records[index]
            if raw is not None:
                record = records[index] = _merge_checkpoint_record(record, raw)
            for key in ("alpha_id", "status", "submission_attempted", "error"):
                if key in row:
                    record[key] = row[key]
            for key in (*_CHECKPOINT_POLL_FIELDS, "simulation_status"):
                if key in row:
                    record[key] = row[key]
                else:
                    record.pop(key, None)
            simulation_slots.reconcile_record(record, account)
            _probe_document_update(document, path)
            cache_store.write_json_atomic(path, document, indent=1)

        outcome = _recover_probe_rows(session, [row for row in active_rows
            if not row.get("multi_request_id") and row["label"] not in native_before],
            on_progress=save_recovered, deadline=deadline)
        for row in active_rows:
            label = row["label"]
            if label in native_before:
                if row.get("train_fitness") is None:
                    outcome["still_missing"].append(label)
                    if row.get("submission_attempted") is False:
                        outcome["not_submitted"].append(label)
                elif row.get("alpha_id"):
                    outcome["matched"][label] = row["alpha_id"]
                    outcome["recovered"] += 1
        _probe_halves(session, records, deadline=deadline)
        return outcome

    now = time.time()
    needs_platform = any(_probe_needs_recovery(row) and (row.get("next_poll_at") or 0) <= now
                         for row in rows)
    needs_halves = any(record.get("alpha_id") and _has_finished_result(record)
        and ((record.get("out_of_sample") or {}).get("test") or {}).get("startDate")
        and (not record.get("half_sharpes") or any(value is None for value in record["half_sharpes"]))
        for record in document["records"])
    outcome = with_session(args, recover, deadline=deadline) if needs_platform or needs_halves else recover(None)
    _probe_reconcile(document)
    _probe_document_update(document, path)
    cache_store.write_json_atomic(path, document, indent=1)
    result = _probe_response(document, path, show_test, manifest, summary=getattr(args, "summary", False))
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
            settings = {**(document.get("settings") or {}), **(arm.get("settings") or {})}
            if "source_file" in arm or ml.is_python_language(settings.get("language")):
                continue
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

    if path.suffix.lower() == ".py" and path.exists():
        raise CliError("INVALID_INPUT", "fmt only formats FASTEXPR")
    # Not a spec file: treat the argument as one expression.
    formatted = expression_lint.pretty_expression(target)
    if args.write:
        raise CliError("USAGE", "--write needs a spec file",
                       "pass the path to a JSON spec instead of a bare expression")
    return {"formatted": formatted}


_PROBE_REFERENCE_LIMIT = 3


def _probe_history_ref(record, *, key=None, train=False):
    """A reusable identity, with at most explicit train statistics."""
    result = {name: record[name] for name in ("alpha_id", "status", "role", "batch", "mechanism")
              if record.get(name) is not None}
    if key is not None:
        result["key"] = key
    if train:
        split = record.get("out_of_sample")
        metrics = split.get("train") if isinstance(split, dict) else None
        result["train"] = {name: value for name, value in metrics.items()
                           if name in METRIC_KEYS and _finite_metric(value)} if isinstance(metrics, dict) else {}
    return result


def _probe_identity(expression, settings):
    language = (settings or {}).get("language")
    return alpha_registry.expression_hash(expression, language=language), _settings_signature(settings)


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

    arm_keys = [None if ml.is_python_language(arm["settings"].get("language"))
                else mechanisms.mechanism_key(arm["expression"]) for arm in arms]
    wanted_mechanisms = {key for key in arm_keys if key}
    wanted_blocks = {key[0] for key in wanted_mechanisms}
    arm_identities = [_probe_identity(arm["expression"], arm["settings"]) for arm in arms]
    wanted_identities = set(arm_identities)
    wanted_settings = {signature for _, signature in arm_identities}

    block_marginals = {}
    snapshot = Path(research_paths.resolve("convergence.json"))
    if snapshot.exists():
        try:
            payload = json.loads(snapshot.read_text(encoding="utf-8"))
            block_marginals = {row["block"]: row for row in payload.get("blocks") or []}
        except (OSError, ValueError):
            block_marginals = {}

    exact_results, structural_results, book_keys = {}, {}, {}
    for entry in latest.values():
        expression = entry.get("expression")
        if not expression:
            continue
        entry_key = entry.get("key") or entry.get("alpha_id")
        prior_settings = entry.get("settings")
        if (not ml.is_python_language(ml.alpha_language(entry))
                and mechanisms.block_of(expression) in wanted_blocks):
            key = mechanisms.mechanism_key(expression)
            if key in wanted_mechanisms:
                structural_results.setdefault(key, []).append(entry)
                if str(entry.get("status") or "").upper() == "ACTIVE":
                    book_keys.setdefault(key, []).append(entry_key)
        if not _has_finished_result(entry):
            continue
        if not isinstance(prior_settings, dict):
            continue
        defaults = (ml.regular_wire_settings({**ml.DEFAULT_SETTINGS, "language": "PYTHON"})
                    if ml.is_python_language(ml.alpha_language(entry)) else ml.DEFAULT_SETTINGS)
        if any(prior_settings.get(key) is None for key in defaults):
            continue
        try:
            resolved = ml.resolve_settings(prior_settings)
        except (ValueError, TypeError):
            continue
        signature = _settings_signature(resolved)
        if signature in wanted_settings:
            identity = _probe_identity(expression, resolved)
            if identity in wanted_identities:
                exact_results.setdefault(identity, []).append(entry)
    verdicts = []
    for key, identity in zip(arm_keys, arm_identities):
        block = key[0] if key else None
        block_row = block_marginals.get(block) or {}
        completed = exact_results.get(identity, [])
        exact = [entry.get("key") or entry.get("alpha_id") for entry in completed]
        exact_keys = set(exact)
        structural = book_keys.get(key, []) if key else []
        similar = [entry for entry in structural_results.get(key, [])
                   if (entry.get("key") or entry.get("alpha_id")) not in exact_keys]
        verdicts.append({
            "available": True,
            "block": block,
            "block_median_marginal": block_row.get("median_marginal"),
            "duplicate_skeleton_of": structural[0] if structural else None,
            "structural_similarity_keys": structural,
            "exact_repeat_keys": exact,
            "already_simulated": bool(exact),
            "history_registry": str(Path(registry).expanduser().resolve()),
            "completed_refs": [_probe_history_ref(entry, key=entry.get("key") or entry.get("alpha_id"),
                                                  train=True)
                               for entry in completed[:_PROBE_REFERENCE_LIMIT]],
            "structural_refs": [_probe_history_ref(entry, key=entry.get("key") or entry.get("alpha_id"))
                                for entry in similar[:_PROBE_REFERENCE_LIMIT]],
            "structural_history_count": len(similar),
        })
    return verdicts


def _probe_checkpoint_refs(target, arms, gate):
    """Look for exact matches at the intended checkpoint without searching the workspace."""
    if not target.is_file():
        return []
    try:
        path, document = _load_probe_batch(target)
    except (CliError, KeyError, TypeError, ValueError):
        return []
    index = {}
    for row, record in zip(document["rows"], document["records"]):
        identity = _probe_identity(row["expression"], row["settings"])
        index.setdefault(identity, []).append((row, record))

    recovery_labels = []
    for arm, verdict in zip(arms, gate):
        identity = _probe_identity(arm["expression"], arm["settings"])
        for row, record in index.get(identity, []):
            ref = {"batch_path": str(path), "run_id": document["run_id"], "label": row["label"],
                   **_probe_history_ref(record, train=True)}
            if _has_finished_result(record) and not row.get("metrics_pending"):
                ref_type = "checkpoint_completed_refs"
            elif _recoverable_probe_reference(row):
                ref_type = "checkpoint_recovery_refs"
            else:
                continue
            refs = verdict.setdefault(ref_type, [])
            if len(refs) < _PROBE_REFERENCE_LIMIT:
                refs.append(ref)
        if verdict.get("checkpoint_recovery_refs"):
            recovery_labels.append(arm["label"])
    if not recovery_labels:
        return []
    return [{"action": "recover", "batch_path": str(path), "platform_read_only": True,
             "matching_arms": recovery_labels,
             "argv": ["python", "research_cli.py", "--research-dir", research_paths.research_dir(),
                      "probe", "--recover", str(path)]}]


def _recoverable_probe_reference(row):
    """Only link to recovery when the saved row identifies an accepted job."""
    if row["submission_state"] != "accepted" or not _probe_needs_recovery(row):
        return False
    if row.get("alpha_id"):
        return True
    try:
        ml._validated_progress_url(row.get("multi_parent_url") or row.get("progress_url"))
    except ml.SimulationError:
        return False
    return True


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
                 if key not in {"settings", "design", "proxy_of", "source_file"}}
        source_file = original.get("source_file")
        if source_file and not ml.is_python_language(overrides.get("language")):
            raise CliError("INVALID_INPUT", "arm %r source_file requires language PYTHON" % original["label"])
        try:
            if source_file:
                source_file = Path(args.spec).expanduser().resolve().parent / Path(source_file).expanduser()
                entry["expression"] = config_io.read_input_text(source_file)
            resolved, pairs = config_io.build_config(overrides, [entry])
        except config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc)) from exc
        expression, decay = pairs[0]
        settings = {**resolved, "decay": decay}
        arm = {**original, "expression": expression, "settings": settings,
               "arm_index": index,
               "ops": None if ml.is_python_language(settings.get("language"))
               else expression_lint.estimate_operator_count(expression)}
        if source_file:
            arm["source_file"] = str(source_file.resolve())
        arms.append(arm)
        group = groups.setdefault(_settings_signature(settings),
                                  {"settings": settings, "arms": [], "pairs": []})
        group["arms"].append(arm)
        group["pairs"].append((expression, decay))
    design = _probe_design({**spec, "arms": arms}, {})
    gate = _novelty_gate(arms, registry_path=getattr(args, "registry", None))
    return spec, name, groups, arms, design, gate


def _request_fingerprint(arms):
    """Hash the ordered, materialized requests saved for a single batch.

    Unlike replay matching, this includes labels and the full resolved
    settings. Source file paths are provenance; saved source is the input.
    """
    request = [{"label": arm["label"], "expression": arm["expression"],
                "settings": arm["settings"]} for arm in arms]
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def cmd_probe(args, sink):
    """Read existing tasks or run a spec through the saved-batch workflow."""
    reference = getattr(args, "from_simulation", None)
    if reference is not None:
        from .simulation_import import read_existing_simulation

        _reject_saved_overrides(args, mode="--from-simulation")
        incompatible = ("spec", "recover", "resume", "dry_run", "check", "show_test", "selection",
                        "name", "batch_path", "registry", "cost_bps", "concurrency", "force",
                        "allow_lint_errors", "strict_gate")
        if any(getattr(args, key, None) is not None and getattr(args, key) is not False
               for key in incompatible):
            raise CliError("USAGE", "--from-simulation is a read-only operation on existing simulations")
        return read_existing_simulation(args, reference)
    _requested_concurrency(args)
    result = _cmd_probe(args, sink)
    if getattr(args, "summary", False):
        _summarize_probe_response(result)
    return result


def _offline_simulation_lint(pairs, settings, scoped=None):
    if ml.is_python_language(settings.get("language")):
        known_fields = (scoped or {}).get("known_fields")
        reports = [lint_python_source(source, known_fields=known_fields, lookback=settings.get("lookback"))
                   for source, _ in pairs]
        return {"ok": all(report["ok"] for report in reports), "language": "PYTHON",
                "errors": [error for report in reports for error in report["errors"]],
                "warnings": [warning for report in reports for warning in report["warnings"]],
                "results": reports}
    if scoped is None:
        return expression_lint.lint_all(pairs, denylist=expression_lint.load_denylist())
    report = expression_lint.lint_all(
        pairs, denylist=expression_lint.load_denylist(),
        known_fields=scoped.get("known_fields"), field_types=scoped.get("field_types"),
        strict_fields=scoped.get("known_fields") is not None)
    attach_field_contracts(report, scoped)
    return report


def _preview_probe_group(args, group, concurrency, multi):
    scheduling = {}
    arms = group["arms"]
    records = ml.simulate_alphas(None, group["pairs"], settings=group["settings"],
        labels=[arm["label"] for arm in arms], dry_run=True, concurrency=concurrency,
        capability_snapshot=getattr(args, "capability_snapshot", None), scheduling_report=scheduling,
        **multi)
    requests = []
    for request in scheduling.pop("multi_requests", []):
        included = [arms[index] for index in request["indices"]]
        requests.append({**request, "indices": [arm["arm_index"] for arm in included],
                         "labels": [arm["label"] for arm in included]})
    return records, scheduling, requests


def _cmd_probe(args, _sink):
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
        _reject_saved_overrides(args)
        return _cmd_probe_recover(args)
    if resume:
        return _cmd_probe_resume(args)
    multi = _multi_options(args)
    spec, name, groups, arms, design, gate = _prepare_probe(args)
    target = Path(getattr(args, "batch_path", None) or research_paths.batch_path(name)).expanduser().resolve()
    source_paths = [Path(arm["source_file"]) for arm in arms if "source_file" in arm]
    if target in source_paths:
        raise CliError("INVALID_INPUT", "batch path would overwrite an input source_file")
    protect_output(args, target, *source_paths)
    context = {key: spec[key] for key in ("idea", "mechanism", "note") if key in spec}
    base = {"name": name, **context, "path": str(target), "arms": len(arms),
            "groups": len(groups), "design": design, "arm_order": [arm["label"] for arm in arms],
            "request_fingerprint": _request_fingerprint(arms),
            "concurrency": _requested_concurrency(args),
            "settings": [group["settings"] for group in groups.values()], **multi}
    gate_blocking = [{"code": "exact_repeat", "label": arm["label"],
                      "evidence_keys": verdict["exact_repeat_keys"]}
                     for arm, verdict in zip(arms, gate)
                     if getattr(args, "strict_gate", False) and verdict.get("already_simulated")]
    if dry_run:
        recovery_actions = _probe_checkpoint_refs(target, arms, gate)
        results = [None] * len(arms)
        reports = []
        requests = []
        for group in groups.values():
            scoped = cached_field_scope(group["settings"])
            report = _offline_simulation_lint(group["pairs"], group["settings"], scoped)
            reports.append({**report, "scope": scoped.get("evidence")})
            preview, scheduling, group_requests = _preview_probe_group(
                args, group, base["concurrency"], multi)
            requests.extend(group_requests)
            reports[-1]["scheduling"] = scheduling
            for record, arm in zip(preview, group["arms"]):
                index = arm["arm_index"]
                results[index] = {**record, "label": arm["label"], "arm_index": index,
                    "role": arm.get("role"),
                    "design": arm.get("design"), "proxy_of": arm.get("proxy_of"),
                    "settings": arm["settings"], "gate": gate[index],
                    "turnover_band": None if ml.is_python_language(arm["settings"].get("language"))
                    else _turnover_band(arm["expression"])}
                if "source_file" in arm:
                    results[index]["source_file"] = arm["source_file"]
        blocking = [item for report in reports for item in report.get("errors", [])]
        return {**base, "dry_run": True, "results": results, "blocking": blocking,
                **({"multi_requests": requests} if multi else {}),
                "gate_blocking": gate_blocking, "would_block": bool(blocking or gate_blocking),
                "preflight_complete": False, "run_id": None, "started_at": None,
                "comparisons": _probe_comparisons(results, design),
                "lint": {"ok": not blocking, "errors": blocking,
                         "warnings": [item for report in reports for item in report.get("warnings", [])],
                         "operator_catalog_checked": False, "field_catalog_checked": False,
                         "hint": "Live catalogs were not checked; run probe --check for exact authenticated preflight.",
                         "groups": reports},
                "next_actions": recovery_actions or [{"action": "check", "spec": str(args.spec), **multi}]}
    if gate_blocking:
        raise CliError("INVALID_INPUT", "exact finished requests repeat under --strict-gate",
                       details={"gate_blocking": gate_blocking})
    if check:
        deadline = _simulation_deadline(args)
        proof_cache = {}

        def check_groups(session):
            base["capability_snapshot"] = _invocation_capabilities(session, args, deadline=deadline)
            reports = []
            for group in groups.values():
                if time.monotonic() >= deadline:
                    break
                reports.append(_preflight_lint(session, group["pairs"], group["settings"], args,
                                               deadline=deadline, proof_cache=proof_cache))
            return reports
        reports = with_session(args, check_groups, deadline=deadline)
        return {**base, "check": True, "preflight": reports,
                "preflight_complete": len(reports) == len(groups)
                and all(bool(report) and report.get("ok", False) for report in reports),
                "run_id": None, "started_at": None,
                "next_actions": [{"action": "run", "spec": str(args.spec), "batch_path": str(target), **multi}]}
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
           "role": alpha_registry.evidence_role(arm, record),
           "design": arm.get("design"), "proxy_of": arm.get("proxy_of"),
           "expression": arm["expression"], "decay": record.get("decay"),
           "settings": dict(record["settings"]),
           "settings_signature": _settings_signature(record["settings"]),
           "status": record.get("status"), "error": record.get("error"),
           "train_sharpe": train.get("sharpe"), "train_fitness": train.get("fitness"),
           "train_turnover": train.get("turnover"), "train_returns": train.get("returns"),
           "train_margin": train.get("margin"),
           "train_sharpe_h1": (record.get("half_sharpes") or (None, None))[0],
           "train_sharpe_h2": (record.get("half_sharpes") or (None, None))[1],
           "ops": arm["ops"], "gate": gate}
    row["incentives"] = incentives.check_summary(record.get("checks"))
    for key in (*_CHECKPOINT_POLL_FIELDS, *_MULTI_CHECKPOINT_FIELDS, "submission_attempted"):
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
    if _terminal_probe_without_alpha(record):
        return "terminal"
    if record.get("alpha_id") or record.get("progress_url") or record.get("multi_parent_url"):
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
           for state in ("accepted", "not_submitted", "unknown_submission", "terminal")}}
    actions = []
    if any(_probe_needs_recovery(row) for row in rows):
        actions.append({"action": "recover", "batch_path": str(target), "platform_read_only": True})
    if any(row["submission_state"] == "not_submitted"
           or (row["submission_state"] == "accepted"
               and (row.get("progress_url") or _native_parent_pending(row)))
           for row in rows):
        actions.append({"action": "resume", "batch_path": str(target), "requires_explicit_request": True})
    if document.get("recording_policy", {}).get("enabled") and not document.get("recording"):
        actions.append({"action": "recover", "batch_path": str(target), "reason": "reconcile_recording"})
    document["next_actions"] = actions
    for action in actions:
        action["argv"] = ["python", "research_cli.py", "probe", "--" + action["action"], str(target)]


def _probe_response(document, path, show_test=False, manifest=None, summary=False):
    keys = ("name", "idea", "mechanism", "note", "run_id", "started_at", "request_fingerprint",
            "arm_order", "design", "complete", "incomplete", "all_empty", "comparisons",
            "execution", "recording", "recording_policy", "next_actions", "preflight",
            "concurrency", "scheduling", "capability_snapshot", "multi_size")
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
    if summary:
        response["summary"] = _probe_summary(response)
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


def _probe_halves(session, records, *, deadline=None):
    for record in records:
        if deadline is not None and time.monotonic() >= deadline:
            break
        if not _has_finished_result(record) or not record.get("alpha_id"):
            continue
        boundary = ((record.get("out_of_sample") or {}).get("test") or {}).get("startDate")
        if not boundary:
            record["half_sharpes"] = [None, None]
            continue
        if record.get("half_sharpes") and all(value is not None for value in record["half_sharpes"]):
            continue
        from requests.exceptions import RequestException

        evidence = {}
        try:
            pnl = ml.fetch_daily_pnl(session, record["alpha_id"],
                cache_dir=Path(research_paths.ensure_dir()) / "pnl", max_age_hours=24, report=evidence,
                deadline=deadline)
        except (ml.BrainError, RequestException, OSError) as exc:
            record["half_sharpes"] = [None, None]
            record["half_sharpes_error"] = str(exc)
        else:
            record["pnl_evidence"] = evidence
            record["half_sharpes"] = half_split_sharpes(pnl, boundary) if evidence.get("complete") is True else (None, None)
            record.pop("half_sharpes_error", None)


def _update_probe_record(document, index, payload, *, submitting=False):
    records = document["records"]
    merged = _merge_checkpoint_record(records[index], payload)
    arm = document["arms"][index]
    merged.update({"settings": dict(arm["settings"]), "expression": arm["expression"],
                   "mechanism": arm["label"], "decay": arm["settings"]["decay"]})
    merged["role"] = alpha_registry.evidence_role(arm, records[index], payload)
    if submitting:
        merged.update({"submission_attempted": True, "status": "UNKNOWN_SUBMISSION"})
    if merged.get("alpha_id") or merged.get("progress_url") or merged.get("multi_parent_url"):
        merged["submission_attempted"] = True
    if _has_finished_result(merged):
        for key in ("progress_url", "next_poll_at", "metrics_pending"):
            if key not in payload:
                merged.pop(key, None)
    records[index] = merged


def _save_probe_checkpoint(document, target):
    _probe_document_update(document, target)
    cache_store.write_json_atomic(target, document, indent=1)


def _save_probe_progress(document, target, indices, kind, payload):
    updates = [(indices[index], row) for index, row in _checkpoint_event_updates(kind, payload, len(indices))]
    if not updates:
        return
    records = document["records"]
    previous = {index: records[index] for index, _row in updates}
    try:
        for index, row in updates:
            _update_probe_record(document, index, row, submitting=kind in {"submitting", "multi_submitting"})
        _save_probe_checkpoint(document, target)
    except BaseException:
        for index, record in previous.items():
            records[index] = record
        _probe_document_update(document, target)
        raise


def _execute_probe(args, document, target, groups):
    deadline = _simulation_deadline(args)
    proof_cache = {}
    records = document["records"]
    state_lock = threading.Lock()
    multi = _multi_options(args, document)

    def save():
        _save_probe_checkpoint(document, target)

    def progress(indices, kind, payload):
        with state_lock:
            _save_probe_progress(document, target, indices, kind, payload)
        note("  %s" % kind)

    def run_group(session, group, *, launching=False, preflight=None, fetch_metrics=True):
        indices = [arm["arm_index"] for arm in group["arms"]]
        scheduling = {}
        document.setdefault("scheduling", []).append(scheduling)
        options = dict(labels=[arm["label"] for arm in group["arms"]],
            concurrency=document["concurrency"], capability_snapshot=document["capability_snapshot"],
            scheduling_report=scheduling, initial_records=[records[index] for index in indices],
            deadline=deadline, fetch_metrics=fetch_metrics,
            on_event=lambda kind, payload: progress(indices, kind, payload), **multi)
        if launching:
            produced = guarded_sim.simulate([expression for expression, _ in group["pairs"]],
                group["settings"], session=session, preflight=preflight,
                force=getattr(args, "allow_lint_errors", False), **options)
        else:
            if multi:
                options["recover_only"] = True
            produced = ml.simulate_alphas(session, group["pairs"], settings=group["settings"], **options)
        with state_lock:
            for index, record in zip(indices, produced):
                _update_probe_record(document, index, record)
            save()
        return produced

    def work(session):
        _validate_saved_native_owner(session, document)
        document["capability_snapshot"] = _invocation_capabilities(session, args, deadline=deadline)
        document.setdefault("preflight", [])
        save()
        def launches(group):
            return any(_submission_state(records[arm["arm_index"]]) == "not_submitted"
                       for arm in group["arms"])

        ordered = sorted(((launches(group), group) for group in groups.values()), key=lambda item: item[0])
        last_launch = len(ordered) - 1 if ordered[-1][0] else -1
        deferred_metrics = []
        for position, (launching, group) in enumerate(ordered):
            if time.monotonic() >= deadline:
                break
            preflight = None
            if launching:
                preflight = _preflight_lint(session, group["pairs"], group["settings"], args,
                                            deadline=deadline, proof_cache=proof_cache)
                document["preflight"].append(preflight)
                save()
            if time.monotonic() >= deadline:
                break
            fetch_metrics = position >= last_launch
            produced = run_group(session, group, launching=launching, preflight=preflight,
                                 fetch_metrics=fetch_metrics)
            if not fetch_metrics:
                waiting = [(arm, pair) for arm, pair, record in zip(group["arms"], group["pairs"], produced)
                           if record.get("alpha_id") and record.get("metrics_pending")
                           and not record.get("progress_url")]
                if waiting:
                    arms, pairs = zip(*waiting)
                    deferred_metrics.append(group if multi else
                        {"settings": group["settings"], "arms": arms, "pairs": pairs})
            if launching and any(
                record.get("progress_url") or record.get("status") in {"QUEUED", "UNKNOWN_SUBMISSION"}
                or (record.get("status") == "PENDING"
                    and not (record.get("alpha_id") and record.get("metrics_pending")))
                for record in produced
            ):
                break
        for group in deferred_metrics:
            if time.monotonic() >= deadline:
                break
            run_group(session, group)
        _probe_halves(session, records, deadline=deadline)

    document["concurrency"] = _requested_concurrency(args, document)
    save()
    if groups:
        with_session(args, work, deadline=deadline)
    _probe_reconcile(document)
    save()
    return _probe_response(document, target, summary=getattr(args, "summary", False))


def _run_probe(args, spec, name, groups, design, gate, target):
    arms = sorted([arm for group in groups.values() for arm in group["arms"]], key=lambda arm: arm["arm_index"])
    document = {"format": "probe-batch", "version": 2, "spec": str(args.spec), "name": name,
        **{key: spec[key] for key in ("idea", "mechanism", "note") if key in spec},
        **_multi_options(args),
        "run_id": str(uuid.uuid4()), "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "request_fingerprint": _request_fingerprint(arms), "arm_order": [arm["label"] for arm in arms],
        "arms": arms, "design": design, "gate": gate, "cost_bps": getattr(args, "cost_bps", None),
        "recording_policy": {"enabled": not getattr(args, "no_record", False),
            "registry": str(Path(getattr(args, "registry", None) or alpha_registry.default_registry_path()).resolve())},
        "records": [{"expression": arm["expression"], "decay": arm["settings"]["decay"],
                     "settings": dict(arm["settings"]), "mechanism": arm["label"], "status": "QUEUED",
                     "role": arm.get("role"),
                     "alpha_id": None, "metrics": None, "submission_attempted": False} for arm in arms]}
    return _execute_probe(args, document, target, groups)


def _reject_saved_overrides(args, mode="resume/recover"):
    changing = ("setting", "region", "universe", "instrument_type", "neutralization",
                "decay", "delay", "truncation", "pasteurization", "nan_handling", "test_period", "multi_size")
    if any(getattr(args, key, None) is not None for key in changing):
        raise CliError("USAGE", f"{mode} uses saved settings; explicit overrides are not allowed")


def _cmd_probe_resume(args):
    _reject_saved_overrides(args)
    changing = ("spec", "name", "batch_path", "registry", "cost_bps", "region", "universe",
                "instrument_type", "neutralization", "decay", "delay", "truncation",
                "pasteurization", "nan_handling", "test_period", "force", "no_record")
    if any(getattr(args, key, None) is not None and getattr(args, key) is not False for key in changing):
        raise CliError("USAGE", "--resume uses saved requests and recording policy; parameters cannot change")
    path, document = _load_probe_batch(args.resume)
    protect_output(args, path)

    def groups_for(saved):
        groups = {}
        native_members = {index for indices in _native_probe_groups(saved) for index in indices}
        for arm, record in zip(saved["arms"], saved["records"]):
            if record.get("multi_request_id"):
                if arm["arm_index"] not in native_members:
                    continue
                state = "native_parent"
            else:
                if _terminal_probe_without_alpha(record) or _has_finished_result(record):
                    continue
                _selection_call(simulation_slots.restore_record, record)
                state = _submission_state(record)
                if state == "unknown_submission":
                    continue
            if not arm.get("settings") or not arm.get("expression"):
                raise CliError("INVALID_INPUT", "legacy batch lacks exact requests; cannot resume")
            if state == "not_submitted" and ml.resolve_settings(arm["settings"]) != arm["settings"]:
                raise CliError("INVALID_INPUT", "legacy batch lacks complete resolved settings; cannot start a new POST")
            group = groups.setdefault((_settings_signature(arm["settings"]), state),
                                      {"settings": arm["settings"], "arms": [], "pairs": []})
            group["arms"].append(arm)
            group["pairs"].append((arm["expression"], arm["settings"].get("decay")))
        return groups

    if getattr(args, "dry_run", False):
        document["concurrency"] = _requested_concurrency(args, document)
        groups = groups_for(document)
        multi = _multi_options(args, document)
        requests = []
        if multi:
            for (_signature, state), group in groups.items():
                if state == "not_submitted":
                    _records, _scheduling, group_requests = _preview_probe_group(
                        args, group, document["concurrency"], multi)
                    requests.extend(group_requests)
        _probe_document_update(document, path)
        return {**_probe_response(document, path, summary=getattr(args, "summary", False)), "dry_run": True, "resume": True,
                **({"multi_requests": requests} if multi else {}),
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
    concurrency = _requested_concurrency(args)
    multi = _multi_options(args)
    metadata = {}
    file_settings, alphas = collect_alphas(args, metadata=metadata)
    settings = merge_settings(file_settings, args)
    paths = [getattr(args, "file", None), *(getattr(args, "files", None) or ())]
    sources = [path for path in paths if path and Path(path).suffix.lower() == ".py"]
    protect_output(args, *sources)
    try:
        settings, pairs = config_io.build_config(settings, alphas)
    except config_io.ResearchError as exc:
        raise CliError("INVALID_INPUT", str(exc))
    labels = ml.alpha_labels(alphas, len(pairs))
    families = ml.alpha_families(alphas, len(pairs))
    candidates = ml.alpha_candidate_ids(alphas, len(pairs))
    roles = [_selection_call(alpha_registry.evidence_role, alpha) if isinstance(alpha, dict) else None
             for alpha in alphas]

    if args.dry_run:
        scheduling = {}
        records = ml.simulate_alphas(
            None, pairs, settings=settings, dry_run=True, labels=labels,
            families=families, candidates=candidates,
            concurrency=concurrency, capability_snapshot=getattr(args, "capability_snapshot", None),
            scheduling_report=scheduling,
            **multi,
        )
        for record, role in zip(records, roles):
            record["role"] = role
        # Offline lint: catches structural problems and learned rejections. The
        # authoritative operator check needs the catalog, so it runs on `lint`
        # and on a real submission below, not here.
        offline_lint = _offline_simulation_lint(pairs, settings)
        return {
            "status": "DRY_RUN",
            "dry_run": True,
            "settings": settings,
            "concurrency": concurrency, "scheduling": scheduling,
            **multi,
            "summary": {"total": len(records), "payloads": len(records)},
            "lint": {
                "operator_catalog_checked": False,
                "hint": ("run `lint` to check Python source and platform fields"
                         if ml.is_python_language(settings.get("language"))
                         else "run `lint` for the authoritative operator check"),
                "ok": offline_lint["ok"],
                "errors": offline_lint["errors"],
                "warnings": offline_lint["warnings"],
            },
            "results": records,
            **metadata,
        }

    checkpoint_path = Path(args.checkpoint or ".brain_checkpoints/simulate-%s.json" % uuid.uuid4().hex)
    checkpoint_path = checkpoint_path.expanduser().resolve()
    with _locked_simulation_checkpoint(checkpoint_path, args):
        if checkpoint_path.exists():
            raise CliError("INVALID_INPUT", "checkpoint already exists; use `resume` or choose another path",
                           str(checkpoint_path))

        deadline = _simulation_deadline(args)
        run = session_reader(args, deadline=deadline)

        def preflight(session):
            snapshot = _invocation_capabilities(session, args, deadline=deadline)
            _preflight_lint(session, pairs, settings, args, deadline=deadline)
            return session, snapshot

        validated_session, snapshot = run(preflight)
        checkpoint = _simulation_checkpoint_document(settings, pairs, concurrency, labels, families, candidates, roles)
        checkpoint.update(multi)
        if metadata:
            checkpoint["metadata"] = metadata
        checkpoint["capability_snapshot"] = snapshot
        checkpoint["scheduling"] = {}
        if getattr(args, "record", False):
            registry = getattr(args, "registry", None) or alpha_registry.default_registry_path()
            args.registry = str(Path(registry).expanduser().resolve())
        checkpoint["recording_policy"] = {key: getattr(args, key, None)
            for key in ("record", "registry", "idea", "mechanism", "family", "note")}
        _write_simulation_checkpoint(checkpoint_path, checkpoint)
        note("simulation checkpoint %s" % checkpoint_path)
        on_event = _simulation_checkpoint_callback(checkpoint_path, checkpoint, sink)

        def simulate(session):
            if session is not validated_session:
                _, checkpoint["capability_snapshot"] = preflight(session)
                _write_simulation_checkpoint(checkpoint_path, checkpoint)
            return ml.simulate_alphas(
                session,
                pairs,
                settings=settings,
                concurrency=concurrency, capability_snapshot=checkpoint["capability_snapshot"],
                scheduling_report=checkpoint["scheduling"],
                on_event=on_event,
                deadline=deadline,
                initial_records=checkpoint["records"],
                labels=labels,
                families=families,
                candidates=candidates,
                **multi,
            )

        records = run(simulate)
        return _finalize_simulation(records, settings, args, checkpoint_path, checkpoint,
                                    explicit=bool(args.checkpoint))


def cmd_resume(args, sink):
    """Resume pending, queued or unpolled simulations from an atomic checkpoint file."""
    _reject_saved_overrides(args)
    path = Path(args.checkpoint).expanduser().resolve()
    with _locked_simulation_checkpoint(path, args):
        return _resume_simulation(args, sink, path)


def _resume_simulation(args, sink, path):
    deadline = _simulation_deadline(args)
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
    multi = _multi_options(args, checkpoint)
    if not multi and any(record.get("multi_request_id") for record in records):
        raise CliError("INVALID_INPUT", "native Multi checkpoint lacks its saved request size")
    concurrency = _requested_concurrency(args, checkpoint)
    checkpoint["concurrency"] = concurrency
    policy = checkpoint.get("recording_policy")
    if isinstance(policy, dict):
        for key, value in policy.items():
            requested = getattr(args, key, None)
            if key == "registry" and requested is not None and value is not None:
                requested = str(Path(requested).expanduser().resolve())
                value = str(Path(value).expanduser().resolve())
            if requested is not None and requested is not False and requested != value:
                raise CliError("USAGE", "resume uses the saved recording policy; %s cannot change" % key)
            setattr(args, key, value)
    for record in records:
        if not record.get("multi_request_id"):
            _selection_call(simulation_slots.restore_record, record)
    if not any(
        record.get("progress_url")
        or record.get("metrics_pending")
        or _native_parent_pending(record)
        or (record.get("multi_request_id") and record.get("status") in {"PENDING", "UNKNOWN_SUBMISSION"})
        or record.get("status") == "QUEUED"
        for record in records
        if isinstance(record, dict)
    ):
        return _finalize_simulation(
            records, settings, args, path, checkpoint, explicit=True
        )
    on_event = _simulation_checkpoint_callback(path, checkpoint, sink)
    checkpoint["scheduling"] = {}
    def resume(session):
        _validate_saved_native_owner(session, checkpoint)
        checkpoint["capability_snapshot"] = _invocation_capabilities(session, args, deadline=deadline)
        return ml.simulate_alphas(session, pairs, settings=settings, concurrency=concurrency,
            capability_snapshot=checkpoint["capability_snapshot"], scheduling_report=checkpoint["scheduling"],
            on_event=on_event, deadline=deadline,
            initial_records=checkpoint["records"], labels=labels, candidates=candidates, **multi)
    records = with_session(args, resume, deadline=deadline)
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
