"""Ablation, expression comparison, summary formatting, and FASTEXPR draft formatting."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import threading
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

import simulation_multi
from . import experiments as _exp
from .runtime import CliError

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
    return _exp.train_half_sharpes([pnl[day] for day in days])

def train_half_sharpes(values):
    """Measure two halves of an already sorted, train-only series."""
    if len(values) < 120:
        return None, None
    middle = len(values) // 2
    return (_exp.stats.sharpe(values[:middle], min_points=2),
            _exp.stats.sharpe(values[middle:], min_points=2))

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
    return (record.get("status") or "").upper() in _exp.alpha_registry.FINISHED_STATUSES

def summarize(records):
    """Count outcomes and rank the scored alphas so a caller can pick winners."""
    completed = [r for r in records if _exp._finished(r)]
    failed_statuses = _exp.simulation_slots.TERMINAL_FAILURE_STATUSES | {"FAILED"}
    failed = [r for r in records if (r.get("status") or "").upper() in failed_statuses]
    scored = [r for r in completed if isinstance(r.get("metrics"), dict)]
    ranking = sorted(
        (
            {
                "alpha_id": r.get("alpha_id"),
                "expression": r.get("expression"),
                **{k: r["metrics"].get(k) for k in _exp.METRIC_KEYS if k in r["metrics"]},
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
    record["metrics"] = {k: metrics.get(k) for k in _exp.METRIC_KEYS if k in metrics}

def cmd_ablate(args, _sink):
    """Write a research file holding full / leave-one-out / solo variants.

    Deliberately does not simulate: generating and spending stay separate, so a
    mis-specified ablation costs nothing and can be inspected first.
    """
    expression = args.expr
    decay = args.decay
    if args.file and expression is not None:
        raise CliError("INVALID_INPUT", "ablate accepts either a file or --expr, not both",
                       "pass either a research file, --expr, or --ids")
    if args.ids and (args.file or expression is not None):
        raise CliError("INVALID_INPUT", "--ids cannot be combined with a file or --expr",
                       "pass --ids alone to generate per-alpha ablation files from the ledger")
    settings = {}
    settings_source = "defaults"
    alphas = None
    if args.file:
        try:
            settings, alphas = _exp.config_io.load_research_file(args.file)
        except _exp.config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc),
                           "pass a valid research file with one composite FASTEXPR expression") from exc
        settings_source = args.file
    resolved = _exp.merge_settings(settings, args)
    if _exp.ml.is_python_language(resolved.get("language")):
        raise CliError("INVALID_INPUT", "ablate only supports FASTEXPR",
                       "use a FASTEXPR expression or omit Python language settings")
    if any(getattr(args, flag, None) is not None for flag in (
            "region", "universe", "instrument_type", "neutralization", "decay",
            "delay", "truncation", "pasteurization", "nan_handling", "test_period")):
        settings_source = "flags"
    if alphas is not None:
        try:
            _settings, pairs = _exp.config_io.build_config(resolved, alphas)
        except _exp.config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc),
                           "fix the composite expression or settings in the research file") from exc
        if len(pairs) != 1:
            raise CliError("INVALID_INPUT",
                           "ablate takes exactly one composite expression, found %d" % len(pairs),
                           "split the file, or pass --expr with one composite")
        expression = pairs[0][0]
        if decay is None:
            decay = pairs[0][1]
    elif expression is not None:
        try:
            _settings, pairs = _exp.config_io.build_config(resolved, [expression])
        except _exp.config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc),
                           "check --expr syntax and settings against `python research_cli.py help ablate`") from exc
        expression, decay = pairs[0]

    if args.ids:
        # Batch form. One file per alpha, because `compare --contributions` groups by
        # candidate_id and reads `full` / `loo:<leg>` / `solo:<leg>`; merging six
        # alphas into one file would collide on those names and silently attribute
        # one alpha's leg contributions to another.
        entries = _exp.alpha_registry.load_entries(getattr(args, "registry", None))
        out_dir = Path(args.out_dir or _exp.research_paths.ensure_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        modes = [part.strip() for part in str(args.modes).split(",") if part.strip()]
        done, skipped = [], []
        for raw in str(args.ids).replace(",", " ").split():
            entry = _exp.alpha_registry.resolve(entries, raw)
            if entry is None:
                skipped.append({"alpha_id": raw, "reason": "not in the ledger"})
                continue
            if _exp.ml.is_python_language(_exp.ml.alpha_language(entry)):
                skipped.append({"alpha_id": raw, "reason": "ablate only supports FASTEXPR"})
                continue
            source = entry.get("expression") or ""
            legs_legend = None
            try:
                plan = _exp.ablation.variants(source, modes)
            except _exp.ablation.AblationError as exc:
                point = _exp.alpha_unit.split(source)
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
                for variant in _exp.alpha_unit.derive(point):
                    if variant.rule != "drop_term":
                        continue
                    name = variant.detail or "t%d" % len(plan)
                    plan.append({"candidate_id": "loo:%s" % name,
                                 "expression": variant.expression})
                if not plan:
                    skipped.append({"alpha_id": raw,
                                    "reason": "unit layer derived no ablations"})
                    continue
            alpha_settings = _exp.merge_settings(entry.get("settings") or {}, args)
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
            _exp.note("wrote %s (%d variants)" % (target, len(plan)))
        return {"mode": "ids", "written": done, "skipped": skipped,
                "count": sum(item["variants"] for item in done),
                "next": "simulate each file, then `compare <results> --contributions`"}

    if not expression:
        raise CliError("USAGE", "nothing to ablate",
                       "pass a research file, --expr, or --ids")

    modes = [part.strip() for part in str(args.modes).split(",") if part.strip()]
    try:
        plan = _exp.ablation.variants(expression, modes)
    except _exp.ablation.AblationError as exc:
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
        _exp.note("wrote %s" % args.out)

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
        raise CliError("USAGE", "compare needs files or --match",
                       "pass result JSON paths or --match 'GLOB' under the research directory")
    if explicit.count("-") > 1:
        raise CliError("INVALID_INPUT", "stdin (-) may be read only once",
                       "pass '-' at most once on the command line")
    if latest is not None and (not isinstance(latest, int) or latest < 1):
        raise CliError("USAGE", "--latest must be a positive integer",
                       "pass --latest 1 or greater (or omit N for 1)")
    if latest is not None and "-" in explicit:
        raise CliError("USAGE", "--latest cannot select from stdin",
                       "pass file paths or --match globs when using --latest")
    root = Path(_exp.research_paths.research_dir()).resolve()
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
                    raise CliError("INVALID_INPUT", "cannot read %s: %s" % (path, exc),
                                   "verify the file path or use --match relative to the research directory") from exc
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
        _canonical, payload = _exp._load_probe_batch(destination)
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
        raise CliError("INVALID_INPUT", "%s is not a simulate or probe result document" % path,
                       "pass a result file written by `simulate` or `probe`")

    if isinstance(payload, dict) and isinstance(payload.get("arms"), list) and len(payload["arms"]) == len(records):
        records = [{**record, "role": _exp.alpha_registry.evidence_role(record, arm)}
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
    train = _exp._train_metrics(record)
    halves = record.get("half_sharpes") or (None, None)
    row = {key: record[key] if record.get(key) is not None else display.get(key)
           for key in ("candidate_id", "alpha_id", "decay", "status")}
    row["role"] = _exp.alpha_registry.evidence_role(record, display)
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
        raise CliError("INVALID_INPUT", str(exc),
                       "check the referenced selection, batch, or slot file before retrying") from exc

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
    cost_bps = list(_exp.costs.DEFAULT_COST_BPS) if cost_bps is None else list(cost_bps)
    manifest = None
    if frozen:
        if not getattr(args, "selection", None):
            raise CliError("USAGE", "--frozen requires --selection from a train --freeze",
                           "freeze a training choice first with `compare ... --freeze choice.json`")
        if getattr(args, "freeze", None):
            raise CliError("USAGE", "cannot freeze a validation view",
                           "omit --freeze when running with --frozen")
        if getattr(args, "sort", None) is not None or getattr(args, "min_fitness", None) is not None:
            raise CliError("USAGE", "frozen validation cannot change --sort or --min-fitness",
                           "omit --sort and --min-fitness so the frozen candidate set and order stay immutable")
        manifest = _exp._selection_call(_exp.selection.load, args.selection, source="compare")
        saved_costs = manifest["criteria"].get("cost_bps", list(_exp.costs.DEFAULT_COST_BPS))
        if getattr(args, "cost_bps", None) is not None and cost_bps != saved_costs:
            raise CliError("USAGE", "cost scenarios conflict with the frozen selection",
                           "omit --cost-bps to reuse the frozen scenarios")
        cost_bps = saved_costs
        paths = list((manifest.get("inputs") or {}).get("files") or [])
        if not paths:
            raise CliError("INVALID_INPUT", "selection has no saved compare input files",
                           "re-create the selection from durable result files")
        provenance = (manifest.get("inputs") or {}).get("provenance") or {}
        supplied = [str(Path(path).resolve()) for path in getattr(args, "files", [])]
        if supplied and supplied != paths or getattr(args, "match", None) or getattr(args, "latest", None):
            raise CliError("USAGE", "frozen comparison reuses the saved input files",
                           "omit positional files, --match, and --latest when passing --frozen")
    else:
        if getattr(args, "selection", None):
            raise CliError("USAGE", "--selection requires --frozen",
                           "add --frozen to validate a saved selection")
        paths, provenance = _exp._compare_inputs(args)
    records = []
    inputs = []
    rows = []
    for path in paths:
        try:
            payload = json.loads(_exp.config_io.read_input_text(path))
        except (_exp.config_io.ResearchError, ValueError) as exc:
            if frozen:
                continue
            raise CliError("INVALID_INPUT", "cannot read %s: %s" % (path, exc),
                           "point at a file written by `simulate` or `probe`") from exc
        file_records, aligned = _exp._compare_document(path, payload)
        records.extend(file_records)
        inputs.append((path, payload, file_records))
        rows.extend(_exp._compare_row(record, display, path, frozen) for record, display in aligned)

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
        row["cost_evidence"] = _exp.costs.evidence(
            {**row["train_period"], **{key: row.get("train_" + key) for key in ("margin", "turnover")}},
            "train", cost_bps)
    if manifest:
        rows = _exp._selection_call(_exp.selection.ordered_rows, manifest, [row for row in rows if row.get("alpha_id")])
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
            raise CliError("USAGE", "freeze requires durable input files, not stdin",
                           "save simulation output to a file before freezing with --freeze")
        manifest = _exp._selection_call(_exp.selection.freeze, args.freeze, source="compare", rows=rows,
            criteria={"min_fitness": args.min_fitness, "sort": sort_key, "scope": "train", "cost_bps": cost_bps},
            inputs={"files": paths, "provenance": provenance}, window={"scope": "train"})
        data["selection_path"] = str(Path(args.freeze).resolve())
    if manifest:
        data["selection_id"] = manifest["selection_id"]
    if frozen:
        _exp.research_paths.log_view("compare", "frozen", [row.get("alpha_id") for row in rows],
                                note={"selection_id": manifest["selection_id"], "files": paths})
    if getattr(args, "per_file", False):
        # Variant ids repeat across files, so keep their contributions separate.
        rules = _exp.alpha_unit.load_rules() or {}
        dataset_family = rules.get("datasets") or {}
        field_family = {field: dataset_family.get(dataset, dataset)
                        for field, dataset in (_exp.alpha_unit.load_field_datasets() or {}).items()}
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
                        sibling_payload = json.loads(_exp.config_io.read_input_text(sibling))
                        if isinstance(sibling_payload, dict):
                            legend = sibling_payload.get("legs")
                    except (_exp.config_io.ResearchError, ValueError):
                        pass
            report = _exp.ablation.report(
                file_records, metrics_of=_exp._train_metrics)
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
        data["contributions"] = _exp.ablation.report(
            records, metrics_of=_exp._train_metrics)
        data["contributions"]["view"] = "train"
    if getattr(args, "summary", False):
        data["rows"] = rows[:20]
        data["rows_omitted"] = max(0, len(rows) - 20)
    if getattr(args, "format", None) == "markdown":
        data["table"] = _exp._train_table(data["rows"])

    return data

def _require_spec(args):
    """`probe --recover` needs no spec; every other mode does."""
    spec = getattr(args, "spec", None)
    if spec:
        return spec
    raise CliError("USAGE", "probe needs a spec file",
                   "pass spec.json, or --recover BATCH_JSON to repair an existing batch")

def _format_fastexpr(expression, label):
    from fastexpr_ast import ParseError

    try:
        return _exp.expression_lint.format_fastexpr(expression, strict=True)
    except (ParseError, RecursionError) as exc:
        raise CliError("INVALID_INPUT", f"{label}: {exc}",
                       "fix the FASTEXPR syntax error before formatting") from exc

def _write_formatted_spec(path, source, document):
    from .campaign_runner import campaign_paths

    root = Path(_exp.research_paths.research_dir()).resolve()
    inbox = root / "backlog"
    try:
        with ExitStack() as locks:
            if path.parent in (inbox, inbox / "done", inbox / "failed"):
                locks.enter_context(_exp.cache_store.file_lock(inbox / ".runner.lock", timeout=0))
                if any(saved.exists() for saved in campaign_paths(root, path)):
                    raise CliError("INVALID_INPUT", "this campaign spec already has saved execution evidence",
                                   "preview it or format a separate draft")
            locks.enter_context(_exp.cache_store.file_lock(str(path) + ".lock", timeout=0))
            if path.read_bytes() != source:
                raise CliError("INVALID_INPUT", "the draft spec changed during formatting",
                               "re-run `fmt` after the concurrent edit finishes")
            _exp.cache_store.write_json_atomic(path, document, mode=path.stat().st_mode & 0o777, indent=2)
    except _exp.cache_store.LockTimeout as exc:
        raise CliError("SERVICE_ERROR", "this draft spec or its queue is already being updated",
                       "wait for the active queue/writer to finish before formatting") from exc

def cmd_fmt(args, _sink):
    """Preview FASTEXPR formatting and write only a verified draft spec."""
    target = args.target
    path = Path(target).expanduser()
    if path.suffix.lower() == ".json":
        path = path.resolve()
        _exp.protect_output(args, path)
        try:
            source = path.read_bytes()
            document = json.loads(source)
        except (OSError, ValueError) as exc:
            raise CliError("INVALID_INPUT", "cannot read %s: %s" % (path, exc),
                           "pass a valid JSON draft spec path")
        if not isinstance(document, dict) or {"format", "records", "rows"} & document.keys():
            raise CliError("INVALID_INPUT", "fmt expects a Probe draft spec",
                           "pass a draft spec JSON (not a saved batch or checkpoint)")
        if args.write:
            spec, _, _, parsed_arms, _ = _exp._probe_inputs(SimpleNamespace(spec=str(path)))
        else:
            spec = _exp._load_probe_spec(path)
            parsed_arms = spec["arms"]
        _exp.protect_output(args, *(path.parent / Path(arm["source_file"]).expanduser()
                               for arm in parsed_arms if "source_file" in arm))
        arms = document["arms"]
        skipped = []
        changed = 0
        for arm, parsed in zip(arms, parsed_arms):
            label = parsed["label"]
            settings = {**spec.get("settings", {}), **parsed.get("settings", {})}
            if "source_file" in parsed or _exp.ml.is_python_language(settings.get("language")):
                skipped.append({"label": label, "reason": "python_source"})
                continue
            original = arm["expression"]
            formatted, reason = _exp._format_fastexpr(original, label)
            if reason:
                skipped.append({"label": label, "reason": reason})
            if formatted != original:
                arm["expression"] = formatted
                changed += 1
        written = bool(args.write and changed)
        if written:
            _exp._write_formatted_spec(path, source, document)
        return {"path": str(path), "arms": len(arms), "changed": changed,
                "written": written, "skipped": skipped, "formatted": document}

    if path.suffix.lower() == ".py" and path.exists():
        _exp.protect_output(args, path)
        raise CliError("INVALID_INPUT", "fmt only formats FASTEXPR",
                       "use `lint --file %s --offline` to check Python Alpha syntax" % path)
    if args.write:
        raise CliError("USAGE", "--write needs a spec file",
                       "pass the path to a JSON spec instead of a bare expression")
    formatted, reason = _exp._format_fastexpr(target, "expression")
    result = {"formatted": formatted}
    if reason:
        result["skipped"] = reason
    return result
