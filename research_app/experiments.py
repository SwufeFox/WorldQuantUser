"""Simulation, resumption, labelled probes, ablation and comparison workflows."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import ablation
import alpha_registry
import alpha_unit
import expression_lint
import machine_lib as ml
import research_config as config_io
import research_paths
from tools import guarded_sim

from .runtime import CliError, METRIC_KEYS, collect_alphas, merge_settings, note, with_session
from .validation import _preflight_lint


#: Operators per arm above which the report says so. Not a hard limit (the platform's
#: is 64); a research limit. The user's rule from 2026-11-07: many operators means more
#: room to fit, and mixed signals blur what a result actually tests.
SIMPLE_OPERATOR_LIMIT = 8


def half_split_sharpes(pnl, test_start=None):
    """Sharpe of the first and second half of the train window.

    Selection uses the train period, and a train Sharpe can be the average of two very
    different halves. `omEaaMl2` read as 2.11 in train while the following year turned
    negative - a fact only visible afterwards, because that year is test data and may
    not drive selection. The halves stay inside train, so this is legal to look at
    *before* choosing, and it says whether the edge is stable or lives in one regime.

    PnL rather than returns: only the PnL series is exposed, and Sharpe is
    scale-invariant, so a within-alpha comparison is unaffected.
    """
    if not pnl:
        return None, None
    days = sorted(pnl)
    if test_start:
        days = [day for day in days if day < test_start]
    if len(days) < 120:
        return None, None
    middle = len(days) // 2

    def sharpe(subset):
        values = [pnl[day] for day in subset]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        if variance <= 0:
            return None
        return mean / (variance ** 0.5) * (252 ** 0.5)

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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".%s." % path.name, suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(checkpoint, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _simulation_checkpoint_callback(path, checkpoint, sink):
    def on_event(kind, payload):
        if kind in {"submitted", "pending", "completed", "failed", "deferred"}:
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
        or record.get("status") in {"PENDING", "QUEUED"}
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
        summary["pending"] or summary["queued"]
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


def cmd_compare(args, _sink):
    """Merge result documents into one table.

    Variants used to live in separate batch files and were aligned by eye, which
    is how a five-leg version scoring below its four-leg parent went unnoticed.

    Two views, because the default one must not leak the test year. ``train`` ranks
    on the train period and does not print the full-window numbers; ``frozen`` shows
    them for a design that is already fixed.
    """
    frozen = bool(getattr(args, "frozen", False))
    def records_from(path, payload):
        """Records from the two shapes this repository writes.

        `simulate` writes ``{"data": {"results": [...]}}`` and the research template
        writes a bare list, so a batch produced by the template could not be compared
        at all - the file was there, the command refused it, and the reason had
        nothing to do with the batch. Rows already flattened by `compare` are not
        accepted on purpose: they carry ``sharpe`` at the top level, and re-reading
        them as records would fill the table with nulls instead of saying so.
        """
        if isinstance(payload, dict) and "data" in payload:
            return records_from(path, payload.get("data"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            return [item for item in payload["results"] if isinstance(item, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            # `probe` writes the raw simulation records next to its display rows, so a
            # batch can be re-read under --frozen without re-running anything.
            return [item for item in payload["records"] if isinstance(item, dict)]
        raise CliError("INVALID_INPUT",
                       "%s is not a simulate result document" % path,
                       "expected {\"results\": [...]} or a bare list of records; "
                       "`ledger list --format json` rows are not records")

    if sum(str(path) == "-" for path in args.files) > 1:
        raise CliError("INVALID_INPUT", "stdin (-) may be read only once")
    records = []
    inputs = []
    for path in args.files:
        try:
            payload = json.loads(config_io.read_input_text(path))
        except (config_io.ResearchError, ValueError) as exc:
            raise CliError("INVALID_INPUT", "cannot read %s: %s" % (path, exc),
                           "point at a file written by `simulate`") from exc
        file_records = records_from(path, payload)
        records.extend(file_records)
        inputs.append((path, payload, file_records))

    if not records:
        raise CliError("NOT_FOUND", "no result records in the given files",
                       "the file must come from `simulate`")

    rows = []
    for record in records:
        metrics = record.get("metrics") or {}
        checks = record.get("checks") or {}
        failed = sorted(name for name, value in checks.items()
                        if isinstance(value, dict)
                        and value.get("result") in ("FAIL", "ERROR"))
        oos = record.get("out_of_sample") or {}
        train = oos.get("train") or {}
        row = {
            "candidate_id": record.get("candidate_id"),
            "alpha_id": record.get("alpha_id"),
            "decay": record.get("decay"),
            # The exploration columns. `sharpe`/`fitness` below are the full window,
            # which contains the test year, so they are only filled in the frozen view.
            "train_sharpe": train.get("sharpe"),
            "train_fitness": train.get("fitness"),
            "train_turnover": train.get("turnover"),
            "train_returns": train.get("returns"),
            "failed_checks": failed,
        }
        if frozen:
            row.update({"sharpe": metrics.get("sharpe"), "fitness": metrics.get("fitness"),
                        "turnover": metrics.get("turnover"),
                        "returns": metrics.get("returns"),
                        "test_sharpe": (oos.get("test") or {}).get("sharpe")})
        rows.append(row)

    fitness_key = "fitness" if frozen else "train_fitness"
    if args.min_fitness is not None:
        rows = [row for row in rows
                if isinstance(row.get(fitness_key), (int, float))
                and row[fitness_key] >= args.min_fitness]

    key = "alpha_id" if args.sort == "id" else args.sort
    if not frozen and key in {"sharpe", "fitness", "turnover", "returns"}:
        key = "train_" + key
    rows.sort(key=lambda row: (row[key] is None, -(row[key] or 0)
                               if isinstance(row[key], (int, float))
                               else str(row[key])))

    data = {"files": args.files, "count": len(rows), "view": "frozen" if frozen else "train",
            "rows": rows}
    if frozen:
        research_paths.log_view("compare", "frozen", [row.get("alpha_id") for row in rows],
                                note=", ".join(str(path) for path in args.files))
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
                file_records, metrics_of=None if frozen else _train_metrics)
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
        data["per_file_view"] = "frozen" if frozen else "train"
    elif args.contributions:
        # The contribution is a selection decision ("does this leg earn its place"),
        # so in the exploration view it has to be computed from train-period fitness.
        # It used to read the full-window block regardless, which put the test year
        # back into the answer the table had just hidden.
        data["contributions"] = ablation.report(
            records, metrics_of=None if frozen else _train_metrics)
        data["contributions"]["view"] = "frozen" if frozen else "train"

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



def _normalized_expression(text):
    """Whitespace-free form so a platform expression matches its spec arm."""
    return "".join((text or "").split())


def _recover_probe_rows(session, rows, limit=400, pages=3):
    """Fill arms that came back empty by matching them to alphas on the platform.

    A batch can lose arms two ways, both seen on 2026-11-05: the whole run stalls and
    gets killed, or individual submissions are rate-limited after the backoff budget.
    Either way the simulations *did* run (or are still running), so the honest repair
    is to read the platform back rather than spend another twenty minutes re-running.

    Matching is by expression, because that is the only stable key: labels live in the
    spec, ids live in a stdout that a killed process never flushed. ``alpha get``
    already exposes ``out_of_sample.train``, so no simulation is needed.
    """
    missing = [row for row in rows if row.get("train_fitness") is None]
    if not missing:
        return {"recovered": 0, "still_missing": [], "matched": {}}
    index = {}
    for page in range(pages):
        batch = ml.list_alphas(session, limit=limit, offset=page * limit,
                               order="-dateCreated", status="UNSUBMITTED")
        for alpha in batch:
            code, _description, _ops = ml.regular_expression(alpha)
            key = _normalized_expression(code)
            if key and key not in index:
                index[key] = alpha.get("id")
        if len(batch) < limit:
            break
    matched, still_missing = {}, []
    for row in missing:
        key = _normalized_expression(row.get("expression"))
        alpha_id = index.get(key)
        if not alpha_id:
            still_missing.append(row["label"])
            continue
        payload = ml.fetch_alpha_metrics(session, alpha_id)
        train = (ml.extract_split(payload) or {}).get("train") or {}
        if train.get("fitness") is None:
            still_missing.append(row["label"])
            continue
        row.update({"alpha_id": alpha_id, "status": payload.get("status"),
                    "train_sharpe": train.get("sharpe"),
                    "train_fitness": train.get("fitness"),
                    "train_turnover": train.get("turnover"),
                    "train_returns": train.get("returns"),
                    "error": None})
        matched[row["label"]] = alpha_id
    return {"recovered": len(matched), "still_missing": still_missing, "matched": matched}


def _cmd_probe_recover(args):
    """Rebuild a batch file's empty rows from the platform, then re-record them."""
    path = Path(args.recover).expanduser().resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CliError("INVALID_INPUT", "cannot read batch file %s" % path, str(exc))
    rows = document.get("rows") or []
    if not rows:
        raise CliError("INVALID_INPUT", "%s has no rows" % path.name)
    outcome = with_session(args, lambda s: _recover_probe_rows(s, rows))
    for row in rows:
        if row.get("train_fitness") is not None:
            note("%-28s %7s %6s %7s" % (row["label"], row.get("train_sharpe"),
                                        row.get("train_fitness"), row.get("train_turnover")))
    document["rows"] = rows
    document["complete"] = not outcome["still_missing"]
    document["incomplete"] = outcome["still_missing"]
    document["recovered"] = outcome["matched"]
    path.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    note("recovered %d arm(s) from the platform; still missing: %s"
         % (outcome["recovered"], ", ".join(outcome["still_missing"]) or "none"))
    note("wrote %s" % path)
    return {"name": document.get("name"), "path": str(path),
            "recovered": outcome["recovered"],
            "still_missing": outcome["still_missing"], "rows": rows}


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


def cmd_probe(args, _sink):
    """Run a batch of arms from one spec file: lint, simulate, table, record.

    Every experiment in this campaign is those four steps, and each was being
    re-written as a throwaway script. The copies drifted in ways that cost real time:
    one printed its table to stdout and broke the one-JSON-document contract, one
    reused its own output filename and silently overwrote the previous batch, one had
    a quoting bug that only surfaced at import, and none logged progress, so a
    four-minute wait looked like a hang.

    A spec file plus this verb removes all four: expressions are JSON strings so
    quoting cannot break, the output name carries the spec name *and* refuses to
    overwrite, progress goes to stderr, and the table shows **train only** unless
    ``--show-test`` is passed - selecting on the test year turns the only
    out-of-sample evidence into training data.
    """
    if getattr(args, "recover", None):
        return _cmd_probe_recover(args)
    _require_spec(args)

    # Resolve every arm before authenticating or spending quota on the first group.
    spec = _load_probe_spec(args.spec)
    settings = dict(spec.get("settings") or {})
    arms = spec["arms"]
    name = args.name or spec.get("name") or ("stdin" if args.spec == "-" else Path(args.spec).stem)

    raw_groups = {}
    for arm in arms:
        overrides = merge_settings({**settings, **arm.get("settings", {})}, args)
        key = json.dumps(overrides, sort_keys=True)
        raw_groups.setdefault(key, {"settings": overrides, "arms": []})["arms"].append(arm)

    groups = {}
    for group in raw_groups.values():
        entries = [{key: value for key, value in arm.items()
                    if key not in {"settings", "design", "proxy_of"}}
                   for arm in group["arms"]]
        try:
            resolved, pairs = config_io.build_config(group["settings"], entries)
        except config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc)) from exc
        for arm, (expression, decay) in zip(group["arms"], pairs):
            # guarded_sim lints strings, so each group must share the effective
            # decay as well as the other request settings. Explicit defaults and
            # per-arm decay overrides then coalesce into identical payload scopes.
            effective = {**resolved, "decay": decay}
            key = json.dumps(effective, sort_keys=True)
            prepared = groups.setdefault(key, {"settings": effective, "arms": [], "pairs": []})
            prepared["arms"].append({**arm, "expression": expression})
            prepared["pairs"].append((expression, decay))

    # Compare the actual requests, including explicit per-arm decay and CLI flags.
    design_arms = [{**arm, "settings": group["settings"]}
                   for group in groups.values() for arm in group["arms"]]
    design = _probe_design({**spec, "arms": design_arms}, {})
    if design["mode"] == "explore":
        note("design: explore -- no control arm, so no arm is being attributed")

    # The output path and its collision rule are decided before anything is spent:
    # an existing batch file used to raise only after the simulations had run, so a
    # same-day re-run paid full quota and then refused to save the result.
    target = Path(research_paths.batch_path(name))
    if target.exists() and not args.force:
        raise CliError("INVALID_INPUT", "%s already exists" % target.name,
                       "pass --name to label this batch, or --force to overwrite it")

    if args.dry_run:
        results = []
        pairs = []
        for group in groups.values():
            labels = [arm["label"] for arm in group["arms"]]
            preview = ml.simulate_alphas(
                None, group["pairs"], settings=group["settings"],
                dry_run=True, labels=labels)
            for record, label in zip(preview, labels):
                record["label"] = label
                record["settings"] = dict(group["settings"])
            results.extend(preview)
            pairs.extend(group["pairs"])
        report = expression_lint.lint_all(
            pairs, denylist=expression_lint.load_denylist())
        blocking = report["errors"]
        note("dry run: %d arm(s) in %d settings group(s), %d blocking finding(s)"
             % (len(arms), len(groups), len(blocking)))
        return {"dry_run": True, "arms": len(arms), "groups": len(groups),
                "blocking": blocking, "design": design,
                "settings": [group["settings"] for group in groups.values()],
                "results": results,
                "lint": {"operator_catalog_checked": False,
                         "field_catalog_checked": False,
                         "hint": "Live operator and field catalogs were not checked; "
                                 "run `lint` for authoritative catalog checks.",
                         "ok": report["ok"], "errors": report["errors"],
                         "warnings": report["warnings"]}}



    def work(session):
        records = []
        for group in groups.values():
            arm_settings = group["settings"]
            note("simulating %d arm(s) under %s/%s/%s"
                 % (len(group["arms"]), arm_settings.get("region"), arm_settings.get("universe"),
                    arm_settings.get("neutralization")))
            produced = guarded_sim.simulate(
                [expression for expression, _decay in group["pairs"]], arm_settings,
                labels=[arm["label"] for arm in group["arms"]], session=session,
                on_event=lambda kind, payload: note("  %s" % kind))
            for record in produced:
                # The ledger must carry the complete settings this arm actually used.
                record["settings"] = dict(arm_settings)
            records.extend(produced)
        # Half-split stability, from the daily PnL the platform already exposes.
        # Inside the session because this is the only place one is open; the fetch is
        # cached, so a re-run costs nothing and no simulation is involved.
        pnl_dir = Path(research_paths.ensure_dir()) / "pnl"
        for record in records:
            if (record.get("status") or "").upper() not in alpha_registry.FINISHED_STATUSES:
                continue
            alpha_id = record.get("alpha_id")
            if not alpha_id:
                continue
            test_start = ((record.get("out_of_sample") or {}).get("test") or {}).get("startDate")
            try:
                pnl = ml.fetch_daily_pnl(session, alpha_id, cache_dir=pnl_dir,
                                         max_age_hours=24)
            except Exception:  # noqa: BLE001 - a cache miss must not fail the batch
                pnl = None
            record["half_sharpes"] = half_split_sharpes(pnl, test_start)
        return records

    records = with_session(args, work)

    rows = []
    note("")
    header = "%-28s %7s %6s %7s %4s %7s %7s" % ("label", "trS", "trF", "trT", "ops",
                                               "trH1", "trH2")
    if args.show_test:
        header += " %7s %6s %7s %7s" % ("S", "F", "T", "testS")
    note(header)
    for record, arm in zip(records, [a for group in groups.values() for a in group["arms"]]):
        metrics = record.get("metrics") or {}
        split = record.get("out_of_sample") or {}
        train = split.get("train") or {}
        failed = sorted(name for name, body in (record.get("checks") or {}).items()
                        if isinstance(body, dict) and body.get("result") == "FAIL")
        # Exploration view: train period only. The platform's headline metrics use
        # the full window, which *contains* the test year, so ranking on them is
        # ranking on test - hiding one column would not have fixed that.
        row = {"label": arm["label"], "alpha_id": record.get("alpha_id"),
               "design": arm.get("design"),
               "expression": arm["expression"],
               "decay": record.get("decay"), "settings": dict(record["settings"]),
               "status": record.get("status"), "error": record.get("error"),
               "train_sharpe": train.get("sharpe"),
               "train_fitness": train.get("fitness"),
               "train_turnover": train.get("turnover"),
               "train_returns": train.get("returns"),
               "train_sharpe_h1": (record.get("half_sharpes") or (None, None))[0],
               "train_sharpe_h2": (record.get("half_sharpes") or (None, None))[1],
               # Simplicity is a design rule here, not an aesthetic: a factor with many
               # operators has more room to fit, and the one operator-heavy candidate of
               # 2026-11-07 was also the one that failed its mechanism-transfer check.
               "ops": expression_lint.estimate_operator_count(arm["expression"]),
               "failed_checks": failed}
        line = "%-28s %7s %6s %7s %4s %7s %7s" % (
            arm["label"], row["train_sharpe"], row["train_fitness"],
            row["train_turnover"], row["ops"],
            None if row["train_sharpe_h1"] is None else round(row["train_sharpe_h1"], 2),
            None if row["train_sharpe_h2"] is None else round(row["train_sharpe_h2"], 2))
        if args.show_test:
            row.update({"sharpe": metrics.get("sharpe"), "fitness": metrics.get("fitness"),
                        "turnover": metrics.get("turnover"),
                        "test_sharpe": (split.get("test") or {}).get("sharpe")})
            line += " %7s %6s %7s %7s" % (row["sharpe"], row["fitness"],
                                          row["turnover"], row["test_sharpe"])
        rows.append(row)
        note(line + ("  fail=" + ",".join(failed) if failed else ""))

    dense = [row["label"] for row in rows if (row.get("ops") or 0) > SIMPLE_OPERATOR_LIMIT]
    if dense:
        note("")
        note("complexity: %d arm(s) above %d operators: %s"
             % (len(dense), SIMPLE_OPERATOR_LIMIT, ", ".join(dense)))
        note("  when two arms score alike, prefer the simpler one - and do not read a "
             "dense arm as evidence for a mechanism until it transfers")

    # An arm can come back QUEUED with no metrics - the platform rate-limits a batch
    # and `guarded_sim` gives up after its backoff budget. The batch file then looks
    # complete while seven rows are empty, which is how a 34-arm sweep quietly became
    # a 27-arm one. Say it out loud, in the table and in the document.
    incomplete = [row["label"] for row in rows if row.get("train_fitness") is None]
    if incomplete:
        note("")
        note("INCOMPLETE: %d arm(s) produced no train metrics: %s"
             % (len(incomplete), ", ".join(incomplete)))
        note("  the platform rate-limited or never finished them; rerun those arms "
             "in a smaller batch (free) rather than reading this table as complete")

    Path(target).write_text(
        json.dumps({"spec": str(args.spec), "name": name, "rows": rows,
                    "design": design, "records": records,
                    "complete": not incomplete, "incomplete": incomplete},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    if args.show_test:
        research_paths.log_view("probe", "test", [row.get("alpha_id") for row in rows],
                                note=name)
    note("wrote %s" % target)

    summary = {"name": name, "path": str(target), "rows": rows,
               "simulated": len(rows), "show_test": bool(args.show_test),
               "design": design, "complete": not incomplete,
               "incomplete": incomplete}
    if not args.no_record:
        summary["registry"] = alpha_registry.record_simulation(
            records, settings, idea=spec.get("idea"), mechanism=spec.get("mechanism"),
            note=spec.get("note"), registry_path=args.registry, source="simulate")

    return summary


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
