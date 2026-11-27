"""Probe spec parsing, design validation, novelty gates, and checkpoint reference matching."""

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

_PROBE_CONTEXT_FIELDS = ("idea", "mechanism", "note", "quick_source", "breeding_source")

def _load_probe_spec(path):
    """Read and validate a probe spec: settings plus a list of labelled arms."""
    try:
        spec = json.loads(_exp.config_io.read_input_text(path))
    except (_exp.config_io.ResearchError, ValueError) as exc:
        raise CliError("INVALID_INPUT", "cannot read spec %s: %s" % (path, exc),
                       "pass a JSON file shaped like "
                       "{\"name\": ..., \"settings\": {...}, \"arms\": [{\"label\", "
                       "\"expression\"}]}") from exc
    return _exp._normalize_probe_spec(spec, path)

def _normalize_probe_spec(spec, path):
    if not isinstance(spec, dict):
        raise CliError("INVALID_INPUT", "spec must be a JSON object",
                       "see `research_cli.py schema probe`")
    try:
        route = _exp.research_route.normalize_route(spec)
    except ValueError as exc:
        raise CliError("INVALID_INPUT", str(exc),
                       "check spec track, settings, and schema via `python research_cli.py schema probe`") from exc
    if route["track"] == "SPC":
        raise CliError("INVALID_INPUT", "SPC uses recorded model outputs",
                       "use campaign preview or run with an SPC experiment spec")
    spec["track"] = route["track"]
    arms = spec.get("arms")
    if not isinstance(arms, list) or not arms:
        raise CliError("INVALID_INPUT", "spec has no arms",
                       "each arm needs at least {\"expression\": ...}")
    known = {"name", "settings", "arms", "design", "track", *_exp._PROBE_CONTEXT_FIELDS}
    unknown = sorted(set(spec) - known)
    if unknown:
        # A typo here is silent otherwise: `"setting"` instead of `"settings"` would
        # run the whole batch on the defaults and look like a failed hypothesis.
        _exp.note("spec: ignoring unknown key(s) %s (known: %s)"
             % (", ".join(unknown), ", ".join(sorted(known))))
    if "settings" in spec and not isinstance(spec["settings"], dict):
        raise CliError("INVALID_INPUT", "spec settings must be an object",
                       "set \"settings\": {...} in the Probe spec")
    if "settings" in spec:
        try:
            spec["settings"] = _exp.ml.normalize_setting_keys(spec["settings"])
        except ValueError as exc:
            raise CliError("INVALID_INPUT", "spec settings: %s" % exc,
                           "run `python research_cli.py schema probe` for valid setting keys") from exc
    for index, arm in enumerate(arms, 1):
        if not isinstance(arm, dict):
            raise CliError("INVALID_INPUT", "arm %d has no expression" % index,
                           "an arm is {\"label\": ..., \"expression\": ..., "
                           "\"settings\": {...} (optional)}")
        if "track" in arm:
            raise CliError("INVALID_INPUT", "declare the research track on the spec, not on individual arms",
                           "move \"track\" to the top level of the spec")
        if "expr" in arm:
            if "expression" in arm and arm["expression"] != arm["expr"]:
                raise CliError("INVALID_INPUT", "arm %d has conflicting expr and expression" % index,
                               "keep only \"expression\" on the arm")
            arm["expression"] = arm.pop("expr")
        _exp._selection_call(_exp.alpha_registry.evidence_role, arm)
        if "source_file" in arm:
            if "expression" in arm:
                raise CliError("INVALID_INPUT", "arm %d must choose expression or source_file" % index,
                               "provide either \"expression\" or \"source_file\", not both")
            if not isinstance(arm["source_file"], str) or not arm["source_file"].strip() \
                    or arm["source_file"] == "-":
                raise CliError("INVALID_INPUT", "arm %d source_file must be a file path" % index,
                               "set \"source_file\" to a .py path relative to the spec")
            if str(path) == "-":
                raise CliError("INVALID_INPUT", "source_file requires a probe spec file",
                               "use inline expression when reading a spec from stdin")
        elif not isinstance(arm.get("expression"), str) or not arm["expression"].strip():
            raise CliError("INVALID_INPUT", "arm %d has no expression" % index,
                           "provide expression or a Python source_file")
        if "settings" in arm and not isinstance(arm["settings"], dict):
            raise CliError("INVALID_INPUT", "arm %d settings must be an object" % index,
                           "set \"settings\": {...} on the arm")
        if "settings" in arm:
            try:
                arm["settings"] = _exp.ml.normalize_setting_keys(arm["settings"])
            except ValueError as exc:
                raise CliError("INVALID_INPUT", "arm %d settings: %s" % (index, exc),
                               "check setting keys against `python research_cli.py schema probe`") from exc
        arm.setdefault("label", "arm%d" % index)
        if not isinstance(arm["label"], str) or not arm["label"].strip():
            raise CliError("INVALID_INPUT", "arm %d label must be a nonempty string" % index,
                           "give every arm a stable non-empty \"label\"")
    labels = [arm["label"] for arm in arms]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise CliError("INVALID_INPUT", "probe arm labels must be unique",
                       "labels identify saved arms; assign unique labels before the first run",
                       details={"reason": "duplicate_arm_labels", "labels": duplicates})
    return spec

_DESIGNS = ("control", "add", "drop", "parallel", "rescue")

def _probe_design(spec, settings):
    """Validate a control and its interventions before authenticating or spending."""
    mode = spec.get("design")
    if mode is not None and mode not in _exp._DESIGNS + ("explore",):
        raise CliError("INVALID_INPUT", "spec design %r is not a design" % (mode,),
                       "use one of " + ", ".join(_exp._DESIGNS + ("explore",)))
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
        if design not in _exp._DESIGNS:
            raise CliError("INVALID_INPUT",
                           "arm %r has design %r" % (arm["label"], design),
                           "use " + ", ".join(_exp._DESIGNS))
        if design == "rescue" and not arm.get("proxy_of"):
            raise CliError(
                "INVALID_INPUT", "rescue arm %r has no proxy_of" % (arm["label"],),
                "name the different implementation of the same economic quantity, "
                "e.g. \"proxy_of\": \"gross_profit/assets\"")
        effective = {**settings, **(arm.get("settings") or {})}
        if _exp.ml.is_python_language(base.get("language")) != _exp.ml.is_python_language(effective.get("language")):
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

def validate_probe_design(spec, settings):
    """Validate probe control/arm structure against spec and settings constraints."""
    return _exp._probe_design(spec, settings)

def _finite_metric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def finite_metric(value):
    """Return True when value is a finite real number (excluding booleans)."""
    return _exp._finite_metric(value)

def _has_finished_result(record):
    status = str(record.get("status") or "").upper()
    if status not in _exp.alpha_registry.FINISHED_STATUSES | {"ACTIVE", "SUBMITTED", "UNSUBMITTED"}:
        return False
    metrics = record.get("metrics") or _exp._train_metrics(record)
    return any(_exp._finite_metric(metrics.get(key)) for key in ("sharpe", "fitness", "returns", "turnover"))

def _batch_incomplete_labels(rows, records):
    """Scientific evidence completeness is separate from terminal execution."""
    return [row["label"] for row in rows if row.get("train_fitness") is None]

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
            elif not _exp._finite_metric(control_value):
                reasons.append("control_train_metric_unavailable")
            if not _exp._finite_metric(arm_value):
                reasons.append("arm_train_metric_unavailable")
            if reasons:
                comparison["unavailable"][metric] = reasons
            else:
                comparison["deltas"][metric] = arm_value - control_value
        result["rows"].append(comparison)
    return result

def _probe_visible_row(row, show_test):
    validation = {"sharpe", "fitness", "turnover", "test_sharpe", "failed_checks", "classification_ids"}
    return {key: value for key, value in row.items() if show_test or key not in validation}

def _turnover_band(expression, *, catalog=None, stats=None):
    """``{low, high, verdict, caution}`` from the fields an expression reads."""
    from . import mine

    band = mine.expected_turnover(expression, catalog=catalog, stats=stats)
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
    if _exp.ml.is_python_language(settings.get("language")):
        settings = _exp.ml.regular_wire_settings(settings)
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))

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
                           if name in _exp.METRIC_KEYS and _exp._finite_metric(value)} if isinstance(metrics, dict) else {}
    return result

def _probe_identity(expression, settings):
    language = (settings or {}).get("language")
    return _exp.alpha_registry.expression_hash(expression, language=language), _exp._settings_signature(settings)

def probe_identity(expression, settings):
    """Return ``(expression_hash, settings_signature)`` identifying a probe arm."""
    return _exp._probe_identity(expression, settings)

def _scan_novelty_history(registry, arm_keys, arm_identities):
    """Scan the authoritative ledger when the derived index is unavailable."""
    from . import mechanisms

    latest = {}
    for entry in _exp.alpha_registry.load_entries(registry):
        key = entry.get("alpha_id") or entry.get("key")
        if key:
            latest[key] = entry

    wanted_mechanisms = {key for key in arm_keys if key}
    wanted_blocks = {key[0] for key in wanted_mechanisms}
    wanted_identities = set(arm_identities)
    wanted_settings = {signature for _, signature in arm_identities}
    exact_results, structural_results, book_keys = {}, {}, {}
    for entry in latest.values():
        expression = entry.get("expression")
        if not expression:
            continue
        entry_key = entry.get("key") or entry.get("alpha_id")
        prior_settings = entry.get("settings")
        if (not _exp.ml.is_python_language(_exp.ml.alpha_language(entry))
                and mechanisms.block_of(expression) in wanted_blocks):
            key = mechanisms.mechanism_key(expression)
            if key in wanted_mechanisms:
                structural_results.setdefault(key, []).append(entry)
                if str(entry.get("status") or "").upper() == "ACTIVE":
                    book_keys.setdefault(key, []).append(entry_key)
        if not _exp._has_finished_result(entry):
            continue
        if not isinstance(prior_settings, dict):
            continue
        defaults = (_exp.ml.regular_wire_settings({**_exp.ml.DEFAULT_SETTINGS, "language": "PYTHON"})
                    if _exp.ml.is_python_language(_exp.ml.alpha_language(entry)) else _exp.ml.DEFAULT_SETTINGS)
        if any(prior_settings.get(key) is None for key in defaults):
            continue
        try:
            resolved = _exp.ml.resolve_settings(prior_settings)
        except (ValueError, TypeError):
            continue
        signature = _exp._settings_signature(resolved)
        if signature in wanted_settings:
            identity = _exp._probe_identity(expression, resolved)
            if identity in wanted_identities:
                exact_results.setdefault(identity, []).append(entry)
    return exact_results, structural_results, book_keys

def _novelty_gate(arms, registry_path=None):
    """Annotate resolved requests with exact-result and structural evidence.

    Missing historical settings never imply today's defaults. Structural similarity
    is independent of an exact repeat and cannot block a simulation.
    """
    from . import mechanisms

    registry = registry_path or _exp.alpha_registry.default_registry_path()
    arm_keys = [None if _exp.ml.is_python_language(arm["settings"].get("language"))
                else mechanisms.mechanism_key(arm["expression"]) for arm in arms]
    arm_identities = [_exp._probe_identity(arm["expression"], arm["settings"]) for arm in arms]
    history = {}
    try:
        with _exp.alpha_registry.indexed_registry(registry) as index:
            if index is None:
                exact_rows, structural_rows, book_keys = _exp._scan_novelty_history(
                    registry, arm_keys, arm_identities)
                for key, identity in zip(arm_keys, arm_identities):
                    completed = exact_rows.get(identity, [])
                    exact = [row.get("key") or row.get("alpha_id") for row in completed]
                    exact_keys = set(exact)
                    similar = [row for row in structural_rows.get(key, [])
                               if (row.get("key") or row.get("alpha_id")) not in exact_keys]
                    history[key, identity] = (exact, completed[:_exp._PROBE_REFERENCE_LIMIT],
                                              len(similar), book_keys.get(key, []),
                                              similar[:_exp._PROBE_REFERENCE_LIMIT])
            else:
                for key, identity in dict.fromkeys(zip(arm_keys, arm_identities)):
                    exact, completed = index.exact_history(identity, limit=_exp._PROBE_REFERENCE_LIMIT)
                    count, structural, similar = index.mechanism_history(
                        key, exclude_keys=exact, limit=_exp._PROBE_REFERENCE_LIMIT)
                    history[key, identity] = (exact, completed, count, structural, similar)
    except OSError as exc:
        return [{"available": False, "reason": str(exc), "already_simulated": False,
                 "exact_repeat_keys": [], "structural_similarity_keys": []}
                for _ in arms]

    block_marginals = {}
    snapshot = Path(_exp.research_paths.resolve("convergence.json"))
    if snapshot.exists():
        try:
            payload = json.loads(snapshot.read_text(encoding="utf-8"))
            block_marginals = {row["block"]: row for row in payload.get("blocks") or []}
        except (OSError, ValueError):
            block_marginals = {}
    verdicts = []
    for key, identity in zip(arm_keys, arm_identities):
        block = key[0] if key else None
        block_row = block_marginals.get(block) or {}
        exact, completed, count, structural, similar = history[key, identity]
        verdicts.append({
            "available": True,
            "block": block,
            "block_median_marginal": block_row.get("median_marginal"),
            "duplicate_skeleton_of": structural[0] if structural else None,
            "structural_similarity_keys": structural,
            "exact_repeat_keys": exact,
            "already_simulated": bool(exact),
            "history_registry": str(Path(registry).expanduser().resolve()),
            "completed_refs": [_exp._probe_history_ref(entry, key=entry.get("key") or entry.get("alpha_id"),
                                                  train=True)
                               for entry in completed],
            "structural_refs": [_exp._probe_history_ref(entry, key=entry.get("key") or entry.get("alpha_id"))
                                for entry in similar],
            "structural_history_count": count,
        })
    return verdicts

def _probe_checkpoint_refs(target, arms, gate):
    """Look for exact matches at the intended checkpoint without searching the workspace."""
    if not target.is_file():
        return []
    try:
        path, document = _exp._load_probe_batch(target)
    except (CliError, KeyError, TypeError, ValueError):
        return []
    index = {}
    for row, record in zip(document["rows"], document["records"]):
        identity = _exp._probe_identity(row["expression"], row["settings"])
        index.setdefault(identity, []).append((row, record))

    recovery_labels = []
    for arm, verdict in zip(arms, gate):
        identity = _exp._probe_identity(arm["expression"], arm["settings"])
        for row, record in index.get(identity, []):
            ref = {"batch_path": str(path), "run_id": document["run_id"], "label": row["label"],
                   **_exp._probe_history_ref(record, train=True)}
            if _exp._has_finished_result(record) and not row.get("metrics_pending"):
                ref_type = "checkpoint_completed_refs"
            elif _exp._recoverable_probe_reference(row):
                ref_type = "checkpoint_recovery_refs"
            else:
                continue
            refs = verdict.setdefault(ref_type, [])
            if len(refs) < _exp._PROBE_REFERENCE_LIMIT:
                refs.append(ref)
        if verdict.get("checkpoint_recovery_refs"):
            recovery_labels.append(arm["label"])
    if not recovery_labels:
        return []
    return [{"action": "recover", "batch_path": str(path), "platform_read_only": True,
             "matching_arms": recovery_labels,
             "argv": ["python", "research_cli.py", "--research-dir", _exp.research_paths.research_dir(),
                      "probe", "--recover", str(path)]}]

def _recoverable_probe_reference(row):
    """Only link to recovery when the saved row identifies an accepted job."""
    if row["submission_state"] != "accepted" or not _exp._probe_needs_recovery(row):
        return False
    if row.get("alpha_id"):
        return True
    try:
        _exp.ml._validated_progress_url(row.get("multi_parent_url") or row.get("progress_url"))
    except _exp.ml.SimulationError:
        return False
    return True

def _probe_inputs(args, document=None):
    """Materialize request settings, source, and controls without searching history."""
    path = _exp._require_spec(args)
    spec = _exp._load_probe_spec(path) if document is None else _exp._normalize_probe_spec(document, path)
    name = getattr(args, "name", None) or spec.get("name") or Path(args.spec).stem
    groups = {}
    arms = []
    for index, original in enumerate(spec["arms"]):
        overrides = _exp.merge_settings({**(spec.get("settings") or {}),
                                    **(original.get("settings") or {})}, args)
        entry = {key: value for key, value in original.items()
                 if key not in {"settings", "design", "proxy_of", "source_file"}}
        source_file = original.get("source_file")
        if source_file and not _exp.ml.is_python_language(overrides.get("language")):
            raise CliError("INVALID_INPUT", "arm %r source_file requires language PYTHON" % original["label"],
                           "set \"language\": \"PYTHON\" in spec or arm settings")
        try:
            if source_file:
                source_file = Path(args.spec).expanduser().resolve().parent / Path(source_file).expanduser()
                entry["expression"] = _exp.config_io.read_input_text(source_file)
            resolved, pairs = _exp.config_io.build_config(overrides, [entry])
        except _exp.config_io.ResearchError as exc:
            raise CliError("INVALID_INPUT", str(exc),
                           "check the arm expression, decay, and settings via `python research_cli.py schema probe`") from exc
        expression, decay = pairs[0]
        settings = {**resolved, "decay": decay}
        arm = {**original, "expression": expression, "settings": settings,
               "arm_index": index,
               "ops": None if _exp.ml.is_python_language(settings.get("language"))
               else _exp.expression_lint.estimate_operator_count(expression)}
        if source_file:
            arm["source_file"] = str(source_file.resolve())
        arms.append(arm)
        group = groups.setdefault(_exp._settings_signature(settings),
                                  {"settings": settings, "arms": [], "pairs": []})
        group["arms"].append(arm)
        group["pairs"].append((expression, decay))
    design = _exp._probe_design({**spec, "arms": arms}, {})
    return spec, name, groups, arms, design

def _prepare_probe(args):
    spec, name, groups, arms, design = _exp._probe_inputs(args)
    gate = _exp._novelty_gate(arms, registry_path=getattr(args, "registry", None))
    return spec, name, groups, arms, design, gate

def probe_spec_identity(path, *, document=None):
    """Bind a materialized file to its checkpoint without repeating discovery."""
    spec, _, _, arms, design = _exp._probe_inputs(SimpleNamespace(spec=str(path)), document)
    return {"track": spec["track"], "design": design,
            **{key: spec[key] for key in ("quick_source", "breeding_source") if key in spec},
            "request_fingerprint": _exp._request_fingerprint(arms), "results": arms}

def preview_probe_spec(path, *, research_dir=None, multi_size=None, strict_gate=True):
    """Preview a BRAIN Probe spec offline in-process without spawning a child CLI."""
    import os
    from . import runtime
    from .parser import build_parser

    argv = ["probe", str(path), "--dry-run"]
    if strict_gate:
        argv.append("--strict-gate")
    if multi_size is not None:
        argv.extend(["--multi-size", str(multi_size)])
    root = (research_dir or os.environ.get(_exp.research_paths.ENV_VAR)
            or os.environ.get("WQ_RESEARCH_DIR"))
    previous_dir = os.environ.get(_exp.research_paths.ENV_VAR)
    previous_quiet = runtime._QUIET
    try:
        if root is not None:
            os.environ[_exp.research_paths.ENV_VAR] = str(Path(root).expanduser().resolve())
        runtime._QUIET = True
        parsed = build_parser().parse_args(argv)
        return _exp.cmd_probe(parsed, None)
    finally:
        runtime._QUIET = previous_quiet
        if previous_dir is None:
            os.environ.pop(_exp.research_paths.ENV_VAR, None)
        else:
            os.environ[_exp.research_paths.ENV_VAR] = previous_dir

def _request_fingerprint(arms):
    """Hash the ordered, materialized requests saved for a single batch.

    Unlike replay matching, this includes labels and the full resolved
    settings. Source file paths are provenance; saved source is the input.
    """
    request = [{"label": arm["label"], "expression": arm["expression"],
                "settings": arm["settings"]} for arm in arms]
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()
