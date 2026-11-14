"""Offline field review, not proof of a new mechanism or permission to simulate.

Separate numeric fields from labels and reductions needing a definition. Show
same-description ledger evidence without claiming alias equivalence. Dataset
turnover is a train-only historical heuristic, not a field's update cadence.
"""

import collections
import glob
import json
import math
import os
import re
import statistics
from datetime import datetime

import alpha_registry
import cache_store
import fastexpr_ast
import machine_lib as ml
import research_paths
from .runtime import CliError, protect_output

#: Types that arithmetic and time-series operators reject outright.
AGGREGATED_TYPES = frozenset({"VECTOR", "EVENT"})

#: Types that are labels, not measurements. A sector code or a type index has no
#: cross-section to rank; emitting an arm for one wastes a simulation and looks silly.
LABEL_TYPES = frozenset({"GROUP", "SYMBOL"})

#: Descriptions that mark a label dressed up as a field.
LABEL_WORDS = ("grouping field", "type ind", "iso code", "zone code", "currency code",
               "categories of instrument", "index vector", "identifier", "forecast type",
               "pricing currency", "city code", "country code", "zip code")


def is_label(meta):
    if (meta.get("type") or "").upper() in LABEL_TYPES:
        return True
    description = (meta.get("description") or "").lower()
    return any(word in description for word in LABEL_WORDS)


def _scope():
    return {key: ml.DEFAULT_SETTINGS[key] for key in
            ("region", "universe", "delay", "instrumentType")}


def _moment(value):
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return moment if moment.tzinfo else None
    except (TypeError, ValueError):
        return None


def load_catalog(report=None):
    """``{field_id: {dataset, type, coverage, description}}`` from the disk cache."""
    fields = {}
    pattern = os.path.join(research_paths.metadata_root(), "datafields", "*.json")
    scope = _scope()
    ignored = 0
    now = cache_store.utc_now()
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
            payload = document.get("payload") or []
        except (OSError, ValueError, AttributeError):
            continue
        if not isinstance(payload, list):
            continue
        envelope = document.get("_cache") or {}
        created = _moment(envelope.get("created_at"))
        expiry = _moment(envelope.get("expires_at"))
        freshness = ("fresh" if expiry > now else "stale") if expiry else "unknown"
        for row in payload:
            if not isinstance(row, dict):
                continue
            field_id = row.get("id")
            if not field_id:
                continue
            declared = {key: row.get(key) for key in scope if row.get(key) is not None}
            if any(value != scope[key] for key, value in declared.items()):
                ignored += 1
                continue
            scope_known = all(key in declared for key in ("region", "universe", "delay"))
            source = {"path": path, "scope_state": "matched" if scope_known else "unknown",
                      "freshness": freshness, "created_at": envelope.get("created_at"),
                      "expires_at": envelope.get("expires_at"),
                      "age_seconds": max(0, (now - created).total_seconds()) if created else None}
            # Prefer explicitly scoped, fresh, newest metadata; never depend on glob order.
            priority = (scope_known, freshness == "fresh", created.timestamp() if created else 0, path)
            previous = fields.get(field_id)
            if previous and priority <= previous["_priority"]:
                continue
            dataset = row.get("dataset") or "?"
            fields[field_id] = {
                "dataset": dataset.get("id", "?") if isinstance(dataset, dict) else dataset,
                "type": (row.get("type") or "").upper() or None,
                "coverage": row.get("coverage"),
                "description": (row.get("description") or "").strip(),
                "source": source,
                "_priority": priority,
            }
    for meta in fields.values():
        meta.pop("_priority", None)
    if report is not None:
        report.update({"scope": scope, "source": "local_metadata_cache",
                       "live_verified": False, "account_verified": False,
                       "fields": len(fields), "ignored_scope_rows": ignored,
                       "freshness": dict(collections.Counter(
                           meta["source"]["freshness"] for meta in fields.values()))})
    return fields


def _fields(expression):
    return set(fastexpr_ast.field_names(fastexpr_ast.parse_program(expression)))


def used_identifiers(registry_path=None):
    """Fields in parseable latest ledger expressions, excluding comments and locals."""
    used = set()
    for entry in alpha_registry.load_entries(registry_path):
        try:
            used.update(_fields(entry.get("expression") or ""))
        except fastexpr_ast.ParseError:
            continue
    return used


def dataset_turnover(entries=None, catalog=None):
    """``{dataset: {n, median}}`` - the turnover this dataset has actually produced.

    Mixed expressions count for each dataset they touch. Their medians are descriptive,
    confounded by construction/settings and do not measure a field's update cadence.
    Missing train metrics remain unknown; full-window metrics are never substituted.
    """
    catalog = load_catalog() if catalog is None else catalog
    if entries is None:
        entries = alpha_registry.load_entries()
    buckets = collections.defaultdict(list)
    missing = collections.Counter()
    for entry in entries:
        try:
            identifiers = _fields(entry.get("expression") or "")
        except fastexpr_ast.ParseError:
            continue
        turnover = ((entry.get("out_of_sample") or {}).get("train") or {}).get("turnover")
        datasets = {catalog[name]["dataset"] for name in identifiers if name in catalog}
        for dataset in datasets:
            if isinstance(turnover, (int, float)) and not isinstance(turnover, bool) \
                    and math.isfinite(turnover) and turnover >= 0:
                buckets[dataset].append(float(turnover))
            else:
                missing[dataset] += 1
    return {dataset: {"n": len(buckets[dataset]),
                      "median": statistics.median(buckets[dataset]) if buckets[dataset] else None,
                      "missing_train": missing[dataset]}
            for dataset in buckets.keys() | missing.keys()}


#: Applying a different shape is a separate experiment: a transformation can change
#: both signal strength and turnover. Compare each against its plain rank control.
SHAPES = {
    "raw": "rank({x})",
    "centre": "-(group_rank({x}, subindustry) - 0.5)",
    "midtier": "-(abs(group_rank({x}, subindustry) - 0.5))",
    "change": "group_rank(ts_mean(ts_delta({x}, 5), 21), subindustry) - 0.5",
    "ratio_assets": "rank({x} / (assets + 1))",
}


def apply_shape(shape, expression):
    template = SHAPES.get(shape)
    if template is None:
        raise ValueError("unknown shape %r (known: %s)" % (shape, ", ".join(sorted(SHAPES))))
    return template.format(x=expression)


def wrapper_for(field_type):
    """A type alone cannot choose a meaningful reduction of vector elements."""
    return None


def _review_reasons(meta):
    reasons = []
    if is_label(meta):
        reasons.append("categorical_or_identifier")
    if meta.get("type") in AGGREGATED_TYPES:
        reasons.append("reduction_semantics_unknown")
    elif meta.get("type") != "MATRIX":
        reasons.append("non_matrix_type")
    description = meta.get("description", "").strip().lower()
    if not description or description in {"estimation value", "no description", "no field description"}:
        reasons.append("measurement_definition_missing")
    if re.search(r"\bdate\b|\btimestamp\b", description) and not re.search(
            r"\bsince\b|\belapsed\b|\bnumber of days\b", description):
        reasons.append("date_encoding_needs_definition")
    return reasons


def _description_key(meta):
    return (meta["dataset"], meta.get("type"),
            " ".join(re.findall(r"[a-z0-9]+", meta.get("description", "").lower())))


def _in_band(value, band):
    low, high = band
    return value is not None and low <= value <= high


def cmd_mine(args, _sink):
    """Review cached fields absent from parsed ledger expressions, optionally draft a scan."""
    protect_output(args, getattr(args, "emit", None))
    evidence = {}
    catalog = load_catalog(evidence)
    if not catalog:
        raise CliError("INVALID_INPUT", "no cached field catalog; run `catalog fields` first")
    entries = alpha_registry.load_entries(getattr(args, "registry", None))
    used = set()
    parse_failures = []
    usage = collections.defaultdict(list)
    for entry in entries:
        try:
            identifiers = _fields(entry.get("expression") or "")
        except fastexpr_ast.ParseError:
            parse_failures.append(entry.get("key") or entry.get("alpha_id"))
            continue
        used.update(identifiers)
        for identifier in identifiers:
            usage[identifier].append({"field": identifier,
                                      "key": entry.get("key") or entry.get("alpha_id"),
                                      "verdict": entry.get("verdict"),
                                      "mechanism": entry.get("mechanism")})
    turnover = dataset_turnover(entries, catalog)

    try:
        band = tuple(float(part) for part in args.turnover_band.split(","))
    except (TypeError, ValueError):
        raise CliError("INVALID_INPUT", "--turnover-band wants finite MIN,MAX with 0 <= MIN <= MAX") from None
    if len(band) != 2 or not all(math.isfinite(value) for value in band) \
            or not 0 <= band[0] <= band[1]:
        raise CliError("INVALID_INPUT", "--turnover-band wants finite MIN,MAX with 0 <= MIN <= MAX")
    if not 0 <= args.min_coverage <= 1 or args.smooth < 0 or args.arms < 1:
        raise CliError("INVALID_INPUT", "--min-coverage must be in [0,1], --smooth >= 0 and --arms >= 1")

    descriptions = collections.defaultdict(list)
    for field_id, meta in catalog.items():
        if field_id in used and _description_key(meta)[2]:
            descriptions[_description_key(meta)].append(field_id)

    datasets = {}
    for field_id, meta in catalog.items():
        if args.dataset and meta["dataset"] != args.dataset:
            continue
        if field_id in used:
            continue
        coverage = meta.get("coverage")
        if args.min_coverage and (coverage is None or coverage < args.min_coverage):
            continue
        bucket = datasets.setdefault(meta["dataset"], {"dataset": meta["dataset"],
                                                       "unused": 0, "candidates": [],
                                                       "needs_definition": []})
        bucket["unused"] += 1
        reasons = _review_reasons(meta)
        related = sorted(descriptions.get(_description_key(meta), []))
        if related:
            reasons.append("same_description_already_used")
        candidate = {
            "id": field_id,
            "type": meta.get("type"),
            "coverage": coverage,
            "wrapper": None,
            "description": meta["description"],
            "review_state": "needs_definition" if reasons else "numeric_review",
            "reasons": reasons,
            "source": meta.get("source"),
            "same_description_fields": related[:5],
            "same_description_count": len(related),
            "related_history": [record for field in related[:5]
                                for record in usage[field][:2]][:5],
        }
        bucket["needs_definition" if reasons else "candidates"].append(candidate)

    rows = []
    for dataset, bucket in datasets.items():
        stats = turnover.get(dataset) or {}
        median = stats.get("median")
        fits = _in_band(median, band)
        candidates = sorted(bucket["candidates"], key=lambda row: -(row["coverage"] or 0))
        review = sorted(bucket["needs_definition"], key=lambda row: -(row["coverage"] or 0))
        rows.append({
            "dataset": dataset,
            "unused": bucket["unused"],
            "candidates": len(candidates),
            "median_turnover": None if median is None else round(median, 3),
            "turnover_samples": stats.get("n", 0),
            "missing_train_turnover": stats.get("missing_train", 0),
            "turnover_in_band": fits,
            "type_mix": dict(collections.Counter(row["type"] or "?" for row in candidates)),
            "top": candidates[: args.limit],
            "needs_definition_count": len(review),
            "needs_definition_reasons": dict(collections.Counter(
                reason for row in review for reason in row["reasons"])),
            "needs_definition": review[: args.limit],
        })
    # A dataset whose alphas have always landed inside the band sorts first; ties break on
    # how much is left untouched.
    rows.sort(key=lambda row: (not row["turnover_in_band"], -row["candidates"]))

    payload = {
        "band": list(band),
        "datasets": rows,
        "catalog": evidence,
        "history": {"source": "latest_ledger_expressions", "entries": len(entries),
                    "parse_failures": parse_failures,
                    "scope": "No claim about unrecorded reports, aliases or economic mechanisms."},
        "metrics_scope": "train",
        "note": ("unused means absent from parseable latest ledger expressions, not a new "
                 "mechanism. numeric_review is a screening lead, not semantic/PIT approval. "
                 "Same descriptions suggest related history, not proven aliases. VECTOR/EVENT "
                 "reductions require an explicit economic choice. median_turnover describes "
                 "train results of mixed constructions, not field cadence or a prediction; "
                 "missing train metrics remain unknown. Cached scope/freshness is not live "
                 "account verification."),
    }

    if getattr(args, "emit", None):
        arms = []
        explicit = [part.strip() for part in (getattr(args, "expr", None) or "").split(";")
                    if part.strip()]
        if explicit:
            shape = getattr(args, "shape", None) or "raw"
            for index, expression in enumerate(explicit):
                arms.append({"label": "%s_%d" % (shape, index + 1),
                             "expression": apply_shape(shape, expression)})
        for row in ([] if explicit else rows):
            for candidate in row["top"]:
                expression = candidate["id"]
                # A raw daily attention/sentiment field reorders every day: the social
                # buzz fields came back at 0.45-0.58 turnover on a first-contact scan,
                # which the floor does not forgive. Smoothing is part of the first shape,
                # not a later refinement, when the field is known to be daily-noisy.
                if args.smooth:
                    expression = "ts_mean(%s, %d)" % (expression, args.smooth)
                arms.append({"label": candidate["id"][:28],
                             "expression": "rank(%s)" % expression})
                if len(arms) >= args.arms:
                    break
            if len(arms) >= args.arms:
                break
        if not arms:
            raise CliError("INVALID_INPUT", "nothing to emit: no numeric-review fields matched; inspect "
                           "needs_definition and use --expr for a reviewed reduction")
        spec = {
            "name": args.name or "field-mine",
            "idea": "field review: exploratory draft, not a verified mechanism",
            "mechanism": ("Explicit expressions" if explicit else "Numeric-review field screen"),
            "note": ("Draft only: define economic direction, units, timing/PIT, controls and "
                     "stopping criteria before simulation; probe --dry-run then --check. "
                     "Dataset train turnover and absent field ids do not establish novelty."),
            "design": "explore",
            "settings": {**_scope(),
                         "neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08,
                         "test_period": "P1Y"},
            "arms": arms,
        }
        path = os.path.abspath(args.emit)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with cache_store.file_lock(path + ".lock"):
            if os.path.exists(path):
                raise CliError("INVALID_INPUT", "refusing to overwrite an existing spec: %s" % path)
            cache_store.write_json_atomic(path, spec, indent=2)
        payload["spec"] = {"path": path, "arms": len(arms)}

    return payload

def expected_turnover(expression, catalog=None, stats=None):
    """Min/max observed train medians of referenced datasets, not a forecast or bound."""
    catalog = catalog if catalog is not None else load_catalog()
    stats = stats if stats is not None else dataset_turnover()
    try:
        identifiers = _fields(expression or "")
    except fastexpr_ast.ParseError:
        return None
    datasets = {catalog[name]["dataset"] for name in identifiers if name in catalog}
    values = sorted(stats[dataset]["median"] for dataset in datasets
                    if dataset in stats and stats[dataset].get("median") is not None)
    if not values:
        return None
    return {"low": round(values[0], 3), "high": round(values[-1], 3),
            "datasets": sorted(datasets)}


_LONG_WINDOW = re.compile(r"ts_(?:delta|mean|std_dev|sum)\([^,]+,\s*(\d{2,})\s*\)")


def long_window_caution(expression, threshold=126):
    """Flag a long window without predicting its effect from dataset medians."""
    windows = [int(value) for value in _LONG_WINDOW.findall(expression or "")]
    if not windows or max(windows) < threshold:
        return None
    return ("long window (%d): measure this construction's turnover; dataset train medians "
            "do not identify the effect of its window"
            % max(windows))


def turnover_verdict(band, floor=0.125):
    """Describe observed train medians; the proposed expression remains unmeasured."""
    if band is None:
        return None
    if band["high"] < 0.05:
        return ("historical floor-taxed observations: dataset train medians are below 0.05; "
                "verify the proposed expression's turnover before applying the Fitness floor")
    if band["low"] >= floor:
        return None
    if band["low"] < 0.05 <= band["high"]:
        return ("mixed observed dataset train medians; field cadence and the proposed "
                "expression's turnover remain unknown")
    return None
