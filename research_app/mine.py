"""Mining the field catalog: which untouched fields are worth a construction.

Three measured facts decide whether a field is worth a probe, and none of them is
"it exists":

* **is it untouched** - a field already in the ledger has been paid for; new work starts
  from the ones that have not;
* **will its cadence land in the useful turnover band** - a dataset whose alphas have
  always come back at 0.02 turnover is paying the ``max(T, 0.125)`` floor tax, and one at
  0.4 has information lasting days. Both were measured the hard way this month: the
  relationship graph came back at 0.41, inventory composition at 0.02, and neither could
  be rescued by shape;
* **what type is it** - VECTOR and EVENT fields are rejected by arithmetic operators, so
  an arm built on one needs `vec_avg` or `vec_sum` before anything else touches it. Five
  arms once went to the simulator to learn this.

`mine` reports those three together and can emit a probe spec, so the gap between
"interesting field" and "batch to simulate" is one command instead of a transcription.
"""

import collections
import glob
import json
import os
import re
import statistics

import alpha_registry
import research_paths

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Types that arithmetic and time-series operators reject outright.
AGGREGATED_TYPES = frozenset({"VECTOR", "EVENT"})

#: Types that are labels, not measurements. A sector code or a type index has no
#: cross-section to rank; emitting an arm for one wastes a simulation and looks silly.
LABEL_TYPES = frozenset({"GROUP", "SYMBOL", "INTEGER"})

#: Descriptions that mark a label dressed up as a field.
LABEL_WORDS = ("grouping field", "type ind", "iso code", "zone code", "currency code",
               "categories of instrument", "index vector", "identifier")


def is_label(meta):
    if (meta.get("type") or "").upper() in LABEL_TYPES:
        return True
    description = (meta.get("description") or "").lower()
    return any(word in description for word in LABEL_WORDS)


def load_catalog():
    """``{field_id: {dataset, type, coverage, description}}`` from the disk cache."""
    fields = {}
    pattern = os.path.join(research_paths.metadata_root(), "datafields", "*.json")
    for path in glob.glob(pattern):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle).get("payload") or []
        except (OSError, ValueError):
            continue
        for row in payload:
            field_id = row.get("id")
            if not field_id:
                continue
            fields[field_id] = {
                "dataset": (row.get("dataset") or {}).get("id") or "?",
                "type": (row.get("type") or "").upper() or None,
                "coverage": row.get("coverage"),
                "description": (row.get("description") or "").strip(),
            }
    return fields


def used_identifiers(registry_path=None):
    """Identifiers the ledger has ever named, minus operators (a field is never `name(`)."""
    used = set()
    for entry in alpha_registry.load_entries(registry_path):
        text = entry.get("expression") or ""
        for match in _TOKEN.finditer(text):
            if text[match.end():].lstrip().startswith("("):
                continue
            used.add(match.group(0))
    return used


def dataset_turnover(entries=None):
    """``{dataset: {n, median}}`` - the turnover this dataset has actually produced.

    Attribution is by the fields an expression names: an alpha built from analyst fields
    reports analyst cadence. Mixed expressions count for every dataset they touch, which
    is why the median (not the mean) is the number to trust.
    """
    catalog = load_catalog()
    if entries is None:
        entries = alpha_registry.load_entries()
    buckets = collections.defaultdict(list)
    for entry in entries:
        turnover = (entry.get("metrics") or {}).get("turnover")
        if turnover is None:
            continue
        text = entry.get("expression") or ""
        datasets = set()
        for match in _TOKEN.finditer(text):
            if text[match.end():].lstrip().startswith("("):
                continue
            meta = catalog.get(match.group(0))
            if meta:
                datasets.add(meta["dataset"])
        for dataset in datasets:
            buckets[dataset].append(float(turnover))
    return {dataset: {"n": len(values), "median": statistics.median(values)}
            for dataset, values in buckets.items()}


def wrapper_for(field_type):
    """How the field must be wrapped before arithmetic touches it."""
    if (field_type or "").upper() in AGGREGATED_TYPES:
        return "vec_avg"
    return None


def _in_band(value, band):
    low, high = band
    return value is not None and low <= value <= high


def cmd_mine(args, _sink):
    """Report untouched fields whose dataset cadence fits the band, and optionally emit a spec."""
    catalog = load_catalog()
    if not catalog:
        raise ValueError("no cached field catalog; run `catalog fields` first")
    used = used_identifiers(getattr(args, "registry", None))
    turnover = dataset_turnover()

    band = tuple(float(part) for part in args.turnover_band.split(","))
    if len(band) != 2:
        raise ValueError("--turnover-band wants MIN,MAX")

    datasets = {}
    for field_id, meta in catalog.items():
        if args.dataset and meta["dataset"] != args.dataset:
            continue
        if field_id in used:
            continue
        if is_label(meta):
            continue
        coverage = meta.get("coverage")
        if args.min_coverage and (coverage is None or coverage < args.min_coverage):
            continue
        bucket = datasets.setdefault(meta["dataset"], {"dataset": meta["dataset"],
                                                       "unused": 0, "candidates": []})
        bucket["unused"] += 1
        bucket["candidates"].append({
            "id": field_id,
            "type": meta.get("type"),
            "coverage": coverage,
            "wrapper": wrapper_for(meta.get("type")),
            "description": meta["description"][:90],
        })

    rows = []
    for dataset, bucket in datasets.items():
        stats = turnover.get(dataset) or {}
        median = stats.get("median")
        fits = _in_band(median, band)
        candidates = sorted(bucket["candidates"], key=lambda row: -(row["coverage"] or 0))
        rows.append({
            "dataset": dataset,
            "unused": bucket["unused"],
            "candidates": len(candidates),
            "median_turnover": None if median is None else round(median, 3),
            "turnover_in_band": fits,
            "type_mix": dict(collections.Counter(row["type"] or "?" for row in candidates)),
            "top": candidates[: args.limit],
        })
    # A dataset whose alphas have always landed inside the band sorts first; ties break on
    # how much is left untouched.
    rows.sort(key=lambda row: (not row["turnover_in_band"], -row["candidates"]))

    payload = {
        "band": list(band),
        "datasets": rows,
        "note": ("median_turnover is what this dataset's alphas actually produced, not a "
                 "guess; a dataset outside the band costs either floor tax or a signal "
                 "that dies in days. `wrapper` is required before arithmetic for VECTOR "
                 "and EVENT fields."),
    }

    if getattr(args, "emit", None):
        arms = []
        for row in rows:
            for candidate in row["top"]:
                expression = candidate["id"]
                if candidate["wrapper"]:
                    expression = "%s(%s)" % (candidate["wrapper"], expression)
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
            raise ValueError("nothing to emit: no candidate fields matched")
        spec = {
            "name": args.name or "field-mine",
            "idea": "field mining: first contact with untouched fields",
            "mechanism": ("Untouched fields ranked by whether their dataset's realised "
                          "turnover lands in %s. Each arm is a plain rank of the field, so "
                          "the first batch answers 'does this field move a cross-section "
                          "at all' before any shape work." % (band,)),
            "note": "first-contact scan; replace rank() with the mechanism's own shape once a field shows life",
            "design": "explore",
            "settings": {"region": "USA", "universe": "TOP3000", "delay": 1,
                         "neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08,
                         "test_period": "P1Y"},
            "arms": arms,
        }
        path = os.path.abspath(args.emit)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(spec, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        payload["spec"] = {"path": path, "arms": len(arms)}

    return payload