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


#: Shapes the campaign has measured. Applying one to a field is a different experiment
#: from applying another, and the same field can go from noise to a live signal across
#: the set: the mid shape took a volume-autocorrelation leg from F -0.70 to +0.90 while
#: leaving volatility clustering worse than its plain group_rank.
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

def expected_turnover(expression, catalog=None, stats=None):
    """The turnover *band* an expression is likely to land in, from its fields.

    A point estimate would be false precision. What the ledger does show is that a
    dataset's cadence bounds what any construction on it can do: quarterly footnotes
    (fundamental2) sit at 0.020 whatever the shape, price-volume at 0.126, options at
    0.142, news at 0.358. The window then moves the arm *within* that band - a 252-day
    difference of a quarterly field reorders once a year, a 5-day difference four times a
    quarter - so the honest output is ``[min, max]`` over the datasets the arms reads,
    plus which side of the floor it is likely to fall on.
    """
    catalog = catalog if catalog is not None else load_catalog()
    stats = stats if stats is not None else dataset_turnover()
    text = " ".join((expression or "").split())
    datasets = set()
    for match in _TOKEN.finditer(text):
        if text[match.end():].lstrip().startswith("("):
            continue
        meta = catalog.get(match.group(0))
        if meta:
            datasets.add(meta["dataset"])
    values = sorted(stats[dataset]["median"] for dataset in datasets
                    if dataset in stats and stats[dataset].get("median") is not None)
    if not values:
        return None
    return {"low": round(values[0], 3), "high": round(values[-1], 3),
            "datasets": sorted(datasets)}


_LONG_WINDOW = re.compile(r"ts_(?:delta|mean|std_dev|sum)\([^,]+,\s*(\d{2,})\s*\)")


def long_window_caution(expression, threshold=126):
    """A long window on a slow field reorders rarely, whatever the dataset median says.

    The measured R&D case is the reason this exists: fundamental fields sit at a 0.118
    median, but `ts_delta(rd_expense, 252)` came back at 0.047 because the annual
    difference of a quarterly field only changes once a year. The median bounds the arm
    from above; the window decides where inside the bound it lands.
    """
    windows = [int(value) for value in _LONG_WINDOW.findall(expression or "")]
    if not windows or max(windows) < threshold:
        return None
    return ("long window (%d) on these fields: expect turnover below the dataset median"
            % max(windows))


def turnover_verdict(band, floor=0.125):
    """What the band implies for Fitness, given the ``max(T, 0.125)`` floor."""
    if band is None:
        return None
    if band["high"] < 0.05:
        return "floor-taxed: every field updates rarely; F is capped near S*sqrt(ret/0.125)"
    if band["low"] >= floor:
        return None
    if band["low"] < 0.05 <= band["high"]:
        return ("mixed: the slow fields will be floor-taxed unless a fast field carries "
                "the turnover")
    return None
