#!/usr/bin/env python3
"""Build two private AQP maps from the visible ACTIVE book and cached training PnL.

An authenticated capture reads the own-Alpha list. Pass --roster to replay its
projected metadata offline. Both maps use the same cohort. No submissions,
metadata updates, simulations, or PnL cache refreshes are involved.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import alpha_registry
import cache_store
import correlation
import fastexpr_ast
import machine_lib as ml
import research_paths
from research_app import capabilities, mechanisms


TEMPLATES = {
    "theme": ("aqp_theme_map.html", "aqp-theme-map.html"),
    "correlation": ("aqp_correlation_map.html", "aqp-correlation-map.html"),
}
ROSTER_SOURCE = "BRAIN 可见的已提交 ACTIVE Alpha"
ALPHA_ID = re.compile(r"[A-Za-z0-9]+")
TAG = re.compile(r"AQP[0-9]{6}", re.IGNORECASE)


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def names(values):
    if not isinstance(values, list):
        return []
    labels = [item.get("name") or item.get("id") if isinstance(item, dict) else item for item in values]
    return sorted({label for label in labels if isinstance(label, str) and label.strip()})


def theme_evidence(row):
    check_rows = (row.get("is") or {}).get("checks") or []
    checks = [check for check in check_rows if check.get("name") == "MATCHES_THEMES"]
    states = sorted({check["result"] for check in checks if isinstance(check.get("result"), str)})
    matched = set(names(row.get("themes")))
    for check in checks:
        if check.get("result") == "PASS":
            matched.update(names(check.get("themes")))
    if matched or "PASS" in states:
        state = "matched"
    elif "PENDING" in states:
        state = "pending"
    elif "WARNING" in states:
        state = "warning"
    elif states == ["FAIL"]:
        state = "not_matched"
    else:
        state = "unknown"
    return {"state": state, "names": sorted(matched), "check_results": states}


def iso_day(value):
    if not isinstance(value, str) or len(value) != 10:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


def training_window(row):
    train, test = row.get("train") or {}, row.get("test") or {}
    start = iso_day(train.get("startDate"))
    # The first test date is the exclusive boundary, including on weekends.
    end = iso_day(test.get("startDate"))
    if not end:
        end = iso_day(train.get("endDate"))
    if not start or not end or start >= end:
        return {"start": None, "end": None, "sharpe": None, "fitness": None}
    return {"start": start, "end": end, "sharpe": number(train.get("sharpe")),
            "fitness": number(train.get("fitness"))}


def capture_roster():
    """Store the minimal metadata needed to reproduce the cohort."""
    rows = []
    with ml.load_session() as session:
        offset = 0
        seen = set()
        while True:
            page = ml.list_alphas(session, status="ACTIVE", region="USA", order="-dateSubmitted",
                                 limit=100, offset=offset)
            for row in page:
                identifier = row.get("id")
                if not isinstance(identifier, str) or not ALPHA_ID.fullmatch(identifier):
                    raise ValueError("BRAIN returned an invalid Alpha ID")
                if identifier in seen:
                    raise ValueError("the live Alpha list changed during pagination")
                seen.add(identifier)
                settings = row.get("settings") or {}
                delay = settings.get("delay")
                if (row.get("status") != "ACTIVE" or row.get("type") != "REGULAR"
                        or not row.get("dateSubmitted") or row.get("hidden")
                        or settings.get("region") != "USA" or settings.get("language") != "FASTEXPR"
                        or type(delay) is not int or delay not in {0, 1}):
                    continue
                tags = row.get("tags")
                rows.append({
                    "id": identifier, "settings": {key: settings.get(key) for key in
                        ("region", "universe", "delay", "instrumentType")},
                    "train": training_window(row),
                    "theme": theme_evidence(row),
                    "aqp_tags": [tag.upper() for tag in names(tags) if TAG.fullmatch(tag)],
                    "tags_known": isinstance(tags, list),
                })
            if len(page) < 100:
                break
            offset += len(page)
    return {"version": 1, "captured_at": cache_store.utc_now().isoformat(),
            "source": ROSTER_SOURCE, "rows": sorted(rows, key=lambda row: row["id"])}


def load_roster(path):
    roster, error = cache_store.load_json(path)
    if error or not isinstance(roster, dict) or roster.get("version") != 1:
        raise ValueError("expected a saved version-1 AQP roster")
    rows = roster.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("AQP roster lacks valid rows")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(alpha_id, str) or not ALPHA_ID.fullmatch(alpha_id) for alpha_id in ids):
        raise ValueError("invalid Alpha ID in AQP roster")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate Alpha ID in AQP roster")
    if not isinstance(roster.get("captured_at"), str):
        raise ValueError("AQP roster lacks capture timestamp")
    return roster


def clean_label(value):
    return " ".join(value.split()) if isinstance(value, str) and value.strip() else None


def structural_annotation(entry):
    expression = entry.get("expression") or ""
    if not expression:
        return None, None, None
    block = mechanisms.block_of(expression)
    try:
        tree = fastexpr_ast.parse(expression)
    except fastexpr_ast.ParseError:
        return block, None, None
    key = json.dumps([block, fastexpr_ast.skeleton(tree)], ensure_ascii=False)
    return block, hashlib.sha256(key.encode()).hexdigest()[:12], fastexpr_ast.field_names(tree)


def dataset_lookup(settings, cache):
    key = tuple(settings.get(k) for k in ("region", "universe", "delay", "instrumentType"))
    if key not in cache:
        scope = capabilities.cached_field_scope(settings)
        mapping = {}
        for row in scope["records"]:
            dataset = row.get("dataset")
            dataset = dataset.get("id") if isinstance(dataset, dict) else dataset
            if isinstance(dataset, str):
                mapping.setdefault(row["id"], set()).add(dataset)
        cache[key] = mapping
    return cache[key]


def train_series(pnl_directory, identifier, start, end):
    if not start or not end:
        return {}, "missing_train_boundary"
    values, error = cache_store.load_json(pnl_directory / f"{identifier}.json")
    if error or not isinstance(values, dict):
        return {}, "missing_pnl"
    coverage = ml.pnl_cache_status(pnl_directory, identifier, values)
    if coverage.get("complete") is not True:
        return {}, "incomplete_pnl"
    series = {}
    for day, value in values.items():
        if iso_day(day) and start <= day < end and number(value) is not None:
            series[day] = value
    return series, "available" if series else "empty_train"


def alpha_annotations(roster):
    ledger = {row["alpha_id"]: row for row in alpha_registry.load_entries() if row.get("alpha_id")}
    dataset_cache = {}
    pnl_directory = Path(research_paths.resolve("pnl"))
    alphas, series_by_id = [], {}
    for row in roster["rows"]:
        identifier = row["id"]
        entry = ledger.get(identifier, {})
        block, structure, fields = structural_annotation(entry)
        settings, train = row["settings"], row["train"]
        mapping = dataset_lookup(settings, dataset_cache) if fields else {}
        datasets = sorted({dataset for field in fields or () for dataset in mapping.get(field, ())})
        dataset_coverage = ("verified" if fields and all(field in mapping for field in fields)
                            else "partial" if datasets else "unknown")
        start, end = train["start"], train["end"]
        series, state = train_series(pnl_directory, identifier, start, end)
        series_by_id[identifier] = series
        alphas.append({
            "id": identifier, "family": clean_label(entry.get("family")),
            "mechanism": clean_label(entry.get("mechanism")), "block": block,
            "structure_id": structure, "datasets": datasets, "dataset_coverage": dataset_coverage,
            "universe": settings.get("universe"), "delay": settings.get("delay"),
            "train_start": start, "train_end": end, "train_sharpe": train["sharpe"],
            "train_fitness": train["fitness"], "pnl_state": state, "pnl_points": len(series),
            "theme": row["theme"], "aqp_tags": row["aqp_tags"], "tags_known": row["tags_known"],
        })
    return alphas, series_by_id


def complete_link_forest(ids, pairs):
    """Merge the groups with the strongest minimum pairwise absolute correlation.

    An absent cross-group measurement precludes a merge. Child identifiers
    and insertion order give a reproducible tie-break, including singletons.
    """
    indices = {identifier: index for index, identifier in enumerate(ids)}
    clusters = {index: {"id": identifier} for identifier, index in indices.items()}
    strengths = {}
    for pair in pairs:
        if pair["r"] is not None:
            a, b = sorted((indices[pair["a"]], indices[pair["b"]]))
            strengths[a, b] = min(1.0, abs(pair["r"]))
    next_index = len(ids)
    while strengths:
        a, b = min(strengths, key=lambda pair: (-strengths[pair], pair))
        node = {"strength": strengths[a, b], "children": [clusters[a], clusters[b]]}
        for other in clusters:
            if other in (a, b):
                continue
            left = strengths.get(tuple(sorted((a, other))))
            right = strengths.get(tuple(sorted((b, other))))
            if left is not None and right is not None:
                strengths[other, next_index] = min(left, right)
        strengths = {key: value for key, value in strengths.items() if a not in key and b not in key}
        del clusters[a]
        del clusters[b]
        clusters[next_index] = node
        next_index += 1
    return list(clusters.values())


def pairwise_map(series_by_id, min_overlap):
    matrix = correlation.matrix(series_by_id, min_overlap=min_overlap)
    pairs = matrix["pairs"]
    unmeasured = []
    for pair in matrix["skipped"]:
        a, b = pair["a"], pair["b"]
        overlap = len(series_by_id[a].keys() & series_by_id[b].keys())
        reason = pair["reason"] if series_by_id[a] and series_by_id[b] else "missing_train_pnl"
        unmeasured.append({**pair, "n": overlap, "r": None, "reason": reason})
    all_pairs = pairs + unmeasured
    return {"min_overlap": min_overlap, "default_threshold": 0.70, "measured": len(pairs),
            "unknown": len(unmeasured), "pairs": all_pairs,
            "forest": complete_link_forest(matrix["ids"], all_pairs)}


def render_map(template, destination, data):
    marker = "<!--MAP_DATA-->"
    markup = template.read_text(encoding="utf-8")
    if markup.count(marker) != 1:
        raise ValueError(f"expected one map data marker in {template.name}")
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("<", "\\u003c")
    payload = f'<script type="application/json" id="map-data">{encoded}</script>'
    cache_store.write_text_atomic(destination, markup.replace(marker, payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", required=True, help="AQP cycle, expressed as YYYYMM")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--research-dir", type=Path, help="Source workspace containing ledger, catalog and PnL cache")
    parser.add_argument("--roster", type=Path, help="Replay a prior aqp-roster.json without network access")
    parser.add_argument("--min-overlap", type=int, default=200, help="Minimum training days shared by each pair")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{4}(?:0[1-9]|1[0-2])", args.cycle):
        parser.error("--cycle must be YYYYMM")
    if args.min_overlap < 3:
        parser.error("--min-overlap must be at least 3")
    if args.research_dir:
        import os
        os.environ[research_paths.ENV_VAR] = str(args.research_dir.resolve())

    roster = load_roster(args.roster) if args.roster else capture_roster()
    alphas, series = alpha_annotations(roster)
    pairwise = pairwise_map(series, args.min_overlap)
    common = {
        "generated_at": cache_store.utc_now().isoformat(),
        "cohort": {"count": len(alphas), "source": roster["source"], "cycle": args.cycle,
                   "captured_at": roster["captured_at"]},
        "alphas": alphas, "links": {key: value[1] for key, value in TEMPLATES.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_store.write_json_atomic(args.output_dir / "aqp-roster.json", roster, indent=2)
    for view, (template, filename) in TEMPLATES.items():
        data = {**common, "pairwise": pairwise} if view == "correlation" else common
        render_map(Path(__file__).with_name(template), args.output_dir / filename, data)
    print(json.dumps({
        "count": len(alphas), "families": Counter(row["family"] for row in alphas),
        "native_theme_states": Counter(row["theme"]["state"] for row in alphas),
        "pairs_measured": pairwise["measured"], "pairs_unknown": pairwise["unknown"],
        "high_correlation_pairs": sum(abs(pair["r"]) >= pairwise["default_threshold"]
                                      for pair in pairwise["pairs"] if pair["r"] is not None),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
