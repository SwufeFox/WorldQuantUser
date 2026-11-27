"""Draft one matched generation from a saved Probe batch's training evidence."""

from collections import Counter, defaultdict
from random import Random

import alpha_registry
import alpha_unit
import machine_lib as ml

from . import experiments
from .runtime import CliError


def _parent_groups(batch):
    """Preserve captured mechanisms and compare like-for-like settings."""
    groups = defaultdict(list)
    omitted = Counter()
    for arm, row, record in zip(batch["arms"], batch["rows"], batch["records"]):
        mechanism = arm.get("mechanism") or batch.get("mechanism")
        settings = ml.normalize_setting_keys(arm["settings"])
        train = {key: row.get("train_" + key) for key in ("sharpe", "fitness")}
        if row.get("role") == "diagnostic" or arm.get("design") == "control":
            omitted["control_or_diagnostic"] += 1
        elif (row.get("metrics_pending")
              or row.get("status") not in alpha_registry.FINISHED_STATUSES | {"UNSUBMITTED", "ACTIVE", "SUBMITTED"}
              or not ml.valid_simulation_id(row.get("alpha_id"))
              or not all(experiments._finite_metric(value) and value > 0 for value in train.values())):
            omitted["no_positive_completed_train_evidence"] += 1
        elif any(settings.get(key) is None for key in ml.DEFAULT_SETTINGS):
            omitted["incomplete_settings"] += 1
        elif ml.is_python_language(settings["language"]) or not isinstance(mechanism, str) or not mechanism.strip():
            omitted["unsupported_source_or_missing_mechanism"] += 1
        else:
            scope = experiments._settings_signature(settings)
            split = record.get("out_of_sample") or {}
            window = tuple(str((split.get(part) or {}).get("startDate") or "") for part in ("train", "test"))
            groups[(mechanism, scope, window)].append({
                "alpha_id": row["alpha_id"], "label": arm["label"], "expression": arm["expression"],
                "settings": settings, "train": train,
            })
    return groups, omitted


def _observed_by_settings(entries, scopes, datasets, omitted):
    observed = defaultdict(set)
    for entry in entries:
        expression = entry.get("expression")
        if not expression or not experiments._has_finished_result(entry):
            continue
        try:
            settings = ml.normalize_setting_keys(entry.get("settings"))
        except (AttributeError, ValueError):
            omitted["history_settings_unusable"] += 1
            continue
        scope = experiments._settings_signature(settings)
        if scope in scopes:
            point = alpha_unit.split(expression, datasets=datasets)
            if point is not None:
                observed[scope].add(point.canonical)
    return observed


def _point_evidence(entries, datasets):
    by_point = {}
    for entry in sorted(entries, key=lambda item: (item["alpha_id"], item["label"])):
        point = alpha_unit.split(entry["expression"], datasets=datasets)
        if point is not None:
            by_point.setdefault(point.canonical, entry)
    return by_point


def _novel_children(parent, corpus, observed):
    unit = corpus.units[parent.unit]
    mutations = alpha_unit.derive(parent, grid=unit.grid, observed=observed)
    crossings = alpha_unit.crossover(unit, corpus, observed=observed, reference=parent)
    variants = {}
    for variant in mutations + crossings:
        if variant.rule in {"data", "port"} and variant.status == "novel":
            variants.setdefault(variant.canonical, variant)
    return variants


def generation(batch, history, *, seed, limit, datasets=None):
    """Randomize one of two parent choices within each mechanism/settings stratum."""
    datasets = alpha_unit.load_field_datasets() if datasets is None else datasets
    groups, omitted = _parent_groups(batch)
    observed = _observed_by_settings(history, {scope for _, scope, _ in groups}, datasets, omitted)
    for (_, scope, _), entries in groups.items():
        observed[scope].update(_point_evidence(entries, datasets))
    arms, comparisons = [], []
    contrasts = 0
    for (mechanism, scope, window), entries in sorted(groups.items()):
        if len(comparisons) >= limit:
            omitted["limit"] += 1
            continue
        by_point = _point_evidence(entries, datasets)
        if len(by_point) < 2:
            omitted["fewer_than_two_distinct_parents"] += 1
            continue
        corpus = alpha_unit.build(by_point.values(), datasets=datasets)
        available = {point.canonical: _novel_children(point, corpus, observed[scope]) for point in corpus.points}
        eligible = [point for point in corpus.points if available[point.canonical]]
        if len(eligible) < 2:
            omitted["fewer_than_two_parents_with_novel_children"] += 1
            continue
        selected = min(eligible, key=lambda point: (
            -by_point[point.canonical]["train"]["fitness"],
            -by_point[point.canonical]["train"]["sharpe"], point.canonical))
        lottery = Random(f"{seed}|{mechanism}|{scope}").choice(
            sorted(eligible, key=lambda point: point.canonical))
        parents = {"selected": selected, "lottery": lottery}
        physical = {}
        group = {"mechanism": mechanism,
                 "train_start": window[0] or None, "test_start": window[1] or None,
                 "eligible": [{key: by_point[point.canonical][key] for key in ("alpha_id", "label", "train")}
                              for point in eligible]}
        for policy, parent in parents.items():
            variants = available[parent.canonical]
            # Identical parents have identical proposal order in both arms.
            rng = Random(f"{seed}|{scope}|{parent.canonical}")
            child = variants[rng.choice(sorted(variants))]
            evidence = by_point[parent.canonical]
            donor = by_point[child.donor]["alpha_id"] if child.donor is not None else None
            if child.canonical not in physical:
                physical[child.canonical] = f"breed-{len(arms) + 1:02d}"
                arms.append({"label": physical[child.canonical], "expression": child.expression,
                             "settings": evidence["settings"], "mechanism": mechanism})
            group[policy] = {"label": physical[child.canonical], "parent_alpha_id": evidence["alpha_id"],
                             "parent_label": evidence["label"], "train": evidence["train"],
                             "donor_alpha_id": donor, "operation": child.rule, "move": child.detail}
        if group["selected"]["label"] != group["lottery"]["label"]:
            contrasts += 1
        comparisons.append(group)
        observed[scope].update(physical)
    if not contrasts:
        raise CliError("INVALID_INPUT", "no contrasting pair of novel children in the saved batch",
                       "inspect `details.omitted` or pass a batch with completed FASTEXPR training arms",
                       details={"matched_pairs": len(comparisons), "omitted": dict(omitted)})
    return arms, {"scope": "train", "rank_metric": "train_fitness", "seed": seed, "limit_per_policy": limit,
                  "proposals_per_policy": len(comparisons), "contrasting_pairs": contrasts,
                  "shared_simulations": 2 * len(comparisons) - len(arms),
                  "comparisons": comparisons, "omitted": dict(omitted)}
