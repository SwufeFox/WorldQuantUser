"""Pure admission checks for a new Quick exploration batch.

This module neither reads caches nor contacts BRAIN. ``analyze_arms`` checks the
whole original input, before limits or dispatch subdivision. ``assess_datasets``
accepts only records the caller has already verified for the current account,
the corresponding arm's resolved scope, and freshness. User-declared ``family``
or ``dataset`` tags are deliberately not evidence.

These reports govern new submissions only. They must not gate recovery of an
already accepted or ambiguous submission.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json

import fastexpr_ast as fa
import machine_lib as ml


MAX_ARMS_PER_SKELETON = 8
MIN_VERIFIED_DATASETS = 3
_MAX_EXPANDED_NODES = 10000
_SCOPE_KEYS = ("region", "universe", "delay", "instrumentType")
_SIGNAL_TYPES = frozenset({"MATRIX", "VECTOR"})
_EXCLUDED_TYPES = frozenset({"GROUP", "UNIVERSE"})
_DIVERSITY_CODES = frozenset({"dataset_diversity_insufficient", "dataset_diversity_unverified"})


class _Unsupported(ValueError):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def _check_statement_order(expression):
    """Check information the existing Program AST intentionally does not retain.

The parser keeps the last output and collects assignments regardless of source
order. Reuse its lexer to reject earlier outputs or assignments after an output;
do not infer field names from source text.
    """
    tokens = fa._tokenize(fa._strip_comments(expression))
    index, has_output = 0, False
    while tokens[index].kind != "eof":
        token = tokens[index]
        if token.text == ";":
            index += 1
            continue
        if token.text == "@":
            # Annotation syntax has already been validated by fa.parse.
            index += 2
            if tokens[index].text == "(":
                depth = 1
                index += 1
                while depth:
                    depth += (tokens[index].text == "(") - (tokens[index].text == ")")
                    index += 1
            continue
        assignment = token.kind == "ident" and tokens[index + 1].text == "="
        if has_output:
            reason = "assignment_after_output" if assignment else "multiple_outputs"
            raise _Unsupported(reason, "Only assignments followed by one final signal are supported.")
        has_output = not assignment
        depth = 0
        while tokens[index].kind != "eof":
            token = tokens[index]
            if token.text == ";" and depth == 0:
                index += 1
                break
            depth += (token.text == "(") - (token.text == ")")
            index += 1


def _final_signal(expression):
    program = fa.parse(expression)
    _check_statement_order(expression)
    if program.signal is None:
        raise _Unsupported("missing_signal", "A final signal expression is required.")
    names = [name for name, _ in program.assignments]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise _Unsupported("duplicate_assignment", "Repeated local assignments: " + ", ".join(duplicates))
    if any(fa.is_named_literal(name) for name in names):
        raise _Unsupported("literal_assignment", "Named literals cannot be local assignment targets.")
    assignments = dict(program.assignments)
    dependencies = {
        name: {node.name for node in value.walk()
               if isinstance(node, fa.Field) and node.name in assignments}
        for name, value in program.assignments
    }
    visiting, visited = set(), set()

    def visit(name):
        if name in visiting:
            raise _Unsupported("cyclic_assignment", "Cyclic local assignment involving " + name)
        if name in visited:
            return
        visiting.add(name)
        for dependency in sorted(dependencies[name]):
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in names:
        visit(name)
    positions = {name: index for index, name in enumerate(names)}
    if any(positions[dependency] > positions[name]
           for name in names for dependency in dependencies[name]):
        raise _Unsupported("forward_reference", "Forward local references have ambiguous assignment semantics.")

    budget = _MAX_EXPANDED_NODES

    def expand(node):
        nonlocal budget
        budget -= 1
        if budget < 0:
            raise _Unsupported("expansion_limit", "Final signal exceeds the supported AST expansion limit.")
        if isinstance(node, fa.Field):
            return expand(assignments[node.name]) if node.name in assignments else node
        if isinstance(node, (fa.Number, fa.String)):
            return node
        if isinstance(node, fa.Call):
            return fa.Call(node.name, tuple(expand(arg) for arg in node.args),
                           tuple((key, expand(value)) for key, value in node.kwargs))
        if isinstance(node, fa.BinOp):
            return fa.BinOp(node.op, expand(node.left), expand(node.right))
        if isinstance(node, fa.UnaryOp):
            return fa.UnaryOp(node.op, expand(node.operand))
        if isinstance(node, fa.Ternary):
            return fa.Ternary(expand(node.condition), expand(node.when_true), expand(node.when_false))
        raise _Unsupported("unsupported_node", "Unsupported node in final signal: " + type(node).__name__)

    return fa.Program(signal=expand(program.signal))


def analyze_arms(arms):
    """Analyze every input row without IO, settings-dependent grouping, or limits.

``expr`` and ``expression`` are accepted aliases. Settings resolve through
``machine_lib``; invalid settings are diagnostics, not an exception. The result
is JSON-safe and contains final-signal fields and inlined AST skeletons.
    """
    if not isinstance(arms, (list, tuple)):
        raise TypeError("arms must be a list or tuple")
    report = {
        "version": 1, "ok": True, "state": "verified",
        "rules": {"idea_required": True, "max_arms_per_skeleton": MAX_ARMS_PER_SKELETON,
                  "min_verified_datasets": MIN_VERIFIED_DATASETS},
        "arms": [], "skeleton_groups": [], "blocking": [],
    }
    groups = {}
    for index, arm in enumerate(arms):
        label = arm.get("label") if isinstance(arm, dict) else None
        item = {"index": index, "label": label if isinstance(label, str) else "arm-%d" % (index + 1),
                "scope": None, "language": None, "skeleton": None, "fields": []}
        report["arms"].append(item)

        def block(code, message, **details):
            report["blocking"].append({"code": code, "index": index, "label": item["label"],
                                       "message": message, **details})

        if not isinstance(arm, dict):
            block("invalid_arm", "Each arm must be an object.")
            continue
        if not isinstance(arm.get("idea"), str) or not arm["idea"].strip():
            block("idea_required", "A nonempty string idea is required before a new submission.")
        try:
            raw_settings = arm.get("settings")
            if raw_settings is not None and not isinstance(raw_settings, dict):
                raise ValueError("settings must be an object")
            settings = ml.resolve_settings(raw_settings)
            item["scope"] = {key: settings[key] for key in _SCOPE_KEYS}
            language = settings.get("language")
        except (TypeError, ValueError, AttributeError) as exc:
            block("invalid_settings", str(exc))
            language = ml.alpha_language(arm)
        item["language"] = language.upper() if isinstance(language, str) else None
        if item["language"] != "FASTEXPR":
            reason = "python" if ml.is_python_language(language) else "language"
            block("unsupported_expression", "New bulk exploration supports FASTEXPR AST policy only.", reason=reason)
            continue
        expression = arm.get("expr", arm.get("expression"))
        if "expr" in arm and "expression" in arm and arm["expr"] != arm["expression"]:
            block("unsupported_expression", "expr and expression disagree.", reason="conflicting_expression_aliases")
            continue
        if not isinstance(expression, str) or not expression.strip():
            block("unsupported_expression", "A nonempty expression is required.", reason="missing_expression")
            continue
        try:
            final = _final_signal(expression)
            item["skeleton"] = fa.skeleton(final)
            item["fields"] = fa.field_names(final)
        except _Unsupported as exc:
            block("unsupported_expression", str(exc), reason=exc.reason)
        except (fa.ParseError, RecursionError) as exc:
            block("unsupported_expression", str(exc), reason="parse_error" if isinstance(exc, fa.ParseError) else "ast_depth")
        if item["skeleton"] is not None:
            group = groups.setdefault(item["skeleton"], {"skeleton": item["skeleton"], "count": 0,
                                                          "indices": [], "labels": []})
            group["count"] += 1
            group["indices"].append(index)
            group["labels"].append(item["label"])
    report["skeleton_groups"] = list(groups.values())
    for group in report["skeleton_groups"]:
        if group["count"] > MAX_ARMS_PER_SKELETON:
            report["blocking"].append({"code": "skeleton_limit_exceeded", **group,
                                       "limit": MAX_ARMS_PER_SKELETON,
                                       "message": "The whole input exceeds the per-skeleton arm limit."})
    report["ok"] = not report["blocking"]
    report["state"] = "verified" if report["ok"] else "blocked"
    return report


def _dataset_id(record):
    dataset = record.get("dataset")
    if isinstance(dataset, dict):
        dataset = dataset.get("id")
    return dataset.strip() if isinstance(dataset, str) and dataset.strip() else None


def _field_evidence(field, signatures):
    result = {"field": field, "state": "unknown"}
    if not signatures:
        return {**result, "reason": "no_verified_record"}
    types = {kind for kind, _ in signatures}
    known_types = types - {None}
    if len(known_types) > 1:
        return {**result, "reason": "conflicting_field_types", "ambiguous": True}
    if None in types:
        return {**result, "reason": "missing_field_type"}
    kind = next(iter(types))
    if kind in _EXCLUDED_TYPES:
        return {**result, "state": "excluded", "type": kind}
    if kind not in _SIGNAL_TYPES:
        return {**result, "reason": "unrecognized_field_type", "type": kind}
    datasets = {dataset for _, dataset in signatures}
    if len(datasets - {None}) > 1:
        return {**result, "reason": "conflicting_dataset_ids", "ambiguous": True, "type": kind}
    if None in datasets:
        return {**result, "reason": "missing_dataset_id", "type": kind}
    return {**result, "state": "verified", "type": kind, "dataset": next(iter(datasets))}


def assess_datasets(analysis, records_by_arm):
    """Add dataset diversity to a structural report, without mutating it.

``records_by_arm`` has one list (or None) per analyzed arm, in input order.
The caller vouches for account, exact resolved scope and freshness of every
supplied record. Matching evidence is pooled only within equal scopes; conflicting
records for the same field cannot count twice. Unused fields and arm-declared
tags never contribute. GROUP/UNIVERSE are excluded even without a dataset ID.

Three verified dataset IDs prove the required lower bound despite other unknown
fields. With fewer than three, complete evidence means ``insufficient`` and
incomplete evidence means ``unknown``. Both block new admission.
    """
    arms = analysis["arms"]
    if not isinstance(records_by_arm, (list, tuple)) or len(records_by_arm) != len(arms):
        raise ValueError("records_by_arm must align one-to-one with analyzed arms")
    report = deepcopy(analysis)
    report["blocking"] = [item for item in report["blocking"] if item["code"] not in _DIVERSITY_CODES]
    scope_keys = [json.dumps(arm["scope"], sort_keys=True) if arm["scope"] is not None else None for arm in arms]
    wanted = {}
    for arm, scope in zip(arms, scope_keys):
        if scope is not None:
            wanted.setdefault(scope, set()).update(arm["fields"])
    evidence = {scope: {} for scope in wanted}
    for scope, records in zip(scope_keys, records_by_arm):
        if records is not None and not isinstance(records, (list, tuple)):
            raise ValueError("each records_by_arm item must be a verified record list or None")
        if scope is None:
            continue
        for record in records or ():
            if not isinstance(record, dict) or not isinstance(record.get("id"), str) or record["id"] not in wanted[scope]:
                continue
            kind = record.get("type")
            kind = kind.strip().upper() if isinstance(kind, str) and kind.strip() else None
            evidence[scope].setdefault(record["id"], set()).add((kind, _dataset_id(record)))
    datasets, unknown, ambiguous, excluded = set(), set(), set(), set()
    by_arm = []
    for arm, scope in zip(arms, scope_keys):
        details = [_field_evidence(field, evidence.get(scope, {}).get(field)) for field in arm["fields"]]
        arm_datasets = {item["dataset"] for item in details if item["state"] == "verified"}
        arm_unknown = {item["field"] for item in details if item["state"] == "unknown"}
        arm_ambiguous = {item["field"] for item in details if item.get("ambiguous")}
        arm_excluded = {item["field"] for item in details if item["state"] == "excluded"}
        datasets.update(arm_datasets)
        unknown.update(arm_unknown)
        ambiguous.update(arm_ambiguous)
        excluded.update(arm_excluded)
        by_arm.append({"index": arm["index"], "label": arm["label"], "scope": deepcopy(arm["scope"]),
                       "verified_datasets": sorted(arm_datasets), "unknown_fields": sorted(arm_unknown),
                       "ambiguous_fields": sorted(arm_ambiguous), "excluded_fields": sorted(arm_excluded),
                       "fields": details})
    complete = not unknown and all(arm["skeleton"] is not None and arm["scope"] is not None for arm in arms)
    state = "verified" if len(datasets) >= MIN_VERIFIED_DATASETS else "insufficient" if complete else "unknown"
    report["diversity"] = {"state": state, "verified_datasets": sorted(datasets), "verified_count": len(datasets),
                           "required_count": MIN_VERIFIED_DATASETS, "unknown_fields": sorted(unknown),
                           "ambiguous_fields": sorted(ambiguous), "excluded_fields": sorted(excluded), "by_arm": by_arm}
    if state != "verified":
        report["blocking"].append({"code": "dataset_diversity_insufficient" if state == "insufficient" else "dataset_diversity_unverified",
                                   "message": "At least three verified signal-field datasets are required.",
                                   "verified_count": len(datasets), "required_count": MIN_VERIFIED_DATASETS,
                                   "unknown_fields": sorted(unknown), "ambiguous_fields": sorted(ambiguous)})
    report["ok"] = not report["blocking"]
    report["state"] = "verified" if report["ok"] else "blocked"
    return report
