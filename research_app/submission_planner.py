"""Submission planner: pick an independent subset and an order, from evidence.

The planner does no research and no scoring of its own.  Weights come from the
caller (for example the marginal Sharpe each candidate would add to the current
book), correlations come from measured daily PnL, and the only computation here
is deterministic: a weighted maximum independent set on the conflict graph
defined by a correlation threshold, plus a submission order.

Two rules keep it honest:

* a candidate without a weight, or a pair without a measured correlation, is
  ``unknown`` -- it is never treated as conflict-free;
* the result is a suggestion with its evidence attached, not an authorization.
  Submitting still needs the platform checks and the user's word.
"""

from __future__ import annotations

import itertools
import math

DEFAULT_MAX_CORRELATION = 0.5


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _key(a, b):
    return (a, b) if a <= b else (b, a)


def conflict_graph(candidate_ids, correlations, max_correlation=DEFAULT_MAX_CORRELATION):
    """Undirected conflict edges plus the pairs that have no measurement.

    ``correlations`` maps ``(a, b)`` to a signed correlation.  A pair is in
    conflict when ``|rho| >= max_correlation``: a strongly negative correlation
    is the same position with the sign flipped, not an independent bet.
    """
    ids = list(dict.fromkeys(candidate_ids))
    edges = {}
    missing = []
    for a, b in itertools.combinations(ids, 2):
        rho = correlations.get((a, b), correlations.get(_key(a, b)))
        if not _finite_number(rho):
            missing.append(_key(a, b))
            continue
        if abs(rho) >= max_correlation:
            edges[_key(a, b)] = float(rho)
    return {"ids": ids, "edges": edges, "missing": missing, "threshold": max_correlation}


def _neighbours(graph):
    out = {}
    # Unmeasured pairs are never assumed independent: two candidates without a
    # measured correlation cannot both enter the same selected subset.
    for (a, b) in itertools.chain(graph["edges"], graph.get("missing") or ()):
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out


def weighted_independent_set(weights, graph):
    """Exact weighted maximum independent set by branch and bound.

    ``weights`` maps id -> float and must cover every candidate.  Returns the
    chosen ids, their total weight, and whether the search proved optimal.  The
    graph is sparse in practice (a book of candidates conflicts only inside a
    family), so the bound prunes most of the tree; ``limit`` is a node budget
    that turns the search into a documented approximate answer instead of a hang.
    """
    ids = [i for i in graph["ids"] if i in weights]
    if len(ids) != len(graph["ids"]):
        missing = sorted(set(graph["ids"]) - set(weights))
        return {"selected": [], "weight": None, "optimal": False,
                "reason": "no weight for %s" % ", ".join(missing)}
    neighbours = _neighbours(graph)
    order = sorted(ids, key=lambda i: (-weights[i], i))

    best = {"weight": 0.0, "selected": []}
    nodes = {"used": 0, "limit": 200000}

    def search(index, chosen, weight, blocked):
        nodes["used"] += 1
        if nodes["used"] > nodes["limit"]:
            return
        if weight > best["weight"]:
            best["weight"] = weight
            best["selected"] = list(chosen)
        if index >= len(order):
            return
        remaining = sum(weights[i] for i in order[index:] if i not in blocked and weights[i] > 0)
        if weight + remaining <= best["weight"]:
            return
        candidate = order[index]
        if candidate not in blocked and weights[candidate] > 0:
            chosen.append(candidate)
            search(index + 1, chosen, weight + weights[candidate],
                   blocked | neighbours.get(candidate, set()))
            chosen.pop()
        search(index + 1, chosen, weight, blocked)

    search(0, [], 0.0, frozenset())
    return {"selected": best["selected"], "weight": best["weight"],
            "optimal": nodes["used"] <= nodes["limit"], "nodes": nodes["used"]}


def plan(candidates, correlations, max_correlation=DEFAULT_MAX_CORRELATION):
    """Select an independent subset and order it for submission.

    ``candidates`` is a list of ``{"id": str, "weight": float}``.  Anything the
    planner cannot measure is reported under ``unknown_pairs`` and cannot be
    co-selected: an unmeasured pair is not evidence of independence.
    """
    weights = {}
    unnamed = []
    for item in candidates or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        weight = item.get("weight")
        if not _finite_number(weight):
            unnamed.append(item.get("id"))
            continue
        weights[item["id"]] = float(weight)
    if unnamed:
        return {"ok": False, "reason": "no weight for %s" % ", ".join(map(str, unnamed)),
                "selected": [], "order": [], "excluded": [], "unknown": []}

    graph = conflict_graph(list(weights), correlations or {}, max_correlation)
    solution = weighted_independent_set(weights, graph)
    if solution.get("reason"):
        return {"ok": False, "reason": solution["reason"], "selected": [],
                "order": [], "excluded": [], "unknown": graph["missing"]}

    selected = set(solution["selected"])
    missing_pairs = set(graph["missing"])
    excluded = []
    for candidate in graph["ids"]:
        if candidate in selected:
            continue
        conflicts = []
        for (a, b), rho in graph["edges"].items():
            if candidate not in (a, b):
                continue
            other = b if a == candidate else a
            if other in selected:
                conflicts.append((other, rho))
        unmeasured = sorted(
            other for other in selected
            if _key(candidate, other) in missing_pairs
        )
        entry = {"id": candidate, "weight": weights.get(candidate),
                 "conflicts": [{"id": other, "rho": rho} for other, rho in conflicts]}
        if unmeasured:
            entry["unmeasured_with"] = unmeasured
        excluded.append(entry)

    order = sorted(selected, key=lambda i: (-weights[i], i))
    return {
        "ok": True,
        "threshold": max_correlation,
        "selected": order,
        "order": order,
        "weight": solution["weight"],
        "optimal": solution["optimal"],
        "excluded": excluded,
        "unknown_pairs": graph["missing"],
        "evidence": {
            "weights": "caller-supplied",
            "correlations": "measured; unknown pairs are listed, never assumed",
            "graph_edges": len(graph["edges"]),
        },
        "note": ("order is by supplied weight; re-measure the marginal against "
                 "the book after each submission"),
    }
