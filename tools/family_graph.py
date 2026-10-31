#!/usr/bin/env python3
"""Build the family graph over the ledger, from structure rather than correlation.

Correlation answers "do these two move together" at a cost of one request per pair
and with no explanation. The tree answers "are these the same construction" for
free, and it says exactly *how* they are related:

``sibling``   identical skeleton -- the same expression with different inputs.
              Salted variants land here, which is why one template can look like
              forty alphas in a count and like one bet in a component.
``extends``   A's leg set is a proper subset of B's: B is A plus a leg. This is the
              "this alpha is that alpha with something added" relation, and it is
              directional.
``shares``    two composites share at least one whole leg without containment.

Components over those edges are the honest unit: two nodes in one component are
related by construction, whatever their PnL says. That is deliberately *not* the
same claim as independence. Measured on 2026-10-30 against the live book: 48 alphas,
183 edges (83 siblings, 36 extends, 64 shares), collapsing to **32** construction
families -- largest 13, 28 singletons -- and candidates rebuilt from those same
families still correlate with the book at 0.85. Structure finds copies; only PnL
finds shared bets.

The family count is 32, not 22, because components use siblings only: an earlier
version also merged on ``extends`` and single linkage then chained unrelated
constructions through a shared ancestor (on the full ledger that produced a
467-member "family").

Usage:
    python3 tools/family_graph.py --book          # the submitted book
    python3 tools/family_graph.py --all           # every non-abandoned entry
    python3 tools/family_graph.py --all --dot out.dot
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import fastexpr_ast as fa  # noqa: E402


def legs(program):
    """One skeleton per assignment, which is what "sharing a leg" means."""
    return frozenset(fa.skeleton(fa.Program((), value)) for _k, value in program.assignments)


def build_nodes(entries):
    """One node per distinct alpha id (or key), carrying its parsed structure.

    Entries whose expression does not parse are skipped rather than raised: a single
    damaged row in the ledger must not stop the graph from being built.
    """

    nodes = {}
    for entry in entries:
        expression = entry.get("expression")
        if not expression:
            continue
        label = entry.get("alpha_id") or entry.get("key") or fa.skeleton(fa.parse(expression))
        if label in nodes:
            continue
        try:
            program = fa.parse(expression)
        except fa.ParseError:
            continue
        nodes[label] = {
            "id": label,
            "expression": expression,
            "skeleton": fa.skeleton(program),
            "fields": tuple(fa.field_names(program)),
            "operators": tuple(fa.operator_names(program)),
            "legs": legs(program),
            "fitness": (entry.get("metrics") or {}).get("fitness"),
            "status": entry.get("status"),
        }
    return nodes


def build_edges(nodes):
    """Every structural relation between two nodes, as ``(a, b, kind)``.

    Siblings are checked first and short-circuit, because an identical skeleton
    already implies a shared construction and reporting both would double-count the
    same fact.
    """

    edges = []
    ids = sorted(nodes)
    for index, left in enumerate(ids):
        a = nodes[left]
        for right in ids[index + 1:]:
            b = nodes[right]
            if a["skeleton"] == b["skeleton"]:
                edges.append((left, right, "sibling"))
                continue
            if a["legs"] and b["legs"]:
                if a["legs"] < b["legs"]:
                    edges.append((left, right, "extends"))
                elif b["legs"] < a["legs"]:
                    edges.append((right, left, "extends"))
                elif a["legs"] & b["legs"]:
                    edges.append((left, right, "shares"))
    return edges


def components(nodes, edges):
    """Families: connected components over **sibling** edges only.

    Two weaker relations are deliberately kept out, and both exclusions were forced
    by measurement rather than taste:

    * ``shares`` -- "one common leg" collapses almost everything into one component.
    * ``extends`` -- leg-set containment is transitive through a common parent, so
      single linkage merges siblings of a sibling: A with one leg, B = A + leg1,
      C = A + leg2 make B and C family although they share no construction. Measured
      on the full ledger that inflated one "family" to 467 of 951 expressions.

    Containment is still useful, but as lineage (which family derives from which),
    not as identity.
    """

    parent = {node: node for node in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for left, right, kind in edges:
        if kind == "sibling":
            union(left, right)
    groups = collections.defaultdict(list)
    for node in nodes:
        groups[find(node)].append(node)
    return list(groups.values())


def report(nodes, edges, stream=None):
    """Print the family summary and return it, so a caller can assert on it."""

    stream = stream or sys.stdout
    kinds = collections.Counter(kind for _l, _r, kind in edges)
    groups = components(nodes, edges)
    groups.sort(key=len, reverse=True)
    print("nodes %d  edges %d  (sibling %d, extends %d, shares %d)"
          % (len(nodes), len(edges), kinds["sibling"], kinds["extends"], kinds["shares"]),
          file=stream)
    print("components %d  largest %d  singleton components %d"
          % (len(groups), len(groups[0]) if groups else 0,
             sum(1 for group in groups if len(group) == 1)), file=stream)
    for group in groups[:8]:
        fitnesses = [nodes[node]["fitness"] for node in group
                     if nodes[node]["fitness"] is not None]
        best = max(fitnesses) if fitnesses else None
        print("\n  [%d] best fitness %s" % (len(group), best), file=stream)
        for node in group[:5]:
            print("      %-10s %s" % (node, nodes[node]["skeleton"][:88]), file=stream)
        if len(group) > 5:
            print("      ... %d more" % (len(group) - 5), file=stream)
    return {"nodes": len(nodes), "edges": len(edges), "kinds": dict(kinds),
            "components": len(groups), "groups": groups}


def to_dot(nodes, edges):
    """Graphviz export; edge style encodes the relation kind."""

    lines = ["graph alphas {", "  node [shape=box, fontsize=9];"]
    for node, data in nodes.items():
        label = "%s\\n%s" % (node, data["skeleton"][:40].replace('"', "'"))
        lines.append('  "%s" [label="%s"];' % (node, label))
    style = {"sibling": "style=dashed", "extends": "style=solid",
             "shares": "style=dotted,color=gray"}
    for left, right, kind in edges:
        lines.append('  "%s" -- "%s" [%s];' % (left, right, style[kind]))
    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", action="store_true", help="only ACTIVE alphas")
    parser.add_argument("--all", action="store_true", help="every non-abandoned entry")
    parser.add_argument("--api-book", action="store_true",
                        help="build over the live ACTIVE book instead of ledger status")
    parser.add_argument("--dot", default=None, help="write a Graphviz file here")
    parser.add_argument("--json", default=None, help="write the summary here")
    args = parser.parse_args()

    entries = alpha_registry.load_entries()
    if args.api_book:
        # The ledger's status column is written at submission time and is not
        # reconciled, so it lags the platform (33 marked ACTIVE against 48 live on
        # 2026-10-30). The graph should follow the platform, not the column.
        import machine_lib as ml

        session = ml.login()
        live, offset = set(), 0
        while True:
            response = session.get(
                "https://api.worldquantbrain.com/users/self/alphas",
                params={"status": "ACTIVE", "limit": 100, "offset": offset})
            payload = response.json() if response.text.strip() else {}
            page = payload.get("results", [])
            live |= {item["id"] for item in page}
            if len(page) < 100:
                break
            offset += 100
        by_id = {e.get("alpha_id"): e for e in entries if e.get("alpha_id")}
        entries = []
        for alpha_id in sorted(live):
            if alpha_id in by_id:
                entries.append(by_id[alpha_id])
            else:
                detail = session.get(
                    "https://api.worldquantbrain.com/alphas/" + alpha_id).json()
                entries.append({"alpha_id": alpha_id,
                                "expression": (detail.get("regular") or {}).get("code"),
                                "status": "ACTIVE"})
        print("live ACTIVE: %d (%d from the ledger, %d fetched)"
              % (len(entries), sum(1 for e in entries if e.get("expression") in
                                   {x.get("expression") for x in by_id.values()}),
                 sum(1 for e in entries if e.get("alpha_id") not in by_id)))
    elif args.book:
        entries = [e for e in entries if (e.get("status") or "").upper() == "ACTIVE"]
    elif args.all:
        entries = [e for e in entries if not e.get("abandoned")]
    else:
        entries = [e for e in entries if not e.get("abandoned")]

    nodes = build_nodes(entries)
    edges = build_edges(nodes)
    summary = report(nodes, edges)

    if args.dot:
        with open(args.dot, "w") as handle:
            handle.write(to_dot(nodes, edges))
        print("\nwrote %s" % args.dot)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"summary": {k: v for k, v in summary.items() if k != "groups"},
                       "groups": summary["groups"]}, handle, ensure_ascii=False, indent=1)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())