#!/usr/bin/env python3
"""Draw the family graph one component per cell, so the structure is readable.

A single spring layout over 48 nodes with 183 edges is a hairball: the large
family collapses into an unreadable blob and the singletons get pushed to the
margins, which hides the only two facts worth seeing -- how many families there
are, and how lopsided their sizes are.

Laying out each component separately and packing the results into a grid fixes
that: cell area is proportional to family size, so a 18-member family and a
singleton look like what they are.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

import alpha_registry  # noqa: E402
import family_graph as fg  # noqa: E402

EDGE_STYLE = {"sibling": ("dashed", 0.8, 0.45),
              "extends": ("solid", 1.6, 0.9),
              "shares": ("dotted", 0.7, 0.30)}


def load_entries(api_book: bool):
    entries = alpha_registry.load_entries()
    if api_book:
        import machine_lib as ml

        session = ml.login()
        response = session.get("https://api.worldquantbrain.com/users/self/alphas",
                               params={"status": "ACTIVE", "limit": 100})
        live = {item["id"] for item in response.json().get("results", [])}
        entries = [e for e in entries if e.get("alpha_id") in live]
    else:
        entries = [e for e in entries if not e.get("abandoned")]
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/data/worldquant-research/family-graph.png")
    parser.add_argument("--api-book", action="store_true")
    parser.add_argument("--max-cells", type=int, default=24)
    args = parser.parse_args()

    nodes = fg.build_nodes(load_entries(args.api_book))
    edges = fg.build_edges(nodes)
    groups = sorted(fg.components(nodes, edges), key=len, reverse=True)
    shown = groups[:args.max_cells]
    hidden = len(groups) - len(shown)

    columns = 4
    rows = max(1, math.ceil(len(shown) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for index, group in enumerate(shown):
        axis = axes[index]
        members = set(group)
        graph = nx.Graph()
        graph.add_nodes_from(members)
        for left, right, kind in edges:
            if left in members and right in members:
                graph.add_edge(left, right, kind=kind)
        position = (nx.spring_layout(graph, seed=11, k=0.7, iterations=120)
                    if len(members) > 1 else {group[0]: (0.0, 0.0)})

        for kind, (style, width, alpha) in EDGE_STYLE.items():
            subset = [(u, v) for u, v, k in graph.edges(data="kind") if k == kind]
            nx.draw_networkx_edges(graph, position, edgelist=subset, ax=axis,
                                   style=style, width=width, alpha=alpha,
                                   edge_color="#555555")

        sizes, colors = [], []
        for node in members:
            fitness = nodes[node]["fitness"]
            sizes.append(160 + 700 * (fitness or 0.0))
            colors.append("#4C78A8" if fitness is None else
                          (("#54A24B" if fitness >= 1.0 else "#F58518")))
        nx.draw_networkx_nodes(graph, position, ax=axis, node_size=sizes,
                               node_color=colors, linewidths=0.8,
                               edgecolors="#222222")
        nx.draw_networkx_labels(graph, position, ax=axis, font_size=6)

        best = max((nodes[node]["fitness"] or 0.0) for node in members)
        axis.set_title("family %d  n=%d  best F=%.2f"
                       % (index + 1, len(members), best), fontsize=9)
        axis.axis("off")

    for axis in axes[len(shown):]:
        axis.axis("off")

    figure.suptitle("book family graph: %d alphas -> %d construction families "
                    "(largest %d)%s"
                    % (len(nodes), len(groups), len(groups[0]),
                       ", %d small families not shown" % hidden if hidden else ""),
                    fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(args.out, dpi=150)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())