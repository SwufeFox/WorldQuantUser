#!/usr/bin/env python3
"""Select a diverse subset by clustering, and compare it with greedy selection.

Method from a widely-shared community post: turn each alpha's PnL into a feature
vector, run K-means with k equal to the number of alphas wanted, then take the best
alpha from each cluster. The claim is that K-means' objective separates styles on its
own, so the chosen set is both diverse and individually strong.

Two embeddings are built, because the post is vague about the vector and the choice
decides what "similar" means:

* ``pnl`` — the z-scored daily PnL series. Groups alphas by co-movement in levels.
* ``corr`` — each alpha's row of the pairwise correlation matrix. Groups alphas by
  who they move with, which is what the platform's SELF_CORRELATION actually measures.

A comment on the post makes the point this tool takes most seriously: do not look only
at the top score inside a cluster. If a cluster's members are all highly correlated
with each other, that whole direction is crowded and its representative is a crowded
bet; the tool therefore reports within-cluster mean absolute correlation next to each
pick.

It also reports the same set's statistics against a greedy pairwise selection, so the
comparison is measured rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import correlation  # noqa: E402
import research_paths  # noqa: E402


def load_pnls(ids, cache_dir):
    out = {}
    for alpha_id in ids:
        path = os.path.join(cache_dir, alpha_id + ".json")
        if not os.path.exists(path):
            continue
        try:
            series = json.loads(open(path, encoding="utf-8").read())
        except ValueError:
            continue
        if series:
            out[alpha_id] = {day: float(v) for day, v in series.items()}
    return out


def pnl_matrix(pnls):
    """z-scored daily PnL, aligned on the dates every alpha shares."""
    ids = sorted(pnls)
    shared = sorted(set.intersection(*(set(pnls[a]) for a in ids)))
    matrix = np.array([[pnls[a][d] for d in shared] for a in ids], dtype=float)
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return ids, (matrix - mean) / std, shared


def corr_matrix(pnls, min_overlap=200):
    """Pairwise Pearson correlation, plus each row as an embedding."""
    ids = sorted(pnls)
    n = len(ids)
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            x, y, _ = correlation.align(pnls[ids[i]], pnls[ids[j]], min_overlap)
            r = correlation.pearson(x, y) if x else 0.0
            m[i, j] = m[j, i] = 0.0 if r is None else r
    return ids, m


def kmeans(matrix, k, seed=42, restarts=20, iterations=300):
    """K-means with k-means++ seeding; returns labels and inertia."""
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    best = None
    for _ in range(restarts):
        # k-means++: spread the initial centroids by distance.
        centres = [matrix[rng.integers(n)]]
        for _ in range(k - 1):
            d = np.min(((matrix[:, None, :] - np.array(centres)[None, :, :]) ** 2)
                       .sum(axis=2), axis=1)
            total = d.sum()
            probs = np.full(n, 1.0 / n) if total <= 0 else d / total
            centres.append(matrix[rng.choice(n, p=probs)])
        centres = np.array(centres)
        labels = np.zeros(n, dtype=int)
        for _ in range(iterations):
            distances = ((matrix[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for index in range(k):
                members = matrix[labels == index]
                if len(members):
                    centres[index] = members.mean(axis=0)
        inertia = float(((matrix - centres[labels]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, labels.copy())
    return best[1], best[0]


def standardize(matrix):
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def mean_abs_corr(ids, cmatrix, members):
    """Within-cluster mean absolute correlation: how crowded the direction is."""
    if len(members) < 2:
        return 0.0
    idx = [ids.index(name) for name in members]
    values = [abs(cmatrix[i, j]) for a, i in enumerate(idx) for j in idx[a + 1:]]
    return float(np.mean(values)) if values else 0.0


def selected_stats(names, cmatrix, ids, fitness):
    """Pairwise |r| summary and mean fitness for a chosen set."""
    idx = [ids.index(n) for n in names]
    pairs = [abs(cmatrix[i, j]) for a, i in enumerate(idx) for j in idx[a + 1:]]
    return {
        "size": len(names),
        "mean_abs_corr": round(float(np.mean(pairs)), 4) if pairs else None,
        "max_abs_corr": round(float(np.max(pairs)), 4) if pairs else None,
        "mean_fitness": round(float(np.mean([fitness[n] for n in names])), 4),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-fitness", type=float, default=1.25)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--embedding", choices=["pnl", "corr", "both"], default="both")
    parser.add_argument("--score", choices=["fitness", "sharpe"], default="fitness")
    parser.add_argument("--min-overlap", type=int, default=200)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    path = args.registry or alpha_registry.default_registry_path()
    entries = [e for e in alpha_registry.unique_entries(alpha_registry.load_entries(path))
               if isinstance((e.get("metrics") or {}).get("fitness"), (int, float))
               and e["metrics"]["fitness"] >= args.min_fitness
               and e.get("alpha_id")]
    by_id = {e["alpha_id"]: e for e in entries}
    cache = os.path.join(research_paths.ensure_dir(), "pnl")
    pnls = load_pnls(sorted(by_id), cache)
    print("pool: %d alphas at fitness >= %s, %d with cached PnL"
          % (len(entries), args.min_fitness, len(pnls)))
    if len(pnls) < args.k:
        raise SystemExit("need at least k=%d alphas with PnL" % args.k)

    fitness = {a: float((by_id[a].get("metrics") or {}).get("fitness") or 0) for a in pnls}
    sharpe = {a: float((by_id[a].get("metrics") or {}).get("sharpe") or 0) for a in pnls}
    score = fitness if args.score == "fitness" else sharpe

    ids_c, cmatrix = corr_matrix(pnls, args.min_overlap)
    results = {}

    embeddings = {}
    if args.embedding in ("pnl", "both"):
        ids_p, pmatrix, shared = pnl_matrix(pnls)
        embeddings["pnl"] = (ids_p, standardize(pmatrix), "%d days" % len(shared))
    if args.embedding in ("corr", "both"):
        embeddings["corr"] = (ids_c, standardize(cmatrix), "%d peers" % len(ids_c))

    for name, (ids, matrix, shape) in embeddings.items():
        labels, inertia = kmeans(matrix, args.k)
        chosen, clusters = [], []
        for index in range(args.k):
            members = [ids[i] for i in range(len(ids)) if labels[i] == index]
            if not members:
                continue
            best = max(members, key=lambda a: score[a])
            chosen.append(best)
            clusters.append({
                "cluster": index,
                "size": len(members),
                "representative": best,
                "representative_fitness": round(fitness[best], 4),
                "within_mean_abs_corr": round(mean_abs_corr(ids, cmatrix, members), 4),
                "members": members,
            })
        stats = selected_stats(chosen, cmatrix, ids_c, fitness)
        results[name] = {"embedding": name, "features": shape,
                         "inertia": round(inertia, 2), "clusters": clusters,
                         "selected": chosen, "stats": stats}
        print()
        print("=== embedding: %s (%s) ===" % (name, shape))
        print("selected %d alphas: mean|r| %.4f, max|r| %.4f, mean fitness %.3f"
              % (stats["size"], stats["mean_abs_corr"] or 0,
                 stats["max_abs_corr"] or 0, stats["mean_fitness"]))
        for cluster in sorted(clusters, key=lambda c: -c["within_mean_abs_corr"]):
            print("  cluster %-2d size %-3d pick %-10s fitness %-5s within|r| %.3f"
                  % (cluster["cluster"], cluster["size"],
                     cluster["representative"], cluster["representative_fitness"],
                     cluster["within_mean_abs_corr"]))

    # Greedy pairwise selection on the same pool, for comparison.
    pairs = [{"a": ids_c[i], "b": ids_c[j], "r": float(cmatrix[i, j])}
             for i in range(len(ids_c)) for j in range(i + 1, len(ids_c))]
    threshold = 0.7
    conflicts = {name: set() for name in ids_c}
    for pair in pairs:
        if abs(pair["r"]) >= threshold:
            conflicts[pair["a"]].add(pair["b"])
            conflicts[pair["b"]].add(pair["a"])
    greedy = []
    for name in sorted(ids_c, key=lambda a: -score[a]):
        if any(other in conflicts[name] for other in greedy):
            continue
        greedy.append(name)
        if len(greedy) >= args.k:
            break
    greedy_stats = selected_stats(greedy, cmatrix, ids_c, fitness)
    print()
    print("=== greedy pairwise (|r| < %.2f), capped at k=%d ===" % (threshold, args.k))
    print("selected %d: mean|r| %.4f, max|r| %.4f, mean fitness %.3f"
          % (greedy_stats["size"], greedy_stats["mean_abs_corr"] or 0,
             greedy_stats["max_abs_corr"] or 0, greedy_stats["mean_fitness"]))
    print("  members:", ", ".join(greedy))

    payload = {"pool": len(pnls), "k": args.k, "score": args.score,
               "embeddings": results, "greedy": {"selected": greedy,
                                                 "stats": greedy_stats}}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
        print()
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())