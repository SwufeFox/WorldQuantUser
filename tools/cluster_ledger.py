#!/usr/bin/env python3
"""Cluster ledger alphas by daily-PnL correlation.

Why this exists: family is a coarse bucket (11 values) and a family can hold one
mechanism or thirty. "How many separate bets do I actually have" is a question
about correlations, not about labels, and the platform's SELF_CORRELATION returns
one opaque number per alpha rather than a structure.

Two numbers are reported on purpose, because neither alone is honest:

* **Connected components at the threshold** merge transitively: A~B and B~C makes
  A, B and C one group even when A and C are unrelated. That *overstates* how
  unified a group is, and it also overstates how many separate groups exist when
  chained through a weak link. It is an upper bound on independence.
* **Greedy maximal independent set** picks alphas that are pairwise below the
  threshold, strongest first. It is a lower bound: a different picking order finds
  a different (sometimes larger) set.

The gap between them is the honest uncertainty, so both are printed with the
members of the component and of the independent set.

``--mode fields`` answers a different question on the same pool: not "do these move
together" but "are these built from the same information". Two alphas can be
uncorrelated in PnL and still read the same fields — the option pair scored 0.88
field overlap while sitting below the 0.7 PnL bar — so PnL independence alone
overstates how many mechanisms a pool really holds. This mode needs no PnL and no
network.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import correlation  # noqa: E402
import machine_lib as ml  # noqa: E402
import research_paths  # noqa: E402


def load_field_ids(path):
    """Datafield ids from the local index, longest first for greedy matching."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    ids = [item.get("id") for item in payload
           if isinstance(item, dict) and item.get("id")]
    ids.sort(key=len, reverse=True)
    return ids


def field_pattern(field_ids):
    """Regex matching a whole datafield id, not a fragment of a longer name.

    The boundaries matter: ``cap`` is a real field and also a substring of dozens
    of others, so an unbounded match would claim nearly every expression shares it.
    """
    return re.compile(r"(?<![A-Za-z0-9_])(" +
                      "|".join(re.escape(name) for name in field_ids) +
                      r")(?![A-Za-z0-9_])")


def load_pnls(session, alpha_ids, cache_dir, pause=0.0):
    """Fetch daily PnL for each id, reusing the cache when present."""
    out, failed = {}, []
    for index, alpha_id in enumerate(alpha_ids, 1):
        series = ml.fetch_daily_pnl(session, alpha_id, cache_dir=cache_dir)
        if series:
            out[alpha_id] = series
        else:
            failed.append(alpha_id)
        if pause:
            time.sleep(pause)
        if index % 25 == 0:
            print("  fetched %d/%d" % (index, len(alpha_ids)), flush=True)
    return out, failed


def components(ids, pairs, threshold):
    """Single-linkage groups: union-find over pairs at or above the threshold."""
    parent = {name: name for name in ids}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for pair in pairs:
        if abs(pair["r"]) >= threshold:
            left, right = find(pair["a"]), find(pair["b"])
            if left != right:
                parent[left] = right

    groups = {}
    for name in ids:
        groups.setdefault(find(name), []).append(name)
    return sorted(groups.values(), key=len, reverse=True)


def components_by(ids, similarity, threshold):
    """Union-find over any pairwise similarity above the threshold."""
    parent = {name: name for name in ids}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1:]:
            if similarity(left, right) >= threshold:
                a, b = find(left), find(right)
                if a != b:
                    parent[a] = b
    groups = {}
    for name in ids:
        groups.setdefault(find(name), []).append(name)
    return sorted(groups.values(), key=len, reverse=True)


def independent_set(ids, pairs, threshold, priority):
    """Greedy maximal set whose members are pairwise below the threshold."""
    conflicts = {name: set() for name in ids}
    for pair in pairs:
        if abs(pair["r"]) >= threshold:
            conflicts[pair["a"]].add(pair["b"])
            conflicts[pair["b"]].add(pair["a"])
    order = sorted(ids, key=lambda name: priority.get(name, 0.0), reverse=True)
    chosen = []
    for name in order:
        if any(other in conflicts[name] for other in chosen):
            continue
        chosen.append(name)
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-fitness", type=float, default=1.25)
    parser.add_argument("--mode", choices=["pnl", "fields"], default="pnl",
                        help="pnl: do they move together; fields: are they built "
                             "from the same information.")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--min-overlap", type=int, default=200)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--out", default=None, help="Write the full result as JSON.")
    parser.add_argument("--drop-common", type=float, default=0.5,
                        help="In fields mode, ignore fields present in more than "
                             "this fraction of the pool: cap or assets appear almost "
                             "everywhere and would make every pair look related.")
    parser.add_argument("--fields-index", default=None,
                        help="Path to the datafield index (fields_live.json).")
    parser.add_argument("--limit-members", type=int, default=6)
    args = parser.parse_args(argv)

    path = args.registry or alpha_registry.default_registry_path()
    entries = alpha_registry.load_entries(path)
    by_id = {}
    for entry in entries:
        alpha_id = entry.get("alpha_id")
        fitness = (entry.get("metrics") or {}).get("fitness")
        if not alpha_id or not isinstance(fitness, (int, float)):
            continue
        if fitness < args.min_fitness:
            continue
        if entry.get("duplicate_of"):
            # An id whose PnL equals another id's is one strategy, not two; leaving
            # it in inflates the header count and lets the greedy set pick the same
            # bet twice under different ids.
            continue
        if alpha_id not in by_id:
            by_id[alpha_id] = entry

    ids = sorted(by_id)
    print("candidates: %d alphas with fitness >= %s" % (len(ids), args.min_fitness))
    if len(ids) < 2:
        print("nothing to compare")
        return 0

    if args.mode == "fields":
        index_path = args.fields_index or os.path.join(
            research_paths.ensure_dir(), "index", "fields_live.json")
        if not os.path.exists(index_path):
            raise SystemExit(
                "no field index at %s; build one with the catalog, or pass "
                "--fields-index" % index_path)
        pattern = field_pattern(load_field_ids(index_path))
        sets = {}
        for alpha_id in ids:
            found = set(pattern.findall(by_id[alpha_id].get("expression") or ""))
            if found:
                sets[alpha_id] = found
        print("expressions with recognised fields: %d/%d" % (len(sets), len(ids)))
        # Ubiquitous fields carry no discrimination: if nine alphas in ten use
        # `cap`, its presence says nothing about whether two of them share an
        # information source, and it inflates every Jaccard towards one.
        frequency = {}
        for found in sets.values():
            for name in found:
                frequency[name] = frequency.get(name, 0) + 1
        limit = args.drop_common * len(sets)
        dropped = sorted((name for name, count in frequency.items()
                          if count > limit), key=lambda name: -frequency[name])
        if dropped:
            print("ignoring %d field(s) present in more than %d%% of the pool: %s"
                  % (len(dropped), int(args.drop_common * 100),
                     ", ".join("%s(%d)" % (name, frequency[name])
                               for name in dropped[:8])))
        sets = {name: (value - set(dropped)) for name, value in sets.items()}
        empty = [name for name, value in sets.items() if not value]
        for name in empty:
            sets.pop(name)
        if empty:
            print("no distinctive field left in %d alpha(s): %s"
                  % (len(empty), ", ".join(empty[:6])))

        def jaccard(left, right):
            a, b = sets[left], sets[right]
            return len(a & b) / len(a | b) if a and b else 0.0

        groups = components_by(list(sets), jaccard, args.threshold)
        shared = sum(1 for left, right in itertools.combinations(sets, 2)
                     if sets[left] & sets[right])
        total_pairs = len(sets) * (len(sets) - 1) // 2
        print()
        print("pairs sharing at least one field: %d/%d" % (shared, total_pairs))
        print("field-overlap groups at Jaccard >= %.2f: %d"
              % (args.threshold, len(groups)))
        for group in groups:
            families = sorted({str((by_id[name].get("family"))) for name in group})
            head = ", ".join(group[:4]) + ("..." if len(group) > 4 else "")
            print("  size %-3d [%s]  %s"
                  % (len(group), ",".join(families), head))
        result = {
            "mode": "fields",
            "threshold": args.threshold,
            "candidates": len(ids),
            "with_fields": len(sets),
            "pairs_sharing": shared,
            "pairs_total": total_pairs,
            "group_count": len(groups),
            "groups": [{"size": len(g), "members": g} for g in groups],
            "fields": {name: sorted(value) for name, value in sets.items()},
        }
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=1)
            print()
            print("wrote", args.out)
        return 0

    session = ml.load_session() or ml.authenticate()
    pnl_dir = os.path.join(research_paths.ensure_dir(), "pnl")
    pnls, failed = load_pnls(session, ids, pnl_dir)
    print("usable: %d, no PnL: %d" % (len(pnls), len(failed)))

    matrix = correlation.matrix(pnls, min_overlap=args.min_overlap)
    pairs = matrix["pairs"]
    priority = {name: (by_id[name].get("metrics") or {}).get("fitness", 0.0)
                for name in pnls}

    groups = components(list(pnls), pairs, args.threshold)
    chosen = independent_set(list(pnls), pairs, args.threshold, priority)
    near = sorted([p for p in pairs if abs(p["r"]) >= 0.9],
                  key=lambda p: -abs(p["r"]))

    def describe(name):
        entry = by_id.get(name) or {}
        return {
            "alpha_id": name,
            "fitness": (entry.get("metrics") or {}).get("fitness"),
            "family": entry.get("family"),
            "mechanism": entry.get("mechanism"),
        }

    print()
    print("pairs measured: %d (skipped for short overlap: %d)"
          % (matrix["measured"], len(matrix["skipped"])))
    print("connected components at |r| >= %.2f: %d" % (args.threshold, len(groups)))
    for group in groups[: args.limit_members + 1]:
        head = describe(group[0])
        print("  size %-3d  %s  %s"
              % (len(group), head["alpha_id"],
                 ", ".join(str(describe(n)["family"]) for n in group[:args.limit_members])))
    if len(groups) > args.limit_members + 1:
        print("  ... %d more groups" % (len(groups) - args.limit_members - 1))

    print()
    print("greedy independent set: %d alphas" % len(chosen))
    for name in chosen[:20]:
        row = describe(name)
        print("  %-10s fitness=%-5s family=%s"
              % (row["alpha_id"], row["fitness"], row["family"]))
    if len(chosen) > 20:
        print("  ... %d more" % (len(chosen) - 20))

    if near:
        print()
        print("near-duplicates (|r| >= 0.90): %d pairs, worst first" % len(near))
        for pair in near[:10]:
            print("  %.4f  %s ~ %s" % (pair["r"], pair["a"], pair["b"]))

    result = {
        "threshold": args.threshold,
        "min_overlap": args.min_overlap,
        "candidates": len(ids),
        "usable": len(pnls),
        "no_pnl": failed,
        "pairs_measured": matrix["measured"],
        "components": [{"size": len(group), "members": group} for group in groups],
        "component_count": len(groups),
        "independent_set": chosen,
        "near_duplicates": near,
        "members": {name: describe(name) for name in pnls},
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=1)
        print()
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
