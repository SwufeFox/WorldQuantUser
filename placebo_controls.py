#!/usr/bin/env python3
"""Build a frozen, stratifiable control pool for matched-field placebos (Q2).

The problem this solves: swapping a load-bearing field for "a random same-dataset
field" is only a placebo if the substitutes are genuinely exchangeable. The first
attempt here drew seven fields that collapsed into four economic clusters -- two
fair-value items were two sides of the same disclosure, three M&A items came from
the same journal entry -- and the top of the null distribution was a single
near-neighbour pair. A percentile computed against that pool means nothing.

So the pool is built **before** any performance is looked at, from metadata only,
and every exclusion carries a reason. What this module can see: dataset, value
type, delay, coverage, descriptions. What it cannot see: the actual values, so
hard requirements from the design -- comparable value domain, comparable missing
pattern, comparable persistence -- stay explicitly `UNASSESSED` rather than being
silently assumed.

Two things it deliberately does **not** do:

* **It does not filter on usage.** `alphaCount == 0` looks neutral and is not: it
  selects for obscurity, which correlates with "the crowd found nothing here".
  Usage is recorded, never used to include or exclude.
* **It does not pick the winner.** Sampling returns a frozen list plus a seed; the
  caller must run exactly that list, in that order.

Near neighbours -- synonyms, mechanical transforms (`total`/`net`/`gross`,
`level1`/`level2`, `current`/`noncurrent`) and term-qualified variants
(`second_year`, `due_2yr`) -- are split out into `near_competitors`. They are
competition, not negative controls: they answer "is this field the best of its
family", not "does this field carry information".
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0])

RULES_VERSION = "2026-11-02.1"

# Words that carry no economic identity: they appear in almost every description.
STOP = frozenset("""
a an the of for in on at to from by with and or as is are was were be been being
that this these those it its their there which what when where how amount amounts
value values total gross sum number percentage rate company companies firm firms
during period periods fiscal year years quarter quarters end begin beginning
""".split())

# Words that qualify *how much* or *which slice* of the same concept, without
# changing which concept it is. Two descriptions that differ only here are the
# same measurement, so they are near neighbours, not independent controls.
QUALIFIERS = frozenset("""
net gross total average avg accumulated cumulative ending beginning current
noncurrent prior previous first second third fourth fifth level level1 level2 level3
due payable receivable paid unpaid deferred recognized reported disclosed
increase decrease changes change net_of before after including excluding
""".split())

_QUALIFIER_PATTERNS = (
    re.compile(r"\b(level|tier|class|stage)\s*\d\b"),
    re.compile(r"\b(first|second|third|fourth|fifth)\s+year\b"),
    re.compile(r"\b(two|three|four|five)\s+years?\b"),
    re.compile(r"\b(20\d\d)\b"),
    re.compile(r"\b(year|yr)\s*\d\b"),
)

_TOKEN = re.compile(r"[a-z0-9]+")

# Surface morphology hides real economic families: `acquire`, `acquired` and
# `acquisition` are three tokens for one concept, which is exactly how the first
# manual audit found an M&A cluster that a pure bag-of-words check had missed.
# A generic stemmer does not fix this (acquisition -> acquis, acquire -> acquir),
# so the families that matter here are listed, and anything unlisted falls back to
# plural stripping only.
_MORPH = {
    "acquire": "acquisition", "acquires": "acquisition", "acquired": "acquisition",
    "acquiring": "acquisition", "acquisitions": "acquisition", "acquisitive": "acquisition",
    "pay": "payment", "pays": "payment", "paid": "payment", "payments": "payment",
    "expense": "expense", "expenses": "expense", "expenditure": "expense",
    "intangibles": "intangible", "amortisation": "amortization",
    "amortized": "amortization", "amortised": "amortization",
    "leases": "lease", "leasehold": "lease", "leased": "lease",
    "liabilities": "liability", "assets": "asset", "revenues": "revenue",
    "impairments": "impairment", "impaired": "impairment",
    "obligations": "obligation", "provisions": "provision", "reserves": "reserve",
    "receivables": "receivable", "payables": "payable", "inventories": "inventory",
}


def normalize(words):
    out = set()
    for word in words:
        if word in _MORPH:
            out.add(_MORPH[word])
        elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            out.add(_MORPH.get(word[:-1], word[:-1]))
        else:
            out.add(word)
    return frozenset(out)


def _load_cached_fields(directory=None):
    """Every datafield the metadata snapshot knows about, without a session."""
    import glob
    import os
    root = directory or os.environ.get("MAGIX_FIELD_CACHE") or (
        "/data/worldquant-research/cache/metadata/datafields")
    fields = []
    for path in sorted(glob.glob(os.path.join(root, "*.json"))):
        try:
            payload = json.load(open(path)).get("payload")
        except Exception:
            continue
        rows = payload if isinstance(payload, list) else [payload]
        fields.extend(row for row in rows if isinstance(row, dict) and row.get("id"))
    return fields


def tokens(text: str) -> frozenset:
    lowered = (text or "").lower()
    for pattern in _QUALIFIER_PATTERNS:
        lowered = pattern.sub(" ", lowered)
    return normalize(word for word in _TOKEN.findall(lowered)
                     if word not in STOP and not word.isdigit())


def core_tokens(text: str) -> frozenset:
    """Identity tokens: qualifiers stripped, so `net sales` and `gross sales` match."""
    return frozenset(word for word in tokens(text) if word not in QUALIFIERS)


def jaccard(left: frozenset, right: frozenset) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def is_near(a: str, b: str, threshold: float = 0.5) -> tuple[bool, str]:
    """Same measurement measured twice? Returns (verdict, why)."""
    ta, tb = tokens(a), tokens(b)
    if ta and ta == tb:
        return True, "identical description tokens"
    core_a, core_b = core_tokens(a), core_tokens(b)
    if core_a and core_a == core_b:
        return True, "same core concept, different qualifier"
    score = jaccard(ta, tb)
    if score >= threshold:
        return True, "description jaccard %.2f" % score
    if core_a and core_b and jaccard(core_a, core_b) >= threshold:
        return True, "core jaccard %.2f" % jaccard(core_a, core_b)
    return False, ""


def profile(field: dict) -> dict:
    return {
        "id": field.get("id"),
        "description": field.get("description") or "",
        "dataset": (field.get("dataset") or {}).get("id"),
        "type": field.get("type"),
        "delay": field.get("delay"),
        "coverage": field.get("coverage"),
        "alpha_count": field.get("alphaCount"),
        "user_count": field.get("userCount"),
        "core": core_tokens(field.get("description") or ""),
    }


def build_pool(fields, target_id, coverage_band=0.05, near_threshold=0.5,
               cluster_threshold=0.35):
    """Frozen control pool for one target field.

    Hard filters are metadata only. `near_competitors` is reported separately and
    must not be used as negative controls.
    """
    by_id = {f["id"]: f for f in (profile(x) for x in fields) if f["id"]}
    target = by_id.get(target_id)
    if target is None:
        raise SystemExit("target field %s not in field set" % target_id)

    pool, near, rejected = [], [], []
    for field in by_id.values():
        if field["id"] == target_id:
            continue
        reason = None
        if field["dataset"] != target["dataset"]:
            reason = "dataset_mismatch:%s" % field["dataset"]
        elif field["type"] != target["type"]:
            reason = "type_mismatch:%s" % field["type"]
        elif field["delay"] != target["delay"]:
            reason = "delay_mismatch:%s" % field["delay"]
        elif abs((field["coverage"] or 0) - (target["coverage"] or 0)) > coverage_band:
            reason = "coverage_out_of_band:%.2f" % (field["coverage"] or 0)
        if reason:
            rejected.append({"id": field["id"], "reason": reason})
            continue
        verdict, why = is_near(target["description"], field["description"], near_threshold)
        if verdict:
            near.append({"id": field["id"], "why": why,
                         "description": field["description"][:90]})
        else:
            pool.append(field)

    # Single-linkage clustering on description similarity: fields that describe the
    # same disclosure family end up in one cluster so the sample cannot spend half
    # its budget inside one accounting note.
    parent = {field["id"]: field["id"] for field in pool}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for index, left in enumerate(pool):
        for right in pool[index + 1:]:
            if jaccard(left["core"], right["core"]) >= cluster_threshold:
                a, b = find(left["id"]), find(right["id"])
                if a != b:
                    parent[a] = b

    clusters = {}
    for field in pool:
        clusters.setdefault(find(field["id"]), []).append(field["id"])

    return {
        "rules_version": RULES_VERSION,
        "target": target,
        "pool": [field["id"] for field in pool],
        "near_competitors": near,
        "rejected": rejected,
        "clusters": {root: sorted(members) for root, members in sorted(clusters.items())},
        "unassessed": ["value_domain", "missing_pattern", "persistence",
                       "cross_stock_coverage", "fiscal_semantics"],
    }


def apply_review(pooled, review):
    """Merge/split clusters per a recorded human judgement.

    Surface similarity cannot see that `payments to acquire businesses`, `cash
    acquired in a business combination` and `costs related to an acquisition` are
    one accounting event -- they score 0.33 against each other and split into three
    clusters. The design requires that judgement to be made from the data dictionary
    *before* any performance is seen, so it lives in a review artifact with reasons,
    not in a threshold.
    """
    merges = review.get("merges") or []
    excluded = review.get("exclusions") or {}
    members = {field: root for root, group in pooled["clusters"].items() for field in group}
    for group in merges:
        roots = {members.get(name) for name in group if name in members}
        roots.discard(None)
        if not roots:
            continue
        anchor = sorted(roots)[0]
        for name in group:
            if name in members:
                members[name] = anchor
    clusters = {}
    for field, root in members.items():
        if field in excluded:
            continue
        clusters.setdefault(root, []).append(field)
    merged = dict(pooled)
    merged["clusters"] = {root: sorted(v) for root, v in sorted(clusters.items())}
    merged["review"] = {"merges": merges, "exclusions": excluded,
                        "reviewer": review.get("reviewer"),
                        "date": review.get("date")}
    return merged


def sample(pooled, draws, seed):
    """At most one field per cluster, deterministic order, no peeking at results."""
    rng = random.Random(seed)
    clusters = list(pooled["clusters"].values())
    rng.shuffle(clusters)
    chosen, shortfall = [], []
    for members in clusters:
        if len(chosen) >= draws:
            break
        options = sorted(members)
        chosen.append(rng.choice(options))
    if len(chosen) < draws:
        shortfall = ["LOW-CONTROL-COUNT: asked %d, %d independent clusters available"
                     % (draws, len(chosen))]
    return {"seed": seed, "draws_requested": draws, "chosen": chosen,
            "flags": shortfall}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--coverage-band", type=float, default=0.05)
    parser.add_argument("--review", default=None,
                        help="frozen cluster review JSON; without it the sample is provisional")
    parser.add_argument("--json", nargs="?", const="-", default=None)
    args = parser.parse_args()

    fields = _load_cached_fields()
    pooled = build_pool(fields, args.target, coverage_band=args.coverage_band)
    review = None
    if args.review:
        review = json.load(open(args.review))
        pooled = apply_review(pooled, review)
    picked = sample(pooled, args.draws, args.seed)
    status = "FROZEN" if review else "REVIEW_REQUIRED"
    picked["status"] = status

    payload = dict(pooled, sample=picked)
    if args.json:
        text = json.dumps(payload, ensure_ascii=False, indent=1, default=list)
        if args.json == "-":
            print(text)
        else:
            open(args.json, "w").write(text)
    print("status %s" % status)
    if not review:
        print("  !! cluster merges are a recorded judgement; run again with --review "
              "before using these draws")
    print("rules %s | target %s (cov %.2f, %s)"
          % (RULES_VERSION, pooled["target"]["id"], pooled["target"]["coverage"] or 0,
             pooled["target"]["dataset"]))
    print("eligible pool %d | near-competitors %d | rejected %d | clusters %d"
          % (len(pooled["pool"]), len(pooled["near_competitors"]),
             len(pooled["rejected"]), len(pooled["clusters"])))
    print("drew %d of %d requested (seed %d)"
          % (len(picked["chosen"]), args.draws, args.seed))
    for name in picked["chosen"]:
        print("  ", name)
    for flag in picked["flags"]:
        print("  !!", flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())