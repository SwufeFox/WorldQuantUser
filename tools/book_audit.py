#!/usr/bin/env python3
"""Reconcile the ledger's verdicts against the actual book.

The ledger's `candidate` verdict means "passed the platform's static checks at simulation
time". It does not mean "entered the book": a candidate can still be refused at submission
for SELF_CORRELATION, and nothing rewrites the verdict afterwards. Measured on 2026-10-23,
159 distinct alphas carried `candidate` while 45 were actually ACTIVE - so the ledger
overstated the book by more than three times, and every hit-rate computed from it was wrong.

Three states, not one:

    research-qualified   metrics and static checks pass          ledger: candidate
    submittable          the platform would accept it            only knowable by trying
    in the book          dateSubmitted is set, status ACTIVE     the account's own list

This tool reads the account, joins it to the ledger, and reports the real numbers. It is
read-only; correcting verdicts is a separate, deliberate act.

Usage:
    python tools/book_audit.py                 # summary
    python tools/book_audit.py --by idea       # hit rate per research direction
    python tools/book_audit.py --stale         # candidates that never entered
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import alpha_registry as ar          # noqa: E402
import machine_lib as ml             # noqa: E402


def load_book(session):
    """The account's actual ACTIVE alphas, with the metrics it reports."""
    book = {}
    for entry in ml.list_alphas(session, limit=400, status="ACTIVE", alpha_type_not=""):
        alpha_id = entry.get("id")
        if alpha_id:
            book[alpha_id] = entry
    return book


def classify(entries, book):
    """Join ledger records to the book.

    Returns ``{alpha_id: {verdict, idea, mechanism, in_book, status}}`` keyed by alpha id,
    newest ledger record winning - the ledger is append-only, so the last word is the
    current one.
    """
    latest = {}
    for entry in entries:
        alpha_id = entry.get("alpha_id")
        if not alpha_id:
            continue
        latest[alpha_id] = entry          # later records overwrite earlier ones
    joined = {}
    for alpha_id, entry in latest.items():
        joined[alpha_id] = {
            "verdict": entry.get("verdict"),
            "idea": (entry.get("idea") or "").strip() or "(unlabelled)",
            "mechanism": entry.get("mechanism") or "",
            "in_book": alpha_id in book,
            "status": (book.get(alpha_id) or {}).get("status") or "NOT_ACTIVE",
            "sharpe": (entry.get("metrics") or {}).get("sharpe"),
            "fitness": (entry.get("metrics") or {}).get("fitness"),
        }
    return joined


def report(joined, group="idea"):
    """Summarise: what the ledger claims versus what the book holds."""
    verdicts = Counter(item["verdict"] for item in joined.values())
    candidates = {k: v for k, v in joined.items() if v["verdict"] == "candidate"}
    in_book = [k for k, v in joined.items() if v["in_book"]]
    stale = [k for k, v in candidates.items() if not v["in_book"]]

    buckets = defaultdict(lambda: {"candidates": 0, "in_book": 0})
    for alpha_id, item in joined.items():
        key = item[group][:40]
        if item["verdict"] == "candidate":
            buckets[key]["candidates"] += 1
        if item["in_book"]:
            buckets[key]["in_book"] += 1

    return {
        "ledger_alphas": len(joined),
        "verdicts": dict(verdicts),
        "marked_candidate": len(candidates),
        "actually_in_book": len(in_book),
        "candidate_but_not_in_book": len(stale),
        "book_entries_missing_from_ledger": sorted(set(k for k in joined if not joined[k]["in_book"]
                                                       and False)) or [],
        "by_group": {k: v for k, v in sorted(buckets.items(),
                                             key=lambda kv: -kv[1]["in_book"])},
        "stale_alpha_ids": sorted(stale),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--by", default="idea",
                        help="Group the hit rate by this ledger field (default idea).")
    parser.add_argument("--stale", action="store_true",
                        help="List candidates that never entered the book.")
    parser.add_argument("--save", default=None)
    args = parser.parse_args(argv)

    session = ml.load_session() or ml.authenticate()
    book = load_book(session)
    joined = classify(ar.load_entries(), book)
    data = report(joined, group=args.by)
    data["book_size"] = len(book)
    data["book_not_in_ledger"] = sorted(set(book) - set(joined))

    if args.save:
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=1)

    print(json.dumps(data if args.stale else {k: v for k, v in data.items()
                                              if k != "stale_alpha_ids"},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())