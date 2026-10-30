#!/usr/bin/env python3
"""Repair two defects found in the ledger on 2026-10-30, without losing history.

Defect 1 -- nine expressions carry a variant index glued to their tail
(``... * -1) 14``). They came from one batch on 2026-10-21 23:28, none of them ever
received an alpha id, and stripping the index does not match any other record, so
they represent work that never ran in any form. The index is removed and the row is
annotated rather than deleted: "this formula was attempted" is information, and a
deleted row silently becomes "never tried".

Defect 2 -- 191 rows have no alpha id at all (QUEUED / PENDING / ERROR), all dated
2026-10-18..23, none younger than a week, so none can still be in flight. They are
marked ``abandoned`` instead of removed, for the same reason.

Both edits are additive except the expression itself, the stored hash and the key,
which must be recomputed when the expression changes. Anything else is left alone.

Usage: ``python3 tools/repair_ledger.py [--apply]``  (dry run without --apply)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import fastexpr_ast  # noqa: E402

TRAILING_INDEX = re.compile(r"\s+\d+\s*$")
ABANDONED_REASON = ("recorded at batch submission; never received an alpha id and "
                    "is older than any in-flight simulation (marked 2026-10-30)")


def _load(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parses(expression):
    try:
        fastexpr_ast.parse(expression)
        return True
    except Exception:  # noqa: BLE001 - any parse failure counts as broken
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="write the changes; without this it is a dry run")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()

    path = args.path or str(alpha_registry.default_registry_path())
    rows = _load(path)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    repaired, abandoned = [], []
    for row in rows:
        expression = row.get("expression") or ""
        if expression and not _parses(expression) and TRAILING_INDEX.search(expression):
            candidate = TRAILING_INDEX.sub("", expression)
            if _parses(candidate):
                row["expression"] = candidate
                row["expression_hash"] = alpha_registry.expression_hash(candidate)
                if not row.get("alpha_id"):
                    row["key"] = alpha_registry.placeholder_key(
                        candidate, row.get("settings") or {})
                row["data_repair"] = {
                    "kind": "trailing_index_removed",
                    "previous_tail": expression[-12:],
                    "repaired_at": stamp,
                    "note": ("variant index had been concatenated onto the formula; "
                             "this row never produced an alpha id"),
                }
                repaired.append(candidate)

        if not row.get("alpha_id") and not row.get("abandoned"):
            row["abandoned"] = True
            row["abandoned_reason"] = ABANDONED_REASON
            row["abandoned_at"] = stamp
            abandoned.append(row.get("key") or row.get("expression_hash"))

    print("rows              : %d" % len(rows))
    print("expressions fixed : %d" % len(repaired))
    print("marked abandoned  : %d" % len(abandoned))

    if not args.apply:
        print("\ndry run: nothing written. Re-run with --apply.")
        return 0

    temp = path + ".tmp"
    with open(temp, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)
    print("\nwritten: %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())