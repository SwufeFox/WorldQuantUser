#!/usr/bin/env python3
"""Backfill reason codes onto historical rejections.

`record_entry` attaches a cause to new rejections, but the 1513 rows already in the
ledger predate that, so "how many failures were low amplitude" is still unanswerable
by query. This fills them in.

Additive only: nothing is removed or reinterpreted. A row that already carries a
`reason_code` is left alone, so re-running after a classifier change does not silently
rewrite history -- if the classification improves, bump the rows deliberately with
`--overwrite` and the change is visible in the diff.

Dry run by default; --apply keeps a copy next to the file first.

Usage:
    python3 tools/backfill_reason_codes.py
    python3 tools/backfill_reason_codes.py --apply
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import rejection_reasons  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="reclassify rows that already carry a reason_code")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()

    path = args.path or str(alpha_registry.default_registry_path())
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    changed = 0
    distribution = collections.Counter()
    for row in rows:
        if row.get("verdict") != "rejected":
            continue
        if row.get("reason_code") and not args.overwrite:
            distribution[row["reason_code"]] += 1
            continue
        code, detail = rejection_reasons.classify(row)
        if code == "unclassified" and row.get("reason_code"):
            distribution[row["reason_code"]] += 1
            continue
        row["reason_code"] = code
        row["reason_detail"] = detail
        row["reason_recorded_at"] = stamp
        distribution[code] += 1
        changed += 1

    print("rows            : %d" % len(rows))
    print("codes written   : %d" % changed)
    print("distribution    :")
    for code, count in distribution.most_common():
        print("   %-22s %5d" % (code, count))

    if not args.apply:
        print("\ndry run: nothing written. Re-run with --apply.")
        return 0

    backup = path + ".before-reasons-" + stamp.replace(":", "")
    shutil.copy2(path, backup)
    temp = path + ".tmp"
    with open(temp, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)
    print("\nwritten %s\nbackup  %s" % (path, backup))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())