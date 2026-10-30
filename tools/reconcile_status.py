#!/usr/bin/env python3
"""Make the ledger's `status` column agree with the platform.

The column is written when a simulation is submitted and never revisited, so it
drifts in one direction: an alpha that later entered the book keeps saying COMPLETE
or UNSUBMITTED. Measured on 2026-10-30: the platform reported 48 ACTIVE, the ledger
claimed 33, and the difference was exactly 15 rows marked COMPLETE or UNSUBMITTED
that had in fact been submitted.

That drift is not cosmetic. Every count derived from the ledger -- how many alphas
the book holds, what the newest family is, whether a candidate would collide -- is
computed against that column.

What this does *not* do: invent a status. If the platform no longer lists an alpha
the ledger calls ACTIVE, the row is marked `not_in_book` with a timestamp instead of
being silently downgraded, because "it left the book" and "it was never in it" are
different facts and only one of them is knowable from here.

Dry run by default; --apply writes atomically after keeping a copy.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import machine_lib as ml  # noqa: E402

BASE = "https://api.worldquantbrain.com"


def fetch_active(session):
    """Every alpha the platform currently reports as submitted."""
    live, offset = {}, 0
    while True:
        response = session.get(BASE + "/users/self/alphas",
                               params={"status": "ACTIVE", "limit": 100, "offset": offset})
        payload = response.json() if response.text.strip() else {}
        page = payload.get("results", [])
        for item in page:
            live[item["id"]] = item
        if len(page) < 100:
            break
        offset += 100
    return live


def _load(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reconcile(rows, live, stamp):
    """Update rows in place; return ``(promoted_ids, demoted_ids)``.

    Pure enough to test: the only inputs are the ledger rows, the platform's ACTIVE
    map, and a timestamp. ``live`` maps alpha id to the platform's own record, which
    is where ``dateSubmitted`` comes from.
    """
    promoted, demoted = [], []
    for row in rows:
        alpha_id = row.get("alpha_id")
        if not alpha_id:
            continue
        status = (row.get("status") or "").upper()
        if alpha_id in live and status != "ACTIVE":
            row["status"] = "ACTIVE"
            row["status_source"] = "reconciled:api"
            row["reconciled_at"] = stamp
            submitted = live[alpha_id].get("dateSubmitted")
            if submitted:
                row["date_submitted"] = submitted
            promoted.append(alpha_id)
        elif alpha_id not in live and status == "ACTIVE":
            row["status_source"] = "reconciled:api"
            row["reconciled_at"] = stamp
            row["not_in_book"] = True
            demoted.append(alpha_id)
    return promoted, demoted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()

    path = args.path or str(alpha_registry.default_registry_path())
    rows = _load(path)
    live = fetch_active(ml.login())
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    promoted, demoted = reconcile(rows, live, stamp)

    print("live ACTIVE on the platform : %d" % len(live))
    print("rows promoted to ACTIVE      : %d" % len(promoted))
    print("rows marked not_in_book      : %d" % len(demoted))
    for alpha_id in promoted[:8]:
        print("   + %s" % alpha_id)
    for alpha_id in demoted[:8]:
        print("   - %s" % alpha_id)

    if not args.apply:
        print("\ndry run: nothing written. Re-run with --apply.")
        return 0

    backup = path + ".before-reconcile-" + stamp.replace(":", "")
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