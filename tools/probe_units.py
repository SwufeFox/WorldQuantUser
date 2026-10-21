#!/usr/bin/env python3
"""Measure a datafield's unit empirically, because the catalog does not declare one.

Why: forget the unit and every cross-field ratio is unverifiable, and the audit
correctly refuses to run it. Half of this campaign's blocked leads were blocked on
exactly that — footnote fields like contingent consideration or acquired goodwill,
where the description says "amount" and nothing says in what.

The measurement is one cheap simulation per field. ``add(ts_backfill(field, 252), 1)``
forces the simulator to reconcile the field's unit with a pure number:

* it completes, so the field is dimensionless;
* it fails with ``expected "Unit[...]", found "Unit[X]"``, which names the unit.

The trick was used by hand earlier to discover that one lease ladder mixes
dimensionless and price-denominated fields in a single table. This makes it
repeatable and keeps the answers in a file instead of in a transcript.

The output is deliberately a plain ``{field_id: unit}`` map, so the audit can take
it as data rather than growing another special case.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import machine_lib as ml  # noqa: E402

# The simulator reports ``expected "Unit[TSPrice:1]", found "Unit[]"`` for
# ``add(field, 1)``. The *expected* side is the field's unit and the *found* side is
# the bare scalar's, so reading "found" inverts every answer. That mistake was made
# once and caught only by printing the raw error, which is why the raw text is kept.
_UNIT_EXPECTED = re.compile(r'expected\s+"?(Unit\[[^\]]*\])"?')
_DIMENSIONLESS = "dimensionless"


def probe_expression(field_id):
    """Force a unit reconciliation: a field plus a pure number."""
    return "add(ts_backfill(%s, 252), 1)" % field_id


def interpret(record):
    """Turn one probe record into ``(unit, note)``.

    A unit mismatch can arrive as WARNING as well as ERROR, so the error text is
    inspected before the status: an alpha that merely carries a unit warning is not
    a failed probe.
    """
    error = str(record.get("error") or "")
    match = _UNIT_EXPECTED.search(error)
    if match:
        unit = match.group(1)
        if unit == "Unit[]":
            return _DIMENSIONLESS, "expected unit is empty; mismatch came from elsewhere"
        return unit, "simulator named the field unit as expected"
    if not error.strip():
        return _DIMENSIONLESS, "add with a scalar succeeded"
    if "unknown variable" in error.lower() or "not found" in error.lower():
        return None, "field does not exist for this account"
    return None, error[:200]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fields", required=True,
                        help="Comma-separated datafield ids.")
    parser.add_argument("--registry", default=None,
                        help="Unused; accepted so callers can pass it uniformly.")
    parser.add_argument("--out", default=None,
                        help="Write/merge the unit map here (default under results tree).")
    parser.add_argument("--keep-errors", action="store_true",
                        help="Write the raw probe errors even when all were resolved.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the probes without submitting them.")
    args = parser.parse_args(argv)

    fields = [part.strip() for part in args.fields.split(",") if part.strip()]
    if not fields:
        raise SystemExit("no fields given")

    import research_paths
    out_path = args.out or os.path.join(research_paths.ensure_dir(), "field-units.json")
    known = {}
    if os.path.exists(out_path):
        try:
            known = json.loads(open(out_path, encoding="utf-8").read())
        except ValueError:
            known = {}

    print("probing %d field(s); %d already measured" % (len(fields), len(known)))
    if args.dry_run:
        for name in fields:
            print("  %s -> %s" % (name, probe_expression(name)))
        return 0

    session = ml.load_session() or ml.authenticate()
    pairs = [(probe_expression(name), 4) for name in fields]
    records = ml.simulate_alphas(session, pairs, settings=dict(
        ml.DEFAULT_SETTINGS, universe="TOP3000", region="USA", delay=1,
        neutralization="SUBINDUSTRY", truncation=0.08),
        batch_size=min(3, len(pairs)), max_wait_seconds=400)

    measured, undetermined = {}, []
    for field_id, record in zip(fields, records):
        unit, note = interpret(record)
        status = (record.get("status") or "").upper()
        print("  %-46s %-22s %s" % (field_id[:46], unit or "-", note))
        if unit:
            measured[field_id] = unit
        else:
            undetermined.append({"field": field_id, "status": status, "note": note})

    known.update(measured)
    # Keep the raw text next to the verdicts: an interpretation that cannot be
    # checked is how a wrong unit gets recorded as fact.
    if undetermined or args.keep_errors:
        errors_path = out_path.replace(".json", "-errors.json")
        with open(errors_path, "w", encoding="utf-8") as handle:
            json.dump({item["field"]: item for item in undetermined},
                      handle, ensure_ascii=False, indent=1)
        print("raw errors written to", errors_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(known, handle, ensure_ascii=False, indent=1, sort_keys=True)
    print()
    print("measured %d, undetermined %d; %d total in %s"
          % (len(measured), len(undetermined), len(known), out_path))
    if undetermined:
        print("undetermined:", ", ".join(item["field"] for item in undetermined))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())