#!/usr/bin/env python3
"""Expand a registered alpha template over fields.

The registry is the reusable structure: a formula with slots plus the evidence
that justifies keeping it. This tool turns (template, fields) into arms for
tools/batch_quick.py or a probe spec, so a structure is written once and
reused without a macro layer.

Examples:
    python3 tools/templates.py list
    python3 tools/templates.py expand short_vs_long_orthogonal \
        --fields mdl110_score,mdl110_quality --region GLB --universe TOP3000 \
        --out arms.json
    # VECTOR fields can be declared with a :vec suffix
    python3 tools/templates.py expand ts_ratio_trend --fields loan_utilization_ratio:vec
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "templates" / "registry.json"

BASE = {
    "instrumentType": "EQUITY", "delay": 1, "decay": 0,
    "neutralization": "SUBINDUSTRY", "truncation": 0.08,
    "pasteurization": "ON", "nanHandling": "ON", "unitHandling": "VERIFY",
    "language": "FASTEXPR", "visualization": False, "testPeriod": "P1Y",
}


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text())


def find(reg: dict, template_id: str) -> dict:
    for entry in reg["templates"]:
        if entry["id"] == template_id:
            return entry
    raise SystemExit(f"unknown template {template_id!r}; try `list`")


def cmd_list(reg: dict) -> int:
    for entry in reg["templates"]:
        slots = ", ".join(
            f"{k}={v}" if isinstance(v, list) else k for k, v in entry.get("slots", {}).items()
        )
        print(f"{entry['id']}")
        print(f"    {entry['name']}")
        print(f"    formula: {entry['formula']}")
        print(f"    slots: {slots or '-'}  operators={entry.get('operators')} fields={entry.get('fields')}")
        print(f"    turnover band: {entry.get('turnover_band')}")
        print(f"    evidence: {entry.get('evidence', '-')}")
        print()
    return 0


def cmd_expand(reg: dict, args) -> int:
    entry = find(reg, args.template)
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    if not fields:
        raise SystemExit("--fields is required")

    slot_values = {}
    for name, allowed in (entry.get("slots") or {}).items():
        given = getattr(args, name, None)
        if given:
            slot_values[name] = [given]
        elif isinstance(allowed, list):
            slot_values[name] = allowed[:1] if args.first_slot_only else allowed
        else:
            slot_values[name] = [allowed]

    keys = list(slot_values)
    combos = list(itertools.product(*[slot_values[k] for k in keys])) if keys else [()]

    arms, seen = [], set()
    for field_spec in fields:
        name, _, kind = field_spec.partition(":")
        reference = f"vec_avg({name})" if kind == "vec" else name
        for combo in combos:
            mapping = dict(zip(keys, combo))
            mapping["f"] = reference
            try:
                expression = entry["formula"].format(**mapping)
            except KeyError as exc:
                raise SystemExit(f"template {entry['id']} needs slot {exc}; pass --{str(exc).strip(chr(39))}")
            suffix = "_".join(str(v) for v in combo) if combo else "base"
            label = f"{entry['id']}_{name[:26]}_{suffix}"[:90]
            if label in seen:
                continue
            seen.add(label)
            settings = dict(BASE, region=args.region, universe=args.universe, delay=args.delay)
            arms.append({"label": label, "expr": expression, "settings": settings})

    out = Path(args.out)
    out.write_text(json.dumps(arms, ensure_ascii=False, indent=2))
    print(f"{entry['id']}: {len(arms)} arms -> {out}")
    for arm in arms[:3]:
        print("  ", arm["label"])
        print("     ", arm["expr"][:120])
    return 0


def main() -> int:
    reg = load_registry()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show registered templates and their evidence")

    ex = sub.add_parser("expand", help="expand a template over fields")
    ex.add_argument("template")
    ex.add_argument("--fields", required=True, help="comma separated; use field:vec for VECTOR fields")
    ex.add_argument("--region", default="USA")
    ex.add_argument("--universe", default="TOP3000")
    ex.add_argument("--delay", type=int, default=1)
    ex.add_argument("--group", default=None)
    ex.add_argument("--days", default=None, type=lambda v: str(v))
    ex.add_argument("--entry", default=None)
    ex.add_argument("--exit", default=None)
    ex.add_argument("--first-slot-only", action="store_true", help="only the first allowed value per slot")
    ex.add_argument("--out", default="arms-template.json")

    args = ap.parse_args()
    if args.cmd == "list":
        return cmd_list(reg)
    return cmd_expand(reg, args)


if __name__ == "__main__":
    sys.exit(main())