#!/usr/bin/env python3
"""SPC (Systematic Predictions Challenge) client and local experiment tools.

Built 2026-11-17 at the user's request: tooling first, submissions only on an
explicit authorisation, same rule as regular alphas.

What the live platform actually exposes (verified against the account, not the
public forum post):
  OPTIONS  /competitions/spc/submissions   -> action schema
  GET      /competitions/spc/submissions   -> this account's submissions
  POST     /competitions/spc/submissions   -> create (guarded here)
  PATCH    /competitions/spc/submissions/{id} -> weight only
  GET      /competitions/spc               -> 404
  GET      /competitions/spc/leaderboard   -> 404
  GET      /competitions/spc/score         -> 404

The forum post's `model` + `modelVersion` pair does not exist here: the schema
requires `modelFamily` and the concrete model id.

Subcommands:
  recon                  Print the schema, the submission list and endpoint reachability.
  validate SPEC.json     Read the live schema and validate local submission specs.
  experiment SPEC.json --run RUN.json [--run MORE.json] [--reference UNIVERSE.json]
                         Compare captured model outputs without platform access.
  submit SPEC.json --yes Perform the POST. Refuses without --yes; that flag is only
                         passed when the user says 提交.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import machine_lib  # noqa: E402
from tools.spc_experiment import experiment, parse_predictions  # noqa: E402

BASE = "https://api.worldquantbrain.com"
SPC = f"{BASE}/competitions/spc/submissions"
REQUIRED = ("name", "prompt", "sampleOutput", "modelFamily", "weight", "updateFrequency")


def session():
    return machine_lib.load_session()


def schema(s) -> dict:
    r = s.options(SPC, timeout=60)
    r.raise_for_status()
    return ((r.json() or {}).get("actions") or {}).get("POST") or {}


def model_index(post: dict) -> set[str]:
    fam = post.get("modelFamily") or {}
    out = set()
    for family in fam.get("choices") or []:
        if family.get("value"):
            out.add(family["value"])
        for item in family.get("items") or []:
            if item.get("value"):
                out.add(item["value"])
    return out


def cmd_recon() -> int:
    s = session()
    post = schema(s)
    print("== POST schema ==")
    for key, spec in post.items():
        if isinstance(spec, dict):
            bits = {k: spec.get(k) for k in ("type", "required", "maxLength") if spec.get(k) is not None}
            print(f"  {key:16s} {bits}")
    families = [(f.get("value"), [i.get("value") for i in (f.get("items") or [])])
                for f in (post.get("modelFamily") or {}).get("choices") or []]
    print("\n== model families ==")
    for fam, items in families:
        print(f"  {fam:10s} {items[:3]}")
    freq = [c.get("value") for c in (post.get("updateFrequency") or {}).get("choices") or []]
    print("\n== updateFrequency ==", freq)

    print("\n== endpoint reachability ==")
    for url in (f"{BASE}/competitions", f"{BASE}/competitions/spc", f"{SPC}?limit=1",
                f"{BASE}/competitions/spc/leaderboard", f"{BASE}/users/self/competitions"):
        try:
            r = s.get(url, timeout=45)
            print(f"  {r.status_code}  {url.replace(BASE, '')}")
        except Exception as exc:  # noqa: BLE001 - recon wants the failure text
            print(f"  ERR  {url.replace(BASE, '')}  {str(exc)[:60]}")

    r = s.get(f"{SPC}?limit=50", timeout=60)
    body = r.json() if r.text else {}
    print(f"\n== submissions: {body.get('count', 0)} ==")
    for sub in body.get("results") or []:
        print(f"  {sub.get('id')} {sub.get('status'):>16s} w={sub.get('weight')} "
              f"{sub.get('updateFrequency')} model={sub.get('modelFamily')} {str(sub.get('name'))[:40]}")
    return 0


def validate_spec(spec: dict, post: dict) -> list[str]:
    if not isinstance(spec, dict):
        return ["submission spec must be a JSON object"]
    problems = []
    for key in REQUIRED:
        value = spec.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            problems.append(f"missing required field: {key}")

    for key, default_limit in (("name", 200), ("prompt", 10000), ("sampleOutput", 100000),
                               ("modelFamily", 100), ("updateFrequency", None)):
        value = spec.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            problems.append(f"{key} must be a string")
            continue
        limit = (post.get(key) or {}).get("maxLength", default_limit)
        if type(limit) is int and len(value) > limit:
            problems.append(f"{key} too long: {len(value)} > {limit}")
        elif key == "sampleOutput" and value.strip():
            _, errors = parse_predictions(value)
            problems.extend(f"sampleOutput: {error}" for error in errors)

    model = spec.get("modelFamily")
    if isinstance(model, str) and model.strip() and model not in model_index(post):
        problems.append(f"modelFamily not offered by the schema: {model}")
    freq = spec.get("updateFrequency")
    allowed = [c.get("value") for c in (post.get("updateFrequency") or {}).get("choices") or []]
    if isinstance(freq, str) and freq.strip() and freq not in allowed:
        problems.append(f"updateFrequency must be one of {allowed}, got {freq!r}")
    weight = spec.get("weight")
    if weight is not None:
        if type(weight) not in (int, float) or not 0 <= weight <= 1:
            problems.append("weight must be a number from 0 to 1")
        elif (decimal_weight := Decimal(str(weight))) != decimal_weight.quantize(Decimal("0.01")):
            problems.append("weight must have at most two decimal places")
    return problems


def cmd_validate(path: str) -> int:
    post = schema(session())
    specs = json.loads(Path(path).read_text())
    if isinstance(specs, dict):
        specs = [specs]
    bad = 0
    for spec in specs:
        problems = validate_spec(spec, post)
        label = (spec.get("name") if isinstance(spec, dict) else None) or "(unnamed)"
        if problems:
            bad += 1
            print(f"FAIL {label}")
            for p in problems:
                print(f"     - {p}")
        else:
            print(f"ok   {label}")
    print(f"\n{len(specs) - bad}/{len(specs)} specs valid")
    return 1 if bad else 0


def cmd_submit(path: str, yes: bool) -> int:
    if not yes:
        print("refused: submission is a public competitive action and needs an explicit 提交 "
              "from the user. Re-run with --yes once that authorisation exists.")
        return 2
    s = session()
    post = schema(s)
    specs = json.loads(Path(path).read_text())
    if isinstance(specs, dict):
        specs = [specs]
    for spec in specs:
        problems = validate_spec(spec, post)
        if problems:
            name = spec.get("name") if isinstance(spec, dict) else None
            print(f"FAIL {name or '(unnamed)'}: {problems}")
            continue
        payload = {k: spec[k] for k in REQUIRED}
        r = s.post(SPC, json=payload, timeout=120)
        print(f"{r.status_code} {spec.get('name')}: {r.text[:160]}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect SPC and compare captured model outputs.")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("recon", help="Read the live platform schema and submission list.")
    validate = commands.add_parser("validate", help="Check a local spec against the live schema.")
    validate.add_argument("spec", metavar="SPEC.json")
    local = commands.add_parser("experiment", help="Compare captured model outputs offline.")
    local.add_argument("spec", type=Path, metavar="SPEC.json", help="One spec with a frozen sampleOutput.")
    local.add_argument("--run", required=True, action="append", type=Path, metavar="RUN.json",
                       help="Raw JSON array from a repeat run. Pass once per capture.")
    local.add_argument("--reference", type=Path, metavar="UNIVERSE.json",
                       help="Optional local ISIN/MIC reference with regions and sectors.")
    submit = commands.add_parser("submit", help="Create a submission with explicit confirmation.")
    submit.add_argument("spec", metavar="SPEC.json")
    submit.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "recon":
        return cmd_recon()
    if args.command == "validate":
        return cmd_validate(args.spec)
    if args.command == "submit":
        return cmd_submit(args.spec, args.yes)
    if args.command == "experiment":
        result = experiment(args.spec, args.run, reference_path=args.reference)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["ok"] else 2
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
