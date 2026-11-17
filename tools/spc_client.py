#!/usr/bin/env python3
"""SPC (Systematic Predictions Challenge) client: recon, validate, submit.

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
  validate SPEC.json     Check one or more specs against the live schema, offline-safe.
  submit SPEC.json --yes Perform the POST. Refuses without --yes; that flag is only
                         passed when the user says 提交.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import machine_lib  # noqa: E402

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
    problems = []
    for key in REQUIRED:
        if spec.get(key) in (None, ""):
            problems.append(f"missing required field: {key}")
    name = spec.get("name") or ""
    if len(name) > 200:
        problems.append(f"name too long: {len(name)} > 200")
    prompt = spec.get("prompt") or ""
    if len(prompt) > 10000:
        problems.append(f"prompt too long: {len(prompt)} > 10000")
    raw = spec.get("sampleOutput")
    if isinstance(raw, str):
        if len(raw) > 100000:
            problems.append(f"sampleOutput too long: {len(raw)} > 100000")
        try:
            parsed = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"sampleOutput is not valid JSON: {exc}")
        else:
            if not isinstance(parsed, list):
                problems.append("sampleOutput must be a JSON array")
    elif raw is not None:
        problems.append("sampleOutput must be a JSON string (the platform takes text)")
    model = spec.get("modelFamily") or ""
    if model and model not in model_index(post):
        problems.append(f"modelFamily not offered by the schema: {model}")
    freq = spec.get("updateFrequency")
    allowed = [c.get("value") for c in (post.get("updateFrequency") or {}).get("choices") or []]
    if freq and freq not in allowed:
        problems.append(f"updateFrequency must be one of {allowed}, got {freq!r}")
    weight = spec.get("weight")
    if weight is not None and not isinstance(weight, (int, float)):
        problems.append("weight must be numeric")
    return problems


def cmd_validate(path: str) -> int:
    post = schema(session())
    specs = json.loads(Path(path).read_text())
    if isinstance(specs, dict):
        specs = [specs]
    bad = 0
    for spec in specs:
        problems = validate_spec(spec, post)
        label = spec.get("name") or "(unnamed)"
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
            print(f"FAIL {spec.get('name')}: {problems}")
            continue
        payload = {k: spec[k] for k in REQUIRED}
        r = s.post(SPC, json=payload, timeout=120)
        print(f"{r.status_code} {spec.get('name')}: {r.text[:160]}")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "recon":
        return cmd_recon()
    if cmd == "validate":
        return cmd_validate(sys.argv[2])
    if cmd == "submit":
        return cmd_submit(sys.argv[2], "--yes" in sys.argv[3:])
    print(f"unknown command {cmd!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())