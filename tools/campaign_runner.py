#!/usr/bin/env python3
"""Run queued regular-alpha experiments and promote eligible research arms."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_app import selection
from tools._runner import research_root, run_cli, run_queue

TRAIN_SHARPE = 1.25
TRAIN_FITNESS = 1.0
MAX_CORRELATION = 0.70
COST_BPS = ("5", "10", "20")


def log(path, record):
    entry = json.dumps({"at": datetime.now(timezone.utc).isoformat(), **record},
                       ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry + "\n")
    print(entry[:400], flush=True)


def require_data(response, operation):
    if not response.ok:
        raise RuntimeError(f"{operation}: {response.detail}")
    return response.data


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def freeze_candidate(root, results, alpha_id):
    """Read train evidence and verify any manifest before allowing promotion.

    Triage returns a decision row even if the gate rejects or defers the
    candidate. Only a passing gate writes a manifest for the validation step.
    """
    path = results / f"campaign-freeze-{uuid4().hex}.json"
    response = run_cli(root, "triage", "--ids", alpha_id, "--against-shelf",
                       "--max-shelf-correlation", str(MAX_CORRELATION),
                       "--min-delta", "0", "--cost-bps", *COST_BPS, "--freeze", path, timeout=900)
    report = require_data(response, f"training evidence for {alpha_id}")
    selection_id = report.get("selection_id")
    rows = report.get("rows") or []
    if selection_id:
        manifest = selection.load(path, source="triage")
        if selection_id != manifest["selection_id"]:
            raise ValueError(f"training selection did not match {alpha_id}")
        rows = manifest["rows"]
    if len(rows) != 1 or rows[0].get("alpha_id") != alpha_id:
        raise ValueError(f"training selection did not match {alpha_id}")
    return (path if selection_id else None), rows[0]


def promote(root, results, alpha_id, label, selection_path):
    output = results / f"promote-{alpha_id}.json"
    response = run_cli(root, "promote", alpha_id, "--note",
                       f"backlog admission: {label}; training gate cleared.",
                       "--selection", selection_path, "--max-self-correlation",
                       str(MAX_CORRELATION), "--save", output, timeout=1800)
    data = require_data(response, f"promote {alpha_id}")
    checks = (data.get("checks") or {}).get("results") or []
    check = next((row for row in checks if row.get("alpha_id") == alpha_id), {})
    validation = data.get("validate") or {}
    return {"alpha_id": alpha_id, "outcome": data.get("outcome"),
            "reasons": data.get("reasons"),
            "self_correlation": ((check.get("checks") or {}).get("SELF_CORRELATION") or {}).get("value"),
            "test_sharpe": validation.get("test_sharpe"), "retention": validation.get("retention")}


def process(root, results, log_entry, spec, data):
    for row in data["rows"]:
        alpha_id, label, role = row.get("alpha_id"), row.get("label"), row.get("role")
        status = row.get("status")
        sharpe, fitness = row.get("train_sharpe"), row.get("train_fitness")
        h1, h2 = row.get("train_sharpe_h1"), row.get("train_sharpe_h2")
        log_entry({"kind": "arm", "spec": spec.name, "label": label, "role": role,
                   "status": status, "train_sharpe": sharpe, "train_fitness": fitness,
                   "h1": h1, "h2": h2, "alpha_id": alpha_id})
        if role == "diagnostic" or status not in {"COMPLETE", "WARNING", "UNSUBMITTED"} or not alpha_id:
            continue
        if not all(map(finite, (sharpe, fitness))):
            raise RuntimeError(f"training metrics incomplete for {alpha_id}")
        if sharpe < TRAIN_SHARPE or fitness < TRAIN_FITNESS:
            continue
        if not all(map(finite, (h1, h2))):
            raise RuntimeError(f"training half-window metrics incomplete for {alpha_id}")
        if h1 <= 0 or h2 <= 0:
            continue

        path, trained = freeze_candidate(root, results, alpha_id)
        gate = trained.get("training_gate") or {}
        shelf = trained.get("shelf_correlation") or {}
        decision = {"spec": spec.name, "label": label, "alpha_id": alpha_id,
                    "selection_path": str(path) if path else None, "delta": trained.get("delta"),
                    "rho_book": trained.get("rho_book"), "training_gate": gate,
                    "max_book_member_rho": trained.get("max_member_rho"),
                    "max_shelf_rho": shelf.get("max_rho"), "max_shelf_member": shelf.get("max_member"),
                    "margin_bps": (trained.get("costs") or {}).get("margin_bps")}
        if gate.get("state") == "rejected":
            log_entry({"kind": "training-rejected", **decision})
            continue
        if (gate.get("state") != "pass" or path is None
                or not finite(decision["delta"]) or decision["delta"] <= 0):
            log_entry({"kind": "training-evidence-deferred", **decision})
            reason = gate.get("reason") or "no admissible frozen training selection"
            raise RuntimeError(f"training evidence for {alpha_id}: {reason}")

        result = promote(root, results, alpha_id, label, path)
        log_entry({"kind": "promote", **decision, **result})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", "--root", dest="root")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    root = research_root(args.root)
    results = root / "backlog-results"
    results.mkdir(parents=True, exist_ok=True)
    report = partial(log, root / "campaign-log.jsonl")

    def paths(spec):
        return (results / f"{spec.stem}-batch.json", results / f"{spec.stem}-result.json")

    return run_queue(root, root / "backlog", paths, report, max_wait=600, interval=30,
                     once=args.once, concurrency=3, consume=partial(process, root, results, report))


if __name__ == "__main__":
    raise SystemExit(main())
