#!/usr/bin/env python3
"""Run queued regular-alpha experiments and promote eligible research arms."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def duplicate_of(root, alpha_id):
    """Find candidate-to-shelf pairs above the campaign's correlation ceiling."""
    shelf = require_data(run_cli(root, "buffer", "list", timeout=300), "read shelf")
    ids = [row["alpha_id"] for row in shelf.get("rows", [])
           if row.get("alpha_id") and row["alpha_id"] != alpha_id]
    if not ids:
        return None
    response = run_cli(root, "alpha", "corr", alpha_id, *ids, "--threshold",
                       str(MAX_CORRELATION), timeout=900)
    pairs = require_data(response, f"check correlation for {alpha_id}")["matrix"]["pairs"]
    return max((pair for pair in pairs if alpha_id in {pair["a"], pair["b"]}
                and pair["r"] >= MAX_CORRELATION), key=lambda pair: pair["r"], default=None)


def promote(root, results, alpha_id, label):
    output = results / f"promote-{alpha_id}.json"
    response = run_cli(root, "promote", alpha_id, "--note",
                       f"backlog admission: {label}; training gate cleared.",
                       "--cost-bps", *COST_BPS, "--save", output, timeout=1800)
    data = require_data(response, f"promote {alpha_id}")
    checks = (data.get("checks") or {}).get("results") or []
    check = next((row for row in checks if row.get("alpha_id") == alpha_id), {})
    validation = data.get("validate") or {}
    return {"alpha_id": alpha_id, "outcome": data.get("outcome"),
            "reasons": data.get("reasons"),
            "self_correlation": ((check.get("checks") or {}).get("SELF_CORRELATION") or {}).get("value"),
            "test_sharpe": validation.get("test_sharpe"), "retention": validation.get("retention")}


def check_marginal(root, alpha_id, result):
    """Apply campaign-specific portfolio rules to a newly shelved candidate."""
    self_corr = result["self_correlation"]
    if isinstance(self_corr, (int, float)) and self_corr >= MAX_CORRELATION:
        note = f"platform self-correlation {self_corr} exceeds the {MAX_CORRELATION} campaign criterion"
        outcome = "dropped-over-gate"
    else:
        response = run_cli(root, "triage", "--ids", alpha_id, "--cost-bps", *COST_BPS,
                           timeout=900)
        rows = require_data(response, f"check marginal contribution for {alpha_id}").get("rows") or []
        marginal = rows[0] if rows else {}
        delta = marginal.get("delta")
        result.update(delta=delta, rho_book=marginal.get("rho_book"),
                      margin_bps=(marginal.get("costs") or {}).get("margin_bps"))
        if not isinstance(delta, (int, float)) or delta > 0:
            return
        note = f"marginal-first: delta {delta} <= 0"
        outcome = "dropped-dilutive"
    drop = run_cli(root, "buffer", "drop", alpha_id, "--note", note, timeout=600)
    require_data(drop, f"drop {alpha_id}")
    result.update(outcome=outcome, drop_ok=True)


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
        if None in (sharpe, fitness, h1, h2):
            continue
        if sharpe < TRAIN_SHARPE or fitness < TRAIN_FITNESS or h1 <= 0 or h2 <= 0:
            continue
        duplicate = duplicate_of(root, alpha_id)
        if duplicate:
            log_entry({"kind": "skipped-duplicate", "spec": spec.name, "label": label,
                       "alpha_id": alpha_id, "train_sharpe": sharpe,
                       "correlates_with": duplicate["a"] if duplicate["b"] == alpha_id else duplicate["b"],
                       "r": duplicate["r"]})
            continue
        result = promote(root, results, alpha_id, label)
        if result["outcome"] == "buffered":
            check_marginal(root, alpha_id, result)
        log_entry({"kind": "promote", "spec": spec.name, **result})


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
