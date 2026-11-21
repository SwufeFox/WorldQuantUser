#!/usr/bin/env python3
"""Run an ATOM / Power Pool scan through the normal probe lifecycle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cache_store
from research_app.research_route import ATOM_RUNNER_MECHANISM
from tools._runner import research_root, run_cli

MECHANISM = ATOM_RUNNER_MECHANISM
IS_METRICS = ("sharpe", "fitness", "turnover", "returns", "margin", "drawdown")


def _probe_spec(source):
    raw = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("arms must be a non-empty JSON list")
    arms = []
    for index, arm in enumerate(raw):
        if not isinstance(arm, dict) or not isinstance(arm.get("expr"), str):
            raise ValueError(f"arm {index + 1} needs an expr string")
        label = arm.get("label") or f"arm{index}"
        settings = arm.get("settings", {})
        if not isinstance(label, str) or not isinstance(settings, dict):
            raise ValueError(f"arm {index + 1} needs a string label and settings object")
        # Some historical scans reuse labels. Probe identities must be unique.
        arms.append({"label": f"{index + 1}: {label}", "expression": arm["expr"],
                     "settings": settings})
    return {"name": f"atom-{source.stem}", "track": "ATOM", "mechanism": MECHANISM,
            "design": "explore", "arms": arms}


def _read_batch(path):
    batch = json.loads(path.read_text(encoding="utf-8"))
    policy = batch.get("recording_policy") if isinstance(batch, dict) else None
    if (not isinstance(batch, dict) or batch.get("format") != "probe-batch"
            or batch.get("mechanism") != MECHANISM or not batch.get("run_id")
            or not isinstance(policy, dict) or policy.get("enabled") is not False):
        raise ValueError(f"{path} is not a batch produced by atom_runner")
    arms, records = batch.get("arms"), batch.get("records")
    if (not isinstance(arms, list) or not isinstance(records, list) or len(arms) != len(records)
            or not all(isinstance(arm, dict) and isinstance(arm.get("label"), str)
                       and isinstance(arm.get("expression"), str) for arm in arms)
            or not all(isinstance(record, dict) for record in records)):
        raise ValueError(f"{path} has inconsistent arm records")
    return batch


def _result_row(batch, index, arm, record):
    metrics = record.get("metrics") or {}
    sharpe, turnover = metrics.get("sharpe"), metrics.get("turnover")
    return {
        "label": arm["label"].split(": ", 1)[-1],
        "alpha_id": record.get("alpha_id"),
        "status": record.get("status"),
        **{key: metrics.get(key) for key in IS_METRICS},
        "checks": record.get("checks") or {},
        **({"classification_ids": record["classification_ids"]} if "classification_ids" in record else {}),
        "expr": arm["expression"],
        "error": record.get("error"),
        "power_pool_sharpe": isinstance(sharpe, (int, float)) and sharpe >= 1.0,
        "turnover_ok": isinstance(turnover, (int, float)) and 0.01 <= turnover <= 0.70,
        "metrics_scope": "full_is",
        "run_id": batch["run_id"],
        "arm_index": index,
    }


def _project(batch_path, output):
    # Read the current checkpoint under the same lock used by probe. This keeps a
    # concurrent resume from making this projection older than the last one.
    with cache_store.file_lock(str(batch_path) + ".lock"):
        batch = _read_batch(batch_path)
        run_id = batch["run_id"]
        rows = [_result_row(batch, index, arm, record)
                for index, (arm, record) in enumerate(zip(batch["arms"], batch["records"]))]
        with cache_store.file_lock(str(output) + ".lock"):
            try:
                previous = output.read_bytes().decode("utf-8")
                mode = stat.S_IMODE(output.stat().st_mode)
            except FileNotFoundError:
                previous, mode = "", 0o600
            latest = {}
            for line in previous.split("\n"):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"invalid result row in {output}")
                    if row.get("run_id") == run_id:
                        latest[row["arm_index"]] = row
            changed = [row for row in rows if row != latest.get(row["arm_index"])]
            if changed:
                separator = "\n" if previous and not previous.endswith("\n") else ""
                delta = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in changed)
                cache_store.write_text_atomic(output, previous + separator + delta, mode=mode)
    return len(changed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("arms", nargs="?", type=Path, help="JSON list of {label, expr, settings}")
    source.add_argument("--resume", type=Path, metavar="BATCH", help="Continue a saved scan")
    source.add_argument("--recover", type=Path, metavar="BATCH", help="Read accepted jobs without starting simulations")
    parser.add_argument("--batch-path", type=Path, help="Checkpoint path for a new scan")
    parser.add_argument("--research-dir", type=Path, help="Workspace for the batch, results and shared simulation state")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-wait", type=float, default=900.0, help="Probe wait budget in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Preview a new or resumed scan without writing results")
    args = parser.parse_args(argv)
    if args.batch_path and not args.arms:
        parser.error("--batch-path requires an arms file")
    if args.dry_run and args.recover:
        parser.error("--dry-run is only available for a new scan or --resume")

    root = research_root(args.research_dir)
    output = root / "atom-results.jsonl"
    options = ["--batch-size", str(args.concurrency), "--max-wait-seconds", str(args.max_wait)]
    try:
        input_text = None
        if args.arms:
            spec = _probe_spec(args.arms.expanduser())
            input_text = json.dumps(spec, ensure_ascii=False, allow_nan=False)
            identity = json.dumps(spec["arms"], sort_keys=True, ensure_ascii=False)
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            batch_path = args.batch_path or root / "checkpoints" / f"atom-{digest}.json"
            batch_path = batch_path.expanduser().resolve()
            if batch_path in {output.resolve(), args.arms.expanduser().resolve()}:
                raise ValueError("checkpoint must differ from the arms file and result ledger")
            if batch_path.exists() and not args.dry_run:
                raise ValueError(f"batch already exists: {batch_path}. Use --resume or --recover with that path.")
            command = ["-", "--no-record", "--batch-path", str(batch_path)]
        else:
            batch_path = (args.resume or args.recover).expanduser().resolve()
            _read_batch(batch_path)
            command = ["--resume" if args.resume else "--recover", str(batch_path)]

        result = run_cli(root, "probe", *command, *options,
                         "--dry-run" if args.dry_run else "--summary", input_text=input_text)
        if not result.ok:
            print(f"{result.detail} (checkpoint: {batch_path})", file=sys.stderr)
            return 1
        report = result.data
        if not args.dry_run:
            report = {**report, "atom_results": str(output), "appended": _project(batch_path, output)}
        print(json.dumps(report, ensure_ascii=False))
        return 0
    except (OSError, ValueError, cache_store.CacheError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
