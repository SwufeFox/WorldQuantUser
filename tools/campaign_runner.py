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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import machine_lib as ml
from research_app import selection
from tools._runner import research_root, run_cli, run_probe, run_queue

TRAIN_SHARPE = 1.25
TRAIN_FITNESS = 1.0
MAX_CORRELATION = 0.70
COST_BPS = ("5", "10", "20")


def log(path, record, *, log_event=None):
    event = {"at": datetime.now(timezone.utc).isoformat(), **record}
    entry = json.dumps(event, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry + "\n")
    if log_event is None:
        print(entry[:400], flush=True)
    else:
        log_event(event)


def require_data(response, operation):
    if not response.ok:
        raise RuntimeError(f"{operation}: {response.detail}")
    return response.data


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _promotion_receipt(path, alpha_id):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict) or payload.get("command") != "promote":
        raise ValueError(f"invalid promotion receipt for {alpha_id}")
    data = payload.get("data")
    if payload.get("ok") is False and data is None:
        return None
    if (not isinstance(data, dict) or data.get("alpha_id") != alpha_id
            or not isinstance(data.get("freeze_path"), str)):
        raise ValueError(f"promotion receipt does not match {alpha_id}")
    return payload


def freeze_candidate(root, results, alpha_id):
    """Read train evidence and verify any manifest before allowing promotion.

    Triage returns a decision row even if the gate rejects or defers the
    candidate. Only a passing gate writes a manifest for the validation step.
    """
    if not ml.valid_simulation_id(alpha_id):
        raise ValueError(f"invalid Alpha ID: {alpha_id!r}")
    path = results / f"campaign-freeze-{alpha_id}.json"
    selection_id = None
    if not path.exists():
        receipt = _promotion_receipt(results / f"promote-{alpha_id}.json", alpha_id)
        if receipt:
            # A legacy worker already bound this candidate to a UUID-named freeze.
            legacy = Path(receipt["data"]["freeze_path"]).resolve()
            if legacy.parent != results.resolve() or not legacy.name.startswith("campaign-freeze-"):
                raise ValueError(f"invalid campaign freeze path for {alpha_id}")
            path = legacy
        else:
            response = run_cli(root, "triage", "--ids", alpha_id, "--against-shelf",
                               "--max-shelf-correlation", str(MAX_CORRELATION),
                               "--min-delta", "0", "--cost-bps", *COST_BPS, "--freeze", path, timeout=900)
            report = require_data(response, f"training evidence for {alpha_id}")
            selection_id = report.get("selection_id")
            if not selection_id:
                rows = report.get("rows") or []
                if len(rows) != 1 or rows[0].get("alpha_id") != alpha_id:
                    raise ValueError(f"training selection did not match {alpha_id}")
                return None, rows[0]
    manifest = selection.load(path, source="triage")
    rows = manifest["rows"]
    if (len(rows) != 1 or rows[0].get("alpha_id") != alpha_id
            or (selection_id and selection_id != manifest["selection_id"])):
        raise ValueError(f"training selection did not match {alpha_id}")
    return path, rows[0]


def _promotion_pending(data):
    reasons = set(data.get("reasons") or [])
    return bool(reasons) and reasons <= {
        "platform_checks_not_resolved", "test_sharpe_missing_or_nonfinite",
        "retention_missing_or_nonfinite", "self_correlation_missing_or_nonfinite",
    }


def promote(root, results, alpha_id, label, selection_path):
    output = results / f"promote-{alpha_id}.json"
    receipt = _promotion_receipt(output, alpha_id)
    data = receipt["data"] if receipt else {}
    if receipt and Path(data["freeze_path"]).resolve() != selection_path.resolve():
        raise ValueError(f"promotion receipt uses a different choice for {alpha_id}")
    if (not receipt or not receipt.get("ok") or data.get("outcome") not in {"buffered", "rejected"}
            or _promotion_pending(data)):
        response = run_cli(root, "promote", alpha_id, "--note",
                           f"backlog admission: {label}; training gate cleared.",
                           "--selection", selection_path, "--max-self-correlation",
                           str(MAX_CORRELATION), "--save", output, timeout=1800)
        data = require_data(response, f"promote {alpha_id}")
    if _promotion_pending(data):
        raise RuntimeError(f"promotion evidence incomplete for {alpha_id}")
    checks = (data.get("checks") or {}).get("results") or []
    check = next((row for row in checks if row.get("alpha_id") == alpha_id), {})
    validation = data.get("validate") or {}
    return {"alpha_id": alpha_id, "outcome": data.get("outcome"),
            "reasons": data.get("reasons"),
            "self_correlation": ((check.get("checks") or {}).get("SELF_CORRELATION") or {}).get("value"),
            "test_sharpe": validation.get("test_sharpe"), "retention": validation.get("retention")}


def process(root, results, log_entry, spec, data, *, validate=True):
    deferred = []
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
        settings = row.get("settings") or {}
        mode = str(settings.get("simulationMode") or settings.get("simulation_mode") or "FULL").strip().upper()
        track = str(data.get("track") or "REGULAR").strip().upper()
        alpha_type = str(settings.get("type") or row.get("type") or "REGULAR").strip().upper()
        if mode != "FULL" or track != "REGULAR" or alpha_type != "REGULAR":
            log_entry({"kind": "research-only", "spec": spec.name, "label": label,
                       "alpha_id": alpha_id, "track": track, "mode": mode, "alpha_type": alpha_type})
            continue
        if not all(map(finite, (sharpe, fitness))):
            deferred.append(f"training metrics incomplete for {alpha_id}")
            continue
        if sharpe < TRAIN_SHARPE or fitness < TRAIN_FITNESS:
            continue
        if not all(map(finite, (h1, h2))):
            deferred.append(f"training half-window metrics incomplete for {alpha_id}")
            continue
        if h1 <= 0 or h2 <= 0:
            continue

        try:
            path, trained = freeze_candidate(root, results, alpha_id)
        except (OSError, RuntimeError, ValueError) as error:
            deferred.append(str(error))
            continue
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
            deferred.append(f"training evidence for {alpha_id}: {reason}")
            continue

        if not validate:
            log_entry({"kind": "training-frozen", **decision})
            continue
        try:
            result = promote(root, results, alpha_id, label, path)
        except (OSError, RuntimeError, ValueError) as error:
            deferred.append(str(error))
            continue
        log_entry({"kind": "promote", **decision, **result})
    if deferred:
        raise RuntimeError("; ".join(deferred))


def campaign_paths(root, spec):
    results = Path(root) / "backlog-results"
    stem = Path(spec).stem
    return results / f"{stem}-batch.json", results / f"{stem}-result.json"


def run_campaign(root, *, once=False, max_wait=600, interval=30, concurrency=3,
                 log_event=None, execute=None, validate=True, multi_size=None,
                 recover_only=False, strict_gate=False):
    results = root / "backlog-results"
    results.mkdir(parents=True, exist_ok=True)
    report = partial(log, root / "campaign-log.jsonl", log_event=log_event)

    if execute is None and (multi_size is not None or recover_only or strict_gate):
        execute = partial(run_probe, multi_size=multi_size, recover_only=recover_only,
                          strict_gate=strict_gate)
    return run_queue(root, root / "backlog", partial(campaign_paths, root), report,
                     max_wait=max_wait, interval=interval,
                     once=once, concurrency=concurrency, execute=execute,
                     recover_first=recover_only,
                     consume=partial(process, root, results, report, validate=validate))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", "--root", dest="root")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    return run_campaign(research_root(args.root), once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
