#!/usr/bin/env python3
"""Long-running research loop: simulate, gate, validate, shelve - unattended.

For every spec in ``backlog/``: lint-check it, simulate its arms with the
platform's own concurrency, then take each candidate arm that clears the train
gate (Sharpe >= 1.25, Fitness >= 1.0, both halves positive) through
``promote``, which freezes the choice, validates on the held-out window and
shelves it only if every platform check passes and retention clears the floor.

Diagnostic arms are never promoted. Failures are logged and skipped; the loop
keeps going until the backlog is empty, then waits for new specs. Every step is
appended to ``campaign-log.jsonl`` so a later session can reconstruct what ran.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CLI = "/data/src/magix/WorldQuantUser/research_cli.py"
PY = sys.executable
CWD = "/data/src/magix/WorldQuantUser"
ROOT = Path(os.environ.get("WQ_RESEARCH_DIR", "/data/worldquant-research"))
BACKLOG = ROOT / "backlog"
RESULTS = ROOT / "backlog-results"
LOG = ROOT / "campaign-log.jsonl"


def buffer_ids() -> list[str]:
    """Current shelf, read from the tool so the runner never guesses."""
    tmp = Path("/tmp/campaign-buffer.json")
    done = run([PY, CLI, "buffer", "list", "--quiet", "--save", str(tmp)], 300)
    if done.returncode != 0:
        return []
    try:
        rows = json.loads(tmp.read_text())["data"]["rows"]
    except Exception:
        return []
    return [r["alpha_id"] for r in rows if r.get("alpha_id")]


def duplicate_of(alpha_id: str, floor: float = 0.70) -> dict | None:
    """The shelf entry this alpha is the same bet as, if any.

    Self-correlation only compares against submitted alphas, so two shelved
    blends that share a leg sail past every platform check and then turn out to
    be one bet (2026-11-17: eight shelf entries inside a single 0.71-0.82
    cluster). This compares against the shelf itself before the alpha is added.
    """
    ids = [i for i in buffer_ids() if i != alpha_id]
    if not ids:
        return None
    tmp = Path("/tmp/campaign-corr.json")
    done = run([PY, CLI, "alpha", "corr", alpha_id, *ids, "--threshold", str(floor),
                "--quiet", "--save", str(tmp)], 900)
    if done.returncode != 0:
        return None
    try:
        pairs = json.loads(tmp.read_text())["data"]["matrix"]["pairs"]
    except Exception:
        return None
    hits = [p for p in pairs if p.get("r", 0) >= floor]
    if not hits:
        return None
    return max(hits, key=lambda p: p["r"])


def marginal_delta(alpha_id: str) -> dict:
    """How much the book's Sharpe moves when this alpha is added.

    The campaign is marginal-first (user decision 2026-11-17): a unit that clears
    every gate but adds nothing to the portfolio is inventory, not a bet.
    """
    tmp = Path("/tmp/campaign-triage.json")
    done = run([PY, CLI, "triage", "--ids", alpha_id, "--cost-bps", "5", "10", "20",
                "--quiet", "--save", str(tmp)], 900)
    if done.returncode != 0:
        return {}
    try:
        rows = json.loads(tmp.read_text())["data"]["rows"]
    except Exception:
        return {}
    return rows[0] if rows else {}
TRAIN_SHARPE = 1.25
TRAIN_FITNESS = 1.0


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(record: dict) -> None:
    record["at"] = now()
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False)[:400], flush=True)


def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=CWD, capture_output=True, text=True, timeout=timeout)


def probe(spec: Path, out: Path) -> dict | None:
    check = run([PY, CLI, "probe", str(spec), "--check", "--quiet"], 600)
    if check.returncode != 0:
        log({"kind": "check-failed", "spec": spec.name,
             "tail": (check.stdout or check.stderr)[-300:]})
        return None
    batch = RESULTS / (spec.stem + "-batch.json")
    done = run([PY, CLI, "probe", str(spec), "--batch-size", "3",
                "--max-wait-seconds", "600", "--batch-path", str(batch),
                "--quiet", "--save", str(out)], 1800)
    if not out.exists():
        log({"kind": "probe-failed", "spec": spec.name,
             "tail": (done.stdout or done.stderr)[-300:]})
        return None
    return json.loads(out.read_text()).get("data") or {}


def promote(alpha_id: str, label: str) -> dict:
    out = RESULTS / f"promote-{alpha_id}.json"
    done = run([PY, CLI, "promote", alpha_id,
                "--note", f"backlog admission: {label}; train gate cleared, frozen validation applied.",
                "--cost-bps", "5", "10", "20", "--quiet", "--save", str(out)], 1800)
    if not out.exists():
        return {"alpha_id": alpha_id, "outcome": "promote-error",
                "tail": (done.stdout or done.stderr)[-200:]}
    data = json.loads(out.read_text()).get("data") or {}
    checks = data.get("checks") or {}
    rows = checks.get("results") or []
    self_corr = None
    if rows:
        self_corr = ((rows[0].get("checks") or {}).get("SELF_CORRELATION") or {}).get("value")
    return {"alpha_id": alpha_id, "outcome": data.get("outcome"),
            "reasons": data.get("reasons"),
            "self_correlation": self_corr,
            "test_sharpe": (data.get("validate") or {}).get("test_sharpe"),
            "retention": (data.get("validate") or {}).get("retention")}


def process(spec: Path) -> None:
    out = RESULTS / (spec.stem + "-result.json")
    data = probe(spec, out)
    if data is None:
        return
    rows = data.get("rows") or []
    for row in rows:
        label = row.get("label")
        role = row.get("role")
        status = row.get("status")
        sharpe = row.get("train_sharpe")
        fitness = row.get("train_fitness")
        h1 = row.get("train_sharpe_h1")
        h2 = row.get("train_sharpe_h2")
        alpha_id = row.get("alpha_id")
        summary = {"kind": "arm", "spec": spec.name, "label": label, "role": role,
                   "status": status, "train_sharpe": sharpe, "train_fitness": fitness,
                   "h1": h1, "h2": h2, "alpha_id": alpha_id}
        log(summary)
        if role == "diagnostic" or status != "COMPLETE" or not alpha_id:
            continue
        if sharpe is None or fitness is None or h1 is None or h2 is None:
            continue
        if sharpe < TRAIN_SHARPE or fitness < TRAIN_FITNESS or h1 <= 0 or h2 <= 0:
            continue
        dup = duplicate_of(alpha_id)
        if dup:
            log({"kind": "skipped-duplicate", "spec": spec.name, "label": label,
                 "alpha_id": alpha_id, "train_sharpe": sharpe,
                 "correlates_with": dup["a"] if dup["b"] == alpha_id else dup["b"],
                 "r": dup["r"]})
            continue
        result = promote(alpha_id, label)
        result["kind"] = "promote"
        result["spec"] = spec.name
        # BRAIN passes self-correlation above the 0.70 line when the alpha's own
        # Sharpe clears an exception rule (expected_result comes back FAIL while
        # the check reports PASS). The campaign's criterion is stricter: value
        # must be under 0.70, so an exception-passed alpha is dropped here rather
        # than counted.
        sc = result.get("self_correlation")
        if result.get("outcome") == "buffered" and isinstance(sc, (int, float)) and sc >= 0.70:
            drop = run([PY, CLI, "buffer", "drop", alpha_id, "--note",
                        f"dropped by the campaign rule: platform self-correlation {sc} passed via the "
                        "exception rule but exceeds the 0.70 criterion", "--quiet"], 600)
            result["outcome"] = "dropped-over-gate"
            result["drop_ok"] = drop.returncode == 0
        # Marginal-first (user decision 2026-11-17): a shelved unit is kept only
        # if it adds portfolio Sharpe. Anything at or below zero is inventory.
        if result.get("outcome") == "buffered":
            marg = marginal_delta(alpha_id)
            result["delta"] = marg.get("delta")
            result["rho_book"] = marg.get("rho_book")
            result["margin_bps"] = (marg.get("costs") or {}).get("margin_bps")
            if isinstance(marg.get("delta"), (int, float)) and marg["delta"] <= 0:
                run([PY, CLI, "buffer", "drop", alpha_id, "--note",
                     f"marginal-first: delta {marg['delta']} <= 0", "--quiet"], 600)
                result["outcome"] = "dropped-dilutive"
        log(result)


def main() -> int:
    BACKLOG.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    done_dir = BACKLOG / "done"
    fail_dir = BACKLOG / "failed"
    done_dir.mkdir(exist_ok=True)
    fail_dir.mkdir(exist_ok=True)
    log({"kind": "start", "backlog": len(list(BACKLOG.glob('*.json')))})
    while True:
        specs = sorted(BACKLOG.glob("*.json"))
        if not specs:
            time.sleep(30)
            continue
        for spec in specs:
            try:
                process(spec)
                shutil.move(str(spec), str(done_dir / spec.name))
            except Exception as exc:  # noqa: BLE001 - one bad spec must not stop the loop
                log({"kind": "spec-crashed", "spec": spec.name, "error": f"{type(exc).__name__}: {exc}"})
                shutil.move(str(spec), str(fail_dir / spec.name))


if __name__ == "__main__":
    raise SystemExit(main())
