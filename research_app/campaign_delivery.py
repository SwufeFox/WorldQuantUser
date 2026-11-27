"""Publication and offline triage for bulk QUICK research campaigns."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import cache_store
import machine_lib as ml
import simulation_capabilities

from . import experiments
from .campaign import _protect_campaign_output, _quick_training
from .runtime import CliError, protect_output, with_session
from tools._runner import probe_poll_at, probe_state, recent_campaign_attempts, research_root, run_cli
from tools.campaign_runner import campaign_paths
from tools.pipeline import batch_policy


PILOT_SIZE = 16
PILOT_SHARPE = 1.0
PILOT_TURNOVER = 0.35


def _locations(args):
    root = research_root(args.research_dir)
    box = Path(args.campaign_dir).expanduser().resolve() if args.campaign_dir else root
    _protect_campaign_output(args, box)
    inbox = box / "backlog"
    if inbox.is_symlink():
        raise CliError("INVALID_INPUT", "backlog must be a real top-level campaign directory",
                       "replace the symlink with a plain directory under the campaign root")
    return root, box, inbox


def _scope(arm):
    settings = arm["settings"]
    return tuple(settings.get(key) for key in ("region", "universe", "instrumentType", "delay"))


def _mechanism(arm):
    value = arm.get("mechanism")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _pilot_evidence(args, arms, track, box):
    if args.phase == "pilot":
        if args.pilot_batch or len(arms) != PILOT_SIZE:
            raise CliError("INVALID_INPUT", f"pilot publication requires exactly {PILOT_SIZE} arms and no --pilot-batch",
                           f"pass a {PILOT_SIZE}-arm spec for --phase pilot or switch to --phase expansion")
        return None
    if not args.pilot_batch:
        raise CliError("USAGE", "expansion requires --pilot-batch with completed QUICK training evidence",
                       "pass --pilot-batch PATH to the completed 16-arm QUICK pilot batch checkpoint")
    source = Path(args.pilot_batch).expanduser().resolve()
    protected = set()
    try:
        path, batch = experiments.load_probe_batch(source, source_paths=protected,
                                                   within=box / "backlog-results")
    finally:
        protect_output(args, source, *protected)
    if not batch.get("complete") or len(batch["arms"]) != PILOT_SIZE or batch.get("track") != track:
        raise CliError("INVALID_INPUT", "expansion requires a completed 16-arm pilot on the same track",
                       "finish or recover the 16-arm pilot batch before publishing an expansion")
    qualified = {}
    for arm, row in zip(batch["arms"], batch["rows"]):
        mechanism = _mechanism(arm)
        if not mechanism or _scope(arm) != _scope(arms[0]) or row.get("role") == "diagnostic" \
                or arm.get("design") == "control":
            continue
        training, _ = _quick_training(arm, row)
        turnover = row.get("train_turnover")
        if (training and training["sharpe"] >= PILOT_SHARPE and training["fitness"] > 0
                and experiments.finite_metric(turnover) and 0 <= turnover <= PILOT_TURNOVER):
            qualified.setdefault(mechanism, []).append(row["alpha_id"])
    unsupported = [arm["label"] for arm in arms if _mechanism(arm) not in qualified]
    if unsupported:
        raise CliError("INVALID_INPUT", "expansion has arms without qualifying same-scope pilot evidence",
                       "limit expansion arms to mechanisms that passed train Sharpe >= 1.0, positive fitness and turnover <= 0.35 in the same scope",
                       details={"unsupported_arms": unsupported, "qualified_mechanisms": sorted(qualified)})
    return {"batch": str(path), "qualifying_alpha_ids_by_mechanism": qualified}


def _published_location(inbox, target, payload):
    """Perform the actual top-level listing and account for an immediate claim."""
    try:
        listing = subprocess.run(["ls", "-d", "--", str(target)], capture_output=True, text=True, check=False)
        listed_ok = listing.returncode == 0
        listed_out = listing.stdout.strip()
        listing_error = listing.stderr.strip()
    except FileNotFoundError:
        listed_ok = target.is_file()
        listed_out = str(target) if listed_ok else ""
        listing_error = "" if listed_ok else "target not visible after link"
    if listed_ok:
        return str(target), "queued", listed_out
    claimed = inbox.parent / "backlog-claimed"
    consumed = inbox.parent / "backlog-consumed"
    locations = ((claimed, "claimed"), (consumed, "consumed"),
                 (inbox / "done", "done"), (inbox / "failed", "failed"))
    for directory, state in locations:
        for path in _specs(directory):
            if path.name != target.name and not path.name.endswith(f"-{target.name}"):
                continue
            try:
                if path.read_bytes() == payload:
                    return str(path), state, None
            except FileNotFoundError:
                continue
    raise CliError("SERVICE_ERROR", "published spec disappeared before its destination could be confirmed",
                   "run `research_cli.py campaign diagnose` to inspect backlog and claimed directories",
                   details={"destination": str(target), "listing_error": listing_error})


def _publish(inbox, target, payload):
    inbox.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=inbox, prefix=".publish-", suffix=".tmp") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        try:
            os.link(handle.name, target)
        except FileExistsError as exc:
            raise CliError("INVALID_INPUT", f"backlog already contains {target.name}",
                           "run `research_cli.py campaign diagnose` or pick a distinct spec filename") from exc
        listing = _published_location(inbox, target, payload)
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(inbox, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    return listing


def cmd_campaign_publish(args, _sink):
    """Check immutable spec bytes and publish only into the top-level inbox."""
    from tools.pipeline.batch_quick import _build_admission

    root, box, inbox = _locations(args)
    source = Path(args.spec).expanduser().resolve()
    if source.suffix != ".json" or source.name.startswith("."):
        raise CliError("INVALID_INPUT", "the spec filename must be a visible .json file",
                       "rename the spec to a non-hidden *.json filename")
    destination = inbox / source.name
    batch, receipt = campaign_paths(box, destination)
    protect_output(args, source, destination, batch, receipt)
    if destination.exists() or batch.exists() or receipt.exists():
        raise CliError("INVALID_INPUT", "spec name already has queued work or saved evidence",
                       "inspect or recover the original batch before choosing a new name")
    try:
        payload = source.read_bytes()
        raw = payload.decode("utf-8-sig")
        document = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise CliError("INVALID_INPUT", f"cannot read spec {source.name}: {exc}",
                       "pass a readable UTF-8 JSON campaign spec file") from exc
    identity = experiments.probe_spec_identity("-", document=document)
    arms = identity["results"]
    if len(arms) < PILOT_SIZE or len({_scope(arm) for arm in arms}) != 1 or any(
        arm["settings"].get("simulationMode") != "QUICK" for arm in arms
    ):
        raise CliError("INVALID_INPUT", "bulk publication requires at least 16 QUICK arms in one exact scope",
                       "publish each region/universe/delay/instrument scope as its own spec")
    structural = batch_policy.analyze_arms(arms)
    if not structural["ok"]:
        raise CliError("INVALID_INPUT", "bulk structural admission blocked",
                       "inspect error.details.blocking (every arm needs an idea and at most 8 arms per normalized skeleton)",
                       details={"blocking": structural["blocking"]})
    pilot = _pilot_evidence(args, arms, identity["track"], box)

    command = ["probe", "-", "--check", "--strict-gate", "--max-wait-seconds", str(args.max_wait_seconds)]
    if args.multi_size is not None:
        command.extend(("--multi-size", str(args.multi_size)))
    for key in ("no_cache", "relogin", "refresh_meta"):
        if getattr(args, key, False):
            command.append("--" + key.replace("_", "-"))
    if args.meta_max_age is not None:
        command.extend(("--meta-max-age", str(args.meta_max_age)))
    result = run_cli(root, *command, input_text=raw, timeout=args.max_wait_seconds + 15)
    if not result.ok:
        raise CliError(result.code or "SERVICE_ERROR", result.detail,
                       "inspect error.details or run `probe SPEC.json --check --strict-gate` directly",
                       details=result.error_details)
    checked = result.data
    if not checked.get("preflight_complete") or checked.get("request_fingerprint") != identity["request_fingerprint"]:
        raise CliError("INVALID_INPUT", "exact probe preflight did not pass",
                       "fix the failing preflight arm in error.details.preflight before publishing",
                       details={"preflight": checked.get("preflight", [])})
    checked_account = (checked.get("capability_snapshot", {}).get("account") or {}).get("fingerprint")

    def catalog_admission(session):
        account = simulation_capabilities.account_provenance(session)
        if not checked_account or account["fingerprint"] != checked_account:
            raise CliError("AUTH_FAILED", "could not bind both checks to the same authenticated BRAIN account",
                           "verify credentials and retry without changing BRAIN account mid-flight")
        return _build_admission(arms, session, account)

    admission = with_session(args, catalog_admission)
    if not admission["ok"]:
        raise CliError("INVALID_INPUT", "verified dataset admission blocked",
                       "ensure at least three actual signal datasets have current account-and-scope catalog evidence",
                       details={"blocking": admission["blocking"], "field_evidence": admission["field_evidence"]})
    try:
        with cache_store.file_lock(inbox / ".publish.lock"):
            if (batch.exists() or receipt.exists() or (inbox / "done" / destination.name).exists()
                    or (inbox / "failed" / destination.name).exists()
                    or any(path.name.endswith(f"-{destination.name}")
                           for path in _specs(box / "backlog-claimed") + _specs(box / "backlog-consumed"))):
                raise CliError("INVALID_INPUT", "spec name is already claimed or archived",
                               "inspect the saved result before choosing a new name")
            location, state, listing = _publish(inbox, destination, payload)
    except (OSError, cache_store.CacheError) as exc:
        raise CliError("SERVICE_ERROR", f"cannot publish {destination.name}: {exc}",
                       "check directory permissions and lock state under the campaign inbox") from exc
    if state == "failed":
        raise CliError("SERVICE_ERROR", "worker archived the spec as failed immediately after publication",
                       "run `research_cli.py campaign diagnose` to inspect the failure reason and receipt",
                       details={"location": location, "next": "campaign diagnose"})
    return {"destination": str(destination), "location": location, "state": state, "top_level_ls": listing,
            "arms": len(arms), "scope": structural["arms"][0]["scope"], "phase": args.phase, "pilot": pilot,
            "preflight_complete": True, "datasets": admission["diversity"]["verified_datasets"],
            "max_skeleton_count": max(group["count"] for group in structural["skeleton_groups"]),
            "request_fingerprint": identity["request_fingerprint"],
            "spec_sha256": hashlib.sha256(payload).hexdigest()}


def _specs(directory):
    return sorted(path for path in directory.glob("*.json")
                  if not path.name.startswith(".") and path.is_file() and not path.is_symlink())


def cmd_campaign_diagnose(args, _sink):
    """Read queue topology, failure records and checkpoint state without authentication."""
    _, box, inbox = _locations(args)
    reserve = box / "backlog-reserve" / "eurusa-approved"
    top, approved = _specs(inbox), _specs(reserve)
    claimed = _specs(box / "backlog-claimed")
    attempts = recent_campaign_attempts(box)
    ready, recovery, invalid = [], [], []
    for path in top:
        if not path.exists():  # A worker may claim a spec during the read.
            continue
        batch, _ = campaign_paths(box, path)
        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(document, dict):
                raise ValueError("expected an object")
            if batch.exists():
                _, checkpoint = experiments.load_probe_batch(batch)
                state, action = probe_state(checkpoint)
                recovery.append({"spec": path.name, "batch": str(batch), "state": state,
                                 "next_action": action, "next_poll_at": probe_poll_at(checkpoint),
                                 "execution": checkpoint["execution"]})
                continue
        except (CliError, OSError, ValueError) as exc:
            invalid.append({"spec": path.name, "error": str(exc)})
            continue
        ready.append(path.name)

    failures = []
    failure_paths = _specs(inbox / "failed")
    for path in sorted(failure_paths, key=lambda item: item.stat().st_mtime_ns, reverse=True)[:args.limit]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = inbox / "failed" / ".reasons" / path.name
        event, _ = cache_store.load_json(manifest)
        if not isinstance(event, dict) or event.get("source_fingerprint") != digest:
            event = next((item for item in reversed(list(attempts.values()))
                          if item.get("source_fingerprint") == digest and item.get("kind") == "failed"), {})
        failures.append({"spec": path.name, "detail": event.get("detail"), "diagnosis": event.get("diagnosis"),
                         "receipt": event.get("receipt")})
    nested = sorted(str(path.relative_to(box)) for path in inbox.glob("*/*.json")
                    if path.parent.name not in {"done", "failed"} and path.is_file())
    issues = []
    if not ready:
        issues.append("no_ready_top_level_specs")
    if not ready and not approved:
        issues.append("approved_reserve_empty")
    if invalid:
        issues.append("unreadable_queue_entries")
    if nested:
        issues.append("specs_outside_top_level")
    if any(item["detail"] is None for item in failures):
        issues.append("failed_specs_missing_recent_reason")
    deadline_path = box / ".native-multi-until"
    try:
        deadline = int(deadline_path.read_text().strip())
    except (OSError, ValueError):
        deadline = None
    if deadline and deadline <= time.time():
        issues.append("native_dispatch_deadline_elapsed")
    return {"inbox": str(inbox), "top_level_count": len(top), "ready": ready[:args.limit],
            "ready_count": len(ready), "checkpoint_count": len(recovery), "checkpoints": recovery[:args.limit],
            "claimed_count": len(claimed), "claimed": [path.name for path in claimed[:args.limit]],
            "approved_reserve": str(reserve), "approved_count": len(approved), "invalid": invalid[:args.limit],
            "failed_count": len(failure_paths), "recent_failures": failures, "nested_specs": nested[:args.limit],
            "native_dispatch_deadline": deadline, "issues": issues}
