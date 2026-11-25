"""A shelf for alphas that passed the submission checks but were not submitted.

Why this exists: a candidate that clears every platform check is finished work, and
submitting it is irreversible. Two things make "hold it" the right default:

* the platform's self-correlation is measured against a snapshot, so a candidate that
  is independent today may collide after a few more submissions - the value of a held
  alpha decays as the book grows;
* a book wants timing. A candidate may offer a return profile the book could need
  later; holding it preserves the choice of when to submit.

So each entry stores the evidence *as it was when the alpha was shelved*, including
the book size at that moment. `buffer list` reports both the stored value and how
stale it is, and `buffer submit` re-checks before spending the irreversible step.

The file is append-only JSONL next to the ledger: a shelf entry is an event
(`shelved`, `consumed`, `dropped`), and the current state is the last event per alpha.
Rewriting a single JSON document would lose why an alpha left the shelf.
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path

import cache_store
from check_evidence import normalize_checks, summarize_checks
import machine_lib as ml
import research_paths

from .runtime import CliError, protect_output, with_session

BUFFER_FILENAME = "submit-buffer.jsonl"
SEEDS_FILENAME = "seeds.jsonl"
SEED_CATEGORIES = ("ANALYST", "FUNDAMENTAL", "PRICE_VOLUME", "SENTIMENT", "VOLUME")

#: Above this, the platform would refuse the submission.
GATE = 0.70


def buffer_path(args=None):
    """Resolve the shelf file, honouring ``--buffer`` for tests and alt workspaces."""
    explicit = getattr(args, "buffer", None) if args is not None else None
    if explicit:
        return Path(explicit).expanduser()
    return Path(research_paths.research_dir()) / BUFFER_FILENAME


def read_events(path):
    """Every shelf event, oldest first. A malformed line is skipped, not fatal."""
    events = []
    if not Path(path).exists():
        return events
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def shelf_state(events):
    """Current shelf: ``{alpha_id: entry}`` for the last event of each alpha."""
    state = {}
    for event in events:
        alpha_id = event.get("alpha_id")
        if not alpha_id:
            continue
        kind = event.get("event")
        if kind == "shelved":
            state[alpha_id] = dict(event)
        elif kind in {"consumed", "dropped"}:
            entry = state.get(alpha_id)
            if entry is not None:
                entry["state"] = kind
                entry["closed_at"] = event.get("at")
    return state


def _append(path, event):
    """Replace the event stream atomically under one dedupe/transition lock."""
    path = Path(path)
    with cache_store.file_lock(str(path) + ".lock"):
        events = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    event_row = json.loads(line)
                    if not isinstance(event_row, dict):
                        raise ValueError("shelf contains a non-object event")
                    events.append(event_row)
        previous = next((row for row in reversed(events)
                         if row.get("alpha_id") == event.get("alpha_id")), None)
        if event.get("event") in {"consumed", "dropped"}:
            if not previous or previous.get("event") != "shelved":
                return False
        comparable = lambda row: {key: value for key, value in row.items()
                                  if key not in {"at", "event_id"}}
        if previous and comparable(previous) == comparable(event):
            return False
        event = dict(event)
        event["event_id"] = hashlib.sha256(json.dumps(
            {"event": comparable(event), "previous": previous}, sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest()
        events.append(event)
        fd, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for row in events:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return True


def consume_submission(args, selection_id=None):
    """Consume only an open matching shelf entry; repeats have no effect."""
    from datetime import datetime, timezone
    path = buffer_path(args)
    if not path.exists():
        return {"complete": True, "consumed": False, "reason": "not shelved"}
    written = _append(path, {"event": "consumed", "alpha_id": args.alpha_id,
                             "at": datetime.now(timezone.utc).isoformat(),
                             "selection_id": selection_id})
    return {"complete": True, "consumed": written, "path": str(path)}


def _book_evidence(args):
    import alpha_registry
    from .book_state import describe_book
    registry = getattr(args, "registry", None) or alpha_registry.default_registry_path()
    return describe_book(alpha_registry.load_entries(registry), registry_path=registry)


def _book_size(args):
    """Local submitted membership count; platform evidence is reported separately."""
    import alpha_registry

    registry = getattr(args, "registry", None) or alpha_registry.default_registry_path()
    try:
        entries = alpha_registry.load_entries(registry)
    except Exception:  # noqa: BLE001 - a missing ledger must not block the shelf
        return None
    latest = {}
    for entry in entries:
        alpha_id = entry.get("alpha_id")
        if alpha_id:
            latest[alpha_id] = entry
    return sum(1 for entry in latest.values()
               if str(entry.get("status") or "").upper() in {"ACTIVE", "SUBMITTED"})


def _retention(entry):
    """test Sharpe / train Sharpe, or None when either is missing."""
    train = (entry.get("train") or {}).get("sharpe")
    test = (entry.get("test") or {}).get("sharpe")
    if train in (None, 0) or test is None:
        return None
    return round(test / train, 2)


def _checks_summary(checks):
    """Keep the shelf's four-key public shape over the shared check semantics."""
    summary = summarize_checks(checks)
    return {key: summary[key] for key in ("passed", "failed", "pending", "warned")}


def seeds_path(args=None):
    explicit = getattr(args, "seeds", None) if args is not None else None
    if explicit:
        return Path(explicit).expanduser()
    return Path(research_paths.research_dir()) / SEEDS_FILENAME


def read_seeds(path):
    """Read seeds without silently discarding damaged or unrelated state."""
    state = {}
    if not Path(path).exists():
        return state
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError("seed store contains invalid JSON at line %d" % number) from exc
        if not isinstance(row, dict) or not ml.valid_simulation_id(row.get("alpha_id")):
            raise ValueError("seed store has no valid Alpha ID at line %d" % number)
        state[row["alpha_id"]] = row
    return state


def validate_seed(name, category, tags, description):
    """Return the list of missing/invalid seed fields; empty means valid."""
    problems = []
    if not isinstance(name, str) or not name.strip():
        problems.append("name")
    if category not in SEED_CATEGORIES:
        problems.append("category (one of %s)" % ", ".join(SEED_CATEGORIES))
    if (not isinstance(tags, (list, tuple)) or not all(isinstance(t, str) for t in tags)
            or not any(t.strip() for t in tags)):
        problems.append("tags (at least one)")
    description_length = len(description.strip()) if isinstance(description, str) else 0
    if description_length < 100:
        problems.append("description (actual=%d characters; minimum=100)" % description_length)
    return problems


def _seed_paths(args, *, selection_path=None):
    """Keep seed writes separate from the shelf and its supporting evidence."""
    import alpha_registry
    from .alphas import _submission_path

    shelf, seeds = buffer_path(args).resolve(), seeds_path(args).resolve()
    protected = {
        "registry": Path(getattr(args, "registry", None) or alpha_registry.default_registry_path()).expanduser().resolve(),
        "submission journal": _submission_path(args.alpha_id).resolve(),
    }
    selected = selection_path or getattr(args, "selection", None)
    if selected:
        protected["selection"] = Path(selected).expanduser().resolve()
    protect_output(args, shelf, seeds, *protected.values())
    if shelf == seeds:
        raise CliError("INVALID_INPUT", "--buffer and --seeds must use distinct paths")
    for name, target in (("buffer", shelf), ("seeds", seeds)):
        for kind, source in protected.items():
            if target == source:
                raise CliError("INVALID_INPUT", "--%s would overwrite the %s" % (name, kind))
    return shelf, seeds


def submission_seed(args, current):
    """Validate local seed evidence for a new submission, without writing it."""
    path = seeds_path(args)
    try:
        seed = read_seeds(path).get(args.alpha_id)
    except (OSError, ValueError) as exc:
        raise CliError("INVALID_INPUT", "cannot read seed store safely: %s" % exc,
                       "repair the seed store before starting a new submission") from exc
    if seed is None:
        raise CliError("INVALID_INPUT", "%s is not in the seed store" % args.alpha_id,
                       "add it with buffer add and its name, category, tags and description first")
    problems = validate_seed(seed.get("name"), seed.get("category"),
                             seed.get("tags"), seed.get("description"))
    if problems:
        raise CliError("INVALID_INPUT", "stored seed missing or invalid: " + ", ".join(problems),
                       "refresh it with buffer add and complete seed metadata")
    if current.get("id") not in (None, args.alpha_id):
        raise CliError("SERVICE_ERROR", "platform Alpha identity differs from the requested seed")
    if seed.get("expression") is not None and not isinstance(seed["expression"], str):
        raise CliError("INVALID_INPUT", "stored seed expression must be text")
    expression, _description, _ops = ml.regular_expression(current)
    if seed.get("expression") and expression:
        import alpha_registry
        language = ml.alpha_language(current)
        if alpha_registry.expression_hash(seed["expression"], language) != alpha_registry.expression_hash(expression, language):
            raise CliError("INVALID_INPUT", "stored seed expression differs from this Alpha ID")
    saved_settings, observed_settings = seed.get("settings"), current.get("settings")
    if saved_settings is not None and not isinstance(saved_settings, dict):
        raise CliError("INVALID_INPUT", "stored seed settings must be an object")
    if isinstance(saved_settings, dict) and isinstance(observed_settings, dict):
        if any(saved_settings[key] != observed_settings[key] for key in saved_settings.keys() & observed_settings.keys()):
            raise CliError("INVALID_INPUT", "stored seed settings differ from this Alpha ID")
    return seed


def _write_seed(path, row):
    path = Path(path)
    if not isinstance(row, dict) or not ml.valid_simulation_id(row.get("alpha_id")):
        raise ValueError("seed requires a valid Alpha ID")
    problems = validate_seed(row.get("name"), row.get("category"), row.get("tags"), row.get("description"))
    if problems:
        raise ValueError("seed missing or invalid: " + ", ".join(problems))
    with cache_store.file_lock(str(path) + ".lock"):
        existing = read_seeds(path)
        if row["alpha_id"] in existing:
            merged = {**existing[row["alpha_id"]], **row}
            row = merged
            comparable = lambda value: {key: item for key, item in value.items() if key != "recorded_at"}
            if comparable(existing[row["alpha_id"]]) == comparable(row):
                return False
        lines = [json.dumps(r, ensure_ascii=False) for r in existing.values()
                 if r.get("alpha_id") != row["alpha_id"]]
        lines.append(json.dumps(row, ensure_ascii=False))
        cache_store.write_text_atomic(path, "\n".join(lines) + "\n")
        return True


def cmd_buffer_add(args, _sink, *, checked=None):
    """Run the checks on a finished alpha and put it on the shelf without submitting."""
    from .alphas import _candidate_role, _reject_quick_simulation, _selection_id
    if not ml.valid_simulation_id(args.alpha_id):
        raise CliError("INVALID_INPUT", "invalid Alpha ID")
    role = _candidate_role(args)
    problems = validate_seed(getattr(args, "name", None), getattr(args, "category", None),
                             getattr(args, "tags", None), getattr(args, "description", None))
    if problems:
        raise CliError("INVALID_INPUT", "seed missing or invalid: " + ", ".join(problems),
                       "supply name, category, at least one tag, and a >=100 character description")
    path, seed_path = _seed_paths(args)
    selection_id = _selection_id(args)
    def work(session):
        payload = ml.fetch_alpha_metrics(session, args.alpha_id)
        if payload.get("id") not in (None, args.alpha_id):
            raise CliError("SERVICE_ERROR", "platform Alpha identity differs from the requested seed")
        if str(payload.get("status") or "").upper() in {"ACTIVE", "SUBMITTED"}:
            raise CliError("INVALID_INPUT", "submitted alphas cannot enter the shelf",
                           "use alpha submit --yes to repair their local records")
        _reject_quick_simulation(args, payload)
        # The body's snapshot usually has SELF_CORRELATION PENDING; the check endpoint
        # computes it. Shelving on the snapshot would record a NULL where the whole
        # point of the shelf is that number.
        wait = getattr(args, "max_wait", None)
        computed = checked if checked is not None else ml.fetch_alpha_checks(
            session, args.alpha_id, max_wait_seconds=60 if wait is None else wait)
        if computed.get("alpha_id") not in (None, args.alpha_id):
            raise CliError("SERVICE_ERROR", "checks belong to a different Alpha ID")
        checks = normalize_checks(computed.get("checks"))
        split = ml.extract_split(payload) or {}
        train = split.get("train") or {}
        test = split.get("test") or {}
        metrics = payload.get("is") or {}
        self_corr = (computed.get("self_correlation") or {}).get("value") \
            if computed.get("self_correlation") else None
        if self_corr is None:
            for name, body in (checks or {}).items():
                if name == "SELF_CORRELATION" and isinstance(body, dict):
                    self_corr = body.get("value")
        code, _description, _ops = ml.regular_expression(payload)
        from . import costs
        from .alphas import _selection_id
        evidence = summarize_checks(checks)
        summary = _checks_summary(checks)
        eligible = bool(computed.get("complete") and not computed.get("error")
                        and evidence["resolved"] and not evidence["failed"] and self_corr is not None)
        return {"event": "shelved", "alpha_id": args.alpha_id,
                "role": role,
                "expression": code,
                "settings": dict(payload.get("settings") or {}),
                "name": payload.get("name"), "category": payload.get("category"),
                "status": payload.get("status"),
                "train": {k: train.get(k) for k in ("sharpe", "fitness", "turnover", "returns", "margin")},
                "test": {k: test.get(k) for k in ("sharpe", "fitness", "turnover")},
                "full": {k: metrics.get(k) for k in ("sharpe", "fitness", "turnover")},
                "checks": summary, "checks_complete": bool(computed.get("complete")),
                "eligible": eligible,
                "eligibility": "eligible" if eligible else "unknown",
                "selection_id": selection_id,
                "costs": costs.evidence(train, "train", getattr(args, "cost_bps", None)),
                "book": _book_evidence(args),
                "self_correlation": self_corr,
                "book_size": _book_size(args),
                "note": args.note}

    from datetime import datetime, timezone
    from .alphas import _submission_path
    with cache_store.file_lock(str(_submission_path(args.alpha_id)) + ".lock"):
        entry = with_session(args, work)
        entry["at"] = datetime.now(timezone.utc).isoformat()
        failed = entry["checks"]["failed"]
        if failed:
            raise CliError("INVALID_INPUT",
                           "%s fails %s" % (args.alpha_id, ", ".join(failed)),
                           "record failed research with ledger record instead")
        seed = {
            "alpha_id": args.alpha_id,
            "name": args.name.strip(),
            "category": args.category,
            "tags": [t.strip() for t in args.tags if t.strip()],
            "description": args.description.strip(),
            "expression": entry.get("expression"),
            "settings": (entry.get("settings") or {}),
            "metrics": {"train": entry.get("train"), "test": entry.get("test"),
                        "full": entry.get("full")},
            "self_corr": entry.get("self_correlation"),
            "recorded_at": entry["at"],
        }
        seed_complete = False
        try:
            seed_written = _write_seed(seed_path, seed)
            seed_complete = True
            written = _append(path, entry)
        except (OSError, ValueError, cache_store.CacheError) as exc:
            raise CliError("SERVICE_ERROR", "cannot finish shelving %s: %s" % (args.alpha_id, exc),
                           "rerun buffer add with the same seed metadata to finish the local writes",
                           details={"alpha_id": args.alpha_id, "partial": seed_complete,
                                    "seed": {"complete": seed_complete, "path": str(seed_path)},
                                    "shelf": {"complete": False, "path": str(path)}}) from exc
    return {"path": str(path), "seeds": str(seed_path), "entry": entry,
            "seed": seed, "written": written, "seed_written": seed_written}


def cmd_buffer_list(args, _sink):
    """What is on the shelf, how good it looked, and how stale that judgement is."""
    path = buffer_path(args)
    state = shelf_state(read_events(path))
    current_book = _book_size(args)
    current_evidence = _book_evidence(args)
    from . import costs
    rows = []
    for alpha_id, entry in sorted(state.items()):
        if entry.get("state") in {"consumed", "dropped"}:
            continue
        shelved_book = entry.get("book_size")
        stored_digest = ((entry.get("book") or {}).get("local") or {}).get("member_digest")
        stale = (stored_digest != current_evidence["local"]["member_digest"]
                 if stored_digest else
                 (shelved_book != current_book if shelved_book is not None and current_book is not None
                  else None))
        rows.append({
            "alpha_id": alpha_id,
            "train_fitness": (entry.get("train") or {}).get("fitness"),
            "train_sharpe": (entry.get("train") or {}).get("sharpe"),
            "turnover": (entry.get("train") or {}).get("turnover"),
            "self_correlation": entry.get("self_correlation"),
            "eligible": bool(entry.get("eligible")) and stale is False,
            "eligibility": "eligible" if entry.get("eligible") and stale is False else "unknown",
            "selection_id": entry.get("selection_id"),
            "costs": costs.evidence(entry.get("train") or {}, "train", getattr(args, "cost_bps", None)),
            "book_evidence_at_shelving": entry.get("book"),
            # How much of the train-period Sharpe survives the test year. The submit gate
            # reads the full window, which is four-fifths train, so a candidate can pass
            # while its out-of-sample behaviour is poor. Shown here, not used to rank:
            # selecting on it would be selecting on test.
            "retention": _retention(entry),
            # A pending check is not a failure, but it is also not a measurement. The
            # submit path re-checks anyway; the flag stops a null being read as "passed".
            "self_correlation_measured": entry.get("self_correlation") is not None,
            # The platform can pass a check above the gate through its exception rule.
            # A shelf that stays quiet about that invites a mistake at submit time.
            "above_gate": (entry.get("self_correlation") is not None
                           and entry.get("self_correlation") > GATE),
            # The recorded number was true when the alpha was shelved. Every submission
            # since then can change it, especially when a similar alpha is submitted.
            # A shelf that prints the old number without saying so invites a decision
            # on stale evidence.
            "stale": stale,
            "book_at_shelving": shelved_book,
            "book_now": current_book,
            # The gate is measured against a snapshot: every alpha added since shifts it.
            "new_alphas_since": (None if shelved_book is None or current_book is None
                                 else current_book - shelved_book),
            "shelved_at": entry.get("at"),
            "note": entry.get("note"),
            "expression": entry.get("expression"),
        })
    return {"path": str(path), "count": len(rows), "gate": GATE, "rows": rows,
            "book": current_evidence, "order": "alpha_id; test metrics never determine order"}


def cmd_buffer_drop(args, _sink):
    """Take an alpha off the shelf without submitting it."""
    path = buffer_path(args)
    state = shelf_state(read_events(path))
    if (args.alpha_id not in state
            or state[args.alpha_id].get("state") in {"consumed", "dropped"}):
        raise CliError("INVALID_INPUT", "%s is not on the shelf" % args.alpha_id,
                       "run `buffer list`")
    from datetime import datetime, timezone
    _append(path, {"event": "dropped", "alpha_id": args.alpha_id,
                   "at": datetime.now(timezone.utc).isoformat(), "note": args.note})
    return {"alpha_id": args.alpha_id, "state": "dropped"}


def cmd_buffer_submit(args, _sink):
    """Use the same durable submit/reconciliation path as alpha submit."""
    from .alphas import submit_workflow
    if getattr(args, "max_wait", None) is not None and args.max_wait < 0:
        raise CliError("USAGE", "--max-wait must be non-negative")
    entry = shelf_state(read_events(buffer_path(args))).get(args.alpha_id)
    if entry is None:
        raise CliError("INVALID_INPUT", "%s has no shelf history" % args.alpha_id,
                       "run buffer list, or use alpha submit for a direct candidate")
    # Closed entries may still need local repair after a confirmed remote submit.
    # The shared journal prevents another POST for an uncertain earlier attempt.
    return with_session(args, lambda session: submit_workflow(session, args, shelf_entry=entry))
