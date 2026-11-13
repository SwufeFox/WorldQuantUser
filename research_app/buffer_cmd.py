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
from pathlib import Path

import machine_lib as ml
import research_paths

from .runtime import CliError, with_session

BUFFER_FILENAME = "submit-buffer.jsonl"

#: Above this, the platform would refuse the submission.
GATE = 0.70


def buffer_path(args=None):
    """Resolve the shelf file, honouring ``--buffer`` for tests and alt workspaces."""
    explicit = getattr(args, "buffer", None) if args is not None else None
    if explicit:
        return Path(explicit).expanduser()
    return Path(research_paths.ensure_dir()) / BUFFER_FILENAME


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _book_size(args):
    """How many alphas are ACTIVE right now; the correlation's frame of reference."""
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
               if str(entry.get("status") or "").upper() == "ACTIVE")


def _retention(entry):
    """test Sharpe / train Sharpe, or None when either is missing."""
    train = (entry.get("train") or {}).get("sharpe")
    test = (entry.get("test") or {}).get("sharpe")
    if train in (None, 0) or test is None:
        return None
    return round(test / train, 2)


def _checks_summary(checks):
    """Split checks into passed / failed / pending.

    PENDING is not a failure: the alpha body carries a cached snapshot in which
    SELF_CORRELATION is usually PENDING until `/alphas/{id}/check` runs. Folding it into
    "failed" made every fresh candidate look unsubmittable.
    """
    passed, failed, pending, warned = [], [], [], []
    for name, body in (checks or {}).items():
        result = (body or {}).get("result") if isinstance(body, dict) else body
        if result == "PASS":
            passed.append(name)
        elif result in (None, "PENDING"):
            pending.append(name)
        elif result == "WARNING":
            # WARNING is not a failure on the platform: the alpha still submits, and
            # UNITS in particular is a unit-inference note, not a rejection. Folding it
            # into "failed" blocked a passing candidate from the shelf.
            warned.append(name)
            passed.append(name)
        else:
            failed.append(name)
    return {"passed": sorted(passed), "failed": sorted(failed),
            "pending": sorted(pending), "warned": sorted(warned)}


def cmd_buffer_add(args, _sink):
    """Run the checks on a finished alpha and put it on the shelf without submitting."""
    def work(session):
        payload = ml.fetch_alpha_metrics(session, args.alpha_id)
        # The body's snapshot usually has SELF_CORRELATION PENDING; the check endpoint
        # computes it. Shelving on the snapshot would record a NULL where the whole
        # point of the shelf is that number.
        computed = ml.fetch_alpha_checks(session, args.alpha_id,
                                         max_wait_seconds=args.max_wait or 60)
        checks = computed.get("checks") or ml._extract_checks(payload)
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
        return {"event": "shelved", "alpha_id": args.alpha_id,
                "expression": code,
                "name": payload.get("name"), "category": payload.get("category"),
                "status": payload.get("status"),
                "train": {k: train.get(k) for k in ("sharpe", "fitness", "turnover", "returns")},
                "test": {k: test.get(k) for k in ("sharpe", "fitness", "turnover")},
                "full": {k: metrics.get(k) for k in ("sharpe", "fitness", "turnover")},
                "checks": _checks_summary(checks),
                "self_correlation": self_corr,
                "book_size": _book_size(args),
                "note": args.note}

    entry = with_session(args, work)
    from datetime import datetime, timezone
    entry["at"] = datetime.now(timezone.utc).isoformat()
    failed = entry["checks"]["failed"]
    if failed:
        # A candidate that fails a check is not "held for later"; it is not submittable.
        raise CliError("INVALID_INPUT",
                       "%s fails %s" % (args.alpha_id, ", ".join(failed)),
                       "fix the construct or shelve it with `ledger record` instead; "
                       "the shelf is for alphas that already pass every check")
    path = buffer_path(args)
    _append(path, entry)
    return {"path": str(path), "entry": entry}


def cmd_buffer_list(args, _sink):
    """What is on the shelf, how good it looked, and how stale that judgement is."""
    path = buffer_path(args)
    state = shelf_state(read_events(path))
    current_book = _book_size(args)
    rows = []
    for alpha_id, entry in sorted(state.items()):
        if entry.get("state") in {"consumed", "dropped"}:
            continue
        shelved_book = entry.get("book_size")
        rows.append({
            "alpha_id": alpha_id,
            "train_fitness": (entry.get("train") or {}).get("fitness"),
            "train_sharpe": (entry.get("train") or {}).get("sharpe"),
            "turnover": (entry.get("train") or {}).get("turnover"),
            "self_correlation": entry.get("self_correlation"),
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
            "stale": bool((None if not shelved_book else current_book - shelved_book) or 0),
            "book_at_shelving": shelved_book,
            "book_now": current_book,
            # The gate is measured against a snapshot: every alpha added since shifts it.
            "new_alphas_since": (None if not shelved_book or current_book is None
                                 else current_book - shelved_book),
            "shelved_at": entry.get("at"),
            "note": entry.get("note"),
            "expression": entry.get("expression"),
        })
    return {"path": str(path), "count": len(rows), "gate": GATE, "rows": rows}


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
    """Submit a shelved alpha, re-checking the one number that decays: independence."""
    path = buffer_path(args)
    state = shelf_state(read_events(path))
    entry = state.get(args.alpha_id)
    if entry is None or entry.get("state") in {"consumed", "dropped"}:
        raise CliError("INVALID_INPUT", "%s is not on the shelf" % args.alpha_id,
                       "run `buffer list`")

    def work(session):
        payload = ml.fetch_alpha_metrics(session, args.alpha_id)
        computed = ml.fetch_alpha_checks(session, args.alpha_id,
                                         max_wait_seconds=args.max_wait or 60)
        fresh = (computed.get("self_correlation") or {}).get("value") \
            if computed.get("self_correlation") else None
        missing = [field for field in ("name", "category") if not payload.get(field)]
        if missing:
            raise CliError("INVALID_INPUT",
                           "%s has no %s" % (args.alpha_id, ", ".join(missing)),
                           "set them first: alpha describe %s --name ... --category ..."
                           % args.alpha_id)
        if fresh is not None and fresh >= GATE and not args.force:
            # The shelf value came from a smaller book. Refuse rather than discover the
            # rejection on the platform, which also costs the irreversible attempt.
            raise CliError("INVALID_INPUT",
                           "%s now correlates %.4f (gate %.2f) - shelved when the book "
                           "was %s" % (args.alpha_id, fresh, GATE, entry.get("book_size")),
                           "pass --force to try anyway, or drop it from the shelf")
        result = ml.submit_alpha(session, args.alpha_id, confirm=args.yes,
                                 max_wait_seconds=args.max_wait or 60)
        result["self_correlation_now"] = fresh
        return result

    result = with_session(args, work)
    if result.get("submitted"):
        from datetime import datetime, timezone
        _append(path, {"event": "consumed", "alpha_id": args.alpha_id,
                       "at": datetime.now(timezone.utc).isoformat(),
                       "self_correlation": result.get("self_correlation_now")})
    return result