"""A durable ledger of alphas worth remembering.

Research findings kept only in a chat window are lost the moment the context is
compacted: the alpha id, the expression that produced it, the settings it was
measured under, and *why it mattered*. This module keeps that in an append-only
JSONL ledger so a later session can reload the state of the research instead of
re-deriving it.

The ledger is deliberately stored **outside the git working tree** (see
``research_paths``): the repository versions the tool, not the research output.
It is still a plain file that can be backed up, moved or diffed on its own.

Design notes:

* **Append-only.** Every write appends one line; :func:`load_entries` collapses
  by key so the newest record wins. History is never rewritten, and a bad write
  cannot destroy earlier work.
* **Keyed by alpha id** when BRAIN assigned one, otherwise by a hash of the
  expression (so rejected-but-informative expressions are also rememberable).
* **Self-describing.** Each entry carries the expression, the full settings, the
  in-sample metrics, the platform checks, the idea it belongs to and the verdict
  plus a short note. That is everything needed to judge it again later without
  re-running anything.
* **JSONL**: one alpha per line, so the file stays greppable and append-only
  edits never rewrite earlier records.
"""

from __future__ import annotations

import datetime
import errno
import hashlib
import json
import math
import re
import os
from contextlib import ExitStack, contextmanager
from functools import partial

import research_paths
import cache_store
import machine_lib as ml
from check_evidence import normalize_checks, platform_check_names, summarize_checks

__all__ = [
    "VERDICTS",
    "QUERY_FIELDS",
    "default_registry_path",
    "expression_hash",
    "same_expression",
    "load_entries",
    "indexed_registry",
    "append_batch",
    "record_entry",
    "render_markdown",
    "summarize",
]

# ``rejected``  - the alpha underperformed; the expression is dead, the
#                 hypothesis was never properly on trial.
# ``falsified`` - a pre-registered hypothesis was put on trial with controls
#                 and the controls won. This is a *result*, not a failure: it
#                 tells a later round not to re-open the mechanism, and it is
#                 the reason `ledger list` exists. Keep the note specific about
#                 which control killed it.
# ``no_go``     - abandoned before a fair test (data blocked, unusable fields).
VERDICTS = ("candidate", "watch", "rejected", "falsified", "no_go")

# Ledger columns available for filtering and grouping.
QUERY_FIELDS = ("status", "verdict", "family", "region", "universe", "neutralization",
                "delay", "decay", "month")

#: Statuses that mean the platform finished the run and every number is final.
#: ``WARNING`` belongs here: BRAIN uses it when a check warns without failing
#: (``UNITS`` is the usual one), and the metrics are as final as ``COMPLETE``.
#: Treating it as unfinished would misclassify a completed simulation in summaries
#: and in the ledger.
FINISHED_STATUSES = frozenset({"COMPLETE", "WARNING"})

def evidence_role(*evidence):
    """Merge explicit intent conservatively; absent legacy intent stays unspecified."""
    roles = [item.get("role") for item in evidence if isinstance(item, dict)]
    if any(role is not None and role not in ("candidate", "diagnostic") for role in roles):
        raise ValueError("role must be candidate or diagnostic")
    return "diagnostic" if "diagnostic" in roles else "candidate" if "candidate" in roles else None


def require_candidate_eligible(rows, entries=None, registry_path=None):
    """Reject diagnostic evidence, including a stale selection of a known alpha."""
    entries = load_entries(registry_path) if entries is None else entries
    diagnostic_ids = {entry.get("alpha_id") for entry in entries
                      if evidence_role(entry) == "diagnostic" and entry.get("alpha_id")}
    for row in rows:
        if evidence_role(row) == "diagnostic" or row.get("alpha_id") in diagnostic_ids:
            raise ValueError("diagnostic evidence cannot be selected, promoted, shelved or submitted: %s"
                             % (row.get("alpha_id") or row.get("label") or "unspecified"))
    return rows

# Second-level grouping: which return source an alpha draws on. ``mechanism``
# is deliberately specific (one label per construction), so counting labels
# overstates diversity when several constructions draw on the same return source.
# ``family`` is the number to quote when the question is "how many independent
# bets do I have".
#
# This is a *declared* field, never inferred. Keyword inference looked
# tempting and is wrong in practice: an alpha built on analyst forward-EPS
# fields is a value bet, not an analyst-revision bet, so guessing from the
# field prefix silently mislabels it.
FAMILIES = (
    "oi",         # overnight vs intraday price formation
    "quality",    # value / profitability / valuation fundamentals
    "leverage",   # balance-sheet leverage, fixed vs operating cost structure
    "risk",       # distress, volatility shape, risk-premium cross-sections
    "option",     # implied volatility level, spread, skew
    "analyst",    # estimate revisions, coverage, dispersion
    "event",      # news, earnings and other dated events
    "reversal",   # short-horizon price reversal
    "momentum",   # price trend continuation
    "attention",  # social volume, short interest, liquidity
    "other",
)

# Metric keys kept from the in-sample block, in reporting order.
_METRIC_KEYS = (
    "sharpe", "fitness", "turnover", "returns", "drawdown", "margin",
    "longCount", "shortCount", "pnl",
)

# Check results that mean "this alpha did not clear the platform bar".
_FAILING_CHECK_RESULTS = frozenset({"FAIL", "ERROR"})


def default_registry_path():
    """Registry location: ``<research_dir>/alpha-registry.jsonl``."""
    return research_paths.resolve("alpha-registry.jsonl")


def expression_hash(expression, language=None):
    """Stable short hash of Alpha source, used in placeholder keys and lookups.

    Python hashes the source under ``py:`` after normalizing line endings.
    Indentation and whitespace inside strings can change the program, so all
    other text is retained.

    FASTEXPR uses ``expr2:`` for canonical trees. This collapses extra parentheses
    and comments. Unparseable source retains the legacy text hash under ``expr:``.
    Existing ledger keys stay intact to preserve historical deduplication.
    """
    if ml.is_python_language(language):
        source = ml.normalize_python_source(expression or "")
        return "py:" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    try:
        import fastexpr_ast

        fastexpr_ast.parse(expression)
        text, prefix = fastexpr_ast.canonical(expression), "expr2:"
    except Exception:  # noqa: BLE001 - unparseable input keeps the old rule
        text, prefix = " ".join((expression or "").split()), "expr:"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return prefix + digest[:16]


def entry_language(entry):
    """Language of normalized ledger evidence, including older top-level markers."""
    return ml.alpha_language(entry)


def entry_expression_hash(entry):
    """Get an identity for matching without rewriting historical ledger keys."""
    language = entry_language(entry)
    if ml.is_python_language(language):
        return expression_hash(entry.get("expression"), language)
    return entry.get("expression_hash")


def same_expression(left, right, language=None):
    """Compare FastExpr trees or exact Python source under one known language."""
    if ml.is_python_language(language):
        return ml.normalize_python_source(left or "") == ml.normalize_python_source(right or "")
    import fastexpr_ast

    return fastexpr_ast.canonical(left) == fastexpr_ast.canonical(right)


#: Settings that change what an expression measures. A record without an alpha id
#: used to key on the expression alone, so the same formula run on TOP3000 and
#: TOP500 collapsed into one entry and the second silently overwrote the first.
_FINGERPRINT_KEYS = ("universe", "region", "delay", "decay", "neutralization",
                     "truncation", "testPeriod", "instrumentType")


def settings_fingerprint(settings):
    """Short settings hash for records without a platform Alpha ID.

    Python includes the REGULAR wire settings, including lookback. FASTEXPR
    keeps the historical subset so legacy placeholder keys continue to match.
    """
    if not settings:
        return "nosettings"
    if ml.is_python_language(settings.get("language")):
        wire = ml.regular_wire_settings(settings)
        text = json.dumps(wire, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    parts = []
    for key in _FINGERPRINT_KEYS:
        if key in settings and settings[key] is not None:
            value = settings[key]
            if isinstance(value, float):
                value = round(value, 6)
            parts.append("%s=%s" % (key, value))
    if not parts:
        return "nosettings"
    digest = hashlib.sha256(";".join(parts).encode("utf-8")).hexdigest()
    return digest[:8]


def placeholder_key(expression, settings):
    """Key for a record that has no alpha id yet.

    Combines the expression with the settings fingerprint so two runs of one
    formula under different universes or decays stay distinct.
    """
    return "%s@%s" % (expression_hash(expression, (settings or {}).get("language")),
                      settings_fingerprint(settings))


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _pick_metrics(metrics):
    """Keep the numeric metrics we report on, dropping the rest."""
    if not isinstance(metrics, dict):
        return {}
    picked = {}
    for key in _METRIC_KEYS:
        if key in metrics:
            picked[key] = metrics[key]
    return picked


def _normalize_checks(checks):
    """Accept either the platform list or the lib's name->record mapping."""
    return normalize_checks(checks)


def _failing_checks(checks):
    return summarize_checks(checks)["failed"]


# These times describe when this ledger observed evidence, never the original
# simulation date. First observations do not date the current target value: a
# FAIL may precede a later PASS. Use checks_observed_at for the current payload.
EVIDENCE_FIELDS = (
    "first_recorded_at", "first_measurement_observed_at", "measurement_observed_at",
    "first_checks_observed_at", "checks_observed_at", "checks_source", "checks_state",
    "last_check_snapshot",
)


def _evidence_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _evidence_identities(entry):
    request = _evidence_digest({"expression": entry.get("expression"),
                                "settings": entry.get("settings") or {}})
    split = entry.get("out_of_sample")
    metrics = {"metrics": _pick_metrics(entry.get("metrics")),
               "out_of_sample": split if isinstance(split, dict) else {}}
    return request, _evidence_digest(metrics), _evidence_digest(_normalize_checks(entry.get("checks")))


def _observation_time(value):
    """Retain valid event times, including legacy naive times, without inventing one."""
    if not isinstance(value, str):
        return None
    try:
        datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value


def evidence_state(entry):
    """Compact replay state; the index need not reread previous ledger payloads."""
    request, measurement, checks = _evidence_identities(entry)
    state = {key: entry.get(key) for key in EVIDENCE_FIELDS if key != "last_check_snapshot"}
    state.update(request_identity=request, measurement_identity=measurement, check_identity=checks)
    current = _normalize_checks(entry.get("checks"))
    snapshot = entry.get("last_check_snapshot")
    if current and entry.get("checks_state") != "partial":
        snapshot = {"checks": current, "observed_at": entry.get("checks_observed_at"),
                    "source": entry.get("checks_source"), "request_identity": request,
                    "simulation_mode": (entry.get("settings") or {}).get("simulationMode")}
    state["snapshot"] = snapshot
    return state


def fold_evidence(entry, previous=None):
    """Fold one event at append and replay boundaries, without rewriting history.

    ``previous`` is evidence_state(), not the previous full payload. Metadata
    events cannot establish a measurement time, even if their copied row has
    metrics. One prior check snapshot survives a missing/partial refresh but is
    never inserted into the current ``checks`` or treated as a current pass.
    """
    entry = dict(entry)
    previous = previous or {}
    request, measurement, check_identity = _evidence_identities(entry)
    same_request = previous.get("request_identity") == request
    source = entry.get("source")
    metadata_only = str(source or "").startswith(("relabel", "dedupe"))
    stamp = _observation_time(entry.get("recorded_at"))
    # Explicit provenance survives exporting the newest row into a fresh ledger.
    saved = entry if entry.get("evidence_version") == 1 else {}
    entry["evidence_version"] = 1
    entry["first_recorded_at"] = (previous.get("first_recorded_at")
                                  or _observation_time(saved.get("first_recorded_at")) or stamp)
    for field in ("first_measurement_observed_at", "first_checks_observed_at"):
        entry[field] = previous.get(field) or _observation_time(saved.get(field))

    split = entry.get("out_of_sample") or {}
    measured = _pick_metrics(entry.get("metrics"))
    if isinstance(split, dict):
        for period in ("train", "test"):
            measured.update({period + ":" + key: value
                             for key, value in _pick_metrics(split.get(period)).items()})
    has_measurement = any(isinstance(value, (int, float)) and not isinstance(value, bool)
                          and math.isfinite(value) for value in measured.values())
    if has_measurement:
        if (same_request and previous.get("measurement_identity") == measurement
                and (previous.get("measurement_observed_at") is not None or metadata_only)):
            observed = previous.get("measurement_observed_at")
        elif not previous and saved:
            observed = _observation_time(saved.get("measurement_observed_at"))
        else:
            observed = None if metadata_only else stamp
    else:
        observed = None
    entry["measurement_observed_at"] = observed
    entry["first_measurement_observed_at"] = entry["first_measurement_observed_at"] or observed

    snapshot = previous.get("snapshot") if same_request else None
    if not previous and saved:
        snapshot = saved.get("last_check_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("request_identity") != request:
        snapshot = None
    checks = _normalize_checks(entry.get("checks"))
    entry["checks"] = checks
    if checks:
        entry["failing_checks"] = _failing_checks(checks)
        if (same_request and previous.get("check_identity") == check_identity
                and (previous.get("checks_observed_at") is not None or metadata_only)):
            observed, origin = previous.get("checks_observed_at"), previous.get("checks_source")
        elif not previous and saved:
            observed, origin = _observation_time(saved.get("checks_observed_at")), saved.get("checks_source")
        else:
            observed, origin = (None, None) if metadata_only else (stamp, source)
        partial = bool(snapshot and platform_check_names(snapshot.get("checks"))
                       - platform_check_names(checks))
        entry.update(checks_state="partial" if partial else "observed",
                     checks_observed_at=observed, checks_source=origin)
        entry["first_checks_observed_at"] = entry["first_checks_observed_at"] or observed
        if not partial:
            snapshot = None
    else:
        entry.setdefault("failing_checks", [])
        entry.update(checks_state="stale" if snapshot else "missing",
                     checks_observed_at=None, checks_source=None)
    if snapshot:
        entry["last_check_snapshot"] = snapshot
    else:
        entry.pop("last_check_snapshot", None)
    # ``ledger record`` has always defaulted to watch, so a legacy nondefault
    # verdict on that path was an explicit CLI judgement even before the flag.
    if source == "record" and entry.get("verdict") not in (None, "watch"):
        entry["verdict_manual"] = True
    # Repair only recognizable automatic defaults in legacy views. A manual
    # provisional candidate label is separate from resolved platform evidence.
    if (entry.get("verdict") == "candidate" and not entry.get("verdict_manual")
            and source in {"simulate", "sync", "submit"}
            and (entry["checks_state"] != "observed" or not summarize_checks(checks)["resolved"])):
        entry["verdict"] = "watch"
    return entry


def build_entry(alpha, idea=None, mechanism=None, verdict="watch", note=None,
                source="manual", extra=None, family=None, batch=None):
    """Build one ledger entry from a normalized alpha fact.

    ``alpha`` is expected to carry ``alpha_id``, ``expression``, ``decay``,
    ``settings``, ``metrics`` and ``checks``. Missing pieces are tolerated, so a
    rejected expression with no alpha id can still be recorded.

    ``family`` groups entries by return source; see ``FAMILIES``. It is
    declared by the caller and never guessed from the expression.
    """
    if verdict not in VERDICTS:
        raise ValueError("verdict must be one of %s, got %r"
                         % (", ".join(VERDICTS), verdict))
    if family is not None and family not in FAMILIES:
        raise ValueError("family must be one of %s, got %r"
                         % (", ".join(FAMILIES), family))
    expression = alpha.get("expression") or ""
    alpha_id = alpha.get("alpha_id")
    metrics = _pick_metrics(alpha.get("metrics"))
    checks = _normalize_checks(alpha.get("checks"))
    settings = alpha.get("settings") or {}
    language = entry_language(alpha)
    if language and not settings.get("language"):
        settings = {**settings, "language": language}
    if alpha.get("decay") is not None and "decay" not in settings:
        settings = dict(settings)
        settings["decay"] = alpha["decay"]

    entry = {
        # A record without an id used to key on the expression alone, so the same
        # formula at a different universe or decay overwrote the earlier one.
        "key": alpha_id or placeholder_key(expression, settings),
        "alpha_id": alpha_id,
        "role": evidence_role(alpha),
        "idea": idea,
        "mechanism": mechanism,
        "family": family,
        "verdict": verdict,
        "note": note,
        "expression": expression,
        "expression_hash": expression_hash(expression, language),
        "settings": settings,
        "settings_fingerprint": settings_fingerprint(settings),
        "metrics": metrics,
        "failing_checks": _failing_checks(checks),
        "checks": checks,
        "status": alpha.get("status"),
        "error": alpha.get("error"),
        "source": source,
        # Which batch produced this row. Without it "where did this number come from"
        # is answered from memory, and memory is exactly what a ledger exists to replace.
        "batch": batch,
        "recorded_at": _now(),
    }
    if extra:
        entry["extra"] = extra
    # Lineage is recorded, never inferred: an ablation variant knows which
    # composite it came from and by which operation, and a research file knows
    # nothing, so the field stays absent rather than guessed.
    parent = alpha.get("parent")
    if parent:
        entry["parent"] = parent
    operation = alpha.get("op")
    if operation:
        entry["op"] = operation
    legs = alpha.get("legs")
    if legs:
        entry["legs"] = list(legs)
    split = alpha.get("out_of_sample")
    if isinstance(split, dict):
        # BRAIN's own train/test split is the only out-of-sample evidence the
        # API exposes, so keep it next to the full-window metrics.
        entry["out_of_sample"] = split
    return entry


def load_entries(path=None):
    """Load the ledger, newest record per key winning.

    Tolerates a missing file (returns ``[]``) and skips malformed lines,
    including a write interrupted in the middle of a UTF-8 character.
    """
    path = path or default_registry_path()
    try:
        handle = open(path, "rb")
    except FileNotFoundError:
        return []
    by_key = {}
    with handle:
        for line in handle:
            key, entry = _parse_line(line)
            if entry is None:
                continue
            previous = by_key.get(key)
            entry = fold_evidence(entry, evidence_state(previous) if previous else None)
            entry["role"] = evidence_role(by_key.get(key), entry)
            by_key[key] = entry
    return list(by_key.values())


def _parse_line(line):
    try:
        entry = json.loads(line.decode("utf-8").strip())
    except ValueError:
        return None, None
    if not isinstance(entry, dict):
        return None, None
    key = entry.get("key") or entry.get("alpha_id")
    if not key:
        expression = entry.get("expression") or ""
        language = entry_language(entry)
        if ml.is_python_language(language):
            settings = {**(entry.get("settings") or {}), "language": language}
            key = placeholder_key(expression, settings)
        else:
            key = expression_hash(expression, language)
    entry.setdefault("key", key)
    return key, entry


@contextmanager
def indexed_registry(path=None):
    """Open an indexed snapshot, yielding None when the local cache is unavailable."""
    from registry_index import index_path, open_index, sqlite3

    path = os.fspath(path or default_registry_path())
    if not os.path.exists(path) and not os.path.exists(index_path(path)):
        yield None
        return
    with ExitStack() as stack, ExitStack() as locked:
        try:
            locked.enter_context(cache_store.file_lock(path + ".lock", timeout=30))
            index = stack.enter_context(open_index(path))
            if index is not None:
                index.begin_snapshot()
        except (cache_store.CacheError, sqlite3.Error, OSError):
            index = None
        locked.close()
        yield index


#: Fields that change on every append and therefore cannot signal "content changed".
_VOLATILE = frozenset({"recorded_at", "source"})


def _comparable(entry):
    """Everything in an entry except the volatile bookkeeping fields.

    This used to be a hand-maintained whitelist, which failed the same way three
    times: a new field was added, the whitelist was not, and a write whose only
    change was that field got skipped as "unchanged". ``family`` lost nine of
    thirteen classifications that way, and ``duplicate_of`` made every dedupe mark
    a no-op that still reported success.

    Deriving the comparison from the record's own keys removes the class of bug:
    anything stored is compared, so nothing can be silently dropped again.
    """
    comparable = {key: value for key, value in entry.items()
                  if key not in _VOLATILE}
    if "expression" in comparable:
        expression = comparable.get("expression") or ""
        comparable["expression"] = (ml.normalize_python_source(expression)
                                    if ml.is_python_language(entry_language(entry))
                                    else " ".join(expression.split()))
    return comparable


def record_entry(entry, path=None, skip_unchanged=True, allow_metrics_loss=False):
    """Append one entry to the ledger.

    With ``skip_unchanged`` (the default) an entry identical to the newest
    stored record for the same key is not rewritten, which keeps bulk account
    syncs from growing the file on every run. Returns ``(entry, written)``.

    ``allow_metrics_loss`` guards the one mistake this module cannot recover from:
    appending a row for a key whose stored row carried metrics while the new one
    carries none. Because the newest row wins, that silently deletes the metrics from
    every view -- it happened on 2026-11-01 with a misread platform payload, and the
    record looked fine afterwards. Refusing is cheap; the caller who genuinely means
    it can say so.
    """
    with append_batch(path) as (_, append):
        return append(entry, skip_unchanged=skip_unchanged, allow_metrics_loss=allow_metrics_loss)


@contextmanager
def append_batch(path=None):
    """Load the latest rows under the append lock for a batch of local writes.

    Use the yielded index for any merge, and the bound append function for
    writes. Successful rows are individually durable and update the index.
    Collect platform data before entering the batch.
    """
    from registry_index import LedgerRows, open_index

    path = os.fspath(path or default_registry_path())
    with cache_store.file_lock(path + ".lock", timeout=30), ExitStack() as stack:
        index = stack.enter_context(open_index(path))
        existing = LedgerRows(index) if index is not None else {
            item["key"]: item for item in load_entries(path)
        }
        handle = None
        check_tail = True

        def write_line(entry):
            nonlocal handle, check_tail
            data = (json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
            if handle is None:
                handle = stack.enter_context(open(path, "a+b", buffering=0))
            if check_tail and handle.seek(0, os.SEEK_END):
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    # Preserve a legacy final record or isolate a torn write.
                    data = b"\n" + data
            check_tail = True
            if handle.write(data) != len(data):
                raise OSError(errno.EIO, "short write to the alpha registry", path)
            os.fsync(handle.fileno())
            check_tail = False

        try:
            yield existing, partial(_append_locked, existing, write_line)
        finally:
            if index is not None and handle is not None:
                index.refresh_after_append()


def _append_locked(existing, write_line, entry, *, skip_unchanged=True, allow_metrics_loss=False):
    entry = dict(entry)
    entry.setdefault("recorded_at", _now())
    key = entry.get("key")
    current = existing.get(key)
    entry = fold_evidence(entry, evidence_state(current) if current else None)
    # Attach the rejection cause while the observed check and metrics are present.
    if entry.get("verdict") == "rejected" and not entry.get("reason_code"):
        try:
            import rejection_reasons

            code, detail = rejection_reasons.classify(entry)
            entry["reason_code"] = code
            entry["reason_detail"] = detail
        except Exception:  # noqa: BLE001 - recording must never fail on this
            pass

    entry["role"] = evidence_role(current, entry)
    if current is not None and not allow_metrics_loss and current.get("metrics") and not entry.get("metrics"):
        raise ValueError(
            "refusing to record %r: the stored row has metrics and this one "
            "does not, and the newest row wins -- the metrics would disappear "
            "from every view. Fix the payload (BRAIN puts them under 'is'), or "
            "pass allow_metrics_loss=True if the loss is intended."
            % (key,))
    if skip_unchanged and current is not None and _comparable(current) == _comparable(entry):
        return current, False
    write_line(entry)
    existing[key] = entry
    return entry, True


def auto_verdict(record):
    """Classify a simulation record without pretending to be a researcher.

    Only a default, for the two paths that persist results before anyone has written
    a judgement: ``simulate --record`` and the research template. Anything that
    cleared the usual bar is a ``candidate``; anything that failed a platform check
    is ``rejected``; the rest stays ``watch`` until someone decides.

    It lives here rather than in the CLI because the ledger is what it classifies,
    and because the same mapping had been written twice - once for the CLI and once
    by hand in each research script that recorded its own results.
    """
    # WARNING is a finished run: the platform marks a simulation WARNING when a check
    # warns without failing (UNITS is the common one), and every metric is final. It
    # must not fall into the "not finished" branch and be filed as `rejected` merely
    # because a non-failing check emitted a warning.
    if (record.get("status") or "").upper() not in FINISHED_STATUSES | {"UNSUBMITTED", "ACTIVE", "SUBMITTED"}:
        return "watch"
    summary = summarize_checks(record.get("checks"))
    if summary["failed"]:
        return "rejected"
    if not summary["resolved"] or record.get("checks_state") in {"missing", "stale", "partial"}:
        return "watch"
    metrics = record.get("metrics") or {}
    sharpe = metrics.get("sharpe")
    fitness = metrics.get("fitness")
    if isinstance(sharpe, (int, float)) and isinstance(fitness, (int, float)):
        if sharpe >= 1.25 and fitness >= 1.0:
            return "candidate"
    return "watch"


def _inherit_judgement(entry, previous, pinned_verdict=False):
    """Keep human labels when a simulation is re-recorded without them.

    Metrics and checks are measurements and are always replaced. Family,
    mechanism, idea, note and lineage are judgement - ``ledger label`` exists
    precisely so a person can set them - and a re-run that carries no opinion must
    not erase them. Before this, re-simulating an id blanked ``family`` and
    ``mechanism`` and dropped ``parent``/``op``, so the ledger's lineage view lost
    the record while the newest-row-wins rule kept the blank one.
    """
    # `batch` belongs here too: a re-record without a batch name is a refresh of an
    # existing row, not evidence that the row came from nowhere.
    manual_fields = set(previous.get("manual_fields") or [])
    entry["role"] = evidence_role(previous, entry)
    if previous.get("source") == "relabel":
        manual_fields.update(("family", "mechanism", "idea", "note"))
    for field in ("family", "mechanism", "idea", "note", "parent", "op", "legs", "batch"):
        if (not entry.get(field) or field in manual_fields) and previous.get(field) is not None:
            entry[field] = previous[field]
    if manual_fields:
        entry["manual_fields"] = sorted(manual_fields)
    # An automatic verdict may not overwrite a verdict a human set; ``relabel``
    # marks those with ``verdict_manual``.
    if (previous.get("verdict_manual") and not pinned_verdict
            and previous.get("verdict")):
        entry["verdict"] = previous["verdict"]
        entry["verdict_manual"] = True
    elif previous.get("verdict_manual"):
        entry["verdict_manual"] = True
    return entry


def record_simulation(records, settings, idea=None, mechanism=None, family=None,
                       note=None, verdict=None, registry_path=None, source="simulate",
                       batch=None):
    """Append one ledger entry per finished simulation record; return a summary.

    ``record["mechanism"]`` and ``record["family"]`` win over the run-wide default,
    so a batch spanning several mechanisms still lands labelled per entry. ``verdict``
    pins the verdict for every entry; without it each record is classified by
    :func:`auto_verdict`, which is a default and not a judgement.

    The research template calls this too. Before it existed the template recorded
    nothing, so an experiment copied from it left no trace in the ledger - the file
    every experiment is copied from was the one path that could silently skip the
    durable record.
    """
    path = registry_path or default_registry_path()
    entries = []
    written = 0
    skipped = []
    deferred = []
    with ExitStack() as stack:
        stored = append = None
        for record in records:
            key = record.get("alpha_id")
            status = record.get("status")
            evidence_status = (status or "").upper()
            if evidence_status not in FINISHED_STATUSES | {"UNSUBMITTED", "ACTIVE", "SUBMITTED"}:
                deferred.append({"alpha_id": key, "label": record.get("mechanism"),
                                 "status": status, "reason": "execution_not_finished"})
                continue
            if not key or not _pick_metrics(record.get("metrics")):
                deferred.append({"alpha_id": key, "label": record.get("mechanism"),
                                 "status": status, "reason": "completed_evidence_missing"})
                continue
            if append is None:
                stored, append = stack.enter_context(append_batch(path))
            previous = stored.get(key)
            chosen_verdict = verdict or auto_verdict(record)
            if previous and (previous.get("status") or "").upper() in {"ACTIVE", "SUBMITTED"}:
                # A simulation cannot un-submit a book alpha. BRAIN hands back the
                # existing id when an identical expression is re-run, so a re-run of a
                # book member arrives here looking like an ordinary new record. Preserve
                # its ACTIVE status and existing verdict rather than demoting it.
                if (status or "").upper() != "ACTIVE":
                    status = previous.get("status")
                if not verdict:
                    chosen_verdict = previous.get("verdict") or chosen_verdict
            entry = build_entry(
                {
                    "alpha_id": key,
                    "role": record.get("role"),
                    "expression": record.get("expression"),
                    "decay": record.get("decay"),
                    "settings": record.get("settings") or settings,
                    "language": entry_language(record),
                    "metrics": record.get("metrics"),
                    "checks": record.get("checks"),
                    "out_of_sample": record.get("out_of_sample"),
                    "status": status,
                    "error": record.get("error"),
                },
                idea=idea,
                mechanism=record.get("mechanism") or mechanism,
                family=record.get("family") or family,
                verdict=chosen_verdict,
                note=note,
                source=source,
                batch=batch or record.get("batch"),
            )
            if previous:
                entry = _inherit_judgement(entry, previous, pinned_verdict=verdict is not None)
                entry["metrics"] = {**(previous.get("metrics") or {}), **entry["metrics"]}
                old_split = previous.get("out_of_sample") or {}
                new_split = entry.get("out_of_sample") or {}
                if old_split:
                    entry["out_of_sample"] = {**old_split, **new_split}
                    for period in ("train", "test"):
                        if old_split.get(period):
                            entry["out_of_sample"][period] = {**old_split[period],
                                **{key: value for key, value in (new_split.get(period) or {}).items()
                                   if value is not None}}
            if verdict is not None:
                entry["verdict_manual"] = True
            try:
                entry, did_write = append(entry)
            except ValueError as exc:
                # `record_entry` refuses to let a metrics-less row erase a metrics-bearing
                # one, which is right. Abandoning the rest of the batch is not: on
                # 2026-11-13 a grid died after its third arm because that arm (a known
                # expression under a new setting) came back without metrics. Keep the
                # stored row, say which arm was skipped, and finish the run.
                skipped.append({"alpha_id": key, "reason": str(exc)})
                continue
            written += 1 if did_write else 0
            entries.append(entry)
    return {
        "registry": path,
        "recorded": len(entries),
        "appended": written,
        "skipped": skipped,
        "deferred": deferred,
        "deferred_count": len(deferred),
        "skipped_count": len(skipped),
        "unchanged": len(entries) - written,
        "by_verdict": {
            name: sum(1 for entry in entries if entry["verdict"] == name)
            for name in VERDICTS
        },
    }


def find_entries(entries, field=None, mechanism=None, idea=None,
                 expression=None, verdicts=None, min_fitness=None,
                 family=None, min_sharpe=None, regex=False, all_fields=None):
    """Filter ledger entries by field usage, mechanism, idea or expression.

    ``field`` is matched as a substring of the recorded expression, which is
    the practical way to answer "has this datafield already been tested?"
    without spending a simulation to find out. Matching is case-insensitive
    and every supplied filter must agree (AND).
    """
    # `all_fields` answers a different question than `field`: not "does this id appear"
    # but "does this expression use both A and B", which is how a leg combination is
    # recognised. `regex` exists because substring matching cannot say "any ts_corr of an
    # absolute series".
    wanted_fields = [part.strip().lower() for part in (all_fields or "").split(",")
                     if part.strip()]
    pattern = None
    if regex and expression:
        try:
            pattern = re.compile(expression, re.IGNORECASE)
        except re.error as exc:
            raise ValueError("bad --regex: %s" % exc) from exc
    needle_field = (field or "").lower()
    needle_mechanism = (mechanism or "").lower()
    needle_idea = (idea or "").lower()
    needle_expression = (expression or "").lower()
    wanted_verdicts = {str(v).lower() for v in (verdicts or [])}
    wanted_family = (family or "").lower()

    matched = []
    for entry in entries:
        code = (entry.get("expression") or "").lower()
        if needle_field and needle_field not in code:
            continue
        if pattern is not None:
            if not pattern.search(entry.get("expression") or ""):
                continue
        elif needle_expression and needle_expression not in code:
            continue
        if wanted_fields and not all(needle in code for needle in wanted_fields):
            continue
        if needle_mechanism and needle_mechanism not in (
                entry.get("mechanism") or "").lower():
            continue
        if needle_idea and needle_idea not in (entry.get("idea") or "").lower():
            continue
        if wanted_family and (entry.get("family") or "").lower() != wanted_family:
            continue
        if wanted_verdicts and (entry.get("verdict") or "").lower() not in wanted_verdicts:
            continue
        if min_fitness is not None:
            fitness = (entry.get("metrics") or {}).get("fitness")
            if not isinstance(fitness, (int, float)) or fitness < min_fitness:
                continue
        if min_sharpe is not None:
            sharpe = (entry.get("metrics") or {}).get("sharpe")
            if not isinstance(sharpe, (int, float)) or sharpe < min_sharpe:
                continue
        matched.append(entry)
    return matched


def summarize(entries):
    """Aggregate counts and the headline list, for a quick orientation read."""
    by_verdict = {verdict: 0 for verdict in VERDICTS}
    by_idea = {}
    by_family = {}
    unclassified = 0
    for entry in entries:
        verdict = entry.get("verdict") or "watch"
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
        idea = entry.get("idea") or "unassigned"
        by_idea[idea] = by_idea.get(idea, 0) + 1
        family = entry.get("family")
        if family:
            by_family[family] = by_family.get(family, 0) + 1
        else:
            unclassified += 1

    def sharpe_key(entry):
        value = (entry.get("metrics") or {}).get("sharpe")
        return value if isinstance(value, (int, float)) else float("-inf")

    ranked = sorted(entries, key=sharpe_key, reverse=True)
    return {
        "total": len(entries),
        "by_verdict": by_verdict,
        "by_idea": dict(sorted(by_idea.items())),
        "by_family": dict(sorted(by_family.items())),
        "family_count": len(by_family),
        "unclassified_family": unclassified,
        "best": [
            {
                "alpha_id": entry.get("alpha_id"),
                "sharpe": (entry.get("metrics") or {}).get("sharpe"),
                "fitness": (entry.get("metrics") or {}).get("fitness"),
                "idea": entry.get("idea"),
                "verdict": entry.get("verdict"),
            }
            for entry in ranked[:10]
        ],
    }


def render_markdown(entries, title="Alpha registry"):
    """Render the ledger as a compact Markdown table for human review."""
    def fmt(value, digits=2):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return ("%%.%df" % digits) % value
        return "—" if value is None else str(value)

    lines = ["# " + title, ""]
    if not entries:
        lines.append("_Empty._")
        return "\n".join(lines) + "\n"
    summary = summarize(entries)
    counts = ", ".join("%s %d" % (name, count)
                       for name, count in summary["by_verdict"].items() if count)
    lines.append("%d alphas recorded (%s)." % (summary["total"], counts or "none"))
    lines.append("")
    lines.append("| alpha | role | sharpe | fitness | turnover | idea | verdict | note |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for entry in sorted(
        entries,
        key=lambda item: (item.get("metrics") or {}).get("sharpe") or float("-inf"),
        reverse=True,
    ):
        metrics = entry.get("metrics") or {}
        note = (entry.get("note") or "").replace("|", "/").replace("\n", " ")
        if len(note) > 80:
            note = note[:77] + "..."
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            entry.get("alpha_id") or "—",
            entry.get("role") or "unspecified",
            fmt(metrics.get("sharpe")),
            fmt(metrics.get("fitness")),
            fmt(metrics.get("turnover"), 3),
            entry.get("idea") or "—",
            entry.get("verdict") or "—",
            note or "—",
        ))
    return "\n".join(lines) + "\n"


def relabel(entry, family=None, verdict=None, mechanism=None, idea=None,
            note=None):
    """Return a copy of ``entry`` with labels replaced.

    Labels are the part of a record that is judgement rather than measurement.
    Missing family labels make independent mechanisms hard to count. Changing a
    label appends a new record under the same key, so history is preserved and the
    newest record wins.
    """
    updated = dict(entry)
    if family is not None:
        if family not in FAMILIES:
            raise ValueError("family must be one of %s, got %r"
                             % (", ".join(FAMILIES), family))
        updated["family"] = family
    if verdict is not None:
        if verdict not in VERDICTS:
            raise ValueError("verdict must be one of %s, got %r"
                             % (", ".join(VERDICTS), verdict))
        updated["verdict"] = verdict
        # A human decision, so a later automatic record must not overwrite it.
        updated["verdict_manual"] = True
    if mechanism is not None:
        updated["mechanism"] = mechanism
    if idea is not None:
        updated["idea"] = idea
    if note is not None:
        updated["note"] = note
    manual_fields = set(updated.get("manual_fields") or [])
    manual_fields.update(key for key, value in {"family": family, "mechanism": mechanism,
                                               "idea": idea, "note": note}.items() if value is not None)
    if manual_fields:
        updated["manual_fields"] = sorted(manual_fields)
    updated["recorded_at"] = _now()
    updated["source"] = "relabel"
    return updated


def derive_legs(expression, language=None):
    """Leg names parsed from an expression, or an empty list.

    Derived on demand rather than migrated into every record: the ledger predates
    the legs column, and a backfill would have to be re-run whenever the parser
    improves. Parsing is cheap and the grammar is the same one the ablation uses.
    """
    if ml.is_python_language(language):
        return []
    try:
        import ablation
        return list(ablation.parse(expression)["used"])
    except Exception:  # noqa: BLE001 - an unparseable expression simply has no legs
        return []


def legs_of(entry):
    """Declared legs, falling back to parsing the recorded expression."""
    legs = entry.get("legs")
    if legs:
        return list(legs)
    return derive_legs(entry.get("expression"), entry_language(entry))


def leg_bodies(entry):
    """``{leg_name: normalized_body}`` for an expression.

    Leg names are positional — ``a`` means whatever the author put there — so a
    name is not an identity. Comparing bodies is what makes "the same leg" mean
    the same sub-expression.
    """
    if ml.is_python_language(entry_language(entry)):
        return {}
    try:
        import ablation
        parsed = ablation.parse(entry.get("expression"))
    except Exception:  # noqa: BLE001 - no legs means no leg-level exposure
        return {}
    return {name: " ".join(body.split())
            for name, body in parsed["legs"].items()}


def shared_legs(entries, reference, min_shared=1):
    """Other entries that stand on the same leg bodies as ``reference``.

    This is the exposure view correlation cannot give: two alphas can have
    uncorrelated PnL and still both stand on the deferred-revenue ratio, in which
    case a judgement about that leg applies to both.
    """
    target = resolve(entries, reference)
    if target is None:
        return []
    target_bodies = {}
    for name, body in leg_bodies(target).items():
        target_bodies[body] = name
    rows = []
    for entry in entries:
        if entry.get("key") == target.get("key"):
            continue
        matched = [body for body in leg_bodies(entry).values()
                   if body in target_bodies]
        if len(matched) >= min_shared:
            rows.append({
                "key": entry.get("key"),
                "alpha_id": entry.get("alpha_id"),
                "verdict": entry.get("verdict"),
                "shared_legs": len(matched),
                "bodies": matched,
            })
    rows.sort(key=lambda row: row["shared_legs"], reverse=True)
    return rows


def leg_index(entries):
    """Map each normalized leg body to the keys that use it."""
    index = {}
    for entry in entries:
        for body in leg_bodies(entry).values():
            index.setdefault(body, []).append(entry.get("key"))
    return {body: sorted(set(keys)) for body, keys in index.items()}


def resolve(entries, reference):
    """Find one entry by alpha id, ledger key or expression hash."""
    if not reference:
        return None
    for entry in entries:
        if reference in (entry.get("alpha_id"), entry.get("key"),
                         entry.get("expression_hash")):
            return entry
        if isinstance(reference, str) and reference.startswith("py:") and reference == entry_expression_hash(entry):
            return entry
    return None


def lineage(entries, reference, max_depth=32):
    """Walk parent edges from ``reference`` upward.

    Returns the chain from the entry itself to the oldest ancestor. A broken or
    missing parent ends the walk instead of looping: the ledger is append-only and
    a record may predate lineage being recorded at all.
    """
    chain = []
    seen = set()
    current = resolve(entries, reference)
    while current is not None and len(chain) < max_depth:
        key = current.get("key") or current.get("alpha_id")
        if key in seen:
            break
        seen.add(key)
        chain.append(current)
        parent = current.get("parent")
        current = resolve(entries, parent) if parent else None
    return chain


def descendants(entries, reference):
    """Every entry that reaches ``reference`` through parent edges."""
    target = resolve(entries, reference)
    if target is None:
        return []
    target_key = target.get("key")
    target_hash = entry_expression_hash(target)
    out = []
    for entry in entries:
        parent = entry.get("parent")
        if parent and parent in (target_key, target_hash, target.get("expression_hash"), target.get("alpha_id")):
            out.append(entry)
    return out


def unlabeled_summary(entries):
    """How much of the ledger still lacks the columns independence is counted on."""
    total = len(entries)
    missing_family = sum(1 for e in entries if not e.get("family"))
    missing_mechanism = sum(1 for e in entries if not e.get("mechanism"))
    return {
        "total": total,
        "missing_family": missing_family,
        "missing_mechanism": missing_mechanism,
        "family_coverage": round(1 - missing_family / total, 3) if total else None,
    }


def mark_duplicate(entry, canonical):
    """Record that this alpha's returns are indistinguishable from another's.

    Two ids can carry the same PnL: submitting the same expression twice yields a
    new id, and an algebraically zero term added to force a fresh simulation does
    the same. Those are not two findings, and leaving them uncounted inflates totals.
    The mark is a field rather than a verdict: being a duplicate says nothing about
    whether the alpha is any good.
    """
    updated = dict(entry)
    updated["duplicate_of"] = canonical
    updated["recorded_at"] = _now()
    updated["source"] = "dedupe"
    return updated


def unique_entries(entries):
    """Entries that are not marked as a duplicate of another."""
    return [entry for entry in entries if not entry.get("duplicate_of")]
