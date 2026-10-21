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
import hashlib
import json
import os

import research_paths
import cache_store

__all__ = [
    "VERDICTS",
    "default_registry_path",
    "expression_hash",
    "load_entries",
    "record_entry",
    "render_markdown",
    "summarize",
]

# ``rejected``  - the alpha underperformed; the expression is dead, the
#                 hypothesis was never properly on trial.
# ``falsified`` - a pre-registered hypothesis was put on trial with controls
#                 and the controls won. This is a *result*, not a failure: it
#                 tells a later round not to re-open the mechanism, and it is
#                 the reason `alpha find` exists. Keep the note specific about
#                 which control killed it.
# ``no_go``     - abandoned before a fair test (data blocked, unusable fields).
VERDICTS = ("candidate", "watch", "rejected", "falsified", "no_go")

# Second-level grouping: which return source an alpha draws on. ``mechanism``
# is deliberately specific (one label per construction), so counting labels
# overstates diversity -- thirteen book alphas carry eleven labels but only
# four families, and six of the labels are the same overnight/intraday bet.
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


def expression_hash(expression):
    """Stable short hash of a normalized expression, used as fallback key."""
    text = " ".join((expression or "").split())
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return "expr:" + digest[:16]


#: Settings that change what an expression measures. A record without an alpha id
#: used to key on the expression alone, so the same formula run on TOP3000 and
#: TOP500 collapsed into one entry and the second silently overwrote the first.
_FINGERPRINT_KEYS = ("universe", "region", "delay", "decay", "neutralization",
                     "truncation", "testPeriod", "instrumentType")


def settings_fingerprint(settings):
    """Short hash of the settings that decide what a simulation measures.

    Only the inputs that alter results are included; cosmetic keys such as
    ``language`` or ``visualization`` would make two identical experiments look
    different.
    """
    if not settings:
        return "nosettings"
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
    return "%s@%s" % (expression_hash(expression), settings_fingerprint(settings))


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
    normalized = {}
    if isinstance(checks, dict):
        for name, record in checks.items():
            if isinstance(record, dict):
                normalized[name] = {
                    "result": record.get("result"),
                    "limit": record.get("limit"),
                    "value": record.get("value"),
                }
            else:
                normalized[name] = {"result": record}
    elif isinstance(checks, list):
        for record in checks:
            if not isinstance(record, dict):
                continue
            name = record.get("name")
            if not name:
                continue
            normalized[name] = {
                "result": record.get("result"),
                "limit": record.get("limit"),
                "value": record.get("value"),
            }
    return normalized


def _failing_checks(checks):
    return sorted(
        name for name, record in checks.items()
        if str(record.get("result") or "").upper() in _FAILING_CHECK_RESULTS
    )


def build_entry(alpha, idea=None, mechanism=None, verdict="watch", note=None,
                source="manual", extra=None, family=None):
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
    if alpha.get("decay") is not None and "decay" not in settings:
        settings = dict(settings)
        settings["decay"] = alpha["decay"]

    entry = {
        # A record without an id used to key on the expression alone, so the same
        # formula at a different universe or decay overwrote the earlier one.
        "key": alpha_id or placeholder_key(expression, settings),
        "alpha_id": alpha_id,
        "idea": idea,
        "mechanism": mechanism,
        "family": family,
        "verdict": verdict,
        "note": note,
        "expression": expression,
        "expression_hash": expression_hash(expression),
        "settings": settings,
        "settings_fingerprint": settings_fingerprint(settings),
        "metrics": metrics,
        "failing_checks": _failing_checks(checks),
        "checks": checks,
        "status": alpha.get("status"),
        "error": alpha.get("error"),
        "source": source,
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

    Tolerates a missing file (returns ``[]``) and skips malformed lines rather
    than failing the caller.
    """
    path = path or default_registry_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return []
    by_key = {}
    order = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entry = json.loads(stripped)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        key = entry.get("key") or entry.get("alpha_id") or \
            expression_hash(entry.get("expression") or "")
        entry.setdefault("key", key)
        if key not in by_key:
            order.append(key)
        by_key[key] = entry
    return [by_key[key] for key in order]


def _comparable(entry):
    """The subset of an entry that decides whether a re-record is needed."""
    return {
        "verdict": entry.get("verdict"),
        "idea": entry.get("idea"),
        "mechanism": entry.get("mechanism"),
        # ``family`` is a curated annotation, not derived from anything else,
        # so it has to count as a change on its own. Leaving it out silently
        # dropped the first attempt to classify the book: nine of thirteen
        # records were skipped as "unchanged" when only the family differed.
        "family": entry.get("family"),
        "note": entry.get("note"),
        "expression": " ".join((entry.get("expression") or "").split()),
        "settings": entry.get("settings") or {},
        "metrics": entry.get("metrics") or {},
        "out_of_sample": entry.get("out_of_sample") or {},
        "failing_checks": entry.get("failing_checks") or [],
    }


def record_entry(entry, path=None, skip_unchanged=True):
    """Append one entry to the ledger.

    With ``skip_unchanged`` (the default) an entry identical to the newest
    stored record for the same key is not rewritten, which keeps bulk account
    syncs from growing the file on every run. Returns ``(entry, written)``.
    """
    path = path or default_registry_path()
    # The dedupe check and the append have to happen under one lock: two
    # concurrent CLI calls otherwise both see "not present" and both append.
    # JSONL appends stay line-atomic because each record is written in a single
    # unbuffered write, so a reader never sees half a line.
    with cache_store.file_lock(path + ".lock", timeout=30):
        if skip_unchanged:
            existing = {item["key"]: item for item in load_entries(path)}
            current = existing.get(entry.get("key"))
            if current is not None and _comparable(current) == _comparable(entry):
                return current, False
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return entry, True


def find_entries(entries, field=None, mechanism=None, idea=None,
                 expression=None, verdicts=None, min_fitness=None,
                 family=None):
    """Filter ledger entries by field usage, mechanism, idea or expression.

    ``field`` is matched as a substring of the recorded expression, which is
    the practical way to answer "has this datafield already been tested?"
    without spending a simulation to find out. Matching is case-insensitive
    and every supplied filter must agree (AND).
    """
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
        if needle_expression and needle_expression not in code:
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
    lines.append("| alpha | sharpe | fitness | turnover | idea | verdict | note |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for entry in sorted(
        entries,
        key=lambda item: (item.get("metrics") or {}).get("sharpe") or float("-inf"),
        reverse=True,
    ):
        metrics = entry.get("metrics") or {}
        note = (entry.get("note") or "").replace("|", "/").replace("\n", " ")
        if len(note) > 80:
            note = note[:77] + "..."
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            entry.get("alpha_id") or "—",
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

    Labels are the only part of a record that is judgement rather than
    measurement, and they are the part that had to be filled in by editing the
    JSONL by hand: 318 of 481 keys carried no ``family``, which made "how many
    independent mechanisms" unanswerable from the ledger. Changing a label
    appends a new record under the same key, so history is preserved and the
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
    if mechanism is not None:
        updated["mechanism"] = mechanism
    if idea is not None:
        updated["idea"] = idea
    if note is not None:
        updated["note"] = note
    updated["recorded_at"] = _now()
    updated["source"] = "relabel"
    return updated


def derive_legs(expression):
    """Leg names parsed from an expression, or an empty list.

    Derived on demand rather than migrated into every record: the ledger predates
    the legs column, and a backfill would have to be re-run whenever the parser
    improves. Parsing is cheap and the grammar is the same one the ablation uses.
    """
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
    return derive_legs(entry.get("expression"))


def leg_bodies(entry):
    """``{leg_name: normalized_body}`` for an expression.

    Leg names are positional — ``a`` means whatever the author put there — so a
    name is not an identity. Comparing bodies is what makes "the same leg" mean
    the same sub-expression.
    """
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


def entries_with_leg(entries, needle):
    """Entries whose expression contains a leg body matching ``needle``."""
    needle_lower = (needle or "").lower()
    out = []
    for entry in entries:
        for body in leg_bodies(entry).values():
            if needle_lower in body.lower():
                out.append(entry)
                break
    return out


def resolve(entries, reference):
    """Find one entry by alpha id, ledger key or expression hash."""
    if not reference:
        return None
    for entry in entries:
        if reference in (entry.get("alpha_id"), entry.get("key"),
                         entry.get("expression_hash")):
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
    target_hash = target.get("expression_hash")
    out = []
    for entry in entries:
        parent = entry.get("parent")
        if parent and parent in (target_key, target_hash, target.get("alpha_id")):
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
