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

VERDICTS = ("candidate", "watch", "rejected", "no_go")

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
                source="manual", extra=None):
    """Build one ledger entry from a normalized alpha fact.

    ``alpha`` is expected to carry ``alpha_id``, ``expression``, ``decay``,
    ``settings``, ``metrics`` and ``checks``. Missing pieces are tolerated, so a
    rejected expression with no alpha id can still be recorded.
    """
    if verdict not in VERDICTS:
        raise ValueError("verdict must be one of %s, got %r"
                         % (", ".join(VERDICTS), verdict))
    expression = alpha.get("expression") or ""
    alpha_id = alpha.get("alpha_id")
    metrics = _pick_metrics(alpha.get("metrics"))
    checks = _normalize_checks(alpha.get("checks"))
    settings = alpha.get("settings") or {}
    if alpha.get("decay") is not None and "decay" not in settings:
        settings = dict(settings)
        settings["decay"] = alpha["decay"]

    entry = {
        "key": alpha_id or expression_hash(expression),
        "alpha_id": alpha_id,
        "idea": idea,
        "mechanism": mechanism,
        "verdict": verdict,
        "note": note,
        "expression": expression,
        "expression_hash": expression_hash(expression),
        "settings": settings,
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


def summarize(entries):
    """Aggregate counts and the headline list, for a quick orientation read."""
    by_verdict = {verdict: 0 for verdict in VERDICTS}
    by_idea = {}
    for entry in entries:
        verdict = entry.get("verdict") or "watch"
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
        idea = entry.get("idea") or "unassigned"
        by_idea[idea] = by_idea.get(idea, 0) + 1

    def sharpe_key(entry):
        value = (entry.get("metrics") or {}).get("sharpe")
        return value if isinstance(value, (int, float)) else float("-inf")

    ranked = sorted(entries, key=sharpe_key, reverse=True)
    return {
        "total": len(entries),
        "by_verdict": by_verdict,
        "by_idea": dict(sorted(by_idea.items())),
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
