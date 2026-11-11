"""Pruning the local result cache against the criteria we hold *today*.

The PnL cache under ``<research>/pnl`` is the working set for every correlation
question, so it grows with every probe - including the probes that were run under
older, laxer bars. When the bar moves (`fitness` 1.0 -> 1.2, a block gets closed, a
shape rule changes), the cache still holds the rejects from the old regime, and they
keep showing up in scans that are meant to answer "what is still worth a slot".

The bar therefore lives in a file, not in code: ``<research>/prune-criteria.json``.
Editing that file and running the same short command is how a new requirement is
applied, which is the whole point - a criterion that can only change by editing a
module is a criterion that quietly stops being applied.

Nothing here is destructive by default. ``prune`` reports; ``prune --apply`` deletes.
Alphas in the book, anything the ledger marks as submitted, and anything explicitly
listed as protected are never touched, because recomputing a book alpha's PnL costs a
network round trip while re-running a rejected arm costs nothing.
"""

import json
import os
from pathlib import PurePosixPath

import alpha_registry
import research_paths

from . import mechanisms

#: Defaults, copied from the platform's own gates plus the campaign's stricter bar for
#: *new* work. A cache entry failing these is not wrong, it is simply no longer a
#: candidate under the current bar.
DEFAULTS = {
    "min_fitness": 1.0,
    "min_sharpe": 1.25,
    "max_turnover": 0.7,
    "min_returns": 0.0,
    "keep_blocks": None,
    "drop_blocks": [],
    "max_operators": None,
    "require_positive_halves": False,
    "protect": [],
    "note": ("Defaults mirror the platform gates (fitness 1.0 / sharpe 1.25 / turnover "
             "0.7). Raise them in prune-criteria.json when the campaign bar moves."),
}


def criteria_path(args=None):
    explicit = getattr(args, "criteria", None)
    if explicit:
        return explicit
    return os.path.join(research_paths.ensure_dir(), "prune-criteria.json")


def load_criteria(path):
    """File over defaults; ``--set`` overrides on top of both."""
    criteria = dict(DEFAULTS)
    source = "defaults"
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                criteria.update({k: v for k, v in payload.items() if k in DEFAULTS})
                source = path
        except (OSError, ValueError) as exc:
            # Refuse rather than fall back. A typo in the criteria file would silently
            # substitute the platform defaults and prune against the wrong bar - the
            # dangerous direction, because `--apply` then deletes by a rule nobody wrote.
            raise ValueError(
                "prune criteria %s is unreadable (%s); fix it or pass --criteria" % (path, exc))
    return criteria, source


def _latest_entries(registry_path=None):
    latest = {}
    for entry in alpha_registry.load_entries(registry_path):
        key = entry.get("alpha_id") or entry.get("key")
        if key:
            latest[key] = entry
    return latest


def _reasons(entry, criteria, known_fields=None):
    """Every criterion this entry fails, in a fixed order for a stable table."""
    metrics = entry.get("metrics") or {}
    reasons = []
    fitness = metrics.get("fitness")
    sharpe = metrics.get("sharpe")
    turnover = metrics.get("turnover")
    returns = metrics.get("returns")
    if criteria.get("min_fitness") is not None:
        if fitness is None or fitness < criteria["min_fitness"]:
            reasons.append("fitness<%.2f" % criteria["min_fitness"])
    if criteria.get("min_sharpe") is not None:
        if sharpe is None or sharpe < criteria["min_sharpe"]:
            reasons.append("sharpe<%.2f" % criteria["min_sharpe"])
    if criteria.get("max_turnover") is not None:
        if turnover is not None and turnover > criteria["max_turnover"]:
            reasons.append("turnover>%.2f" % criteria["max_turnover"])
    if criteria.get("min_returns"):
        if returns is None or returns < criteria["min_returns"]:
            reasons.append("returns<%.3f" % criteria["min_returns"])

    expression = entry.get("expression") or ""
    block = mechanisms.block_of(expression) if expression else None
    keep_blocks = criteria.get("keep_blocks")
    if keep_blocks is not None and block not in keep_blocks:
        reasons.append("block=%s" % block)
    if block in (criteria.get("drop_blocks") or []):
        reasons.append("block=%s dropped" % block)

    max_operators = criteria.get("max_operators")
    if max_operators is not None:
        operators = expression.count("(")
        if operators > max_operators:
            reasons.append("operators>%d" % max_operators)

    if criteria.get("require_positive_halves"):
        halves = ((entry.get("metrics") or {}).get("sharpe_h1"),
                  (entry.get("metrics") or {}).get("sharpe_h2"))
        if None not in halves and not (halves[0] > 0 and halves[1] > 0):
            reasons.append("halves not both positive")
    return reasons


def _is_protected(entry, criteria):
    status = str(entry.get("status") or "").upper()
    if status in ("ACTIVE", "SUBMITTED"):
        return "in the book"
    if entry.get("alpha_id") in (criteria.get("protect") or []):
        return "listed in protect"
    if str(entry.get("verdict") or "").lower() in ("candidate",):
        return "verdict candidate"
    return None


def cmd_prune(args, _sink):
    """Report - or with ``--apply`` delete - cached results that fail the current bar."""
    path = criteria_path(args)
    criteria, source = load_criteria(path)
    for pair in getattr(args, "set", None) or []:
        if "=" not in pair:
            raise ValueError("--set expects key=value, got %r" % pair)
        key, value = pair.split("=", 1)
        key = key.strip()
        if key not in DEFAULTS:
            raise ValueError("unknown criterion %r (known: %s)"
                             % (key, ", ".join(sorted(DEFAULTS))))
        try:
            criteria[key] = json.loads(value)
        except ValueError:
            criteria[key] = value

    if getattr(args, "show_criteria", False):
        return {"criteria_path": path, "source": source, "criteria": criteria}

    pnl_dir = os.path.join(research_paths.ensure_dir(), "pnl")
    latest = _latest_entries(getattr(args, "registry", None))

    keep, drop, reasons_by_reason = [], [], {}
    freed_bytes = 0
    for name in sorted(os.listdir(pnl_dir)) if os.path.isdir(pnl_dir) else []:
        if not name.endswith(".json"):
            continue
        alpha_id = name[:-5]
        entry = latest.get(alpha_id)
        if entry is None:
            # A PnL file with no ledger row is not a reject, it is an orphan; report it
            # separately rather than deleting evidence we cannot even classify.
            drop.append({"alpha_id": alpha_id, "reasons": ["no ledger row"]})
            continue
        protected = _is_protected(entry, criteria)
        if protected:
            keep.append({"alpha_id": alpha_id, "why": protected})
            continue
        reasons = _reasons(entry, criteria)
        if not reasons:
            keep.append({"alpha_id": alpha_id, "why": "passes"})
            continue
        drop.append({"alpha_id": alpha_id, "reasons": reasons,
                     "fitness": (entry.get("metrics") or {}).get("fitness"),
                     "sharpe": (entry.get("metrics") or {}).get("sharpe")})
        for reason in reasons:
            reasons_by_reason[reason] = reasons_by_reason.get(reason, 0) + 1
        freed_bytes += os.path.getsize(os.path.join(pnl_dir, name))

    removed = 0
    if getattr(args, "apply", False):
        for row in drop:
            target = os.path.join(pnl_dir, row["alpha_id"] + ".json")
            if os.path.exists(target):
                os.unlink(target)
                removed += 1

    return {
        "criteria_path": path,
        "criteria_source": source,
        "criteria": criteria,
        "applied": bool(getattr(args, "apply", False)),
        "cached": len(keep) + len(drop),
        "kept": len(keep),
        "dropped": len(drop),
        "removed": removed,
        "freed_mb": round(freed_bytes / (1024.0 * 1024.0), 2),
        "by_reason": reasons_by_reason,
        "dropped_sample": drop[: args.limit],
    }