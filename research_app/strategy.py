"""Explicit research strategy: state, candidates, selection and updates.

The loop keeps the parts a reviewer needs to reproduce a decision - the candidate
identity, the policy that chose it, the evaluator context that made its score
comparable, and the state the update produced. Nothing here talks to BRAIN.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import semantic_identity

STATE_FILENAME = "state.json"
STRATEGY_DIRNAME = "strategy"
SCORE_KEYS = ("fitness", "sharpe")


def strategy_dir(root):
    """Return the directory that keeps strategy state for one research root."""
    return Path(root) / STRATEGY_DIRNAME


def context_of(entry):
    """Return the comparison context a recorded evaluation belongs to."""
    settings = entry.get("settings") or {}
    return {
        "region": settings.get("region"),
        "universe": settings.get("universe"),
        "delay": settings.get("delay"),
        "train_start": settings.get("startDate"),
        "train_end": settings.get("endDate"),
        "checks_version": entry.get("checks_version") or "unknown",
        "cost_bps": entry.get("cost_bps", 0),
    }


def identity_of(expression, *, neutralization="INDUSTRY"):
    """Semantic identity of a candidate expression; blank expressions stay unique."""
    if not expression:
        return None
    try:
        return semantic_identity.fingerprint(expression, neutralization=neutralization)
    except Exception:
        return None


def score_of(entry):
    """Comparable score of one evaluation: fitness first, then Sharpe."""
    metrics = entry.get("metrics") or {}
    for key in SCORE_KEYS:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def candidate_of(entry):
    """Project one ledger entry into the candidate shape the selectors consume."""
    settings = entry.get("settings") or {}
    return {
        "alpha_id": entry.get("alpha_id"),
        "expr": entry.get("expression"),
        "identity": entry.get("identity") or identity_of(
            entry.get("expression"), neutralization=settings.get("neutralization")),
        "score": score_of(entry),
        "context": context_of(entry),
    }


def _ranked(candidates):
    return sorted((c for c in candidates if c.get("score") is not None),
                  key=lambda c: c["score"], reverse=True)


def select_top_k(candidates, k):
    """Plain score ranking."""
    return _ranked(candidates)[:k]


def select_unique(candidates, k, *, seen=()):
    """Score ranking that spends one slot per semantic identity."""
    seen = set(seen or ())
    picked, used = [], set(seen)
    for candidate in _ranked(candidates):
        identity = candidate.get("identity")
        if identity and identity in used:
            continue
        picked.append(candidate)
        if identity:
            used.add(identity)
        if len(picked) == k:
            break
    return picked


def select_with_exploration(candidates, k, *, state=None, exploration_every=0):
    """Unique ranking that forces one fresh identity every Nth slot."""
    state = state or {}
    seen = set(state.get("seen_identities") or ())
    if not exploration_every or exploration_every <= 0:
        return select_unique(candidates, k, seen=seen)
    picked = select_unique(candidates, k, seen=seen)
    if len(picked) < k:
        return picked
    fresh = [c for c in _ranked(candidates)
             if c.get("identity") and c["identity"] not in seen
             and c["identity"] not in {p.get("identity") for p in picked}]
    for offset, candidate in enumerate(fresh):
        slot = exploration_every - 1
        while slot < k:
            if slot < len(picked):
                picked[slot] = candidate
            else:
                break
            slot += exploration_every
            break
        if offset + 1 >= max(1, k // exploration_every):
            break
    return picked[:k]


SELECTORS = {
    "top-k": select_top_k,
    "unique": select_unique,
}


def load_policy(reference):
    """Resolve `pkg.module:function` into a callable selector."""
    if reference in SELECTORS:
        return SELECTORS[reference]
    module_name, _, function_name = reference.partition(":")
    if not module_name or not function_name:
        raise ValueError("policy must be `module:function` or a built-in selector name")
    module = __import__(module_name, fromlist=[function_name])
    return getattr(module, function_name)


def load_state(root):
    """Read the persisted strategy state, or a fresh one."""
    path = strategy_dir(root) / STATE_FILENAME
    if not path.is_file():
        return {"counters": {"arms": 0, "pass": 0, "explore": 0},
                "parents": [], "seen_identities": [], "history": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(root, state):
    """Write the strategy state atomically."""
    directory = strategy_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / STATE_FILENAME
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    temp.replace(path)
    return path


def update(state, evaluations):
    """Fold evaluations into a new state; the only place counters move."""
    state = json.loads(json.dumps(state))
    counters = state.setdefault("counters", {"arms": 0, "pass": 0, "explore": 0})
    seen = list(state.get("seen_identities") or [])
    parents = list(state.get("parents") or [])
    for item in evaluations:
        counters["arms"] += 1
        if item.get("verdict") == "pass":
            counters["pass"] += 1
        identity = item.get("identity")
        if identity and identity not in seen:
            seen.append(identity)
        alpha_id = item.get("alpha_id")
        if alpha_id and item.get("verdict") == "pass" and alpha_id not in parents:
            parents.append(alpha_id)
    state["seen_identities"] = seen
    state["parents"] = parents
    return state


def replay(entries, k, *, trials=200, seed=0):
    """Compare selection policies on already measured entries. Costs no simulation."""
    candidates = [candidate_of(entry) for entry in entries]
    candidates = [c for c in candidates if c.get("score") is not None]
    if len(candidates) < k:
        return {"available": len(candidates), "k": k, "insufficient": True}
    rng = random.Random(seed)
    report = {name: {"score": 0.0, "duplicates": 0, "identities": 0}
              for name in ("top-k", "unique", "random")}
    for _ in range(trials):
        sample = rng.sample(candidates, min(len(candidates), max(k * 4, k)))
        for name, picker in (("top-k", lambda: select_top_k(sample, k)),
                             ("unique", lambda: select_unique(sample, k)),
                             ("random", lambda: rng.sample(sample, k))):
            picked = picker()
            if not picked:
                continue
            report[name]["score"] += sum(c["score"] for c in picked) / len(picked)
            report[name]["duplicates"] += len(picked) - len({c.get("identity") for c in picked if c.get("identity")})
            report[name]["identities"] += len({c.get("identity") for c in picked if c.get("identity")})
    for name, bucket in report.items():
        bucket["mean_score"] = round(bucket["score"] / trials, 4)
        bucket["duplicate_rate"] = round(bucket["duplicates"] / (trials * k), 4)
        bucket["mean_identities"] = round(bucket["identities"] / trials, 2)
        bucket.pop("score", None)
        bucket.pop("duplicates", None)
        bucket.pop("identities", None)
    return {"available": len(candidates), "k": k, "trials": trials, "policies": report}


def cmd_strategy_replay(args, _sink):
    """Replay selection policies over recorded entries without running simulations."""
    path = Path(args.ledger)
    entries = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
    return replay(entries, int(args.k), trials=int(args.trials), seed=int(args.seed))


def cmd_strategy_select(args, _sink):
    """Select a batch from a candidate file using one explicit policy."""
    payload = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    rows = payload.get("candidates") if isinstance(payload, dict) else payload
    candidates = [candidate_of(row) for row in rows or []]
    state = load_state(args.research_dir) if getattr(args, "research_dir", None) else {}
    policy = load_policy(getattr(args, "policy", "unique"))
    picked = policy(candidates, int(args.k))
    return {"policy": getattr(args, "policy", "unique"), "k": int(args.k),
            "picked": picked, "counts": {"candidates": len(candidates), "picked": len(picked)}}
