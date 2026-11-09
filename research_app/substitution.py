"""Whole-token substitution: the "change exactly one input" experiment shape.

Two things are load-bearing here and both were learned the hard way. Substitution is
textual and *exact*, because the claim is that only this token changed. And whole
identifiers only: `str.replace("sales", "revenue")` also rewrites `sales_growth`, so a
"single-factor" swap silently changed several inputs while the lint still passed - which
is why an arm that was supposed to isolate one field could not be attributed.

A target that does not appear in the parent raises instead of returning a no-op, because
a batch of identical expressions looks exactly like a batch of variants that all failed.
"""

import itertools
import re

import alpha_registry


def expand(expression, substitutions):
    """One ``(label, expression)`` pair per combination of replacement values.

    Substitution is textual and exact, which is the point: the experiment claims that
    *only* this token changed. A target that does not appear in the parent is an error
    rather than a silent no-op, because a batch of identical expressions looks exactly
    like a batch of variants that all failed.
    """
    targets = list(substitutions)
    patterns = {}
    for target in targets:
        # Whole identifiers only. `str.replace("sales", "revenue")` also rewrites
        # `sales_growth` and `sales_number`, so a "single-factor" swap silently
        # changed several inputs - and both fields exist, so the lint passed.
        pattern = re.compile(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(target))
        if not pattern.search(expression):
            raise ValueError("%r is not in the parent expression; nothing to swap"
                            % target)
        patterns[target] = pattern
    pairs = []
    for combination in itertools.product(*(substitutions[t] for t in targets)):
        text = expression
        for target, value in zip(targets, combination):
            text = patterns[target].sub(value, text)
        pairs.append((",".join("%s=%s" % (t, v) for t, v in zip(targets, combination)),
                      text))
    return pairs


def parent_experiment(alpha_id, registry_path=None):
    """``(settings, pairs)`` from a ledger entry, for the swapped-token shape."""
    entries = alpha_registry.load_entries(registry_path)
    entry = next((item for item in entries if item.get("alpha_id") == alpha_id), None)
    if entry is None:
        raise ValueError("%s is not in the ledger; record it first, or use EXPRESSIONS"
                        % alpha_id)
    expression = entry.get("expression") or ""
    if not expression:
        raise ValueError("%s has no expression in the ledger" % alpha_id)
    settings = {key: value for key, value in (entry.get("settings") or {}).items()
                if key in SETTINGS_KEYS}
    if not SUBSTITUTIONS:
        raise ValueError("PARENT is set but SUBSTITUTIONS is empty")
    # The parent expression is already compiled; running it through the sugar pass
    # again would re-render it and weaken the claim that only the token changed.
    return settings, expand(expression, SUBSTITUTIONS)
