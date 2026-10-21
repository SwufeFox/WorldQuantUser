#!/usr/bin/env python3
"""Ranked text search over catalog records, with no built-in vocabulary.

Substring matching is the wrong primitive for this catalog. ``--search cfps
count`` should find the field that counts CFPS estimates even though those two
words never appear next to each other in the id; ``--search perc`` should find
``percent``; and a one-letter typo should not return nothing. So this ranks
instead of filtering: every query term must match somewhere, and the score says
how well.

There is deliberately no synonym table, no abbreviation map, and no stopword
list here. Each of those would encode one snapshot of someone's naming and then
rot silently. The matching rules below are mechanical:

* an exact token match beats a prefix match, which beats a substring match;
* a token match in the id beats the same match in the description;
* ``-term`` excludes instead of requiring;
* ``--fuzzy`` allows one edit for longer tokens, which covers typos.

Callers that know the domain can still narrow results themselves; the tool just
refuses to pretend it knows which words mean the same thing.
"""

from __future__ import annotations

import re

__all__ = ["parse_query", "score_record", "search_records", "absent_terms",
           "edit_distance"]

_SPLIT = re.compile(r"[^a-z0-9]+")

# Weights are relative, not calibrated. The ordering they encode is the part
# that matters: exact > prefix > substring, and id > description.
_W_EXACT_ID = 6
_W_PREFIX_ID = 4
_W_REVERSE_PREFIX = 3
_W_SUBSTRING_ID = 2
_W_EXACT_DESC = 3
_W_PREFIX_DESC = 2
_W_SUBSTRING_DESC = 1
_W_FUZZY_ID = 2
_W_FUZZY_DESC = 1
_MIN_FUZZY_LENGTH = 4
# Reverse prefix means "the query spells out what the id abbreviates", as in
# cashflow vs cash. Below four characters the id token is usually a fragment
# ("per" in "percent"), and letting it match invents results that look like hits.
_MIN_REVERSE_PREFIX_TOKEN = 4


def _tokens(text):
    if not text:
        return []
    return [token for token in _SPLIT.split(str(text).lower()) if token]


def parse_query(text):
    """Split a query into ``(required, excluded)`` token lists.

    ``-term`` means "must not appear". Everything else must appear. Multi-word
    queries are AND by construction; callers that want OR can lower the
    requirement themselves.
    """
    required, excluded = [], []
    for raw in str(text or "").split():
        negated = raw.startswith("-") and len(raw) > 1
        term = raw[1:] if negated else raw
        for token in _tokens(term):
            (excluded if negated else required).append(token)
    return required, excluded


def edit_distance(left, right):
    """Levenshtein distance, iteratively, single row."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, lch in enumerate(left, start=1):
        current = [i]
        for j, rch in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (lch != rch)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def _term_score(term, id_tokens, desc_tokens, fuzzy):
    """Best score for one term against one record, 0 meaning no match."""
    best = 0
    for token in id_tokens:
        if token == term:
            best = max(best, _W_EXACT_ID)
        elif token.startswith(term):
            best = max(best, _W_PREFIX_ID)
        elif term.startswith(token) and len(token) >= _MIN_REVERSE_PREFIX_TOKEN:
            # Query spelled out what the id abbreviates ("cashflow" vs "cash").
            best = max(best, _W_REVERSE_PREFIX)
        elif term in token:
            best = max(best, _W_SUBSTRING_ID)
        elif fuzzy and len(term) >= _MIN_FUZZY_LENGTH and len(token) >= _MIN_FUZZY_LENGTH:
            if edit_distance(term, token) <= 1:
                best = max(best, _W_FUZZY_ID)
    if best >= _W_EXACT_ID:
        return best
    for token in desc_tokens:
        if token == term:
            best = max(best, _W_EXACT_DESC)
        elif token.startswith(term):
            best = max(best, _W_PREFIX_DESC)
        elif term in token:
            best = max(best, _W_SUBSTRING_DESC)
        elif fuzzy and len(term) >= _MIN_FUZZY_LENGTH and len(token) >= _MIN_FUZZY_LENGTH:
            if edit_distance(term, token) <= 1:
                best = max(best, _W_FUZZY_DESC)
    return best


def _is_excluded(record, excluded):
    """True when an excluded term appears in the id or description."""
    if not excluded:
        return False
    field_id = record.get("id") or record.get("name") or ""
    description = str(record.get("description") or "").lower()
    id_tokens = _tokens(field_id)
    for term in excluded:
        if any(term in token for token in id_tokens) or term in description:
            return True
    return False


def score_record(record, required, excluded, fuzzy=False, match_all=True):
    """Return ``(score, matched_terms)`` for one record; score 0 means filtered out.

    ``match_all`` is the AND/OR switch: with it, every required term must match
    somewhere; without it, one match is enough and the score is the sum of what
    did match.
    """
    field_id = record.get("id") or record.get("name") or ""
    description = record.get("description") or ""
    id_tokens = _tokens(field_id)
    desc_tokens = _tokens(description)

    if _is_excluded(record, excluded):
        return 0, []

    if not required:
        return 0, []

    total = 0
    matched = []
    for term in required:
        score = _term_score(term, id_tokens, desc_tokens, fuzzy)
        if score:
            total += score
            matched.append(term)
        elif match_all:
            return 0, []
    return total, matched


def search_records(records, query, limit=None, fuzzy=False, match_all=True,
                   rank_by_coverage=True):
    """Rank ``records`` against ``query`` and return annotated copies.

    Each hit carries ``score`` and ``matched_terms`` so a caller can see *why*
    something ranked where it did instead of trusting an opaque order. An empty
    query returns the input order untouched, which keeps ``--search`` optional.
    """
    required, excluded = parse_query(query)
    if not required and not excluded:
        hits = [dict(record) for record in records or []]
        for hit in hits:
            hit["score"] = 0
            hit["matched_terms"] = []
        return hits[:limit] if limit else hits

    hits = []
    for record in records or []:
        if _is_excluded(record, excluded):
            continue
        if required:
            score, matched = score_record(
                record, required, (), fuzzy, match_all)
            if not score:
                continue
        else:
            # Exclusion-only query: a filter, not a ranking. Everything that
            # survived is kept and nothing is ordered by relevance.
            score, matched = 0, []
        item = dict(record)
        item["score"] = score
        item["matched_terms"] = matched
        hits.append(item)

    def sort_key(item):
        coverage = item.get("coverage")
        try:
            coverage = float(coverage)
        except (TypeError, ValueError):
            coverage = -1.0
        if rank_by_coverage:
            return (-item["score"], -coverage, str(item.get("id") or ""))
        return (-item["score"], str(item.get("id") or ""))

    hits.sort(key=sort_key)
    return hits[:limit] if limit else hits


def absent_terms(records, query, fuzzy=False):
    """Required terms that match nothing anywhere in ``records``.

    Exists so an empty result can explain itself. The usual cause is a word the
    catalog does not use -- asking for "count" when every field spells it
    "number" -- and the honest answer is to name the term, not to quietly treat
    it as a synonym and return something that merely looks close.
    """
    required, _ = parse_query(query)
    absent = []
    for term in required:
        found = False
        for record in records or []:
            field_id = record.get("id") or record.get("name") or ""
            if _term_score(term, _tokens(field_id), _tokens(record.get("description")),
                           fuzzy):
                found = True
                break
        if not found:
            absent.append(term)
    return absent
