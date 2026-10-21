#!/usr/bin/env python3
"""Live field-name verification, plus generic token grouping for comparison.

Two problems this addresses, both of which cost real simulations before it
existed.

**Wrong names.** A field identifier typed from memory — say
``min_useful_life_buildings_land`` when the catalog only has
``min_useful_life_furniture_fixtures`` — passes every offline check and then
fails at the simulator with ``Attempted to use unknown variable``. The operator
catalog cannot catch that because such a name is not an operator. Verifying
identifiers against the live field list before submitting turns that round trip
into a local error.

**Duplicate concepts.** The catalog names the same quantity several ways, with
materially different coverage, and nothing in the ids says so. Grouping ids by
shared tokens puts the variants next to each other so the choice is deliberate.

Design note: the grouping here carries **no knowledge of BRAIN's naming**.
There is deliberately no baked-in prefix table, no acronym map, no stopword
list. Vocabulary normalization is passed in at call time (``ignore_words``)
because a curated table inside the tool would encode one snapshot of the
catalog, rot silently as families are renamed, and quietly mis-group the things
that matter. The tool stays mechanical; the judgment stays with the caller.
"""

from __future__ import annotations

import re

__all__ = [
    "tokenize",
    "concept_key",
    "group_by_concept",
    "duplicate_concepts",
    "field_ids",
    "fetch_all_fields",
]

_SPLIT = re.compile(r"[^a-z0-9]+")


def tokenize(field_id: str):
    """Split a field id into lower-case word tokens.

    Purely mechanical: every alphanumeric run is a token. Digits are kept,
    because whether ``_2`` is a vintage marker or a real index is a judgment the
    caller makes with ``ignore_words`` rather than one the tool guesses.
    """
    if not field_id:
        return []
    return [token for token in _SPLIT.split(str(field_id).lower()) if token]


def concept_key(field_id: str, ignore_words=(), drop_numeric=False, tail_tokens=0):
    """Return a comparison key: the field's tokens minus ignored words, sorted.

    Sorting makes the key order-insensitive, so two families that name the same
    quantity in a different word order still collapse together once the shared
    boilerplate tokens are ignored.

    ``drop_numeric`` removes purely numeric tokens. That is a mechanical rule,
    not catalog knowledge: it exists because outlets commonly append a vintage
    number to an otherwise identical id, and whether a given number is a vintage
    or part of the name is the caller's call, so the flag defaults off.

    ``tail_tokens`` keeps only the trailing N tokens, which is the mechanical
    version of "leading words are family boilerplate, trailing words are the
    concept". It is a positional rule rather than a vocabulary: no list of
    family names is needed, and it survives families being renamed.
    """
    ignored = {word.lower() for word in (ignore_words or ())}
    tokens = [token for token in tokenize(field_id) if token not in ignored]
    if drop_numeric:
        tokens = [token for token in tokens if not token.isdigit()]
    if tail_tokens and len(tokens) > tail_tokens:
        tokens = tokens[-tail_tokens:]
    if not tokens:
        return ""
    return " ".join(sorted(tokens))


def _coverage(record) -> float:
    value = record.get("coverage")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def group_by_concept(fields, ignore_words=(), drop_numeric=False, tail_tokens=0):
    """Return ``{concept_key: [field, ...]}``, best-covered field first."""
    groups = {}
    for record in fields or []:
        field_id = record.get("id")
        if not field_id:
            continue
        key = concept_key(field_id, ignore_words, drop_numeric, tail_tokens)
        groups.setdefault(key, []).append(record)
    for members in groups.values():
        members.sort(key=lambda r: (-_coverage(r), r.get("alphaCount") or 0, r.get("id") or ""))
    return groups


def duplicate_concepts(fields, ignore_words=(), min_members=2, min_spread=0.0,
                       field_type=None, drop_numeric=False, tail_tokens=0):
    """Groups served by more than one field, widest coverage spread first.

    ``coverage_spread`` is best member minus worst. A wide spread is the
    dangerous case: choosing the wrong variant silently discards that share of
    the universe.
    """
    groups = group_by_concept(fields, ignore_words, drop_numeric, tail_tokens)
    out = []
    for key, members in groups.items():
        if field_type:
            members = [m for m in members if m.get("type") == field_type]
        if len(members) < min_members:
            continue
        spread = round(_coverage(members[0]) - _coverage(members[-1]), 4)
        if spread < min_spread:
            continue
        out.append(
            {
                "concept": key,
                "members": len(members),
                "best": members[0]["id"],
                "best_coverage": _coverage(members[0]),
                "worst": members[-1]["id"],
                "worst_coverage": _coverage(members[-1]),
                "coverage_spread": spread,
                "fields": [
                    {
                        "id": m["id"],
                        "coverage": _coverage(m),
                        "alphaCount": m.get("alphaCount"),
                        "dataset": m.get("dataset"),
                        "type": m.get("type"),
                    }
                    for m in members
                ],
            }
        )
    out.sort(key=lambda item: -item["coverage_spread"])
    return out


def field_ids(fields):
    """Just the ids, as a set — the shape ``lint_all(known_fields=...)`` wants."""
    return {record["id"] for record in fields or [] if record.get("id")}


def fetch_all_fields(session, region="USA", universe="TOP3000", delay=1,
                     instrument_type="EQUITY", dataset_ids=None):
    """Fetch every datafield for a scope, across all of its datasets.

    This is the authoritative list for name verification. It rides the existing
    catalog cache, so repeated calls inside one session are cheap.
    """
    import machine_lib as ml

    if dataset_ids is None:
        records = ml.get_datasets(
            session, instrument_type=instrument_type, region=region,
            universe=universe, delay=delay,
        ) or []
        dataset_ids = [r.get("id") for r in records if r.get("id")]

    fields = []
    for dataset_id in dataset_ids:
        for record in ml.get_datafields(
            session, instrument_type=instrument_type, region=region,
            delay=delay, universe=universe, dataset_id=dataset_id,
        ) or []:
            item = dict(record)
            item.setdefault("dataset", dataset_id)
            fields.append(item)
    return fields
