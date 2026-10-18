"""Structured field contracts for BRAIN datafields.

The BRAIN catalog reliably tells you a field's ``id``, ``dataset``, ``type``,
``coverage`` and ``delay``. It does *not* tell you the unit, the currency, the
fiscal period, the sign convention or how corporate actions are handled -- and
those are exactly the unknowns that killed Idea 47 (share counts: no split
basis) and Idea 50 (CapEx/OCF/Assets: no confirmed fiscal alignment, no sign
convention).

Hand-auditing those questions in prose for every idea does not scale and does
not survive a compacted context. This module turns the audit into a
machine-readable contract, and is deliberately explicit about the difference
between what the catalog *declared* and what we merely *inferred*:

* ``declared`` -- quoted from the raw catalog record, with the source key.
* ``inferred`` -- derived by a named rule from the field id or the description.
* ``unknown``  -- nobody said, and no rule applies. Never guessed.

:func:`audit_fields` then answers the question a research idea actually asks:
*can I combine these fields for this operation without lying to myself?* The
answer is a ``verdict`` plus the exact list of ``blocking_unknowns``.

The module is pure: it takes raw catalog dicts and never touches the network.
"""

from __future__ import annotations

import re

__all__ = [
    "audit_field",
    "audit_fields",
    "audit_operation",
    "OPERATIONS",
]

# Operations we know how to reason about.
#
# ``ts_difference``  -- same field differenced over time (delta, growth rate)
# ``ratio``          -- one field divided by another
# ``difference``     -- one field minus another
# ``sum``            -- two or more fields added
OPERATIONS = ("ts_difference", "ratio", "difference", "sum")

# Unit families that can be meaningfully divided or subtracted from each other.
_CURRENCY_UNITS = frozenset({"currency_amount", "currency_per_share"})

# Field ids whose unit is a share count need a split basis before any
# time-series arithmetic; the BRAIN catalog does not expose one.
_SHARE_COUNT_UNITS = frozenset({"share_count"})

# Ordered; first match wins. Each entry is (regex on field id, period, rule id).
_PERIOD_RULES = (
    # ``fnd6_newqv1300_cshopq`` / ``fnd6_newa2v1300_oancf``: family token, an
    # optional statement counter and an optional ``vNNNN`` version segment.
    (r"newq\d*(?:v\d+)?_", "quarterly", "id_family_newq"),
    (r"cptnewq|cptmfmq", "quarterly", "id_family_newq"),
    (r"newa\d*(?:v\d+)?_", "annual", "id_family_newa"),
    (r"cptnewa", "annual", "id_family_cptnewa"),
    (r"q$", "quarterly", "id_suffix_q"),
    (r"a$", "annual", "id_suffix_a"),
)

# Ordered; first match wins. Each entry is (regex on description, unit, rule).
_UNIT_RULES = (
    (r"per share", "currency_per_share", "desc_per_share"),
    (r"\bshares?\b", "share_count", "desc_shares"),
    (r"number of\b|\bcount\b", "count", "desc_count"),
    (r"\bratio\b|percent|percentage|\brate\b|\bindex\b|\bscore\b|z-score",
     "dimensionless", "desc_ratio"),
    (r"usd|us\$|\bdollar|amount|\bvalue\b|expense|expenditure|cost|"
     r"sales|revenue|income|cash|liabilit|asset|debt|dividend|payment",
     "currency_amount", "desc_currency"),
)

# Region -> reporting currency. Only the unambiguous ones.
_REGION_CURRENCY = {
    "USA": "USD",
    "EUR": "EUR",
    "JPN": "JPY",
    "KOR": "KRW",
    "TWN": "TWD",
    "HKG": "HKD",
    "IND": "INR",
    "CHN": "CNY",
}

# Fields whose sign convention is known to be ambiguous in the catalog.
_SIGN_AMBIGUOUS = re.compile(r"capx|capex|chg|change|delta|net", re.IGNORECASE)


def _attr(state, value=None, rule=None, confidence=None, note=None):
    """Build one attribute record. ``unknown`` attrs carry no value."""
    record = {"state": state}
    if value is not None:
        record["value"] = value
    if rule is not None:
        record["rule"] = rule
    if confidence is not None:
        record["confidence"] = confidence
    if note is not None:
        record["note"] = note
    return record


def _unknown(note=None):
    record = {"state": "unknown"}
    if note:
        record["note"] = note
    return record


def infer_period(field_id):
    """Infer the reporting period from the field id.

    Only field-id families and suffixes are used, never the free-text
    description, because the description wording is inconsistent. The result is
    always ``inferred`` with ``low`` or ``medium`` confidence -- the catalog
    never declares a period outright.
    """
    if not field_id:
        return _unknown("no field id")
    lowered = field_id.lower()
    for pattern, period, rule in _PERIOD_RULES:
        if re.search(pattern, lowered):
            # A family prefix such as ``newq``/``newa`` is a stronger signal
            # than a bare trailing letter.
            confidence = "medium" if rule.startswith("id_family") else "low"
            return _attr(
                "inferred",
                period,
                rule=rule,
                confidence=confidence,
                note="derived from the field id only; the catalog does not "
                     "declare a reporting period",
            )
    return _unknown(
        "no period marker in the field id; catalog does not declare one"
    )


def infer_unit(description, field_id=""):
    """Infer the unit family from the catalog description."""
    text = (description or "").lower()
    if not text or text in {"no field description", "no description"}:
        return _unknown("catalog record has no usable description")
    for pattern, unit, rule in _UNIT_RULES:
        if re.search(pattern, text):
            return _attr(
                "inferred",
                unit,
                rule=rule,
                confidence="medium",
                note="derived from description keywords, not a declared unit",
            )
    return _unknown("no unit keyword in the description")


def infer_currency(unit_attr, region):
    """Infer the reporting currency for currency-denominated units."""
    value = unit_attr.get("value")
    if value not in _CURRENCY_UNITS:
        return _unknown("field is not currency denominated")
    currency = _REGION_CURRENCY.get((region or "").upper())
    if not currency:
        return _unknown("no reporting currency known for region %r" % (region,))
    return _attr(
        "inferred",
        currency,
        rule="region_reporting_currency",
        confidence="medium",
        note="from the data region, not declared per field",
    )


def infer_sign(field_id, description):
    """Infer the sign convention. Almost always unknown -- on purpose."""
    if _SIGN_AMBIGUOUS.search(field_id or "") or _SIGN_AMBIGUOUS.search(description or ""):
        return _unknown(
            "catalog does not state whether this is stored as a positive "
            "expenditure/balance or as a signed cash-flow change"
        )
    return _unknown("catalog does not declare a sign convention")


def corporate_action_risk(unit_attr):
    """Flag fields that need a split basis before time-series arithmetic."""
    if unit_attr.get("value") in _SHARE_COUNT_UNITS:
        return {
            "flagged": True,
            "reason": "share counts require a split basis; the catalog exposes "
                      "no adjustment rule for this field",
        }
    return {"flagged": False}


def audit_field(raw):
    """Turn one raw catalog record into a field contract.

    ``raw`` is a record as returned by ``GET /data-fields``.
    """
    field_id = raw.get("id") or ""
    description = raw.get("description") or ""
    dataset = raw.get("dataset") or {}
    period = infer_period(field_id)
    unit = infer_unit(description, field_id)
    currency = infer_currency(unit, raw.get("region"))

    attributes = {
        "period": period,
        "unit": unit,
        "currency": currency,
        "sign": infer_sign(field_id, description),
        # The catalog never exposes a fiscal period end or a reporting lag.
        "fiscal_alignment": _unknown(
            "catalog exposes no fiscal period end or reporting lag"
        ),
    }

    blocking = []
    if period["state"] == "unknown":
        blocking.append("period")
    if unit["state"] == "unknown":
        blocking.append("unit")

    return {
        "field_id": field_id,
        "description": description,
        "dataset": dataset.get("id"),
        "category": (raw.get("category") or {}).get("id"),
        "subcategory": (raw.get("subcategory") or {}).get("id"),
        "type": raw.get("type"),
        "region": raw.get("region"),
        "delay": {
            "state": "declared" if raw.get("delay") is not None else "unknown",
            "value": raw.get("delay"),
            "source_key": "delay",
        },
        "coverage": raw.get("coverage"),
        "date_coverage": raw.get("dateCoverage"),
        "attributes": attributes,
        "corporate_action_risk": corporate_action_risk(unit),
        "blocking_unknowns": blocking,
        "verdict": "blocked" if blocking else "ready",
    }


def _units_compatible(contracts):
    """True when every field shares one unit family (dimensionless collapses)."""
    values = set()
    for contract in contracts:
        unit = contract["attributes"]["unit"]
        if unit["state"] == "unknown":
            return False, "unit unknown for %s" % contract["field_id"]
        values.add(unit["value"])
    if len(values) == 1:
        return True, None
    return False, "incompatible units: %s" % ", ".join(sorted(values))


def _periods_compatible(contracts):
    values = {}
    for contract in contracts:
        period = contract["attributes"]["period"]
        if period["state"] == "unknown":
            return False, "period unknown for %s" % contract["field_id"]
        values.setdefault(period["value"], []).append(contract["field_id"])
    if len(values) == 1:
        return True, None
    return False, "mixed periods: %s" % ", ".join(
        "%s(%s)" % (period, ",".join(ids)) for period, ids in sorted(values.items())
    )


def audit_operation(contracts, operation):
    """Judge whether one operation over these fields is safe to run.

    ``ts_difference`` reasons about a single field over time; the other
    operations reason about several fields at once.
    """
    if operation not in OPERATIONS:
        raise ValueError("unknown operation %r; expected one of %s"
                         % (operation, ", ".join(OPERATIONS)))

    findings = []
    blocking = []
    warnings = []

    if operation == "ts_difference":
        # Every field is differenced against itself, so several fields are fine:
        # each one is judged on its own, then the results are aggregated.
        empty = True
        for contract in contracts:
            empty = False
            period = contract["attributes"]["period"]
            if period["state"] == "unknown":
                blocking.append("period")
                findings.append("%s: reporting period unknown, cannot difference "
                                "over time" % contract["field_id"])
            if contract["corporate_action_risk"]["flagged"]:
                blocking.append("corporate_action_adjustment")
                findings.append(
                    "%s: %s" % (contract["field_id"],
                                contract["corporate_action_risk"]["reason"])
                )
        if empty:
            raise ValueError("ts_difference needs at least one field")
    else:
        units_ok, units_reason = _units_compatible(contracts)
        if not units_ok:
            blocking.append("unit")
            findings.append(units_reason)
        periods_ok, periods_reason = _periods_compatible(contracts)
        if not periods_ok:
            blocking.append("period")
            findings.append(periods_reason)
        if units_ok and periods_ok and len(contracts) > 1:
            # Periods agree by label, but the catalog never confirms that the
            # underlying fiscal windows line up.
            warnings.append("fiscal_period_alignment")
            findings.append(
                "period labels agree, but fiscal period ends/lags are not "
                "declared, so cross-field alignment is unverified"
            )
        unsigned = [contract["field_id"] for contract in contracts
                    if contract["attributes"]["sign"]["state"] == "unknown"]
        if unsigned:
            # A sum or difference of fields with undeclared sign is unusable: a
            # single negative stored value inverts the whole signal. For a ratio
            # it stays a warning, with a note to audit as ``difference`` first
            # when the ratio hides a subtraction, e.g. (CapEx - OCF) / Assets.
            if operation in {"difference", "sum"}:
                blocking.append("sign_convention")
            else:
                warnings.append("sign_convention")
            note = "a negative stored value would flip the sign of the result"
            if operation == "ratio":
                note += "; if the ratio hides a subtraction, audit it as " \
                        "`difference` first"
            findings.append("sign convention undeclared for %s; %s"
                            % (", ".join(unsigned), note))

    verdict = "blocked" if blocking else ("needs_review" if warnings else "ready")
    return {
        "operation": operation,
        "fields": [contract["field_id"] for contract in contracts],
        "verdict": verdict,
        "blocking_unknowns": sorted(set(blocking)),
        "warnings": warnings,
        "findings": findings,
    }


def audit_fields(raw_records, operation=None):
    """Audit a set of raw catalog records for one intended operation.

    Returns a document with the per-field contracts plus the operation verdict,
    which is what a research note should quote instead of prose hand-waving.
    """
    contracts = [audit_field(raw) for raw in raw_records]
    document = {
        "field_count": len(contracts),
        "fields": contracts,
    }
    if operation:
        document["operation"] = audit_operation(contracts, operation)
        document["verdict"] = document["operation"]["verdict"]
    else:
        worst = "ready"
        for contract in contracts:
            if contract["verdict"] == "blocked":
                worst = "blocked"
                break
            if contract["verdict"] == "needs_review":
                worst = "needs_review"
        document["verdict"] = worst
    return document