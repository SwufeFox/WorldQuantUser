"""Bounded views of provider incentive metadata; no eligibility or reward formula."""

from __future__ import annotations

import math


MATCH_CHECKS = ("MATCHES_PYRAMID", "MATCHES_THEMES")
SUMMARY_LIMIT = 20
MEMBER_LIMIT = 5


def _multiplier(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def _entity(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    result = {key: value[key] for key in ("id", "name") if key in value}
    if isinstance(result.get("name"), str) and len(result["name"]) > 160:
        result["name"] = result["name"][:160]
        result["name_truncated"] = True
    if "multiplier" in value:
        result["multiplier"] = _multiplier(value["multiplier"])
    omitted = sum(key not in {"id", "name", "multiplier"} for key in value)
    if omitted:
        result["fields_omitted"] = omitted
    return result


def _members(value):
    if not isinstance(value, list):
        return {"state": "unknown", "count": None, "items": None, "omitted": 0}
    return {"state": "reported", "count": len(value),
            "items": [_entity(item) for item in value[:MEMBER_LIMIT]],
            "omitted": max(0, len(value) - MEMBER_LIMIT)}


def catalog_item(record, scope):
    """Keep native category/scope and distinguish missing metadata from explicit zero/empty."""
    result = {key: record[key] for key in ("id", "type", "coverage", "dateCoverage",
              "userCount", "alphaCount", "fieldCount") if key in record}
    for key in ("name", "description"):
        if isinstance(record.get(key), str):
            result[key] = record[key][:240]
    if "dataset" in record:
        result["dataset"] = _entity(record["dataset"])
    multiplier = _multiplier(record.get("pyramidMultiplier"))
    result["pyramid"] = {
        "region": record.get("region", scope.get("region")),
        "delay": record.get("delay", scope.get("delay")),
        "category": _entity(record.get("category")),
        "state": "reported" if multiplier is not None else "unknown",
        "multiplier": multiplier,
    }
    result["themes"] = _members(record.get("themes"))
    return result


def catalog_summary(data, key):
    """Keep query/cache evidence while bounding displayed catalog rows."""
    records = data[key]
    data[key] = [catalog_item(record, data) for record in records[:SUMMARY_LIMIT]]
    data.update(summary=True, displayed_count=len(data[key]),
                rows_omitted=max(0, len(records) - SUMMARY_LIMIT),
                incentive_basis="catalog metadata; per-alpha eligibility and reward arithmetic are not inferred")
    return data


def check_summary(checks, basis="alpha_snapshot"):
    """Preserve provider statuses and effective values without turning them into booleans."""
    checks = checks if isinstance(checks, dict) else {}
    matching = {}
    for name in MATCH_CHECKS:
        check = checks.get(name)
        if not isinstance(check, dict):
            continue
        result = {key: check[key] for key in ("result", "effective", "date") if key in check}
        result["multiplier"] = _multiplier(check.get("multiplier"))
        for key in ("pyramids", "themes", "competitions"):
            if key in check:
                result[key] = _members(check[key])
        matching[name] = result
    return {"state": "reported" if matching else "unknown", "basis": basis, "checks": matching}
