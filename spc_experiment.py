"""Inspect captured SPC predictions and compare them offline."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import mean


_FIELDS = {
    "company_name", "ticker", "isin", "mic", "confidence_score",
    "investment_horizon", "return_prediction",
}
_ISIN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")
_MIC = re.compile(r"[A-Z0-9]{4}")
_UNPAIRED_SURROGATE = re.compile(r"[\ud800-\udfff]")
_SEGMENTS = {
    "region": ("Americas", "EMEA", "APAC"),
    "sector": ("Information Technology", "Health Care", "Financials", "Consumer Discretionary",
               "Communication Services", "Industrials", "Consumer Staples", "Energy", "Materials",
               "Utilities", "Real Estate"),
}
_SEGMENT_NAMES = {field: {name.casefold(): name for name in names}
                  for field, names in _SEGMENTS.items()}


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _nonfinite(_value):
    raise ValueError("non-finite JSON number")


def _json_float(value):
    number = float(value)
    if not math.isfinite(number):
        _nonfinite(value)
    return number


def _loads(raw):
    return json.loads(raw, object_pairs_hook=_object, parse_constant=_nonfinite, parse_float=_json_float)


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and not _UNPAIRED_SURROGATE.search(value)


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _valid_isin(value):
    if not isinstance(value, str) or not _ISIN.fullmatch(value):
        return False
    # ISO 6166 expands letters to decimal digits before applying Luhn.
    digits = "".join(str(ord(char) - 55) if char.isalpha() else char for char in value)
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char) * (2 if index % 2 else 1)
        total += digit if digit < 10 else digit - 9
    return total % 10 == 0


def _instrument_errors(row):
    errors = []
    if not _valid_isin(row.get("isin")):
        errors.append("isin must be an uppercase ISIN with a valid check digit")
    mic = row.get("mic")
    if not isinstance(mic, str) or not _MIC.fullmatch(mic):
        errors.append("mic must have four uppercase ASCII letters or digits")
    return errors


def _prediction_errors(row):
    missing = sorted(_FIELDS - row.keys())
    extra = sorted(row.keys() - _FIELDS)
    if missing or extra:
        errors = []
        if missing:
            errors.append("missing fields: " + ", ".join(missing))
        if extra:
            errors.append("unexpected fields: " + ", ".join(ascii(field) for field in extra))
        return errors
    errors = _instrument_errors(row)
    for field in ("company_name", "ticker"):
        if not _text(row[field]):
            errors.append(f"{field} must be a non-empty UTF-8 string")
    score = row["confidence_score"]
    if not _finite(score) or not -1 <= score <= 1:
        errors.append("confidence_score must be a finite number from -1 to 1")
    horizon = row["investment_horizon"]
    if type(horizon) is not int or not _finite(horizon) or horizon <= 0:
        errors.append("investment_horizon must be a positive finite integer")
    prediction = row["return_prediction"]
    if not _finite(prediction) or prediction <= 0:
        errors.append("return_prediction must be a finite positive number")
    return errors


def _key(row):
    return row["isin"], row["mic"]


def parse_predictions(raw: str):
    """Return structurally valid rows and errors. Any error rejects the output."""
    if not isinstance(raw, str):
        return [], ["prediction output must be a JSON string"]
    try:
        payload = _loads(raw)
    except (ValueError, RecursionError) as error:
        return [], [f"invalid prediction JSON: {error}"]
    if not isinstance(payload, list) or not payload:
        return [], ["prediction output must be a non-empty JSON array"]

    rows, errors, seen = [], [], set()
    for index, row in enumerate(payload, start=1):
        issues = _prediction_errors(row) if isinstance(row, dict) else ["expected an object"]
        if not issues and _key(row) in seen:
            issues.append("duplicate ISIN and MIC pair")
        if issues:
            errors.extend(f"row {index}: {issue}" for issue in issues)
        else:
            seen.add(_key(row))
            rows.append(row)
    return rows, errors


def _read(path):
    path = Path(path)
    source = {"path": str(path), "sha256": None}
    try:
        raw = path.read_bytes()
    except (OSError, ValueError) as error:
        return source, None, [f"cannot read file: {error}"]
    source["sha256"] = sha256(raw).hexdigest()
    try:
        return source, raw.decode("utf-8-sig"), []
    except UnicodeDecodeError:
        return source, None, ["file must be UTF-8"]


def _reference(raw):
    try:
        payload = _loads(raw)
    except (ValueError, RecursionError) as error:
        return {}, [f"invalid reference JSON: {error}"]
    if not isinstance(payload, list) or not payload:
        return {}, ["reference must be a non-empty JSON array"]

    records, errors = {}, []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            errors.append(f"reference row {index}: expected an object")
            continue
        issues = _instrument_errors(row)
        segments = {}
        for field, names in _SEGMENT_NAMES.items():
            value = row.get(field)
            if value is None:
                continue
            if _text(value) and (name := names.get(value.strip().casefold())):
                segments[field] = name
            else:
                issues.append(f"{field} must match an SPC segment name or be null")
        if not issues and _key(row) in records:
            issues.append("duplicate ISIN and MIC pair")
        if issues:
            errors.extend(f"reference row {index}: {issue}" for issue in issues)
        else:
            records[_key(row)] = segments
    return records, errors


def _coverage(rows, reference):
    if reference is None:
        return None
    positions = [row for row in rows if row["confidence_score"] != 0]
    matched = [reference[_key(row)] for row in positions if _key(row) in reference]
    denominator = len(positions)
    result = {"denominator": denominator, "matched": len(matched), "missing": denominator - len(matched)}
    for field in ("region", "sector"):
        counts = Counter(row[field] for row in matched if field in row)
        result[field] = {
            "unknown": denominator - sum(counts.values()),
            "counts": dict(sorted(counts.items())),
            "qualifying": sorted(name for name, count in counts.items() if count * 10 >= denominator),
        }
    return result


def _side(score):
    if score > 0:
        return "long"
    if score < 0:
        return "short"
    return "neutral"


def _summary(rows, errors, reference):
    result = {"ok": not errors, "errors": errors, "count": None, "unique_isins": None,
              "sides": None, "coverage": None}
    if errors:
        return result
    counts = Counter(_side(row["confidence_score"]) for row in rows)
    return {**result, "count": len(rows), "unique_isins": len({row["isin"] for row in rows}),
            "sides": {side: counts[side] for side in ("long", "short", "neutral")},
            "coverage": _coverage(rows, reference)}


def _compare(sample_rows, run_rows, run_path):
    sample = {_key(row): row for row in sample_rows}
    run = {_key(row): row for row in run_rows}
    shared = sample.keys() & run.keys()
    union = len(sample.keys() | run.keys())
    side_changes = sum(_side(sample[key]["confidence_score"]) != _side(run[key]["confidence_score"])
                       for key in shared)
    movement = {}
    for field, name in (("confidence_score", "confidence"), ("investment_horizon", "horizon"),
                        ("return_prediction", "return")):
        movement[f"mean_abs_{name}_change"] = (
            mean(abs(sample[key][field] - run[key][field]) for key in shared) if shared else None
        )
    return {"run_path": run_path, "shared": len(shared), "union": union,
            "jaccard": len(shared) / union, "side_changes": side_changes if shared else None, **movement}


def _spec(raw):
    try:
        spec = _loads(raw)
    except (ValueError, RecursionError) as error:
        return {}, [f"invalid spec JSON: {error}"]
    if not isinstance(spec, dict):
        return {}, ["experiment spec must be a JSON object"]
    errors = [f"{field} must be a non-empty UTF-8 string"
              for field in ("name", "prompt", "modelFamily", "updateFrequency") if not _text(spec.get(field))]
    return spec, errors


def experiment(spec_path, run_paths, reference_path=None):
    """Compare trial outputs with the sample in a frozen, caller-supplied spec."""
    spec_source, spec_raw, errors = _read(spec_path)
    spec = {}
    if not errors:
        spec, errors = _spec(spec_raw)
    spec_source.update({field: spec[field] for field in ("name", "modelFamily", "updateFrequency", "as_of")
                        if _text(spec.get(field))})
    if _finite(spec.get("weight")):
        spec_source["weight"] = spec["weight"]
    if _text(spec.get("prompt")):
        spec_source["prompt_sha256"] = sha256(spec["prompt"].encode("utf-8")).hexdigest()

    reference, reference_report = None, None
    if reference_path is not None:
        reference_source, reference_raw, reference_errors = _read(reference_path)
        if not reference_errors:
            reference, reference_errors = _reference(reference_raw)
        if reference_errors:
            reference = None
        reference_report = {**reference_source, "ok": not reference_errors, "errors": reference_errors,
                            "count": len(reference) if reference is not None else None}

    sample_rows, sample_errors = parse_predictions(spec.get("sampleOutput"))
    if isinstance(spec.get("sampleOutput"), str) and not _UNPAIRED_SURROGATE.search(spec["sampleOutput"]):
        sample_hash = sha256(spec["sampleOutput"].encode("utf-8")).hexdigest()
    else:
        sample_hash = None
    sample = {"sha256": sample_hash, **_summary(sample_rows, sample_errors, reference)}
    runs, comparisons = [], []
    for path in run_paths:
        source, raw, read_errors = _read(path)
        rows, run_errors = ([], read_errors) if read_errors else parse_predictions(raw)
        runs.append({**source, **_summary(rows, run_errors, reference)})
        if sample["ok"] and not run_errors:
            comparisons.append(_compare(sample_rows, rows, source["path"]))
    if not runs:
        errors.append("at least one run output file is required")
    ok = not errors and sample["ok"] and all(run["ok"] for run in runs)
    return {"ok": ok and (reference_report is None or reference_report["ok"]), "errors": errors,
            "spec": spec_source, "reference": reference_report, "sample": sample,
            "runs": runs, "comparisons": comparisons}
