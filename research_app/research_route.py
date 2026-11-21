"""Normalize experiment routing without assigning platform eligibility."""

from collections.abc import Mapping
from decimal import Decimal
import json

import machine_lib


TRACKS = ("REGULAR", "POWER_POOL", "ATOM", "SPC")
ATOM_RUNNER_MECHANISM = "ATOM / Power Pool scan"
SPEC_CONTRACT = {
    "track": {"choices": list(TRACKS), "default_for_probe": "REGULAR"},
    "brain_simulation": {"required": ["arms"], "optional": ["settings", "design"],
                         "native_type": "REGULAR", "languages": ["FASTEXPR", "PYTHON"]},
    "spc": {"required": ["name", "prompt", "sampleOutput", "modelFamily", "weight",
                         "updateFrequency", "runs"], "optional": ["reference", "as_of"],
            "captures": "paths relative to the spec or absolute paths"},
}


def _track(value):
    if not isinstance(value, str):
        raise ValueError("track must be REGULAR, POWER_POOL, ATOM or SPC")
    name = "_".join(value.upper().replace("-", " ").split())
    if name == "PP":
        name = "POWER_POOL"
    if name not in TRACKS:
        raise ValueError("track must be REGULAR, POWER_POOL, ATOM or SPC")
    return name


def _setting_token(value, key):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string when specified")
    return value.strip().upper()


def _validate_spc(spec):
    from tools.spc_experiment import _spec, parse_predictions

    # Reuse the same source and sample validation as the offline SPC experiment
    # command. Model and frequency availability belong to the provider schema.
    _, errors = _spec(json.dumps(dict(spec), allow_nan=False, ensure_ascii=False))
    _, sample_errors = parse_predictions(spec.get("sampleOutput"))
    errors.extend(f"sampleOutput: {error}" for error in sample_errors)
    weight = spec.get("weight")
    if type(weight) not in (int, float) or not 0 <= weight <= 1:
        errors.append("weight must be a number from 0 to 1")
    elif (decimal_weight := Decimal(str(weight))) != decimal_weight.quantize(Decimal("0.01")):
        errors.append("weight must have at most two decimal places")
    if errors:
        raise ValueError("invalid SPC spec: " + ", ".join(errors))


def normalize_route(spec, *, settings=None):
    """Return normalized routing metadata, leaving spec and wire settings intact.

    The optional settings argument is the effective arm settings after CLI
    overrides. An omitted simulation mode leaves the native default in effect.
    A track states research intent. Capability and classification evidence
    determine what the account and a simulated alpha actually qualify for.
    """
    if not isinstance(spec, Mapping):
        raise ValueError("experiment spec must be an object")

    raw_track = spec.get("track")
    spc_payload = "prompt" in spec and "sampleOutput" in spec
    if raw_track is None:
        if spc_payload and "arms" not in spec:
            track = "SPC"
        else:
            track = "ATOM" if spec.get("mechanism") == ATOM_RUNNER_MECHANISM else "REGULAR"
    else:
        track = _track(raw_track)
    if track == "SPC":
        if "arms" in spec or "settings" in spec or settings is not None:
            raise ValueError("SPC uses a prompt and sample output, without simulation arms or settings")
        _validate_spc(spec)
        return {
            "track": track,
            "protocol": "spc_submission",
            "simulation_type": None,
            "language": None,
            "simulation_mode": None,
        }

    if spc_payload and "arms" not in spec:
        raise ValueError("an SPC prompt and sample output require the SPC track")
    if settings is None:
        settings = spec.get("settings", {})
    if not isinstance(settings, Mapping):
        raise ValueError("simulation settings must be an object")
    settings = machine_lib.normalize_setting_keys(settings)
    language = _setting_token(settings.get("language"), "language")
    if language is None:
        language = machine_lib.DEFAULT_SETTINGS["language"]
    return {
        "track": track,
        "protocol": "brain_simulation",
        "simulation_type": "REGULAR",
        "language": language,
        "simulation_mode": _setting_token(settings.get("simulationMode"), "simulationMode"),
    }
