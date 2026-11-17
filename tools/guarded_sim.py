#!/usr/bin/env python3
"""Check Alpha source and settings before dispatching a BRAIN simulation.

Probe supplies its scoped preflight report. This wrapper checks that the report
matches the resolved settings and normalized source submitted to the client.
Direct callers can let the wrapper check the inputs:

    from guarded_sim import simulate
    records = simulate(["rank(close)"], {"universe": "TOP500"})

Direct FASTEXPR checks use operator and field catalogs. Python also requires
verified language access for the dispatching account and validates declared
fields through the CLI preflight.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import expression_lint  # noqa: E402
import field_index  # noqa: E402
import machine_lib as ml  # noqa: E402


class UnknownNameError(RuntimeError):
    """Raised when the expression references a name the catalog does not have."""


def _catalog(session, settings):
    """Return (operator_index, known_fields, field_types) from cache, fetching on miss."""
    operators = ml.fetch_operators(session)
    operator_index = expression_lint.build_operator_index(operators)

    records = field_index.fetch_all_fields(
        session,
        instrument_type=settings.get("instrumentType", ml.DEFAULT_SETTINGS["instrumentType"]),
        region=settings.get("region", ml.DEFAULT_SETTINGS["region"]),
        delay=settings.get("delay", ml.DEFAULT_SETTINGS["delay"]),
        universe=settings.get("universe", ml.DEFAULT_SETTINGS["universe"]),
    )
    known = field_index.field_ids(records)
    types = {r.get("id"): r.get("type") for r in records if r.get("id")}
    return operator_index, known, types


def check(expressions, settings=None, session=None):
    """Lint against the live catalog and return the report (no simulation)."""
    settings = dict(settings or {})
    session = session or ml.login()
    if ml.is_python_language(settings.get("language")):
        return _check_python(expressions, ml.resolve_settings(settings), session)
    operator_index, known, types = _catalog(session, settings)
    pairs = [(expr, settings.get("decay", 4)) for expr in expressions]
    return expression_lint.lint_all(
        pairs,
        operator_index=operator_index,
        denylist=expression_lint.load_denylist(),
        known_fields=known,
        strict_fields=True,
        field_types=types,
        field_units=None,
    )


def _check_python(expressions, settings, session, capability_snapshot=None):
    from research_app.runtime import CliError
    from research_app.validation import _preflight_lint

    pairs = ml.normalize_alphas(expressions, settings)
    try:
        return _preflight_lint(session, pairs, settings, SimpleNamespace(),
                               capability_snapshot=capability_snapshot)
    except CliError as exc:
        if exc.details is None:
            raise
        return exc.details


def simulate(expressions, settings=None, labels=None, force=False, session=None,
             preflight=None, **kwargs):
    """Validate once, reusing only a report bound to these exact resolved requests."""
    settings = ml.resolve_settings(settings or {})
    python = ml.is_python_language(settings.get("language"))
    if preflight is not None:
        expected = {"settings": settings,
                    "pairs": [list(pair) for pair in ml.normalize_alphas(expressions, settings)]}
        if (not isinstance(preflight, dict) or preflight.get("checked_requests") != expected
                or not isinstance(preflight.get("errors"), list)):
            raise UnknownNameError("preflight does not match the exact simulation requests")
        report = preflight
        blocking = report["errors"]
    elif python:
        session = session or ml.login()
        report = _check_python(expressions, settings, session, kwargs.get("capability_snapshot"))
        blocking = report["errors"]
    else:
        session = session or ml.login()
        report = check(expressions, settings, session=session)
        blocking = [e for e in report["errors"]
                    if e["code"] in {"unknown_operator", "unknown_identifier",
                                     "unused_variable", "operator_arg_constraint",
                                     "signal_type", "undefined_variable"}]
    if blocking and (python or not force or any(e.get("code") == "unsupported_setting" for e in blocking)):
        lines = []
        for item in blocking:
            subject = item.get("identifier") or item.get("operator") or item["code"]
            lines.append("  %-14s %s" % (item["code"], subject))
            if item.get("message"):
                lines.append("      " + item["message"])
            # `check` already knows which expression each finding came from; dropping
            # it here left a batch of eight with one bad name and no way to tell
            # which -- you had to rerun `check` yourself to find out.
            if item.get("expression"):
                lines.append("      in: " + " ".join(item["expression"].split())[:100])
        raise UnknownNameError(
            "refusing to spend a simulation: %d blocking finding(s)\n%s"
            % (len(blocking), "\n".join(lines))
        )
    session = session or ml.login()
    if python:
        import simulation_capabilities

        # Preflight may have been supplied by the caller. Bind the evidence
        # to the session that will actually send the Python request.
        claimed = (report.get("capability_snapshot") or {}).get("account") or {}
        account = simulation_capabilities.account_provenance(session)
        if ((report.get("language_capability") or {}).get("state") != "verified"
                or not claimed.get("fingerprint") or claimed["fingerprint"] != account.get("fingerprint")):
            raise UnknownNameError("Python preflight does not verify availability for this account")
    return ml.simulate_alphas(
        session,
        expressions,
        settings=settings,
        labels=labels,
        **kwargs,
    )


if __name__ == "__main__":
    # Running this file is a mistake worth naming: it is a library with no CLI, and
    # silence looks like "the check passed" to anyone who typed it expecting output.
    import sys as _sys

    print("guarded_sim is a library, not a command:\n"
          "  from guarded_sim import simulate, check\n"
          "  CLI entry point: research_cli.py probe SPEC.json",
          file=_sys.stderr)
    raise SystemExit(2)
