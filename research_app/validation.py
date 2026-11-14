"""Expression lint commands and simulation preflight checks."""

from __future__ import annotations

import expression_lint
import expression_units
import field_contract
import machine_lib as ml
import research_config as config_io

from . import capabilities
from .capabilities import cached_field_scope
from .runtime import CliError, collect_alphas, merge_settings, with_session


def _preflight_lint(session, pairs, settings, args):
    """Require complete scoped metadata before spending simulation quota."""
    operator_index = None
    operator_evidence = {"state": "unknown", "complete": False}
    fields = capabilities.field_scope([], {
        "state": "unknown", "scope": capabilities.scope_settings(settings),
        "account": capabilities.account_provenance(session),
        "coverage": {"complete": False}, "freshness": {"state": "unknown"}})
    try:
        records = ml.fetch_operators(session, refresh=True, report=operator_evidence)
        if (records and operator_evidence.get("complete") is True
                and not operator_evidence.get("stale") and not operator_evidence.get("error")
                and not operator_evidence.get("degraded") and not operator_evidence.get("hit")):
            operator_index = expression_lint.build_operator_index(records)
        operator_evidence["state"] = "verified" if operator_index else "unknown"
        operator_evidence["complete"] = bool(operator_index)
    except Exception as exc:  # noqa: BLE001 - preserve failed metadata evidence
        operator_evidence.update(state="unknown", complete=False, error=str(exc))
    try:
        fields = _field_scope(session, settings)
    except Exception as exc:  # noqa: BLE001 - a failure must never become a green light
        fields["evidence"]["error"] = str(exc)
    report = expression_lint.lint_all(
        pairs, operator_index=operator_index,
        denylist=expression_lint.load_denylist(),
        known_fields=fields["known_fields"], strict_fields=True,
        field_types=fields["field_types"], market_fields=fields["market_fields"],
        field_units=expression_units.load_field_units())
    attach_field_contracts(report, fields)
    report["operator_catalog_size"] = len(operator_index or {})
    report["operator_catalog"] = operator_evidence
    report["field_catalog"] = fields["evidence"]
    report["coverage"] = {"operators": operator_evidence["state"],
                          "fields": fields["evidence"]["state"]}
    for catalog, state in report["coverage"].items():
        if state != "verified":
            report["errors"].append({"code": "catalog_unverified", "catalog": catalog,
                                     "message": "required %s catalog is unavailable, empty, stale or incomplete" % catalog})
    report["ok"] = not report["errors"]
    report["allow_lint_errors"] = bool(getattr(args, "allow_lint_errors", False))
    if report["ok"] or report["allow_lint_errors"]:
        return report
    raise CliError("INVALID_INPUT", "expression preflight failed; nothing was submitted",
                   "refresh the required catalogs and fix expression errors; --allow-lint-errors explicitly overrides this guard",
                   details=report)


def attach_field_contracts(report, fields):
    """Attach economic evidence without confusing unknown semantics with type errors."""
    contracts = []
    for result in report.get("results", []):
        evidence = field_contract.audit_expression(result["expression"], fields.get("records") or [])
        result["field_contract"] = evidence
        contracts.append({"expression": result["expression"], **evidence})
        if evidence["state"] != "reviewed":
            warning = {"code": "field_contract_unknown", "severity": "warning",
                       "message": "economic units, periods or nested operation semantics remain unverified"}
            result["warnings"].append(warning)
            report["warnings"].append({**warning, "expression": result["expression"]})
    report["field_contracts"] = contracts
    return report


def _collect_lint_pairs(args):
    """Return ``(settings, [(expression, decay), ...])`` for the lint target."""
    file_settings, alphas = collect_alphas(args)
    settings = merge_settings(file_settings, args)
    try:
        return config_io.build_config(settings, alphas)
    except config_io.ResearchError as exc:
        raise CliError("INVALID_INPUT", str(exc)) from exc


def _operator_index_from_catalog(session):
    return expression_lint.build_operator_index(ml.fetch_operators(session))


def _field_scope(session, settings):
    scope = capabilities.scope_settings(settings)
    evidence = {}
    try:
        records = capabilities.fetch_fields(
            session, region=scope["region"], universe=scope["universe"],
            delay=scope["delay"], instrument_type=scope["instrumentType"],
            refresh=True, report=evidence)
    except Exception as exc:
        evidence.update(state="unknown", error=str(exc))
        evidence.setdefault("scope", scope)
        evidence.setdefault("coverage", {"complete": False})
        evidence.setdefault("freshness", {"state": "unknown"})
        records = []
    return capabilities.field_scope(records, evidence)


def cmd_lint(args, sink):
    """Lint FASTEXPR expressions against operator/field catalogs, learned denylists, and syntax rules."""
    settings, pairs = _collect_lint_pairs(args)
    denylist = expression_lint.load_denylist(getattr(args, "denylist", None))
    operator_index = None
    fields = cached_field_scope(settings, getattr(args, "meta_max_age", None))
    catalog_checked = False
    if not getattr(args, "offline", False):
        def work(session):
            return _operator_index_from_catalog(session), _field_scope(session, settings)

        operator_index, fields = with_session(args, work)
        catalog_checked = bool(operator_index) and fields["evidence"]["state"] == "verified"
    known_fields = fields["known_fields"]
    field_types = fields["field_types"]
    market_fields = fields["market_fields"]

    results = []
    for expression, decay in pairs:
        report = expression_lint.lint_expression(
            expression,
            operator_index=operator_index,
            denylist=denylist,
            known_fields=known_fields,
            strict_fields=getattr(args, "strict_fields", False),
            field_types=field_types,
            market_fields=market_fields,
            field_units=expression_units.load_field_units(),
        )
        report["expression"] = expression
        if decay is not None:
            report["decay"] = decay
        results.append(report)
    contract_report = attach_field_contracts({"results": results, "warnings": []}, fields)

    failed = [item for item in results if not item["ok"]]
    document = {
        "expressions": len(results),
        "ok": not failed,
        "catalog_checked": catalog_checked,
        "operator_catalog_size": len(operator_index or {}),
        "denylist_size": len(denylist),
        "known_field_count": None if known_fields is None else len(known_fields),
        "settings": settings or None,
        "field_catalog": fields["evidence"],
        "field_contracts": contract_report["field_contracts"],
        "summary": {
            "errors": sum(len(item["errors"]) for item in results),
            "warnings": sum(len(item["warnings"]) for item in results),
        },
        "results": results,
    }

    if failed:
        raise CliError(
            "INVALID_INPUT",
            "%d of %d expression(s) failed lint" % (len(failed), len(results)),
            "see data.results[].errors; `unknown_operator` means the operator is "
            "not in the BRAIN operator list even when a documentation page exists",
            details=document,
        )
    return document
