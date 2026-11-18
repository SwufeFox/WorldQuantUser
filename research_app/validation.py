"""Expression lint commands and simulation preflight checks."""

from __future__ import annotations

import ast
import copy

import expression_lint
import expression_units
import fastexpr_ast
import field_contract
import machine_lib as ml
import research_config as config_io
import simulation_capabilities

from . import capabilities
from .capabilities import cached_field_scope
from .runtime import CliError, collect_alphas, merge_settings, with_session


def _python_finding(code, message, severity="error", **location):
    return {"code": code, "message": message, "severity": severity, **location}


def _python_history_warnings(function, data_argument, fields, lookback):
    """Compare direct, literal history access to the maximum available window."""
    if type(lookback) is not int or lookback < 0:
        return []
    available = lookback + 1
    findings = []
    pending = list(reversed(function.body))
    while pending:
        node = pending.pop()
        # Nested scopes may shadow the decorated function's data parameter.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                             ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            if getattr(node, "name", None) == data_argument:
                return []
            continue
        if (isinstance(node, ast.Name) and node.id == data_argument
                and isinstance(node.ctx, (ast.Store, ast.Del))):
            return []
        pending.extend(reversed(list(ast.iter_child_nodes(node))))
        if not isinstance(node, ast.Subscript) or not isinstance(node.ctx, ast.Load):
            continue
        field = node.value
        if not (isinstance(field, ast.Attribute) and field.attr in fields
                and isinstance(field.value, ast.Name) and field.value.id == data_argument):
            continue
        bound = node.slice
        slice_access = isinstance(bound, ast.Slice)
        if slice_access:
            if bound.upper is not None or bound.step is not None:
                continue
            bound = bound.lower
        if not (isinstance(bound, ast.UnaryOp) and isinstance(bound.op, ast.USub)
                and isinstance(bound.operand, ast.Constant) and type(bound.operand.value) is int):
            continue
        required = bound.operand.value
        if required <= available:
            continue
        if slice_access:
            code = "python_history_slice_truncated"
            message = f"{field.attr}[-{required}:] requests {required} rows but can return at most {available}"
        else:
            code = "python_history_index_out_of_range"
            message = f"{field.attr}[-{required}] needs {required} rows but can have at most {available}"
        findings.append(_python_finding(
            code, message, "warning", identifier=field.attr, line=node.lineno,
            column=node.col_offset + 1, lookback=lookback, required_rows=required, available_rows=available))
    return findings


def _parse_python_source(source, *, lookback=None):
    """Read the static BRAIN Alpha interface without importing analyst code.

    Literal data declarations allow the online preflight to verify the exact
    submitted field set. Return-array shape and dtype are runtime properties.
    """
    result = {"language": "PYTHON", "expression": source, "declared_fields": None,
              "operator_count": None, "operator_estimate": None, "operators": [],
              "calls": [], "denylist_unverified": [], "details": None,
              "errors": [], "warnings": []}
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        result["errors"].append(_python_finding(
            "python_syntax", getattr(exc, "msg", str(exc)),
            line=getattr(exc, "lineno", None), column=getattr(exc, "offset", None)))
        return result

    names = {"alpha"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "brain.alphas" and not node.level:
            names.update(alias.asname or alias.name for alias in node.names if alias.name == "alpha")
    decorators = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Name) and target.id in names:
                decorators.append((node, decorator))
    if len(decorators) != 1:
        result["errors"].append(_python_finding(
            "python_alpha_entrypoint", "one top-level function must use @alpha(data=[...])"))
        return result

    function, decorator = decorators[0]
    arguments = function.args
    positional = arguments.posonlyargs + arguments.args
    if (len(positional) != 2 or arguments.kwonlyargs
            or arguments.vararg or arguments.kwarg):
        result["errors"].append(_python_finding(
            "python_alpha_signature", "the @alpha function requires exactly two positional arguments",
            line=function.lineno))
    keywords = decorator.keywords if isinstance(decorator, ast.Call) else ()
    fields = [keyword.value for keyword in keywords if keyword.arg == "data"]
    literal = fields[0] if len(fields) == 1 else None
    if (not isinstance(literal, ast.List) or any(keyword.arg is None for keyword in keywords)
            or any(not isinstance(item, ast.Constant) or not isinstance(item.value, str)
                   or not item.value or item.value != item.value.strip() for item in literal.elts)):
        result["errors"].append(_python_finding(
            "python_data_declaration", "declare data in @alpha as a literal list of field names",
            line=decorator.lineno))
        return result

    declared = [item.value for item in literal.elts]
    if "universe" in declared:
        result["errors"].append(_python_finding(
            "python_implicit_field", "data.universe is supplied automatically; remove universe from the data list",
            line=literal.lineno, identifier="universe"))
    else:
        result["declared_fields"] = sorted(set(declared))
    if not result["errors"]:
        result["warnings"].extend(_python_history_warnings(
            function, positional[0].arg, set(declared), lookback))
    return result


def _attach_python_fields(result, known_fields):
    declared = result["declared_fields"]
    state = "unknown"
    if declared == []:
        state = "not_applicable"
    elif declared is not None:
        if known_fields is None:
            result["warnings"].append(_python_finding(
                "python_fields_unverified", "account access to the declared fields has not been verified", "warning"))
        else:
            missing = sorted(set(declared).difference(known_fields))
            state = "unsupported" if missing else "verified"
            result["errors"].extend(_python_finding(
                "unknown_identifier", "declared field is absent from the scoped field catalog",
                identifier=field) for field in missing)
    result["fields_state"] = state
    result["ok"] = not result["errors"]
    return result


def lint_python_source(source, known_fields=None, *, lookback=None):
    """Parse and check declared Python Alpha fields without loading the code."""
    return _attach_python_fields(_parse_python_source(source, lookback=lookback), known_fields)


def _python_batch(results, known_fields):
    errors, warnings = [], []
    for result in results:
        _attach_python_fields(result, known_fields)
        errors.extend(dict(item, expression=result["expression"]) for item in result["errors"])
        warnings.extend(dict(item, expression=result["expression"]) for item in result["warnings"])
    return {"ok": not errors, "expressions": len(results), "results": results,
            "errors": errors, "warnings": warnings, "field_contracts": []}


def _preflight_lint(session, pairs, settings, args, capability_snapshot=None):
    """Require fresh field proofs and check supplied or cached settings evidence."""
    settings = ml.resolve_settings(settings)
    pairs = [[pair[0], pair[1]] if isinstance(pair, (tuple, list))
             else [pair, settings["decay"]] for pair in pairs]
    python = ml.is_python_language(settings.get("language"))
    snapshot = capability_snapshot if capability_snapshot is not None else getattr(args, "capability_snapshot", None)
    if snapshot is None:
        snapshot = simulation_capabilities.fetch(session) if python else simulation_capabilities.cached()
    account = simulation_capabilities.account_provenance(session)
    if (not account.get("fingerprint") or not isinstance(snapshot.get("account"), dict)
            or snapshot["account"].get("fingerprint") != account.get("fingerprint")):
        snapshot = {**snapshot, "state": "unknown", "reasons": ["account_mismatch"]}
    requests = [ml.regular_settings_evidence({**settings, "decay": pair[1]}, snapshot) for pair in pairs]
    settings_report = ml.regular_settings_evidence(settings, snapshot) if not requests else {
        **requests[0],
        "state": "unsupported" if any(row["unsupported"] for row in requests) else
                 "unknown" if any(row["state"] == "unknown" for row in requests) else "verified",
        "checked": sorted({key for row in requests for key in row["checked"]}),
        "unsupported": [{**item, "pair_index": index} for index, row in enumerate(requests)
                        for item in row["unsupported"]],
        "unknown": [{**item, "pair_index": index} for index, row in enumerate(requests)
                    for item in row["unknown"]],
        "requests": requests}
    operator_index = None
    operator_evidence = {"state": "unknown", "complete": False}
    fields = capabilities.field_scope([], {
        "state": "unknown", "scope": capabilities.scope_settings(settings),
        "account": account,
        "coverage": {"complete": False}, "freshness": {"state": "unknown"}})
    if python:
        # Python declarations provide field names directly; operator metadata
        # and FASTEXPR type inference do not apply.
        operator_evidence.update(state="not_applicable")
        results = [_parse_python_source(expression, lookback=settings.get("lookback"))
                   for expression, _ in pairs]
        required = {field for result in results for field in (result["declared_fields"] or [])}
        if required:
            fields = _field_scope(session, settings, required_fields=required)
        elif all(result["declared_fields"] == [] for result in results):
            fields["evidence"]["state"] = "not_applicable"
        report = _python_batch(results, fields["known_fields"])
        schema = snapshot.get("settings_schema")
        advertised = (simulation_capabilities.choice_values(schema.get("language"), settings)
                      if snapshot.get("state") == "verified" and isinstance(schema, dict) else None)
        language_state = ("unknown" if advertised is None else
                          "verified" if settings["language"] in advertised else "unsupported")
        if advertised is None:
            report["errors"].append(_python_finding(
                "python_language_unverified", "the account's simulation settings do not verify Python language availability"))
        report["language"] = "PYTHON"
        report["language_capability"] = {"state": language_state, "source": snapshot.get("source")}
    else:
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
        required = set()
        for expression, _ in pairs:
            try:
                required.update(fastexpr_ast.field_names(fastexpr_ast.parse(expression)))
            except fastexpr_ast.ParseError:
                # The linter reports syntax errors without a catalog lookup.
                continue
        if required:
            try:
                fields = _field_scope(session, settings, required_fields=required)
            except ml.AuthError:
                raise
            except Exception as exc:  # noqa: BLE001 - retain failed capability evidence
                fields["evidence"]["error"] = str(exc)
        else:
            fields["evidence"]["state"] = "not_applicable"
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
    report["settings_capabilities"] = settings_report
    report["capability_snapshot"] = copy.deepcopy(snapshot)
    report["checked_requests"] = {"settings": copy.deepcopy(settings), "pairs": copy.deepcopy(pairs)}
    for unsupported in settings_report["unsupported"]:
        report["errors"].append({"code": "unsupported_setting", **unsupported})
    report["coverage"] = {"operators": operator_evidence["state"],
                          "fields": fields["evidence"]["state"]}
    for catalog, state in report["coverage"].items():
        if state not in {"verified", "not_applicable"}:
            report["errors"].append({"code": "catalog_unverified", "catalog": catalog,
                                     "message": "required %s catalog is unavailable, empty, stale or incomplete" % catalog})
    report["ok"] = not report["errors"]
    report["allow_lint_errors"] = bool(getattr(args, "allow_lint_errors", False))
    if report["ok"] or (not python and report["allow_lint_errors"] and not settings_report["unsupported"]):
        return report
    hint = ("verify account language availability and declare data fields in a literal list" if python else
            "refresh required catalogs and fix expression or advertised setting errors; --allow-lint-errors overrides lint only")
    raise CliError("INVALID_INPUT", "simulation preflight failed; nothing was submitted",
                   hint, details=report)


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


def _field_scope(session, settings, required_fields=None):
    scope = capabilities.scope_settings(settings)
    evidence = {}
    try:
        if required_fields:
            return capabilities.required_field_scope(session, settings, required_fields)
        records = capabilities.fetch_fields(
            session, region=scope["region"], universe=scope["universe"],
            delay=scope["delay"], instrument_type=scope["instrumentType"],
            refresh=True, report=evidence)
    except ml.AuthError:
        raise
    except Exception as exc:
        evidence.update(state="unknown", error=str(exc))
        evidence.setdefault("scope", scope)
        evidence.setdefault("coverage", {"complete": False})
        evidence.setdefault("freshness", {"state": "unknown"})
        records = []
    return capabilities.field_scope(records, evidence)


def _cmd_lint_python(args, settings, pairs):
    offline = bool(getattr(args, "offline", False))
    if offline:
        fields = cached_field_scope(settings, getattr(args, "meta_max_age", None))
        reports = [_parse_python_source(expression, lookback=settings.get("lookback"))
                   for expression, _ in pairs]
        result = _python_batch(reports, fields["known_fields"])
        result["field_catalog"] = fields["evidence"]
        result["language_capability"] = {"state": "unknown", "source": None}
    else:
        def work(session):
            try:
                return _preflight_lint(session, pairs, settings, args)
            except CliError as exc:
                if exc.details is None:
                    raise
                return exc.details

        result = with_session(args, work)
    for report, (_, decay) in zip(result["results"], pairs):
        report["decay"] = decay
    result.update(language="PYTHON", settings=settings, operator_catalog_size=0, denylist_size=0,
                  known_field_count=result["field_catalog"].get("coverage", {}).get("field_count"),
                  catalog_checked=not offline and result["field_catalog"]["state"] in {"verified", "not_applicable"},
                  summary={"errors": len(result["errors"]), "warnings": len(result["warnings"])})
    if not result["ok"]:
        raise CliError("INVALID_INPUT", "Python Alpha lint failed",
                       "inspect the syntax, field declarations and language availability in details",
                       details=result)
    return result


def cmd_lint(args, sink):
    """Lint code with the checks and catalogs for its configured language."""
    settings, pairs = _collect_lint_pairs(args)
    if ml.is_python_language(settings.get("language")):
        return _cmd_lint_python(args, settings, pairs)
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
