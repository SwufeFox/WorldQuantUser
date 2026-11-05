"""Expression lint commands and simulation preflight checks."""

from __future__ import annotations

import expression_lint
import expression_units
import field_index
import machine_lib as ml
import research_config as config_io

from .runtime import CliError, collect_alphas, merge_settings, note, with_session


def _preflight_lint(session, pairs, settings, args):
    """Lint against the operator catalog before spending simulation quota.

    Two catalogs are consulted, not one. The operator index catches
    ``regression_neut``-style names that have docs but cannot be simulated. The
    field index catches identifiers that were typed from memory and never
    existed, which the operator list cannot see because such a name is not an
    operator at all -- one such typo cost a full simulation round trip before
    this check existed.

    Raises ``INVALID_INPUT`` on a hard lint failure unless the caller passed
    ``--allow-lint-errors``; a clean pass is reported on stderr as an event so
    stdout stays a single JSON document.
    """
    operator_index = None
    catalog_note = "checked"
    try:
        operator_index = _operator_index_from_catalog(session)
    except Exception as exc:  # noqa: BLE001
        # The operator list is metadata. A hiccup there must not block a run that
        # is otherwise fine, but the degradation is announced rather than hidden
        # so nobody mistakes a partial lint for a full one.
        catalog_note = "unavailable: %s: %s" % (type(exc).__name__, exc)
        note("lint: operator catalog %s; continuing with the learned denylist only"
             % catalog_note)

    known_fields = None
    field_types = None
    market_fields = None
    field_note = "unavailable"
    try:
        # One fetch, three views. The dataset split feeds the turnover check: a construct
        # with no daily leg turned over 0.006-0.010 against a 0.01 floor, while the same
        # disclosure fields with such a leg cleared it comfortably and entered the book.
        # The list is kept here, visible, rather than buried in the linter.
        slow_datasets = {"fundamental2", "fundamental6", "analyst4"}
        records = field_index.fetch_all_fields(
            session,
            # Lint against the settings the simulation will actually use. Reading the
            # raw CLI flags instead meant a research file asking for TOP500 was still
            # linted against TOP3000: different field catalogue, so the check could
            # reject a valid field or wave an unavailable one through.
            region=settings.get("region") or ml.DEFAULT_SETTINGS["region"],
            universe=settings.get("universe") or ml.DEFAULT_SETTINGS["universe"],
            delay=settings.get("delay") if settings.get("delay") is not None else ml.DEFAULT_SETTINGS["delay"],
            instrument_type=settings.get("instrumentType") or ml.DEFAULT_SETTINGS["instrumentType"],
        )
        known_fields = field_index.field_ids(records)
        field_note = "checked (%d names)" % len(known_fields)
        # Field type decides whether an operator can read the field at all: a third
        # of the catalogue is VECTOR and most operators reject event inputs.
        field_types = {record.get("id"): record.get("type")
                       for record in records if record.get("id")}
        market_fields = set()
        for record in records:
            field_id = record.get("id")
            if not field_id:
                continue
            dataset = record.get("dataset")
            dataset_id = dataset.get("id") if isinstance(dataset, dict) else dataset
            if dataset_id not in slow_datasets:
                market_fields.add(field_id)
    except Exception as exc:  # noqa: BLE001
        field_note = "unavailable: %s: %s" % (type(exc).__name__, exc)
        note("lint: field catalog unavailable (%s); field names are NOT verified "
             "for this run, so a typo will only surface at the simulator" % exc)

    report = expression_lint.lint_all(
        pairs,
        operator_index=operator_index,
        denylist=expression_lint.load_denylist(),
        known_fields=known_fields,
        strict_fields=bool(known_fields),
        field_types=field_types,
        market_fields=market_fields,
        field_units=expression_units.load_field_units(),
    )
    report["operator_catalog_size"] = len(operator_index or {})
    report["operator_catalog"] = catalog_note
    report["field_catalog"] = field_note
    if report["ok"]:
        return report
    if getattr(args, "allow_lint_errors", False):
        return report
    unknown = sorted({item.get("operator") for item in report["errors"]
                      if item.get("code") == "unknown_operator"})
    unknown_fields = sorted({item.get("identifier") for item in report["errors"]
                             if item.get("code") == "unknown_identifier"})
    hint = "fix the expressions, or pass --allow-lint-errors to submit anyway"
    if unknown:
        hint = ("operator(s) %s are not in the BRAIN operator list; a docs page "
                "existing does not make them simulatable" % ", ".join(unknown))
    if unknown_fields:
        hint = ("field name(s) %s do not exist in the live catalog for this scope; "
                "run `catalog concepts --search <word>` to find the real variants "
                "instead of typing a name from memory" % ", ".join(unknown_fields))
    raise CliError(
        "INVALID_INPUT",
        "expression lint failed; nothing was submitted",
        hint,
        details={
            "operator_catalog_size": report["operator_catalog_size"],
            "field_catalog": field_note,
            "unknown_fields": unknown_fields,
            "errors": report["errors"],
            "warnings": report["warnings"],
        },
    )


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


DISCLOSURE_DATASETS = frozenset({"fundamental2", "fundamental6", "analyst4"})


def _dataset_id(record):
    """Dataset id from a catalogue record, which may nest it or not."""
    dataset = record.get("dataset")
    return dataset.get("id") if isinstance(dataset, dict) else dataset


def _field_scope(session, settings):
    """Return ``(known_field_ids, market_field_ids)`` from one catalogue fetch.

    Two views on purpose: the first answers "is this a real field", the second "does
    anything here move with the market". Fetching twice would let them disagree.
    """
    records = field_index.fetch_all_fields(
        session,
        region=settings["region"],
        universe=settings["universe"],
        delay=settings["delay"],
        instrument_type=settings["instrumentType"],
    )
    known = field_index.field_ids(records)
    market = {record.get("id") for record in records
              if record.get("id") and _dataset_id(record) not in DISCLOSURE_DATASETS}
    return known, market


def cmd_lint(args, sink):
    settings, pairs = _collect_lint_pairs(args)
    denylist = expression_lint.load_denylist(getattr(args, "denylist", None))
    operator_index = None
    known_fields = None
    market_fields = None
    catalog_checked = False
    if not getattr(args, "offline", False):
        def work(session):
            known, market = _field_scope(session, settings)
            return _operator_index_from_catalog(session), known, market

        operator_index, known_fields, market_fields = with_session(args, work)
        catalog_checked = operator_index is not None

    results = []
    for expression, decay in pairs:
        report = expression_lint.lint_expression(
            expression,
            operator_index=operator_index,
            denylist=denylist,
            known_fields=known_fields,
            strict_fields=getattr(args, "strict_fields", False),
            market_fields=market_fields,
            field_units=expression_units.load_field_units(),
        )
        report["expression"] = expression
        if decay is not None:
            report["decay"] = decay
        results.append(report)

    failed = [item for item in results if not item["ok"]]
    document = {
        "expressions": len(results),
        "ok": not failed,
        "catalog_checked": catalog_checked,
        "operator_catalog_size": len(operator_index or {}),
        "denylist_size": len(denylist),
        "known_field_count": None if known_fields is None else len(known_fields),
        "settings": settings or None,
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
