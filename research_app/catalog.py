"""Quota-free dataset, field and operator catalog commands."""

from __future__ import annotations

from collections import Counter

import field_contract
import field_index
import field_search
import machine_lib as ml

from .runtime import CliError, session_reader, with_session
from . import capabilities, incentives


def _catalog_access(args):
    run = session_reader(args)
    access = ml.CatalogAccess(ml.configured_username(), run)
    if getattr(args, "no_cache", False) or getattr(args, "relogin", False):
        run(lambda _session: None)
    return access


def cmd_catalog_operators(args, _sink):
    """List supported operators with ranked local search and optional documentation fetch."""
    cache_report = {}

    def work(session):
        # Fetch the whole list and rank locally for the same reason fields are
        # ranked locally: the endpoint matches one token, so a two-word intent
        # could not be expressed at all.
        catalog = ml.fetch_operators(session, search="", group=args.group, report=cache_report)
        operators = catalog
        scanned = len(operators)
        search_note = None
        if args.search:
            required, excluded = field_search.parse_query(args.search)
            operators = field_search.search_records(
                operators, args.search, limit=args.limit, fuzzy=args.fuzzy,
                match_all=not args.any_term, rank_by_coverage=False)
            absent = field_search.absent_terms(catalog, args.search, fuzzy=args.fuzzy) if not operators else []
            search_note = {
                "query": args.search, "required_terms": required,
                "excluded_terms": excluded, "absent_terms": absent,
                "mode": "any" if args.any_term else "all",
                "fuzzy": bool(args.fuzzy),
            }
            if absent:
                search_note["hint"] = "no operator uses these words"
        elif args.limit:
            operators = operators[: args.limit]
        data = {"count": len(operators), "scanned": scanned,
                "operators": operators, "cache": cache_report}
        if search_note:
            data["search"] = search_note
        if args.docs:
            data["docs"] = {n: ml.fetch_operator_doc(session, n) for n in args.docs}
        return data

    data = with_session(args, work)

    return data


def cmd_catalog_datasets(args, _sink):
    """List accessible datasets for a region, universe, delay and instrument type."""
    cache_report = {}
    datasets = ml.get_datasets(
        _catalog_access(args), instrument_type=args.instrument_type, region=args.region,
        delay=args.delay, universe=args.universe, limit=args.limit,
        report=cache_report,
        theme=None if getattr(args, "theme", None) is None else args.theme == "true",
    )
    data = {
        "region": args.region, "universe": args.universe, "delay": args.delay,
        "instrument_type": args.instrument_type, "theme": getattr(args, "theme", None),
        "count": len(datasets), "datasets": datasets, "cache": cache_report,
    }
    if not datasets:
        data["hint"] = ("no datasets were returned for this scope; inspect cache freshness and "
                        "enumeration completeness before drawing an access conclusion")

    return incentives.catalog_summary(data, "datasets") if getattr(args, "summary", False) else data


def cmd_catalog_fields(args, _sink):
    """List a dataset's fields, with ranked local search.

    Search runs here rather than on the server. The endpoint takes a single
    token, so ``--search cfps count`` used to be a usage error and any multi-word
    intent was impossible to express. Fetching the dataset (cached) and ranking
    locally supports AND terms, exclusions, prefixes and typos, and reports why
    each hit matched.
    """
    cache_report = {}
    everything = ml.get_datafields(
        _catalog_access(args), instrument_type=args.instrument_type, region=args.region,
        delay=args.delay, universe=args.universe, dataset_id=args.dataset,
        search="", field_type=args.field_type, limit=None,
        report=cache_report,
    )

    fields = everything
    search_note = None
    if args.search:
        required, excluded = field_search.parse_query(args.search)
        fields = field_search.search_records(
            everything, args.search, limit=args.limit,
            fuzzy=args.fuzzy, match_all=not args.any_term,
        )
        absent = field_search.absent_terms(
            everything, args.search, fuzzy=args.fuzzy) if not fields else []
        search_note = {
            "query": args.search,
            "required_terms": required,
            "excluded_terms": excluded,
            "absent_terms": absent,
            "mode": "any" if args.any_term else "all",
            "fuzzy": bool(args.fuzzy),
            "note": "ranked locally: exact token > prefix > substring, id beats description",
        }
        if absent:
            search_note["hint"] = (
                "the catalog does not use these words anywhere in this dataset; "
                "try --any-term or different wording")
        elif not fields and len(required) > 1:
            # Every word exists somewhere, just never together in one field. That
            # is the common case when asking for an idea spelled across two
            # naming families, and it deserves a different suggestion than a
            # missing word does.
            search_note["hint"] = (
                "each term appears somewhere in this dataset but no single field "
                "contains all of them; try --any-term, or search the terms one at "
                "a time with `catalog concepts`")
    elif args.limit:
        fields = fields[: args.limit]

    type_counts = Counter(field["type"] for field in fields if isinstance(field.get("type"), str))
    data = {
        "dataset": args.dataset, "region": args.region, "universe": args.universe,
        "delay": args.delay, "count": len(fields), "scanned": len(everything),
        "instrument_type": args.instrument_type,
        "type_counts": dict(sorted(type_counts.items())),
        "fields": fields, "cache": cache_report,
    }
    if search_note:
        data["search"] = search_note
    if args.expressions:
        selected = fields[:incentives.SUMMARY_LIMIT] if getattr(args, "summary", False) else fields
        data["expressions"] = ml.process_datafields(selected)
        if getattr(args, "summary", False):
            data["expressions_scope"] = "displayed_fields"

    return incentives.catalog_summary(data, "fields") if getattr(args, "summary", False) else data


def _catalog_complete(report):
    return report.get("complete") is True and not any(
        report.get(key) for key in ("stale", "degraded", "error"))


def _find_field_matches(fields, dataset, args, max_alpha_count):
    category = dataset.get("category")
    if isinstance(category, dict):
        category = category.get("name")
    if args.query:
        fields = field_search.search_records(fields, args.query, fuzzy=args.fuzzy,
                                             match_all=not args.any_term)

    matches = []
    unknown_alpha_count = 0
    for field in fields:
        coverage = field.get("coverage")
        if type(coverage) not in (int, float) or not 0 <= coverage <= 1:
            coverage = None
        if args.min_coverage is not None and (coverage is None or coverage < args.min_coverage):
            continue
        alpha_count = field.get("alphaCount")
        if type(alpha_count) is not int or alpha_count < 0:
            alpha_count = None
        if max_alpha_count is not None:
            if alpha_count is None:
                unknown_alpha_count += 1
                continue
            if alpha_count > max_alpha_count:
                continue
        description = field.get("description")
        matches.append({
            "id": field["id"], "dataset": dataset["id"], "category": category,
            "type": field.get("type"), "coverage": coverage, "date_coverage": field.get("dateCoverage"),
            "alpha_count": alpha_count, "user_count": field.get("userCount"),
            "pyramid_multiplier": incentives.reported_multiplier(field.get("pyramidMultiplier")),
            "description": description.strip()[:120] if isinstance(description, str) else "",
        })
    return matches, unknown_alpha_count


def _find_in_dataset(access, scope, dataset, args, filters):
    metadata = {}
    fields = ml.get_datafields(access, **scope, dataset_id=dataset["id"],
                               field_type=filters["field_type"], report=metadata)
    if not isinstance(fields, list) or any(
        not isinstance(field, dict) or not isinstance(field.get("id"), str) or not field["id"]
        for field in fields
    ):
        raise ml.ServiceError("catalog returned an invalid field ID or collection")
    matches, unknown_alpha_count = _find_field_matches(fields, dataset, args, filters["max_alpha_count"])
    complete = _catalog_complete(metadata)
    return matches, {
        "dataset": dataset["id"], "state": "verified" if complete and not metadata.get("hit") else "unknown",
        "complete": complete, "fields_scanned": len(fields), "metadata": metadata,
        "matches": len(matches), "alpha_count_unknown": unknown_alpha_count,
    }


def cmd_catalog_find(args, sink):
    """Search or filter scoped field catalogs and retain per-dataset evidence."""
    query = args.query
    max_alpha_count = getattr(args, "max_alpha_count", None)
    field_type = getattr(args, "field_type", None)
    sort_by = getattr(args, "sort", "relevance")
    if query is None and max_alpha_count is None and args.min_coverage is None and not field_type:
        raise CliError("INVALID_INPUT", "provide a search query or a field filter",
                       "use --max-alpha-count, --min-coverage or --type to scan without a query")
    if max_alpha_count is not None and max_alpha_count < 0:
        raise CliError("INVALID_INPUT", "--max-alpha-count must be zero or greater")
    if args.min_coverage is not None and not 0 <= args.min_coverage <= 1:
        raise CliError("INVALID_INPUT", "--min-coverage must be between 0 and 1")

    scope = {"region": args.region, "universe": args.universe, "delay": args.delay,
             "instrument_type": args.instrument_type}
    filters = {"field_type": field_type, "max_alpha_count": max_alpha_count, "min_coverage": args.min_coverage}
    if sink is not None:
        sink("catalog_scope_started", scope)
    cache_report = {}
    access = _catalog_access(args)
    datasets = ml.get_datasets(access, **scope, report=cache_report)
    wanted = dict.fromkeys(part.strip() for part in (args.dataset or "").split(",") if part.strip())
    if not isinstance(datasets, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
        for row in datasets
    ):
        raise CliError("SERVICE_ERROR", "catalog returned an invalid dataset ID", details={"cache": cache_report})
    available = {row["id"] for row in datasets}
    missing = [dataset_id for dataset_id in wanted if dataset_id not in available]
    if wanted:
        datasets = [row for row in datasets if row["id"] in wanted]
    if not datasets:
        raise CliError("INVALID_INPUT", "no datasets for %s/%s/delay %s"
                       % (args.region, args.universe, args.delay),
                       details={"cache": cache_report, "missing_datasets": missing})

    rows, catalogs = [], []
    rate_limited = False
    for index, dataset in enumerate(datasets, 1):
        dataset_id = dataset["id"]
        if rate_limited:
            catalogs.append({"dataset": dataset_id, "state": "not_scanned", "matches": None,
                             "fields_scanned": 0, "error_code": "RATE_LIMITED"})
            continue
        progress = {"dataset": dataset_id, "index": index, "total": len(datasets)}
        if sink is not None:
            sink("catalog_scan_started", progress)
        try:
            matches, evidence = _find_in_dataset(access, scope, dataset, args, filters)
        except ml.AuthError:
            raise
        except ml.BrainError as exc:
            matches = []
            evidence = {"dataset": dataset_id, "state": "unknown", "error": str(exc),
                        "error_code": exc.code, "fields_scanned": 0, "matches": None}
        rows.extend(matches)
        catalogs.append(evidence)
        metadata = evidence.get("metadata", {})
        error_code = evidence.get("error_code") or metadata.get("error")
        rate_limited = error_code == "RATE_LIMITED"
        if sink is not None:
            sink("catalog_scan_finished", {**progress, "fields_scanned": evidence["fields_scanned"],
                                           "matches": evidence["matches"], "cache_hit": bool(metadata.get("hit")),
                                           "error_code": error_code})
    attempted = sum(row["state"] != "not_scanned" for row in catalogs)
    catalogs.extend({"dataset": dataset_id, "state": "unknown", "fields_scanned": 0, "matches": None}
                    for dataset_id in missing)

    lowered = (query or "").strip().lower()
    if sort_by == "pyramid-multiplier":
        rows.sort(key=lambda row: (row["pyramid_multiplier"] is None, -(row["pyramid_multiplier"] or 0),
                                   -(row["coverage"] or 0), row["id"]))
        ranking = "reported pyramid multiplier descending, then coverage, then id"
    else:
        rows.sort(key=lambda row: (row["id"].lower() != lowered, -(row["coverage"] or 0), row["id"]))
        ranking = "exact id, then coverage, then id"

    complete = _catalog_complete(cache_report) and all(row.get("complete") for row in catalogs)
    live_verified = complete and not cache_report.get("hit") and all(row["state"] == "verified" for row in catalogs)
    data = {"query": query, "region": args.region, "universe": args.universe, "delay": args.delay,
            "filters": filters, "datasets_scanned": attempted,
            "fields_scanned": sum(row["fields_scanned"] for row in catalogs), "count": len(rows),
            "fields": rows[:args.limit], "cache": cache_report, "ranking": ranking,
            "discovery": {"state": "verified" if live_verified else "unknown", "complete": complete,
                          "full_scope": not bool(wanted), "catalogs": catalogs, "missing_datasets": missing}}
    if not rows:
        if not complete:
            data["hint"] = "scan incomplete; no matches were observed in the available results"
        elif query:
            data["hint"] = "nothing matched; try --fuzzy or --any-term, or adjust the filters"
        else:
            data["hint"] = "nothing matched the filters in the available catalog records"
    return data


def cmd_catalog_concepts(args, _sink):
    """Group fields by shared tokens and show coverage side by side.

    The tool brings no vocabulary of its own: pass ``--ignore-words`` for the
    boilerplate tokens of the family you are looking at, and everything left is
    compared literally. Baking catalog knowledge into the source would encode one
    snapshot of the naming and rot silently, so the judgment stays here at the
    command line.
    """
    evidence = {}
    records = with_session(args, lambda s: capabilities.fetch_fields(
        s, region=args.region, universe=args.universe, delay=args.delay,
        instrument_type=args.instrument_type,
        dataset_ids=[args.dataset] if args.dataset else None,
        report=evidence,
    ))
    concepts = field_index.duplicate_concepts(
        records, ignore_words=args.ignore_words,
        min_members=args.min_members, min_spread=args.min_spread,
        field_type=args.field_type, drop_numeric=args.drop_numeric,
        tail_tokens=args.tail_tokens,
    )
    if args.search:
        # Rank concepts, not raw fields: the concept key is the thing being
        # asked about, and the member ids are folded into the description so a
        # query can hit either the grouping or a specific variant.
        haystack = [
            {
                "id": item["concept"],
                "description": " ".join(member["id"] for member in item["fields"]),
                "coverage": item["best_coverage"],
                "_concept": item,
            }
            for item in concepts
        ]
        ranked = field_search.search_records(
            haystack, args.search, limit=args.limit, fuzzy=args.fuzzy,
            match_all=not args.any_term,
        )
        concepts = []
        for hit in ranked:
            item = dict(hit["_concept"])
            item["score"] = hit["score"]
            item["matched_terms"] = hit["matched_terms"]
            concepts.append(item)
    limit = args.limit or len(concepts)
    data = {
        "region": args.region, "universe": args.universe, "delay": args.delay,
        "dataset": args.dataset,
        "ignore_words": list(args.ignore_words or []),
        "drop_numeric": bool(args.drop_numeric),
        "tail_tokens": args.tail_tokens,
        "capabilities": evidence,
        "fields_scanned": len(records),
        "concepts_with_variants": len(concepts),
        "shown": min(limit, len(concepts)),
        "concepts": concepts[:limit],
        "note": ("grouping is literal token matching after --ignore-words; "
                 "coverage_spread is best minus worst member"),
    }

    return data


def cmd_catalog_verify(args, _sink):
    """Check field identifiers against the live catalog, quota-free.

    This is the check that catches a name typed from memory. Such a name is not
    an operator, so the operator catalog cannot see it, and the first symptom is
    a simulator error after the submission has already been spent.
    """
    wanted = []
    for chunk in args.fields or []:
        for piece in chunk.replace(",", " ").split():
            if piece and piece not in wanted:
                wanted.append(piece)
    if not wanted:
        raise CliError("USAGE", "nothing to verify",
                       "pass --fields with one or more field ids, comma or space separated")

    evidence = {}
    known = with_session(args, lambda s: field_index.field_ids(
        capabilities.fetch_fields(
            s, region=args.region, universe=args.universe, delay=args.delay,
            instrument_type=args.instrument_type, report=evidence,
        )
    ))
    verified = evidence.get("state") == "verified"
    unknown = [name for name in wanted if name not in known] if verified else list(wanted)
    found = [name for name in wanted if name in known] if verified else []
    data = {
        "region": args.region, "universe": args.universe, "delay": args.delay,
        "capabilities": evidence,
        "catalog_size": len(known),
        "checked": len(wanted),
        "found": found,
        "unknown": unknown,
        "verdict": "unverified_catalog" if not verified else "ok" if not unknown else "unknown_fields",
    }
    if unknown and args.strict:
        raise CliError(
            "INVALID_INPUT",
            "catalog could not verify field name(s): %s" % ", ".join(unknown),
            "run `catalog concepts --search <word>` to see the real variants for that idea",
            details=data,
        )

    return data


def cmd_catalog_unused(args, _sink):
    """Count cached fields across scopes that the latest ledger has not referenced."""
    from . import mine

    fields = mine.load_catalog(all_scopes=True)
    if not fields:
        raise ValueError("no cached catalog fields; run `catalog fields` first")
    used_fields = mine.used_identifiers(getattr(args, "registry", None)) & fields.keys()
    bound_count = sum(row["source"]["account_state"] == "matched" for row in fields.values())

    by_dataset = {}
    for field_id, row in fields.items():
        dataset = row["dataset"]
        bucket = by_dataset.setdefault(dataset, {"dataset": dataset, "fields": 0,
                                                 "used": 0, "unused": []})
        bucket["fields"] += 1
        if field_id in used_fields:
            bucket["used"] += 1
        else:
            bucket["unused"].append({"id": field_id,
                                     "coverage": row["coverage"],
                                     "description": row["description"][:70]})

    rows = []
    for dataset, bucket in by_dataset.items():
        sample = sorted(bucket["unused"], key=lambda r: -(r["coverage"] or 0))
        rows.append({"dataset": dataset,
                     "fields": bucket["fields"],
                     "used": bucket["used"],
                     "unused": len(bucket["unused"]),
                     "unused_coverage_median": None if not sample else round(
                         sorted([r["coverage"] or 0 for r in sample])[len(sample) // 2], 3),
                     "top_unused": sample[: args.top]})
    rows.sort(key=lambda r: -r["unused"])
    return {"catalog_fields": len(fields), "used_fields": len(used_fields),
            "unused_fields": len(fields) - len(used_fields),
            "catalog_evidence": {"account_matched": bound_count,
                                 "account_unknown": len(fields) - bound_count,
                                 "live_verified": False},
            "datasets": rows}


def cmd_catalog_categories(args, _sink):
    """List data categories with field and alpha counts indicating namespace crowding."""
    cache_report = {}
    categories = ml.fetch_data_categories(
        _catalog_access(args), region=args.region, universe=args.universe, delay=args.delay,
        report=cache_report)
    data = {
        "region": args.region, "universe": args.universe, "delay": args.delay,
        "count": len(categories), "categories": categories, "cache": cache_report,
        "note": "alphaCount is a crowding signal: higher means more mined",
    }

    return data


def _fetch_raw_fields(session, args, dataset=None, search=""):
    return ml.get_datafields(
        session,
        instrument_type=getattr(args, "instrument_type", ml.DEFAULT_SETTINGS["instrumentType"]),
        region=getattr(args, "region", ml.DEFAULT_SETTINGS["region"]),
        delay=getattr(args, "delay", ml.DEFAULT_SETTINGS["delay"]),
        universe=getattr(args, "universe", ml.DEFAULT_SETTINGS["universe"]),
        dataset_id=dataset or "",
        search=search,
        limit=getattr(args, "limit", None),
    )


def _resolve_fields_by_id(session, args, field_ids):
    """Locate raw catalog records for specific field ids.

    Tries ``--dataset`` first, then walks the region's datasets until every id
    is found. Dataset listing and field listing are both quota-free, so a miss
    costs a request rather than simulation quota.
    """
    wanted = list(field_ids)
    found = []
    if getattr(args, "dataset", None):
        records = _fetch_raw_fields(session, args, dataset=args.dataset)
        by_id = {record.get("id"): record for record in records}
        found = [by_id[field_id] for field_id in wanted if field_id in by_id]
        remaining = [field_id for field_id in wanted if field_id not in by_id]
        if not remaining:
            return found, []
        wanted = remaining

    datasets = ml.get_datasets(
        session,
        instrument_type=getattr(args, "instrument_type", ml.DEFAULT_SETTINGS["instrumentType"]),
        region=getattr(args, "region", ml.DEFAULT_SETTINGS["region"]),
        delay=getattr(args, "delay", ml.DEFAULT_SETTINGS["delay"]),
        universe=getattr(args, "universe", ml.DEFAULT_SETTINGS["universe"]),
    )
    remaining = set(wanted)
    for dataset in datasets:
        dataset_id = dataset.get("id")
        if not dataset_id or not remaining:
            continue
        records = _fetch_raw_fields(session, args, dataset=dataset_id)
        for record in records:
            field_id = record.get("id")
            if field_id in remaining:
                found.append(record)
                remaining.discard(field_id)
        if not remaining:
            break
    order = {field_id: index for index, field_id in enumerate(wanted)}
    found.sort(key=lambda record: order.get(record.get("id"), len(order)))
    return found, sorted(remaining)


def cmd_catalog_field_audit(args, sink):
    field_ids = [item.strip()
                 for item in (getattr(args, "fields", "") or "").split(",")
                 if item.strip()]
    if not field_ids and not getattr(args, "dataset", None):
        raise CliError(
            "USAGE", "field-audit needs --fields and/or --dataset",
            "e.g. `catalog field-audit --fields fnd6_newa1v1300_capx,"
            "fnd6_newa2v1300_oancf --operation ratio`")

    def work(session):
        if field_ids:
            return _resolve_fields_by_id(session, args, field_ids)
        return _fetch_raw_fields(session, args), []

    raw_records, missing = with_session(args, work)
    if not raw_records:
        raise CliError(
            "NOT_FOUND", "no matching datafields",
            "check the field ids and the region/universe/delay; field listing is "
            "quota-free, so a wrong id costs a retry rather than a simulation")

    document = field_contract.audit_fields(
        raw_records, operation=getattr(args, "operation", None))
    document["catalog"] = {
        "region": getattr(args, "region", ml.DEFAULT_SETTINGS["region"]),
        "universe": getattr(args, "universe", ml.DEFAULT_SETTINGS["universe"]),
        "delay": getattr(args, "delay", ml.DEFAULT_SETTINGS["delay"]),
        "instrument_type": getattr(args, "instrument_type", ml.DEFAULT_SETTINGS["instrumentType"]),
    }
    if field_ids:
        document["requested_fields"] = field_ids
    if missing:
        document["missing_fields"] = missing

    return document
