"""Quota-free dataset, field and operator catalog commands."""

from __future__ import annotations

import field_contract
import field_index
import field_search
import machine_lib as ml

from .runtime import CliError, with_session
from . import capabilities


def cmd_catalog_operators(args, _sink):
    """List supported operators with ranked local search and optional documentation fetch."""
    cache_report = {}

    def work(session):
        # Fetch the whole list and rank locally for the same reason fields are
        # ranked locally: the endpoint matches one token, so a two-word intent
        # could not be expressed at all.
        operators = ml.fetch_operators(session, search="", group=args.group,
                                       report=cache_report)
        scanned = len(operators)
        search_note = None
        if args.search:
            required, excluded = field_search.parse_query(args.search)
            operators = field_search.search_records(
                operators, args.search, limit=args.limit, fuzzy=args.fuzzy,
                match_all=not args.any_term, rank_by_coverage=False)
            absent = field_search.absent_terms(
                operators or ml.fetch_operators(session, search="", group=args.group),
                args.search, fuzzy=args.fuzzy) if not operators else []
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
    datasets = with_session(args, lambda s: ml.get_datasets(
        s, instrument_type=args.instrument_type, region=args.region,
        delay=args.delay, universe=args.universe, limit=args.limit,
        report=cache_report,
    ))
    data = {
        "region": args.region, "universe": args.universe, "delay": args.delay,
        "count": len(datasets), "datasets": datasets, "cache": cache_report,
    }
    if not datasets:
        data["hint"] = ("no datasets for this combination; the account may lack "
                        "permission (a TUTORIAL account only sees USA)")

    return data


def cmd_catalog_fields(args, _sink):
    """List a dataset's fields, with ranked local search.

    Search runs here rather than on the server. The endpoint takes a single
    token, so ``--search cfps count`` used to be a usage error and any multi-word
    intent was impossible to express. Fetching the dataset (cached) and ranking
    locally supports AND terms, exclusions, prefixes and typos, and reports why
    each hit matched.
    """
    cache_report = {}
    everything = with_session(args, lambda s: ml.get_datafields(
        s, instrument_type=args.instrument_type, region=args.region,
        delay=args.delay, universe=args.universe, dataset_id=args.dataset,
        search="", field_type=args.field_type, limit=None,
        report=cache_report,
    ))

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

    data = {
        "dataset": args.dataset, "region": args.region, "universe": args.universe,
        "delay": args.delay, "count": len(fields), "scanned": len(everything),
        "type_counts": {
            kind: len([f for f in fields if f.get("type") == kind])
            for kind in ("MATRIX", "VECTOR", "GROUP")
        },
        "fields": fields, "cache": cache_report,
    }
    if search_note:
        data["search"] = search_note
    if args.expressions:
        data["expressions"] = ml.process_datafields(fields)

    return data


def cmd_catalog_find(args, _sink):
    """Search every dataset's fields at once, ranked, with the dataset in each row.

    `catalog fields` needs `--dataset`, so "is there a short-term debt field anywhere?"
    meant paging fourteen datasets by hand - which is what happened four times on
    2026-11-08 while looking for `debt_st`, `debt_lt`, `*_oi` and growth forecasts.
    Every field record is already cached per dataset, so the only thing missing was a
    loop and a ranking rule.

    Ranking: an exact id match first, then coverage (how much of the universe actually
    has the field), then the local search score. Coverage is the tie-break that matters
    in practice - a field that exists for 12% of the universe cannot carry an alpha.
    """
    cache_report = {}
    datasets = with_session(args, lambda s: ml.get_datasets(
        s, instrument_type=args.instrument_type, region=args.region,
        delay=args.delay, universe=args.universe, limit=None, report=cache_report,
    ))
    wanted = [part.strip() for part in (args.dataset or "").split(",") if part.strip()]
    if wanted:
        datasets = [d for d in datasets if d.get("id") in wanted]
    if not datasets:
        raise CliError("INVALID_INPUT", "no datasets for %s/%s/delay %s"
                       % (args.region, args.universe, args.delay))

    rows, scanned = [], 0
    for dataset in datasets:
        dataset_id = dataset.get("id")
        try:
            fields = with_session(args, lambda s, d=dataset_id: ml.get_datafields(
                s, instrument_type=args.instrument_type, region=args.region,
                delay=args.delay, universe=args.universe, dataset_id=d,
                search="", field_type=None, limit=None))
        except Exception:  # noqa: BLE001 - one unreadable dataset must not sink the search
            continue
        scanned += len(fields)
        hits = field_search.search_records(
            fields, args.query, limit=None, fuzzy=args.fuzzy,
            match_all=not args.any_term)
        for hit in hits:
            coverage = hit.get("coverage")
            if args.min_coverage is not None and (coverage or 0) < args.min_coverage:
                continue
            rows.append({
                "id": hit.get("id"), "dataset": dataset_id,
                "category": (dataset.get("category") or {}).get("name")
                if isinstance(dataset.get("category"), dict) else dataset.get("category"),
                "type": hit.get("type"), "coverage": coverage,
                "description": (hit.get("description") or "").strip()[:120],
            })
    lowered = args.query.strip().lower()
    rows.sort(key=lambda row: (0 if (row["id"] or "").lower() == lowered else 1,
                               -(row["coverage"] or 0), row["id"] or ""))
    limited = rows[: args.limit]
    data = {"query": args.query, "region": args.region, "universe": args.universe,
            "delay": args.delay, "datasets_scanned": len(datasets),
            "fields_scanned": scanned, "count": len(rows), "fields": limited,
            "cache": cache_report,
            "ranking": "exact id, then coverage, then id"}
    if not rows:
        data["hint"] = ("nothing matched across %d datasets; try --fuzzy or --any-term"
                        % len(datasets))
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
    """Which catalog fields this book has never touched, by dataset.

    "We only use 13% of the catalog" is true and useless on its own - the interesting
    part is *which* families are untouched and what they let you ask. This counts the
    difference between the cached catalog and every identifier that appears in the
    ledger, per dataset, so the next mining round starts from a measurement instead of
    from a hunch.

    Operator names are excluded by the same rule the block classifier uses: a field is an
    identifier that is not followed by `(`.
    """
    import glob
    import json
    import os
    import re

    import alpha_registry
    import research_paths

    fields = {}
    cache_dir = os.path.join(research_paths.metadata_root(), "datafields")
    for path in glob.glob(os.path.join(cache_dir, "*.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle).get("payload") or []
        except (OSError, ValueError):
            continue
        for row in payload:
            field_id = row.get("id")
            if field_id:
                fields[field_id] = row
    if not fields:
        raise ValueError("no cached catalog fields; run `catalog fields` first")

    token = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    used = set()
    for entry in alpha_registry.load_entries(getattr(args, "registry", None)):
        text = entry.get("expression") or ""
        for match in token.finditer(text):
            if text[match.end():].lstrip().startswith("("):
                continue
            used.add(match.group(0))
    used_fields = used & set(fields)

    by_dataset = {}
    for field_id, row in fields.items():
        dataset = (row.get("dataset") or {}).get("id") or "?"
        bucket = by_dataset.setdefault(dataset, {"dataset": dataset, "fields": 0,
                                                 "used": 0, "unused": []})
        bucket["fields"] += 1
        if field_id in used_fields:
            bucket["used"] += 1
        else:
            bucket["unused"].append({"id": field_id,
                                     "coverage": row.get("coverage"),
                                     "description": (row.get("description") or "")[:70]})

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
            "datasets": rows}


def cmd_catalog_categories(args, _sink):
    """List data categories with field and alpha counts indicating namespace crowding."""
    cache_report = {}
    categories = with_session(args, lambda s: ml.fetch_data_categories(
        s, region=args.region, universe=args.universe, delay=args.delay,
        report=cache_report))
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
