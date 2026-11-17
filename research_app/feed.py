"""Collect bounded, attributable research evidence without executing source claims.

Provider ordering and source coverage remain explicit. Selected evidence is archived
before draft/seen state; all reading and strategy decisions require human review.
"""

import hashlib
import json
import math
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack

import cache_store
import research_paths
import source_io

from . import runtime
from .runtime import CliError
from .validation import cached_field_scope


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")

#: Words too common to make a query specific.
STOPWORDS = frozenset("""
the and for with from that this into over under about across after before between
study evidence effect effects using based does does not its our their their on in of to
""".split())



def _seen_path(args=None):
    explicit = getattr(args, "seen", None)
    if explicit:
        return explicit
    return research_paths.resolve("inspiration", "seen.jsonl")


def _read_jsonl(path):
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except FileNotFoundError:
        return b""


def _seen_keys(content):
    seen = set()
    for line in content.splitlines():
        try:
            record = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        key = record.get("key") if isinstance(record, dict) else None
        if isinstance(key, str) and key:
            seen.add(_canonical_key(key))
        aliases = record.get("aliases") if isinstance(record, dict) else None
        if isinstance(aliases, list):
            seen.update(_canonical_key(alias) for alias in aliases
                        if isinstance(alias, str) and alias.strip())
    return seen


def load_seen(path):
    """Read prior picks; unreadable state must never masquerade as an empty store."""
    return _seen_keys(_read_jsonl(path))


def _append_jsonl_atomic(path, records, previous=None):
    """Preserve prior bytes and commit one flushed JSONL file under the caller's lock."""
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    previous = _read_jsonl(path) if previous is None else previous
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + os.path.basename(path) + ".", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(previous)
            if previous and not previous.endswith(b"\n"):
                handle.write(b"\n")
            for record in records:
                handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _canonical_key(value):
    """Fold DOI spellings while retaining unrecognized historical keys."""
    value = value.strip()
    doi = re.sub(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)", "", value,
                 flags=re.IGNORECASE)
    if re.match(r"^10\.\d+/\S+$", doi, re.IGNORECASE):
        return doi.lower()
    match = re.fullmatch(r"(?:https://openalex\.org/)?[wW](\d+)", value)
    return "W" + match.group(1) if match else value


def _title_key(paper):
    return re.sub(r"[^a-z0-9]+", "", (paper.get("title") or "").lower())


def paper_key(paper):
    """Prefer a canonical identifier; only unidentified works use title folding."""
    doi = (paper.get("doi") or "").strip()
    if doi:
        return _canonical_key(doi).lower()
    identifier = paper.get("openalex_id")
    if identifier:
        return _canonical_key(identifier)
    title = _title_key(paper)
    return title or (paper.get("url") or "").strip().lower()


def _paper_keys(paper):
    keys = [paper_key(paper)]
    identifier = paper.get("openalex_id")
    if isinstance(identifier, str):
        identifier = _canonical_key(identifier)
        if re.fullmatch(r"W\d+", identifier) and identifier not in keys:
            keys.append(identifier)
    return keys


def _paper_identifier(paper):
    """Only validated provider identifiers become executable reading arguments."""
    import paper_search

    for value in (paper.get("openalex_id"), paper.get("doi")):
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            return paper_search.normalize_identifier(_canonical_key(value))
        except (ValueError, paper_search.PaperError):
            continue
    return None


#: An optional hand-written English query on a thread line:
#: ``- [open] **问题** — 证据；下一步：…；query: earnings dispersion analyst coverage``
#: The question text stays in whatever language it was written in; the query is the
#: author's own translation, so nothing is invented on the reader's behalf.
_EXPLICIT_QUERY = re.compile(r"(?:^|[;；]\s*)query\s*[:：]\s*(.+?)\s*$", re.IGNORECASE)


def explicit_query(line):
    """The hand-written ``query:`` on a thread line, or ``None``."""
    match = _EXPLICIT_QUERY.search(line.strip())
    if not match:
        return None
    text = " ".join(match.group(1).split())
    return text or None


def _open_threads(path):
    """Parse a heading and its authored query together in one read."""
    questions = []
    if not os.path.exists(path):
        return questions
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            bullet = re.match(r"^\s*-\s+(.*)$", line)
            if not bullet:
                continue
            text = bullet.group(1).strip()
            tag = re.match(r"^\[(open|closed)\]\s*", text, re.IGNORECASE)
            if tag:
                if tag.group(1).lower() == "closed":
                    continue
                text = text[tag.end():]
            stated = explicit_query(text)
            heading_text = _EXPLICIT_QUERY.sub("", text).strip()
            heading = re.match(r"^\*\*(.+?)\*\*", heading_text)
            if heading:
                question = heading.group(1).strip()
            elif tag or re.search(r"[:：]", heading_text):
                question = re.split(r"\s*[—–]+\s*|[:：]", heading_text, maxsplit=1)[0].strip()
            else:
                continue
            if question:
                questions.append({"question": question, "query": stated})
    return questions


def open_questions(path):
    """Read tagged bold/plain questions and legacy bullet headings."""
    return [row["question"] for row in _open_threads(path)]


_CODE = re.compile(r"^h-?\d+[a-z]?$|^\d+$|^20\d\d$")

#: Platform identifiers - operators and field-name fragments - that never appear in the
#: finance prose a query is looking for. Ordinary words that happen to be field names
#: (eps, news, turnover) stay searchable; they are ordinary English too.
#: On 2026-11-12 the derived queries were ["eps", "rank", "delay", "stale"] and the
#: digest answered with a biofilm paper, a bone-biology paper, photoemission physics and
#: bread staling - which is why a derived query now needs two content words.
TECHNICAL_TOKENS = frozenset("""
backfill decay group_rank neutralization sharesout sortino subindustry
truncation ts ts_backfill ts_delta ts_mean ts_std_dev vec_avg vec_sum winsorize
""".split())

#: A one-word query is a topic, not a question. ``inspiration/topics.md`` exists for
#: those, written by hand and in English; a derived query has to earn two content words.
MIN_QUERY_WORDS = 2


def query_from_question(question, max_words=8):
    """A search string of the distinctive Latin words in a question.

    Chinese headings yield no query: inventing a translation would obscure the
    source question. The digest reports these questions for an explicit query.
    """
    words = [w for w in _TOKEN.findall(question.lower())
             if w not in STOPWORDS and not _CODE.match(w)
             and w not in TECHNICAL_TOKENS]
    if len(words) < MIN_QUERY_WORDS:
        return ""
    return " ".join(words[:max_words])


def topics_file_queries(path):
    """Hand-written English queries, one per line, ``#`` for comments."""
    queries = []
    if not os.path.exists(path):
        return queries
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text and not text.startswith("#"):
                queries.append(text)
    return queries



def score(paper):
    """Legacy citation/year score; provider relevance order determines selection."""
    citations = paper.get("citations") or 0
    year = paper.get("year") or 0
    return citations + max(0, year - 1990) * 3


def _closure_cautions(paper, register):
    """Only an explicit family can satisfy the register's supported match."""
    family = paper.get("family")
    cautions = []
    if not isinstance(family, str) or not family.strip():
        return cautions
    for index, entry in enumerate(register.get("closed") or []):
        if not isinstance(entry, dict):
            continue
        match = entry.get("match")
        if not isinstance(match, dict) or set(match) != {"family"}:
            continue
        if not isinstance(match["family"], str) or match["family"] != family:
            continue
        cautions.append({"source": register["path"], "entry_index": index,
                         "id": entry.get("id"), "match": match,
                         "reason": entry.get("reason"),
                         "scope": "declared family; review the recorded construction"})
    return cautions


def _retrieval_job(value):
    if isinstance(value, dict):
        return dict(value)
    return {"kind": "query", "query": value, "source": "openalex", "source_id": "paper",
            "ordering_basis": "provider relevance"}


def _build_jobs(args, topics, seeds, relation):
    """Validate the whole configuration before creating bounded retrieval jobs."""
    import research_sources
    from market_sources import validate_market

    try:
        depth = getattr(args, "book_depth", 10)
        if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 50:
            raise source_io.SourceError("INVALID_INPUT", "Book depth must be an integer from 1 to 50.")
        sources = research_sources.load_sources(getattr(args, "sources_file", None))
        markets = []
        for value in getattr(args, "market", None) or []:
            if not isinstance(value, str) or ":" not in value:
                raise source_io.SourceError("INVALID_INPUT", "Market must be VENUE:SYMBOL.")
            venue, symbol = value.split(":", 1)
            spec = validate_market(venue, symbol, depth=getattr(args, "book_depth", 10),
                                   feed=getattr(args, "market_feed", "iex"))
            suffix = hashlib.sha256(spec["symbol"].encode("utf-8")).hexdigest()[:16]
            definition = {"id": "market-%s-%s" % (spec["venue"], suffix),
                          "kind": "market", "adapter": "market", **spec}
            if definition not in markets:
                markets.append(definition)
        requested = getattr(args, "source", None) or []
        if requested:
            selected = list(dict.fromkeys(identifier for value in requested
                                         for identifier in (sources if value == "all" else [value])))
        elif getattr(args, "sources_file", None):
            selected = [identifier for identifier in sources
                        if identifier not in research_sources.BUILTIN_SOURCES]
        else:
            selected = ["paper"] if topics or seeds or not markets else []
        if any(identifier not in sources for identifier in selected):
            raise source_io.SourceError("INVALID_INPUT", "Unknown source ID.",
                                        hint="Use a built-in ID or one defined in --sources-file.")
        per_source = []
        for identifier in selected:
            definition = sources[identifier]
            if identifier == "paper":
                paper_jobs = [_retrieval_job(topic) for topic in topics]
                if not topics and not seeds and requested:
                    paper_jobs.append({"kind": "query", "source_id": "paper", "source": "openalex",
                                       "query": None, "_input_required": True})
                per_source.append(paper_jobs)
                continue
            query_based = research_sources.requires_query(definition)
            queries = topics if query_based and topics else [None]
            per_source.append([{
                "kind": "source", "source_id": identifier, "source": identifier,
                "query": query, "ordering_basis": "provider order",
                "descriptor": research_sources.describe_source(definition),
                "_definition": definition,
                **({"_input_required": True} if query_based and not topics else {}),
            } for query in queries])
        # First give every selected source a job, then take the next topic for each.
        jobs = [bucket[index] for index in range(max(map(len, per_source), default=0))
                for bucket in per_source if index < len(bucket)]
        jobs.extend({"kind": "seed", "source_id": "paper", "seed": seed, "relation": relation,
                     "source": "openalex", "query": "%s of %s" % (relation, seed),
                     "ordering_basis": "provider order"} for seed in seeds)
        jobs.extend({"kind": "market", "source_id": definition["id"], "source": definition["id"],
                     "query": None, "market_spec": {key: definition[key] for key in
                                                    ("venue", "symbol", "depth", "feed")},
                     "ordering_basis": "one requested snapshot", "_definition": definition,
                     "descriptor": research_sources.describe_source(definition)}
                    for definition in markets)
        return jobs
    except source_io.SourceError as error:
        raise CliError(error.code, str(error), hint=error.hint) from error


def _public_job(job):
    return {key: value for key, value in job.items() if not key.startswith("_")}


def _search_topics(topics, args):
    """Use one worker pool for all providers; retain deterministic job order."""
    import paper_search
    import research_sources

    def search(value):
        job = _retrieval_job(value)
        definition = job.pop("_definition", None)
        missing_query = job.pop("_input_required", False)
        if missing_query:
            return job, [], {"code": "INPUT_REQUIRED", "hint": "Supply --topic for this source."}
        try:
            if definition is not None:
                found = research_sources.fetch_source(definition, query=job["query"], limit=args.limit,
                                                       timeout=getattr(args, "timeout", 30))
                rows = found.get("items") if isinstance(found, dict) else None
            else:
                options = dict(limit=args.limit, from_year=args.from_year,
                               min_citations=args.min_citations, timeout=getattr(args, "timeout", 30))
                found = (paper_search.related_papers(job["seed"], relation=job["relation"], **options)
                         if job["kind"] == "seed" else paper_search.search_papers(job["query"], **options))
                rows = found.get("papers") if isinstance(found, dict) else found
            if not isinstance(rows, list):
                return job, [], {"code": "INVALID_RESPONSE"}
            if isinstance(found, dict):
                for key in ("source", "seed", "relation", "coverage", "cached", "total_available",
                            "ordering_basis", "collected_at", "as_of", "age_seconds"):
                    if key in found:
                        job[key] = found[key]
                if job["kind"] == "seed":
                    job["query"] = found.get("query") or job["query"]
                job["ordering_basis"] = (job.get("coverage") or {}).get("ordering") or job["ordering_basis"]
            # Older paper caches do not record collection time. Do not label them live.
            job.setdefault("collected_at", None if job.get("cached") else source_io.utc_now())
            job.setdefault("coverage", {"scope": "bounded provider results", "ordering": job["ordering_basis"]})
            if len(rows) > args.limit:
                job["coverage"] = {**job["coverage"], "omitted_by_limit": len(rows) - args.limit}
            return job, rows[:args.limit], None
        except Exception as error:  # noqa: BLE001 - one provider cannot cancel peers
            if isinstance(error, source_io.SourceError):
                return job, [], error.public()
            if isinstance(error, paper_search.PaperError):
                return job, [], {"code": error.code, "hint": error.hint, "type": type(error).__name__}
            return job, [], {"code": "SEARCH_FAILED", "type": type(error).__name__}

    if not topics:
        return []
    outcomes = [None] * len(topics)
    with ThreadPoolExecutor(max_workers=min(getattr(args, "workers", 4), len(topics))) as pool:
        pending = {pool.submit(search, topic): index for index, topic in enumerate(topics)}
        for completed, future in enumerate(as_completed(pending), 1):
            index = pending[future]
            outcomes[index] = future.result()
            status = "failed" if outcomes[index][2] else "complete"
            runtime.note("feed: search %d/%d %s (%d finished)" %
                         (index + 1, len(topics), status, completed))
    return outcomes


def _safe_url(value):
    try:
        return source_io.public_url(value) if value else None
    except (source_io.SourceError, TypeError, ValueError):
        return None


def _item_keys(item):
    if item.get("kind", "paper") == "paper":
        return _paper_keys(item)
    kind, provider = item["kind"], item["provider"]
    identifier = item.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Missing native identity")
    native = "%s:%s:id:%s" % (kind, provider, identifier)
    if kind == "market":
        return [native]  # Endpoint URLs identify feeds, never individual snapshots.
    native = "%s:%s:%s:id:%s" % (kind, provider, item["source_id"], identifier)
    if provider == "local_file":
        # Repeated pasted exports are one source, even under different filenames/config IDs.
        content = {key: item.get(key) for key in
                   ("id", "title", "excerpt", "author", "parent_id", "published_at_raw", "url")}
        content["excerpt"] = item["_evidence"].get("excerpt")
        fingerprint = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        native = "%s:local_file:%s" % (kind, fingerprint)
    url = _safe_url(item.get("url"))
    return ["%s:url:%s" % (kind, url), native] if url else [native]


def _normalize_item(row, job):
    """Keep full evidence separately from the bounded display fields."""
    item = dict(row)
    is_paper = job.get("source_id", "paper") == "paper"
    kind = "paper" if is_paper else item.get("kind")
    if kind not in {"paper", "news", "social", "forum", "trend", "web", "market"}:
        raise ValueError("Unknown evidence kind")
    item["kind"] = kind
    item["provider"] = item.get("provider") or item.get("source") or job["source"]
    if not isinstance(item.get("title"), str) or not isinstance(item["provider"], str):
        raise ValueError("Missing title or provider")
    text = item.get("abstract") if is_paper else item.get("excerpt")
    text = "" if text is None else text
    if not isinstance(text, str):
        raise ValueError("Invalid evidence text")
    if is_paper:
        rank = score(item)
        if not math.isfinite(rank):
            raise ValueError("Invalid paper metadata")
        timestamp = source_io.parse_timestamp(item.get("publication_date") or
                                               (str(item["year"]) if item.get("year") else None))
        item.update(published_at=timestamp["value"], published_at_raw=timestamp["raw"],
                    time_precision=timestamp["precision"], rank=rank,
                    evidence={"basis": "OpenAlex metadata and available abstract",
                              "claim_state": "unverified", "full_text_read": False},
                    engagement=None)
    item.setdefault("collected_at", job.get("collected_at"))
    item.setdefault("published_at", None)
    item.setdefault("published_at_raw", None)
    item.setdefault("time_precision", "unknown")
    item.setdefault("author", None)
    item.setdefault("engagement", None)
    item.setdefault("evidence", {"basis": "provider metadata", "claim_state": "unverified"})
    item.update(source=job["source"], source_id=job.get("source_id", "paper"),
                coverage=job.get("coverage"), cached=job.get("cached", False))
    for field in ("url", "landing_page_url", "pdf_url", "cited_by_api_url"):
        if field in item:
            item[field] = _safe_url(item[field])
    if isinstance(item.get("id"), str) and item["id"].lower().startswith(("http://", "https://")):
        item["id"] = _safe_url(item["id"])
    item["excerpt"] = text
    full = dict(item)
    item.update(excerpt=text[:600], excerpt_chars=len(text), excerpt_truncated=len(text) > 600,
                evidence_path=None, _evidence=full)
    item.pop("raw", None)
    if is_paper:
        item.update(abstract=text[:600], abstract_chars=len(text), abstract_truncated=len(text) > 600,
                    abstract_available=bool(text), reading=row.get("reading"), full_text_read=False,
                    reference_count=len(row["referenced_works"])
                    if isinstance(row.get("referenced_works"), list) else None)
        item.pop("referenced_works", None)
    if kind == "market":
        if not isinstance(item.get("market"), dict):
            raise ValueError("Missing snapshot")
        item["market"] = {key: value for key, value in item["market"].items()
                          if key not in {"bids", "asks", "raw"}}
    item["unknowns"] = ["claim validity", "historical first availability", "backtest applicability"]
    if not item["published_at"]:
        item["unknowns"].append("exact source publication time")
    if item["collected_at"] is None:
        item["unknowns"].append("original collection time")
    return item


def _select_items(outcomes, seen, total, register):
    """Round-robin all jobs without converting popularity into evidence strength."""
    candidates, identities, buckets, reports, repeats = {}, {}, [], [], set()
    for value, rows, error in outcomes:
        job = _retrieval_job(value)
        topic = job.get("query")
        report = {**_public_job(job), "status": "failure" if error else "success" if rows else "empty",
                  "returned": len(rows), "eligible": 0, "seen_before": 0,
                  "invalid": 0, "picked": 0}
        if error:
            report["error"] = error
        reports.append(report)
        bucket, bucket_keys = [], set()
        buckets.append(bucket)
        for provider_rank, row in enumerate(rows, 1):
            try:
                if not isinstance(row, dict):
                    raise ValueError("Invalid item")
                item = _normalize_item(row, job)
                keys = _item_keys(item)
                key = keys[0]
                if not key:
                    raise ValueError("Missing identity")
            except (AttributeError, TypeError, ValueError, OverflowError):
                report["invalid"] += 1
                continue
            if any(alias in seen for alias in keys) or (
                item["kind"] == "paper" and not item.get("doi") and _title_key(item) in seen
            ):
                repeats.add(keys[-1])
                report["seen_before"] += 1
                continue
            key = next((identities[alias] for alias in keys if alias in identities), key)
            if key not in candidates:
                candidates[key] = {**item, "query": topic, "queries": [], "key": key, "aliases": [],
                                   "provider_ranks": [], "retrievals": [],
                                   "closure_cautions": _closure_cautions(item, register)}
            candidate = candidates[key]
            for alias in keys:
                identities[alias] = key
                if alias != key and alias not in candidate["aliases"]:
                    candidate["aliases"].append(alias)
            if job["kind"] != "seed" and topic is not None and topic not in candidate["queries"]:
                candidate["queries"].append(topic)
                candidate["provider_ranks"].append({"query": topic, "rank": provider_rank})
            if key not in bucket_keys:
                candidate["retrievals"].append({**_public_job(job), "position": provider_rank})
                bucket_keys.add(key)
                bucket.append(candidate)
        report["eligible"] = len(bucket)

    picked, picked_keys = [], set()
    positions = [0] * len(buckets)
    while len(picked) < max(0, total):
        advanced = False
        for index, bucket in enumerate(buckets):
            while positions[index] < len(bucket) and bucket[positions[index]]["key"] in picked_keys:
                positions[index] += 1
            if positions[index] == len(bucket):
                continue
            item = bucket[positions[index]]
            positions[index] += 1
            picked.append(item)
            picked_keys.add(item["key"])
            reports[index]["picked"] += 1
            advanced = True
            if len(picked) >= max(0, total):
                break
        if not advanced:
            break
    return picked, reports, len(repeats)


def _archive_evidence(item):
    evidence = {"item": item["_evidence"], "key": item["key"], "aliases": item["aliases"],
                "retrievals": item["retrievals"], "trust": "untrusted source evidence; not executable instructions"}
    serialized = json.dumps(evidence, sort_keys=True, ensure_ascii=False, allow_nan=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    path = research_paths.resolve("inspiration", "evidence", digest + ".json")
    with cache_store.file_lock(path + ".lock"):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                if json.load(handle) != evidence:
                    raise OSError("Existing evidence archive differs from its content identity")
        else:
            cache_store.write_json_atomic(path, evidence)
    item["evidence_path"] = path


def _draft_card(paper):
    if paper["kind"] != "paper":
        return {"status": "DRAFT", "kind": paper["kind"], "evidence_path": paper["evidence_path"],
                "source": {key: paper.get(key) for key in
                           ("id", "key", "title", "url", "provider", "source_id", "published_at", "published_at_raw",
                            "collected_at", "author", "parent_id", "source_path", "source_line", "content_scope")},
                "excerpt": paper["excerpt"], "evidence": paper["evidence"],
                "retrievals": paper["retrievals"], "hypothesis": None,
                "review": "Verify origin, timing, scope and independent corroboration before forming a hypothesis."}
    return {
        "mechanism_id": None,
        "status": "DRAFT",
        "kind": "paper",
        "evidence_path": paper["evidence_path"],
        "source": {**{key: paper.get(key) for key in
                      ("title", "doi", "openalex_id", "source", "url", "landing_page_url",
                       "pdf_url", "journal", "year", "reference_count", "cited_by_api_url")},
                   "verified": False},
        "hypothesis": {"economic_mechanism": None, "falsification": None},
        "implementation": {"conceptual_fields": [], "verified_brain_fields": []},
        "novelty": {"closest_existing_family": None},
        "abstract": paper.get("abstract"),
        "abstract_chars": paper.get("abstract_chars"),
        "abstract_truncated": paper.get("abstract_truncated"),
        "abstract_available": paper.get("abstract_available"),
        "reading": paper.get("reading"),
        "full_text_read": False,
        "queries": paper.get("queries"),
        "retrievals": paper.get("retrievals"),
    }


def _feed_argv(args, topics, seen_path):
    """Exact CLI arguments for a bounded retry using this invocation's research context."""
    argv = ["--research-dir", os.fspath(research_paths.research_dir()), "feed"]
    jobs = [_retrieval_job(topic) for topic in topics]
    for job in jobs:
        if job["kind"] == "market":
            spec = job["market_spec"]
            argv.extend(("--market", "%s:%s" % (spec["venue"], spec["symbol"]),
                         "--book-depth", str(spec["depth"]), "--market-feed", spec["feed"]))
        else:
            argv.extend(("--source", job.get("source_id", "paper")))
            if getattr(args, "sources_file", None):
                argv.extend(("--sources-file", os.path.abspath(args.sources_file)))
            if job["kind"] == "seed":
                argv.extend(("--seed", job["seed"], "--follow", job["relation"]))
            elif job.get("query") is not None:
                argv.extend(("--topic", job["query"]))
    for option, value in (("--limit", args.limit), ("--total", args.total),
                          ("--max-topics", len(topics)), ("--from-year", args.from_year),
                          ("--min-citations", args.min_citations),
                          ("--workers", getattr(args, "workers", 4)),
                          ("--timeout", getattr(args, "timeout", 30)), ("--seen", seen_path)):
        if value is not None:
            argv.extend((option, str(value)))
    for option, attribute in (("--region", "region"), ("--universe", "universe"),
                              ("--delay", "delay"), ("--instrument-type", "instrument_type"),
                              ("--meta-max-age", "meta_max_age")):
        value = getattr(args, attribute, None)
        if value is not None:
            argv.extend((option, str(value)))
    if args.draft_cards:
        argv.extend(("--draft-cards", os.path.abspath(args.draft_cards)))
    for option, attribute in (("--dry-run", "dry_run"), ("--action", "action"), ("--summary", "summary")):
        if getattr(args, attribute, False):
            argv.append(option)
    return argv


# Keywords describe possible reading needs, never account entitlements.
DATA_NEEDS = (
    (("conference call", "transcript", "managerial tone", "tone of"), "call transcripts"),
    (("patent", "innovation and productivity", "citation-weighted"), "patent data"),
    (("short interest", "securities lending", "lending market", "borrow fee"), "lending and short interest"),
    (("corporate bond", "credit spread", "bond return"), "bond prices"),
    (("supply chain", "supplier", "bullwhip", "customer-supplier"), "supply-chain links"),
    (("index reconstitution", "index addition", "index deletion", "indexing"), "index membership history"),
    (("monetary policy", "macro variable", "macroeconomic"), "macro series"),
    (("earnings announcement", "announcement date", "post-earnings"), "announcement dates"),
    (("analyst forecast", "forecast dispersion", "estimate dispersion", "revision"), "analyst estimates"),
    (("accrual", "inventory", "deferred revenue", "earnings quality", "fundamental",
      "r&d", "research and development", "capex", "dividend", "payout"), "fundamentals"),
    (("implied volatility", "option", "skew", "open interest"), "options"),
    (("news", "sentiment", "social media", "attention", "tweet"), "news and sentiment"),
    (("turnover", "volume", "momentum", "reversal", "liquidity", "price"), "prices and volume"),
)


def _has_keyword(text, phrase):
    """Match a reading cue as words, allowing a simple plural at the end.

    Substrings made technology *adoption* look like *option* research and
    *steps* look like *EPS*. These hints must not attach unrelated ledger work.
    """
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?:s|es)?(?!\w)", text) is not None


def data_fit(paper, field_scope=None):
    """Separate inferred reading needs from explicitly named field availability."""
    text = ("%s %s" % (paper.get("title") or "", paper.get("abstract") or paper.get("excerpt") or "")).lower()
    needs = [{"need": need, "state": "inferred", "verdict": "unknown", "basis": "title_evidence_keywords"}
             for phrases, need in DATA_NEEDS if any(_has_keyword(text, phrase) for phrase in phrases)]
    supplied = paper.get("required_fields")
    valid = supplied is None or (isinstance(supplied, list) and all(isinstance(item, str) and item.strip() for item in supplied))
    required = sorted({item.strip() for item in supplied}) if valid and supplied else []
    scope = field_scope or {}
    evidence = scope.get("evidence") or {"state": "unknown", "reasons": ["no_scoped_catalog"]}
    known = scope.get("known_fields")
    verified = evidence.get("state") == "verified" and known is not None
    available = sorted(set(required) & set(known)) if verified else []
    missing = sorted(set(required) - set(known)) if verified else []
    verdict = "unknown"
    if required and verified:
        verdict = "required_fields_missing" if missing else "fields_available"
    return {"verdict": verdict, "needs": needs, "catalog": evidence,
            "required_fields": required, "available_fields": available,
            "missing_fields": missing, "unknown_fields": required if not verified else [],
            "requirement_state": "declared" if required else "invalid" if not valid else "unspecified",
            "why": ("Only explicitly required field IDs can be checked against a complete current scoped catalog. "
                    "Keywords suggest reading needs. Available fields do not prove a valid signal; "
                    "missing direct inputs do not rule out a separately justified proxy.")}


def _next_actions(args, reports, unresolved, deferred, items, seen_path):
    """Executable retries target exactly one job; source text never supplies argv."""
    actions = []
    for report in [row for row in reports if row["status"] == "failure"][:3]:
        actions.append({"kind": "supply_query" if report["error"]["code"] == "INPUT_REQUIRED" else "retry_search",
                        "query": report.get("query"),
                        "retrieval": {key: report[key] for key in
                                      ("kind", "source", "source_id", "seed", "relation", "market_spec") if key in report},
                        "error": report["error"], "argv": _feed_argv(args, [report], seen_path)})
    for question in unresolved[:3]:
        actions.append({"kind": "supply_query", **question,
                        "instruction": "Add '; query:' followed by your own English search terms to this open-thread line."})
    for value in deferred[:3]:
        job = _retrieval_job(value)
        actions.append({"kind": "search_deferred", "queries": [job["query"]] if job.get("query") and job["kind"] != "seed" else [],
                        "retrievals": [_public_job(job)], "remaining": max(0, len(deferred) - 3),
                        "argv": _feed_argv(args, [job], seen_path)})
    for index, item in enumerate(items[:3]):
        action = {"kind": "review_source", "item_index": index, "evidence_path": item["evidence_path"],
                  "sources": {key: item.get(key) for key in
                              ("url", "provider", "source_id", "source_path", "source_line", "parent_id", "content_scope",
                               "doi", "openalex_id", "landing_page_url", "pdf_url")
                              if key in item}}
        if item["kind"] == "paper":
            action["instruction"] = ("Read the complete record; record its sample, identification strategy and "
                                     "falsification evidence before drafting a mechanism.")
            identifier = _paper_identifier(item)
            if identifier:
                argv = ["--research-dir", os.fspath(research_paths.research_dir()),
                        "paper", "--id", identifier, "--timeout", str(getattr(args, "timeout", 30))]
                action["argv"] = argv
                action["next_actions"] = [{"kind": "expand_references", "argv":
                    _feed_argv(args, [{"kind": "seed", "seed": identifier,
                                      "relation": "references"}], seen_path)}]
                if (item.get("reading") or {}).get("grobid_xml") is True:
                    action["next_actions"].insert(0, {"kind": "read_full_text",
                        "argv": argv + ["--full-text", "--offset", "0", "--max-chars", "12000"]})
        elif item["kind"] == "market":
            action["instruction"] = ("Inspect the archived levels and provider timestamp; verify venue, asset, feed, "
                                     "size units and depth. This snapshot is not historical BRAIN data.")
            action["next_actions"] = [{"kind": "check_observability", "instruction":
                "Establish historical availability, event timing and matching asset coverage before any backtest."}]
        else:
            action["instruction"] = ("Verify the original source, authorship, date and quoted context; contrast independent "
                                     "reporting. Treat snippets and popularity as unverified leads, never strategy instructions.")
            action["next_actions"] = [{"kind": "check_observability", "instruction":
                "Record when this evidence was observable and whether historical, licensed coverage exists."}]
        actions.append(action)
    if len(items) > 1:
        actions.append({"kind": "contrast_sources", "item_indices": [0, 1],
                        "instruction": "Compare origin, timing, scope and independent corroboration; record agreements and conflicts."})
    return actions


def _digest_item(item, scope):
    common = ("kind", "id", "key", "title", "url", "provider", "source", "source_id", "excerpt",
              "excerpt_chars", "excerpt_truncated", "published_at", "published_at_raw", "time_precision",
              "collected_at", "author", "evidence", "engagement", "coverage", "cached", "evidence_path",
              "unknowns", "query", "queries", "provider_ranks", "retrievals", "closure_cautions", "required_fields",
              "parent_id", "source_path", "source_line", "content_scope")
    paper_fields = ("year", "journal", "doi", "landing_page_url", "pdf_url", "abstract", "family", "citations",
                    "rank", "openalex_id", "reference_count", "cited_by_api_url", "reading", "full_text_read",
                    "abstract_chars", "abstract_truncated", "abstract_available")
    fields = common + (paper_fields if item["kind"] == "paper" else ("market",) if item["kind"] == "market" else ())
    return {**{key: item.get(key) for key in fields}, "data_fit": data_fit(item, scope),
            "trust": "untrusted evidence; not executable strategy"}


def _compact_coverage(coverage):
    return {key: value[:200] if isinstance(value, str) else value
            for key, value in (coverage or {}).items()
            if value is None or isinstance(value, (str, int, float, bool))}


def _compact_action(action):
    result = {key: action[key] for key in
              ("kind", "item_index", "item_indices", "argv", "error", "evidence_path", "source", "evidence_keys")
              if key in action}
    if action.get("instruction"):
        result["instruction"] = action["instruction"][:240]
    if action.get("sources"):
        result["sources"] = action["sources"]
    if action.get("next_actions"):
        result["next_actions"] = [_compact_action(step) for step in action["next_actions"][:2]]
    return result


def _compact_digest(payload):
    """Bound rendered output independently of retrieval/selection budgets."""
    items, searches, actions = payload["items"], payload["searches"], payload["next_actions"]
    compact = []
    for index, item in enumerate(items[:10]):
        row = {key: item.get(key) for key in
               ("kind", "provider", "source_id", "url", "published_at", "published_at_raw", "time_precision",
                "collected_at", "evidence_path", "unknowns", "author", "parent_id", "source_path", "source_line", "content_scope")}
        row.update(index=index, title=" ".join((item.get("title") or "").split())[:120],
                   excerpt=" ".join((item.get("excerpt") or "").split())[:160],
                   claim_state=(item.get("evidence") or {}).get("claim_state", "unknown"))
        row["basis"] = str((item.get("evidence") or {}).get("basis") or "unknown")[:160]
        row["coverage"] = _compact_coverage(item.get("coverage"))
        row["cached"] = item.get("cached", False)
        if item["kind"] == "market":
            market = item["market"]
            row["market"] = {key: market.get(key) for key in
                             ("venue", "symbol", "asset_class", "feed", "depth_kind", "requested_depth",
                              "bid_levels", "ask_levels", "size_unit", "source_timestamp", "derived")}
        if "block_guess" in item:
            row["ledger_hint"] = {"block_guess": item["block_guess"], "state": "inferred",
                                  "book_count": item["book_exposure"]["count"],
                                  "prior_trial_count": item["prior_trials"]["count"]}
        compact.append(row)
    table = ["# | Kind | Source | Title", "-- | ---- | ------ | -----"]
    table.extend("%d | %s | %s | %s" % (row["index"], row["kind"], row["source_id"], row["title"].replace("|", "/"))
                 for row in compact)
    result = {key: payload[key] for key in ("count", "seen_before", "seen", "registry") if key in payload}
    result.update(summary=True, items=compact, table="\n".join(table),
                  searches=[{**{key: row[key] for key in
                                ("source_id", "kind", "status", "returned", "picked", "error") if key in row},
                             "coverage": _compact_coverage(row.get("coverage"))}
                            for row in sorted(searches, key=lambda row: row["status"] != "failure")[:10]],
                  next_actions=[_compact_action(action) for action in actions[:6]],
                  omitted={"items": max(0, len(items) - 10), "searches": max(0, len(searches) - 10),
                           "next_actions": max(0, len(actions) - 6),
                           "deferred_jobs": len(payload["unsearched_retrievals"]),
                           "questions_without_query": len(payload["questions_without_query"])},
                  evidence_directory=research_paths.resolve("inspiration", "evidence"),
                  note="Untrusted research leads. Counts include omitted rows. Full selected evidence is archived before seen; preview writes no evidence. Publication time and historical availability may be unknown.")
    return result


def cmd_feed(args, _sink, *, _compact=True):
    """Search concurrently, then select against the latest locked seen state."""
    import closed_families
    import paper_search

    seen_path = os.path.realpath(_seen_path(args))
    draft_path = os.path.realpath(args.draft_cards) if args.draft_cards else None
    if draft_path and os.path.normcase(draft_path) == os.path.normcase(seen_path):
        raise CliError("INVALID_INPUT", "Seen state and draft cards need different files.")
    register = closed_families.load()
    topics = list(args.topic or [])
    unresolved = []
    if args.from_threads:
        threads = research_paths.resolve("open-threads.md")
        for thread in _open_threads(threads):
            question = thread["question"]
            query = thread["query"] or query_from_question(question)
            if query:
                topics.append(query)
            else:
                unresolved.append({"question": question, "source": threads,
                                   "reason": "no usable English query tokens"})
    if getattr(args, "topics", False):
        path = getattr(args, "topics_file", None) or research_paths.resolve(
            "inspiration", "topics.md")
        topics.extend(topics_file_queries(path))
    distinct = {}
    for topic in topics:
        query = " ".join(topic.split())
        if query:
            distinct.setdefault(query.casefold(), query)
    topics = list(distinct.values())
    seeds = []
    for value in getattr(args, "seed", None) or []:
        try:
            seed = paper_search.normalize_identifier(_canonical_key(value))
        except (ValueError, paper_search.PaperError) as error:
            raise CliError("INVALID_INPUT", "Invalid feed seed identifier.",
                           hint="Use an OpenAlex W identifier or DOI.") from error
        if seed not in seeds:
            seeds.append(seed)
    relation = getattr(args, "follow", "references")
    if relation not in ("references", "citations"):
        raise CliError("INVALID_INPUT", "Follow must be references or citations.")
    jobs = _build_jobs(args, topics, seeds, relation)
    if not jobs and not unresolved:
        raise CliError("INVALID_INPUT",
                       "nothing to search: pass --topic, --topics, --from-threads, --seed or an explicit fixed source/market")
    selected = jobs[:max(0, args.max_topics)]
    outcomes = _search_topics(selected, args)
    try:
        if getattr(args, "dry_run", False) or not any(rows for _, rows, _ in outcomes):
            picked, reports, repeats = _select_items(outcomes, load_seen(seen_path), args.total, register)
        else:
            # All provider I/O is finished. Stable lock ordering also protects a shared
            # draft destination used with different seen stores.
            locks = {os.path.normcase(path) + ".lock" for path in (seen_path, draft_path) if path}
            with ExitStack() as stack:
                for path in sorted(locks):
                    stack.enter_context(cache_store.file_lock(path))
                previous = _read_jsonl(seen_path)
                picked, reports, repeats = _select_items(
                    outcomes, _seen_keys(previous), args.total, register)
                if picked:
                    # Every immutable evidence file must exist before either mutable state file.
                    for item in picked:
                        _archive_evidence(item)
                    if draft_path:
                        _append_jsonl_atomic(draft_path, (_draft_card(paper) for paper in picked))
                    _append_jsonl_atomic(seen_path, (
                        {"key": paper["key"], "title": paper.get("title"), "query": paper.get("query"),
                         "kind": paper["kind"], "evidence_path": paper["evidence_path"],
                         **({"aliases": paper["aliases"]} if paper["aliases"] else {})}
                        for paper in picked), previous=previous)
    except (OSError, ValueError, TypeError, cache_store.LockTimeout) as error:
        raise CliError("INVALID_INPUT", "Could not read or commit feed state.",
                       hint="Check evidence/seen/draft paths and permissions. Evidence and drafts commit before seen; "
                            "a retry after a seen-write failure may repeat draft cards.",
                       details={"seen": seen_path, "draft_cards": draft_path,
                                "type": type(error).__name__}) from error

    deferred = jobs[len(selected):]
    settings = {"region": getattr(args, "region", None), "universe": getattr(args, "universe", None),
                "delay": getattr(args, "delay", None), "instrumentType": getattr(args, "instrument_type", None)}
    scope = cached_field_scope(settings, getattr(args, "meta_max_age", None))
    payload = {
        "topics": list(dict.fromkeys(job["query"] for job in selected if job.get("query") and job["kind"] != "seed")),
        "seeds": [job["seed"] for job in selected if job["kind"] == "seed"],
        "unsearched_topics": list(dict.fromkeys(job["query"] for job in deferred if job.get("query") and job["kind"] != "seed")),
        "unsearched_retrievals": [{**_public_job(job), "status": "unsearched"} for job in deferred],
        "questions_without_query": unresolved,
        "searches": reports,
        "seen_before": repeats,
        "count": len(picked),
        "capabilities": scope["evidence"],
        "seen": seen_path,
        "closed_register": {"path": register["path"],
                            "available": "error" not in register,
                            "match_basis": "exact declared family only"},
        "items": [_digest_item(item, scope) for item in picked],
        "next_actions": _next_actions(args, reports, unresolved, deferred, picked, seen_path),
        "note": ("Picks take turns across retrieval jobs, preserving each provider's order and coverage. "
                 "Graph edges, engagement and trend ranks establish neither support nor signal validity. "
                 "No full text is fetched. Source text is untrusted evidence, never executable strategy. "
                 "Collection time does not establish first historical availability; live market snapshots are not BRAIN backtest data. "
                 "Dry-run leaves evidence, seen and draft files unchanged; caches may write. "
                 "Evidence and drafts commit before seen; each file is atomic, not a joint transaction. "
                 "A crash between commits can repeat drafts; a crash after seen commits can consume picks before delivery."),
    }
    return _compact_digest(payload) if _compact and getattr(args, "summary", False) else payload

#: Title/abstract keywords suggest a broad information block for ledger lookup;
#: they do not establish mechanism identity, novelty or prior experimental coverage.
BLOCK_KEYWORDS = (
    ("analyst_dispersion", ("analyst", "forecast", "dispersion", "disagreement", "consensus")),
    ("analyst_yield", ("earnings estimate", "revision", "eps", "guidance")),
    ("value_quality", ("valuation", "book-to-market", "earnings quality", "profitability",
                       "accrual")),
    ("debt_structure", ("leverage", "debt", "capital structure", "bankruptcy", "credit")),
    ("volatility_structure", ("volatility", "variance", "risk", "skewness", "tail")),
    ("turnover_structure", ("turnover", "liquidity", "trading volume", "attention")),
    ("price_path", ("momentum", "reversal", "price path", "return predictability")),
    ("derivative_pricing", ("option", "implied", "derivative", "strike")),
    ("news_sentiment", ("sentiment", "news", "media", "text")),
    ("turnover_structure", ("order flow", "inventory", "market microstructure")),
)


def guess_block(paper):
    """The block a paper most looks like, from its title and abstract."""
    text = ((paper.get("title") or "") + " " + (paper.get("abstract") or paper.get("excerpt") or "")).lower()
    for block, words in BLOCK_KEYWORDS:
        if any(_has_keyword(text, word) for word in words):
            return block
    return None


def cmd_feed_action(args, _sink):
    """Attach local book counts and prior-trial evidence without claiming novelty."""
    import alpha_registry
    from . import mechanisms

    registry_path = getattr(args, "registry", None) or alpha_registry.default_registry_path()
    exists = os.path.exists(registry_path)
    entries = alpha_registry.load_entries(registry_path)
    by_block = {}
    for entry in entries:
        block = mechanisms.block_of(entry.get("expression") or "")
        if block != "other":
            by_block.setdefault(block, []).append(entry)

    payload = cmd_feed(args, _sink, _compact=False)
    rows = []
    for paper in payload["items"]:
        block = guess_block(paper)
        known = exists and block is not None
        matches = by_block.get(block, []) if known else []
        book = [entry for entry in matches
                if str(entry.get("status") or "").upper() in ("ACTIVE", "SUBMITTED")]
        trials = [entry for entry in matches
                  if str(entry.get("status") or "").upper() in
                  alpha_registry.FINISHED_STATUSES | {"ACTIVE", "SUBMITTED"}]
        rows.append({
            **paper,
            "block_guess": block,
            "book_exposure": {"source": registry_path, "count": len(book) if known else None,
                              "basis": "local ACTIVE/SUBMITTED entries in guessed block"},
            "prior_trials": {"source": registry_path,
                             "count": len(trials) if known else None,
                             "ledger_entries": len(matches) if known else None,
                             "evidence_shown": min(5, len(trials)) if known else None,
                             "evidence": [{
                                 **{k: entry.get(k) for k in
                                    ("key", "alpha_id", "status", "verdict", "family", "batch")},
                                 "mechanism": (entry.get("mechanism") or "")[:160] or None,
                                 "note": (entry.get("note") or "")[:240] or None,
                             } for entry in trials[-5:]]},
            "next": ("Read the source and prior evidence, state the mechanism and closest "
                     "family, then choose a discriminating control and falsification condition."),
        })
    payload["items"] = rows
    payload["registry"] = {"path": registry_path, "exists": exists,
                           "entries": len(entries)}
    for index, paper in enumerate(rows):
        evidence = paper["prior_trials"]["evidence"]
        if evidence:
            payload["next_actions"].append({
                "kind": "contrast_prior_evidence", "item_index": index,
                "source": registry_path, "evidence_keys": [entry.get("key") for entry in evidence],
                "instruction": "Read these prior trials and compare their recorded mechanisms, "
                               "controls and failure conditions with this source; a shared block "
                               "does not establish the same mechanism.",
            })
            break
    payload["note"] += (" Digest order is preserved. Block guesses guide local ledger lookup; "
                        "book exposure and prior trials are separate evidence. Missing sources "
                        "or unknown blocks remain unknown. Zero local matches do not establish "
                        "an unexplored mechanism or a marginal contribution to the book.")
    return _compact_digest(payload) if getattr(args, "summary", False) else payload
