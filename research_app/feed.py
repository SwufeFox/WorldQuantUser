"""Turn open questions into an attributable, balanced literature digest.

Search results retain their evidence and query provenance. Local closure decisions
are cautions only when their declared match applies; they never suppress papers.
Draft cards still require a deliberate mechanism and falsification review.
"""

import json
import math
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack

import cache_store
import research_paths

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
            seen.add(key)
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


def paper_key(paper):
    """A key that survives punctuation drift between mirror records.

    The same paper came back twice with titles differing only in leading dots
    ("… and the Cross-Section" vs ". . . and the Cross-Section"), so the title is folded
    to letters and digits before it is used as a key.
    """
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        return doi
    title = re.sub(r"[^a-z0-9]+", "", (paper.get("title") or "").lower())
    return title or (paper.get("url") or "").strip().lower()


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
    """Rank what to read first: recent and cited, with a floor for relevance by year."""
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


def _search_topics(topics, args):
    """Overlap provider I/O and keep outcomes in input order, without shared sessions."""
    import paper_search

    def search(topic):
        try:
            found = paper_search.search_papers(
                topic, limit=args.limit, from_year=args.from_year,
                min_citations=args.min_citations, timeout=getattr(args, "timeout", 30))
            rows = found.get("papers") if isinstance(found, dict) else found
            if not isinstance(rows, list):
                return topic, [], {"code": "INVALID_RESPONSE"}
            return topic, rows, None
        except Exception as error:  # noqa: BLE001 - one provider failure must not cancel peers
            # Messages may contain request URLs, credentials or response bodies.
            return topic, [], {"code": "SEARCH_FAILED", "type": type(error).__name__}

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


def _select_papers(outcomes, seen, total, register):
    candidates, buckets, reports, repeats = {}, [], [], set()
    for topic, rows, error in outcomes:
        report = {"query": topic, "status": "failure" if error else "success" if rows else "empty",
                  "returned": len(rows), "eligible": 0, "seen_before": 0,
                  "invalid": 0, "picked": 0}
        if error:
            report["error"] = error
        reports.append(report)
        bucket = []
        buckets.append(bucket)
        bucket_keys = set()
        for paper in rows:
            if not isinstance(paper, dict):
                report["invalid"] += 1
                continue
            try:
                key = paper_key(paper)
                rank = score(paper)
                abstract = paper.get("abstract") or ""
                valid = bool(key) and math.isfinite(rank) and isinstance(abstract, str)
            except (AttributeError, TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                report["invalid"] += 1
                continue
            if key in seen:
                repeats.add(key)
                report["seen_before"] += 1
                continue
            if key not in candidates:
                candidates[key] = {**paper, "query": topic, "queries": [], "key": key,
                                   "rank": rank, "abstract": abstract[:600],
                                   "closure_cautions": _closure_cautions(paper, register)}
            candidate = candidates[key]
            if topic not in candidate["queries"]:
                candidate["queries"].append(topic)
            if key not in bucket_keys:
                bucket_keys.add(key)
                bucket.append(candidate)
        bucket.sort(key=lambda row: -row["rank"])
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
            paper = bucket[positions[index]]
            positions[index] += 1
            picked.append(paper)
            picked_keys.add(paper["key"])
            reports[index]["picked"] += 1
            advanced = True
            if len(picked) >= max(0, total):
                break
        if not advanced:
            break
    return picked, reports, len(repeats)


def _draft_card(paper):
    return {
        "mechanism_id": None,
        "status": "DRAFT",
        "source": {**{key: paper.get(key) for key in
                      ("title", "doi", "url", "landing_page_url", "pdf_url", "journal", "year")},
                   "verified": False},
        "hypothesis": {"economic_mechanism": None, "falsification": None},
        "implementation": {"conceptual_fields": [], "verified_brain_fields": []},
        "novelty": {"closest_existing_family": None},
        "abstract": paper.get("abstract"),
    }


def _feed_argv(args, topics, seen_path):
    """Exact CLI arguments for a bounded retry using this invocation's research context."""
    argv = ["--research-dir", os.fspath(research_paths.research_dir()), "feed"]
    for topic in topics:
        argv.extend(("--topic", topic))
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
    for option, attribute in (("--dry-run", "dry_run"), ("--action", "action")):
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


def data_fit(paper, field_scope=None):
    """Separate inferred reading needs from explicitly named field availability."""
    text = ("%s %s" % (paper.get("title") or "", paper.get("abstract") or "")).lower()
    needs = [{"need": need, "state": "inferred", "verdict": "unknown", "basis": "title_abstract_keywords"}
             for phrases, need in DATA_NEEDS if any(phrase in text for phrase in phrases)]
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


def _next_actions(args, reports, unresolved, deferred, papers, seen_path):
    """At most eleven evidence-based actions; the action view may add one ledger contrast."""
    actions = []
    for report in [row for row in reports if row["status"] == "failure"][:3]:
        actions.append({"kind": "retry_search", "query": report["query"],
                        "error": report["error"],
                        "argv": _feed_argv(args, [report["query"]], seen_path)})
    for question in unresolved[:3]:
        actions.append({"kind": "supply_query", **question,
                        "instruction": "Add '; query:' followed by your own English search "
                                       "terms to this open-thread line."})
    if deferred:
        actions.append({"kind": "search_deferred", "queries": deferred[:3],
                        "remaining": max(0, len(deferred) - 3),
                        "argv": _feed_argv(args, deferred[:3], seen_path)})
    for index, paper in enumerate(papers[:3]):
        actions.append({"kind": "review_source", "paper_index": index,
                        "sources": {key: paper.get(key) for key in
                                    ("doi", "url", "landing_page_url", "pdf_url")},
                        "instruction": "Read the source; record its sample, identification "
                                       "strategy and falsification evidence before drafting a mechanism."})
    if len(papers) > 1:
        actions.append({"kind": "contrast_sources", "paper_indices": [0, 1],
                        "instruction": "Compare these sources' samples, identification and "
                                       "falsification evidence; record agreements and conflicts."})
    return actions


def cmd_feed(args, _sink):
    """Search concurrently, then select against the latest locked seen state."""
    import closed_families

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
    if not topics and not unresolved:
        raise CliError("INVALID_INPUT",
                       "nothing to search: pass --topic, --topics or --from-threads")
    selected = topics[:max(0, args.max_topics)]
    outcomes = _search_topics(selected, args)
    try:
        if getattr(args, "dry_run", False) or not any(rows for _, rows, _ in outcomes):
            picked, reports, repeats = _select_papers(outcomes, load_seen(seen_path), args.total, register)
        else:
            # All provider I/O is finished. Stable lock ordering also protects a shared
            # draft destination used with different seen stores.
            locks = {os.path.normcase(path) + ".lock" for path in (seen_path, draft_path) if path}
            with ExitStack() as stack:
                for path in sorted(locks):
                    stack.enter_context(cache_store.file_lock(path))
                previous = _read_jsonl(seen_path)
                picked, reports, repeats = _select_papers(
                    outcomes, _seen_keys(previous), args.total, register)
                if picked:
                    # A failed draft write must leave every unseen pick available.
                    if draft_path:
                        _append_jsonl_atomic(draft_path, (_draft_card(paper) for paper in picked))
                    _append_jsonl_atomic(seen_path, (
                        {"key": paper["key"], "title": paper.get("title"), "query": paper.get("query")}
                        for paper in picked), previous=previous)
    except (OSError, cache_store.LockTimeout) as error:
        raise CliError("INVALID_INPUT", "Could not read or commit feed state.",
                       hint="Check seen/draft paths and permissions. Drafts commit before seen; "
                            "a retry after a seen-write failure may repeat draft cards.",
                       details={"seen": seen_path, "draft_cards": draft_path,
                                "type": type(error).__name__}) from error

    deferred = topics[len(selected):]
    settings = {"region": getattr(args, "region", None), "universe": getattr(args, "universe", None),
                "delay": getattr(args, "delay", None), "instrumentType": getattr(args, "instrument_type", None)}
    scope = cached_field_scope(settings, getattr(args, "meta_max_age", None))
    return {
        "topics": selected,
        "unsearched_topics": deferred,
        "questions_without_query": unresolved,
        "searches": reports,
        "seen_before": repeats,
        "count": len(picked),
        "capabilities": scope["evidence"],
        "seen": seen_path,
        "closed_register": {"path": register["path"],
                            "available": "error" not in register,
                            "match_basis": "exact declared family only"},
        "papers": [{**{k: paper.get(k) for k in
                     ("title", "year", "journal", "doi", "url", "landing_page_url", "pdf_url",
                      "abstract", "family", "citations", "query", "queries", "rank", "closure_cautions", "required_fields")},
                    "data_fit": data_fit(paper, scope)}
                   for paper in picked],
        "next_actions": _next_actions(args, reports, unresolved, deferred, picked, seen_path),
        "note": ("Picks take turns across queries, ranked within each query. Closure "
                 "cautions require a declared family; no title or relevance filter applies. "
                 "Dry-run leaves seen and draft files unchanged; search caching may write. "
                 "Drafts require a falsification condition and closest family before promotion. "
                 "Drafts commit before seen; each file is atomic, not a joint transaction. "
                 "A crash between commits can repeat drafts; a crash after seen commits can "
                 "consume picks before delivery. Next actions are bounded suggestions for review."),
    }

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
    text = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).lower()
    for block, words in BLOCK_KEYWORDS:
        if any(word in text for word in words):
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

    payload = cmd_feed(args, _sink)
    rows = []
    for paper in payload["papers"]:
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
    payload["papers"] = rows
    payload["registry"] = {"path": registry_path, "exists": exists,
                           "entries": len(entries)}
    for index, paper in enumerate(rows):
        evidence = paper["prior_trials"]["evidence"]
        if evidence:
            payload["next_actions"].append({
                "kind": "contrast_prior_evidence", "paper_index": index,
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
    return payload
