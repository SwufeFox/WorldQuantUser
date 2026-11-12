"""Turn open questions into an attributable, balanced literature digest.

Search results retain their evidence and query provenance. Local closure decisions
are cautions only when their declared match applies; they never suppress papers.
Draft cards still require a deliberate mechanism and falsification review.
"""

import json
import os
import re

import research_paths

from .runtime import CliError


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


def load_seen(path):
    """DOIs and titles already surfaced, so a daily run is incremental."""
    seen = set()
    if not os.path.exists(path):
        return seen
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                key = record.get("key")
                if key:
                    seen.add(key)
    except OSError:
        return seen
    return seen


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


def open_questions(path):
    """Read tagged bold/plain questions and legacy bullet headings."""
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
            heading = re.match(r"^\*\*(.+?)\*\*", text)
            if heading:
                question = heading.group(1).strip()
            elif tag or re.search(r"[:：]", text):
                question = re.split(r"\s*[—–]+\s*|[:：]", text, maxsplit=1)[0].strip()
            else:
                continue
            if question:
                questions.append(question)
    return questions


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


def cmd_feed(args, _sink):
    """Search every distinct selected query, then take ranked turns per topic."""
    import closed_families
    import paper_search

    seen_path = _seen_path(args)
    seen = load_seen(seen_path)
    register = closed_families.load()
    topics = list(args.topic or [])
    unresolved = []
    if args.from_threads:
        threads = research_paths.resolve("open-threads.md")
        explicit = {}
        if os.path.exists(threads):
            with open(threads, encoding="utf-8") as handle:
                for line in handle:
                    match = re.match(r"^\s*-\s+(.*)$", line)
                    if not match:
                        continue
                    heading = re.match(r"^\[(open|closed)\]\s*\*\*(.+?)\*\*", match.group(1),
                                       re.IGNORECASE)
                    if heading and heading.group(1).lower() == "open":
                        stated = explicit_query(match.group(1))
                        if stated:
                            explicit[heading.group(2).strip()] = stated
        for question in open_questions(threads):
            query = explicit.get(question) or query_from_question(question)
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
    candidates, buckets, reports, repeats = {}, [], [], set()
    for topic in selected:
        report = {"query": topic, "status": "failure", "returned": 0,
                  "eligible": 0, "seen_before": 0, "invalid": 0, "picked": 0}
        reports.append(report)
        bucket = []
        buckets.append(bucket)
        try:
            options = {"limit": args.limit, "from_year": args.from_year,
                       "min_citations": args.min_citations}
            found = paper_search.search_papers(topic, **options)
            rows = found.get("papers") if isinstance(found, dict) else found
            if not isinstance(rows, list):
                report["error"] = {"code": "INVALID_RESPONSE"}
                continue
        except Exception as error:  # noqa: BLE001 - retain other queries on failure
            # Exception messages may contain request URLs, credentials or response bodies.
            report["error"] = {"code": "SEARCH_FAILED", "type": type(error).__name__}
            continue
        report["returned"] = len(rows)
        report["status"] = "success" if rows else "empty"
        bucket_keys = set()
        for paper in rows:
            if not isinstance(paper, dict):
                report["invalid"] += 1
                continue
            key = paper_key(paper)
            if not key:
                report["invalid"] += 1
                continue
            if key in seen:
                repeats.add(key)
                report["seen_before"] += 1
                continue
            if key not in candidates:
                candidates[key] = {**paper, "query": topic, "queries": [], "key": key,
                                   "rank": score(paper),
                                   "abstract": (paper.get("abstract") or "")[:600],
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
    while len(picked) < max(0, args.total):
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
            if len(picked) >= max(0, args.total):
                break
        if not advanced:
            break

    if picked and not getattr(args, "dry_run", False):
        os.makedirs(os.path.dirname(seen_path) or ".", exist_ok=True)
        with open(seen_path, "a", encoding="utf-8") as handle:
            for paper in picked:
                handle.write(json.dumps({"key": paper["key"],
                                         "title": paper.get("title"),
                                         "query": paper.get("query")},
                                        ensure_ascii=False) + "\n")
        if args.draft_cards:
            os.makedirs(os.path.dirname(args.draft_cards) or ".", exist_ok=True)
            with open(args.draft_cards, "a", encoding="utf-8") as handle:
                for paper in picked:
                    handle.write(json.dumps({
                        "mechanism_id": None,
                        "status": "DRAFT",
                        "source": {"title": paper.get("title"), "doi": paper.get("doi"),
                                   "url": paper.get("url"), "journal": paper.get("journal"),
                                   "year": paper.get("year"), "verified": False},
                        "hypothesis": {"economic_mechanism": None, "falsification": None},
                        "implementation": {"conceptual_fields": [],
                                           "verified_brain_fields": []},
                        "novelty": {"closest_existing_family": None},
                        "abstract": paper.get("abstract"),
                    }, ensure_ascii=False) + "\n")

    return {
        "topics": selected,
        "unsearched_topics": topics[len(selected):],
        "questions_without_query": unresolved,
        "searches": reports,
        "seen_before": len(repeats),
        "count": len(picked),
        "seen": seen_path,
        "closed_register": {"path": register["path"],
                            "available": "error" not in register,
                            "match_basis": "exact declared family only"},
        "papers": [{k: paper.get(k) for k in
                    ("title", "year", "journal", "doi", "url", "abstract", "family",
                     "citations", "query", "queries", "rank", "closure_cautions")}
                   for paper in picked],
        "note": ("Picks take turns across queries, ranked within each query. Closure "
                 "cautions require a declared family; no title or relevance filter applies. "
                 "Dry-run leaves seen and draft files unchanged; search caching may write. "
                 "Drafts require a falsification condition and closest family before promotion."),
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
    payload["note"] = ("Digest order is preserved. Block guesses guide local ledger lookup; "
                       "book exposure and prior trials are separate evidence. Missing sources "
                       "or unknown blocks remain unknown. Zero local matches do not establish "
                       "an unexplored mechanism or a marginal contribution to the book.")
    return payload
