"""An input stream: turn open questions into fresh literature, without repeating.

Research went stale in a specific way twice this month. The same directions were
re-probed because nothing remembered that they had been closed, and the ideas that did
arrive came from a person pasting a repo into the chat. Inspiration needs a pipe, not a
tap: a standing list of open questions, a query per question, a seen-store so tomorrow
does not re-read today, and a filter that drops what the closed register already killed.

What this deliberately does *not* do is write mechanism cards automatically. A card needs
a falsification condition and a closest-existing-family claim, and a title plus an abstract
does not supply either. The stream proposes; a person (or a deliberate pass) promotes.
"""

import json
import os
import re

import research_paths

#: A paper is only worth reading if it is not about a direction already closed. These are
#: the phrases the closed register uses; matching is on the title only, so a paper that
#: merely mentions the topic is not discarded.
CLOSED_MARKERS = (
    "pead", "post-earnings", "post earnings announcement",
    "placebo", "reversal of fortune",
)

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")

#: Words too common to make a query specific.
STOPWORDS = frozenset("""
the and for with from that this into over under about across after before between
study evidence effect effects using based does does not its our their their on in of to
""".split())


def _queue_path(args=None):
    explicit = getattr(args, "queue", None)
    if explicit:
        return explicit
    return os.path.join(research_paths.ensure_dir(), "inspiration", "queue.jsonl")


def _seen_path(args=None):
    explicit = getattr(args, "seen", None)
    if explicit:
        return explicit
    return os.path.join(research_paths.ensure_dir(), "inspiration", "seen.jsonl")


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


def open_questions(path):
    """The live questions, one per line, from ``open-threads.md``.

    Only the leading bolded heading of each bullet is used: the rest of the bullet is the
    evidence, which is exactly what makes a bad search query.
    """
    questions = []
    if not os.path.exists(path):
        return questions
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text.startswith("- "):
                continue
            if "[closed]" in text.lower():
                continue
            heading = re.match(r"- \**([^*]{4,120}?)\**[:：]", text)
            if heading:
                questions.append(heading.group(1).strip())
    return questions


_CODE = re.compile(r"^h-?\d+[a-z]?$|^\d+$|^20\d\d$")


def query_from_question(question, max_words=8):
    """A search string of the distinctive Latin words in a question.

    Chinese headings yield nothing here on purpose: OpenAlex is queried in English, and a
    query made of "h-040" would return noise while looking like a real search. When there
    are no usable tokens the caller falls back to the topics file.
    """
    words = [w for w in _TOKEN.findall(question.lower())
             if w not in STOPWORDS and not _CODE.match(w)]
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


#: A paper has to be about *pricing* to be useful here. OpenAlex happily returns supply
#: chain management and sustainability papers for a "supplier network" query, and reading
#: those wastes the one resource this stream is meant to protect.
RELEVANCE_MARKERS = ("return", "cross-section", "cross section", "anomaly", "predict",
                     "portfolio", "asset pricing", "alpha", "mispricing", "premium",
                     "accrual", "earnings quality", "earnings response", "information risk")


def is_relevant(paper):
    text = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).lower()
    return any(marker in text for marker in RELEVANCE_MARKERS)


def is_closed(title):
    lowered = (title or "").lower()
    return any(marker in lowered for marker in CLOSED_MARKERS)


def score(paper):
    """Rank what to read first: recent and cited, with a floor for relevance by year."""
    citations = paper.get("citations") or 0
    year = paper.get("year") or 0
    return citations + max(0, year - 1990) * 3


def cmd_feed(args, _sink):
    """One digest: questions -> queries -> papers -> a ranked, deduped list."""
    import paper_search

    queue_path, seen_path = _queue_path(args), _seen_path(args)
    seen = load_seen(seen_path)

    topics = []
    if args.topic:
        topics = list(args.topic)
    if args.from_threads:
        threads = os.path.join(research_paths.ensure_dir(), "open-threads.md")
        for question in open_questions(threads):
            query = query_from_question(question)
            if query:
                topics.append(query)
    if getattr(args, "topics", False):
        path = getattr(args, "topics_file", None) or os.path.join(
            research_paths.ensure_dir(), "inspiration", "topics.md")
        topics.extend(topics_file_queries(path))
    if not topics:
        raise ValueError("nothing to search: pass --topic or --from-threads")

    papers, dropped_closed, repeats = [], [], []
    for topic in topics[: args.max_topics]:
        try:
            found = paper_search.search_papers(
                topic, limit=args.limit, from_year=args.from_year,
                min_citations=args.min_citations)
        except Exception:  # noqa: BLE001 - a network hiccup must not kill the digest
            continue
        rows = found.get("papers") if isinstance(found, dict) else found
        for paper in rows or []:
            key = paper_key(paper)
            if not key:
                continue
            if key in seen:
                repeats.append(paper.get("title"))
                continue
            if is_closed(paper.get("title")):
                dropped_closed.append(paper.get("title"))
                continue
            if not is_relevant(paper):
                dropped_closed.append("off-topic: %s" % (paper.get("title") or "")[:60])
                continue
            seen.add(key)
            papers.append({**paper, "query": topic, "key": key,
                           "rank": score(paper)})
        if len(papers) >= args.total:
            break

    papers.sort(key=lambda row: -row["rank"])
    picked = papers[: args.total]

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
                    # A draft, not a card: the two fields that make a card checkable
                    # cannot come from a title, so they are left empty on purpose and
                    # `ledger cards --add` will refuse the file until they are filled.
                    handle.write(json.dumps({
                        "mechanism_id": None,
                        "status": "DRAFT",
                        "source": {"title": paper.get("title"), "doi": paper.get("doi"),
                                   "journal": paper.get("journal"),
                                   "year": paper.get("year"), "verified": False},
                        "hypothesis": {"economic_mechanism": None, "falsification": None},
                        "implementation": {"conceptual_fields": [],
                                           "verified_brain_fields": []},
                        "novelty": {"closest_existing_family": None},
                        "abstract": (paper.get("abstract") or "")[:600],
                    }, ensure_ascii=False) + "\n")

    return {
        "topics": topics[: args.max_topics],
        "seen_before": len(repeats),
        "dropped_closed": dropped_closed[:5],
        "count": len(picked),
        "queue": queue_path,
        "seen": seen_path,
        "papers": [{k: paper.get(k) for k in
                    ("title", "year", "journal", "doi", "citations", "query", "rank")}
                   for paper in picked],
        "note": ("drafts are deliberately un-promotable: `ledger cards --add` will refuse "
                 "them until the falsification condition and closest family are filled."),
    }