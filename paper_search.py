#!/usr/bin/env python3
"""Find papers, not web pages.

The general search in ``deepseek_search`` is a web search with reranking. Measured over
three mechanism queries it returned zero to one academic domain out of ten: the rest were
research-blog posts, accounting explainers and retail brokerage pages. That is fine for
looking up a definition and useless for finding the source of an anomaly.

OpenAlex is free, needs no key, and answers with what a paper search should: journal,
year, DOI and citation count. Two properties matter more than the metadata:

* sorting by relevance rather than citations. Sorting by citations answers the query "net
  operating assets stock returns" with the most-cited finance papers ever written - the
  Fama-French mutual fund study among them - because citation count is not relevance. The
  default ranking returns the actual NOA paper.
* abstracts arrive as an inverted index, ``{"word": [positions]}``. Reconstructing them is
  required; without it every result is a title and the reason to prefer this over the web
  search disappears.

No abstract means the publisher withheld it. That is reported as absent rather than filled
with a guess.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BASE = "https://api.openalex.org/works"
#: OpenAlex asks clients to identify themselves; a project string plus the environment's
#: contact is enough to land in the polite pool.
USER_AGENT = "alpha-machine/1.0 (research; mailto:research@example.invalid)"
CACHE_TTL_SECONDS = 7 * 24 * 3600


def reconstruct_abstract(work):
    """Turn OpenAlex's inverted index into text, or return None."""
    inverted = work.get("abstract_inverted_index")
    if not inverted:
        return None
    positions = {}
    for word, indexes in inverted.items():
        for index in indexes:
            positions[index] = word
    return " ".join(positions[index] for index in sorted(positions))


def _authors(work):
    names = []
    for item in work.get("authorships") or []:
        name = ((item.get("author") or {}).get("display_name") or "").strip()
        if name:
            names.append(name)
    return names


def normalise(work):
    """One paper as this project reports it."""
    source = ((work.get("primary_location") or {}).get("source") or {})
    return {
        "title": (work.get("title") or work.get("display_name") or "").strip(),
        "year": work.get("publication_year"),
        "journal": source.get("display_name"),
        "doi": work.get("doi"),
        "url": work.get("id"),
        "citations": work.get("cited_by_count"),
        "authors": _authors(work)[:6],
        "abstract": reconstruct_abstract(work),
        "open_access": bool((work.get("open_access") or {}).get("is_oa")),
    }


def _cache_path(query, limit, from_year, min_citations):
    try:
        import cache_store
        directory = cache_store.default_dir()
    except Exception:  # noqa: BLE001 - a missing cache just means no caching
        return None
    safe = "".join(char if char.isalnum() else "-" for char in query.lower())[:60]
    name = "papers-%s-%s-%s-%s-%s.json" % (safe, limit, from_year or 0, min_citations or 0,
                                           int(time.time() // CACHE_TTL_SECONDS))
    return os.path.join(directory, name)


def search_papers(query, limit=10, from_year=None, min_citations=None, timeout=30):
    """Return papers matching ``query``, ranked by relevance.

    ``min_citations`` filters after the fetch: OpenAlex can filter on it server-side, but
    the filter interacts badly with relevance ranking on narrow topics, so the ranking is
    left alone and the threshold applied here.
    """
    if not (query or "").strip():
        raise ValueError("query must not be empty")
    limit = max(1, min(int(limit), 50))
    cache = _cache_path(query, limit, from_year, min_citations)
    if cache and os.path.exists(cache):
        try:
            with open(cache, encoding="utf-8") as handle:
                cached = json.load(handle)
            cached["cached"] = True
            return cached
        except (OSError, ValueError):
            pass

    params = {"search": query.strip(), "per-page": str(min(50, limit * 3 if min_citations else limit))}
    if from_year:
        params["filter"] = "from_publication_date:%d-01-01" % int(from_year)
    url = BASE + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    papers = [normalise(work) for work in payload.get("results") or []]
    papers = [paper for paper in papers if paper["title"]]
    if min_citations:
        papers = [paper for paper in papers if (paper.get("citations") or 0) >= int(min_citations)]
    papers = papers[:limit]
    result = {
        "query": query.strip(),
        "total_available": (payload.get("meta") or {}).get("count"),
        "returned": len(papers),
        "source": "openalex",
        "papers": papers,
        "cached": False,
    }
    if cache:
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False)
        except OSError:
            pass
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Search OpenAlex for papers.")
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--from-year", type=int)
    parser.add_argument("--min-citations", type=int)
    args = parser.parse_args(argv)
    data = search_papers(args.query, limit=args.limit, from_year=args.from_year,
                         min_citations=args.min_citations)
    for paper in data["papers"]:
        print("[%s] %s %s" % (paper.get("citations"), paper.get("year"), paper["title"]))
        print("    %s" % (paper.get("journal") or "-"))
        if paper.get("doi"):
            print("    %s" % paper["doi"])
        if paper.get("abstract"):
            print("    %s" % paper["abstract"][:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())