"""Literature search and paper discovery commands."""

from __future__ import annotations

import deepseek_search
import paper_search
import research_paths

from .runtime import CliError


def _search_failure(exc):
    """Map a DeepSeek search error onto the CLI's documented codes."""
    if isinstance(exc, deepseek_search.MissingSearchKey):
        return CliError(
            "AUTH_MISSING", str(exc),
            "put DEEPSEEK_API_KEY in .env next to machine_lib.py, or export it; "
            "never paste the key into chat or a commit")
    if isinstance(exc, deepseek_search.SearchAuthError):
        return CliError("AUTH_FAILED", str(exc),
                        "check that the key is valid and the account can use the "
                        "Search API")
    if isinstance(exc, deepseek_search.SearchInsufficientBalance):
        return CliError("SERVICE_ERROR", str(exc),
                        "top up the DeepSeek account; Search and Fetch are free "
                        "until billing starts on 2026-11-01 Beijing time")
    if isinstance(exc, deepseek_search.SearchRateLimited):
        return CliError("RATE_LIMITED", str(exc), "wait a moment and retry")
    if isinstance(exc, deepseek_search.SearchBlocked):
        return CliError("NOT_FOUND", str(exc),
                        "the page was refused or had no usable text; try another "
                        "source")
    return CliError("SERVICE_ERROR", str(exc), "retry, or narrow the query")


def cmd_search(args, _sink):
    """Literature search or single-page fetch, quota-free on the BRAIN side.

    Kept in this CLI rather than a browser because the point is to connect a
    mechanism idea to the paper that tests it, and to leave a record of what was
    read. Results keep the provider's ranking and its ``rerank_score`` /
    ``authority`` / ``date`` fields so a journal article can be preferred over a
    blog restating it.
    """
    if not deepseek_search.available():
        raise _search_failure(deepseek_search.MissingSearchKey(
            "No DeepSeek key configured (DEEPSEEK_API_KEY)"))

    if args.fetch_url:
        try:
            page = deepseek_search.fetch(args.fetch_url)
        except deepseek_search.DeepSeekSearchError as exc:
            raise _search_failure(exc) from exc
        content = page.get("content") or ""
        data = {
            "mode": "fetch",
            "url": args.fetch_url,
            "title": page.get("title"),
            "content": content,
            "chars": len(content),
        }

        return data

    if not args.query:
        raise CliError("USAGE", "nothing to search",
                       "pass a query, or --fetch URL to read one page")

    cached = False
    try:
        if args.no_cache:
            results = deepseek_search.search(
                args.query, count=args.count, need_content=args.content)
        else:
            results, cached = deepseek_search.search_cached(
                research_paths.ensure_dir(), args.query, count=args.count,
                need_content=args.content, max_age_hours=args.max_age_hours)
    except deepseek_search.DeepSeekSearchError as exc:
        raise _search_failure(exc) from exc

    total = len(results)
    kept = deepseek_search.high_quality(
        results, min_score=args.min_score, min_authority=args.min_authority)
    data = {
        "mode": "search",
        "query": args.query,
        "total": total,
        "returned": len(kept),
        "cached": cached,
        "filters": {"min_score": args.min_score,
                    "min_authority": args.min_authority},
        "results": [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "summary": item.get("summary"),
                "site_name": item.get("site_name"),
                "date": item.get("date"),
                "rerank_score": item.get("rerank_score"),
                "authority": item.get("authority"),
                "content": item.get("content") if args.content else None,
            }
            for item in kept
        ],
        "note": ("rerank_score 1.5+ is high quality and authority 3+ is "
                 "authoritative; prefer the paper over posts about it"),
    }

    return data


def cmd_paper(args, _sink):
    """Paper search: the scholarly counterpart to `search`.

    `search` is a web search with reranking and, measured over three mechanism queries,
    returned zero to one academic domain in ten. This one talks to OpenAlex, so every row
    carries a journal, a year and a citation count, which is what makes a result worth
    chasing rather than inferring a formula from a blog post.
    """
    try:
        data = paper_search.search_papers(args.query, limit=args.limit,
                                          from_year=args.from_year,
                                          min_citations=args.min_citations)
    except ValueError as exc:
        raise CliError("INVALID_INPUT", str(exc), "query must not be empty") from exc
    except Exception as exc:  # noqa: BLE001 - network and DNS both land here
        raise CliError("SERVICE_ERROR", "OpenAlex request failed: %s" % exc,
                       "the API is public and needs no key; retry shortly") from exc
    return data
