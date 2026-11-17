"""Literature search and paper discovery commands."""

from __future__ import annotations

import json

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
                        "check the provider's current balance and pricing")
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
    """Discover, inspect, read or expand one attributable scholarly source."""
    query = getattr(args, "query", None)
    identifier = getattr(args, "paper_id", None)
    relation = getattr(args, "related", None)
    full_text = getattr(args, "full_text", False)
    offset = getattr(args, "offset", None)
    max_chars = getattr(args, "max_chars", None)
    timeout = getattr(args, "timeout", 30)
    if (query is None) == (identifier is None):
        raise CliError("USAGE", "provide either a keyword query or --id ID",
                       'use paper "analyst revisions" or paper --id W2741809807')
    if query is not None and not query.strip():
        raise CliError("INVALID_INPUT", "query must not be empty")
    if (relation or full_text) and identifier is None:
        raise CliError("INVALID_INPUT", "--related and --full-text require --id")
    if relation and full_text:
        raise CliError("INVALID_INPUT", "choose --related or --full-text for this invocation")
    if not full_text and (offset is not None or max_chars is not None):
        raise CliError("INVALID_INPUT", "--offset and --max-chars apply only to --full-text")
    if identifier is not None and not relation and any(
        getattr(args, name, None) is not None for name in ("from_year", "min_citations")
    ):
        raise CliError("INVALID_INPUT", "year/citation filters apply to search or --related")
    try:
        if full_text:
            data = paper_search.read_content(identifier, timeout=timeout,
                offset=offset if offset is not None else 0,
                max_chars=max_chars if max_chars is not None else 12000)
        elif relation:
            data = paper_search.related_papers(identifier, relation=relation, limit=args.limit,
                from_year=args.from_year, min_citations=args.min_citations, timeout=timeout)
        elif identifier is not None:
            data = paper_search.get_paper(identifier, timeout=timeout)
        else:
            data = paper_search.search_papers(query, limit=args.limit,
                from_year=args.from_year, min_citations=args.min_citations, timeout=timeout)
    except paper_search.PaperError as exc:
        raise CliError(exc.code, str(exc), getattr(exc, "hint", None)) from None
    except json.JSONDecodeError:
        raise CliError("SERVICE_ERROR", "OpenAlex returned invalid JSON") from None
    except ValueError:
        raise CliError("INVALID_INPUT", "invalid paper query, identifier or reading window") from None
    except Exception as exc:
        raise CliError("SERVICE_ERROR", "Paper retrieval failed.",
                       "Check connectivity and the provider's current access requirements.",
                       details={"type": type(exc).__name__}) from None
    content = data.get("content") or {}
    if full_text and content.get("next_offset") is not None:
        paper_id = (data.get("paper") or {}).get("openalex_id") or identifier
        data["next_actions"] = [{"kind": "read_next", "argv": [
            "--research-dir", research_paths.research_dir(), "paper", "--id", paper_id,
            "--full-text", "--offset", str(content["next_offset"]),
            "--max-chars", str(max_chars if max_chars is not None else 12000),
            "--timeout", str(timeout)]}]
    return data
