"""OpenAlex paper discovery and reading commands."""

from __future__ import annotations

import json

import paper_search
import research_paths

from .runtime import CliError


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
