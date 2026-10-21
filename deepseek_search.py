#!/usr/bin/env python3
"""DeepSeek Search and Fetch: literature lookup for the research loop.

Why this exists: the missing link in this toolkit was between a mechanism idea
and the paper that tests it. Catalog metadata says a field exists; it cannot say
whether the economic claim behind it has already been rejected. Reading papers by
hand through a browser does not scale and leaves no record, so the search lives
here, next to the research commands, and returns the same one-JSON-document shape
as everything else.

Uses DeepSeek's own endpoints:

    POST https://api.deepseek.com/v1/tools/search
    POST https://api.deepseek.com/v1/tools/fetch

Both take the same bearer key as the rest of the DeepSeek API. The key is read
from ``DEEPSEEK_API_KEY`` after loading the existing ``.env`` loader in
``machine_lib``; it is never written to the repository or printed.

Results are ranked by the provider's own ``rerank_score`` (0-3, with 1.5+ treated
as high quality) and carry ``authority`` (0-4, 3+ authoritative) and ``date``,
which is what makes it possible to prefer an actual journal article over a blog
post repeating it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

import machine_lib

__all__ = [
    "DeepSeekSearchError",
    "MissingSearchKey",
    "SearchAuthError",
    "SearchRateLimited",
    "SearchInsufficientBalance",
    "SearchBlocked",
    "SearchServiceError",
    "api_key",
    "available",
    "search",
    "fetch",
    "high_quality",
]

SEARCH_URL = "https://api.deepseek.com/v1/tools/search"
FETCH_URL = "https://api.deepseek.com/v1/tools/fetch"
MAX_QUERY_CHARS = 128


class DeepSeekSearchError(RuntimeError):
    """Base class so callers can catch one exception type."""


class MissingSearchKey(DeepSeekSearchError):
    """No key configured."""


class SearchAuthError(DeepSeekSearchError):
    """Key rejected, or the account cannot use the endpoint."""


class SearchInsufficientBalance(DeepSeekSearchError):
    """HTTP 402: the account needs a top-up."""


class SearchRateLimited(DeepSeekSearchError):
    """HTTP 429: query-per-second limit hit."""


class SearchBlocked(DeepSeekSearchError):
    """HTTP 403/404: the target page was refused or had no usable content."""


class SearchServiceError(DeepSeekSearchError):
    """HTTP 5xx or a transport failure."""


def api_key():
    """Return the DeepSeek key, or raise with an actionable message.

    ``WEBSEARCH_API_KEY`` is accepted as an alias because other tools in this
    space use that name; the explicit name wins when both are set.
    """
    machine_lib._load_dotenv()
    for name in ("DEEPSEEK_API_KEY", "WEBSEARCH_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise MissingSearchKey(
        "No DeepSeek key found. Add DEEPSEEK_API_KEY to the .env file next to "
        "machine_lib.py (copy .env.example and fill it in), or export it in the "
        "shell. The key is only read from the environment and is never written "
        "to the repository.")


def available():
    """True when a key is configured, without calling the network."""
    try:
        api_key()
        return True
    except MissingSearchKey:
        return False


def _post(url, payload, timeout=60):
    try:
        response = requests.post(
            url,
            headers={"Authorization": "Bearer " + api_key(),
                     "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise SearchServiceError("request failed: %s" % exc) from exc

    if response.status_code == 200:
        try:
            return response.json()
        except ValueError as exc:
            raise SearchServiceError("response was not JSON") from exc

    detail = ""
    try:
        body = response.json()
        detail = ((body.get("error") or {}).get("message") or "").strip()
    except ValueError:
        detail = response.text.strip()[:200]

    code = response.status_code
    if code == 401:
        raise SearchAuthError("key rejected (401): %s" % (detail or "unauthorized"))
    if code == 402:
        raise SearchInsufficientBalance(
            "insufficient balance (402): top up the DeepSeek account")
    if code == 429:
        raise SearchRateLimited("rate limited (429): %s" % (detail or "slow down"))
    if code in (403, 404):
        raise SearchBlocked("refused (HTTP %d): %s" % (code, detail or "no content"))
    raise SearchServiceError("HTTP %d: %s" % (code, detail or "service error"))


def search(query, count=10, need_content=False, ori_question=None,
           localization=""):
    """Run one search and return the provider's result list.

    The query is passed through with its operators intact, so ``site:``,
    ``"exact phrase"``, ``lang:``, ``after:`` and ``before:`` all work. A query
    longer than the documented 128-character limit is rejected here rather than
    by the server, because the server's error does not say what to cut.
    """
    text = (query or "").strip()
    if not text:
        raise SearchServiceError("query is empty")
    if len(text) > MAX_QUERY_CHARS:
        raise SearchServiceError(
            "query is %d characters; the API accepts at most %d"
            % (len(text), MAX_QUERY_CHARS))

    payload = {"query": text, "count": max(1, min(50, int(count))),
               "need_content": bool(need_content)}
    if ori_question:
        payload["ori_question"] = str(ori_question)[:MAX_QUERY_CHARS]
    if localization:
        payload["localization"] = localization

    body = _post(SEARCH_URL, payload)
    return body.get("results") or []


def fetch(url):
    """Fetch one page as Markdown: ``{title, content}``."""
    target = (url or "").strip()
    if not target.lower().startswith(("http://", "https://")):
        raise SearchServiceError("fetch needs an http(s) URL")
    return _post(FETCH_URL, {"url": target})


def high_quality(results, min_score=1.5, min_authority=None):
    """Filter to results worth reading, preserving order.

    The provider already ranks, so this is a floor rather than a re-sort: a
    sub-1.5 ``rerank_score`` usually means the query matched loosely, and an
    ``authority`` floor removes blog restatements of a paper.
    """
    kept = []
    for item in results or []:
        if (item.get("rerank_score") or 0) < min_score:
            continue
        if min_authority is not None and (item.get("authority") or -1) < min_authority:
            continue
        kept.append(item)
    return kept


def cache_path(research_dir, query):
    """Where a query's results may be cached inside the results tree."""
    import hashlib
    digest = hashlib.sha1((query or "").encode("utf-8")).hexdigest()[:16]
    return Path(research_dir) / "search-cache" / (digest + ".json")


def search_cached(research_dir, query, count=10, need_content=False,
                  ori_question=None, max_age_hours=168):
    """Search, reusing a cached answer when it is fresh enough.

    A literature query's answer does not change hour to hour, so re-running the
    same query in a later session should not cost a request. ``max_age_hours``
    defaults to a week.
    """
    path = cache_path(research_dir, query)
    if path.is_file():
        age = time.time() - path.stat().st_mtime
        if age < max_age_hours * 3600:
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("count") == count:
                    return cached.get("results") or [], True
            except (OSError, ValueError):
                pass

    results = search(query, count=count, need_content=need_content,
                     ori_question=ori_question)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"query": query, "count": count,
                                    "results": results}, ensure_ascii=False),
                        encoding="utf-8")
    except OSError:
        pass
    return results, False