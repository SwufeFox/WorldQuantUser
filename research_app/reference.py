"""CLI reads for BRAIN's Zendesk community and Help Center."""

from __future__ import annotations

import re

from .reference_cache import read_cached
from .runtime import CliError, with_session
from .support_reader import support_get, support_pagination, with_support


def _resource_id(value):
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise CliError("INVALID_INPUT", "use the numeric ID from the BRAIN support link")
    return value


def _check_limit(value, maximum):
    if not 1 <= value <= maximum:
        raise CliError("INVALID_INPUT", f"--limit must be between 1 and {maximum}")


def _cursor_params(args):
    _check_limit(args.limit, 100)
    params = {"page[size]": args.limit}
    if args.cursor is not None:
        if not args.cursor.strip():
            raise CliError("INVALID_INPUT", "--cursor must be a non-empty after_cursor value")
        params["page[after]"] = args.cursor
    return params


def _page_params(args, maximum):
    _check_limit(args.limit, maximum)
    if args.page < 1:
        raise CliError("INVALID_INPUT", "--page must be one or greater")
    return {"page": args.page, "per_page": args.limit}


def _read(args, path, params, project):
    def read(brain_session):
        with with_support(brain_session) as support_session:
            return project(support_get(support_session, path, params=params))

    return read_cached(args, "support", {"path": path, "params": params or {}},
                       lambda: with_session(args, read))


def _records(payload, key, fields):
    rows = payload.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise CliError("SERVICE_ERROR", f"BRAIN support returned an invalid {key} list")
    return {key: [_project(row, fields) for row in rows], "count": payload.get("count"),
            "pagination": support_pagination(payload)}


def _record(payload, key, fields):
    row = payload.get(key)
    if not isinstance(row, dict) or row.get("id") is None:
        raise CliError("SERVICE_ERROR", f"BRAIN support returned an invalid {key}")
    return {key: _project(row, fields)}


def _project(row, fields):
    return {key: row[key] for key in fields if key in row}


_TITLE_FIELDS = ("id", "title", "html_url", "created_at", "updated_at")
_POST_FIELDS = (*_TITLE_FIELDS, "topic_id", "comment_count", "status", "pinned", "closed")
_ARTICLE_FIELDS = (*_TITLE_FIELDS, "section_id", "locale")


def cmd_forum_topics(args, _sink):
    return _read(args, "/api/v2/community/topics.json", _cursor_params(args),
                 lambda payload: _records(payload, "topics", ("id", "name", "html_url", "description")))


def cmd_forum_search(args, _sink):
    query = args.query.strip()
    if not query:
        raise CliError("INVALID_INPUT", "forum search needs a non-empty query")
    params = {**_page_params(args, 25), "query": query}
    return _read(args, "/api/v2/help_center/community_posts/search.json", params,
                 lambda payload: {"query": query, **_records(payload, "results", (*_POST_FIELDS, "snippet"))})


def cmd_forum_get(args, _sink):
    post_id = _resource_id(args.post_id)
    return _read(args, f"/api/v2/community/posts/{post_id}.json", None,
                 lambda payload: _record(payload, "post", (*_POST_FIELDS, "details")))


def cmd_forum_comments(args, _sink):
    post_id = _resource_id(args.post_id)
    fields = ("id", "post_id", "html_url", "body", "official", "created_at", "updated_at")
    return _read(args, f"/api/v2/community/posts/{post_id}/comments.json", _cursor_params(args),
                 lambda payload: {"post_id": post_id, **_records(payload, "comments", fields)})


def cmd_docs_articles(args, _sink):
    category_id = _resource_id(args.category_id)
    path = f"/api/v2/help_center/en-us/categories/{category_id}/articles.json"
    return _read(args, path, _page_params(args, 100),
                 lambda payload: {"category_id": category_id, **_records(payload, "articles", _ARTICLE_FIELDS)})


def cmd_docs_article(args, _sink):
    article_id = _resource_id(args.article_id)
    return _read(args, f"/api/v2/help_center/en-us/articles/{article_id}.json", None,
                 lambda payload: _record(payload, "article", (*_ARTICLE_FIELDS, "body")))
