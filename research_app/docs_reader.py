"""Read BRAIN's tutorial catalog and original document content blocks."""

from __future__ import annotations

import re

import requests

import machine_lib as ml

from .docs_text import render_blocks
from .reference_cache import read_cached
from .runtime import CliError, session_reader


_CATALOG = "/tutorials"
_DOCUMENTS = "/tutorial-pages/"
_PLATFORM_DOCS = "https://platform.worldquantbrain.com/learn/documentation"
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _matches(query, *values):
    return any(isinstance(value, str) and query in value.casefold() for value in values)


def _page_url(section_id, page_id):
    return f"{_PLATFORM_DOCS}/{section_id}/{page_id}"


def _valid_slug(value):
    return isinstance(value, str) and _SLUG.fullmatch(value) is not None


def _fetch_catalog(session):
    categories = []
    for batch in ml._catalog_pages(session, _CATALOG, params={"limit": 50}):
        for section in batch:
            if not isinstance(section, dict):
                raise CliError("SERVICE_ERROR", "BRAIN returned an invalid tutorial category")
            section_id = section.get("id")
            pages = section.get("pages")
            if not _valid_slug(section_id) or not isinstance(pages, list):
                raise CliError("SERVICE_ERROR", "BRAIN returned an invalid tutorial category")
            linked_pages = []
            for page in pages:
                if not isinstance(page, dict) or not _valid_slug(page.get("id")):
                    raise CliError("SERVICE_ERROR", "BRAIN returned an invalid tutorial page")
                linked_pages.append({**page, "url": _page_url(section_id, page["id"])})
            categories.append({**section, "pages": linked_pages})
    return {"categories": categories}


def _cached_catalog(args, read=None):
    read = read if read is not None else session_reader(args)

    def fetch():
        try:
            return read(_fetch_catalog)
        except requests.RequestException as exc:
            raise CliError("SERVICE_ERROR", "BRAIN tutorial catalog request failed") from exc

    return read_cached(args, "brain-tutorial-catalog", {}, fetch)


def _query(query):
    if query is not None:
        query = query.strip()
        if not query:
            raise CliError("INVALID_INPUT", "documentation query must not be empty")
    return query


def _filter_catalog(sections, query):
    needle = query.casefold() if query else None
    categories = []
    for section in sections:
        section_matches = needle is None or _matches(
            needle, section["id"], section.get("title"), section.get("category"))
        pages = [page for page in section["pages"]
                 if section_matches or _matches(needle, page["id"], page.get("title"))]
        if section_matches or pages:
            categories.append({**section, "pages": pages})
    return categories


def _page_summaries(categories):
    return [{"id": page["id"], "title": page.get("title"), "url": page["url"]}
            for section in categories for page in section["pages"]]


def cmd_docs_list(args, _sink):
    """List the catalog, optionally filtering its titles and identifiers."""
    query = _query(getattr(args, "query", None))
    cached = _cached_catalog(args)
    categories = _filter_catalog(cached["categories"], query)
    result = {"query": query, "category_count": len(categories),
              "page_count": sum(len(c["pages"]) for c in categories), "cache": cached["cache"]}
    if getattr(args, "summary", False):
        return {**result, "pages": _page_summaries(categories)}
    return {**result, "categories": categories}


def _lookup_page(args, query, read):
    categories = _filter_catalog(_cached_catalog(args, read)["categories"], query)
    pages = _page_summaries(categories)
    needle = query.casefold()
    exact = [page for page in pages
             if any(isinstance(value, str) and value.casefold() == needle
                    for value in (page["id"], page["title"]))]
    matches = exact or pages
    if not matches:
        raise CliError("NOT_FOUND", "no BRAIN documentation page matches this query",
                       "try a shorter keyword or use `docs list --summary`")
    if len(matches) > 1:
        raise CliError("INVALID_INPUT", "the documentation query matches multiple pages",
                       "choose a page ID, then use `docs get PAGE_ID --text`",
                       details={"pages": matches})
    return matches[0]


def _document_url(args, session, slug):
    # The article endpoint supplies a grouping label (e.g. Getting Started).
    # The web URL also needs the catalog section ID (e.g. discover-brain).
    try:
        categories = _cached_catalog(args, lambda work: work(session))["categories"]
    except (CliError, ml.BrainError):
        # A failed metadata lookup should still leave the successfully read
        # article and its authoritative API URL available.
        return None
    return next((page["url"] for section in categories for page in section["pages"]
                 if page["id"] == slug), None)


def cmd_docs_get(args, _sink):
    """Read a page by ID or catalog keyword, optionally rendering its body."""
    query = _query(getattr(args, "query", None))
    read = session_reader(args)
    page = _lookup_page(args, query, read) if query is not None else None
    slug = page["id"] if page is not None else args.slug
    if not _valid_slug(slug):
        raise CliError("INVALID_INPUT", "provide a BRAIN documentation slug",
                       "use `docs list` to find the page ID")
    endpoint = _DOCUMENTS + slug

    def work(session):
        document = ml.request_json(session, "GET", endpoint)
        if (not isinstance(document, dict) or document.get("id") != slug
                or not isinstance(document.get("content"), list)):
            raise CliError("SERVICE_ERROR", "BRAIN returned an invalid tutorial document")
        url = page["url"] if page is not None else _document_url(args, session, slug)
        return {**document, "url": url,
                "api_url": ml.BRAIN_API_URL + endpoint}

    def fetch():
        try:
            return read(work)
        except requests.RequestException as exc:
            raise CliError("SERVICE_ERROR", "BRAIN tutorial document request failed") from exc

    document = read_cached(args, "brain-tutorial-page", {"slug": slug}, fetch)
    if page is not None and document.get("url") != page["url"]:
        document = {**document, "url": page["url"]}
    if not getattr(args, "text", False):
        return document
    text, unsupported = render_blocks(document["content"])
    return {**{key: value for key, value in document.items() if key != "content"},
            "text": text, "unsupported_blocks": unsupported}
