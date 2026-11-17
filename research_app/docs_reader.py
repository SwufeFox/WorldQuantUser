"""Read BRAIN's tutorial catalog and original document content blocks."""

from __future__ import annotations

import re

import requests

import machine_lib as ml

from .runtime import CliError, with_session


_CATALOG = "/tutorials"
_DOCUMENTS = "/tutorial-pages/"
_PLATFORM_DOCS = "https://platform.worldquantbrain.com/learn/documentation"
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _matches(query, *values):
    return any(isinstance(value, str) and query in value.casefold() for value in values)


def _catalog(session):
    # Use the same offset walker as BRAIN's other catalogs. Its requests keep
    # the endpoint fixed even when a response includes an absolute next URL.
    return ml._catalog_pages(session, _CATALOG, params={"limit": 50})


def _page_url(section_id, page_id):
    return f"{_PLATFORM_DOCS}/{section_id}/{page_id}"


def _valid_slug(value):
    return isinstance(value, str) and _SLUG.fullmatch(value) is not None


def cmd_docs_list(args, _sink):
    """List the catalog, optionally filtering its titles and identifiers."""
    query = getattr(args, "query", None)
    if query is not None:
        query = query.strip()
        if not query:
            raise CliError("INVALID_INPUT", "documentation query must not be empty")
    needle = query.casefold() if query else None

    def work(session):
        categories = []
        for batch in _catalog(session):
            for section in batch:
                section_id = section.get("id")
                pages = section.get("pages")
                if not _valid_slug(section_id) or not isinstance(pages, list):
                    raise CliError("SERVICE_ERROR", "BRAIN returned an invalid tutorial category")
                section_matches = needle is None or _matches(
                    needle, section_id, section.get("title"), section.get("category"))
                found = []
                for page in pages:
                    if not isinstance(page, dict) or not _valid_slug(page.get("id")):
                        raise CliError("SERVICE_ERROR", "BRAIN returned an invalid tutorial page")
                    if section_matches or _matches(needle, page["id"], page.get("title")):
                        found.append({**page, "url": _page_url(section_id, page["id"])})
                if section_matches or found:
                    categories.append({**section, "pages": found})
        return {"query": query, "category_count": len(categories),
                "page_count": sum(len(c["pages"]) for c in categories), "categories": categories}

    try:
        return with_session(args, work)
    except requests.RequestException as exc:
        raise CliError("SERVICE_ERROR", "BRAIN tutorial catalog request failed") from exc


def _document_url(session, slug):
    # The article endpoint supplies a grouping label (e.g. Getting Started).
    # The web URL also needs the catalog section ID (e.g. discover-brain).
    try:
        for batch in _catalog(session):
            for section in batch:
                section_id = section.get("id")
                pages = section.get("pages")
                if _valid_slug(section_id) and isinstance(pages, list) and any(
                    isinstance(page, dict) and page.get("id") == slug for page in pages
                ):
                    return _page_url(section_id, slug)
    except (ml.BrainError, requests.RequestException):
        # A failed metadata lookup should still leave the successfully read
        # article and its authoritative API URL available.
        return None
    return None


def cmd_docs_get(args, _sink):
    """Read a known documentation slug without converting its content blocks."""
    slug = args.slug
    if not _valid_slug(slug):
        raise CliError("INVALID_INPUT", "provide a BRAIN documentation slug",
                       "use `docs list` to find the page ID")
    endpoint = _DOCUMENTS + slug

    def work(session):
        document = ml.request_json(session, "GET", endpoint)
        if (not isinstance(document, dict) or document.get("id") != slug
                or not isinstance(document.get("content"), list)):
            raise CliError("SERVICE_ERROR", "BRAIN returned an invalid tutorial document")
        return {**document, "url": _document_url(session, slug), "api_url": ml.BRAIN_API_URL + endpoint}

    try:
        return with_session(args, work)
    except requests.RequestException as exc:
        raise CliError("SERVICE_ERROR", "BRAIN tutorial document request failed") from exc
