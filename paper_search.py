#!/usr/bin/env python3
"""Search and read research papers through OpenAlex.

Metadata queries support optional OPENALEX_API_KEY authentication. Full-text downloads
require a key. Search follows provider relevance and reconstructs abstracts from their
inverted indexes. Missing abstracts and content remain unknown.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import gzip
import io
import re
import urllib.error
import xml.etree.ElementTree as ET
import os
import sys
import urllib.parse
import urllib.request

import cache_store
import research_paths

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BASE = "https://api.openalex.org/works"
USER_AGENT = "alpha-machine/1.0 (research)"
CACHE_TTL_SECONDS = 7 * 24 * 3600
CONTENT_BASE = "https://content.openalex.org/works/"
MAX_CONTENT_BYTES = 16 * 1024 * 1024


def _record_path(identifier, kind="records"):
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    return os.path.join(research_paths.cache_root(), "papers", kind, digest + ".json")


def _read_cache(path, kind):
    try:
        entry = cache_store.read_entry(path)
        if entry["fresh"] and entry["kind"] == kind and entry["expires_in"] is not None:
            return entry["payload"]
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        pass
    return None


def _save_cache(path, kind, payload):
    try:
        cache_store.write_json_atomic(path, cache_store.build_entry(
            kind, payload, ttl=CACHE_TTL_SECONDS, source=BASE))
        return path
    except OSError:
        return None


def _store_record(work):
    paper = normalise(work)
    key = paper["openalex_id"]
    if not key:
        return None
    # List responses can omit authors/locations; do not promote incomplete lists.
    if work.get("is_authors_truncated") or (work.get("locations_count") or 0) > len(work.get("locations") or []):
        return None
    aliases = [key]
    for value in [work.get("doi")] + [location.get("doi") for location in work.get("locations") or []]:
        alias = _identifier(value)
        if alias and alias not in aliases:
            aliases.append(alias)
    payload = {"version": 1, "paper": paper, "identifiers": aliases}
    path = _save_cache(_record_path(key), "paper_record", payload)
    if path:
        for alias in aliases[1:]:
            _save_cache(_record_path(alias), "paper_alias", {"identifier": alias, "openalex_id": key})
    return path


def _lookup_record(identifier):
    key = identifier
    if not identifier.startswith("W"):
        alias = _read_cache(_record_path(identifier), "paper_alias")
        if not isinstance(alias, dict) or alias.get("identifier") != identifier:
            return None
        key = alias.get("openalex_id")
        if not isinstance(key, str) or not re.fullmatch(r"W[0-9]+", key):
            return None
    path = _record_path(key)
    payload = _read_cache(path, "paper_record")
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return None
    paper = payload.get("paper")
    if (not isinstance(paper, dict) or paper.get("openalex_id") != key
            or identifier not in payload.get("identifiers", [])
            or not {"reading", "referenced_works", "abstract", "source"} <= paper.keys()):
        return None
    return {"source": "openalex", "paper": paper, "cached": True, "cache_path": path}


def get_paper(identifier, timeout=30):
    """Retrieve one attributable work; aliases require returned record evidence."""
    key = normalize_identifier(identifier)
    cached = _lookup_record(key)
    if cached:
        return cached
    target = key if key.startswith("W") else "doi:" + key
    work = _request(BASE + "/" + urllib.parse.quote(target, safe="/:"), timeout)
    if not isinstance(work, dict):
        raise PaperError("SERVICE_ERROR", "OpenAlex returned an invalid work record.")
    paper = normalise(work)
    identities = {paper["openalex_id"], _identifier(work.get("doi"))}
    identities.update(_identifier(location.get("doi")) for location in work.get("locations") or [])
    if not paper["openalex_id"] or key not in identities:
        raise PaperError("SERVICE_ERROR", "OpenAlex returned a different work identity; no alias was cached.")
    path = _store_record(work)
    return {"source": "openalex", "paper": paper, "cached": False, "cache_path": path}


def related_papers(identifier, relation="references", limit=10, from_year=None,
                   min_citations=None, timeout=30):
    """One bounded graph page, with explicit ordering and coverage."""
    if relation not in {"references", "citations"}:
        raise PaperError("INVALID_INPUT", "Relation must be references or citations.")
    limit = max(1, min(int(limit), 50))
    from_year = int(from_year) if from_year else None
    min_citations = int(min_citations) if min_citations else None
    seed = get_paper(identifier, timeout=timeout)["paper"]
    key = seed["openalex_id"]
    query = "%s of %s" % (relation, key)
    identity = json.dumps([key, relation, limit, from_year, min_citations])
    cache = _record_path(identity, "graph")
    cached = _read_cache(cache, "paper_graph")
    if isinstance(cached, dict) and cached.get("query") == query:
        return {**cached, "cached": True}
    references = seed["referenced_works"]
    coverage = {"ordering": "referenced_works_order" if relation == "references" else "provider_order",
                "from_year": from_year, "min_citations": min_citations,
                "request_limit": limit, "max_candidates": 100,
                "references_known": references is not None,
                "edge_meaning": "citation link only; no support or contradiction inferred"}
    candidates = []
    total = None
    if relation == "references":
        selected = list(dict.fromkeys(references or []))[:100]
        total = len(list(dict.fromkeys(references))) if references is not None else None
        if selected:
            payload = _request(BASE + "?" + urllib.parse.urlencode({
                "filter": "ids.openalex:" + "|".join(selected), "per-page": 100}), timeout)
            works = payload.get("results") or []
            by_id = {_identifier(work.get("id")): work for work in works}
            candidates = [by_id[item] for item in selected if item in by_id]
        coverage.update({"requested_candidates": len(selected),
                         "unresolved_candidates": len(selected) - len(candidates),
                         "unfetched_candidates": max(0, (total or 0) - len(selected))})
    else:
        filters = ["cites:" + key]
        if from_year:
            filters.append("from_publication_date:%d-01-01" % from_year)
        if min_citations:
            filters.append("cited_by_count:>%d" % (min_citations - 1))
        payload = _request(BASE + "?" + urllib.parse.urlencode({
            "filter": ",".join(filters), "per-page": limit}), timeout)
        candidates = (payload.get("results") or [])[:limit]
        total = (payload.get("meta") or {}).get("count")
        coverage["total_scope"] = "provider query after filters"
    papers = []
    for work in candidates:
        _store_record(work)
        paper = normalise(work)
        if (from_year and (paper["year"] is None or paper["year"] < from_year)) or (
                min_citations and (paper["citations"] is None or paper["citations"] < min_citations)):
            continue
        papers.append(paper)
    coverage.update({"fetched_candidates": len(candidates),
                     "filtered_out": len(candidates) - len(papers),
                     "eligible_in_page": len(papers),
                     "truncated": len(papers) > limit or (total is not None and total > len(candidates))})
    if relation == "references":
        coverage["total_scope"] = "seed referenced_works before filters"
    result = {"query": query, "source": "openalex", "seed": key,
              "relation": relation, "papers": papers[:limit], "returned": min(limit, len(papers)),
              "total_available": total, "cached": False, "coverage": coverage}
    _save_cache(cache, "paper_graph", result)
    return result


def _extract_tei(data):
    if data.startswith(b"\x1f\x8b"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as compressed:
                data = compressed.read(MAX_CONTENT_BYTES + 1)
        except (OSError, EOFError):
            raise PaperError("SERVICE_ERROR", "OpenAlex returned invalid compressed content.") from None
    if len(data) > MAX_CONTENT_BYTES:
        raise PaperError("SERVICE_ERROR", "OpenAlex XML exceeds the extracted size limit.")
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise PaperError("SERVICE_ERROR", "OpenAlex XML contains unsupported declarations.")
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        raise PaperError("SERVICE_ERROR", "OpenAlex returned invalid XML content.") from None
    local = lambda element: element.tag.rsplit("}", 1)[-1].lower()
    tei = next((element for element in root.iter() if local(element) == "tei"), None)
    if tei is None:
        raise PaperError("SERVICE_ERROR", "OpenAlex content is not TEI XML.")
    text_root = next((element for element in tei.iter() if local(element) == "text"), None)
    if text_root is None:
        return ""
    blocks = []

    def collect(element):
        if local(element) in {"head", "p", "item", "ab", "formula", "biblstruct"}:
            text = " ".join("".join(element.itertext()).split())
            if text:
                blocks.append(text)
            return
        for child in element:
            collect(child)

    collect(text_root)
    return "\n\n".join(blocks)


def read_content(identifier, timeout=30, offset=0, max_chars=12000):
    """Read advertised TEI explicitly, returning a bounded character window."""
    if not isinstance(offset, int) or not isinstance(max_chars, int) or offset < 0 or not 1 <= max_chars <= 100000:
        raise PaperError("INVALID_INPUT", "offset must be nonnegative and max_chars must be between 1 and 100000.")
    record = get_paper(identifier, timeout=timeout)
    paper = record["paper"]
    reading = paper["reading"]
    source_url = reading["grobid_xml_url"]
    content = {"state": reading["full_text_state"], "format": None, "source_url": source_url,
               "text": None, "offset": offset, "returned_chars": 0, "total_chars": None,
               "truncated": False, "next_offset": None, "cache_path": None, "cached": False}
    result = {"source": "openalex", "paper": paper, "content": content}
    if reading["grobid_xml"] is not True:
        content["reason"] = "Structured TEI is not advertised." if reading["grobid_xml"] is None else "OpenAlex reports no structured TEI content."
        return result
    path = _record_path(paper["openalex_id"], "content")
    cached = _read_cache(path, "paper_content")
    if (isinstance(cached, dict) and cached.get("openalex_id") == paper["openalex_id"]
            and cached.get("source_url") == source_url and isinstance(cached.get("text"), str)):
        text = cached["text"]
        content.update({"cached": True, "cache_path": path})
    else:
        try:
            text = _extract_tei(_request(source_url, timeout, content=True))
        except PaperError as error:
            if error.code != "NOT_FOUND":
                raise
            content.update({"state": "unavailable", "reason": "OpenAlex content endpoint returned HTTP 404."})
            return result
        if not text:
            content.update({"state": "unavailable", "reason": "Retrieved TEI contains no extractable text."})
            return result
        content["cache_path"] = _save_cache(path, "paper_content", {
            "openalex_id": paper["openalex_id"], "source_url": source_url, "text": text})
    window = text[offset:offset + max_chars]
    next_offset = offset + len(window) if offset + len(window) < len(text) else None
    content.update({"state": "read", "format": "tei", "text": window,
                    "returned_chars": len(window), "total_chars": len(text),
                    "truncated": offset > 0 or next_offset is not None, "next_offset": next_offset})
    return result


class PaperError(ValueError):
    """Provider failure safe to display without request credentials."""

    def __init__(self, code, message, hint=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


def normalize_identifier(value):
    """Accept only work IDs and DOIs, never arbitrary fetch URLs."""
    if not isinstance(value, str):
        raise PaperError("INVALID_INPUT", "Expected an OpenAlex work ID or DOI.")
    value = value.strip()
    if value.lower().startswith("https://"):
        parsed = urllib.parse.urlsplit(value)
        if (parsed.netloc.lower() not in {"openalex.org", "doi.org"}
                or parsed.query or parsed.fragment):
            raise PaperError("INVALID_INPUT", "Identifier URL must be an OpenAlex or DOI HTTPS URL without credentials or query.")
        host = parsed.netloc.lower()
        value = parsed.path.lstrip("/")
        if host == "openalex.org" and not re.fullmatch(r"[Ww][0-9]+", value):
            raise PaperError("INVALID_INPUT", "Expected an OpenAlex work ID.")
        if host == "doi.org" and not value.startswith("10."):
            raise PaperError("INVALID_INPUT", "Expected a DOI.")
    if re.fullmatch(r"[Ww][0-9]+", value):
        return value.upper()
    if value.lower().startswith("doi:"):
        value = value[4:]
    if re.fullmatch(r"10\.[0-9]{4,9}/[^\s?#]+", value):
        return value.lower()
    raise PaperError("INVALID_INPUT", "Expected W followed by digits or a valid DOI.")


def _identifier(value):
    try:
        return normalize_identifier(value)
    except PaperError:
        return None


def _safe_url(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urllib.parse.urlsplit(value)
        if (parsed.scheme not in {"https", "http"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            return None
    except ValueError:
        return None
    return value


def _api_key():
    from machine_lib import _load_dotenv
    _load_dotenv()
    return os.environ.get("OPENALEX_API_KEY", "").strip()


def _request(url, timeout, content=False):
    key = _api_key()
    if content and not key:
        raise PaperError("AUTH_MISSING", "OpenAlex structured content requires an API key.", "Set OPENALEX_API_KEY in the environment or .env.")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    # urllib omits unredirected headers when following a redirect to storage.
    if key:
        request.add_unredirected_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if content:
                data = response.read(MAX_CONTENT_BYTES + 1)
                if len(data) > MAX_CONTENT_BYTES:
                    raise PaperError("SERVICE_ERROR", "OpenAlex content exceeds the download size limit.")
                return data
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        code = {400: "INVALID_INPUT", 401: "AUTH_FAILED" if key else "AUTH_MISSING",
                403: "AUTH_FAILED" if key else "AUTH_MISSING", 404: "NOT_FOUND",
                429: "RATE_LIMITED"}.get(error.code, "SERVICE_ERROR")
        raise PaperError(code, "OpenAlex request failed (HTTP %d)." % error.code,
                         "Check OPENALEX_API_KEY." if code.startswith("AUTH") else None) from None
    except urllib.error.URLError:
        raise PaperError("SERVICE_ERROR", "OpenAlex could not be reached.") from None


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
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    locations = [primary, work.get("best_oa_location") or {}]
    locations.extend(work.get("locations") or [])
    landing_page = next((_safe_url(location.get("landing_page_url")) for location in locations
                         if location and _safe_url(location.get("landing_page_url"))), None)
    pdf = next((_safe_url(location.get("pdf_url")) for location in locations
                if location and _safe_url(location.get("pdf_url"))), None)
    work_id = _identifier(work.get("id"))
    if work_id and not work_id.startswith("W"):
        work_id = None
    content = work.get("has_content") or {}
    tei = content.get("grobid_xml") if isinstance(content.get("grobid_xml"), bool) else None
    pdf_available = content.get("pdf") if isinstance(content.get("pdf"), bool) else None
    abstract = reconstruct_abstract(work)
    references = work.get("referenced_works")
    paper = {
        "title": (work.get("title") or work.get("display_name") or "").strip(),
        "year": work.get("publication_year"),
        "journal": source.get("display_name"),
        "doi": _safe_url(work.get("doi")),
        "url": landing_page or _safe_url(work.get("doi")) or ("https://openalex.org/" + work_id if work_id else None),
        "citations": work.get("cited_by_count"),
        "authors": _authors(work)[:6],
        "abstract": abstract,
        "open_access": (work.get("open_access") or {}).get("is_oa"),
        "openalex_id": work_id,
        "source": "openalex",
        "referenced_works": [key for item in references if (key := _identifier(item)) and key.startswith("W")] if isinstance(references, list) else None,
        "cited_by_api_url": BASE + "?filter=cites:" + work_id if work_id else None,
        "reading": {"abstract_available": bool(abstract), "grobid_xml": tei,
                    "pdf": pdf_available,
                    "full_text_state": "available" if tei else "unavailable" if tei is False else "unknown",
                    "grobid_xml_url": CONTENT_BASE + work_id + ".grobid-xml" if tei and work_id else None},
    }
    if landing_page:
        paper["landing_page_url"] = landing_page
    if pdf:
        paper["pdf_url"] = pdf
    return paper


def _cache_path(query, limit, from_year, min_citations):
    request = {"query": query, "limit": limit, "from_year": from_year,
               "min_citations": min_citations}
    digest = hashlib.sha256(json.dumps(request, sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    return os.path.join(research_paths.cache_root(), "papers", digest + ".json")


def _cached_result(path, query):
    try:
        entry = cache_store.read_entry(path)
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    result = entry.get("payload")
    if (not entry["fresh"] or entry["kind"] != "papers"
            or entry["expires_in"] is None or entry["age_seconds"] is None
            or not isinstance(result, dict) or result.get("query") != query
            or result.get("source") != "openalex"
            or not isinstance(result.get("papers"), list)
            or result.get("returned") != len(result["papers"])
            or any(not isinstance(paper, dict) or not paper.get("title")
                   for paper in result["papers"])):
        return None
    return {**result, "cached": True}


def search_papers(query, limit=10, from_year=None, min_citations=None, timeout=30):
    """Return papers matching ``query``, ranked by relevance.

    ``min_citations`` filters after the fetch: OpenAlex can filter on it server-side, but
    the filter interacts badly with relevance ranking on narrow topics, so the ranking is
    left alone and the threshold applied here.
    """
    if not (query or "").strip():
        raise ValueError("query must not be empty")
    query = query.strip()
    limit = max(1, min(int(limit), 50))
    from_year = (int(from_year) or None) if from_year else None
    min_citations = (int(min_citations) or None) if min_citations else None
    cache = _cache_path(query, limit, from_year, min_citations)
    cached = _cached_result(cache, query)
    if cached is not None:
        return cached

    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(cache_store.file_lock(cache + ".lock"))
        except (OSError, cache_store.LockTimeout):
            pass  # Retrieval still works when the cache cannot be locked.
        cached = _cached_result(cache, query)
        if cached is not None:
            return cached

        params = {"search": query, "per-page": str(min(50, limit * 3 if min_citations else limit))}
        if from_year:
            params["filter"] = "from_publication_date:%d-01-01" % from_year
        url = BASE + "?" + urllib.parse.urlencode(params)
        payload = _request(url, timeout)

        papers = [normalise(work) for work in payload.get("results") or []]
        for work in payload.get("results") or []:
            _store_record(work)
        papers = [paper for paper in papers if paper["title"]]
        if min_citations:
            papers = [paper for paper in papers if (paper.get("citations") or 0) >= min_citations]
        papers = papers[:limit]
        result = {
            "query": query,
            "total_available": (payload.get("meta") or {}).get("count"),
            "returned": len(papers),
            "source": "openalex",
            "papers": papers,
            "cached": False,
        }
        try:
            cache_store.write_json_atomic(cache, cache_store.build_entry(
                "papers", result, ttl=CACHE_TTL_SECONDS, source=BASE))
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
