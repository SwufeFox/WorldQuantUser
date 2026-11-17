"""Bounded public research leads with explicit coverage and source provenance.

RSS follows https://www.rssboard.org/rss-specification and RFC 4287.
Provider schemas: https://endpoints.bsky.app/openapi.bluesky-app.json,
https://hn.algolia.com/api.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, quote, quote_plus, urlencode, urljoin, urlsplit
import xml.etree.ElementTree as ET

import cache_store
import research_paths
from source_io import (SourceError, parse_timestamp, public_url,
                       request_bytes, request_json, utc_now, validate_url)

CACHE_TTL_SECONDS = 300
MAX_FILE_BYTES = 4 * 1024 * 1024
_GOOGLE_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_BLUESKY = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"
_HN = "https://hn.algolia.com/api/v1/search"
_SECRET_KEYS = {"api_key", "apikey", "key", "token", "access_token", "auth", "signature"}
BUILTIN_SOURCES = {
    "paper": {"id": "paper", "kind": "paper", "adapter": "paper"},
    "news": {"id": "news", "kind": "news", "adapter": "rss", "url": _GOOGLE_RSS},
    "social": {"id": "social", "kind": "social", "adapter": "bluesky"},
    "forum": {"id": "forum", "kind": "forum", "adapter": "hn"},
    "trends": {"id": "trends", "kind": "trend", "adapter": "hn_frontpage"},
}
_ADAPTER_KEYS = {"rss": {"url"}, "bluesky": set(),
                 "hn": set(), "hn_frontpage": set(),
                 "market": {"venue", "symbol", "depth", "feed"}, "file": {"path"}, "paper": set()}
_KINDS = {"news", "social", "forum", "trend", "web"}


def _invalid(message):
    raise SourceError("INVALID_INPUT", message)


def _validate_definition(source, custom=False):
    if not isinstance(source, dict):
        _invalid("Each source definition must be an object.")
    adapter = source.get("adapter")
    if not isinstance(adapter, str) or adapter not in _ADAPTER_KEYS:
        _invalid("Source adapter must be rss, bluesky, hn, hn_frontpage, file or market.")
    if set(source) - ({"id", "kind", "adapter"} | _ADAPTER_KEYS[adapter]):
        _invalid("Source definition contains unsupported keys for its adapter.")
    identifier = source.get("id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", identifier):
        _invalid("Source id must be 1..64 letters, digits, dots, underscores or hyphens.")
    if identifier == "all":
        _invalid("Source id 'all' is reserved for selecting every source.")
    if custom and identifier in BUILTIN_SOURCES:
        _invalid("Custom source ids must not collide with built-in ids.")
    kind = source.get("kind")
    expected = {"bluesky": "social", "hn": "forum", "hn_frontpage": "trend",
                "market": "market", "paper": "paper"}.get(adapter)
    if not isinstance(kind, str) or (kind != expected if expected else kind not in _KINDS):
        _invalid("Source kind is incompatible with its adapter.")
    result = copy.deepcopy(source)
    if adapter == "paper":
        if custom or source != BUILTIN_SOURCES["paper"]:
            _invalid("Paper retrieval is configured through the built-in paper source.")
    elif adapter == "rss":
        url = source.get("url")
        if not isinstance(url, str):
            _invalid("RSS sources require a URL.")
        probe = url.replace("{query}", "topic")
        if "{" in probe or "}" in probe:
            _invalid("RSS templates support only the literal {query} placeholder.")
        validate_url(probe)
        if any(key.lower() in _SECRET_KEYS for key, _ in parse_qsl(urlsplit(probe).query)):
            _invalid("Source URLs cannot contain credential parameters; use public feed URLs.")
    elif adapter == "file":
        value = source.get("path")
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            _invalid("File sources require a nonempty local path.")
        if Path(value).suffix.lower() not in {".jsonl", ".txt", ".md"}:
            _invalid("File sources support only .jsonl, .txt and .md exports.")
        try:
            result["path"] = str(Path(value).resolve())
        except (OSError, ValueError, RuntimeError):
            _invalid("File source path could not be resolved.")
    elif adapter == "market":
        from market_sources import validate_market
        normalized = validate_market(source.get("venue"), source.get("symbol"),
                                     depth=source.get("depth", 10), feed=source.get("feed", "iex"))
        result.update(normalized)
    return result


def load_sources(path=None):
    """Return built-ins plus a strictly validated UTF-8/BOM source file."""
    result = copy.deepcopy(BUILTIN_SOURCES)
    if path is None:
        return result
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, UnicodeError, TypeError):
        raise SourceError("INVALID_INPUT", "Sources file must be readable UTF-8 JSON.") from None
    if not isinstance(payload, dict) or set(payload) != {"sources"} or not isinstance(payload["sources"], list):
        _invalid("Sources file must contain exactly a sources array.")
    for source in payload["sources"]:
        if isinstance(source, dict) and source.get("adapter") == "file" and isinstance(source.get("path"), str):
            source = dict(source, path=str(Path(path).resolve().parent / source["path"]))
        normalized = _validate_definition(source, custom=True)
        if normalized["id"] in result:
            _invalid("Source ids must be unique.")
        result[normalized["id"]] = normalized
    return result


def requires_query(source):
    """Whether the feed scheduler must create one job per topic."""
    adapter = source.get("adapter")
    return adapter in {"paper", "bluesky", "hn"} or (
        adapter == "rss" and "{query}" in source.get("url", ""))


def describe_source(source):
    """Return safe configuration and scheduling metadata, never credentials."""
    result = _validate_definition(source)
    if "url" in result:
        result["url"] = public_url(result["url"].replace("{query}", "SOURCEQUERYPLACEHOLDER")).replace(
            "SOURCEQUERYPLACEHOLDER", "{query}")
    result["requires_query"] = requires_query(source)
    result["cache_ttl_seconds"] = 0 if result["adapter"] in {"market", "file"} else CACHE_TTL_SECONDS
    return result


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"p", "div", "br", "li", "section", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif tag in {"p", "div", "li", "section"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _text(value):
    if not isinstance(value, str):
        return ""
    parser = _Text()
    parser.feed(value)
    return " ".join("".join(parser.parts).split())


def _url(value, base=None):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return public_url(urljoin(base, value.strip()) if base else value.strip())
    except (SourceError, ValueError):
        return None


def _safe_raw(value):
    # Preserve selected public record fields while removing credentials from links.
    if isinstance(value, dict):
        return {key: _safe_raw(item) for key, item in value.items() if key.lower() not in _SECRET_KEYS}
    if isinstance(value, list):
        return [_safe_raw(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"https?://[^\s<>\"']+", lambda match: _url(match.group()) or "[invalid URL]", value)
    return value


def _item(source, provider, identifier, title, url, excerpt, date, collected, author, raw,
          basis, limitations, engagement=None):
    stamp = parse_timestamp(date)
    title = title if provider in {"bluesky", "local_file"} else _text(title)
    excerpt = excerpt if provider in {"bluesky", "local_file"} else _text(excerpt)
    return {"kind": source["kind"], "id": _safe_raw(str(identifier)), "title": _safe_raw(title),
            "url": _url(url), "excerpt": _safe_raw(excerpt),
            "published_at": stamp["value"], "published_at_raw": stamp["raw"],
            "time_precision": stamp["precision"], "collected_at": collected,
            "author": _safe_raw(author) if isinstance(author, str) else None, "provider": provider,
            "evidence": {"basis": basis, "claim_state": "unverified_claim", "limitations": limitations},
            "engagement": engagement, "raw": _safe_raw(raw)}


def _packet(source, query, items, collected, limit, provider_count, invalid, endpoint,
            scope, ordering, limitations, has_more=None, **extra):
    if provider_count and invalid == provider_count:
        raise SourceError("FORMAT_ERROR", "Source returned records, but none had a usable required record shape.")
    omitted = max(0, len(items) - limit)
    items = items[:limit]
    coverage = {"scope": scope, "ordering": ordering, "requested_limit": limit,
                "returned_count": len(items), "provider_returned_count": provider_count,
                "invalid_count": invalid, "omitted_count": omitted, "partial": bool(invalid),
                "has_more": True if omitted else has_more, "limitations": limitations,
                "endpoint": public_url(endpoint) if endpoint else None, "cache_ttl_seconds": CACHE_TTL_SECONDS,
                "cache_age_seconds": 0.0, "as_of": collected}
    coverage.update(extra)
    return {"items": items, "source": source, "query": query, "cached": False,
            "collected_at": collected, "coverage": coverage}


def _file(source, limit):
    """Import user-held exports as text evidence; never execute embedded content."""
    path = Path(source["path"])
    try:
        with path.open("rb") as handle:
            body = handle.read(MAX_FILE_BYTES + 1)
    except (OSError, ValueError):
        raise SourceError("FILE_ACCESS_ERROR", "Local source export could not be read.",
                          hint="Check that the configured export exists and is readable.") from None
    if len(body) > MAX_FILE_BYTES:
        raise SourceError("RESPONSE_TOO_LARGE", "Local source export exceeds the 4 MiB import bound.",
                          hint="Split the export into smaller files.")
    collected = utc_now()
    limitations = ["User-supplied local export only; no live platform access or full-thread coverage is implied.",
                   "Collection time is import time, not publication time or first historical availability.",
                   "Claims, comments, metrics and embedded code are unverified text; code was not executed.",
                   "Relative publication labels are retained without inferring calendar dates."]
    rows, invalid, provider_count = [], 0, 0
    if path.suffix.lower() == ".jsonl":
        def reject_constant(value):
            raise ValueError("Non-JSON numeric constant")

        for line_number, line in enumerate(body.splitlines(), 1):
            try:
                decoded = line.decode("utf-8-sig")
                if not decoded.strip():
                    continue
                provider_count += 1
                record = json.loads(decoded, parse_constant=reject_constant)
            except UnicodeError:
                provider_count += 1
                invalid += 1
                continue
            except (ValueError, RecursionError):
                invalid += 1
                continue
            rows.append((line_number, record))
    else:
        try:
            text = body.decode("utf-8-sig")
        except UnicodeError:
            raise SourceError("FORMAT_ERROR", "Local text export must be UTF-8, with an optional BOM.") from None
        if not text.strip():
            raise SourceError("FORMAT_ERROR", "Local text export contains no captured text.")
        rows = [(None, {"text": text})]
        provider_count = 1
    items = []
    for line_number, record in rows:
        if not isinstance(record, dict):
            invalid += 1
            continue
        text = record.get("text") if record.get("text") is not None else record.get("content")
        if not isinstance(text, str) or not text.strip():
            invalid += 1
            continue
        text_fields = ("title", "url", "author", "published_at", "published_at_raw", "content_scope")
        if any(record.get(key) is not None and not isinstance(record[key], str) for key in text_fields):
            invalid += 1
            continue
        if any(record.get(key) is not None and (isinstance(record[key], bool)
                or not isinstance(record[key], (str, int))) for key in ("id", "parent_id")):
            invalid += 1
            continue
        if record.get("url") and _url(record["url"]) is None:
            invalid += 1
            continue
        title = record.get("title") or next(part.strip() for part in text.splitlines() if part.strip())[:160]
        identity = record.get("id")
        if identity is None or identity == "":
            identity = hashlib.sha256(json.dumps({"path": str(path), "record": record},
                                                 sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        date = record.get("published_at") or record.get("published_at_raw")
        item = _item(source, "local_file", identity, title, record.get("url"), text, date,
                     collected, record.get("author"), record, "user_supplied_export", limitations)
        if record.get("published_at_raw") is not None:
            item["published_at_raw"] = record["published_at_raw"]
        item.update(parent_id=str(record["parent_id"]) if record.get("parent_id") is not None else None,
                    source_path=str(path), source_line=line_number,
                    content_scope=record.get("content_scope") or (
                        "captured_text" if line_number is None else "user_supplied_export"))
        items.append(item)
    return _packet("local_file", None, items, collected, limit, provider_count, invalid, None,
                   "User-supplied local export", "file_order", limitations, has_more=False,
                   source_path=str(path), content_scope="user_supplied_export", cache_ttl_seconds=0,
                   imported_at=collected, max_file_bytes=MAX_FILE_BYTES)


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _child(element, *names):
    for name in names:
        for child in element:
            if _local(child.tag) == name:
                return child
    return None


def _xml_text(element):
    if element is None:
        return None
    if len(element):
        return (element.text or "") + "".join(ET.tostring(child, encoding="unicode") for child in element)
    return element.text


def _rss(source, query, limit, timeout):
    endpoint = source["url"].replace("{query}", quote_plus(query or ""))
    body, meta = request_bytes(endpoint, timeout=timeout)
    if b"<!DOCTYPE" in body.upper() or b"<!ENTITY" in body.upper():
        raise SourceError("FORMAT_ERROR", "RSS/Atom declarations with entities or document types are unsupported.")
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        raise SourceError("FORMAT_ERROR", "Source did not return complete, valid RSS or Atom XML.") from None
    atom = _local(root.tag) == "feed"
    if atom:
        rows = [node for node in root if _local(node.tag) == "entry"]
    elif _local(root.tag) == "rss" and _child(root, "channel") is not None:
        rows = [node for node in _child(root, "channel") if _local(node.tag) == "item"]
    else:
        raise SourceError("FORMAT_ERROR", "Source XML is neither an RSS 2 channel nor an Atom feed.")
    google = urlsplit(endpoint).hostname == "news.google.com"
    provider = "google_news" if google else "rss"
    limitations = ["Only the returned feed entries; no full-site or platform coverage.",
                   "Syndicated text and headlines are unverified claims; linked articles were not downloaded."]
    if google:
        limitations += ["Google News is an aggregator, not the primary publisher.",
                        "Google's feed declares personal, non-commercial feed-reader use only."]
    items, invalid = [], 0
    for row in rows:
        title = _xml_text(_child(row, "title"))
        excerpt = _xml_text(_child(row, "encoded", "content", "summary", "description"))
        link = None
        base_key = "{http://www.w3.org/XML/1998/namespace}base"
        base = urljoin(endpoint, root.get(base_key, ""))
        base = urljoin(base, row.get(base_key, ""))
        for child in row:
            if _local(child.tag) == "link" and (not atom or child.get("rel", "alternate") == "alternate"):
                link = child.get("href") if atom else child.text
                if link:
                    base = urljoin(base, child.get(base_key, ""))
                    break
        link = _url(link, base)
        native_id = _xml_text(_child(row, "id", "guid"))
        date = _xml_text(_child(row, "published", "pubDate", "date"))
        author_node = _child(row, "author", "creator")
        if atom and author_node is None:
            author_node = _child(root, "author")
        author = (_xml_text(_child(author_node, "name")) if atom and author_node is not None
                  else _xml_text(author_node))
        if not _text(title) and not _text(excerpt):
            invalid += 1
            continue
        raw = {"title": title, "link": link, "id": native_id, "content": excerpt,
               "published": date, "updated": _xml_text(_child(row, "updated")), "author": author,
               "publisher": _xml_text(_child(row, "source"))}
        identifier = native_id or link or hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
        item = _item(source, provider, identifier, title or _text(excerpt)[:160], link, excerpt,
                     date, meta["collected_at"], author, raw, "aggregator_headline" if google else "syndicated_text",
                     limitations)
        items.append(item)
    return _packet(provider, query, items, meta["collected_at"], limit, len(rows), invalid,
                   endpoint, "Google News RSS search (en-US/US)" if google else "Configured RSS/Atom feed",
                   "provider_feed_order", limitations)


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _bluesky(source, query, limit, timeout):
    endpoint = _BLUESKY + "?" + urlencode({"q": query, "sort": "latest", "limit": min(limit, 100)})
    payload, meta = request_json(endpoint, timeout=timeout)
    if not isinstance(payload, dict) or not isinstance(payload.get("posts"), list):
        raise SourceError("FORMAT_ERROR", "Bluesky response must contain a posts array.")
    limitations = ["One public AppView search page, not all social media or all Bluesky posts.",
                   "The provider may restrict search access; no authentication bypass is attempted.",
                   "Post text and engagement are not verified facts or alpha-validity scores."]
    items, invalid = [], 0
    for row in payload["posts"]:
        if not isinstance(row, dict):
            invalid += 1
            continue
        record, author, uri = row.get("record"), row.get("author"), row.get("uri")
        match = re.fullmatch(r"at://([^/]+)/app\.bsky\.feed\.post/([^/]+)", uri or "") if isinstance(uri, str) else None
        if not isinstance(record, dict) or not isinstance(record.get("text"), str) or not isinstance(author, dict) or not match:
            invalid += 1
            continue
        actor, rkey = match.groups()
        link = "https://bsky.app/profile/" + quote(actor, safe=":") + "/post/" + quote(rkey, safe="")
        engagement = {name: _count(row.get(name)) for name in ("likeCount", "replyCount", "repostCount", "quoteCount")}
        engagement["meaning"] = "Native Bluesky counts at collection; not evidential strength."
        item = _item(source, "bluesky", uri, record["text"][:160], link, record["text"], record.get("createdAt"),
                     meta["collected_at"], author.get("handle") or author.get("did"), row, "social_post",
                     limitations, engagement)
        item["author_did"] = author.get("did") if isinstance(author.get("did"), str) else None
        items.append(item)
    return _packet("bluesky", query, items, meta["collected_at"], limit, len(payload["posts"]), invalid,
                   endpoint, "Bluesky public AppView indexed posts", "latest", limitations,
                   has_more=bool(payload.get("cursor")), provider_total=_count(payload.get("hitsTotal")),
                   provider_request_limit=min(limit, 100))


def _hn(source, query, limit, timeout):
    frontpage = source["adapter"] == "hn_frontpage"
    params = {"tags": "front_page" if frontpage else "story", "hitsPerPage": min(limit, 100), "page": 0}
    if not frontpage:
        params["query"] = query
    endpoint = _HN + "?" + urlencode(params)
    payload, meta = request_json(endpoint, timeout=timeout)
    if not isinstance(payload, dict) or not isinstance(payload.get("hits"), list):
        raise SourceError("FORMAT_ERROR", "Hacker News search response must contain a hits array.")
    limitations = ["One Algolia index page of Hacker News stories; not all forums.",
                   "Points and comments describe native engagement, not evidential strength."]
    if frontpage:
        limitations.append("front_page is an index tag; response order is not the current HN front-page rank or global trends.")
    items, invalid = [], 0
    for row in payload["hits"]:
        if not isinstance(row, dict) or not isinstance(row.get("objectID"), str) or not row["objectID"].isdigit() or not _text(row.get("title")):
            invalid += 1
            continue
        discussion = "https://news.ycombinator.com/item?id=" + row["objectID"]
        engagement = {"points": _count(row.get("points")), "comments": _count(row.get("num_comments")),
                      "meaning": "Native Hacker News points and comment counts at collection."}
        item = _item(source, "hacker_news", row["objectID"], row["title"], discussion,
                     row.get("story_text"), row.get("created_at"), meta["collected_at"], row.get("author"),
                     row, "forum_submission", limitations, engagement)
        item["external_url"] = _url(row.get("url"))
        item["discussion_url"] = discussion
        items.append(item)
    pages = _count(payload.get("nbPages"))
    return _packet("hacker_news", query, items, meta["collected_at"], limit, len(payload["hits"]), invalid,
                   endpoint, "HN front_page tagged stories" if frontpage else "HN indexed story search",
                   "algolia_relevance", limitations, has_more=pages > 1 if pages is not None else None,
                   provider_total=_count(payload.get("nbHits")), provider_request_limit=min(limit, 100))


def _cache_hit(path):
    try:
        entry = cache_store.read_entry(path)
        payload = entry.get("payload")
        if (not entry["fresh"] or entry["kind"] != "research-source-v1" or entry["expires_in"] is None
                or entry["age_seconds"] is None or entry["age_seconds"] >= CACHE_TTL_SECONDS
                or not isinstance(payload, dict) or not isinstance(payload.get("items"), list)
                or not isinstance(payload.get("coverage"), dict) or payload["coverage"].get("partial")
                or not isinstance(payload.get("collected_at"), str)):
            return None
        result = copy.deepcopy(payload)
        moment = datetime.fromisoformat(result["collected_at"].replace("Z", "+00:00"))
        result["cached"] = True
        result["coverage"]["cache_age_seconds"] = max(0.0, (datetime.now(timezone.utc) - moment).total_seconds())
        result["coverage"]["as_of"] = result["collected_at"]
        return result
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return None


def fetch_source(source, query=None, limit=5, timeout=30):
    """Fetch one bounded job or raise a safe, concrete :class:`SourceError`."""
    source = _validate_definition(source)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        _invalid("Source item limit must be a positive integer.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        _invalid("Source timeout must be a finite positive number.")
    if query is not None and not isinstance(query, str):
        _invalid("Source query must be text.")
    query = query.strip() if query else None
    if requires_query(source) and not query:
        raise SourceError("QUERY_REQUIRED", "This source requires an explicit research topic.")
    if not requires_query(source):
        query = None
    adapter = source["adapter"]
    if adapter == "paper":
        _invalid("Paper retrieval is handled by the existing paper feed adapter.")
    if adapter == "file":
        return _file(source, limit)
    if adapter == "market":
        from market_sources import fetch_market
        return fetch_market(source["venue"], source["symbol"], depth=source.get("depth", 10),
                            feed=source.get("feed", "iex"), timeout=timeout)
    identity = {"version": 1, "source": source, "query": query, "limit": limit}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = Path(research_paths.cache_root()) / "sources" / (digest + ".json")
    cached = _cache_hit(path)
    if cached is not None:
        return cached
    with contextlib.ExitStack() as stack:
        cache_problem = False
        try:
            stack.enter_context(cache_store.file_lock(str(path) + ".lock", timeout=timeout))
        except (OSError, cache_store.CacheError):
            cache_problem = True
        cached = _cache_hit(path)
        if cached is not None:
            return cached
        fetcher = {"rss": _rss, "bluesky": _bluesky, "hn": _hn, "hn_frontpage": _hn}[adapter]
        packet = fetcher(source, query, limit, timeout)
        if cache_problem:
            packet["coverage"]["cache_unavailable"] = True
        if not packet["coverage"]["partial"] and not cache_problem:
            try:
                cache_store.write_json_atomic(path, cache_store.build_entry(
                    "research-source-v1", packet, ttl=CACHE_TTL_SECONDS, source=packet["source"]))
            except OSError:
                packet["coverage"]["cache_unavailable"] = True
        return packet
