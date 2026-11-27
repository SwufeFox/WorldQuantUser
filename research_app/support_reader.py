"""Read WorldQuant's Zendesk community and help center through BRAIN SSO."""

from __future__ import annotations

from contextlib import contextmanager
import re
from urllib.parse import parse_qs, urljoin, urlsplit

import requests

import machine_lib as ml

from .runtime import CliError


_ORIGIN = "https://support.worldquantbrain.com"
_IDENTITY = _ORIGIN + "/api/v2/users/me.json"
_SUPPORT_HOSTS = {"support.worldquantbrain.com", "worldquantbrain.zendesk.com"}
_PLATFORM_HOST = "platform.worldquantbrain.com"
_REDIRECTS = {301, 302, 303, 307, 308}
_REDIRECT_LIMIT = 8
_TIMEOUT = 12
_JSON_HEADERS = {"Accept": "application/json"}
_API_PATH = re.compile(r"/api/v2/(?:community|help_center)(?:/[A-Za-z0-9_.-]+)+")


def _send(session, url, **kwargs):
    try:
        return session.get(url, allow_redirects=False, timeout=_TIMEOUT, **kwargs)
    except requests.RequestException:
        # Request exceptions can contain the signed SSO redirect URL.
        raise CliError("SERVICE_ERROR", "could not reach BRAIN Support",
                       "check network connectivity and retry, or pass --offline if cached") from None


def _check_status(response, *, brain=False):
    status = response.status_code
    if response.headers.get("cf-mitigated", "").lower() == "challenge":
        raise CliError("SERVICE_ERROR", "BRAIN Support requires browser verification",
                       "open the BRAIN Community in your browser", details={"http_status": status})
    if brain and status in (401, 403):
        raise ml.AuthError("BRAIN rejected the session used to open Support")
    if status != 200 and status not in _REDIRECTS:
        codes = {401: "AUTH_FAILED", 403: "AUTH_FAILED", 404: "NOT_FOUND", 429: "RATE_LIMITED"}
        hints = {
            401: "retry with --relogin to acquire a fresh BRAIN session",
            403: "check account permissions or retry with --relogin",
            404: "verify the Support post, article, or category ID",
            429: "wait briefly before retrying Support requests",
        }
        raise CliError(codes.get(status, "SERVICE_ERROR"), f"BRAIN Support returned HTTP {status}",
                       hints.get(status, "retry shortly or pass --offline if cached"),
                       details={"http_status": status})


def _support_url(base, location, *, first_hop=False):
    try:
        if not isinstance(location, str) or not location.strip() or any(
            ord(character) <= 32 or character == "\\" for character in location
        ):
            raise ValueError
        url = urljoin(base, location)
        target = urlsplit(url)
        if (target.scheme != "https" or target.port not in (None, 443)
                or target.username is not None or target.password is not None or target.fragment):
            raise ValueError
        if first_hop and target.hostname == _PLATFORM_HOST:
            raise ml.AuthError("BRAIN Support requested a fresh BRAIN session")
        if target.hostname not in _SUPPORT_HOSTS:
            raise ValueError
    except ValueError:
        raise CliError("SERVICE_ERROR", "BRAIN Support returned an unexpected redirect",
                       "retry with --relogin or open BRAIN Community in a browser") from None
    return url


def _json_object(response):
    _check_status(response)
    if response.status_code in _REDIRECTS:
        raise CliError("SERVICE_ERROR", "BRAIN Support API returned a redirect",
                       "retry with --relogin to refresh the Support session",
                       details={"http_status": response.status_code})
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json" and not content_type.endswith("+json"):
        raise CliError("SERVICE_ERROR", "BRAIN Support API returned a non-JSON response",
                       "retry with --relogin or check whether Support requires browser verification")
    try:
        payload = response.json()
    except ValueError:
        raise CliError("SERVICE_ERROR", "BRAIN Support API returned malformed JSON",
                       "retry the Support request shortly") from None
    if not isinstance(payload, dict):
        raise CliError("SERVICE_ERROR", "BRAIN Support API returned an unexpected JSON object",
                       "retry the Support request shortly")
    return payload


@contextmanager
def with_support(brain_session):
    """Yield a verified Zendesk session with cookies acquired through SSO."""
    url = ml.BRAIN_API_URL + "/authentication/support"
    response = _send(brain_session, url, params={"return_to": _IDENTITY})
    _check_status(response, brain=True)
    if response.status_code not in _REDIRECTS:
        raise CliError("SERVICE_ERROR", "BRAIN Support did not return an SSO redirect",
                       "retry with --relogin or verify BRAIN Support access in a browser")
    # The BRAIN client may use Basic Auth and domainless cookies. Follow the
    # signed redirect with a fresh session to keep those credentials on BRAIN.
    with requests.Session() as support_session:
        for hop in range(_REDIRECT_LIMIT):
            if response.status_code not in _REDIRECTS:
                break
            url = _support_url(url, response.headers.get("Location"), first_hop=hop == 0)
            headers = _JSON_HEADERS if url == _IDENTITY else None
            response = _send(support_session, url, headers=headers)
            _check_status(response)
        if response.status_code in _REDIRECTS:
            raise CliError("SERVICE_ERROR", "BRAIN Support SSO exceeded its redirect limit",
                           "retry with --relogin")
        if url != _IDENTITY:
            response = _send(support_session, _IDENTITY, headers=_JSON_HEADERS)
        user = _json_object(response).get("user")
        if (not isinstance(user, dict) or type(user.get("id")) is not int or user["id"] <= 0
                or not user.get("role") or user["role"] == "anonymous"):
            raise CliError("AUTH_FAILED", "BRAIN Support SSO did not produce an authenticated session",
                           "retry with --relogin to refresh your BRAIN session")
        yield support_session


def support_get(session, path, params=None):
    """Get a JSON object from a fixed read-only Zendesk API path."""
    if (not isinstance(path, str) or not _API_PATH.fullmatch(path)
            or any(part in (".", "..") for part in path.split("/"))):
        raise CliError("INVALID_INPUT", "invalid BRAIN Support API path",
                       "use the `docs` or `forum` CLI subcommands with valid numeric IDs")
    return _json_object(_send(session, _ORIGIN + path, params=params, headers=_JSON_HEADERS))


def support_pagination(data):
    """Expose the next cursor or page number while keeping redirect URLs private."""
    meta = data.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("has_more"), bool):
        cursor = meta.get("after_cursor")
        return {"has_more": meta["has_more"],
                "after_cursor": cursor if meta["has_more"] and isinstance(cursor, str) and cursor else None,
                "next_page": None}
    page = None
    link = data.get("next_page")
    if isinstance(link, str) and link:
        try:
            url = urlsplit(_support_url(_ORIGIN, link))
            pages = parse_qs(url.query, max_num_fields=32).get("page", [])
            if _API_PATH.fullmatch(url.path) and len(pages) == 1 and re.fullmatch(r"[0-9]{1,9}", pages[0]):
                page = int(pages[0]) or None
        except (CliError, ValueError):
            pass
    return {"has_more": page is not None, "after_cursor": None, "next_page": page}
