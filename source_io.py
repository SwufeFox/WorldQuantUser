"""Bounded, read-only HTTP and provenance helpers for research sources."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import urllib.error
import urllib.parse
import urllib.request


MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_SECRET_PARAMETERS = frozenset({"api_key", "apikey", "key", "token", "access_token", "auth", "signature"})
_TRACKING_PARAMETERS = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid"})


class SourceError(RuntimeError):
    """A source failure whose public fields cannot contain response bodies or keys."""

    def __init__(self, code, message, hint=None, retry_after_seconds=None):
        super().__init__(message)
        self.code = code
        self.hint = hint
        self.retry_after_seconds = retry_after_seconds

    def public(self):
        result = {"code": self.code, "message": str(self)}
        if self.hint:
            result["hint"] = self.hint
        if self.retry_after_seconds is not None:
            result["retry_after_seconds"] = self.retry_after_seconds
        return result


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_url(value):
    if not isinstance(value, str) or any(character.isspace() for character in value):
        raise SourceError("INVALID_INPUT", "Source URLs must be HTTP(S) URLs without whitespace.")
    try:
        parts = urllib.parse.urlsplit(value)
        valid = (parts.scheme.lower() in {"http", "https"} and parts.hostname
                 and parts.username is None and parts.password is None)
        parts.port
    except ValueError:
        valid = False
    if not valid:
        raise SourceError("INVALID_INPUT", "Source URLs must be HTTP(S) URLs without embedded credentials.")
    return value


def canonical_url(value):
    parts = urllib.parse.urlsplit(validate_url(value))
    query = [(key, item) for key, item in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
             if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_PARAMETERS]
    return urllib.parse.urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                                   parts.path or "/", urllib.parse.urlencode(query), ""))


def public_url(value):
    parts = urllib.parse.urlsplit(canonical_url(value))
    query = [(key, "[redacted]" if key.lower() in _SECRET_PARAMETERS else item)
             for key, item in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)]
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path,
                                   urllib.parse.urlencode(query), ""))


def parse_timestamp(value):
    """Retain date precision; never replace an unknown source time with collection time."""
    raw = value.strip() if isinstance(value, str) else None
    result = {"value": None, "raw": raw, "precision": "unknown"}
    if not raw:
        return result
    if len(raw) == 4 and raw.isdigit():
        result["precision"] = "year"
        return result
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(raw)
        except (ValueError, TypeError, OverflowError):
            return result
    if len(raw) == 10:
        result["precision"] = "date"
    elif parsed.tzinfo is not None:
        result.update(value=parsed.astimezone(timezone.utc).isoformat(), precision="timestamp")
    return result


def _retry_after(headers):
    value = headers.get("Retry-After") if headers else None
    if value is None:
        return None
    try:
        number = float(value)
        return max(0.0, number) if math.isfinite(number) else None
    except (TypeError, ValueError):
        parsed = parse_timestamp(value)["value"]
        if parsed:
            return max(0.0, (datetime.fromisoformat(parsed) - datetime.now(timezone.utc)).total_seconds())
    return None


def request_bytes(url, timeout=30, headers=None, max_bytes=MAX_RESPONSE_BYTES):
    """GET one bounded response; caller-supplied auth headers do not follow redirects."""
    validate_url(url)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise SourceError("INVALID_INPUT", "Source timeout must be finite and positive.")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise SourceError("INVALID_INPUT", "Response size bound must be a positive integer.")
    request = urllib.request.Request(url, headers={"User-Agent": "WorldQuantUser/research-sources"})
    for name, value in (headers or {}).items():
        if (not isinstance(name, str) or not isinstance(value, str)
                or any(character in name + value for character in "\r\n")):
            raise SourceError("INVALID_INPUT", "Invalid source request header.")
        request.add_unredirected_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(max_bytes + 1)
            content_type = (getattr(response, "headers", None) or {}).get("Content-Type")
    except urllib.error.HTTPError as error:
        code = ({401: "AUTH_FAILED", 403: "AUTH_FAILED", 404: "NOT_FOUND", 429: "RATE_LIMITED"}
                .get(error.code, "SERVICE_ERROR"))
        raise SourceError(code, "Source returned HTTP %d." % error.code,
                          "Check this source's access requirements or retry later.",
                          _retry_after(error.headers)) from None
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        raise SourceError("SERVICE_ERROR", "Source request could not be completed.",
                          "Check connectivity and the source's access requirements.") from None
    if len(body) > max_bytes:
        raise SourceError("SERVICE_ERROR", "Source response exceeds the configured size bound.")
    return body, {"url": public_url(url), "collected_at": utc_now(), "content_type": content_type}


def _finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def _invalid_json_constant(value):
    raise ValueError("non-standard JSON constant")


def request_json(url, timeout=30, headers=None, max_bytes=MAX_RESPONSE_BYTES):
    body, evidence = request_bytes(url, timeout=timeout, headers=headers, max_bytes=max_bytes)
    try:
        payload = json.loads(body.decode("utf-8-sig"), parse_float=_finite_json_float,
                             parse_constant=_invalid_json_constant)
    except (ValueError, UnicodeError):
        raise SourceError("SERVICE_ERROR", "Source returned invalid JSON.") from None
    return payload, evidence
