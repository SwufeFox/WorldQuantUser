"""Offline regressions for BRAIN documents and isolated Zendesk reads."""

import json
from types import SimpleNamespace
from unittest import mock
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

import machine_lib as ml
from research_app import docs_reader, support_reader
from research_app.runtime import CliError

SUPPORT = "https://support.worldquantbrain.com"
IDENTITY = SUPPORT + "/api/v2/users/me.json"
BODY_PATH = "/api/v2/community/posts/42.json"
SIGNED_ENTRY = "https://worldquantbrain.zendesk.com/access/jwt?jwt=PRIVATE_FIXTURE"


def response(status=200, payload=None, location=None):
    reply = requests.Response()
    reply.status_code = status
    reply._content = json.dumps(payload).encode()
    reply._content_consumed = True
    reply.headers["Content-Type"] = "application/json"
    if location is not None:
        reply.headers["Location"] = location
    return reply


def test_support_sso_keeps_brain_credentials_on_brain_origin():
    post = {"post": {"id": 42, "details": "<p>Sample <code>x</code></p>"}}
    exchanges = iter([
        response(302, location=SIGNED_ENTRY),
        response(302, location=IDENTITY),
        response(payload={"user": {"id": 7, "role": "end-user"}}),
        response(payload=post),
    ])
    requests_seen = []

    def send(request, **kwargs):
        requests_seen.append(request)
        assert kwargs["timeout"] > 0
        reply = next(exchanges)
        reply.url, reply.request = request.url, request
        return reply

    with requests.Session() as brain, mock.patch.object(requests.adapters.HTTPAdapter, "send", side_effect=send):
        brain.auth = ("fixture-user", "fixture-password")
        brain.cookies.set("brain-session", "fixture-cookie")
        with support_reader.with_support(brain) as forum:
            assert forum is not brain
            assert support_reader.support_get(forum, BODY_PATH) == post

    entry, *forum_requests = requests_seen
    entry_url = urlsplit(entry.url)
    assert entry_url.hostname == "api.worldquantbrain.com"
    assert entry_url.path == "/authentication/support"
    assert parse_qs(entry_url.query) == {"return_to": [IDENTITY]}
    assert entry.headers["Authorization"].startswith("Basic ")
    assert entry.headers["Cookie"] == "brain-session=fixture-cookie"
    assert [request.url for request in forum_requests] == [SIGNED_ENTRY, IDENTITY, SUPPORT + BODY_PATH]
    assert all(request.method == "GET" for request in requests_seen)
    assert all("Authorization" not in request.headers and "Cookie" not in request.headers
               for request in forum_requests)


@pytest.mark.parametrize("destination", [
    "https://outside.invalid/login?token=PRIVATE_FIXTURE",
    "http://support.worldquantbrain.com/login?token=PRIVATE_FIXTURE",
    "https://support.worldquantbrain.com@outside.invalid/login?token=PRIVATE_FIXTURE",
    "https://support.worldquantbrain.com:9443/login?token=PRIVATE_FIXTURE",
])
def test_support_sso_blocks_unapproved_redirects_without_echoing_them(destination):
    brain = mock.Mock()
    brain.get.return_value = response(302, location=destination)
    with (mock.patch.object(requests.adapters.HTTPAdapter, "send") as send,
          pytest.raises(CliError) as error, support_reader.with_support(brain)):
        pytest.fail("an unapproved redirect established a support session")
    send.assert_not_called()
    assert "PRIVATE_FIXTURE" not in str(error.value)
    assert "PRIVATE_FIXTURE" not in repr(error.value.details)


def test_transport_error_during_sso_does_not_expose_signed_location():
    brain = mock.Mock()
    brain.get.return_value = response(302, location=SIGNED_ENTRY)
    failure = requests.ConnectionError(f"unable to connect to {SIGNED_ENTRY}")
    with (mock.patch.object(requests.adapters.HTTPAdapter, "send", side_effect=failure),
          pytest.raises(CliError) as error, support_reader.with_support(brain)):
        pytest.fail("a failed redirect established a support session")
    assert error.value.code == "SERVICE_ERROR"
    assert "PRIVATE_FIXTURE" not in str(error.value)
    assert "PRIVATE_FIXTURE" not in repr(error.value.details)


@pytest.mark.parametrize(("payload", "expected"), [
    ({"meta": {"has_more": True, "after_cursor": "cursor-42"}},
     {"has_more": True, "after_cursor": "cursor-42", "next_page": None}),
    ({"next_page": SUPPORT + "/api/v2/help_center/community_posts/search.json?page=2&query=topic"},
     {"has_more": True, "after_cursor": None, "next_page": 2}),
    ({"next_page": "https://outside.invalid/api/v2/community/posts.json?page=2"},
     {"has_more": False, "after_cursor": None, "next_page": None}),
])
def test_support_pagination_returns_only_a_trusted_continuation(payload, expected):
    assert support_reader.support_pagination(payload) == expected


def test_platform_doc_get_preserves_structure_and_renders_canonical_link():
    session = object()
    slug = "test-research-workflow"
    detail = {
        "id": slug,
        "title": "Workflow",
        "category": "QUICK_START",
        "lastModified": "2026-11-17T12:00:00Z",
        "content": [
            {"type": "TEXT", "content": "<p>A table follows.</p>"},
            {"type": "TABLE", "content": [["Step", "Result"], ["Run", "OK"]]},
            {"type": "NEW_BLOCK", "content": {"sample": "preserve unknown block types"}},
        ],
    }
    directory = {"count": 1, "next": None, "results": [
        {"id": "getting-started", "title": "Getting Started", "category": "QUICK_START",
         "pages": [{"id": slug, "title": "Workflow"}]},
    ]}
    with mock.patch.object(ml, "load_session", return_value=session) as load, \
            mock.patch.object(ml, "request_json", side_effect=[detail, directory]) as request:
        result = docs_reader.cmd_docs_get(SimpleNamespace(slug=slug), None)
    assert load.call_count == 1
    assert [call.args[:3] for call in request.call_args_list] == [
        (session, "GET", f"/tutorial-pages/{slug}"), (session, "GET", "/tutorials"),
    ]
    assert {key: result[key] for key in detail} == detail
    assert result["url"].endswith(f"/learn/documentation/getting-started/{slug}")
    assert result["api_url"] == f"https://api.worldquantbrain.com/tutorial-pages/{slug}"


def test_platform_doc_list_casefolds_query_and_omits_unmatched_pages():
    session = object()
    directory = {"count": 1, "next": None, "results": [
        {"id": "getting-started", "title": "Getting Started", "category": "QUICK_START",
         "pages": [{"id": "workflow", "title": "Strategy Workflow"},
                   {"id": "glossary", "title": "Glossary"}]},
    ]}
    with mock.patch.object(ml, "load_session", return_value=session), \
            mock.patch.object(ml, "request_json", return_value=directory) as request:
        result = docs_reader.cmd_docs_list(SimpleNamespace(query="WORKFLOW"), None)
    assert request.call_args.args[:3] == (session, "GET", "/tutorials")
    assert request.call_args.kwargs["params"] == {"limit": 50}
    assert result["category_count"] == result["page_count"] == 1
    assert [page["id"] for page in result["categories"][0]["pages"]] == ["workflow"]


def test_platform_doc_slug_rejects_an_absolute_url_before_authentication():
    with mock.patch.object(ml, "load_session") as load, pytest.raises(CliError) as error:
        docs_reader.cmd_docs_get(SimpleNamespace(slug="https://outside.invalid/file"), None)
    load.assert_not_called()
    assert error.value.code == "INVALID_INPUT"
