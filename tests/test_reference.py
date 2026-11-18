"""Offline regressions for BRAIN documents and isolated Zendesk reads."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

import cache_store
import machine_lib as ml
import research_paths
from research_app import docs_reader, reference, support_reader
from research_app.runtime import CliError, OutputConflict

SUPPORT = "https://support.worldquantbrain.com"
IDENTITY = SUPPORT + "/api/v2/users/me.json"
BODY_PATH = "/api/v2/community/posts/42.json"
SIGNED_ENTRY = "https://worldquantbrain.zendesk.com/access/jwt?jwt=PRIVATE_FIXTURE"


@pytest.fixture(autouse=True)
def reference_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv(research_paths.ENV_VAR, str(tmp_path))
    monkeypatch.setenv("BRAIN_USERNAME", "fixture-reference-account")
    monkeypatch.setattr(ml, "_load_dotenv", lambda: None)


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


def test_platform_docs_share_catalog_across_queries_and_canonical_urls():
    detail = {"id": "workflow", "content": []}
    directory = {"count": 1, "next": None, "results": [
        {"id": "getting-started", "pages": [
            {"id": "workflow", "title": "Workflow"}, {"id": "glossary", "title": "Glossary"}]},
    ]}
    with mock.patch.object(ml, "load_session", return_value=object()) as load, \
            mock.patch.object(ml, "request_json", side_effect=[directory, detail]) as request:
        first = docs_reader.cmd_docs_list(SimpleNamespace(query="workflow"), None)
        other = docs_reader.cmd_docs_list(SimpleNamespace(query="glossary", offline=True), None)
        article = docs_reader.cmd_docs_get(SimpleNamespace(slug="workflow"), None)
        warm = docs_reader.cmd_docs_get(SimpleNamespace(slug="workflow", offline=True), None)

    assert first["cache"]["hit"] is False
    assert other["cache"]["hit"] is True
    assert [p["id"] for p in other["categories"][0]["pages"]] == ["glossary"]
    assert article["url"].endswith("/getting-started/workflow")
    assert warm["url"] == article["url"]
    assert warm["cache"]["hit"] is True
    assert load.call_count == 2    # catalog and article, neither hit
    assert [call.args[2] for call in request.call_args_list] == ["/tutorials", "/tutorial-pages/workflow"]


def test_doc_save_cannot_overwrite_the_shared_catalog(monkeypatch, tmp_path):
    load = mock.Mock(side_effect=AssertionError("cache output is protected before login"))
    monkeypatch.setattr(ml, "load_session", load)
    catalog = tmp_path / "cache/metadata/reference-brain-tutorial-catalog/existing.json"
    opts = SimpleNamespace(slug="workflow", save=str(catalog))
    with pytest.raises(OutputConflict):
        docs_reader.cmd_docs_get(opts, None)
    load.assert_not_called()


@pytest.fixture
def support_transport(monkeypatch):
    load = mock.Mock(return_value=object())
    sso = mock.MagicMock()
    get = mock.Mock()
    monkeypatch.setattr(ml, "load_session", load)
    monkeypatch.setattr(reference, "with_support", sso)
    monkeypatch.setattr(reference, "support_get", get)
    return load, sso, get


def test_forum_cache_is_lazy_and_account_scoped(support_transport, monkeypatch, tmp_path):
    load, sso, get = support_transport
    get.side_effect = [
        {"post": {"id": 42, "details": "<p>Account one</p>", "extra_private": "IGNORE_ME"}},
        {"post": {"id": 42, "details": "<p>Account two</p>"}},
    ]
    opts = SimpleNamespace(post_id="42")
    cached_only = SimpleNamespace(post_id="42", offline=True)
    assert reference.cmd_forum_get(opts, None)["cache"]["hit"] is False
    assert reference.cmd_forum_get(cached_only, None)["post"]["details"] == "<p>Account one</p>"
    assert (load.call_count, sso.call_count, get.call_count) == (1, 1, 1)

    monkeypatch.setenv("BRAIN_USERNAME", "second-reference-account")
    with pytest.raises(CliError) as error:
        reference.cmd_forum_get(cached_only, None)
    assert error.value.code == "NOT_FOUND"
    assert get.call_count == 1
    assert reference.cmd_forum_get(opts, None)["post"]["details"] == "<p>Account two</p>"
    assert load.call_args.kwargs["expected_username"] == "second-reference-account"

    cache_files = list(tmp_path.glob("cache/metadata/reference-support/*.json"))
    assert len(cache_files) == 2
    for path in cache_files:
        saved = path.read_text()
        assert "fixture-reference-account" not in saved
        assert "second-reference-account" not in saved
        assert "IGNORE_ME" not in saved


def test_forum_pagination_and_refresh_use_exact_cache_keys(support_transport):
    load, sso, get = support_transport
    get.side_effect = [{"results": [{"id": value}]} for value in (1, 2, 3, 4)]

    def search(page, **kwargs):
        return reference.cmd_forum_search(SimpleNamespace(query="alpha", limit=10, page=page, **kwargs), None)

    assert search(1)["results"] == [{"id": 1}]
    assert search(2)["results"] == [{"id": 2}]
    assert search(1, refresh_meta=True)["results"] == [{"id": 3}]
    assert search(1, no_cache=True)["results"] == [{"id": 4}]
    assert search(1, offline=True)["results"] == [{"id": 3}]
    assert search(2, offline=True)["results"] == [{"id": 2}]
    assert [call.kwargs["params"]["page"] for call in get.call_args_list] == [1, 2, 1, 1]
    assert load.call_args.kwargs["use_cache"] is False
    assert sso.call_count == 4


def test_offline_cache_miss_and_expiry_never_authenticate(support_transport, monkeypatch):
    load, _, get = support_transport
    current = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(cache_store, "utc_now", lambda: current)
    get.side_effect = [{"article": {"id": 42, "body": text}} for text in ("first", "second")]
    opts = SimpleNamespace(article_id="42", meta_max_age=10)
    offline = SimpleNamespace(article_id="42", meta_max_age=10, offline=True)

    with pytest.raises(CliError) as missing:
        reference.cmd_docs_article(offline, None)
    assert missing.value.code == "NOT_FOUND"
    load.assert_not_called()
    assert reference.cmd_docs_article(opts, None)["article"]["body"] == "first"

    current += timedelta(seconds=11)
    with pytest.raises(CliError) as expired:
        reference.cmd_docs_article(offline, None)
    assert expired.value.code == "NOT_FOUND"
    assert load.call_count == 1
    assert reference.cmd_docs_article(opts, None)["article"]["body"] == "second"
    assert load.call_count == 2


@pytest.mark.parametrize("flag", ["refresh_meta", "relogin", "no_cache"])
def test_offline_cache_rejects_network_flags(support_transport, flag):
    load, sso, get = support_transport
    opts = SimpleNamespace(post_id="42", offline=True, **{flag: True})
    with pytest.raises(CliError) as error:
        reference.cmd_forum_get(opts, None)
    assert error.value.code == "INVALID_INPUT"
    load.assert_not_called()
    sso.assert_not_called()
    get.assert_not_called()


def test_invalid_forum_response_is_not_cached(support_transport):
    load, _, get = support_transport
    opts = SimpleNamespace(post_id="42")
    get.side_effect = [{"post": None}, {"post": {"id": 42, "details": "example"}}]

    with pytest.raises(CliError) as invalid:
        reference.cmd_forum_get(opts, None)
    assert invalid.value.code == "SERVICE_ERROR"
    assert reference.cmd_forum_get(opts, None)["cache"]["hit"] is False
    assert reference.cmd_forum_get(opts, None)["cache"]["hit"] is True
    assert load.call_count == 2


def test_corrupted_cached_data_is_an_offline_miss(support_transport, tmp_path):
    load, _, get = support_transport
    get.return_value = {"post": {"id": 42, "details": "valid"}}
    reference.cmd_forum_get(SimpleNamespace(post_id="42"), None)
    path, = tmp_path.glob("cache/metadata/reference-support/*.json")
    entry = json.loads(path.read_text())
    entry["payload"]["data"] = {}
    path.write_text(json.dumps(entry))

    with pytest.raises(CliError) as missing:
        reference.cmd_forum_get(SimpleNamespace(post_id="42", offline=True), None)
    assert missing.value.code == "NOT_FOUND"
    assert load.call_count == 1
