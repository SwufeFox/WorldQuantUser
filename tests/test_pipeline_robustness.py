"""Exercise the shared robustness runner through both legacy entry points."""

import json
import math
from unittest import mock

import requests

import machine_lib as ml
from tools.pipeline import robustness, robustness_pnl


def _response(payload, *, status=200, retry_after=None):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode()
    if retry_after is not None:
        response.headers["Retry-After"] = str(retry_after)
    return response


def test_native_and_cumulative_keep_their_existing_yearly_rules():
    values = [0.2, 0.2, 1.0, 1.0, 1.0]
    reported = [[year, 0, 0, 0, 0, 0, value] for year, value in enumerate(values, 2020)]
    cumulative = 0.0
    pnl = [["2019-12-31", cumulative]]
    for year, value in enumerate(values, 2020):
        for day in range(1, 21):
            cumulative += value / math.sqrt(252) + (-1 if day % 2 else 1)
            pnl.append([f"{year}-01-{day:02d}", cumulative])
    session = mock.Mock()
    session.get.side_effect = [_response({"records": reported}), _response({"records": pnl})]

    original = robustness.fetch(session, "A")
    derived = robustness_pnl.yearly_sharpe(session, "A")

    assert original["years"] == derived["years"] == [str(year) for year in range(2020, 2025)]
    assert original["sharpe_by_year"] == derived["sharpe_by_year"] == values
    assert original["flags"] == []
    assert original["max_year"] == 1.0
    assert derived["flags"] == ["back_loaded"]
    assert "max_year" not in derived
    assert [call.args[0].split("/")[-1] for call in session.get.call_args_list] == ["yearly-stats", "pnl"]


def test_report_preserves_recordset_readiness_and_refuses_truncation(monkeypatch):
    session = mock.Mock()
    row = [2024, 0, 0, 0, 0, 0, 0.4]
    session.get.side_effect = [
        _response({"records": [[2024, 0, 0, 0, 0, 0, 99]]}, retry_after=1),
        _response({"records": [row], "count": 1}),
        _response({"records": [row], "count": 2}),
    ]
    wait = mock.Mock()
    monkeypatch.setattr(ml, "sleep", wait)
    assert robustness.fetch(session, "A")["sharpe_by_year"] == [0.4]
    wait.assert_called_once_with(1.0)
    assert robustness.fetch(session, "A")["error"] == "incomplete yearly-stats recordset"
    assert all(call.kwargs["allow_redirects"] is False for call in session.get.call_args_list)


def test_report_cli_reuses_login_with_isolated_sessions_and_ordered_jsonl(tmp_path, monkeypatch):
    ids_file, report = tmp_path / "ids.txt", tmp_path / "reports/years.jsonl"
    ids_file.write_text("Beta Alpha\n", encoding="utf-8")
    original = requests.Session()
    original.auth = ("user", "synthetic-password")
    original.cookies.set("session", "synthetic-cookie")
    login = mock.Mock(return_value=original)
    monkeypatch.setattr(ml, "load_session", login)
    seen = []

    def get(session, url, **kwargs):
        assert session.auth == original.auth
        assert session.cookies.get("session") == "synthetic-cookie"
        assert session.adapters["https://"] is not original.adapters["https://"]
        seen.append(session)
        session.cookies.set("session", "rotated")
        if "/Beta/" in url:
            return _response({}, status=404)
        return _response({"records": [[2024, 0, 0, 0, 0, 0, 0.4]]})

    monkeypatch.setattr(requests.Session, "get", get)
    assert robustness.main(["Alpha", "--ids-file", str(ids_file), "--out", str(report)]) == 0
    rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert [row["alpha_id"] for row in rows] == ["Alpha", "Beta"]
    assert rows[0]["min_year"] == 0.4
    assert rows[1]["error"] == "HTTP 404"
    assert len(seen) == len({id(session) for session in seen}) == 2
    assert original.cookies.get("session") == "synthetic-cookie"
    login.assert_called_once_with()
