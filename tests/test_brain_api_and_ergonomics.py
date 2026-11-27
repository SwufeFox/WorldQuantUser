"""Offline tests for WorldQuant BRAIN API wiring, BrainClient, and CLI ergonomics."""

from __future__ import annotations

import contextlib
from datetime import date, timedelta
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import alpha_registry
import cache_store
import machine_lib as ml
import simulation_slots
from research_app import app as cli_app
from research_app import alphas
from research_app import promote as cli_promote
from research_app.runtime import CliError


def invoke_cli(*arguments: str) -> tuple[int, dict, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = cli_app.main(list(arguments))
    return code, json.loads(stdout.getvalue()), stderr.getvalue()


class FakeResponse:
    def __init__(self, status_code=200, body=None, *, headers=None, text=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.headers = dict(headers or {})
        self._body = body
        if text is not None:
            self.text = text
        elif body is None:
            self.text = ""
        else:
            self.text = json.dumps(body)
        self.content = self.text.encode("utf-8")

    def json(self):
        if self._body is None and not self.text.strip():
            raise ValueError("empty body")
        return self._body if self._body is not None else json.loads(self.text)

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, routes=None, *, account_id="USER123"):
        self.routes = dict(routes or {})
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []
        self._brain_account_id = account_id
        self.auth = ("researcher@example.com", "secret")
        self.closed = False

    def _dispatch(self, method, url, params=None, json_body=None):
        self.calls.append((method.upper(), url, params, json_body))
        path = url.replace(ml.BRAIN_API_URL, "")
        key = (method.upper(), path)
        handler = self.routes.get(key) or self.routes.get(path)
        if handler is None:
            return FakeResponse(404, {"detail": f"unhandled {method} {path}"})
        if isinstance(handler, list):
            item = handler.pop(0) if len(handler) > 1 else handler[0]
            return item(params, json_body) if callable(item) else item
        return handler(params, json_body) if callable(handler) else handler

    def request(self, method, url, params=None, json=None, timeout=None, **kwargs):
        return self._dispatch(method, url, params=params, json_body=json)

    def get(self, url, params=None, timeout=None, **kwargs):
        return self._dispatch("GET", url, params=params)

    def patch(self, url, json=None, timeout=None, **kwargs):
        return self._dispatch("PATCH", url, json_body=json)

    def delete(self, url, timeout=None, **kwargs):
        return self._dispatch("DELETE", url)

    def close(self):
        self.closed = True


def test_account_profile_competitions_consultant_and_remote_logout(tmp_path):
    cache_file = tmp_path / "session.json"
    entry = cache_store.build_entry(
        "brain-session",
        {"cookies": {"t": "tok123"}, "user_id": "WQ_U1", "username": "researcher@example.com"},
        ttl=3600,
        source="login",
    )
    cache_store.write_json_atomic(cache_file, entry, mode=0o600)

    session = FakeSession({
        ("GET", "/users/self"): FakeResponse(200, {"id": "WQ_U1", "country": "US"}),
        ("GET", "/users/self/consultant"): FakeResponse(200, {"level": "GOLD"}),
        ("GET", "/users/self/pyramids"): FakeResponse(403, {"detail": "level gated"}),
        ("GET", "/users/self/activities"): FakeResponse(404, {"detail": "absent"}),
        ("GET", "/users/self/competitions"): FakeResponse(200, {"results": [{"id": "iac", "name": "IQC"}]}),
        ("GET", "/competitions"): FakeResponse(200, {"competitions": [{"id": "iac"}, {"id": "pp"}]}),
        ("GET", "/competitions/pp"): FakeResponse(200, {"id": "pp", "status": "ACTIVE"}),
        ("DELETE", "/authentication"): FakeResponse(204),
    })

    with mock.patch.object(ml, "load_session", return_value=session):
        code, doc, _ = invoke_cli("auth", "profile", "--consultant", "--research-dir", str(tmp_path))
        assert code == 0
        assert doc["data"]["profile"]["id"] == "WQ_U1"
        assert doc["data"]["surfaces"]["consultant"]["state"] == "available"
        assert doc["data"]["surfaces"]["pyramids"]["state"] == "level_gated"
        assert doc["data"]["surfaces"]["activities"]["state"] == "absent"

        code, doc, _ = invoke_cli("auth", "competitions", "--research-dir", str(tmp_path))
        assert code == 0
        assert doc["data"]["scope"] == "enrolled"
        assert doc["data"]["count"] == 1

        code, doc, _ = invoke_cli("auth", "competitions", "--all", "--research-dir", str(tmp_path))
        assert code == 0
        assert doc["data"]["scope"] == "all"
        assert doc["data"]["count"] == 2

        code, doc, _ = invoke_cli("auth", "competitions", "--id", "pp", "--research-dir", str(tmp_path))
        assert code == 0
        assert doc["data"]["competition"]["id"] == "pp"

    res = ml.logout_session(session, cache_path=cache_file, remote=True)
    assert res["logged_out"] is True
    assert res["remote_revoked"] is True
    assert res["http_status"] == 204
    assert not cache_file.exists()


def test_catalog_point_lookups_and_positional_dataset(tmp_path):
    session = FakeSession({
        ("GET", "/data-sets/fundamental6"): FakeResponse(
            200, {"id": "fundamental6", "name": "Fundamental 6", "fieldCount": 120}
        ),
        ("GET", "/data-fields/fnd6_newa1v1300_assets"): FakeResponse(
            200,
            {
                "id": "fnd6_newa1v1300_assets",
                "type": "MATRIX",
                "dataset": {"id": "fundamental6"},
                "description": "Total corporate assets",
                "coverage": 0.94,
            },
        ),
        ("GET", "/data-fields"): FakeResponse(
            200,
            {
                "count": 1,
                "results": [
                    {
                        "id": "fnd6_newa1v1300_assets",
                        "type": "MATRIX",
                        "description": "Total corporate assets",
                        "coverage": 0.94,
                    }
                ],
            },
        ),
    })

    with mock.patch.object(ml, "load_session", return_value=session), \
            mock.patch.object(ml, "configured_username", return_value="researcher@example.com"):
        code, ds_doc, _ = invoke_cli("catalog", "dataset", "fundamental6", "--research-dir", str(tmp_path))
        assert code == 0
        assert ds_doc["data"]["dataset"]["id"] == "fundamental6"

        code, fld_doc, _ = invoke_cli("catalog", "field", "fnd6_newa1v1300_assets", "--research-dir", str(tmp_path))
        assert code == 0
        assert fld_doc["data"]["field"]["dataset"]["id"] == "fundamental6"
        assert fld_doc["data"]["expression"] == "winsorize(ts_backfill(fnd6_newa1v1300_assets, 120), std=4)"

        code, fields_doc, _ = invoke_cli("catalog", "fields", "fundamental6", "--expressions", "--research-dir", str(tmp_path))
        assert code == 0
        assert fields_doc["data"]["dataset"] == "fundamental6"
        assert fields_doc["data"]["count"] == 1

        code, err_doc, _ = invoke_cli(
            "catalog", "fields", "fundamental6", "--dataset", "analyst4", "--research-dir", str(tmp_path)
        )
        assert code == 2
        assert err_doc["error"]["code"] == "USAGE"


def test_correlations_archive_roundtrip_and_status_errors(tmp_path):
    prod_payload = {
        "schema": {
            "name": "prodCorrelation",
            "properties": [{"name": "min"}, {"name": "max"}, {"name": "alphas"}],
        },
        "max": 0.62,
        "min": -0.35,
        "records": [[-1.0, -0.3, 1], [-0.3, 0.5, 14], [0.5, 0.7, 3], [0.7, 1.0, 0]],
    }
    pp_payload = {
        "schema": {"name": "powerPoolCorrelation"},
        "max": 0.44,
        "min": -0.2,
        "records": [["PP1", "peer", "ACTIVE", "USA", "TOP3000", 0.44, 1.8]],
    }
    session = FakeSession({
        ("GET", "/alphas/A100"): FakeResponse(
            200,
            {
                "id": "A100",
                "status": "UNSUBMITTED",
                "regular": {"code": "rank(close)", "operatorCount": 1},
                "settings": dict(ml.DEFAULT_SETTINGS),
                "is": {"sharpe": 1.6, "fitness": 1.2, "turnover": 0.15, "checks": []},
            },
        ),
        ("GET", "/alphas/A100/correlations/prod"): [
            FakeResponse(200, None, headers={"Retry-After": "0.01"}, text=""),
            FakeResponse(200, prod_payload),
        ],
        ("GET", "/alphas/A100/correlations/power-pool"): FakeResponse(200, pp_payload),
        ("GET", "/alphas/A100/correlations/self"): FakeResponse(
            200, {"max": 0.31, "min": -0.1, "records": [["B1", "book", "ACTIVE", "USA", "TOP3000", 0.31, 1.5]]}
        ),
        ("GET", "/alphas/GATED/correlations/self"): FakeResponse(200, {"max": 0.2, "min": 0.0, "records": []}),
        ("GET", "/alphas/GATED/correlations/prod"): FakeResponse(403, {"detail": "level gated"}),
        ("GET", "/alphas/GATED/correlations/power-pool"): FakeResponse(200, pp_payload),
        ("GET", "/alphas/GATED"): FakeResponse(
            200, {"id": "GATED", "status": "UNSUBMITTED", "regular": "rank(close)", "is": {"sharpe": 1.4}}
        ),
        ("GET", "/alphas/MISSING/correlations/self"): FakeResponse(404, {"detail": "not found"}),
    })

    with pytest.raises(ml.NotFound):
        ml.fetch_self_correlation(session, "MISSING", tries=2, max_wait=0.1)

    archive_file = tmp_path / "pc_archive.json"
    with mock.patch.object(ml, "load_session", return_value=session):
        code, prod_doc, _ = invoke_cli(
            "alpha", "prod-corr", "A100", "--save-archive", str(archive_file),
            "--research-dir", str(tmp_path),
        )
        assert code == 0
        assert prod_doc["data"]["max_correlation"] == 0.62
        assert prod_doc["data"]["count_in_bins_at_or_above_limit"] == 0
        assert archive_file.is_file()

        # Verify archive is accepted verbatim by promote._read_production_distribution!
        reviewed = cli_promote._read_production_distribution(archive_file, "A100", 0.7)
        assert reviewed["state"] == "available"
        assert reviewed["max"] == 0.62
        assert reviewed["count_in_bins_at_or_above_formal_limit"] == 0

        code, pp_doc, _ = invoke_cli("alpha", "pp-corr", "A100", "--research-dir", str(tmp_path))
        assert code == 0
        assert pp_doc["data"]["verdict"] == "clear"
        assert pp_doc["data"]["blocked"] is False

        # Strict < 0.5 check: 0.50 must block!
        strict_verdict = ml.power_pool_correlation_verdict({"max": 0.5, "records": []}, limit=0.5)
        assert strict_verdict["blocked"] is True

        # alpha get --correlations handles level-gated prod gracefully alongside self & power-pool
        code, get_doc, _ = invoke_cli("alpha", "get", "GATED", "--correlations", "--research-dir", str(tmp_path))
        assert code == 0
        corrs = get_doc["data"]["correlations"]
        assert corrs["self"]["state"] == "available"
        assert corrs["prod"]["state"] == "level_gated"
        assert corrs["power-pool"]["state"] == "available"


def test_cancel_simulation_releases_local_slot(tmp_path):
    with mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": str(tmp_path)}):
        session = FakeSession({
            ("DELETE", "/simulations/SIM999"): FakeResponse(204),
        }, account_id="ACCT_SLOT")
        import simulation_capabilities

        account = simulation_capabilities.account_provenance(session)
        progress_url = f"{ml.BRAIN_API_URL}/simulations/SIM999"
        record = {}

        def save_intent(reservation):
            record["slot_reservation"] = reservation

        identity, _ = simulation_slots.prepare_submission(account, 3, save_intent)
        assert identity is not None
        record["progress_url"] = progress_url
        simulation_slots.reconcile_record(record, account, outcome="accepted")
        assert simulation_slots.confirmed_accepted_count(account) == 1

        session.routes[("DELETE", "/simulations/SIM999")] = FakeResponse(202)
        pending = ml.cancel_simulation(session, "SIM999")
        assert pending["state"] == "cancellation_pending"
        assert pending["released_reservations"] == []
        assert simulation_slots.confirmed_accepted_count(account) == 1

        session.routes[("DELETE", "/simulations/SIM999")] = FakeResponse(204)
        with mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": str(tmp_path)}), \
                mock.patch.object(ml, "load_session", return_value=session):
            canceled = alphas.cmd_alpha_cancel_sim(
                SimpleNamespace(reference="SIM999", no_cache=False, relogin=False, meta_max_age=None),
                None,
            )
        assert canceled["state"] == "cancelled"
        assert canceled["cancelled"] is True
        assert identity["id"] in canceled["released_reservations"]
        assert simulation_slots.confirmed_accepted_count(account) == 0


def test_alpha_readiness_composes_and_applies_verified_description(tmp_path):
    alpha_state = {
        "id": "PP777",
        "status": "UNSUBMITTED",
        "name": "Asset Turnover Signal",
        "category": "FUNDAMENTAL",
        "tags": ["quality"],
        "regular": {
            "code": "rank(ts_backfill(fnd6_newa1v1300_assets, 120))",
            "description": None,
            "operatorCount": 2,
        },
        "settings": {**ml.DEFAULT_SETTINGS, "simulationMode": "FULL"},
        "is": {"sharpe": 1.45, "fitness": 1.15, "turnover": 0.12},
    }

    def patch_alpha(_params, body):
        if "regular" in body and "description" in body["regular"]:
            alpha_state["regular"]["description"] = body["regular"]["description"]
        for k in ("name", "category", "tags", "color", "hidden"):
            if k in body:
                alpha_state[k] = body[k]
        return FakeResponse(200, alpha_state)

    def check_alpha(_params, _body):
        desc = alpha_state["regular"].get("description") or ""
        return FakeResponse(200, {
            "id": "PP777",
            "is": {
                "checks": [
                    {"name": "LOW_SHARPE", "result": "PASS", "value": 1.45, "limit": 1.0},
                    {"name": "HIGH_TURNOVER", "result": "PASS", "value": 0.12, "limit": 0.7},
                    {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS", "value": 1.1, "limit": 0.8},
                    {"name": "POWER_POOL_CORRELATION", "result": "PASS", "value": 0.32, "limit": 0.5},
                    {"name": "MATCHES_THEMES", "result": "PASS", "themes": [{"name": "Fundamental"}]},
                    {
                        "name": "POWER_POOL_DESCRIPTION_LENGTH",
                        "result": "PASS" if len(desc) >= 100 else "FAIL",
                        "value": len(desc),
                        "limit": 100,
                    },
                ]
            },
        })

    session = FakeSession({
        ("GET", "/alphas/PP777"): lambda _p, _b: FakeResponse(200, alpha_state),
        ("PATCH", "/alphas/PP777"): patch_alpha,
        ("GET", "/alphas/PP777/check"): check_alpha,
        ("GET", "/data-fields/fnd6_newa1v1300_assets"): FakeResponse(
            200,
            {
                "id": "fnd6_newa1v1300_assets",
                "type": "MATRIX",
                "dataset": {"id": "fundamental6"},
                "description": "Total reported corporate assets",
            },
        ),
        ("GET", "/operators"): FakeResponse(
            200,
            [
                {"name": "rank", "category": "Cross Sectional", "description": "ranks values cross-sectionally"},
                {"name": "ts_backfill", "category": "Time Series", "description": "fills missing values from recent history"},
            ],
        ),
    })

    with mock.patch.object(ml, "load_session", return_value=session), \
            mock.patch.object(ml, "configured_username", return_value="researcher@example.com"):
        code, doc, _ = invoke_cli(
            "alpha", "readiness", "PP777",
            "--track", "POWER_POOL",
            "--idea", "Cross-sectional variation in corporate asset base captures firm scale resilience.",
            "--apply-description",
            "--research-dir", str(tmp_path),
        )
    assert code == 0
    data = doc["data"]
    assert data["description_applied"] is True
    assert data["draft_description"]["ok"] is True
    assert "Idea:" in data["expression_description"]
    assert "Rationale for data used:" in data["expression_description"]
    assert "Rationale for operators used:" in data["expression_description"]
    assert data["power_pool_budget"]["budget"]["operators"] == 1  # ts_backfill exempt!
    assert data["power_pool_budget"]["budget"]["single_dataset"] is True
    assert data["ready"] is True
    assert data["blocking"] == []


def test_alpha_list_filters_and_describe_color_hide_tags(tmp_path):
    alpha_state = {
        "id": "A55",
        "name": "Old",
        "category": "PRICE_VOLUME",
        "tags": [],
        "color": None,
        "hidden": False,
        "regular": {"code": "rank(close)", "description": "existing"},
        "is": {"sharpe": 1.3},
    }

    def patch_alpha(_params, body):
        for k, v in body.items():
            if k == "regular":
                alpha_state["regular"].update(v)
            else:
                alpha_state[k] = v
        return FakeResponse(200, alpha_state)

    session = FakeSession({
        ("GET", "/users/self/alphas"): FakeResponse(200, {"results": [alpha_state]}),
        ("GET", "/alphas/A55"): lambda _p, _b: FakeResponse(200, alpha_state),
        ("PATCH", "/alphas/A55"): patch_alpha,
    })

    with mock.patch.object(ml, "load_session", return_value=session):
        code, list_doc, _ = invoke_cli(
            "alpha", "list",
            "--delay", "1",
            "--category", "PRICE_VOLUME",
            "--color", "GREEN",
            "--tag", "rev",
            "--max-turnover", "0.4",
            "--since", "2026-01-01",
            "--research-dir", str(tmp_path),
        )
        assert code == 0
        sent_params = session.calls[0][2]
        assert sent_params["settings.delay"] == 1
        assert sent_params["category"] == "PRICE_VOLUME"
        assert sent_params["color"] == "GREEN"
        assert sent_params["tag"] == "rev"
        assert sent_params["is.turnover<"] == 0.4
        assert sent_params["dateCreated>="] == "2026-01-01"

        with mock.patch.dict(os.environ, {"BRAIN_RESEARCH_DIR": str(tmp_path)}), \
                mock.patch.object(ml, "load_session", return_value=session):
            desc_data = alphas.cmd_alpha_describe(
                SimpleNamespace(
                    alpha_id="A55",
                    name=None,
                    category=None,
                    tags=["alpha,beta", "gamma"],
                    description=None,
                    color="GREEN",
                    hide=True,
                    unhide=False,
                    no_cache=False,
                    relogin=False,
                    meta_max_age=None,
                ),
                None,
            )
        assert desc_data["tags"] == ["alpha", "beta", "gamma"]
        assert desc_data["color"] == "GREEN"
        assert desc_data["hidden"] is True


def test_alpha_describe_rejects_old_template_before_the_remote_write():
    args = SimpleNamespace(
        alpha_id="A55",
        description="IDEA:\n- " + "quality " * 20 + "\n\nRATIONALE:\n- " + "rank " * 20,
        name=None,
        category=None,
        tags=None,
        color=None,
        hide=False,
        unhide=False,
    )
    with mock.patch.object(alphas, "with_session") as session:
        with pytest.raises(CliError) as caught:
            alphas.cmd_alpha_describe(args, None)
    assert "official Power Pool template" in caught.value.message
    assert "Rationale for operators used:" in caught.value.hint
    session.assert_not_called()


def test_ergonomic_cli_shortcuts_propose_triage_and_near(tmp_path):
    # 1. Removed v3 simulation/triage CLI entrypoints fail cleanly with USAGE
    for removed in (["simulate", "rank(close)", "--quick"], ["probe", "x.json"], ["triage", "CAND1"]):
        code, err_doc, _ = invoke_cli(*removed, "--research-dir", str(tmp_path))
        assert code == 2
        assert err_doc["error"]["code"] == "USAGE"

    # 2. `propose` supports non-blocking spec submission with --name override
    spec = tmp_path / "quick.spec.json"
    spec.write_text(json.dumps({
        "name": "ergonomic-probe",
        "idea": "Cross-sectional liquidity",
        "design": "explore",
        "arms": [{"label": "c1", "expression": "rank(close)"}],
    }), encoding="utf-8")
    code, proposed, _ = invoke_cli("propose", str(spec), "--name", "custom-run", "--research-dir", str(tmp_path))
    assert code == 0, proposed
    assert proposed["data"]["name"] == "custom-run"
    assert proposed["data"]["state"] == "AUTO_PREFLIGHT"

    # 3. Positional target on `alpha near` and internal `triage` scorer
    registry = tmp_path / "alpha-registry.jsonl"
    entry = alpha_registry.build_entry(
        {
            "alpha_id": "CAND1",
            "expression": "rank(volume)",
            "settings": ml.resolve_settings({"testPeriod": "P1Y"}),
            "status": "UNSUBMITTED",
            "metrics": {"sharpe": 1.5, "fitness": 1.1, "turnover": 0.2},
            "out_of_sample": {
                "test_period": "P1Y",
                "train": {"sharpe": 1.5, "fitness": 1.1, "turnover": 0.2, "startDate": "2019-01-01", "endDate": "2022-12-31"},
                "test": {"sharpe": 1.2, "fitness": 0.9, "turnover": 0.2, "startDate": "2023-01-01", "endDate": "2023-12-31"},
            },
        },
        verdict="candidate",
    )
    book_days = [(date(2022, 1, 3) + timedelta(days=index)).isoformat()
                 for index in range(120)]
    (tmp_path / "pnl").mkdir(exist_ok=True)
    (tmp_path / "pnl" / "BOOK1.json").write_text(
        json.dumps({day: 1.0 if index % 2 else 0.4 for index, day in enumerate(book_days)}),
        encoding="utf-8",
    )
    (tmp_path / "pnl" / "CAND1.json").write_text(
        json.dumps({day: -0.7 if index % 2 else 0.9 for index, day in enumerate(book_days)}),
        encoding="utf-8",
    )
    book_entry = alpha_registry.build_entry(
        {
            "alpha_id": "BOOK1",
            "expression": "ts_delta(cash, 5)",
            "settings": ml.resolve_settings({"testPeriod": "P1Y"}),
            "status": "ACTIVE",
            "metrics": {"sharpe": 1.0, "fitness": 1.0, "turnover": 0.2},
            "out_of_sample": {
                "test_period": "P1Y",
                "train": {"sharpe": 1.0, "fitness": 1.0, "turnover": 0.2, "startDate": "2019-01-01", "endDate": "2022-12-31"},
                "test": {"sharpe": 1.0, "fitness": 1.0, "turnover": 0.2, "startDate": "2023-01-01", "endDate": "2023-12-31"},
            },
        },
        verdict="watch",
    )
    with alpha_registry.append_batch(str(registry)) as (_entries, append):
        append(book_entry)
        append(entry)

    code, near_expr, _ = invoke_cli("alpha", "near", "rank(close)", "--registry", str(registry), "--research-dir", str(tmp_path))
    assert code == 0
    assert near_expr["data"]["same_skeleton"] == ["CAND1"]

    code, near_id, _ = invoke_cli("alpha", "near", "CAND1", "--registry", str(registry), "--research-dir", str(tmp_path))
    assert code == 0
    assert near_id["data"]["fields"] == ["volume"]

    from research_app import triage
    triage_doc = triage.triage(["CAND1"], registry_path=str(registry), pnl_dir=str(tmp_path / "pnl"))
    assert [row["alpha_id"] for row in triage_doc["rows"]] == ["CAND1"]


def test_brain_client_lazy_session_offline_preview_and_auto_refresh(tmp_path):
    # 1. Dry-run never opens a session
    with mock.patch.object(ml, "load_session", side_effect=AssertionError("should not login")):
        with ml.BrainClient() as client:
            preview = client.simulate_one("rank(close)", mode="QUICK", dry_run=True)
            assert preview["payload"]["settings"]["simulationMode"] == "QUICK"
            assert preview["expression"] == "rank(close)"

    # 2. Expired session refreshes automatically once on HTTP 401
    stale = FakeSession({
        ("GET", "/users/self"): FakeResponse(401, {"detail": "expired"}),
    })
    fresh = FakeSession({
        ("GET", "/users/self"): FakeResponse(200, {"id": "WQ_U9", "username": "quant"}),
    })
    sessions = [stale, fresh]

    def fake_load_session(**kwargs):
        return sessions.pop(0)

    with mock.patch.object(ml, "load_session", side_effect=fake_load_session) as loader:
        with ml.BrainClient() as client:
            profile = client.profile()
            assert profile["id"] == "WQ_U9"
            assert loader.call_count == 2
        assert fresh.closed is True
