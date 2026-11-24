"""Offline contract for releasing a reservation whose simulation no longer exists."""

import json
from pathlib import Path

import simulation_slots as slots

ACCOUNT = {"fingerprint": "release-url-test"}
URL = "https://api.worldquantbrain.com/simulations/gone"


def _journal(tmp_path, monkeypatch, reservations):
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    path, _lock = slots._paths(slots._fingerprint(ACCOUNT))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "version": 1, "account_fingerprint": slots._fingerprint(ACCOUNT),
        "reservations": reservations,
    }))
    return Path(path)


def test_release_url_drops_only_the_matching_reservation(tmp_path, monkeypatch):
    path = _journal(tmp_path, monkeypatch, {
        "gone": {"state": "accepted", "progress_url": URL},
        "alive": {"state": "accepted", "progress_url": URL + "-alive"},
    })
    assert slots.release_url(URL) == ["gone"]
    journal = json.loads(path.read_text())
    assert list(journal["reservations"]) == ["alive"]
    assert journal["quick_resolved"]["gone"]["state"] == "released"
    assert slots.release_url(URL) == []


def test_release_url_ignores_non_http_values(tmp_path, monkeypatch):
    _journal(tmp_path, monkeypatch, {})
    assert slots.release_url(None) == []
    assert slots.release_url("simulations/4") == []
