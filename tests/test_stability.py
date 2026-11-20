#!/usr/bin/env python3
"""Tests for `ledger stability`: does a train Sharpe live in both halves?

The verb exists because a single train number hid the fact that a candidate's edge was
one regime. These tests pin the two states the book is judged on: a stable alpha and a
sign-flipping one, plus the late-bloomer shape that neither call should label a flip.
"""

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import research_paths
from research_app import app, ledger as cli_ledger, pnl_snapshot


def _dates(year, count):
    import datetime
    start = datetime.date(year, 1, 1)
    return [(start + datetime.timedelta(days=offset)).isoformat() for offset in range(count)]


def _series(days, first, second):
    out = {}
    for index, day in enumerate(days):
        out[day] = first if index % 2 == 0 else second
    return out


class StabilityTests(unittest.TestCase):
    def _run(self, series, min_move=0.3):
        tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp, "pnl"), exist_ok=True)
        entries = []
        for name, pnl in series.items():
            with open(os.path.join(tmp, "pnl", "%s.json" % name), "w") as handle:
                json.dump(pnl, handle)
            entries.append({"alpha_id": name, "status": "ACTIVE",
                            "out_of_sample": {"test": {"startDate": "2023-01-01"}}})
        args = argparse.Namespace(where=None, status="ACTIVE", min_move=min_move,
                                  limit=10, list=True, registry=None)
        with mock.patch.object(cli_ledger, "_ledger", lambda a: (None, entries)), \
                mock.patch.object(research_paths, "ensure_dir", return_value=tmp):
            return cli_ledger.cmd_ledger_stability(args, mock.MagicMock())

    def test_a_steady_alpha_is_not_a_flip(self):
        days = _dates(2020, 199)
        data = self._run({"STEADY": _series(days, 1.0, 0.5)})
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["flips"], 0)

    def test_a_sign_flip_between_halves_is_reported(self):
        first = _dates(2020, 99)
        second = _dates(2021, 99)
        flipping = {}
        for index, day in enumerate(first):
            flipping[day] = 9.0 if index % 2 else 3.0
        for index, day in enumerate(second):
            flipping[day] = -9.0 if index % 2 else -3.0
        data = self._run({"FLIP": flipping})
        self.assertEqual(data["flips"], 1)
        self.assertLess(data["flip_list"][0]["h1"] * data["flip_list"][0]["h2"], 0)

    def test_a_quiet_first_half_is_reported_as_a_gap_not_a_flip(self):
        first = _dates(2020, 99)
        second = _dates(2021, 99)
        late = {day: (0.1 if index % 2 else -0.1) for index, day in enumerate(first)}
        late.update({day: (12.0 if index % 2 else 4.0) for index, day in enumerate(second)})
        data = self._run({"LATE": late})
        self.assertEqual(data["flips"], 0)
        self.assertGreater(data["worst"][0]["gap"], 1.0)


def test_polars_stability_reuses_and_invalidates_train_only_snapshot(tmp_path, monkeypatch, capsys):
    pl = pytest.importorskip("polars")
    monkeypatch.setenv("BRAIN_RESEARCH_DIR", str(tmp_path))
    registry = tmp_path / "alpha-registry.jsonl"
    pnl_dir = tmp_path / "pnl"
    pnl_dir.mkdir()
    rows = [{"alpha_id": key, "status": status, "out_of_sample": {"test": {"startDate": cutoff}}}
            for key, status, cutoff in [("Good", "ACTIVE", "2020-06-01T00:00:00Z"),
                ("Short", "ACTIVE", None), ("Absent", "ACTIVE", "2020-06-01"),
                ("Malformed", "UNSUBMITTED", "2020-06-01")]]

    def write_registry():
        registry.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    write_registry()
    good = pnl_dir / "Good.json"
    good.write_text(json.dumps(_series(_dates(2020, 151)[::-1], 1.0, 2.0)
                    | {"2020-06-01": 1e12, "2020-06-02": -1e12}), encoding="utf-8")
    (pnl_dir / "Short.json").write_text('{"2020-01-01": 1}', encoding="utf-8")
    (pnl_dir / "Malformed.json").write_text('["not a series"]', encoding="utf-8")

    def invoke(*args):
        assert app.main(["ledger", "stability", "--quiet", *args]) == 0
        return json.loads(capsys.readouterr().out)["data"]

    expected = invoke("--list")
    assert expected["count"] == 2
    assert next(row for row in expected["worst"] if row["alpha_id"] == "Short")["gap"] is None
    first = invoke("--polars", "--list")
    evidence = first.pop("pnl_snapshot")
    assert evidence["state"] == "rebuilt"
    assert first == expected
    path = Path(evidence["path"])
    assert pl.read_parquet(path)["day"].max() < "2020-06-01"

    with monkeypatch.context() as warm:
        warm.setitem(sys.modules, "polars", None)
        warm.setitem(sys.modules, "numpy", None)
        with mock.patch.object(cli_ledger.experiments, "train_half_sharpes",
                               side_effect=AssertionError("warm stability remeasured PnL")):
            reuse = invoke("--polars", "--list")
        assert reuse.pop("pnl_snapshot")["state"] == "hit"
        assert reuse == expected

    subset = invoke("--polars", "--status", "ACTIVE")
    assert subset.pop("pnl_snapshot")["state"] == "hit"
    assert subset == invoke("--status", "ACTIVE")
    rows[0]["out_of_sample"]["test"]["startDate"] = "2020-05-10"
    write_registry()
    changed_cutoff = invoke("--polars")
    assert changed_cutoff.pop("pnl_snapshot")["state"] == "rebuilt"
    assert changed_cutoff == invoke()
    assert pl.read_parquet(path)["day"].max() < "2020-05-10"

    good.unlink()
    changed_pnl = invoke("--polars")
    assert changed_pnl.pop("pnl_snapshot")["state"] == "rebuilt"
    assert changed_pnl == invoke()
    path.write_text("damaged", encoding="utf-8")
    assert invoke("--polars")["pnl_snapshot"]["state"] == "rebuilt"

    manifest = path.with_suffix(".json")
    damaged = json.loads(manifest.read_text(encoding="utf-8"))
    damaged["half_sharpes"]["Short"] = [True, None]
    manifest.write_text(json.dumps(damaged), encoding="utf-8")
    assert invoke("--polars")["pnl_snapshot"]["state"] == "rebuilt"

    rows.append({"alpha_id": "Extreme", "status": "UNSUBMITTED",
                 "out_of_sample": {"test": {"startDate": "2021-01-01"}}})
    write_registry()
    (pnl_dir / "Extreme.json").write_text(
        json.dumps(_series(_dates(2020, 150), 1e180, -1e180)), encoding="utf-8")
    filtered = invoke("--polars", "--status", "ACTIVE")
    assert filtered.pop("pnl_snapshot")["state"] == "rebuilt"
    assert filtered == invoke("--status", "ACTIVE")


def test_polars_snapshot_refresh_reads_only_changed_series(tmp_path):
    pl = pytest.importorskip("polars")
    pnl_dir = tmp_path / "pnl"
    pnl_dir.mkdir()
    path = tmp_path / "cache" / "pnl-train.parquet"
    days = _dates(2020, 145)
    names = ("Unchanged", "Beta", "Alpha", "RemovedId", "DeletedFile")
    series = {name: _series(days[::-1], index + 1.0, index + 2.0)
              for index, name in enumerate(names)}
    cutoffs = dict.fromkeys(series, days[140])

    def write_source(alpha_id):
        (pnl_dir / f"{alpha_id}.json").write_text(json.dumps(series[alpha_id]), encoding="utf-8")

    for alpha_id in series:
        write_source(alpha_id)
    _, state = pnl_snapshot.training_half_sharpes(pnl_dir, path, cutoffs, cutoffs)
    assert state == "rebuilt"

    series["Alpha"][days[0]] = -12.0
    write_source("Alpha")
    cutoffs["Beta"] = days[122]
    series["Middle"] = _series(days[::-1], 1.5, 3.0)
    cutoffs["Middle"] = days[135]
    write_source("Middle")
    del cutoffs["RemovedId"]
    (pnl_dir / "DeletedFile.json").unlink()

    present = ("Alpha", "Beta", "Middle", "Unchanged")
    expected_rows = sorted((alpha_id, day, pnl) for alpha_id in present
                           for day, pnl in series[alpha_id].items() if day < cutoffs[alpha_id])
    with mock.patch.object(pnl_snapshot, "read_series", wraps=pnl_snapshot.read_series) as read_series:
        result, state = pnl_snapshot.training_half_sharpes(pnl_dir, path, cutoffs, cutoffs)
        assert state == "rebuilt"
        assert sorted(call.args[0].name for call in read_series.call_args_list) == [
            "Alpha.json", "Beta.json", "Middle.json"]
        assert pl.read_parquet(path).rows() == expected_rows
        assert result.keys() == set(present)
        for alpha_id, pair in result.items():
            expected = cli_ledger.experiments.half_split_sharpes(series[alpha_id], cutoffs[alpha_id])
            assert tuple(pair) == pytest.approx(expected)

        read_series.reset_mock()
        warm, state = pnl_snapshot.training_half_sharpes(pnl_dir, path, cutoffs, cutoffs)
        assert state == "hit"
        assert {alpha_id: tuple(pair) for alpha_id, pair in warm.items()} == {
            alpha_id: tuple(pair) for alpha_id, pair in result.items()}
        read_series.assert_not_called()

        (pnl_dir / "Middle.json").write_text("[]", encoding="utf-8")
        result, state = pnl_snapshot.training_half_sharpes(pnl_dir, path, cutoffs, cutoffs)
        assert state == "rebuilt"
        read_series.assert_called_once_with(pnl_dir / "Middle.json")
        assert result.keys() == {"Alpha", "Beta", "Unchanged"}
        assert pl.read_parquet(path).rows() == [row for row in expected_rows if row[0] != "Middle"]


if __name__ == "__main__":
    unittest.main()
