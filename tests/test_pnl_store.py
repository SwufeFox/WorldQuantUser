"""Regression coverage for committed daily PnL and its analysis readers."""

from datetime import date, timedelta
from unittest import mock

import pytest

import cache_store
import machine_lib as ml
import pnl_store
from research_app import pnl_snapshot
from tools import cluster_ledger


def test_compaction_uses_manifest_and_never_resurrects_replaced_dates(tmp_path):
    root = tmp_path / "pnl"
    pnl_store.write_pnl(root, "A", {"2020-01-01": 1}, {"complete": True})
    pnl_store.maybe_compact(root, force=True)
    pnl_store.write_pnl(root, "A", {"2021-01-01": 2}, {"complete": True})
    pnl_store.write_pnl(root, "B", {}, {"complete": True})
    cache_store.write_json_atomic(root / "A.json", {"2020-01-01": 5})
    cache_store.write_json_atomic(root / "B.json", {"2020-01-01": 5})
    orphan = pnl_store._publish_delta_file(root, "A", {"2020-01-01": 99.0})

    assert pnl_snapshot.load_series_map(root, end_day="2021-01-01") == {}
    assert pnl_snapshot.load_series_map(root) == {"A": {"2021-01-01": 2}}
    with mock.patch.object(pnl_store, "_publish_frame", side_effect=OSError("write interrupted")):
        with pytest.raises(OSError, match="write interrupted"):
            pnl_store.maybe_compact(root, force=True)
    assert pnl_snapshot.load_series_map(root) == {"A": {"2021-01-01": 2}}

    assert pnl_store.maybe_compact(root, force=True)["deltas_merged"] == 2
    assert orphan.exists()
    with pnl_store.backup_members(root) as members:
        assert len(members) == 1
        assert (root / members[0]).is_file()
        assert orphan.relative_to(root) not in members
    assert pnl_store.read_pnl(root, "B") == {}
    assert ml.pnl_cache_status(root, "B")["complete"] is True
    assert pnl_snapshot.load_series_map(root) == {"A": {"2021-01-01": 2}}


def test_migration_preserves_current_and_previous_digest_bound_evidence(tmp_path):
    root = tmp_path / "pnl"
    archive = tmp_path / "archive"
    for alpha_id in ("A", "B", "C"):
        data = {"2020-01-01": 1}
        cache_store.write_json_atomic(root / f"{alpha_id}.json", data)
        proof = {"format": ml._PNL_EVIDENCE_FORMAT, "alpha_id": alpha_id,
                 "complete": True, "data_sha256": ml._pnl_digest(data), "as_of": "2025-01-01"}
        cache_store.write_json_atomic(root / ".meta" / f"{alpha_id}.json",
                                      {**proof, "data_sha256": "unrelated", "previous": proof if alpha_id == "A" else None})
    assert pnl_store.migrate_legacy_pnl_dir(root, archive_dir=archive, dry_run=True)["migrated"] == 3
    assert not (root / "meta.sqlite").exists()
    pnl_store.write_pnl(root, "C", {"2020-01-01": 3}, {"complete": True})

    cache_store.write_json_atomic(archive / "B.json", {"keep": 7})
    migrated = pnl_store.migrate_legacy_pnl_dir(root, archive_dir=archive)
    assert migrated["migrated"] == 2
    assert migrated["archive_conflicts"] == 1
    assert cache_store.load_json(archive / "B.json")[0] == {"keep": 7}
    assert (root / "B.json").exists()
    assert ml.pnl_cache_status(root, "A")["complete"] is True
    assert ml.pnl_cache_status(root, "B")["complete"] is None
    assert pnl_store.read_pnl(root, "C") == {"2020-01-01": 3}
    assert (root / "C.json").exists()
    assert (archive / ".meta" / "A.json").is_file()
    assert (archive / "A.json").is_file()
    assert pnl_store.migrate_legacy_pnl_dir(root, archive_dir=archive)["migrated"] == 0


def test_yearly_stability_and_clustering_read_committed_pnl(tmp_path):
    root = tmp_path / "pnl"
    cutoff = "2022-01-01"
    start = date(2019, 1, 1)
    days = [(start + timedelta(days=offset)).isoformat() for offset in range(3 * 366)]
    series = {day: (-1) ** index + (0.25 if day < cutoff else -1000)
              for index, day in enumerate(days)}
    pnl_store.write_pnl(root, "A", series, {"complete": True, "as_of": "2025-01-01"})
    rows = [{"alpha_id": "A", "train_start": start.isoformat(), "test_start": cutoff}]
    [yearly] = pnl_snapshot.yearly_train_batch(rows, root)
    assert yearly["state"] == "pass"
    assert [entry["year"] for entry in yearly["years"]] == [2019, 2020, 2021]
    assert min(entry["sharpe"] for entry in yearly["years"]) > 0

    cutoffs = {"A": cutoff}
    path = tmp_path / "train.parquet"
    training, state = pnl_snapshot.training_half_sharpes(root, path, cutoffs, cutoffs)
    assert state == "rebuilt"
    assert min(training["A"]) > 0
    pnl_store.maybe_compact(root, force=True)
    warm, state = pnl_snapshot.training_half_sharpes(root, path, cutoffs, cutoffs)
    assert state == "hit"
    assert tuple(warm["A"]) == training["A"]
    with mock.patch.object(ml, "fetch_daily_pnl", side_effect=AssertionError("unexpected fetch")):
        assert cluster_ledger.load_pnls(None, ["A"], root) == ({"A": series}, [])
