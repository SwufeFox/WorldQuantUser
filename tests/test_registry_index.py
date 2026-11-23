import json
from unittest.mock import patch

import alpha_registry
import machine_lib
import registry_index
from mechanism_identity import mechanism_key
from research_app import mine


def _row(key, **fields):
    return {
        "key": key,
        "alpha_id": key,
        "expression": "rank(close)",
        "settings": {**machine_lib.DEFAULT_SETTINGS, "simulationMode": "FULL"},
        "metrics": {"sharpe": 1.8},
        "status": "COMPLETE",
        **fields,
    }


def _append_raw(path, *rows):
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _snapshot(path):
    with alpha_registry.indexed_registry(path) as index:
        keys = index.connection.execute("SELECT key FROM registry_entries ORDER BY first_offset")
        return [index.entry(key) for (key,) in keys]


def test_projection_replays_append_without_scanning_existing_rows(tmp_path):
    path = tmp_path / "registry.jsonl"
    _append_raw(path, _row("A", role="diagnostic"), _row("B"))
    assert _snapshot(path) == alpha_registry.load_entries(path)
    _append_raw(path, _row("A", role="candidate"), _row("C"))
    with patch.object(alpha_registry, "_parse_line", wraps=alpha_registry._parse_line) as parse:
        with alpha_registry.indexed_registry(path) as index:
            assert parse.call_count == 2
            assert index.entry("A")["role"] == "diagnostic"
    with patch.object(alpha_registry, "load_entries", side_effect=AssertionError("full scan")):
        entry, written = alpha_registry.record_entry(_row("A", extra={"value": 1}), path)
        assert written and entry["role"] == "diagnostic"
        _, written = alpha_registry.record_entry(_row("A", extra={"value": 1}), path)
        assert not written
    assert _snapshot(path) == alpha_registry.load_entries(path)


def test_projection_rebuilds_replaced_truncated_and_rewritten_files(tmp_path):
    path = tmp_path / "registry.jsonl"
    _append_raw(path, _row("A", role="diagnostic"), _row("B"))
    assert len(_snapshot(path)) == 2

    # A same-size in-place edit removes sticky role history and changes the indexed key.
    path.write_bytes(path.read_bytes().replace(b'"A"', b'"X"'))
    assert _snapshot(path) == alpha_registry.load_entries(path)
    assert _snapshot(path)[0]["key"] == "X"

    replacement = path.with_suffix(".new")
    _append_raw(replacement, _row("A"))
    replacement.replace(path)
    assert _snapshot(path) == alpha_registry.load_entries(path)
    assert _snapshot(path)[0]["role"] is None

    path.write_bytes(b"")
    assert _snapshot(path) == []
    path.unlink()
    assert _snapshot(path) == []


def test_projection_keeps_legacy_tail_semantics_and_reader_snapshot(tmp_path):
    path = tmp_path / "registry.jsonl"
    row = _row("A")
    path.write_text(json.dumps(row), encoding="utf-8")
    assert _snapshot(path) == alpha_registry.load_entries(path)
    alpha_registry.record_entry(_row("B"), path)
    assert _snapshot(path) == alpha_registry.load_entries(path)

    with alpha_registry.indexed_registry(path) as previous:
        alpha_registry.record_entry(_row("A", role="diagnostic"), path)
        assert previous.entry("A")["role"] is None
        with alpha_registry.indexed_registry(path) as current:
            assert current.entry("A")["role"] == "diagnostic"
    assert _snapshot(path) == alpha_registry.load_entries(path)


def test_projection_qualifies_and_previews_novelty_in_legacy_order(tmp_path):
    path = tmp_path / "registry.jsonl"
    _append_raw(path,
                _row("first", alpha_id="shared", status="ACTIVE"),
                _row("second", status="ACTIVE"),
                _row("third", alpha_id="shared"),
                _row("partial", settings={"region": "USA"}),
                _row("warning", status="WARNING"),
                _row("pending", status="PENDING"),
                _row("missing", metrics={}),
                _row("first", alpha_id="shared", expression="rank(open)", status="ACTIVE"))
    with alpha_registry.indexed_registry(path) as index:
        identity = registry_index.known_request_identity(_row("x"))
        keys, exact = index.exact_history(identity, limit=1)
        assert keys == ["third", "second", "warning"]
        assert [entry["key"] for entry in exact] == ["third"]
        with patch.object(index, "_read_entry", wraps=index._read_entry) as read:
            count, active, preview = index.mechanism_history(
                mechanism_key("rank(close)"), exclude_keys=keys, limit=2
            )
            assert count == 3 and active == ["second"]
            assert [entry["key"] for entry in preview] == ["partial", "pending"]
            assert read.call_count == 2


def test_corrupt_projection_keeps_ledger_reads_and_writes_available(tmp_path):
    path = tmp_path / "registry.jsonl"
    _append_raw(path, _row("A", role="diagnostic"))
    with open(registry_index.index_path(path), "wb") as cache:
        cache.write(b"invalid database")
    with alpha_registry.indexed_registry(path) as index:
        assert index is None
    entry, written = alpha_registry.record_entry(_row("A", note="new"), path)
    assert written and entry["role"] == "diagnostic"
    assert alpha_registry.load_entries(path) == [entry]


def test_indexed_turnover_counts_each_dataset_once_and_tracks_replaced_fields(tmp_path):
    path = tmp_path / "registry.jsonl"
    _append_raw(path,
                _row("A", expression="x = close + volume; rank(x + turnover)",
                     out_of_sample={"train": {"turnover": 0.1}}),
                _row("B", expression="rank(volume + turnover)",
                     out_of_sample={"train": {"turnover": 0.3}}),
                _row("C", expression="rank(volume)", metrics={"turnover": 0.9}),
                _row("D", expression="rank(close)", out_of_sample={"train": {"turnover": -0.1}}),
                _row("E", expression="rank(turnover)", out_of_sample={"train": {"turnover": True}}))
    catalog = {"close": {"dataset": "price"}, "volume": {"dataset": "price"},
               "turnover": {"dataset": "activity"}, "x": {"dataset": "local"}}
    actual = mine.dataset_turnover(catalog=catalog, registry_path=path)
    assert actual == mine.dataset_turnover(alpha_registry.load_entries(path), catalog)
    assert actual == {
        "price": {"n": 2, "median": 0.2, "missing_train": 2},
        "activity": {"n": 2, "median": 0.2, "missing_train": 1},
    }
    alpha_registry.record_entry(_row("B", expression="rank(turnover)",
                                     out_of_sample={"train": {"turnover": 0.3}}), path)
    assert mine.dataset_turnover(catalog=catalog, datasets={"price"}, registry_path=path) == {
        "price": {"n": 1, "median": 0.1, "missing_train": 2},
    }


def test_open_missing_registry_creates_no_directory(tmp_path):
    root = tmp_path / "unused"
    with alpha_registry.indexed_registry(root / "registry.jsonl") as index:
        assert index is None
    assert not root.exists()
