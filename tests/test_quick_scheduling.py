"""Offline coverage for shared batch entry points and independent metrics reads."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

from tools.pipeline import async_batch, batch_quick


def test_late_metrics_do_not_block_next_simulation():
    metrics_started = Event()
    next_simulation_done = Event()
    arms = [{"label": name, "expr": name} for name in ("first", "second")]

    def run(arm, *, fetch_metrics):
        assert not fetch_metrics
        if arm["label"] == "first":
            return {**arm, "metrics_pending": True, "status": "PENDING", "alpha_id": "FirstAlpha"}
        assert metrics_started.wait(5)
        next_simulation_done.set()
        return {**arm, "status": "COMPLETE"}

    def finish_metrics(row):
        metrics_started.set()
        assert next_simulation_done.wait(5)
        return {"label": row["label"], "status": "COMPLETE", "alpha_id": row["alpha_id"]}

    runner = Mock(run=run, finish_metrics=finish_metrics)
    with ThreadPoolExecutor(max_workers=1) as simulations, ThreadPoolExecutor(max_workers=1) as metrics:
        rows = [row for _, row in batch_quick._completed_arms(runner, arms, simulations, metrics)]

    assert {row["label"]: row["status"] for row in rows} == {"first": "COMPLETE", "second": "COMPLETE"}


def test_metrics_timeout_is_written_once_as_pending():
    arm = {"label": "first", "expr": "rank(close)"}
    pending = {**arm, "status": "PENDING", "metrics_pending": True, "alpha_id": "FirstAlpha"}
    runner = Mock()
    runner.run.return_value = pending
    runner.finish_metrics.return_value = pending

    with ThreadPoolExecutor(max_workers=1) as simulations, ThreadPoolExecutor(max_workers=1) as metrics:
        assert list(batch_quick._completed_arms(runner, [arm], simulations, metrics)) == [(arm, pending)]

    runner.finish_metrics.assert_called_once_with(pending)


def test_async_entry_point_preserves_legacy_cli_defaults(monkeypatch):
    shared_main = Mock(return_value=0)
    monkeypatch.setattr(async_batch, "run_batch", shared_main)
    assert async_batch.main() == 0
    shared_main.assert_called_once_with(default_out="ledgers/async.jsonl", default_concurrency=6)


def test_interrupted_consumer_leaves_unstarted_arms_unscheduled():
    arms = [{"label": name, "expr": name} for name in ("first", "second", "third")]
    runner = Mock()
    runner.run.side_effect = lambda arm, **_: {**arm, "status": "COMPLETE"}
    with ThreadPoolExecutor(max_workers=1) as simulations, ThreadPoolExecutor(max_workers=1) as metrics:
        rows = batch_quick._completed_arms(runner, arms, simulations, metrics, workers=1)
        assert next(rows)[0] == arms[0]
        rows.close()
    runner.run.assert_called_once_with(arms[0], fetch_metrics=False)
