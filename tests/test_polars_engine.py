"""Tests for Polars catalog DataFrames, unified incremental Parquet PnL cache, and vectorized BLAS correlation."""

import argparse
from datetime import date, timedelta
import json
import math
from pathlib import Path
from random import Random
import sys
import tempfile
import unittest
from unittest import mock

import polars as pl

import correlation
import machine_lib as ml
from research_app import ledger_analysis, pairs, pnl_snapshot
import research_paths
from tools import cluster_ledger


def _make_series(seed, count=260, *, start=date(2020, 1, 1), offset=0.0, scale=1.0, skip_every=None):
    rng = Random(seed)
    series = {}
    for idx in range(count):
        if skip_every and idx % skip_every == 0:
            continue
        day = (start + timedelta(days=idx)).isoformat()
        series[day] = offset + scale * (rng.uniform(-1.5, 1.5) + 0.05)
    return series


class CatalogPolarsTests(unittest.TestCase):
    """fields_to_df and process_datafields use Polars cleanly without pandas."""

    def test_fields_to_df_and_process_datafields_round_trip(self):
        rows = [
            {"id": f"field_{i}", "type": "MATRIX" if i % 2 == 0 else "VECTOR",
             "dataset": {"id": "fundamental6", "name": "Fund 6"},
             "extra": f"val_{i}" if i == 115 else None}
            for i in range(120)
        ]
        df = ml.fields_to_df(rows)
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.height, 120)
        snippets = ml.process_datafields(df)
        self.assertEqual(len(snippets), 120)
        self.assertIn("winsorize(ts_backfill(field_0, 120), std=4)", snippets)
        self.assertIn("vec_avg(field_1)", snippets)

        empty_df = ml.fields_to_df([])
        self.assertIsInstance(empty_df, pl.DataFrame)
        self.assertEqual(empty_df.height, 0)
        self.assertEqual(ml.process_datafields(empty_df), [])


class UnifiedPnlParquetTests(unittest.TestCase):
    """Incremental <pnl_dir>/.parquet_cache/pnl.parquet sync, hit reuse, updates, and slicing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pnl_dir = Path(self.tmp.name) / "pnl"
        self.pnl_dir.mkdir()
        self.data = {}
        for idx in range(5):
            aid = f"ALPHA_{idx}"
            series = _make_series(100 + idx, count=260)
            self.data[aid] = series
            (self.pnl_dir / f"{aid}.json").write_text(json.dumps(series), encoding="utf-8")

    def test_cold_build_warm_hit_incremental_update_and_cutoffs(self):
        with mock.patch.object(pnl_snapshot, "read_series", wraps=pnl_snapshot.read_series) as cold_read:
            loaded = pnl_snapshot.load_series_map(self.pnl_dir)
        self.assertEqual(len(loaded), 5)
        self.assertEqual(cold_read.call_count, 5)
        parquet_file = self.pnl_dir / ".parquet_cache" / "pnl.parquet"
        self.assertTrue(parquet_file.exists())

        with mock.patch.object(pnl_snapshot, "read_series", wraps=pnl_snapshot.read_series) as warm_read:
            loaded_warm = pnl_snapshot.load_series_map(
                self.pnl_dir,
                ["ALPHA_0", "ALPHA_2"],
                cutoffs={"ALPHA_0": "2020-04-01", "ALPHA_2": "2020-06-01"},
            )
        self.assertEqual(warm_read.call_count, 0)
        self.assertEqual(set(loaded_warm), {"ALPHA_0", "ALPHA_2"})
        self.assertTrue(all(day < "2020-04-01" for day in loaded_warm["ALPHA_0"]))
        self.assertTrue(all(day < "2020-06-01" for day in loaded_warm["ALPHA_2"]))

        # Mutate one JSON file and delete another: only the modified file is re-read.
        updated = {**self.data["ALPHA_1"], "2021-12-31": 9.5}
        (self.pnl_dir / "ALPHA_1.json").write_text(json.dumps(updated), encoding="utf-8")
        (self.pnl_dir / "ALPHA_4.json").unlink()
        with mock.patch.object(pnl_snapshot, "read_series", wraps=pnl_snapshot.read_series) as inc_read:
            reloaded = pnl_snapshot.load_series_map(self.pnl_dir)
        self.assertEqual([call.args[0].stem for call in inc_read.call_args_list], ["ALPHA_1"])
        self.assertEqual(set(reloaded), {"ALPHA_0", "ALPHA_1", "ALPHA_2", "ALPHA_3"})
        self.assertEqual(reloaded["ALPHA_1"]["2021-12-31"], 9.5)

    def test_polars_unavailable_falls_back_to_json(self):
        with mock.patch.dict(sys.modules, {"polars": None}):
            loaded = pnl_snapshot.load_series_map(self.pnl_dir, ["ALPHA_0"], end_day="2020-03-01")
        self.assertEqual(set(loaded), {"ALPHA_0"})
        self.assertTrue(all(day < "2020-03-01" for day in loaded["ALPHA_0"]))


class VectorizedCorrelationParityTests(unittest.TestCase):
    """Vectorized BLAS/Polars correlation matches scalar alignment + Pearson bit-for-bit."""

    @staticmethod
    def _scalar_matrix(pnls, min_overlap=200):
        ids = sorted(pnls)
        pairs_out, skipped = [], []
        for i, left in enumerate(ids):
            for right in ids[i + 1:]:
                x, y, dates = correlation.align(pnls[left], pnls[right], min_overlap)
                r = correlation.pearson(x, y) if x else None
                if r is None:
                    skipped.append({"a": left, "b": right, "n": len(dates),
                                    "reason": "short_overlap" if not x else "undefined"})
                else:
                    pairs_out.append({"a": left, "b": right, "r": r, "n": len(dates)})
        return pairs_out, skipped

    def test_matrix_and_max_against_parity_with_edge_cases(self):
        pnls = {
            f"S_{i:02d}": _make_series(10 + i, count=340, skip_every=(5 if i % 3 == 0 else None))
            for i in range(15)
        }
        # Mirror, flat/constant, short overlap, large offset, extreme scales.
        pnls["MIRROR"] = {d: -2.0 * v for d, v in pnls["S_01"].items()}
        pnls["FLAT"] = {d: 3.14 for d in pnls["S_02"]}
        pnls["SHORT"] = _make_series(999, count=120)
        pnls["BIG_OFFSET"] = _make_series(77, count=340, offset=1e9, scale=1e-2)
        pnls["TINY_SCALE"] = _make_series(88, count=340, scale=1e-100)
        pnls["HUGE_SCALE"] = _make_series(88, count=340, scale=1e100)

        got = correlation.matrix(pnls, min_overlap=200)
        exp_pairs, exp_skipped = self._scalar_matrix(pnls, min_overlap=200)

        self.assertEqual(got["skipped"], exp_skipped)
        self.assertEqual(len(got["pairs"]), len(exp_pairs))
        for g, e in zip(got["pairs"], exp_pairs):
            self.assertEqual((g["a"], g["b"], g["n"]), (e["a"], e["b"], e["n"]))
            self.assertAlmostEqual(g["r"], e["r"], places=9)

        # Extreme scales identical underlying series => +1.0
        tiny_huge = next(p for p in got["pairs"] if {p["a"], p["b"]} == {"HUGE_SCALE", "TINY_SCALE"})
        self.assertAlmostEqual(tiny_huge["r"], 1.0, places=9)

        # Compare matrix_against_refs and max_against parity
        refs = {f"S_{i:02d}": pnls[f"S_{i:02d}"] for i in range(12)}
        batch = correlation.matrix_against_refs(
            {"MIRROR": pnls["MIRROR"], "BIG_OFFSET": pnls["BIG_OFFSET"]},
            refs,
            min_overlap=200,
        )
        for tid in ("MIRROR", "BIG_OFFSET"):
            with mock.patch.dict(sys.modules, {"numpy": None}):
                scalar_ref = correlation.max_against(pnls[tid], refs, min_overlap=200)
            self.assertEqual([r["with"] for r in batch[tid]["all"]], [r["with"] for r in scalar_ref["all"]])
            for g, e in zip(batch[tid]["all"], scalar_ref["all"]):
                self.assertEqual(g["n"], e["n"])
                self.assertAlmostEqual(g["r"], e["r"], places=9)


class WorkflowParityTests(unittest.TestCase):
    """Yearly batch analysis, vectorized pairs, and cluster_ledger cache reuse."""

    def test_yearly_train_batch_matches_scalar_yearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            pnl_dir = Path(tmp) / "pnl"
            (pnl_dir / ".meta").mkdir(parents=True)

            def write_alpha(aid, values, *, with_meta=True):
                (pnl_dir / f"{aid}.json").write_text(json.dumps(values), encoding="utf-8")
                if with_meta:
                    meta = {"format": ml._PNL_EVIDENCE_FORMAT, "alpha_id": aid,
                            "complete": True, "as_of": "2025-01-01",
                            "data_sha256": ml._pnl_digest(values)}
                    (pnl_dir / ".meta" / f"{aid}.json").write_text(json.dumps(meta), encoding="utf-8")

            good, bad, short = {}, {}, {}
            cur = date(2019, 1, 1)
            while cur < date(2023, 1, 1):
                if cur.weekday() < 5:
                    d = cur.isoformat()
                    good[d] = (-1) ** cur.toordinal() + 0.25 if cur.year < 2022 else -500.0
                    bad[d] = (-1) ** cur.toordinal() - 0.35 if cur.year < 2022 else 500.0
                    if cur.year == 2020:
                        short[d] = 0.5
                cur += timedelta(days=1)

            write_alpha("A_GOOD", good)
            write_alpha("A_FAIL", bad)
            write_alpha("A_SHORT", short)
            write_alpha("A_NOMETA", good, with_meta=False)
            (pnl_dir / "A_CORRUPT.json").write_text("{bad json", encoding="utf-8")

            rows = [
                {"alpha_id": aid, "train_start": "2019-01-01", "test_start": "2022-01-01"}
                for aid in ("A_GOOD", "A_FAIL", "A_SHORT", "A_NOMETA", "A_CORRUPT", "A_MISSING", "bad/id")
            ]
            batch = pnl_snapshot.yearly_train_batch(rows, pnl_dir)
            for row, got in zip(rows, batch):
                expected = ledger_analysis._yearly(row, pnl_dir)
                self.assertEqual(got.keys(), expected.keys(), msg=f"keys for {row['alpha_id']}")
                self.assertEqual(got["state"], expected["state"], msg=f"state for {row['alpha_id']}")
                if "years" in expected:
                    self.assertEqual(len(got["years"]), len(expected["years"]))
                    for gy, ey in zip(got["years"], expected["years"]):
                        self.assertEqual((gy["year"], gy["points"]), (ey["year"], ey["points"]))
                        if ey["sharpe"] is None:
                            self.assertIsNone(gy["sharpe"])
                        else:
                            self.assertAlmostEqual(gy["sharpe"], ey["sharpe"], places=12)
                for key in ("reason", "as_of", "three_weak_years"):
                    if key in expected:
                        self.assertEqual(got[key], expected[key])
                for key in ("worst_sharpe", "recent_three_mean"):
                    if key in expected:
                        self.assertAlmostEqual(got[key], expected[key], places=12)

    def test_vectorized_pairs_and_cluster_ledger_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pnl_dir = root / "pnl"
            pnl_dir.mkdir()
            registry = root / "alpha-registry.jsonl"
            rng = Random(42)
            days = [(date(2020, 1, 1) + timedelta(days=k)).isoformat() for k in range(200)]

            book = {d: rng.uniform(-1.0, 1.0) + 0.05 for d in days}
            p1 = {d: (-book[d] + rng.uniform(-0.2, 0.2)) for d in days}
            p2 = {d: (book[d] + rng.uniform(-0.2, 0.2)) for d in days}
            p3 = {d: rng.uniform(-1.0, 1.0) + 0.12 for d in days}
            series_map = {"BOOK1": book, "POOL1": p1, "POOL2": p2, "POOL3": p3}

            entries = []
            for aid, vals in series_map.items():
                (pnl_dir / f"{aid}.json").write_text(json.dumps(vals), encoding="utf-8")
                entries.append({
                    "key": aid, "alpha_id": aid,
                    "status": "ACTIVE" if aid == "BOOK1" else "COMPLETE",
                    "expression": "rank(close)",
                    "metrics": {"sharpe": 1.2, "returns": 0.05, "turnover": 0.2,
                                "longCount": 100, "shortCount": 100},
                })
            registry.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

            # Test cluster_ledger.load_pnls hits cache without calling ml.fetch_daily_pnl for cached IDs.
            with mock.patch.object(ml, "fetch_daily_pnl", side_effect=AssertionError("unexpected fetch")) as mocked:
                loaded, failed = cluster_ledger.load_pnls(None, ["BOOK1", "POOL1", "POOL3"], str(pnl_dir))
            self.assertEqual(mocked.call_count, 0)
            self.assertEqual(failed, [])
            self.assertEqual(set(loaded), {"BOOK1", "POOL1", "POOL3"})

            # Test pairs.cmd_pairs vectorized results.
            args = argparse.Namespace(
                registry=str(registry),
                min_leg_sharpe=None, max_leg_sharpe=None,
                max_book_rho=0.85, min_marginal=None, min_composite_sharpe=None,
                min_support_ratio=None, plain_legs=False, rank="marginal", limit=10,
            )
            with mock.patch.object(research_paths, "ensure_dir", lambda: str(root)):
                out = pairs.cmd_pairs(args, None)
            self.assertEqual(out["pool"], 3)
            self.assertEqual(out["book"], 1)
            self.assertGreaterEqual(len(out["pairs"]), 1)
            for row in out["pairs"]:
                self.assertLessEqual(row["max_book_rho"], 0.85)
                self.assertTrue(math.isfinite(row["composite_sharpe"]))


if __name__ == "__main__":
    unittest.main()
