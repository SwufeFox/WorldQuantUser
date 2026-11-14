"""Offline field review must not turn unknown semantics into simulation arms."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import research_paths
from research_app import mine
from research_app.runtime import CliError


class MineTests(unittest.TestCase):
    def _workspace(self, fields, expressions=(), entries=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp = temporary.name
        directory = os.path.join(tmp, "datafields")
        os.makedirs(directory)
        with open(os.path.join(directory, "a.json"), "w", encoding="utf-8") as handle:
            json.dump({"payload": fields}, handle)
        registry = os.path.join(tmp, "r.jsonl")
        if entries is None:
            entries = [{"alpha_id": "A%d" % index, "expression": expression,
                        "out_of_sample": {"train": {"turnover": 0.12}}}
                       for index, expression in enumerate(expressions)]
        with open(registry, "w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry) + "\n")
        return tmp, registry

    def _args(self, registry, **overrides):
        values = dict(dataset=None, min_coverage=0.5, turnover_band="0.08,0.25",
                      limit=20, arms=20, smooth=0, emit=None, name="t",
                      registry=registry, expr=None, shape="raw")
        values.update(overrides)
        return SimpleNamespace(**values)

    def _field(self, field_id, description, field_type="MATRIX", dataset="analyst4"):
        return {"id": field_id, "dataset": {"id": dataset}, "type": field_type,
                "coverage": 0.9, "description": description}

    def test_numeric_counts_are_emitted_but_labels_vectors_and_unknowns_need_definition(self):
        fields = [
            self._field("numeric_value", "Annual sales revenue in reporting currency"),
            self._field("analyst_count", "Number of contributing analyst estimates"),
            self._field("estimate_vec", "Individual analyst estimate values", "VECTOR"),
            self._field("report_events", "Reported earnings observations", "EVENT"),
            self._field("peer_group", "Industry grouping field", "GROUP"),
            self._field("forecast_flag", "Forecast type: revision, new or confirmed"),
            self._field("currency_id", "Pricing currency code"),
            self._field("report_date", "Date of the latest financial report"),
            self._field("mystery_value", "Estimation value"),
        ]
        tmp, registry = self._workspace(fields)
        emitted = os.path.join(tmp, "spec.json")
        with mock.patch.object(research_paths, "metadata_root", return_value=tmp):
            data = mine.cmd_mine(self._args(registry, emit=emitted), None)
        with open(emitted, encoding="utf-8") as handle:
            spec = json.load(handle)
        row = data["datasets"][0]
        self.assertEqual(row["unused"], 9)
        self.assertEqual(row["candidates"], 2)
        self.assertEqual(row["needs_definition_count"], 7)
        self.assertEqual({item["id"] for item in row["top"]},
                         {"numeric_value", "analyst_count"})
        review = {item["id"]: item for item in row["needs_definition"]}
        for field_id in ("estimate_vec", "report_events"):
            self.assertIn("reduction_semantics_unknown", review[field_id]["reasons"])
            self.assertIsNone(review[field_id]["wrapper"])
        for field_id in ("forecast_flag", "currency_id", "peer_group"):
            self.assertIn("categorical_or_identifier", review[field_id]["reasons"])
        self.assertIn("date_encoding_needs_definition", review["report_date"]["reasons"])
        self.assertIn("measurement_definition_missing", review["mystery_value"]["reasons"])
        self.assertEqual({arm["expression"] for arm in spec["arms"]},
                         {"rank(numeric_value)", "rank(analyst_count)"})
        self.assertEqual(data["spec"]["arms"], 2)
        self.assertFalse(data["catalog"]["live_verified"])
        self.assertFalse(data["catalog"]["account_verified"])

    def test_an_explicit_vector_reduction_preserves_the_users_choice(self):
        tmp, registry = self._workspace([
            self._field("estimate_vec", "Individual analyst estimate values", "VECTOR"),
        ])
        emitted = os.path.join(tmp, "explicit.json")
        args = self._args(registry, emit=emitted, expr="vec_sum(estimate_vec)")
        with mock.patch.object(research_paths, "metadata_root", return_value=tmp):
            data = mine.cmd_mine(args, None)
        with open(emitted, encoding="utf-8") as handle:
            spec = json.load(handle)
        self.assertEqual([arm["expression"] for arm in spec["arms"]],
                         ["rank(vec_sum(estimate_vec))"])
        self.assertEqual(data["datasets"][0]["candidates"], 0)
        self.assertEqual(data["datasets"][0]["needs_definition_count"], 1)

    def test_only_undefined_reductions_cannot_silently_emit_a_spec(self):
        tmp, registry = self._workspace([
            self._field("estimate_vec", "Individual analyst estimate values", "VECTOR"),
            self._field("report_events", "Reported earnings observations", "EVENT"),
        ])
        emitted = os.path.join(tmp, "unsafe.json")
        with mock.patch.object(research_paths, "metadata_root", return_value=tmp):
            with self.assertRaisesRegex(CliError, "nothing to emit") as caught:
                mine.cmd_mine(self._args(registry, emit=emitted), None)
        self.assertEqual(caught.exception.code, "INVALID_INPUT")
        self.assertFalse(os.path.exists(emitted))

    def test_emitting_refuses_to_replace_a_previously_reviewed_spec(self):
        tmp, registry = self._workspace([
            self._field("numeric_value", "Annual sales revenue in reporting currency"),
        ])
        emitted = os.path.join(tmp, "reviewed.json")
        original = '{"name":"already reviewed","arms":[{"expression":"rank(assets)"}]}\n'
        with open(emitted, "w", encoding="utf-8") as handle:
            handle.write(original)
        with mock.patch.object(research_paths, "metadata_root", return_value=tmp):
            with self.assertRaisesRegex(CliError, "overwrite") as caught:
                mine.cmd_mine(self._args(registry, emit=emitted), None)
        self.assertEqual(caught.exception.code, "INVALID_INPUT")
        with open(emitted, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), original)

    def test_smoothing_is_part_of_the_first_shape(self):
        fields = [self._field("buzz", "Relative daily attention volume", dataset="socialmedia12")]
        tmp, registry = self._workspace(fields, [])
        args = self._args(registry, dataset="socialmedia12", smooth=21,
                          emit=os.path.join(tmp, "s.json"))
        with mock.patch.object(research_paths, "metadata_root", return_value=tmp):
            mine.cmd_mine(args, None)
        with open(args.emit, encoding="utf-8") as handle:
            spec = json.load(handle)
        self.assertEqual(spec["arms"][0]["expression"], "rank(ts_mean(buzz, 21))")

    def test_dataset_turnover_uses_train_and_counts_missing_or_invalid_train(self):
        catalog = {"slow_field": {"dataset": "fundamental2"}}
        entries = [
            {"expression": "rank(slow_field)", "metrics": {"turnover": 0.8},
             "out_of_sample": {"train": {"turnover": 0.02}}},
            {"expression": "rank(slow_field)", "metrics": {"turnover": 0.7}},
            {"expression": "rank(slow_field)", "metrics": {"turnover": 0.6},
             "out_of_sample": {"train": {"turnover": True}}},
            {"expression": "rank(slow_field)", "metrics": {"turnover": 0.5},
             "out_of_sample": {"train": {"turnover": float("nan")}}},
        ]
        stats = mine.dataset_turnover(entries, catalog=catalog)
        self.assertEqual(stats["fundamental2"]["n"], 1)
        self.assertEqual(stats["fundamental2"]["median"], 0.02)
        self.assertEqual(stats["fundamental2"]["missing_train"], 3)

    def test_full_window_turnover_alone_stays_unknown(self):
        stats = mine.dataset_turnover(
            [{"expression": "rank(slow_field)", "metrics": {"turnover": 0.12}}],
            catalog={"slow_field": {"dataset": "fundamental2"}})
        self.assertIsNone(stats["fundamental2"]["median"])
        self.assertEqual(stats["fundamental2"]["n"], 0)
        self.assertEqual(stats["fundamental2"]["missing_train"], 1)

    def test_explicit_registry_controls_both_unused_fields_and_train_turnover(self):
        fields = [self._field("assets", "Total assets", dataset="fundamental2"),
                  self._field("cash", "Cash and cash equivalents", dataset="fundamental2")]
        tmp, registry = self._workspace(fields, entries=[
            {"alpha_id": "EXPLICIT", "expression": "rank(assets)",
             "metrics": {"turnover": 0.8},
             "out_of_sample": {"train": {"turnover": 0.02}}},
        ])
        default_registry = os.path.join(tmp, "default.jsonl")
        with open(default_registry, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"alpha_id": "DEFAULT", "expression": "rank(cash)",
                                     "out_of_sample": {"train": {"turnover": 0.2}}}) + "\n")
        with mock.patch.object(research_paths, "metadata_root", return_value=tmp), \
                mock.patch.object(mine.alpha_registry, "default_registry_path", return_value=default_registry):
            data = mine.cmd_mine(self._args(registry, turnover_band="0.01,0.06"), None)
        row = data["datasets"][0]
        self.assertEqual([item["id"] for item in row["top"]], ["cash"])
        self.assertEqual(row["median_turnover"], 0.02)
        self.assertEqual(row["turnover_samples"], 1)
        self.assertTrue(row["turnover_in_band"])

    def test_ast_usage_ignores_comments_strings_and_local_variables(self):
        expression = ('local_shadow = assets; '
                      'group_rank(local_shadow, bucket(rank(assets), range="ghost_in_string")) '
                      '/* ghost_in_comment */')
        _, registry = self._workspace([], [expression])
        self.assertEqual(mine.used_identifiers(registry), {"assets"})

    def test_same_description_history_is_reviewed_without_claiming_an_alias(self):
        fields = [
            self._field("used_measure", "Future minimum lease payments.", dataset="fundamental2"),
            self._field("alternate_measure", "  FUTURE   minimum lease payments  ", dataset="fundamental2"),
            self._field("new_measure", "Annual amortization expense", dataset="fundamental2"),
        ]
        tmp, registry = self._workspace(fields, entries=[
            {"alpha_id": "USED", "expression": "rank(used_measure)",
             "verdict": "rejected", "mechanism": "lease_level",
             "out_of_sample": {"train": {"turnover": 0.03}}},
        ])
        emitted = os.path.join(tmp, "reviewed.json")
        with mock.patch.object(research_paths, "metadata_root", return_value=tmp):
            data = mine.cmd_mine(self._args(registry, emit=emitted), None)
        row = data["datasets"][0]
        self.assertEqual(row["unused"], 2)
        self.assertEqual(row["candidates"], 1)
        self.assertEqual(row["needs_definition_count"], 1)
        review = row["needs_definition"][0]
        self.assertEqual(review["id"], "alternate_measure")
        self.assertIn("same_description_already_used", review["reasons"])
        self.assertEqual(review["same_description_fields"], ["used_measure"])
        self.assertEqual(review["related_history"], [
            {"field": "used_measure", "key": "USED", "verdict": "rejected",
             "mechanism": "lease_level"},
        ])
        self.assertIn("not proven aliases", data["note"])
        with open(emitted, encoding="utf-8") as handle:
            spec = json.load(handle)
        self.assertEqual([arm["expression"] for arm in spec["arms"]], ["rank(new_measure)"])


class CatalogEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = temporary.name
        self.directory = os.path.join(self.root, "datafields")
        os.makedirs(self.directory)
        self.now = datetime(2026, 11, 15, 12, tzinfo=timezone.utc)

    def _write(self, name, rows, created=None, expires=None):
        document = {"payload": rows}
        if created or expires:
            document["_cache"] = {"created_at": created, "expires_at": expires}
        path = os.path.join(self.directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return path

    def _row(self, **overrides):
        row = {"id": "amount", "dataset": "fundamental2", "type": "MATRIX",
               "coverage": 0.8, "description": "Total reported income",
               "region": "USA", "universe": "TOP3000", "delay": 1,
               "instrumentType": "EQUITY"}
        row.update(overrides)
        return row

    def _load(self, report):
        scope = {"region": "USA", "universe": "TOP3000", "delay": 1,
                 "instrumentType": "EQUITY"}
        with mock.patch.object(research_paths, "metadata_root", return_value=self.root), \
                mock.patch.object(mine.cache_store, "utc_now", return_value=self.now), \
                mock.patch.dict(mine.ml.DEFAULT_SETTINGS, scope):
            return mine.load_catalog(report)

    def test_known_wrong_scopes_are_excluded_from_default_catalog(self):
        self._write("scope.json", [self._row(), self._row(id="wrong_pool", universe="TOP500"),
                                   self._row(id="wrong_region", region="EUR")])
        evidence = {}
        catalog = self._load(evidence)
        self.assertEqual(set(catalog), {"amount"})
        self.assertEqual(evidence["scope"]["universe"], "TOP3000")
        self.assertEqual(evidence["ignored_scope_rows"], 2)
        self.assertEqual(catalog["amount"]["source"]["scope_state"], "matched")
        self.assertFalse(evidence["live_verified"])
        self.assertFalse(evidence["account_verified"])

    def test_latest_scoped_cache_wins_even_when_older_file_sorts_last(self):
        newest = self._write("a_new.json", [self._row(coverage=0.8)],
                             created="2026-11-15T11:00:00Z", expires="2026-11-15T16:00:00Z")
        self._write("z_old.json", [self._row(coverage=0.4)],
                    created="2026-11-15T10:00:00Z", expires="2026-11-15T16:00:00Z")
        # A later, fresh TOP500 record cannot override the target-scope record.
        self._write("zz_other_pool.json", [self._row(coverage=1.0, universe="TOP500")],
                    created="2026-11-15T11:59:00Z", expires="2026-11-15T17:00:00Z")
        evidence = {}
        catalog = self._load(evidence)
        self.assertEqual(catalog["amount"]["coverage"], 0.8)
        source = catalog["amount"]["source"]
        self.assertEqual(source["path"], newest)
        self.assertEqual(source["created_at"], "2026-11-15T11:00:00Z")
        self.assertEqual(source["age_seconds"], 3600.0)
        self.assertEqual(source["expires_at"], "2026-11-15T16:00:00Z")
        self.assertEqual(source["freshness"], "fresh")
        self.assertEqual(evidence["freshness"], {"fresh": 1})
        self.assertEqual(evidence["ignored_scope_rows"], 1)

    def test_envelope_expiry_and_legacy_unknown_scope_remain_visible(self):
        stale = self._write("stale.json", [self._row()],
                            created="2026-11-15T09:00:00Z", expires="2026-11-15T11:00:00Z")
        self._write("legacy.json", [{"id": "legacy_count", "dataset": "analyst4",
                                    "type": "MATRIX", "coverage": 0.9,
                                    "description": "Number of analyst estimates"}])
        evidence = {}
        catalog = self._load(evidence)
        self.assertEqual(catalog["amount"]["source"]["path"], stale)
        self.assertEqual(catalog["amount"]["source"]["freshness"], "stale")
        self.assertEqual(catalog["amount"]["source"]["age_seconds"], 10800.0)
        self.assertEqual(catalog["amount"]["source"]["expires_at"], "2026-11-15T11:00:00Z")
        legacy_source = catalog["legacy_count"]["source"]
        self.assertEqual(legacy_source["scope_state"], "unknown")
        self.assertEqual(legacy_source["freshness"], "unknown")
        self.assertIsNone(legacy_source["created_at"])
        self.assertIsNone(legacy_source["age_seconds"])
        self.assertIsNone(legacy_source["expires_at"])
        self.assertEqual(evidence["freshness"], {"stale": 1, "unknown": 1})
        self.assertEqual(evidence["source"], "local_metadata_cache")
        self.assertFalse(evidence["live_verified"])


class ShapeTests(unittest.TestCase):
    """Shape-first mining: keep the construction, change the field."""

    def test_the_known_shapes_apply_to_a_sub_expression(self):
        expression = "deferred_tax_assets_net_value / (assets + 1)"
        self.assertEqual(mine.apply_shape("centre", expression),
                         "-(group_rank(%s, subindustry) - 0.5)" % expression)
        self.assertEqual(mine.apply_shape("midtier", expression),
                         "-(abs(group_rank(%s, subindustry) - 0.5))" % expression)
        self.assertIn("ts_delta(", mine.apply_shape("change", expression))

    def test_an_unknown_shape_is_refused(self):
        with self.assertRaises(ValueError):
            mine.apply_shape("magic", "rank(close)")


class TurnoverBandTests(unittest.TestCase):
    """A field's cadence bounds the turnover any construction on it can reach."""

    def test_a_single_dataset_gives_a_point_band(self):
        import research_app.mine as mine_module
        catalog = {"assets": {"dataset": "fundamental6", "type": "MATRIX",
                              "coverage": 1.0, "description": ""}}
        stats = {"fundamental6": {"n": 5, "median": 0.02}}
        band = mine_module.expected_turnover("rank(assets)", catalog=catalog, stats=stats)
        self.assertEqual(band["low"], 0.02)
        self.assertEqual(band["high"], 0.02)

    def test_a_long_window_on_slow_fields_is_flagged(self):
        caution = mine.long_window_caution("rank(ts_delta(ts_backfill(x, 252), 252))")
        self.assertIsNotNone(caution)
        self.assertIsNone(mine.long_window_caution("rank(ts_delta(x, 5))"))

    def test_a_band_that_straddles_the_floor_is_reported_as_mixed(self):
        verdict = mine.turnover_verdict({"low": 0.02, "high": 0.12})
        self.assertIn("mixed", verdict)
