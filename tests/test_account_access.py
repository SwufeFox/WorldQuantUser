"""Account discovery must not turn invalid defaults or outages into absence."""

from types import SimpleNamespace
import unittest
from unittest import mock

import machine_lib as ml
from research_app import account


def options(regions=("USA",), universes=None, delays=None):
    universes = universes or {region: ["TOP3000"] for region in regions}
    delays = delays or {region: [1, 0] for region in regions}
    return {"actions": {"POST": {"settings": {"children": {
        "instrumentType": {"choices": [{"value": "EQUITY"}]},
        "region": {"choices": {"instrumentType": {"EQUITY": [{"value": region} for region in regions]}}},
        "delay": {"choices": {"instrumentType": {"EQUITY": {"region": {
            region: [{"value": value} for value in delays[region]] for region in regions}}}}},
        "universe": {"choices": {"instrumentType": {"EQUITY": {"region": {
            region: [{"value": value} for value in universes[region]] for region in regions}}}}},
    }}}}}


class AccountAccessTests(unittest.TestCase):
    def fresh(self, session, **kwargs):
        self.assertTrue(kwargs["refresh"])
        kwargs["report"].update(complete=True, hit=False)
        return [{"id": "sample_dataset"}]

    def test_only_advertised_regions_are_probed_with_live_metadata(self):
        session = object()
        with mock.patch.object(ml, "request_json", return_value=options()) as request, \
                mock.patch.object(ml, "get_datasets", side_effect=self.fresh) as datasets:
            result = account._data_access(session)
        request.assert_called_once_with(session, "OPTIONS", "/simulations")
        self.assertEqual(datasets.call_count, 1)
        self.assertEqual(datasets.call_args.kwargs["region"], "USA")
        self.assertEqual(result["regions"], ["USA"])
        self.assertEqual(result["state"], "verified")
        self.assertIn("not all combinations", result["coverage"])

    def test_region_specific_universe_and_delay_replace_unsupported_defaults(self):
        schema = options(("EUR",), {"EUR": ["TOP2500", "TOP1200"]}, {"EUR": [0]})
        with mock.patch.object(ml, "request_json", return_value=schema), \
                mock.patch.object(ml, "get_datasets", side_effect=self.fresh) as datasets:
            result = account._data_access(object())
        self.assertEqual(datasets.call_args.kwargs["universe"], "TOP2500")
        self.assertEqual(datasets.call_args.kwargs["delay"], 0)
        self.assertEqual(result["regions"], ["EUR"])

    def test_empty_catalog_is_only_an_empty_sample(self):
        def catalogs(session, **kwargs):
            kwargs["report"].update(complete=True, hit=False)
            return [] if kwargs["region"] == "EUR" else [{"id": "sample"}]
        with mock.patch.object(ml, "request_json", return_value=options(("USA", "EUR"))), \
                mock.patch.object(ml, "get_datasets", side_effect=catalogs):
            result = account._data_access(object())
        self.assertEqual(result["regions"], ["USA"])
        self.assertEqual([row["state"] for row in result["probes"]], ["available", "empty"])
        self.assertEqual(result["state"], "verified")

    def test_cached_positive_results_do_not_prove_live_access(self):
        for flags in ({"hit": True}, {"stale": True}, {"degraded": True},
                      {"error": "RATE_LIMITED"}):
            with self.subTest(flags=flags):
                def catalogs(session, **kwargs):
                    kwargs["report"].update({"complete": True, "hit": False, **flags})
                    return [{"id": "old_dataset"}]
                with mock.patch.object(ml, "request_json", return_value=options()), \
                        mock.patch.object(ml, "get_datasets", side_effect=catalogs):
                    result = account._data_access(object())
                self.assertEqual(result["regions"], [])
                self.assertEqual(result["state"], "unknown")
                self.assertEqual(result["probes"][0]["state"], "unknown")

    def test_live_partial_positive_catalog_proves_existence(self):
        def catalogs(session, **kwargs):
            kwargs["report"].update(complete=False, hit=False)
            return [{"id": "first_dataset"}]
        with mock.patch.object(ml, "request_json", return_value=options()), \
                mock.patch.object(ml, "get_datasets", side_effect=catalogs):
            result = account._data_access(object())
        self.assertEqual(result["regions"], ["USA"])
        self.assertEqual(result["probes"][0]["state"], "available")
        self.assertFalse(result["probes"][0]["metadata"]["complete"])

    def test_live_incomplete_empty_catalog_cannot_prove_absence(self):
        def catalogs(session, **kwargs):
            kwargs["report"].update(complete=False, hit=False)
            return []
        with mock.patch.object(ml, "request_json", return_value=options()), \
                mock.patch.object(ml, "get_datasets", side_effect=catalogs):
            result = account._data_access(object())
        self.assertEqual(result["regions"], [])
        self.assertEqual(result["probes"][0]["state"], "unknown")
        self.assertEqual(result["state"], "unknown")

    def test_one_catalog_error_preserves_other_regions_and_unknown_receipt(self):
        def catalogs(session, **kwargs):
            if kwargs["region"] == "EUR":
                raise ml.RateLimited("synthetic rate limit")
            return self.fresh(session, **kwargs)
        with mock.patch.object(ml, "request_json", return_value=options(("USA", "EUR"))), \
                mock.patch.object(ml, "get_datasets", side_effect=catalogs):
            result = account._data_access(object())
        self.assertEqual(result["regions"], ["USA"])
        self.assertEqual(result["probes"][1]["error"]["code"], "RATE_LIMITED")
        self.assertEqual(result["probes"][1]["state"], "unknown")
        self.assertEqual(result["state"], "unknown")

    def test_unavailable_options_do_not_trigger_guessed_scope_probes(self):
        with mock.patch.object(ml, "request_json", side_effect=ml.RateLimited("synthetic")), \
                mock.patch.object(ml, "get_datasets") as datasets:
            result = account._data_access(object())
        datasets.assert_not_called()
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(result["settings"]["error"]["code"], "RATE_LIMITED")

    def test_authentication_failures_are_not_silenced_as_empty_regions(self):
        for during_catalog in (False, True):
            with self.subTest(during_catalog=during_catalog), \
                    mock.patch.object(ml, "request_json", return_value=options(),
                                      side_effect=None if during_catalog else ml.AuthError("synthetic")), \
                    mock.patch.object(ml, "get_datasets", side_effect=ml.AuthError("synthetic")):
                with self.assertRaises(ml.AuthError):
                    account._data_access(object())

    def test_missing_choice_contract_stays_unknown_without_defaults(self):
        schema = options()
        del schema["actions"]["POST"]["settings"]["children"]["universe"]
        with mock.patch.object(ml, "request_json", return_value=schema), \
                mock.patch.object(ml, "get_datasets") as datasets:
            result = account._data_access(object())
        datasets.assert_not_called()
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(result["settings"]["issues"][0]["reason"], "universe_choices_unavailable")

    def test_delay_dependent_universe_choices_are_resolved(self):
        schema = options(("EUR",), {"EUR": ["TOP2500"]}, {"EUR": [1]})
        schema["actions"]["POST"]["settings"]["children"]["universe"]["choices"]["instrumentType"]["EQUITY"]["region"]["EUR"] = {
            "delay": {"1": [{"value": "TOP2500"}]}}
        with mock.patch.object(ml, "request_json", return_value=schema), \
                mock.patch.object(ml, "get_datasets", side_effect=self.fresh) as datasets:
            account._data_access(object())
        self.assertEqual(datasets.call_args.kwargs["universe"], "TOP2500")

    def test_auth_command_keeps_regions_and_exposes_scope_evidence(self):
        args = SimpleNamespace(no_cache=False, relogin=False)
        with mock.patch.object(ml, "load_session", return_value=object()), \
                mock.patch.object(ml, "session_status", return_value={"user_id": "synthetic"}), \
                mock.patch.object(ml, "request_json", return_value=options()), \
                mock.patch.object(ml, "get_datasets", side_effect=self.fresh):
            result = account.cmd_auth_check(args, None)
        self.assertTrue(result["authenticated"])
        self.assertEqual(result["data_regions"], ["USA"])
        self.assertEqual(result["data_scope_evidence"]["state"], "verified")
        self.assertIn("unprobed", result["note"])

    def test_shared_deadline_preserves_earlier_region_and_marks_unprobed_unknown(self):
        clock = [100.0]

        def catalogs(session, **kwargs):
            clock[0] = 106.0
            return self.fresh(session, **kwargs)

        session = object()
        with mock.patch.object(account.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(ml, "request_json", return_value=options(("USA", "EUR"))) as request, \
                mock.patch.object(ml, "get_datasets", side_effect=catalogs) as datasets:
            result = account._data_access(session, deadline=105.0)
        request.assert_called_once_with(session, "OPTIONS", "/simulations", deadline=105.0, max_retries=1)
        self.assertEqual(datasets.call_count, 1)
        self.assertEqual(datasets.call_args.kwargs["deadline"], 105.0)
        self.assertEqual(result["regions"], ["USA"])
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(result["probes"][1]["reason"], "diagnostic_budget_expired")

    def test_dataset_catalog_uses_probe_deadline_and_does_not_retry(self):
        session = object()
        payload = {"results": [{"id": "sample"}], "count": 1}
        with mock.patch.object(ml, "request_json", return_value=payload) as request:
            self.assertEqual(ml.get_datasets(session, limit=1, deadline=105.0), payload["results"])
        self.assertEqual(request.call_args.args, (session, "GET", "/data-sets"))
        self.assertEqual(request.call_args.kwargs["deadline"], 105.0)
        self.assertEqual(request.call_args.kwargs["max_retries"], 1)

    def test_malformed_settings_do_not_make_up_regions(self):
        for schema in ([], {"actions": []}, {"actions": {"POST": {"settings": "unknown"}}}):
            with self.subTest(schema=schema), \
                    mock.patch.object(ml, "request_json", return_value=schema), \
                    mock.patch.object(ml, "get_datasets") as datasets:
                result = account._data_access(object())
            datasets.assert_not_called()
            self.assertEqual(result["state"], "unknown")

    def test_malformed_catalog_does_not_prove_available_or_empty(self):
        def catalogs(session, **kwargs):
            kwargs["report"].update(complete=True, hit=False)
            return [{"name": "no identifier"}]
        with mock.patch.object(ml, "request_json", return_value=options()), \
                mock.patch.object(ml, "get_datasets", side_effect=catalogs):
            result = account._data_access(object())
        self.assertEqual(result["regions"], [])
        self.assertEqual(result["probes"][0]["state"], "unknown")

    def test_new_region_universe_and_delay_are_discovered_without_historical_lists(self):
        schema = options(("NEWREGION",), {"NEWREGION": ["NEWUNIVERSE"]}, {"NEWREGION": [5]})
        with mock.patch.object(ml, "request_json", return_value=schema), \
                mock.patch.object(ml, "get_datasets", side_effect=self.fresh) as datasets:
            result = account._data_access(object())
        self.assertEqual(result["regions"], ["NEWREGION"])
        self.assertEqual(datasets.call_args.kwargs["delay"], 5)
        self.assertEqual(datasets.call_args.kwargs["universe"], "NEWUNIVERSE")
        self.assertEqual(result["capabilities"]["concurrency"]["state"], "unknown")

    def test_advertised_empty_instruments_are_distinct_from_unpublished(self):
        schema = options()
        schema["actions"]["POST"]["settings"]["children"]["instrumentType"]["choices"] = []
        with mock.patch.object(ml, "request_json", return_value=schema), \
                mock.patch.object(ml, "get_datasets") as datasets:
            result = account._data_access(object())
        datasets.assert_not_called()
        self.assertEqual(result["state"], "verified")
        self.assertFalse(result["settings"]["equity_supported"])


if __name__ == "__main__":
    unittest.main()
