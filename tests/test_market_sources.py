"""Offline market semantics, validation, authentication, and identity coverage."""

import copy
import json
import os
import unittest
from unittest import mock

import market_sources
from source_io import SourceError


class MarketSourcesTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {"collected_at": "2026-11-17T12:00:00+00:00"}
        self.book = {"sequence": 100, "bids": [["99", "2", 7], ["100", "3", 2]],
                     "asks": [["102", "1", 1], ["101", "4", 9]], "auction_mode": False}
        self.quote = {"symbol": "AAPL", "quote": {
            "bp": 100, "bs": 2, "ap": 101, "as": 3, "bx": "V", "ax": "V",
            "t": "2026-11-17T06:59:00.123456789-05:00", "c": ["R"], "z": "C"}}
        self.dotenv = mock.patch("machine_lib._load_dotenv")
        self.dotenv.start()
        self.addCleanup(self.dotenv.stop)
        self.credentials = mock.patch.dict(os.environ, {
            "APCA_API_KEY_ID": "offline-key-value", "APCA_API_SECRET_KEY": "offline-secret-value"})
        self.credentials.start()
        self.addCleanup(self.credentials.stop)

    def fetch(self, payload=None, venue="coinbase", symbol="BTC-USD", **kwargs):
        with mock.patch("market_sources.request_json", return_value=(
                self.book if payload is None else payload, self.metadata)) as request:
            packet = market_sources.fetch_market(venue, symbol, **kwargs)
        return packet, request

    def test_l2_is_aggregated_full_provider_book_with_local_depth_cap(self):
        packet, request = self.fetch(depth=1, timeout=4)
        item = packet["items"][0]
        market = item["market"]
        request.assert_called_once_with("https://api.exchange.coinbase.com/products/BTC-USD/book?level=2",
                                        timeout=4, headers=None)
        self.assertEqual((market["asset_class"], market["feed"], market["depth_kind"]),
                         ("crypto", "exchange", "aggregated_l2"))
        self.assertEqual(market["bids"], [[100.0, 3.0]])
        self.assertEqual(market["asks"], [[101.0, 4.0]])
        self.assertEqual(market["provider_bid_levels"], 2)
        self.assertEqual(market["valid_bid_levels"], 2)
        self.assertEqual(market["bid_levels"], 1)
        self.assertEqual(market["omitted_bid_levels"], 1)
        self.assertEqual(market["derived"]["mid"], 100.5)
        self.assertEqual(market["derived"]["spread"], 1)
        self.assertAlmostEqual(market["derived"]["imbalance"], -1 / 7)
        self.assertAlmostEqual(market["derived"]["spread_bps"], 1 / 100.5 * 10000)
        self.assertFalse(market["derived"]["crossed"])
        self.assertEqual(market["size_unit"], "provider_native_size")
        self.assertEqual(market["price_unit"], "USD")
        self.assertEqual(item["raw"], self.book)
        self.assertFalse(packet["cached"])

    def test_missing_source_time_stays_unknown(self):
        packet, _ = self.fetch()
        item = packet["items"][0]
        self.assertIsNone(item["published_at"])
        self.assertIsNone(item["published_at_raw"])
        self.assertEqual(item["time_precision"], "unknown")
        self.assertEqual(item["collected_at"], self.metadata["collected_at"])
        self.assertIsNone(item["market"]["source_timestamp"])

    def test_l1_iex_is_explicit_and_retains_round_lots(self):
        packet, request = self.fetch(self.quote, "alpaca", "AAPL", depth=50)
        market = packet["items"][0]["market"]
        self.assertEqual(request.call_args.args[0],
                         "https://data.alpaca.markets/v2/stocks/AAPL/quotes/latest?feed=iex")
        self.assertEqual(request.call_args.kwargs["headers"], {
            "APCA-API-KEY-ID": "offline-key-value", "APCA-API-SECRET-KEY": "offline-secret-value"})
        self.assertEqual((market["asset_class"], market["feed"], market["depth_kind"]),
                         ("equity", "iex", "feed_l1"))
        self.assertIn("not national NBBO", market["scope"])
        self.assertEqual(market["requested_depth"], 50)
        self.assertEqual(market["bid_levels"], 1)
        self.assertEqual(market["size_unit"], "round_lots")
        self.assertEqual(market["bids"], [[100.0, 2.0]])
        self.assertEqual(market["bid_exchange"], "V")
        self.assertEqual(market["quote_conditions"], ["R"])
        self.assertEqual(market["source_timestamp"], "2026-11-17T11:59:00.123456+00:00")
        self.assertEqual(market["source_timestamp_raw"], self.quote["quote"]["t"])
        serialized = json.dumps(packet)
        self.assertNotIn("offline-key-value", serialized)
        self.assertNotIn("offline-secret-value", serialized)

    def test_explicit_feed_and_no_quote_time(self):
        self.quote["quote"].pop("t")
        packet, request = self.fetch(self.quote, "alpaca", "AAPL", feed="sip")
        self.assertTrue(request.call_args.args[0].endswith("?feed=sip"))
        market = packet["items"][0]["market"]
        self.assertEqual(market["feed"], "sip")
        self.assertIn("subscription required", market["scope"])
        self.assertIsNone(market["source_timestamp"])
        self.assertEqual(market["time_precision"], "unknown")

    def test_missing_credentials_never_issue_http_or_echo_values(self):
        for values in ({"APCA_API_KEY_ID": "", "APCA_API_SECRET_KEY": ""},
                       {"APCA_API_KEY_ID": "sensitive-partial-key", "APCA_API_SECRET_KEY": ""},
                       {"APCA_API_KEY_ID": "", "APCA_API_SECRET_KEY": "sensitive-partial-secret"}):
            with self.subTest(values=list(values)), mock.patch.dict(os.environ, values), \
                    mock.patch("market_sources.request_json") as request:
                with self.assertRaises(SourceError) as raised:
                    market_sources.fetch_market("alpaca", "AAPL")
                self.assertEqual(raised.exception.code, "AUTH_MISSING")
                self.assertNotIn("sensitive-partial", json.dumps(raised.exception.public()))
                request.assert_not_called()

    def test_invalid_configuration_never_issues_http(self):
        cases = [("unknown", "AAPL", 10, "iex"), ("alpaca", "AAPL?token=secret", 10, "iex"),
                 ("coinbase", "BTC/USD", 10, "iex"), ("alpaca", "", 10, "iex"),
                 ("alpaca", "AAPL", True, "iex"), ("alpaca", "AAPL", 1.5, "iex"),
                 ("alpaca", "AAPL", 0, "iex"), ("alpaca", "AAPL", 51, "iex"),
                 ("alpaca", "AAPL", 10, "secret-feed"), ("coinbase", "BTC-USD", 10, "sip")]
        with mock.patch("market_sources.request_json") as request:
            for venue, symbol, depth, feed in cases:
                with self.subTest(venue=venue, depth=depth), self.assertRaises(SourceError) as raised:
                    market_sources.fetch_market(venue, symbol, depth, feed)
                self.assertEqual(raised.exception.code, "INVALID_INPUT")
                self.assertNotIn("secret", str(raised.exception))
            request.assert_not_called()
        self.assertEqual(market_sources.validate_market(" Coinbase ", "btc-usd"),
                         {"venue": "coinbase", "symbol": "BTC-USD", "depth": 10, "feed": "exchange"})

    def test_snapshot_identity_uses_native_state_not_endpoint_or_collection(self):
        first, _ = self.fetch(depth=1)
        self.metadata["collected_at"] = "2026-11-17T12:01:00+00:00"
        second, _ = self.fetch(depth=2)
        first_id = first["items"][0]["id"]
        self.assertEqual(first_id, second["items"][0]["id"])
        for changed in ({**self.book, "sequence": 101},
                        {**self.book, "time": "2026-11-17T12:00:00.123456789Z"},
                        {**self.book, "bids": [["98", "2", 7], ["100", "3", 2]]}):
            packet, _ = self.fetch(changed, depth=1)
            self.assertNotEqual(first_id, packet["items"][0]["id"])
        same_reordered = dict(reversed(list(self.book.items())))
        packet, _ = self.fetch(same_reordered)
        self.assertEqual(first_id, packet["items"][0]["id"])

    def test_quote_identity_changes_with_nanoseconds_and_feed(self):
        first, _ = self.fetch(self.quote, "alpaca", "AAPL")
        changed = copy.deepcopy(self.quote)
        changed["quote"]["t"] = "2026-11-17T06:59:00.123456790-05:00"
        second, _ = self.fetch(changed, "alpaca", "AAPL")
        third, _ = self.fetch(self.quote, "alpaca", "AAPL", feed="sip")
        self.assertNotEqual(first["items"][0]["id"], second["items"][0]["id"])
        self.assertNotEqual(first["items"][0]["id"], third["items"][0]["id"])

    def test_empty_one_sided_zero_and_crossed_snapshots(self):
        for bids, asks, expected in [([], [], None), ([[100, 0]], [], None),
                                     ([[100, 0]], [[101, 0]], False),
                                     ([[102, 2]], [[101, 1]], True)]:
            with self.subTest(bids=bids, asks=asks):
                packet, _ = self.fetch({"bids": bids, "asks": asks})
                derived = packet["items"][0]["market"]["derived"]
                self.assertEqual(derived["crossed"], expected)
                if expected is None:
                    self.assertIsNone(derived["mid"])
                    self.assertIsNone(derived["imbalance"])
                elif expected:
                    self.assertEqual(derived["spread"], -1)
                    self.assertLess(derived["spread_bps"], 0)
                else:
                    self.assertIsNone(derived["imbalance"])

    def test_malformed_levels_are_omitted_with_truthful_partial_coverage(self):
        invalid_rows = [[0, 1], [-1, 1], ["NaN", 1], ["Infinity", 1],
                        [100, -2], [100, True], [True, 2], [101], None, {}, ["bad", 2]]
        packet, _ = self.fetch({"bids": [[100, 2]] + invalid_rows, "asks": [[101, 3]]})
        market = packet["items"][0]["market"]
        self.assertEqual(market["bids"], [[100, 2]])
        self.assertEqual(market["invalid_bid_levels"], len(invalid_rows))
        self.assertEqual(market["provider_bid_levels"], len(invalid_rows) + 1)
        self.assertTrue(packet["coverage"]["partial"])
        self.assertTrue(all(value is None for value in market["derived"].values()))
        self.assertEqual(packet["items"][0]["raw"]["bids"][1:], invalid_rows)

    def test_duplicate_price_is_not_double_counted(self):
        packet, _ = self.fetch({"bids": [[100, 2], [100, 4]], "asks": [[101, 3]]})
        market = packet["items"][0]["market"]
        self.assertEqual(market["invalid_bid_levels"], 1)
        self.assertEqual(market["bids"], [[100, 2]])
        self.assertIsNone(market["derived"]["imbalance"])

    def test_invalid_envelopes_fail_safely(self):
        for payload in ([], {}, {"bids": None, "asks": []}, {"bids": [[0, 1]], "asks": []}):
            with self.subTest(payload=payload), self.assertRaises(SourceError) as raised:
                self.fetch(payload)
            self.assertEqual(raised.exception.code, "INVALID_RESPONSE")
        for payload in ({}, {"quote": None}, {"quote": {}}, {**self.quote, "symbol": "MSFT"}):
            with self.subTest(payload=payload), self.assertRaises(SourceError) as raised:
                self.fetch(payload, "alpaca", "AAPL")
            self.assertEqual(raised.exception.code, "INVALID_RESPONSE")

    def test_large_finite_values_do_not_overflow_derived_measures(self):
        packet, _ = self.fetch({"bids": [[1e308, 1e308], [9e307, 1e308]],
                                "asks": [[1.1e308, 1e308], [1.2e308, 1e308]]})
        derived = packet["items"][0]["market"]["derived"]
        self.assertIsNotNone(derived["mid"])
        self.assertEqual(derived["imbalance"], 0)
        json.dumps(derived, allow_nan=False)

    def test_auction_mode_is_preserved_and_indicative(self):
        packet, _ = self.fetch({**self.book, "auction_mode": True, "auction": {"can_open": "no"}})
        item = packet["items"][0]
        self.assertTrue(item["market"]["auction_mode"])
        self.assertIn("indicative auction", " ".join(item["evidence"]["limitations"]))
        self.assertEqual(item["raw"]["auction"], {"can_open": "no"})

    def test_provider_failures_propagate_without_retry(self):
        failure = SourceError("AUTH_FAILED", "The source rejected authentication.")
        with mock.patch("market_sources.request_json", side_effect=failure) as request:
            with self.assertRaises(SourceError) as raised:
                market_sources.fetch_market("alpaca", "AAPL")
            self.assertIs(raised.exception, failure)
            request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
