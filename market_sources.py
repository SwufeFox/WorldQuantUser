"""Read-only, uncached market observations with explicit venue and feed scope.

Coinbase's level=2 endpoint returns the full aggregated venue book; ``depth``
only caps retained normalized levels. Alpaca supplies one latest quote per side,
never an order book. Source time and collection time remain separate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from urllib.parse import quote, urlencode

from source_io import SourceError, parse_timestamp, public_url, request_json


_COINBASE_DOC = "https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductbook"
_ALPACA_DOC = "https://docs.alpaca.markets/reference/stocklatestquotes-1"
_ALPACA_QUOTE_DOC = "https://docs.alpaca.markets/docs/real-time-stock-pricing-data"
_ALPACA_SINGLE_DOC = "https://github.com/alpacahq/alpaca-trade-api-python/blob/master/alpaca_trade_api/rest.py"
_FEED_SCOPES = {
    "iex": "IEX single-exchange latest best bid/ask; not national NBBO",
    "sip": "SIP consolidated US-exchange latest best bid/ask; subscription required",
    "delayed_sip": "SIP consolidated US-exchange best bid/ask delayed by 15 minutes",
    "boats": "Blue Ocean ATS overnight latest best bid/ask",
    "overnight": "Alpaca derived overnight latest best bid/ask",
    "otc": "Over-the-counter exchanges latest best bid/ask",
}


def validate_market(venue, symbol, depth=10, feed="iex"):
    """Validate before I/O and return normalized fetch_market keyword arguments."""
    if not isinstance(venue, str) or venue.strip().lower() not in {"coinbase", "alpaca"}:
        raise SourceError("INVALID_INPUT", "Market venue must be coinbase or alpaca.")
    venue = venue.strip().lower()
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 50:
        raise SourceError("INVALID_INPUT", "Market depth must be an integer from 1 to 50.")
    if not isinstance(symbol, str):
        raise SourceError("INVALID_INPUT", "Market symbol must be a string.")
    symbol = symbol.strip().upper()
    pattern = r"[A-Z0-9]{1,20}-[A-Z0-9]{1,20}" if venue == "coinbase" else r"[A-Z][A-Z0-9]{0,14}(?:[.-][A-Z0-9]{1,5})?"
    if not re.fullmatch(pattern, symbol):
        raise SourceError("INVALID_INPUT", "Malformed market symbol for the selected venue.")
    if not isinstance(feed, str):
        raise SourceError("INVALID_INPUT", "Market feed must be a string.")
    feed = feed.strip().lower()
    if venue == "coinbase":
        if feed not in {"iex", "exchange"}:
            raise SourceError("INVALID_INPUT", "Coinbase uses only its exchange book feed.")
        feed = "exchange"
    elif feed not in _FEED_SCOPES:
        raise SourceError("INVALID_INPUT", "Unsupported Alpaca stock quote feed.")
    return {"venue": venue, "symbol": symbol, "depth": depth, "feed": feed}


def _number(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        return None
    return number


def _levels(rows, depth, reverse):
    if not isinstance(rows, list):
        raise SourceError("INVALID_RESPONSE", "Market response has no valid bid/ask arrays.")
    levels, seen, invalid = [], set(), 0
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            invalid += 1
            continue
        price, size = _number(row[0], positive=True), _number(row[1])
        if price is None or size is None or price in seen:
            invalid += 1
            continue
        seen.add(price)
        levels.append([price, size])
    levels.sort(key=lambda level: level[0], reverse=reverse)
    return levels[:depth], {"provider": len(rows), "valid": len(levels), "invalid": invalid,
                            "omitted": max(0, len(levels) - depth)}


def _finite(value):
    return value if math.isfinite(value) else None


def _derived(bids, asks, invalid):
    result = dict.fromkeys(("best_bid", "best_ask", "mid", "spread", "spread_bps", "imbalance", "crossed"))
    # Invalid/duplicate rows may hide a better price or distort size totals.
    if invalid:
        return result
    best_bid, best_ask = (bids[0][0] if bids else None), (asks[0][0] if asks else None)
    result.update(best_bid=best_bid, best_ask=best_ask)
    if best_bid is None or best_ask is None:
        return result
    mid = best_bid / 2 + best_ask / 2
    spread = best_ask - best_bid
    result.update(mid=_finite(mid), spread=_finite(spread), crossed=best_bid > best_ask)
    if mid > 0:
        result["spread_bps"] = _finite(spread / mid * 10000)
    # Scaling avoids overflow while retaining the ratio of displayed sizes.
    scale = max(level[1] for level in bids + asks)
    if scale:
        bid_size = sum(level[1] / scale for level in bids)
        ask_size = sum(level[1] / scale for level in asks)
        result["imbalance"] = (bid_size - ask_size) / (bid_size + ask_size)
    return result


def fetch_market(venue, symbol, depth=10, feed="iex", timeout=30):
    """Fetch exactly one public book or credentialed quote, with no cache reuse."""
    config = validate_market(venue, symbol, depth, feed)
    venue, symbol, depth, feed = (config[key] for key in ("venue", "symbol", "depth", "feed"))
    headers = None
    if venue == "coinbase":
        endpoint = "https://api.exchange.coinbase.com/products/%s/book?level=2" % quote(symbol, safe="")
    else:
        from machine_lib import _load_dotenv

        _load_dotenv()
        key = os.environ.get("APCA_API_KEY_ID", "").strip()
        secret = os.environ.get("APCA_API_SECRET_KEY", "").strip()
        if not key or not secret:
            raise SourceError("AUTH_MISSING", "Alpaca stock quotes require credentials.",
                              "Configure APCA_API_KEY_ID and APCA_API_SECRET_KEY; feed access depends on subscription.")
        headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        endpoint = "https://data.alpaca.markets/v2/stocks/%s/quotes/latest?%s" % (
            quote(symbol, safe=""), urlencode({"feed": feed}))
    payload, metadata = request_json(endpoint, timeout=timeout, headers=headers)
    if not isinstance(payload, dict):
        raise SourceError("INVALID_RESPONSE", "Market response must be an object.")
    limitations = ["One snapshot, not historical availability or backtest data.",
                   "Collection time does not establish when source data first became available."]
    if venue == "coinbase":
        bid_rows, ask_rows = payload.get("bids"), payload.get("asks")
        timestamp = parse_timestamp(payload.get("time"))
        scope = "Coinbase Exchange aggregated L2 book for this crypto product only"
        # The book reference defines aggregation but not a size-to-currency conversion.
        size_unit = "provider_native_size"
        limitations.append("Size is the provider's aggregated size at each price; no unit conversion is assumed.")
        auction_mode = payload.get("auction_mode") if isinstance(payload.get("auction_mode"), bool) else None
        if auction_mode is True:
            limitations.append("Auction mode: prices are indicative auction quotes, not firm continuous-market quotes.")
        elif auction_mode is None:
            limitations.append("Auction mode was not supplied by the provider.")
        extras = {"sequence": payload.get("sequence"), "auction_mode": auction_mode,
                  "bid_exchange": None, "ask_exchange": None, "quote_conditions": None, "tape": None}
        docs = [_COINBASE_DOC]
    else:
        native_quote = payload.get("quote")
        if not isinstance(native_quote, dict) or not native_quote:
            raise SourceError("INVALID_RESPONSE", "Alpaca response has no latest quote object.")
        if payload.get("symbol", symbol) != symbol:
            raise SourceError("INVALID_RESPONSE", "Alpaca response symbol does not match the request.")
        bid_rows = [[native_quote.get("bp"), native_quote.get("bs")]]
        ask_rows = [[native_quote.get("ap"), native_quote.get("as")]]
        timestamp = parse_timestamp(native_quote.get("t"))
        scope = _FEED_SCOPES[feed]
        size_unit = "round_lots"
        limitations.extend(["One latest feed quote per side; requested depth cannot create L2 levels.",
                            "Quote sizes retain documented round lots; no shares-per-lot conversion is assumed.",
                            "Latest quote may precede collection by a substantial interval; freshness is not guaranteed."])
        extras = {"sequence": None, "auction_mode": None,
                  "bid_exchange": native_quote.get("bx"), "ask_exchange": native_quote.get("ax"),
                  "quote_conditions": native_quote.get("c"), "tape": native_quote.get("z")}
        docs = [_ALPACA_DOC, _ALPACA_QUOTE_DOC, _ALPACA_SINGLE_DOC]
    bids, bid_counts = _levels(bid_rows, depth, True)
    asks, ask_counts = _levels(ask_rows, depth, False)
    invalid = bid_counts["invalid"] + ask_counts["invalid"]
    if invalid:
        limitations.append("Invalid or duplicate price levels were omitted; derived measures are unknown.")
    if not bids and not asks and invalid:
        raise SourceError("INVALID_RESPONSE", "Market response contains no valid price levels.")
    if timestamp["value"] is None:
        limitations.append("Provider observation timestamp is unknown.")
    market = {
        "venue": venue, "symbol": symbol, "asset_class": "crypto" if venue == "coinbase" else "equity",
        "feed": feed, "scope": scope, "depth_kind": "aggregated_l2" if venue == "coinbase" else "feed_l1",
        "provider_level": 2 if venue == "coinbase" else 1, "requested_depth": depth,
        "bid_levels": len(bids), "ask_levels": len(asks), "bids": bids, "asks": asks,
        "source_timestamp": timestamp["value"], "source_timestamp_raw": timestamp["raw"],
        "time_precision": timestamp["precision"], "size_unit": size_unit,
        "price_unit": symbol.split("-")[1] if venue == "coinbase" else "USD",
        "derived": _derived(bids, asks, invalid),
        "derived_scope": "Retained levels only; imbalance=(sum bid sizes-sum ask sizes)/(sum bid sizes+sum ask sizes).",
        "limitations": limitations, **extras,
    }
    for side, counts in (("bid", bid_counts), ("ask", ask_counts)):
        for count_kind, count in counts.items():
            market["%s_%s_levels" % (count_kind, side)] = count
    # Full native state (including sequence, timestamp, and uncapped levels) defines
    # identity. Collection time and local depth do not create a new observation.
    native_identity = json.dumps({"venue": venue, "symbol": symbol, "feed": feed, "snapshot": payload},
                                 sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    identity = hashlib.sha256(native_identity.encode("utf-8")).hexdigest()
    collected_at = metadata["collected_at"]
    coverage = {"scope": scope, "ordering": "bids descending; asks ascending",
                "partial": bool(invalid), "requested_depth": depth,
                "provider_bid_levels": bid_counts["provider"], "provider_ask_levels": ask_counts["provider"],
                "returned_bid_levels": len(bids), "returned_ask_levels": len(asks),
                "invalid_levels": invalid, "limitations": limitations, "documentation": docs,
                "cache_policy": "No cache reuse; one provider request per observation."}
    item = {
        "kind": "market", "id": "%s:%s:%s:%s" % (venue, symbol, feed, identity),
        "title": "%s %s %s snapshot" % (venue, symbol, market["depth_kind"]),
        "url": public_url(endpoint), "excerpt": scope,
        "published_at": timestamp["value"], "published_at_raw": timestamp["raw"],
        "time_precision": timestamp["precision"], "collected_at": collected_at,
        "author": None, "provider": venue, "engagement": None,
        "evidence": {"basis": "provider_market_snapshot", "claim_state": "observed_snapshot",
                     "scope": scope, "limitations": limitations, "documentation": docs},
        "market": market, "raw": payload,
    }
    return {"items": [item], "source": venue, "query": None, "cached": False,
            "collected_at": collected_at, "coverage": coverage}
