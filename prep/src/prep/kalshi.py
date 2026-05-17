"""Minimal anonymous Kalshi API client.

The Kalshi public REST API serves `events`, `markets`, and per-market data
without authentication. See:
  https://trading-api.readme.io/reference/getevents

We deliberately keep this thin: no auth, no caching, no retries beyond
basic error handling. The point is to be the smallest dependency that
lets us snapshot open markets for backtest collection.
"""

from __future__ import annotations

import time
from typing import Any

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


def _get(path: str, params: dict[str, Any] | None = None, timeout: int = 30) -> dict:
    """GET with exponential backoff for 429 / 5xx / network errors.

    Designed to survive transient network blips that happen when a laptop
    suspends, the wifi flaps, or Kalshi rate-limits — important for a
    long-running cron that walks tens of thousands of paginated rows.
    """
    backoff = 1.0
    last_exc: Exception | None = None
    for attempt in range(8):
        try:
            resp = requests.get(BASE_URL + path, params=params or {}, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last_exc = e
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
    if last_exc:
        raise last_exc
    resp.raise_for_status()
    return resp.json()


def _paginate(path: str, params: dict[str, Any], key: str, pause: float = 0.3) -> list[dict]:
    items: list[dict] = []
    cursor: str | None = None
    while True:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        data = _get(path, p)
        items.extend(data.get(key, []))
        cursor = data.get("cursor") or None
        if not cursor:
            break
        time.sleep(pause)
    return items


def list_events(
    *,
    status: str = "open",
    limit: int = 200,
    min_close_ts: int | None = None,
    max_close_ts: int | None = None,
) -> list[dict]:
    """List events. status: 'open' | 'closed' | 'settled'."""
    params: dict[str, Any] = {"status": status, "limit": limit}
    if min_close_ts is not None:
        params["min_close_ts"] = min_close_ts
    if max_close_ts is not None:
        params["max_close_ts"] = max_close_ts
    return _paginate("/events", params, "events")


def list_markets(
    *,
    event_ticker: str | None = None,
    status: str = "open",
    limit: int = 200,
    min_close_ts: int | None = None,
    max_close_ts: int | None = None,
) -> list[dict]:
    """List markets. status: 'open' | 'closed' | 'settled' | 'unopened'."""
    params: dict[str, Any] = {"status": status, "limit": limit}
    if event_ticker:
        params["event_ticker"] = event_ticker
    if min_close_ts is not None:
        params["min_close_ts"] = min_close_ts
    if max_close_ts is not None:
        params["max_close_ts"] = max_close_ts
    return _paginate("/markets", params, "markets")


def _normalize_market(m: dict) -> dict:
    """Normalize a live Kalshi market record to the historical schema our
    backtest data and downstream code expect.

    Kalshi changed the wire format: prices come back as `*_dollars`
    (0.0-1.0 floats) and volume as `*_fp`. Older snapshots stored in
    `data/external/subset_*` use the prior `yes_ask` (cents 0-100) /
    `volume_24h` shape. To avoid two code paths everywhere, we add the
    legacy keys back here when they're missing.
    """
    if "yes_ask" not in m and "yes_ask_dollars" in m:
        v = m.get("yes_ask_dollars")
        if v is not None:
            m["yes_ask"] = round(float(v) * 100)
    if "no_ask" not in m and "no_ask_dollars" in m:
        v = m.get("no_ask_dollars")
        if v is not None:
            m["no_ask"] = round(float(v) * 100)
    if "yes_bid" not in m and "yes_bid_dollars" in m:
        v = m.get("yes_bid_dollars")
        if v is not None:
            m["yes_bid"] = round(float(v) * 100)
    if "no_bid" not in m and "no_bid_dollars" in m:
        v = m.get("no_bid_dollars")
        if v is not None:
            m["no_bid"] = round(float(v) * 100)
    if "last_price" not in m and "last_price_dollars" in m:
        v = m.get("last_price_dollars")
        if v is not None:
            m["last_price"] = round(float(v) * 100)
    if "volume_24h" not in m and "volume_24h_fp" in m:
        v = m.get("volume_24h_fp")
        if v is not None:
            m["volume_24h"] = float(v)
    if "volume" not in m and "volume_fp" in m:
        v = m.get("volume_fp")
        if v is not None:
            m["volume"] = float(v)
    if "liquidity" not in m and "liquidity_dollars" in m:
        v = m.get("liquidity_dollars")
        if v is not None:
            m["liquidity"] = float(v)
    return m


def get_market(ticker: str) -> dict | None:
    try:
        data = _get(f"/markets/{ticker}")
        market = data.get("market") or data
        return _normalize_market(market) if market else None
    except requests.HTTPError:
        return None


def list_markets_normalized(**kwargs) -> list[dict]:
    """Like list_markets but normalizes each record. Use this anywhere
    you would have iterated raw API records directly."""
    return [_normalize_market(m) for m in list_markets(**kwargs)]
