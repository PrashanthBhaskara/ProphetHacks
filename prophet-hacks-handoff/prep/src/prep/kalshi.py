"""Minimal Kalshi API client.

The Kalshi public REST API serves `events`, `markets`, and per-market data
without authentication. See:
  https://trading-api.readme.io/reference/getevents

We deliberately keep this thin: no caching, no generated SDK, and no trading
methods. The point is to be the smallest dependency that lets us snapshot
markets and backfill public market-data history for research.
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


def _paginate(
    path: str,
    params: dict[str, Any],
    key: str,
    pause: float = 0.3,
    max_items: int | None = None,
) -> list[dict]:
    items: list[dict] = []
    cursor: str | None = None
    while True:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        data = _get(path, p)
        items.extend(data.get(key, []))
        if max_items is not None and len(items) >= max_items:
            return items[:max_items]
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
    status: str | None = "open",
    limit: int = 200,
    min_close_ts: int | None = None,
    max_close_ts: int | None = None,
    min_created_ts: int | None = None,
    max_created_ts: int | None = None,
    min_settled_ts: int | None = None,
    max_settled_ts: int | None = None,
    series_ticker: str | None = None,
    mve_filter: str | None = None,
    max_items: int | None = None,
) -> list[dict]:
    """List markets. status: 'open' | 'closed' | 'settled' | 'unopened'."""
    params: dict[str, Any] = {"limit": limit}
    if status:
        params["status"] = status
    if event_ticker:
        params["event_ticker"] = event_ticker
    if series_ticker:
        params["series_ticker"] = series_ticker
    if min_close_ts is not None:
        params["min_close_ts"] = min_close_ts
    if max_close_ts is not None:
        params["max_close_ts"] = max_close_ts
    if min_created_ts is not None:
        params["min_created_ts"] = min_created_ts
    if max_created_ts is not None:
        params["max_created_ts"] = max_created_ts
    if min_settled_ts is not None:
        params["min_settled_ts"] = min_settled_ts
    if max_settled_ts is not None:
        params["max_settled_ts"] = max_settled_ts
    if mve_filter:
        params["mve_filter"] = mve_filter
    return _paginate("/markets", params, "markets", max_items=max_items)


def get_market(ticker: str) -> dict | None:
    try:
        data = _get(f"/markets/{ticker}")
        return data.get("market") or data
    except requests.HTTPError:
        return None


def historical_cutoff() -> dict:
    """Return Kalshi's live-vs-historical cutoff timestamps."""
    return _get("/historical/cutoff")


def list_historical_markets(
    *,
    event_ticker: str | None = None,
    series_ticker: str | None = None,
    tickers: list[str] | None = None,
    exclude_mve: bool = True,
    limit: int = 1000,
    pause: float = 0.3,
    max_items: int | None = None,
) -> list[dict]:
    """List markets archived to Kalshi's historical tier."""
    params: dict[str, Any] = {"limit": limit}
    if event_ticker:
        params["event_ticker"] = event_ticker
    if series_ticker:
        params["series_ticker"] = series_ticker
    if tickers:
        params["tickers"] = ",".join(tickers)
    if exclude_mve:
        params["mve_filter"] = "exclude"
    return _paginate("/historical/markets", params, "markets", pause=pause, max_items=max_items)


def list_trades(
    *,
    ticker: str | None = None,
    min_ts: int | None = None,
    max_ts: int | None = None,
    historical: bool = False,
    limit: int = 1000,
    pause: float = 0.2,
) -> list[dict]:
    """List Kalshi trade tape rows from the live or historical tier.

    Kalshi partitions trades by fill time. Use `historical=True` for rows
    before `historical_cutoff()["trades_created_ts"]`; use the live tier for
    recent rows, then concatenate/dedupe by `trade_id`.
    """
    params: dict[str, Any] = {"limit": limit}
    if ticker:
        params["ticker"] = ticker
    if min_ts is not None:
        params["min_ts"] = min_ts
    if max_ts is not None:
        params["max_ts"] = max_ts
    path = "/historical/trades" if historical else "/markets/trades"
    return _paginate(path, params, "trades", pause=pause)


def get_market_candlesticks(
    *,
    ticker: str,
    series_ticker: str | None,
    start_ts: int,
    end_ts: int,
    period_interval: int = 1,
    historical: bool = False,
) -> dict:
    """Return Kalshi OHLCV/top-of-book candles for one market.

    `period_interval` is minutes and must be 1, 60, or 1440. Historical
    archived markets use a different path and do not require `series_ticker`.
    """
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_interval,
    }
    if historical:
        return _get(f"/historical/markets/{ticker}/candlesticks", params)
    if not series_ticker:
        raise ValueError("series_ticker is required for live market candlesticks")
    return _get(f"/series/{series_ticker}/markets/{ticker}/candlesticks", params)


def get_orderbook(ticker: str, *, depth: int = 0) -> dict:
    """Return the current full-depth order book for a market.

    Kalshi returns YES bids and NO bids. A YES ask at price X is implied by
    the best NO bid at `1 - X`.
    """
    return _get(f"/markets/{ticker}/orderbook", {"depth": depth})
