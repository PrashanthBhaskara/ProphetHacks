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


def get_market(ticker: str) -> dict | None:
    try:
        data = _get(f"/markets/{ticker}")
        return data.get("market") or data
    except requests.HTTPError:
        return None
