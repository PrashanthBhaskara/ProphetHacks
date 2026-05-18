"""Minimal Kalshi API client.

The Kalshi public REST API serves `events`, `markets`, and per-market data
without authentication. See:
  https://trading-api.readme.io/reference/getevents

We deliberately keep this thin: no caching, no generated SDK, and no trading
methods. The point is to be the smallest dependency that lets us snapshot
markets and backfill public market-data history for research.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_ATTEMPTS = 2


def _get(
    path: str,
    params: dict[str, Any] | None = None,
    timeout: float | None = None,
    max_attempts: int | None = None,
) -> dict:
    """GET with exponential backoff for 429 / 5xx / network errors.

    Designed to survive transient network blips that happen when a laptop
    suspends, the wifi flaps, or Kalshi rate-limits — important for a
    long-running cron that walks tens of thousands of paginated rows.
    """
    timeout = timeout if timeout is not None else float(os.environ.get("KALSHI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    max_attempts = max_attempts if max_attempts is not None else int(os.environ.get("KALSHI_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS))
    max_attempts = max(1, max_attempts)
    backoff = 1.0
    last_exc: Exception | None = None
    resp = None
    for attempt in range(max_attempts):
        try:
            resp = requests.get(BASE_URL + path, params=params or {}, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < max_attempts - 1:
                    time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last_exc = e
            if attempt < max_attempts - 1:
                time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
    if last_exc:
        raise last_exc
    if resp is None:
        raise RuntimeError("Kalshi request did not produce a response")
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
