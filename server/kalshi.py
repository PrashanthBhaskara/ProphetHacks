"""Anonymous Kalshi REST client.

Vendored from prep/src/prep/kalshi.py so the server is self-contained
and Render can build it without reaching into the prep/ tree.

The public Kalshi API serves /events and /markets without auth.
See https://trading-api.readme.io/reference/getevents.
"""

from __future__ import annotations

import time
from typing import Any

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


def _get(path: str, params: dict[str, Any] | None = None, timeout: int = 30) -> dict:
    backoff = 1.0
    last_exc: Exception | None = None
    for _ in range(8):
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


def get_market(ticker: str) -> dict | None:
    try:
        data = _get(f"/markets/{ticker}")
        return data.get("market") or data
    except requests.HTTPError:
        return None
