"""Public Kalshi market retrieval for forecast-event creation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

from .config import load_local_env
from .features import normalize_category
from .kalshi_auth import kalshi_auth_headers, kalshi_credential_status


DEFAULT_BASE_URL = "https://api.elections.kalshi.com"


def list_markets(
    *,
    status: str = "open",
    limit: int = 200,
    max_items: int | None = None,
    max_close_ts: int | None = None,
    min_close_ts: int | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"status": status, "limit": limit}
    if max_close_ts is not None:
        params["max_close_ts"] = max_close_ts
    if min_close_ts is not None:
        params["min_close_ts"] = min_close_ts
    markets = _paginate("/trade-api/v2/markets", params, "markets", max_items=max_items)
    if category:
        markets = [
            market for market in markets
            if normalize_category(market.get("category"), market.get("event_ticker")) == category
        ]
    return markets


def market_to_arena_event(market: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_ticker": market.get("event_ticker") or "",
        "market_ticker": market.get("ticker") or "",
        "title": market.get("title") or market.get("subtitle") or "",
        "subtitle": market.get("subtitle") or market.get("yes_sub_title"),
        "description": market.get("rules_primary"),
        "category": normalize_category(market.get("category"), market.get("event_ticker")),
        "rules": market.get("rules_primary") or market.get("rules_secondary"),
        "close_time": market.get("close_time"),
        "outcomes": ["YES", "NO"],
    }


def retrieve_events(
    *,
    deadline: str | None = None,
    status: str = "open",
    max_items: int = 100,
    category: str | None = None,
) -> list[dict[str, Any]]:
    max_close_ts = None
    if deadline:
        from .features import parse_dt

        parsed = parse_dt(deadline)
        if parsed is None:
            raise ValueError(f"invalid deadline: {deadline}")
        max_close_ts = int(parsed.timestamp())
    markets = list_markets(
        status=status,
        max_items=max_items,
        max_close_ts=max_close_ts,
        category=category,
    )
    return [market_to_arena_event(market) for market in markets]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline")
    parser.add_argument("--status", default="open")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--category")
    parser.add_argument("-o", "--output", type=Path, default=Path("events.json"))
    args = parser.parse_args()
    events = retrieve_events(
        deadline=args.deadline,
        status=args.status,
        max_items=args.max_items,
        category=args.category,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(events, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "n_events": len(events),
        "kalshi_credentials": kalshi_credential_status(),
    }, indent=2, sort_keys=True))
    return 0


def _paginate(path: str, params: dict[str, Any], key: str, *, max_items: int | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor = None
    while True:
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor
        data = _get(path, page_params)
        out.extend(data.get(key) or [])
        if max_items is not None and len(out) >= max_items:
            return out[:max_items]
        cursor = data.get("cursor")
        if not cursor:
            return out
        time.sleep(0.2)


def _get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    load_local_env()
    base_url = _base_url()
    headers = kalshi_auth_headers("GET", path)
    response = requests.get(base_url + path, params=params, headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()


def _base_url() -> str:
    load_local_env()
    import os

    return (os.environ.get("KALSHI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


if __name__ == "__main__":
    raise SystemExit(main())
