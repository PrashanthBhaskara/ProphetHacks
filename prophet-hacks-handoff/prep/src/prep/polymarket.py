"""Polymarket cross-venue price prior for Kalshi markets.

Mapping cache is built offline by scripts/poly_match.py (kalshi_ticker ->
polymarket conditionId + outcome label). At runtime get_market_priors()
reads the cache and fetches current Polymarket prices via Gamma.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import requests

from .schemas import KalshiQuote

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "kalshi_polymarket"
MAP_CSV = DATA_DIR / "map.csv"
NEG_CSV = DATA_DIR / "rejected.csv"
GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"


class MarketPrior(NamedTuple):
    quote: KalshiQuote
    exchange: str   # "polymarket"
    title: str      # Polymarket market title


def _append_row(path: Path, header: list[str], row: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)


def _load_map() -> dict[str, tuple[str, str]]:
    """Returns {kalshi_ticker: (poly_condition_id, poly_outcome)}."""
    if not MAP_CSV.exists():
        return {}
    with MAP_CSV.open() as f:
        return {
            row["kalshi_ticker"]: (row["poly_condition_id"], row["poly_outcome"])
            for row in csv.DictReader(f)
        }


def _load_negative() -> set[str]:
    if not NEG_CSV.exists():
        return set()
    with NEG_CSV.open() as f:
        return {row["kalshi_ticker"] for row in csv.DictReader(f)}


def _parse_outcomes(market: dict) -> list[str]:
    raw = market.get("outcomes")
    if isinstance(raw, str):
        try:
            return [str(o) for o in json.loads(raw)]
        except json.JSONDecodeError:
            return []
    return [str(o) for o in raw] if isinstance(raw, list) else []


def get_market_priors(market: dict) -> list[MarketPrior]:
    """Return at most one MarketPrior for a Kalshi market, or [].

    Looks up the polymarket conditionId from MAP_CSV, fetches the current
    Polymarket market via Gamma, and converts the quoted side to a KalshiQuote.
    If our stored outcome is the complement (e.g., we matched "Republicans win X"
    to Polymarket's "Democrats win X" with outcome="No"), flip the prices.
    """
    ticker = market.get("ticker") or market.get("market_ticker")
    if not ticker:
        return []
    cache = _load_map()
    if ticker not in cache:
        return []
    poly_cid, outcome = cache[ticker]

    try:
        rs = requests.get(GAMMA_MARKETS, params={"condition_ids": poly_cid}, timeout=15)
        rs.raise_for_status()
        markets = rs.json()
    except requests.RequestException:
        return []
    if not isinstance(markets, list) or not markets:
        return []

    m = markets[0]
    if not m.get("active") or m.get("closed") or m.get("archived"):
        return []

    outcomes = _parse_outcomes(m)
    if outcome not in outcomes:
        return []

    # Gamma's bestBid/bestAsk/lastTradePrice quote the first outcome.
    # If our recorded outcome is the second one, flip via 1 - x.
    try:
        bid = float(m.get("bestBid") or 0)
        ask = float(m.get("bestAsk") or 0)
        last = float(m.get("lastTradePrice") or 0)
    except (TypeError, ValueError):
        return []

    if outcomes.index(outcome) == 1:
        bid, ask = 1.0 - ask, 1.0 - bid
        last = 1.0 - last if last else 0.0

    if ask <= 0.01 or ask >= 0.99:
        return []

    quote = KalshiQuote(
        yes_bid=bid,
        yes_ask=ask,
        no_bid=1.0 - ask,
        no_ask=1.0 - bid,
        last_price=last or None,
        snapshot_time=datetime.now(timezone.utc).isoformat(),
    )
    return [MarketPrior(quote=quote, exchange="polymarket", title=m.get("question", ""))]
