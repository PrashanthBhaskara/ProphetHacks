"""Build canonical market packets from historical rows or live Kalshi data."""

from __future__ import annotations

from datetime import datetime, timezone

from .data import Sample
from .schemas import KalshiQuote, MarketPacket


def _cent_to_prob(value) -> float | None:
    if value is None:
        return None
    return float(value) / 100.0


def packet_from_sample(sample: Sample) -> MarketPacket:
    mi = sample.market_info or {}
    as_of = mi.get("snapshot_time") or datetime.now(timezone.utc).isoformat()
    yes_ask = _cent_to_prob(mi.get("yes_ask"))
    no_ask = _cent_to_prob(mi.get("no_ask"))
    quote = KalshiQuote(
        yes_bid=None if no_ask is None else max(0.0, 1.0 - no_ask),
        yes_ask=yes_ask,
        no_bid=None if yes_ask is None else max(0.0, 1.0 - yes_ask),
        no_ask=no_ask,
        last_price=_cent_to_prob(mi.get("last_price")),
        volume=mi.get("volume"),
        open_interest=mi.get("open_interest"),
        snapshot_time=as_of,
    )
    event = sample.event
    return MarketPacket(
        as_of=as_of,
        event_ticker=event.get("event_ticker") or "",
        market_ticker=event.get("market_ticker") or "",
        title=event.get("title") or "",
        subtitle=event.get("subtitle"),
        rules=event.get("rules"),
        category=event.get("category") or "Other",
        close_time=event.get("close_time"),
        kalshi=quote,
        retrieval={},
    )
