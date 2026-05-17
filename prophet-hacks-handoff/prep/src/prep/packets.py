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
    # Propagate outcomes if the event carries them (Prophet Arena dataset tasks),
    # otherwise default to binary YES/NO (Kalshi historical snapshots).
    outcomes = event.get("outcomes") or ["YES", "NO"]
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
        outcomes=list(outcomes),
        retrieval={},
    )


def packet_from_arena_event(arena_event: dict) -> MarketPacket:
    """Build a MarketPacket from a Prophet Arena `Event` JSON (POST /predict body).

    Arena events don't include Kalshi quote data — set an empty KalshiQuote.
    Trading-track code that reads .kalshi.* should null-check; forecasting
    only needs the event fields + outcomes.
    """
    retrieval = dict(arena_event.get("retrieval") or {})
    for key in ("description", "sources", "market_data", "market_implied_probabilities"):
        if arena_event.get(key) is not None and key not in retrieval:
            retrieval[key] = arena_event.get(key)
    return MarketPacket(
        as_of=arena_event.get("snapshot_time") or arena_event.get("predict_by") or datetime.now(timezone.utc).isoformat(),
        event_ticker=arena_event.get("event_ticker") or arena_event.get("task_id") or "",
        market_ticker=arena_event.get("market_ticker") or arena_event.get("task_id") or "",
        title=arena_event.get("title") or "",
        subtitle=arena_event.get("subtitle"),
        rules=arena_event.get("rules") or arena_event.get("context"),
        category=arena_event.get("category") or (arena_event.get("metadata") or {}).get("category") or "Other",
        close_time=arena_event.get("close_time") or arena_event.get("predict_by"),
        kalshi=KalshiQuote(),
        outcomes=list(arena_event.get("outcomes") or ["YES", "NO"]),
        retrieval=retrieval,
    )
