"""Build canonical market packets from live Prophet Arena events."""

from __future__ import annotations

from datetime import datetime, timezone

from .schemas import KalshiQuote, MarketPacket


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
