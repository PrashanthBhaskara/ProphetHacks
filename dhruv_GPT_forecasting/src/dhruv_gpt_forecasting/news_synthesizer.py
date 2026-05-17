"""Compact local news digests for PIT evidence archives."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .features import parse_dt
from .sentiment import aggregate_sentiment, annotate_record_sentiment


def synthesize_news_digest(
    records: list[dict[str, Any]],
    packet: Any,
    query: str,
    *,
    max_records: int = 8,
    max_chars: int = 1800,
) -> dict[str, Any] | None:
    """Create an extractive digest so GPT does not read every article row.

    This deliberately avoids a paid LLM call. If we later add a Hugging Face
    model, it should consume this short candidate list rather than raw records.
    """
    ranked = _rank(records, query, getattr(packet, "outcomes", []) or [])
    if not ranked:
        return None
    selected = [annotate_record_sentiment(row) for row in ranked[:max_records]]
    source_counts = Counter(str(row.get("source") or "unknown") for row in selected)
    sentiment = aggregate_sentiment(selected)
    bullets: list[str] = []
    latest_published: datetime | None = None
    for row in selected:
        published = parse_dt(row.get("published_at") or row.get("created_at") or row.get("timestamp"))
        if published and (latest_published is None or published > latest_published):
            latest_published = published
        title = _clean(row.get("title") or row.get("text") or row.get("summary") or "")
        if not title:
            continue
        source = row.get("domain") or row.get("subreddit") or row.get("source") or "source"
        stamp = published.isoformat() if published else "unknown_time"
        bullets.append(f"{stamp} [{source}] {title[:220]}")
    summary = "\n".join(f"- {item}" for item in bullets)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."
    as_of = getattr(packet, "as_of", None)
    latest = latest_published.isoformat() if latest_published else as_of
    return {
        "source": "pit_news_digest",
        "published_at": latest,
        "collected_at": _now(),
        "title": f"PIT news digest for {getattr(packet, 'market_ticker', '')}",
        "text": summary,
        "summary": summary,
        "url": None,
        "target_event_ticker": getattr(packet, "event_ticker", None),
        "target_market_ticker": getattr(packet, "market_ticker", None),
        "event_ticker": getattr(packet, "event_ticker", None),
        "market_ticker": getattr(packet, "market_ticker", None),
        "forecast_as_of": as_of,
        "forecast_close_time": getattr(packet, "close_time", None),
        "forecast_title": getattr(packet, "title", None),
        "forecast_query": query,
        "pit_mode": "extractive_news_digest_v1",
        "strict_pit_eligible": False,
        "published_at_pit_eligible": True,
        "synthesizer": "extractive_local_v1",
        "n_source_records": len(selected),
        "source_counts": dict(source_counts),
        "sentiment": sentiment,
        "source_urls": [row.get("url") for row in selected if row.get("url")][:max_records],
    }


def _rank(records: list[dict[str, Any]], query: str, outcomes: list[str]) -> list[dict[str, Any]]:
    q_tokens = set(_tokens(query))
    outcome_tokens = set()
    for outcome in outcomes:
        outcome_tokens.update(_tokens(str(outcome)))

    def score(row: dict[str, Any]) -> tuple[float, str]:
        text = " ".join(str(row.get(key) or "") for key in ("title", "text", "summary"))
        tokens = set(_tokens(text))
        overlap = len(tokens & q_tokens)
        outcome_overlap = len(tokens & outcome_tokens)
        engagement = 0.0
        for key in ("score", "num_comments", "like_count", "retweet_count", "reply_count"):
            try:
                engagement += max(0.0, float(row.get(key) or 0.0)) ** 0.25
            except (TypeError, ValueError):
                continue
        published = str(row.get("published_at") or row.get("created_at") or "")
        return (overlap + 2.0 * outcome_overlap + 0.05 * engagement, published)

    ranked = [row for row in records if _clean(row.get("title") or row.get("text") or "")]
    ranked.sort(key=score, reverse=True)
    return ranked


def _tokens(value: str) -> list[str]:
    import re

    stop = {"the", "and", "for", "with", "from", "this", "that", "will", "yes", "no"}
    return [tok for tok in re.findall(r"[a-z0-9]+", value.lower()) if len(tok) > 2 and tok not in stop]


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
