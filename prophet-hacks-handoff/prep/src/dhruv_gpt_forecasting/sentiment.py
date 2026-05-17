"""Lightweight timestamped sentiment features for external evidence archives."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]*")

POSITIVE_TERMS = {
    "advance",
    "advances",
    "beat",
    "beats",
    "boost",
    "bullish",
    "clinch",
    "clinches",
    "confirm",
    "confirmed",
    "edge",
    "gain",
    "gains",
    "good",
    "healthy",
    "improve",
    "improved",
    "leads",
    "positive",
    "rally",
    "recover",
    "recovered",
    "rise",
    "rises",
    "strong",
    "surge",
    "up",
    "win",
    "wins",
}

NEGATIVE_TERMS = {
    "bearish",
    "decline",
    "declines",
    "defeat",
    "defeated",
    "delay",
    "delayed",
    "doubt",
    "down",
    "drop",
    "drops",
    "fall",
    "falls",
    "injury",
    "injured",
    "loss",
    "miss",
    "misses",
    "negative",
    "risk",
    "risks",
    "slump",
    "uncertain",
    "weak",
    "worse",
}


def score_text_sentiment(text: Any) -> dict[str, Any]:
    """Return a small lexicon sentiment score in [-1, 1].

    This is intentionally cheap and deterministic. It is not a trading signal
    by itself; it gives GPT a compact timestamped prior over the tone of the
    archived evidence without spending tokens on every raw article.
    """
    tokens = TOKEN_RE.findall(str(text or "").lower())
    if not tokens:
        return {"score": 0.0, "label": "neutral", "positive_terms": 0, "negative_terms": 0}
    counts = Counter(tokens)
    pos = sum(counts[token] for token in POSITIVE_TERMS)
    neg = sum(counts[token] for token in NEGATIVE_TERMS)
    denom = pos + neg
    score = 0.0 if denom == 0 else (pos - neg) / denom
    if score >= 0.20:
        label = "positive"
    elif score <= -0.20:
        label = "negative"
    else:
        label = "neutral"
    return {
        "score": round(score, 4),
        "label": label,
        "positive_terms": int(pos),
        "negative_terms": int(neg),
    }


def record_text(record: dict[str, Any]) -> str:
    return " ".join(
        str(record.get(key) or "")
        for key in ("title", "text", "body", "summary", "claim")
    )


def annotate_record_sentiment(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    sentiment = score_text_sentiment(record_text(record))
    enriched.setdefault("sentiment_score", sentiment["score"])
    enriched.setdefault("sentiment_label", sentiment["label"])
    enriched.setdefault("sentiment_positive_terms", sentiment["positive_terms"])
    enriched.setdefault("sentiment_negative_terms", sentiment["negative_terms"])
    enriched.setdefault("sentiment_model", "lexicon_v1")
    return enriched


def aggregate_sentiment(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for record in records:
        if record.get("sentiment_score") is None:
            record = annotate_record_sentiment(record)
        try:
            scored.append(float(record.get("sentiment_score") or 0.0))
        except (TypeError, ValueError):
            scored.append(0.0)
    if not scored:
        return {"mean_score": 0.0, "label_counts": {}, "n_scored": 0}
    labels = Counter(str(record.get("sentiment_label") or "neutral") for record in records)
    return {
        "mean_score": round(sum(scored) / len(scored), 4),
        "label_counts": dict(labels),
        "n_scored": len(scored),
    }
