"""Deterministic Prophet Arena priors for live forecasts."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from .arena_types import ArenaForecastPacket, ArenaPrior
from .constraints import enforce_constraints, normalize_distribution
from .features import classify_event_structure, normalize_category, parse_dt
from .kalshi_contracts import parse_kalshi_multileg_contract


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class HistoricalRecord:
    market_ticker: str
    event_ticker: str
    title: str
    rules: str | None
    subtitle: str | None
    category: str
    label: str
    outcome: int
    tokens: frozenset[str]


def build_arena_packet(event: dict[str, Any], *, include_historical_analogs: bool = False) -> ArenaForecastPacket:
    outcomes = _clean_outcomes(event.get("outcomes"))
    title = str(event.get("title") or "")
    description = event.get("description") or event.get("context")
    category = normalize_category(event.get("category"), event.get("event_ticker"))
    as_of = (
        event.get("as_of")
        or event.get("snapshot_time")
        or event.get("created_at")
        or datetime.now(timezone.utc).isoformat()
    )
    close_time = event.get("close_time") or event.get("predict_by")
    as_of_dt = parse_dt(as_of)
    close_dt = parse_dt(close_time)
    horizon_hours = None
    if as_of_dt and close_dt:
        horizon_hours = max(0.0, (close_dt - as_of_dt).total_seconds() / 3600.0)
    event_structure = classify_event_structure(outcomes, title, event.get("rules"))
    packet = ArenaForecastPacket(
        as_of=as_of,
        event_ticker=str(event.get("event_ticker") or event.get("task_id") or ""),
        market_ticker=str(event.get("market_ticker") or event.get("task_id") or event.get("event_ticker") or ""),
        title=title,
        subtitle=event.get("subtitle"),
        description=description,
        context=event.get("context"),
        category=category,
        rules=event.get("rules"),
        close_time=close_time,
        outcomes=outcomes,
        event_structure=event_structure,
        horizon_hours=horizon_hours,
        extracted_entities=_extract_entities(event, outcomes),
        features=dict(event.get("features") or {}),
    )
    inline_market_probability = _inline_market_probability(event)
    if inline_market_probability is not None:
        packet.features["inline_market_quote"] = {
            "yes_probability": inline_market_probability,
            "source": "event_payload_quote",
        }
    if include_historical_analogs:
        packet.historical_analogs = historical_analogs(packet)
    return packet


def deterministic_arena_prior(packet: ArenaForecastPacket) -> ArenaPrior:
    outcomes = packet.outcomes
    if not outcomes:
        return ArenaPrior({}, 0.0, 1.0, "empty_outcomes", ["empty_outcomes"])

    uniform = {outcome: 1.0 / len(outcomes) for outcome in outcomes}
    category_p = _category_yes_rate(packet.category)
    analog_p = _analog_yes_rate(packet.historical_analogs)
    entity_probs = _entity_distribution(packet)
    live_probs = _live_probability_distribution(packet)

    reason_codes = ["uniform_anchor"]
    diagnostics: dict[str, Any] = {
        "category_yes_rate": category_p,
        "analog_yes_rate": analog_p,
        "entity_distribution": entity_probs,
        "live_distribution": live_probs,
        "n_historical_analogs": len(packet.historical_analogs),
    }

    if _is_yes_no(outcomes):
        p_yes = 0.50
        total_weight = 1.0
        if category_p is not None:
            p_yes += 0.25 * (category_p - 0.50)
            total_weight += 0.25
            reason_codes.append("category_base_rate")
        if analog_p is not None:
            p_yes += 0.45 * (analog_p - 0.50)
            total_weight += 0.45
            reason_codes.append("nearest_neighbor_analogs")
        if live_probs is None:
            p_yes = max(0.12, min(0.88, p_yes))
        probs = {"YES": p_yes, "NO": 1.0 - p_yes}
        probs = _blend_distribution(uniform, probs, min(0.80, total_weight / 2.20))
    elif entity_probs:
        probs = _blend_distribution(uniform, entity_probs, 0.65)
        reason_codes.append("entity_historical_rates")
    else:
        probs = dict(uniform)

    if live_probs:
        probs = _blend_distribution(probs, live_probs, 0.80 if _has_market_quote_probability(packet) else 0.45)
        reason_codes.append("live_evidence_probabilities")

    probs = enforce_constraints(probs, outcomes, packet.event_structure, lo=0.001, hi=0.999)
    confidence = _prior_confidence(packet, reason_codes, live_probs is not None)
    uncertainty = 1.0 - confidence
    packet.deterministic_priors = probs
    packet.features.update({
        "prior_confidence": confidence,
        "prior_uncertainty": uncertainty,
        "prior_reason_codes": reason_codes,
    })
    return ArenaPrior(
        probabilities=probs,
        confidence=confidence,
        uncertainty=uncertainty,
        source="deterministic_arena_prior",
        reason_codes=reason_codes,
        diagnostics=diagnostics,
    )


def historical_analogs(packet: ArenaForecastPacket, *, limit: int = 8) -> list[dict[str, Any]]:
    query_tokens = _event_tokens(packet.title, packet.rules, packet.description, packet.category)
    if not query_tokens:
        return []
    rows = []
    for record in _historical_records():
        if record.market_ticker == packet.market_ticker or record.event_ticker == packet.event_ticker:
            continue
        overlap = len(query_tokens & record.tokens)
        if overlap == 0:
            continue
        union = len(query_tokens | record.tokens)
        score = overlap / union if union else 0.0
        if record.category == packet.category:
            score += 0.08
        for outcome in packet.outcomes:
            if _label_key(outcome) == _label_key(record.label):
                score += 0.18
                break
        if score < 0.10:
            continue
        rows.append((score, record))
    rows.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "market_ticker": record.market_ticker,
            "event_ticker": record.event_ticker,
            "title": record.title,
            "subtitle": record.subtitle,
            "category": record.category,
            "label": record.label,
            "resolved_yes": bool(record.outcome),
            "similarity": round(score, 4),
        }
        for score, record in rows[:limit]
    ]


def _clean_outcomes(raw: Any) -> list[str]:
    values = [str(item) for item in (raw or ["YES", "NO"]) if str(item)]
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out or ["YES", "NO"]


def _extract_entities(event: dict[str, Any], outcomes: list[str]) -> dict[str, Any]:
    text = " ".join(str(event.get(key) or "") for key in ("title", "subtitle", "description", "context", "rules"))
    multileg = parse_kalshi_multileg_contract({**event, "outcomes": outcomes})
    return {
        "outcome_labels": outcomes,
        "outcome_keys": [_label_key(outcome) for outcome in outcomes],
        "numbers": re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))[:12],
        "series_prefix": str(event.get("event_ticker") or event.get("market_ticker") or "").split("-", 1)[0],
        "tokens": sorted(_tokens(text))[:80],
        "category_hooks": _category_hooks(normalize_category(event.get("category"), event.get("event_ticker")), text),
        "kalshi_multileg_contract": multileg,
    }


def _category_hooks(category: str, text: str) -> dict[str, Any]:
    lowered = text.lower()
    if category == "Sports":
        return {
            "preferred_live_sources": ["oddspipe", "espn", "lseg"],
            "signals": [signal for signal in ("injury", "lineup", "ranking", "odds", "spread", "moneyline") if signal in lowered],
        }
    if category in {"Politics", "Elections"}:
        return {
            "preferred_live_sources": ["lseg", "gdelt", "reddit"],
            "signals": [signal for signal in ("poll", "approval", "election", "candidate", "vote") if signal in lowered],
        }
    if category in {"Economics", "Financials", "Crypto", "Commodities"}:
        return {
            "preferred_live_sources": ["fred", "bea", "eia", "polygon", "wrds", "lseg"],
            "signals": [
                signal
                for signal in ("cpi", "inflation", "unemployment", "fed", "gdp", "oil", "bitcoin", "ethereum", "rate")
                if signal in lowered
            ],
        }
    if category in {"Entertainment", "Culture"}:
        return {
            "preferred_live_sources": ["lseg", "gdelt", "reddit"],
            "signals": [signal for signal in ("award", "box office", "spoiler", "release", "nomination") if signal in lowered],
        }
    return {"preferred_live_sources": ["lseg", "gdelt"], "signals": []}


def _event_tokens(*parts: str | None) -> frozenset[str]:
    return frozenset(_tokens(" ".join(part or "" for part in parts)))


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "will", "who", "what", "when", "where", "for", "and", "or", "to", "of", "on", "in",
        "a", "an", "be", "by", "with", "resolve", "resolves", "official", "market", "predict",
    }
    return {tok for tok in TOKEN_RE.findall(text.lower()) if len(tok) > 1 and tok not in stop}


def _label_key(label: str | None) -> str:
    return " ".join(TOKEN_RE.findall(str(label or "").lower()))


def _is_yes_no(outcomes: list[str]) -> bool:
    return [outcome.upper() for outcome in outcomes] == ["YES", "NO"]


def _category_yes_rate(category: str) -> float | None:
    yes, total = _category_stats().get(category, (0, 0))
    if total <= 0:
        return None
    global_yes, global_total = _category_stats().get("__GLOBAL__", (0, 0))
    global_rate = global_yes / global_total if global_total else 0.5
    return (yes + 20.0 * global_rate) / (total + 20.0)


def _analog_yes_rate(analogs: list[dict[str, Any]]) -> float | None:
    if not analogs:
        return None
    weighted = 0.0
    total_weight = 0.0
    for item in analogs:
        weight = max(0.01, float(item.get("similarity") or 0.0))
        weighted += weight * (1.0 if item.get("resolved_yes") else 0.0)
        total_weight += weight
    return weighted / total_weight if total_weight else None


def _entity_distribution(packet: ArenaForecastPacket) -> dict[str, float] | None:
    if _is_yes_no(packet.outcomes):
        return None
    category_rate = _category_yes_rate(packet.category) or 0.5
    scores: dict[str, float] = {}
    used = False
    for outcome in packet.outcomes:
        yes, total = _entity_stats().get(_label_key(outcome), (0, 0))
        if total:
            used = True
        scores[outcome] = (yes + 8.0 * category_rate) / (total + 8.0) if total else category_rate
    if not used:
        return None
    return normalize_distribution(scores, packet.outcomes, lo=0.001, hi=0.999)


def _live_probability_distribution(packet: ArenaForecastPacket) -> dict[str, float] | None:
    inline_quote = packet.features.get("inline_market_quote") if isinstance(packet.features, dict) else None
    if _is_yes_no(packet.outcomes) and isinstance(inline_quote, dict) and _is_number(inline_quote.get("yes_probability")):
        p_yes = float(inline_quote["yes_probability"])
        return enforce_constraints({"YES": p_yes, "NO": 1.0 - p_yes}, packet.outcomes, "binary")
    for item in packet.live_evidence:
        raw = item.get("probabilities") or item.get("market_probabilities")
        if isinstance(raw, dict):
            return enforce_constraints(
                {str(k): float(v) for k, v in raw.items() if _is_number(v)},
                packet.outcomes,
                packet.event_structure,
                lo=0.001,
                hi=0.999,
            )
        if _is_yes_no(packet.outcomes) and _is_number(item.get("yes_probability")):
            p_yes = float(item["yes_probability"])
            return enforce_constraints({"YES": p_yes, "NO": 1.0 - p_yes}, packet.outcomes, "binary")
    return None


def _has_market_quote_probability(packet: ArenaForecastPacket) -> bool:
    inline_quote = packet.features.get("inline_market_quote") if isinstance(packet.features, dict) else None
    if isinstance(inline_quote, dict) and _is_number(inline_quote.get("yes_probability")):
        return True
    market_sources = {
        "kalshi_public_market",
        "kalshi_random_pit_market_snapshot",
        "polymarket_public_search",
    }
    return any(
        isinstance(item, dict)
        and item.get("source") in market_sources
        and (
            _is_number(item.get("yes_probability"))
            or isinstance(item.get("probabilities") or item.get("market_probabilities"), dict)
        )
        for item in packet.live_evidence
    )


def _inline_market_probability(event: dict[str, Any]) -> float | None:
    for key in ("yes_probability", "market_yes_probability", "yes_mid", "market_mid", "last_price"):
        if _is_number(event.get(key)):
            return _normalize_quote_value(float(event[key]))
    yes_bid = _quote_field(event, "yes_bid", "yes_bid_dollars")
    yes_ask = _quote_field(event, "yes_ask", "yes_ask_dollars")
    no_bid = _quote_field(event, "no_bid", "no_bid_dollars")
    no_ask = _quote_field(event, "no_ask", "no_ask_dollars")
    if yes_bid is not None and yes_ask is not None:
        return (yes_bid + yes_ask) / 2.0
    if yes_ask is not None and no_ask is not None:
        return (yes_ask + (1.0 - no_ask)) / 2.0
    if yes_bid is not None and no_bid is not None:
        return (yes_bid + (1.0 - no_bid)) / 2.0
    return None


def _quote_field(event: dict[str, Any], cents_key: str, dollars_key: str) -> float | None:
    if _is_number(event.get(dollars_key)):
        return _normalize_quote_value(float(event[dollars_key]))
    if _is_number(event.get(cents_key)):
        return _normalize_quote_value(float(event[cents_key]))
    return None


def _normalize_quote_value(value: float) -> float:
    if value > 1.0:
        value /= 100.0
    return max(0.001, min(0.999, value))


def _blend_distribution(
    left: dict[str, float],
    right: dict[str, float],
    weight: float,
) -> dict[str, float]:
    outcomes = list(left.keys())
    weight = max(0.0, min(1.0, weight))
    raw = {outcome: (1.0 - weight) * left[outcome] + weight * right.get(outcome, left[outcome]) for outcome in outcomes}
    return normalize_distribution(raw, outcomes, lo=0.001, hi=0.999)


def _prior_confidence(packet: ArenaForecastPacket, reason_codes: list[str], has_live_probs: bool) -> float:
    confidence = 0.24
    if "category_base_rate" in reason_codes:
        confidence += 0.08
    if "nearest_neighbor_analogs" in reason_codes:
        confidence += min(0.16, 0.02 * len(packet.historical_analogs))
    if "entity_historical_rates" in reason_codes:
        confidence += 0.12
    if has_live_probs:
        confidence += 0.20
    if packet.horizon_hours is None:
        confidence -= 0.03
    elif packet.horizon_hours < 1:
        confidence += 0.04
    elif packet.horizon_hours > 24 * 30:
        confidence -= 0.04
    if len(packet.outcomes) > 4:
        confidence -= 0.05
    return max(0.10, min(0.80, confidence))


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=1)
def _category_stats() -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in _historical_records():
        counts[record.category][1] += 1
        counts[record.category][0] += record.outcome
        counts["__GLOBAL__"][1] += 1
        counts["__GLOBAL__"][0] += record.outcome
    return {key: (value[0], value[1]) for key, value in counts.items()}


@lru_cache(maxsize=1)
def _entity_stats() -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in _historical_records():
        key = _label_key(record.label)
        if not key:
            continue
        counts[key][1] += 1
        counts[key][0] += record.outcome
    return {key: (value[0], value[1]) for key, value in counts.items()}


@lru_cache(maxsize=1)
def _historical_records() -> tuple[HistoricalRecord, ...]:
    return tuple()
