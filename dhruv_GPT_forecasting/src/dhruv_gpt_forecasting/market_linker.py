"""Linked prediction-market inference model.

This module converts related prediction markets into a compact probability
distribution. It is intentionally deterministic and point-in-time: callers pass
only evidence that has already been truncated to the forecast `as_of`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .constraints import normalize_distribution
from .schemas import FeaturePacket, clamp_prob


LINKABLE_CONTEXT_SOURCES = {"kalshi_nonbinary_context", "kalshi_topvol_same_event"}


@dataclass(frozen=True)
class LinkedMarketForecast:
    target_ticker: str
    event_ticker: str
    probabilities: dict[str, float]
    component_distribution: list[dict[str, Any]]
    confidence: float
    uncertainty: float
    quality: float
    inferred_structure: str
    reason_codes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_evidence(self) -> dict[str, Any]:
        return {
            "source": "linked_market_model",
            "family": "secondary_model",
            "claim": "Linked prediction markets imply a point-in-time probability distribution.",
            "target_ticker": self.target_ticker,
            "event_ticker": self.event_ticker,
            "probabilities": self.probabilities,
            "component_distribution": self.component_distribution[:8],
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "quality": self.quality,
            "inferred_structure": self.inferred_structure,
            "reason_codes": self.reason_codes,
            "diagnostics": self.diagnostics,
        }


def infer_linked_market_distribution(
    packet: FeaturePacket,
    context_evidence: list[dict[str, Any]] | None = None,
) -> LinkedMarketForecast | None:
    """Infer p(YES) from linked prediction-market components.

    Same-event component markets often describe a mutually exclusive event
    distribution, such as a game winner, award winner, or threshold/range
    ladder. The model normalizes component market mids when the group is
    coherent and falls back to the target component's own market mid otherwise.
    """
    evidence = context_evidence if context_evidence is not None else packet.evidence_digest
    candidates = [
        _forecast_from_context_item(packet, item)
        for item in evidence
        if isinstance(item, dict) and item.get("source") in LINKABLE_CONTEXT_SOURCES
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.quality, item.confidence))


def _forecast_from_context_item(packet: FeaturePacket, item: dict[str, Any]) -> LinkedMarketForecast | None:
    derived = item.get("derived") or {}
    target_ticker = packet.market_ticker
    target_prob = _float_or_none(derived.get("target_normalized_probability"))
    target_mid = _float_or_none(derived.get("target_yes_market_mid"))
    sum_mid = _float_or_none(derived.get("sum_yes_market_mid"))
    priced_count = int(derived.get("priced_component_count") or 0)
    component_count = int(derived.get("component_count") or 0)
    distribution = _component_distribution(item)
    structure = _infer_structure(sum_mid, priced_count, component_count)
    quality = _quality(sum_mid, priced_count, component_count, distribution, target_prob, target_mid)
    if target_prob is None and target_mid is not None:
        target_prob = target_mid
    if target_prob is None:
        return None

    # If the linked group is a strong mutually-exclusive distribution, use the
    # normalized component probability. If not, mix toward the target market mid.
    if target_mid is not None and structure != "mutually_exclusive_component_distribution":
        weight = min(0.50, 0.25 + 0.25 * quality)
        target_prob = (1.0 - weight) * target_mid + weight * target_prob
    target_prob = clamp_prob(target_prob, lo=0.001, hi=0.999)
    reason_codes_extra: list[str] = []
    if packet.is_binary_yes_no:
        probs = {"YES": target_prob, "NO": 1.0 - target_prob}
    else:
        probs = linked_distribution_for_outcomes(packet.outcomes, distribution)
        if probs is None:
            probs = {outcome: 1.0 / max(1, len(packet.outcomes)) for outcome in packet.outcomes}
        else:
            reason_codes_extra.append("named_outcome_component_match")
    confidence = max(0.05, min(0.85, 0.20 + 0.55 * quality))
    reason_codes = ["linked_prediction_markets", structure]
    reason_codes.extend(reason_codes_extra)
    if target_mid is not None:
        reason_codes.append("target_component_quote")
    if derived.get("target_rank_by_normalized_probability") == 1:
        reason_codes.append("target_component_favorite")
    return LinkedMarketForecast(
        target_ticker=target_ticker,
        event_ticker=packet.event_ticker,
        probabilities=probs,
        component_distribution=distribution,
        confidence=confidence,
        uncertainty=1.0 - confidence,
        quality=quality,
        inferred_structure=structure,
        reason_codes=reason_codes,
        diagnostics={
            "source": item.get("source"),
            "relation": item.get("relation"),
            "group_key": item.get("group_key"),
            "sum_yes_market_mid": sum_mid,
            "priced_component_count": priced_count,
            "component_count": component_count,
            "target_yes_market_mid": target_mid,
            "target_normalized_probability": target_prob,
            "target_rank_by_normalized_probability": derived.get("target_rank_by_normalized_probability"),
            "target_gap_to_favorite_probability": derived.get("target_gap_to_favorite_probability"),
            "normalized_probability_entropy": derived.get("normalized_probability_entropy"),
        },
    )


def _component_distribution(item: dict[str, Any]) -> list[dict[str, Any]]:
    derived = item.get("derived") or {}
    provided = derived.get("normalized_distribution_top")
    if isinstance(provided, list) and provided:
        return [
            {
                "ticker": row.get("ticker"),
                "label": row.get("yes_sub_title"),
                "probability": clamp_prob(row.get("normalized_probability"), lo=0.001, hi=0.999),
                "market_mid": _float_or_none(row.get("market_mid")),
            }
            for row in provided
            if isinstance(row, dict)
        ]
    priced = []
    for component in item.get("components") or []:
        if not isinstance(component, dict):
            continue
        mid = _float_or_none((component.get("pre_as_of_quote") or {}).get("market_mid"))
        if mid is None:
            continue
        priced.append((component, mid))
    total = sum(mid for _, mid in priced)
    if total <= 0.0:
        return []
    rows = [
        {
            "ticker": component.get("ticker"),
            "label": component.get("yes_sub_title"),
            "probability": clamp_prob(mid / total, lo=0.001, hi=0.999),
            "market_mid": mid,
        }
        for component, mid in priced
    ]
    return sorted(rows, key=lambda row: row["probability"], reverse=True)[:8]


def _infer_structure(sum_mid: float | None, priced_count: int, component_count: int) -> str:
    if priced_count >= 2 and sum_mid is not None and 0.75 <= sum_mid <= 1.25:
        return "mutually_exclusive_component_distribution"
    if priced_count >= 2 and component_count >= 2:
        return "soft_component_distribution"
    return "single_linked_market_quote"


def _quality(
    sum_mid: float | None,
    priced_count: int,
    component_count: int,
    distribution: list[dict[str, Any]],
    target_prob: float | None,
    target_mid: float | None,
) -> float:
    if target_prob is None and target_mid is None:
        return 0.0
    coverage = priced_count / max(1, component_count)
    count_quality = min(1.0, priced_count / 4.0)
    sum_quality = 0.55
    if sum_mid is not None:
        sum_quality = max(0.0, 1.0 - min(1.0, abs(sum_mid - 1.0)))
    dist_quality = 0.20 if distribution else 0.0
    return max(0.0, min(1.0, 0.35 * coverage + 0.25 * count_quality + 0.25 * sum_quality + dist_quality))


def linked_distribution_for_outcomes(
    outcomes: list[str],
    component_distribution: list[dict[str, Any]],
) -> dict[str, float] | None:
    """Map component labels into a named-outcome Arena distribution when possible."""
    if not outcomes or not component_distribution:
        return None
    scores = {outcome: 0.0 for outcome in outcomes}
    matched = 0
    for outcome in outcomes:
        outcome_key = _label_key(outcome)
        for row in component_distribution:
            label_key = _label_key(row.get("label"))
            if not outcome_key or not label_key:
                continue
            if outcome_key == label_key or outcome_key in label_key or label_key in outcome_key:
                scores[outcome] += float(row.get("probability") or 0.0)
                matched += 1
    if matched == 0 or sum(scores.values()) <= 0.0:
        return None
    return normalize_distribution(scores, outcomes, lo=0.001, hi=0.999)


def _label_key(value: Any) -> str:
    import re

    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
