"""Probability normalization and structural constraints."""

from __future__ import annotations

import re
from typing import Any

from .schemas import EventStructure, clamp_prob


def normalize_distribution(
    probs: dict[str, float],
    outcomes: list[str],
    *,
    lo: float = 0.001,
    hi: float = 0.999,
) -> dict[str, float]:
    if not outcomes:
        return {}
    cleaned = {outcome: clamp_prob(probs.get(outcome, 1.0 / len(outcomes)), lo=lo, hi=hi) for outcome in outcomes}
    total = sum(cleaned.values())
    if total <= 0:
        return {outcome: 1.0 / len(outcomes) for outcome in outcomes}
    return {outcome: value / total for outcome, value in cleaned.items()}


def align_probabilities(raw: dict[str, Any], outcomes: list[str]) -> dict[str, float]:
    """Map model-returned labels onto exact outcome labels using casefold fallback."""
    if not outcomes:
        return {}
    exact: dict[str, float] = {}
    case_map = {str(k).casefold(): k for k in raw.keys()}
    for outcome in outcomes:
        value = raw.get(outcome)
        if value is None:
            raw_key = case_map.get(outcome.casefold())
            value = raw.get(raw_key) if raw_key is not None else None
        if value is not None:
            try:
                exact[outcome] = float(value)
            except (TypeError, ValueError):
                pass
    return exact


def threshold_value(label: str) -> tuple[str, float] | None:
    text = label.lower()
    match = re.search(r"(-?\d+(?:\.\d+)?)", text.replace(",", ""))
    if not match:
        return None
    value = float(match.group(1))
    direction = "above" if any(token in text for token in ("above", "over", "at least", ">")) else "below"
    if any(token in text for token in ("below", "under", "less than", "<")):
        direction = "below"
    return direction, value


def enforce_threshold_monotonicity(
    probs: dict[str, float],
    *,
    lo: float = 0.001,
    hi: float = 0.999,
) -> dict[str, float]:
    parsed = [(outcome, threshold_value(outcome), probs[outcome]) for outcome in probs]
    usable = [(outcome, parsed_value, prob) for outcome, parsed_value, prob in parsed if parsed_value is not None]
    if len(usable) < 2:
        return probs
    direction_counts = {"above": 0, "below": 0}
    for _, parsed_value, _ in usable:
        direction_counts[parsed_value[0]] += 1
    direction = "above" if direction_counts["above"] >= direction_counts["below"] else "below"
    ordered = sorted(
        [(outcome, parsed_value[1], prob) for outcome, parsed_value, prob in usable if parsed_value[0] == direction],
        key=lambda x: x[1],
    )
    adjusted = dict(probs)
    if direction == "above":
        running = 1.0
        for outcome, _, prob in ordered:
            running = min(running, prob)
            adjusted[outcome] = running
    else:
        running = 0.0
        for outcome, _, prob in ordered:
            running = max(running, prob)
            adjusted[outcome] = running
    return {outcome: clamp_prob(value, lo=lo, hi=hi) for outcome, value in adjusted.items()}


def enforce_constraints(
    probabilities: dict[str, float],
    outcomes: list[str],
    event_structure: EventStructure,
    *,
    lo: float = 0.001,
    hi: float = 0.999,
) -> dict[str, float]:
    aligned = align_probabilities(probabilities, outcomes)
    if not aligned:
        aligned = {outcome: 1.0 / max(1, len(outcomes)) for outcome in outcomes}
    if event_structure == "threshold_ladder":
        ladder = enforce_threshold_monotonicity(
            {outcome: aligned.get(outcome, 0.5) for outcome in outcomes},
            lo=lo,
            hi=hi,
        )
        return normalize_distribution(ladder, outcomes, lo=lo, hi=hi)
    if event_structure in {"binary", "mutually_exclusive", "range_bucket"}:
        return normalize_distribution(aligned, outcomes, lo=lo, hi=hi)
    return {outcome: clamp_prob(aligned.get(outcome, 0.5), lo=lo, hi=hi) for outcome in outcomes}
