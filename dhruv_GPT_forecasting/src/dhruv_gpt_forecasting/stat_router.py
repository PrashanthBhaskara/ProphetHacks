"""Statistical model candidates and routing for binary market forecasts."""

from __future__ import annotations

import math
from typing import Callable

from .config import StatConfig
from .constraints import enforce_constraints
from .schemas import FeaturePacket, StatForecast, clamp_prob
from .stat_lane import forecast_stat, inv_logit, logit


Predictor = Callable[[FeaturePacket], float]


def model_registry(*, include_context: bool) -> dict[str, Predictor]:
    """Return deterministic binary p(YES) predictors used offline and live."""
    models: dict[str, Predictor] = {
        "market_mid": lambda packet: packet.quote.market_mid,
        "stat_default": lambda packet: _stat_with_caps(packet, default_dev=0.02, max_dev=0.04),
        "stat_cap_1pp": lambda packet: _stat_with_caps(packet, default_dev=0.01, max_dev=0.01),
        "stat_cap_2pp": lambda packet: _stat_with_caps(packet, default_dev=0.02, max_dev=0.02),
        "stat_cap_3pp": lambda packet: _stat_with_caps(packet, default_dev=0.03, max_dev=0.03),
        "stat_cap_6pp": lambda packet: _stat_with_caps(packet, default_dev=0.03, max_dev=0.06),
        "stat_spread_shrink": _spread_shrunk_stat,
        "ewma_logit_fast": lambda packet: _ewma_logit(packet, half_life=4.0, momentum_scale=0.08, cap=0.04),
        "ewma_logit_slow": lambda packet: _ewma_logit(packet, half_life=16.0, momentum_scale=0.04, cap=0.03),
        "volatility_shrunk_market": _volatility_shrunk_market,
        "momentum_follow_5pct": lambda packet: _momentum(packet, scale=0.05, cap=0.03),
        "momentum_follow_10pct": lambda packet: _momentum(packet, scale=0.10, cap=0.04),
        "momentum_revert_5pct": lambda packet: _momentum(packet, scale=-0.05, cap=0.03),
        "recent_follow_10pct": lambda packet: _recent_momentum(packet, scale=0.10, cap=0.03),
        "recent_revert_10pct": lambda packet: _recent_momentum(packet, scale=-0.10, cap=0.03),
        "near_close_calibrated": _near_close_or_market,
    }
    if include_context:
        models.update({
            "context_implied": _context_or_market,
            "context_norm_25pct": lambda packet: _context_normalized(packet, weight=0.25),
            "context_norm_50pct": lambda packet: _context_normalized(packet, weight=0.50),
            "context_norm_75pct": lambda packet: _context_normalized(packet, weight=0.75),
            "stat_context_norm_25pct": lambda packet: _blend(
                _spread_shrunk_stat(packet),
                _context_normalized(packet, weight=0.50),
                weight=0.25,
            ),
            "stat_context_norm_50pct": lambda packet: _blend(
                _spread_shrunk_stat(packet),
                _context_normalized(packet, weight=0.50),
                weight=0.50,
            ),
        })
    return models


def predict_with_model(name: str, packet: FeaturePacket, *, include_context: bool = True) -> float:
    models = model_registry(include_context=include_context)
    predictor = models.get(name) or models["stat_spread_shrink"]
    return clamp_prob(predictor(packet))


def heuristic_model_name(packet: FeaturePacket, *, include_context: bool = True) -> str:
    """Conservative category defaults before an OOS-trained route file exists."""
    has_context_prob = _context_target_prob(packet) is not None
    if include_context and has_context_prob and packet.category == "Sports":
        return "stat_context_norm_25pct"
    if include_context and has_context_prob:
        return "stat_context_norm_25pct"
    if packet.category == "Crypto":
        return "momentum_follow_10pct"
    if packet.category == "Sports":
        return "near_close_calibrated"
    if packet.category == "Other":
        return "ewma_logit_slow"
    if packet.category in {"Economics", "Financials", "Commodities"}:
        return "stat_cap_2pp"
    if packet.category in {"Politics", "Elections", "Entertainment"}:
        return "stat_spread_shrink"
    if packet.category in {"Climate and Weather", "Weather"}:
        return "ewma_logit_slow"
    return "stat_spread_shrink"


def forecast_stat_routed(
    packet: FeaturePacket,
    cfg: StatConfig | None = None,
    *,
    model_name: str | None = None,
    include_context: bool = True,
) -> StatForecast:
    """Return a StatForecast using the selected category/context model."""
    cfg = cfg or StatConfig()
    selected = model_name or heuristic_model_name(packet, include_context=include_context)
    p_yes = predict_with_model(selected, packet, include_context=include_context)
    base = forecast_stat(packet, cfg)
    probs = {"YES": p_yes, "NO": 1.0 - p_yes} if packet.is_binary_yes_no else base.probabilities
    probs = enforce_constraints(probs, packet.outcomes, packet.event_structure)
    uncertainty = _routed_uncertainty(packet, selected, base.uncertainty)
    confidence = max(0.0, min(1.0, 1.0 - uncertainty))
    return StatForecast(
        probabilities=probs,
        market_prior=base.market_prior,
        calibrated_probability=p_yes,
        uncertainty=uncertainty,
        confidence=confidence,
        reason_codes=[*base.reason_codes, "stat_model_router", f"stat_model:{selected}"],
        diagnostics={
            **base.diagnostics,
            "routed_model": selected,
            "routed_probability": p_yes,
            "context_target_probability": _context_target_prob(packet),
            "context_distribution_quality": _context_quality(packet),
        },
    )


def _stat_with_caps(packet: FeaturePacket, *, default_dev: float, max_dev: float) -> float:
    cfg = StatConfig(default_market_deviation=default_dev, max_market_deviation=max_dev)
    stat = forecast_stat(packet, cfg)
    return stat.probabilities.get("YES", stat.calibrated_probability)


def _spread_shrunk_stat(packet: FeaturePacket) -> float:
    market = packet.quote.market_mid
    stat = forecast_stat(packet, StatConfig(default_market_deviation=0.03, max_market_deviation=0.06))
    p_stat = stat.probabilities.get("YES", stat.calibrated_probability)
    spread = float(packet.quote.spread or 0.10)
    n = int(packet.features.get("n_snapshots") or 0)
    quality = max(0.0, min(1.0, 1.0 - spread / 0.30))
    depth = max(0.25, min(1.0, n / 20.0))
    shrink = 0.65 * quality * depth
    return clamp_prob(market + shrink * (p_stat - market))


def _ewma_logit(packet: FeaturePacket, *, half_life: float, momentum_scale: float, cap: float) -> float:
    points = packet.price_trajectory
    if not points:
        return packet.quote.market_mid
    alpha = 1.0 - math.exp(math.log(0.5) / max(1.0, half_life))
    state = logit(points[0]["market_mid"])
    prev = state
    for point in points[1:]:
        prev = state
        state = alpha * logit(point["market_mid"]) + (1.0 - alpha) * state
    momentum = momentum_scale * (state - prev) if len(points) >= 2 else 0.0
    return inv_logit(state + max(-cap, min(cap, momentum)))


def _volatility_shrunk_market(packet: FeaturePacket) -> float:
    points = packet.price_trajectory
    market = packet.quote.market_mid
    if len(points) < 6:
        return market
    diffs = [
        float(points[i]["market_mid"] - points[i - 1]["market_mid"])
        for i in range(1, len(points))
    ]
    vol = math.sqrt(sum(diff * diff for diff in diffs) / len(diffs))
    # High short-window noise usually means the current quote should be shrunk
    # toward 50 rather than extrapolated.
    shrink = max(0.0, min(0.35, vol * 1.5))
    return clamp_prob((1.0 - shrink) * market + shrink * 0.5)


def _near_close_or_market(packet: FeaturePacket) -> float:
    stat = forecast_stat(packet, StatConfig(near_close_brier_enabled=True))
    return stat.probabilities.get("YES", stat.calibrated_probability)


def _momentum(packet: FeaturePacket, *, scale: float, cap: float) -> float:
    points = packet.price_trajectory
    if len(points) < 2:
        return packet.quote.market_mid
    market = packet.quote.market_mid
    move = float(points[-1]["market_mid"] - points[0]["market_mid"])
    return clamp_prob(market + max(-cap, min(cap, scale * move)))


def _recent_momentum(packet: FeaturePacket, *, scale: float, cap: float) -> float:
    points = packet.price_trajectory
    if len(points) < 5:
        return packet.quote.market_mid
    market = packet.quote.market_mid
    move = float(points[-1]["market_mid"] - points[-5]["market_mid"])
    return clamp_prob(market + max(-cap, min(cap, scale * move)))


def _context_or_market(packet: FeaturePacket) -> float:
    return _context_target_prob(packet) or packet.quote.market_mid


def _context_normalized(packet: FeaturePacket, *, weight: float) -> float:
    market = packet.quote.market_mid
    context_prob = _context_target_prob(packet)
    if context_prob is None:
        return market
    quality = _context_quality(packet)
    return clamp_prob(_blend(market, context_prob, weight * quality))


def _context_target_prob(packet: FeaturePacket) -> float | None:
    for item in packet.evidence_digest:
        if item.get("source") == "linked_market_model":
            probs = item.get("probabilities") or {}
            if isinstance(probs, dict) and probs.get("YES") is not None:
                return clamp_prob(float(probs["YES"]))
        derived = item.get("derived") or {}
        for key in ("target_normalized_probability", "target_yes_normalized_probability"):
            if derived.get(key) is not None:
                return clamp_prob(float(derived[key]))
        target_mid = derived.get("target_yes_market_mid")
        sum_mid = derived.get("sum_yes_market_mid")
        if target_mid is None or sum_mid is None:
            continue
        sum_mid = float(sum_mid)
        if 0.25 <= sum_mid <= 4.0:
            return clamp_prob(float(target_mid) / sum_mid)
    return None


def _context_quality(packet: FeaturePacket) -> float:
    quality = 0.0
    for item in packet.evidence_digest:
        derived = item.get("derived") or {}
        priced = int(derived.get("priced_component_count") or 0)
        component_count = int(derived.get("component_count") or 0)
        sum_mid = derived.get("sum_yes_market_mid")
        if priced <= 0:
            continue
        coverage = priced / max(1, component_count)
        sum_quality = 1.0
        if sum_mid is not None:
            sum_quality = max(0.0, 1.0 - min(1.0, abs(float(sum_mid) - 1.0)))
        quality = max(quality, min(1.0, 0.60 * coverage + 0.40 * sum_quality))
    spread = float(packet.quote.spread or 0.10)
    quote_quality = max(0.15, min(1.0, 1.0 - spread / 0.30))
    return max(0.0, min(1.0, quality * quote_quality))


def _blend(left: float, right: float, weight: float) -> float:
    weight = max(0.0, min(1.0, weight))
    return clamp_prob((1.0 - weight) * left + weight * right)


def _routed_uncertainty(packet: FeaturePacket, model_name: str, base_uncertainty: float) -> float:
    uncertainty = base_uncertainty
    if model_name.startswith("context") or "context" in model_name:
        uncertainty -= 0.10 * _context_quality(packet)
    if model_name.startswith("ewma") and packet.category in {"Crypto", "Climate and Weather", "Weather"}:
        uncertainty -= 0.03
    if packet.quote.spread is not None and packet.quote.spread > 0.20:
        uncertainty += 0.08
    return max(0.05, min(0.95, uncertainty))
