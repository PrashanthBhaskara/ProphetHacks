"""Deterministic no-key forecaster used for smoke tests and backtests."""

from __future__ import annotations

import hashlib

from .base import (
    ForecasterConfig,
    clamped_market_plus_edge,
    forecast_from_response,
    heuristic_trade_recommendation,
    stable_prompt_hash,
)
from prep.schemas import MarketPacket


def _jitter_bps(config: ForecasterConfig, packet: MarketPacket) -> float:
    seed = f"{config.name}:{packet.market_ticker}:{packet.as_of}".encode()
    digest = hashlib.sha256(seed).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return (bucket - 0.5) * 300.0


def _build_distribution(packet: MarketPacket, p_yes: float) -> dict[str, float]:
    """Build a probabilities dict over packet.outcomes from a single edge-adjusted p_yes.

    Binary case: {"YES": p_yes, "NO": 1-p_yes}.
    Multi-outcome: split the (1 - p_yes) mass uniformly across remaining outcomes
    when YES is in the outcomes set; otherwise distribute uniformly (mock has no
    independent reasoning per outcome, so this is just an infrastructure stub).
    """
    outs = packet.outcomes
    if not outs:
        return {"YES": p_yes, "NO": 1.0 - p_yes}
    if tuple(outs) == ("YES", "NO"):
        return {"YES": p_yes, "NO": 1.0 - p_yes}
    if "YES" in outs:
        rest = [o for o in outs if o != "YES"]
        share = (1.0 - p_yes) / max(1, len(rest))
        return {"YES": p_yes, **{o: share for o in rest}}
    # Pure multi-outcome — mock has no per-outcome signal; uniform with first-outcome bias
    bias = p_yes
    others_share = (1.0 - bias) / max(1, len(outs) - 1)
    return {o: (bias if i == 0 else others_share) for i, o in enumerate(outs)}


def forecast(config: ForecasterConfig, packet: MarketPacket):
    p_yes = clamped_market_plus_edge(packet, config.mock_edge_bps + _jitter_bps(config, packet))
    probabilities = _build_distribution(packet, p_yes)
    response = {
        "forecast": {
            "probabilities": probabilities,
            "confidence": 0.45,
            "uncertainty": 0.35,
            "fair_yes_price": p_yes,
            "max_yes_buy_price": max(0.01, p_yes - 0.05),
            "max_no_buy_price": max(0.01, (1.0 - p_yes) - 0.05),
            "trade_recommendation": heuristic_trade_recommendation(p_yes, packet),
        },
        "reasoning_track": {
            "summary": "Mock forecast anchored on Kalshi midpoint with deterministic model-specific jitter.",
            "base_rate": "No external base-rate data used in mock mode.",
            "market_analysis": f"Market midpoint used as anchor: {packet.kalshi.market_mid:.3f}.",
            "key_evidence": [],
            "counterarguments": [{"claim": "Mock mode has no independent evidence.", "impact": "defer to market"}],
            "assumptions": ["Used only for infrastructure tests."],
            "information_gaps": ["No live retrieval or model call was made."],
            "what_would_change_my_mind": ["Replace mock adapter with Gemini/OpenRouter forecast."],
        },
        "diagnostics": {
            "evidence_quality": "low",
            "rules_clarity": "medium",
            "liquidity_quality": "medium",
            "market_disagreement_reason": "deterministic mock jitter",
            "should_defer_to_market": True,
        },
        "prompt_hash": stable_prompt_hash(packet, config),
    }
    return forecast_from_response(
        provider="mock",
        model_id=config.model,
        packet=packet,
        response=response,
        raw_response=response,
    )
