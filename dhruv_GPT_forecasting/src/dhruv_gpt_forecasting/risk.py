"""Deterministic trading gate and sizing."""

from __future__ import annotations

from .config import RiskConfig
from .schemas import FeaturePacket, TradeDecision


def _kelly_fraction(prob: float, price: float) -> float:
    if price <= 0.0 or price >= 1.0:
        return 0.0
    b = (1.0 - price) / price
    q = 1.0 - prob
    return max(0.0, (b * prob - q) / b)


def decide_trade(
    packet: FeaturePacket,
    probabilities: dict[str, float],
    confidence: float,
    uncertainty: float,
    cfg: RiskConfig,
) -> TradeDecision:
    spread = packet.quote.spread
    if spread is not None and spread >= cfg.hard_no_trade_spread:
        return TradeDecision("NONE", None, 0.0, cfg.min_edge, 0.0, "hard_no_trade_wide_spread")
    if not packet.is_binary_yes_no:
        return TradeDecision("NONE", None, 0.0, cfg.min_edge, 0.0, "non_binary_no_trade")
    yes_ask = packet.quote.executable_yes
    no_ask = packet.quote.executable_no
    if yes_ask is None or no_ask is None:
        return TradeDecision("NONE", None, 0.0, cfg.min_edge, 0.0, "missing_executable_price")
    p_yes = probabilities.get("YES", 0.5)
    p_no = probabilities.get("NO", 1.0 - p_yes)
    threshold = cfg.min_edge + cfg.fee_buffer + cfg.uncertainty_buffer * uncertainty
    if spread is not None:
        threshold += cfg.spread_buffer * spread
        if spread >= cfg.wide_spread:
            threshold += 0.04
    threshold += max(0.0, 0.50 - confidence) * 0.04
    yes_edge = p_yes - yes_ask
    no_edge = p_no - no_ask
    best_edge = max(yes_edge, no_edge)
    if best_edge < threshold:
        return TradeDecision("NONE", None, best_edge, threshold, 0.0, "no_edge_after_buffers")
    max_stake = cfg.starting_equity * cfg.max_equity_fraction_per_market
    if yes_edge >= no_edge:
        kelly = _kelly_fraction(p_yes, yes_ask) * cfg.kelly_fraction
        stake = min(max_stake, max_stake * kelly)
        return TradeDecision("YES", yes_ask, yes_edge, threshold, stake, "yes_edge_cleared")
    kelly = _kelly_fraction(p_no, no_ask) * cfg.kelly_fraction
    stake = min(max_stake, max_stake * kelly)
    return TradeDecision("NO", no_ask, no_edge, threshold, stake, "no_edge_cleared")


def advisory_from_decision(decision: TradeDecision) -> str:
    if decision.side == "YES":
        return "BUY_YES" if decision.edge >= decision.threshold * 1.75 else "BUY_YES_SMALL"
    if decision.side == "NO":
        return "BUY_NO" if decision.edge >= decision.threshold * 1.75 else "BUY_NO_SMALL"
    return "NO_TRADE"
