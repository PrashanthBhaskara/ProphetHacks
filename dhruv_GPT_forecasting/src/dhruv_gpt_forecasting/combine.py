"""Combine market, stat, cheap-lane, and supervisor forecasts."""

from __future__ import annotations

from .constraints import enforce_constraints
from .risk import advisory_from_decision, decide_trade
from .schemas import FeaturePacket, LaneForecast, StatForecast, SupervisorDecision, clamp_prob
from .config import RiskConfig


def combine_stat_and_lane(
    packet: FeaturePacket,
    stat: StatForecast,
    lane: LaneForecast | None,
) -> tuple[dict[str, float], float, float, str]:
    if lane is None:
        return stat.probabilities, stat.confidence, stat.uncertainty, "stat_only"
    if packet.is_binary_yes_no:
        market = stat.market_prior
        lane_yes = lane.probabilities.get("YES", stat.calibrated_probability)
        shrink = 0.20 + 0.35 * max(0.0, min(1.0, lane.confidence))
        if lane.defer_to_market:
            shrink *= 0.55
        final_yes = clamp_prob(market + shrink * (lane_yes - market))
        probs = {"YES": final_yes, "NO": 1.0 - final_yes}
    else:
        probs = lane.probabilities or stat.probabilities
    probs = enforce_constraints(probs, packet.outcomes, packet.event_structure)
    confidence = max(0.0, min(1.0, (stat.confidence + lane.confidence) / 2.0))
    uncertainty = max(stat.uncertainty, lane.uncertainty)
    return probs, confidence, uncertainty, "cheap_lane"


def make_supervisor_decision(
    packet: FeaturePacket,
    stat: StatForecast,
    risk_cfg: RiskConfig,
    *,
    cheap: LaneForecast | None = None,
    supervisor: LaneForecast | None = None,
    audit: dict | None = None,
) -> SupervisorDecision:
    if supervisor is not None:
        probs = enforce_constraints(supervisor.probabilities, packet.outcomes, packet.event_structure)
        confidence = supervisor.confidence
        uncertainty = supervisor.uncertainty
        source = "supervisor_lane"
    else:
        probs, confidence, uncertainty, source = combine_stat_and_lane(packet, stat, cheap)
    trade = decide_trade(packet, probs, confidence, uncertainty, risk_cfg)
    return SupervisorDecision(
        probabilities=probs,
        confidence=confidence,
        uncertainty=uncertainty,
        source=source,
        trade_recommendation=advisory_from_decision(trade),
        trade_decision=trade,
        audit_summary=audit or {},
    )

