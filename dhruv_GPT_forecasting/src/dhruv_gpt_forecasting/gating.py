"""LLM call gates."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import GateConfig
from .schemas import FeaturePacket, LaneForecast, StatForecast


@dataclass
class GateDecision:
    call_cheap: bool
    call_supervisor: bool
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "call_cheap": self.call_cheap,
            "call_supervisor": self.call_supervisor,
            "reason_codes": self.reason_codes,
        }


def cheap_gate(packet: FeaturePacket, stat: StatForecast, cfg: GateConfig) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    spread = packet.quote.spread
    stat_delta = abs(stat.calibrated_probability - stat.market_prior)
    tight = spread is not None and spread <= cfg.cheap_tight_spread
    near_settled = packet.horizon_hours is not None and packet.horizon_hours <= 3.0
    evidence_reasons = _evidence_signal_reasons(packet, stat)
    has_evidence_signal = bool(evidence_reasons)
    if tight and near_settled and stat_delta <= cfg.cheap_delta_pp and not has_evidence_signal:
        return False, ["market_tight_near_settled"]
    if spread is None or spread >= cfg.cheap_high_spread:
        reasons.append("unusual_or_wide_spread")
    if stat_delta > cfg.cheap_delta_pp:
        reasons.append("stat_market_disagreement")
    if evidence_reasons:
        reasons.extend(evidence_reasons)
    if packet.rules is None or len(packet.rules.strip()) < 20:
        reasons.append("rules_sparse")
    if packet.category in cfg.cheap_categories and reasons:
        reasons.append("category_needs_reasoning")
    return bool(reasons), reasons or ["no_llm_edge_signal"]


def _evidence_signal_reasons(packet: FeaturePacket, stat: StatForecast) -> list[str]:
    """Return evidence reasons strong enough to spend an LLM call.

    Structural related-market context is often attached to every market. That
    should help prompts when another gate is already open, but should not by
    itself trigger the LLM unless it is cross-venue, real-time, or materially
    disagrees with the target quote.
    """
    reasons: list[str] = []
    structural_sources = {
        "kalshi_nonbinary_context",
        "kalshi_topvol_same_event",
        "kalshi_polymarket_map_gap",
    }
    for item in packet.evidence_digest:
        source = str(item.get("source") or "")
        if source == "kalshi_polymarket_map":
            reasons.append("cross_venue_context_present")
            continue
        if source and source not in structural_sources:
            reasons.append("external_realtime_evidence_present")
            continue
        if source == "kalshi_topvol_same_event":
            continue
        derived = item.get("derived") or {}
        target_mid = derived.get("target_yes_market_mid")
        sum_mid = derived.get("sum_yes_market_mid")
        if target_mid is None or sum_mid is None:
            continue
        try:
            normalized = float(target_mid) / float(sum_mid)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if abs(normalized - stat.market_prior) >= 0.05:
            reasons.append("related_market_disagreement")
    return list(dict.fromkeys(reasons))


def supervisor_gate(
    packet: FeaturePacket,
    stat: StatForecast,
    cheap: LaneForecast | None,
    cfg: GateConfig,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if cheap is not None and packet.is_binary_yes_no:
        cheap_yes = cheap.probabilities.get("YES", stat.calibrated_probability)
        if abs(cheap_yes - stat.calibrated_probability) >= cfg.supervisor_disagreement_pp:
            reasons.append("cheap_stat_disagreement")
    yes_ask = packet.quote.executable_yes
    no_ask = packet.quote.executable_no
    if packet.is_binary_yes_no and yes_ask is not None and no_ask is not None:
        p_yes = cheap.probabilities.get("YES", stat.calibrated_probability) if cheap else stat.calibrated_probability
        edge = max(p_yes - yes_ask, (1.0 - p_yes) - no_ask)
        if edge >= cfg.supervisor_min_edge:
            reasons.append("preliminary_edge_clears")
    liquidity = packet.quote.liquidity or packet.quote.volume or 0.0
    if liquidity >= cfg.high_notional_liquidity and cheap is not None and cheap.confidence >= 0.65:
        reasons.append("high_notional_high_confidence")
    return bool(reasons), reasons or ["supervisor_not_needed"]


def decide_gates(
    packet: FeaturePacket,
    stat: StatForecast,
    cfg: GateConfig,
    cheap: LaneForecast | None = None,
) -> GateDecision:
    call_cheap, cheap_reasons = cheap_gate(packet, stat, cfg)
    call_supervisor = False
    supervisor_reasons: list[str] = []
    if cheap is not None:
        call_supervisor, supervisor_reasons = supervisor_gate(packet, stat, cheap, cfg)
    return GateDecision(
        call_cheap=call_cheap,
        call_supervisor=call_supervisor,
        reason_codes=cheap_reasons + supervisor_reasons,
    )
