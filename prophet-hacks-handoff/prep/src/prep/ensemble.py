"""Supervisor aggregation for model forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .calibration import CalibrationConfig, calibrate_to_market, inv_logit, logit
from .schemas import MarketPacket, ModelForecast, SupervisorForecast, clamp_prob


@dataclass
class EnsembleMember:
    forecast: ModelForecast
    configured_weight: float = 1.0

    @property
    def effective_weight(self) -> float:
        diag = self.forecast.diagnostics
        f = self.forecast.forecast
        quality = {"low": 0.55, "medium": 0.85, "high": 1.0}.get(diag.evidence_quality, 0.85)
        clarity = {"low": 0.65, "medium": 0.85, "high": 1.0}.get(diag.rules_clarity, 0.85)
        defer = 0.75 if diag.should_defer_to_market else 1.0
        return max(0.01, self.configured_weight * quality * clarity * defer * (0.5 + f.confidence / 2.0))


def aggregate_forecasts(
    packet: MarketPacket,
    members: list[EnsembleMember],
    calibration: CalibrationConfig | None = None,
    *,
    market_anchor_weight: float = 1.5,
) -> SupervisorForecast:
    if not members:
        raw_p = packet.kalshi.market_mid
        assessments: list[dict[str, Any]] = []
    else:
        weighted_logits = [market_anchor_weight * logit(packet.kalshi.market_mid)]
        weights = [market_anchor_weight]
        assessments = []
        for member in members:
            w = member.effective_weight
            weighted_logits.append(w * logit(member.forecast.p_yes))
            weights.append(w)
            assessments.append({
                "model_id": member.forecast.model_id,
                "provider": member.forecast.provider,
                "p_yes": member.forecast.p_yes,
                "configured_weight": member.configured_weight,
                "effective_weight": w,
                "confidence": member.forecast.forecast.confidence,
                "summary": member.forecast.reasoning_track.summary,
                "defer_to_market": member.forecast.diagnostics.should_defer_to_market,
            })
        raw_p = inv_logit(sum(weighted_logits) / sum(weights))

    calibration = calibration or CalibrationConfig()
    calibrated_p, shrink_weight = calibrate_to_market(raw_p, packet, calibration)
    ps = [m.forecast.p_yes for m in members]
    disagreement = (max(ps) - min(ps)) if ps else 0.0
    confidence = clamp_prob(1.0 - disagreement, lo=0.0, hi=1.0)

    if disagreement > 0.20:
        disagreement_summary = f"High model disagreement: range {min(ps):.3f}-{max(ps):.3f}."
    elif disagreement > 0.08:
        disagreement_summary = f"Moderate model disagreement: range {min(ps):.3f}-{max(ps):.3f}."
    else:
        disagreement_summary = "Models are broadly aligned." if ps else "No model forecasts; using market anchor."

    thesis = (
        f"Market anchor {packet.kalshi.market_mid:.3f}; raw ensemble {raw_p:.3f}; "
        f"calibrated {calibrated_p:.3f} after shrink weight {shrink_weight:.3f}."
    )
    risk_notes = []
    if packet.kalshi.spread is not None and packet.kalshi.spread > 0.08:
        risk_notes.append(f"Wide spread: {packet.kalshi.spread:.3f}.")
    if disagreement > 0.20:
        risk_notes.append("Large model disagreement; reduce size or no-trade.")

    return SupervisorForecast(
        market_ticker=packet.market_ticker,
        raw_p_yes=raw_p,
        calibrated_p_yes=calibrated_p,
        confidence=confidence,
        model_assessment=assessments,
        disagreement_summary=disagreement_summary,
        final_trade_thesis=thesis,
        risk_notes=risk_notes,
    )
