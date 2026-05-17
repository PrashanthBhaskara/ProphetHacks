"""Supervisor aggregation for model forecasts.

Per Prophet Arena dev docs, predictions are a probability *distribution* over
the event's `outcomes` list. We logit-pool per outcome across lanes, then
renormalize. Binary events (outcomes=["YES","NO"]) reduce cleanly to the
old logit-pool of a single p_yes.

Market anchor only applies when the packet has a Kalshi quote (binary YES/NO).
For multi-outcome events without market data, we anchor toward a uniform prior
with a small weight (so a single noisy model can't dominate).

CHANGE vs. original: when a fitted `RecommendedPredictor` is passed in via
`data_baseline=`, the binary YES/NO anchor uses its calibrated p_yes
(Platt + event-size + logit-shrink α=0.5) instead of raw market_mid.
The same calibrated anchor is reused as the shrink target so we're not
anchoring to two different prices. Falls back to old behavior when
`data_baseline=None`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .baselines.fair_price import RecommendedPredictor  # NEW
from .calibration import CalibrationConfig, calibrate_to_market, inv_logit, logit
from .schemas import (
    MarketPacket,
    ModelForecast,
    SupervisorForecast,
    clamp_prob,
    normalize_distribution,
)


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


def _resolve_binary_anchor(
    packet: MarketPacket,
    data_baseline: RecommendedPredictor | None,
) -> float:
    """Anchor p_yes for binary Kalshi events.

    If a fitted data baseline is provided, run market_mid through it
    (event_size_platt + α=0.5 logit-shrink). Otherwise return market_mid raw.
    """
    if data_baseline is None:
        return packet.kalshi.market_mid
    # Preflight: RecommendedPredictor's _q_of reads only yes_ask/no_ask.
    # If either is missing, fall back to packet.kalshi.market_mid, which
    # has a last_price fallback. Avoids Platt-correcting from 0.5 on
    # illiquid markets.
    if packet.kalshi.yes_ask is None or packet.kalshi.no_ask is None:
        return packet.kalshi.market_mid
    event_dict = {
        "event_ticker": packet.event_ticker,
        "market_ticker": packet.market_ticker,
    }
    return data_baseline.predict(event_dict, packet.kalshi.to_dict())


def _anchor_distribution(
    packet: MarketPacket,
    data_baseline: RecommendedPredictor | None = None,  # CHANGED: new optional arg
) -> dict[str, float]:
    """Prior distribution used as the market-anchor in the logit pool.

    Binary Kalshi events: YES = anchor_p_yes, NO = 1 - anchor_p_yes.
    Multi-outcome: uniform over the listed outcomes.
    """
    outs = packet.outcomes or ["YES", "NO"]
    if tuple(outs) == ("YES", "NO") and packet.kalshi is not None:
        p = _resolve_binary_anchor(packet, data_baseline)  # CHANGED: was packet.kalshi.market_mid
        return {"YES": p, "NO": 1.0 - p}
    n = max(1, len(outs))
    return {o: 1.0 / n for o in outs}


def _pool_distributions(
    distributions: list[tuple[dict[str, float], float]],
    outcomes: list[str],
) -> dict[str, float]:
    """Weighted logit-pool, per outcome. (Unchanged from original.)"""
    if not outcomes:
        return {}
    n = len(outcomes)
    uniform = 1.0 / n
    raw: dict[str, float] = {}
    for outcome in outcomes:
        weighted_sum = 0.0
        total_w = 0.0
        for probs, w in distributions:
            p = probs.get(outcome)
            if p is None or p <= 0 or p >= 1:
                p = clamp_prob(p if p is not None else uniform)
            weighted_sum += w * logit(p)
            total_w += w
        if total_w <= 0:
            raw[outcome] = uniform
        else:
            raw[outcome] = inv_logit(weighted_sum / total_w)
    return normalize_distribution(raw)


def aggregate_forecasts(
    packet: MarketPacket,
    members: list[EnsembleMember],
    calibration: CalibrationConfig | None = None,
    *,
    market_anchor_weight: float = 1.5,
    data_baseline: RecommendedPredictor | None = None,  # NEW
) -> SupervisorForecast:
    outcomes = packet.outcomes or ["YES", "NO"]
    is_binary_kalshi = (
        tuple(outcomes) == ("YES", "NO") and packet.kalshi is not None
    )

    # NEW: compute the anchor p_yes once so the pool and the shrinkage
    # both target the same number (no double-anchoring to two different prices).
    anchor_p_yes = _resolve_binary_anchor(packet, data_baseline) if is_binary_kalshi else None
    if is_binary_kalshi:
        anchor = {"YES": anchor_p_yes, "NO": 1.0 - anchor_p_yes}
    else:
        anchor = _anchor_distribution(packet)  # uniform path; baseline doesn't apply

    if not members:
        raw_dist = dict(anchor)
        assessments: list[dict[str, Any]] = []
    else:
        contributions: list[tuple[dict[str, float], float]] = [(anchor, market_anchor_weight)]
        assessments = []
        for member in members:
            w = member.effective_weight
            mp = dict(member.forecast.probabilities) or anchor
            contributions.append((mp, w))
            assessments.append({
                "model_id": member.forecast.model_id,
                "provider": member.forecast.provider,
                "probabilities": mp,
                "p_yes": member.forecast.p_yes,
                "configured_weight": member.configured_weight,
                "effective_weight": w,
                "confidence": member.forecast.forecast.confidence,
                "summary": member.forecast.reasoning_track.summary,
                "defer_to_market": member.forecast.diagnostics.should_defer_to_market,
            })
        raw_dist = _pool_distributions(contributions, outcomes)

    calibration = calibration or CalibrationConfig()
    if is_binary_kalshi:
        shrink_weight = calibration.shrink_weight(packet)
        # CHANGED: shrink toward the same anchor_p_yes used in the pool above,
        # rather than calling calibrate_to_market which re-reads market_mid.
        cal_yes = clamp_prob(
            anchor_p_yes + shrink_weight * (raw_dist.get("YES", 0.5) - anchor_p_yes)
        )
        calibrated_dist = normalize_distribution({"YES": cal_yes, "NO": 1.0 - cal_yes})
    else:
        shrink_weight = calibration.shrink_weight(packet)
        anchor_share = 1.0 / max(1, len(outcomes))
        calibrated_dist = normalize_distribution({
            o: anchor_share + shrink_weight * (raw_dist.get(o, anchor_share) - anchor_share)
            for o in outcomes
        })

    if members:
        top_outcome = max(raw_dist, key=raw_dist.get)
        ps = [m.forecast.probabilities.get(top_outcome, 1.0 / len(outcomes)) for m in members]
        disagreement = (max(ps) - min(ps)) if ps else 0.0
    else:
        disagreement = 0.0
    confidence = clamp_prob(1.0 - disagreement, lo=0.0, hi=1.0)

    if disagreement > 0.20:
        disagreement_summary = f"High model disagreement on {top_outcome}: range {min(ps):.3f}-{max(ps):.3f}."
    elif disagreement > 0.08:
        disagreement_summary = f"Moderate disagreement on {top_outcome}: range {min(ps):.3f}-{max(ps):.3f}."
    else:
        disagreement_summary = "Models are broadly aligned." if members else "No model forecasts; using anchor."

    top = max(calibrated_dist, key=calibrated_dist.get) if calibrated_dist else "?"
    thesis = (
        f"Distribution over {len(outcomes)} outcomes; "
        f"calibrated top={top} @ {calibrated_dist.get(top, 0):.3f} "
        f"(raw {raw_dist.get(top, 0):.3f}, shrink weight {shrink_weight:.3f})."
    )
    risk_notes = []
    if packet.kalshi and packet.kalshi.spread is not None and packet.kalshi.spread > 0.08:
        risk_notes.append(f"Wide spread: {packet.kalshi.spread:.3f}.")
    if disagreement > 0.20:
        risk_notes.append("Large model disagreement; reduce size or no-trade.")

    return SupervisorForecast(
        market_ticker=packet.market_ticker,
        raw_probabilities=raw_dist,
        calibrated_probabilities=calibrated_dist,
        confidence=confidence,
        model_assessment=assessments,
        disagreement_summary=disagreement_summary,
        final_trade_thesis=thesis,
        risk_notes=risk_notes,
    )
