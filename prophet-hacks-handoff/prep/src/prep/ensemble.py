"""Supervisor aggregation for model forecasts.

Per Prophet Arena dev docs, predictions are a probability *distribution* over
the event's `outcomes` list. We logit-pool per outcome across lanes, then
renormalize. Binary events (outcomes=["YES","NO"]) reduce cleanly to the
old logit-pool of a single p_yes.

Market anchor only applies when the packet has a Kalshi quote (binary YES/NO).
For multi-outcome events without market data, we anchor toward a uniform prior
with a small weight (so a single noisy model can't dominate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def _anchor_distribution(packet: MarketPacket) -> dict[str, float]:
    """Prior distribution used as the market-anchor in the logit pool.

    Binary Kalshi events: YES = market_mid, NO = 1 - market_mid.
    Multi-outcome: uniform over the listed outcomes.
    """
    outs = packet.outcomes or ["YES", "NO"]
    if tuple(outs) == ("YES", "NO") and packet.kalshi is not None:
        mid = packet.kalshi.market_mid
        return {"YES": mid, "NO": 1.0 - mid}
    n = max(1, len(outs))
    return {o: 1.0 / n for o in outs}


def _pool_distributions(
    distributions: list[tuple[dict[str, float], float]],
    outcomes: list[str],
) -> dict[str, float]:
    """Weighted logit-pool, per outcome.

    `distributions` is a list of (probs, weight). For each outcome label we
    average weighted logits, then inv-logit, then renormalize across outcomes.
    Missing outcomes in a lane's distribution fall back to uniform (1/N).

    For binary YES/NO events specifically, a case-insensitive secondary
    lookup is used if the exact-case match misses, so a lane that returned
    {"Yes": 0.7} against canonical ["YES", "NO"] still contributes its
    signal rather than being silently substituted with uniform. Multi-
    outcome events are untouched (avoids collision risk on labels that
    differ only by case).
    """
    if not outcomes:
        return {}
    n = len(outcomes)
    uniform = 1.0 / n
    is_binary = tuple(outcomes) == ("YES", "NO")
    raw: dict[str, float] = {}
    for outcome in outcomes:
        weighted_sum = 0.0
        total_w = 0.0
        for probs, w in distributions:
            p = probs.get(outcome)
            if p is None and is_binary:
                folded = outcome.casefold()
                for k, v in probs.items():
                    if isinstance(k, str) and k.casefold() == folded:
                        p = v
                        break
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
) -> SupervisorForecast:
    outcomes = packet.outcomes or ["YES", "NO"]
    anchor = _anchor_distribution(packet)

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
    # Calibration shrinks each outcome toward the market anchor by the same
    # per-event weight. Multi-outcome non-Kalshi events get the uniform anchor.
    if tuple(outcomes) == ("YES", "NO") and packet.kalshi is not None:
        # Reuse existing binary calibrate_to_market on YES side, mirror to NO
        cal_yes, shrink_weight = calibrate_to_market(raw_dist.get("YES", 0.5), packet, calibration)
        calibrated_dist = normalize_distribution({"YES": cal_yes, "NO": 1.0 - cal_yes})
    else:
        # Multi-outcome shrinkage: pull each prob toward uniform by shrink_weight
        shrink_weight = calibration.shrink_weight(packet)
        anchor_share = 1.0 / max(1, len(outcomes))
        calibrated_dist = normalize_distribution({
            o: anchor_share + shrink_weight * (raw_dist.get(o, anchor_share) - anchor_share)
            for o in outcomes
        })

    # Disagreement: max range of p across lanes, for the most-likely outcome
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
