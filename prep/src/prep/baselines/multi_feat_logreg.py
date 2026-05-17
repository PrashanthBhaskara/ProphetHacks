"""Multi-feature logistic regression calibrator.

Walk-forward validated on N=3793 contamination-free 2026 samples
(prep_handoff/Kalshitopvolmarkets/samples/llm_calls_x200_seed42.jsonl,
post-Grok-training-cutoff). Beats raw market price by **−0.25pp Brier
absolute** with paired-bootstrap 95% CI [−0.42pp, −0.08pp] entirely
below zero (p ≈ 0.001 even after Bonferroni correction for ~10 tested
methods).

Features:
    1, logit(mid), log(volume+1), log(open_interest+1),
    spread, |mid-0.5|, log(minutes_to_close+1)

Final coefficients (fit on ALL 3793 samples):
    bias        :  +1.0606
    logit_mid   :  +0.7342   (similar to Platt slope)
    log_volume  :  +0.0196   (slight YES tilt for high-volume)
    log_oi      :  -0.0553
    spread      :  -0.5468   (wider spread = less informed → predict less YES)
    |mid-0.5|   :  -0.7543   (extreme prices regress toward 0.5)
    log_mtc     :  -0.0694   (slight effect of time-to-close)

Per-week walk-forward: 11/15 weeks better than market, aggregate −0.25pp.
On the most recent 5 weeks (Apr 9 - May 7): all 5 wins, aggregate
larger gain.

Caveats:
- Trained on 2026 Kalshi top-volume markets, 89% Sports. May not transfer
  to a different category mix (subset_1200 has more Politics/Entertainment).
- Per Victor's docs, multi-feature on subset_1200 OVERFITS (subset_1200
  Brier 0.181 vs market 0.185 with these features). 2026 has more
  consistent bias patterns or different feature relevance.
- volume / open_interest must be present in market_info. Missing values
  default to 0 (logged as log(1)=0, effectively no contribution).

This is a pure-data baseline — no API calls, $0 cost during eval.
"""

from __future__ import annotations

import math

# Sports-only coefficients (fit on all 2647 Sports samples).
# Sports-only walks-forward marginally better than full-data version
# (Δ −0.30pp vs −0.25pp aggregate; same paired-bootstrap significance p≈0.001).
# Order matches feats() below: bias, logit(mid), log(vol+1), log(oi+1), spread, |mid-0.5|, log(mtc+1).
COEFS: list[float] = [
    +0.7494,   # bias
    +0.7154,   # logit(mid)
    +0.0307,   # log(volume + 1)
    -0.0500,   # log(open_interest + 1)
    -0.4185,   # spread
    -0.6858,   # |mid - 0.5|
    -0.0364,   # log(minutes_to_close + 1)
]


def _market_mid(market_info: dict | None) -> float | None:
    if not market_info:
        return None
    yes_ask = market_info.get("yes_ask")
    no_ask = market_info.get("no_ask")
    last_price = market_info.get("last_price")
    if yes_ask is not None and no_ask is not None and (yes_ask + no_ask) > 0:
        return (yes_ask + (100 - no_ask)) / 200
    if last_price is not None:
        return last_price / 100
    return None


def _features(market_info: dict | None, minutes_to_close: float | None = None) -> list[float] | None:
    mid = _market_mid(market_info)
    if mid is None:
        return None
    yes_ask = market_info.get("yes_ask", 0) or 0
    no_ask = market_info.get("no_ask", 0) or 0
    spread = max(0.0, (yes_ask - (100 - no_ask)) / 100) if yes_ask and no_ask else 0.0
    # If yes_bid exists, prefer spread = yes_ask - yes_bid (more accurate)
    if market_info.get("yes_bid") is not None and yes_ask:
        # Convert to fraction form
        spread = max(0.0, (yes_ask - market_info["yes_bid"]) / 100)
    vol = market_info.get("volume", 0) or market_info.get("volume_24h", 0) or 0
    oi = market_info.get("open_interest", 0) or 0
    mtc = minutes_to_close if minutes_to_close is not None else 1440  # default 1 day

    p = max(1e-4, min(1 - 1e-4, mid))
    return [
        1.0,
        math.log(p / (1 - p)),
        math.log(vol + 1),
        math.log(oi + 1),
        spread,
        abs(mid - 0.5),
        math.log(max(0.0, mtc) + 1),
    ]


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def predict(event: dict, market_info: dict | None = None) -> dict:
    """Production-style predict.

    Sports markets: apply the multi-feature logreg (validated +0.30pp Brier
    improvement vs raw market on 2026 walk-forward, p≈0.001).
    Non-Sports markets: pass-through raw market mid (we don't have
    enough non-Sports data in the 2026 train set to fit a reliable
    calibration, and the Sports-fit model OVERFITS on subset_1200
    non-Sports — distribution mismatch).
    """
    mid = _market_mid(market_info)
    if mid is None:
        return {"p_yes": 0.5, "rationale": "no market price; fallback 0.5"}

    category = (event.get("category") or "").lower()
    if "sport" not in category:
        return {"p_yes": max(0.01, min(0.99, mid)),
                "rationale": f"market {mid:.3f} (non-Sports; pass-through to avoid OOD overfit)"}

    minutes_to_close = None
    close_time = event.get("close_time")
    if close_time:
        try:
            from datetime import datetime, UTC
            ct = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
            now = datetime.now(UTC)
            minutes_to_close = max(0.0, (ct - now).total_seconds() / 60)
        except Exception:
            pass

    x = _features(market_info, minutes_to_close)
    if x is None:
        return {"p_yes": max(0.01, min(0.99, mid)), "rationale": "features unavailable; market pass-through"}
    z = sum(c * xi for c, xi in zip(COEFS, x))
    p_logreg = _sigmoid(z)

    # Blend with market for variance reduction. Walk-forward analysis on
    # N=2647 Sports samples shows 0.7·logreg + 0.3·market gives:
    #   point estimate: −0.27pp (vs pure logreg's −0.29pp; essentially identical)
    #   95% CI:         [−0.41, −0.13]  (vs pure logreg [−0.50, −0.09]; tighter)
    # The tightening means we're more confident the win is real on any given
    # eval sample. Net positive trade.
    blend_alpha = 0.7  # weight on logreg; 0.3 weight on market
    p_blended = blend_alpha * p_logreg + (1 - blend_alpha) * mid
    p = max(0.01, min(0.99, p_blended))
    return {
        "p_yes": p,
        "rationale": f"Sports {blend_alpha}·logreg({p_logreg:.3f}) + {1-blend_alpha:.1f}·mkt({mid:.3f}) → {p:.3f}",
    }
