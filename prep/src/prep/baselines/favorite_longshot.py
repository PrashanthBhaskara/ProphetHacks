"""Parsimonious favorite-longshot bias correction for Sports markets.

This is the LIVE-SUBMISSION-SAFE variant of `multi_feat_logreg`. Empirically
80% of multi_feat's gain comes from these 2 features (Platt-style slope
correction + favorite-longshot bias regression), and the 2-feature model
generalizes more reliably under distribution shift.

Walk-forward validated on Sports portion of N=3793 2026 samples:
  Market alone:        Brier 0.2216
  multi_feat (6 feat): Brier 0.2186  (−0.30pp, CI [−0.49, −0.09])
  THIS (2 feat):       Brier 0.2189  (−0.27pp, CI [−0.45, −0.09])

Per-week, the 2-feature variant wins 8 / loses 6 / ties 1 vs the 6-feature
variant. They agree closely. Recommended for the live submission because
the simpler model has less risk of breaking under feature drift.

Final coefficients (fit on all 3378 Sports samples):
  bias       : +0.1015
  logit(mid) : +0.7161    (slight overconfidence — scale logit toward 0)
  |mid - 0.5|: -0.6702    (extreme prices regress toward 0.5)

Non-Sports markets pass through raw market price (Sports-only fit;
distribution mismatch on non-Sports per subset_1200 validation).
"""

from __future__ import annotations

import math

COEFS = {
    "bias":      +0.1015,
    "logit_mid": +0.7161,
    "abs_mid":   -0.6702,
}


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


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def predict(event: dict, market_info: dict | None = None) -> dict:
    """For Sports markets: apply 2-feature logreg (favorite-longshot correction).
    For non-Sports: pass-through raw market price.
    """
    mid = _market_mid(market_info)
    if mid is None:
        return {"p_yes": 0.5, "rationale": "no market price; fallback 0.5"}

    category = (event.get("category") or "").lower()
    if "sport" not in category:
        return {"p_yes": max(0.01, min(0.99, mid)),
                "rationale": f"market {mid:.3f} (non-Sports; pass-through)"}

    p = max(1e-4, min(1 - 1e-4, mid))
    z = (COEFS["bias"]
         + COEFS["logit_mid"] * math.log(p / (1 - p))
         + COEFS["abs_mid"] * abs(mid - 0.5))
    p_raw = _sigmoid(z)
    # Blend with market for variance reduction (0.7·model + 0.3·market) —
    # tighter CI without losing point-estimate gain. See multi_feat_logreg.py
    # for the supporting walk-forward analysis.
    blend_alpha = 0.7
    p_cal = max(0.01, min(0.99, blend_alpha * p_raw + (1 - blend_alpha) * mid))
    return {"p_yes": p_cal,
            "rationale": f"Sports favorite-longshot 2-feat: {blend_alpha}·model({p_raw:.3f}) + {1-blend_alpha:.1f}·mkt({mid:.3f}) → {p_cal:.3f}"}
