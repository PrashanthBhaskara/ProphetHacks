"""Per-category calibrated-market baseline.

Empirical finding from 2026 data (`prep_handoff/Kalshitopvolmarkets/samples/llm_calls_x50_seed42.jsonl`,
N=948 post-Grok-training-cutoff):

  Sports market is overconfident (Platt slope ~0.59 → market logit
  should be scaled down by ~40%). The best simple data-only intervention
  found was: shrink market toward Sports base rate (~0.467) by α≈0.10.

This is small (within noise floor) but consistent across many α and
fits. Worth turning on for the live submission because cost is zero
and downside is minimal.

Behavior:
  Sports markets   → logit-pool with `base_rate`, weight 1−α on market
  Other markets    → market price unchanged
  No market info   → 0.5 fallback

Default α and base rates are tuned on the N=948 2026 Sports-heavy sample;
override via constants at the top of this file or by environment
variables `CALIBRATED_SHRINK_ALPHA` / `CALIBRATED_BASE_RATE_SPORTS`.

This is a pure-data baseline — no API calls, $0 cost during eval.
"""

from __future__ import annotations

import math
import os

# Empirical from 2026 sample (Sports market shows overconfidence).
DEFAULT_ALPHA_SPORTS = 0.10
DEFAULT_BASE_RATE_SPORTS = 0.467

# Empirical: Other category was already well-calibrated; no shrinkage.
DEFAULT_ALPHA_OTHER = 0.0


def _logit(p: float, eps: float = 1e-4) -> float:
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


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


def predict(event: dict, market_info: dict | None = None) -> dict:
    q = _market_mid(market_info)
    if q is None:
        return {"p_yes": 0.5, "rationale": "no market price; fallback to 0.5"}

    category = (event.get("category") or "").lower()
    alpha = float(os.environ.get("CALIBRATED_SHRINK_ALPHA", DEFAULT_ALPHA_SPORTS))
    base = float(os.environ.get("CALIBRATED_BASE_RATE_SPORTS", DEFAULT_BASE_RATE_SPORTS))

    if "sport" in category:
        # Linear (probability-space) shrink toward base rate.
        p = (1 - alpha) * q + alpha * base
        return {"p_yes": max(0.01, min(0.99, p)),
                "rationale": f"market {q:.3f} shrunk α={alpha} toward Sports base rate {base:.3f}"}
    # Default: market unchanged
    return {"p_yes": max(0.01, min(0.99, q)),
            "rationale": f"market {q:.3f} (category={category!r}, no shrinkage)"}
