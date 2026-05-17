"""Pure-stdlib port of Victor's `fair_price.RecommendedPredictor`.

Uses the **published coefficients** from `prophet-hacks-handoff/FORECAST_BENCHMARKS.md`:
documented Brier 0.171 on subset_1200 holdout (N=2,090), beating market's 0.185.
No training, no numpy. Just plug in market price + event size.

Formula:
    p_es     = sigmoid(0.61 + 1.17 · logit(q) − 0.46 · log(N_event))
    p_final  = sigmoid(0.5 · logit(p_es) + 0.5 · logit(q))     # α=0.5 shrinkage

Where:
    q       = bid/ask midpoint
    N_event = number of binary markets sharing the same `event_ticker`
              (in the live candidate set; defaults to 1 if unknown)

Two usage patterns:
    # batch (preferred — n_event computed from the candidate set)
    preds = predict_batch(samples)

    # single market (n_event fallback to prefix lookup or default)
    p = predict(event, market_info, n_event=…)

Caveats from the source benchmark file:
- Without `candidate_set`, n_event defaults to 1 → predictor collapses to
  a Platt of q (still better than raw q, but ~half the alpha).
- At N=200 noise floor σ ≈ 0.011. Don't read fractional Brier deltas
  as real.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable, Sequence

# Coefficients from b1f32b3 (Victor's verified benchmark).
BIAS = 0.61
LOGIT_Q_SLOPE = 1.17
LOG_EVENT_SIZE_SLOPE = -0.46
SHRINK_ALPHA = 0.5  # logit-space blend toward q

CLAMP_LOW = 0.01
CLAMP_HIGH = 0.99


def _logit(p: float) -> float:
    p = max(1e-4, min(1 - 1e-4, p))
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def _q_of(market_info: dict | None) -> float | None:
    """Bid/ask midpoint in [0, 1]. None if market_info has no usable price."""
    if not market_info:
        return None
    yes_ask = market_info.get("yes_ask")
    no_ask = market_info.get("no_ask")
    yes_bid = market_info.get("yes_bid")
    no_bid = market_info.get("no_bid")
    last_price = market_info.get("last_price")
    # Prefer spread-corrected midpoint
    if yes_ask is not None and no_ask is not None and (yes_ask + no_ask) > 0:
        return (yes_ask + (100 - no_ask)) / 200
    # Fallback to yes_bid/yes_ask midpoint
    if yes_bid is not None and yes_ask is not None:
        return (yes_bid + yes_ask) / 200
    if last_price is not None:
        return last_price / 100
    return None


def fair_price_one(q: float, n_event: float = 1.0) -> float:
    """Single-market fair price given market mid q and event size N."""
    z = BIAS + LOGIT_Q_SLOPE * _logit(q) + LOG_EVENT_SIZE_SLOPE * math.log(max(1.0, n_event))
    p_es = _sigmoid(z)
    z_final = SHRINK_ALPHA * _logit(p_es) + (1 - SHRINK_ALPHA) * _logit(q)
    return max(CLAMP_LOW, min(CLAMP_HIGH, _sigmoid(z_final)))


def predict(event: dict, market_info: dict | None, n_event: int | None = None) -> dict:
    """Production-shape predictor: returns {p_yes, rationale}.

    If `n_event` is provided, uses it. Otherwise defaults to 1 (Platt-only
    mode — see module docstring).
    """
    q = _q_of(market_info)
    if q is None:
        return {"p_yes": 0.5, "rationale": "fair_price: no market price available, fell back to 0.5"}
    n = max(1, int(n_event)) if n_event else 1
    p = fair_price_one(q, n)
    return {"p_yes": p, "rationale": f"fair_price(q={q:.3f}, N={n}) → {p:.3f}"}


def predict_batch(samples: Iterable) -> dict[str, float]:
    """Run over a candidate set; computes n_event from sibling count.

    `samples` must have `.event['event_ticker']` and `.event['market_ticker']`
    and `.market_info`. Compatible with `prep.data.Sample`.
    """
    samples = list(samples)
    sizes: Counter[str] = Counter()
    for s in samples:
        et = s.event.get("event_ticker") or ""
        sizes[et] += 1

    out: dict[str, float] = {}
    for s in samples:
        et = s.event.get("event_ticker") or ""
        n = max(1, sizes.get(et, 1))
        q = _q_of(s.market_info)
        ticker = s.event.get("market_ticker") or ""
        if q is None or not ticker:
            continue
        out[ticker] = fair_price_one(q, n)
    return out
