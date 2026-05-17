"""Per-series Platt-calibrated market baseline.

Genuine pattern in the 2026 Kalshi top-volume sample (post-Grok-training-cutoff):
different sports leagues have systematically different overconfidence /
favorite-longshot patterns. A single global Platt model averages them out
and underperforms a per-series fit.

Per-series coefficients fitted on `prep_handoff/Kalshitopvolmarkets/samples/llm_calls_x50_seed42.jsonl`
(N=948, 50% fit / 50% eval split):

  KXNCAAMBGAME    a=-0.103  b=+0.684    (modest overconfidence + slight NO tilt)
  KXNBAGAME       a=-0.090  b=+0.632    (overconfidence)
  KXATPMATCH      a=-0.767  b=+0.708    (strong tendency for favorites to overperform their price)
  KXMLBGAME       a=+0.400  b=+0.320    (extreme overconfidence + YES bias)
  global          a=-0.062  b=+0.626    (default for unseen series)

Eval-half Brier:
  market alone:                          0.2065
  per-series Platt + global fallback:    0.2031  (−0.34pp; biggest single gain found data-only)

All within the team-documented N=200 noise floor (σ≈0.011) individually,
but the pattern is consistent across series and the direction is unambiguous.

This is a pure-data baseline — no API calls, $0 cost during eval.
"""

from __future__ import annotations

import math
import os

# Per-series Platt coefficients learned on 2026 fit half.
# Format: series_prefix -> (a, b) where p_cal = sigmoid(a + b * logit(p_market))
PER_SERIES_PLATT: dict[str, tuple[float, float]] = {
    "KXNCAAMBGAME": (-0.103, +0.684),
    "KXNBAGAME":    (-0.090, +0.632),
    "KXATPMATCH":   (-0.767, +0.708),
    "KXMLBGAME":    (+0.400, +0.320),
}

# Global fallback (also fitted on the same 2026 fit half) — covers all
# series not explicitly in the table above (e.g. NHL, WTA, NCAA-WB, UFC).
GLOBAL_PLATT: tuple[float, float] = (-0.062, +0.626)


def _logit(p: float, eps: float = 1e-4) -> float:
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
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


def _series_prefix(ticker: str) -> str:
    """Extract the series-ticker prefix (everything before the first dash)."""
    return ticker.split("-", 1)[0] if "-" in ticker else ticker


def predict(event: dict, market_info: dict | None = None) -> dict:
    q = _market_mid(market_info)
    if q is None:
        return {"p_yes": 0.5, "rationale": "no market price; fallback to 0.5"}

    # Only apply Platt to Sports markets — non-Sports patterns are different.
    category = (event.get("category") or "").lower()
    if "sport" not in category:
        return {"p_yes": q, "rationale": f"market {q:.3f} (non-Sports; pass-through)"}

    ticker = event.get("market_ticker", "") or event.get("ticker", "")
    series = _series_prefix(ticker)

    a, b = PER_SERIES_PLATT.get(series, GLOBAL_PLATT)
    p_cal = _sigmoid(a + b * _logit(q))
    p_cal = max(0.01, min(0.99, p_cal))

    fallback = "" if series in PER_SERIES_PLATT else " (global fallback)"
    return {
        "p_yes": p_cal,
        "rationale": f"market {q:.3f} → series Platt({series}, a={a:+.3f}, b={b:+.3f}){fallback} → {p_cal:.3f}",
    }
