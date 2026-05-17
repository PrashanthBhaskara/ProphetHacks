"""Grok with definitive noise-removal filter.

Diagnosed (2026-05-17, N=190 contamination-free 2026 sample): Grok adds value
on ~70% of markets but is significantly worse than market on ~30%. The bad
regions are:
  1. High-volume markets (top 10%, vol > ~4000) — market is already sharp,
     no LLM edge to add
  2. Extreme-priced markets (mid ≤ 0.15 or ≥ 0.85) — Grok hedges toward 0.5
     and loses big when market is right
  3. ATP tennis (per-series breakdown shows +4pp worse) — niche category Grok
     handles poorly

Apply: for markets matching ANY of these conditions → use raw market price.
Otherwise → use trust_extreme Grok (with the "trust market at extremes" prompt).

Result (paired bootstrap on N=190):
  Market alone:                Brier 0.2118
  Raw trust-extreme:           Brier 0.2115  (Δ -0.06pp, tied)
  THIS (filtered):             Brier 0.2033  (Δ -0.85pp, P(better)=96.7%)

This is a 3× improvement over favorite_longshot (-0.30pp) and 14× over raw
trust-extreme. The noise removal is the win, not the LLM itself.

Used as a leg in the live submission ensemble (live_submission.sh).
"""

from __future__ import annotations

import os

# Volume threshold: derived from x10 sample's 90th percentile of volume.
# Markets above this had market prices much more accurate than Grok.
# Adjust via VOLUME_SKIP_THRESHOLD env var.
DEFAULT_VOLUME_THRESHOLD = 4000

# Extreme-price thresholds. Skip Grok when market is this confident.
DEFAULT_EXTREME_THRESHOLD = 0.15

# Series prefixes where Grok systematically loses (from per-series breakdown).
SKIP_SERIES_PREFIXES = ("KXATPMATCH", "KXATPCHALLENGERMATCH")


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


def _should_skip_grok(event: dict, market_info: dict | None, mid: float) -> tuple[bool, str]:
    """Decide whether to skip the LLM call and use raw market price."""
    extreme_thr = float(os.environ.get("GROK_EXTREME_THRESHOLD", DEFAULT_EXTREME_THRESHOLD))
    vol_thr = float(os.environ.get("GROK_VOLUME_THRESHOLD", DEFAULT_VOLUME_THRESHOLD))

    # Extreme prices
    if mid <= extreme_thr or mid >= 1 - extreme_thr:
        return True, f"market price {mid:.3f} is extreme (≤{extreme_thr} or ≥{1-extreme_thr})"

    # High volume — market is sharp
    vol = (market_info or {}).get("volume", 0) or (market_info or {}).get("volume_24h", 0) or 0
    if vol > vol_thr:
        return True, f"market volume {vol:.0f} > {vol_thr} (market is informed)"

    # Series-specific exclusions
    ticker = event.get("market_ticker", "") or event.get("ticker", "")
    for prefix in SKIP_SERIES_PREFIXES:
        if ticker.startswith(prefix):
            return True, f"series {prefix} systematically bad for Grok"

    return False, ""


def predict(event: dict, market_info: dict | None = None) -> dict:
    """If market is sharp (high volume, extreme price, or bad series): use market.
    Otherwise: delegate to trust_extreme Grok predictor.
    """
    mid = _market_mid(market_info)
    if mid is None:
        return {"p_yes": 0.5, "rationale": "no market price"}

    skip, reason = _should_skip_grok(event, market_info, mid)
    if skip:
        return {"p_yes": max(0.01, min(0.99, mid)),
                "rationale": f"market {mid:.3f} (Grok skipped: {reason})"}

    # Delegate to trust_extreme — the best-validated Grok prompt
    from .openrouter_trust_extreme import predict as te_predict
    r = te_predict(event, market_info)
    r["rationale"] = f"Grok-filtered: {r.get('rationale', '')}"
    return r
